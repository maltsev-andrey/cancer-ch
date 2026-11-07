#!/usr/bin/env python3
"""
cancer-ch / Stage 6: Bayesian age-period-cohort model (NumPyro).

Data: processed/apc_incidence_grid.parquet (NKRS, Switzerland), one fit per
site x sex x indicator on the 18 age x 6 period grid (23 birth cohorts).

Model (per cell a, p; cohort c = p - a):
    y_ap ~ Poisson(PY_ap * exp(eta_ap))
    eta_ap = alpha_a + delta * (p - p_mean) + beta_p + gamma_c + eps_ap
    alpha : RW2 with free level and slope (age curve)
    delta : net drift (common linear period + cohort trend, identifiable)
    beta  : RW2, linear trend removed  -> period curvature only
    gamma : RW2, linear trend removed  -> cohort curvature only
    eps   : iid N(0, s_eps), extra-Poisson variation
The linear trend of period vs cohort is not identifiable from the data; only
drift, curvatures and the age curve are reported, as in Holford / Riebler & Held.

Modes
  python pipeline/s06_apc.py --site "Lunge, Bronchien, Luftröhre" --sex M --indicator incidence
  python pipeline/s06_apc.py --all                     # every site x sex x indicator with data
  python pipeline/s06_apc.py --list                    # print the fit list (for Slurm arrays)
  python pipeline/s06_apc.py --index 7                 # fit number 7 of that list
  --holdout       fit 1993-2017, predict 2018-2022 (out-of-sample check)
  --future 2      project 2 periods (2023-27, 2028-32) by extrapolating the RW2 priors

Outputs: data/models/apc/<slug>/  posterior.nc (ArviZ), summary.json, fit.png
         data/models/apc/index.parquet (one row per fit: diagnostics + headline numbers)
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- config ----
REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    """paths.data from sources.yaml, relative to the repo root unless absolute;
    the environment variable CANCER_CH_DATA overrides both."""
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()
DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
GRID = DATA / "processed" / "apc_incidence_grid.parquet"
OUT = DATA / "models" / "apc"
REGION = "Schweiz"
MIN_CASES_PER_YEAR = 30          # below this the grid is too sparse for a 3-effect model
PERIODS = ["1993-1997", "1998-2002", "2003-2007", "2008-2012", "2013-2017", "2018-2022"]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def slug(site, sex, ind):
    s = re.sub(r"[^a-z0-9]+", "-", site.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")).strip("-")
    return f"{s}__{sex}__{ind}"


# ============================================================ data ==========
def load_cells(site, sex, indicator, holdout=False):
    g = pd.read_parquet(GRID)
    g = g[(g.region == REGION) & (g.site == site) & (g.sex == sex) & (g.indicator == indicator)
          & ~g.structural_zero & g.person_years.gt(0)].copy()
    g["a"] = g.age5.map(lambda s: int(re.match(r"\d+", s).group()) // 5)      # 0..17
    g["p"] = g.period.map(PERIODS.index)                                      # 0..5
    g["y"] = np.rint(g.cases).astype(int)
    g = g.sort_values(["p", "a"])
    train = g[g.p < 5] if holdout else g
    return g, train


# ============================================================ model =========
def rw2_curve(d2, n, scale):
    """Second-order random walk from its second differences d2 (length n-2), first two
    values pinned at 0. The double cumsum of n iid N(0,1) has amplitude ~ n^1.5, so it is
    normalised: `scale` is the prior SD of the curve itself, whatever n is."""
    import jax.numpy as jnp
    inc = jnp.concatenate([jnp.zeros(2), d2])
    return jnp.cumsum(jnp.cumsum(inc)) * scale / (n ** 1.5)


def detrend(x, n_obs):
    """Remove the OLS line fitted on the first n_obs points from all points, so that the
    curve has zero mean and zero slope on the observed range and the linear part lives
    only in `drift`."""
    import jax.numpy as jnp
    t = jnp.arange(x.shape[0], dtype=jnp.float32)
    t_obs, x_obs = t[:n_obs], x[:n_obs]
    tm, xm = t_obs.mean(), x_obs.mean()
    slope = ((t_obs - tm) * (x_obs - xm)).sum() / ((t_obs - tm) ** 2).sum()
    return x - xm - slope * (t - tm)


def build_model():
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def rw2(name, n, scale):
        d2 = numpyro.sample(f"{name}_d2", dist.Normal(0, 1).expand([max(n - 2, 0)]))
        return rw2_curve(d2, n, scale)

    def model(a_idx, p_idx, c_idx, log_py, y, A, P, C, n_future):
        Pf, Cf = P + n_future, C + n_future
        # age: level + first-order random walk (increments of ~0.3-0.8 per 5-y step are
        # typical, so a non-centred RW1 is well conditioned; an RW2 on a 6-log-unit curve
        # needed second differences far outside their N(0,1) prior and chains did not mix)
        # The age curve is pinned by the data (6 periods per age), so it is a free vector
        # with a weak prior: centred parameterisation, no hyperparameter to create a funnel.
        alpha = numpyro.sample("age_free", dist.Normal(-8.0, 5.0).expand([A]))
        # drift + detrended period and cohort curvatures
        drift = numpyro.sample("drift", dist.Normal(0.0, 0.5))
        s_per = numpyro.sample("s_period", dist.HalfNormal(0.5))
        s_coh = numpyro.sample("s_cohort", dist.HalfNormal(0.5))
        beta = detrend(rw2("period", Pf, s_per), P)
        gamma = detrend(rw2("cohort", Cf, s_coh), C)
        numpyro.deterministic("alpha", alpha)
        numpyro.deterministic("beta", beta)
        numpyro.deterministic("gamma", gamma)
        # extra-Poisson variation on observed cells
        s_eps = numpyro.sample("s_eps", dist.HalfNormal(0.1))
        eps = numpyro.sample("eps", dist.Normal(0, 1).expand([a_idx.shape[0]])) * s_eps
        eta = (alpha[a_idx] + drift * (p_idx - (P - 1) / 2.0) + beta[p_idx] + gamma[c_idx] + eps)
        numpyro.deterministic("log_rate", eta)
        numpyro.sample("y", dist.Poisson(jnp.exp(eta + log_py)), obs=y)
        # rates on the full grid (observed + future periods), no eps
        aa, pp = jnp.meshgrid(jnp.arange(A), jnp.arange(Pf), indexing="ij")
        cc = pp - aa + (A - 1)
        numpyro.deterministic("rate_grid", jnp.exp(alpha[aa] + drift * (pp - (P - 1) / 2.0)
                                                   + beta[pp] + gamma[cc]) * 1e5)

    return model


# ============================================================ fit ===========
def fit_one(site, sex, indicator, args):
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS, init_to_median

    g, train = load_cells(site, sex, indicator, args.holdout)
    if g.empty or g.cases_per_year.sum() / g.period.nunique() < MIN_CASES_PER_YEAR:
        log(f"skip {site}/{sex}/{indicator}: too few cases")
        return None
    A, P = 18, 6
    C = A + P - 1
    a, p = train.a.values, train.p.values
    c = p - a + (A - 1)
    data = dict(a_idx=jax.numpy.array(a), p_idx=jax.numpy.array(p), c_idx=jax.numpy.array(c),
                log_py=jax.numpy.array(np.log(train.person_years.values)),
                y=jax.numpy.array(train.y.values), A=A, P=P, C=C, n_future=args.future)

    kernel = NUTS(build_model(), target_accept_prob=args.target_accept, max_tree_depth=10,
                  init_strategy=init_to_median(num_samples=50))
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains,
                chain_method="vectorized" if jax.devices()[0].platform == "gpu" else "parallel",
                progress_bar=False)
    t0 = datetime.now()
    mcmc.run(jax.random.PRNGKey(args.seed), extra_fields=("energy", "diverging", "num_steps"), **data)
    secs = (datetime.now() - t0).total_seconds()
    idata = az.from_numpyro(mcmc)
    smry = az.summary(idata, var_names=["drift", "s_period", "s_cohort", "s_eps"])
    div = int(idata.sample_stats.diverging.sum())
    diag = diagnostics(idata)
    rhat_by_var, rhat_max = diag["rhat_by_var"], diag["rhat_max"]

    post = idata.posterior
    rate = post.rate_grid.values.reshape(-1, A, P + args.future)          # draws x A x P'
    drift_yr = (np.exp(post.drift.values.ravel() / 5) - 1) * 100         # % per year
    out = {"site": site, "sex": sex, "indicator": indicator, "is_holdout": args.holdout,
           "n_cells": int(len(train)), "cases_per_year": float(g.cases_per_year.sum() / 6),
           "chains": args.chains, "samples": args.samples, "seconds": round(secs, 1),
           "device": jax.devices()[0].platform, "divergences": div, "rhat_max": round(rhat_max, 3),
           "rhat_by_var": rhat_by_var, "ess_bulk_min": diag["ess_bulk_min"],
           "ess_tail_min": diag["ess_tail_min"], "ess_by_var": diag["ess_by_var"],
           "bfmi_min": diag["bfmi_min"], "tree_depth_max": diag["tree_depth_max"],
           "converged": converged(div, rhat_max, diag["ess_bulk_min"], diag["ess_tail_min"], diag["bfmi_min"]),
           "net_drift_pct_per_year": [float(np.percentile(drift_yr, q)) for q in (2.5, 50, 97.5)],
           "period_curv_sd": float(post.beta.std(("chain", "draw")).mean()),
           "cohort_curv_sd": float(post.gamma.std(("chain", "draw")).mean())}

    # observed vs fitted on the full grid, holdout error if requested
    obs = np.full((A, P), np.nan)
    obs[g.a.values, g.p.values] = g.cases_per_year.values / g.person_years.values * 5e5
    fit_med = np.median(rate, axis=0)
    ok = np.isfinite(obs) & (obs > 0)
    cpy = np.full((A, P), np.nan); cpy[g.a.values, g.p.values] = g.cases_per_year.values
    big = ok & (np.nan_to_num(cpy) >= 20)
    out["mape_fitted_pct_all"] = float(np.mean(np.abs(fit_med[:, :P][ok] / obs[ok] - 1)) * 100)
    out["mape_fitted_pct"] = float(np.mean(np.abs(fit_med[:, :P][big] / obs[big] - 1)) * 100)
    if args.holdout:
        test = g[g.p == 5]
        pred = rate[:, test.a.values, 5]                                   # draws x cells
        # posterior predictive: parameter uncertainty + extra-Poisson eps + Poisson noise
        rng = np.random.default_rng(args.seed)
        s_eps = post.s_eps.values.ravel()[:, None]
        mu = pred / 1e5 * test.person_years.values * np.exp(rng.normal(0, 1, pred.shape) * s_eps)
        pred_cases = rng.poisson(mu)
        lo, hi = np.percentile(pred_cases, [5, 95], axis=0)
        inside = ((test.cases.values >= lo) & (test.cases.values <= hi)).mean()
        asr_obs = asr(obs[:, 5]); asr_pred = asr(np.median(rate[:, :, 5], axis=0))
        out["holdout_check"] = {"period": "2018-2022", "asr_observed": asr_obs, "asr_predicted": asr_pred,
                          "asr_error_pct": (asr_pred / asr_obs - 1) * 100,
                          "cells_in_90pct_interval": float(inside)}
    if args.future:
        out["projection"] = {f"{2023 + 5 * k}-{2027 + 5 * k}":
                             [float(np.percentile([asr(r[:, P + k]) for r in rate], q)) for q in (2.5, 50, 97.5)]
                             for k in range(args.future)}
    out["asr_fitted_by_period"] = {per: asr(fit_med[:, i]) for i, per in enumerate(PERIODS)}

    d = OUT / (slug(site, sex, indicator) + ("_holdout" if args.holdout else ""))
    d.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(d / "posterior.nc")
    (d / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    smry.to_csv(d / "hyper_summary.csv")
    plot_fit(g, rate, post, A, P, args.future, site, sex, indicator, d / "fit.png")
    flag = "OK" if out["converged"] else "CHECK"
    log(f"{flag} {site}/{sex}/{indicator}: {secs:.0f}s on {out['device']}, div={div}, "
        f"rhat={rhat_max:.3f}, ess={diag['ess_bulk_min']:.0f}/{diag['ess_tail_min']:.0f}, "
        f"bfmi={diag['bfmi_min']:.2f}, drift={out['net_drift_pct_per_year'][1]:+.2f}%/yr"
        + (f", holdout ASR err {out['holdout_check']['asr_error_pct']:+.1f}%" if args.holdout else ""))
    return out


DIAG_VARS = ["alpha", "beta", "gamma", "drift", "s_period", "s_cohort", "s_eps"]
ESS_MIN = 400          # Vehtari et al. 2021: 100 per chain, 4 chains
BFMI_MIN = 0.3         # Betancourt 2016: below 0.3 the energy transitions are too sticky
RHAT_MAX = 1.01        # was 1.05; the sweep rerun shows what this costs


def diagnostics(idata):
    """R-hat, ESS (bulk, tail), BFMI per chain and max tree depth, per variable and overall.
    All reductions are nanmax/nanmin over the parameter vectors."""
    import arviz as az
    rh = az.rhat(idata, var_names=DIAG_VARS)
    eb = az.ess(idata, var_names=DIAG_VARS, method="bulk")
    et = az.ess(idata, var_names=DIAG_VARS, method="tail")
    f = lambda ds, fn: {v: round(float(fn(np.asarray(ds[v]))), 3) for v in ds.data_vars}
    rhat_by_var, ess_b, ess_t = f(rh, np.nanmax), f(eb, np.nanmin), f(et, np.nanmin)
    ss = idata.sample_stats
    # energy is only there if mcmc.run() asked for it in extra_fields; without it BFMI is
    # unknown and the gate treats it as failed (nan compares False)
    if "energy" in ss:
        b = az.bfmi(idata)                       # ndarray (arviz<1) or DataTree (arviz>=1)
        bfmi = np.asarray(b["energy"].values if hasattr(b, "children") else b, dtype=float)
    else:
        bfmi = np.array([np.nan])
    depth = ss["tree_depth"] if "tree_depth" in ss else None   # from_numpyro derives it from num_steps
    return {"rhat_by_var": rhat_by_var, "rhat_max": max(rhat_by_var.values()),
            "ess_by_var": {v: [ess_b[v], ess_t[v]] for v in ess_b},
            "ess_bulk_min": min(ess_b.values()), "ess_tail_min": min(ess_t.values()),
            "bfmi_by_chain": [round(float(x), 3) for x in bfmi],
            "bfmi_min": round(float(np.min(bfmi)), 3),
            "tree_depth_max": int(depth.max()) if depth is not None else None}


def converged(div, rhat_max, ess_bulk_min, ess_tail_min, bfmi_min):
    return bool(div <= 10 and rhat_max < RHAT_MAX and ess_bulk_min >= ESS_MIN
                and ess_tail_min >= ESS_MIN and bfmi_min >= BFMI_MIN)


ESP = np.array([8000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 6000, 5000,
                4000, 3000, 2000, 1000, 1000], dtype=float)


def asr(rates_by_age):
    r = np.asarray(rates_by_age, dtype=float)
    ok = ~np.isnan(r)
    return float((r[ok] * ESP[ok]).sum() / ESP[ok].sum())


# ============================================================ plot ==========
def plot_fit(g, rate, post, A, P, nf, site, sex, ind, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ages = [f"{5 * i}-{5 * i + 4}" for i in range(17)] + ["85+"]
    coh = [1993 - 5 * (A - 1) + 5 * k for k in range(A + P - 1 + nf)]      # cohort start years
    col = "#2a78d6" if sex == "M" else "#eb6834"
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), facecolor="#fcfcfb")
    fig.suptitle(f"{site}, {'men' if sex == 'M' else 'women'}, {ind}: age-period-cohort fit",
                 x=0.02, ha="left", fontweight="bold")
    # age curves per period: observed dots, fitted lines
    a0 = ax[0, 0]
    for i, per in enumerate(PERIODS):
        obs = g[g.p == i]
        a0.plot(obs.a, obs.cases_per_year / obs.person_years * 5e5, "o", ms=3, color=col, alpha=0.25 + 0.15 * i)
        a0.plot(range(A), np.median(rate[:, :, i], axis=0), color=col, alpha=0.25 + 0.15 * i, lw=1.5,
                label=per if i in (0, 5) else None)
    a0.set_yscale("log"); a0.set_xticks(range(0, A, 3), ages[::3]); a0.legend(frameon=False, fontsize=8)
    a0.set_title("Age-specific rate per 100,000 (dots observed, lines fitted)", loc="left", fontsize=10)
    # ASR by period, observed vs fitted + projection
    a1 = ax[0, 1]
    asr_fit = np.array([[asr(r[:, i]) for i in range(P + nf)] for r in rate])
    x = np.arange(P + nf)
    lo, md, hi = np.percentile(asr_fit, [2.5, 50, 97.5], axis=0)
    a1.fill_between(x, lo, hi, color=col, alpha=0.15); a1.plot(x, md, color=col)
    obs_asr = []
    for i in range(P):
        o = np.full(A, np.nan); gg = g[g.p == i]
        o[gg.a.values] = gg.cases_per_year / gg.person_years * 5e5
        obs_asr.append(asr(o))
    a1.plot(range(P), obs_asr, "o", color="#0b0b0b", ms=5, label="observed")
    labels = PERIODS + [f"{2023 + 5 * k}-{2027 + 5 * k}" for k in range(nf)]
    a1.set_xticks(x, labels, rotation=30, ha="right", fontsize=8); a1.set_ylim(bottom=0)
    a1.set_title("ASR (Europe 1976), fitted with 95% interval" + (", projected" if nf else ""), loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    # period and cohort curvature as RR
    for axx, name, xs, ttl in ((ax[1, 0], "beta", labels, "Period effect (curvature), RR"),
                               (ax[1, 1], "gamma", [str(c) for c in coh], "Cohort effect (curvature), RR vs linear trend")):
        v = np.exp(post[name].values.reshape(-1, post[name].shape[-1]))
        lo, md, hi = np.percentile(v, [2.5, 50, 97.5], axis=0)
        xi = np.arange(len(md))
        axx.fill_between(xi, lo, hi, color=col, alpha=0.15); axx.plot(xi, md, color=col)
        axx.axhline(1, color="#c3c2b7", lw=1)
        step = max(1, len(xs) // 8)
        axx.set_xticks(xi[::step], xs[::step], rotation=30, ha="right", fontsize=8)
        axx.set_title(ttl, loc="left", fontsize=10)
    for a_ in ax.ravel():
        a_.grid(color="#e1e0d9", lw=0.6); a_.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "Source: NKRS/BFS px-x-1403030300_161, model: Bayesian APC (NumPyro)",
             color="#898781", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=160); plt.close(fig)


# ============================================================ main ==========
def fit_list():
    g = pd.read_parquet(GRID)
    g = g[(g.region == REGION) & ~g.structural_zero & ~g.site.str.startswith(" ") & g.sex.isin(["M", "F"])]
    s = g.groupby(["site", "sex", "indicator"]).cases_per_year.sum() / 6
    s = s[s >= MIN_CASES_PER_YEAR].reset_index()
    return s.sort_values("cases_per_year", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site"); ap.add_argument("--sex"); ap.add_argument("--indicator", default="incidence")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--holdout", action="store_true"); ap.add_argument("--future", type=int, default=0)
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-accept", dest="target_accept", type=float, default=0.9)
    args = ap.parse_args()

    fl = fit_list()
    if args.list:
        print(fl.to_string()); print(f"{len(fl)} fits"); return
    if args.index is not None:
        jobs = [fl.iloc[args.index]]
    elif args.all:
        jobs = [r for _, r in fl.iterrows()]
    else:
        if not (args.site and args.sex):
            sys.exit("give --site and --sex, or --all / --index / --list")
        jobs = [pd.Series({"site": args.site, "sex": args.sex, "indicator": args.indicator})]

    import jax
    import numpyro
    numpyro.set_host_device_count(args.chains)
    numpyro.enable_x64()
    log(f"jax {jax.__version__}, devices: {jax.devices()}")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for r in jobs:
        try:
            res = fit_one(r.site, r.sex, r.indicator, args)
            if res:
                results.append(res)
        except Exception as e:
            log(f"FAILED {r.site}/{r.sex}/{r.indicator}: {type(e).__name__}: {e}")
    if results:
        idx = OUT / "index.parquet"
        new = pd.json_normalize(results)
        if idx.exists():
            old = pd.read_parquet(idx)
            key = ["site", "sex", "indicator", "is_holdout"]
            old = old.merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)
            old = old[old._merge == "left_only"].drop(columns="_merge")
            new = pd.concat([old, new], ignore_index=True)
        new.to_parquet(idx, index=False)
        log(f"{len(results)} fit(s) written, index: {len(new)} rows")


if __name__ == "__main__":
    main()
