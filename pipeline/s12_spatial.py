#!/usr/bin/env python
"""
s12_spatial: BYM2 spatial model of mortality by canton (Riebler et al. 2016).

For every sex x 5-year period the observed deaths in canton c are modelled against the deaths
expected from the national age-specific rates of that sex and period (indirect
standardisation, internal standard, so the relative risks average to 1):

    y_c ~ Poisson( E_c * exp(alpha + phi_c) )
    phi = sigma * ( sqrt(rho) * u / sqrt(s)  +  sqrt(1 - rho) * v )
    u   ~ ICAR on the canton graph (pairwise differences on the 57 edges, sum-to-zero)
    v   ~ N(0, 1) iid
    s   = ICAR scaling factor from s11 (generalised variance of u is 1 after scaling)
    rho ~ Beta(1, 1): share of the variation that is spatially structured
    sigma ~ HalfNormal(0.5)

Outputs per fit: RR_c = exp(phi_c) with 95% interval, P(RR_c > 1), rho, sigma, diagnostics.

Data:  --indicator allcause   (default) deaths from px-x-0102020000_104_deaths (BEVNAT, open)
       --indicator <cause>    a cantonal cause-of-death table once BFS delivers it (s13, later)
Population: 31 Dec stocks from px-x-0102020000_104 as in s04 (mid-year = mean of y-1 and y).
Both deaths and stocks in this cube are tabulated by age reached on 31 December, so the
convention cancels in the rate.

Outputs
  data/models/spatial/<indicator>__<sex>__<period>/summary.json
  data/models/spatial/rr.parquet                 canton x sex x period: SMR, RR, interval, P(RR>1)
  assets/fig16_cantonal_rr_<indicator>.png       maps of RR, small multiples by period, per sex
  assets/fig17_spatial_params_<indicator>.png    rho and sigma over time
  data/audit/s12_checks.md

Usage
  python pipeline/s12_spatial.py --synthetic           # recover a known field on the real graph
  python pipeline/s12_spatial.py                       # all-cause, both sexes, 1990-2024
  python pipeline/s12_spatial.py --sex F --period 2020-2024
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

REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()


DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
PX = DATA / "interim" / "pxweb"
GEO = DATA / "processed" / "geo"
OUT = DATA / "models" / "spatial"
AUDIT = DATA / "audit"
ASSETS = REPO / "assets"

PERIODS = [f"{y}-{y + 4}" for y in range(1990, 2024, 5)]            # 1990-1994 .. 2020-2024
SEX = {"Mann": "M", "Frau": "F"}
ESP_AGES = [f"{5 * i}-{5 * i + 4}" for i in range(17)] + ["85+"]

SURFACE, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEX_COLOR = {"M": "#2a78d6", "F": "#eb6834"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def age5(label):
    m = re.match(r"^(\d+)", str(label))
    if not m:
        return None
    a = int(m.group(1))
    return "85+" if a >= 85 else f"{a // 5 * 5}-{a // 5 * 5 + 4}"


CANTON_ALIASES = {"Appenzell A.Rh.": "Appenzell Ausserrhoden", "Appenzell I.Rh.": "Appenzell Innerrhoden",
                  "Appenzell A. Rh.": "Appenzell Ausserrhoden", "Appenzell I. Rh.": "Appenzell Innerrhoden"}


def canton_key(name):
    """'Bern / Berne' -> 'Bern'; BFS abbreviations of the two Appenzell -> full names."""
    k = str(name).split(" / ")[0].strip()
    return CANTON_ALIASES.get(k, k)
def period_of(year):
    y = int(year)
    for per in PERIODS:
        a, b = map(int, per.split("-"))
        if a <= y <= b:
            return per
    return None


# ---------------------------------------------------------------- data --------
def load_graph():
    g = json.loads((GEO / "nodes_edges.json").read_text())
    cant = pd.read_csv(GEO / "cantons.csv")
    return cant, np.array(g["edges"], dtype=int), float(g["icar_scaling_factor"])


def load_cells(indicator):
    """Deaths and person-years by canton x sex x 5-y age x period."""
    cant, _, _ = load_graph()
    names = set(cant.name_de)
    if indicator == "allcause":
        d = pd.read_parquet(PX / "px-x-0102020000_104_deaths.parquet")
        d = d[d.Geschlecht.isin(SEX)].assign(sex=lambda x: x.Geschlecht.map(SEX),
                                             canton=lambda x: x.Kanton.map(canton_key),
                                             age5=lambda x: x.Alter.map(age5),
                                             year=lambda x: x.Jahr.astype(int))
        d = d[d.canton.isin(names) & d.age5.notna()]
        deaths = d.groupby(["canton", "sex", "age5", "year"], observed=True).value.sum().rename("deaths")
    else:
        raise SystemExit(f"indicator {indicator}: cantonal cause-of-death tables are not ingested yet (see docs/bfs_request)")
    p = pd.read_parquet(PX / "px-x-0102020000_104.parquet")
    p = p[p.Geschlecht.isin(SEX) & (p["Demografische Komponente"] == "Bestand am 31. Dezember")]
    p = p.assign(sex=lambda x: x.Geschlecht.map(SEX), canton=lambda x: x.Kanton.map(canton_key),
                 age5=lambda x: x.Alter.map(age5), year=lambda x: x.Jahr.astype(int))
    p = p[p.canton.isin(names) & p.age5.notna()]
    dec31 = p.groupby(["canton", "sex", "age5", "year"], observed=True).value.sum().rename("dec31")
    df = pd.concat([deaths, dec31], axis=1).reset_index().sort_values(["canton", "sex", "age5", "year"])
    df["dec31_prev"] = df.groupby(["canton", "sex", "age5"], observed=True).dec31.shift(1)
    df = df[df.dec31_prev.notna() & df.deaths.notna()]
    df["py"] = (df.dec31 + df.dec31_prev) / 2
    df["period"] = df.year.map(period_of)
    df = df[df.period.notna()]
    cells = df.groupby(["canton", "sex", "age5", "period"], observed=True)[["deaths", "py"]].sum().reset_index()
    return cells


def expected(cells, sex, period):
    """Indirectly standardised expected deaths per canton, internal standard."""
    c = cells[(cells.sex == sex) & (cells.period == period)]
    nat = c.groupby("age5")[["deaths", "py"]].sum()
    rate = (nat.deaths / nat.py).rename("rate")
    c = c.join(rate, on="age5")
    out = c.assign(E=c.py * c.rate).groupby("canton")[["deaths", "E", "py"]].sum()
    cant, _, _ = load_graph()
    missing = set(cant.name_de) - set(out.index)
    if missing:
        raise SystemExit(f"{sex}/{period}: no data for cantons {sorted(missing)}; check canton labels")
    return out.reindex(cant.name_de)


# ---------------------------------------------------------------- model -------
def build_model(edges, scale):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    ei, ej = jnp.array(edges[:, 0]), jnp.array(edges[:, 1])

    def model(E, y, n):
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
        rho = numpyro.sample("rho", dist.Beta(1.0, 1.0))
        v = numpyro.sample("v", dist.Normal(0.0, 1.0).expand([n]))
        u = numpyro.sample("u_raw", dist.ImproperUniform(dist.constraints.real, (), (n,)))
        # ICAR: pairwise differences on the edges, soft sum-to-zero
        numpyro.factor("icar", -0.5 * jnp.sum((u[ei] - u[ej]) ** 2))
        numpyro.factor("icar_sum", dist.Normal(0.0, 0.001 * n).log_prob(jnp.sum(u)))
        phi = sigma * (jnp.sqrt(rho / scale) * u + jnp.sqrt(1.0 - rho) * v)
        numpyro.deterministic("phi", phi)
        numpyro.deterministic("rr", jnp.exp(phi))
        numpyro.deterministic("structured_share", rho)
        numpyro.sample("y", dist.Poisson(E * jnp.exp(alpha + phi)), obs=y)

    return model


def fit(E, y, edges, scale, args, seed=1):
    import jax
    import arviz as az
    from numpyro.infer import MCMC, NUTS, init_to_median
    n = len(E)
    kernel = NUTS(build_model(edges, scale), target_accept_prob=args.target_accept,
                  init_strategy=init_to_median(num_samples=50))
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains,
                chain_method="parallel", progress_bar=False)
    t0 = datetime.now()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("energy", "diverging"),
             E=jax.numpy.array(E, dtype=float), y=jax.numpy.array(y, dtype=int), n=n)
    secs = (datetime.now() - t0).total_seconds()
    idata = az.from_numpyro(mcmc)
    post = idata.posterior
    div = int(idata.sample_stats.diverging.sum())
    rh = az.rhat(idata, var_names=["alpha", "sigma", "rho", "phi"])
    rhat = max(float(np.nanmax(np.asarray(rh[v]))) for v in rh.data_vars)
    ess = az.ess(idata, var_names=["alpha", "sigma", "rho", "phi"], method="bulk")
    ess_min = min(float(np.nanmin(np.asarray(ess[v]))) for v in ess.data_vars)
    rr = post.rr.values.reshape(-1, n)
    return {"rr_med": np.median(rr, 0), "rr_lo": np.percentile(rr, 2.5, 0), "rr_hi": np.percentile(rr, 97.5, 0),
            "p_above_1": (rr > 1).mean(0), "rho": np.percentile(post.rho.values.ravel(), [2.5, 50, 97.5]),
            "sigma": np.percentile(post.sigma.values.ravel(), [2.5, 50, 97.5]),
            "div": div, "rhat": rhat, "ess_min": ess_min, "seconds": secs, "post": post}


# ---------------------------------------------------------------- synthetic ---
def synthetic(args):
    import jax
    rng = np.random.default_rng(7)
    cant, edges, scale = load_graph()
    n = len(cant)
    # draw a structured field from the ICAR prior (pseudo-inverse) and an iid one
    Q = np.zeros((n, n))
    for a, b in edges:
        Q[a, a] += 1; Q[b, b] += 1; Q[a, b] -= 1; Q[b, a] -= 1
    cov = np.linalg.pinv(Q)
    u = rng.multivariate_normal(np.zeros(n), cov); u -= u.mean()
    v = rng.standard_normal(n)
    rho_true, sigma_true = 0.7, 0.25
    phi = sigma_true * (np.sqrt(rho_true / scale) * u + np.sqrt(1 - rho_true) * v)
    cells = load_cells("allcause")
    E = expected(cells, "M", "2015-2019").E.values            # realistic expected counts
    y = rng.poisson(E * np.exp(phi))
    r = fit(E, y, edges, scale, args)
    inside = np.mean((np.exp(phi) >= r["rr_lo"]) & (np.exp(phi) <= r["rr_hi"]))
    corr = np.corrcoef(np.log(r["rr_med"]), phi)[0, 1]
    log(f"synthetic: rho true {rho_true} est {r['rho'][1]:.2f} ({r['rho'][0]:.2f}..{r['rho'][2]:.2f}); "
        f"sigma true {sigma_true} est {r['sigma'][1]:.3f}; RR 95% coverage {inside:.2f}; "
        f"corr(log RR_hat, phi) {corr:.3f}; div {r['div']}, rhat {r['rhat']:.3f}, ess {r['ess_min']:.0f}, {r['seconds']:.0f}s")
    ok = r["rho"][0] <= rho_true <= r["rho"][2] and inside >= 0.85 and corr > 0.9 and r["div"] <= 10
    print("SYNTHETIC", "OK" if ok else "CHECK")
    return ok


# ---------------------------------------------------------------- figures -----
def fig_maps(rr, indicator, cant):
    try:
        import geopandas as gpd
    except ImportError:
        log("geopandas not available: maps skipped"); return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    g = gpd.read_file(GEO / "cantons_2056.geojson")
    periods = sorted(rr.period.unique())
    for sex in ("M", "F"):
        d = rr[rr.sex == sex]
        if d.empty:
            continue
        ncol = min(4, len(periods)); nrow = int(np.ceil(len(periods) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow + 1.0), facecolor=SURFACE)
        axes = np.atleast_1d(axes).ravel()
        vmax = max(float(np.max(np.abs(np.log(d.rr_med)))), 0.1)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        for ax, per in zip(axes, periods):
            m = g.drop(columns=["abbr"]).merge(d[d.period == per], left_on="name_de", right_on="canton", how="left")
            m["logrr"] = np.log(m.rr_med)
            m.plot(column="logrr", cmap="RdBu_r", norm=norm, linewidth=0.4, edgecolor=INK2, ax=ax)
            sig = m[(m.p_above_1 > 0.95) | (m.p_above_1 < 0.05)]
            for r in sig.itertuples():
                c = r.geometry.representative_point()
                ax.annotate(r.abbr, (c.x, c.y), ha="center", va="center", fontsize=6.5, color=INK,
                            fontweight="bold")
            ax.set_axis_off(); ax.set_title(per, fontsize=10, loc="left", pad=2)
        for ax in axes[len(periods):]:
            ax.set_axis_off()
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.14, wspace=0.02, hspace=0.12)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm); sm.set_array(np.linspace(-vmax, vmax, 10))
        cax = fig.add_axes([0.30, 0.075, 0.40, 0.02])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        ticks = [t for t in (0.8, 0.9, 1.0, 1.1, 1.25) if abs(np.log(t)) <= vmax]
        cb.set_ticks(np.log(ticks)); cb.set_ticklabels([f"{t:.2f}" for t in ticks])
        cb.set_label("relative risk vs Switzerland, same sex and period", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.suptitle(f"{'Men' if sex == 'M' else 'Women'}: {indicator} mortality by canton, BYM2 relative risk "
                     f"after age standardisation", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=13)
        fig.text(0.01, 0.925, "Label: posterior probability of RR above or below 1 exceeds 95%. Internal standard: "
                 "RR averages to 1 within each sex and period.", color=INK2, fontsize=9)
        fig.text(0.01, 0.005, "Source: BFS px-x-0102020000_104 (BEVNAT), swisstopo swissBOUNDARIES3D; BYM2 in NumPyro",
                 color=MUTED, fontsize=8)
        ASSETS.mkdir(exist_ok=True)
        fig.savefig(ASSETS / f"fig16_cantonal_rr_{indicator}_{sex}.png", dpi=180, facecolor=SURFACE); plt.close(fig)
        log(f"  -> assets/fig16_cantonal_rr_{indicator}_{sex}.png")


def fig_params(params, indicator):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for sex in ("M", "F"):
        d = params[params.sex == sex].sort_values("period")
        if d.empty:
            continue
        x = np.arange(len(d))
        axes[0].errorbar(x, d.rho, yerr=[d.rho - d.rho_lo, d.rho_hi - d.rho], fmt="o-", color=SEX_COLOR[sex], label=sex, capsize=2)
        axes[1].errorbar(x, d.sigma, yerr=[d.sigma - d.sigma_lo, d.sigma_hi - d.sigma], fmt="o-", color=SEX_COLOR[sex], label=sex, capsize=2)
        for ax in axes:
            ax.set_xticks(x, [p[:4] for p in d.period], fontsize=8)
    axes[0].set_ylim(0, 1); axes[0].set_title("rho: share of between-canton variation that is spatial", loc="left", fontsize=10)
    axes[1].set_title("sigma: total between-canton SD of log RR", loc="left", fontsize=10)
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(color=GRID_C, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"BYM2 hyperparameters over time, {indicator} mortality", x=0.01, ha="left", fontweight="bold")
    fig.savefig(ASSETS / f"fig17_spatial_params_{indicator}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    log(f"  -> assets/fig17_spatial_params_{indicator}.png")


# ---------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicator", default="allcause")
    ap.add_argument("--sex", choices=["M", "F"]); ap.add_argument("--period")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--maps-only", dest="maps_only", action="store_true", help="redraw figures from rr.parquet, no fitting")
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--target-accept", dest="target_accept", type=float, default=0.95)
    args = ap.parse_args()
    import numpyro
    numpyro.set_host_device_count(args.chains); numpyro.enable_x64()
    if args.synthetic:
        sys.exit(0 if synthetic(args) else 1)
    if args.maps_only:
        cant, _, _ = load_graph()
        rr = pd.read_parquet(OUT / "rr.parquet"); pr = pd.read_parquet(OUT / f"params_{args.indicator}.parquet")
        fig_maps(rr[rr.indicator == args.indicator], args.indicator, cant); fig_params(pr, args.indicator)
        return

    cant, edges, scale = load_graph()
    cells = load_cells(args.indicator)
    periods = [args.period] if args.period else [p for p in PERIODS if p in set(cells.period)]
    sexes = [args.sex] if args.sex else ["M", "F"]
    log(f"{args.indicator}: {len(cells):,} cells, periods {periods[0]}..{periods[-1]}, graph {len(cant)} nodes / {len(edges)} edges, scale {scale:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, params = [], []
    for sex in sexes:
        for per in periods:
            ex = expected(cells, sex, per)
            r = fit(ex.E.values, ex.deaths.values, edges, scale, args)
            flag = "OK" if r["div"] <= 10 and r["rhat"] < 1.01 and r["ess_min"] >= 400 else "CHECK"
            log(f"{flag} {args.indicator}/{sex}/{per}: {r['seconds']:.0f}s, div={r['div']}, rhat={r['rhat']:.3f}, "
                f"ess={r['ess_min']:.0f}, rho={r['rho'][1]:.2f} ({r['rho'][0]:.2f}..{r['rho'][2]:.2f}), sigma={r['sigma'][1]:.3f}, "
                f"RR range {r['rr_med'].min():.2f}..{r['rr_med'].max():.2f}")
            for i, c in enumerate(cant.itertuples()):
                rows.append({"indicator": args.indicator, "sex": sex, "period": per, "canton": c.name_de, "abbr": c.abbr,
                             "deaths": float(ex.deaths.values[i]), "expected": float(ex.E.values[i]),
                             "smr": float(ex.deaths.values[i] / ex.E.values[i]),
                             "rr_med": float(r["rr_med"][i]), "rr_lo": float(r["rr_lo"][i]), "rr_hi": float(r["rr_hi"][i]),
                             "p_above_1": float(r["p_above_1"][i])})
            params.append({"indicator": args.indicator, "sex": sex, "period": per,
                           "rho": r["rho"][1], "rho_lo": r["rho"][0], "rho_hi": r["rho"][2],
                           "sigma": r["sigma"][1], "sigma_lo": r["sigma"][0], "sigma_hi": r["sigma"][2],
                           "divergences": r["div"], "rhat_max": r["rhat"], "ess_min": r["ess_min"],
                           "converged": flag == "OK", "seconds": r["seconds"]})
            d = OUT / f"{args.indicator}__{sex}__{per}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps({**params[-1], "rr": {c.abbr: float(r["rr_med"][i]) for i, c in enumerate(cant.itertuples())}}, indent=1))
    rr = pd.DataFrame(rows); pr = pd.DataFrame(params)
    rr_path = OUT / "rr.parquet"
    if rr_path.exists() and (args.sex or args.period):
        old = pd.read_parquet(rr_path)
        key = ["indicator", "sex", "period"]
        old = old.merge(rr[key].drop_duplicates(), on=key, how="left", indicator=True)
        rr = pd.concat([old[old._merge == "left_only"].drop(columns="_merge"), rr], ignore_index=True)
    rr.to_parquet(rr_path, index=False)
    pr.to_parquet(OUT / f"params_{args.indicator}.parquet", index=False)
    fig_maps(rr[rr.indicator == args.indicator], args.indicator, cant)
    fig_params(pr, args.indicator)

    AUDIT.mkdir(parents=True, exist_ok=True)
    last = rr[(rr.period == periods[-1])]
    md = ["# s12 spatial checks", "", f"- indicator: {args.indicator}; fits: {len(pr)}, converged: {int(pr.converged.sum())}",
          f"- SMR mean over cantons (should be ~1): {rr.groupby(['sex', 'period']).apply(lambda x: (x.deaths.sum() / x.expected.sum())).round(4).to_dict()}",
          "", "## hyperparameters", "", pr[["sex", "period", "rho", "rho_lo", "rho_hi", "sigma", "divergences", "rhat_max", "ess_min"]].round(3).to_markdown(index=False),
          "", f"## relative risk, {periods[-1]}", "",
          last.sort_values(["sex", "rr_med"])[["sex", "abbr", "deaths", "expected", "smr", "rr_med", "rr_lo", "rr_hi", "p_above_1"]].round(3).to_markdown(index=False), ""]
    (AUDIT / "s12_checks.md").write_text("\n".join(md))
    log(f"written: {rr_path}, audit/s12_checks.md")


if __name__ == "__main__":
    main()
