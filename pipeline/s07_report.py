#!/usr/bin/env python3
"""
cancer-ch / Stage 7: sweep report.

Rebuilds models/apc/index.parquet from every summary.json (robust to concurrent writers),
extracts cohort peaks from the posteriors, and writes the sweep figures to assets/:
  fig07_drift_by_site.png        net drift (%/yr) per site, men vs women, incidence
  fig08_cohort_peaks.png         birth cohort with the highest relative risk, per site and sex
  fig09_holdout_krg.png          out-of-sample error 2018-2022 across all fits (registration-act test)
  fig10_lung_cohorts.png         lung cancer cohort curves, men vs women, with the cohort of peak
  fig11_projection_2030.png      projected ASR change 2018-22 -> 2028-32 per site
plus assets/sweep_summary.json and data/audit/s07_checks.md.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    """paths.data from sources.yaml, relative to the repo root unless absolute;
    the environment variable CANCER_CH_DATA overrides both."""
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()
DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
MOD = DATA / "models" / "apc"
ASSETS = REPO / "assets"
AUDIT = DATA / "audit"
PERIODS = ["1993-1997", "1998-2002", "2003-2007", "2008-2012", "2013-2017", "2018-2022"]
A, P = 18, 6
COHORT0 = 1993 - 5 * (A - 1)                     # start year of cohort index 0

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEX_COLOR = {"M": "#2a78d6", "F": "#eb6834"}
SEX_NAME = {"M": "Men", "F": "Women"}
SITE_EN = {"Alle Krebsarten (total)": "All cancers", "Mundhöhle und Rachen": "Oral cavity & pharynx",
           "Speiseröhre": "Oesophagus", "Magen": "Stomach", "Dünndarm": "Small intestine",
           "Dickdarm": "Colorectum", "Anus": "Anus", "Leber": "Liver",
           "Gallenblase und extrahepatische Gallenwege": "Gallbladder", "Bauchspeicheldrüse": "Pancreas",
           "Kehlkopf": "Larynx", "Lunge, Bronchien, Luftröhre": "Lung",
           "Brustfell (Pleuramesotheliom)": "Pleural mesothelioma", "Knochen, Gelenke, Knorpel": "Bone",
           "Hautmelanom": "Skin melanoma", "Bindegewebe": "Soft tissue", "Brust": "Breast",
           "Gebärmutterhals": "Cervix uteri", "Gebärmutterkörper": "Corpus uteri", "Eierstock": "Ovary",
           "Prostata": "Prostate", "Hoden": "Testis", "Niere": "Kidney", "Harnblase": "Bladder",
           "Ableitende Harnwege": "Urinary tract", "Auge": "Eye", "Gehirn und Zentralnervensystem": "Brain & CNS",
           "Schilddrüse": "Thyroid", "Hodgkin-Lymphom": "Hodgkin lymphoma",
           "Non-Hodgkin-Lymphom": "Non-Hodgkin lymphoma", "Multiples Myelom": "Multiple myeloma",
           "Leukämie": "Leukaemia", "Andere Leukämien": "Other leukaemias", "Andere Krebsarten": "Other sites"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 26, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False})


def en(site):
    return SITE_EN.get(site, site)


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def source(fig, text):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8, ha="left", va="top")


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  -> assets/{name}")


# ============================================================ index =========
def rebuild_index():
    rows = []
    for f in sorted(MOD.glob("*/summary.json")):
        d = json.loads(f.read_text())
        d["dir"] = f.parent.name
        rows.append(d)
    idx = pd.json_normalize(rows)
    idx["is_holdout"] = idx["dir"].str.endswith("_holdout")      # robust to older summary.json
    idx["drift_lo"] = idx.net_drift_pct_per_year.str[0]
    idx["drift"] = idx.net_drift_pct_per_year.str[1]
    idx["drift_hi"] = idx.net_drift_pct_per_year.str[2]
    idx["converged"] = (idx.rhat_max < 1.05) & (idx.divergences <= 10)
    idx.to_parquet(MOD / "index.parquet", index=False)
    return idx


def cohort_peak(d):
    """Cohort (start year) with max median RR and its RR, from the posterior; None if not stored."""
    import xarray as xr
    try:
        ds = xr.open_dataset(MOD / d / "posterior.nc", group="posterior", engine="h5netcdf")
    except Exception:
        return None
    g = np.exp(ds["gamma"].median(("chain", "draw")).values)
    lo = np.exp(ds["gamma"].quantile(0.025, ("chain", "draw")).values)
    hi = np.exp(ds["gamma"].quantile(0.975, ("chain", "draw")).values)
    n_obs = A + P - 1                                  # observed cohorts only
    # cells per cohort: 1,2,3,4,5,6,...,6,5,4,3,2,1. Edge cohorts rest on 1-3 cells and the
    # RW2 is nearly unconstrained there, so the peak is searched where >= 4 cells exist.
    n_cells = np.array([min(k + 1, n_obs - k, P) for k in range(n_obs)])
    inner = np.where(n_cells >= 4)[0]
    k = int(inner[np.argmax(g[inner])])
    at_edge = k in (inner[0], inner[-1])
    return {"peak_cohort": COHORT0 + 5 * k, "peak_rr": float(g[k]), "peak_rr_lo": float(lo[k]),
            "peak_rr_hi": float(hi[k]), "peak_at_edge": bool(at_edge),
            "gamma_med": g[:n_obs].tolist(), "gamma_lo": lo[:n_obs].tolist(), "gamma_hi": hi[:n_obs].tolist()}


# ============================================================ figures =======
def fig_drift(full, summary):
    d = full[(full.indicator == "incidence") & (full.cases_per_year >= 100)].copy()
    order = d.groupby("site").drift.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 2))
    for i, site in enumerate(order):
        for sex, dy in (("M", 0.15), ("F", -0.15)):
            r = d[(d.site == site) & (d.sex == sex)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.drift_lo, r.drift_hi], [i + dy] * 2, color=SEX_COLOR[sex], lw=1.5, alpha=0.6)
            ax.plot(r.drift, i + dy, "o", color=SEX_COLOR[sex], ms=6 if r.converged else 4,
                    mfc=SEX_COLOR[sex] if r.converged else SURFACE)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), [en(s) for s in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Net drift, % per year (95% interval)")
    for sex in ("M", "F"):
        ax.plot([], [], "o", color=SEX_COLOR[sex], label=SEX_NAME[sex])
    ax.plot([], [], "o", mfc=SURFACE, color=MUTED, label="not converged")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("How fast incidence is changing, by cancer site")
    subtitle(ax, "Net drift = the identifiable linear trend of period and cohort combined, 1993–2022, sites with ≥100 cases/yr")
    source(fig, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC model per site and sex")
    summary["drift_incidence"] = {en(r.site) + "/" + r.sex: round(r.drift, 2) for r in d.itertuples()}
    save(fig, "fig07_drift_by_site.png")


def fig_cohort_peaks(full, summary):
    d = full[(full.indicator == "incidence") & (full.cases_per_year >= 300) & full.converged
             & full.peak_cohort.notna()].copy()
    d = d[(d.peak_rr_lo > 1.0) & ~d.peak_at_edge.astype(bool)]   # clear interior peak only
    order = d.groupby("site").peak_cohort.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 2))
    for i, site in enumerate(order):
        for sex, dy in (("M", 0.15), ("F", -0.15)):
            r = d[(d.site == site) & (d.sex == sex)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot(r.peak_cohort + 2, i + dy, "o", color=SEX_COLOR[sex], ms=4 + 6 * min(np.log(r.peak_rr), 1))
            ax.text(r.peak_cohort + 5.5, i + dy, f"RR {r.peak_rr:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [en(s) for s in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Birth cohort with the highest risk relative to trend (5-year cohorts)")
    for sex in ("M", "F"):
        ax.plot([], [], "o", color=SEX_COLOR[sex], label=SEX_NAME[sex])
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Which generation carries the highest risk")
    subtitle(ax, "Interior cohorts (≥4 observed cells) with a peak clearly above the linear trend (lower 95% bound of RR > 1); dot size = RR")
    source(fig, "Source: NKRS/BFS px-x-1403030300_161, cohort curvature from the APC model")
    summary["cohort_peaks"] = {en(r.site) + "/" + r.sex: {"cohort": int(r.peak_cohort), "rr": round(r.peak_rr, 2)}
                               for r in d.itertuples()}
    save(fig, "fig08_cohort_peaks.png")


def fig_holdout(hold, summary):
    col = "holdout_check.asr_error_pct"
    if col not in hold:
        hold = hold.assign(**{col: np.nan})
    d = hold[hold.converged & (hold.cases_per_year >= 100) & hold[col].notna()].copy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    for ind, mk in (("incidence", "o"), ("mortality", "s")):
        for sex in ("M", "F"):
            r = d[(d.indicator == ind) & (d.sex == sex)]
            ax[0].plot(r.cases_per_year, r[col], mk, color=SEX_COLOR[sex], ms=5, alpha=0.8,
                       mfc=SEX_COLOR[sex] if ind == "incidence" else SURFACE,
                       label=f"{SEX_NAME[sex]}, {ind}")
    ax[0].axhline(0, color=AXIS, lw=1); ax[0].set_xscale("log")
    ax[0].set_xlabel("Cases per year (log)"); ax[0].set_ylabel("Predicted vs observed ASR 2018–2022, %")
    ax[0].legend(fontsize=8); ax[0].set_ylim(-25, 25)
    ax[0].set_title("Out-of-sample error by fit")
    subtitle(ax[0], "Negative = the 2018–2022 rate came in above the trend fitted on 1993–2017")
    for ind, c in (("incidence", "#1c5cab"), ("mortality", "#86b6ef")):
        v = d[d.indicator == ind][col].clip(-25, 25)
        ax[1].hist(v, bins=np.arange(-25, 26, 2.5), alpha=0.7, color=c, label=f"{ind} (median {v.median():+.1f}%)")
    ax[1].axvline(0, color=AXIS, lw=1); ax[1].set_xlabel("ASR error, %"); ax[1].legend(fontsize=8.5)
    ax[1].set_title("Distribution across sites")
    med_inc = d[d.indicator == "incidence"][col].median(); med_mort = d[d.indicator == "mortality"][col].median()
    subtitle(ax[1], f"Incidence median {med_inc:+.1f}%, mortality median {med_mort:+.1f}%")
    source(fig, "Source: NKRS/BFS px-x-1403030300_161. Fits on 1993–2017, prediction for 2018–2022. "
                "A systematic incidence-only shift would point to the Cancer Registration Act (2020).")
    summary["holdout"] = {"n": int(len(d)), "incidence_median_pct": round(med_inc, 2),
                          "mortality_median_pct": round(med_mort, 2),
                          "incidence_share_negative": round(float((d[d.indicator == "incidence"][col] < 0).mean()), 2)}
    save(fig, "fig09_holdout_krg.png")


def fig_lung(full, summary):
    fig, ax = plt.subplots(figsize=(10, 5))
    yrs = [COHORT0 + 5 * k + 2 for k in range(A + P - 1)]
    for sex in ("M", "F"):
        r = full[(full.site == "Lunge, Bronchien, Luftröhre") & (full.sex == sex) & (full.indicator == "incidence")]
        if r.empty or r.iloc[0].gamma_med is None:
            continue
        r = r.iloc[0]
        ax.fill_between(yrs, r.gamma_lo, r.gamma_hi, color=SEX_COLOR[sex], alpha=0.15)
        ax.plot(yrs, r.gamma_med, color=SEX_COLOR[sex], lw=2, label=f"{SEX_NAME[sex]}: peak cohort {int(r.peak_cohort)}–{int(r.peak_cohort) + 4}, RR {r.peak_rr:.2f}, drift {r.drift:+.1f}%/yr")
        ax.plot(r.peak_cohort + 2, r.peak_rr, "o", color=SEX_COLOR[sex], ms=8, mec=SURFACE, mew=2)
    ax.axhline(1, color=AXIS, lw=1); ax.set_ylabel("Relative risk vs linear trend"); ax.set_xlabel("Birth cohort")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Lung cancer: the smoking generations, men and women")
    subtitle(ax, "Cohort curvature from the APC model, incidence 1993\u20132022, 95% interval")
    source(fig, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC model, incidence 1993–2022")
    save(fig, "fig10_lung_cohorts.png")


def fig_projection(full, summary):
    d = full[(full.indicator == "incidence") & (full.cases_per_year >= 100) & full.converged].copy()
    d = d[d["projection.2028-2032"].notna()]
    d["asr_now"] = d["asr_fitted_by_period.2018-2022"]
    d["asr_2030"] = d["projection.2028-2032"].str[1]
    d["lo"] = d["projection.2028-2032"].str[0]; d["hi"] = d["projection.2028-2032"].str[2]
    d["chg"] = (d.asr_2030 / d.asr_now - 1) * 100
    d["chg_lo"] = (d.lo / d.asr_now - 1) * 100; d["chg_hi"] = (d.hi / d.asr_now - 1) * 100
    order = d.groupby("site").chg.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 2))
    for i, site in enumerate(order):
        for sex, dy in (("M", 0.15), ("F", -0.15)):
            r = d[(d.site == site) & (d.sex == sex)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.chg_lo, r.chg_hi], [i + dy] * 2, color=SEX_COLOR[sex], lw=1.5, alpha=0.6)
            ax.plot(r.chg, i + dy, "o", color=SEX_COLOR[sex], ms=6)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), [en(s) for s in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Projected change in ASR, 2018–2022 → 2028–2032, % (95% interval)")
    for sex in ("M", "F"):
        ax.plot([], [], "o", color=SEX_COLOR[sex], label=SEX_NAME[sex])
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Where incidence is heading by 2030")
    subtitle(ax, "Extrapolation of the fitted drift and cohort/period curvature; intervals widen with the horizon")
    source(fig, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC projection, rates per 100,000 (Europe 1976)")
    summary["projection_2028_32_pct"] = {en(r.site) + "/" + r.sex: round(r.chg, 1) for r in d.itertuples()}
    save(fig, "fig11_projection_2030.png")


# ============================================================ main ==========
def main():
    idx = rebuild_index()
    print(f"index: {len(idx)} fits, converged {int(idx.converged.sum())}")
    peaks = []
    for r in idx.itertuples():
        pk = cohort_peak(r.dir) if not r.is_holdout else None
        peaks.append(pk or {})
    idx = pd.concat([idx.reset_index(drop=True), pd.DataFrame(peaks)], axis=1)
    full, hold = idx[~idx.is_holdout], idx[idx.is_holdout]

    summary = {"n_fits": int(len(idx)), "n_full": int(len(full)), "n_holdout": int(len(hold)),
               "converged_full": int(full.converged.sum()), "converged_holdout": int(hold.converged.sum())}
    for f in (fig_drift, fig_cohort_peaks, fig_holdout, fig_lung, fig_projection):
        print(f.__name__)
        try:
            f(full if f is not fig_holdout else hold, summary)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    (ASSETS / "sweep_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    bad = idx[~idx.converged][["site", "sex", "indicator", "is_holdout", "rhat_max", "divergences"]]
    md = [f"# s07 checks", "", f"- fits: {len(idx)} (full {len(full)}, holdout {len(hold)})",
          f"- converged (rhat<1.05, div<=10): {int(idx.converged.sum())}",
          f"- not converged: {len(bad)}", bad.to_markdown(index=False) if len(bad) else "",
          f"- holdout: {summary.get('holdout')}"]
    AUDIT.mkdir(exist_ok=True); (AUDIT / "s07_checks.md").write_text("\n".join(md))
    print("\n".join(md[2:5]))
    if len(bad):
        rerun = [i for i, r in enumerate(idx.itertuples()) if not r.converged]
        print("rerun indices (fit list order may differ, use --site/--sex):")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
