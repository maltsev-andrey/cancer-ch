#!/usr/bin/env python
"""
s10_prior_sensitivity: do the headline estimates move when the priors move?

Compares, for every site x sex x indicator that has prior-variant fits (dirs <slug>_prior-<tag>,
written by s06 --tag with --drift-sd/--rw-sd/--eps-sd), the baseline fit against each variant on
  - net drift, % per year, median and 95% interval
  - peak cohort (interior cohorts, >= 4 observed cells) and its RR versus trend
  - ASR projected for 2028-2032 if the fit has one

Pass criterion (per variant): the variant's drift median lies inside the baseline 95% interval
and the peak cohort is the same or adjacent (+-5 years).

Variants used in v1.2 (run_prior_sensitivity.sh):
  tight   drift_sd 0.15, rw_sd 0.15, eps_sd 0.03     (x0.3)
  wide    drift_sd 1.5,  rw_sd 1.5,  eps_sd 0.3      (x3)
  curvy   drift_sd 0.15, rw_sd 1.5,  eps_sd 0.1      (push variation from drift into curvature)

Outputs
  assets/fig14_prior_sensitivity.png
  assets/prior_sensitivity.json
  data/audit/s10_checks.md
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()


DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
MOD = DATA / "models" / "apc"
ASSETS = REPO / "assets"
AUDIT = DATA / "audit"
A, P = 18, 6
C = A + P - 1
COHORT_START = 1993 - 5 * (A - 1)
SITE_EN = {"Alle Krebsarten (total)": "All cancers", "Lunge, Bronchien, Luftröhre": "Lung",
           "Hautmelanom": "Skin melanoma", "Prostata": "Prostate", "Brust": "Breast",
           "Dickdarm": "Colorectum", "Schilddrüse": "Thyroid", "Bauchspeicheldrüse": "Pancreas"}
SURFACE, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VARIANT_COLOR = {"baseline": INK, "tight": "#2a78d6", "wide": "#eb6834", "curvy": "#8e63c9"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False})


def cohort_peak(d):
    """Interior cohort (>=4 observed cells, index 3..C-4) with the highest median RR vs trend."""
    ds = xr.open_dataset(d / "posterior.nc", group="posterior", engine="h5netcdf")
    g = np.exp(ds["gamma"].median(("chain", "draw")).values)[:C]
    lo = np.exp(ds["gamma"].quantile(0.025, ("chain", "draw")).values)[:C]
    interior = np.arange(3, C - 3)
    k = interior[np.argmax(g[interior])]
    return {"cohort": int(COHORT_START + 5 * k), "rr": round(float(g[k]), 3), "rr_lo": round(float(lo[k]), 3)}


def load():
    rows = []
    for d in sorted(MOD.glob("*_prior-*")):
        m = re.match(r"(.+?)_prior-([a-z0-9]+)$", d.name)
        if not m or "_holdout" in d.name:
            continue
        base = MOD / m.group(1)
        if not (base / "summary.json").exists():
            continue
        for tag, dd in (("baseline", base), (m.group(2), d)):
            s = json.loads((dd / "summary.json").read_text())
            pk = cohort_peak(dd)
            proj = s.get("projection", {}).get("2028-2032")
            rows.append({"slug": m.group(1), "site": s["site"], "sex": s["sex"], "indicator": s["indicator"],
                         "variant": tag, "converged": bool(s.get("converged", True)),
                         "drift_lo": s["net_drift_pct_per_year"][0], "drift": s["net_drift_pct_per_year"][1],
                         "drift_hi": s["net_drift_pct_per_year"][2],
                         "peak_cohort": pk["cohort"], "peak_rr": pk["rr"], "peak_rr_lo": pk["rr_lo"],
                         "asr_2030": proj[1] if proj else np.nan,
                         "priors": json.dumps(s.get("priors", {}))})
    df = pd.DataFrame(rows).drop_duplicates(["slug", "variant"])
    return df


def evaluate(df):
    out = []
    for slug, g in df.groupby("slug"):
        b = g[g.variant == "baseline"]
        if b.empty:
            continue
        b = b.iloc[0]
        for r in g[g.variant != "baseline"].itertuples():
            drift_ok = b.drift_lo <= r.drift <= b.drift_hi
            cohort_ok = abs(r.peak_cohort - b.peak_cohort) <= 5
            out.append({"slug": slug, "site": SITE_EN.get(b.site, b.site), "sex": b.sex, "indicator": b.indicator,
                        "variant": r.variant, "converged": r.converged,
                        "drift_base": b.drift, "drift_var": r.drift, "drift_shift": r.drift - b.drift,
                        "drift_inside_base_ci": bool(drift_ok),
                        "cohort_base": b.peak_cohort, "cohort_var": r.peak_cohort, "cohort_ok": bool(cohort_ok),
                        "rr_base": b.peak_rr, "rr_var": r.peak_rr,
                        "asr2030_base": b.asr_2030, "asr2030_var": r.asr_2030,
                        "pass": bool(drift_ok and cohort_ok and r.converged)})
    return pd.DataFrame(out)


def fig(df, ev):
    labels = [f"{SITE_EN.get(r.site, r.site)} {r.sex} {r.indicator[:3]}" for r in
              df[df.variant == "baseline"].sort_values("slug").itertuples()]
    slugs = sorted(df.slug.unique())
    variants = ["baseline"] + [v for v in ("tight", "wide", "curvy") if v in set(df.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(slugs) + 2.5))
    off = np.linspace(-0.27, 0.27, len(variants))
    for j, v in enumerate(variants):
        for i, slug in enumerate(slugs):
            r = df[(df.slug == slug) & (df.variant == v)]
            if r.empty:
                continue
            r = r.iloc[0]; y = i + off[j]; col = VARIANT_COLOR.get(v, MUTED)
            axes[0].plot([r.drift_lo, r.drift_hi], [y, y], color=col, lw=1.5 if v == "baseline" else 1, alpha=0.9)
            axes[0].plot(r.drift, y, "o", color=col, ms=5 if v == "baseline" else 4, label=v if i == 0 else None)
            axes[1].plot(r.peak_cohort, y, "o", color=col, ms=4 + 6 * (r.peak_rr - 1), label=v if i == 0 else None)
    for ax in axes:
        ax.set_yticks(range(len(slugs)), labels, fontsize=9); ax.invert_yaxis()
    axes[0].axvline(0, color=INK2, lw=1); axes[0].set_xlabel("net drift, % per year (95% interval)")
    axes[0].set_title("Net drift under each prior"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("peak birth cohort (dot size = RR vs trend)"); axes[1].set_title("Peak cohort under each prior")
    n_pass, n = int(ev["pass"].sum()), len(ev)
    fig.suptitle("Prior sensitivity: tight (x0.3), wide (x3) and curvature-favouring priors vs baseline",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, 0.91, f"{n_pass} of {n} variant fits pass: drift median inside the baseline 95% interval and the "
             "same or adjacent peak cohort.", color=INK2, fontsize=9)
    fig.text(0.01, -0.05, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC model, s06 --drift-sd/--rw-sd/--eps-sd",
             color=MUTED, fontsize=8, va="top")
    fig.subplots_adjust(top=0.80, wspace=0.45)
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "fig14_prior_sensitivity.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  -> assets/fig14_prior_sensitivity.png")


def main():
    df = load()
    if df.empty:
        raise SystemExit("no prior-variant fits found; run scripts/run_prior_sensitivity.sh first")
    ev = evaluate(df)
    print(f"prior sensitivity: {df.slug.nunique()} fits x {df.variant.nunique() - 1} variants")
    fig(df, ev)
    summary = {"n_variant_fits": int(len(ev)), "n_pass": int(ev["pass"].sum()),
               "max_abs_drift_shift_pct_per_year": round(float(ev.drift_shift.abs().max()), 3),
               "by_variant": {v: {"n": int(len(g)), "pass": int(g["pass"].sum()),
                                  "median_abs_drift_shift": round(float(g.drift_shift.abs().median()), 3),
                                  "cohort_changed": int((~g.cohort_ok).sum())}
                              for v, g in ev.groupby("variant")},
               "fits": ev.drop(columns=["slug"]).round(3).to_dict(orient="records")}
    (ASSETS / "prior_sensitivity.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    AUDIT.mkdir(parents=True, exist_ok=True)
    cols = ["site", "sex", "indicator", "variant", "drift_base", "drift_var", "drift_inside_base_ci",
            "cohort_base", "cohort_var", "rr_base", "rr_var", "asr2030_base", "asr2030_var", "converged", "pass"]
    md = ["# s10 prior sensitivity", "", f"- {summary['n_pass']} of {summary['n_variant_fits']} variant fits pass",
          f"- largest drift shift: {summary['max_abs_drift_shift_pct_per_year']} %/yr", "",
          ev[cols].round(2).to_markdown(index=False), ""]
    (AUDIT / "s10_checks.md").write_text("\n".join(md))
    print(ev[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
