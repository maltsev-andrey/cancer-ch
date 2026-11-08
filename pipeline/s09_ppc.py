#!/usr/bin/env python
"""
s09_ppc: posterior predictive check by age group and period.

Question: is the 90 % posterior predictive interval calibrated uniformly over the grid, or
does the free age vector (18 parameters pinned by 6 cells each) overfit the oldest or the
sparsest cells? Overall coverage was already checked in s06 (holdout); this breaks it down.

For every full fit (no _holdout suffix) and every observed cell:
  predictive draws  = Poisson( rate_grid/1e5 * person_years * exp(N(0,1) * s_eps) )
  inside            = observed cases within the 5th..95th percentile of the draws
  PIT               = share of draws <= observed (uniform on [0,1] if calibrated)
  z                 = (observed - median) / sd of the draws

Outputs
  assets/fig13_ppc.png             coverage by age, by period, by cell size; PIT histogram
  assets/ppc_summary.json
  data/audit/s09_checks.md
  data/models/apc/ppc_cells.parquet   one row per cell, for anyone who wants to dig

Usage
  python pipeline/s09_ppc.py
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
GRID = DATA / "processed" / "apc_incidence_grid.parquet"
ASSETS = REPO / "assets"
AUDIT = DATA / "audit"
REGION = "Schweiz"
PERIODS = ["1993-1997", "1998-2002", "2003-2007", "2008-2012", "2013-2017", "2018-2022"]
AGES = [f"{5 * i}-{5 * i + 4}" for i in range(17)] + ["85+"]
NOMINAL = 0.90
N_DRAWS = 2000            # predictive draws per cell, subsampled from the posterior
SEED = 1

SURFACE, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
IND_COLOR = {"incidence": "#2a78d6", "mortality": "#8e63c9"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False})


def cells_for(grid, site, sex, indicator):
    g = grid[(grid.region == REGION) & (grid.site == site) & (grid.sex == sex)
             & (grid.indicator == indicator) & ~grid.structural_zero & grid.person_years.gt(0)].copy()
    g["a"] = g.age5.map(lambda s: int(re.match(r"\d+", s).group()) // 5)
    g["p"] = g.period.map(PERIODS.index)
    return g


def ppc_one(d, grid, rng):
    s = json.loads((d / "summary.json").read_text())
    if not s.get("converged", True):
        return None
    post = xr.open_dataset(d / "posterior.nc", group="posterior", engine="h5netcdf")
    rate = post.rate_grid.values                       # chain x draw x A x Pf
    rate = rate.reshape(-1, *rate.shape[2:])
    s_eps = post.s_eps.values.ravel()
    take = rng.choice(rate.shape[0], size=min(N_DRAWS, rate.shape[0]), replace=False)
    rate, s_eps = rate[take], s_eps[take][:, None]
    g = cells_for(grid, s["site"], s["sex"], s["indicator"])
    if g.empty:
        return None
    pred_rate = rate[:, g.a.values, g.p.values]                                   # draws x cells
    mu = pred_rate / 1e5 * g.person_years.values * np.exp(rng.normal(0, 1, pred_rate.shape) * s_eps)
    pred = rng.poisson(mu)
    obs = np.rint(g.cases.values).astype(int)
    lo, hi = np.percentile(pred, [5, 95], axis=0)
    med, sd = np.median(pred, axis=0), pred.std(axis=0)
    # randomised PIT for a discrete outcome (Czado, Gneiting & Held 2009)
    below, equal = (pred < obs).mean(0), (pred == obs).mean(0)
    pit = below + rng.uniform(0, 1, len(obs)) * equal
    return pd.DataFrame({
        "site": s["site"], "sex": s["sex"], "indicator": s["indicator"], "dir": d.name,
        "age5": g.age5.values, "a": g.a.values, "period": g.period.values, "p": g.p.values,
        "cases": obs, "cases_per_year": g.cases_per_year.values,
        "inside": (obs >= lo) & (obs <= hi), "pit": pit,
        "z": np.where(sd > 0, (obs - med) / np.where(sd > 0, sd, 1), 0.0),
        "pred_median": med})


def binom_band(n, p=NOMINAL, z=1.96):
    se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
    return p - z * se, p + z * se


def fig_ppc(cells, summary):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    # 1. coverage by age
    ax = axes[0]
    for ind, col in IND_COLOR.items():
        c = cells[cells.indicator == ind]
        if c.empty:
            continue
        by = c.groupby("a").inside.agg(["mean", "size"]).reindex(range(18))
        lo, hi = binom_band(by["size"].values)
        ax.fill_between(range(18), lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(range(18), by["mean"], "o-", color=col, ms=4, label=f"{ind} (n={int(by['size'].sum())} cells)")
    ax.axhline(NOMINAL, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(0, 18, 2), AGES[::2], fontsize=8)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("share of cells inside the 90% predictive interval")
    ax.set_title("Coverage by age group"); ax.legend(fontsize=8, loc="lower left")
    # 2. coverage by period
    ax = axes[1]
    for ind, col in IND_COLOR.items():
        c = cells[cells.indicator == ind]
        if c.empty:
            continue
        by = c.groupby("p").inside.agg(["mean", "size"]).reindex(range(6))
        lo, hi = binom_band(by["size"].values)
        ax.fill_between(range(6), lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(range(6), by["mean"], "o-", color=col, ms=4)
    ax.axhline(NOMINAL, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(6), [p[:4] for p in PERIODS], fontsize=8); ax.set_ylim(0.6, 1.0)
    ax.set_title("By period")
    # 3. coverage by cell size
    ax = axes[2]
    bins = [0, 5, 20, 100, 500, 1e9]; labels = ["<5", "5–20", "20–100", "100–500", ">500"]
    cells["size_bin"] = pd.cut(cells.cases_per_year, bins, labels=labels, right=False)
    for i, (ind, col) in enumerate(IND_COLOR.items()):
        c = cells[cells.indicator == ind]
        if c.empty:
            continue
        by = c.groupby("size_bin", observed=False).inside.agg(["mean", "size"])
        x = np.arange(len(labels)) + (i - 0.5) * 0.3
        ax.bar(x, by["mean"], width=0.28, color=col, alpha=0.8)
        for xi, (m, n) in zip(x, by.itertuples(index=False)):
            if n:
                ax.text(xi, m + 0.01, f"{int(n)}", ha="center", fontsize=7, color=INK2)
    ax.axhline(NOMINAL, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(labels)), labels, fontsize=8); ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("cases per year in the cell"); ax.set_title("By cell size")
    # 4. PIT histogram
    ax = axes[3]
    for ind, col in IND_COLOR.items():
        c = cells[cells.indicator == ind]
        if c.empty:
            continue
        ax.hist(c.pit, bins=10, range=(0, 1), density=True, histtype="step", lw=1.8, color=col)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("PIT = P(predicted ≤ observed)"); ax.set_title("Calibration (PIT)")
    ax.set_ylim(0, 2)
    fig.suptitle("Posterior predictive check: is the 90% interval honest everywhere on the grid?",
                 x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.01, 0.90, f"All converged full fits, {cells.dir.nunique()} fits, {len(cells)} cells. "
             "Dashed line: nominal 90%; shaded: ±2 SE binomial band. A flat PIT histogram means calibrated; "
             "a U shape means too narrow, a hump too wide.", color=INK2, fontsize=9)
    fig.text(0.01, -0.04, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC model, posterior predictive with "
             "extra-Poisson and Poisson noise", color=MUTED, fontsize=8, va="top")
    fig.subplots_adjust(top=0.74, wspace=0.28)
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "fig13_ppc.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  -> assets/fig13_ppc.png")


def main():
    grid = pd.read_parquet(GRID)
    rng = np.random.default_rng(SEED)
    dirs = [d for d in sorted(MOD.iterdir()) if d.is_dir() and "_holdout" not in d.name and "_prior-" not in d.name
            and (d / "posterior.nc").exists()]
    parts = [x for x in (ppc_one(d, grid, rng) for d in dirs) if x is not None]
    if not parts:
        raise SystemExit("no full fits with posterior.nc found")
    cells = pd.concat(parts, ignore_index=True)
    print(f"ppc: {cells.dir.nunique()} fits, {len(cells)} cells")

    summary = {"n_fits": int(cells.dir.nunique()), "n_cells": int(len(cells)), "nominal": NOMINAL}
    for ind in ("incidence", "mortality"):
        c = cells[cells.indicator == ind]
        if c.empty:
            continue
        by_age = c.groupby("age5").inside.mean().reindex(AGES)
        summary[ind] = {
            "coverage_overall": round(float(c.inside.mean()), 3),
            "coverage_by_age": {k: round(float(v), 3) for k, v in by_age.items()},
            "coverage_by_period": {k: round(float(v), 3) for k, v in c.groupby("period").inside.mean().items()},
            "coverage_by_cell_size": {str(k): round(float(v), 3)
                                      for k, v in c.groupby(pd.cut(c.cases_per_year, [0, 5, 20, 100, 500, 1e9],
                                                                   right=False), observed=False).inside.mean().items()},
            "coverage_85plus": round(float(c[c.a == 17].inside.mean()), 3),
            "coverage_under_40": round(float(c[c.a < 8].inside.mean()), 3),
            "pit_ks_distance": round(float(np.max(np.abs(np.sort(c.pit.values) - np.linspace(0, 1, len(c))))), 3),
            "z_sd": round(float(c.z.std()), 3),
            "worst_fits_by_coverage": c.groupby("dir").inside.mean().nsmallest(5).round(2).to_dict()}
    fig_ppc(cells, summary)
    (ASSETS / "ppc_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    cells.drop(columns=["size_bin"], errors="ignore").to_parquet(MOD / "ppc_cells.parquet", index=False)

    AUDIT.mkdir(parents=True, exist_ok=True)
    md = ["# s09 posterior predictive checks", "",
          f"- fits: {summary['n_fits']}, cells: {summary['n_cells']}, nominal interval {NOMINAL:.0%}", ""]
    for ind in ("incidence", "mortality"):
        if ind not in summary:
            continue
        s = summary[ind]
        md += [f"## {ind}", "",
               f"- overall coverage {s['coverage_overall']:.3f}; under 40: {s['coverage_under_40']:.3f}; "
               f"85+: {s['coverage_85plus']:.3f}; PIT KS distance {s['pit_ks_distance']:.3f}; sd of z {s['z_sd']:.2f}",
               "", "| age | coverage |", "|---|---|",
               *[f"| {k} | {v:.3f} |" for k, v in s["coverage_by_age"].items()], "",
               "| period | coverage |", "|---|---|",
               *[f"| {k} | {v:.3f} |" for k, v in s["coverage_by_period"].items()], "",
               "| cases/yr in cell | coverage |", "|---|---|",
               *[f"| {k} | {v:.3f} |" for k, v in s["coverage_by_cell_size"].items()], "",
               "lowest-coverage fits: " + ", ".join(f"{k} ({v:.2f})" for k, v in s["worst_fits_by_coverage"].items()), ""]
    (AUDIT / "s09_checks.md").write_text("\n".join(md))
    for ind in ("incidence", "mortality"):
        if ind in summary:
            s = summary[ind]
            print(f"{ind}: coverage {s['coverage_overall']:.3f}, <40 {s['coverage_under_40']:.3f}, "
                  f"85+ {s['coverage_85plus']:.3f}, PIT KS {s['pit_ks_distance']:.3f}, z sd {s['z_sd']:.2f}")


if __name__ == "__main__":
    main()
