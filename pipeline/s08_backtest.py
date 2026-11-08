#!/usr/bin/env python
"""
s08_backtest: rolling-origin backtest report.

Reads every backtest fit written by s06 --train-until (directories <slug>_holdout2007,
_holdout2012 and _holdout) and scores the age-standardised rate predicted for each held-out
period against the observed one, by origin (last training period) and horizon (periods ahead).

Question it answers: is the +2.5 % / +2.9 % shift seen when predicting 2018-2022 from
1993-2017 a property of the model (it would show at every origin) or of those years
(it would show only at origin 2017)?

Outputs
  assets/fig12_backtest.png            error distribution by origin x horizon, incidence and mortality
  assets/backtest_summary.json         medians, MAD, coverage per origin x horizon
  data/audit/s08_checks.md

Usage
  python pipeline/s08_backtest.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
IND_COLOR = {"incidence": "#2a78d6", "mortality": "#8e63c9"}
MIN_CASES = 100          # same floor as fig09: below this the 5-year count noise dominates
MIN_FITS_PER_CELL = 3    # fewer than this and a box is not a distribution

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 26, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False})


def load_backtests():
    rows = []
    for f in sorted(MOD.glob("*_holdout*/summary.json")):
        if "_prior-" in f.parent.name:
            continue
        d = json.loads(f.read_text())
        if "backtest" not in d:                       # pre-backtest holdout: single horizon
            if "holdout_check" in d:
                d["backtest"] = {d["holdout_check"]["period"]: {**d["holdout_check"], "horizon": 1}}
            else:
                continue
        conv = d.get("converged", d.get("rhat_max", 9) < 1.05 and d.get("divergences", 99) <= 10)
        for per, c in d["backtest"].items():
            rows.append({"site": d["site"], "sex": d["sex"], "indicator": d["indicator"],
                         "train_until": d.get("train_until", 2017), "period": per,
                         "horizon": c["horizon"], "err_pct": c["asr_error_pct"],
                         "coverage": c["cells_in_90pct_interval"],
                         "cases_per_year": d["cases_per_year"], "converged": bool(conv),
                         "dir": f.parent.name})
    return pd.DataFrame(rows)


def table(bt):
    g = bt[bt.converged & (bt.cases_per_year >= MIN_CASES)]
    out = {}
    for (ind, tu, h), x in g.groupby(["indicator", "train_until", "horizon"]):
        e = x.err_pct.values
        out[f"{ind}/{tu}/h{h}"] = {
            "n": int(len(e)), "period": x.period.iloc[0],
            "median_pct": round(float(np.median(e)), 2),
            "mad_pct": round(float(np.median(np.abs(e - np.median(e)))), 2),
            "share_within_10pct": round(float((np.abs(e) <= 10).mean()), 2),
            "share_negative": round(float((e < 0).mean()), 2),
            "cell_coverage_90": round(float(x.coverage.mean()), 2)}
    return out


def fig_backtest(bt, summary):
    g = bt[bt.converged & (bt.cases_per_year >= MIN_CASES)]
    origins = sorted(g.train_until.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, ind in zip(axes, ["incidence", "mortality"]):
        d = g[g.indicator == ind]
        pos, labels, data = [], [], []
        x = 0
        for tu in origins:
            for h in sorted(d[d.train_until == tu].horizon.unique()):
                e = d[(d.train_until == tu) & (d.horizon == h)].err_pct.values
                if len(e) < MIN_FITS_PER_CELL:
                    continue
                pos.append(x); labels.append(f"{tu}\nh{h}"); data.append(e); x += 1
            x += 0.6
        if not data:
            ax.text(0.5, 0.5, "fewer than 3 converged fits per cell", transform=ax.transAxes,
                    ha="center", color=MUTED)
            ax.set_title(f"{ind.capitalize()}: predicted vs observed ASR")
            continue
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=AXIS),
                        capprops=dict(color=AXIS), boxprops=dict(facecolor=IND_COLOR[ind], alpha=0.35,
                                                                 edgecolor=IND_COLOR[ind]))
        for p_, e in zip(pos, data):
            ax.plot(np.random.default_rng(0).normal(p_, 0.06, len(e)), e, "o", ms=3,
                    color=IND_COLOR[ind], alpha=0.5)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(pos, labels, fontsize=8)
        ax.set_title(f"{ind.capitalize()}: predicted vs observed ASR")
        ax.set_xlabel("Training window end (origin) and horizon in 5-year periods")
    axes[0].set_ylabel("ASR error, % (predicted / observed − 1)")
    fig.suptitle("Rolling-origin backtest: does the model's error depend on the years, or on the model?",
                 x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.01, 0.885, f"Fits with ≥{MIN_CASES} cases/yr; positive = observed rate came in below the trend "
             "fitted on the training window. Boxes: quartiles; dots: individual site × sex fits.",
             color=INK2, fontsize=9)
    fig.text(0.01, -0.03, "Source: NKRS/BFS px-x-1403030300_161, Bayesian APC model refitted at each origin "
             "(s06 --train-until)", color=MUTED, fontsize=8, va="top")
    fig.subplots_adjust(top=0.76, wspace=0.08)
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "fig12_backtest.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  -> assets/fig12_backtest.png")


def main():
    bt = load_backtests()
    if bt.empty:
        raise SystemExit("no backtest fits found; run s06 --train-until 2007/2012/2017 first")
    print(f"backtest rows: {len(bt)} ({bt.dir.nunique()} fits, origins {sorted(bt.train_until.unique())})")
    summary = table(bt)
    fig_backtest(bt, summary)
    (ASSETS / "backtest_summary.json").write_text(json.dumps(summary, indent=1))
    bt.to_parquet(MOD / "backtest.parquet", index=False)

    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = ["| indicator | origin | horizon | period | n | median % | MAD % | within ±10% | negative | cell cov. 90% |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        ind, tu, h = k.split("/")
        rows.append(f"| {ind} | {tu} | {h} | {v['period']} | {v['n']} | {v['median_pct']:+.1f} | {v['mad_pct']:.1f} "
                    f"| {v['share_within_10pct']:.2f} | {v['share_negative']:.2f} | {v['cell_coverage_90']:.2f} |")
    nonconv = bt[~bt.converged].dir.unique()
    md = ["# s08 backtest checks", "", f"- backtest fits: {bt.dir.nunique()}, non-converged: {len(nonconv)}",
          f"- scored: fits with ≥{MIN_CASES} cases/yr and converged", "", *rows, ""]
    if len(nonconv):
        md += ["non-converged (excluded):", *[f"- {d}" for d in nonconv], ""]
    (AUDIT / "s08_checks.md").write_text("\n".join(md))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
