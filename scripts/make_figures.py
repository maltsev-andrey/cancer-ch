#!/usr/bin/env python3
"""
cancer-ch: checkpoint figures for the wiki (data layer, stages s00-s03).

Reads the interim Parquet files and writes PNGs to repo/assets/:
  fig01_data_coverage.png        which source covers which years
  fig02_cancer_mortality_asr.png cancer mortality ASR 1970-2024, raw vs ICD-bridged,
                                 validated against the BFS-corrected APC series
  fig03_icd_bridge.png           ICD-8 -> ICD-10 factors per cause, two methods
  fig04_lung_m_vs_f.png          lung cancer: mortality (annual) + incidence (NKRS periods)
  fig05_incidence_vs_mortality.png  new cases vs deaths per site, 2018-2022
  fig06_age_profile.png          age-specific incidence and mortality, 2018-2022
plus assets/figures_summary.json with the headline numbers quoted in the wiki.

Run:  python scripts/make_figures.py
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

# ---------------------------------------------------------------- config ----
REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    """paths.data from sources.yaml, relative to the repo root unless absolute;
    the environment variable CANCER_CH_DATA overrides both."""
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()
DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
MORT = DATA / "interim" / "mortality"
PX = DATA / "interim" / "pxweb"
OUT = REPO / "assets"
DPI = 200

# palette: validated categorical slots 1-2 (adjacent CVD dE 24.7, normal 33.6)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEX_COLOR = {"M": "#2a78d6", "F": "#eb6834"}          # color follows the entity everywhere
SEX_NAME = {"M": "Men", "F": "Women"}
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"          # sequential blue 250 / 550

SITE_EN = {"Mundhöhle und Rachen": "Oral cavity & pharynx", "Speiseröhre": "Oesophagus",
           "Magen": "Stomach", "Dünndarm": "Small intestine", "Dickdarm": "Colorectum",
           "Anus": "Anus", "Leber": "Liver", "Gallenblase und extrahepatische Gallenwege": "Gallbladder",
           "Bauchspeicheldrüse": "Pancreas", "Kehlkopf": "Larynx",
           "Lunge, Bronchien, Luftröhre": "Lung", "Brustfell (Pleuramesotheliom)": "Pleural mesothelioma",
           "Knochen, Gelenke, Knorpel": "Bone", "Hautmelanom": "Skin melanoma",
           "Bindegewebe": "Soft tissue", "Brust": "Breast", "Gebärmutterhals": "Cervix uteri",
           "Gebärmutterkörper": "Corpus uteri", "Eierstock": "Ovary", "Prostata": "Prostate",
           "Hoden": "Testis", "Niere": "Kidney", "Harnblase": "Bladder",
           "Ableitende Harnwege": "Urinary tract", "Auge": "Eye",
           "Gehirn und Zentralnervensystem": "Brain & CNS", "Schilddrüse": "Thyroid",
           "Hodgkin-Lymphom": "Hodgkin lymphoma", "Non-Hodgkin-Lymphom": "Non-Hodgkin lymphoma",
           "Multiples Myelom": "Multiple myeloma", "Leukämie": "Leukaemia"}

CAUSE_NAME = {"all": "All causes", "cancer": "All cancers", "stomach": "Stomach",
              "colon": "Colorectal", "lung": "Lung", "breast": "Breast",
              "cervix": "Cervix uteri", "prostate": "Prostate"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def source(fig, text):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8, ha="left", va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> assets/{name}")


def load_px(cube):
    d = pd.read_parquet(PX / f"{cube}.parquet")
    return d.drop(columns=[c for c in d if c.endswith("_code")])


def bridged_series():
    """je-03/04 published rate (European-standard ASR) with the ICD bridge applied <=1994."""
    s = pd.read_parquet(MORT / "mort_series.parquet")
    s = s[s.measure == "rate"][["year", "sex", "cause_id", "value"]]
    b = pd.read_parquet(MORT / "icd_bridge.parquet")[["cause_id", "sex", "year", "factor"]]
    s = s.merge(b, on=["cause_id", "sex", "year"], how="left")
    s["factor"] = s["factor"].fillna(1.0)
    s["bridged"] = s["value"] * s["factor"]
    return s


# ============================================================== figures =====
def fig01_coverage(summary):
    rows = []
    s = pd.read_parquet(MORT / "mort_series.parquet")
    a = pd.read_parquet(MORT / "mort_age.parquet")
    ap = pd.read_parquet(MORT / "apc_period.parquet")
    nk = load_px("px-x-1403030300_161")
    per = sorted({int(p[:4]) for p in nk["Periode"].unique()})
    pop_c = load_px("px-x-0102020000_104")
    pop_y = pd.to_numeric(pop_c["Jahr"], errors="coerce")
    rows = [
        ("Incidence, NKRS cube (5-y periods,\nregion x site x sex x age)", [(p, p + 5) for p in per], "period"),
        ("Mortality by cause, annual\n(sex; decades before 1990)",
         [(y, y + 1) for y in sorted(s.year.unique())], "annual"),
        ("Mortality by cause x age, annual\n(6 broad age groups)",
         [(a.year.min(), a.year.max() + 1)], "annual"),
        ("Mortality APC tables\n(5-y periods x 5-y ages)",
         [(int(p[:4]), int(p[:4]) + 5) for p in sorted(ap.period.unique())], "period"),
        ("Population by canton x sex x age",
         [(int(pop_y.min()), int(pop_y.max()) + 1)], "annual"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (label, spans, kind) in enumerate(rows[::-1]):
        for x0, x1 in spans:
            ax.barh(i, x1 - x0 - 0.15, left=x0, height=0.5,
                    color=BLUE_DARK if kind == "period" else BLUE_LIGHT, edgecolor=SURFACE)
        ax.text(1968, i, label, ha="right", va="center", fontsize=9, color=INK2)
    ax.axvline(1995, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(1995.4, -0.62, "ICD-8 → ICD-10", color=INK2, fontsize=8.5)
    ax.axvline(2020, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(2019.6, -0.62, "Cancer Registration Act", color=INK2, fontsize=8.5, ha="right")
    ax.set_yticks([]); ax.set_xlim(1968, 2027); ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("What the public Swiss data actually covers")
    subtitle(ax, "Dark = 5-year periods, light = annual. No public source has cantonal cancer data.")
    source(fig, "Sources: BFS/NKRS PxWeb px-x-1403030300_161, BFS CoD je-d-14.03.04.01.01-04, "
                "su-d-14.03.04.01.11-20, STATPOP px-x-0102020000_104")
    save(fig, "fig01_data_coverage.png")


def fig02_mortality(summary):
    s = bridged_series()
    c = s[s.cause_id == "cancer"].sort_values("year")
    apc = pd.read_parquet(MORT / "apc_asr.parquet")
    apc = apc[(apc.cause_id == "cancer") & (apc.measure == "asr_corr")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for sex in ("M", "F"):
        d = c[c.sex == sex]
        pre = d[d.year <= 1994]
        ax.plot(pre.year, pre.value, color=SEX_COLOR[sex], lw=1.2, ls=(0, (2, 2)), alpha=0.8)
        ax.plot(d.year, d.bridged, color=SEX_COLOR[sex], lw=2,
                label=f"{SEX_NAME[sex]}, ICD-10 equivalent")
        v = apc[apc.sex == sex]
        ax.plot(v.year, v.value, ls="none", marker="o", ms=4.5, mfc=SURFACE,
                mec=SEX_COLOR[sex], mew=1.2)
        last = d.iloc[-1]
        ax.text(last.year + 0.6, last.bridged, f"{SEX_NAME[sex]}  {last.bridged:.0f}",
                color=INK, fontsize=9, va="center")
        peak = d.loc[d.bridged.idxmax()]
        summary[f"cancer_asr_{sex}"] = {"peak_year": int(peak.year), "peak": round(peak.bridged, 1),
                                        "last_year": int(last.year), "last": round(last.bridged, 1),
                                        "change_pct": round((last.bridged / peak.bridged - 1) * 100, 1)}
    ax.axvline(1995, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(1995.6, 8, "ICD-8 → ICD-10", color=INK2, fontsize=8.5, va="bottom")
    ax.plot([], [], color=MUTED, lw=1.2, ls=(0, (2, 2)), label="as published (ICD-8 coding)")
    ax.plot([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, label="BFS corrected APC series (check)")
    ax.set_ylim(bottom=0); ax.set_xlim(1968, 2030)
    ax.set_ylabel("Deaths per 100,000, age-standardised (European standard)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Cancer mortality in Switzerland, 1970–2024")
    subtitle(ax, "ICD-8 years scaled by the BFS bridge factor; open circles = BFS' own corrected series")
    source(fig, "Source: BFS cause-of-death statistics (je-d-14.03.04.01.03/04, su-d-14.03.04.01.14). "
                "Points before 1990: 1970 and 1980 only.")
    save(fig, "fig02_cancer_mortality_asr.png")


def fig03_bridge(summary):
    b = pd.read_parquet(MORT / "icd_bridge.parquet")
    g = b.groupby(["cause_id", "sex"]).agg(f=("factor", "mean"), chk=("count_ratio_check", "first"),
                                           method=("method", "first")).reset_index()
    g = g[g.cause_id != "all"]
    order = g.groupby("cause_id").f.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, cause in enumerate(order):
        for sex, dy in (("M", 0.12), ("F", -0.12)):
            r = g[(g.cause_id == cause) & (g.sex == sex)]
            if r.empty or r.method.iloc[0] == "from_F":
                continue
            r = r.iloc[0]
            ax.plot(r.f, i + dy, marker="o", ms=8, color=SEX_COLOR[sex], mec=SURFACE, mew=2)
            if not np.isnan(r.chk):
                ax.plot(r.chk, i + dy, marker="|", ms=12, mew=1.5, color=INK2)
        fmean = g[g.cause_id == cause].f.mean()
        ax.text(1.015, i, f"{(fmean - 1) * 100:+.0f}%", va="center", fontsize=9, color=INK2)
    ax.axvline(1.0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), [CAUSE_NAME.get(c, c) for c in order])
    ax.set_xlim(0.82, 1.05); ax.grid(axis="y", visible=False)
    ax.set_xlabel("ICD-10 equivalent / ICD-8 coded deaths, 1970–1994")
    ax.plot([], [], marker="o", ls="none", color=SEX_COLOR["M"], label="Men")
    ax.plot([], [], marker="o", ls="none", color=SEX_COLOR["F"], label="Women")
    ax.plot([], [], marker="|", ls="none", ms=10, color=INK2, label="independent check (count ratio 1990–94)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("How much the ICD-8 → ICD-10 switch moved each cause")
    subtitle(ax, "Two independent estimates agree to 0.001: ASR ratio (dots) and publication cross-check (ticks)")
    source(fig, "Source: BFS su-d-14.03.04.01.14-20 (Tab. 7) vs je-d-14.03.04.01.03/04. "
                "Stomach: factor 1.00 (no measurable coding effect).")
    summary["bridge"] = {c: round(g[g.cause_id == c].f.mean(), 3) for c in order}
    save(fig, "fig03_icd_bridge.png")


def fig04_lung(summary):
    s = bridged_series()
    l = s[s.cause_id == "lung"].sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 5))
    for sex in ("M", "F"):
        d = l[l.sex == sex]
        ax.plot(d.year, d.bridged, color=SEX_COLOR[sex], lw=2, label=f"{SEX_NAME[sex]}, mortality (annual)")
        last = d.iloc[-1]
        ax.text(last.year + 0.6, last.bridged, f"{last.bridged:.1f}", color=INK, fontsize=9, va="center")
        summary[f"lung_mort_{sex}"] = {int(y): round(v, 1) for y, v in
                                       d[d.year.isin([1970, 1990, 2000, 2010, 2024])][["year", "bridged"]].values}
    try:
        r = load_px("px-x-1403030300_162")
        kz = [k for k in r["Kennzahl"].unique() if re.search("standard", k, re.I)]
        r = r[(r.Sprachregion == "Schweiz") & r.Krebsart.str.startswith("Lunge") &
              (r.Indikator == "Inzidenz") & r.Kennzahl.isin(kz[:1])]
        for sex, lab in (("M", "Mann"), ("F", "Frau")):
            d = r[r.Geschlecht == lab]
            for p, v in zip(d.Periode, d.value):
                y0 = int(p[:4])
                ax.plot([y0, y0 + 4.8], [v, v], color=SEX_COLOR[sex], lw=5, alpha=0.35,
                        solid_capstyle="butt")
            summary[f"lung_inc_{sex}"] = {p: round(v, 1) for p, v in zip(d.Periode, d.value)}
        ax.plot([], [], color=MUTED, lw=5, alpha=0.5, label="incidence, 5-year mean (NKRS)")
    except Exception as e:                               # figure still useful without overlay
        print(f"  (incidence overlay skipped: {e})")
    ax.axvline(1995, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_ylim(bottom=0); ax.set_xlim(1968, 2028)
    ax.set_ylabel("per 100,000, age-standardised (European standard)")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.55), fontsize=8.5)
    ax.set_title("Lung cancer: men down 60% since 1980, women flat since 2010")
    subtitle(ax, "Female incidence is still rising while female mortality has stopped rising. "
                 "Main target for the APC model.")
    source(fig, "Sources: BFS CoD je-d-14.03.04.01.03/04 (ICD-bridged), NKRS px-x-1403030300_162")
    save(fig, "fig04_lung_m_vs_f.png")


def fig05_inc_mort(summary):
    d = load_px("px-x-1403030300_161")
    d = d[(d.Sprachregion == "Schweiz") & (d.Geschlecht.str.contains("Total")) &
          (d.Periode == "2018-2022") & d.Altersklasse.str.contains("Total") &
          d.Kennzahl.str.startswith("Anzahl")]
    d = d[~d.Krebsart.str.contains(r"^\s|Alle|Andere", regex=True)]
    w = d.pivot_table(index="Krebsart", columns="Indikator", values="value").dropna()
    w = w.sort_values("Inzidenz").tail(18)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    y = np.arange(len(w))
    ax.barh(y, w["Inzidenz"], height=0.7, color=BLUE_LIGHT, label="New cases per year")
    ax.barh(y, w["Mortalität"], height=0.36, color=BLUE_DARK, label="Deaths per year")
    for i, (inc, mort) in enumerate(zip(w["Inzidenz"], w["Mortalität"])):
        ax.text(inc + w["Inzidenz"].max() * 0.01, i, f"{inc:,.0f}   M:I {mort / inc:.2f}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [SITE_EN.get(s, s) for s in w.index]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, w["Inzidenz"].max() * 1.28)
    ax.set_xlabel("Annual mean, 2018–2022, both sexes")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("New cases vs deaths by cancer site, 2018–2022")
    subtitle(ax, "Prostate and breast are the most frequent, lung kills the most. "
                 "M:I = deaths / new cases, high = poor survival.")
    source(fig, "Source: NKRS/BFS PxWeb px-x-1403030300_161")
    summary["mi_ratio_2018_22"] = {SITE_EN.get(k, k): round(r["Mortalität"] / r["Inzidenz"], 2) for k, r in w.iterrows()}
    save(fig, "fig05_incidence_vs_mortality.png")


def fig06_age(summary):
    d = load_px("px-x-1403030300_161")
    d = d[(d.Sprachregion == "Schweiz") & d.Krebsart.str.startswith("Alle") &
          (d.Periode == "2018-2022") & ~d.Altersklasse.str.contains("Total") &
          d.Kennzahl.str.contains("Rate")]
    ages = [a for a in d.Altersklasse.unique()]
    ages.sort(key=lambda a: int(re.match(r"\d+", a).group()))
    x = np.arange(len(ages))
    fig, ax = plt.subplots(figsize=(10, 5))
    for sex, lab in (("M", "Mann"), ("F", "Frau")):
        for ind, ls, name in (("Inzidenz", "-", "incidence"), ("Mortalität", (0, (4, 2)), "mortality")):
            v = d[(d.Geschlecht == lab) & (d.Indikator == ind)].set_index("Altersklasse").reindex(ages)
            ax.plot(x, v.value, color=SEX_COLOR[sex], ls=ls, label=f"{SEX_NAME[sex]}, {name}")
    ax.set_yscale("log")
    ax.set_xticks(x, ages, rotation=45, ha="right")
    ax.set_ylabel("per 100,000 per year (log scale)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_title("Cancer risk by age, 2018–2022")
    subtitle(ax, "Women have higher incidence than men at ages 25\u201354 (breast, thyroid, melanoma); "
                 "men overtake from 55.")
    source(fig, "Source: NKRS/BFS PxWeb px-x-1403030300_161, age-specific rates")
    save(fig, "fig06_age_profile.png")


def main():
    summary = {}
    for f in (fig01_coverage, fig02_mortality, fig03_bridge, fig04_lung, fig05_inc_mort, fig06_age):
        print(f.__name__)
        try:
            f(summary)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    (OUT / "figures_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("  -> assets/figures_summary.json")


if __name__ == "__main__":
    main()
