#!/usr/bin/env python3
"""
cancer-ch / Stage 3: parse BFS APC mortality tables su-d-14.03.04.01.11..20 (1970-2009).

Sheets are identified by their title row, not by tab number (breast/cervix/prostate
files have only one sex):
  "... nach Periode und Altersgruppe, <sex>"            period x age x {count, rate}
  "... Todesfälle nach Geburtsjahrgang ..., <sex>"       cohort x age, counts
  "... Mortalitätsraten nach Geburtsjahrgang ..., <sex>" cohort x age, rates
  "... rohe und standardisierte Mortalitätsraten"        year x sex x {crude, asr, asr_corr}

Outputs (interim/mortality/):
  apc_period.parquet  apc_cohort.parquet  apc_asr.parquet
  icd_bridge.parquet  ICD-8 -> ICD-10 factor per cause/sex/year = asr_corr / asr (1970-94)

The files are legacy .xls (BIFF) despite the .xlsx name -> read with calamine.
"""
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
SOURCES = yaml.safe_load((REPO / "config" / "sources.yaml").read_text())
DATA = data_dir(Path(SOURCES["paths"]["data"]))
RAW = DATA / "raw" / "bfs_dam" / "mortality"
OUT = DATA / "interim" / "mortality"
AUDIT = DATA / "audit"
XL_ENGINE = "calamine"

CAUSE_BY_ORDER = {"su-d-14.03.04.01.11": "all", "su-d-14.03.04.01.14": "cancer",
                  "su-d-14.03.04.01.15": "stomach", "su-d-14.03.04.01.16": "colon",
                  "su-d-14.03.04.01.17": "lung", "su-d-14.03.04.01.18": "breast",
                  "su-d-14.03.04.01.19": "cervix", "su-d-14.03.04.01.20": "prostate"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------- helpers -----
def txt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    t = str(x).strip().replace("'", "").replace(",", ".")
    if t in ("-", "–"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return np.nan


def norm_age(label):
    """'unter 5 Jahre' -> 0-4, '5 bis 9 Jahre' / '5-9 Jahre' -> 5-9, '95-jährig und älter' -> 95+."""
    t = txt(label).lower()
    t = re.sub(r"\s*\d\)$", "", t)
    if t.startswith("alle"):
        return "Total"
    if t.startswith("unter"):
        upper = int(re.search(r"\d+", t).group())
        return f"0-{upper - 1}"
    m = re.match(r"^(\d+)\s*(?:bis|-)\s*(\d+)", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d+)\D*(?:älter|und mehr|\+)", t)
    if m:
        return f"{m.group(1)}+"
    return None


def age_start(a):
    return np.nan if a in (None, "Total") else int(re.match(r"\d+", a).group())


def sex_of(title):
    t = title.lower()
    return "M" if "männer" in t else "F" if "frauen" in t else None


def find_row(df, pred, limit=12):
    for i in range(min(limit, len(df))):
        if pred([txt(v) for v in df.iloc[i]]):
            return i
    raise ValueError("header row not found")


# --------------------------------------------------------------- parsers ---
def parse_period(df, cause, sex):
    hp = find_row(df, lambda r: r[0].startswith("Periode"))
    per = {}
    cur = None
    for j in range(1, df.shape[1]):
        p = txt(df.iat[hp, j])
        if re.match(r"^\d{4}-\d{4}$", p):
            cur = p
        m = txt(df.iat[hp + 1, j]).lower()
        if cur and (m.startswith("anzahl") or m.startswith("rate")):
            per[j] = (cur, "count" if m.startswith("anzahl") else "rate")
    rows = []
    for i in range(hp + 2, len(df)):
        age = norm_age(df.iat[i, 0])
        if age is None:
            continue
        for j, (p, meas) in per.items():
            rows.append({"cause_id": cause, "sex": sex, "period": p,
                         "period_start": int(p[:4]), "age": age, "age_start": age_start(age),
                         "measure": meas, "value": num(df.iat[i, j])})
    return rows


def parse_cohort(df, cause, sex, measure):
    ha = find_row(df, lambda r: r[0].startswith("Altersgruppe"))
    ages = {j: norm_age(df.iat[ha, j]) for j in range(1, df.shape[1])}
    ages = {j: a for j, a in ages.items() if a}
    rows = []
    for i in range(ha + 1, len(df)):
        c = txt(df.iat[i, 0])
        if not re.match(r"^\d{4}-\d{4}$", c):
            continue
        for j, a in ages.items():
            v = num(df.iat[i, j])
            if not np.isnan(v):          # empty cells = outside the observed Lexis window
                rows.append({"cause_id": cause, "sex": sex, "cohort": c,
                             "cohort_start": int(c[:4]), "age": a, "age_start": age_start(a),
                             "measure": measure, "value": v})
    return rows


def parse_asr(df, cause):
    hs = find_row(df, lambda r: any(x in ("Männer", "Frauen") for x in r))
    cols, sex = {}, None
    for j in range(1, df.shape[1]):
        s = txt(df.iat[hs, j])
        if s in ("Männer", "Frauen"):
            sex = "M" if s == "Männer" else "F"
        m = txt(df.iat[hs + 1, j]).lower()
        if sex and m:
            meas = ("asr_corr" if m.startswith("korrigiert") else
                    "asr" if m.startswith("standardisiert") else
                    "crude" if m.startswith("roh") else None)
            if meas:
                cols[j] = (sex, meas)
    rows = []
    for i in range(hs + 2, len(df)):
        y = txt(df.iat[i, 0])
        if not re.match(r"^(19|20)\d{2}(\.0)?$", y):
            continue
        for j, (s, meas) in cols.items():
            rows.append({"cause_id": cause, "sex": s, "year": int(float(y)),
                         "measure": meas, "value": num(df.iat[i, j])})
    return rows


# ------------------------------------------------------------------ main ---
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per, coh, asr = [], [], []
    for onr, cause in CAUSE_BY_ORDER.items():
        files = sorted(RAW.glob(f"{onr}_*.xlsx"))
        if not files:
            log(f"{onr}: missing")
            continue
        book = pd.read_excel(files[-1], sheet_name=None, header=None, dtype=object,
                             engine=XL_ENGINE)
        kinds = []
        for name, df in book.items():
            title = txt(df.iat[0, 0]) if len(df) else ""
            t, sex = title.lower(), sex_of(title)
            if "nach periode" in t:
                per += parse_period(df, cause, sex); kinds.append(f"{name}:period/{sex}")
            elif "geburtsjahrgang" in t and "todesfälle" in t:
                coh += parse_cohort(df, cause, sex, "count"); kinds.append(f"{name}:cohort_n/{sex}")
            elif "geburtsjahrgang" in t and "mortalitätsraten" in t:
                coh += parse_cohort(df, cause, sex, "rate"); kinds.append(f"{name}:cohort_r/{sex}")
            elif "standardisierte" in t:
                asr += parse_asr(df, cause); kinds.append(f"{name}:asr")
        log(f"{files[-1].name} [{cause}]: {', '.join(kinds)}")

    per, coh, asr = pd.DataFrame(per), pd.DataFrame(coh), pd.DataFrame(asr)
    per.to_parquet(OUT / "apc_period.parquet", index=False)
    coh.to_parquet(OUT / "apc_cohort.parquet", index=False)
    asr.to_parquet(OUT / "apc_asr.parquet", index=False)

    w = asr.pivot_table(index=["cause_id", "sex", "year"], columns="measure",
                        values="value").reset_index()
    bridge = build_bridge(per, w)
    bridge.to_parquet(OUT / "icd_bridge.parquet", index=False)
    log(f"apc_period {len(per):,}  apc_cohort {len(coh):,}  apc_asr {len(asr):,}  bridge {len(bridge):,}")
    sys.exit(checks(per, coh, w, bridge))


def count_ratio(per):
    """Corrected APC deaths 1990-94 / raw je-03/04 annual deaths 1990-94, per cause and sex."""
    sp = DATA / "interim" / "mortality" / "mort_series.parquet"
    if not sp.exists():
        return pd.Series(dtype=float)
    s = pd.read_parquet(sp)
    raw = s[(s.measure == "count") & s.year.between(1990, 1994)] \
        .groupby(["cause_id", "sex"])["value"].sum()
    apc = per[(per.measure == "count") & (per.age == "Total") & (per.period == "1990-1994")] \
        .set_index(["cause_id", "sex"])["value"]
    return (apc / raw).dropna()


def build_bridge(per, w):
    """One row per cause x sex x year 1970-1994 with the ICD-8 -> ICD-10 factor.
    method: asr_ratio   asr_corr / asr from the yearly table (2014 edition)
            identity    all causes: total deaths do not depend on coding
            count_ratio corrected APC 1990-94 / raw je-03/04 1990-94 (2008 edition, no raw ASR)
            from_F      male breast: rates ~0.1/100k, ratio is rounding noise -> female factor"""
    years = pd.DataFrame({"year": range(1970, 1995)})
    cr = count_ratio(per)
    out = []
    for (cause, sex), g in per.groupby(["cause_id", "sex"]):
        a = w[(w.cause_id == cause) & (w.sex == sex) & (w.year <= 1994)]
        if cause == "all":
            f = years.assign(factor=1.0, method="identity")
        elif cause == "breast" and sex == "M":
            continue                                   # filled from F below
        elif "asr_corr" in a and a["asr_corr"].notna().any():
            f = a.assign(factor=a.asr_corr / a.asr)[["year", "factor"]].assign(method="asr_ratio")
        elif (cause, sex) in cr.index:
            f = years.assign(factor=cr[(cause, sex)], method="count_ratio")
        else:
            log(f"bridge: no factor for {cause}/{sex}")
            continue
        out.append(f.assign(cause_id=cause, sex=sex,
                            count_ratio_check=cr.get((cause, sex), np.nan)))
    b = pd.concat(out, ignore_index=True)
    fem = b[(b.cause_id == "breast") & (b.sex == "F")]
    if len(fem):
        b = pd.concat([b, fem.assign(sex="M", method="from_F",
                                     count_ratio_check=cr.get(("breast", "M"), np.nan))],
                      ignore_index=True)
    return b[["cause_id", "sex", "year", "factor", "method", "count_ratio_check"]]


# ---------------------------------------------------------------- checks ---
def checks(per, coh, asr_w, bridge):
    md, hard = [f"# s03 checks ({datetime.now():%Y-%m-%d %H:%M})", ""], 0

    # 1. age groups sum to "Alle Altersgruppen" (counts)
    c = per[per.measure == "count"]
    tot = c[c.age == "Total"].set_index(["cause_id", "sex", "period"])["value"]
    parts = c[c.age != "Total"].groupby(["cause_id", "sex", "period"])["value"].sum()
    d = (parts - tot).dropna()
    # 1970-94 counts are ICD-corrected (fractional, then rounded): allow +-1 there
    pre95 = d.index.get_level_values("period").str[:4].astype(int) < 1995
    # 20 independently rounded cells: sd of the sum ~1.3, so +-3 is the 2-sigma band
    bad = d[(d.abs() > 0.5) & ~pre95 | (d.abs() > 3.0) & pre95]
    rnd = int(((d.abs() > 0.5) & pre95).sum())
    md.append(f"- period table: age groups sum to total {len(d) - len(bad)}/{len(d)} "
              f"({rnd} pre-1995 cells off by 1 from rounding of corrected counts)")
    if len(bad):
        md.append("  - " + "; ".join(f"{k}: {v:+.0f}" for k, v in bad.head(8).items()))
    hard += len(bad) > 0

    # 2. period vs cohort layout (informational). The cohort table holds 8 synthetic
    #    cohorts c = p - a per age group and has no 95+ column, so edge Lexis triangles
    #    are dropped by design: expect it a few % below the period table, never above.
    pc = c[~c.age.isin(["Total", "95+"])].groupby(["cause_id", "sex"])["value"].sum()
    cc = coh[coh.measure == "count"].groupby(["cause_id", "sex"])["value"].sum()
    j = pd.concat([pc.rename("period"), cc.rename("cohort")], axis=1).dropna()
    j["diff_pct"] = (j.cohort / j.period - 1) * 100
    md.append("- period vs cohort table, total deaths 1970-2009 (info, cohort table drops edges):")
    md.append(j.round(2).to_markdown())
    hard += int((j.diff_pct > 0.5).any())        # cohort table above period = parsing error

    # 3. ICD-10 era: APC period sums vs je-03/04 annual counts (1995-2009, no correction)
    sp = DATA / "interim" / "mortality" / "mort_series.parquet"
    if sp.exists():
        s = pd.read_parquet(sp)
        s = s[(s.measure == "count") & s.year.between(1995, 2009)].copy()
        s["period"] = s.year.map(lambda y: f"{y - (y - 1995) % 5}-{y - (y - 1995) % 5 + 4}")
        sj = s.groupby(["cause_id", "sex", "period"])["value"].sum()
        k = pd.concat([tot.rename("apc"), sj.rename("je_series")], axis=1, join="inner").dropna()
        k["diff_pct"] = (k.apc / k.je_series - 1) * 100
        md.append(f"- APC table vs je-03/04, periods 1995-2009: {len(k)} cells, "
                  f"max |diff| {k.diff_pct.abs().max():.2f}%")
        if (k.diff_pct.abs() > 2).any():
            md.append(k[k.diff_pct.abs() > 2].round(2).to_markdown())
        small = k.index.get_level_values("cause_id").isin(["breast"]) & \
                (k.index.get_level_values("sex") == "M")
        hard += int(((k.diff_pct.abs() > 2) & ~small).any())   # male breast: ~10 deaths/yr
    else:
        md.append("- je-03/04 cross-check skipped (run s02 first)")

    # 4. bridge factors: sane range, and the two independent methods agree
    if not bridge.empty:
        r = bridge.groupby(["cause_id", "sex", "method"]).agg(
            fmin=("factor", "min"), fmean=("factor", "mean"), fmax=("factor", "max"),
            count_ratio_check=("count_ratio_check", "first"))
        md.append("- ICD-8 -> ICD-10 bridge factor, 1970-1994 "
                  "(count_ratio_check = corrected APC / raw je-03/04, 1990-94):")
        md.append(r.round(3).to_markdown())
        hard += int(((r.fmin < 0.5) | (r.fmax > 1.5)).any())
        asr_m = r.xs("asr_ratio", level="method", drop_level=False) \
            if "asr_ratio" in r.index.get_level_values("method") else r.iloc[0:0]
        dev = (asr_m.fmean - asr_m.count_ratio_check).abs()
        md.append(f"- asr_ratio vs count_ratio agreement: max |diff| {dev.max():.3f}")
        if (dev > 0.03).any():
            md.append("  - disagree > 0.03: " + ", ".join(map(str, dev[dev > 0.03].index)))
        post = asr_w[asr_w.year >= 1995]
        if {"asr", "asr_corr"} <= set(post):
            dv = (post.asr_corr - post.asr).abs().max()
            md.append(f"- asr_corr == asr from 1995: max |diff| {dv:.3f}")
            hard += int(dv > 0.05)
    md.append(f"- editions: all/stomach = 2008 edition (data to 2004), others = 2014 (to 2009)")

    (AUDIT / "s03_checks.md").write_text("\n".join(md))
    print("\n".join(md))
    return 1 if hard else 0


if __name__ == "__main__":
    main()
