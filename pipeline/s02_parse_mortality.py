#!/usr/bin/env python3
"""
cancer-ch / Stage 2: parse BFS cause-of-death XLSX (je-d-14.03.04.01.01..04).

je-..01/02  one sheet per year (1990..), causes x 6 age groups + Total,
            two blocks: "Anzahl Sterbefälle" (counts) and "Sterbeziffer" (rates)
            -> interim/mortality/mort_age.parquet
je-..03/04  sheets "Anzahl Fälle" / "Sterbeziffern", causes x years (1970, 1980, 1990..)
            -> interim/mortality/mort_series.parquet

Checks (written to audit/s02_checks.md, non-zero exit on hard failures):
  - sum of age groups == Total (counts)
  - davon-children <= parent (counts)
  - je-01/02 Total == je-03/04 count for the same year, sex, cause
  - every row label maps to a canonical cause (config/causes.yaml)

su-d-14.03.04.01.* (APC tables 1970-2009) are parsed separately, layout differs.
"""
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
SOURCES = yaml.safe_load((REPO / "config" / "sources.yaml").read_text())
CAUSES = yaml.safe_load((REPO / "config" / "causes.yaml").read_text())["causes"]
DATA = data_dir(Path(SOURCES["paths"]["data"]))
RAW = DATA / "raw" / "bfs_dam" / "mortality"
OUT = DATA / "interim" / "mortality"
AUDIT = DATA / "audit"

SEX_BY_ORDER = {"je-d-14.03.04.01.01": "M", "je-d-14.03.04.01.02": "F",
                "je-d-14.03.04.01.03": "M", "je-d-14.03.04.01.04": "F"}
AGE_ORDER = ["0", "1-14", "15-44", "45-64", "65-84", "85+", "Total"]
XL_ENGINE = "openpyxl"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------- cleaning ----
RE_FOOT = re.compile(r"\s*\d\)\s*")          # "Suizid 4)", "0  1)", "Rate 2)"


def clean_label(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return re.sub(r"\s+", " ", RE_FOOT.sub(" ", str(x))).strip()


def to_num(x):
    """BFS cells: 1'002 | 2'401.0 | 25,3 | '-' (zero) | '*' (not applicable)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    t = str(x).strip().replace("'", "").replace("’", "").replace("\xa0", "").replace(" ", "")
    if t in ("-", "–"):
        return 0.0
    if t in ("*", "...", "…", "x", ""):
        return np.nan
    t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def cause_id(label):
    hits = [c["id"] for c in CAUSES if re.search(c["match"], label)]
    if len(hits) > 1:                 # e.g. "^Lunge" would also catch "Lungenembolie"
        raise ValueError(f"label '{label}' matches several causes: {hits}, tighten causes.yaml")
    return hits[0] if hits else None


def is_data_row(row):
    lab = clean_label(row.iloc[0])
    if not lab or lab.lower().startswith(("davon", "(pro", "stand der", "quelle", "©", "auskunft")):
        return False
    if re.match(r"^\d\)", str(row.iloc[0]).strip()):
        return False
    return any(isinstance(v, (int, float)) and not pd.isna(v) or
               (isinstance(v, str) and re.match(r"^[\d'.,\- *–]+$", v.strip()))
               for v in row.iloc[1:])


def footnotes(df):
    return [str(v).strip() for v in df.iloc[:, 0].dropna()
            if re.match(r"^\d\)\s", str(v).strip())]


# ----------------------------------------------------- je-01/02 (by age) ----
def parse_age_sheet(df, year, sex, src):
    # age header: the row that contains "1-14"
    hdr = next(i for i in range(min(15, len(df)))
               if any(clean_label(v) == "1-14" for v in df.iloc[i]))
    cols = {}
    for j in range(1, df.shape[1]):
        lab = clean_label(df.iat[hdr, j])
        if not lab:  # the Total column has its label a couple of rows higher
            above = [clean_label(df.iat[i, j]) for i in range(max(0, hdr - 4), hdr)]
            lab = "Total" if "Total" in above else ""
        if lab in AGE_ORDER:
            cols[j] = lab
    rows, measure = [], None
    for i in range(hdr + 1, len(df)):
        lab0 = clean_label(df.iat[i, 0])
        if lab0.startswith("Anzahl Sterbef"):
            measure = "count"
            continue
        if lab0.startswith("Sterbeziffer"):
            measure = "rate"
            continue
        if measure is None or not is_data_row(df.iloc[i]):
            continue
        cid = cause_id(lab0)
        for j, age in cols.items():
            rows.append({"year": year, "sex": sex, "cause_id": cid, "cause_label": lab0,
                         "age": age, "measure": measure, "value": to_num(df.iat[i, j]),
                         "source": src})
    return rows


# ---------------------------------------------------- je-03/04 (series) ----
def parse_series_sheet(df, sex, measure, src):
    hdr = next(i for i in range(min(15, len(df)))
               if sum(bool(re.match(r"^(19|20)\d{2}$", clean_label(v)))
                      for v in df.iloc[i]) >= 3)
    years = {j: int(clean_label(df.iat[hdr, j])) for j in range(1, df.shape[1])
             if re.match(r"^(19|20)\d{2}$", clean_label(df.iat[hdr, j]))}
    rows = []
    for i in range(hdr + 1, len(df)):
        if not is_data_row(df.iloc[i]):
            continue
        lab0 = clean_label(df.iat[i, 0])
        cid = cause_id(lab0)
        for j, y in years.items():
            rows.append({"year": y, "sex": sex, "cause_id": cid, "cause_label": lab0,
                         "measure": measure, "value": to_num(df.iat[i, j]), "source": src})
    return rows


# ---------------------------------------------------------------- main -----
def latest_file(order_nr):
    files = sorted(RAW.glob(f"{order_nr}_*.xlsx"))
    return files[-1] if files else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    age_rows, ser_rows, notes = [], [], {}
    for onr, sex in SEX_BY_ORDER.items():
        f = latest_file(onr)
        if f is None:
            log(f"{onr}: file missing in {RAW}")
            continue
        book = pd.read_excel(f, sheet_name=None, header=None, dtype=object, engine=XL_ENGINE)
        if onr.endswith((".01", ".02")):
            for name, df in book.items():
                if re.match(r"^(19|20)\d{2}$", name.strip()):
                    age_rows += parse_age_sheet(df, int(name), sex, f.name)
            notes[onr] = footnotes(next(iter(book.values())))
            log(f"{f.name}: {sum(1 for _ in book)} sheets")
        else:
            for name, df in book.items():
                m = ("count" if name.startswith("Anzahl") else
                     "rate" if name.startswith("Sterbeziff") else None)
                if m:
                    ser_rows += parse_series_sheet(df, sex, m, f.name)
                    notes[f"{onr}/{name}"] = [clean_label(df.iat[0, 0])] + footnotes(df)
            log(f"{f.name}: sheets {list(book)}")

    age = add_flags(pd.DataFrame(age_rows))
    ser = add_flags(pd.DataFrame(ser_rows))
    if not age.empty:
        age["age"] = pd.Categorical(age["age"], AGE_ORDER, ordered=True)
        age.to_parquet(OUT / "mort_age.parquet", index=False)
    if not ser.empty:
        ser.to_parquet(OUT / "mort_series.parquet", index=False)
    (OUT / "mort_footnotes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1))
    log(f"mort_age: {len(age):,} rows  mort_series: {len(ser):,} rows")
    sys.exit(checks(age, ser))


def add_flags(df):
    """ICD-8 until 1994, ICD-10 from 1995 (series break, BFS footnote 1).
    1995-1997: je-01/02 and je-03/04 disagree for 'all' and 'alc_cirrh' (later recoding)."""
    if df.empty:
        return df
    df["icd_era"] = np.where(df["year"] <= 1994, "ICD-8", "ICD-10")
    df["revision_1995_97"] = df["year"].between(1995, 1997) & df["cause_id"].isin(["all", "alc_cirrh"])
    return df


# --------------------------------------------------------------- checks ----
def checks(age, ser):
    md, hard = [f"# s02 checks ({datetime.now():%Y-%m-%d %H:%M})", ""], 0
    parent = {c["id"]: c["parent"] for c in CAUSES}

    for name, df in (("mort_age", age), ("mort_series", ser)):
        if df.empty:
            md.append(f"- {name}: EMPTY")
            hard += 1
            continue
        key = [k for k in ("year", "sex", "cause_id", "age", "measure") if k in df]
        dup = df[df.duplicated(key, keep=False) & df.cause_id.notna()]
        md.append(f"- {name}: duplicate keys: {len(dup)}"
                  + (f" e.g. {dup.cause_label.unique()[:5].tolist()}" if len(dup) else ""))
        if len(dup):
            hard += 1
            continue
        unk = df[df.cause_id.isna()].cause_label.unique()
        md.append(f"- {name}: unmapped labels: {list(unk) or 'none'}")
        hard += len(unk) > 0
        md.append(f"- {name}: years {df.year.min()}..{df.year.max()} "
                  f"({df.year.nunique()}), sexes {sorted(df.sex.unique())}")

    if hard:
        (AUDIT / "s02_checks.md").write_text("\n".join(md))
        print("\n".join(md))
        return 1
    if not age.empty:
        c = age[age.measure == "count"]
        w = c.pivot_table(index=["year", "sex", "cause_id"], columns="age",
                          values="value", observed=True, aggfunc="sum")
        parts = w[[a for a in AGE_ORDER[:-1] if a in w]].sum(axis=1, min_count=1)
        bad = w[(parts - w["Total"]).abs() > 0.5]
        md.append(f"- age groups sum to Total: {len(w) - len(bad)}/{len(w)} ok")
        if len(bad):
            md.append("  - first mismatches:\n" + bad.head(5).to_markdown())
        hard += len(bad) > 0

        tot = c[c.age == "Total"].set_index(["year", "sex", "cause_id"])["value"]
        viol = []
        for (y, s, cid), v in tot.items():
            p = parent.get(cid)
            if p and (y, s, p) in tot.index and v > tot[(y, s, p)] + 0.5:
                viol.append((y, s, cid, v, p, tot[(y, s, p)]))
        md.append(f"- children <= parent: {len(viol)} violations")
        for y, s, cid, v, p, pv in viol[:20]:
            md.append(f"  - {y} {s} {cid}={v:.0f} > {p}={pv:.0f}")
        hard += len(viol) > 0

    if not age.empty and not ser.empty:
        a = age[(age.measure == "count") & (age.age == "Total")] \
            .set_index(["year", "sex", "cause_id"])["value"]
        s = ser[ser.measure == "count"].set_index(["year", "sex", "cause_id"])["value"]
        j = pd.concat([a.rename("by_age"), s.rename("series")], axis=1, join="inner").dropna()
        diff = j[(j.by_age - j.series).abs() > 0.5]
        md.append(f"- je-01/02 Total vs je-03/04 count: {len(j) - len(diff)}/{len(j)} equal")
        known = diff.index.get_level_values("year").isin([1995, 1996, 1997]) & \
                diff.index.get_level_values("cause_id").isin(["all", "alc_cirrh"])
        md.append(f"  - of the differences, known 1995-97 recoding: {known.sum()}, unexpected: {(~known).sum()}")
        hard += int((~known).sum() > 0)
        if len(diff):
            md.append("  - first differences:\n" + diff.head(8).to_markdown())

    (AUDIT / "s02_checks.md").write_text("\n".join(md))
    print("\n".join(md))
    return 1 if hard else 0


if __name__ == "__main__":
    main()
