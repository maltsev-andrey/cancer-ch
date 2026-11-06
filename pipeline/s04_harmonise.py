#!/usr/bin/env python3
"""
cancer-ch / Stage 4: harmonise.

Builds the analysis-ready tables and proves them against BFS before any model runs.

1. Population: mid-year population (mean of 1 Jan and 31 Dec stock) by canton x sex x 5-y age,
   1981-2025, from px-x-0102020000_104; cross-checked against STATPOP px-x-0102010000_102.
2. Language regions: the canton -> region mapping (bilingual BE, FR, VS) is chosen by fitting
   the population that NKRS itself implies (cases / rate) for each region and period.
3. Validation: our crude rates and ASR (European standard 1976) against px-x-1403030300_162;
   cause-of-death ASR (je-03/04) against NKRS mortality ASR.
4. Outputs (data/processed/):
     population_5y.parquet        year x region x sex x age5, mid-year population
     apc_incidence_grid.parquet   region x site x sex x period x age5: cases, person-years, rates
     mortality_icd10.parquet      annual 1970-2024 cause x sex, published + ICD-10 equivalent
   and data/audit/s04_checks.md; exit 1 on a hard failure.
"""
import itertools
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
PX = DATA / "interim" / "pxweb"
MORT = DATA / "interim" / "mortality"
OUT = DATA / "processed"
AUDIT = DATA / "audit"

# European standard population 1976 (per 100,000), 18 groups + 85+
ESP1976 = {"0-4": 8000, "5-9": 7000, "10-14": 7000, "15-19": 7000, "20-24": 7000,
           "25-29": 7000, "30-34": 7000, "35-39": 7000, "40-44": 7000, "45-49": 7000,
           "50-54": 7000, "55-59": 6000, "60-64": 5000, "65-69": 4000, "70-74": 3000,
           "75-79": 2000, "80-84": 1000, "85+": 1000}
AGE5 = list(ESP1976)
LATIN_FIXED = ["Genève", "Vaud", "Neuchâtel", "Jura", "Ticino"]   # always Westschweiz + Tessin
BILINGUAL = ["Bern", "Fribourg", "Valais"]                         # fitted
SEX = {"Mann": "M", "Frau": "F", "Geschlecht - Total": "T"}
R_CH, R_DE, R_LAT = "Schweiz", "Deutschschweiz", "Westschweiz und Tessin"

# tolerances
TOL_POP_SOURCES = 0.0005     # STATPOP vs demographic balance, 31 Dec stock
TOL_CRUDE = 0.01             # our crude rate vs NKRS
TOL_ASR_WEIGHTS = 0.005      # ESP1976 applied to NKRS age rates vs NKRS ASR
TOL_ASR = 0.02               # our ASR vs NKRS


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_px(cube):
    d = pd.read_parquet(PX / f"{cube}.parquet")
    return d.drop(columns=[c for c in d if c.endswith("_code")])


def age5(label):
    """'0 Jahre' / '1 Jahr' / '99 Jahre und mehr' / '100 Jahre und mehr' -> 5-y group."""
    m = re.match(r"^(\d+)", str(label))
    if not m:
        return None                       # 'Alter - Total', 'Ohne Angabe'
    a = int(m.group(1))
    return "85+" if a >= 85 else f"{a // 5 * 5}-{a // 5 * 5 + 4}"


def canton_key(name):
    return str(name).split(" / ")[0].strip()


# ============================================================ population ====
def build_population():
    d = load_px("px-x-0102020000_104")
    d = d[d.Geschlecht.isin(["Mann", "Frau"])].copy()
    d["sex"] = d.Geschlecht.map(SEX)
    d["year"] = d.Jahr.astype(int)
    d["age5"] = d.Alter.map(age5)
    unknown_age = d[d.age5.isna() & (d.Alter != "Alter - Total")]
    d = d[d.age5.notna()]
    comp = {"Bestand am 1. Januar": "jan1", "Bestand am 31. Dezember": "dec31"}
    d = d[d["Demografische Komponente"].isin(comp)]
    d["comp"] = d["Demografische Komponente"].map(comp)
    g = d.groupby(["year", "Kanton", "sex", "age5", "comp"], observed=True)["value"].sum() \
         .unstack("comp").reset_index()
    # BFS convention in the demographic balance: every component of year y, the 1 January
    # stock included, is tabulated by the age reached on 31 December of year y. So jan1 at
    # label a is NOT the completed-age-a population on 1 January (jan1 at '0 Jahre' is 0).
    # Person-years at completed age a = mean of the 31 Dec stocks of y-1 and y, both
    # tabulated by completed age at their own date.
    g = g.sort_values(["Kanton", "sex", "age5", "year"])
    g["dec31_prev"] = g.groupby(["Kanton", "sex", "age5"], observed=True).dec31.shift(1)
    g = g[g.dec31_prev.notna()]                       # drops 1981
    g["midyear"] = (g.dec31_prev + g.dec31) / 2
    g["canton"] = g.Kanton.map(canton_key)
    ua = unknown_age.groupby("Demografische Komponente")["value"].sum().to_dict()
    return g.drop(columns="Kanton"), ua


def check_population(pop, md):
    hard = 0
    # 1. cantons + 'Ohne Angabe' sum to Schweiz
    ch = pop[pop.canton == "Schweiz"].groupby("year").dec31.sum()
    ct = pop[pop.canton != "Schweiz"].groupby("year").dec31.sum()
    dev = ((ct - ch) / ch).abs().max()
    md.append(f"- cantons sum to Schweiz (31 Dec, all years): max rel. diff {dev:.2e}")
    hard += dev > 1e-4                      # a lost canton would be >= 0.3%
    # 2. age convention: jan1 at '0-4' must be ~4/5 of dec31 (age 0 is empty on 1 Jan),
    #    and jan1 totals equal the previous 31 Dec totals (same people, shifted labels)
    w = pop[pop.canton == "Schweiz"].groupby("year")[["jan1", "dec31", "dec31_prev"]].sum()
    cont = ((w.dec31_prev - w.jan1) / w.jan1).abs().max()
    y0 = pop[(pop.canton == "Schweiz") & (pop.age5 == "0-4")].groupby("year")[["jan1", "dec31"]].sum()
    md.append(f"- 1 Jan stock = previous 31 Dec stock (totals): max rel. diff {cont:.2e}; "
              f"1 Jan / 31 Dec at age 0-4: {(y0.jan1 / y0.dec31).mean():.3f} "
              f"(0.8 expected: the balance tabulates 1 Jan by age reached on 31 Dec, age 0 empty)")
    hard += int(abs((y0.jan1 / y0.dec31).mean() - 0.8) > 0.03)
    # 3. STATPOP cube, 31 Dec stock, 2010-2025
    s = load_px("px-x-0102010000_102")
    s = s[(s.Kanton == "Schweiz") & s.Geschlecht.isin(["Mann", "Frau"]) & (s.Alter != "Alter - Total")]
    s = s.groupby(s.Jahr.astype(int))["value"].sum()
    j = pd.concat([w.dec31.rename("balance"), s.rename("statpop")], axis=1, join="inner")
    j["rel"] = (j.balance / j.statpop - 1)
    md.append(f"- demographic balance vs STATPOP, 31 Dec, {j.index.min()}-{j.index.max()}: "
              f"max rel. diff {j.rel.abs().max():.2e}")
    if j.rel.abs().max() > TOL_POP_SOURCES:
        md.append(j[j.rel.abs() > TOL_POP_SOURCES].to_markdown())
        hard += 1
    return hard


# ============================================================ regions =======
def nkrs_implied_population():
    """NKRS counts / NKRS crude rate -> the population NKRS divided by, per region/sex/period."""
    d = load_px("px-x-1403030300_161")
    d = d[d.Krebsart.str.startswith("Alle") & d.Altersklasse.str.contains("Total") &
          (d.Indikator == "Inzidenz") & d.Geschlecht.isin(["Mann", "Frau"])]
    w = d.pivot_table(index=["Sprachregion", "Geschlecht", "Periode"], columns="Kennzahl",
                      values="value").reset_index()
    cnt = [c for c in w if str(c).startswith("Anzahl")][0]
    rate = [c for c in w if str(c).startswith("Alters")][0]
    w["implied_pop"] = w[cnt] / w[rate] * 1e5
    w["sex"] = w.Geschlecht.map(SEX)
    return w[["Sprachregion", "sex", "Periode", "implied_pop"]]


def region_of(canton, latin_bilingual):
    if canton in LATIN_FIXED or any(canton.startswith(b) for b in latin_bilingual):
        return R_LAT
    return R_DE


def period_pop(pop, mapping):
    """Mean mid-year population per region x sex x NKRS period x age5."""
    p = pop[(pop.canton != "Schweiz") & (pop.canton != "Ohne Angabe")].copy()
    p["region"] = p.canton.map(lambda c: region_of(c, mapping))
    p = pd.concat([p, p.assign(region=R_CH)])
    p["period"] = p.year.map(lambda y: next((f"{s}-{s + 4}" for s in range(1993, 2023, 5)
                                             if s <= y <= s + 4), None))
    p = p[p.period.notna()]
    # sum cantons within a year first, then average the years of the period
    yearly = p.groupby(["region", "sex", "period", "year", "age5"], observed=True)["midyear"].sum()
    return yearly.groupby(["region", "sex", "period", "age5"], observed=True).mean().reset_index()


def fit_regions(pop, md):
    imp = nkrs_implied_population()
    best = None
    rows = []
    for k in range(len(BILINGUAL) + 1):
        for combo in itertools.combinations(BILINGUAL, k):
            pp = period_pop(pop, combo).groupby(["region", "sex", "period"])["midyear"].sum()
            j = imp.set_index(["Sprachregion", "sex", "Periode"])["implied_pop"]
            j.index.names = ["region", "sex", "period"]
            e = ((pp / j - 1).abs()).dropna()
            e = e[e.index.get_level_values("region") != R_CH]
            rows.append({"latin_bilingual": "+".join(combo) or "none",
                         "mean_abs_err_pct": e.mean() * 100, "max_abs_err_pct": e.max() * 100})
            if best is None or e.mean() < best[1]:
                best = (combo, e.mean())
    t = pd.DataFrame(rows).sort_values("mean_abs_err_pct")
    md.append("- canton -> language region: fitted against the population NKRS implies "
              "(cases / crude rate), all candidate mappings:")
    md.append(t.round(3).to_markdown(index=False))
    md.append(f"- chosen: bilingual cantons in '{R_LAT}': {list(best[0]) or 'none'}")
    hard = int(best[1] > 0.01)                   # best mapping must reproduce NKRS within 1%
    return best[0], hard


# ============================================================ incidence =====
def build_grid(pop, mapping):
    d = load_px("px-x-1403030300_161")
    d = d[~d.Altersklasse.str.contains("Total") & ~d.Krebsart.str.startswith(" ")]
    d["sex"] = d.Geschlecht.map(SEX)
    d["measure"] = np.where(d.Kennzahl.str.startswith("Anzahl"), "cases_per_year", "rate_nkrs")
    w = d.pivot_table(index=["Sprachregion", "Krebsart", "sex", "Periode", "Altersklasse", "Indikator"],
                      columns="measure", values="value", dropna=False).reset_index()
    w = w.rename(columns={"Sprachregion": "region", "Krebsart": "site", "Periode": "period",
                          "Altersklasse": "age5", "Indikator": "indicator"})
    pp = period_pop(pop, mapping)
    tot = pp.groupby(["region", "period", "age5"], as_index=False)["midyear"].sum().assign(sex="T")
    pp = pd.concat([pp, tot]).rename(columns={"sex": "pop_sex"})
    # sites with cases for one sex only (prostate, testis, cervix, corpus, ovary): NKRS divides
    # the 'Total' row by the population of the sex at risk, not by both sexes
    has = w[w.sex.isin(["M", "F"])].groupby(["site", "sex"]).cases_per_year \
           .apply(lambda s: s.notna().any()).unstack()
    single = {site: ("M" if r.get("M") else "F") for site, r in has.iterrows()
              if bool(r.get("M")) != bool(r.get("F"))}
    w["pop_sex"] = np.where((w.sex == "T") & w.site.isin(single),
                            w.site.map(single), w.sex)
    w = w.merge(pp, on=["region", "pop_sex", "period", "age5"], how="left")
    w["person_years"] = w.midyear * 5
    w["cases"] = w.cases_per_year * 5
    w["rate_ours"] = w.cases_per_year / w.midyear * 1e5
    w["indicator"] = w.indicator.map({"Inzidenz": "incidence", "Mortalität": "mortality"})
    w["structural_zero"] = w.cases_per_year.isna()
    return w.drop(columns=["midyear", "pop_sex"])


def asr(df, rate_col):
    x = df.assign(w=df.age5.map(ESP1976))
    x = x[x[rate_col].notna()]
    return (x[rate_col] * x.w).sum() / x.w.sum()


def validate_rates(grid, md):
    ref = load_px("px-x-1403030300_162")
    ref = ref.assign(sex=ref.Geschlecht.map(SEX),
                     indicator=ref.Indikator.map({"Inzidenz": "incidence", "Mortalität": "mortality"}))
    kz = {"Rohe Rate": "crude", "Europa-standardisierte Rate (1976)": "asr_e76"}
    ref = ref[ref.Kennzahl.isin(kz)].assign(k=lambda d: d.Kennzahl.map(kz)) \
             .pivot_table(index=["Sprachregion", "Krebsart", "sex", "Periode", "indicator"],
                          columns="k", values="value")
    ref.index.names = ["region", "site", "sex", "period", "indicator"]

    rows = []
    for key, g in grid.groupby(["region", "site", "sex", "period", "indicator"]):
        if g.cases_per_year.isna().all():
            continue
        rows.append({**dict(zip(["region", "site", "sex", "period", "indicator"], key)),
                     "crude_ours": g.cases_per_year.sum() / g.person_years.sum() * 5e5,
                     "asr_nkrs_rates": asr(g, "rate_nkrs"),
                     "asr_ours": asr(g, "rate_ours"),
                     "cases_per_year": g.cases_per_year.sum()})
    v = pd.DataFrame(rows).set_index(["region", "site", "sex", "period", "indicator"]).join(ref)
    v["d_crude"] = v.crude_ours / v.crude - 1
    v["d_weights"] = v.asr_nkrs_rates / v.asr_e76 - 1
    v["d_asr"] = v.asr_ours / v.asr_e76 - 1
    # the cube stores rates to 0.1 and counts to 1: on a rate of 2.0 that alone is 2.5%.
    # Hard criterion only where rounding is negligible (ASR >= 20, i.e. <= 0.25%).
    big = v[(v.cases_per_year >= 100) & (v.asr_e76 >= 20)]
    small = v[(v.cases_per_year >= 100) & (v.asr_e76 < 20)]

    hard = 0
    md.append(f"- rate validation vs px-x-1403030300_162: {len(v)} cells, {len(big)} with >=100 cases/yr "
              f"and ASR >= 20 (hard), {len(small)} with >=100 cases/yr and ASR < 20 (info, "
              f"median |ASR diff| {small.d_asr.abs().median() * 100:.2f}%, rounding-dominated):")
    for col, tol, what, is_hard in (
            ("d_weights", TOL_ASR_WEIGHTS, "ESP1976 on NKRS age rates vs NKRS ASR (info: cube rates "
                                           "are rounded to 0.1)", False),
            ("d_crude", TOL_CRUDE, "our crude rate vs NKRS crude", True),
            ("d_asr", TOL_ASR, "our ASR vs NKRS ASR", True)):
        a = big[col].abs()
        share = (a <= tol).mean() * 100
        md.append(f"  - {what}: median |diff| {a.median() * 100:.2f}%, p95 {a.quantile(.95) * 100:.2f}%, "
                  f"within {tol * 100:.1f}%: {share:.1f}%")
        hard += int(is_hard and share < 95)
    key = v.loc[(R_CH, "Alle Krebsarten (total)")].reset_index()
    key = key[key.period == "2018-2022"][["sex", "indicator", "crude", "crude_ours", "asr_e76", "asr_ours"]]
    md.append("- headline check, Switzerland, all cancers, 2018-2022:")
    md.append(key.round(1).to_markdown(index=False))
    x = grid[(grid.region == R_CH) & (grid.site == "Alle Krebsarten (total)") &
             (grid.period == "2018-2022") & (grid.sex == "M") & (grid.indicator == "incidence")].copy()
    x["pop_ours"] = x.person_years / 5
    x["pop_nkrs"] = x.cases_per_year / x.rate_nkrs * 1e5
    x["ratio"] = x.pop_ours / x.pop_nkrs
    x = x.sort_values("age5", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    md.append("- age profile of denominators, Switzerland, men, 2018-2022 (ratio must be ~1 at all ages):")
    md.append(x[["age5", "rate_nkrs", "rate_ours", "pop_ours", "pop_nkrs", "ratio"]]
              .round(3).to_markdown(index=False))
    worst = big.reindex(big.d_asr.abs().sort_values(ascending=False).index).head(8)
    md.append("- largest ASR differences among the hard cells:")
    md.append(worst[["cases_per_year", "asr_e76", "asr_ours", "d_asr"]].round(3).to_markdown())
    return v.reset_index(), hard


# ============================================================ mortality =====
def build_mortality(md):
    s = pd.read_parquet(MORT / "mort_series.parquet")
    b = pd.read_parquet(MORT / "icd_bridge.parquet")[["cause_id", "sex", "year", "factor", "method"]]
    s = s.merge(b, on=["cause_id", "sex", "year"], how="left")
    s["factor"] = np.where(s.year <= 1994, s.factor, 1.0)
    s["bridged"] = s.factor.notna()
    s["value_icd10"] = s.value * s.factor.fillna(1.0)
    with_bridge = sorted(s[s.factor.notna() & (s.year <= 1994)].cause_id.unique())
    md.append(f"- mortality: {s.year.min()}-{s.year.max()}; ICD bridge applied before 1995 to "
              f"{with_bridge}; all other causes kept as published (bridged=False)")
    return s


def validate_mortality(mort, md):
    """Cause-of-death ASR (je-03/04, 5-year mean) vs NKRS mortality ASR (Europe 1976)."""
    ref = load_px("px-x-1403030300_162")
    ref = ref[(ref.Sprachregion == R_CH) & (ref.Indikator == "Mortalität") &
              (ref.Kennzahl == "Europa-standardisierte Rate (1976)") & ref.Geschlecht.isin(["Mann", "Frau"])]
    site = {"cancer": "Alle Krebsarten (total)", "lung": "Lunge, Bronchien, Luftröhre",
            "stomach": "Magen", "breast": "Brust", "prostate": "Prostata",
            "cervix": "Gebärmutterhals", "colon": "Dickdarm"}
    m = mort[mort.measure == "rate"].copy()
    m["period"] = m.year.map(lambda y: next((f"{s}-{s + 4}" for s in range(1993, 2023, 5)
                                             if s <= y <= s + 4), None))
    m = m[m.period.notna() & m.cause_id.isin(site)]
    m5 = m.groupby(["cause_id", "sex", "period"])["value_icd10"].mean().reset_index()
    rows = []
    for r in m5.itertuples():
        x = ref[(ref.Krebsart == site[r.cause_id]) & (ref.Geschlecht.map(SEX) == r.sex) &
                (ref.Periode == r.period)]
        if len(x):
            rows.append({"cause": r.cause_id, "sex": r.sex, "period": r.period,
                         "cod_asr": r.value_icd10, "nkrs_asr": x.value.iloc[0]})
    v = pd.DataFrame(rows)
    v["diff_pct"] = (v.cod_asr / v.nkrs_asr - 1) * 100
    md.append("- cause-of-death ASR (5-y mean) vs NKRS mortality ASR, % difference by cause (info). "
              "Two independent systems; 'colon' differs by definition: CoD Dickdarm = C18 only, "
              "NKRS Dickdarm = C18-C20:")
    md.append(v.groupby(["cause", "sex"]).diff_pct.agg(["mean", "min", "max"]).round(2).to_markdown())
    return v


# ================================================================== main ====
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, hard = [f"# s04 checks ({datetime.now():%Y-%m-%d %H:%M})", ""], 0

    pop, unknown_age = build_population()
    md.append(f"- population 'Ohne Angabe' age (excluded): {unknown_age}")
    hard += check_population(pop, md)

    mapping, h = fit_regions(pop, md)
    hard += h
    p5 = period_pop(pop, mapping)
    yr = pop[(pop.canton != "Schweiz") & (pop.canton != "Ohne Angabe")].copy()
    yr["region"] = yr.canton.map(lambda c: region_of(c, mapping))
    yr = pd.concat([yr, yr.assign(region=R_CH)]) \
           .groupby(["year", "region", "sex", "age5"], observed=True)["midyear"].sum().reset_index()
    yr.to_parquet(OUT / "population_5y.parquet", index=False)

    grid = build_grid(pop, mapping)
    grid.to_parquet(OUT / "apc_incidence_grid.parquet", index=False)
    v, h = validate_rates(grid, md)
    hard += h
    v.to_parquet(OUT / "validation_rates.parquet", index=False)

    mort = build_mortality(md)
    mort.to_parquet(OUT / "mortality_icd10.parquet", index=False)
    validate_mortality(mort, md).to_parquet(OUT / "validation_mortality.parquet", index=False)

    md.append(f"- outputs: population_5y {len(yr):,}, apc_incidence_grid {len(grid):,}, "
              f"mortality_icd10 {len(mort):,} rows")
    (AUDIT / "s04_checks.md").write_text("\n".join(md))
    print("\n".join(md))
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
