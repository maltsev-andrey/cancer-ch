# 00 · Overview

**cancer-ch**: cancer incidence and mortality in Switzerland, rebuilt from official public data
and modelled on the home HPC cluster.

| | |
|---|---|
| Status | Data layer (s00–s04) and APC model sweep (s06–s07) complete, 2026-11-06 |
| Next | public README and repository; batched GPU model; incidence × mortality joint cohort model |
| Environment | Python 3.11, pandas 3.x, NumPyro (conda-forge or venv) |
| Code | this repository |
| Data | `../data` next to the repository (not in git; `CANCER_CH_DATA` overrides) |

## Goal

Describe how cancer incidence and mortality in Switzerland have changed by site, sex and age,
and separate the three time effects behind those trends:

- **age:** risk rises with age;
- **period:** something changes for everybody at once (screening, treatment, coding rules);
- **cohort:** a generation carries its exposure with it (smoking is the textbook case).

The main scientific target is lung cancer, where men and women follow opposite paths:
the female rise is a pure cohort effect peaking for women born 1953–1957
(see [08 Results](/books/cancer-ch/page/08-results)).

## Scope and non-goals

- **Switzerland and two language regions only.** No public source publishes cancer incidence or
  cause-specific mortality by canton (see [01 Data sources](01_data_sources.md)).
- **Aggregated public data only.** No individual records. The BFS order form for microdata
  exists but is out of scope for a public repository.
- **No causal claims from correlations.** An early idea to correlate cantonal PM2.5 or smoking with
  cancer was dropped: 26 cantons, a 15–30 year latency between exposure and diagnosis, and
  aggregated data make such correlations an ecological fallacy, not evidence.

## Chapters

| # | Chapter | Content |
|---|---|---|
| 00 | Overview | this page |
| 01 | [Data sources](01_data_sources.md) | what exists publicly, what does not |
| 02 | [Data audit (s00)](02_data_audit.md) | how the sources were found and measured |
| 03 | [Ingest (s01)](03_ingest.md) | PxWeb cubes and DAM tables into Parquet |
| 04 | [Mortality parsing (s02)](04_mortality_parsing.md) | two BFS cause-of-death tables, cross-checked |
| 05 | [ICD-8 → ICD-10 bridge (s03)](05_icd_bridge.md) | the 1995 coding break, measured two ways |
| 06 | [Harmonisation (s04)](06_harmonisation.md) | denominators, language regions, validation against NKRS |
| 07 | [APC model (s06)](/books/cancer-ch/page/07-age-period-cohort-model-s06) | Bayesian age-period-cohort model, 214 fits, GPU vs CPU |
| 08 | [Results](/books/cancer-ch/page/08-results) | trends, cohorts, projections to 2030 |
| 09 | [Lessons learned](09_lessons_learned.md) | traps in the BFS APIs and file formats |

## Pipeline

```
s00 audit      DAM catalog + PxWeb discovery -> coverage report
s01 ingest     PxWeb cubes (JSON-stat2) -> Parquet;  DAM XLSX -> raw/
s02 mortality  je-d-14.03.04.01.01..04 -> mort_age / mort_series
s03 apc        su-d-14.03.04.01.11..20 -> apc_period / apc_cohort / apc_asr / icd_bridge
s04 harmonise  mid-year population, language regions, ASR validated against NKRS
s06 apc        Bayesian APC per site x sex x indicator (NumPyro), full + holdout
s07 report     index of all fits, cohort peaks, sweep figures
```

Every stage is idempotent, writes checks to `data/audit/sNN_checks.md` and exits non-zero on a
hard failure. Run with `set -o pipefail` so `| tee` does not hide the exit code.
