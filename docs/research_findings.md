# Research findings

What this project has established, sorted by how much it depends on the model. Every line
carries its evidence class and where to reproduce it. Numbers are per 100,000, age-standardised
to the European standard population 1976 (ESP1976), Switzerland, unless stated otherwise.

| Class | Meaning | Reproduce with |
|---|---|---|
| **A. Observed** | Read directly from BFS/NKRS tables after harmonisation; no statistical model beyond age standardisation | `s02`–`s04`, `scripts/make_figures.py`, `assets/figures_summary.json` |
| **B. Model estimate** | Posterior from the Bayesian APC model fitted to 1993–2022 (incidence) or the mortality grid; 95% intervals | `s06`, `s07`, `assets/sweep_summary.json`, `data/models/apc/<slug>/summary.json` |
| **C. Projection** | Extrapolation of class-B estimates to 2028–2032; assumes no new screening, coding or exposure change | `s06 --future 2`, `fig11` |
| **D. Hypothesis** | An interpretation the data are consistent with but cannot establish | none; listed so they are not mistaken for A–C |
| **R. Rejected** | A hypothesis the data contradict | the test that rejected it |

Status as of v1.1 (sweep of 2026-11-07, 214 fits, all converged under the v1.1 gate).

---

## A. Observed

**A1. Cancer mortality has fallen by close to half.** Men peaked at 244.9 in 1990 and stood at
131.3 in 2024 (−46%); women at 153.6 in 1970 and 93.2 in 2024 (−39%). ICD-10 equivalent series;
the BFS corrected series overlaps ours in every year. `fig02`.

**A2. Lung cancer runs in opposite directions by sex.** Men: mortality 57.1 (1970) → 64.0
(1990) → 26.5 (2024); incidence 66.9 (1993–97) → 45.9 (2018–22). Women: mortality 4.9 (1970)
→ 18.7 (2010) → 16.9 (2024); incidence 18.8 (1993–97) → 32.2 (2018–22). `fig04`, `fig10`.

**A3. Frequency and lethality are different rankings.** 2018–2022 mortality-to-incidence
ratio: pancreas 0.82, brain 0.81, liver 0.73, oesophagus 0.72, lung 0.64; thyroid 0.07, melanoma
0.08, prostate 0.16, breast 0.20. Prostate and breast are the most frequent cancers; lung kills
the most. `fig05`.

**A4. The ICD-8 → ICD-10 break (1994/95) is real and cause-specific.** Bridge factors from the
BFS APC tables: colorectal 0.85, prostate 0.86, breast 0.89, all cancers 0.93, cervix 0.95,
lung 0.96, stomach 1.00; two independent methods agree to 0.001. Any 1970–2024 series that
ignores this overstates the pre-1995 level for colorectal, prostate and breast by 11–15%.
`s03`, `fig03`.

**A5. The two mortality systems agree.** Cause-of-death ASR (BFS) and registry mortality ASR
(NKRS) differ by −0.4% (all cancers), −0.3% (lung), +0.4% (stomach), −0.7% (prostate) over
2018–2022. `s04 validate_mortality`.

**A6. Our rates reproduce the registry's.** Crude 557.0 / 617.9 / 497.1 and ASR 384.6 / 440.6
/ 339.2 (both / men / women, all cancers, 2018–22) match NKRS exactly; 419 cells with ≥100
cases/yr are within 1% (crude) and 2% (ASR). `s04 validate_rates`.

**A7. The public data has no cantonal incidence.** The BFS API carries Switzerland and two
language regions only. Cantonal registries publish incidence as PDF reports (Zürich since 1980,
Zug since 2011, most others since 2020). Cantonal cause-of-death *is* in the API. `s00`.

---

## B. Model estimates

Net drift = the identifiable linear trend of period and cohort combined, % per year, with 95%
interval. Cohort RR = relative risk of a birth cohort versus the linear trend (curvature only);
a "peak cohort" is the generation most above its trend line, not the one with the highest
absolute rate. Fit identifiers are `<site>__<sex>__<indicator>`.

**B1. All-cancer incidence is flat; all-cancer mortality falls 2–2.5% a year.** Incidence drift
M −0.19, F +0.36; mortality drift M −2.47, F −1.77. `alle-krebsarten-total__*`.

**B2. Lung cancer in women has no trend; its rise is a cohort effect.** Incidence drift
F −0.05 (interval straddles 0), M −2.31. Cohort RR peaks for women born 1953–1957 at 1.94
(interval entirely above 1.5) and for men born 1953–1957 at 1.29; every cohort after 1960 sits
below its trend line in both sexes. Mortality drift M −3.93, F −2.06. `lunge-bronchien-luftroehre__*`, `fig10`.

**B3. The 1950s cohort peaks at every tobacco- and alcohol-related site at once.** Peak cohort
1953–1957 for lung (M, F), oesophagus (M, RR 1.42), liver (M, 1.17), non-Hodgkin lymphoma
(F, 1.09); 1958–1962 for NHL (M, 1.15). `fig08`, `sweep_summary.cohort_peaks`.

**B4. Melanoma has its own generation.** Peak cohort 1973–1977 in both sexes, RR 1.74 (M) and
1.86 (F), the second-largest cohort effect in the sweep after female lung. Net drift +0.81 /
+0.60. `hautmelanom__*`.

**B5. Bladder cancer's generation is earlier.** Peak cohort 1943–1947 (M, RR 1.12); drift
M −1.51, F −0.68. `harnblase__*`.

**B6. Screening-type signatures.** Incidence rises while mortality falls for colorectum
(+1.68 / +1.56 vs −1.91 / −1.78), breast (+1.08 vs −2.50) and prostate (+1.13 vs −2.95). The
model does not attribute cause; this is the pattern screening and earlier detection would
produce. `dickdarm__*`, `brust__F__*`, `prostata__M__*`.

**B7. Fastest rises.** Thyroid +2.83 (M) / +2.37 (F), small intestine +1.97 (M), liver +1.72
(F), pancreas +1.67 (F) / +1.02 (M). Thyroid's mortality drift is −3.26 (F): more diagnoses,
fewer deaths. `fig07`.

**B8. Fastest falls.** Larynx −3.20 (M), cervix −2.84, lung −2.31 (M), other sites −1.92 (M),
oesophagus −1.74 (M), bladder −1.51 (M), stomach −1.25 (M). `fig07`.

**B9. No measurable registration jump in 2020.** Fits on 1993–2017 predict 2018–2022 with a
median error of +2.4% (incidence) and +2.7% (mortality) over 87 fits with ≥100 cases/yr; 36%
of incidence errors are negative. Both indicators came in slightly *below* trend. An incidence
inflation from the Cancer Registration Act would show as incidence above trend with mortality
on trend. `fig09`, `sweep_summary.holdout`.

**B10. Holdout error scales with count, not with site.** Fits with ≥1000 cases/yr are within
±8% out of sample; fits with 100–200 cases/yr scatter over ±20–30% (female melanoma mortality
+33%, female myeloma mortality +23%, male gallbladder mortality −22%). This is sampling noise
on 2020–2022 counts, not a model failure, and sets the floor for what rolling-origin
backtesting can resolve. `fig09`, left panel.

**B11. The model recovers known truth.** On a synthetic 18 × 6 grid with drift −1.5%/yr and a
cohort bump at 1968 it returns −1.48 (−1.60…−1.35) and 1968; run as `pytest -m slow`.

**B12. The estimates are robust to sampler settings.** Between the v1.0 sweep (4 × 1000 draws,
target acceptance 0.9) and v1.1 (4 × 2000, 0.99, stricter gate) no drift moved by more than
0.1 %/yr and no cohort peak moved. `docs/wiki/07_apc_model.md`.

**B13. The 2018–2022 shift is period-specific, not a model bias.** Rolling-origin backtest:
models trained to 2007 and 2012 predict 2008–2017 with medians −2.5…0% (observed at or
slightly above trend); every prediction of 2018–2022, from origins 2007, 2012 and 2017, is
positive: incidence −0.2 / +0.7 / +2.4, mortality +2.6 / +2.0 / +2.7 (n = 37–50 per cell).
`s08_backtest`, `fig12`, `backtest_summary.json`.

**B14. Projection error doubles per five-year horizon.** Median absolute deviation of the ASR
error: 4–6% one period ahead, 7–8% at two, 12–15% at three; share of fits within ±10%:
74–86% → 53–65% → 38–42%. A three-period training window cannot separate a trend from a
cohort effect: trained to 2007, female lung incidence reads the cohort rise as +2.4%/yr
drift and overshoots 2018–2022 by 9%. The 2028–2032 projections (C) are horizon 2.
`s08_backtest`.

---

## C. Projections, 2018–2022 → 2028–2032

Extrapolation of the fitted drift and random-walk curvatures. Intervals widen with horizon and
with small counts; the ones below are the medians, and most intervals include a flat outcome.

| Rising | | Falling | |
|---|---|---|---|
| Thyroid | +31% M, +18% F | Larynx (M) | −30% |
| Small intestine (M) | +23% | Cervix uteri | −24% |
| Pancreas | +15% M, +18% F | Lung (M) | −21% |
| Liver (F) | +18% | Oesophagus (M) | −19% |
| Melanoma (M) | +15% | Oral cavity (M) | −17% |
| Breast | +9% | Ovary | −13% |

**C1.** Female lung incidence turns down (−4%) as the post-1960 cohorts replace the 1950s
cohort at the ages where lung cancer occurs. Lung (M) continues to fall.

**C2.** Thyroid, pancreas and liver (F) keep rising. For thyroid the mortality trend (−3.3%/yr)
says the rise is diagnostic rather than lethal; for pancreas (M:I 0.82) it is not.

What a projection assumes: no new screening programme, no coding revision, no change in
exposure beyond what the cohorts already carry, and COVID-19 as a one-off. These hold for five
years more often than not, and never for twenty.

---

## D. Hypotheses the data are consistent with

**D1.** The 1953–1957 cohort peak across lung, oesophagus, liver and NHL reflects the smoking
and drinking habits of Swiss adults who were 20 in the mid-1970s, the top of Swiss tobacco
consumption. The data show the cohort; they do not show the exposure.

**D2.** The 1973–1977 melanoma cohort reflects childhood UV exposure in the 1980s (package
holidays, no sun-protection culture yet). Same caveat.

**D3.** The below-trend 2018–2022 (+2.4 / +2.7% at horizon 1, and from every earlier origin,
B13) is the COVID-19 years: fewer diagnoses in 2020–21 and, for mortality, competing deaths.
The backtest rules out the alternative (a model that always overshoots), it does not identify
the cause. The next NKRS release (2023–27 period) can: if 2018–22 was a dip, the next period
should come in above trend.

**D4.** Female liver incidence +1.7%/yr with projection +18% is worth a closer look at metabolic
causes, since hepatitis C treatment and alcohol trends would predict the opposite. Not testable
with this data.

---

## R. Rejected

**R1. "Women's lung cancer peaks one generation after men's."** Both sexes peak in the same
cohort, 1953–1957 (B2). The delay is in the *age curve* and the *absolute level*, not in the
cohort: women of that generation smoked less and started later, so their rates are lower and
their peak ages are reached later in calendar time, but the generation is the same.

**R2. "The Cancer Registration Act inflated incidence from 2020."** Holdout error is the same
sign and size for incidence and mortality (B9); an incidence-only shift is not there.

**R3. "Cancer is getting more common."** Age-standardised incidence is flat (B1) and mortality
is down by 39–46% (A1). The rise in *counts* is ageing and population growth.

**R4. "A GPU speeds up this model."** The same fit took 694 s on a Tesla P100 and ~20 s on
four CPU cores. 108 data cells give NUTS nothing to parallelise. A batched all-sites model
would change that; the per-site model does not.

---

## Open questions (tracked as v1.1 issues)

1. ~~Rolling-origin backtesting~~: answered, B13–B14. The shift is tied to 2018–2022 at
   every origin; it is the years, not the model.
2. **Prior sensitivity.** Do drift and cohort RR hold when the random-walk and drift scales are
   varied ×0.3 and ×3? Ten key sites.
3. **Posterior predictive by age.** Is predictive coverage uniform across age groups, or is the
   free age vector overfitting the oldest cells?
4. **Cantonal mortality.** Cause-of-death by canton 1990–2024 is in the API; a BYM2 spatial
   model would show whether the national trends are uniform across cantons.

---

## How to cite a line from this file

Give the class and the identifier: "cancer-ch B2 (lunge-bronchien-luftroehre__F__incidence,
v1.1)". The identifier resolves to `data/models/apc/<slug>/summary.json`, which records the
data, settings, seed and diagnostics of that fit.
