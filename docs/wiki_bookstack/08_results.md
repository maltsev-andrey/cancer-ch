# 08 · Results

All rates per 100,000, age-standardised to the European standard population (1976).
Descriptive results come from the data layer; trends, cohorts and projections from the APC
model (chapter 07). Switzerland as a whole, 1993–2022 for the model, 1970–2024 for mortality.

## 1. Cancer mortality fell by 40–46% since its peak

![Cancer mortality](fig02_cancer_mortality_asr.png)

| | Peak | 2024 | Change |
|---|---|---|---|
| Men | ≈260 in the mid-1980s (BFS APC series); 244.9 in 1990 | 131.3 | −46% vs 1990 |
| Women | 153.6 in 1970 | 93.2 | −39% |

The ICD-bridged series and BFS' own corrected series overlap in every year where both exist.

## 2. Incidence is flat overall, mortality keeps falling

![Net drift by site](fig07_drift_by_site.png)

Net drift from the APC model, incidence 1993–2022, % per year:

| Rising | | Falling | |
|---|---|---|---|
| Thyroid | +2.8 M / +2.4 F | Larynx (M) | −3.2 |
| Small intestine (M) | +2.0 | Cervix uteri | −2.8 |
| Colorectum | +1.7 M / +1.6 F | Lung (M) | −2.3 |
| Pancreas | +1.0 M / +1.7 F | Oesophagus (M) | −1.7 |
| Liver (F) | +1.7 | Bladder (M) | −1.5 |
| Prostate | +1.1 | Stomach | −1.3 M / −0.7 F |
| Breast | +1.1 | Ovary | −1.1 |
| Melanoma | +0.8 M / +0.6 F | | |

All cancers: **−0.2%/yr in men, +0.4%/yr in women** for incidence; **−2.5% and −1.8%/yr** for
mortality. Colorectal, breast and prostate incidence rise while their mortality falls
(−1.9, −2.5, −3.0%/yr): the signature of screening and earlier detection. Thyroid is the
textbook overdiagnosis case. Pancreas is one of the few sites where mortality rises too.

## 3. Lung cancer: the smoking generations

![Lung cohorts](fig10_lung_cohorts.png)

| | Incidence drift | Mortality drift | Peak cohort | RR vs trend |
|---|---|---|---|---|
| Men | −2.3%/yr | −3.9%/yr | 1953–1957 | 1.29 |
| Women | −0.1%/yr | −2.1%/yr | 1953–1957 | **1.94** |

- **Women:** no linear trend at all. The rise from 18.8 (1993–97) to 32.2 (2018–22) is entirely a
  cohort effect, peaking for women born 1953–1957 and falling for every later generation.
  The model projects female incidence to plateau and turn down (−4% by 2028–32).
- **Men:** the 1950s cohorts also sit above the trend (RR 1.29), but on top of a steep decline
  of −2.3%/yr. The absolute male peak (cohorts around 1900–1920) lies outside the 1993–2022
  observation window and cannot be seen by this model.
- Female mortality falls at −2.1%/yr while incidence is flat: survival is improving.
- The initial hypothesis that the female peak comes a generation after the male one is **not**
  supported in curvature terms: both peak in the same 5-year cohort.

## 4. Which generation carries the highest risk

![Cohort peaks](fig08_cohort_peaks.png)

Sites with a clear interior cohort peak (lower 95% bound of RR above 1):

| Peak cohort | Sites |
|---|---|
| 1943–1947 | bladder (M) |
| 1953–1957 | lung (M, F), oesophagus (M), liver (M), non-Hodgkin lymphoma (F) |
| 1958–1962 | non-Hodgkin lymphoma (M) |
| 1973–1977 | skin melanoma (M RR 1.74, F RR 1.86) |

The 1950s generation shows up at every tobacco- and alcohol-related site at once. Melanoma has
its own generation, born in the 1970s.

## 5. Where incidence is heading by 2030

![Projection](fig11_projection_2030.png)

Projected change in ASR, 2018–2022 → 2028–2032 (median; intervals in the figure):

| Rising | | Falling | |
|---|---|---|---|
| Thyroid | +31% M / +18% F | Larynx (M) | −30% |
| Small intestine (M) | +22% | Cervix uteri | −24% |
| Pancreas | +15% M / +18% F | Lung (M) | −21% |
| Liver (F) | +18% | Oesophagus (M) | −19% |
| Melanoma (M) | +15% | Oral cavity (M) | −17% |
| Breast | +9% | Ovary | −13% |
| All cancers | −2% M / +3% F | Pleural mesothelioma (M) | −12% |

Projections extrapolate the fitted drift and curvatures; they assume no new screening programme,
no new coding change and no new exposure. Intervals widen with the horizon and with small counts.

## 6. Out-of-sample check and the registration act

![Holdout](fig09_holdout_krg.png)

Fits on 1993–2017 predict 2018–2022 with a median error of +2.5% (incidence) and +2.9%
(mortality) over 87 fits. Both indicators came in slightly below trend; there is no
incidence-only jump that the Cancer Registration Act (2020) would have produced.

![Rolling-origin backtest](fig12_backtest.png)

Repeating the test from origins 2007 and 2012 (chapter 07) shows that the shift belongs to the
years 2018–2022 and not to the model: predictions of 2008–2017 from the earlier origins are
unbiased or slightly negative, while every prediction of 2018–2022 comes in positive, +0.7…+2.4%
for incidence and +2.0…+2.7% for mortality. The size is modest and the mortality signal is the
clearer one, which is what fewer diagnoses in 2020–2021 and competing causes of death would
produce. The backtest also calibrates the projection in section 5: the median absolute error is
about 8% two periods ahead, which is the horizon of the 2028–2032 figures.

## 7. Frequency is not lethality (2018–2022)

![Incidence vs mortality](fig05_incidence_vs_mortality.png)

- Most frequent: prostate (8,297 new cases per year), breast (6,819). **Most deaths: lung**
  (≈3,240 per year, M:I 0.64).
- Poorest survival proxy: pancreas (M:I 0.82), brain and CNS (0.81), liver (0.73),
  oesophagus (0.72). Best: thyroid (0.07), melanoma (0.08), prostate (0.16), breast (0.20).
- Women have higher incidence than men between 25 and 54; men overtake from 55
  (figure 6, chapter 01).

## 8. Data-quality results

- Two independent BFS cause-of-death tables agree for every cancer row, 1990–2024.
- The ICD-8 → ICD-10 bridge is confirmed by two methods to 0.001 ([chapter 05](/books/cancer-ch/page/05-icd-8-icd-10-bridge-s03)).
- Our crude and age-standardised rates reproduce NKRS to the decimal ([chapter 06](/books/cancer-ch/page/06-harmonisation-s04)).
- Cause-of-death and cancer-registry mortality agree within 1% for all major sites.
