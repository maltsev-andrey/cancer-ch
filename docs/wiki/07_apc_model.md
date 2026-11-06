# 07 · Age-period-cohort model (s06)

`pipeline/s06_apc.py` fits one Bayesian APC model per site × sex × indicator on the NKRS grid:
18 five-year age groups × 6 five-year periods (1993–2022), 23 birth cohorts. 107 fits qualify
(≥30 cases per year), each run twice: full (with projection) and holdout.

## Model

```
y_ap ~ Poisson(PY_ap · exp(η_ap))
η_ap = α_a + δ·(p − p̄) + β_p + γ_c + ε_ap          c = p − a
```

| Term | Prior | Meaning |
|---|---|---|
| α_a | free vector, N(−8, 5) | age curve (pinned by the data, no hyperparameter) |
| δ | N(0, 0.5) | **net drift**: the identifiable linear trend of period and cohort combined |
| β_p | RW2, linear trend removed | period curvature only |
| γ_c | RW2, linear trend removed | cohort curvature only |
| ε_ap | N(0, s_ε) | extra-Poisson variation |

The split of a linear trend between period and cohort is not identifiable from any APC data
(Holford 1983; Riebler & Held 2017). The model does not pretend otherwise: it reports drift,
curvatures and the age curve, which are identifiable, and shows cohort effects as **relative
risk versus the linear trend**. A cohort "peak" is therefore a generation that lies above the
trend line, not necessarily the generation with the highest absolute rate.

NUTS (NumPyro), 4 chains, 1000 warm-up + 1000 samples; fits with divergences were rerun at
`target_accept` 0.97–0.99. Result: **211 of 214 fits converged** (r̂ < 1.05, ≤ 10 divergences);
the three remaining holdout fits have r̂ 1.00 and 18–29 divergences and are excluded from the
holdout summary only.

## What it took to make it sample

| Problem | Symptom | Fix |
|---|---|---|
| RW2 built as a double cumsum of N(0,1) | amplitude ~ n^1.5; one chain ran to 10^23 | normalise by n^1.5 so the scale parameter is the curve's SD |
| RW2 on the age curve | all-cancers age curve spans 6 log units; second differences far outside their N(0,1) prior; r̂ 1.4 | age as a free vector (centred); the data pin 18 parameters with 6 periods each |
| Non-centred RW1 on age | funnel between s_age and the increments with strong data; r̂ 2.1 on synthetic data | same fix: centred, no hyperparameter |
| Predictive interval without noise | 39% holdout coverage of a 90% interval | add ε and Poisson noise to the posterior predictive |
| Cohort peak at the grid edge | "peak cohort 2020, RR 2.7" (one observed cell) | search peaks only among cohorts with ≥ 4 observed cells |

Every change was first checked on a synthetic grid with a known drift (−1.5%/yr) and a known
cohort bump (1968): the model recovers −1.48%/yr (−1.60…−1.35) and the bump at 1968.

## Validation

- **Holdout:** fit on 1993–2017, predict 2018–2022. Over 84 converged fits with ≥100 cases per
  year the median error is **+2.5% for incidence and +3.0% for mortality**: the 2018–2022 rates
  came in slightly *below* the extrapolated trend for both indicators.
- This also answers the registration-act question. If the Cancer Registration Act (2020) had
  inflated registered incidence, incidence would sit systematically *above* trend while
  mortality would not. It does not: 38% of incidence fits are below trend, 62% above, and
  mortality moves the same way. Lung cancer in men (−4%, observed above trend) is the
  exception, not the rule. The more likely reading of the small positive medians is the
  COVID-19 dip in diagnoses in 2020–2021 and a mortality decline that keeps accelerating.

## GPU vs CPU

The same lung-cancer fit took **694 s on the Tesla P100** (4 chains × 3000 draws) and
**~20 s on 4 CPU cores** (4 × 2000 draws): per draw the GPU is about 20× slower. A model
with 108 data cells and ~70 parameters gives NUTS nothing to parallelise within a step, and
kernel-launch overhead dominates. The sweep therefore ran on a 20-core CPU host with 4 fits
in parallel (16 cores, 4.5 GB RAM): 214 fits in about 45 minutes. A batched model that fits
all 107 sites in one vectorised program is the way to make the GPU useful here, and is left
for a later iteration.
