# 11 · Cantonal model (s11, s12)

The national APC model (chapter 07) says what happens to the whole country. This chapter
builds the machinery for the cantonal question, whether the national trends are uniform across
the 26 cantons, and runs it on the one cantonal dataset that is open: all-cause deaths by
canton, sex and age (BEVNAT). Cantonal cause-of-death tables are not published by BFS and have
been requested (`docs/bfs_request_2026-11.md`); when they arrive they go through exactly this
code with `--indicator <cause>`.

## What is open and what is not

| Need | Open source | Status |
|---|---|---|
| Canton polygons | swisstopo swissBOUNDARIES3D, GeoPackage LV95, layer `tlm_kantonsgebiet` | used |
| Deaths by canton × sex × single age × year | `px-x-0102020000_104`, component *Todesfall*, 1981–2025 | used (dry run) |
| Population by canton × sex × age | same cube, 31 December stocks, as in chapter 06 | used |
| Deaths by **cause** × canton | not in PxWeb, not in the DAM catalogue after 1972 | requested from BFS |
| Cancer incidence by canton | cantonal registry reports, PDF, most since 2020 | later |

The DAM catalogue does hold two historical tables, cancer deaths by canton 1901–1968
(`hs-d-14.03.03.02.03.16`) and 1959/62 and 1969/72 (`B048-00-df`), which could make a
chapter of their own but are not comparable with ICD-10 data.

## s11: the canton graph

Two sources, checked against each other. The canonical adjacency list (land borders between
cantons, 57 edges, hard-coded by BFS canton number) is what the model uses. The geometry is
used for maps and as a check: cantons whose polygons intersect after a 50 m buffer are
neighbours. The two agree exactly: 57 edges from each, none only in one. The 50 m buffer
matters, because the TLM polygons of neighbouring cantons are separated by a river centreline
in places; the first attempt with vertex-sharing (queen) contiguity on the buffered polygons
found zero neighbours, since buffered polygons share no vertices.

Degrees run from 1 (Basel-Stadt, Genève) to 11 (Bern). The ICAR scaling factor on this graph
is 0.368: the generalised variance of the structured effect under the unscaled ICAR is 0.368,
so the structured component is divided by √0.368 to put it on the same scale as the iid one
(Riebler et al. 2016; the factor is the geometric mean of the marginal variances of the
constrained pseudo-inverse, computed as `(Q + 11ᵀ/n)⁻¹` minus `1/n`).

Canton labels differ between BFS cubes: `Appenzell A.Rh.` / `Appenzell I.Rh.` in the
population cube, full names elsewhere. `canton_key()` normalises both.

## s12: BYM2

Per sex and 5-year period (1990–94 … 2020–24), deaths in canton *c* against the deaths
expected from the national age-specific rates of that sex and period (indirect
standardisation, internal standard):

```
y_c  ~ Poisson( E_c · exp(α + φ_c) )
φ    = σ · ( √ρ · u/√s  +  √(1−ρ) · v )
u    ~ ICAR on the 57 edges, soft sum-to-zero;  v ~ N(0,1) iid
ρ    ~ Beta(1,1)        share of between-canton variation that is spatially structured
σ    ~ HalfNormal(0.5)  total between-canton SD of log RR
```

NUTS, 4 chains × 2000 + 2000, target acceptance 0.95, same gate as the APC model. The ICAR is
written as a `numpyro.factor` on the pairwise differences, so no precision matrix is formed.
Both deaths and stocks in the cube are tabulated by the age reached on 31 December (chapter
06), so the convention cancels in the rate; mid-year population is the mean of the 31 December
stocks of *y−1* and *y*, as in s04.

**Synthetic check** on the real graph with the real expected counts (men, 2015–19): a field
with ρ = 0.7, σ = 0.25 is recovered with corr(log RR̂, φ) = 1.00, 95% coverage 1.00, σ̂ 0.22,
ρ interval 0.03–0.97. The last number is the point: **ρ is not identifiable from 26 areas.**
With so few regions the data cannot tell a smooth field from an independent one, and the
posterior of ρ is close to its prior whatever the truth. The relative risks are identified
regardless, so the maps are sound; statements about "spatial clustering" are not available
from this model at this resolution, and chapter 10 lists that as a limitation rather than a
finding.

## Dry run: all-cause mortality

![Men](fig16_cantonal_rr_allcause_M.png)
![Women](fig16_cantonal_rr_allcause_F.png)

14 fits, all converged (R-hat ≤ 1.004, ESS > 1000, no divergences). The SMR averages to 1.000
in every sex × period, as the internal standard requires. Between-canton variation is small
after age standardisation: σ of log RR is 0.05–0.07 for men and 0.045–0.055 for women, so
cantons sit within about ±10% of the national rate. Shrinkage behaves as it should: Glarus with
1,010 male deaths in 2020–24 and SMR 1.126 is pulled to RR 1.09, Appenzell Innerrhoden with
385 deaths and SMR 1.003 gets an interval of 0.92–1.09, Zürich with 28,768 deaths barely moves.

What the maps show (a check that the pipeline reproduces known geography, not a result of
this project): Ticino, Genève and Zug below the national rate throughout, Ticino by 10–14% for
women in every period; Fribourg and Valais above for men in the 1990s and 2000s, Fribourg
converging to the mean after 2010; Jura, Neuchâtel, Basel-Stadt and Glarus above in recent
periods for both sexes. This matches the Obsan cantonal mortality indicators. σ for men rises
from 0.05 to 0.07 over the 30 years; for women it is flat. Whether that holds for cancer
mortality is a question for the cause-specific tables.

![Hyperparameters](fig17_spatial_params_allcause.png)

## Engineering notes

- 14 fits took 24 minutes of wall time for 3 minutes of sampling: the model is rebuilt and
  JIT-compiled for every sex × period. For 16 cancer sites × 2 sexes × 6 periods that would be
  an hour of compilation. The cause-specific version will be one model vectorised over periods
  (and, for a joint space-time model, with a period random walk shared across cantons), which
  also answers the real question, whether the cantonal pattern changes over time, inside one
  posterior.
- 26 nodes is far too small for a GPU to pay off; the statement in chapter 07 stands. The
  candidate for a GPU is a cantonal **APC** model (26 × 18 × 7 cells per site), not BYM2.
- Small cantons: AI (16,000 residents), OW, NW, UR and GL each have 400–1,000 deaths per
  five years for all causes. For a single cancer site that is 20–100 deaths, and their RR will
  be mostly prior. The maps mark only cantons with P(RR ≠ 1) > 95%, which keeps them honest.
- The ecological trap: a map invites a second map (smoking, altitude, income) next to it. This
  project does not do that, and chapter 10 keeps a class-R entry for it.
