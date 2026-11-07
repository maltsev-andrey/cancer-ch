"""End-to-end: simulate an 18 x 6 grid with a known net drift of -1.5 %/yr and a cohort
bump centred on the 1968 cohort, fit the real model, check both are recovered.
About a minute on 4 CPU cores. Run with: pytest -m slow"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.slow

A, P = 18, 6
C = A + P - 1
TRUE_DRIFT_PER_YEAR = -1.5           # percent
BUMP_COHORT = 1968                   # cohort start year; cohorts run 1908 .. 2018 in 5-y steps
COHORT_START = 1993 - 5 * (A - 1)    # first period 1993-97, oldest age 85+


def simulate(seed=1):
    rng = np.random.default_rng(seed)
    aa, pp = np.meshgrid(np.arange(A), np.arange(P), indexing="ij")
    cc = pp - aa + (A - 1)
    alpha = -11 + 0.45 * np.arange(A)                         # log rate rising with age
    alpha = np.minimum(alpha, -6.0)
    drift5 = 5 * np.log1p(TRUE_DRIFT_PER_YEAR / 100)           # per 5-y period
    cohort_years = COHORT_START + 5 * np.arange(C)
    gamma = 0.35 * np.exp(-0.5 * ((cohort_years - BUMP_COHORT) / 7.0) ** 2)
    gamma -= np.polyval(np.polyfit(np.arange(C), gamma, 1), np.arange(C))   # keep it curvature-only
    eta = alpha[aa] + drift5 * (pp - (P - 1) / 2) + gamma[cc]
    py = np.full((A, P), 5 * 250_000.0)                         # 250k persons per age group, 5 years
    y = rng.poisson(np.exp(eta) * py)
    return aa.ravel(), pp.ravel(), cc.ravel(), np.log(py).ravel(), y.ravel(), cohort_years


def test_recovers_drift_and_cohort_bump(s06):
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_median
    jax.config.update("jax_enable_x64", True)
    a, p, c, log_py, y, cohort_years = simulate()
    kernel = NUTS(s06.build_model(), target_accept_prob=0.9,
                  init_strategy=init_to_median(num_samples=50))
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), a_idx=jnp.array(a), p_idx=jnp.array(p), c_idx=jnp.array(c),
             log_py=jnp.array(log_py), y=jnp.array(y), A=A, P=P, C=C, n_future=0)
    s = mcmc.get_samples()
    drift_yr = (np.exp(np.asarray(s["drift"]) / 5) - 1) * 100
    lo, med, hi = np.percentile(drift_yr, [2.5, 50, 97.5])
    assert lo < TRUE_DRIFT_PER_YEAR < hi, (lo, med, hi)
    assert abs(med - TRUE_DRIFT_PER_YEAR) < 0.3, med

    gamma = np.asarray(s["gamma"]).mean(0)
    interior = slice(3, C - 3)                                  # edge cohorts have 1-3 cells
    peak = cohort_years[interior][np.argmax(gamma[interior])]
    assert abs(peak - BUMP_COHORT) <= 5, peak                   # within one cohort step

    div = int(mcmc.get_extra_fields()["diverging"].sum())
    assert div <= 10
