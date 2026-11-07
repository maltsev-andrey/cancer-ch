"""APC parameterisation: the RW2 normalisation, the detrending that makes drift the only
linear term, and the model's forward pass. Pure JAX, seconds on CPU."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def test_rw2_scale_is_independent_of_length(s06):
    """Without the n^1.5 normalisation the prior SD of the curve grows with n; with it, the
    empirical SD across random draws stays within a factor ~1.5 from n=6 to n=40."""
    rng = np.random.default_rng(0)
    sds = {}
    for n in (6, 12, 23, 40):
        curves = np.stack([np.asarray(s06.rw2_curve(jnp.array(rng.standard_normal(n - 2)), n, 1.0))
                           for _ in range(400)])
        sds[n] = curves.std()
    assert max(sds.values()) / min(sds.values()) < 1.6, sds


def test_rw2_starts_at_zero_and_scales_linearly(s06):
    d2 = jnp.array([1.0, -1.0, 0.5, 0.0])
    c1 = np.asarray(s06.rw2_curve(d2, 6, 1.0))
    c2 = np.asarray(s06.rw2_curve(d2, 6, 2.0))
    assert c1[0] == 0 and c1[1] == 0
    assert np.allclose(c2, 2 * c1)


def test_detrend_removes_mean_and_slope_on_observed_range(s06):
    x = jnp.array([1.0, 2.5, 2.0, 4.0, 5.5, 5.0, 9.0, 9.5])
    y = np.asarray(s06.detrend(x, 6))
    t = np.arange(6)
    assert y[:6].mean() == pytest.approx(0, abs=1e-5)
    assert np.polyfit(t, y[:6], 1)[0] == pytest.approx(0, abs=1e-5)


def test_detrend_of_a_line_is_zero_everywhere_including_future(s06):
    x = jnp.arange(8, dtype=jnp.float32) * 0.7 + 3.0
    y = np.asarray(s06.detrend(x, 6))
    assert np.allclose(y, 0, atol=1e-5)          # future points too: the line is extrapolated


def test_detrend_keeps_curvature(s06):
    t = np.arange(6, dtype=np.float32)
    x = jnp.array(t ** 2)
    y = np.asarray(s06.detrend(x, 6))
    assert np.allclose(y, (t ** 2) - np.polyval(np.polyfit(t, t ** 2, 1), t), atol=1e-4)


def test_model_forward_pass_shapes(s06):
    """Trace one prior draw through the model: deterministic sites exist with the right
    shapes and cohort index c = p - a + (A-1) stays inside 0..C-1."""
    from numpyro import handlers
    A, P, nf = 18, 6, 2
    C = A + P - 1
    aa, pp = np.meshgrid(np.arange(A), np.arange(P), indexing="ij")
    a_idx, p_idx = jnp.array(aa.ravel()), jnp.array(pp.ravel())
    c_idx = p_idx - a_idx + (A - 1)
    assert int(c_idx.min()) == 0 and int(c_idx.max()) == C - 1
    log_py = jnp.full(A * P, np.log(1e5))
    y = jnp.zeros(A * P, dtype=jnp.int32)
    tr = handlers.trace(handlers.seed(s06.build_model(), 0)).get_trace(
        a_idx, p_idx, c_idx, log_py, y, A, P, C, nf)
    assert tr["alpha"]["value"].shape == (A,)
    assert tr["beta"]["value"].shape == (P + nf,)
    assert tr["gamma"]["value"].shape == (C + nf,)
    assert tr["rate_grid"]["value"].shape == (A, P + nf)
    assert tr["log_rate"]["value"].shape == (A * P,)
    beta = np.asarray(tr["beta"]["value"])
    assert beta[:P].mean() == pytest.approx(0, abs=1e-4)      # detrended on observed range


def test_convergence_gate(s06):
    assert s06.converged(0, 1.005, 800, 600, 0.9)
    assert not s06.converged(11, 1.005, 800, 600, 0.9)      # divergences
    assert not s06.converged(0, 1.02, 800, 600, 0.9)        # rhat
    assert not s06.converged(0, 1.005, 300, 600, 0.9)       # bulk ESS
    assert not s06.converged(0, 1.005, 800, 300, 0.9)       # tail ESS
    assert not s06.converged(0, 1.005, 800, 600, 0.2)       # BFMI
