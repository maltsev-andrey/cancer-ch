"""Age standardisation. Two implementations exist (s04 on DataFrames, s06 on arrays);
both must use the same ESP1976 weights and agree with each other and with NKRS."""
import numpy as np
import pandas as pd
import pytest

# NKRS, Switzerland, all cancers, 2018-2022, as published in px-x-1403030300_162
NKRS_ASR = {"T": 384.6, "M": 440.6, "F": 339.2}
NKRS_CRUDE = {"T": 557.0, "M": 617.9, "F": 497.1}


def test_esp1976_sums_to_100000(s04, s06):
    assert sum(s04.ESP1976.values()) == 100_000
    assert s06.ESP.sum() == 100_000
    assert len(s04.ESP1976) == 18 and len(s06.ESP) == 18


def test_esp_weights_identical_in_both_modules(s04, s06):
    assert np.array_equal(np.array(list(s04.ESP1976.values()), dtype=float), s06.ESP)


def test_asr_of_constant_rate_is_that_rate(s04, s06):
    df = pd.DataFrame({"age5": s04.AGE5, "rate": 12.5})
    assert s04.asr(df, "rate") == pytest.approx(12.5)
    assert s06.asr(np.full(18, 12.5)) == pytest.approx(12.5)


def test_asr_hand_computed(s04, s06):
    rates = np.arange(18, dtype=float) * 10            # 0, 10, ..., 170
    w = s06.ESP
    expected = (rates * w).sum() / w.sum()
    assert s06.asr(rates) == pytest.approx(expected)
    df = pd.DataFrame({"age5": s04.AGE5, "rate": rates})
    assert s04.asr(df, "rate") == pytest.approx(expected)


def test_asr_ignores_missing_ages_by_renormalising(s06):
    """A sex-specific site has NaN at no age, but a sparse cell can; s06.asr renormalises
    the weights over the ages present rather than treating NaN as zero."""
    r = np.full(18, 10.0)
    r[0] = np.nan
    assert s06.asr(r) == pytest.approx(10.0)


def test_asr_is_lower_than_crude_for_an_old_population(s06):
    """Swiss population is older than ESP1976, so ASR < crude for any rate rising with age."""
    rates = np.exp(np.linspace(0, 6, 18))              # rises steeply with age
    pop_old = np.linspace(1, 3, 18)                     # more old people than ESP
    crude = (rates * pop_old).sum() / pop_old.sum()
    assert s06.asr(rates) < crude


@pytest.mark.parametrize("sex", ["T", "M", "F"])
def test_nkrs_reference_values_are_consistent(sex):
    """Guard against someone editing the reference constants: ASR/crude ratio for
    Switzerland 2018-2022 lies in 0.66-0.72 for every sex."""
    assert 0.66 < NKRS_ASR[sex] / NKRS_CRUDE[sex] < 0.72
