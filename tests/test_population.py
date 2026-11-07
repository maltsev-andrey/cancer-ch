"""The BFS demographic-balance convention that cost us a 1-3% bias in every ASR:
every component of year y, the 1 January stock included, is tabulated by the age reached
on 31 December of y. Person-years at completed age a must therefore be the mean of the
31 December stocks of y-1 and y, never (jan1 + dec31) / 2 of the same row."""
import pandas as pd
import pytest


def balance_frame(years, dec31_by_age, jan1_by_age, canton="Zürich / Zurich", sex="Mann"):
    """Build a px-x-0102020000_104-shaped long frame for one canton and sex."""
    rows = []
    for y in years:
        for age_label, v in dec31_by_age[y].items():
            rows.append((y, canton, sex, age_label, "Bestand am 31. Dezember", v))
        for age_label, v in jan1_by_age[y].items():
            rows.append((y, canton, sex, age_label, "Bestand am 1. Januar", v))
        rows.append((y, canton, sex, "Alter - Total", "Bestand am 31. Dezember", sum(dec31_by_age[y].values())))
    return pd.DataFrame(rows, columns=["Jahr", "Kanton", "Geschlecht", "Alter",
                                       "Demografische Komponente", "value"])


@pytest.fixture
def two_year_balance():
    # ages 0..4 only; 2019 and 2020. jan1 at '0 Jahre' is 0 by BFS convention.
    dec31 = {2019: {f"{a} Jahre": 1000 + 10 * a for a in range(5)},
             2020: {f"{a} Jahre": 1100 + 10 * a for a in range(5)}}
    jan1 = {2019: {"0 Jahre": 0, **{f"{a} Jahre": 900 + 10 * a for a in range(1, 5)}},
            2020: {"0 Jahre": 0, **{f"{a} Jahre": 1000 + 10 * a for a in range(1, 5)}}}
    return balance_frame([2019, 2020], dec31, jan1)


def test_age5_labels(s04):
    assert s04.age5("0 Jahre") == "0-4"
    assert s04.age5("1 Jahr") == "0-4"
    assert s04.age5("4 Jahre") == "0-4"
    assert s04.age5("5 Jahre") == "5-9"
    assert s04.age5("84 Jahre") == "80-84"
    assert s04.age5("85 Jahre") == "85+"
    assert s04.age5("100 Jahre und mehr") == "85+"
    assert s04.age5("Alter - Total") is None
    assert s04.age5("Ohne Angabe") is None


def test_midyear_uses_previous_dec31_not_jan1(s04, two_year_balance, monkeypatch):
    monkeypatch.setattr(s04, "load_px", lambda cube: two_year_balance)
    pop, unknown = s04.build_population()
    r = pop[(pop.year == 2020) & (pop.age5 == "0-4")].iloc[0]
    dec31_2019 = sum(1000 + 10 * a for a in range(5))     # 5100
    dec31_2020 = sum(1100 + 10 * a for a in range(5))     # 5600
    jan1_2020 = sum(1000 + 10 * a for a in range(1, 5))   # 4060, the trap
    assert r.midyear == pytest.approx((dec31_2019 + dec31_2020) / 2)
    assert r.midyear != pytest.approx((jan1_2020 + dec31_2020) / 2)
    assert r.dec31_prev == dec31_2019


def test_first_year_is_dropped(s04, two_year_balance, monkeypatch):
    monkeypatch.setattr(s04, "load_px", lambda cube: two_year_balance)
    pop, _ = s04.build_population()
    assert set(pop.year) == {2020}            # 2019 has no previous 31 Dec stock


def test_total_rows_are_not_counted_as_an_age(s04, two_year_balance, monkeypatch):
    monkeypatch.setattr(s04, "load_px", lambda cube: two_year_balance)
    pop, _ = s04.build_population()
    assert pop.midyear.sum() == pytest.approx((5100 + 5600) / 2)


def test_canton_key_strips_bilingual_suffix(s04):
    assert s04.canton_key("Zürich / Zurich") == "Zürich"
    assert s04.canton_key("Fribourg / Freiburg") == "Fribourg"
    assert s04.canton_key("Bern") == "Bern"
