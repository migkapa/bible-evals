import pytest

from bible_eval.core.stats import bootstrap_ci, wilson_interval


def test_wilson_bounds_are_ordered_and_clamped():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo <= hi <= 1.0
    # Point estimate should sit inside its interval.
    assert lo <= 0.8 <= hi


def test_wilson_extremes_stay_in_unit_interval():
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0  # clamped at the floor
    assert 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_interval(10, 10)
    assert hi1 == pytest.approx(1.0)  # ceiling (modulo float rounding)
    assert 0.0 < lo1 < 1.0


def test_wilson_zero_n_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_with_more_data():
    lo_small, hi_small = wilson_interval(8, 10)
    lo_big, hi_big = wilson_interval(800, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_bootstrap_is_seeded_and_deterministic():
    values = [0.1, 0.2, 0.0, 0.5, 0.3, 0.1, 0.4, 0.2]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_bootstrap_brackets_the_mean():
    values = [0.1, 0.2, 0.0, 0.5, 0.3, 0.1, 0.4, 0.2]
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values)
    assert lo <= mean <= hi


def test_bootstrap_degenerate_inputs():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.42]) == (0.42, 0.42)
