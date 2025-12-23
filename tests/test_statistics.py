"""Tests for the statistics module."""
import math

from bible_eval.core.statistics import (
    ConfidenceInterval,
    compute_confidence_interval,
    compute_proportion_ci,
    format_ci,
)


def test_confidence_interval_perfect_data() -> None:
    """All same values should have zero std dev."""
    values = [0.5, 0.5, 0.5, 0.5, 0.5]
    ci = compute_confidence_interval(values)
    assert ci is not None
    assert ci.mean == 0.5
    assert ci.std_dev == 0.0
    assert ci.lower == 0.5
    assert ci.upper == 0.5


def test_confidence_interval_normal_data() -> None:
    """Test CI with varied data."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ci = compute_confidence_interval(values)
    assert ci is not None
    assert abs(ci.mean - 0.5) < 0.001
    assert ci.lower < ci.mean
    assert ci.upper > ci.mean
    assert ci.n == 9


def test_confidence_interval_insufficient_data() -> None:
    """Single value should return None."""
    assert compute_confidence_interval([0.5]) is None
    assert compute_confidence_interval([]) is None


def test_proportion_ci_zero() -> None:
    """Zero successes should have lower bound near 0."""
    ci = compute_proportion_ci(0, 10)
    assert ci is not None
    assert ci.mean == 0.0
    assert ci.lower == 0.0
    assert ci.upper > 0.0  # Wilson interval never exactly 0


def test_proportion_ci_all() -> None:
    """All successes should have upper bound near 1."""
    ci = compute_proportion_ci(10, 10)
    assert ci is not None
    assert ci.mean == 1.0
    assert ci.lower < 1.0  # Wilson interval never exactly 1
    assert ci.upper == 1.0


def test_proportion_ci_half() -> None:
    """50% success rate."""
    ci = compute_proportion_ci(50, 100)
    assert ci is not None
    assert abs(ci.mean - 0.5) < 0.001
    # 95% CI for n=100, p=0.5 should be roughly [0.4, 0.6]
    assert 0.35 < ci.lower < 0.45
    assert 0.55 < ci.upper < 0.65


def test_format_ci_as_percent() -> None:
    """Test percentage formatting."""
    ci = ConfidenceInterval(
        mean=0.75,
        lower=0.65,
        upper=0.85,
        std_dev=0.1,
        n=50,
    )
    formatted = format_ci(ci, as_percent=True)
    assert "75.0%" in formatted
    assert "65.0" in formatted
    assert "85.0" in formatted


def test_format_ci_none() -> None:
    """None CI should return N/A."""
    assert format_ci(None) == "N/A"
