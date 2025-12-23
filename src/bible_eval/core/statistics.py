"""
Statistical utilities for evaluation metrics.

Provides confidence intervals and statistical significance testing
for evaluation results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ConfidenceInterval:
    """Represents a confidence interval for a metric."""

    mean: float
    lower: float
    upper: float
    std_dev: float
    n: int
    confidence_level: float = 0.95

    @property
    def margin_of_error(self) -> float:
        return (self.upper - self.lower) / 2

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "std_dev": self.std_dev,
            "n": self.n,
            "confidence_level": self.confidence_level,
            "margin_of_error": self.margin_of_error,
        }


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """
    Approximate t-critical value for given degrees of freedom.

    Uses a simplified approximation that works well for df >= 2.
    For publication-quality results, use scipy.stats.t.ppf.
    """
    # For 95% confidence (two-tailed), alpha = 0.05
    # Common t-values for reference:
    # df=10: 2.228, df=20: 2.086, df=30: 2.042, df=50: 2.009, df=100: 1.984
    # df=inf (z): 1.96

    if df <= 0:
        return float("inf")
    if df == 1:
        return 12.71  # 95% CI for df=1
    if df == 2:
        return 4.303

    # Approximation formula for larger df
    # Based on: t ≈ z + (z + z^3) / (4 * df) for large df
    z = 1.96  # z-score for 95% CI

    if df >= 120:
        return z

    # More accurate approximation for moderate df
    t = z + (z + z**3) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    return t


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Optional[ConfidenceInterval]:
    """
    Compute confidence interval for a list of values.

    Uses t-distribution for small samples (n < 30) and
    normal approximation for larger samples.

    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval or None if insufficient data
    """
    n = len(values)
    if n < 2:
        return None

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return ConfidenceInterval(
            mean=mean,
            lower=mean,
            upper=mean,
            std_dev=0.0,
            n=n,
            confidence_level=confidence,
        )

    # Standard error of the mean
    sem = std_dev / math.sqrt(n)

    # t-critical value
    df = n - 1
    t_crit = _t_critical(df, confidence)

    margin = t_crit * sem
    lower = mean - margin
    upper = mean + margin

    return ConfidenceInterval(
        mean=mean,
        lower=lower,
        upper=upper,
        std_dev=std_dev,
        n=n,
        confidence_level=confidence,
    )


def compute_proportion_ci(
    successes: int, total: int, confidence: float = 0.95
) -> Optional[ConfidenceInterval]:
    """
    Compute confidence interval for a proportion using Wilson score interval.

    The Wilson score interval is preferred over the normal approximation
    because it:
    - Works well for small samples
    - Never produces intervals outside [0, 1]
    - Is more accurate when p is close to 0 or 1

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        ConfidenceInterval or None if insufficient data
    """
    if total == 0:
        return None

    p = successes / total

    # Z-score for confidence level
    z = 1.96  # 95% CI
    if confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    # Approximate standard deviation for binomial
    std_dev = math.sqrt(p * (1 - p) / total) if total > 0 else 0.0

    return ConfidenceInterval(
        mean=p,
        lower=lower,
        upper=upper,
        std_dev=std_dev,
        n=total,
        confidence_level=confidence,
    )


@dataclass(frozen=True)
class MetricsSummary:
    """Summary statistics for a set of evaluation results."""

    strict_accuracy: ConfidenceInterval
    avg_wer: ConfidenceInterval
    avg_cer: ConfidenceInterval
    avg_semantic_similarity: ConfidenceInterval
    hallucination_rate: ConfidenceInterval

    # Category counts (raw, for display)
    n_verbatim: int
    n_verbatim_with_extras: int
    n_minor_deviation: int
    n_omission: int
    n_paraphrase: int
    n_partial_recall: int
    n_hallucination: int
    n_total: int

    def to_dict(self) -> dict:
        return {
            "strict_accuracy": self.strict_accuracy.to_dict() if self.strict_accuracy else None,
            "avg_wer": self.avg_wer.to_dict() if self.avg_wer else None,
            "avg_cer": self.avg_cer.to_dict() if self.avg_cer else None,
            "avg_semantic_similarity": (
                self.avg_semantic_similarity.to_dict() if self.avg_semantic_similarity else None
            ),
            "hallucination_rate": self.hallucination_rate.to_dict() if self.hallucination_rate else None,
            "category_counts": {
                "verbatim": self.n_verbatim,
                "verbatim_with_extras": self.n_verbatim_with_extras,
                "minor_deviation": self.n_minor_deviation,
                "omission": self.n_omission,
                "paraphrase": self.n_paraphrase,
                "partial_recall": self.n_partial_recall,
                "hallucination": self.n_hallucination,
                "total": self.n_total,
            },
        }


def compute_metrics_summary(results: List[dict]) -> Optional[MetricsSummary]:
    """
    Compute summary statistics with confidence intervals from evaluation results.

    Args:
        results: List of result dictionaries containing 'scores' field

    Returns:
        MetricsSummary or None if insufficient data
    """
    if not results:
        return None

    n = len(results)

    # Extract metrics
    wers = [r["scores"]["wer"] for r in results]
    cers = [r["scores"]["cer"] for r in results]
    semantic_sims = [r["scores"].get("semantic_similarity", 0.0) for r in results]

    # Count categories
    categories = [r["scores"].get("error_category", r["scores"].get("label", "unknown")) for r in results]

    n_verbatim = sum(1 for c in categories if c == "verbatim")
    n_verbatim_with_extras = sum(1 for c in categories if c == "verbatim_with_extras")
    n_minor_deviation = sum(1 for c in categories if c == "minor_deviation")
    n_omission = sum(1 for c in categories if c == "omission")
    n_paraphrase = sum(1 for c in categories if c == "paraphrase")
    n_partial_recall = sum(1 for c in categories if c == "partial_recall")
    n_hallucination = sum(1 for c in categories if c == "total_hallucination")

    # Strict accuracy: WER=0 and CER=0
    strict_hits = sum(1 for r in results if r["scores"]["wer"] == 0.0 and r["scores"]["cer"] == 0.0)

    return MetricsSummary(
        strict_accuracy=compute_proportion_ci(strict_hits, n),
        avg_wer=compute_confidence_interval(wers),
        avg_cer=compute_confidence_interval(cers),
        avg_semantic_similarity=compute_confidence_interval(semantic_sims),
        hallucination_rate=compute_proportion_ci(n_hallucination, n),
        n_verbatim=n_verbatim,
        n_verbatim_with_extras=n_verbatim_with_extras,
        n_minor_deviation=n_minor_deviation,
        n_omission=n_omission,
        n_paraphrase=n_paraphrase,
        n_partial_recall=n_partial_recall,
        n_hallucination=n_hallucination,
        n_total=n,
    )


def format_ci(ci: Optional[ConfidenceInterval], as_percent: bool = False) -> str:
    """Format confidence interval for display."""
    if ci is None:
        return "N/A"

    mult = 100.0 if as_percent else 1.0
    suffix = "%" if as_percent else ""

    return f"{ci.mean * mult:.1f}{suffix} [{ci.lower * mult:.1f}-{ci.upper * mult:.1f}]"
