from __future__ import annotations

import random
from math import sqrt
from typing import Callable, Sequence, Tuple

# 97.5th percentile of the standard normal — two-sided 95% interval.
Z_95 = 1.959963984540054


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def wilson_interval(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Unlike the naive normal approximation, Wilson behaves sensibly at the
    extremes (0/n, n/n) and for small n — which is exactly the regime this
    benchmark runs in. Returns (lo, hi) clamped to [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * sqrt(phat * (1.0 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence. q in [0, 1]."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic (default: mean) of ``values``.

    Seeded for reproducibility, so re-running a scored run yields identical
    bands. Returns (lo, hi). Degenerate inputs (0 or 1 values) return a
    point interval.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    return (_percentile(stats, alpha), _percentile(stats, 1.0 - alpha))
