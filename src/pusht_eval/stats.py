"""Small, dependency-free statistics for paired policy evaluation."""

from __future__ import annotations

import math
import random
from math import comb, sqrt

Z95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("expected 0 <= successes <= trials and trials > 0")
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    half = z * sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_two_sided_p(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial test (sum of outcomes no more likely than k)."""
    probs = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(q for q in probs if q <= observed * (1 + 1e-9)))


def mcnemar_exact(base: list[bool], other: list[bool]) -> dict:
    """Exact McNemar test on paired success flags (same initial states)."""
    if len(base) != len(other):
        raise ValueError("paired lists must have equal length")
    lost = sum(1 for a, b in zip(base, other) if a and not b)
    gained = sum(1 for a, b in zip(base, other) if b and not a)
    n = lost + gained
    p = 1.0 if n == 0 else binom_two_sided_p(min(lost, gained), n, 0.5)
    return {"lost": lost, "gained": gained, "discordant": n, "p_value": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    order = sorted(p_values, key=p_values.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        adjusted[key] = running
    return adjusted


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    if not diffs:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(math.floor(alpha / 2 * n_boot))]
    hi = means[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return lo, hi
