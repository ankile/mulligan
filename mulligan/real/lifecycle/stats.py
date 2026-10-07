"""Paired-eval statistics shared across real task lines.

Used by the held-out eval ingest, the start design and the paper's real-robot
tables. Outputs are pinned by golden fixtures in
``tests/real/test_real_lifecycle_golden.py``.
"""

from __future__ import annotations

import math

import numpy as np

Z_95 = 1.959963984540054
Z_1SE = 1.0

# Granular normalized-stage headline pins:
# fixed-seed percentile bootstrap, 20k resamples.
STAGE_BOOTSTRAPS = 20_000
STAGE_BOOTSTRAP_SEED = 20260720


def t_interval(x: np.ndarray, ci: float = 0.95) -> tuple[float, float, float, float]:
    """Student-t interval on a sample mean: ``(mean, se, lo, hi)`` (df = n-1).

    ``n < 2`` raises: a one-episode
    cell has no dispersion estimate and must fail loud, not return NaN bands.
    """
    from scipy import stats as sp_stats

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError(f"t_interval requires a 1-D sample with n >= 2, got shape {x.shape}")
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n))
    tcrit = float(sp_stats.t.ppf(0.5 + ci / 2.0, df=n - 1))
    return mean, se, mean - tcrit * se, mean + tcrit * se


def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int = STAGE_BOOTSTRAPS,
    seed: int = STAGE_BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on a sample mean: ``(mean, lo, hi)``.

    Fresh pinned-seed RNG per call so every (round, arm) cell is reproducible
    in isolation.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"bootstrap_mean_ci requires a 1-D non-empty sample, got shape {x.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    alpha = 100.0 * (1.0 - ci) / 2.0
    lo, hi = np.percentile(means, [alpha, 100.0 - alpha])
    return float(x.mean()), float(lo), float(hi)


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Score-Wilson binomial CI (default 95%)."""
    if n <= 0:
        raise ValueError("Wilson CI requires n > 0")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wilson_1se(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval with ``z=1`` (one-standard-error analogue).

    Unlike the plug-in Wald interval ``p_hat +/- sqrt(p_hat(1-p_hat)/n)``,
    this remains non-degenerate when the observed count is zero or ``n``.
    """

    return wilson_ci(successes, n, z=Z_1SE)


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Exact two-tailed McNemar p-value from the discordant-pair counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_paired_delta(
    a_success: np.ndarray,
    b_success: np.ndarray,
    n_boot: int = 20_000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Paired success-rate delta ``mean(a) - mean(b)`` with bootstrap CI.

    Resamples paired rounds with replacement. Returns
    ``(delta, ci_low, ci_high)`` in probability units.
    """
    a = np.asarray(a_success, dtype=np.float64)
    b = np.asarray(b_success, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError(f"paired arrays required, got shapes {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (a[idx] - b[idx]).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def signflip_permutation_pvalue(
    deltas: np.ndarray,
    n_perm: int = 20_000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value for ``mean(paired deltas) == 0``.

    Generalizes exact McNemar from binary paired outcomes to GRADED paired
    scores (e.g. routing's 0..2 clip score): under H0 each paired delta is
    symmetric around 0, so its sign is exchangeable. Uses Monte-Carlo sign
    flips with a pinned seed and the add-one correction
    ``(1 + #{|perm| >= |obs|}) / (1 + n_perm)`` so the estimate is never 0.
    All-zero deltas (fully concordant pairs) return 1.0, matching
    :func:`exact_mcnemar_pvalue` with no discordant pairs.
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError(f"1-D non-empty paired-delta array required, got shape {d.shape}")
    if np.all(d == 0.0):
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    # 1e-12: |obs| itself is one of the permutations up to float roundoff.
    extreme = int((perm >= observed - 1e-12).sum())
    return float((1 + extreme) / (1 + n_perm))


def bootstrap_paired_delta_stratified(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int = 20_000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Pooled paired delta across strata, resampling pairs WITHIN each stratum.

    ``pairs`` is one ``(a_success, b_success)`` tuple per stratum (e.g. one per
    paired eval block). The point estimate and every bootstrap replicate pool
    the per-pair deltas across all strata; resampling stays within-stratum so
    the stratum sizes are preserved. With a single stratum the CI bounds are
    identical to :func:`bootstrap_paired_delta` for the same seed (the
    point estimate can differ in the last ulp from summation order). Returns
    ``(delta, ci_low, ci_high)`` in probability units.
    """
    if not pairs:
        raise ValueError("at least one stratum is required")
    rng = np.random.default_rng(seed)
    total_n = 0
    delta_sums = np.zeros(n_boot, dtype=np.float64)
    point_sum = 0.0
    for a_success, b_success in pairs:
        a = np.asarray(a_success, dtype=np.float64)
        b = np.asarray(b_success, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
            raise ValueError(f"paired arrays required, got shapes {a.shape} vs {b.shape}")
        d = a - b
        n = len(d)
        idx = rng.integers(0, n, size=(n_boot, n))
        delta_sums += d[idx].sum(axis=1)
        point_sum += float(d.sum())
        total_n += n
    deltas = delta_sums / total_n
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return point_sum / total_n, float(lo), float(hi)
