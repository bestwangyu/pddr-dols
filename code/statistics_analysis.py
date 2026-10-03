"""Paired statistical comparisons for repeated algorithm experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def _paired_arrays(first: Iterable[float], second: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert two paired score sequences to finite equal-length arrays."""
    left = np.asarray(list(first), dtype=float)
    right = np.asarray(list(second), dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise ValueError("paired samples must be equal-length 1-D arrays")
    if left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired samples must contain at least two finite values")
    return left, right


def paired_wilcoxon(
    first: Iterable[float], second: Iterable[float], *, alternative: str = "two-sided"
) -> dict[str, float | int]:
    """Run a paired Wilcoxon signed-rank test and report the median delta."""
    left, right = _paired_arrays(first, second)
    if np.allclose(left, right, rtol=0.0, atol=1e-12):
        # SciPy can return NaN rather than raising when every paired difference
        # is zero. This is a complete tie, so no difference is detectable.
        statistic, pvalue = 0.0, 1.0
    else:
        try:
            result = wilcoxon(left, right, alternative=alternative, zero_method="wilcox")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, pvalue = 0.0, 1.0
    if not np.isfinite(statistic) or not np.isfinite(pvalue):
        # Defensive fallback for SciPy edge cases involving zero variance.
        statistic, pvalue = 0.0, 1.0
    return {
        "n": int(left.size),
        "statistic": statistic,
        "pvalue": pvalue,
        "median_delta": float(np.median(left - right)),
    }


def holm_correction(pvalues: Iterable[float], *, alpha: float = 0.05) -> dict[str, list]:
    """Apply Holm correction while preserving input order."""
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a non-empty finite 1-D sequence")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rejected, corrected, _, _ = multipletests(values, alpha=alpha, method="holm")
    return {
        "raw_pvalues": values.tolist(),
        "holm_pvalues": corrected.tolist(),
        "reject": rejected.tolist(),
        "alpha": [float(alpha)] * len(values),
    }


def win_tie_loss(
    first: Iterable[float],
    second: Iterable[float],
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    """Count paired wins, ties and losses for one metric."""
    left, right = _paired_arrays(first, second)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    delta = left - right
    ties = np.abs(delta) <= tolerance
    if higher_is_better:
        wins = delta > tolerance
        losses = delta < -tolerance
    else:
        wins = delta < -tolerance
        losses = delta > tolerance
    return {"win": int(wins.sum()), "tie": int(ties.sum()), "loss": int(losses.sum())}


def friedman_test(scores: np.ndarray) -> dict[str, float | int]:
    """Run Friedman test on rows=instances and columns=algorithms."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("scores must have at least two instances and two algorithms")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    if np.allclose(values, values[:, [0]], rtol=0.0, atol=1e-12):
        statistic, pvalue = 0.0, 1.0
    else:
        result = friedmanchisquare(
            *[values[:, column] for column in range(values.shape[1])]
        )
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        if not np.isfinite(statistic) or not np.isfinite(pvalue):
            statistic, pvalue = 0.0, 1.0
    return {
        "n_instances": int(values.shape[0]),
        "n_algorithms": int(values.shape[1]),
        "statistic": statistic,
        "pvalue": pvalue,
    }


def compare_against_reference(
    scores: dict[str, Iterable[float]],
    *,
    reference: str,
    higher_is_better: bool,
    alpha: float = 0.05,
) -> list[dict]:
    """Compare every algorithm to a reference with Holm-adjusted p-values."""
    if reference not in scores:
        raise ValueError(f"reference {reference!r} is not present in scores")
    reference_values = np.asarray(list(scores[reference]), dtype=float)
    comparisons = []
    for name, values in scores.items():
        if name == reference:
            continue
        test = paired_wilcoxon(reference_values, values)
        wtl = win_tie_loss(
            reference_values, values, higher_is_better=higher_is_better
        )
        comparisons.append({"algorithm": name, **test, **wtl})
    if comparisons:
        correction = holm_correction([row["pvalue"] for row in comparisons], alpha=alpha)
        for row, corrected, rejected in zip(
            comparisons, correction["holm_pvalues"], correction["reject"]
        ):
            row["holm_pvalue"] = float(corrected)
            row["reject_holm"] = bool(rejected)
    return comparisons


__all__ = [
    "compare_against_reference",
    "friedman_test",
    "holm_correction",
    "paired_wilcoxon",
    "win_tie_loss",
]
