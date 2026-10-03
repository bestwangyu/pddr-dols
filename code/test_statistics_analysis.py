"""Tests for paired statistical analysis helpers."""

import numpy as np
import pytest

from statistics_analysis import (
    compare_against_reference,
    friedman_test,
    holm_correction,
    paired_wilcoxon,
    win_tie_loss,
)


def test_wilcoxon_and_win_tie_loss_for_minimization_metric() -> None:
    """Lower values win for IGD+ and the paired delta is reported."""
    first = [0.1, 0.2, 0.3, 0.4]
    second = [0.2, 0.2, 0.4, 0.5]
    result = paired_wilcoxon(first, second)
    wtl = win_tie_loss(first, second, higher_is_better=False)

    assert result["n"] == 4
    assert result["median_delta"] < 0
    assert wtl == {"win": 3, "tie": 1, "loss": 0}


def test_holm_and_reference_comparison_preserve_rows() -> None:
    """Holm correction returns one adjusted p-value per comparison."""
    corrected = holm_correction([0.01, 0.2, 0.03])
    comparisons = compare_against_reference(
        {
            "pddr": [0.1, 0.2, 0.3],
            "nsga2": [0.2, 0.2, 0.4],
            "moead": [0.1, 0.3, 0.4],
        },
        reference="pddr",
        higher_is_better=False,
    )

    assert len(corrected["holm_pvalues"]) == 3
    assert len(comparisons) == 2
    assert {row["algorithm"] for row in comparisons} == {"nsga2", "moead"}
    assert all("holm_pvalue" in row for row in comparisons)


def test_friedman_test_accepts_instance_by_algorithm_matrix() -> None:
    """Friedman output identifies the matrix dimensions."""
    scores = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.2, 0.4], [0.15, 0.3, 0.35], [0.12, 0.25, 0.4]]
    )
    result = friedman_test(scores)
    assert result["n_instances"] == 4
    assert result["n_algorithms"] == 3
    assert np.isfinite(result["pvalue"])


def test_all_ties_return_finite_no_difference_statistics() -> None:
    """Complete ties are represented as finite, non-significant results."""
    wilcoxon = paired_wilcoxon([1, 1, 1], [1, 1, 1])
    friedman = friedman_test(np.ones((4, 3)))

    assert wilcoxon["statistic"] == 0.0
    assert wilcoxon["pvalue"] == 1.0
    assert friedman["statistic"] == 0.0
    assert friedman["pvalue"] == 1.0


def test_statistics_reject_invalid_inputs() -> None:
    """Reject unpaired arrays, invalid alpha and unknown references."""
    with pytest.raises(ValueError, match="equal-length"):
        paired_wilcoxon([1, 2], [1])
    with pytest.raises(ValueError, match="alpha"):
        holm_correction([0.1], alpha=1.0)
    with pytest.raises(ValueError, match="reference"):
        compare_against_reference({"a": [1, 2]}, reference="missing", higher_is_better=True)
