"""Unit tests for the independent PDDR-FF implementation."""

import numpy as np
import pytest

from pddrff import dominance_matrix, dominates, nondominated_mask, pddr_ff


def test_known_front_and_dominated_point() -> None:
    """Three trade-off points are non-dominated; the fourth is dominated."""
    objectives = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 4.0],
        ]
    )

    eval_values, q, p = pddr_ff(objectives)

    assert np.array_equal(nondominated_mask(objectives), [True, True, True, False])
    assert np.array_equal(q, [0.0, 0.0, 0.0, 3.0])
    assert np.array_equal(p, [1.0, 1.0, 1.0, 0.0])
    assert np.all(eval_values[:3] <= 1.0)
    assert eval_values[3] > 1.0


def test_duplicate_points_do_not_dominate_each_other() -> None:
    """Strict improvement prevents duplicate objective points from dominating."""
    objectives = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    matrix = dominance_matrix(objectives)
    eval_values, q, _ = pddr_ff(objectives)

    assert not matrix[0, 1]
    assert not matrix[1, 0]
    assert np.array_equal(q, [0.0, 0.0, 2.0])
    assert np.array_equal(nondominated_mask(objectives), [True, True, False])
    assert np.all(eval_values[:2] <= 1.0)


def test_random_matrix_has_exact_eval_frontier_equivalence() -> None:
    """For any finite matrix, eval <= 1 is equivalent to q == 0."""
    rng = np.random.default_rng(20260803)
    objectives = rng.random((40, 3))
    eval_values, q, _ = pddr_ff(objectives)

    assert np.array_equal(eval_values <= 1.0, q == 0.0)
    assert np.array_equal(nondominated_mask(objectives), q == 0.0)


def test_invalid_inputs_raise_clear_errors() -> None:
    """Reject malformed or non-finite objective matrices early."""
    with pytest.raises(ValueError, match="2-D"):
        pddr_ff(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        pddr_ff(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="same shape"):
        dominates(np.array([1.0, 2.0]), np.array([1.0]))
