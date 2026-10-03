"""Tests for common normalization, HV and IGD+ calculations."""

import numpy as np
import pytest

from metrics import (
    build_normalization_context,
    compute_front_metrics,
    merge_reference_front,
    nondominated_front,
    normalize_objectives,
)


def test_context_is_shared_across_all_fronts() -> None:
    """The context sees the global ideal and nadir, not one front at a time."""
    fronts = {
        "a": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "b": np.array([[0.5, 0.5], [2.0, 2.0]]),
    }
    context = build_normalization_context(fronts)

    assert np.array_equal(context.ideal, [0.0, 0.0])
    assert np.array_equal(context.nadir, [2.0, 2.0])
    assert np.array_equal(context.span, [2.0, 2.0])
    assert np.array_equal(context.reference_point, [1.1, 1.1])
    assert np.allclose(normalize_objectives(fronts["a"], context)[0], [0.0, 0.5])


def test_nondominated_and_merged_reference_front() -> None:
    """Dominated rows are removed from each and the merged reference front."""
    front = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    clean = nondominated_front(front)
    merged = merge_reference_front({"a": front, "b": np.array([[0.5, 0.5]])})

    assert clean.shape == (2, 2)
    assert merged.shape == (3, 2)


def test_metrics_are_finite_and_hv_is_nonnegative() -> None:
    """Common HV/IGD+ output has one record per algorithm."""
    fronts = {
        "strong": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "weak": np.array([[0.5, 0.9], [0.9, 0.5]]),
    }
    context, reference, metrics = compute_front_metrics(fronts)

    assert reference.shape == (4, 2)
    assert context.reference_point.shape == (2,)
    assert set(metrics) == {"strong", "weak"}
    assert all(np.isfinite(row["hypervolume"]) for row in metrics.values())
    assert all(row["hypervolume"] >= 0.0 for row in metrics.values())
    assert all(np.isfinite(row["igd_plus"]) for row in metrics.values())


def test_metric_inputs_are_rejected() -> None:
    """Reject empty collections and inconsistent objective dimensions."""
    with pytest.raises(ValueError, match="at least one"):
        build_normalization_context({})
    with pytest.raises(ValueError, match="same number"):
        build_normalization_context({"a": np.ones((2, 2)), "b": np.ones((2, 3))})
