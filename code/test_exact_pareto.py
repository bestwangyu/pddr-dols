"""Tests for exact small-pool Pareto enumeration."""

import numpy as np
import pytest

from ensemble_pruning import EnsembleEvaluator
from exact_pareto import enumerate_exact_pareto, save_exact_pareto
from pddrff import nondominated_mask


@pytest.fixture
def small_evaluator() -> EnsembleEvaluator:
    """A three-classifier prediction cache with seven non-empty subsets."""
    predictions = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]]
    )
    labels = np.array([0, 1, 1, 0])
    return EnsembleEvaluator(predictions, labels, np.array([1.0, 2.0, 3.0]))


def test_exact_enumeration_evaluates_all_nonempty_subsets(
    small_evaluator: EnsembleEvaluator,
) -> None:
    """Three classifiers produce exactly seven evaluated non-empty subsets."""
    result = enumerate_exact_pareto(small_evaluator)

    assert result.n_classifiers == 3
    assert result.evaluated_subsets == 7
    assert result.pareto_masks.shape[1] == 3
    assert result.pareto_objectives.shape[1] == 3
    assert len(result.pareto_masks) >= 1
    assert np.all(nondominated_mask(result.pareto_objectives))
    assert len(np.unique(result.pareto_objectives, axis=0)) == len(
        result.pareto_objectives
    )


def test_exact_enumeration_has_explicit_safety_limits(
    small_evaluator: EnsembleEvaluator,
) -> None:
    """Callers cannot silently enumerate beyond configured resources."""
    with pytest.raises(ValueError, match="at most"):
        enumerate_exact_pareto(small_evaluator, max_classifiers=2)
    with pytest.raises(ValueError, match="max_subsets"):
        enumerate_exact_pareto(small_evaluator, max_subsets=3)


def test_exact_front_can_be_saved_as_json(
    small_evaluator: EnsembleEvaluator, tmp_path
) -> None:
    """The saved record contains masks and objectives for later IGD evaluation."""
    result = enumerate_exact_pareto(small_evaluator)
    output = tmp_path / "exact_front.json"
    save_exact_pareto(result, output)
    assert output.exists()
    assert "pareto_objectives" in output.read_text(encoding="utf-8")
