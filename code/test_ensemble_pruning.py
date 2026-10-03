"""Unit tests for the cached ensemble-pruning objective evaluator."""

import numpy as np
import pytest

from ensemble_pruning import EnsembleEvaluator


@pytest.fixture
def evaluator() -> EnsembleEvaluator:
    """Create a small prediction cache with a known majority-vote outcome."""
    predictions = np.array(
        [
            [0, 0, 1, 1],  # classifier 0
            [0, 1, 1, 0],  # classifier 1
            [1, 1, 1, 1],  # classifier 2
        ]
    )
    labels = np.array([0, 1, 1, 0])
    costs = np.array([1.0, 2.0, 4.0])
    return EnsembleEvaluator(predictions, labels, costs)


def test_hard_vote_and_objectives(evaluator: EnsembleEvaluator) -> None:
    """Evaluate two classifiers and verify error, cost and margin loss."""
    result = evaluator.evaluate([1, 1, 0])

    # Votes are [0, 0, 1, 0], so one of four samples is incorrect.
    assert np.array_equal(result.predictions, [0, 0, 1, 0])
    assert result.validation_error == pytest.approx(0.25)
    assert result.normalized_cost == pytest.approx(3.0 / 7.0)
    assert result.margin_loss == pytest.approx(0.5)
    assert result.selected_indices == (0, 1)
    assert not result.repaired_empty_subset


def test_empty_subset_is_repaired_to_cheapest_classifier(
    evaluator: EnsembleEvaluator,
) -> None:
    """An empty binary subset becomes the deterministic cheapest subset."""
    result = evaluator.evaluate(np.array([0, 0, 0]))

    assert result.repaired_empty_subset
    assert result.selected_indices == (0,)
    assert result.normalized_cost == pytest.approx(1.0 / 7.0)
    assert evaluator.cache_size == 1


def test_empty_subset_and_repaired_subset_share_cache(evaluator: EnsembleEvaluator) -> None:
    """The repair is part of the cache key, preventing duplicate evaluations."""
    first = evaluator.evaluate([0, 0, 0])
    second = evaluator.evaluate([1, 0, 0])

    assert first.objectives is second.objectives
    assert evaluator.cache_size == 1


def test_objective_wrapper_returns_copy(evaluator: EnsembleEvaluator) -> None:
    """The pymoo-facing wrapper must not expose mutable cached state."""
    objectives = evaluator.evaluate_objectives([1, 0, 1])
    objectives[0] = 999.0
    assert evaluator.evaluate([1, 0, 1]).validation_error != 999.0


def test_invalid_evaluator_inputs_are_rejected() -> None:
    """Reject shape, label and cost errors before any optimization starts."""
    with pytest.raises(ValueError, match="same samples"):
        EnsembleEvaluator(np.zeros((2, 3)), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        EnsembleEvaluator(np.zeros((2, 2)), np.zeros(2), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="one value per classifier"):
        EnsembleEvaluator(np.zeros((2, 2)), np.zeros(2), np.ones(3))
