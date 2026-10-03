"""Tests for the pymoo ensemble-pruning problem wrapper."""

import numpy as np

from ensemble_problem import EnsemblePruningProblem
from ensemble_pruning import EnsembleEvaluator


def test_problem_evaluates_binary_population() -> None:
    """A batch of pymoo decisions returns one three-objective row per subset."""
    predictions = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
    labels = np.array([0, 1, 1])
    evaluator = EnsembleEvaluator(predictions, labels, np.array([1.0, 2.0, 3.0]))
    problem = EnsemblePruningProblem(evaluator)
    decisions = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)

    objectives = problem.evaluate(decisions)

    assert objectives.shape == (3, 3)
    assert np.all(np.isfinite(objectives))
    assert np.array_equal(objectives[0], evaluator.evaluate_objectives(decisions[0]))
    assert evaluator.cache_size == 2  # Empty mask repairs to the first subset.


def test_problem_metadata_matches_evaluator() -> None:
    """The wrapper declares the correct variable and objective dimensions."""
    evaluator = EnsembleEvaluator(
        np.zeros((4, 5), dtype=int), np.zeros(5, dtype=int), np.ones(4)
    )
    problem = EnsemblePruningProblem(evaluator)

    assert problem.n_var == 4
    assert problem.n_obj == 3
    assert problem.vtype is bool
