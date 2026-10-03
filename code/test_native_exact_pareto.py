"""Tests for PEP/MDEP native exact references."""

import numpy as np

from ensemble_pruning import EnsembleEvaluator
from native_exact_pareto import enumerate_mdep_exact, enumerate_pep_exact


def _evaluator() -> EnsembleEvaluator:
    predictions = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0]])
    labels = np.array([0, 0, 1, 1])
    return EnsembleEvaluator(predictions, labels, np.ones(3))


def test_native_exact_front_dimensions_and_subset_count() -> None:
    evaluator = _evaluator()
    pep = enumerate_pep_exact(evaluator)
    mdep = enumerate_mdep_exact(evaluator)
    assert pep.evaluated_subsets == 7
    assert pep.pareto_objectives.shape[1] == 2
    assert mdep.evaluated_subsets == 7
    assert mdep.pareto_objectives.shape[1] == 3
    assert len(pep.pareto_masks) == len(pep.pareto_objectives)
    assert len(mdep.pareto_masks) == len(mdep.pareto_objectives)


def test_native_exact_fronts_are_non_dominated() -> None:
    evaluator = _evaluator()
    for result in (enumerate_pep_exact(evaluator), enumerate_mdep_exact(evaluator)):
        for i, left in enumerate(result.pareto_objectives):
            for j, right in enumerate(result.pareto_objectives):
                if i == j:
                    continue
                assert not (np.all(left <= right) and np.any(left < right))
