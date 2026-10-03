"""Unit tests for the paper-faithful PEP implementation."""

import numpy as np

from ensemble_pruning import EnsembleEvaluator
from pep_paper import (
    PEPPaperProblem,
    default_pep_iterations,
    generate_vds_neighbors,
    mutate_bitwise,
    pareto_update,
    pep_dominates,
    run_pep_paper,
    select_final_solution,
)


def _evaluator() -> EnsembleEvaluator:
    predictions = np.array(
        [
            [0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    labels = np.array([0, 0, 1, 1, 0, 1])
    return EnsembleEvaluator(predictions, labels, np.ones(4))


def test_pep_dominance_and_archive_update() -> None:
    assert pep_dominates(np.array([0.1, 0.3]), np.array([0.2, 0.3]))
    masks, values, inserted = pareto_update(
        [np.array([1, 0]), np.array([1, 1])],
        [np.array([0.2, 0.5]), np.array([0.1, 0.8])],
        np.array([0, 1]),
        np.array([0.1, 0.4]),
    )
    assert inserted
    assert len(masks) == len(values) == 1
    assert any(np.array_equal(mask, [0, 1]) for mask in masks)


def test_mutation_is_reproducible_and_vds_flips_unique_positions() -> None:
    mask = np.array([True, False, True, False])
    assert np.array_equal(mutate_bitwise(mask, np.random.default_rng(5)), mutate_bitwise(mask, np.random.default_rng(5)))
    problem = PEPPaperProblem(_evaluator())
    generated, evaluations = generate_vds_neighbors(problem, mask)
    assert len(generated) == 4
    assert evaluations == 10  # 4 + 3 + 2 + 1 candidate evaluations
    assert all(np.count_nonzero(generated[i] != mask) >= 1 for i in range(4))
    assert all(np.count_nonzero(generated[i] != generated[i - 1]) == 1 for i in range(1, 4))


def test_empty_subset_uses_paper_policy_and_final_selection_is_non_empty() -> None:
    problem = PEPPaperProblem(_evaluator())
    objective = problem.evaluate(np.zeros(4, dtype=bool))
    assert np.isinf(objective[0]) and objective[1] == 0
    mask, value = select_final_solution(
        [np.zeros(4, dtype=bool), np.array([1, 0, 0, 0], dtype=bool)],
        [np.array([np.inf, 0.0]), np.array([0.2, 0.25])],
    )
    assert mask.any() and np.isfinite(value[0])


def test_run_is_reproducible_and_records_budget() -> None:
    first = run_pep_paper(_evaluator(), iterations=8, seed=17)
    second = run_pep_paper(_evaluator(), iterations=8, seed=17)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.F, second.F)
    assert first.F.shape[1] == 2
    assert first.algorithm.iterations == 8
    assert first.algorithm.evaluation_count >= 9  # initial point plus iterations
    assert np.all(first.X.any(axis=1))
    assert default_pep_iterations(4) == 23
