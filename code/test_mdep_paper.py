"""Unit tests for the paper-faithful MDEP implementation."""

import numpy as np
from pymoo.core.population import Population

from ensemble_pruning import EnsembleEvaluator
from mdep_paper import (
    MDEPProblem,
    MDEPRepair,
    build_mdep_algorithm,
    margin_ratio,
    run_mdep_paper,
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


def test_margin_ratio_is_finite_for_nonempty_binary_subset() -> None:
    evaluator = _evaluator()
    value = margin_ratio(evaluator.predictions, evaluator.labels, np.array([1, 1, 0, 0]))
    assert np.isfinite(value)
    assert value >= 0


def test_mdep_problem_has_three_objectives_and_excludes_empty_subset() -> None:
    problem = MDEPProblem(_evaluator())
    assert problem.n_var == 4
    out = {}
    problem._evaluate(np.zeros(4, dtype=bool), out)
    assert np.array_equal(out["F"], [1e6, 1e6, 1e6])


def test_final_selection_uses_validation_error_then_size() -> None:
    masks = [np.array([1, 1, 0]), np.array([1, 0, 0])]
    objectives = [np.array([0.1, 0.4, 0.67]), np.array([0.1, 0.8, 0.33])]
    mask, value = select_final_solution(masks, objectives)
    assert np.array_equal(mask, masks[1])
    assert np.array_equal(value, objectives[1])


def test_initial_singletons_are_preserved_but_offspring_are_repaired() -> None:
    problem = MDEPProblem(_evaluator())
    algorithm = build_mdep_algorithm(problem, pop_size=4, seed=11)

    initial = algorithm.initialization.do(problem, 4).get("X")
    assert np.any(np.sum(initial, axis=1) == 1)
    assert isinstance(algorithm.mating.repair, MDEPRepair)

    invalid_offspring = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0]], dtype=bool
    )
    repaired = algorithm.mating.repair.do(
        problem, Population.new("X", invalid_offspring)
    ).get("X")
    assert np.all(np.sum(repaired, axis=1) >= 2)


def test_mdep_smoke_is_reproducible() -> None:
    first = run_mdep_paper(_evaluator(), pop_size=4, n_gen=2, seed=11)
    second = run_mdep_paper(_evaluator(), pop_size=4, n_gen=2, seed=11)
    assert first.F.shape[1] == 3
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.F, second.F)
    assert np.all(np.sum(first.X, axis=1) >= 1)
