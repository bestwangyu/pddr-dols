"""Tests for matched NSGA-II, reference-direction and MOEA/D baselines."""

import numpy as np
import pytest

from ensemble_pruning import EnsembleEvaluator
from fair_baselines import (
    MOEAD_REFERENCE_SEED,
    make_reference_directions,
    run_baseline,
    run_fair_comparison,
)


@pytest.fixture
def small_evaluator() -> EnsembleEvaluator:
    """Create a small deterministic three-objective pruning problem."""
    predictions = np.array(
        [
            [0, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 1],
        ]
    )
    labels = np.array([0, 1, 1, 1, 0, 1])
    return EnsembleEvaluator(predictions, labels, np.array([1.0, 2.0, 3.0, 4.0]))


def test_reference_directions_are_simplex_points() -> None:
    """Directions are non-negative and each sums to one."""
    directions = make_reference_directions(8, n_obj=3, seed=5)
    assert directions.shape == (8, 3)
    assert np.all(directions >= 0.0)
    assert np.allclose(directions.sum(axis=1), 1.0)
    assert all(
        any(np.allclose(direction, axis) for direction in directions)
        for axis in np.eye(3)
    )


def test_reference_directions_are_frozen_across_optimizer_seeds() -> None:
    """MOEA/D decomposition geometry is independent of search replication."""
    first = make_reference_directions(8, n_obj=3, seed=MOEAD_REFERENCE_SEED)
    second = make_reference_directions(8, n_obj=3, seed=MOEAD_REFERENCE_SEED)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("variant", ["nsga2", "reference", "moead"])
def test_each_baseline_runs_with_shared_de_variation(
    small_evaluator: EnsembleEvaluator, variant: str
) -> None:
    """Every baseline returns finite three-objective solutions."""
    _, problem, result = run_baseline(
        small_evaluator, variant=variant, pop_size=8, n_gen=2, seed=13
    )
    assert result.problem is problem
    assert result.F.shape[1] == 3
    assert result.X.shape[1] == 4
    assert np.isfinite(result.F).all()
    assert np.all(np.asarray(result.X).sum(axis=1) >= 1)


def test_fair_comparison_returns_all_requested_variants(
    small_evaluator: EnsembleEvaluator,
) -> None:
    """The common runner labels all three baselines distinctly."""
    result = run_fair_comparison(
        small_evaluator, pop_size=8, n_gen=2, seed=13
    )
    assert set(result) == {"nsga2", "reference", "moead"}
    assert all(np.isfinite(value.F).all() for value in result.values())


def test_baseline_rejects_unknown_variant(small_evaluator: EnsembleEvaluator) -> None:
    """Unknown baseline names cannot silently change the comparison."""
    with pytest.raises(ValueError, match="variant"):
        run_baseline(small_evaluator, variant="pddr", pop_size=8, n_gen=1)
