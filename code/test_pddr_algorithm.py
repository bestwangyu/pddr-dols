"""Tests for the binary PDDR/DA-PDDR optimization pilot."""

import numpy as np
import pytest

from ensemble_pruning import EnsembleEvaluator
from pddr_algorithm import run_binary_pddr


@pytest.fixture
def small_evaluator() -> EnsembleEvaluator:
    """Return a deterministic cached-prediction problem with four classifiers."""
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


@pytest.mark.parametrize(
    "variant",
    [
        "pddr",
        "da_pddr",
        "da_pddr_obj",
        "rank_pddr",
        "hybrid_pddr",
        "extreme_pddr",
    ],
)
@pytest.mark.parametrize("variation", ["ga", "de"])
def test_binary_driver_runs_both_survival_variants(
    small_evaluator: EnsembleEvaluator, variant: str, variation: str
) -> None:
    """Both variants produce finite three-objective non-dominated results."""
    algorithm, problem, result = run_binary_pddr(
        small_evaluator,
        variant=variant,
        pop_size=8,
        n_gen=3,
        variation=variation,
        seed=9,
    )

    assert result.problem is problem
    assert result.F.shape[1] == 3
    assert result.X.shape[1] == 4
    assert np.isfinite(result.F).all()
    assert len(result.X) >= 1
    assert np.all(np.asarray(result.X).sum(axis=1) >= 1)


def test_differential_crossover_returns_binary_children() -> None:
    """The HMODE-style donor is thresholded to a valid binary matrix."""
    from pddr_algorithm import BinaryDifferentialCrossover

    crossover = BinaryDifferentialCrossover(F=0.5, K=0.5, CR=0.9)
    parents = np.array(
        [
            [[0, 0, 0], [1, 1, 1]],
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 1]],
            [[0, 1, 1], [1, 0, 0]],
        ],
        dtype=bool,
    )
    children = crossover._do(None, parents)

    assert children.shape == (1, 2, 3)
    assert np.isin(children, [False, True]).all()


def test_binary_driver_rejects_unknown_variant(small_evaluator: EnsembleEvaluator) -> None:
    """Variant names must be explicit to keep result labels unambiguous."""
    with pytest.raises(ValueError, match="variant"):
        run_binary_pddr(small_evaluator, variant="nsga2", n_gen=1)


def test_objective_variant_receives_boundary_tolerance(
    small_evaluator: EnsembleEvaluator,
) -> None:
    """The experiment-facing parameter reaches objective-space survival."""
    algorithm, _, _ = run_binary_pddr(
        small_evaluator,
        variant="da_pddr_obj",
        boundary_tolerance=0.025,
        pop_size=8,
        n_gen=1,
        seed=9,
    )

    assert algorithm.survival.boundary_tolerance == pytest.approx(0.025)
