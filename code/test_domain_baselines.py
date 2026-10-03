"""Tests for PEP and MDEP objective-modeling baselines."""

import numpy as np
import pytest

from domain_baselines import DomainPruningProblem, run_domain_baseline
from ensemble_pruning import EnsembleEvaluator


@pytest.fixture
def small_evaluator() -> EnsembleEvaluator:
    """Create a deterministic evaluator for domain objective tests."""
    predictions = np.array(
        [[0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    )
    labels = np.array([0, 1, 1, 1, 0])
    return EnsembleEvaluator(predictions, labels, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("variant, n_obj", [("pep", 2), ("mdep", 3)])
def test_domain_problem_has_expected_objective_model(
    small_evaluator: EnsembleEvaluator, variant: str, n_obj: int
) -> None:
    """PEP uses count as its second target; MDEP uses margin and count."""
    problem = DomainPruningProblem(small_evaluator, variant=variant)
    values = problem.evaluate(np.array([[1, 0, 0], [1, 1, 0]], dtype=bool))

    assert problem.n_obj == n_obj
    assert values.shape == (2, n_obj)
    assert np.all(np.isfinite(values))
    assert np.all(values[:, -1] > 0)


@pytest.mark.parametrize("variant", ["pep", "mdep"])
def test_domain_baseline_runs(
    small_evaluator: EnsembleEvaluator, variant: str
) -> None:
    """Both objective-modeling baselines run with the common binary DE."""
    _, problem, result = run_domain_baseline(
        small_evaluator, variant=variant, pop_size=8, n_gen=2, seed=21
    )
    assert result.problem is problem
    assert result.F.shape[1] == (2 if variant == "pep" else 3)
    assert np.isfinite(result.F).all()


def test_domain_baseline_rejects_unknown_variant(small_evaluator: EnsembleEvaluator) -> None:
    """Only explicit PEP and MDEP objective models are accepted."""
    with pytest.raises(ValueError, match="variant"):
        DomainPruningProblem(small_evaluator, variant="unknown")
