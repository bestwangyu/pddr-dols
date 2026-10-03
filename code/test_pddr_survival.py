"""Tests for direct PDDR-FF survival selection."""

import numpy as np
import pytest
from pymoo.core.population import Population

from pddr_survival import PDDRSurvival, select_pddr_survivors


def test_survival_selects_lowest_pddr_values_stably() -> None:
    """The direct baseline selects the first stable members at a tie boundary."""
    objectives = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 4.0],
        ]
    )
    result = select_pddr_survivors(objectives, n_survive=2)

    assert np.array_equal(result.selected_indices, [0, 1])
    assert result.nondominated_ratio == pytest.approx(0.75)
    assert result.unique_eval_ratio == pytest.approx(0.5)
    assert result.boundary_tie_size == 3


def test_all_non_dominated_population_exposes_tie_degeneracy() -> None:
    """A population with identical trade-off points has one PDDR value."""
    objectives = np.tile(np.array([[1.0, 1.0, 1.0]]), (6, 1))
    result = select_pddr_survivors(objectives, n_survive=3)

    assert np.array_equal(result.selected_indices, [0, 1, 2])
    assert result.nondominated_ratio == pytest.approx(1.0)
    assert result.unique_eval_ratio == pytest.approx(1.0 / 6.0)
    assert result.boundary_tie_size == 6


def test_pymoo_population_adapter_returns_selected_population() -> None:
    """The adapter preserves the selected objective rows in a pymoo Population."""
    objectives = np.array(
        [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]]
    )
    population = Population.new("X", np.arange(4).reshape(-1, 1), "F", objectives)
    survival = PDDRSurvival()
    selected = survival.do(None, population, n_survive=2)

    assert len(selected) == 2
    assert np.array_equal(selected.get("X").ravel(), [0, 1])
    assert np.array_equal(selected.get("F"), objectives[[0, 1]])
    assert survival.last_result is not None
    assert survival.last_result.boundary_tie_size == 3


def test_survival_rejects_invalid_requests() -> None:
    """Reject malformed populations and impossible survivor counts."""
    with pytest.raises(ValueError, match="non-empty"):
        select_pddr_survivors(np.empty((0, 2)), n_survive=1)
    with pytest.raises(ValueError, match="between 1"):
        select_pddr_survivors(np.ones((2, 2)), n_survive=3)
    with pytest.raises(ValueError, match="integer"):
        select_pddr_survivors(np.ones((2, 2)), n_survive=1.5)
