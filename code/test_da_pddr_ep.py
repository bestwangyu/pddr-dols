"""Tests for degeneracy-aware PDDR survival."""

import numpy as np
import pytest
from pymoo.core.population import Population

from da_pddr_ep import (
    DAPDDRSurvival,
    ObjectiveSpaceDASurvival,
    select_da_pddr_survivors,
    select_objective_da_pddr_survivors,
)
from pddr_survival import select_pddr_survivors


def test_hamming_compensation_changes_a_degenerate_boundary() -> None:
    """DA selection spreads survivors when all PDDR values are tied."""
    objectives = np.ones((4, 3))
    decisions = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
    )

    pure = select_pddr_survivors(objectives, n_survive=2)
    aware = select_da_pddr_survivors(objectives, decisions, n_survive=2)

    assert np.array_equal(pure.selected_indices, [0, 1])
    assert np.array_equal(aware.selected_indices, [0, 2])
    assert aware.compensation_activated
    assert aware.boundary_tie_size == 4
    assert aware.boundary_slots == 2
    assert aware.selected_mean_hamming == pytest.approx(1.0)


def test_non_boundary_pddr_order_is_preserved() -> None:
    """Individuals strictly better than the cutoff cannot be displaced."""
    objectives = np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 4.0],
        ]
    )
    decisions = np.array(
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]]
    )
    result = select_da_pddr_survivors(objectives, decisions, n_survive=2)

    assert result.selected_indices[0] == 0
    assert result.compensation_activated


def test_no_compensation_when_entire_boundary_fits() -> None:
    """A non-truncated boundary retains deterministic PDDR order."""
    objectives = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    decisions = np.eye(3, dtype=int)
    result = select_da_pddr_survivors(objectives, decisions, n_survive=3)

    assert np.array_equal(result.selected_indices, [0, 1, 2])
    assert not result.compensation_activated


def test_pymoo_adapter_uses_objectives_and_decisions() -> None:
    """The pymoo adapter returns the same diverse boundary survivors."""
    objectives = np.ones((4, 3))
    decisions = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0]]
    )
    population = Population.new("X", decisions, "F", objectives)
    survival = DAPDDRSurvival()
    selected = survival.do(None, population, n_survive=2)

    assert np.array_equal(selected.get("X"), decisions[[0, 2]])
    assert survival.last_result is not None
    assert survival.last_result.compensation_activated


def test_objective_space_compensation_prefers_target_diversity() -> None:
    """Objective-space DA selects separated objective vectors at the boundary."""
    objectives = np.array(
        [
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.5, 0.5, 0.5],
        ]
    )
    decisions = np.eye(4, dtype=int)
    result = select_objective_da_pddr_survivors(
        objectives, decisions, n_survive=2, boundary_tolerance=1.0
    )

    assert result.compensation_activated
    assert len(result.selected_indices) == 2
    assert result.selected_mean_hamming > 0.0


def test_objective_space_adapter_records_result() -> None:
    """The objective-space pymoo adapter exposes its diagnostics."""
    objectives = np.ones((4, 3))
    decisions = np.eye(4, dtype=int)
    population = Population.new("X", decisions, "F", objectives)
    survival = ObjectiveSpaceDASurvival(boundary_tolerance=0.05)
    selected = survival.do(None, population, n_survive=2)

    assert len(selected) == 2
    assert survival.last_result is not None


def test_invalid_decision_matrix_is_rejected() -> None:
    """Decision vectors must align with objectives and remain binary."""
    with pytest.raises(ValueError, match="one row"):
        select_da_pddr_survivors(np.ones((3, 2)), np.ones((2, 4)), 2)
    with pytest.raises(ValueError, match="binary"):
        select_da_pddr_survivors(np.ones((3, 2)), np.full((3, 4), 0.5), 2)
