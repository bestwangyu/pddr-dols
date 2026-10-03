"""Tests for rank-preserving PDDR survival."""

import numpy as np
import pytest
from pymoo.core.population import Population

from pddr_survival import select_pddr_survivors
from rank_pddr_survival import (
    ExtremePDDRSurvival,
    HybridPDDRSurvival,
    RankPDDRSurvival,
    extreme_pddr_parent_keys,
    hybrid_pddr_parent_keys,
    rank_pddr_parent_keys,
    select_extreme_pddr_survivors,
    select_hybrid_pddr_survivors,
    select_rank_pddr_survivors,
)


def test_rank_pddr_keeps_unique_extreme_tradeoffs() -> None:
    """Duplicate early indices cannot displace unique non-dominated extremes."""
    objectives = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    decisions = np.eye(5, dtype=int)

    pure = select_pddr_survivors(objectives, n_survive=3)
    result = select_rank_pddr_survivors(objectives, decisions, n_survive=3)

    assert np.array_equal(pure.selected_indices, [0, 1, 2])
    assert set(result.selected_indices) == {0, 3, 4}
    assert result.selected_unique_objective_count == 3


def test_rank_pddr_does_not_skip_a_better_pareto_layer() -> None:
    """Dominated candidates cannot replace distinct rank-zero objectives."""
    objectives = np.array(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    )
    decisions = np.eye(4, dtype=int)
    result = select_rank_pddr_survivors(objectives, decisions, n_survive=2)

    assert set(result.selected_indices) == {0, 1}
    assert np.all(result.pareto_ranks[result.selected_indices] == 0)


def test_rank_pddr_adapter_returns_selected_population() -> None:
    """The pymoo adapter returns the same selected survivors and diagnostics."""
    objectives = np.array(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    decisions = np.eye(4, dtype=int)
    population = Population.new("X", decisions, "F", objectives)
    survival = RankPDDRSurvival()
    selected = survival.do(None, population, n_survive=2)

    assert {tuple(row) for row in selected.get("F")} == {(0.0, 1.0), (1.0, 0.0)}
    assert survival.last_result is not None
    assert survival.last_result.selected_unique_objective_count == 2


def test_parent_keys_prioritize_rank_then_objective_spacing() -> None:
    """Parent tournaments can protect non-dominated separated candidates."""
    objectives = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0]]
    )
    decisions = np.eye(4, dtype=int)
    ranks, spacing, pddr_values, hamming = rank_pddr_parent_keys(
        objectives, decisions
    )

    assert np.array_equal(ranks, [0, 0, 0, 1])
    assert spacing[0] > 0.0
    assert spacing[3] > 0.0
    assert np.isfinite(pddr_values).all()
    assert np.isfinite(hamming).all()


def test_hybrid_keeps_convergence_priority_with_limited_spreading() -> None:
    """Hybrid selection keeps most slots on the best PDDR candidates."""
    objectives = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.2, 0.8, 0.8],
            [0.4, 0.6, 0.6],
            [0.6, 0.4, 0.4],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )
    decisions = np.eye(len(objectives), dtype=int)
    result = select_hybrid_pddr_survivors(
        objectives, decisions, n_survive=4, diversity_fraction=0.25
    )

    assert len(result.selected_indices) == 4
    assert result.selected_unique_objective_count == 4
    assert np.all(result.pareto_ranks[result.selected_indices] == 0)


def test_hybrid_rejects_invalid_diversity_fraction() -> None:
    """Diversity allocation is explicitly bounded."""
    objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
    decisions = np.eye(2, dtype=int)
    with pytest.raises(ValueError, match="diversity_fraction"):
        select_hybrid_pddr_survivors(
            objectives, decisions, n_survive=1, diversity_fraction=1.5
        )


def test_hybrid_adapter_uses_configured_fraction() -> None:
    """The pymoo adapter exposes its hybrid selection diagnostics."""
    objectives = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    decisions = np.eye(3, dtype=int)
    population = Population.new("X", decisions, "F", objectives)
    survival = HybridPDDRSurvival(diversity_fraction=0.25)
    selected = survival.do(None, population, n_survive=2)

    assert len(selected) == 2
    assert survival.last_result is not None
    assert survival.last_result.selected_unique_objective_count == 2


def test_hybrid_parent_keys_keep_pddr_as_primary_after_rank() -> None:
    """Hybrid parent keys return finite rank, PDDR and spacing arrays."""
    objectives = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    decisions = np.eye(3, dtype=int)
    ranks, pddr_values, spacing, hamming = hybrid_pddr_parent_keys(
        objectives, decisions
    )
    assert np.array_equal(ranks, [0, 0, 0])
    assert np.isfinite(pddr_values).all()
    assert np.isfinite(spacing).all()
    assert np.isfinite(hamming).all()


def test_extreme_pddr_forces_each_objective_minimum() -> None:
    """Three objective-wise minima survive a three-slot truncation."""
    objectives = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.4, 0.4, 0.4],
            [0.5, 0.3, 0.5],
        ]
    )
    decisions = np.eye(len(objectives), dtype=int)
    result = select_extreme_pddr_survivors(objectives, decisions, n_survive=3)

    assert set(result.selected_indices) == {0, 1, 2}
    assert result.selected_unique_objective_count == 3


def test_extreme_parent_keys_expose_rank_crowding_then_pddr() -> None:
    """Parent selection receives valid rank, crowding and PDDR arrays."""
    objectives = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0]]
    )
    decisions = np.eye(4, dtype=int)
    ranks, crowding, pddr_values, hamming = extreme_pddr_parent_keys(
        objectives, decisions
    )

    assert np.array_equal(ranks, [0, 0, 0, 1])
    assert np.isinf(crowding[0])
    assert np.isinf(crowding[1])
    assert np.isfinite(pddr_values).all()
    assert np.isfinite(hamming).all()


def test_extreme_pddr_adapter_preserves_unique_objectives() -> None:
    """The pymoo adapter removes objective duplicates before filling."""
    objectives = np.array(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    )
    decisions = np.eye(4, dtype=int)
    population = Population.new("X", decisions, "F", objectives)
    survival = ExtremePDDRSurvival()
    selected = survival.do(None, population, n_survive=3)

    assert len(selected) == 3
    assert survival.last_result is not None
    assert survival.last_result.selected_unique_objective_count == 3
