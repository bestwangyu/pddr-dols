"""Tests for PDDR-triggered structured subset neighborhood search."""

import numpy as np

from ensemble_pruning import EnsembleEvaluator
from pddr_local_search import generate_structured_neighbors, run_pddr_local_search


def _small_evaluator() -> EnsembleEvaluator:
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


def test_structured_neighbors_are_unique_nonempty_and_budget_bounded() -> None:
    decisions = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=bool,
    )
    objectives = np.array(
        [
            [0.1, 0.8, 0.8],
            [0.2, 0.6, 0.6],
            [0.3, 0.4, 0.4],
            [0.4, 0.2, 0.2],
        ]
    )
    neighbors, operations = generate_structured_neighbors(
        decisions, objectives, n_neighbors=9
    )

    assert 0 < len(neighbors) <= 9
    assert len(neighbors) == len(operations)
    assert np.all(neighbors.sum(axis=1) >= 1)
    assert len(np.unique(neighbors, axis=0)) == len(neighbors)
    existing = {tuple(row.tolist()) for row in decisions}
    assert not any(tuple(row.tolist()) in existing for row in neighbors)
    assert set(operations).issubset({"add", "delete", "swap"})


def test_structured_neighbors_respect_operation_ablation() -> None:
    """Component ablations emit only their selected neighborhood operation."""
    decisions = np.array(
        [[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=bool
    )
    objectives = np.array(
        [[0.1, 0.8, 0.8], [0.2, 0.6, 0.6], [0.3, 0.4, 0.4]]
    )
    for operation in ("add", "delete", "swap"):
        _, operations = generate_structured_neighbors(
            decisions, objectives, n_neighbors=4, allowed_operations=(operation,)
        )
        assert operations
        assert set(operations) == {operation}


def test_local_search_replaces_offspring_without_changing_population_budget() -> None:
    _, _, result = run_pddr_local_search(
        _small_evaluator(),
        pop_size=8,
        n_gen=5,
        seed=19,
        local_fraction=0.5,
        pddr_unique_threshold=1.0,
        objective_unique_threshold=1.0,
        stagnation_patience=1,
    )
    diagnostics = result.algorithm.search_diagnostics

    assert len(result.algorithm.pop) == 8
    assert diagnostics["trigger_count"] == 4
    assert diagnostics["local_candidate_count"] > 0
    assert sum(diagnostics["operation_counts"].values()) == diagnostics[
        "local_candidate_count"
    ]
    assert len(diagnostics["events"]) == diagnostics["trigger_count"]
    assert np.isfinite(result.F).all()


def test_always_local_search_triggers_each_infill_generation() -> None:
    """Always-on ablation uses the same local fraction on every generation."""
    _, _, result = run_pddr_local_search(
        _small_evaluator(),
        pop_size=8,
        n_gen=5,
        seed=19,
        local_fraction=0.5,
        trigger_mode="always",
    )
    diagnostics = result.algorithm.search_diagnostics

    assert diagnostics["trigger_mode"] == "always"
    assert diagnostics["trigger_count"] == 4
    assert diagnostics["reason_counts"]["always_on"] == 4
    assert 0 < diagnostics["local_candidate_count"] <= 16
    assert len(result.algorithm.pop) == 8
    assert all(event["trigger_mode"] == "always" for event in diagnostics["events"])


def test_component_local_search_records_allowed_operations() -> None:
    """A component run preserves the common evaluation budget and audit."""
    _, _, result = run_pddr_local_search(
        _small_evaluator(),
        pop_size=8,
        n_gen=4,
        seed=19,
        trigger_mode="pddr",
        allowed_operations=("swap",),
    )
    diagnostics = result.algorithm.search_diagnostics
    assert diagnostics["allowed_operations"] == ["swap"]
    assert diagnostics["operation_counts"]["add"] == 0
    assert diagnostics["operation_counts"]["delete"] == 0
    assert diagnostics["operation_counts"]["swap"] == diagnostics[
        "local_candidate_count"
    ]
