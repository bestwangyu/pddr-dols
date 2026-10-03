"""Tests for per-instance unified metric aggregation."""

import numpy as np

from compute_unified_metrics import _merge_record_sources, compute_unified_metrics


def _record(algorithm: str, n_objectives: int, front: list[list[float]]) -> dict:
    """Build one compact synthetic batch record."""
    return {
        "dataset": "toy",
        "seed": 7,
        "fold": 0,
        "algorithm": algorithm,
        "n_objectives": n_objectives,
        "raw_front": front,
        "test_error": 0.1,
        "test_margin_loss": 0.2,
        "selected_count": 2,
    }


def test_exact_reference_is_used_for_three_objective_group() -> None:
    """Exact reference metadata is propagated to every three-objective method."""
    context = "shared-context"
    batch = [
        {**_record("pddr", 3, [[0.2, 0.2, 0.2], [0.5, 0.1, 0.4]]), "objective_context_id": context},
        {**_record("da_pddr", 3, [[0.3, 0.2, 0.2], [0.4, 0.1, 0.5]]), "objective_context_id": context},
        _record("pep", 2, [[0.2, 0.4], [0.3, 0.2]]),
    ]
    exact = [
        {
            "dataset": "toy",
            "seed": 7,
            "fold": 0,
            "algorithm": "exact_pareto",
            "raw_front": [[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]],
            "objective_context_id": context,
        }
    ]

    result = compute_unified_metrics(batch, exact)
    rows = result["records"]
    assert len(rows) == 3
    assert {row["reference_type"] for row in rows if row["n_objectives"] == 3} == {
        "exact_pareto"
    }
    pep = next(row for row in rows if row["algorithm"] == "pep")
    assert pep["n_objectives"] == 2
    assert pep["reference_type"] == "merged_algorithm_front"
    assert all(np.isfinite(row["hypervolume"]) for row in rows)
    assert all(np.isfinite(row["igd_plus"]) for row in rows)


def test_exact_reference_rejects_mismatched_context() -> None:
    """A separately measured cost space cannot be used as an exact reference."""
    batch = [{**_record("pddr", 3, [[0.2, 0.2, 0.2]]), "objective_context_id": "batch"}]
    exact = [{
        "dataset": "toy", "seed": 7, "fold": 0, "algorithm": "exact_pareto",
        "raw_front": [[0.1, 0.2, 0.3]], "objective_context_id": "exact",
    }]
    import pytest

    with pytest.raises(ValueError, match="objective_context_id"):
        compute_unified_metrics(batch, exact)


def test_missing_exact_reference_falls_back_to_merged_front() -> None:
    """Three-objective groups without exact files use same-dimension merging."""
    batch = [
        _record("pddr", 3, [[0.2, 0.2, 0.2]]),
        _record("mdep", 3, [[0.3, 0.3, 0.3]]),
    ]
    result = compute_unified_metrics(batch)
    assert {row["reference_type"] for row in result["records"]} == {
        "merged_algorithm_front"
    }


def test_merge_accepts_distinct_optimization_seeds() -> None:
    """Replicated runs are unique by optimization seed, not only outer fold."""
    first = {**_record("nsga2", 3, [[0.2, 0.2, 0.2]]), "optimization_seed": 101}
    second = {**_record("nsga2", 3, [[0.3, 0.2, 0.2]]), "optimization_seed": 102}

    merged = _merge_record_sources([first, second])

    assert len(merged) == 2
