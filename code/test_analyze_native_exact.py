"""Tests for PEP/MDEP native objective-space analysis."""

import numpy as np

from analyze_native_exact import analyze_native_exact


def _row(dataset, algorithm, n_objectives, front, *, optimization_seed=None):
    return {
        "dataset": dataset,
        "seed": 1,
        "fold": 0,
        "optimization_seed": optimization_seed,
        "algorithm": algorithm,
        "n_objectives": n_objectives,
        "raw_front": front,
        "evaluated_subsets": 7,
        "objective_context_id": "ctx",
        "test_error": 0.1,
        "selected_count": 1,
        "algorithm_runtime_seconds": 0.01,
    }


def test_native_analysis_keeps_pep_and_mdep_spaces_separate() -> None:
    records = [
        _row("toy", "exact_pep_native", 2, [[0.0, 0.5], [0.2, 0.25]]),
        _row("toy", "pep_paper", 2, [[0.0, 0.5]], optimization_seed=10),
        _row("toy", "exact_mdep_native", 3, [[0.0, 0.5, 0.5], [0.2, 0.2, 0.25]]),
        _row("toy", "mdep_paper", 3, [[0.0, 0.5, 0.5]], optimization_seed=10),
    ]
    report = analyze_native_exact(records)
    assert report["integrity"]["native_metric_records"] == 2
    assert {row["native_objective_space"] for row in report["records"]} == {
        "pep_native",
        "mdep_native",
    }
    assert all(np.isfinite(row["igd_plus"]) for row in report["records"])
