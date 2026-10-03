"""Tests for nested-fold batch orchestration."""

import json

from batch_runner import BatchConfig, run_batch


def test_small_batch_runs_outer_folds_and_domain_baseline(tmp_path) -> None:
    """One small dataset produces one record per fold and algorithm."""
    output = tmp_path / "batch.json"
    config = BatchConfig(
        datasets=("iris",),
        algorithms=("pddr", "pep"),
        seeds=(7,),
        n_splits=2,
        pool_size=3,
        pop_size=6,
        n_gen=1,
        timing_repeats=1,
    )
    records = run_batch(config, output)

    assert len(records) == 4
    assert {(row["fold"], row["algorithm"]) for row in records} == {
        (0, "pddr"),
        (0, "pep"),
        (1, "pddr"),
        (1, "pep"),
    }
    assert all(row["split_sizes"]["test"] == 75 for row in records)
    assert json.loads(output.read_text(encoding="utf-8"))[0]["dataset"] == "iris"
    assert all(row["objective_context_id"] for row in records)
    assert all(len(row["raw_masks"]) == len(row["raw_front"]) for row in records)


def test_joint_batch_shares_context_with_exact_front(tmp_path) -> None:
    """Algorithms and exact enumeration in one run share the cost context."""
    output = tmp_path / "joint.json"
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("pddr",),
            seeds=(7,),
            n_splits=2,
            pool_size=3,
            pop_size=6,
            n_gen=1,
            timing_repeats=1,
            include_exact_front=True,
            exact_max_classifiers=3,
        ),
        output,
    )
    assert len(records) == 4
    for fold in range(2):
        rows = [row for row in records if row["fold"] == fold]
        assert len({row["objective_context_id"] for row in rows}) == 1
        exact = next(row for row in rows if row["algorithm"] == "exact_pareto")
        assert exact["evaluated_subsets"] == 7
        assert len(exact["pareto_masks"]) == len(exact["raw_front"])


def test_batch_config_rejects_unknown_algorithm() -> None:
    """Misspelled algorithm names fail before data loading."""
    try:
        run_batch(BatchConfig(datasets=("iris",), algorithms=("bad",)))
    except ValueError as exc:
        assert "unknown algorithms" in str(exc)
    else:
        raise AssertionError("invalid batch configuration was accepted")


def test_batch_expands_and_records_objective_tolerances(tmp_path) -> None:
    """Sensitivity variants share folds but retain unambiguous labels."""
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("da_pddr_obj",),
            seeds=(7,),
            n_splits=2,
            pool_size=3,
            pop_size=6,
            n_gen=1,
            timing_repeats=1,
            objective_boundary_tolerances=(0.01, 0.025),
        ),
        tmp_path / "sensitivity.json",
    )

    assert len(records) == 4
    assert {row["algorithm"] for row in records} == {
        "da_pddr_obj_tol_0p01",
        "da_pddr_obj_tol_0p025",
    }
    assert {
        row["algorithm_parameters"]["boundary_tolerance"] for row in records
    } == {0.01, 0.025}


def test_batch_records_pddr_local_search_diagnostics(tmp_path) -> None:
    """The new search mechanism retains its trigger audit in batch JSON."""
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("pddr_local_search",),
            seeds=(7,),
            n_splits=2,
            pool_size=4,
            pop_size=8,
            n_gen=3,
            timing_repeats=1,
        ),
        tmp_path / "local_search.json",
    )

    assert len(records) == 2
    assert all("search_diagnostics" in row for row in records)
    assert all(
        row["algorithm_parameters"]["local_fraction"] == 0.5
        for row in records
    )
    assert all(
        row["search_diagnostics"]["local_candidate_count"] >= 0
        for row in records
    )


def test_batch_records_equal_budget_local_search_ablation(tmp_path) -> None:
    """The three ablation labels share folds and retain trigger metadata."""
    algorithms = ("nsga2", "local_search_always", "pddr_local_search")
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=algorithms,
            seeds=(7,),
            n_splits=2,
            pool_size=4,
            pop_size=8,
            n_gen=3,
            timing_repeats=1,
        ),
        tmp_path / "ablation.json",
    )

    assert len(records) == 6
    assert {(row["fold"], row["algorithm"]) for row in records} == {
        (fold, algorithm) for fold in range(2) for algorithm in algorithms
    }
    local_rows = [
        row for row in records if row["algorithm"] != "nsga2"
    ]
    assert {row["algorithm_parameters"]["trigger_mode"] for row in local_rows} == {
        "always",
        "pddr",
    }
    assert all(row["evaluation_count"] == 24 for row in records)
    assert all(0 < row["population_size"] <= 8 for row in records)


def test_batch_reuses_outer_folds_for_independent_optimization_seeds(tmp_path) -> None:
    """Optimizer seeds multiply algorithm rows without changing data folds."""
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("nsga2", "pddr_local_search_medium"),
            seeds=(7,),
            optimization_seeds=(101, 102, 103),
            n_splits=2,
            pool_size=3,
            pop_size=6,
            n_gen=1,
            timing_repeats=1,
        ),
        tmp_path / "optimization_seeds.json",
    )

    assert len(records) == 12
    assert {row["fold"] for row in records} == {0, 1}
    assert {row["optimization_seed"] for row in records} == {101, 102, 103}
    assert all(row["evaluation_count"] == 6 for row in records)
    assert {
        row["algorithm_parameters"]["pddr_unique_threshold"]
        for row in records
        if row["algorithm"] == "pddr_local_search_medium"
    } == {0.25}


def test_batch_records_frozen_moead_protocol(tmp_path) -> None:
    """Formal MOEA/D rows expose their fixed decomposition and variation."""
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("moead",),
            seeds=(7,),
            optimization_seeds=(101, 102),
            n_splits=2,
            pool_size=3,
            pop_size=6,
            n_gen=2,
            timing_repeats=1,
        ),
        tmp_path / "moead.json",
    )
    assert len(records) == 4
    assert {row["method_name"] for row in records} == {"MOEA/D"}
    assert {row["evaluation_count"] for row in records} == {12}
    parameters = [row["algorithm_parameters"] for row in records]
    assert {row["reference_direction_method"] for row in parameters} == {"energy"}
    assert {row["reference_direction_seed"] for row in parameters} == {20260820}
    assert len({row["reference_direction_sha256"] for row in parameters}) == 1
    assert {row["decomposition"] for row in parameters} == {"PBI"}
