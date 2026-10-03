"""Tests for formal PEP/MDEP joint statistics."""

from analyze_domain_baseline_statistics import analyze_domain_baseline_statistics


def _common(algorithm, opt_seed, *, dataset="toy", selected_count=2):
    return {
        "dataset": dataset, "seed": 1, "fold": 0, "optimization_seed": opt_seed,
        "algorithm": algorithm, "test_error": 0.1, "selected_count": selected_count,
        "algorithm_runtime_seconds": 0.2,
    }


def _native(space, algorithm, opt_seed):
    return {
        "dataset": "toy", "seed": 1, "fold": 0, "optimization_seed": opt_seed,
        "algorithm": algorithm, "native_objective_space": space,
        "hypervolume": 0.5, "igd_plus": 0.1, "hypervolume_gap": 0.05,
        "exact_point_coverage": 0.8, "test_error": 0.1, "selected_count": 2,
    }


def test_common_and_native_layers_are_separate():
    common = [
        _common(a, seed, dataset=dataset)
        for dataset in ("toy_a", "toy_b")
        for seed in (10, 11)
        for a in ("pep_paper", "mdep_paper", "nsga2", "pddr_local_search_delete_only")
    ]
    native = [
        _native(space, algorithm, seed)
        for seed in range(125)
        for space, algorithm in (("pep_native", "pep_paper"), ("mdep_native", "mdep_paper"))
    ]
    report = analyze_domain_baseline_statistics(common, native, bootstrap_samples=100, bootstrap_seed=7)
    assert len(report["common"]["datasets"]) == 2
    assert report["common"]["cross_dataset"]["test_error"]["n_datasets"] == 2
    assert len(report["native"]["groups"]) == 2
    assert report["native"]["protocol"]["cross_space_ranking"] is False


def test_comparisons_are_candidate_oriented_at_both_analysis_levels():
    selected = {
        "pep_paper": 1,
        "mdep_paper": 3,
        "nsga2": 2,
        "pddr_local_search_delete_only": 2,
    }
    common = [
        _common(
            algorithm,
            opt_seed,
            dataset=dataset,
            selected_count=selected[algorithm],
        )
        for dataset in ("toy_a", "toy_b")
        for opt_seed in (10, 11)
        for algorithm in selected
    ]
    native = [
        _native(space, algorithm, seed)
        for seed in range(125)
        for space, algorithm in (
            ("pep_native", "pep_paper"),
            ("mdep_native", "mdep_paper"),
        )
    ]
    report = analyze_domain_baseline_statistics(
        common, native, bootstrap_samples=100, bootstrap_seed=7
    )

    within = report["common"]["datasets"][0]["comparisons_to_nsga2"][
        "selected_count"
    ][0]
    assert within["algorithm"] == "pep_paper"
    assert (within["win"], within["tie"], within["loss"]) == (2, 0, 0)
    assert within["mean_improvement"] == 1.0

    cross = report["common"]["cross_dataset"]["selected_count"][
        "comparisons_to_nsga2"
    ][0]
    assert cross["algorithm"] == "pep_paper"
    assert (cross["win"], cross["tie"], cross["loss"]) == (2, 0, 0)
    assert cross["mean_improvement"] == 1.0
