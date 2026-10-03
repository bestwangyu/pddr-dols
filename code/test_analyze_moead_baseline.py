"""Tests for the formal MOEA/D general-baseline report."""

from analyze_moead_baseline import ALL_ALGORITHMS, analyze_moead_baseline


def _moead_parameters() -> dict:
    return {
        "method_name": "MOEA/D",
        "decomposition": "PBI",
        "pbi_theta": 5.0,
        "n_neighbors": 20,
        "prob_neighbor_mating": 0.9,
        "reference_direction_method": "energy",
        "reference_direction_count": 40,
        "reference_direction_seed": 20260820,
        "variation": "de",
        "de_F": 0.5,
        "de_K": 0.5,
        "de_CR": 0.9,
    }


def test_formal_report_validates_counts_and_compares_moead() -> None:
    datasets = ("a", "b")
    optimization_seeds = (11, 12)
    raw = []
    metrics = []
    for dataset_index, dataset in enumerate(datasets):
        for fold in range(2):
            context_id = f"{dataset}-{fold}"
            context = {
                "inference_seconds": [1.0, 2.0],
                "storage_bytes": [10.0, 20.0],
                "combined_costs": [0.25, 0.75],
            }
            raw.append(
                {
                    "dataset": dataset,
                    "seed": 7,
                    "fold": fold,
                    "optimization_seed": None,
                    "algorithm": "exact_pareto",
                    "objective_context_id": context_id,
                }
            )
            for optimization_seed in optimization_seeds:
                for algorithm in ALL_ALGORITHMS:
                    offset = {
                        "pddr_local_search_delete_only": 0.0,
                        "nsga2": 0.04,
                        "moead": 0.03,
                        "local_search_always_delete_only": 0.01,
                    }[algorithm]
                    raw.append(
                        {
                            "dataset": dataset,
                            "seed": 7,
                            "fold": fold,
                            "optimization_seed": optimization_seed,
                            "algorithm": algorithm,
                            "evaluation_count": 2000,
                            "objective_context_id": context_id,
                            "objective_context": context,
                            "algorithm_parameters": (
                                _moead_parameters() if algorithm == "moead" else {}
                            ),
                            "best_mask": [1, 0],
                            "selected_count": 1,
                            "algorithm_runtime_seconds": 1.0 + offset,
                        }
                    )
                    metrics.append(
                        {
                            "dataset": dataset,
                            "seed": 7,
                            "fold": fold,
                            "optimization_seed": optimization_seed,
                            "algorithm": algorithm,
                            "reference_type": "exact_pareto",
                            "igd_plus": 0.10 + offset + dataset_index * 0.001,
                            "hypervolume": 0.90 - offset - dataset_index * 0.001,
                            "test_error": 0.20 + offset,
                        }
                    )

    report = analyze_moead_baseline(
        {"records": metrics},
        raw,
        datasets=datasets,
        data_seed=7,
        optimization_seeds=optimization_seeds,
        n_splits=2,
        expected_evaluations=2000,
        bootstrap_samples=1000,
        bootstrap_seed=3,
    )

    assert report["integrity"]["raw_algorithm_records"] == 32
    assert report["integrity"]["exact_records"] == 4
    assert report["moead_evidence"]["igd_plus_nonloss_datasets"] == 2
    assert report["moead_evidence"]["hypervolume_nonloss_datasets"] == 2
    assert {
        row["algorithm"]
        for row in report["main_front_metrics"]["igd_plus"][
            "comparisons_from_pddr"
        ]
    } == {"nsga2", "moead"}
