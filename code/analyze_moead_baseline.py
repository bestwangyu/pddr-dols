"""Paper-scale analysis of PDDR-DOLS against NSGA-II and MOEA/D.

The independent inference unit is one dataset mean over five outer folds and
five optimization seeds.  Always delete-only is retained as a mechanism
ablation but is not mixed into the three-algorithm general-baseline family.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from statistics_analysis import (
    compare_against_reference,
    friedman_test,
    paired_wilcoxon,
    win_tie_loss,
)


PDDR = "pddr_local_search_delete_only"
NSGA2 = "nsga2"
MOEAD = "moead"
ALWAYS = "local_search_always_delete_only"
ALL_ALGORITHMS = (NSGA2, MOEAD, PDDR, ALWAYS)
MAIN_ALGORITHMS = (PDDR, NSGA2, MOEAD)
DISPLAY_NAMES = {
    PDDR: "PDDR-DOLS",
    NSGA2: "NSGA-II",
    MOEAD: "MOEA/D",
    ALWAYS: "Always delete-only",
}
FRONT_METRICS = {
    "igd_plus": False,
    "hypervolume": True,
    "test_error": False,
}
RESOURCE_METRICS = {
    "selected_count": False,
    "inference_seconds": False,
    "storage_bytes": False,
    "combined_cost": False,
    "algorithm_runtime_seconds": False,
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        int(row["fold"]),
        int(row["optimization_seed"]),
    )


def _rank_biserial(first, second, *, higher_is_better: bool) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    direction = 1.0 if higher_is_better else -1.0
    differences = (left - right) * direction
    nonzero = differences[np.abs(differences) > 1e-12]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float(
        (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    )


def _bootstrap_ci(values, *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap input must contain at least two finite values")
    if samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _add_comparison_details(
    comparisons: list[dict],
    scores: dict[str, np.ndarray],
    *,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    reference = scores[PDDR]
    direction = 1.0 if higher_is_better else -1.0
    for index, comparison in enumerate(comparisons):
        baseline_name = str(comparison["algorithm"])
        baseline = scores[baseline_name]
        raw_delta = reference - baseline
        oriented = raw_delta * direction
        comparison.update(
            {
                "reference_display_name": DISPLAY_NAMES[PDDR],
                "algorithm_display_name": DISPLAY_NAMES[baseline_name],
                "reference_mean": float(reference.mean()),
                "algorithm_mean": float(baseline.mean()),
                "mean_raw_delta": float(raw_delta.mean()),
                "mean_oriented_improvement": float(oriented.mean()),
                "mean_raw_delta_ci95": _bootstrap_ci(
                    raw_delta,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + index,
                ),
                "rank_biserial": _rank_biserial(
                    reference,
                    baseline,
                    higher_is_better=higher_is_better,
                ),
            }
        )
        if not higher_is_better and np.all(baseline > 0.0):
            reduction = 100.0 * (baseline - reference) / baseline
            comparison["mean_relative_reduction_percent"] = float(
                reduction.mean()
            )
            comparison["mean_relative_reduction_ci95"] = _bootstrap_ci(
                reduction,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 100 + index,
            )


def _representative_resource(row: dict, metric: str) -> float:
    if metric == "selected_count":
        return float(row["selected_count"])
    if metric == "algorithm_runtime_seconds":
        return float(row["algorithm_runtime_seconds"])
    mask = np.asarray(row["best_mask"], dtype=bool)
    context = row["objective_context"]
    source_name = {
        "inference_seconds": "inference_seconds",
        "storage_bytes": "storage_bytes",
        "combined_cost": "combined_costs",
    }[metric]
    costs = np.asarray(context[source_name], dtype=float)
    if mask.ndim != 1 or costs.shape != mask.shape:
        raise ValueError(
            f"representative mask/cost mismatch for {_key(row)} {row['algorithm']}"
        )
    return float(costs[mask].sum())


def _pairwise_ablation(
    pddr: np.ndarray,
    always: np.ndarray,
    *,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    direction = 1.0 if higher_is_better else -1.0
    raw_delta = pddr - always
    return {
        **paired_wilcoxon(pddr, always),
        **win_tie_loss(pddr, always, higher_is_better=higher_is_better),
        "reference": PDDR,
        "algorithm": ALWAYS,
        "reference_mean": float(pddr.mean()),
        "algorithm_mean": float(always.mean()),
        "mean_raw_delta": float(raw_delta.mean()),
        "mean_oriented_improvement": float((raw_delta * direction).mean()),
        "mean_raw_delta_ci95": _bootstrap_ci(
            raw_delta,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "rank_biserial": _rank_biserial(
            pddr,
            always,
            higher_is_better=higher_is_better,
        ),
    }


def analyze_moead_baseline(
    metrics_payload: dict,
    raw_records: list[dict],
    *,
    datasets: tuple[str, ...],
    data_seed: int,
    optimization_seeds: tuple[int, ...],
    n_splits: int,
    expected_evaluations: int,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260822,
) -> dict:
    """Validate and summarize the frozen four-arm joint experiment."""
    if len(datasets) < 2 or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must contain unique names")
    expected_instances = {
        (dataset, data_seed, fold, optimization_seed)
        for dataset in datasets
        for fold in range(n_splits)
        for optimization_seed in optimization_seeds
    }

    metric_groups: dict[tuple[str, int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in metrics_payload.get("records", []):
        algorithm = str(row.get("algorithm"))
        if algorithm not in ALL_ALGORITHMS:
            raise ValueError(f"unexpected metric algorithm: {algorithm!r}")
        key = _key(row)
        if algorithm in metric_groups[key]:
            raise ValueError(f"duplicate metric record for {key} {algorithm}")
        if row.get("reference_type") != "exact_pareto":
            raise ValueError(f"metric record lacks exact Pareto reference: {key}")
        values = np.asarray([row.get(metric) for metric in FRONT_METRICS], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite metric record for {key} {algorithm}")
        metric_groups[key][algorithm] = row
    if set(metric_groups) != expected_instances:
        missing = sorted(expected_instances - set(metric_groups))
        extra = sorted(set(metric_groups) - expected_instances)
        raise ValueError(f"metric instance mismatch; missing={missing}, extra={extra}")
    for key, rows in metric_groups.items():
        if set(rows) != set(ALL_ALGORITHMS):
            raise ValueError(f"incomplete metric algorithms for {key}: {sorted(rows)}")

    algorithm_rows = [
        row for row in raw_records if row.get("algorithm") in ALL_ALGORITHMS
    ]
    exact_rows = [row for row in raw_records if row.get("algorithm") == "exact_pareto"]
    raw_index = {(_key(row), str(row["algorithm"])): row for row in algorithm_rows}
    expected_algorithm_records = len(expected_instances) * len(ALL_ALGORITHMS)
    expected_exact_records = len(datasets) * n_splits
    if len(raw_index) != expected_algorithm_records:
        raise ValueError(
            f"raw algorithm records={len(raw_index)}, expected={expected_algorithm_records}"
        )
    if len(exact_rows) != expected_exact_records:
        raise ValueError(
            f"exact records={len(exact_rows)}, expected={expected_exact_records}"
        )

    for key in sorted(expected_instances):
        rows = [raw_index[(key, algorithm)] for algorithm in ALL_ALGORITHMS]
        if {int(row["evaluation_count"]) for row in rows} != {expected_evaluations}:
            raise ValueError(f"unequal evaluation budget for {key}")
        contexts = {str(row["objective_context_id"]) for row in rows}
        if len(contexts) != 1:
            raise ValueError(f"objective context mismatch for {key}")
        moead_parameters = raw_index[(key, MOEAD)]["algorithm_parameters"]
        expected_moead = {
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
        for name, expected in expected_moead.items():
            if moead_parameters.get(name) != expected:
                raise ValueError(
                    f"MOEA/D parameter mismatch for {key}: "
                    f"{name}={moead_parameters.get(name)!r}, expected={expected!r}"
                )

    dataset_means: dict[str, dict[str, dict[str, float]]] = {}
    dataset_rows: list[dict] = []
    for dataset in datasets:
        keys = sorted(key for key in expected_instances if key[0] == dataset)
        measures: dict[str, dict[str, float]] = {}
        for metric in FRONT_METRICS:
            measures[metric] = {
                algorithm: float(
                    np.mean([metric_groups[key][algorithm][metric] for key in keys])
                )
                for algorithm in ALL_ALGORITHMS
            }
        for metric in RESOURCE_METRICS:
            measures[metric] = {
                algorithm: float(
                    np.mean(
                        [
                            _representative_resource(
                                raw_index[(key, algorithm)],
                                metric,
                            )
                            for key in keys
                        ]
                    )
                )
                for algorithm in ALL_ALGORITHMS
            }
        dataset_means[dataset] = measures
        flat = {"dataset": dataset, "instances": len(keys)}
        for metric, algorithm_values in measures.items():
            for algorithm, value in algorithm_values.items():
                flat[f"{metric}__{algorithm}"] = value
        dataset_rows.append(flat)

    def score_map(metric: str, algorithms: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {
            algorithm: np.asarray(
                [dataset_means[dataset][metric][algorithm] for dataset in datasets],
                dtype=float,
            )
            for algorithm in algorithms
        }

    main_results = {}
    ablation_results = {}
    for metric_index, (metric, higher_is_better) in enumerate(FRONT_METRICS.items()):
        scores = score_map(metric, MAIN_ALGORITHMS)
        matrix = np.column_stack([scores[algorithm] for algorithm in MAIN_ALGORITHMS])
        ranks = np.vstack(
            [
                rankdata(-row if higher_is_better else row, method="average")
                for row in matrix
            ]
        )
        comparisons = compare_against_reference(
            scores,
            reference=PDDR,
            higher_is_better=higher_is_better,
        )
        _add_comparison_details(
            comparisons,
            scores,
            higher_is_better=higher_is_better,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1000 * metric_index,
        )
        main_results[metric] = {
            "higher_is_better": higher_is_better,
            "friedman": friedman_test(matrix),
            "mean_ranks": {
                algorithm: float(ranks[:, index].mean())
                for index, algorithm in enumerate(MAIN_ALGORITHMS)
            },
            "comparisons_from_pddr": comparisons,
        }
        ablation_scores = score_map(metric, (PDDR, ALWAYS))
        ablation_results[metric] = _pairwise_ablation(
            ablation_scores[PDDR],
            ablation_scores[ALWAYS],
            higher_is_better=higher_is_better,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 5000 + metric_index,
        )

    resource_results = {}
    for metric_index, (metric, higher_is_better) in enumerate(RESOURCE_METRICS.items()):
        scores = score_map(metric, MAIN_ALGORITHMS)
        comparisons = compare_against_reference(
            scores,
            reference=PDDR,
            higher_is_better=higher_is_better,
        )
        _add_comparison_details(
            comparisons,
            scores,
            higher_is_better=higher_is_better,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 10_000 + 1000 * metric_index,
        )
        resource_results[metric] = {
            "higher_is_better": higher_is_better,
            "comparisons_from_pddr": comparisons,
        }

    def comparison(metric: str, baseline: str) -> dict:
        return next(
            row
            for row in main_results[metric]["comparisons_from_pddr"]
            if row["algorithm"] == baseline
        )

    required_nonloss = math.ceil(0.75 * len(datasets))
    moead_igd = comparison("igd_plus", MOEAD)
    moead_hv = comparison("hypervolume", MOEAD)
    directional = all(
        item["win"] + item["tie"] >= required_nonloss
        for item in (moead_igd, moead_hv)
    )
    significant_positive = any(
        item["reject_holm"] and item["mean_oriented_improvement"] > 0.0
        for item in (moead_igd, moead_hv)
    )
    evidence_status = (
        "SUPPORTED"
        if directional and significant_positive
        else "DIRECTIONAL"
        if directional
        else "MIXED"
    )

    return {
        "protocol": {
            "purpose": "general multi-objective baseline confirmation with frozen MOEA/D",
            "datasets": list(datasets),
            "data_seed": data_seed,
            "optimization_seeds": list(optimization_seeds),
            "n_splits": n_splits,
            "algorithms": list(ALL_ALGORITHMS),
            "main_comparison_algorithms": list(MAIN_ALGORITHMS),
            "ablation_algorithm": ALWAYS,
            "evaluation_count": expected_evaluations,
            "inference_unit": "dataset mean over outer folds and optimization seeds",
            "multiple_comparisons": (
                "Holm correction across PDDR-DOLS vs NSGA-II and MOEA/D "
                "within each main metric"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "integrity": {
            "datasets": len(datasets),
            "instances": len(expected_instances),
            "metric_records": sum(len(rows) for rows in metric_groups.values()),
            "raw_algorithm_records": len(algorithm_rows),
            "exact_records": len(exact_rows),
            "records_per_dataset_algorithm": n_splits * len(optimization_seeds),
            "expected_total_raw_records": (
                expected_algorithm_records + expected_exact_records
            ),
        },
        "main_front_metrics": main_results,
        "representative_resource_metrics": resource_results,
        "mechanism_ablation": ablation_results,
        "dataset_means": dataset_rows,
        "moead_evidence": {
            "status": evidence_status,
            "required_nonloss_datasets": required_nonloss,
            "igd_plus_nonloss_datasets": moead_igd["win"] + moead_igd["tie"],
            "hypervolume_nonloss_datasets": moead_hv["win"] + moead_hv["tie"],
            "at_least_one_significant_positive_front_metric": significant_positive,
            "interpretation": (
                "PDDR-DOLS has paper-level support against the decomposition baseline."
                if evidence_status == "SUPPORTED"
                else "PDDR-DOLS has directional but not confirmatory support against MOEA/D."
                if evidence_status == "DIRECTIONAL"
                else "The PDDR-DOLS versus MOEA/D evidence is mixed and requires claim revision or method improvement."
            ),
        },
    }


def _text(report: dict) -> str:
    lines = [
        "PDDR-DOLS general-baseline confirmation",
        "",
        f"integrity: {report['integrity']}",
        f"MOEA/D evidence: {report['moead_evidence']}",
        "",
        "Main front metrics (dataset means are independent paired units):",
    ]
    for metric, item in report["main_front_metrics"].items():
        lines.append(
            f"  {metric}: Friedman={item['friedman']}; ranks={item['mean_ranks']}"
        )
        for row in item["comparisons_from_pddr"]:
            lines.append(
                "    PDDR-DOLS vs {algorithm_display_name}: "
                "W/T/L={win}/{tie}/{loss}, p={pvalue:.6g}, "
                "Holm={holm_pvalue:.6g}, reject={reject_holm}, "
                "r_rb={rank_biserial:+.4f}, improvement={mean_oriented_improvement:+.6g}, "
                "CI95={mean_raw_delta_ci95}".format(**row)
            )
    lines.extend(["", "Representative-resource metrics:"])
    for metric, item in report["representative_resource_metrics"].items():
        lines.append(f"  {metric}:")
        for row in item["comparisons_from_pddr"]:
            reduction = row.get("mean_relative_reduction_percent")
            reduction_text = (
                f", relative reduction={reduction:+.3f}%"
                if reduction is not None
                else ""
            )
            lines.append(
                "    PDDR-DOLS vs {algorithm_display_name}: "
                "W/T/L={win}/{tie}/{loss}, Holm={holm_pvalue:.6g}, "
                "r_rb={rank_biserial:+.4f}{reduction}".format(
                    reduction=reduction_text,
                    **row,
                )
            )
    lines.extend(["", "Mechanism ablation (separate hypothesis family):"])
    for metric, row in report["mechanism_ablation"].items():
        lines.append(
            f"  {metric}: PDDR-DOLS vs Always delete-only "
            f"W/T/L={row['win']}/{row['tie']}/{row['loss']}, "
            f"p={row['pvalue']:.6g}, r_rb={row['rank_biserial']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def _write_dataset_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--raw-prefix", default="joint_moead")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument(
        "--optimization-seeds",
        default="20260820,20260821,20260822,20260823,20260824",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--expected-evaluations", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    args = parser.parse_args()

    raw_records: list[dict] = []
    for path in sorted(args.raw_dir.glob(f"{args.raw_prefix}_*_seed_*.json")):
        payload = _load(path)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        raw_records.extend(payload)
    report = analyze_moead_baseline(
        _load(args.metrics),
        raw_records,
        datasets=_csv_values(args.datasets),
        data_seed=args.data_seed,
        optimization_seeds=tuple(
            int(value) for value in _csv_values(args.optimization_seeds)
        ),
        n_splits=args.n_splits,
        expected_evaluations=args.expected_evaluations,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    _write_dataset_csv(args.dataset_csv, report["dataset_means"])
    print(f"MOEAD_EVIDENCE_{report['moead_evidence']['status']}")
    print(f"datasets = {report['integrity']['datasets']}")
    print(f"raw_records = {report['integrity']['expected_total_raw_records']}")
    print(f"output = {args.output}")
    print(f"dataset_csv = {args.dataset_csv}")


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_ALGORITHMS",
    "MAIN_ALGORITHMS",
    "analyze_moead_baseline",
]
