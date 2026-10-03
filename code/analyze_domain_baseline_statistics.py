"""Formal statistics for PEP/MDEP joint experiments.

Two analysis layers are deliberately kept separate:

* Common downstream outcomes (test error, selected count and runtime) are
  paired across ``pep_paper``, ``mdep_paper``, ``nsga2`` and
  ``pddr_local_search_delete_only`` using the same dataset/seed/fold/
  optimization-seed key.
* Native Pareto metrics are summarized independently for PEP and MDEP.  PEP
  uses its two-objective exact front and MDEP uses its three-objective exact
  front; these spaces are never pooled into one statistical ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from statistics_analysis import (
    friedman_test,
    holm_correction,
    paired_wilcoxon,
    win_tie_loss,
)


COMMON_ALGORITHMS = (
    "pep_paper",
    "mdep_paper",
    "nsga2",
    "pddr_local_search_delete_only",
)
COMMON_METRICS = {
    "test_error": False,
    "selected_count": False,
    "algorithm_runtime_seconds": False,
}
NATIVE_METRICS = (
    "hypervolume",
    "igd_plus",
    "hypervolume_gap",
    "exact_point_coverage",
)


def _load_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _load_dict_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path} must contain an object with a records list")
    return payload["records"]


def _key(row: dict) -> tuple[str, int, int, int]:
    value = row.get("optimization_seed")
    if value is None:
        raise ValueError(f"algorithm row has null optimization_seed: {row.get('algorithm')}")
    return str(row["dataset"]), int(row["seed"]), int(row["fold"]), int(value)


def _rank_biserial(first: np.ndarray, second: np.ndarray, *, higher_is_better: bool) -> float:
    direction = 1.0 if higher_is_better else -1.0
    differences = (first - second) * direction
    nonzero = differences[np.abs(differences) > 1e-12]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def _bootstrap_ci(values: np.ndarray, *, seed: int, samples: int) -> list[float]:
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _stable_seed(base: int, *parts: str) -> int:
    """Derive a reproducible bootstrap seed without Python's randomized hash."""
    payload = "|".join((str(base),) + tuple(parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _compare_candidates_to_reference(
    scores: dict[str, np.ndarray],
    *,
    reference: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
    seed_parts: tuple[str, ...],
) -> list[dict]:
    """Return candidate-oriented paired comparisons against one reference."""
    reference_values = np.asarray(scores[reference], dtype=float)
    comparisons = []
    direction = 1.0 if higher_is_better else -1.0
    for algorithm, values in scores.items():
        if algorithm == reference:
            continue
        candidate_values = np.asarray(values, dtype=float)
        raw_delta = candidate_values - reference_values
        improvement = raw_delta * direction
        comparison = {
            "algorithm": algorithm,
            "reference": reference,
            **paired_wilcoxon(candidate_values, reference_values),
            **win_tie_loss(
                candidate_values,
                reference_values,
                higher_is_better=higher_is_better,
            ),
            "comparison_direction": "candidate_vs_reference",
            "win_definition": "candidate_better_than_reference",
            "mean_delta_candidate_minus_reference": float(np.mean(raw_delta)),
            "mean_delta_reference_minus_candidate": float(np.mean(-raw_delta)),
            "mean_improvement": float(np.mean(improvement)),
            "mean_improvement_ci95": _bootstrap_ci(
                improvement,
                seed=_stable_seed(
                    bootstrap_seed, *seed_parts, algorithm, reference
                ),
                samples=bootstrap_samples,
            ),
            "rank_biserial": _rank_biserial(
                candidate_values,
                reference_values,
                higher_is_better=higher_is_better,
            ),
        }
        comparisons.append(comparison)
    correction = holm_correction(
        [row["pvalue"] for row in comparisons], alpha=0.05
    )
    for row, corrected, rejected in zip(
        comparisons, correction["holm_pvalues"], correction["reject"]
    ):
        row["holm_pvalue"] = float(corrected)
        row["reject_holm"] = bool(rejected)
    return comparisons


def _common_statistics(
    records: Iterable[dict], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    grouped: dict[tuple[str, int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in records:
        algorithm = str(row.get("algorithm"))
        if algorithm not in COMMON_ALGORITHMS:
            continue
        key = _key(row)
        if algorithm in grouped[key]:
            raise ValueError(f"duplicate common record for {key} / {algorithm}")
        grouped[key][algorithm] = row
    if not grouped:
        raise ValueError("no common PEP/MDEP/NSGA-II/PDDR records found")
    expected = set(COMMON_ALGORITHMS)
    incomplete = [(key, sorted(rows)) for key, rows in grouped.items() if set(rows) != expected]
    if incomplete:
        raise ValueError(f"incomplete common pairing, first={incomplete[0]}")

    datasets = sorted({key[0] for key in grouped})
    dataset_reports = []
    for dataset in datasets:
        keys = sorted(key for key in grouped if key[0] == dataset)
        report = {
            "dataset": dataset,
            "n_instances": len(keys),
            "n_outer_folds": len({key[2] for key in keys}),
            "n_optimization_seeds": len({key[3] for key in keys}),
            "means": {},
            "friedman": {},
            "comparisons_to_nsga2": {},
        }
        for metric, higher_is_better in COMMON_METRICS.items():
            scores = {
                algorithm: np.asarray(
                    [float(grouped[key][algorithm][metric]) for key in keys], dtype=float
                )
                for algorithm in COMMON_ALGORITHMS
            }
            if not all(np.isfinite(values).all() for values in scores.values()):
                raise ValueError(f"non-finite common metric {metric} in {dataset}")
            report["means"][metric] = {
                algorithm: float(np.mean(values)) for algorithm, values in scores.items()
            }
            report["friedman"][metric] = friedman_test(
                np.column_stack([scores[algorithm] for algorithm in COMMON_ALGORITHMS])
            )
            comparisons = _compare_candidates_to_reference(
                scores,
                reference="nsga2",
                higher_is_better=higher_is_better,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                seed_parts=("within_dataset", dataset, metric),
            )
            report["comparisons_to_nsga2"][metric] = comparisons
        dataset_reports.append(report)

    # The primary inference unit is one mean per dataset.  Outer folds and
    # optimizer seeds are repeated measurements nested within that dataset.
    cross_dataset = {}
    for metric, higher_is_better in COMMON_METRICS.items():
        scores = {
            algorithm: np.asarray(
                [report["means"][metric][algorithm] for report in dataset_reports],
                dtype=float,
            )
            for algorithm in COMMON_ALGORITHMS
        }
        matrix = np.column_stack(
            [scores[algorithm] for algorithm in COMMON_ALGORITHMS]
        )
        ranks = np.vstack(
            [
                rankdata(-row if higher_is_better else row, method="average")
                for row in matrix
            ]
        )
        cross_dataset[metric] = {
            "n_datasets": len(datasets),
            "dataset_order": datasets,
            "means": {
                algorithm: float(np.mean(values))
                for algorithm, values in scores.items()
            },
            "dataset_means": {
                algorithm: values.tolist() for algorithm, values in scores.items()
            },
            "friedman": friedman_test(matrix),
            "mean_ranks": {
                algorithm: float(ranks[:, index].mean())
                for index, algorithm in enumerate(COMMON_ALGORITHMS)
            },
            "comparisons_to_nsga2": _compare_candidates_to_reference(
                scores,
                reference="nsga2",
                higher_is_better=higher_is_better,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                seed_parts=("cross_dataset", metric),
            ),
        }

    pooled = {}
    keys = sorted(grouped)
    for metric in COMMON_METRICS:
        scores = {
            algorithm: np.asarray(
                [float(grouped[key][algorithm][metric]) for key in keys], dtype=float
            )
            for algorithm in COMMON_ALGORITHMS
        }
        pooled[metric] = {
            "n_instances": len(keys),
            "means": {algorithm: float(np.mean(values)) for algorithm, values in scores.items()},
        }
    return {
        "protocol": {
            "primary_inference_unit": "dataset mean over outer folds and optimization seeds",
            "n_primary_units": len(datasets),
            "repeated_run_unit": "dataset x outer-fold x optimization-seed",
            "algorithms": list(COMMON_ALGORITHMS),
            "reference": "nsga2",
            "primary_metric": "test_error",
            "secondary_metrics": ["selected_count", "algorithm_runtime_seconds"],
            "primary_tests": "Friedman omnibus across dataset means; paired two-sided Wilcoxon candidate versus NSGA-II; Holm correction across three candidate comparisons within each metric",
            "within_dataset_view": "secondary repeated-run diagnostic; optimizer seeds are nested within outer folds",
            "pooled_view": "descriptive support only; no pooled inferential p-values",
            "comparison_direction": "W/T/L and positive improvement are always candidate relative to NSGA-II",
        },
        "cross_dataset": cross_dataset,
        "datasets": dataset_reports,
        "pooled_descriptive": pooled,
    }


def _native_statistics(records: Iterable[dict], *, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        space = str(row.get("native_objective_space"))
        algorithm = str(row.get("algorithm"))
        if space not in {"pep_native", "mdep_native"}:
            raise ValueError(f"unexpected native objective space: {space}")
        grouped[(space, algorithm)].append(row)
    expected = {("pep_native", "pep_paper"), ("mdep_native", "mdep_paper")}
    if set(grouped) != expected:
        raise ValueError(f"native groups mismatch: {sorted(grouped)}")

    reports = []
    for (space, algorithm), values in sorted(grouped.items()):
        datasets = sorted({str(row["dataset"]) for row in values})
        counts_by_dataset = {
            dataset: sum(str(row["dataset"]) == dataset for row in values)
            for dataset in datasets
        }
        if not counts_by_dataset or min(counts_by_dataset.values()) < 2:
            raise ValueError(f"native groups need at least two paired instances: {space}/{algorithm}")
        if len(set(counts_by_dataset.values())) != 1:
            raise ValueError(f"inconsistent native records per dataset: {space}/{counts_by_dataset}")
        dataset_reports = []
        for dataset in datasets:
            rows = [row for row in values if str(row["dataset"]) == dataset]
            summary = {
                "dataset": dataset,
                "n_instances": len(rows),
                "native_objective_space": space,
                "algorithm": algorithm,
                "means": {},
                "bootstrap_ci95": {},
            }
            for offset, metric in enumerate(NATIVE_METRICS):
                array = np.asarray([float(row[metric]) for row in rows], dtype=float)
                if not np.isfinite(array).all():
                    raise ValueError(f"non-finite native metric {metric} for {space}/{dataset}")
                summary["means"][metric] = float(np.mean(array))
                summary["bootstrap_ci95"][metric] = _bootstrap_ci(
                    array,
                    seed=_stable_seed(bootstrap_seed + offset, space, dataset, metric),
                    samples=bootstrap_samples,
                )
            summary["test_error_mean"] = float(np.mean([float(row["test_error"]) for row in rows]))
            summary["selected_count_mean"] = float(np.mean([float(row["selected_count"]) for row in rows]))
            dataset_reports.append(summary)
        reports.append(
            {
                "native_objective_space": space,
                "algorithm": algorithm,
                "n_records": len(values),
                "datasets": dataset_reports,
            }
        )
    return {
        "protocol": {
            "unit": "dataset x outer-fold x optimization-seed",
            "pep_objectives": ["validation_error", "selected_count_fraction"],
            "mdep_objectives": ["validation_error", "margin_ratio", "selected_count_fraction"],
            "reference": "exhaustive exact native Pareto front for the same outer fold",
            "cross_space_ranking": False,
            "inference_note": "native metrics are effect summaries relative to an exact reference; no PEP-vs-MDEP significance test is performed across different objective spaces",
        },
        "groups": reports,
    }


def _text_report(report: dict) -> str:
    lines = [
        "Formal PEP/MDEP joint statistics",
        "",
        "[Primary cross-dataset inference: dataset means are paired units]",
    ]
    for metric, item in report["common"]["cross_dataset"].items():
        fr = item["friedman"]
        lines.append(
            f"  {metric} (n={item['n_datasets']} datasets): "
            f"Friedman p={fr['pvalue']:.6g}; "
            f"means={item['means']}; ranks={item['mean_ranks']}"
        )
        for row in item["comparisons_to_nsga2"]:
            ci = row["mean_improvement_ci95"]
            lines.append(
                f"    {row['algorithm']} vs nsga2 W/T/L="
                f"{row['win']}/{row['tie']}/{row['loss']} "
                f"p={row['pvalue']:.6g} Holm={row['holm_pvalue']:.6g} "
                f"reject={row['reject_holm']} improvement={row['mean_improvement']:+.6g} "
                f"CI95=[{ci[0]:+.6g}, {ci[1]:+.6g}] "
                f"r_rb={row['rank_biserial']:+.4f}"
            )
    lines.extend(
        [
            "",
            "W/T/L definition: candidate algorithm relative to NSGA-II.",
            "Positive improvement means the candidate is better.",
            "",
            "[Within-dataset repeated-run diagnostics: secondary]",
        ]
    )
    for dataset in report["common"]["datasets"]:
        lines.append(
            f"{dataset['dataset']} (n={dataset['n_instances']}):"
        )
        for metric in COMMON_METRICS:
            fr = dataset["friedman"][metric]
            lines.append(
                f"  {metric}: Friedman p={fr['pvalue']:.6g}; means={dataset['means'][metric]}"
            )
            for row in dataset["comparisons_to_nsga2"][metric]:
                lines.append(
                    f"    {row['algorithm']} W/T/L={row['win']}/{row['tie']}/{row['loss']} "
                    f"p={row['pvalue']:.6g} Holm={row.get('holm_pvalue', float('nan')):.6g} "
                    f"improvement={row['mean_improvement']:+.6g} "
                    f"r_rb={row['rank_biserial']:+.4f}"
                )
    lines.append("\n[Pooled common metrics: descriptive only]")
    for metric, item in report["common"]["pooled_descriptive"].items():
        lines.append(f"  {metric}: n={item['n_instances']} means={item['means']}")
    lines.append("\n[Native objective-space summaries]")
    for group in report["native"]["groups"]:
        lines.append(f"{group['native_objective_space']} / {group['algorithm']}:")
        for row in group["datasets"]:
            lines.append(
                f"  {row['dataset']} (n={row['n_instances']}): "
                f"HV={row['means']['hypervolume']:.6f}, "
                f"IGD+={row['means']['igd_plus']:.6f}, "
                f"gap={row['means']['hypervolume_gap']:.6f}, "
                f"coverage={row['means']['exact_point_coverage']:.4f}"
            )
    lines.append("\nBoundary: PEP native two-objective and MDEP native three-objective metrics are reported separately.")
    return "\n".join(lines) + "\n"


def _cross_dataset_text_report(report: dict) -> str:
    common = report["common"]
    lines = [
        "PEP/MDEP cross-dataset statistics",
        "",
        f"Inference unit: {common['protocol']['primary_inference_unit']}",
        f"Independent units: n={common['protocol']['n_primary_units']} datasets",
        f"Tests: {common['protocol']['primary_tests']}",
        "W/T/L definition: candidate algorithm relative to NSGA-II.",
        "Positive improvement means the candidate is better.",
        "",
    ]
    for metric, item in common["cross_dataset"].items():
        lines.append(
            f"{metric}: Friedman={item['friedman']}; means={item['means']}; "
            f"ranks={item['mean_ranks']}"
        )
        for row in item["comparisons_to_nsga2"]:
            lines.append(
                "  {algorithm} vs nsga2: W/T/L={win}/{tie}/{loss}, "
                "p={pvalue:.6g}, Holm={holm_pvalue:.6g}, "
                "reject={reject_holm}, improvement={mean_improvement:+.6g}, "
                "CI95={mean_improvement_ci95}, r_rb={rank_biserial:+.4f}".format(
                    **row
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def analyze_domain_baseline_statistics(
    joint_records: Iterable[dict], native_records: Iterable[dict], *,
    bootstrap_samples: int = 10_000, bootstrap_seed: int = 20260902,
) -> dict:
    common = _common_statistics(
        joint_records, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
    )
    native = _native_statistics(
        native_records, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed + 1_000_000
    )
    return {
        "protocol": {
            "scope": "formal PEP/MDEP joint statistics with native-space separation",
            "native_space_separation": True,
        },
        "common": common,
        "native": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-input", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--cross-dataset-output", type=Path)
    parser.add_argument("--cross-dataset-text-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if bool(args.cross_dataset_output) != bool(args.cross_dataset_text_output):
        parser.error(
            "--cross-dataset-output and --cross-dataset-text-output must be used together"
        )
    report = analyze_domain_baseline_statistics(
        _load_list(args.joint_input),
        _load_dict_records(args.native_input),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    args.text_output.write_text(_text_report(report), encoding="utf-8")
    if args.cross_dataset_output is not None:
        cross_dataset_report = {
            "protocol": report["common"]["protocol"],
            "cross_dataset": report["common"]["cross_dataset"],
        }
        args.cross_dataset_output.parent.mkdir(parents=True, exist_ok=True)
        args.cross_dataset_text_output.parent.mkdir(parents=True, exist_ok=True)
        args.cross_dataset_output.write_text(
            json.dumps(cross_dataset_report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        args.cross_dataset_text_output.write_text(
            _cross_dataset_text_report(report), encoding="utf-8"
        )
    print(f"common_datasets = {len(report['common']['datasets'])}")
    print(f"common_pooled_instances = {report['common']['pooled_descriptive']['test_error']['n_instances']}")
    print(f"native_groups = {len(report['native']['groups'])}")
    print(f"cross_dataset_units = {report['common']['protocol']['n_primary_units']}")
    print(f"output = {args.output}")
    print(f"text_output = {args.text_output}")


if __name__ == "__main__":
    main()


__all__ = ["analyze_domain_baseline_statistics"]
