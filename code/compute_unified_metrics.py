"""Compute per-instance HV and IGD+ metrics from batch JSON results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from metrics import (
    build_normalization_context,
    merge_reference_front,
    nondominated_front,
    normalize_objectives,
)


GroupKey = tuple[str, int, int, int]
OuterKey = tuple[str, int, int]


def _load_json_records(paths: Iterable[Path]) -> list[dict]:
    """Load JSON lists and preserve their source-independent record format."""
    records: list[dict] = []
    for path in sorted(paths):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        records.extend(payload)
    return records


def _merge_record_sources(*sources: Iterable[dict]) -> list[dict]:
    """Combine sources while rejecting duplicate algorithm instances."""
    records: list[dict] = []
    seen: set[tuple[str, int, int, int, str]] = set()
    for source in sources:
        for record in source:
            optimization_seed = record.get("optimization_seed")
            if optimization_seed is None:
                optimization_seed = -1
            key = (
                str(record.get("dataset")),
                int(record.get("seed")),
                int(record.get("fold")),
                int(optimization_seed),
                str(record.get("algorithm")),
            )
            if key in seen:
                raise ValueError(f"duplicate result record for {key}")
            seen.add(key)
            records.append(record)
    return records


def _group_key(record: dict) -> GroupKey:
    """Return the dataset/seed/fold/optimizer key for algorithm records."""
    try:
        optimization_seed = record.get("optimization_seed")
        if optimization_seed is None:
            optimization_seed = -1
        return (
            str(record["dataset"]),
            int(record["seed"]),
            int(record["fold"]),
            int(optimization_seed),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("record is missing a valid dataset, seed or fold") from exc


def _metric_rows(
    fronts: dict[str, np.ndarray],
    source_records: dict[str, dict],
    *,
    exact_front: np.ndarray | None,
) -> list[dict]:
    """Compute shared-normalization metrics for one instance and objective count."""
    objective_count = next(iter(fronts.values())).shape[1]
    if exact_front is not None:
        if objective_count != 3:
            raise ValueError("exact Pareto references are supported only for 3 objectives")
        reference_type = "exact_pareto"
        reference_front = nondominated_front(exact_front)
        context_fronts = {**fronts, "exact_pareto": exact_front}
    else:
        reference_type = "merged_algorithm_front"
        reference_front = merge_reference_front(fronts)
        context_fronts = fronts

    context = build_normalization_context(context_fronts)
    normalized_reference = normalize_objectives(reference_front, context)
    hypervolume = HV(ref_point=context.reference_point)
    igd_plus = IGDPlus(normalized_reference)
    rows: list[dict] = []
    for algorithm in sorted(fronts):
        clean_front = nondominated_front(fronts[algorithm])
        normalized_front = normalize_objectives(clean_front, context)
        source = source_records[algorithm]
        rows.append(
            {
                "dataset": source["dataset"],
                "seed": int(source["seed"]),
                "fold": int(source["fold"]),
                "optimization_seed": (
                    int(source["optimization_seed"])
                    if source.get("optimization_seed") is not None
                    else -1
                ),
                "algorithm": algorithm,
                "n_objectives": int(objective_count),
                "reference_type": reference_type,
                "reference_front_size": int(len(reference_front)),
                "front_size": int(len(clean_front)),
                "hypervolume": float(hypervolume.do(normalized_front)),
                "igd_plus": float(igd_plus.do(normalized_front)),
                "test_error": float(source["test_error"]),
                "test_margin_loss": float(source["test_margin_loss"]),
                "selected_count": int(source["selected_count"]),
                "normalization": {
                    "ideal": context.ideal.tolist(),
                    "nadir": context.nadir.tolist(),
                    "span": context.span.tolist(),
                    "reference_point": context.reference_point.tolist(),
                },
            }
        )
    return rows


def compute_unified_metrics(
    batch_records: Iterable[dict], exact_records: Iterable[dict] = ()
) -> dict:
    """Compute per-instance metrics without pooling across datasets or folds."""
    grouped: dict[GroupKey, dict[str, dict]] = defaultdict(dict)
    for record in batch_records:
        if record.get("algorithm") == "exact_pareto":
            continue
        key = _group_key(record)
        algorithm = str(record.get("algorithm", ""))
        if not algorithm:
            raise ValueError("batch record has an empty algorithm name")
        if algorithm in grouped[key]:
            raise ValueError(f"duplicate batch algorithm for {key}: {algorithm}")
        grouped[key][algorithm] = record

    exact_by_outer: dict[OuterKey, tuple[np.ndarray, str | None]] = {}
    for record in exact_records:
        if record.get("algorithm") != "exact_pareto":
            continue
        key = (
            str(record["dataset"]),
            int(record["seed"]),
            int(record["fold"]),
        )
        front = np.asarray(record.get("raw_front"), dtype=float)
        if front.ndim != 2 or front.shape[1] != 3 or front.shape[0] == 0:
            raise ValueError(f"invalid exact Pareto front for {key}")
        if key in exact_by_outer:
            raise ValueError(f"duplicate exact Pareto record for {key}")
        exact_by_outer[key] = (front, record.get("objective_context_id"))

    output_rows: list[dict] = []
    for key in sorted(grouped):
        records_by_algorithm = grouped[key]
        by_objective_count: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        source_by_objective_count: dict[int, dict[str, dict]] = defaultdict(dict)
        for algorithm, record in records_by_algorithm.items():
            objective_count = int(record["n_objectives"])
            front = np.asarray(record["raw_front"], dtype=float)
            if front.ndim != 2 or front.shape[1] != objective_count or front.shape[0] == 0:
                raise ValueError(f"invalid front for {key} algorithm={algorithm}")
            by_objective_count[objective_count][algorithm] = front
            source_by_objective_count[objective_count][algorithm] = record

        for objective_count in sorted(by_objective_count):
            exact_front = None
            outer_key = key[:3]
            if objective_count == 3 and outer_key in exact_by_outer:
                candidate_front, exact_context_id = exact_by_outer[outer_key]
                context_ids = {
                    source.get("objective_context_id")
                    for source in source_by_objective_count[objective_count].values()
                }
                if not exact_context_id or context_ids != {exact_context_id}:
                    raise ValueError(
                        "exact Pareto and batch records must share a non-empty "
                        f"objective_context_id for {key}; got exact="
                        f"{exact_context_id!r}, batch={sorted(context_ids, key=str)!r}"
                    )
                exact_front = candidate_front
            output_rows.extend(
                _metric_rows(
                    by_objective_count[objective_count],
                    source_by_objective_count[objective_count],
                    exact_front=exact_front,
                )
            )

    return {
        "protocol": {
            "scope": "per dataset, data seed, outer fold, optimization seed and algorithm",
            "exact_reference_policy": "use exact Pareto when supplied; otherwise merge same-dimension algorithm fronts",
            "pep_policy": "PEP remains a separate two-objective group",
            "cross_dataset_summary": False,
        },
        "records": output_rows,
    }


def main() -> None:
    """Load batch/exact files and write per-instance metric records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("results/raw_data")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw_data/unified_metrics.json"),
    )
    args = parser.parse_args()

    joint_records = _load_json_records(args.raw_dir.glob("joint_*_seed_*.json"))
    batch_records = _merge_record_sources(
        _load_json_records(args.raw_dir.glob("batch_*_seed_*.json")),
        joint_records,
    )
    exact_records = _merge_record_sources(
        _load_json_records(args.raw_dir.glob("exact_*.json")),
        joint_records,
    )
    result = compute_unified_metrics(batch_records, exact_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(f"records = {len(result['records'])}")
    print(f"output = {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["compute_unified_metrics"]
