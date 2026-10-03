"""Audit the P0 experiment protocol and write a reproducibility lock.

P0 validates protocol integrity only. It does not compare algorithm quality.
The lock records the objective context and cost vector used by every algorithm
on each outer fold, so later reports can prove that comparisons were made in a
shared cost/prediction space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("comma-separated value must not be empty")
    return items


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _context_digest(context: dict) -> str:
    encoded = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_p0_protocol(
    records: Iterable[dict],
    *,
    datasets: tuple[str, ...],
    data_seed: int,
    optimization_seeds: tuple[int, ...],
    algorithms: tuple[str, ...],
    n_splits: int,
    pop_size: int,
    n_gen: int,
    include_exact: bool = True,
    stage: str = "P0",
    purpose: str = "freeze and audit shared objective contexts",
) -> dict:
    """Validate record uniqueness, shared contexts, budgets and exact fronts."""
    rows = list(records)
    if not rows:
        raise ValueError("no records supplied")
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be unique")
    if len(set(optimization_seeds)) != len(optimization_seeds):
        raise ValueError("optimization_seeds must be unique")
    if not algorithms:
        raise ValueError("algorithms must not be empty")

    expected_algorithm_set = set(algorithms)
    expected_keys = {
        (dataset, data_seed, fold, opt_seed, algorithm)
        for dataset in datasets
        for fold in range(n_splits)
        for opt_seed in optimization_seeds
        for algorithm in algorithms
    }
    seen_keys: set[tuple[str, int, int, int, str]] = set()
    lock_folds: list[dict] = []
    algorithm_rows = [row for row in rows if row.get("algorithm") != "exact_pareto"]
    exact_rows = [row for row in rows if row.get("algorithm") == "exact_pareto"]

    for row in algorithm_rows:
        try:
            key = (
                str(row["dataset"]),
                int(row["seed"]),
                int(row["fold"]),
                int(row["optimization_seed"]),
                str(row["algorithm"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("algorithm row is missing a valid protocol key") from exc
        if key in seen_keys:
            raise ValueError(f"duplicate algorithm record: {key}")
        seen_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected algorithm record: {key}")
        if int(row.get("evaluation_count", -1)) != pop_size * n_gen:
            raise ValueError(
                f"evaluation budget mismatch for {key}: "
                f"{row.get('evaluation_count')} != {pop_size * n_gen}"
            )
        parameters = row.get("algorithm_parameters", {})
        if int(parameters.get("pop_size", -1)) != pop_size:
            raise ValueError(f"pop_size mismatch for {key}")
        if int(parameters.get("n_gen", -1)) != n_gen:
            raise ValueError(f"n_gen mismatch for {key}")

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        extra = sorted(seen_keys - expected_keys)
        raise ValueError(f"algorithm coverage mismatch; missing={missing}, extra={extra}")

    expected_exact_keys = {
        (dataset, data_seed, fold) for dataset in datasets for fold in range(n_splits)
    }
    exact_keys: set[tuple[str, int, int]] = set()
    for row in exact_rows:
        key = (str(row["dataset"]), int(row["seed"]), int(row["fold"]))
        if key in exact_keys:
            raise ValueError(f"duplicate exact Pareto record: {key}")
        exact_keys.add(key)
        if key not in expected_exact_keys:
            raise ValueError(f"unexpected exact Pareto record: {key}")
        front = np.asarray(row.get("raw_front"), dtype=float)
        masks = np.asarray(row.get("pareto_masks"))
        if front.ndim != 2 or front.shape[1] != 3 or len(front) == 0:
            raise ValueError(f"invalid exact front for {key}")
        if masks.ndim != 2 or len(masks) != len(front):
            raise ValueError(f"exact masks/front mismatch for {key}")

    if include_exact and exact_keys != expected_exact_keys:
        raise ValueError(
            f"exact Pareto coverage mismatch; missing={sorted(expected_exact_keys - exact_keys)}"
        )
    if not include_exact and exact_rows:
        raise ValueError("exact Pareto rows present although include_exact=False")

    for dataset in datasets:
        for fold in range(n_splits):
            fold_rows = [
                row
                for row in algorithm_rows
                if str(row["dataset"]) == dataset
                and int(row["seed"]) == data_seed
                and int(row["fold"]) == fold
            ]
            contexts = {row.get("objective_context_id") for row in fold_rows}
            if len(contexts) != 1 or None in contexts:
                raise ValueError(f"objective context mismatch for {(dataset, fold)}")
            context_values = fold_rows[0].get("objective_context", {})
            if _context_digest(context_values) != str(next(iter(contexts))):
                raise ValueError(f"objective context digest mismatch for {(dataset, fold)}")
            costs = np.asarray(context_values.get("combined_costs"), dtype=float)
            if costs.ndim != 1 or len(costs) < 2 or not np.isfinite(costs).all():
                raise ValueError(f"invalid combined costs for {(dataset, fold)}")
            if np.any(costs < 0) or not np.isclose(float(costs.sum()), 1.0, atol=1e-8):
                raise ValueError(f"combined costs are not normalized for {(dataset, fold)}")
            names = context_values.get("classifier_names", [])
            if len(names) != len(set(names)) or len(names) != len(costs):
                raise ValueError(f"classifier/cost alignment mismatch for {(dataset, fold)}")
            if include_exact:
                exact = next(
                    row
                    for row in exact_rows
                    if str(row["dataset"]) == dataset
                    and int(row["seed"]) == data_seed
                    and int(row["fold"]) == fold
                )
                if exact.get("objective_context_id") != next(iter(contexts)):
                    raise ValueError(f"exact context mismatch for {(dataset, fold)}")
            lock_folds.append(
                {
                    "dataset": dataset,
                    "data_seed": data_seed,
                    "fold": fold,
                    "objective_context_id": next(iter(contexts)),
                    "classifier_names": names,
                    "combined_costs": costs.tolist(),
                    "inference_seconds": context_values.get("inference_seconds"),
                    "storage_bytes": context_values.get("storage_bytes"),
                }
            )

    return {
        "protocol": {
            "status": f"{stage}_PASS",
            "purpose": purpose,
            "data_seed": data_seed,
            "optimization_seeds": list(optimization_seeds),
            "datasets": list(datasets),
            "algorithms": list(algorithms),
            "n_splits": n_splits,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "evaluation_count_per_algorithm_fold": pop_size * n_gen,
            "include_exact": include_exact,
        },
        "fold_contexts": lock_folds,
        "counts": {
            "raw_records": len(rows),
            "algorithm_records": len(algorithm_rows),
            "exact_records": len(exact_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--optimization-seeds", required=True)
    parser.add_argument("--algorithms", required=True)
    parser.add_argument("--n-splits", type=int, required=True)
    parser.add_argument("--pop-size", type=int, required=True)
    parser.add_argument("--n-gen", type=int, required=True)
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--stage", default="P0")
    parser.add_argument("--purpose", default="freeze and audit shared objective contexts")
    args = parser.parse_args()
    lock = validate_p0_protocol(
        _load(args.input),
        datasets=_csv(args.datasets),
        data_seed=args.data_seed,
        optimization_seeds=tuple(int(value) for value in _csv(args.optimization_seeds)),
        algorithms=_csv(args.algorithms),
        n_splits=args.n_splits,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        include_exact=not args.no_exact,
        stage=args.stage,
        purpose=args.purpose,
    )
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_output.write_text(json.dumps(lock, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(lock["protocol"]["status"])
    print(f"raw_records = {lock['counts']['raw_records']}")
    print(f"fold_contexts = {len(lock['fold_contexts'])}")
    print(f"lock_output = {args.lock_output}")


if __name__ == "__main__":
    main()


__all__ = ["validate_p0_protocol"]
