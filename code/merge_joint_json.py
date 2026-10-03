"""Combine dataset-wise joint experiments, rejecting gaps and duplicate records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def merge(directory: Path, datasets: tuple[str, ...]) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str, int, int, int, str]] = set()
    expected_counts = {
        "exact_pep_native": 5,
        "exact_mdep_native": 5,
        "pep_paper": 25,
        "mdep_paper": 25,
        "nsga2": 25,
        "pddr_local_search_delete_only": 25,
    }
    for dataset in datasets:
        path = directory / f"joint_{dataset}_seed_20260820.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or Counter(row["algorithm"] for row in rows) != expected_counts:
            raise ValueError(f"{path}: incomplete expected algorithm/exact coverage")
        contexts: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            if row["dataset"] != dataset or int(row["seed"]) != 20260820:
                raise ValueError(f"{path}: mismatched dataset {row['dataset']}")
            fold = int(row["fold"])
            if fold not in range(5):
                raise ValueError(f"{path}: invalid fold {fold}")
            contexts[fold].add(str(row["objective_context_id"]))
            seed = row.get("optimization_seed")
            if row["algorithm"].startswith("exact_"):
                if seed is not None:
                    raise ValueError(f"{path}: exact front has optimizer seed")
            else:
                if seed not in range(20260820, 20260825):
                    raise ValueError(f"{path}: unexpected optimizer seed {seed}")
                expected_evaluations = 1976 if row["algorithm"] == "pep_paper" else 2000
                if int(row["evaluation_count"]) != expected_evaluations:
                    raise ValueError(f"{path}: unequal evaluation count")
                if row["algorithm"] == "mdep_paper" and row["algorithm_parameters"].get(
                    "implementation_revision"
                ) != "mdep_singleton_initialization_fix_v1":
                    raise ValueError(f"{path}: MDEP initialization revision mismatch")
            key = (dataset, int(row["seed"]), fold,
                   -1 if seed is None else int(seed), str(row["algorithm"]))
            if key in seen:
                raise ValueError(f"duplicate record: {key}")
            seen.add(key)
        if set(contexts) != set(range(5)) or any(len(value) != 1 for value in contexts.values()):
            raise ValueError(f"{path}: incomplete or mismatched objective contexts")
        records.extend(rows)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if not datasets or len(set(datasets)) != len(datasets):
        parser.error("datasets must be nonempty and unique")
    records = merge(args.raw_dir, datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"records = {len(records)}")
    print(f"output = {args.output}")


if __name__ == "__main__":
    main()
