"""Analyze PEP/MDEP in their native objective spaces against exact fronts.

The input is the joint JSON produced by ``run_native_exact_joint.py``.  PEP
is evaluated only in its native two-objective space
``(validation_error, selected_count)`` and MDEP only in its native
three-objective space ``(validation_error, margin_ratio, selected_count)``.
The two spaces are never pooled or ranked together.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from metrics import build_normalization_context, nondominated_front, normalize_objectives


PAIRING = {
    "exact_pep_native": ("pep_paper", "pep_native"),
    "exact_mdep_native": ("mdep_paper", "mdep_native"),
}


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _outer_key(row: dict) -> tuple[str, int, int]:
    return str(row["dataset"]), int(row["seed"]), int(row["fold"])


def _finite_front(row: dict, field: str = "raw_front") -> np.ndarray:
    front = np.asarray(row.get(field), dtype=float)
    if front.ndim != 2 or front.shape[0] == 0:
        raise ValueError(f"invalid {field} for {row.get('algorithm')}")
    if front.shape[1] != int(row["n_objectives"]):
        raise ValueError(f"objective dimension mismatch for {row.get('algorithm')}")
    if not np.isfinite(front).all():
        raise ValueError(f"non-finite {field} for {row.get('algorithm')}")
    return front


def _weakly_covers(reference_front: np.ndarray, candidate_front: np.ndarray, tolerance: float = 1e-12) -> float:
    """Return the fraction of reference points weakly covered by a candidate.

    All objectives are minimized.  A reference point is covered when at least
    one candidate point is no worse in every objective, within ``tolerance``.
    """
    covered = [
        np.any(np.all(candidate_front <= point + tolerance, axis=1))
        for point in reference_front
    ]
    return float(np.mean(covered))


def _analyze_pair(exact_row: dict, algorithm_rows: Iterable[dict], *, native_name: str) -> list[dict]:
    exact_front = nondominated_front(_finite_front(exact_row))
    objective_count = exact_front.shape[1]
    rows = list(algorithm_rows)
    if not rows:
        raise ValueError(f"no algorithm rows for {_outer_key(exact_row)} / {native_name}")
    context_id = exact_row.get("objective_context_id")
    if not context_id:
        raise ValueError(f"exact row lacks objective_context_id for {_outer_key(exact_row)}")
    for row in rows:
        if int(row["n_objectives"]) != objective_count:
            raise ValueError(
                f"{native_name} objective dimension mismatch: exact={objective_count}, "
                f"algorithm={row.get('n_objectives')}"
            )
        if row.get("objective_context_id") != context_id:
            raise ValueError(f"objective context mismatch for {_outer_key(exact_row)} / {native_name}")

    fronts = {"exact_native": exact_front}
    for row in rows:
        fronts[f"algorithm_{row['optimization_seed']}"] = nondominated_front(_finite_front(row))
    normalization = build_normalization_context(fronts)
    normalized_exact = normalize_objectives(exact_front, normalization)
    hv = HV(ref_point=normalization.reference_point)
    igd = IGDPlus(normalized_exact)
    exact_hv = float(hv.do(normalized_exact))

    output = []
    for row in rows:
        clean = nondominated_front(_finite_front(row))
        normalized = normalize_objectives(clean, normalization)
        value = float(hv.do(normalized))
        output.append(
            {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "optimization_seed": int(row["optimization_seed"]),
                "algorithm": row["algorithm"],
                "native_objective_space": native_name,
                "n_objectives": objective_count,
                "reference_type": "exact_native",
                "evaluated_subsets": int(exact_row["evaluated_subsets"]),
                "exact_front_size": int(len(exact_front)),
                "algorithm_front_size": int(len(clean)),
                "hypervolume": value,
                "exact_hypervolume": exact_hv,
                "hypervolume_gap": float(exact_hv - value),
                "igd_plus": float(igd.do(normalized)),
                "exact_point_coverage": _weakly_covers(exact_front, clean),
                "test_error": float(row["test_error"]),
                "selected_count": int(row["selected_count"]),
                "algorithm_runtime_seconds": float(row.get("algorithm_runtime_seconds", 0.0)),
                "objective_context_id": context_id,
            }
        )
    return output


def _summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["algorithm"]))].append(row)
    result = []
    for (dataset, algorithm), values in sorted(grouped.items()):
        def mean(name: str) -> float:
            return float(np.mean([float(v[name]) for v in values]))

        def median(name: str) -> float:
            return float(np.median([float(v[name]) for v in values]))

        result.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "native_objective_space": values[0]["native_objective_space"],
                "n": len(values),
                "hypervolume_mean": mean("hypervolume"),
                "hypervolume_median": median("hypervolume"),
                "exact_hypervolume_mean": mean("exact_hypervolume"),
                "hypervolume_gap_mean": mean("hypervolume_gap"),
                "igd_plus_mean": mean("igd_plus"),
                "igd_plus_median": median("igd_plus"),
                "exact_point_coverage_mean": mean("exact_point_coverage"),
                "algorithm_front_size_mean": mean("algorithm_front_size"),
                "exact_front_size_mean": mean("exact_front_size"),
                "test_error_mean": mean("test_error"),
                "selected_count_mean": mean("selected_count"),
                "runtime_seconds_mean": mean("algorithm_runtime_seconds"),
            }
        )
    return result


def analyze_native_exact(records: Iterable[dict]) -> dict:
    """Return native-space metric records and dataset-level summaries."""
    all_records = list(records)
    exact_by_key: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    algorithms_by_key: dict[tuple[str, int, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in all_records:
        algorithm = str(row.get("algorithm", ""))
        key = _outer_key(row)
        if algorithm in PAIRING:
            if row.get("optimization_seed") is not None:
                raise ValueError(f"exact row must have null optimization_seed: {key} / {algorithm}")
            if algorithm in exact_by_key[key]:
                raise ValueError(f"duplicate exact row for {key} / {algorithm}")
            exact_by_key[key][algorithm] = row
        elif algorithm in {pair[0] for pair in PAIRING.values()}:
            algorithms_by_key[key][algorithm].append(row)

    output_rows = []
    for key in sorted(exact_by_key):
        for exact_algorithm, (algorithm, native_name) in PAIRING.items():
            exact = exact_by_key[key].get(exact_algorithm)
            candidates = algorithms_by_key[key].get(algorithm, [])
            if exact is None or not candidates:
                raise ValueError(f"missing exact or algorithm rows for {key} / {native_name}")
            seen_seeds = set()
            for row in candidates:
                opt_seed = int(row["optimization_seed"])
                if opt_seed in seen_seeds:
                    raise ValueError(f"duplicate optimization seed for {key} / {algorithm}: {opt_seed}")
                seen_seeds.add(opt_seed)
            output_rows.extend(_analyze_pair(exact, candidates, native_name=native_name))

    return {
        "protocol": {
            "description": "PEP/MDEP native objective-space analysis against exhaustive non-empty-subset references",
            "pep_objectives": ["validation_error", "selected_count_fraction"],
            "mdep_objectives": ["validation_error", "margin_ratio", "selected_count_fraction"],
            "reference_policy": "separate exact native Pareto front for each method and outer fold",
            "coverage_definition": "fraction of exact points weakly dominated by at least one algorithm point",
            "normalization": "shared per outer fold between the algorithm front and its native exact front",
            "cross_space_ranking": False,
        },
        "integrity": {
            "native_metric_records": len(output_rows),
            "outer_instances": len({(r["dataset"], r["seed"], r["fold"]) for r in output_rows}),
            "algorithms": sorted({r["algorithm"] for r in output_rows}),
            "all_finite": bool(all(np.isfinite(float(r[name])) for r in output_rows for name in ("hypervolume", "igd_plus", "hypervolume_gap", "exact_point_coverage"))),
        },
        "records": output_rows,
        "summary": _summary(output_rows),
    }


def _text_report(report: dict) -> str:
    lines = ["PEP/MDEP native objective-space analysis", "", f"records: {report['integrity']['native_metric_records']}", ""]
    for row in report["summary"]:
        lines.append(
            f"{row['dataset']} | {row['algorithm']} | n={row['n']} | "
            f"HV={row['hypervolume_mean']:.6f} | IGD+={row['igd_plus_mean']:.6f} | "
            f"coverage={row['exact_point_coverage_mean']:.4f} | "
            f"front={row['algorithm_front_size_mean']:.2f}/{row['exact_front_size_mean']:.2f} | "
            f"test_error={row['test_error_mean']:.6f} | selected={row['selected_count_mean']:.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze_native_exact(_load_records(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(_text_report(report), encoding="utf-8")
    print(f"records = {report['integrity']['native_metric_records']}")
    print(f"summary_records = {len(report['summary'])}")
    print(f"output = {args.output}")
    if args.text_output is not None:
        print(f"text_output = {args.text_output}")


if __name__ == "__main__":
    main()


__all__ = ["analyze_native_exact"]
