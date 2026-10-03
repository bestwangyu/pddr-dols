"""Integrity checks for dataset-wise domain experiment merging."""

import json

import pytest

from merge_joint_json import merge


def test_merge_requires_all_expected_records(tmp_path) -> None:
    path = tmp_path / "joint_iris_seed_20260820.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete expected"):
        merge(tmp_path, ("iris",))


def test_merge_rejects_wrong_mdep_revision(tmp_path) -> None:
    rows = []
    for fold in range(5):
        for algorithm in ("exact_pep_native", "exact_mdep_native"):
            rows.append({"dataset": "iris", "seed": 20260820, "fold": fold,
                         "algorithm": algorithm, "optimization_seed": None,
                         "objective_context_id": f"context-{fold}"})
        for seed in range(20260820, 20260825):
            for algorithm in ("pep_paper", "mdep_paper", "nsga2",
                              "pddr_local_search_delete_only"):
                rows.append({"dataset": "iris", "seed": 20260820, "fold": fold,
                             "algorithm": algorithm, "optimization_seed": seed,
                             "objective_context_id": f"context-{fold}",
                             "evaluation_count": 1976 if algorithm == "pep_paper" else 2000,
                             "algorithm_parameters": {"implementation_revision": "wrong"}})
    path = tmp_path / "joint_iris_seed_20260820.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="MDEP initialization revision mismatch"):
        merge(tmp_path, ("iris",))
    for row in rows:
        if row["algorithm"] == "mdep_paper":
            row["algorithm_parameters"]["implementation_revision"] = (
                "mdep_singleton_initialization_fix_v1"
            )
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert len(merge(tmp_path, ("iris",))) == 110
