"""Smoke checks for the joint domain-baseline experiment."""

import json

from batch_runner import BatchConfig, run_batch


def test_joint_domain_methods_share_context_and_objective_groups(tmp_path) -> None:
    output = tmp_path / "joint.json"
    records = run_batch(
        BatchConfig(
            datasets=("iris",),
            algorithms=("pep_paper", "mdep_paper", "nsga2", "pddr"),
            seeds=(7,),
            optimization_seeds=(101, 102),
            n_splits=2,
            pool_size=4,
            pop_size=4,
            n_gen=2,
            pep_iterations=8,
            timing_repeats=1,
            include_exact_front=True,
            exact_max_classifiers=4,
        ),
        output,
    )
    assert len(records) == 2 * 2 * 4 + 2
    algorithm_rows = [r for r in records if r["algorithm"] != "exact_pareto"]
    assert {r["algorithm"] for r in algorithm_rows} == {
        "pep_paper", "mdep_paper", "nsga2", "pddr"
    }
    for fold in range(2):
        rows = [r for r in algorithm_rows if r["fold"] == fold]
        assert len({r["objective_context_id"] for r in rows}) == 1
        assert {r["n_objectives"] for r in rows} == {2, 3}
    assert json.loads(output.read_text(encoding="utf-8"))
