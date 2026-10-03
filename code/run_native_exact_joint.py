"""Run joint algorithms and native PEP/MDEP exact references in one context."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from batch_runner import (
    BatchConfig,
    _algorithm_runs,
    _make_objective_context,
    _record_result,
    _run_algorithm,
    _select_specs,
)
from classifier_pool import train_classifier_pool
from data_loader import load_experiment_dataset, make_fold_split, make_outer_splits
from ensemble_pruning import EnsembleEvaluator
from native_exact_pareto import enumerate_mdep_exact, enumerate_pep_exact


def _exact_record(result, *, dataset, fold, seed, split_sizes, context):
    return {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "optimization_seed": None,
        "algorithm": f"exact_{result.objective_name}",
        "algorithm_parameters": {
            "implementation_type": "native_exact_reference",
            "objective_definition": result.objective_name,
            "evaluated_subsets": result.evaluated_subsets,
            "max_classifiers": result.n_classifiers,
        },
        "split_sizes": split_sizes,
        "n_objectives": int(result.pareto_objectives.shape[1]),
        "evaluated_subsets": int(result.evaluated_subsets),
        "raw_front": result.pareto_objectives.tolist(),
        "pareto_masks": result.pareto_masks.astype(int).tolist(),
        "objective_context_id": context["id"],
        "objective_context": context["values"],
    }


def run_native_exact_joint(config: BatchConfig, output: str | Path) -> list[dict]:
    if config.include_exact_front is False:
        raise ValueError("include_exact_front must be enabled for native references")
    records: list[dict] = []
    for dataset_name in config.datasets:
        dataset = load_experiment_dataset(dataset_name, data_home=config.data_home)
        for seed in config.seeds:
            folds = make_outer_splits(
                dataset.features, dataset.labels, n_splits=config.n_splits, random_state=seed
            )
            for fold, (outer_train, outer_test) in enumerate(folds):
                split = make_fold_split(
                    dataset.features,
                    dataset.labels,
                    outer_train,
                    outer_test,
                    val_size=config.val_size,
                    random_state=seed + fold,
                )
                pool = train_classifier_pool(
                    split.x_train,
                    split.y_train,
                    split.x_val,
                    split.x_test,
                    specs=_select_specs(config.pool_size, seed + fold),
                    timing_repeats=config.timing_repeats,
                )
                val_evaluator = EnsembleEvaluator(pool.val_predictions, split.y_val, pool.combined_costs)
                test_evaluator = EnsembleEvaluator(pool.test_predictions, split.y_test, pool.combined_costs)
                context = _make_objective_context(pool, split.y_val, split.y_test)
                split_sizes = {
                    "train": int(len(split.y_train)),
                    "validation": int(len(split.y_val)),
                    "test": int(len(split.y_test)),
                }
                # Native references are generated from the same validation
                # predictions and context as the algorithm results.
                records.append(_exact_record(
                    enumerate_pep_exact(val_evaluator, max_classifiers=config.exact_max_classifiers),
                    dataset=dataset.name, fold=fold, seed=seed,
                    split_sizes=split_sizes, context=context,
                ))
                records.append(_exact_record(
                    enumerate_mdep_exact(val_evaluator, max_classifiers=config.exact_max_classifiers),
                    dataset=dataset.name, fold=fold, seed=seed,
                    split_sizes=split_sizes, context=context,
                ))
                optimizer_seeds = config.optimization_seeds or (seed + fold,)
                for optimization_seed in optimizer_seeds:
                    for label, algorithm, tolerance in _algorithm_runs(config):
                        started_at = time.perf_counter()
                        result = _run_algorithm(
                            val_evaluator,
                            algorithm,
                            pop_size=config.pop_size,
                            n_gen=config.n_gen,
                            seed=optimization_seed,
                            boundary_tolerance=tolerance,
                            pep_iterations=config.pep_iterations,
                        )
                        params = {
                            "pop_size": config.pop_size,
                            "n_gen": config.n_gen,
                            "variation": "de",
                        }
                        if algorithm == "pep_paper":
                            params.update({
                                "implementation_type": "paper_reimplementation",
                                "source_paper": "Qian et al. (AAAI 2015), Pareto Ensemble Pruning",
                                "objective_definition": "validation_error_and_selected_count",
                                "vds_enabled": True,
                                "empty_subset_policy": "paper_infinite_error",
                                "iterations": config.pep_iterations,
                            })
                        if algorithm == "mdep_paper":
                            params.update({
                                "implementation_type": "paper_reimplementation",
                                "source_paper": "Wu et al. (2022), Multi-objective Evolutionary Ensemble Pruning Guided by Margin Distribution",
                                "objective_definition": "validation_error_margin_ratio_selected_count",
                                "moea": "NSGA-III",
                                "variation": "uniform_crossover",
                                "crossover": "uniform",
                                "crossover_probability": 0.7,
                                "mutation_probability_per_bit": 1.0 / config.pool_size,
                                "initialization_policy": "retain_non_dominated_singletons",
                                "offspring_repair_policy": "repair_size_zero_or_one",
                                "implementation_revision": "mdep_singleton_initialization_fix_v1",
                            })
                        record = _record_result(
                            result,
                            test_evaluator,
                            dataset=dataset.name,
                            fold=fold,
                            seed=seed,
                            algorithm=label,
                            split_sizes=split_sizes,
                            objective_context=context,
                            algorithm_parameters=params,
                        )
                        record["optimization_seed"] = int(optimization_seed)
                        record["algorithm_runtime_seconds"] = float(time.perf_counter() - started_at)
                        records.append(record)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="iris,banknote,segment,spambase")
    parser.add_argument("--seeds", default="20260803,20260804,20260805,20260806,20260807")
    parser.add_argument("--optimization-seeds", default="20260803,20260804,20260805,20260806,20260807")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--n-gen", type=int, default=50)
    parser.add_argument("--pep-iterations", type=int, default=13)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--data-home", default="data/cache")
    parser.add_argument("--exact-max-classifiers", type=int, default=6)
    args = parser.parse_args()
    config = BatchConfig(
        datasets=tuple(v.strip() for v in args.datasets.split(",") if v.strip()),
        algorithms=("pep_paper", "mdep_paper", "nsga2", "pddr_local_search_delete_only"),
        seeds=tuple(int(v) for v in args.seeds.split(",") if v.strip()),
        optimization_seeds=tuple(int(v) for v in args.optimization_seeds.split(",") if v.strip()),
        n_splits=args.n_splits,
        val_size=args.val_size,
        pool_size=args.pool_size,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        timing_repeats=args.timing_repeats,
        include_exact_front=True,
        exact_max_classifiers=args.exact_max_classifiers,
        data_home=args.data_home,
        pep_iterations=args.pep_iterations,
    )
    records = run_native_exact_joint(config, args.output)
    print(f"records = {len(records)}")
    print(f"output = {args.output}")


if __name__ == "__main__":
    main()
