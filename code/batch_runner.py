"""Multi-dataset nested-fold experiment orchestration without checkpointing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from classifier_pool import build_base_classifier_specs, train_classifier_pool
from data_loader import (
    load_experiment_dataset,
    make_fold_split,
    make_outer_splits,
)
from domain_baselines import run_domain_baseline
from ensemble_pruning import EnsembleEvaluator
from exact_pareto import enumerate_exact_pareto
from fair_baselines import (
    MOEAD_MAX_NEIGHBORS,
    MOEAD_NEIGHBOR_MATING_PROBABILITY,
    MOEAD_PBI_THETA,
    MOEAD_REFERENCE_METHOD,
    MOEAD_REFERENCE_SEED,
    make_reference_directions,
    run_baseline,
)
from pddr_algorithm import run_binary_pddr
from pddr_local_search import (
    PDDR_DOLS_INTERNAL_ID,
    PDDR_DOLS_NAME,
    run_pddr_local_search,
)
from pep_paper import run_pep_paper
from mdep_paper import run_mdep_paper


ALL_ALGORITHMS = (
    "pddr",
    "da_pddr",
    "da_pddr_obj",
    "rank_pddr",
    "hybrid_pddr",
    "extreme_pddr",
    "pddr_local_search",
    "pddr_local_search_sparse",
    "pddr_local_search_medium",
    "pddr_local_search_add_only",
    "pddr_local_search_delete_only",
    "pddr_local_search_swap_only",
    "pddr_local_search_add_delete",
    "pddr_local_search_add_swap",
    "pddr_local_search_delete_swap",
    "local_search_always",
    "local_search_always_delete_only",
    "local_search_always_add_delete",
    "nsga2",
    "reference",
    "moead",
    "pep",
    "pep_paper",
    "mdep",
    "mdep_paper",
)
DEFAULT_ALGORITHMS = tuple(
    algorithm
    for algorithm in ALL_ALGORITHMS
    if algorithm not in {"pep_paper", "mdep_paper"}
)

# Canonical names for presentation.  Internal ids are retained for backward
# compatibility with the frozen archives and analysis scripts.
ALGORITHM_DISPLAY_NAMES = {
    PDDR_DOLS_INTERNAL_ID: PDDR_DOLS_NAME,
    "local_search_always_delete_only": "Always delete-only",
    "nsga2": "NSGA-II",
    "moead": "MOEA/D",
}


LOCAL_SEARCH_PARAMETERS = {
    "pddr_local_search": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.35,
        "objective_unique_threshold": 0.5,
        "stagnation_patience": 3,
        "allowed_operations": ("add", "delete", "swap"),
    },
    "pddr_local_search_sparse": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.15,
        "objective_unique_threshold": 0.25,
        "stagnation_patience": 6,
        "allowed_operations": ("add", "delete", "swap"),
    },
    "pddr_local_search_medium": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("add", "delete", "swap"),
    },
    "pddr_local_search_add_only": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("add",),
    },
    "pddr_local_search_delete_only": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("delete",),
    },
    "pddr_local_search_swap_only": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("swap",),
    },
    "pddr_local_search_add_delete": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("add", "delete"),
    },
    "pddr_local_search_add_swap": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("add", "swap"),
    },
    "pddr_local_search_delete_swap": {
        "trigger_mode": "pddr",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.25,
        "objective_unique_threshold": 0.4,
        "stagnation_patience": 4,
        "allowed_operations": ("delete", "swap"),
    },
    "local_search_always": {
        "trigger_mode": "always",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.35,
        "objective_unique_threshold": 0.5,
        "stagnation_patience": 3,
        "allowed_operations": ("add", "delete", "swap"),
    },
    "local_search_always_delete_only": {
        "trigger_mode": "always",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.35,
        "objective_unique_threshold": 0.5,
        "stagnation_patience": 3,
        "allowed_operations": ("delete",),
    },
    "local_search_always_add_delete": {
        "trigger_mode": "always",
        "local_fraction": 0.5,
        "pddr_unique_threshold": 0.35,
        "objective_unique_threshold": 0.5,
        "stagnation_patience": 3,
        "allowed_operations": ("add", "delete"),
    },
}


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for one reproducible batch invocation."""

    datasets: tuple[str, ...] = ("iris", "wine", "breast_cancer")
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    seeds: tuple[int, ...] = (20260803,)
    # Optional independent optimizer seeds. Outer folds remain controlled by
    # ``seeds``; when omitted, the historical seed+fold behavior is retained.
    optimization_seeds: tuple[int, ...] = ()
    n_splits: int = 5
    val_size: float = 0.2
    pool_size: int = 6
    pop_size: int = 20
    n_gen: int = 5
    timing_repeats: int = 1
    include_exact_front: bool = False
    exact_max_classifiers: int = 20
    data_home: str | None = None
    objective_boundary_tolerances: tuple[float, ...] = (0.05,)
    # ``None`` uses the paper's ceil(n^2 log n) iterations.  A positive value
    # is useful for matched-budget smoke tests.
    pep_iterations: int | None = None


def _tolerance_label(value: float) -> str:
    """Return a stable filesystem- and JSON-friendly tolerance label."""
    text = format(float(value), ".12g").replace("-", "m").replace(".", "p")
    return f"da_pddr_obj_tol_{text}"


def _algorithm_runs(config: BatchConfig) -> tuple[tuple[str, str, float | None], ...]:
    """Expand an objective-space method into separately labelled tolerances."""
    runs: list[tuple[str, str, float | None]] = []
    expand = len(config.objective_boundary_tolerances) > 1
    for algorithm in config.algorithms:
        if algorithm != "da_pddr_obj":
            runs.append((algorithm, algorithm, None))
            continue
        for tolerance in config.objective_boundary_tolerances:
            label = _tolerance_label(tolerance) if expand else algorithm
            runs.append((label, algorithm, float(tolerance)))
    return tuple(runs)


def _select_specs(pool_size: int, seed: int):
    """Select heterogeneous configurations spread over the base grid."""
    specs = build_base_classifier_specs(random_state=seed)
    if not 2 <= pool_size <= len(specs):
        raise ValueError(f"pool_size must be between 2 and {len(specs)}")
    indices = np.linspace(0, len(specs) - 1, pool_size, dtype=int)
    return tuple(specs[int(index)] for index in indices)


def _validate_config(config: BatchConfig) -> None:
    """Reject invalid batch settings before any model training starts."""
    if not config.datasets or not config.seeds:
        raise ValueError("datasets and seeds must be non-empty")
    if any(not isinstance(seed, int) for seed in config.optimization_seeds):
        raise ValueError("optimization_seeds must contain integers")
    if len(set(config.optimization_seeds)) != len(config.optimization_seeds):
        raise ValueError("optimization_seeds must be unique")
    unknown = set(config.algorithms) - set(ALL_ALGORITHMS)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    if config.n_splits < 2 or config.pop_size < 2 or config.n_gen < 1:
        raise ValueError("n_splits, pop_size and n_gen have invalid values")
    if config.pep_iterations is not None and config.pep_iterations < 1:
        raise ValueError("pep_iterations must be positive when provided")
    tolerances = config.objective_boundary_tolerances
    if not tolerances:
        raise ValueError("objective_boundary_tolerances must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("objective boundary tolerances must be finite and non-negative")
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("objective boundary tolerances must be unique")


def _run_algorithm(
    evaluator: EnsembleEvaluator,
    algorithm: str,
    *,
    pop_size: int,
    n_gen: int,
    seed: int,
    boundary_tolerance: float | None = None,
    pep_iterations: int | None = None,
):
    """Dispatch one algorithm while preserving a common evaluation budget."""
    if algorithm in LOCAL_SEARCH_PARAMETERS:
        local_config = LOCAL_SEARCH_PARAMETERS[algorithm]
        _, _, result = run_pddr_local_search(
            evaluator,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            trigger_mode=local_config["trigger_mode"],
            pddr_unique_threshold=local_config["pddr_unique_threshold"],
            objective_unique_threshold=local_config["objective_unique_threshold"],
            stagnation_patience=local_config["stagnation_patience"],
            allowed_operations=local_config["allowed_operations"],
        )
    elif algorithm in {
        "pddr",
        "da_pddr",
        "da_pddr_obj",
        "rank_pddr",
        "hybrid_pddr",
        "extreme_pddr",
    }:
        _, _, result = run_binary_pddr(
            evaluator,
            variant=algorithm,
            variation="de",
            pop_size=pop_size,
            n_gen=n_gen,
            boundary_tolerance=(
                0.05 if boundary_tolerance is None else boundary_tolerance
            ),
            seed=seed,
        )
    elif algorithm in {"nsga2", "reference", "moead"}:
        _, _, result = run_baseline(
            evaluator,
            variant=algorithm,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
    elif algorithm in {"pep", "mdep"}:
        _, _, result = run_domain_baseline(
            evaluator,
            variant=algorithm,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
    elif algorithm == "pep_paper":
        return run_pep_paper(
            evaluator,
            iterations=pep_iterations,
            seed=seed,
            vds_enabled=True,
            empty_subset_policy="paper_infinite_error",
        )
    elif algorithm == "mdep_paper":
        return run_mdep_paper(
            evaluator,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return result


def _record_result(
    result,
    test_evaluator: EnsembleEvaluator,
    *,
    dataset: str,
    fold: int,
    seed: int,
    algorithm: str,
    split_sizes: dict[str, int],
    objective_context: dict,
    algorithm_parameters: dict,
) -> dict:
    """Serialize one algorithm/fold result without using test labels for choice."""
    if algorithm == "mdep_paper":
        # MDEP's paper-level eval rule is validation error, then size; margin
        # ratio is not used to choose the final representative solution.
        best_index = int(
            min(
                range(len(result.X)),
                key=lambda index: (
                    float(result.F[index][0]),
                    float(result.F[index][2]),
                    index,
                ),
            )
        )
    else:
        best_index = int(
            min(
                range(len(result.X)),
                key=lambda index: tuple(
                    [float(value) for value in result.F[index]] + [index]
                ),
            )
        )
    best_mask = np.asarray(result.X[best_index], dtype=bool)
    test_result = test_evaluator.evaluate(best_mask)
    raw_masks = np.asarray(result.X, dtype=bool)
    record = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "algorithm": algorithm,
        "method_name": ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
        "algorithm_parameters": algorithm_parameters,
        "split_sizes": split_sizes,
        "n_objectives": int(result.F.shape[1]),
        "population_size": int(len(result.X)),
        "evaluation_count": _result_evaluation_count(result),
        "raw_front": np.asarray(result.F, dtype=float).tolist(),
        "raw_masks": raw_masks.astype(int).tolist(),
        "best_mask": best_mask.astype(int).tolist(),
        "best_validation_objectives": result.F[best_index].tolist(),
        "test_error": test_result.validation_error,
        "test_margin_loss": test_result.margin_loss,
        "selected_count": len(test_result.selected_indices),
        "objective_context_id": objective_context["id"],
        "objective_context": objective_context["values"],
    }
    search_diagnostics = getattr(result.algorithm, "search_diagnostics", None)
    if search_diagnostics is not None:
        record["search_diagnostics"] = search_diagnostics
    if hasattr(result.algorithm, "iterations") and hasattr(
        result.algorithm, "vds_evaluations"
    ):
        record["pep_search_stats"] = asdict(result.algorithm)
    return record


def _result_evaluation_count(result) -> int | None:
    """Extract evaluation accounting from pymoo and standalone algorithms."""
    algorithm = getattr(result, "algorithm", None)
    direct = getattr(algorithm, "evaluation_count", None)
    if direct is not None:
        return int(direct)
    evaluator = getattr(algorithm, "evaluator", None)
    n_eval = getattr(evaluator, "n_eval", None)
    return int(n_eval) if n_eval is not None else None


def _array_digest(values: np.ndarray) -> str:
    """Return a stable digest for a prediction or label array."""
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _make_objective_context(pool, val_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    """Build a reproducibility record for one shared classifier-pool cost space."""
    values = {
        "classifier_names": list(pool.names),
        "inference_seconds": np.asarray(pool.inference_seconds, dtype=float).tolist(),
        "storage_bytes": np.asarray(pool.storage_bytes, dtype=int).tolist(),
        "combined_costs": np.asarray(pool.combined_costs, dtype=float).tolist(),
        "cost_scale": float(np.asarray(pool.combined_costs, dtype=float).sum()),
        "inference_weight": 0.5,
        "storage_weight": 0.5,
        "validation_predictions_sha256": _array_digest(pool.val_predictions),
        "validation_labels_sha256": _array_digest(val_labels),
        "test_predictions_sha256": _array_digest(pool.test_predictions),
        "test_labels_sha256": _array_digest(test_labels),
    }
    encoded = json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "values": values,
    }


def run_batch(config: BatchConfig, output_path: str | Path | None = None) -> list[dict]:
    """Run all configured datasets, folds, seeds and algorithms in memory.

    This function intentionally has no checkpoint/resume behavior.  Each run
    writes one complete result list at the end, as requested for the initial
    batch implementation.
    """
    _validate_config(config)
    records: list[dict] = []
    for dataset_name in config.datasets:
        dataset = load_experiment_dataset(dataset_name, data_home=config.data_home)
        for seed in config.seeds:
            folds = make_outer_splits(
                dataset.features,
                dataset.labels,
                n_splits=config.n_splits,
                random_state=seed,
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
                val_evaluator = EnsembleEvaluator(
                    pool.val_predictions, split.y_val, pool.combined_costs
                )
                test_evaluator = EnsembleEvaluator(
                    pool.test_predictions, split.y_test, pool.combined_costs
                )
                objective_context = _make_objective_context(
                    pool, split.y_val, split.y_test
                )
                split_sizes = {
                    "train": int(len(split.y_train)),
                    "validation": int(len(split.y_val)),
                    "test": int(len(split.y_test)),
                }
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
                        algorithm_runtime_seconds = time.perf_counter() - started_at
                        parameters = {
                            "pop_size": config.pop_size,
                            "n_gen": config.n_gen,
                            "variation": "de",
                            "de_F": 0.5,
                            "de_K": 0.5,
                            "de_CR": 0.9,
                            "mutation": "bitflip",
                            "mutation_probability_per_bit": 0.02,
                            "repair": "cheapest_classifier_for_empty_subset",
                            "method_name": ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
                        }
                        if algorithm == "moead":
                            reference_directions = make_reference_directions(
                                config.pop_size,
                                n_obj=3,
                                seed=MOEAD_REFERENCE_SEED,
                            )
                            parameters.update(
                                {
                                    "implementation_type": "shared_binary_de_baseline",
                                    "decomposition": "PBI",
                                    "pbi_theta": MOEAD_PBI_THETA,
                                    "n_neighbors": min(
                                        MOEAD_MAX_NEIGHBORS,
                                        config.pop_size,
                                    ),
                                    "prob_neighbor_mating": (
                                        MOEAD_NEIGHBOR_MATING_PROBABILITY
                                    ),
                                    "reference_direction_method": (
                                        MOEAD_REFERENCE_METHOD
                                    ),
                                    "reference_direction_count": config.pop_size,
                                    "reference_direction_seed": (
                                        MOEAD_REFERENCE_SEED
                                    ),
                                    "reference_direction_sha256": _array_digest(
                                        reference_directions
                                    ),
                                }
                            )
                        if algorithm == "pep_paper":
                            parameters.update(
                                {
                                    "implementation_type": "paper_reimplementation",
                                    "source_paper": "Qian et al. (AAAI 2015), Pareto Ensemble Pruning",
                                    "objective_definition": "validation_error_and_selected_count",
                                    "vds_enabled": True,
                                    "empty_subset_policy": "paper_infinite_error",
                                    "iterations": (
                                        config.pep_iterations
                                        if config.pep_iterations is not None
                                        else "ceil(n_classifiers^2 log n_classifiers)"
                                    ),
                                }
                            )
                        if algorithm == "mdep_paper":
                            parameters.update(
                                {
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
                                }
                            )
                        if tolerance is not None:
                            parameters["boundary_tolerance"] = tolerance
                        if algorithm in LOCAL_SEARCH_PARAMETERS:
                            local_config = LOCAL_SEARCH_PARAMETERS[algorithm]
                            parameters.update(
                                {
                                    "trigger_mode": local_config["trigger_mode"],
                                    "local_fraction": local_config["local_fraction"],
                                    "pddr_unique_threshold": local_config[
                                        "pddr_unique_threshold"
                                    ],
                                    "objective_unique_threshold": local_config[
                                        "objective_unique_threshold"
                                    ],
                                "stagnation_patience": local_config[
                                    "stagnation_patience"
                                ],
                                "allowed_operations": list(
                                    local_config["allowed_operations"]
                                ),
                                }
                            )
                        record = _record_result(
                            result,
                            test_evaluator,
                            dataset=dataset.name,
                            fold=fold,
                            seed=seed,
                            algorithm=label,
                            split_sizes=split_sizes,
                            objective_context=objective_context,
                            algorithm_parameters=parameters,
                        )
                        record["optimization_seed"] = int(optimization_seed)
                        record["algorithm_runtime_seconds"] = float(
                            algorithm_runtime_seconds
                        )
                        records.append(record)
                if config.include_exact_front:
                    exact = enumerate_exact_pareto(
                        val_evaluator,
                        max_classifiers=config.exact_max_classifiers,
                    )
                    records.append(
                        {
                            "dataset": dataset.name,
                            "fold": fold,
                            "seed": seed,
                            "optimization_seed": None,
                            "algorithm": "exact_pareto",
                            "split_sizes": split_sizes,
                            "n_objectives": 3,
                            "evaluated_subsets": exact.evaluated_subsets,
                            "raw_front": exact.pareto_objectives.tolist(),
                            "pareto_masks": exact.pareto_masks.astype(int).tolist(),
                            "objective_context_id": objective_context["id"],
                            "objective_context": objective_context["values"],
                        }
                    )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
    return records


__all__ = ["ALL_ALGORITHMS", "DEFAULT_ALGORITHMS", "BatchConfig", "run_batch"]
