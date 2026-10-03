"""Native exact Pareto references for PEP and MDEP objective spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ensemble_pruning import EnsembleEvaluator
from mdep_paper import margin_ratio
from pddrff import nondominated_mask


@dataclass(frozen=True)
class NativeExactParetoResult:
    """Exact non-dominated subsets for one objective definition."""

    objective_name: str
    n_classifiers: int
    evaluated_subsets: int
    pareto_masks: np.ndarray
    pareto_objectives: np.ndarray


def _mask_from_integer(value: int, n_classifiers: int) -> np.ndarray:
    bits = (value >> np.arange(n_classifiers)) & 1
    return bits.astype(bool)


def enumerate_native_exact_pareto(
    evaluator: EnsembleEvaluator,
    objective_fn: Callable[[np.ndarray], np.ndarray],
    *,
    objective_name: str,
    max_classifiers: int = 20,
    max_subsets: int = 1_048_576,
) -> NativeExactParetoResult:
    """Enumerate every non-empty subset and retain unique non-dominated points."""
    n = evaluator.n_classifiers
    if n > max_classifiers:
        raise ValueError(f"exact enumeration supports at most {max_classifiers} classifiers")
    total = (1 << n) - 1
    if total > max_subsets:
        raise ValueError(f"enumeration requires {total} subsets, above max_subsets={max_subsets}")
    masks = np.empty((total, n), dtype=bool)
    objectives: list[np.ndarray] = []
    for row, value in enumerate(range(1, total + 1)):
        mask = _mask_from_integer(value, n)
        masks[row] = mask
        objective = np.asarray(objective_fn(mask), dtype=float)
        if objective.ndim != 1 or not np.isfinite(objective).all():
            raise ValueError(f"objective_fn returned invalid value for subset {value}")
        objectives.append(objective)
    objective_matrix = np.vstack(objectives)
    front = nondominated_mask(objective_matrix)
    front_masks = masks[front]
    front_objectives = objective_matrix[front]
    _, unique_indices = np.unique(front_objectives, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return NativeExactParetoResult(
        objective_name=objective_name,
        n_classifiers=n,
        evaluated_subsets=total,
        pareto_masks=front_masks[unique_indices],
        pareto_objectives=front_objectives[unique_indices],
    )


def pep_objectives(evaluator: EnsembleEvaluator, mask: np.ndarray) -> np.ndarray:
    """Return PEP's native ``(validation_error, selected_count/n)`` objectives."""
    result = evaluator.evaluate(mask)
    return np.array(
        [result.validation_error, len(result.selected_indices) / evaluator.n_classifiers],
        dtype=float,
    )


def mdep_objectives(evaluator: EnsembleEvaluator, mask: np.ndarray) -> np.ndarray:
    """Return MDEP's native ``(validation_error, margin_ratio, selected_count/n)``."""
    result = evaluator.evaluate(mask)
    return np.array(
        [
            result.validation_error,
            margin_ratio(evaluator.predictions, evaluator.labels, mask),
            len(result.selected_indices) / evaluator.n_classifiers,
        ],
        dtype=float,
    )


def enumerate_pep_exact(evaluator: EnsembleEvaluator, **kwargs) -> NativeExactParetoResult:
    return enumerate_native_exact_pareto(
        evaluator, lambda mask: pep_objectives(evaluator, mask), objective_name="pep_native", **kwargs
    )


def enumerate_mdep_exact(evaluator: EnsembleEvaluator, **kwargs) -> NativeExactParetoResult:
    return enumerate_native_exact_pareto(
        evaluator, lambda mask: mdep_objectives(evaluator, mask), objective_name="mdep_native", **kwargs
    )


__all__ = [
    "NativeExactParetoResult",
    "enumerate_mdep_exact",
    "enumerate_native_exact_pareto",
    "enumerate_pep_exact",
    "mdep_objectives",
    "pep_objectives",
]
