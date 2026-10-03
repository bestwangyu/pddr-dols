"""Exact Pareto-front enumeration for small classifier pools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from ensemble_pruning import EnsembleEvaluator
from pddrff import nondominated_mask


@dataclass(frozen=True)
class ExactParetoResult:
    """All evaluated subset count and unique exact Pareto solutions."""

    n_classifiers: int
    evaluated_subsets: int
    pareto_masks: np.ndarray
    pareto_objectives: np.ndarray


def _mask_from_integer(value: int, n_classifiers: int) -> np.ndarray:
    """Convert one positive integer to a fixed-width binary decision vector."""
    bits = (value >> np.arange(n_classifiers)) & 1
    return bits.astype(bool)


def enumerate_exact_pareto(
    evaluator: EnsembleEvaluator,
    *,
    max_classifiers: int = 20,
    max_subsets: int = 1_048_576,
) -> ExactParetoResult:
    """Enumerate every non-empty subset and return the exact unique objective front.

    The returned front is exact for the supplied prediction cache and objective
    definitions, not a generalization estimate.  Empty subsets are excluded
    because the evaluator's repair rule makes them duplicates of a valid subset.
    """
    n_classifiers = evaluator.n_classifiers
    if n_classifiers > max_classifiers:
        raise ValueError(
            f"exact enumeration supports at most {max_classifiers} classifiers"
        )
    total_subsets = (1 << n_classifiers) - 1
    if total_subsets > max_subsets:
        raise ValueError(
            f"enumeration requires {total_subsets} subsets, above max_subsets={max_subsets}"
        )

    masks = np.empty((total_subsets, n_classifiers), dtype=bool)
    objectives = np.empty((total_subsets, 3), dtype=float)
    for row, value in enumerate(range(1, total_subsets + 1)):
        mask = _mask_from_integer(value, n_classifiers)
        masks[row] = mask
        objectives[row] = evaluator.evaluate_objectives(mask)

    front_mask = nondominated_mask(objectives)
    front_masks = masks[front_mask]
    front_objectives = objectives[front_mask]
    # Duplicate objective points are equivalent for the Pareto front metric;
    # retain the first corresponding subset for reproducible reporting.
    _, unique_indices = np.unique(
        front_objectives, axis=0, return_index=True
    )
    unique_indices = np.sort(unique_indices)
    return ExactParetoResult(
        n_classifiers=n_classifiers,
        evaluated_subsets=total_subsets,
        pareto_masks=front_masks[unique_indices],
        pareto_objectives=front_objectives[unique_indices],
    )


def save_exact_pareto(result: ExactParetoResult, path: str | Path) -> None:
    """Save an exact front as a portable JSON record."""
    output = {
        "n_classifiers": result.n_classifiers,
        "evaluated_subsets": result.evaluated_subsets,
        "pareto_masks": result.pareto_masks.astype(int).tolist(),
        "pareto_objectives": result.pareto_objectives.tolist(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


__all__ = ["ExactParetoResult", "enumerate_exact_pareto", "save_exact_pareto"]
