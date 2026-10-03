"""Common multi-objective metrics with one normalization protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass(frozen=True)
class NormalizationContext:
    """Shared affine normalization and hypervolume reference point."""

    ideal: np.ndarray
    nadir: np.ndarray
    span: np.ndarray
    reference_point: np.ndarray


def _validate_front(front: np.ndarray) -> np.ndarray:
    """Validate one objective front and return a floating matrix."""
    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("front must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("front must contain only finite values")
    return values


def build_normalization_context(
    fronts: dict[str, np.ndarray], *, reference_margin: float = 0.1
) -> NormalizationContext:
    """Build one ideal/nadir/reference point from all compared fronts.

    All objectives are minimized.  The context must be built before computing
    any algorithm-specific metric so no method receives a different scale.
    """
    if not fronts:
        raise ValueError("fronts must contain at least one algorithm")
    if reference_margin <= 0:
        raise ValueError("reference_margin must be positive")
    values = [_validate_front(front) for front in fronts.values()]
    n_obj = values[0].shape[1]
    if any(front.shape[1] != n_obj for front in values):
        raise ValueError("all fronts must have the same number of objectives")
    combined = np.vstack(values)
    ideal = combined.min(axis=0)
    nadir = combined.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    reference = np.ones(n_obj, dtype=float) * (1.0 + reference_margin)
    return NormalizationContext(ideal, nadir, span, reference)


def normalize_objectives(
    front: np.ndarray, context: NormalizationContext
) -> np.ndarray:
    """Normalize a front with an already shared context."""
    values = _validate_front(front)
    if values.shape[1] != context.ideal.size:
        raise ValueError("front objective count does not match normalization context")
    return (values - context.ideal) / context.span


def nondominated_front(front: np.ndarray) -> np.ndarray:
    """Return only non-dominated rows, preserving objective values."""
    values = _validate_front(front)
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    # Duplicate objective points do not dominate each other; remove them here
    # so front size, IGD+ references and serialized diagnostics are not inflated.
    return np.unique(values[np.asarray(indices, dtype=int)], axis=0)


def merge_reference_front(fronts: dict[str, np.ndarray]) -> np.ndarray:
    """Build the cross-algorithm merged non-dominated reference front."""
    if not fronts:
        raise ValueError("fronts must contain at least one algorithm")
    return nondominated_front(np.vstack([_validate_front(front) for front in fronts.values()]))


def compute_front_metrics(
    fronts: dict[str, np.ndarray], *, reference_margin: float = 0.1
) -> tuple[NormalizationContext, np.ndarray, dict[str, dict[str, float]]]:
    """Compute common HV and IGD+ for every algorithm front.

    IGD+ uses the merged non-dominated front as a relative reference when no
    exact Pareto front is available.  For a small exhaustive problem, callers
    should replace it with the exact front explicitly.
    """
    context = build_normalization_context(fronts, reference_margin=reference_margin)
    merged_reference = merge_reference_front(fronts)
    normalized_reference = normalize_objectives(merged_reference, context)
    hypervolume = HV(ref_point=context.reference_point)
    igd_plus = IGDPlus(normalized_reference)
    metrics: dict[str, dict[str, float]] = {}
    for name, front in fronts.items():
        clean_front = nondominated_front(front)
        normalized = normalize_objectives(clean_front, context)
        metrics[name] = {
            "hypervolume": float(hypervolume.do(normalized)),
            "igd_plus": float(igd_plus.do(normalized)),
            "front_size": float(len(clean_front)),
        }
    return context, merged_reference, metrics


__all__ = [
    "NormalizationContext",
    "build_normalization_context",
    "compute_front_metrics",
    "merge_reference_front",
    "nondominated_front",
    "normalize_objectives",
]
