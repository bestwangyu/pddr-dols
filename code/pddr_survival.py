"""PDDR-FF survival selection and tie-degeneracy diagnostics.

This module implements the direct PDDR-FF survival baseline (PDDR-EP).  The
DA-PDDR-EP secondary tie-breaker is intentionally kept for the next stage, so
that the effect of the original survival rule can be measured separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.core.survival import Survival

from pddrff import pddr_ff


@dataclass(frozen=True)
class SurvivalResult:
    """Selected indices and diagnostics from one survival operation."""

    selected_indices: np.ndarray
    eval_values: np.ndarray
    q: np.ndarray
    p: np.ndarray
    nondominated_ratio: float
    unique_eval_ratio: float
    boundary_tie_size: int


def _validate_survival_inputs(
    objectives: np.ndarray, n_survive: int
) -> np.ndarray:
    """Validate a survival request and return a floating objective matrix."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("objectives must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("objectives must contain only finite values")
    if not isinstance(n_survive, (int, np.integer)):
        raise ValueError("n_survive must be an integer")
    if not 1 <= int(n_survive) <= values.shape[0]:
        raise ValueError("n_survive must be between 1 and the population size")
    return values


def select_pddr_survivors(
    objectives: np.ndarray, n_survive: int
) -> SurvivalResult:
    """Select the lowest PDDR-FF values with stable index tie handling.

    The direct baseline does not use crowding distance or a secondary
    diversity rule.  If the truncation boundary contains equal PDDR-FF values,
    lower original indices are selected first and the complete boundary tie is
    reported in ``boundary_tie_size`` for later DA-PDDR analysis.
    """
    values = _validate_survival_inputs(objectives, n_survive)
    eval_values, q, p = pddr_ff(values)

    # np.lexsort uses the last key as primary: eval is primary and the original
    # index is the stable deterministic tie-breaker.
    original_indices = np.arange(values.shape[0])
    order = np.lexsort((original_indices, eval_values))
    selected = order[: int(n_survive)]
    cutoff = eval_values[selected[-1]]
    boundary = np.flatnonzero(np.isclose(eval_values, cutoff, rtol=0.0, atol=1e-12))

    return SurvivalResult(
        selected_indices=selected.astype(int, copy=True),
        eval_values=eval_values,
        q=q,
        p=p,
        nondominated_ratio=float(np.mean(q == 0.0)),
        unique_eval_ratio=float(np.unique(eval_values).size / values.shape[0]),
        boundary_tie_size=int(boundary.size),
    )


class PDDRSurvival(Survival):
    """pymoo adapter for direct PDDR-FF survival selection."""

    def __init__(self) -> None:
        # Feasibility filtering is handled by the current experiment protocol;
        # this adapter only performs objective-based survival selection.
        super().__init__(filter_infeasible=False)
        self.last_result: SurvivalResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """Select a pymoo Population using its stored objective matrix."""
        if n_survive is None:
            raise ValueError("n_survive is required for PDDR survival")
        result = select_pddr_survivors(pop.get("F"), n_survive)
        self.last_result = result
        return pop[result.selected_indices]


__all__ = ["PDDRSurvival", "SurvivalResult", "select_pddr_survivors"]
