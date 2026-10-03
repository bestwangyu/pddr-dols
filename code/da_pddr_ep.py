"""Degeneracy-aware PDDR-FF survival for binary decision vectors.

Only individuals tied at the PDDR-FF truncation boundary are re-ranked.  The
secondary rule greedily maximizes Hamming diversity, preserving the original
PDDR ordering everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.core.survival import Survival

from pddrff import pddr_ff


@dataclass(frozen=True)
class DASurvivalResult:
    """Selection result and diagnostics for one DA-PDDR survival operation."""

    selected_indices: np.ndarray
    eval_values: np.ndarray
    q: np.ndarray
    p: np.ndarray
    nondominated_ratio: float
    unique_eval_ratio: float
    boundary_tie_size: int
    boundary_slots: int
    compensation_activated: bool
    selected_mean_hamming: float


def _validate_inputs(
    objectives: np.ndarray, decisions: np.ndarray, n_survive: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate objective and binary decision matrices."""
    objective_values = np.asarray(objectives, dtype=float)
    decision_values = np.asarray(decisions)
    if objective_values.ndim != 2 or objective_values.shape[0] == 0:
        raise ValueError("objectives must be a non-empty 2-D array")
    if not np.isfinite(objective_values).all():
        raise ValueError("objectives must contain only finite values")
    if decision_values.ndim != 2 or decision_values.shape[0] != objective_values.shape[0]:
        raise ValueError("decisions must have one row per objective vector")
    if decision_values.shape[1] == 0 or not np.all(np.isin(decision_values, [0, 1])):
        raise ValueError("decisions must be a non-empty binary matrix")
    if not isinstance(n_survive, (int, np.integer)):
        raise ValueError("n_survive must be an integer")
    if not 1 <= int(n_survive) <= objective_values.shape[0]:
        raise ValueError("n_survive must be between 1 and the population size")
    return objective_values, decision_values.astype(bool, copy=False)


def _normalized_hamming(first: np.ndarray, second: np.ndarray) -> float:
    """Return normalized Hamming distance in the interval [0, 1]."""
    return float(np.mean(first != second))


def _mean_pairwise_hamming(decisions: np.ndarray) -> float:
    """Return mean normalized Hamming distance over all unordered pairs."""
    if decisions.shape[0] < 2:
        return 0.0
    distances = [
        _normalized_hamming(decisions[i], decisions[j])
        for i in range(decisions.shape[0])
        for j in range(i + 1, decisions.shape[0])
    ]
    return float(np.mean(distances))


def _normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """Normalize objective columns for distance comparisons."""
    low = objectives.min(axis=0)
    span = objectives.max(axis=0) - low
    span = np.where(span > 0.0, span, 1.0)
    return (objectives - low) / span


def _select_objective_diverse_boundary(
    objectives: np.ndarray,
    decisions: np.ndarray,
    candidates: np.ndarray,
    already_selected: list[int],
    slots: int,
) -> list[int]:
    """Select boundary candidates by max-min normalized objective distance."""
    normalized = _normalize_objectives(objectives)
    remaining = [int(index) for index in candidates]
    chosen: list[int] = []

    while len(chosen) < slots:
        references = already_selected + chosen
        scored: list[tuple[float, float, float, int]] = []
        for candidate in remaining:
            if references:
                primary = min(
                    float(np.linalg.norm(normalized[candidate] - normalized[index]))
                    for index in references
                )
            else:
                others = [index for index in remaining if index != candidate]
                primary = (
                    float(
                        np.mean(
                            [
                                np.linalg.norm(
                                    normalized[candidate] - normalized[index]
                                )
                                for index in others
                            ]
                        )
                    )
                    if others
                    else 0.0
                )

            objective_mean = float(
                np.mean(
                    [
                        np.linalg.norm(normalized[candidate] - normalized[index])
                        for index in candidates
                        if index != candidate
                    ]
                    or [0.0]
                )
            )
            hamming_mean = float(
                np.mean(
                    [
                        _normalized_hamming(decisions[candidate], decisions[index])
                        for index in candidates
                        if index != candidate
                    ]
                    or [0.0]
                )
            )
            # Objective distance is primary; Hamming distance only resolves
            # objective ties so the ablation remains target-space driven.
            scored.append((primary, objective_mean, hamming_mean, -candidate))

        best_position = max(range(len(scored)), key=scored.__getitem__)
        chosen.append(remaining.pop(best_position))
    return chosen


def select_objective_da_pddr_survivors(
    objectives: np.ndarray,
    decisions: np.ndarray,
    n_survive: int,
    *,
    boundary_tolerance: float = 0.05,
) -> DASurvivalResult:
    """Apply PDDR ranking with objective-space boundary compensation.

    The PDDR-FF order is preserved away from the truncation boundary.  Near
    the boundary, survivors are selected by max-min normalized objective
    distance, with Hamming distance used only as a deterministic tie-breaker.
    """
    values, binary = _validate_inputs(objectives, decisions, n_survive)
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be finite and non-negative")

    eval_values, q, p = pddr_ff(values)
    original_indices = np.arange(values.shape[0])
    order = np.lexsort((original_indices, eval_values))
    cutoff = float(eval_values[order[int(n_survive) - 1]])

    strictly_better = order[eval_values[order] < cutoff - boundary_tolerance]
    boundary = original_indices[
        (eval_values >= cutoff - boundary_tolerance)
        & (eval_values <= cutoff + boundary_tolerance)
    ]
    slots = int(n_survive) - len(strictly_better)
    if slots < 0 or len(boundary) < slots:
        raise RuntimeError("objective-space boundary construction is inconsistent")

    selected = [int(index) for index in strictly_better]
    compensation_activated = slots < len(boundary)
    if compensation_activated:
        selected.extend(
            _select_objective_diverse_boundary(
                values, binary, boundary, selected, slots
            )
        )
    else:
        selected.extend(int(index) for index in boundary[:slots])

    selected_array = np.asarray(selected, dtype=int)
    return DASurvivalResult(
        selected_indices=selected_array,
        eval_values=eval_values,
        q=q,
        p=p,
        nondominated_ratio=float(np.mean(q == 0.0)),
        unique_eval_ratio=float(np.unique(eval_values).size / values.shape[0]),
        boundary_tie_size=int(len(boundary)),
        boundary_slots=int(slots),
        compensation_activated=compensation_activated,
        selected_mean_hamming=_mean_pairwise_hamming(binary[selected_array]),
    )


def _select_diverse_boundary(
    decisions: np.ndarray,
    candidates: np.ndarray,
    already_selected: list[int],
    slots: int,
) -> list[int]:
    """Greedily select boundary candidates with max-min Hamming diversity."""
    remaining = [int(index) for index in candidates]
    chosen: list[int] = []

    while len(chosen) < slots:
        references = already_selected + chosen
        scored: list[tuple[float, float, int]] = []
        for candidate in remaining:
            # The primary score is distance to the closest selected survivor.
            # If no survivor exists yet, use mean distance to the full boundary.
            if references:
                primary = min(
                    _normalized_hamming(decisions[candidate], decisions[index])
                    for index in references
                )
            else:
                other_candidates = [index for index in remaining if index != candidate]
                primary = (
                    float(
                        np.mean(
                            [
                                _normalized_hamming(
                                    decisions[candidate], decisions[index]
                                )
                                for index in other_candidates
                            ]
                        )
                    )
                    if other_candidates
                    else 0.0
                )

            # Mean boundary distance breaks equal max-min scores.  The negative
            # index in max() makes the final tie deterministic toward low index.
            secondary = float(
                np.mean(
                    [
                        _normalized_hamming(decisions[candidate], decisions[index])
                        for index in candidates
                        if index != candidate
                    ]
                    or [0.0]
                )
            )
            scored.append((primary, secondary, -candidate))

        best_position = max(range(len(scored)), key=scored.__getitem__)
        chosen_index = remaining.pop(best_position)
        chosen.append(chosen_index)
    return chosen


def select_da_pddr_survivors(
    objectives: np.ndarray, decisions: np.ndarray, n_survive: int
) -> DASurvivalResult:
    """Apply PDDR-FF selection plus boundary-only Hamming compensation."""
    values, binary = _validate_inputs(objectives, decisions, n_survive)
    eval_values, q, p = pddr_ff(values)
    original_indices = np.arange(values.shape[0])
    order = np.lexsort((original_indices, eval_values))
    cutoff = eval_values[order[int(n_survive) - 1]]

    strictly_better = order[eval_values[order] < cutoff - 1e-12]
    boundary = original_indices[
        np.isclose(eval_values, cutoff, rtol=0.0, atol=1e-12)
    ]
    slots = int(n_survive) - len(strictly_better)
    selected = [int(index) for index in strictly_better]

    compensation_activated = slots < len(boundary)
    if compensation_activated:
        selected.extend(
            _select_diverse_boundary(binary, boundary, selected, slots)
        )
    else:
        selected.extend(int(index) for index in boundary[:slots])

    selected_array = np.asarray(selected, dtype=int)
    return DASurvivalResult(
        selected_indices=selected_array,
        eval_values=eval_values,
        q=q,
        p=p,
        nondominated_ratio=float(np.mean(q == 0.0)),
        unique_eval_ratio=float(np.unique(eval_values).size / values.shape[0]),
        boundary_tie_size=int(len(boundary)),
        boundary_slots=int(slots),
        compensation_activated=compensation_activated,
        selected_mean_hamming=_mean_pairwise_hamming(binary[selected_array]),
    )


class DAPDDRSurvival(Survival):
    """pymoo adapter for degeneracy-aware PDDR survival."""

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)
        self.last_result: DASurvivalResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            raise ValueError("n_survive is required for DA-PDDR survival")
        result = select_da_pddr_survivors(pop.get("F"), pop.get("X"), n_survive)
        self.last_result = result
        return pop[result.selected_indices]


class ObjectiveSpaceDASurvival(Survival):
    """PDDR survival with normalized objective-space diversity compensation."""

    def __init__(self, *, boundary_tolerance: float = 0.05) -> None:
        super().__init__(filter_infeasible=False)
        self.boundary_tolerance = float(boundary_tolerance)
        self.last_result: DASurvivalResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            raise ValueError("n_survive is required for objective-space DA-PDDR")
        result = select_objective_da_pddr_survivors(
            pop.get("F"),
            pop.get("X"),
            n_survive,
            boundary_tolerance=self.boundary_tolerance,
        )
        self.last_result = result
        return pop[result.selected_indices]


__all__ = [
    "DAPDDRSurvival",
    "DASurvivalResult",
    "ObjectiveSpaceDASurvival",
    "select_da_pddr_survivors",
    "select_objective_da_pddr_survivors",
]
