"""Rank-preserving PDDR survival for discrete objective-space degeneracy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.core.survival import Survival

from pddrff import dominance_matrix, pddr_ff


@dataclass(frozen=True)
class RankPDDRResult:
    """Selected survivors and diversity diagnostics for one environmental step."""

    selected_indices: np.ndarray
    pareto_ranks: np.ndarray
    pddr_values: np.ndarray
    unique_objective_count: int
    selected_unique_objective_count: int


def _validate_inputs(
    objectives: np.ndarray, decisions: np.ndarray, n_survive: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite objectives, aligned binary decisions and survivor count."""
    values = np.asarray(objectives, dtype=float)
    masks = np.asarray(decisions)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("objectives must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("objectives must contain only finite values")
    if masks.ndim != 2 or masks.shape[0] != values.shape[0]:
        raise ValueError("decisions must have one row per objective vector")
    if masks.shape[1] == 0 or not np.all(np.isin(masks, [0, 1])):
        raise ValueError("decisions must be a non-empty binary matrix")
    if not isinstance(n_survive, (int, np.integer)):
        raise ValueError("n_survive must be an integer")
    if not 1 <= int(n_survive) <= len(values):
        raise ValueError("n_survive must be between 1 and the population size")
    return values, masks.astype(bool, copy=False)


def _pareto_ranks(objectives: np.ndarray) -> np.ndarray:
    """Return non-dominated ranks using the same strict dominance relation as PDDR."""
    matrix = dominance_matrix(objectives)
    remaining = set(range(len(objectives)))
    ranks = np.full(len(objectives), -1, dtype=int)
    rank = 0
    while remaining:
        front = [
            index
            for index in sorted(remaining)
            if not any(matrix[other, index] for other in remaining if other != index)
        ]
        if not front:
            raise RuntimeError("Pareto sorting could not find a non-dominated front")
        ranks[front] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """Normalize each objective safely for Euclidean distance comparisons."""
    low = objectives.min(axis=0)
    span = objectives.max(axis=0) - low
    return (objectives - low) / np.where(span > 0.0, span, 1.0)


def rank_pddr_parent_keys(
    objectives: np.ndarray, decisions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rank, objective spacing, PDDR and Hamming keys for tournaments."""
    values, masks = _validate_inputs(objectives, decisions, len(objectives))
    ranks = _pareto_ranks(values)
    pddr_values, _, _ = pddr_ff(values)
    normalized = _normalize_objectives(values)
    objective_spacing = np.zeros(len(values), dtype=float)
    hamming_spacing = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        others = [other for other in range(len(values)) if other != index]
        if not others:
            continue
        objective_spacing[index] = min(
            float(np.linalg.norm(normalized[index] - normalized[other]))
            for other in others
        )
        hamming_spacing[index] = _mean_hamming(masks[index], others, masks)
    return ranks, objective_spacing, pddr_values, hamming_spacing


def hybrid_pddr_parent_keys(
    objectives: np.ndarray, decisions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return parent-selection keys with convergence before spacing.

    Unlike the rank-preserving exploratory variant, PDDR is the primary key
    inside a Pareto layer. Objective spacing is only a later tie-breaker.
    """
    ranks, objective_spacing, pddr_values, hamming_spacing = rank_pddr_parent_keys(
        objectives, decisions
    )
    return ranks, pddr_values, objective_spacing, hamming_spacing


def _crowding_distances(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Return normalized NSGA-II-style crowding within each Pareto layer."""
    distances = np.zeros(len(objectives), dtype=float)
    for rank in range(int(ranks.max()) + 1):
        front = np.flatnonzero(ranks == rank).tolist()
        if len(front) <= 2:
            distances[front] = np.inf
            continue
        for objective in range(objectives.shape[1]):
            order = sorted(
                front,
                key=lambda index: (float(objectives[index, objective]), index),
            )
            low = float(objectives[order[0], objective])
            high = float(objectives[order[-1], objective])
            if high <= low:
                continue
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            for position in range(1, len(order) - 1):
                index = order[position]
                if np.isinf(distances[index]):
                    continue
                previous_value = float(objectives[order[position - 1], objective])
                next_value = float(objectives[order[position + 1], objective])
                distances[index] += (next_value - previous_value) / (high - low)
    return distances


def extreme_pddr_parent_keys(
    objectives: np.ndarray, decisions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rank, crowding, PDDR and Hamming keys for parent tournaments."""
    values, masks = _validate_inputs(objectives, decisions, len(objectives))
    ranks = _pareto_ranks(values)
    crowding = _crowding_distances(values, ranks)
    pddr_values, _, _ = pddr_ff(values)
    hamming_spacing = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        others = [other for other in range(len(values)) if other != index]
        hamming_spacing[index] = _mean_hamming(masks[index], others, masks)
    return ranks, crowding, pddr_values, hamming_spacing


def _mean_hamming(first: np.ndarray, others: list[int], decisions: np.ndarray) -> float:
    """Return mean normalized Hamming distance to indexed decisions."""
    if not others:
        return 0.0
    return float(np.mean([np.mean(first != decisions[index]) for index in others]))


def _representatives(
    candidates: list[int],
    objectives: np.ndarray,
    decisions: np.ndarray,
    selected: list[int],
    pddr_values: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Choose one decision-diverse representative for every exact objective vector."""
    groups: dict[tuple[float, ...], list[int]] = {}
    for index in candidates:
        groups.setdefault(tuple(objectives[index].tolist()), []).append(index)

    representatives: list[int] = []
    duplicates: list[int] = []
    for group in groups.values():
        representative = max(
            group,
            key=lambda index: (
                _mean_hamming(decisions[index], selected, decisions),
                -float(pddr_values[index]),
                -index,
            ),
        )
        representatives.append(representative)
        duplicates.extend(index for index in group if index != representative)
    return representatives, duplicates


def _select_spread(
    candidates: list[int],
    slots: int,
    normalized_objectives: np.ndarray,
    decisions: np.ndarray,
    selected: list[int],
    pddr_values: np.ndarray,
) -> list[int]:
    """Use global objective max-min distance, then PDDR and Hamming tie-breaks."""
    remaining = list(candidates)
    chosen: list[int] = []
    while len(chosen) < slots:
        references = selected + chosen
        scores: list[tuple[float, float, float, float, int]] = []
        for candidate in remaining:
            others = [index for index in remaining if index != candidate]
            if references:
                primary = min(
                    float(
                        np.linalg.norm(
                            normalized_objectives[candidate]
                            - normalized_objectives[index]
                        )
                    )
                    for index in references
                )
                hamming = _mean_hamming(decisions[candidate], references, decisions)
            else:
                primary = float(
                    np.mean(
                        [
                            np.linalg.norm(
                                normalized_objectives[candidate]
                                - normalized_objectives[index]
                            )
                            for index in others
                        ]
                        or [0.0]
                    )
                )
                hamming = _mean_hamming(decisions[candidate], others, decisions)
            secondary = float(
                np.mean(
                    [
                        np.linalg.norm(
                            normalized_objectives[candidate]
                            - normalized_objectives[index]
                        )
                        for index in others
                    ]
                    or [0.0]
                )
            )
            scores.append(
                (primary, secondary, -float(pddr_values[candidate]), hamming, -candidate)
            )
        position = max(range(len(scores)), key=scores.__getitem__)
        chosen.append(remaining.pop(position))
    return chosen


def select_rank_pddr_survivors(
    objectives: np.ndarray, decisions: np.ndarray, n_survive: int
) -> RankPDDRResult:
    """Preserve Pareto layers and unique objectives before applying PDDR ties.

    Exact duplicate objective vectors are represented once while capacity remains.
    When a Pareto layer overflows, selection covers the complete normalized
    objective space of that layer.  PDDR-FF and decision Hamming distance only
    resolve equal diversity choices.
    """
    values, masks = _validate_inputs(objectives, decisions, n_survive)
    ranks = _pareto_ranks(values)
    pddr_values, _, _ = pddr_ff(values)
    normalized = _normalize_objectives(values)
    selected: list[int] = []
    deferred_duplicates: list[int] = []

    for rank in range(int(ranks.max()) + 1):
        front = np.flatnonzero(ranks == rank).tolist()
        representatives, duplicates = _representatives(
            front, values, masks, selected, pddr_values
        )
        slots = int(n_survive) - len(selected)
        if len(representatives) >= slots:
            selected.extend(
                _select_spread(
                    representatives,
                    slots,
                    normalized,
                    masks,
                    selected,
                    pddr_values,
                )
            )
            break
        selected.extend(
            _select_spread(
                representatives,
                len(representatives),
                normalized,
                masks,
                selected,
                pddr_values,
            )
        )
        deferred_duplicates.extend(duplicates)

    if len(selected) < n_survive:
        selected.extend(
            _select_spread(
                deferred_duplicates,
                int(n_survive) - len(selected),
                normalized,
                masks,
                selected,
                pddr_values,
            )
        )

    selected_array = np.asarray(selected, dtype=int)
    return RankPDDRResult(
        selected_indices=selected_array,
        pareto_ranks=ranks,
        pddr_values=pddr_values,
        unique_objective_count=int(len(np.unique(values, axis=0))),
        selected_unique_objective_count=int(
            len(np.unique(values[selected_array], axis=0))
        ),
    )


def _select_convergence_first(
    candidates: list[int], slots: int, pddr_values: np.ndarray
) -> list[int]:
    """Select the best PDDR candidates with deterministic index ties."""
    order = sorted(candidates, key=lambda index: (float(pddr_values[index]), index))
    return order[:slots]


def _select_hybrid_truncated(
    candidates: list[int],
    slots: int,
    normalized_objectives: np.ndarray,
    decisions: np.ndarray,
    selected: list[int],
    pddr_values: np.ndarray,
    diversity_fraction: float,
) -> list[int]:
    """Use PDDR for most slots and restricted spacing for the remainder."""
    if slots <= 1:
        return _select_convergence_first(candidates, slots, pddr_values)
    n_diverse = min(slots, max(1, int(round(slots * diversity_fraction))))
    n_convergence = slots - n_diverse
    convergence = _select_convergence_first(candidates, n_convergence, pddr_values)
    remaining = [index for index in candidates if index not in convergence]
    if not remaining or n_diverse == 0:
        return convergence

    # Restrict diversity to the best PDDR candidates so spacing cannot pull in
    # a poor-convergence solution merely because it is geometrically distant.
    pool_size = min(len(remaining), max(n_diverse, 2 * n_diverse))
    restricted_pool = sorted(
        remaining, key=lambda index: (float(pddr_values[index]), index)
    )[:pool_size]
    spread = _select_spread(
        restricted_pool,
        min(n_diverse, len(restricted_pool)),
        normalized_objectives,
        decisions,
        selected + convergence,
        pddr_values,
    )
    return convergence + spread


def select_hybrid_pddr_survivors(
    objectives: np.ndarray,
    decisions: np.ndarray,
    n_survive: int,
    *,
    diversity_fraction: float = 0.2,
) -> RankPDDRResult:
    """Select with convergence priority and limited objective-space spread.

    Complete Pareto layers are retained while capacity permits. Only the
    truncated layer receives diversity allocation; at most
    ``diversity_fraction`` of its slots use restricted max-min spacing. Exact
    duplicate objective vectors are represented once before duplicate filling.
    """
    if not 0.0 <= float(diversity_fraction) <= 1.0:
        raise ValueError("diversity_fraction must be between 0 and 1")
    values, masks = _validate_inputs(objectives, decisions, n_survive)
    ranks = _pareto_ranks(values)
    pddr_values, _, _ = pddr_ff(values)
    normalized = _normalize_objectives(values)
    selected: list[int] = []
    deferred_duplicates: list[int] = []

    for rank in range(int(ranks.max()) + 1):
        front = np.flatnonzero(ranks == rank).tolist()
        representatives, duplicates = _representatives(
            front, values, masks, selected, pddr_values
        )
        slots = int(n_survive) - len(selected)
        if slots <= 0:
            break
        if len(representatives) > slots:
            selected.extend(
                _select_hybrid_truncated(
                    representatives,
                    slots,
                    normalized,
                    masks,
                    selected,
                    pddr_values,
                    float(diversity_fraction),
                )
            )
            break
        selected.extend(_select_convergence_first(representatives, len(representatives), pddr_values))
        deferred_duplicates.extend(duplicates)

    if len(selected) < n_survive:
        selected.extend(
            _select_convergence_first(
                deferred_duplicates, int(n_survive) - len(selected), pddr_values
            )
        )

    selected_array = np.asarray(selected, dtype=int)
    return RankPDDRResult(
        selected_indices=selected_array,
        pareto_ranks=ranks,
        pddr_values=pddr_values,
        unique_objective_count=int(len(np.unique(values, axis=0))),
        selected_unique_objective_count=int(
            len(np.unique(values[selected_array], axis=0))
        ),
    )


def _objective_minimum_indices(
    candidates: list[int], objectives: np.ndarray, pddr_values: np.ndarray
) -> list[int]:
    """Choose one deterministic representative for each objective minimum."""
    extrema: list[int] = []
    for objective in range(objectives.shape[1]):
        minimum = min(float(objectives[index, objective]) for index in candidates)
        tied = [
            index
            for index in candidates
            if np.isclose(
                float(objectives[index, objective]), minimum, rtol=0.0, atol=1e-12
            )
        ]
        representative = min(
            tied, key=lambda index: (float(pddr_values[index]), index)
        )
        if representative not in extrema:
            extrema.append(representative)
    return extrema


def _select_extreme_crowding(
    candidates: list[int],
    slots: int,
    objectives: np.ndarray,
    decisions: np.ndarray,
    selected: list[int],
    ranks: np.ndarray,
    pddr_values: np.ndarray,
) -> list[int]:
    """Force objective minima, then use crowding with PDDR as tie-breaker."""
    crowding = _crowding_distances(objectives, ranks)
    extrema = _objective_minimum_indices(candidates, objectives, pddr_values)
    chosen = sorted(
        extrema, key=lambda index: (float(pddr_values[index]), index)
    )[:slots]
    remaining = [index for index in candidates if index not in chosen]
    while len(chosen) < slots:
        references = selected + chosen
        candidate = max(
            remaining,
            key=lambda index: (
                float(crowding[index]),
                -float(pddr_values[index]),
                _mean_hamming(decisions[index], references, decisions),
                -index,
            ),
        )
        chosen.append(candidate)
        remaining.remove(candidate)
    return chosen


def select_extreme_pddr_survivors(
    objectives: np.ndarray, decisions: np.ndarray, n_survive: int
) -> RankPDDRResult:
    """Preserve extremes and crowding, using PDDR only as a secondary key."""
    values, masks = _validate_inputs(objectives, decisions, n_survive)
    ranks = _pareto_ranks(values)
    pddr_values, _, _ = pddr_ff(values)
    selected: list[int] = []
    deferred_duplicates: list[int] = []

    for rank in range(int(ranks.max()) + 1):
        front = np.flatnonzero(ranks == rank).tolist()
        representatives, duplicates = _representatives(
            front, values, masks, selected, pddr_values
        )
        slots = int(n_survive) - len(selected)
        if slots <= 0:
            break
        if len(representatives) > slots:
            selected.extend(
                _select_extreme_crowding(
                    representatives,
                    slots,
                    values,
                    masks,
                    selected,
                    ranks,
                    pddr_values,
                )
            )
            break
        selected.extend(representatives)
        deferred_duplicates.extend(duplicates)

    if len(selected) < n_survive:
        crowding = _crowding_distances(values, ranks)
        deferred_duplicates.sort(
            key=lambda index: (
                int(ranks[index]),
                -float(crowding[index]),
                float(pddr_values[index]),
                index,
            )
        )
        selected.extend(deferred_duplicates[: int(n_survive) - len(selected)])

    selected_array = np.asarray(selected, dtype=int)
    return RankPDDRResult(
        selected_indices=selected_array,
        pareto_ranks=ranks,
        pddr_values=pddr_values,
        unique_objective_count=int(len(np.unique(values, axis=0))),
        selected_unique_objective_count=int(
            len(np.unique(values[selected_array], axis=0))
        ),
    )


class RankPDDRSurvival(Survival):
    """pymoo adapter for rank-preserving PDDR environmental selection."""

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)
        self.last_result: RankPDDRResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            raise ValueError("n_survive is required for rank-preserving PDDR")
        result = select_rank_pddr_survivors(pop.get("F"), pop.get("X"), n_survive)
        self.last_result = result
        return pop[result.selected_indices]


class HybridPDDRSurvival(Survival):
    """Convergence-first PDDR survival with restricted diversity allocation."""

    def __init__(self, diversity_fraction: float = 0.2) -> None:
        super().__init__(filter_infeasible=False)
        self.diversity_fraction = float(diversity_fraction)
        self.last_result: RankPDDRResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            raise ValueError("n_survive is required for hybrid PDDR")
        result = select_hybrid_pddr_survivors(
            pop.get("F"),
            pop.get("X"),
            n_survive,
            diversity_fraction=self.diversity_fraction,
        )
        self.last_result = result
        return pop[result.selected_indices]


class ExtremePDDRSurvival(Survival):
    """Extreme-preserving crowding survival with PDDR tie-breaking."""

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)
        self.last_result: RankPDDRResult | None = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            raise ValueError("n_survive is required for extreme PDDR")
        result = select_extreme_pddr_survivors(
            pop.get("F"), pop.get("X"), n_survive
        )
        self.last_result = result
        return pop[result.selected_indices]


__all__ = [
    "RankPDDRResult",
    "RankPDDRSurvival",
    "HybridPDDRSurvival",
    "ExtremePDDRSurvival",
    "extreme_pddr_parent_keys",
    "hybrid_pddr_parent_keys",
    "rank_pddr_parent_keys",
    "select_hybrid_pddr_survivors",
    "select_extreme_pddr_survivors",
    "select_rank_pddr_survivors",
]
