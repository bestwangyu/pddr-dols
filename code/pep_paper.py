"""Paper-faithful implementation of Pareto Ensemble Pruning (PEP).

This module follows Algorithm 1 and the VDS procedure described by Qian,
Yu, and Zhou, AAAI 2015.  It intentionally keeps PEP separate from the
project's existing ``pep`` objective-modeling proxy: the reproduced method
optimizes exactly two objectives, validation error and selected ensemble
size.  Predictions are supplied by the shared :class:`EnsembleEvaluator`,
so no classifier is retrained during subset search.

The original paper assigns infinite validation error to the empty subset.
``empty_subset_policy='paper_infinite_error'`` preserves that definition;
``'project_repair'`` is available only for controlled compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np

from ensemble_pruning import EnsembleEvaluator


def pep_dominates(left: np.ndarray, right: np.ndarray, *, strict: bool = True) -> bool:
    """Return whether ``left`` dominates ``right`` for minimization.

    With ``strict=True`` this is strict Pareto dominance (no worse in every
    objective and strictly better in at least one); with ``strict=False`` it
    is weak dominance (no worse in every objective).
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != (2,) or b.shape != (2,):
        raise ValueError("PEP objective vectors must have shape (2,)")
    if strict:
        return bool(np.all(a <= b) and np.any(a < b))
    return bool(np.all(a <= b))


def _same_mask(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right))


def pareto_update(
    archive_masks: Iterable[np.ndarray],
    archive_objectives: Iterable[np.ndarray],
    candidate_mask: np.ndarray,
    candidate_objective: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    """Insert one candidate into a bi-objective Pareto archive.

    A candidate is rejected if it is strictly dominated or duplicates an
    existing mask.  Existing points weakly dominated by the candidate are
    removed, matching the archive update in the PEP algorithm.
    """
    masks = [np.asarray(mask, dtype=bool).copy() for mask in archive_masks]
    objectives = [np.asarray(value, dtype=float).copy() for value in archive_objectives]
    candidate_mask = np.asarray(candidate_mask, dtype=bool).copy()
    candidate_objective = np.asarray(candidate_objective, dtype=float).copy()
    if candidate_objective.shape != (2,):
        raise ValueError("candidate_objective must have shape (2,)")
    for mask, objective in zip(masks, objectives):
        if _same_mask(mask, candidate_mask) or pep_dominates(objective, candidate_objective):
            return masks, objectives, False
    kept = [
        (mask, objective)
        for mask, objective in zip(masks, objectives)
        if not pep_dominates(candidate_objective, objective, strict=False)
    ]
    kept_masks = [mask for mask, _ in kept]
    kept_objectives = [objective for _, objective in kept]
    kept_masks.append(candidate_mask)
    kept_objectives.append(candidate_objective)
    return kept_masks, kept_objectives, True


def mutate_bitwise(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip each bit independently with probability ``1/n``."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mask must be a non-empty 1-D vector")
    flips = rng.random(values.size) < (1.0 / values.size)
    return np.logical_xor(values, flips)


@dataclass
class PEPSearchStats:
    """Search accounting retained for reproducibility and fair comparisons."""

    iterations: int = 0
    objective_evaluations: int = 0
    vds_evaluations: int = 0
    archive_updates: int = 0
    empty_subset_evaluations: int = 0
    empty_subset_policy: str = "paper_infinite_error"
    vds_enabled: bool = True

    @property
    def evaluation_count(self) -> int:
        return self.objective_evaluations


@dataclass
class PEPResult:
    """Small result object compatible with the project's batch serializer."""

    X: np.ndarray
    F: np.ndarray
    algorithm: PEPSearchStats = field(default_factory=PEPSearchStats)
    final_selection: str = "validation_error"


class PEPPaperProblem:
    """Bi-objective objective wrapper used by the standalone PEP driver."""

    def __init__(
        self,
        evaluator: EnsembleEvaluator,
        *,
        empty_subset_policy: str = "paper_infinite_error",
        stats: PEPSearchStats | None = None,
    ) -> None:
        if empty_subset_policy not in {"paper_infinite_error", "project_repair"}:
            raise ValueError(
                "empty_subset_policy must be 'paper_infinite_error' or 'project_repair'"
            )
        self.evaluator = evaluator
        self.empty_subset_policy = empty_subset_policy
        self.stats = stats if stats is not None else PEPSearchStats(
            empty_subset_policy=empty_subset_policy
        )
        self.stats.empty_subset_policy = empty_subset_policy
        self._objective_cache: dict[tuple[int, ...], np.ndarray] = {}

    @property
    def n_var(self) -> int:
        return self.evaluator.n_classifiers

    def evaluate(self, mask: np.ndarray) -> np.ndarray:
        """Evaluate ``(validation_error, selected_count/n)`` exactly."""
        values = np.asarray(mask, dtype=bool)
        if values.ndim != 1 or values.size != self.n_var:
            raise ValueError("mask has the wrong number of classifiers")
        # Count objective calls, including calls served by the prediction
        # cache.  This is the quantity needed for paper-style budget reports.
        self.stats.objective_evaluations += 1
        key = tuple(int(value) for value in values)
        cached = self._objective_cache.get(key)
        if cached is not None:
            return cached.copy()
        if not values.any():
            self.stats.empty_subset_evaluations += 1
            if self.empty_subset_policy == "paper_infinite_error":
                objective = np.array([np.inf, 0.0], dtype=float)
                self._objective_cache[key] = objective
                return objective.copy()
            result = self.evaluator.evaluate(values)
            objective = np.array([result.validation_error, 1.0 / self.n_var], dtype=float)
            self._objective_cache[key] = objective
            return objective.copy()
        result = self.evaluator.evaluate(values)
        objective = np.array(
            [result.validation_error, len(result.selected_indices) / self.n_var],
            dtype=float,
        )
        self._objective_cache[key] = objective
        return objective.copy()


def generate_vds_neighbors(
    problem: PEPPaperProblem,
    start_mask: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """Run PEP's Variable-Depth Search and return every generated solution.

    At each depth the best validation-error neighbor is selected, while a bit
    position can be flipped at most once in the current VDS trajectory.
    """
    current = np.asarray(start_mask, dtype=bool).copy()
    if current.ndim != 1 or current.size != problem.n_var:
        raise ValueError("start_mask has the wrong number of classifiers")
    used_positions: set[int] = set()
    generated: list[np.ndarray] = []
    evaluations = 0
    while len(used_positions) < problem.n_var:
        candidates: list[tuple[float, int, np.ndarray]] = []
        for index in range(problem.n_var):
            if index in used_positions:
                continue
            candidate = current.copy()
            candidate[index] = ~candidate[index]
            objective = problem.evaluate(candidate)
            evaluations += 1
            candidates.append((float(objective[0]), index, candidate))
        if not candidates:
            break
        _, chosen_index, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        used_positions.add(chosen_index)
        current = chosen
        generated.append(current.copy())
    problem.stats.vds_evaluations += evaluations
    return generated, evaluations


def _generate_vds_candidates(
    problem: PEPPaperProblem, start_mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Internal VDS variant retaining objective values for archive updates."""
    current = np.asarray(start_mask, dtype=bool).copy()
    used_positions: set[int] = set()
    generated: list[tuple[np.ndarray, np.ndarray]] = []
    while len(used_positions) < problem.n_var:
        candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for index in range(problem.n_var):
            if index in used_positions:
                continue
            candidate = current.copy()
            candidate[index] = ~candidate[index]
            objective = problem.evaluate(candidate)
            candidates.append((float(objective[0]), index, candidate, objective))
        if not candidates:
            break
        _, chosen_index, chosen, objective = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        used_positions.add(chosen_index)
        current = chosen
        generated.append((current.copy(), objective.copy()))
    problem.stats.vds_evaluations += sum(
        problem.n_var - depth for depth in range(len(generated))
    )
    return generated


def default_pep_iterations(n_classifiers: int) -> int:
    """Return the paper's ``ceil(n^2 log n)`` iteration count."""
    if n_classifiers < 2:
        raise ValueError("PEP requires at least two classifiers")
    return max(1, int(math.ceil(n_classifiers**2 * math.log(n_classifiers))))


def select_final_solution(
    masks: Iterable[np.ndarray], objectives: Iterable[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Select the final PEP solution by minimum validation error (paper eval)."""
    pairs = [
        (np.asarray(mask, dtype=bool), np.asarray(objective, dtype=float))
        for mask, objective in zip(masks, objectives)
        if np.asarray(mask, dtype=bool).any() and np.isfinite(objective[0])
    ]
    if not pairs:
        raise ValueError("PEP archive contains no non-empty finite solution")
    return min(pairs, key=lambda pair: (float(pair[1][0]), float(pair[1][1])))


def run_pep_paper(
    evaluator: EnsembleEvaluator,
    *,
    iterations: int | None = None,
    seed: int = 20260803,
    vds_enabled: bool = True,
    empty_subset_policy: str = "paper_infinite_error",
) -> PEPResult:
    """Run the paper PEP search and return its two-objective Pareto archive."""
    if iterations is not None and (not isinstance(iterations, int) or iterations < 1):
        raise ValueError("iterations must be a positive integer")
    rng = np.random.default_rng(seed)
    stats = PEPSearchStats(
        empty_subset_policy=empty_subset_policy, vds_enabled=bool(vds_enabled)
    )
    problem = PEPPaperProblem(
        evaluator, empty_subset_policy=empty_subset_policy, stats=stats
    )
    initial = rng.integers(0, 2, size=problem.n_var, dtype=np.int8).astype(bool)
    archive_masks: list[np.ndarray] = []
    archive_objectives: list[np.ndarray] = []
    initial_objective = problem.evaluate(initial)
    archive_masks, archive_objectives, _ = pareto_update(
        archive_masks, archive_objectives, initial, initial_objective
    )
    n_iterations = default_pep_iterations(problem.n_var) if iterations is None else iterations
    for _ in range(n_iterations):
        stats.iterations += 1
        parent_index = int(rng.integers(len(archive_masks)))
        candidate = mutate_bitwise(archive_masks[parent_index], rng)
        candidate_objective = problem.evaluate(candidate)
        archive_masks, archive_objectives, inserted = pareto_update(
            archive_masks, archive_objectives, candidate, candidate_objective
        )
        if inserted:
            stats.archive_updates += 1
        if vds_enabled:
            generated = _generate_vds_candidates(problem, candidate)
            for vds_mask, vds_objective in generated:
                archive_masks, archive_objectives, inserted = pareto_update(
                    archive_masks, archive_objectives, vds_mask, vds_objective
                )
                if inserted:
                    stats.archive_updates += 1
    order = sorted(
        range(len(archive_masks)),
        key=lambda index: (float(archive_objectives[index][0]), float(archive_objectives[index][1]), index),
    )
    # The paper gives the empty subset infinite error.  It can transiently be
    # present when a very short smoke run has not yet discovered a non-empty
    # point, but it is never a valid reported ensemble.  Exclude it from the
    # returned front (and provide a deterministic singleton fallback).
    finite_order = [
        index
        for index in order
        if archive_masks[index].any() and np.isfinite(archive_objectives[index][0])
    ]
    if not finite_order:
        fallback = np.zeros(problem.n_var, dtype=bool)
        fallback[0] = True
        fallback_objective = problem.evaluate(fallback)
        finite_order = [len(archive_masks)]
        archive_masks.append(fallback)
        archive_objectives.append(fallback_objective)
    masks = np.asarray([archive_masks[index] for index in finite_order], dtype=bool)
    objectives = np.asarray([archive_objectives[index] for index in finite_order], dtype=float)
    return PEPResult(X=masks, F=objectives, algorithm=stats)


__all__ = [
    "PEPPaperProblem",
    "PEPResult",
    "PEPSearchStats",
    "default_pep_iterations",
    "generate_vds_neighbors",
    "mutate_bitwise",
    "pareto_update",
    "pep_dominates",
    "run_pep_paper",
    "select_final_solution",
]
