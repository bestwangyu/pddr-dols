"""PDDR-DOLS structured neighborhood search for ensemble pruning.

PDDR is used only to detect a degenerate or stagnant search state. Pareto rank
and crowding retain responsibility for parent and environmental selection.
When triggered, part of the ordinary binary-DE offspring budget is replaced by
add, delete and swap neighbors of diverse non-dominated ensemble masks.
"""

from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
from pymoo.core.population import Population
from pymoo.optimize import minimize

from ensemble_problem import EnsemblePruningProblem
from ensemble_pruning import EnsembleEvaluator
from fair_baselines import BinaryNSGA2
from metrics import nondominated_front
from pddrff import pddr_ff


# Public display name used by the manuscript and new result records.  The
# historical internal id remains unchanged so archived JSON files stay usable.
PDDR_DOLS_NAME = "PDDR-DOLS"
PDDR_DOLS_INTERNAL_ID = "pddr_local_search_delete_only"


def _front_signature(objectives: np.ndarray) -> str:
    """Return a stable signature for the unique current non-dominated front."""
    front = np.ascontiguousarray(np.round(nondominated_front(objectives), 12))
    return hashlib.sha256(front.tobytes()).hexdigest()


def _ordered_seed_indices(objectives: np.ndarray) -> list[int]:
    """Cover objective minima first, then the remaining non-dominated masks."""
    values = np.asarray(objectives, dtype=float)
    _, q, _ = pddr_ff(values)
    nondominated = np.flatnonzero(q == 0.0).tolist()
    if not nondominated:
        raise RuntimeError("a finite population must contain a non-dominated member")

    ordered: list[int] = []
    for objective in range(values.shape[1]):
        index = min(
            nondominated,
            key=lambda candidate: (float(values[candidate, objective]), candidate),
        )
        if index not in ordered:
            ordered.append(index)

    remaining = [index for index in nondominated if index not in ordered]
    if remaining:
        permutation = np.random.permutation(remaining).tolist()
        ordered.extend(int(index) for index in permutation)
    return ordered


def generate_structured_neighbors(
    decisions: np.ndarray,
    objectives: np.ndarray,
    n_neighbors: int,
    allowed_operations: tuple[str, ...] = ("add", "delete", "swap"),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Generate unique add/delete/swap neighbors of non-dominated masks.

    No candidate is evaluated inside this function. Returned masks replace the
    same number of ordinary offspring, preserving the optimization evaluation
    budget used by the fair NSGA-II baseline.
    """
    masks = np.asarray(decisions, dtype=bool)
    values = np.asarray(objectives, dtype=float)
    if masks.ndim != 2 or values.ndim != 2 or len(masks) != len(values):
        raise ValueError("decisions and objectives must be aligned 2-D arrays")
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 0:
        raise ValueError("n_neighbors must be a non-negative integer")
    allowed = tuple(dict.fromkeys(allowed_operations))
    if not allowed or not set(allowed).issubset({"add", "delete", "swap"}):
        raise ValueError(
            "allowed_operations must contain one or more of add, delete and swap"
        )
    if n_neighbors == 0:
        return np.empty((0, masks.shape[1]), dtype=bool), ()

    existing = {tuple(row.tolist()) for row in masks}
    buckets: dict[str, list[np.ndarray]] = {"add": [], "delete": [], "swap": []}
    seen = set(existing)
    for seed_index in _ordered_seed_indices(values):
        seed = masks[seed_index]
        selected = np.flatnonzero(seed).tolist()
        unselected = np.flatnonzero(~seed).tolist()

        for add_index in np.random.permutation(unselected).tolist():
            if "add" not in allowed:
                break
            candidate = seed.copy()
            candidate[int(add_index)] = True
            key = tuple(candidate.tolist())
            if key not in seen:
                buckets["add"].append(candidate)
                seen.add(key)

        if len(selected) > 1 and "delete" in allowed:
            for delete_index in np.random.permutation(selected).tolist():
                candidate = seed.copy()
                candidate[int(delete_index)] = False
                key = tuple(candidate.tolist())
                if key not in seen:
                    buckets["delete"].append(candidate)
                    seen.add(key)

        pairs = (
            [(remove, add) for remove in selected for add in unselected]
            if "swap" in allowed
            else []
        )
        if pairs:
            for position in np.random.permutation(len(pairs)).tolist():
                remove_index, add_index = pairs[int(position)]
                candidate = seed.copy()
                candidate[int(remove_index)] = False
                candidate[int(add_index)] = True
                key = tuple(candidate.tolist())
                if key not in seen:
                    buckets["swap"].append(candidate)
                    seen.add(key)

    chosen: list[np.ndarray] = []
    operations: list[str] = []
    operation_order = tuple(
        operation for operation in ("add", "delete", "swap") if operation in allowed
    )
    while len(chosen) < n_neighbors and any(buckets.values()):
        for operation in operation_order:
            if buckets[operation] and len(chosen) < n_neighbors:
                chosen.append(buckets[operation].pop(0))
                operations.append(operation)

    if not chosen:
        return np.empty((0, masks.shape[1]), dtype=bool), ()
    return np.asarray(chosen, dtype=bool), tuple(operations)


class PDDRLocalSearchNSGA2(BinaryNSGA2):
    """NSGA-II with budget-neutral PDDR-DOLS subset neighborhood search."""

    def __init__(
        self,
        *,
        pop_size: int,
        local_fraction: float = 0.5,
        pddr_unique_threshold: float = 0.35,
        objective_unique_threshold: float = 0.5,
        stagnation_patience: int = 3,
        trigger_mode: str = "pddr",
        allowed_operations: tuple[str, ...] = ("add", "delete", "swap"),
    ) -> None:
        if not 0.0 < local_fraction <= 1.0:
            raise ValueError("local_fraction must be in (0, 1]")
        if not 0.0 <= pddr_unique_threshold <= 1.0:
            raise ValueError("pddr_unique_threshold must be in [0, 1]")
        if not 0.0 <= objective_unique_threshold <= 1.0:
            raise ValueError("objective_unique_threshold must be in [0, 1]")
        if not isinstance(stagnation_patience, int) or stagnation_patience < 1:
            raise ValueError("stagnation_patience must be a positive integer")
        if trigger_mode not in {"pddr", "always"}:
            raise ValueError("trigger_mode must be 'pddr' or 'always'")
        allowed_operations = tuple(dict.fromkeys(allowed_operations))
        if not allowed_operations or not set(allowed_operations).issubset(
            {"add", "delete", "swap"}
        ):
            raise ValueError(
                "allowed_operations must contain one or more of add, delete and swap"
            )
        super().__init__(pop_size=pop_size, variation="de")
        self.local_fraction = float(local_fraction)
        self.pddr_unique_threshold = float(pddr_unique_threshold)
        self.objective_unique_threshold = float(objective_unique_threshold)
        self.stagnation_patience = int(stagnation_patience)
        self.trigger_mode = trigger_mode
        self.allowed_operations = allowed_operations
        self._previous_signature: str | None = None
        self._stagnant_generations = 0
        self.search_diagnostics = {
            "trigger_mode": self.trigger_mode,
            "allowed_operations": list(self.allowed_operations),
            "trigger_count": 0,
            "local_candidate_count": 0,
            "operation_counts": {"add": 0, "delete": 0, "swap": 0},
            "reason_counts": {
                "pddr_degeneracy": 0,
                "objective_degeneracy": 0,
                "stagnation": 0,
                "always_on": 0,
            },
            "events": [],
        }

    def _search_state(self) -> dict:
        """Measure PDDR degeneracy, objective duplication and front stagnation."""
        objectives = np.asarray(self.pop.get("F"), dtype=float)
        eval_values, _, _ = pddr_ff(objectives)
        pddr_unique_ratio = float(
            len(np.unique(np.round(eval_values, 12))) / len(objectives)
        )
        objective_unique_ratio = float(
            len(np.unique(np.round(objectives, 12), axis=0)) / len(objectives)
        )
        signature = _front_signature(objectives)
        if signature == self._previous_signature:
            self._stagnant_generations += 1
        else:
            self._stagnant_generations = 0
        self._previous_signature = signature

        reasons = []
        if pddr_unique_ratio <= self.pddr_unique_threshold:
            reasons.append("pddr_degeneracy")
        if objective_unique_ratio <= self.objective_unique_threshold:
            reasons.append("objective_degeneracy")
        if self._stagnant_generations >= self.stagnation_patience:
            reasons.append("stagnation")
        return {
            "pddr_unique_ratio": pddr_unique_ratio,
            "objective_unique_ratio": objective_unique_ratio,
            "stagnant_generations": self._stagnant_generations,
            "reasons": reasons,
        }

    def _infill(self):
        """Replace a fixed share of ordinary offspring under the chosen trigger."""
        ordinary = super()._infill()
        if ordinary is None or len(ordinary) == 0:
            return ordinary
        state = self._search_state()
        reasons = state["reasons"] if self.trigger_mode == "pddr" else ["always_on"]
        if not reasons:
            return ordinary

        requested = max(1, int(round(self.n_offsprings * self.local_fraction)))
        neighbors, operations = generate_structured_neighbors(
            self.pop.get("X"),
            self.pop.get("F"),
            requested,
            allowed_operations=self.allowed_operations,
        )
        injected = len(neighbors)
        if injected == 0:
            return ordinary
        local_population = Population.new("X", neighbors)
        combined = Population.merge(ordinary[: len(ordinary) - injected], local_population)

        self.search_diagnostics["trigger_count"] += 1
        self.search_diagnostics["local_candidate_count"] += injected
        operation_counts = Counter(operations)
        for operation, count in operation_counts.items():
            self.search_diagnostics["operation_counts"][operation] += int(count)
        for reason in reasons:
            self.search_diagnostics["reason_counts"][reason] += 1
        self.search_diagnostics["events"].append(
            {
                "generation": int(self.n_gen),
                "pddr_unique_ratio": state["pddr_unique_ratio"],
                "objective_unique_ratio": state["objective_unique_ratio"],
                "stagnant_generations": state["stagnant_generations"],
                "reasons": list(reasons),
                "trigger_mode": self.trigger_mode,
                "requested_candidates": requested,
                "injected_candidates": injected,
                "operation_counts": {
                    operation: int(operation_counts.get(operation, 0))
                    for operation in ("add", "delete", "swap")
                },
            }
        )
        return combined


def run_pddr_local_search(
    evaluator: EnsembleEvaluator,
    *,
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 20260817,
    local_fraction: float = 0.5,
    pddr_unique_threshold: float = 0.35,
    objective_unique_threshold: float = 0.5,
    stagnation_patience: int = 3,
    trigger_mode: str = "pddr",
    allowed_operations: tuple[str, ...] = ("add", "delete", "swap"),
):
    """Run budget-neutral structured local search with PDDR or always-on trigger."""
    problem = EnsemblePruningProblem(evaluator)
    algorithm = PDDRLocalSearchNSGA2(
        pop_size=pop_size,
        local_fraction=local_fraction,
        pddr_unique_threshold=pddr_unique_threshold,
        objective_unique_threshold=objective_unique_threshold,
        stagnation_patience=stagnation_patience,
        trigger_mode=trigger_mode,
        allowed_operations=allowed_operations,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    return algorithm, problem, result


# Public API alias.  The historical function name remains available for
# compatibility with frozen experiment scripts.
run_pddr_dols = run_pddr_local_search


__all__ = [
    "PDDR_DOLS_NAME",
    "PDDR_DOLS_INTERNAL_ID",
    "PDDRLocalSearchNSGA2",
    "generate_structured_neighbors",
    "run_pddr_local_search",
    "run_pddr_dols",
]
