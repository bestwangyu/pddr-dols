"""Binary evolutionary drivers using Pure PDDR or DA-PDDR survival.

This is the first executable optimization baseline.  It deliberately uses
binary genetic variation so that the survival mechanisms can be validated in
isolation before adding the HMODE-style DE variation.
"""

from __future__ import annotations

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.crossover import Crossover
from pymoo.core.selection import Selection
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput

from da_pddr_ep import DAPDDRSurvival, ObjectiveSpaceDASurvival
from ensemble_pruning import EnsembleEvaluator
from ensemble_problem import EnsemblePruningProblem
from pddr_survival import PDDRSurvival
from pddrff import pddr_ff
from rank_pddr_survival import (
    ExtremePDDRSurvival,
    HybridPDDRSurvival,
    RankPDDRSurvival,
    extreme_pddr_parent_keys,
    hybrid_pddr_parent_keys,
    rank_pddr_parent_keys,
)


def _compare_pddr(pop, permutations, **kwargs):
    """Choose lower PDDR-FF values in each tournament, stably by index."""
    eval_values, _, _ = pddr_ff(pop.get("F"))
    winners = []
    for candidates in permutations:
        winners.append(min(candidates, key=lambda index: (eval_values[index], index)))
    return np.asarray(winners, dtype=int)


class PDDRTournamentSelection(Selection):
    """Tournament selection that uses PDDR-FF instead of rank/crowding."""

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        n_random = n_select * n_parents * 2
        permutations = np.random.permutation(len(pop))
        repeats = int(np.ceil(n_random / len(pop)))
        candidates = np.tile(permutations, repeats)[:n_random]
        candidates = candidates.reshape((n_select * n_parents, 2))
        winners = _compare_pddr(pop, candidates, **kwargs)
        return winners.reshape((n_select, n_parents))


class RankPDDRTournamentSelection(Selection):
    """Tournament selection aligned with rank-preserving PDDR survival."""

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        n_random = n_select * n_parents * 2
        permutations = np.random.permutation(len(pop))
        repeats = int(np.ceil(n_random / len(pop)))
        candidates = np.tile(permutations, repeats)[:n_random]
        candidates = candidates.reshape((n_select * n_parents, 2))
        ranks, objective_spacing, pddr_values, hamming_spacing = (
            rank_pddr_parent_keys(pop.get("F"), pop.get("X"))
        )
        winners = [
            min(
                pair,
                key=lambda index: (
                    int(ranks[index]),
                    -float(objective_spacing[index]),
                    float(pddr_values[index]),
                    -float(hamming_spacing[index]),
                    int(index),
                ),
            )
            for pair in candidates
        ]
        return np.asarray(winners, dtype=int).reshape((n_select, n_parents))


class HybridPDDRTournamentSelection(Selection):
    """Tournament selection prioritizing convergence within each rank."""

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        n_random = n_select * n_parents * 2
        permutations = np.random.permutation(len(pop))
        repeats = int(np.ceil(n_random / len(pop)))
        candidates = np.tile(permutations, repeats)[:n_random]
        candidates = candidates.reshape((n_select * n_parents, 2))
        ranks, pddr_values, objective_spacing, hamming_spacing = (
            hybrid_pddr_parent_keys(pop.get("F"), pop.get("X"))
        )
        winners = [
            min(
                pair,
                key=lambda index: (
                    int(ranks[index]),
                    float(pddr_values[index]),
                    -float(objective_spacing[index]),
                    -float(hamming_spacing[index]),
                    int(index),
                ),
            )
            for pair in candidates
        ]
        return np.asarray(winners, dtype=int).reshape((n_select, n_parents))


class ExtremePDDRTournamentSelection(Selection):
    """Tournament selection using rank, crowding and then PDDR."""

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        n_random = n_select * n_parents * 2
        permutations = np.random.permutation(len(pop))
        repeats = int(np.ceil(n_random / len(pop)))
        candidates = np.tile(permutations, repeats)[:n_random]
        candidates = candidates.reshape((n_select * n_parents, 2))
        ranks, crowding, pddr_values, hamming_spacing = extreme_pddr_parent_keys(
            pop.get("F"), pop.get("X")
        )
        winners = [
            min(
                pair,
                key=lambda index: (
                    int(ranks[index]),
                    -float(crowding[index]),
                    float(pddr_values[index]),
                    -float(hamming_spacing[index]),
                    int(index),
                ),
            )
            for pair in candidates
        ]
        return np.asarray(winners, dtype=int).reshape((n_select, n_parents))


class BinaryDifferentialCrossover(Crossover):
    """Binary HMODE-style differential variation.

    Four parents are interpreted as ``s``, ``s3``, ``s1`` and ``s2``.  The
    continuous donor follows ``s + K*(s3-s) + F*(s1-s2)`` and is then converted
    to a binary child by binomial crossover and a 0.5 threshold.
    """

    def __init__(self, F: float = 0.5, K: float = 0.5, CR: float = 0.9) -> None:
        super().__init__(n_parents=4, n_offsprings=1)
        if not 0.0 <= F <= 2.0 or not 0.0 <= K <= 2.0:
            raise ValueError("F and K must be in [0, 2]")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        self.F = float(F)
        self.K = float(K)
        self.CR = float(CR)

    def _do(self, problem, X, **kwargs):
        """Create one binary child for every four-parent mating."""
        # Pymoo stores binary variables as bool; differential arithmetic must
        # happen in floating point before the donor is thresholded back.
        base, guide, first, second = (np.asarray(parent, dtype=float) for parent in X)
        donor = base + self.K * (guide - base) + self.F * (first - second)
        donor = np.clip(donor, 0.0, 1.0)
        n_matings, n_var = base.shape
        mask = np.random.random((n_matings, n_var)) < self.CR
        # Guarantee at least one donor position per child, as in binomial DE.
        forced = np.random.randint(0, n_var, size=n_matings)
        mask[np.arange(n_matings), forced] = True
        child = np.where(mask, donor >= 0.5, base >= 0.5)
        return child[None, :, :]


class NonEmptyBinaryRepair(Repair):
    """Repair all-zero subsets using the cheapest classifier in the pool."""

    def _do(self, problem, X, **kwargs):
        decisions = np.asarray(X, dtype=bool).copy()
        empty_rows = np.flatnonzero(decisions.sum(axis=1) == 0)
        cheapest = int(np.argmin(problem.evaluator.classifier_costs))
        decisions[empty_rows, cheapest] = True
        return decisions


class BinaryPDDREvolution(GeneticAlgorithm):
    """Shared binary variation driver for PDDR-EP and DA-PDDR-EP."""

    def __init__(
        self,
        *,
        survival,
        pop_size: int = 40,
        variation: str = "ga",
        selection: Selection | None = None,
        **kwargs,
    ) -> None:
        if variation not in {"ga", "de"}:
            raise ValueError("variation must be 'ga' or 'de'")
        crossover = (
            TwoPointCrossover(prob=0.9)
            if variation == "ga"
            else BinaryDifferentialCrossover(F=0.5, K=0.5, CR=0.9)
        )
        super().__init__(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=BinaryRandomSampling(),
            selection=PDDRTournamentSelection() if selection is None else selection,
            crossover=crossover,
            # DE still receives a small bit-flip mutation so binary masks can
            # escape a frozen donor threshold; both survival variants share it.
            mutation=BitflipMutation(prob=1.0, prob_var=0.02),
            survival=survival,
            repair=NonEmptyBinaryRepair(),
            output=MultiObjectiveOutput(),
            eliminate_duplicates=False,
            advance_after_initial_infill=True,
            **kwargs,
        )
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = "pddr"
        self.variation = variation

    def _set_optimum(self, **kwargs):
        """Expose the current non-dominated set without rank/crowding fields."""
        _, q, _ = pddr_ff(self.pop.get("F"))
        self.opt = self.pop[q == 0]


class PDDREP(BinaryPDDREvolution):
    """Direct PDDR-FF survival baseline."""

    def __init__(self, *, pop_size: int = 40, variation: str = "ga", **kwargs) -> None:
        super().__init__(
            pop_size=pop_size, survival=PDDRSurvival(), variation=variation, **kwargs
        )


class DAPDDREP(BinaryPDDREvolution):
    """Degeneracy-aware PDDR survival variant."""

    def __init__(self, *, pop_size: int = 40, variation: str = "ga", **kwargs) -> None:
        super().__init__(
            pop_size=pop_size, survival=DAPDDRSurvival(), variation=variation, **kwargs
        )


class ObjectiveSpaceDAPDDREP(BinaryPDDREvolution):
    """Objective-space diversity compensation ablation."""

    def __init__(
        self,
        *,
        pop_size: int = 40,
        variation: str = "ga",
        boundary_tolerance: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(
            pop_size=pop_size,
            survival=ObjectiveSpaceDASurvival(
                boundary_tolerance=boundary_tolerance
            ),
            variation=variation,
            **kwargs,
        )


class RankPDDREP(BinaryPDDREvolution):
    """Pareto-rank-preserving PDDR survival and parent selection variant."""

    def __init__(self, *, pop_size: int = 40, variation: str = "ga", **kwargs) -> None:
        super().__init__(
            pop_size=pop_size,
            survival=RankPDDRSurvival(),
            selection=RankPDDRTournamentSelection(),
            variation=variation,
            **kwargs,
        )


class HybridPDDREP(BinaryPDDREvolution):
    """Convergence-first survival with limited objective-space spreading."""

    def __init__(
        self,
        *,
        pop_size: int = 40,
        variation: str = "ga",
        diversity_fraction: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(
            pop_size=pop_size,
            survival=HybridPDDRSurvival(diversity_fraction=diversity_fraction),
            selection=HybridPDDRTournamentSelection(),
            variation=variation,
            **kwargs,
        )


class ExtremePDDREP(BinaryPDDREvolution):
    """Extreme-preserving crowding survival with PDDR tie-breaking."""

    def __init__(self, *, pop_size: int = 40, variation: str = "ga", **kwargs) -> None:
        super().__init__(
            pop_size=pop_size,
            survival=ExtremePDDRSurvival(),
            selection=ExtremePDDRTournamentSelection(),
            variation=variation,
            **kwargs,
        )


def run_binary_pddr(
    evaluator: EnsembleEvaluator,
    *,
    variant: str = "pddr",
    pop_size: int = 40,
    n_gen: int = 20,
    variation: str = "ga",
    boundary_tolerance: float = 0.05,
    seed: int = 20260803,
    verbose: bool = False,
):
    """Run a small binary PDDR or DA-PDDR optimization pilot."""
    variants = {
        "pddr",
        "da_pddr",
        "da_pddr_obj",
        "rank_pddr",
        "hybrid_pddr",
        "extreme_pddr",
    }
    if variant not in variants:
        raise ValueError(
            "variant must be one of " + ", ".join(sorted(variants))
        )
    if n_gen < 1:
        raise ValueError("n_gen must be positive")
    problem = EnsemblePruningProblem(evaluator)
    algorithm_class = {
        "pddr": PDDREP,
        "da_pddr": DAPDDREP,
        "da_pddr_obj": ObjectiveSpaceDAPDDREP,
        "rank_pddr": RankPDDREP,
        "hybrid_pddr": HybridPDDREP,
        "extreme_pddr": ExtremePDDREP,
    }[variant]
    algorithm_kwargs = {"pop_size": pop_size, "variation": variation}
    if variant == "da_pddr_obj":
        algorithm_kwargs["boundary_tolerance"] = boundary_tolerance
    algorithm = algorithm_class(**algorithm_kwargs)
    return algorithm, problem, minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )


__all__ = [
    "BinaryPDDREvolution",
    "BinaryDifferentialCrossover",
    "DAPDDREP",
    "NonEmptyBinaryRepair",
    "ObjectiveSpaceDAPDDREP",
    "PDDREP",
    "PDDRTournamentSelection",
    "RankPDDREP",
    "RankPDDRTournamentSelection",
    "HybridPDDREP",
    "HybridPDDRTournamentSelection",
    "ExtremePDDREP",
    "ExtremePDDRTournamentSelection",
    "run_binary_pddr",
]
