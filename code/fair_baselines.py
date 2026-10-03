"""Fair binary baselines sharing the same problem, budget and DE variation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.decomposition.pbi import PBI
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.algorithms.moo.rnsga2 import RankAndModifiedCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ensemble_pruning import EnsembleEvaluator
from ensemble_problem import EnsemblePruningProblem
from pddr_algorithm import (
    BinaryDifferentialCrossover,
    NonEmptyBinaryRepair,
)
from pddrff import pddr_ff


# The decomposition geometry is part of the frozen MOEA/D method definition.
# It must not change with the independent optimization seed, otherwise each
# replicate would compare PDDR-DOLS with a different MOEA/D decomposition.
MOEAD_REFERENCE_METHOD = "energy"
MOEAD_REFERENCE_SEED = 20260820
MOEAD_PBI_THETA = 5.0
MOEAD_NEIGHBOR_MATING_PROBABILITY = 0.9
MOEAD_MAX_NEIGHBORS = 20


@lru_cache(maxsize=16)
def _cached_reference_directions(
    n_points: int,
    n_obj: int,
    seed: int,
) -> np.ndarray:
    """Build one immutable-in-practice reference geometry per process."""
    if n_points < n_obj or n_obj < 2:
        raise ValueError("n_points must be at least n_obj and n_obj must be >= 2")
    # pymoo 0.6.1's energy-direction implementation draws from NumPy's
    # module-level RNG in part of its construction.  Isolate and restore that
    # state so the returned geometry is deterministic without perturbing the
    # optimizer's random stream.
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        directions = np.asarray(
            get_reference_directions(
                MOEAD_REFERENCE_METHOD,
                n_obj,
                n_points,
                seed=seed,
                sampling="construction",
            ),
            dtype=float,
        )
    finally:
        np.random.set_state(rng_state)
    if directions.shape != (n_points, n_obj):
        raise ValueError(
            "reference-direction generator returned an unexpected shape: "
            f"{directions.shape}"
        )
    if not np.isfinite(directions).all() or np.any(directions < 0.0):
        raise ValueError("reference directions must be finite and non-negative")
    if not np.allclose(directions.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("every reference direction must lie on the unit simplex")
    directions.setflags(write=False)
    return directions


def make_reference_directions(
    n_points: int, n_obj: int = 3, seed: int = MOEAD_REFERENCE_SEED
) -> np.ndarray:
    """Return fixed, well-spread energy reference directions for MOEA/D."""
    # Return a copy so callers cannot modify the cached protocol definition.
    return _cached_reference_directions(n_points, n_obj, seed).copy()


class BinaryNSGA2(GeneticAlgorithm):
    """NSGA-II-style survival with shared binary variation."""

    def __init__(self, *, pop_size: int, variation: str = "de") -> None:
        if variation != "de":
            raise ValueError("fair baseline currently requires variation='de'")
        super().__init__(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=BinaryRandomSampling(),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=BinaryDifferentialCrossover(),
            mutation=BitflipMutation(prob=1.0, prob_var=0.02),
            survival=RankAndCrowding(),
            repair=NonEmptyBinaryRepair(),
            output=None,
            eliminate_duplicates=False,
            advance_after_initial_infill=True,
        )
        self.tournament_type = "comp_by_dom_and_crowding"


class NormalizedReferenceSurvival(Survival):
    """Reference-direction survival with zero-range-safe objective scaling."""

    def __init__(self, ref_dirs: np.ndarray) -> None:
        super().__init__(filter_infeasible=False)
        self.inner = RankAndModifiedCrowdingSurvival(
            ref_dirs,
            epsilon=0.01,
            weights=None,
            normalization="no",
            extreme_points_as_reference_points=False,
        )

    def _do(self, problem, pop, n_survive=None, **kwargs):
        """Normalize temporary objective values before reference selection."""
        raw = np.asarray(pop.get("F"), dtype=float).copy()
        low = raw.min(axis=0)
        span = raw.max(axis=0) - low
        span = np.where(span > 0.0, span, 1.0)
        pop.set("F", (raw - low) / span)
        try:
            selected = self.inner._do(problem, pop, n_survive=n_survive, **kwargs)
        finally:
            # Restore raw objectives because all downstream metrics use the
            # original error/cost/margin definitions.
            pop.set("F", raw)
        return selected


class BinaryReferenceDirection(GeneticAlgorithm):
    """Reference-direction survival with shared binary DE variation."""

    def __init__(self, *, pop_size: int, ref_dirs: np.ndarray) -> None:
        if len(ref_dirs) != pop_size:
            raise ValueError("ref_dirs length must equal pop_size")
        super().__init__(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=BinaryRandomSampling(),
            selection=RandomSelection(),
            crossover=BinaryDifferentialCrossover(),
            mutation=BitflipMutation(prob=1.0, prob_var=0.02),
            survival=NormalizedReferenceSurvival(ref_dirs),
            repair=NonEmptyBinaryRepair(),
            output=None,
            eliminate_duplicates=False,
            advance_after_initial_infill=True,
        )


class BinaryMOEAD(MOEAD):
    """MOEA/D with binary sampling, shared DE crossover and bit repair."""

    def __init__(self, *, ref_dirs: np.ndarray) -> None:
        pop_size = len(ref_dirs)
        super().__init__(
            ref_dirs=ref_dirs,
            n_neighbors=min(MOEAD_MAX_NEIGHBORS, pop_size),
            decomposition=PBI(theta=MOEAD_PBI_THETA),
            prob_neighbor_mating=MOEAD_NEIGHBOR_MATING_PROBABILITY,
            sampling=BinaryRandomSampling(),
            crossover=BinaryDifferentialCrossover(),
            mutation=BitflipMutation(prob=1.0, prob_var=0.02),
            repair=NonEmptyBinaryRepair(),
            output=None,
        )


def _make_algorithm(
    variant: str,
    pop_size: int,
    *,
    reference_seed: int = MOEAD_REFERENCE_SEED,
):
    """Build one baseline algorithm with deterministic reference directions."""
    if variant == "nsga2":
        return BinaryNSGA2(pop_size=pop_size)
    ref_dirs = make_reference_directions(
        pop_size,
        n_obj=3,
        seed=reference_seed,
    )
    if variant == "reference":
        return BinaryReferenceDirection(pop_size=pop_size, ref_dirs=ref_dirs)
    if variant == "moead":
        return BinaryMOEAD(ref_dirs=ref_dirs)
    raise ValueError("variant must be 'nsga2', 'reference' or 'moead'")


def run_baseline(
    evaluator: EnsembleEvaluator,
    *,
    variant: str,
    pop_size: int = 40,
    n_gen: int = 20,
    seed: int = 20260803,
    reference_seed: int = MOEAD_REFERENCE_SEED,
):
    """Run one fair baseline using the common binary DE variation."""
    if n_gen < 1:
        raise ValueError("n_gen must be positive")
    problem = EnsemblePruningProblem(evaluator)
    algorithm = _make_algorithm(
        variant,
        pop_size,
        reference_seed=reference_seed,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    return algorithm, problem, result


def run_fair_comparison(
    evaluator: EnsembleEvaluator,
    *,
    variants: tuple[str, ...] = ("nsga2", "reference", "moead"),
    pop_size: int = 40,
    n_gen: int = 20,
    seed: int = 20260803,
    reference_seed: int = MOEAD_REFERENCE_SEED,
) -> dict:
    """Run all requested baselines with matched seed and evaluation budget."""
    comparison = {}
    for variant in variants:
        _, _, result = run_baseline(
            evaluator,
            variant=variant,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            reference_seed=reference_seed,
        )
        comparison[variant] = result
    return comparison


__all__ = [
    "BinaryMOEAD",
    "BinaryNSGA2",
    "BinaryReferenceDirection",
    "MOEAD_MAX_NEIGHBORS",
    "MOEAD_NEIGHBOR_MATING_PROBABILITY",
    "MOEAD_PBI_THETA",
    "MOEAD_REFERENCE_METHOD",
    "MOEAD_REFERENCE_SEED",
    "make_reference_directions",
    "run_baseline",
    "run_fair_comparison",
]
