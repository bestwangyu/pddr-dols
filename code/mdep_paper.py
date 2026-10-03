"""Paper-faithful implementation of MDEP (Wu et al., 2022).

MDEP formulates ensemble pruning as a three-objective minimization problem:
validation error, margin ratio, and ensemble size.  This implementation uses
NSGA-III as in the paper's main comparison, with the paper's special
singleton initialization, uniform crossover, bit-wise mutation, and repair
that excludes offspring of size zero or one.

The original paper trains 100 homogeneous Bagging C4.5 trees and focuses on
binary classification.  The current project supplies a cached heterogeneous
classifier pool, so the search operators and objective definition are
reproduced while the data/pool protocol is explicitly recorded as an
adaptation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ensemble_pruning import EnsembleEvaluator


def margin_ratio(
    predictions: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Compute the MDEP margin ratio for a selected subset.

    For binary data this reduces to the paper's signed voting margin
    ``2 * correct_vote_fraction - 1``.  For multiclass data, the compatible
    extension uses true-class vote fraction minus the strongest incorrect
    class vote fraction, matching the project's margin definition.
    """
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or not values.any():
        raise ValueError("MDEP margin ratio requires a non-empty mask")
    prediction_values = np.asarray(predictions)
    label_values = np.asarray(labels)
    selected = prediction_values[values]
    classes = np.unique(np.concatenate((label_values, prediction_values.reshape(-1))))
    counts = np.vstack([(selected == label).sum(axis=0) for label in classes])
    fractions = counts / float(selected.shape[0])
    true_rows = np.searchsorted(classes, label_values)
    true_fraction = fractions[true_rows, np.arange(label_values.size)]
    if len(classes) == 1:
        strongest_incorrect = np.zeros(label_values.size)
    else:
        incorrect = fractions.copy()
        incorrect[true_rows, np.arange(label_values.size)] = -np.inf
        strongest_incorrect = incorrect.max(axis=0)
    margins = true_fraction - strongest_incorrect
    mean_margin = float(np.mean(margins))
    std_margin = float(np.std(margins, ddof=1)) if margins.size > 1 else 0.0
    denominator = max(abs(mean_margin), epsilon)
    return float(min(std_margin / denominator, 1e6))


class MDEPProblem(ElementwiseProblem):
    """Expose validation error, margin ratio, and size as three objectives."""

    def __init__(self, evaluator: EnsembleEvaluator) -> None:
        self.evaluator = evaluator
        super().__init__(
            n_var=evaluator.n_classifiers,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.zeros(evaluator.n_classifiers),
            xu=np.ones(evaluator.n_classifiers),
            vtype=bool,
        )

    def _evaluate(self, decision, out, *args, **kwargs) -> None:
        mask = np.asarray(decision, dtype=bool)
        if not mask.any():
            out["F"] = np.array([1e6, 1e6, 1e6], dtype=float)
            return
        result = self.evaluator.evaluate(mask)
        out["F"] = np.array(
            [
                result.validation_error,
                margin_ratio(self.evaluator.predictions, self.evaluator.labels, mask),
                len(result.selected_indices) / self.evaluator.n_classifiers,
            ],
            dtype=float,
        )


class MDEPInitialization(Sampling):
    """Initialize with all non-dominated singleton subsets, then random masks."""

    def __init__(self, problem: MDEPProblem, pop_size: int, seed: int) -> None:
        super().__init__()
        self.problem = problem
        self.pop_size = int(pop_size)
        self.seed = int(seed)

    def _do(self, _, n_samples, **kwargs):
        if n_samples != self.pop_size:
            raise ValueError("MDEP initialization requires a fixed population size")
        rng = np.random.default_rng(self.seed)
        singleton_masks = []
        singleton_objectives = []
        for index in range(self.problem.n_var):
            mask = np.zeros(self.problem.n_var, dtype=bool)
            mask[index] = True
            singleton_masks.append(mask)
            singleton_objectives.append(self.problem.evaluate(mask))
        keep = []
        for i, objective in enumerate(singleton_objectives):
            if not any(
                np.all(other <= objective) and np.any(other < objective)
                for j, other in enumerate(singleton_objectives)
                if i != j
            ):
                keep.append(i)
        masks = [singleton_masks[i] for i in keep]
        seen = {tuple(mask.astype(int)) for mask in masks}
        while len(masks) < n_samples:
            mask = rng.integers(0, 2, size=self.problem.n_var, dtype=np.int8).astype(bool)
            if mask.sum() <= 1:
                on = rng.choice(self.problem.n_var, size=2, replace=False)
                mask[on] = True
            key = tuple(mask.astype(int))
            if key not in seen:
                seen.add(key)
                masks.append(mask)
        return np.asarray(masks, dtype=bool)


class MDEPRepair(Repair):
    """Apply the paper's repeated mutation until offspring size exceeds one."""

    def __init__(self, *, mutation_probability: float | None = None) -> None:
        super().__init__()
        self.mutation_probability = mutation_probability

    def _do(self, problem, X, **kwargs):
        values = np.asarray(X, dtype=bool).copy()
        probability = (
            1.0 / problem.n_var
            if self.mutation_probability is None
            else float(self.mutation_probability)
        )
        for row in values:
            attempts = 0
            while row.sum() <= 1 and attempts < 1000:
                row ^= np.random.random(problem.n_var) < probability
                attempts += 1
            if row.sum() <= 1:
                row[: min(2, problem.n_var)] = True
        return values


def select_final_solution(
    masks: Iterable[np.ndarray], objectives: Iterable[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Select minimum validation error, breaking ties by smaller size."""
    pairs = list(zip(masks, objectives))
    if not pairs:
        raise ValueError("MDEP result contains no solutions")
    return min(pairs, key=lambda pair: (float(pair[1][0]), float(pair[1][2])))


def build_mdep_algorithm(
    problem: MDEPProblem,
    *,
    pop_size: int,
    seed: int,
    crossover_probability: float = 0.7,
) -> NSGA3:
    """Build MDEP while keeping initialization and offspring repair separate.

    ``pymoo`` applies an algorithm-level repair to both the initial population
    and later offspring.  MDEP needs a narrower policy: non-dominated
    singleton solutions are retained at initialization, while offspring with
    zero or one selected classifier are repaired.  The mating-level repair
    below implements that distinction explicitly.
    """
    sampling = MDEPInitialization(problem, pop_size, seed)
    ref_dirs = get_reference_directions(
        "das-dennis",
        3,
        n_partitions=max(1, int(round((pop_size ** (1 / 2)) - 1))),
    )
    # Ensure the requested population size is respected even when the
    # reference-direction construction yields a different count.
    if len(ref_dirs) != pop_size:
        ref_dirs = np.random.default_rng(seed).dirichlet(np.ones(3), size=pop_size)
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=sampling,
        crossover=UniformCrossover(prob=crossover_probability),
        mutation=BitflipMutation(prob=1.0, prob_var=1.0 / problem.n_var),
        eliminate_duplicates=False,
    )
    # NSGA3's initialization keeps its default NoRepair.  Only mating output
    # receives the MDEP size constraint.
    algorithm.mating.repair = MDEPRepair()
    algorithm.final_selection = "validation_error_then_size"
    algorithm.implementation_type = "paper_reimplementation"
    algorithm.source_paper = (
        "Wu et al. (2022), Multi-objective Evolutionary Ensemble Pruning "
        "Guided by Margin Distribution"
    )
    algorithm.objective_definition = "validation_error_margin_ratio_selected_count"
    algorithm.vds_enabled = False
    algorithm.initialization_policy = "retain_non_dominated_singletons"
    algorithm.offspring_repair_policy = "repair_size_zero_or_one"
    return algorithm


def run_mdep_paper(
    evaluator: EnsembleEvaluator,
    *,
    pop_size: int = 100,
    n_gen: int = 500,
    seed: int = 20260803,
    crossover_probability: float = 0.7,
) :
    """Run MDEP with NSGA-III and return a pymoo-compatible result."""
    if evaluator.n_classifiers < 2:
        raise ValueError("MDEP requires at least two classifiers")
    if pop_size < 2 or n_gen < 1:
        raise ValueError("pop_size and n_gen must be positive")
    problem = MDEPProblem(evaluator)
    algorithm = build_mdep_algorithm(
        problem,
        pop_size=pop_size,
        seed=seed,
        crossover_probability=crossover_probability,
    )
    result = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return result


__all__ = [
    "MDEPInitialization",
    "MDEPProblem",
    "MDEPRepair",
    "build_mdep_algorithm",
    "margin_ratio",
    "run_mdep_paper",
    "select_final_solution",
]
