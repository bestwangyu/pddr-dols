"""PEP/MDEP objective-modeling baselines for ensemble pruning.

These wrappers reproduce the target definitions used by the domain methods,
while keeping the current project optimizer and data protocol fixed.  They are
not claims of full source-level reimplementation of every original operator.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from ensemble_pruning import EnsembleEvaluator
from fair_baselines import BinaryNSGA2


class DomainPruningProblem(ElementwiseProblem):
    """Expose PEP or MDEP objective modeling over the shared evaluator."""

    def __init__(self, evaluator: EnsembleEvaluator, *, variant: str) -> None:
        if variant not in {"pep", "mdep"}:
            raise ValueError("variant must be 'pep' or 'mdep'")
        self.evaluator = evaluator
        self.variant = variant
        super().__init__(
            n_var=evaluator.n_classifiers,
            n_obj=2 if variant == "pep" else 3,
            n_ieq_constr=0,
            xl=np.zeros(evaluator.n_classifiers),
            xu=np.ones(evaluator.n_classifiers),
            vtype=bool,
        )

    def _evaluate(self, decision, out, *args, **kwargs) -> None:
        """Map a subset to PEP or MDEP's documented target combination."""
        result = self.evaluator.evaluate(decision)
        selected_count = len(result.selected_indices) / self.evaluator.n_classifiers
        if self.variant == "pep":
            out["F"] = np.array([result.validation_error, selected_count])
        else:
            out["F"] = np.array(
                [result.validation_error, result.margin_loss, selected_count]
            )


def run_domain_baseline(
    evaluator: EnsembleEvaluator,
    *,
    variant: str,
    pop_size: int = 40,
    n_gen: int = 20,
    seed: int = 20260803,
):
    """Run one PEP/MDEP objective-modeling baseline."""
    if n_gen < 1:
        raise ValueError("n_gen must be positive")
    problem = DomainPruningProblem(evaluator, variant=variant)
    algorithm = BinaryNSGA2(pop_size=pop_size, variation="de")
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    return algorithm, problem, result


__all__ = ["DomainPruningProblem", "run_domain_baseline"]
