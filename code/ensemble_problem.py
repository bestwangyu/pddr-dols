"""pymoo problem wrapper for cached binary ensemble pruning."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ensemble_pruning import EnsembleEvaluator


class EnsemblePruningProblem(ElementwiseProblem):
    """Expose EnsembleEvaluator as a three-objective binary pymoo problem."""

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
        """Evaluate one binary mask through the shared prediction cache."""
        out["F"] = self.evaluator.evaluate_objectives(decision)


__all__ = ["EnsemblePruningProblem"]
