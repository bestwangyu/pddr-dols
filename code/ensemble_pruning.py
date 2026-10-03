"""Objective evaluation for binary heterogeneous ensemble pruning.

The evaluator operates on cached predictions instead of retraining a model for
every binary subset.  This keeps objective evaluation deterministic and makes
the later evolutionary algorithms share exactly the same prediction cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnsembleEvaluation:
    """All information produced for one repaired binary subset."""

    objectives: np.ndarray
    predictions: np.ndarray
    selected_indices: tuple[int, ...]
    repaired_empty_subset: bool

    @property
    def validation_error(self) -> float:
        """Validation error, the first minimization objective."""
        return float(self.objectives[0])

    @property
    def normalized_cost(self) -> float:
        """Normalized inference/storage cost, the second objective."""
        return float(self.objectives[1])

    @property
    def margin_loss(self) -> float:
        """Margin loss, the third minimization objective."""
        return float(self.objectives[2])


class EnsembleEvaluator:
    """Evaluate binary subsets using cached classifier predictions.

    Parameters
    ----------
    predictions:
        Matrix with shape ``(n_classifiers, n_samples)``.  Each row contains
        hard predictions from one already-trained classifier on ``D_val``.
    labels:
        Ground-truth labels for the same samples.
    classifier_costs:
        One positive cost per classifier.  The caller should combine measured
        inference time and model-storage cost before constructing the class.
    cost_scale:
        Positive normalization constant.  By default it is the cost of the
        full ensemble, so the full ensemble has cost objective 1.0.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        classifier_costs: np.ndarray,
        *,
        cost_scale: float | None = None,
    ) -> None:
        prediction_values = np.asarray(predictions)
        label_values = np.asarray(labels)
        costs = np.asarray(classifier_costs, dtype=float)

        if prediction_values.ndim != 2:
            raise ValueError("predictions must have shape (n_classifiers, n_samples)")
        if label_values.ndim != 1:
            raise ValueError("labels must be a 1-D array")
        if prediction_values.shape[1] != label_values.size:
            raise ValueError("predictions and labels must contain the same samples")
        if costs.shape != (prediction_values.shape[0],):
            raise ValueError("classifier_costs must contain one value per classifier")
        if not np.isfinite(costs).all() or np.any(costs <= 0):
            raise ValueError("classifier_costs must be finite and strictly positive")
        if label_values.size == 0:
            raise ValueError("labels must contain at least one sample")

        self.predictions = prediction_values
        self.labels = label_values
        self.classifier_costs = costs
        self.n_classifiers = prediction_values.shape[0]
        self.n_samples = prediction_values.shape[1]
        self.classes = np.unique(
            np.concatenate((label_values, prediction_values.reshape(-1)))
        )

        full_cost = float(costs.sum())
        self.cost_scale = full_cost if cost_scale is None else float(cost_scale)
        if not np.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("cost_scale must be finite and strictly positive")

        # The cache key is the repaired binary mask.  An empty subset and its
        # deterministic repair therefore share one cached evaluation.
        self._cache: dict[tuple[int, ...], EnsembleEvaluation] = {}

    def _normalize_mask(self, mask: np.ndarray) -> tuple[np.ndarray, bool]:
        """Validate a binary mask and repair an empty subset deterministically."""
        values = np.asarray(mask)
        if values.ndim != 1 or values.size != self.n_classifiers:
            raise ValueError(
                "mask must be a 1-D vector with one value per classifier"
            )
        if values.dtype.kind == "b":
            binary = values.astype(bool, copy=True)
        else:
            if not np.all(np.isin(values, [0, 1])):
                raise ValueError("mask must contain only 0/1 or boolean values")
            binary = values.astype(bool, copy=True)

        repaired = False
        if not binary.any():
            # Choosing the cheapest classifier is deterministic and guarantees
            # that every evaluated ensemble contains at least one classifier.
            binary[int(np.argmin(self.classifier_costs))] = True
            repaired = True
        return binary, repaired

    def _hard_vote(self, selected_indices: tuple[int, ...]) -> np.ndarray:
        """Return deterministic majority-vote predictions for a subset."""
        selected = self.predictions[list(selected_indices)]
        counts = np.vstack(
            [(selected == class_label).sum(axis=0) for class_label in self.classes]
        )
        # np.argmax returns the first maximum, and self.classes is sorted by
        # np.unique, so ties are resolved consistently across all algorithms.
        return self.classes[np.argmax(counts, axis=0)]

    def _margin_loss(self, selected_indices: tuple[int, ...]) -> float:
        """Return mean ``1 - margin`` from hard-vote class proportions.

        For each sample, margin is the vote fraction of the true class minus
        the largest vote fraction of any incorrect class.  Lower loss is
        better, and the definition works for binary and multiclass labels.
        """
        selected = self.predictions[list(selected_indices)]
        counts = np.vstack(
            [(selected == class_label).sum(axis=0) for class_label in self.classes]
        )
        fractions = counts / float(len(selected_indices))
        true_rows = np.searchsorted(self.classes, self.labels)
        true_fraction = fractions[true_rows, np.arange(self.n_samples)]
        if fractions.shape[0] == 1:
            # With one class there is no incorrect-class vote; define its
            # strongest incorrect fraction as zero instead of -infinity.
            strongest_incorrect = np.zeros(self.n_samples)
        else:
            incorrect = fractions.copy()
            incorrect[true_rows, np.arange(self.n_samples)] = -np.inf
            strongest_incorrect = incorrect.max(axis=0)
        margins = true_fraction - strongest_incorrect
        return float(np.mean(1.0 - margins))

    def evaluate(self, mask: np.ndarray) -> EnsembleEvaluation:
        """Evaluate one binary subset and return its three objectives."""
        binary, repaired = self._normalize_mask(mask)
        key = tuple(int(value) for value in binary)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        selected_indices = tuple(np.flatnonzero(binary).tolist())
        predictions = self._hard_vote(selected_indices)
        validation_error = float(np.mean(predictions != self.labels))
        normalized_cost = float(self.classifier_costs[list(selected_indices)].sum())
        normalized_cost /= self.cost_scale
        margin_loss = self._margin_loss(selected_indices)
        result = EnsembleEvaluation(
            objectives=np.array(
                [validation_error, normalized_cost, margin_loss], dtype=float
            ),
            predictions=predictions,
            selected_indices=selected_indices,
            repaired_empty_subset=repaired,
        )
        self._cache[key] = result
        return result

    def evaluate_objectives(self, mask: np.ndarray) -> np.ndarray:
        """Return only the objective vector for pymoo-compatible callers."""
        return self.evaluate(mask).objectives.copy()

    @property
    def cache_size(self) -> int:
        """Number of unique repaired subsets currently cached."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear cached subset evaluations."""
        self._cache.clear()


__all__ = ["EnsembleEvaluation", "EnsembleEvaluator"]
