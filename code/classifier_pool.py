"""Heterogeneous base-classifier definitions, training and prediction cache."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ClassifierSpec:
    """A named, cloneable sklearn estimator configuration."""

    name: str
    estimator: BaseEstimator


@dataclass(frozen=True)
class TrainedClassifierPool:
    """Trained models, cached predictions and measured cost components."""

    names: tuple[str, ...]
    models: tuple[BaseEstimator, ...]
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    inference_seconds: np.ndarray
    storage_bytes: np.ndarray
    combined_costs: np.ndarray


def build_base_classifier_specs(
    *, random_state: int = 20260803
) -> tuple[ClassifierSpec, ...]:
    """Return the 24 base configurations defined in the experiment plan."""
    specs: list[ClassifierSpec] = []

    for n_estimators in (50, 100, 200):
        specs.append(
            ClassifierSpec(
                f"rf_n{n_estimators}",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=1,
                ),
            )
        )
    for c_value in (0.1, 1.0, 10.0):
        for gamma in (0.01, 0.1, 1.0):
            specs.append(
                ClassifierSpec(
                    f"svm_rbf_c{c_value:g}_g{gamma:g}",
                    SVC(C=c_value, gamma=gamma, kernel="rbf"),
                )
            )
    for c_value in (0.1, 1.0, 10.0):
        specs.append(
            ClassifierSpec(
                f"logreg_c{c_value:g}",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    random_state=random_state,
                ),
            )
        )
    for neighbors in (3, 5, 7, 11):
        specs.append(
            ClassifierSpec(
                f"knn_k{neighbors}", KNeighborsClassifier(n_neighbors=neighbors)
            )
        )
    for depth in (3, 5, 10, None):
        depth_name = "none" if depth is None else str(depth)
        specs.append(
            ClassifierSpec(
                f"tree_depth{depth_name}",
                DecisionTreeClassifier(
                    max_depth=depth, random_state=random_state
                ),
            )
        )
    specs.append(ClassifierSpec("gaussian_nb", GaussianNB()))
    return tuple(specs)


def _measure_batch_prediction_time(
    model: BaseEstimator, features: np.ndarray, repeats: int
) -> float:
    """Warm up and return median wall time for one prediction batch."""
    model.predict(features)
    measurements = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict(features)
        measurements.append(perf_counter() - start)
    return max(float(np.median(measurements)), np.finfo(float).eps)


def train_classifier_pool(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    specs: tuple[ClassifierSpec, ...] | None = None,
    timing_repeats: int = 5,
    inference_weight: float = 0.5,
    storage_weight: float = 0.5,
) -> TrainedClassifierPool:
    """Train a shared pool and cache validation/test hard predictions.

    The combined cost is a weighted sum of each model's share of total batch
    inference time and share of total serialized size.  It is suitable for a
    smoke pilot; formal timing experiments should repeat measurement under the
    fixed single-process protocol.
    """
    if not isinstance(timing_repeats, int) or timing_repeats < 1:
        raise ValueError("timing_repeats must be a positive integer")
    if inference_weight < 0 or storage_weight < 0:
        raise ValueError("cost weights must be non-negative")
    if not np.isclose(inference_weight + storage_weight, 1.0):
        raise ValueError("inference_weight and storage_weight must sum to 1")

    selected_specs = build_base_classifier_specs() if specs is None else specs
    if not selected_specs:
        raise ValueError("at least one classifier specification is required")
    names = [spec.name for spec in selected_specs]
    if len(names) != len(set(names)):
        raise ValueError("classifier specification names must be unique")

    models: list[BaseEstimator] = []
    val_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    inference_seconds: list[float] = []
    storage_bytes: list[int] = []

    for spec in selected_specs:
        model = clone(spec.estimator)
        model.fit(x_train, y_train)
        models.append(model)
        val_predictions.append(np.asarray(model.predict(x_val)))
        test_predictions.append(np.asarray(model.predict(x_test)))
        inference_seconds.append(
            _measure_batch_prediction_time(model, x_val, timing_repeats)
        )
        storage_bytes.append(
            len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        )

    time_values = np.asarray(inference_seconds, dtype=float)
    storage_values = np.asarray(storage_bytes, dtype=float)
    time_share = time_values / time_values.sum()
    storage_share = storage_values / storage_values.sum()
    combined_costs = (
        inference_weight * time_share + storage_weight * storage_share
    )
    return TrainedClassifierPool(
        names=tuple(names),
        models=tuple(models),
        val_predictions=np.vstack(val_predictions),
        test_predictions=np.vstack(test_predictions),
        inference_seconds=time_values,
        storage_bytes=storage_values,
        combined_costs=combined_costs,
    )


__all__ = [
    "ClassifierSpec",
    "TrainedClassifierPool",
    "build_base_classifier_specs",
    "train_classifier_pool",
]
