"""Tests for heterogeneous classifier specifications and prediction cache."""

import numpy as np
import pytest

from classifier_pool import build_base_classifier_specs, train_classifier_pool
from data_loader import load_builtin_dataset, make_three_way_split


def test_base_pool_contains_24_unique_configurations() -> None:
    """The base grid matches the documented 3+9+3+4+4+1 configurations."""
    specs = build_base_classifier_specs(random_state=7)
    names = [spec.name for spec in specs]

    assert len(specs) == 24
    assert len(set(names)) == 24
    assert names[0] == "rf_n50"
    assert names[-1] == "gaussian_nb"


def test_small_pool_trains_and_caches_predictions() -> None:
    """A reduced Iris pool produces aligned predictions and positive costs."""
    dataset = load_builtin_dataset("iris")
    split = make_three_way_split(dataset.features, dataset.labels, random_state=3)
    specs = build_base_classifier_specs(random_state=3)[:3]
    pool = train_classifier_pool(
        split.x_train,
        split.y_train,
        split.x_val,
        split.x_test,
        specs=specs,
        timing_repeats=1,
    )

    assert len(pool.models) == 3
    assert pool.val_predictions.shape == (3, len(split.y_val))
    assert pool.test_predictions.shape == (3, len(split.y_test))
    assert np.all(pool.inference_seconds > 0)
    assert np.all(pool.storage_bytes > 0)
    assert np.all(pool.combined_costs > 0)
    assert pool.combined_costs.sum() == pytest.approx(1.0)


def test_invalid_cost_configuration_is_rejected() -> None:
    """Cost weights must be a non-negative convex combination."""
    dataset = load_builtin_dataset("iris")
    split = make_three_way_split(dataset.features, dataset.labels)
    specs = build_base_classifier_specs()[:1]
    with pytest.raises(ValueError, match="sum to 1"):
        train_classifier_pool(
            split.x_train,
            split.y_train,
            split.x_val,
            split.x_test,
            specs=specs,
            inference_weight=0.7,
            storage_weight=0.7,
        )
