"""Tests for local datasets and leakage-free splitting."""

import numpy as np
import pytest

from data_loader import (
    load_experiment_dataset,
    load_builtin_dataset,
    make_fold_split,
    make_outer_splits,
    make_three_way_split,
)


def test_iris_three_way_split_has_no_overlap() -> None:
    """Every Iris row belongs to exactly one deterministic split."""
    dataset = load_builtin_dataset("iris")
    split = make_three_way_split(dataset.features, dataset.labels)

    assert split.x_train.shape == (90, 4)
    assert split.x_val.shape == (30, 4)
    assert split.x_test.shape == (30, 4)
    assert not set(split.train_indices) & set(split.val_indices)
    assert not set(split.train_indices) & set(split.test_indices)
    assert not set(split.val_indices) & set(split.test_indices)
    assert len(np.unique(np.concatenate(
        [split.train_indices, split.val_indices, split.test_indices]
    ))) == 150


def test_scaler_is_fitted_on_train_only() -> None:
    """The transformed train mean is zero while test data is only transformed."""
    dataset = load_builtin_dataset("wine")
    split = make_three_way_split(dataset.features, dataset.labels)

    assert np.allclose(split.x_train.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(
        split.scaler.mean_, dataset.features[split.train_indices].mean(axis=0)
    )


def test_split_is_reproducible_and_stratified() -> None:
    """A fixed seed reproduces indices and retains every class in each split."""
    dataset = load_builtin_dataset("breast_cancer")
    first = make_three_way_split(dataset.features, dataset.labels, random_state=17)
    second = make_three_way_split(dataset.features, dataset.labels, random_state=17)

    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.val_indices, second.val_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    expected_classes = set(np.unique(dataset.labels))
    assert set(np.unique(first.y_train)) == expected_classes
    assert set(np.unique(first.y_val)) == expected_classes
    assert set(np.unique(first.y_test)) == expected_classes


def test_unknown_dataset_and_invalid_sizes_are_rejected() -> None:
    """Invalid dataset names and impossible split fractions fail early."""
    with pytest.raises(ValueError, match="supported datasets"):
        load_builtin_dataset("unknown")
    dataset = load_builtin_dataset("iris")
    with pytest.raises(ValueError, match="less than 1"):
        make_three_way_split(
            dataset.features, dataset.labels, val_size=0.5, test_size=0.5
        )


def test_dry_bean_uses_stable_uci_id(monkeypatch, tmp_path) -> None:
    """Dry Bean loads through its stable UCI repository identifier."""
    calls = []

    class UCIData:
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = np.array([["SEKER"], ["BARBUNYA"], ["SEKER"]])

    class UCIResult:
        data = UCIData()

    def fake_fetch_ucirepo(**kwargs):
        calls.append(kwargs)
        return UCIResult()

    monkeypatch.setattr("data_loader.fetch_ucirepo", fake_fetch_ucirepo)
    cache_home = str(tmp_path / "cache")
    dataset = load_experiment_dataset("dry_bean", data_home=cache_home)

    assert calls == [{"id": 602}]
    assert dataset.features.shape == (3, 2)
    assert dataset.target_names == ("BARBUNYA", "SEKER")
    assert np.array_equal(dataset.labels, np.array([1, 0, 1]))

    monkeypatch.setattr(
        "data_loader.fetch_ucirepo",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    cached = load_experiment_dataset("dry_bean", data_home=cache_home)
    assert np.array_equal(cached.features, dataset.features)
    assert np.array_equal(cached.labels, dataset.labels)


def test_outer_splits_and_fold_split_preserve_global_test_indices() -> None:
    """Nested fold splitting keeps outer test rows isolated from preprocessing."""
    dataset = load_builtin_dataset("iris")
    folds = make_outer_splits(dataset.features, dataset.labels, n_splits=5, random_state=4)
    train_indices, test_indices = folds[0]
    split = make_fold_split(
        dataset.features,
        dataset.labels,
        train_indices,
        test_indices,
        random_state=4,
    )

    assert len(folds) == 5
    assert len(split.y_test) == len(test_indices)
    assert set(split.test_indices) == set(test_indices)
    assert not set(split.train_indices) & set(split.test_indices)
    assert not set(split.val_indices) & set(split.test_indices)
