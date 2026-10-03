"""Local dataset loading and leakage-free train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetBundle:
    """A complete supervised classification dataset before splitting."""

    name: str
    features: np.ndarray
    labels: np.ndarray
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class DataSplit:
    """Leakage-free three-way split and the train-fitted scaler."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    scaler: StandardScaler


# Numeric public datasets are selected first so the initial batch runner can
# use the same StandardScaler path without introducing categorical preprocessing.
OPENML_DATASETS = {
    "banknote": "banknote-authentication",
    "segment": "segment",
    "spambase": "spambase",
    "magic": "MagicTelescope",
    "optdigits": "optdigits",
    # P7 independent confirmation datasets. These were not used for P2-P6
    # algorithm selection or parameter freezing.
    "vehicle": "vehicle",
    "page_blocks": "page-blocks",
    "letter": "letter",
    "semeion": "semeion",
    "yeast": "yeast",
}

UCI_DATASETS = {
    "dry_bean": 602,
}


def _uci_cache_path(name: str, data_home: str | None) -> Path:
    """Return the local cache path for one fixed UCI dataset."""
    root = Path(
        data_home
        or os.environ.get("SCIKIT_LEARN_DATA", "data/cache")
    )
    return root / "uci" / f"{name}_id{UCI_DATASETS[name]}.npz"


def _load_uci_dataset(name: str, *, data_home: str | None) -> DatasetBundle:
    """Load a UCI dataset from an atomic local cache or fetch it once."""
    cache_path = _uci_cache_path(name, data_home)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            return DatasetBundle(
                name=name,
                features=np.asarray(cached["features"], dtype=float),
                labels=np.asarray(cached["labels"], dtype=int),
                target_names=tuple(str(value) for value in cached["target_names"]),
            )

    raw = fetch_ucirepo(id=UCI_DATASETS[name])
    features = np.asarray(raw.data.features, dtype=float)
    target_values = np.asarray(raw.data.targets)
    if target_values.ndim == 2 and target_values.shape[1] == 1:
        target_values = target_values[:, 0]
    if target_values.ndim != 1 or target_values.size != features.shape[0]:
        raise ValueError(f"dataset {name!r} must contain one target column")
    unique_labels, encoded_labels = np.unique(target_values, return_inverse=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=features,
                labels=encoded_labels.astype(int),
                target_names=np.asarray(
                    [str(value) for value in unique_labels], dtype=str
                ),
            )
        temporary_path.replace(cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return DatasetBundle(
        name=name,
        features=features,
        labels=encoded_labels,
        target_names=tuple(str(value) for value in unique_labels),
    )


def load_builtin_dataset(name: str) -> DatasetBundle:
    """Load a small sklearn dataset without network access."""
    normalized_name = name.strip().lower().replace("-", "_").replace(" ", "_")
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    if normalized_name not in loaders:
        supported = ", ".join(sorted(loaders))
        raise ValueError(f"unknown dataset {name!r}; supported datasets: {supported}")

    raw = loaders[normalized_name]()
    return DatasetBundle(
        name=normalized_name,
        features=np.asarray(raw.data, dtype=float),
        labels=np.asarray(raw.target),
        target_names=tuple(str(value) for value in raw.target_names),
    )


def load_experiment_dataset(name: str, *, data_home: str | None = None) -> DatasetBundle:
    """Load a registered built-in, OpenML or UCI classification dataset."""
    normalized_name = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_name in {"iris", "wine", "breast_cancer"}:
        return load_builtin_dataset(normalized_name)
    if normalized_name in UCI_DATASETS:
        return _load_uci_dataset(normalized_name, data_home=data_home)
    if normalized_name not in OPENML_DATASETS:
        supported = ", ".join(
            ["iris", "wine", "breast_cancer", *OPENML_DATASETS, *UCI_DATASETS]
        )
        raise ValueError(f"unknown experiment dataset {name!r}; supported: {supported}")

    raw = fetch_openml(
        name=OPENML_DATASETS[normalized_name],
        version=1,
        as_frame=False,
        data_home=data_home,
    )
    try:
        features = np.asarray(raw.data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"dataset {normalized_name!r} contains non-numeric features; "
            "add an explicit categorical preprocessing path before using it"
        ) from exc
    labels = np.asarray(raw.target)
    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    return DatasetBundle(
        name=normalized_name,
        features=features,
        labels=encoded_labels,
        target_names=tuple(str(value) for value in unique_labels),
    )


def make_outer_splits(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 20260803,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified outer train/test indices for repeated experiments."""
    x = np.asarray(features)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("features and labels must have aligned sample rows")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return [
        (train_indices.astype(int), test_indices.astype(int))
        for train_indices, test_indices in splitter.split(x, y)
    ]


def make_fold_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    val_size: float = 0.2,
    random_state: int = 20260803,
) -> DataSplit:
    """Create inner train/validation data and transform the untouched outer test.

    The returned indices are global indices from the original dataset.  The
    scaler is fitted only on the inner training subset.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    outer_train = np.asarray(train_indices, dtype=int)
    outer_test = np.asarray(test_indices, dtype=int)
    if outer_train.size == 0 or outer_test.size == 0:
        raise ValueError("outer train and test indices must be non-empty")
    inner_train, inner_val = train_test_split(
        outer_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y[outer_train],
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[inner_train])
    x_val = scaler.transform(x[inner_val])
    x_test = scaler.transform(x[outer_test])
    return DataSplit(
        x_train=x_train,
        y_train=y[inner_train],
        x_val=x_val,
        y_val=y[inner_val],
        x_test=x_test,
        y_test=y[outer_test],
        train_indices=inner_train,
        val_indices=inner_val,
        test_indices=outer_test,
        scaler=scaler,
    )


def make_three_way_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 20260803,
) -> DataSplit:
    """Create stratified D_train/D_val/D_test and scale without leakage.

    ``val_size`` and ``test_size`` are fractions of the full dataset.  Test
    indices are held out first; validation indices are then drawn only from the
    remaining training portion.  StandardScaler is fitted on D_train only.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("features and labels must have aligned sample rows")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty 2-D matrix")
    if not np.isfinite(x).all():
        raise ValueError("features must contain only finite values")
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1")

    indices = np.arange(x.shape[0])
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y[train_val_indices],
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_indices])
    x_val = scaler.transform(x[val_indices])
    x_test = scaler.transform(x[test_indices])
    return DataSplit(
        x_train=x_train,
        y_train=y[train_indices],
        x_val=x_val,
        y_val=y[val_indices],
        x_test=x_test,
        y_test=y[test_indices],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        scaler=scaler,
    )


__all__ = [
    "DataSplit",
    "DatasetBundle",
    "OPENML_DATASETS",
    "UCI_DATASETS",
    "load_builtin_dataset",
    "load_experiment_dataset",
    "make_fold_split",
    "make_outer_splits",
    "make_three_way_split",
]
