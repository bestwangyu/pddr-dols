"""PDDR-FF for minimization multi-objective problems.

This module only implements the relationship-based fitness calculation.
Ensemble-pruning objectives and evolutionary operators are intentionally kept
outside this file so that the core formula can be tested independently.
"""

from __future__ import annotations

import numpy as np


def _validate_objectives(objectives: np.ndarray) -> np.ndarray:
    """Return a validated 2-D floating-point objective matrix."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a 2-D array with shape (n, m)")
    if values.shape[1] == 0:
        raise ValueError("objectives must contain at least one objective")
    if not np.isfinite(values).all():
        raise ValueError("objectives must contain only finite values")
    return values


def dominates(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether ``first`` strictly Pareto-dominates ``second``.

    All objectives are minimized.  A vector dominates another vector when it
    is no worse in every objective and strictly better in at least one.  The
    strict-improvement condition means duplicate objective points do not
    dominate each other.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("objective vectors must have the same shape")
    return bool(np.all(first <= second) and np.any(first < second))


def dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    """Build the strict dominance matrix for a minimization problem.

    ``matrix[i, j]`` is ``True`` exactly when individual ``i`` dominates
    individual ``j``.  The implementation is deliberately explicit and easy
    to audit; later optimization can reuse this matrix for all diagnostics.
    """
    values = _validate_objectives(objectives)
    n = values.shape[0]
    matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = dominates(values[i], values[j])
    return matrix


def pddr_ff(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PDDR-FF values and its two relationship counts.

    Parameters
    ----------
    objectives:
        A ``(n, m)`` matrix.  Every objective is minimized.

    Returns
    -------
    eval_values:
        ``q + 1 / (p + 1)`` for every individual.
    q:
        Number of individuals that dominate each individual.
    p:
        Number of individuals dominated by each individual.
    """
    matrix = dominance_matrix(objectives)

    # Columns count dominators of each individual; rows count individuals it
    # dominates.  This follows the notation used in the proposal document.
    q = matrix.sum(axis=0).astype(float)
    p = matrix.sum(axis=1).astype(float)
    eval_values = q + 1.0 / (p + 1.0)
    return eval_values, q, p


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return a boolean mask for the current non-dominated set."""
    _, q, _ = pddr_ff(objectives)
    return q == 0


__all__ = ["dominates", "dominance_matrix", "nondominated_mask", "pddr_ff"]
