"""Exact competition metric utilities."""

from __future__ import annotations

import numpy as np


def weighted_pearson(y_true, y_pred, *, mask=None) -> float:
    """Return global weighted Pearson correlation for one target.

    Values are clipped to [-2, 2] and weights are abs(clipped target), as in
    the competition scorer. Degenerate inputs return 0.0 rather than NaN.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} != {p.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != y.shape:
            raise ValueError(f"mask shape mismatch: {m.shape} != {y.shape}")
        y, p = y[m], p[m]
    if y.size == 0 or not (np.isfinite(y).all() and np.isfinite(p).all()):
        return 0.0
    y = np.clip(y, -2.0, 2.0)
    p = np.clip(p, -2.0, 2.0)
    w = np.abs(y)
    total = w.sum()
    if total <= 0.0:
        return 0.0
    y_bar = np.dot(w, y) / total
    p_bar = np.dot(w, p) / total
    yc, pc = y - y_bar, p - p_bar
    denom = np.sqrt(np.dot(w, yc * yc) * np.dot(w, pc * pc))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(w, yc * pc) / denom)


def global_weighted_pearson(y_true, y_pred, *, mask=None) -> float:
    """Average the two pooled target correlations (columns t0 and t1)."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.ndim != 2 or p.shape != y.shape or y.shape[1] != 2:
        raise ValueError("expected y_true and y_pred with shape (n_rows, 2)")
    return float(np.mean([weighted_pearson(y[:, i], p[:, i], mask=mask) for i in range(2)]))
