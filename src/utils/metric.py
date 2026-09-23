"""Global Weighted Pearson (WP), matching the starter pack scorer.

``weighted_pearson`` follows ``utils.weighted_pearson`` from the starter pack
operation for operation: inputs are cast to float32, targets and predictions
are clipped to [-2, 2], weights are ``abs(clipped target)``, arithmetic is
float64, a total weight below 1e-8 or a weighted standard deviation of at most
1e-8 gives 0, and the result is clipped to [-1, 1]. Nonfinite inputs raise.

``global_wp`` is the in-memory reference from METRIC.md. ``WPAccumulator``
computes the same score with bounded memory from chunks of any size, and
accumulators merge, so shards can be scored in parallel.
"""

from __future__ import annotations

import numpy as np

METRIC_CLIP = 2.0
EPS = 1e-8


def weighted_pearson(y_true, y_pred, *, mask=None) -> float:
    """Weighted Pearson correlation of one target over the selected rows."""
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} != {p.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != y.shape:
            raise ValueError(f"mask shape mismatch: {m.shape} != {y.shape}")
        y, p = y[m], p[m]
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("target/prediction contains nonfinite values")
    y = np.clip(y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.clip(p, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    w = np.abs(y)
    total = w.sum()
    if total < EPS:
        return 0.0
    yc = y - np.sum(w * y) / total
    pc = p - np.sum(w * p) / total
    covariance = np.sum(w * yc * pc) / total
    y_std = np.sqrt(np.sum(w * yc**2) / total)
    p_std = np.sqrt(np.sum(w * pc**2) / total)
    if y_std <= EPS or p_std <= EPS:
        return 0.0
    return float(np.clip(covariance / (y_std * p_std), -1.0, 1.0))


def global_weighted_pearson(y_true, y_pred, *, mask=None) -> float:
    """Mean of the two target-wise correlations (columns t0 and t1)."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.ndim != 2 or p.shape != y.shape or y.shape[1] != 2:
        raise ValueError("expected y_true and y_pred with shape (n_rows, 2)")
    return float(np.mean([weighted_pearson(y[:, k], p[:, k], mask=mask) for k in range(2)]))


def _check_rows(targets, predictions, need_prediction, is_scored):
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    need = np.asarray(need_prediction, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape:
        raise ValueError("expected targets and predictions of shape (N, 2)")
    scored = need if is_scored is None else np.asarray(is_scored, dtype=bool)
    if need.shape != (len(y),) or scored.shape != need.shape:
        raise ValueError("incorrect mask shape")
    if not np.isfinite(y).all() or not np.isfinite(p[need]).all():
        raise ValueError("nonfinite target or required prediction")
    return y, p, need & scored


def global_wp(targets, predictions, need_prediction, is_scored=None) -> float:
    """Score all rows of an evaluated dataset at once (METRIC.md reference).

    Rows count when ``need_prediction AND is_scored``. Without ``is_scored``,
    as for training data, every required row counts.
    """
    y, p, mask = _check_rows(targets, predictions, need_prediction, is_scored)
    if not mask.any():
        raise ValueError("no rows selected by the scoring mask")
    return float(np.mean([weighted_pearson(y[mask, k], p[mask, k]) for k in range(2)]))


class WPAccumulator:
    """Global WP from chunks of any size, by merging centered weighted moments.

    Chunks may split or span sequences; the score equals ``global_wp`` over
    all rows added. Merging moments, never per-chunk scores, keeps the
    between-chunk covariance that makes the metric global.
    """

    def __init__(self):
        self.selected_rows = 0
        self.weight = np.zeros(2)
        self.mean_y = np.zeros(2)
        self.mean_p = np.zeros(2)
        self.m2_y = np.zeros(2)
        self.m2_p = np.zeros(2)
        self.cross = np.zeros(2)

    def update(self, targets, predictions, need_prediction, is_scored=None) -> "WPAccumulator":
        y, p, mask = _check_rows(targets, predictions, need_prediction, is_scored)
        if not mask.any():
            return self
        y = np.clip(y[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(p[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        chunk = WPAccumulator()
        chunk.selected_rows = len(y)
        w = np.abs(y)
        chunk.weight = w.sum(axis=0)
        safe = np.where(chunk.weight > 0, chunk.weight, 1.0)
        chunk.mean_y = np.sum(w * y, axis=0) / safe
        chunk.mean_p = np.sum(w * p, axis=0) / safe
        yc, pc = y - chunk.mean_y, p - chunk.mean_p
        chunk.m2_y = np.sum(w * yc * yc, axis=0)
        chunk.m2_p = np.sum(w * pc * pc, axis=0)
        chunk.cross = np.sum(w * yc * pc, axis=0)
        return self.merge(chunk)

    def merge(self, other: "WPAccumulator") -> "WPAccumulator":
        """Fold another accumulator's rows into this one (parallel-variance merge)."""
        self.selected_rows += other.selected_rows
        total = self.weight + other.weight
        live = other.weight > 0
        share = np.divide(other.weight, total, out=np.zeros(2), where=live)
        correction = self.weight * share
        dy = other.mean_y - self.mean_y
        dp = other.mean_p - self.mean_p
        self.m2_y = np.where(live, self.m2_y + other.m2_y + dy * dy * correction, self.m2_y)
        self.m2_p = np.where(live, self.m2_p + other.m2_p + dp * dp * correction, self.m2_p)
        self.cross = np.where(live, self.cross + other.cross + dy * dp * correction, self.cross)
        self.mean_y = np.where(live, self.mean_y + dy * share, self.mean_y)
        self.mean_p = np.where(live, self.mean_p + dp * share, self.mean_p)
        self.weight = total
        return self

    def result(self) -> dict:
        if not self.selected_rows:
            raise ValueError("no rows selected by the scoring mask")
        score = np.zeros(2)
        for k in range(2):
            total = self.weight[k]
            if total < EPS:
                continue
            y_std = np.sqrt(max(0.0, self.m2_y[k] / total))
            p_std = np.sqrt(max(0.0, self.m2_p[k] / total))
            if y_std > EPS and p_std > EPS:
                score[k] = np.clip(self.cross[k] / total / (y_std * p_std), -1.0, 1.0)
        return {"t0": float(score[0]), "t1": float(score[1]),
                "weighted_pearson": float(score.mean()), "selected_rows": self.selected_rows}
