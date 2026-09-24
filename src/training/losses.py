"""Training losses over (batch, time, 2) predictions and targets.

``mask`` selects rows that count: rows needing a prediction (step >= 99).
"""

import torch

CLIP = 2.0


def mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error against targets clipped to the metric range."""
    error = (prediction - target.clamp(-CLIP, CLIP)) ** 2
    return error[mask].mean()


def pearson(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1 - weighted Pearson correlation, as the metric computes it, over the masked rows.

    Weights are |clip(target)| and targets are clipped to [-2, 2], like Global WP.
    Predictions are not clipped, so rows past the clip range still get gradients.
    Computed per target over all rows of the window, then averaged over t0 and t1.
    """
    y = target.clamp(-CLIP, CLIP)[mask]
    p = prediction[mask]
    w = y.abs()
    total = w.sum(0).clamp_min(1e-8)
    yc = y - (w * y).sum(0) / total
    pc = p - (w * p).sum(0) / total
    cov = (w * yc * pc).sum(0) / total
    var = ((w * yc * yc).sum(0) / total) * ((w * pc * pc).sum(0) / total)
    return 1 - (cov / var.clamp_min(1e-12).sqrt()).mean()


def hybrid(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """0.8 weighted Pearson loss + 0.2 MSE: the correlation objective with MSE anchoring scale."""
    return 0.8 * pearson(prediction, target, mask) + 0.2 * mse(prediction, target, mask)


LOSSES = {"mse": mse, "pearson": pearson, "hybrid": hybrid}
