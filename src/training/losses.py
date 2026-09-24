"""Training losses over (batch, time, 2) predictions and targets.

``mask`` selects rows that count: rows needing a prediction (step >= 99).
"""

import torch

CLIP = 2.0


def mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error against targets clipped to the metric range."""
    error = (prediction - target.clamp(-CLIP, CLIP)) ** 2
    return error[mask].mean()


LOSSES = {"mse": mse}
