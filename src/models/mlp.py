"""A stateless regressor on the current raw row: an ensemble member that costs no recurrent state.

At batch 1 every GRU op with its state costs 5-10 us in the package, whatever
its width (see experiments.md, Axis 4), so a second recurrent member does not
fit the latency ceiling. A small MLP does.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.features import N_RAW, FeatureLayer


class RowMLP(nn.Module):
    """2 tanh(W2 relu(W1 x + b1) + b2) / tau) on each row alone.

    The output layer starts at zero, so the model starts by predicting 0: as a
    residual member it leaves its partner's predictions exactly as they were.
    """

    def __init__(self, width: int = 64, tanh_tau: float = 8.0):
        super().__init__()
        self.input_dim, self.width, self.tanh_tau = N_RAW, width, tanh_tau
        self.features = FeatureLayer([])
        self.hidden_layer = nn.Linear(N_RAW, width)
        self.out = nn.Linear(width, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def initial_state(self, batch: int) -> list[torch.Tensor]:
        return []

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        z = self.out(torch.relu(self.hidden_layer(x)))
        return 2.0 * torch.tanh(z / self.tanh_tau), []

    def step(self, x: torch.Tensor, state: list[torch.Tensor], scale: float = 1.0):
        """``forward`` for one row, with 1/tanh_tau folded into the output layer."""
        weight, bias = self.out.weight / self.tanh_tau, self.out.bias / self.tanh_tau
        z = nn.functional.linear(torch.relu(self.hidden_layer(x)), weight, bias)
        return (2.0 * scale) * torch.tanh(z), []
