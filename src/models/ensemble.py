"""A weighted sum of models, exported as one graph: an average, or a model plus a residual.

Members run side by side on the same raw row, each with its own recurrent
state (possibly none); the package carries all the states. In the exported
one-row graph the weights fold into each member's output scale, so combining
costs one Add per extra member.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.features import N_RAW, FeatureLayer


class Ensemble(nn.Module):
    def __init__(self, members, weights):
        super().__init__()
        if len(members) != len(weights):
            raise ValueError(f"need one weight per member; got {weights}")
        self.members = nn.ModuleList(members)
        self.weights = [float(w) for w in weights]
        self.input_dim = N_RAW
        self.features = FeatureLayer([])  # members compute their own features
        self._states = [len(m.initial_state(1)) for m in members]

    def initial_state(self, batch: int) -> list[torch.Tensor]:
        return [s for member in self.members for s in member.initial_state(batch)]

    def _split(self, state):
        start = 0
        for count in self._states:
            yield state[start:start + count]
            start += count

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        out, new_state = 0, []
        for member, weight, own in zip(self.members, self.weights, self._split(state), strict=True):
            prediction, own = member(x, own)
            out = out + weight * prediction
            new_state += own
        return out, new_state

    def step(self, x: torch.Tensor, state: list[torch.Tensor]):
        """``forward`` for one row, with each weight folded into its member's output scale."""
        out, new_state = None, []
        for member, weight, own in zip(self.members, self.weights, self._split(state), strict=True):
            prediction, own = member.step(x, own, scale=weight)
            out = prediction if out is None else out + prediction
            new_state += own
        return out, new_state
