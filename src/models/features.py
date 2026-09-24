"""Engineered input features, computed inside the model and so inside the exported graph.

Training and serving run the same torch code, since the feature ops are part of
model.onnx; the solution only feeds raw rows. Each feature is an ``nn.Module``
registered by name in ``FEATURES``, with a ``width`` attribute giving how many
columns it appends to the 112 raw inputs.

Raw inputs are per-column rank-Gaussian transforms of the original book and
trade data (values near N(0, 1), saturating at +-5.199), so volume-like
columns can be negative and price-like columns are not ordered by level.
Column layout per instrument (i0 at offset 0, i1 at offset 52): price-like bid
0-10 and ask 11-21, volume-like bid 22-32 and ask 33-43, trade price 44-47,
trade volume 48-51. The additional features a0-a7 are columns 104-111.
"""

from __future__ import annotations

import torch
from torch import nn

N_RAW = 112
INSTRUMENTS = (0, 52)
P_BID, P_ASK = range(0, 11), range(11, 22)
V_BID, V_ASK = range(22, 33), range(33, 44)
DP, DV = range(44, 48), range(48, 52)
CLAMP = 10.0  # engineered features are clamped so no input row can blow up the recurrence

FEATURES: dict[str, type[nn.Module]] = {}


def register(cls: type[nn.Module]) -> type[nn.Module]:
    FEATURES[cls.name] = cls
    return cls


def columns(part: range, offset: int) -> list[int]:
    return [offset + i for i in part]


class FeatureLayer(nn.Module):
    """Append the named features to raw inputs of shape (..., 112)."""

    def __init__(self, names=()):
        super().__init__()
        unknown = [n for n in names if n not in FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}; known: {sorted(FEATURES)}")
        self.names = list(names)
        self.blocks = nn.ModuleList(FEATURES[n]() for n in self.names)
        self.width = N_RAW + sum(block.width for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.names:
            return x
        extra = torch.cat([block(x) for block in self.blocks], dim=-1).clamp(-CLAMP, CLAMP)
        return torch.cat([x, extra], dim=-1)
