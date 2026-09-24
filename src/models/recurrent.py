"""Stacked recurrent regressor, weight-compatible with the starter pack GRU baseline.

The baseline ONNX (``baseline/baseline.onnx``) is two single-layer GRU blocks
of width 128 (PyTorch gate convention, ``linear_before_reset=1``) followed by
a linear head, fed the raw 112 features. ``load_baseline_onnx`` copies its
weights into this module so experiments can warm-start from it; engineered
features (``src/models/features.py``) get zero input weights, so the
warm-started model initially reproduces the baseline exactly. The model always
takes the 112 raw inputs; features are computed inside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.features import N_RAW, FeatureLayer


class RecurrentRegressor(nn.Module):
    def __init__(self, features=(), hidden: int = 128, layers: int = 2):
        super().__init__()
        self.input_dim, self.hidden, self.layers = N_RAW, hidden, layers
        self.features = FeatureLayer(features)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"gru": nn.GRU(self.features.width if i == 0 else hidden, hidden,
                                         batch_first=True)})
            for i in range(layers))
        self.reg_head = nn.Linear(hidden, 2)

    def initial_state(self, batch: int) -> list[torch.Tensor]:
        return [torch.zeros(1, batch, self.hidden) for _ in range(self.layers)]

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        """x: (batch, time, 112) raw inputs; state: per-block (1, batch, hidden)."""
        x = self.features(x)
        new_state = []
        for block, h in zip(self.blocks, state, strict=True):
            x, h = block["gru"](x, h)
            new_state.append(h)
        return self.reg_head(x), new_state


def _onnx_gates_to_torch(w: np.ndarray, hidden: int) -> np.ndarray:
    """ONNX stacks GRU gates as (z, r, h); PyTorch as (r, z, n)."""
    z, r, n = w[:hidden], w[hidden:2 * hidden], w[2 * hidden:]
    return np.concatenate([r, z, n])


def load_baseline_onnx(model: RecurrentRegressor, path: str | Path) -> RecurrentRegressor:
    """Copy the baseline's weights into ``model`` (zero weights for any extra inputs)."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    weights = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    grus = [n for n in graph.node if n.op_type == "GRU"]
    head = next(n for n in graph.node if n.op_type == "MatMul")
    if len(grus) != model.layers:
        raise ValueError(f"baseline has {len(grus)} GRU blocks, model has {model.layers}")
    with torch.no_grad():
        for block, node in zip(model.blocks, grus, strict=True):
            gru, hidden = block["gru"], model.hidden
            w, r, b = (weights[name] for name in node.input[1:4])
            if r.shape[-1] != hidden:
                raise ValueError(f"baseline width is {r.shape[-1]}, model width is {hidden}")
            w_ih = _onnx_gates_to_torch(w[0], hidden)
            gru.weight_ih_l0.zero_()
            gru.weight_ih_l0[:, :w_ih.shape[1]] = torch.from_numpy(w_ih)
            gru.weight_hh_l0.copy_(torch.from_numpy(_onnx_gates_to_torch(r[0], hidden)))
            gru.bias_ih_l0.copy_(torch.from_numpy(_onnx_gates_to_torch(b[0, :3 * hidden], hidden)))
            gru.bias_hh_l0.copy_(torch.from_numpy(_onnx_gates_to_torch(b[0, 3 * hidden:], hidden)))
        model.reg_head.weight.copy_(torch.from_numpy(weights[head.input[1]].T.copy()))
        model.reg_head.bias.copy_(torch.from_numpy(weights["reg_head.bias"].copy()))
    return model
