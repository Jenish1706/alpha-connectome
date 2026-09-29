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
    def __init__(self, features=(), hidden: int = 128, layers: int = 2,
                 tanh_tau: float | None = None):
        super().__init__()
        self.input_dim, self.hidden, self.layers = N_RAW, hidden, layers
        # Optional soft clamp of the outputs into the metric's [-2, 2]: 2 tanh(z / tau).
        self.tanh_tau = tanh_tau
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
        out = self.reg_head(x)
        if self.tanh_tau:
            out = 2.0 * torch.tanh(out / self.tanh_tau)
        return out, new_state

    def step(self, x: torch.Tensor, state: list[torch.Tensor], scale: float = 1.0):
        """``forward`` for one row, x (1, 1, 112), written for a lean exported graph.

        For a single step a GRU's final state is its output, so each GRU op's
        state output feeds the next layer directly. Nothing is transposed for
        batch_first or squeezed off the op's direction axis, and the head
        absorbs 1/tanh_tau. This leaves the two GRU ops and four head ops.
        The output is multiplied by ``scale`` at no cost (for ensembles).
        """
        x = self.features(x)  # (1, 1, width): the same as (time, batch, features)
        new_state = []
        for block, h in zip(self.blocks, state, strict=True):
            gru = block["gru"]
            _, h = torch._VF.gru(x, h, gru._flat_weights, gru.bias, gru.num_layers, 0.0, False,
                                 gru.bidirectional, False)
            new_state.append(h)
            x = h
        weight, bias = self.reg_head.weight, self.reg_head.bias
        if self.tanh_tau:
            weight, bias = weight / self.tanh_tau, bias / self.tanh_tau
        elif scale != 1.0:
            weight, bias = weight * scale, bias * scale
        out = nn.functional.linear(x, weight, bias)
        if self.tanh_tau:
            out = (2.0 * scale) * torch.tanh(out)
        return out, new_state


def _onnx_gates_to_torch(w: np.ndarray, hidden: int) -> np.ndarray:
    """ONNX stacks GRU gates as (z, r, h); PyTorch as (r, z, n)."""
    z, r, n = w[:hidden], w[hidden:2 * hidden], w[2 * hidden:]
    return np.concatenate([r, z, n])


def _kept_units(weights: dict, grus: list, head, keep: int) -> list[np.ndarray]:
    """Units to keep per block when narrowing the baseline: the most-used ones.

    The last block's units are ranked by their output-head weights; each
    earlier block's units by the next block's input weights on them. Keeping
    every unit keeps them in order.
    """
    kept = [None] * len(grus)
    kept[-1] = np.argsort(-np.linalg.norm(weights[head.input[1]], axis=1))[:keep]
    for i in range(len(grus) - 2, -1, -1):
        w_next = weights[grus[i + 1].input[1]][0]  # (3 * hidden, hidden), columns = block i units
        kept[i] = np.argsort(-np.linalg.norm(w_next, axis=0))[:keep]
    return [np.sort(k) for k in kept]


def _gate_rows(units: np.ndarray, hidden: int) -> np.ndarray:
    return np.concatenate([g * hidden + units for g in range(3)])


def load_baseline_onnx(model: RecurrentRegressor, path: str | Path) -> RecurrentRegressor:
    """Copy the baseline's weights into ``model`` (zero weights for any extra inputs).

    A narrower model keeps the baseline's most-used hidden units (see
    ``_kept_units``), slicing every gate consistently.
    """
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    weights = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    grus = [n for n in graph.node if n.op_type == "GRU"]
    head = next(n for n in graph.node if n.op_type == "MatMul")
    if len(grus) != model.layers:
        raise ValueError(f"baseline has {len(grus)} GRU blocks, model has {model.layers}")
    base = weights[grus[0].input[2]].shape[-1]
    if model.hidden > base:
        raise ValueError(f"baseline width is {base}, model width is {model.hidden}")
    kept = _kept_units(weights, grus, head, model.hidden)
    with torch.no_grad():
        for i, (block, node) in enumerate(zip(model.blocks, grus, strict=True)):
            gru = block["gru"]
            w, r, b = (weights[name] for name in node.input[1:4])
            rows = _gate_rows(kept[i], base)
            w_ih = _onnx_gates_to_torch(w[0], base)[rows]
            if i > 0:
                w_ih = w_ih[:, kept[i - 1]]
            gru.weight_ih_l0.zero_()
            gru.weight_ih_l0[:, :w_ih.shape[1]] = torch.from_numpy(np.ascontiguousarray(w_ih))
            w_hh = _onnx_gates_to_torch(r[0], base)[rows][:, kept[i]]
            gru.weight_hh_l0.copy_(torch.from_numpy(np.ascontiguousarray(w_hh)))
            gru.bias_ih_l0.copy_(torch.from_numpy(_onnx_gates_to_torch(b[0, :3 * base], base)[rows]))
            gru.bias_hh_l0.copy_(torch.from_numpy(_onnx_gates_to_torch(b[0, 3 * base:], base)[rows]))
        head_w = weights[head.input[1]].T[:, kept[-1]]
        model.reg_head.weight.copy_(torch.from_numpy(np.ascontiguousarray(head_w)))
        model.reg_head.bias.copy_(torch.from_numpy(weights["reg_head.bias"].copy()))
    return model
