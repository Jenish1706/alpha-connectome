"""Export a trained model as a self-contained submission package.

A package directory holds solution.py (from src/solution_template.py),
model.onnx (a one-step graph with explicit recurrent state; engineered
features are part of the graph) and model.json. Zipping the directory gives an
uploadable submission.

Export traces the model, so Python branches that depend on tensor values get
frozen. Tracer warnings raised from this repository's code are errors;
recurrent state must be float32 tensors (write complex recurrences with real
and imaginary parts).
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import torch
from torch import nn

SRC = Path(__file__).resolve().parent


class _Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *state):
        prediction, new_state = self.model(features, list(state))
        return (prediction, *new_state)


def export_package(model, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    state = model.initial_state(1)
    if any(s.dtype != torch.float32 for s in state):
        raise TypeError("recurrent state must be float32 tensors")
    names = [f"hidden_{i}" for i in range(len(state))]
    example = torch.zeros(1, 1, model.input_dim)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(_Step(model), (example, *state), directory / "model.onnx",
                          input_names=["features", *names],
                          output_names=["prediction", *[f"next_{n}" for n in names]],
                          opset_version=17, dynamo=False)
    # nn.GRU's own shape checks warn from site-packages; anything else is a frozen branch.
    ours = [w for w in caught if issubclass(w.category, torch.jit.TracerWarning)
            and "site-packages" not in str(w.filename)]
    if ours:
        raise RuntimeError(f"tracing froze a value-dependent branch: {ours[0].filename}:{ours[0].lineno} "
                           f"{ours[0].message}")
    (directory / "model.json").write_text(json.dumps({
        "features": list(model.features.names), "state_names": names,
        "state_shapes": [list(s.shape) for s in state]}, indent=2))
    shutil.copy(SRC / "solution_template.py", directory / "solution.py")
    return directory
