"""Stateful ONNX solution written by the experiment harness (src/export.py).

Ships with model.onnx and model.json in the same directory. Engineered
features are computed inside model.onnx, so each call only feeds the raw row.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = Path(__file__).resolve().parent


class PredictionModel:
    def __init__(self):
        config = json.loads((HERE / "model.json").read_text())
        self.state_names = config["state_names"]
        self.state_shapes = [tuple(s) for s in config["state_shapes"]]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(HERE / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.seq_ix = None
        self.states = []

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]
        feeds = dict(zip(self.state_names, self.states, strict=True))
        feeds["features"] = np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, -1)
        prediction, *self.states = self.session.run(None, feeds)
        if not data_point.need_prediction:
            return None
        return prediction[0, 0]
