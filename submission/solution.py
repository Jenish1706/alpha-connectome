"""Minimal deterministic submission scaffold."""
import numpy as np


class PredictionModel:
    def __init__(self):
        self.seq_ix = None
        self.ema = np.zeros(2, dtype=np.float32)

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.ema[:] = 0
        x = np.asarray(data_point.state, dtype=np.float32)
        # Replace with an exported model once trained. This is finite and fast.
        self.ema = (0.995 * self.ema + 0.005 * x[:2]).astype(np.float32)
        if not data_point.need_prediction:
            return None
        return np.clip(self.ema, -2.0, 2.0).astype(np.float32, copy=True)
