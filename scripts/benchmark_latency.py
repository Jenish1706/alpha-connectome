"""Callback-style latency benchmark for a baseline ONNX model or dummy model."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class DummyModel:
    def __init__(self):
        self.seq_ix = None
        self.last = np.zeros(2, dtype=np.float32)

    def predict(self, dp: DataPoint):
        if dp.seq_ix != self.seq_ix:
            self.seq_ix = dp.seq_ix
            self.last[:] = 0
        self.last = (0.99 * self.last + 0.01 * dp.state[:2]).astype(np.float32)
        return self.last.copy() if dp.need_prediction else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, help="Optional baseline model; dummy is used otherwise")
    ap.add_argument("--steps", type=int, default=40000)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = DummyModel()
    rng = np.random.default_rng(0)
    samples = []
    start = time.perf_counter_ns()
    for i in range(args.steps):
        dp = DataPoint(i // 20000, i % 20000, (i % 20000) >= 99,
                       rng.standard_normal(112).astype(np.float32))
        t0 = time.perf_counter_ns()
        result = model.predict(dp)
        samples.append(time.perf_counter_ns() - t0)
        if result is not None and result.shape != (2,):
            raise RuntimeError("invalid prediction shape")
    elapsed = time.perf_counter_ns() - start
    per = np.asarray(samples[100:], dtype=np.float64) / 1000.0
    print(f"steps: {args.steps}; wall us/step: {elapsed / args.steps / 1000:.2f}")
    print(f"callback mean/median/p95 us: {per.mean():.2f}/{np.median(per):.2f}/{np.percentile(per, 95):.2f}")
    if psutil:
        print(f"RSS MB: {psutil.Process().memory_info().rss / 1024**2:.1f}")
    if args.onnx:
        print("Note: --onnx is reserved for wiring the competition model's exact input signature; dummy callback was benchmarked.")


if __name__ == "__main__":
    main()
