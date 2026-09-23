"""Replay rows through a solution's callback, check its outputs and time it.

    python scripts/benchmark_latency.py                    # submission/solution.py, synthetic rows
    python scripts/benchmark_latency.py --data datasets/valid.parquet --sequences 2 --score
    python scripts/benchmark_latency.py \\
        --solution wnn_connectome_starterpack/baseline/solution.py \\
        --data datasets/valid.parquet --sequences 0 --score   # 0 = every sequence

Rows reach ``PredictionModel.predict`` the way the scorer sends them: one
DataPoint per row, in order, sequence by sequence. Outputs must follow the
submission contract: None during warm-up, two finite values on every required
row. ``--score`` adds Global WP over ``need_prediction AND is_scored`` (every
required row for training files). The scorer allows 60 minutes for the whole
test set, whose size is hidden; the projection assumes it matches validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # the scorer runs on one vCPU

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import (N_FEATURES, SEQUENCE_LENGTH, SEQUENCES, WARMUP,  # noqa: E402
                             DataPoint, Kind, Sequence, iter_sequences)
from src.utils.metric import WPAccumulator  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

TIME_LIMIT_MINUTES = 60
PROJECTED_ROWS = SEQUENCES[Kind.VALID] * SEQUENCE_LENGTH
WARM_CALLS = 100  # first calls excluded from latency statistics


def load_solution(path: Path):
    """Import a solution.py the way the scorer does and build its model."""
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module.PredictionModel()


def synthetic_sequences(rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP
    for seq, start in enumerate(range(0, rows, SEQUENCE_LENGTH)):
        n = min(SEQUENCE_LENGTH, rows - start)
        features = rng.standard_normal((n, N_FEATURES), dtype=np.float32)
        yield Sequence(seq, features, need[:n], None, None)


def replay(model, sequences, *, score: bool = False):
    """Feed every row to ``model``; return (latencies in ns, rows, accumulator or None)."""
    latencies = []
    accumulator = WPAccumulator() if score else None
    for seq in sequences:
        n = len(seq.features)
        spent = np.empty(n, dtype=np.int64)
        predictions = np.zeros((n, 2), dtype=np.float32)
        for step in range(n):
            need = bool(seq.need_prediction[step])
            point = DataPoint(seq.seq_ix, step, need, seq.features[step])
            start = time.perf_counter_ns()
            value = model.predict(point)
            spent[step] = time.perf_counter_ns() - start
            if not need:
                if value is not None:
                    raise ValueError(f"seq {seq.seq_ix} step {step}: return None during warm-up")
                continue
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"seq {seq.seq_ix} step {step}: need two finite values, got {value!r}")
            predictions[step] = value
        latencies.append(spent)
        if accumulator is not None:
            if seq.targets is None:
                raise ValueError("--score needs a file with t0 and t1")
            accumulator.update(seq.targets, predictions, seq.need_prediction, seq.is_scored)
    spent = np.concatenate(latencies) if latencies else np.empty(0, dtype=np.int64)
    return spent, int(spent.size), accumulator


def summarize(latencies_ns: np.ndarray, rows: int, wall_s: float) -> dict:
    per = latencies_ns[min(WARM_CALLS, rows // 10):] / 1000.0
    mean = float(per.mean())
    report = {
        "rows": rows,
        "wall_us_per_row": wall_s * 1e6 / rows,
        "callback_us": {"mean": mean, "median": float(np.median(per)),
                        "p95": float(np.percentile(per, 95)),
                        "p99": float(np.percentile(per, 99)), "max": float(per.max())},
        "projected_minutes": mean * PROJECTED_ROWS / 60e6,
        "projected_rows": PROJECTED_ROWS,
        "time_limit_minutes": TIME_LIMIT_MINUTES,
    }
    if psutil:
        report["rss_mb"] = psutil.Process().memory_info().rss / 1024**2
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solution", type=Path, default=ROOT / "submission" / "solution.py")
    ap.add_argument("--data", type=Path, help="train- or valid-layout parquet; synthetic rows if omitted")
    ap.add_argument("--sequences", type=int, default=1, help="sequences to replay from --data; 0 = all")
    ap.add_argument("--steps", type=int, default=40_000, help="synthetic rows when --data is omitted")
    ap.add_argument("--score", action="store_true", help="report Global WP (needs --data)")
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    args = ap.parse_args(argv)
    if args.score and args.data is None:
        ap.error("--score needs --data")
    if args.steps < 1 or args.sequences < 0:
        ap.error("--steps must be positive and --sequences non-negative")

    model = load_solution(args.solution)
    if args.data is None:
        sequences = synthetic_sequences(args.steps)
    else:
        import pyarrow.parquet as pq
        available = pq.ParquetFile(args.data).metadata.num_row_groups
        count = available if args.sequences == 0 else min(args.sequences, available)
        sequences = iter_sequences(args.data, range(count))

    start = time.perf_counter()
    latencies, rows, accumulator = replay(model, sequences, score=args.score)
    report = summarize(latencies, rows, time.perf_counter() - start)
    report["solution"] = str(args.solution)
    report["data"] = str(args.data) if args.data else "synthetic"
    if accumulator is not None:
        report["score"] = accumulator.result()

    if args.json:
        print(json.dumps(report))
        return 0
    cb = report["callback_us"]
    print(f"solution: {report['solution']}\ndata: {report['data']}; rows: {rows:,}")
    print(f"callback us mean/median/p95/p99/max: {cb['mean']:.2f}/{cb['median']:.2f}/"
          f"{cb['p95']:.2f}/{cb['p99']:.2f}/{cb['max']:.2f}; wall us/row: {report['wall_us_per_row']:.2f}")
    print(f"projected callback time for {PROJECTED_ROWS:,} rows: "
          f"{report['projected_minutes']:.1f} of {TIME_LIMIT_MINUTES} min")
    if "rss_mb" in report:
        print(f"RSS MB: {report['rss_mb']:.1f}")
    if accumulator is not None:
        s = report["score"]
        print(f"Global WP: {s['weighted_pearson']:.6f} (t0 {s['t0']:.6f}, t1 {s['t1']:.6f}; "
              f"{s['selected_rows']:,} scored rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
