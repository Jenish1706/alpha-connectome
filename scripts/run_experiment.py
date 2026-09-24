#!/usr/bin/env python3
"""Train, score, export and judge one experiment against the current champion.

    python scripts/run_experiment.py [--config configs/experiment.yaml]

Exit status: 0 accepted (configs/champion.json and runs/champion/ updated),
1 rejected, 2 failed (errors, broken bookkeeping, or a disallowed first run).

Protocol
  1. Fit on the leading sequences of datasets/train_head.parquet with a fixed
     schedule and keep the final weights. The last ``holdout`` sample sequences
     are scored along the way as a diagnostic only.
  2. Score Global WP on the full validation set (need_prediction AND is_scored).
     WP over every required row is recorded too, as a check that does not
     depend on the public mask.
  3. Export an ONNX package. Replay the first validation sequences through it
     row by row in an isolated subprocess started in the package directory,
     and require its predictions to match the batched ones.
  4. Latency: mean callback time from scripts/benchmark_latency.py over two
     validation sequences. The 60 us ceiling was set when the starter pack
     solution measured 54 us on the host of the time; hosts differ in speed, so
     two checks apply. As measured: the median of three unpinned runs must be at
     most 60 us. Rescaled to the 54 us host: five alternating pairs of candidate
     and starter pack runs, pinned to one core, give per-pair ratios, and
     54 us times their median must be at most 60 us.
  5. Accept when both latency checks pass, WP beats the champion, and a paired
     bootstrap over validation sequences gives P(candidate > champion) >= 0.95.
     The first champion can only be the untrained starter pack baseline.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.schema import SEQUENCE_LENGTH, WARMUP, iter_sequences  # noqa: E402
from src.export import export_package  # noqa: E402
from src.models.recurrent import RecurrentRegressor, load_baseline_onnx  # noqa: E402
from src.training.fit import TrainConfig, predict, sequence_stats, train  # noqa: E402
from src.utils.metric import EPS, WPAccumulator  # noqa: E402

RUNS = ROOT / "runs"
CHAMPION = ROOT / "configs" / "champion.json"
CHAMPION_DIR = RUNS / "champion"
TRAIN = ROOT / "datasets" / "train_head.parquet"
VALID = ROOT / "datasets" / "valid.parquet"
BASELINE = ROOT / "wnn_connectome_starterpack" / "baseline" / "baseline.onnx"
BENCHMARK = ROOT / "scripts" / "benchmark_latency.py"
REFERENCE_SOLUTION = ROOT / "wnn_connectome_starterpack" / "baseline" / "solution.py"
REFERENCE_US = 54.0  # the starter pack solution on the host where the ceiling was set
LATENCY_LIMIT_US = 60.0
CONFIDENCE = 0.95
BOOTSTRAP = 2000
CHECK_SEQUENCES = 4
CHECK_TOLERANCE = 1e-4
FIRST_CHAMPION = {"init": "baseline", "features": [], "model": {"hidden": 128, "layers": 2}}

REPLAY = r"""
import sys
from types import SimpleNamespace
import numpy as np
import solution
x, ids = np.load(sys.argv[1]), np.load(sys.argv[2])
model = solution.PredictionModel()  # one instance across sequences, like the scorer
out = np.empty((x.shape[0], x.shape[1] - WARMUP, 2), np.float32)
for s in range(x.shape[0]):
    for step in range(x.shape[1]):
        need = step >= WARMUP
        value = model.predict(SimpleNamespace(seq_ix=int(ids[s]), step_in_seq=step,
                                              need_prediction=need, state=x[s, step]))
        if not need:
            assert value is None, "prediction returned during warm-up"
            continue
        value = np.asarray(value, dtype=np.float32)
        assert value.shape == (2,) and np.isfinite(value).all(), f"bad prediction {value!r}"
        out[s, step - WARMUP] = value
np.save(sys.argv[3], out)
""".replace("WARMUP", str(WARMUP))


def log(message: str) -> None:
    print(message, flush=True)


def build_model(config: dict) -> RecurrentRegressor:
    model = RecurrentRegressor(features=config["features"], **config.get("model", {}))
    if config["init"] == "baseline":
        load_baseline_onnx(model, BASELINE)
    elif config["init"] != "scratch":
        raise ValueError(f"unknown init {config['init']!r}")
    return model


def wp_from_stats(stats: np.ndarray) -> np.ndarray:
    """Global WP from summed raw moments (..., 2, 6), with the scorer's thresholds."""
    w, sy, sp, syy, spp, syp = np.moveaxis(stats, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        my, mp = sy / w, sp / w
        vy, vp = np.maximum(syy / w - my * my, 0), np.maximum(spp / w - mp * mp, 0)
        ok = (w >= EPS) & (np.sqrt(vy) > EPS) & (np.sqrt(vp) > EPS)
        corr = np.where(ok, (syp / w - my * mp) / np.sqrt(vy * vp), 0.0)
    return np.clip(corr, -1, 1).mean(-1)


def paired_bootstrap(candidate: np.ndarray, champion: np.ndarray, seed: int = 0):
    """Resample validation sequences; return P(delta > 0) and the delta's spread."""
    n = len(candidate)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(row, minlength=n) for row in rng.integers(0, n, (BOOTSTRAP, n))])
    shape = (BOOTSTRAP, *candidate.shape[1:])
    delta = (wp_from_stats((counts @ candidate.reshape(n, -1)).reshape(shape))
             - wp_from_stats((counts @ champion.reshape(n, -1)).reshape(shape)))
    return float((delta > 0).mean()), float(delta.std())


def replay_package(package: Path, sequences: int) -> np.ndarray:
    """Required-row predictions of the package, fed row by row in a clean interpreter.

    The subprocess runs with -I in the package directory, so the package must be
    self-contained: nothing from this repository is importable there.
    """
    seqs = list(iter_sequences(VALID, range(sequences)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "x.npy", np.stack([s.features for s in seqs]))
        np.save(tmp / "ids.npy", np.array([s.seq_ix for s in seqs]))
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, '.')\n" + REPLAY,
                        str(tmp / "x.npy"), str(tmp / "ids.npy"), str(tmp / "out.npy")],
                       cwd=package, env=env, check=True, capture_output=True, text=True)
        return np.load(tmp / "out.npy")


def _callback_mean(solution: Path, core: int | None) -> float:
    command = [sys.executable, str(BENCHMARK), "--solution", str(solution),
               "--data", str(VALID), "--sequences", "2", "--json"]
    if core is not None:
        command = ["taskset", "-c", str(core), *command]
    out = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)["callback_us"]["mean"]


def measure_latency(package: Path) -> dict:
    """Raw unpinned latency, plus the ratio to the starter pack solution from pinned pairs."""
    candidate = package / "solution.py"
    raw = [_callback_mean(candidate, None) for _ in range(3)]
    core = max(os.sched_getaffinity(0))
    try:
        _callback_mean(REFERENCE_SOLUTION, core)
    except (subprocess.CalledProcessError, FileNotFoundError):
        core = None  # pinning unavailable: pair unpinned runs instead
    pairs = []
    for i in range(5):  # alternate the order so drift within a pair cancels
        first, second = (candidate, REFERENCE_SOLUTION) if i % 2 == 0 else (REFERENCE_SOLUTION, candidate)
        a, b = _callback_mean(first, core), _callback_mean(second, core)
        pairs.append((a, b) if first == candidate else (b, a))
    ratio = float(np.median([c / r for c, r in pairs]))
    return {"latency_us": float(np.median(raw)), "latency_runs_us": raw, "latency_pairs_us": pairs,
            "latency_core": core, "latency_ratio": ratio, "latency_scaled_us": ratio * REFERENCE_US,
            "latency_pinned_us": float(np.median([c for c, _ in pairs])),
            "reference_latency_us": float(np.median([r for _, r in pairs]))}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def run(config: dict, run_dir: Path) -> dict:
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    for name in untracked:  # new files are not in git_diff; keep their contents with the run
        target = run_dir / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target)
    record = {"git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
              "git_diff": git("diff", "HEAD"), "untracked_files": untracked,
              "train_file": {"row_groups": pq.ParquetFile(TRAIN).metadata.num_row_groups,
                             "bytes": TRAIN.stat().st_size}}
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    model = build_model(config)
    train_config = TrainConfig(**config.get("train", {}))
    started = time.time()
    if train_config.epochs > 0:
        log(f"training {config['name']} on {TRAIN.name}")
        model, history = train(model, TRAIN, train_config, seed=config["seed"], log=log)
        record["holdout"] = history.records
    record["train_seconds"] = round(time.time() - started, 1)
    torch.save(model.state_dict(), run_dir / "model.pt")

    log("scoring the full validation set")
    started = time.time()
    groups = list(range(pq.ParquetFile(VALID).metadata.num_row_groups))
    stats, first, all_rows = [], {}, WPAccumulator()
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP

    def keep(batch, predictions):
        stats.append(sequence_stats(batch.targets, predictions, batch.is_scored & need))
        all_rows.update(batch.targets.reshape(-1, 2), predictions.reshape(-1, 2),
                        np.broadcast_to(need, batch.is_scored.shape).reshape(-1))
        for i, group in enumerate(batch.groups):
            if group < CHECK_SEQUENCES:
                first[group] = predictions[i, WARMUP:].copy()

    result = predict(model, VALID, groups, on_batch=keep).result()
    stats = np.concatenate(stats)
    np.save(run_dir / "val_stats.npy", stats)
    record.update(wp=result["weighted_pearson"], t0=result["t0"], t1=result["t1"],
                  selected_rows=result["selected_rows"],
                  wp_all_rows=all_rows.result()["weighted_pearson"],
                  eval_seconds=round(time.time() - started, 1))
    summed = float(wp_from_stats(stats.sum(0)))
    if abs(summed - record["wp"]) > 1e-9:
        raise RuntimeError(f"moment statistics disagree with the accumulator: {summed} vs {record['wp']}")
    log(f"validation WP {record['wp']:.6f} (t0 {record['t0']:.6f}, t1 {record['t1']:.6f}); "
        f"all required rows {record['wp_all_rows']:.6f}")

    package = export_package(model, run_dir / "package")
    replayed = replay_package(package, CHECK_SEQUENCES)
    batched = np.stack([first[g] for g in range(CHECK_SEQUENCES)])
    record["export_max_abs_diff"] = float(np.abs(replayed - batched).max())
    log(f"exported package vs batched predictions: max abs diff {record['export_max_abs_diff']:.2e}")
    if record["export_max_abs_diff"] > CHECK_TOLERANCE:
        raise RuntimeError("exported package does not reproduce the scored predictions")

    record.update(measure_latency(package))
    log(f"latency {record['latency_us']:.2f} us unpinned; pinned {record['latency_pinned_us']:.2f} us vs "
        f"starter pack {record['reference_latency_us']:.2f} us, ratio {record['latency_ratio']:.3f}, "
        f"so {record['latency_scaled_us']:.2f} us on the 54 us host")
    return record


def decide(record: dict, champion: dict | None, run_dir: Path) -> tuple[bool, str]:
    summary = ""
    if champion is not None:  # computed whatever the outcome, so every attempt logs it
        stats = np.load(run_dir / "val_stats.npy")
        record["p_better"], record["delta_sd"] = paired_bootstrap(
            stats, np.load(CHAMPION_DIR / "val_stats.npy"))
        record["delta"] = record["wp"] - champion["wp"]
        record["champion"] = champion["name"]
        summary = (f"delta {record['delta']:+.6f} (sd {record['delta_sd']:.6f}), "
                   f"P(better) {record['p_better']:.3f}")
    slow = [k for k in ("latency_us", "latency_scaled_us") if record[k] > LATENCY_LIMIT_US]
    record["latency_ok"] = not slow
    if slow:
        return False, (f"{slow[0]} {record[slow[0]]:.2f} us exceeds {LATENCY_LIMIT_US:.0f} us"
                       + (f"; {summary}" if summary else ""))
    if champion is None:
        return True, "first champion: the untrained starter pack baseline"
    if record["delta"] <= 0 or record["p_better"] < CONFIDENCE:
        return False, f"not better than {champion['name']}: {summary}"
    return True, f"beats {champion['name']}: {summary}"


def load_champion() -> dict | None:
    """The champion record, checked against runs/champion/; raises if they disagree."""
    have_record, have_dir = CHAMPION.exists(), (CHAMPION_DIR / "result.json").exists()
    if not have_record and not have_dir:
        return None
    if have_record != have_dir:
        raise RuntimeError(f"{CHAMPION.relative_to(ROOT)} and {CHAMPION_DIR.relative_to(ROOT)}/ "
                           "disagree: one is missing. Restore both, or rerun the champion's config.")
    champion = json.loads(CHAMPION.read_text())
    stored = json.loads((CHAMPION_DIR / "result.json").read_text())
    if stored.get("run_dir") != champion.get("run_dir") or not (CHAMPION_DIR / "val_stats.npy").exists():
        raise RuntimeError(f"{CHAMPION.relative_to(ROOT)} names {champion.get('run_dir')} but "
                           f"{CHAMPION_DIR.relative_to(ROOT)}/ holds {stored.get('run_dir')}")
    return champion


def is_first_champion(config: dict) -> bool:
    return ({k: config.get(k) for k in FIRST_CHAMPION} == FIRST_CHAMPION
            and config.get("train", {}).get("epochs", 0) == 0)


def install_champion(record: dict, run_dir: Path) -> None:
    """Swap run_dir in as runs/champion/ and rewrite configs/champion.json."""
    fresh, old = RUNS / "champion.new", RUNS / "champion.old"
    for leftover in (fresh, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.copytree(run_dir, fresh)
    if CHAMPION_DIR.exists():
        CHAMPION_DIR.rename(old)
    fresh.rename(CHAMPION_DIR)
    if old.exists():
        shutil.rmtree(old)
    keys = ("name", "description", "wp", "t0", "t1", "wp_all_rows", "latency_us", "latency_scaled_us",
            "latency_ratio", "latency_pinned_us", "reference_latency_us", "run_dir", "config",
            "finished")
    part = CHAMPION.with_suffix(".json.part")
    part.write_text(json.dumps({k: record.get(k) for k in keys}, indent=2, default=float) + "\n")
    part.replace(CHAMPION)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    args = ap.parse_args(argv)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another run_experiment.py is running")
            return 2
        return _main(args.config)


def _main(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = RUNS / f"{stamp}-{config['name']}"
    run_dir.mkdir(parents=True)
    record = {"name": config["name"], "description": config.get("description", ""), "config": config,
              "run_dir": str(run_dir.relative_to(ROOT)),
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        champion = load_champion()
        if champion is None and not is_first_champion(config):
            raise RuntimeError("no champion yet: register the untrained baseline first "
                               "(init: baseline, features: [], hidden 128, layers 2, epochs: 0)")
        record.update(run(config, run_dir))
        accepted, reason = decide(record, champion, run_dir)
        status = 0 if accepted else 1
    except Exception as exc:  # a failed run is logged and reported, never accepted
        record["error"] = "".join(traceback.format_exception(exc))
        accepted, reason, status = False, f"failed: {exc}", 2
        log(record["error"])
    record.update(decision=["ACCEPTED", "REJECTED", "FAILED"][status], reason=reason,
                  finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        (run_dir / "result.json").write_text(json.dumps(record, indent=2, default=float))
        if accepted:
            install_champion(record, run_dir)
    except Exception as exc:
        record.update(decision="FAILED", reason=f"failed while recording: {exc}")
        accepted, status, reason = False, 2, record["reason"]
        log("".join(traceback.format_exception(exc)))
        (run_dir / "result.json").write_text(json.dumps(record, indent=2, default=float))
    with open(RUNS / "experiments.jsonl", "a") as out:
        out.write(json.dumps(record, default=float) + "\n")
    wp = record.get("wp")
    wp_text = "n/a" if wp is None else f"{wp:.6f}"
    raw, scaled = (record.get(k, float("nan")) for k in ("latency_us", "latency_scaled_us"))
    log(f"RESULT {record['decision']} {config.get('name')}: WP {wp_text}, latency {raw:.2f} us "
        f"({scaled:.2f} us scaled); {reason}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # anything unexpected is a failure, never a rejection
        traceback.print_exc()
        sys.exit(2)
