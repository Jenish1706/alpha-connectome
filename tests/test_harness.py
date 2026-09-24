"""Experiment harness: features, export parity, bootstrap and training loop."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.data.schema import SEQUENCE_LENGTH, WARMUP, DataPoint, Kind, iter_sequences
from src.export import export_package
from src.models.features import N_RAW, FeatureLayer
from src.models.recurrent import RecurrentRegressor, load_baseline_onnx
from src.training.fit import TrainConfig, predict, sequence_stats, train
from src.utils.metric import global_wp

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_no_features_is_identity():
    x = torch.randn(3, 5, N_RAW)
    layer = FeatureLayer([])
    assert layer(x) is x and layer.width == N_RAW
    with pytest.raises(ValueError, match="unknown features"):
        FeatureLayer(["nope"])


def test_moment_statistics_reproduce_global_wp():
    run = _script("run_experiment")
    rng = np.random.default_rng(1)
    y = (rng.standard_normal((4, SEQUENCE_LENGTH, 2)) * 1.5).astype(np.float32)
    p = (0.3 * y + rng.standard_normal(y.shape)).astype(np.float32)
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP
    scored = need & (rng.random((4, SEQUENCE_LENGTH)) < 0.1)
    stats = sequence_stats(y, p, scored)
    expected = global_wp(y.reshape(-1, 2), p.reshape(-1, 2), np.tile(need, 4), scored.reshape(-1))
    assert float(run.wp_from_stats(stats.sum(0))) == pytest.approx(expected, abs=1e-12)

    same, _ = run.paired_bootstrap(stats, stats)
    assert same == 0.0  # identical predictions never count as better
    better = sequence_stats(y, (0.6 * y + rng.standard_normal(y.shape)).astype(np.float32), scored)
    assert run.paired_bootstrap(better, stats)[0] > 0.99


def test_exported_package_matches_torch(tmp_path):
    bench = _script("benchmark_latency")
    torch.manual_seed(0)
    model = RecurrentRegressor(hidden=16).eval()
    package = export_package(model, tmp_path / "package")
    solution = bench.load_solution(package / "solution.py")
    x = np.random.default_rng(2).standard_normal((300, N_RAW)).astype(np.float32)
    with torch.no_grad():
        expected, _ = model(torch.from_numpy(x)[None], model.initial_state(1))
    got = [solution.predict(DataPoint(5, i, i >= WARMUP, x[i])) for i in range(300)]
    assert all(g is None for g in got[:WARMUP])
    assert np.abs(np.stack(got[WARMUP:]) - expected[0, WARMUP:].numpy()).max() < 1e-5


def test_baseline_weights_port_exactly(valid_path, baseline_solution):
    onnx_path = baseline_solution.parent / "baseline.onnx"
    model = load_baseline_onnx(RecurrentRegressor(), onnx_path).eval()
    x = next(iter_sequences(valid_path, [0])).features[:500]
    with torch.no_grad():
        ours, _ = model(torch.from_numpy(x)[None], model.initial_state(1))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    h0 = h1 = np.zeros((1, 1, 128), np.float32)
    reference = []
    for row in x:
        out, h0, h1 = session.run(None, {"features": row.reshape(1, 1, -1), "hidden_0": h0, "hidden_1": h1})
        reference.append(out[0, 0])
    assert np.abs(ours[0].numpy() - np.stack(reference)).max() < 1e-5


def test_training_keeps_the_best_holdout_checkpoint(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2, 3))
    torch.manual_seed(0)
    model = RecurrentRegressor(hidden=8)
    config = TrainConfig(epochs=1, batch_sequences=2, chunk=5_000, fit_sequences=2, holdout=1,
                         eval_every=1, lr=1e-3)
    model, history = train(model, path, config, seed=0, log=lambda *_: None)
    assert [r["step"] for r in history.records] == [0, 4]
    rescored = predict(model, path, [2]).result()["weighted_pearson"]
    assert rescored == pytest.approx(history.records[-1]["holdout_wp"], abs=1e-9)  # final weights kept


def test_export_rejects_value_dependent_branches(tmp_path):
    class Guarded(RecurrentRegressor):
        def forward(self, x, state):
            if x.abs().max() > 4:  # frozen by tracing: must not export silently
                x = x.clamp(-4, 4)
            return super().forward(x, state)

    with pytest.raises(RuntimeError, match="froze a value-dependent branch"):
        export_package(Guarded(hidden=8), tmp_path / "package")


def test_first_champion_must_be_the_untrained_baseline():
    run = _script("run_experiment")
    base = {"init": "baseline", "features": [], "model": {"hidden": 128, "layers": 2},
            "train": {"epochs": 0}}
    assert run.is_first_champion(base)
    assert not run.is_first_champion({**base, "features": ["ofi"]})
    assert not run.is_first_champion({**base, "train": {"epochs": 1}})


def test_champion_bookkeeping_must_agree(tmp_path, monkeypatch):
    run = _script("run_experiment")
    record, directory = tmp_path / "champion.json", tmp_path / "champion"
    monkeypatch.setattr(run, "CHAMPION", record)
    monkeypatch.setattr(run, "CHAMPION_DIR", directory)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    assert run.load_champion() is None
    record.write_text('{"name": "a", "run_dir": "runs/a", "wp": 0.6}')
    with pytest.raises(RuntimeError, match="one is missing"):
        run.load_champion()
    directory.mkdir()
    (directory / "result.json").write_text('{"run_dir": "runs/b"}')
    np.save(directory / "val_stats.npy", np.zeros((1, 2, 6)))
    with pytest.raises(RuntimeError, match="names runs/a"):
        run.load_champion()
    (directory / "result.json").write_text('{"run_dir": "runs/a"}')
    assert run.load_champion()["name"] == "a"


def test_training_rejects_bad_schedules(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2, 3))
    model = RecurrentRegressor(hidden=8)
    with pytest.raises(ValueError, match="without scored rows"):
        train(model, path, TrainConfig(epochs=1, chunk=99, fit_sequences=2, holdout=1), seed=0)
    with pytest.raises(ValueError, match="do not fit"):
        train(model, path, TrainConfig(epochs=1, fit_sequences=3, holdout=1), seed=0)
    with pytest.raises(ValueError, match="cannot fill a batch"):
        train(model, path, TrainConfig(epochs=1, batch_sequences=4, fit_sequences=2, holdout=1), seed=0)
