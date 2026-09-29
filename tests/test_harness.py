"""Experiment harness: features, export parity, bootstrap and training loop."""

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.data.schema import SEQUENCE_LENGTH, WARMUP, DataPoint, Kind, iter_sequences
from src.export import export_package
from src.models.features import N_RAW, FeatureLayer
from src.models.recurrent import RecurrentRegressor, load_baseline_onnx
from src.training.fit import TrainConfig, predict, sequence_stats, train
from src.training.losses import hybrid, mse, pearson
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


@pytest.mark.parametrize("options", [{}, {"tanh_tau": 8.0}])
def test_exported_package_matches_torch(tmp_path, options):
    bench = _script("benchmark_latency")
    torch.manual_seed(0)
    model = RecurrentRegressor(hidden=16, **options).eval()
    package = export_package(model, tmp_path / "package")
    ops = {n.op_type for n in onnx.load(str(package / "model.onnx")).graph.node} - {"Constant"}
    assert not ops & {"Transpose", "Squeeze", "Div"}  # the lean one-row graph
    solution = bench.load_solution(package / "solution.py")
    x = np.random.default_rng(2).standard_normal((300, N_RAW)).astype(np.float32)
    with torch.no_grad():
        expected, _ = model(torch.from_numpy(x)[None], model.initial_state(1))
    got = [solution.predict(DataPoint(5, i, i >= WARMUP, x[i])) for i in range(300)]
    assert all(g is None for g in got[:WARMUP])
    assert np.abs(np.stack(got[WARMUP:]) - expected[0, WARMUP:].numpy()).max() < 1e-5


def test_export_checks_the_one_row_path(tmp_path):
    class Broken(RecurrentRegressor):
        def forward(self, x, state):
            return super().forward(x, state)

        def step(self, x, state):
            out, state = super().step(x, state)
            return out + 1e-3, state

    with pytest.raises(RuntimeError, match="disagrees with forward"):
        export_package(Broken(hidden=8), tmp_path / "package")


def test_ensembles_average_their_members_and_export(tmp_path):
    run = _script("run_experiment")
    bench = _script("benchmark_latency")
    torch.manual_seed(0)
    config = {"features": [], "init": "scratch", "model": {"hidden": 8, "tanh_tau": 8.0}}
    partner = run.build_model(config).eval()
    (tmp_path / "partner").mkdir()
    torch.save(partner.state_dict(), tmp_path / "partner" / "model.pt")
    (tmp_path / "partner" / "result.json").write_text(json.dumps({"config": config}))
    member = run.build_model({**config, "model": {"hidden": 4, "layers": 1}}).eval()
    ensemble_config = {**config, "model": {"hidden": 4, "layers": 1},
                       "ensemble_with": str(tmp_path / "partner"), "ensemble_weight": 0.25}
    model = run.assemble(member, ensemble_config)
    x = torch.randn(2, 300, N_RAW)
    with torch.no_grad():
        got, _ = model(x, model.initial_state(2))
        a, _ = partner(x, partner.initial_state(2))
        b, _ = member(x, member.initial_state(2))
    torch.testing.assert_close(got, 0.75 * a + 0.25 * b)

    package = export_package(model, tmp_path / "package")  # also checks step against forward
    ops = Counter(n.op_type for n in onnx.load(str(package / "model.onnx")).graph.node)
    assert ops["GRU"] == 3 and ops["Add"] == 3 and "Div" not in ops  # two heads' biases, one average
    solution = bench.load_solution(package / "solution.py")
    rows = x[0].numpy()
    replayed = [solution.predict(DataPoint(3, i, i >= WARMUP, rows[i])) for i in range(300)]
    assert np.abs(np.stack(replayed[WARMUP:]) - got[0, WARMUP:].numpy()).max() < 1e-5

    (tmp_path / "ensemble").mkdir()  # a finished ensemble run rebuilds from its record
    torch.save(model.state_dict(), tmp_path / "ensemble" / "model.pt")
    (tmp_path / "ensemble" / "result.json").write_text(json.dumps({"config": ensemble_config}))
    with torch.no_grad():
        again, _ = (rebuilt := run.load_run_model(tmp_path / "ensemble"))(x, rebuilt.initial_state(2))
    torch.testing.assert_close(again, got, rtol=0, atol=0)


def test_narrow_warm_start_keeps_the_most_used_units(baseline_solution):
    onnx_path = baseline_solution.parent / "baseline.onnx"
    full = load_baseline_onnx(RecurrentRegressor(), onnx_path)
    narrow = load_baseline_onnx(RecurrentRegressor(hidden=64), onnx_path)
    k1 = full.reg_head.weight.norm(dim=0).argsort(descending=True)[:64].sort().values
    k0 = full.blocks[1]["gru"].weight_ih_l0.norm(dim=0).argsort(descending=True)[:64].sort().values

    def rows(k):
        return torch.cat([g * 128 + k for g in range(3)])

    first, second = (block["gru"] for block in full.blocks)
    small0, small1 = (block["gru"] for block in narrow.blocks)
    torch.testing.assert_close(narrow.reg_head.weight, full.reg_head.weight[:, k1], rtol=0, atol=0)
    torch.testing.assert_close(small1.weight_hh_l0, second.weight_hh_l0[rows(k1)][:, k1], rtol=0, atol=0)
    torch.testing.assert_close(small1.weight_ih_l0, second.weight_ih_l0[rows(k1)][:, k0], rtol=0, atol=0)
    torch.testing.assert_close(small0.weight_ih_l0, first.weight_ih_l0[rows(k0)], rtol=0, atol=0)
    torch.testing.assert_close(small0.bias_hh_l0, first.bias_hh_l0[rows(k0)], rtol=0, atol=0)
    with pytest.raises(ValueError, match="baseline width"):
        load_baseline_onnx(RecurrentRegressor(hidden=160), onnx_path)


def test_residual_member_trains_on_the_sum_with_its_partner_frozen(tmp_path, write_dataset):
    run = _script("run_experiment")
    torch.manual_seed(0)
    gru_config = {"features": [], "init": "scratch", "model": {"hidden": 8, "tanh_tau": 8.0}}
    partner = run.build_model(gru_config).eval()
    (tmp_path / "partner").mkdir()
    torch.save(partner.state_dict(), tmp_path / "partner" / "model.pt")
    (tmp_path / "partner" / "result.json").write_text(json.dumps({"config": gru_config}))
    config = {"features": [], "init": "scratch", "model": {"kind": "row_mlp", "width": 16},
              "ensemble_with": str(tmp_path / "partner"), "ensemble_residual": True}
    model = run.assemble(run.build_model(config), config)
    x = torch.randn(2, 300, N_RAW)
    with torch.no_grad():
        start, _ = model(x, model.initial_state(2))
        alone, _ = partner(x, partner.initial_state(2))
    # The member starts at zero. Not bit-exact: torch runs a frozen GRU on another CPU kernel.
    torch.testing.assert_close(start, alone, rtol=0, atol=1e-6)

    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2, 3))
    config_train = TrainConfig(epochs=1, batch_sequences=2, chunk=5_000, fit_sequences=2, holdout=1,
                               lr=1e-3, loss="hybrid")
    model, _ = train(model, path, config_train, seed=0, log=lambda *_: None)
    for trained, frozen in zip(model.members[0].parameters(), partner.parameters(), strict=True):
        assert torch.equal(trained, frozen)
    assert model.members[1].out.weight.abs().sum() > 0
    package = export_package(model.eval(), tmp_path / "package")  # checks step against forward
    ops = Counter(n.op_type for n in onnx.load(str(package / "model.onnx")).graph.node)
    assert ops["GRU"] == 2 and ops["Relu"] == 1
    with pytest.raises(ValueError, match="from scratch"):
        run.build_model({**config, "init": "baseline"})


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


def test_ema_replaces_the_final_weights(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2, 3))
    config = TrainConfig(epochs=1, batch_sequences=2, chunk=5_000, fit_sequences=2, holdout=1,
                         eval_every=1, lr=1e-3)

    def fit(**changes):
        torch.manual_seed(0)
        return train(RecurrentRegressor(hidden=8), path, replace(config, **changes), seed=0,
                     log=lambda *_: None)

    raw, _ = fit()
    last, _ = fit(ema_decay=0.5, ema_from=1.0)  # averaging starts at the final step
    for a, b in zip(raw.parameters(), last.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    averaged, history = fit(ema_decay=0.5, ema_from=0.5)
    assert [(r["step"], r.get("ema", False)) for r in history.records] == [(0, False), (4, False), (4, True)]
    assert any((a != b).any() for a, b in zip(raw.parameters(), averaged.parameters(), strict=True))
    rescored = predict(averaged, path, [2]).result()["weighted_pearson"]
    assert rescored == pytest.approx(history.records[-1]["holdout_wp"], abs=1e-9)


def test_hybrid_alpha_interpolates_the_losses():
    torch.manual_seed(0)
    y, p = torch.randn(2, 300, 2), torch.randn(2, 300, 2)
    mask = torch.ones(2, 300, dtype=torch.bool)
    assert hybrid(p, y, mask) == 0.8 * pearson(p, y, mask) + 0.2 * mse(p, y, mask)
    torch.testing.assert_close(hybrid(p, y, mask, alpha=1.0), pearson(p, y, mask))
    torch.testing.assert_close(hybrid(p, y, mask, alpha=0.0), mse(p, y, mask))


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
    with pytest.raises(ValueError, match="does not apply"):
        train(model, path, TrainConfig(epochs=1, fit_sequences=2, holdout=1, hybrid_alpha=0.9), seed=0)
