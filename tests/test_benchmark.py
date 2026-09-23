import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_latency.py"


def _bench(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          check=False)


def _report(*args: str) -> dict:
    result = _bench(*args, "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_synthetic_replay_of_repo_submission():
    report = _report("--steps", "2000")
    assert report["rows"] == 2000 and report["data"] == "synthetic"
    assert 0 < report["callback_us"]["median"] <= report["callback_us"]["max"]
    assert report["projected_rows"] == 37_460_000


@pytest.mark.parametrize("body, message", [
    ("return np.zeros(2, np.float32)", "return None during warm-up"),
    ("return None if not dp.need_prediction else np.array([np.nan, 0.0])", "two finite values"),
    ("return None if not dp.need_prediction else np.zeros(3)", "two finite values"),
])
def test_contract_violations_fail_the_run(tmp_path, body, message):
    solution = tmp_path / "solution.py"
    solution.write_text("import numpy as np\n\nclass PredictionModel:\n"
                        f"    def predict(self, dp):\n        {body}\n")
    result = _bench("--solution", str(solution), "--steps", "500")
    assert result.returncode != 0 and message in result.stderr


def test_repo_submission_scores_on_validation(valid_path):
    report = _report("--data", str(valid_path), "--sequences", "1", "--score")
    first = pq.ParquetFile(valid_path).read_row_group(0, columns=["is_scored"])
    assert report["rows"] == 20_000
    assert report["score"]["selected_rows"] == int(np.sum(first["is_scored"].to_numpy(zero_copy_only=False)))
    assert -1.0 <= report["score"]["weighted_pearson"] <= 1.0


def test_baseline_score_matches_starterpack_scorer(tmp_path, valid_path, official, baseline_solution):
    subset = tmp_path / "valid_subset.parquet"
    source = pq.ParquetFile(valid_path)
    with pq.ParquetWriter(subset, source.schema_arrow) as writer:
        for group in range(2):
            writer.write_table(source.read_row_group(group), row_group_size=20_000)
    ours = _report("--solution", str(baseline_solution), "--data", str(subset),
                   "--sequences", "0", "--score")

    spec = importlib.util.spec_from_file_location("baseline_solution", baseline_solution)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = official.ScorerStepByStep(subset).score(module.PredictionModel())
    assert ours["rows"] == reference["rows_seen"] == 40_000
    assert ours["score"]["selected_rows"] == reference["selected_rows"]
    for key in ("t0", "t1", "weighted_pearson"):
        assert ours["score"][key] == pytest.approx(reference[key], abs=1e-9)
