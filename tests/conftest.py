"""Shared fixtures.

Tests that need the starter pack skip when it is absent. Set
ALPHA_CONNECTOME_DATA to use another datasets directory, and
FULL_DATA_CHECKS=1 to check every sequence rather than the first few.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.schema import COLUMNS, FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP, Kind

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ALPHA_CONNECTOME_DATA", ROOT / "datasets"))
PACK = ROOT / "wnn_connectome_starterpack"
FULL = os.environ.get("FULL_DATA_CHECKS") == "1"
SAMPLE_SEQUENCES = 4


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} not found (see README: starter pack)")
    return path


def sequence_table(kind: Kind = Kind.VALID, seq: int = 7, **overrides) -> pa.Table:
    """A synthetic sequence that satisfies the data contract."""
    rng = np.random.default_rng(seq)
    steps = np.arange(SEQUENCE_LENGTH)
    columns = {
        "seq_ix": np.full(SEQUENCE_LENGTH, seq, np.int64),
        "step_in_seq": steps.astype(np.int64),
        "need_prediction": steps >= WARMUP,
        "is_scored": (steps >= WARMUP) & (steps % 3 == 0),
        "t0": rng.standard_normal(SEQUENCE_LENGTH).astype(np.float32),
        "t1": rng.standard_normal(SEQUENCE_LENGTH).astype(np.float32),
    }
    columns.update({c: rng.standard_normal(SEQUENCE_LENGTH).astype(np.float32) for c in FEATURE_COLUMNS})
    columns.update(overrides)
    return pa.table({c: columns[c] for c in COLUMNS[kind]})


@pytest.fixture
def write_dataset(tmp_path):
    """Write synthetic sequences as a Parquet file, one row group per sequence."""
    def write(kind: Kind = Kind.VALID, seq_ids=(11, 5, 42), name: str = "data.parquet") -> Path:
        path = tmp_path / name
        tables = [sequence_table(kind, seq) for seq in seq_ids]
        with pq.ParquetWriter(path, tables[0].schema) as writer:
            for table in tables:
                writer.write_table(table, row_group_size=SEQUENCE_LENGTH)
        return path
    return write


@pytest.fixture(scope="session")
def check_groups():
    """Row groups to check in a real file: all with FULL_DATA_CHECKS=1, else the first few."""
    def pick(total: int) -> range:
        return range(total if FULL else min(SAMPLE_SEQUENCES, total))
    return pick


@pytest.fixture(scope="session")
def valid_path() -> Path:
    return _need(DATA / "valid.parquet")


@pytest.fixture(scope="session")
def valid_mask_path() -> Path:
    return _need(DATA / "valid_mask.parquet")


@pytest.fixture(scope="session")
def train_path() -> Path:
    """The full training file if present, else its leading-sequence sample."""
    full = DATA / "train.parquet"
    return full if full.exists() else _need(DATA / "train_head.parquet")


@pytest.fixture(scope="session")
def train_metadata() -> pq.FileMetaData:
    """Footer of the full training file, from the file itself or the saved footer."""
    full = DATA / "train.parquet"
    return pq.read_metadata(full if full.exists() else _need(DATA / "train.parquet.footer"))


@pytest.fixture(scope="session")
def official():
    """The starter pack's utils.py: reference scorer and column definitions."""
    spec = importlib.util.spec_from_file_location("starterpack_utils", _need(PACK / "utils.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def baseline_solution() -> Path:
    return _need(PACK / "baseline" / "solution.py")
