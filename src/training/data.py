"""Batches of whole sequences from a train- or valid-layout Parquet file.

Each batch holds B complete 20,000-row sequences as dense arrays, so a
training step can run B sequences side by side over one time window. The
next batch is read on a background thread while the current one is used.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP, SchemaError

NEED = np.arange(SEQUENCE_LENGTH) >= WARMUP


@dataclass
class Batch:
    groups: list[int]
    seq_ix: np.ndarray  # (B,)
    features: np.ndarray  # (B, 20000, 112) float32, raw inputs
    targets: np.ndarray  # (B, 20000, 2) float32
    is_scored: np.ndarray | None  # (B, 20000) bool, validation files only


def columns_for(parquet: pq.ParquetFile) -> list[str]:
    scored = "is_scored" in parquet.schema_arrow.names
    return ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS] + \
        (["is_scored"] if scored else [])


def read_batch(parquet: pq.ParquetFile, groups: Sequence[int], remote=None) -> Batch:
    """Read whole sequences; groups at or past the local file's end come from ``remote``.

    Each row group is decoded straight into its slot of preallocated batch
    arrays, so reading peaks near the batch's own size rather than a multiple.
    Rows follow ``groups``.
    """
    columns = columns_for(parquet)
    local = parquet.metadata.num_row_groups
    far = [g for g in groups if g >= local]
    if far and remote is None:
        raise SchemaError(f"row groups {far[:3]}... are not in the local file")
    n = len(groups)
    x = np.empty((n, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), np.float32)
    y = np.empty((n, SEQUENCE_LENGTH, len(TARGET_COLUMNS)), np.float32)
    scored = np.empty((n, SEQUENCE_LENGTH), bool) if "is_scored" in columns else None
    ids = np.empty(n, np.int64)
    for i, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=columns) if group < local \
            else remote.get(group, columns=columns)
        ids[i] = _fill(table, x[i], y[i], None if scored is None else scored[i])
    return Batch(list(groups), ids, x, y, scored)


def _fill(table: pa.Table, x: np.ndarray, y: np.ndarray, scored: np.ndarray | None) -> int:
    """Copy one sequence's row group into its slots of the batch arrays; return its seq_ix."""
    if table.num_rows != SEQUENCE_LENGTH:
        raise SchemaError(f"expected a whole sequence, got {table.num_rows} rows")
    ids = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    need = table["need_prediction"].to_numpy(zero_copy_only=False)
    if (ids != ids[0]).any() or (steps != np.arange(SEQUENCE_LENGTH)).any() or (need != NEED).any():
        raise SchemaError("row groups break the one-sequence-per-group contract")
    np.stack([table[c].to_numpy() for c in FEATURE_COLUMNS], axis=1, out=x)
    np.stack([table[c].to_numpy() for c in TARGET_COLUMNS], axis=1, out=y)
    if not np.isfinite(x).all():
        raise SchemaError("nonfinite features")
    if not np.isfinite(y).all():
        raise SchemaError("nonfinite targets")
    if scored is not None:
        scored[:] = table["is_scored"].to_numpy(zero_copy_only=False)
    return int(ids[0])


def iter_batches(path: str | Path, batches: Sequence[Sequence[int]], remote=None) -> Iterator[Batch]:
    """Yield batches in order, reading the next one while the caller works."""
    parquet = pq.ParquetFile(path)
    result: dict = {}

    def load(i: int) -> None:
        try:
            result[i] = read_batch(parquet, batches[i], remote)
        except BaseException as exc:  # re-raised in the caller's thread
            result[i] = exc

    thread = None
    for i in range(len(batches)):
        if thread is None:
            load(i)
        else:
            thread.join()
        thread = None
        if i + 1 < len(batches):
            thread = threading.Thread(target=load, args=(i + 1,), daemon=True)
            thread.start()
        batch = result.pop(i)
        if isinstance(batch, BaseException):
            raise batch
        yield batch
