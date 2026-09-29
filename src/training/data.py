"""Batches of whole sequences from a train- or valid-layout Parquet file.

Each batch holds B complete 20,000-row sequences as dense arrays, so a
training step can run B sequences side by side over one time window. The
next batch is read on a background thread while the current one is used.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

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
    """Read whole sequences; groups at or past the local file's end come from ``remote``."""
    columns = columns_for(parquet)
    local = [g for g in groups if g < parquet.metadata.num_row_groups]
    far = [g for g in groups if g >= parquet.metadata.num_row_groups]
    if far and remote is None:
        raise SchemaError(f"row groups {far[:3]}... are not in the local file")
    tables = [parquet.read_row_groups(local, columns=columns)] if local else []
    tables += [remote.get(g, columns=columns) for g in far]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return table_to_batch(table, local + far)


def table_to_batch(table: pa.Table, groups: Sequence[int]) -> Batch:
    """Dense arrays for whole sequences stored back to back in ``table``, one per group."""
    scored = "is_scored" in table.column_names
    n = len(groups)
    if table.num_rows != n * SEQUENCE_LENGTH:
        raise SchemaError(f"expected {n} whole sequences, got {table.num_rows} rows")
    ids = table["seq_ix"].to_numpy().reshape(n, SEQUENCE_LENGTH)
    steps = table["step_in_seq"].to_numpy().reshape(n, SEQUENCE_LENGTH)
    need = table["need_prediction"].to_numpy(zero_copy_only=False).reshape(n, SEQUENCE_LENGTH)
    if (ids != ids[:, :1]).any() or (steps != np.arange(SEQUENCE_LENGTH)).any() or (need != NEED).any():
        raise SchemaError("row groups break the one-sequence-per-group contract")
    cols = [table[c].to_numpy() for c in FEATURE_COLUMNS]
    # Check column by column: one column's temporaries, not a full-batch copy.
    if not all(np.isfinite(c).all() for c in cols):
        raise SchemaError("nonfinite features")
    with warnings.catch_warnings():  # Arrow buffers are read-only; they are only read here
        warnings.simplefilter("ignore", UserWarning)
        # torch.stack interleaves the columns in parallel, several times faster than numpy.
        x = torch.stack([torch.from_numpy(c) for c in cols], dim=1).numpy()
    y = np.stack([table[t].to_numpy() for t in TARGET_COLUMNS], axis=-1).astype(np.float32)
    mask = table["is_scored"].to_numpy(zero_copy_only=False).reshape(n, SEQUENCE_LENGTH) if scored else None
    del cols, table  # release the Arrow buffers before the next batch is read
    if not np.isfinite(y).all():
        raise SchemaError("nonfinite targets")
    return Batch(list(groups), ids[:, 0].copy(), x.reshape(n, SEQUENCE_LENGTH, -1),
                 y.reshape(n, SEQUENCE_LENGTH, 2), mask)  # rows follow ``groups``


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
