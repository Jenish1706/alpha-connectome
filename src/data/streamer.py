"""Streaming row-group reader with TBPTT chunk support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

_META = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}


@dataclass
class SequenceChunk:
    seq_ix: int
    start: int
    features: torch.Tensor
    targets: torch.Tensor | None
    need_prediction: torch.Tensor
    is_scored: torch.Tensor | None


class ParquetSequenceDataset(IterableDataset):
    """Yield ordered, fixed-size chunks from one row group at a time.

    A row group is read only for the current sequence, and each yielded chunk
    is suitable for one TBPTT update. Hidden state management belongs to the
    training loop: detach it after every yielded chunk and reset per sequence.
    """

    def __init__(self, path: str | Path, chunk_size: int = 512,
                 columns: Sequence[str] | None = None):
        self.path = str(path)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.columns = list(columns) if columns is not None else None

    def __iter__(self) -> Iterator[SequenceChunk]:
        parquet = pq.ParquetFile(self.path)
        names = parquet.schema_arrow.names
        feature_names = self.columns or [n for n in names if n not in _META]
        has_targets = "t0" in names and "t1" in names
        has_scored = "is_scored" in names
        read_columns = ["seq_ix", "step_in_seq", "need_prediction", *feature_names]
        if has_targets:
            read_columns += ["t0", "t1"]
        if has_scored:
            read_columns.append("is_scored")
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=read_columns)
            seq = int(table["seq_ix"][0].as_py())
            x = np.column_stack([table[n].to_numpy(zero_copy_only=False) for n in feature_names]).astype(np.float32, copy=False)
            need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
            y = None
            if has_targets:
                y = np.column_stack([table["t0"].to_numpy(zero_copy_only=False), table["t1"].to_numpy(zero_copy_only=False)]).astype(np.float32, copy=False)
            scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False) if has_scored else None
            for start in range(0, len(x), self.chunk_size):
                stop = min(start + self.chunk_size, len(x))
                yield SequenceChunk(seq, start, torch.from_numpy(x[start:stop]),
                    None if y is None else torch.from_numpy(y[start:stop]),
                    torch.from_numpy(need[start:stop]),
                    None if scored is None else torch.from_numpy(scored[start:stop]))
