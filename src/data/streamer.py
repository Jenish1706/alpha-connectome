"""Streaming row-group reader with TBPTT chunk support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.schema import (FEATURE_COLUMNS, SEQUENCE_LENGTH, Kind, SchemaError, read_sequence,
                             validate_schema)


@dataclass
class SequenceChunk:
    seq_ix: int
    start: int  # 0 marks the first chunk of a sequence: reset the hidden state
    features: torch.Tensor  # (rows, n_features) float32
    targets: torch.Tensor | None  # (rows, 2) float32
    need_prediction: torch.Tensor  # (rows,) bool
    is_scored: torch.Tensor | None  # (rows,) bool, validation files only


class ParquetSequenceDataset(IterableDataset):
    """Yield ordered, fixed-size chunks from one row group (= one sequence) at a time.

    Each chunk suits one TBPTT update. Hidden state management belongs to the
    training loop: detach it after every chunk and reset it when ``start == 0``.
    Rows are validated against the competition contract as they are read.
    Under a multi-worker ``DataLoader`` each worker streams a disjoint share of
    the row groups, so each sequence stays in order within one worker; use
    ``batch_size=None`` to keep chunks whole.
    """

    def __init__(self, path: str | Path, chunk_size: int = 512,
                 columns: Sequence[str] | None = None, *,
                 row_groups: Sequence[int] | None = None, validate: bool = True):
        self.path = str(path)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.features = tuple(columns) if columns is not None else FEATURE_COLUMNS
        parquet = pq.ParquetFile(self.path)
        if validate_schema(parquet.schema_arrow) is Kind.VALID_MASK:
            raise SchemaError("the validation mask file has no features")
        groups = range(parquet.num_row_groups)
        self.row_groups = list(groups if row_groups is None else row_groups)
        self.validate = validate

    def __len__(self) -> int:
        """Number of chunks in one pass over the selected row groups."""
        return len(self.row_groups) * math.ceil(SEQUENCE_LENGTH / self.chunk_size)

    def __iter__(self) -> Iterator[SequenceChunk]:
        groups = self.row_groups
        worker = get_worker_info()
        if worker is not None:
            groups = groups[worker.id::worker.num_workers]
        parquet = pq.ParquetFile(self.path)
        for group in groups:
            seq = read_sequence(parquet, group, features=self.features, validate=self.validate)
            x = torch.from_numpy(seq.features)
            y = None if seq.targets is None else torch.from_numpy(seq.targets)
            need = torch.from_numpy(seq.need_prediction)
            scored = None if seq.is_scored is None else torch.from_numpy(seq.is_scored)
            for start in range(0, len(x), self.chunk_size):
                stop = start + self.chunk_size
                yield SequenceChunk(seq.seq_ix, start, x[start:stop],
                                    None if y is None else y[start:stop], need[start:stop],
                                    None if scored is None else scored[start:stop])
