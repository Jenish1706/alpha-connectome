"""Dataset and callback contract for the Alpha Connectome competition files.

Mirrors the starter pack's ``utils.py`` and ``docs/data_overview.md``: column
order and types, one Parquet row group per 20,000-row sequence, and the
99-step warm-up. Validation comes at two depths:

* ``validate_metadata`` checks a whole file from its footer alone: schema,
  sequence count and 20,000 rows per row group. When the writer stored
  column statistics it also proves one sequence per group, full step
  coverage and no nulls. It runs in seconds even on the 29 GB training file,
  or on the saved footer ``datasets/train.parquet.footer``.
* ``sequence_ids`` reads only the ``seq_ix`` column, proving one sequence
  per row group and no repeated ids. The distributed files carry no
  statistics, so this is how their layout is proven.
* ``validate_sequence`` checks the rows of one sequence: step order, the
  warm-up contract, the scoring mask and finite values.

``read_sequence`` and ``iter_sequences`` turn row groups into validated numpy
arrays; the torch streamer and the latency benchmark both build on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence as SequenceOf

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"i{i}_{group}{j}" for i in range(2)
     for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4)) for j in range(count)]
    + [f"a{i}" for i in range(8)])
TARGET_COLUMNS: tuple[str, ...] = ("t0", "t1")
ID_COLUMNS: tuple[str, ...] = ("seq_ix", "step_in_seq")
FLAG_COLUMNS: tuple[str, ...] = ("need_prediction", "is_scored")


class Kind(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    VALID_MASK = "valid_mask"


COLUMNS: dict[Kind, tuple[str, ...]] = {
    Kind.TRAIN: (*ID_COLUMNS, "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS),
    Kind.VALID: (*ID_COLUMNS, "need_prediction", "is_scored", *FEATURE_COLUMNS, *TARGET_COLUMNS),
    Kind.VALID_MASK: (*ID_COLUMNS, "is_scored"),
}

# Documented sequence counts of the distributed files.
SEQUENCES: dict[Kind, int] = {Kind.TRAIN: 10_607, Kind.VALID: 1_873, Kind.VALID_MASK: 1_873}


class SchemaError(ValueError):
    """A file or table breaks the competition data contract."""


@dataclass(frozen=True)
class DataPoint:
    """The object the scorer passes to ``PredictionModel.predict``."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


@dataclass(frozen=True)
class FileSummary:
    kind: Kind
    sequences: int
    rows: int
    seq_ix: np.ndarray | None  # one id per row group, when the footer has statistics


@dataclass(frozen=True)
class Sequence:
    """One validated sequence as numpy arrays."""

    seq_ix: int
    features: np.ndarray  # (20000, n_features) float32
    need_prediction: np.ndarray  # (20000,) bool
    targets: np.ndarray | None  # (20000, 2) float32, absent in test-like data
    is_scored: np.ndarray | None  # (20000,) bool, validation only


def detect_kind(names: SequenceOf[str]) -> Kind:
    names = tuple(names)
    for kind, expected in COLUMNS.items():
        if names == expected:
            return kind
    raise SchemaError(f"columns match no dataset layout: {_first_difference(names)}")


def _first_difference(names: tuple[str, ...]) -> str:
    best = max(COLUMNS.values(), key=lambda cols: len(set(cols) & set(names)))
    missing = [c for c in best if c not in names]
    extra = [c for c in names if c not in best]
    if missing or extra:
        return f"missing {missing[:5]}, unexpected {extra[:5]}"
    if len(names) != len(best):
        return f"{len(names)} columns where {len(best)} are expected (repeated names)"
    i = next(i for i, (a, b) in enumerate(zip(names, best, strict=True)) if a != b)
    return f"position {i} holds {names[i]!r}, expected {best[i]!r} (features must not be sorted)"


def _type_ok(name: str, dtype: pa.DataType) -> bool:
    if name in ID_COLUMNS:
        return pa.types.is_integer(dtype)
    if name in FLAG_COLUMNS:
        return pa.types.is_boolean(dtype)
    return dtype == pa.float32()


def validate_schema(schema: pa.Schema, kind: Kind | str | None = None) -> Kind:
    """Check column order and types; return the dataset kind."""
    found = detect_kind(schema.names)
    if kind is not None and found != Kind(kind):
        raise SchemaError(f"expected a {Kind(kind).value} file, found {found.value} columns")
    bad = [f"{f.name}: {f.type}" for f in schema if not _type_ok(f.name, f.type)]
    if bad:
        raise SchemaError(f"unexpected column types: {bad[:5]}")
    return found


def _min_max(group, column: int):
    stats = group.column(column).statistics
    if stats is None or not stats.has_min_max:
        return None
    return stats.min, stats.max


def _check_unique(ids: np.ndarray) -> None:
    values, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        raise SchemaError(f"seq_ix {values[counts > 1][:5].tolist()} spans several row groups")


def validate_metadata(meta: pq.FileMetaData, kind: Kind | str | None = None, *,
                      sequences: int | None = None) -> FileSummary:
    """Validate a whole file from its footer, without reading any data pages."""
    kind = validate_schema(meta.schema.to_arrow_schema(), kind)
    position = {name: i for i, name in enumerate(COLUMNS[kind])}
    ids = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows != SEQUENCE_LENGTH:
            raise SchemaError(f"row group {g} has {group.num_rows} rows, not {SEQUENCE_LENGTH}")
        seq = _min_max(group, position["seq_ix"])
        if seq is not None:
            if seq[0] != seq[1]:
                raise SchemaError(f"row group {g} mixes sequences {seq[0]}..{seq[1]}")
            ids.append(seq[0])
        steps = _min_max(group, position["step_in_seq"])
        if steps is not None and steps != (0, SEQUENCE_LENGTH - 1):
            raise SchemaError(f"row group {g} covers steps {steps}")
        need = _min_max(group, position["need_prediction"]) if "need_prediction" in position else None
        if need is not None and need != (False, True):
            raise SchemaError(f"row group {g} need_prediction range is {need}")
        for j in range(group.num_columns):
            stats = group.column(j).statistics
            if stats is not None and stats.has_null_count and stats.null_count:
                raise SchemaError(f"row group {g}: {stats.null_count} nulls in {COLUMNS[kind][j]}")
    seq_ix = np.asarray(ids, dtype=np.int64) if len(ids) == meta.num_row_groups else None
    if seq_ix is not None:
        _check_unique(seq_ix)
    if sequences is not None and meta.num_row_groups != sequences:
        raise SchemaError(f"expected {sequences} sequences, found {meta.num_row_groups}")
    return FileSummary(kind, meta.num_row_groups, meta.num_rows, seq_ix)


def sequence_ids(parquet: pq.ParquetFile) -> np.ndarray:
    """Read only ``seq_ix``: prove one id per row group and no repeats; return the ids."""
    ids = np.empty(parquet.metadata.num_row_groups, dtype=np.int64)
    for g in range(len(ids)):
        column = parquet.read_row_group(g, columns=["seq_ix"])["seq_ix"].to_numpy()
        if not (column == column[0]).all():
            raise SchemaError(f"row group {g} mixes several seq_ix values")
        ids[g] = column[0]
    _check_unique(ids)
    return ids


def validate_sequence(table: pa.Table) -> int:
    """Check one row group's rows against the contract; return its seq_ix.

    Works on any column subset that includes ``seq_ix`` and ``step_in_seq``.
    """
    if table.num_rows != SEQUENCE_LENGTH:
        raise SchemaError(f"sequence has {table.num_rows} rows, not {SEQUENCE_LENGTH}")
    ids = table["seq_ix"].to_numpy()
    if not (ids == ids[0]).all():
        raise SchemaError("several seq_ix values in one sequence")
    seq = int(ids[0])
    if not np.array_equal(table["step_in_seq"].to_numpy(), np.arange(SEQUENCE_LENGTH)):
        raise SchemaError(f"sequence {seq}: steps are not 0..{SEQUENCE_LENGTH - 1} in order")
    names = set(table.column_names)
    if "need_prediction" in names:
        need = table["need_prediction"].to_numpy(zero_copy_only=False)
        if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= WARMUP):
            raise SchemaError(f"sequence {seq}: need_prediction breaks the {WARMUP}-step warm-up")
    if "is_scored" in names:
        scored = table["is_scored"].to_numpy(zero_copy_only=False)
        if scored[:WARMUP].any():
            raise SchemaError(f"sequence {seq}: warm-up rows are marked is_scored")
    for name in names.difference(ID_COLUMNS, FLAG_COLUMNS):
        if not np.isfinite(table[name].to_numpy()).all():
            raise SchemaError(f"sequence {seq}: nonfinite values in {name}")
    return seq


def _stack(table: pa.Table, names: SequenceOf[str]) -> np.ndarray:
    return np.column_stack([table[n].to_numpy() for n in names]).astype(np.float32, copy=False)


def read_sequence(parquet: pq.ParquetFile, group: int, *,
                  features: SequenceOf[str] = FEATURE_COLUMNS, validate: bool = True) -> Sequence:
    """Read row group ``group`` as one sequence, with features in the given order."""
    unknown = [f for f in features if f not in FEATURE_COLUMNS]
    if unknown:
        raise SchemaError(f"not feature columns: {unknown[:5]}")
    names = set(parquet.schema_arrow.names)
    has_targets = names.issuperset(TARGET_COLUMNS)
    has_scored = "is_scored" in names
    columns = [*ID_COLUMNS, "need_prediction", *features]
    columns += list(TARGET_COLUMNS) if has_targets else []
    columns += ["is_scored"] if has_scored else []
    table = parquet.read_row_group(group, columns=columns)
    seq = validate_sequence(table) if validate else int(table["seq_ix"][0].as_py())
    return Sequence(
        seq_ix=seq,
        features=_stack(table, features),
        need_prediction=table["need_prediction"].to_numpy(zero_copy_only=False),
        targets=_stack(table, TARGET_COLUMNS) if has_targets else None,
        is_scored=table["is_scored"].to_numpy(zero_copy_only=False) if has_scored else None,
    )


def iter_sequences(path: str | Path, row_groups: Iterable[int] | None = None, *,
                   features: SequenceOf[str] = FEATURE_COLUMNS,
                   validate: bool = True) -> Iterator[Sequence]:
    """Yield sequences of a train- or valid-layout file, one row group at a time."""
    parquet = pq.ParquetFile(path)
    kind = validate_schema(parquet.schema_arrow)
    if kind is Kind.VALID_MASK:
        raise SchemaError("the validation mask file has no features")
    groups = range(parquet.num_row_groups) if row_groups is None else row_groups
    for group in groups:
        yield read_sequence(parquet, group, features=features, validate=validate)
