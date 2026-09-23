import math

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from conftest import sequence_table
from torch.utils.data import DataLoader

from src.data.schema import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, Kind, SchemaError
from src.data.streamer import ParquetSequenceDataset


def _reassemble(chunks):
    """Group chunks by sequence, checking they arrive in order."""
    by_seq = {}
    for chunk in chunks:
        parts = by_seq.setdefault(chunk.seq_ix, [])
        assert chunk.start == sum(len(c.features) for c in parts)
        parts.append(chunk)
    return by_seq


def test_chunks_cover_each_sequence_in_order(write_dataset):
    path = write_dataset(Kind.VALID, seq_ids=(11, 5, 42))
    dataset = ParquetSequenceDataset(path, chunk_size=6_000)
    chunks = list(dataset)
    assert len(chunks) == len(dataset) == 3 * 4
    assert [c.start for c in chunks[:4]] == [0, 6_000, 12_000, 18_000]
    assert len(chunks[3].features) == 2_000
    by_seq = _reassemble(chunks)
    assert list(by_seq) == [11, 5, 42]
    table = pq.ParquetFile(path).read_row_group(1)
    seq5 = by_seq[5]
    features = torch.cat([c.features for c in seq5]).numpy()
    assert features.dtype == np.float32 and features.shape == (SEQUENCE_LENGTH, N_FEATURES)
    assert np.array_equal(features, np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]))
    assert torch.cat([c.targets for c in seq5]).shape == (SEQUENCE_LENGTH, 2)
    assert torch.cat([c.is_scored for c in seq5]).numpy().tolist() == table["is_scored"].to_pylist()
    assert seq5[0].need_prediction.dtype == torch.bool


def test_train_layout_and_feature_subset(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2))
    chunks = list(ParquetSequenceDataset(path, chunk_size=SEQUENCE_LENGTH,
                                         columns=["a7", "i0_p0"], row_groups=[1]))
    assert len(chunks) == 1 and chunks[0].seq_ix == 2 and chunks[0].is_scored is None
    table = pq.ParquetFile(path).read_row_group(1)
    assert np.array_equal(chunks[0].features[:, 0].numpy(), table["a7"].to_numpy())


def test_workers_stream_disjoint_sequences(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(1, 2, 3, 4, 5))
    dataset = ParquetSequenceDataset(path, chunk_size=5_000)
    chunks = list(DataLoader(dataset, batch_size=None, num_workers=2))
    assert sorted((c.seq_ix, c.start) for c in chunks) == [
        (s, start) for s in range(1, 6) for start in range(0, SEQUENCE_LENGTH, 5_000)]
    _reassemble(chunks)


def test_contract_violations_raise(tmp_path, write_dataset):
    path = tmp_path / "broken.parquet"
    steps = np.arange(SEQUENCE_LENGTH)
    pq.write_table(sequence_table(Kind.TRAIN, need_prediction=steps >= 10), path,
                   row_group_size=SEQUENCE_LENGTH)
    with pytest.raises(SchemaError, match="warm-up"):
        list(ParquetSequenceDataset(path))
    mask = tmp_path / "mask.parquet"
    pq.write_table(sequence_table(Kind.VALID_MASK), mask)
    with pytest.raises(SchemaError, match="no features"):
        ParquetSequenceDataset(mask)
    with pytest.raises(ValueError, match="chunk_size"):
        ParquetSequenceDataset(write_dataset(), chunk_size=0)
    shuffled = tmp_path / "sorted.parquet"
    table = sequence_table(Kind.TRAIN)
    pq.write_table(table.select(sorted(table.column_names)), shuffled)
    with pytest.raises(SchemaError):
        ParquetSequenceDataset(shuffled)


def test_real_validation_chunks(valid_path):
    dataset = ParquetSequenceDataset(valid_path, chunk_size=512, row_groups=[0, 1])
    chunks = list(dataset)
    assert len(chunks) == len(dataset) == 2 * math.ceil(SEQUENCE_LENGTH / 512)
    by_seq = _reassemble(chunks)
    parquet = pq.ParquetFile(valid_path)
    for group, (seq, parts) in enumerate(by_seq.items()):
        table = parquet.read_row_group(group)
        assert seq == table["seq_ix"][0].as_py()
        expected = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS])
        assert np.array_equal(torch.cat([c.features for c in parts]).numpy(), expected)
        targets = torch.cat([c.targets for c in parts]).numpy()
        assert np.array_equal(targets[:, 1], table["t1"].to_numpy())
        scored = torch.cat([c.is_scored for c in parts]).numpy()
        assert np.array_equal(scored, table["is_scored"].to_numpy(zero_copy_only=False))


def test_real_train_chunks(train_path):
    chunks = list(ParquetSequenceDataset(train_path, chunk_size=SEQUENCE_LENGTH, row_groups=[0]))
    assert len(chunks) == 1 and chunks[0].is_scored is None
    assert chunks[0].features.shape == (SEQUENCE_LENGTH, N_FEATURES)
    assert torch.isfinite(chunks[0].targets).all()
