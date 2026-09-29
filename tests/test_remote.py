"""Streaming training row groups from the archive: gzip resume, tar parsing, decoding."""

import gzip
import io
import tarfile

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.data.schema import Kind, SchemaError
from src.models.features import N_RAW
from src.training.data import columns_for, read_batch
from src.training.fit import TrainConfig, train
from src.training.remote import MEMBER, RemoteRowGroups, ResumableGzipStream, row_group_spans


def _footer(path, out):
    """Save a file's footer the way the fetcher does: PAR1 + footer + length + PAR1."""
    data = path.read_bytes()
    length = int.from_bytes(data[-8:-4], "little")
    out.write_bytes(b"PAR1" + data[-(length + 8):])
    return out


def _archive(tmp_path, parquet):
    """A starter-pack-shaped .tar.gz holding ``parquet`` as train.parquet; returns its URL."""
    archive = tmp_path / "pack.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        readme = tarfile.TarInfo("wnn_connectome_starterpack/README.md")
        readme.size = 6
        tar.addfile(readme, io.BytesIO(b"hello\n"))
        tar.add(parquet, arcname=MEMBER)
    return archive.as_uri()


def test_gzip_stream_resumes_from_a_snapshot(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 16, 6 << 20, dtype=np.uint8).tobytes()  # compresses about 2:1
    path = tmp_path / "blob.gz"
    path.write_bytes(gzip.compress(data, compresslevel=1))
    stream = ResumableGzipStream(path.as_uri(), snapshot_every=256 << 10, fail_after=2 << 20)
    with stream:
        assert stream.read() == data
    assert stream.resumes == 1


def test_gzip_stream_gives_up_after_its_retries(tmp_path):
    path = tmp_path / "cut.gz"
    path.write_bytes(gzip.compress(b"x" * (1 << 20))[:-100])  # truncated: never reaches the end
    with ResumableGzipStream(path.as_uri(), retries=2) as stream, pytest.raises(OSError, match="2 times"):
        stream.read()


def test_remote_row_groups_match_the_file(tmp_path, write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(3, 1, 4, 2), name="train_full.parquet")
    footer = _footer(path, tmp_path / "train.parquet.footer")
    assert row_group_spans(pq.read_metadata(footer))[-1][1] + 8 + footer.stat().st_size - 12 == \
        path.stat().st_size
    local = pq.ParquetFile(path)
    columns = columns_for(local)
    remote = RemoteRowGroups(footer, [3, 1], url=_archive(tmp_path, path), log=lambda *_: None,
                             stream_options={"snapshot_every": 1 << 20, "fail_after": 3 << 20})
    try:
        for group in (1, 3):
            assert remote.get(group, columns=columns).equals(local.read_row_group(group, columns=columns))
        with pytest.raises(OSError, match="without row group 0"):
            remote.get(0)
    finally:
        remote.close()


def test_remote_reader_checks_the_archive_holds_this_file(tmp_path, write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(3, 1), name="train_full.parquet")
    other = write_dataset(Kind.TRAIN, seq_ids=(5,), name="other.parquet")
    remote = RemoteRowGroups(_footer(path, tmp_path / "train.parquet.footer"), [1],
                             url=_archive(tmp_path, other), log=lambda *_: None)
    try:
        with pytest.raises(OSError, match="unavailable") as error:
            remote.get(1)
        assert "footer implies" in str(error.value.__cause__)
    finally:
        remote.close()


def test_batches_mix_local_and_remote_sequences(tmp_path, write_dataset):
    full = write_dataset(Kind.TRAIN, seq_ids=(3, 1, 4, 2), name="train_full.parquet")
    head = write_dataset(Kind.TRAIN, seq_ids=(3, 1), name="train_head.parquet")
    remote = RemoteRowGroups(_footer(full, tmp_path / "train.parquet.footer"), [2, 3],
                             url=_archive(tmp_path, full), log=lambda *_: None)
    try:
        mixed = read_batch(pq.ParquetFile(head), [3, 1, 2], remote)
    finally:
        remote.close()
    expected = read_batch(pq.ParquetFile(full), [3, 1, 2])
    assert mixed.groups == expected.groups == [3, 1, 2]
    assert mixed.seq_ix.tolist() == [2, 1, 4]
    np.testing.assert_array_equal(mixed.features, expected.features)
    np.testing.assert_array_equal(mixed.targets, expected.targets)
    with pytest.raises(SchemaError, match="not in the local file"):
        read_batch(pq.ParquetFile(head), [0, 2])


class _Linear(torch.nn.Module):
    """A stateless stand-in for the GRU: the schedule, not the model, is under test."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(N_RAW, 2)

    def initial_state(self, n):
        return [torch.zeros(n, 1)]

    def forward(self, x, state):
        return self.head(x), state


def test_full_data_epochs_stream_the_rest_of_the_file(tmp_path, write_dataset):
    full = write_dataset(Kind.TRAIN, seq_ids=(3, 1, 4, 2, 6), name="train_full.parquet")
    head = write_dataset(Kind.TRAIN, seq_ids=(3, 1, 4), name="train_head.parquet")
    _footer(full, tmp_path / "train.parquet.footer")
    torch.manual_seed(0)
    model = _Linear()
    config = TrainConfig(epochs=2, batch_sequences=1, chunk=10_000, fit_sequences=2, holdout=1,
                         eval_every=100, lr=1e-3, min_lr=1e-4, full_data=True, local_first=1)
    lines = []
    model, history = train(model, head, config, seed=0, log=lines.append,
                           archive=_archive(tmp_path, full))
    # 2 local + 2 remote sequences per epoch, one per batch, two steps each
    assert [r["step"] for r in history.records] == [0, 16]
    assert sum("remote: read 2 row groups" in line for line in lines) == 2
    with pytest.raises(ValueError, match="whole epochs"):
        train(model, head, TrainConfig(**{**config.__dict__, "epochs": 1.5}), seed=0)
