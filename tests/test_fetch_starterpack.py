"""End-to-end tests of scripts/fetch_starterpack.py on a synthetic archive."""

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_starterpack.py"
PACK = "wnn_connectome_starterpack"
ROWS = 20_000
GROUPS = 5


def _train_parquet(path: Path, columns: int = 5) -> Path:
    rng = np.random.default_rng(0)
    schema = pa.schema([("seq_ix", pa.int32()), ("step_in_seq", pa.int32()),
                        *[(f"f{i}", pa.float32()) for i in range(columns)]])
    with pq.ParquetWriter(path, schema) as writer:
        for group in range(GROUPS):
            data = {"seq_ix": np.full(ROWS, 100 + group, np.int32),
                    "step_in_seq": np.arange(ROWS, dtype=np.int32)}
            data.update({f"f{i}": rng.standard_normal(ROWS).astype(np.float32) for i in range(columns)})
            writer.write_table(pa.table(data, schema=schema), row_group_size=ROWS)
    return path


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(PACK)
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _run(root: Path, *args: str, stdin: bytes | None = None):
    return subprocess.run([sys.executable, str(SCRIPT), *args, "--root", str(root)],
                          input=stdin, capture_output=True, check=False)


def _group_end(meta, i: int) -> int:
    """Byte offset just past row group i (its dictionary pages precede the data pages)."""
    group = meta.row_group(i)
    ends = []
    for j in range(group.num_columns):
        col = group.column(j)
        start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
        ends.append(start + col.total_compressed_size)
    return max(ends)


def test_full_extraction_layout_from_stdin(tmp_path):
    train = _train_parquet(tmp_path / "train.parquet").read_bytes()
    archive = _archive(tmp_path / "pack.tar.gz", {
        f"._{PACK}": b"apple", f"{PACK}/README.md": b"readme", f"{PACK}/._README.md": b"apple",
        f"{PACK}/docs/faq.md": b"faq", f"{PACK}/datasets/train.parquet": train,
        f"{PACK}/datasets/valid.parquet": b"valid", f"{PACK}/datasets/._valid.parquet": b"apple",
    })
    out = tmp_path / "repo"
    result = _run(out, "-", stdin=archive.read_bytes())
    assert result.returncode == 0, result.stderr.decode()
    assert (out / "datasets" / "train.parquet").read_bytes() == train
    assert (out / "datasets" / "valid.parquet").read_bytes() == b"valid"
    assert (out / PACK / "README.md").read_bytes() == b"readme"
    assert (out / PACK / "docs" / "faq.md").read_bytes() == b"faq"
    leftovers = [p for p in out.rglob("*") if p.name.startswith("._") or p.name.endswith(".part")]
    assert leftovers == []


def test_train_sample_keeps_leading_sequences_and_full_footer(tmp_path):
    source = _train_parquet(tmp_path / "train.parquet")
    meta = pq.read_metadata(source)
    archive = _archive(tmp_path / "pack.tar.gz", {f"{PACK}/datasets/train.parquet": source.read_bytes()})
    # The head covers two whole row groups and part of the third.
    head = _group_end(meta, 1) + (64 << 10) + 1000
    assert head < _group_end(meta, 2)
    out = tmp_path / "repo"
    result = _run(out, str(archive), "--train-sample", "3", "--train-head-bytes", str(head))
    assert result.returncode == 0, result.stderr.decode()
    assert b"kept 2 of 5 training sequences (requested 3)" in result.stderr

    datasets = out / "datasets"
    assert not (datasets / "train.parquet").exists()
    sample = pq.ParquetFile(datasets / "train_head.parquet")
    original = pq.ParquetFile(source)
    assert sample.metadata.num_row_groups == 2
    kept_bytes = _group_end(meta, 1)
    assert (datasets / "train_head.parquet").read_bytes()[:kept_bytes] == source.read_bytes()[:kept_bytes]
    assert sample.schema_arrow == original.schema_arrow
    for i in range(2):
        assert sample.read_row_group(i).equals(original.read_row_group(i))
    footer = pq.read_metadata(datasets / "train.parquet.footer")
    assert footer.num_row_groups == GROUPS and footer.num_rows == GROUPS * ROWS
    assert footer.schema.to_arrow_schema() == original.schema_arrow
    assert sorted(p.name for p in datasets.iterdir()) == ["train.parquet.footer", "train_head.parquet"]


def test_rejects_path_traversal(tmp_path):
    archive = _archive(tmp_path / "pack.tar.gz", {f"{PACK}/../evil.txt": b"x"})
    out = tmp_path / "repo"
    result = _run(out, str(archive))
    assert result.returncode != 0
    assert b"unsafe archive member" in result.stderr
    assert not (tmp_path / "evil.txt").exists()


def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_starterpack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("statistics", [True, False])
def test_truncate_footer_keeps_any_prefix_of_row_groups(tmp_path, statistics):
    fetch = _load_script()
    groups, rows = 17, 1000  # >= 15 exercises the long list-size form
    path = tmp_path / "many.parquet"
    rng = np.random.default_rng(1)
    schema = pa.schema([("seq_ix", pa.int32()), ("x", pa.float32()), ("flag", pa.bool_())])
    with pq.ParquetWriter(path, schema, write_statistics=statistics) as writer:
        for g in range(groups):
            writer.write_table(pa.table({"seq_ix": np.full(rows, g, np.int32),
                                         "x": rng.standard_normal(rows).astype(np.float32),
                                         "flag": rng.random(rows) < 0.5}, schema=schema))
    data = path.read_bytes()
    footer_len = int.from_bytes(data[-8:-4], "little")
    footer = data[-(footer_len + 8):-8]
    meta = pq.read_metadata(path)
    for keep in (1, 2, 14, 15, 16, groups):
        new = fetch.truncate_footer(footer, keep, keep * rows)
        end = _group_end(meta, keep - 1)
        out = tmp_path / f"keep{keep}.parquet"
        out.write_bytes(data[:end] + new + len(new).to_bytes(4, "little") + b"PAR1")
        cut = pq.ParquetFile(out)
        assert cut.metadata.num_row_groups == keep and cut.metadata.num_rows == keep * rows
        assert cut.schema_arrow.equals(pq.read_schema(path))
        assert cut.read().equals(pq.read_table(path).slice(0, keep * rows))


def test_cut_stream_still_finishes_sample_from_saved_footer(tmp_path):
    # ~40 MB, so the head spans several 8 MB read chunks, as with the real file.
    source = _train_parquet(tmp_path / "train.parquet", columns=100)
    meta = pq.read_metadata(source)
    archive = _archive(tmp_path / "pack.tar.gz", {f"{PACK}/datasets/train.parquet": source.read_bytes()})
    cut = tmp_path / "cut.tar.gz"
    cut.write_bytes(archive.read_bytes()[:int(archive.stat().st_size * 0.8)])  # dies inside train.parquet
    out = tmp_path / "repo"
    (out / "datasets").mkdir(parents=True)
    data = source.read_bytes()
    footer_len = int.from_bytes(data[-8:-4], "little")
    (out / "datasets" / "train.parquet.footer").write_bytes(b"PAR1" + data[-(footer_len + 8):])
    head = _group_end(meta, 1) + 1000
    result = _run(out, str(cut), "--train-sample", "2", "--train-head-bytes", str(head))
    assert result.returncode == 1 and b"finishing the training sample" in result.stderr
    sample = out / "datasets" / "train_head.parquet"
    assert pq.ParquetFile(sample).metadata.num_row_groups == 2
    assert sample.read_bytes()[:_group_end(meta, 1)] == data[:_group_end(meta, 1)]
