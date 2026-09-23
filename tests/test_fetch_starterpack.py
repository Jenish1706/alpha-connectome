"""End-to-end tests of scripts/fetch_starterpack.py on a synthetic archive."""

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_starterpack.py"
PACK = "wnn_connectome_starterpack"
ROWS = 20_000
GROUPS = 5


def _train_parquet(path: Path) -> Path:
    rng = np.random.default_rng(0)
    schema = pa.schema([("seq_ix", pa.int32()), ("step_in_seq", pa.int32()),
                        *[(f"f{i}", pa.float32()) for i in range(5)]])
    with pq.ParquetWriter(path, schema) as writer:
        for group in range(GROUPS):
            columns = {"seq_ix": np.full(ROWS, 100 + group, np.int32),
                       "step_in_seq": np.arange(ROWS, dtype=np.int32)}
            columns.update({f"f{i}": rng.standard_normal(ROWS).astype(np.float32) for i in range(5)})
            writer.write_table(pa.table(columns, schema=schema), row_group_size=ROWS)
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
    group = meta.row_group(i)
    return max(group.column(j).data_page_offset + group.column(j).total_compressed_size
               for j in range(group.num_columns))


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
