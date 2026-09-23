#!/usr/bin/env python3
"""Stream the competition starter pack into this repository's data layout.

    curl -fL https://files.wundernn.io/wnn_connectome_starterpack.tar.gz \\
        | python scripts/fetch_starterpack.py

Files under the pack's ``datasets/`` land in ``datasets/`` at the repository
root; everything else lands in ``wnn_connectome_starterpack/``. Both paths are
gitignored. macOS ``._*`` metadata entries are skipped.

The archive is about 34 GB and ``datasets/train.parquet`` alone is about
29 GB. On a disk that cannot hold it, pass ``--train-sample N``: the stream is
still read end to end, but only the first N training sequences are written
(``datasets/train_head.parquet``, one row group per sequence as in the
original), together with the complete training footer
(``datasets/train.parquet.footer``). ``pyarrow.parquet.read_metadata`` reads
that footer, so the structure of every training sequence can still be
validated without the data.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import time
from collections import deque
from pathlib import Path, PurePosixPath

PACK = "wnn_connectome_starterpack"
TRAIN = ("datasets", "train.parquet")
TRAIN_SEQUENCES = 10_607  # documented count; only used to size the head buffer
CHUNK = 8 << 20
RESERVE = 1 << 30  # never fill the disk completely
READ_MARGIN = 64 << 10  # slack between the last kept row group and the head end
MAGIC = b"PAR1"


def destination(name: str, root: Path) -> Path | None:
    """Map an archive member name to a local path, or None to skip it."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    parts = path.parts
    if parts and parts[0] == PACK:
        parts = parts[1:]
    if not parts or any(part.startswith("._") for part in parts):
        return None  # the pack root itself, or AppleDouble metadata
    if parts[0] == "datasets":
        return root.joinpath(*parts)
    return root.joinpath(PACK, *parts)


def ensure_space(root: Path, needed: int, what: str) -> None:
    free = shutil.disk_usage(root).free
    if needed > free - RESERVE:
        raise SystemExit(
            f"{what} needs {needed / 1e9:.1f} GB but {free / 1e9:.1f} GB is free; "
            "free space or rerun with --train-sample N")


class Progress:
    def __init__(self, every: int = 2 << 30):
        self.start = time.monotonic()
        self.done = 0
        self.every = every
        self.next = every

    def __call__(self, n: int) -> None:
        self.done += n
        if self.done >= self.next:
            self.next += self.every
            rate = self.done / max(time.monotonic() - self.start, 1e-9) / 1e6
            print(f"  {self.done / 1e9:6.1f} GB extracted ({rate:.0f} MB/s)",
                  file=sys.stderr, flush=True)


def copy_member(src, dest: Path, size: int, progress: Progress) -> None:
    """Write a member through a .part file so a cut stream leaves no valid-looking file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    with open(part, "wb") as out:
        while chunk := src.read(CHUNK):
            out.write(chunk)
            written += len(chunk)
            progress(len(chunk))
    if written != size:
        raise OSError(f"{dest.name}: got {written} of {size} bytes")
    part.replace(dest)


class HeadTailCapture:
    """Keep the first ``head_limit`` bytes on disk and the last ``tail_limit`` in memory."""

    def __init__(self, head_path: Path, head_limit: int, tail_limit: int):
        self.head_path = head_path
        self.head = open(head_path, "wb")
        self.head_limit = head_limit
        self.tail_limit = tail_limit
        self.tail: deque[bytes] = deque()
        self.tail_len = 0
        self.size = 0

    def write(self, chunk: bytes) -> None:
        if self.size < self.head_limit:
            self.head.write(chunk[: self.head_limit - self.size])
        self.size += len(chunk)
        self.tail.append(chunk)
        self.tail_len += len(chunk)
        while self.tail_len - len(self.tail[0]) >= self.tail_limit:
            self.tail_len -= len(self.tail.popleft())

    def finish(self) -> bytes:
        self.head.close()
        return b"".join(self.tail)[-self.tail_limit:]


class HeadTailFile(io.RawIOBase):
    """Seekable read-only view of a file of which only a prefix and a suffix are known."""

    def __init__(self, head_path: Path, tail: bytes, size: int):
        self._head = open(head_path, "rb")
        self._head_len = head_path.stat().st_size
        self._tail = memoryview(tail)
        self._tail_start = size - len(tail)
        self._size = size
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, buf) -> int:
        view = memoryview(buf).cast("B")
        n = min(len(view), max(0, self._size - self._pos))
        done = 0
        while done < n:
            pos = self._pos + done
            if pos < self._head_len:
                self._head.seek(pos)
                got = self._head.readinto(view[done:done + min(n - done, self._head_len - pos)])
                if not got:
                    raise OSError("short read from captured head")
                done += got
            elif pos >= self._tail_start:
                offset = pos - self._tail_start
                view[done:n] = self._tail[offset:offset + n - done]
                done = n
            else:
                raise OSError(f"byte {pos} of {self._size} was not captured")
        self._pos += n
        return n

    def close(self) -> None:
        self._head.close()
        super().close()


def leading_complete_row_groups(meta, limit: int, wanted: int) -> int:
    """Count leading row groups whose column chunks all end at or before ``limit``."""
    count = 0
    for i in range(min(wanted, meta.num_row_groups)):
        group = meta.row_group(i)
        end = 0
        for j in range(group.num_columns):
            col = group.column(j)
            start = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset:
                start = min(start, col.dictionary_page_offset)
            end = max(end, start + col.total_compressed_size)
        if end > limit:
            break
        count += 1
    return count


def write_train_sample(head_path: Path, tail: bytes, size: int, wanted: int,
                       sample: Path, footer: Path) -> tuple[int, int]:
    """Write the footer file and the leading-sequence sample; return (kept, total) groups."""
    import pyarrow.parquet as pq

    if tail[-4:] != MAGIC:
        raise ValueError("training file does not end with the Parquet magic")
    footer_len = int.from_bytes(tail[-8:-4], "little")
    if footer_len + 8 > len(tail):
        raise ValueError(f"footer is {footer_len} bytes; rerun with --tail-bytes above that")
    footer_part = footer.with_name(footer.name + ".part")
    footer_part.write_bytes(MAGIC + tail[-(footer_len + 8):])
    footer_part.replace(footer)

    head_len = head_path.stat().st_size
    limit = size if head_len >= size else head_len - READ_MARGIN
    with HeadTailFile(head_path, tail, size) as source:
        parquet = pq.ParquetFile(source)
        meta = parquet.metadata
        kept = leading_complete_row_groups(meta, limit, wanted)
        if kept == 0:
            raise ValueError("no complete training sequence fits in the captured head")
        codec = meta.row_group(0).column(0).compression.lower()
        codec = "none" if codec == "uncompressed" else codec
        sample_part = sample.with_name(sample.name + ".part")
        with pq.ParquetWriter(sample_part, parquet.schema_arrow, compression=codec) as writer:
            for i in range(kept):
                table = parquet.read_row_group(i, use_threads=False)
                writer.write_table(table, row_group_size=table.num_rows)
        total = meta.num_row_groups
    if pq.ParquetFile(sample_part).metadata.num_row_groups != kept:
        raise ValueError("training sample was written with the wrong row-group layout")
    sample_part.replace(sample)
    return kept, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", nargs="?", default="-",
                    help="starter pack .tar.gz path, or - to read stdin (default)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repository root to populate (default: this repository)")
    ap.add_argument("--train-sample", type=int, metavar="N",
                    help="keep only the first N training sequences plus the full footer")
    ap.add_argument("--tail-bytes", type=int, default=1536 << 20,
                    help="bytes kept from the end of train.parquet to recover its footer")
    ap.add_argument("--train-head-bytes", type=int, help=argparse.SUPPRESS)  # test override
    args = ap.parse_args(argv)
    if args.train_sample is not None and args.train_sample < 1:
        ap.error("--train-sample must be positive")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    train_dest = root.joinpath(*TRAIN)
    datasets = train_dest.parent
    progress = Progress()
    captured = None
    written: list[Path] = []

    source = sys.stdin.buffer if args.archive == "-" else open(args.archive, "rb")
    with source, tarfile.open(fileobj=source, mode="r|gz", bufsize=CHUNK) as tar:
        for member in tar:
            dest = destination(member.name, root)
            if dest is None:
                continue
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                print(f"skipping non-regular member {member.name}", file=sys.stderr)
                continue
            fileobj = tar.extractfile(member)
            if dest == train_dest and args.train_sample is not None:
                head = args.train_head_bytes or int(
                    member.size * args.train_sample / TRAIN_SEQUENCES * 1.2) + (64 << 20)
                head = min(head, member.size)
                ensure_space(root, head * 2, f"a {args.train_sample}-sequence training sample")
                datasets.mkdir(parents=True, exist_ok=True)
                capture = HeadTailCapture(datasets / ".train.head.part", head, args.tail_bytes)
                while chunk := fileobj.read(CHUNK):
                    capture.write(chunk)
                    progress(len(chunk))
                if capture.size != member.size:
                    raise OSError(f"train.parquet: got {capture.size} of {member.size} bytes")
                tail = capture.finish()
                # Keep the tail on disk too, so a failed finalize can be redone by hand.
                (datasets / ".train.tail.part").write_bytes(tail)
                captured = (capture.head_path, tail, member.size)
                print(f"captured train.parquet head ({head / 1e9:.2f} GB) and tail; "
                      "continuing the stream", file=sys.stderr, flush=True)
                continue
            ensure_space(root, member.size, dest.name)
            copy_member(fileobj, dest, member.size, progress)
            written.append(dest)

    if captured is not None:
        head_path, tail, size = captured
        sample = datasets / "train_head.parquet"
        footer = datasets / "train.parquet.footer"
        kept, total = write_train_sample(head_path, tail, size, args.train_sample, sample, footer)
        head_path.unlink()
        (datasets / ".train.tail.part").unlink()
        written += [sample, footer]
        note = "" if kept == args.train_sample else f" (requested {args.train_sample})"
        print(f"kept {kept} of {total} training sequences{note}", file=sys.stderr)

    for path in written:
        print(f"{path.stat().st_size:>15,d}  {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
