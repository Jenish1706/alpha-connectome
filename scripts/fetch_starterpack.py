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
(``datasets/train_head.parquet``: the original bytes of those row groups under
a rewritten footer), together with the complete training footer
(``datasets/train.parquet.footer``). ``pyarrow.parquet.read_metadata`` reads
that footer, so the structure of every training sequence can still be
validated without the data.
"""

from __future__ import annotations

import argparse
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


def leading_complete_row_groups(meta, limit: int, wanted: int) -> int:
    """Count leading row groups whose column chunks all end at or before ``limit``."""
    count = 0
    for i in range(min(wanted, meta.num_row_groups)):
        if _group_end(meta.row_group(i)) > limit:
            break
        count += 1
    return count


def _group_end(group) -> int:
    """Byte offset just past a row group's last column chunk."""
    end = 0
    for j in range(group.num_columns):
        col = group.column(j)
        start = col.data_page_offset
        if col.has_dictionary_page and col.dictionary_page_offset:
            start = min(start, col.dictionary_page_offset)
        end = max(end, start + col.total_compressed_size)
    return end


# Thrift compact protocol, enough to cut the row-group list of a Parquet footer.
CT_STOP, CT_TRUE, CT_FALSE, CT_BYTE, CT_I16, CT_I32, CT_I64 = 0, 1, 2, 3, 4, 5, 6
CT_DOUBLE, CT_BINARY, CT_LIST, CT_SET, CT_MAP, CT_STRUCT = 7, 8, 9, 10, 11, 12


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _list_header(buf: bytes, pos: int) -> tuple[int, int, int]:
    size, element = buf[pos] >> 4, buf[pos] & 0x0F
    pos += 1
    if size == 15:
        size, pos = _varint(buf, pos)
    return size, element, pos


def _skip(buf: bytes, pos: int, ctype: int, in_container: bool = False) -> int:
    if ctype in (CT_TRUE, CT_FALSE):
        return pos + 1 if in_container else pos  # struct fields carry bools in the type
    if ctype == CT_BYTE:
        return pos + 1
    if ctype in (CT_I16, CT_I32, CT_I64):
        return _varint(buf, pos)[1]
    if ctype == CT_DOUBLE:
        return pos + 8
    if ctype == CT_BINARY:
        length, pos = _varint(buf, pos)
        return pos + length
    if ctype in (CT_LIST, CT_SET):
        size, element, pos = _list_header(buf, pos)
        for _ in range(size):
            pos = _skip(buf, pos, element, True)
        return pos
    if ctype == CT_MAP:
        size, pos = _varint(buf, pos)
        if size:
            key, value = buf[pos] >> 4, buf[pos] & 0x0F
            pos += 1
            for _ in range(size):
                pos = _skip(buf, _skip(buf, pos, key, True), value, True)
        return pos
    if ctype == CT_STRUCT:
        while buf[pos] != CT_STOP:
            header = buf[pos]
            pos += 1
            if not header >> 4:
                pos = _varint(buf, pos)[1]  # long-form field id
            pos = _skip(buf, pos, header & 0x0F)
        return pos + 1
    raise ValueError(f"unknown thrift compact type {ctype}")


def truncate_footer(footer: bytes, keep: int, num_rows: int) -> bytes:
    """Re-serialize a FileMetaData keeping its first ``keep`` row groups.

    Only field 3 (num_rows) and field 4 (row_groups) change; every other byte,
    including the schema and ARROW:schema metadata, is copied as is.
    """
    out, last, pos, field, seen = bytearray(), 0, 0, 0, set()
    while footer[pos] != CT_STOP:
        header = footer[pos]
        pos += 1
        ctype = header & 0x0F
        if header >> 4:
            field += header >> 4
        else:
            raw, pos = _varint(footer, pos)
            field = (raw >> 1) ^ -(raw & 1)
        if field == 3 and ctype == CT_I64:
            end = _skip(footer, pos, ctype)
            out += footer[last:pos] + _encode_varint(num_rows << 1)
            last, pos = end, end
            seen.add(3)
        elif field == 4 and ctype == CT_LIST:
            size, element, body = _list_header(footer, pos)
            if element != CT_STRUCT or not 0 < keep <= size:
                raise ValueError(f"cannot keep {keep} of {size} row groups")
            cut = body
            for i in range(size):
                if i == keep:
                    cut_at = cut
                cut = _skip(footer, cut, CT_STRUCT)
            if keep == size:
                cut_at = cut
            header_bytes = bytes([keep << 4 | CT_STRUCT]) if keep < 15 else \
                bytes([0xF0 | CT_STRUCT]) + _encode_varint(keep)
            out += footer[last:pos] + header_bytes + footer[body:cut_at]
            last, pos = cut, cut
            seen.add(4)
        else:
            pos = _skip(footer, pos, ctype)
    if seen != {3, 4}:
        raise ValueError("footer has no num_rows or row_groups field")
    return bytes(out + footer[last:])


def write_train_sample(head_path: Path, tail: bytes, wanted: int,
                       sample: Path, footer: Path) -> tuple[int, int]:
    """Turn the captured head into a valid file holding its leading whole row groups.

    The kept row groups stay byte-identical to the original; only the footer
    is rewritten. Also writes the complete original footer. Returns
    (kept, total) row groups.
    """
    import pyarrow.parquet as pq

    if tail[-4:] != MAGIC:
        raise ValueError("training file does not end with the Parquet magic")
    footer_len = int.from_bytes(tail[-8:-4], "little")
    if footer_len + 8 > len(tail):
        raise ValueError(f"footer is {footer_len} bytes; rerun with --tail-bytes above that")
    original = tail[-(footer_len + 8):-8]
    footer_part = footer.with_name(footer.name + ".part")
    footer_part.write_bytes(MAGIC + original + tail[-8:])
    meta = pq.read_metadata(footer_part)
    footer_part.replace(footer)

    head_len = head_path.stat().st_size
    kept = leading_complete_row_groups(meta, head_len, wanted)
    if kept == 0:
        raise ValueError("no complete training sequence fits in the captured head")
    end = max(_group_end(meta.row_group(i)) for i in range(kept))
    rows = sum(meta.row_group(i).num_rows for i in range(kept))
    new_footer = truncate_footer(original, kept, rows)
    with open(head_path, "r+b") as out:
        out.truncate(end)
        out.seek(end)
        out.write(new_footer + len(new_footer).to_bytes(4, "little") + MAGIC)
    check = pq.ParquetFile(head_path)
    if check.metadata.num_row_groups != kept or check.metadata.num_rows != rows:
        raise ValueError("rewritten training footer does not describe the kept row groups")
    if not check.schema_arrow.equals(meta.schema.to_arrow_schema()):
        raise ValueError("rewritten training footer changed the schema")
    for group in {0, kept - 1}:
        check.read_row_group(group)  # decodes, so a bad cut fails here
    head_path.replace(sample)
    return kept, meta.num_row_groups


class _Stream:
    """What the extraction loop has captured so far."""

    def __init__(self):
        self.capture: HeadTailCapture | None = None
        self.captured: tuple[Path, bytes] | None = None
        self.written: list[Path] = []


def _extract(tar, args, root: Path, progress: Progress, state: _Stream) -> None:
    train_dest = root.joinpath(*TRAIN)
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
                member.size * args.train_sample / TRAIN_SEQUENCES * 1.05) + (64 << 20)
            head = min(head, member.size)
            ensure_space(root, head, f"a {args.train_sample}-sequence training sample")
            dest.parent.mkdir(parents=True, exist_ok=True)
            state.capture = HeadTailCapture(dest.parent / ".train.head.part", head, args.tail_bytes)
            while chunk := fileobj.read(CHUNK):
                state.capture.write(chunk)
                progress(len(chunk))
            if state.capture.size != member.size:
                raise OSError(f"train.parquet: got {state.capture.size} of {member.size} bytes")
            tail = state.capture.finish()
            # Keep the tail on disk too, so a failed finalize can be redone by hand.
            (dest.parent / ".train.tail.part").write_bytes(tail)
            state.captured = (state.capture.head_path, tail)
            print(f"captured train.parquet head ({head / 1e9:.2f} GB) and tail; "
                  "continuing the stream", file=sys.stderr, flush=True)
            continue
        if args.skip_existing and dest.exists() and dest.stat().st_size == member.size:
            print(f"keeping existing {dest.relative_to(root)}", file=sys.stderr)
            continue
        ensure_space(root, member.size, dest.name)
        copy_member(fileobj, dest, member.size, progress)
        state.written.append(dest)


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
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep local files whose size matches the archive entry")
    ap.add_argument("--train-head-bytes", type=int, help=argparse.SUPPRESS)  # test override
    args = ap.parse_args(argv)
    if args.train_sample is not None and args.train_sample < 1:
        ap.error("--train-sample must be positive")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = root.joinpath(*TRAIN).parent
    footer = datasets / "train.parquet.footer"
    state, stream_error = _Stream(), None
    source = sys.stdin.buffer if args.archive == "-" else open(args.archive, "rb")
    try:
        with source, tarfile.open(fileobj=source, mode="r|gz", bufsize=CHUNK) as tar:
            _extract(tar, args, root, Progress(), state)
    except (tarfile.ReadError, EOFError) as exc:  # the download was cut short
        stream_error = exc
    head = state.capture
    if (state.captured is None and head is not None and head.size >= head.head_limit
            and footer.exists()):
        # The stream broke after the head was complete: finish from the saved footer.
        head.head.close()
        state.captured = (head.head_path, footer.read_bytes())
        print(f"stream ended early ({stream_error}); finishing the training sample from "
              f"{footer.name}", file=sys.stderr)

    if state.captured is not None:
        head_path, tail = state.captured
        sample = datasets / "train_head.parquet"
        kept, total = write_train_sample(head_path, tail, args.train_sample, sample, footer)
        (datasets / ".train.tail.part").unlink(missing_ok=True)
        state.written += [sample, footer]
        note = "" if kept == args.train_sample else f" (requested {args.train_sample})"
        print(f"kept {kept} of {total} training sequences{note}", file=sys.stderr)

    for path in state.written:
        print(f"{path.stat().st_size:>15,d}  {path.relative_to(root)}")
    if stream_error is not None:
        print(f"the archive stream failed ({stream_error}); later entries are missing. "
              "Rerun with --skip-existing to fetch them.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
