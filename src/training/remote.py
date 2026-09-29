"""Stream training row groups straight from the starter pack archive, never touching disk.

The 29 GB ``train.parquet`` does not fit on this disk. The archive server
honours byte ranges over HTTP/1.1, but the archive is one gzip stream, so it
can only be read in order. ``RemoteRowGroups`` reads it once per epoch in a
background thread. It finds ``train.parquet``'s data in the tar stream, and
cuts out the bytes of each wanted row group using the offsets in the saved
footer (``datasets/train.parquet.footer``). ``decode_row_group`` then turns
one row group's bytes into a table, with the footer as metadata.

Three facts about the transfer shape the reader:

* A stream left idle for 60 s is dropped (30 s is fine), so the reader never
  stops reading. When its buffer is full it trickles 1 MB every 0.5 s.
* Connections can break. ``ResumableGzipStream`` snapshots the decompressor
  every 256 MB of input and resumes from the latest snapshot with a range
  request. curl's own ``--retry`` is not used: a retried transfer restarts its
  range and would splice repeated bytes into the stream.
* Bytes can arrive corrupt anyway (one run of A3b met an invalid deflate
  code 17 GB in). So every row group is decoded and checked before it is
  handed over, and one that fails, or a stream that stops inflating, is read
  again from the latest snapshot before the row group's start.
"""

from __future__ import annotations

import bisect
import io
import subprocess
import tarfile
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.data.schema import FEATURE_COLUMNS, SEQUENCE_LENGTH, SchemaError

URL = "https://files.wundernn.io/wnn_connectome_starterpack.tar.gz"
MEMBER = "wnn_connectome_starterpack/datasets/train.parquet"
CHUNK = 1 << 20
TAR_BUFFER = 8 << 20
FETCH_ATTEMPTS = 4  # reads of one row group before giving up on it
FEATURE_BOUND = 8.0  # rank-Gaussian inputs saturate at 5.2; corrupt floats rarely stay inside


class CorruptStream(OSError):
    """The compressed bytes stopped inflating: they must be read again."""


class StreamFailed(OSError):
    """The archive cannot be read: out of restarts, or it ended early."""


class ResumableGzipStream:
    """The uncompressed bytes of a remote .gz, read in order, with restarts from snapshots.

    ``position`` is the uncompressed offset of the next byte ``read`` returns.
    A broken transfer resumes at ``position`` from the latest snapshot at or
    before it; ``rewind`` re-reads from an earlier offset the same way.
    """

    def __init__(self, url: str, *, snapshot_every: int = 256 << 20, retries: int = 16,
                 fail_after: int | None = None, corrupt_after: int | None = None):
        self.url, self.snapshot_every, self.retries = url, snapshot_every, retries
        # Test hooks, each firing once past a compressed offset: a broken transfer, corrupt bytes.
        self._fail_after, self._corrupt_after = fail_after, corrupt_after
        self._decompressor = zlib.decompressobj(wbits=31)
        self._snapshots = [(0, 0, self._decompressor.copy())]  # (input offset, output offset, state)
        self._in = 0  # compressed bytes fed to the decompressor
        self._made = 0  # uncompressed offset of the decompressor's next output byte
        self.position = 0
        self._pending = memoryview(b"")  # decompressed bytes from ``position`` on
        self.restarts = 0
        self._stopped = False
        self._proc = self._spawn(0)

    def _spawn(self, start: int) -> subprocess.Popen:
        command = ["curl", "-sS", "-fL", "--http1.1", "-r", f"{start}-", self.url]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self, n: int = -1) -> bytes:
        """Up to ``n`` bytes from ``position`` on (all of them if n < 0); b"" at the end."""
        if n < 0:
            return b"".join(iter(lambda: self.read(TAR_BUFFER), b""))
        while not self._pending:
            if not self._fill():
                return b""
        out = bytes(self._pending[:n])
        self._pending = self._pending[n:]
        self.position += len(out)
        return out

    def _fill(self) -> bool:
        """Inflate the next compressed chunk; False at the end of the stream."""
        if self._decompressor.eof:
            return False
        chunk = self._proc.stdout.read(CHUNK)
        if self._fail_after is not None and self._in > self._fail_after:
            self._fail_after = None
            self._proc.kill()
            chunk = b""
        if not chunk:
            self._proc.wait()
            if self._stopped:
                return False
            self._restart(self.position, "the transfer ended early")
            return True
        if self._corrupt_after is not None and self._in > self._corrupt_after:
            self._corrupt_after = None
            chunk = bytes(b ^ 0x5A for b in chunk[:4096]) + chunk[4096:]
        self._in += len(chunk)
        try:
            out = self._decompressor.decompress(chunk)
        except zlib.error as exc:
            raise CorruptStream(f"deflate error near compressed offset {self._in}: {exc}") from exc
        start, self._made = self._made, self._made + len(out)
        if self._made > self.position:  # after a restart, output before ``position`` is dropped
            self._pending = memoryview(out)[max(0, self.position - start):]
        if self._in - self._snapshots[-1][0] >= self.snapshot_every:
            self._snapshots.append((self._in, self._made, self._decompressor.copy()))
        return True

    def rewind(self, offset: int, older: int = 0) -> None:
        """Continue reading from uncompressed ``offset``, which may be behind ``position``.

        Restarts from the latest snapshot at or before ``offset``, or ``older``
        snapshots before that one, in case bad bytes reached a snapshot before
        they stopped inflating.
        """
        self._restart(offset, f"re-reading from {offset}", older)

    def _restart(self, offset: int, reason: str, older: int = 0) -> None:
        if self.restarts >= self.retries:
            raise StreamFailed(f"archive stream restarted {self.restarts} times; last: {reason}")
        self.restarts += 1
        # Snapshots past the restart point may have seen bad bytes: they are retaken on the way.
        keep = bisect.bisect_right([made for _, made, _ in self._snapshots], offset)
        del self._snapshots[max(1, keep - older):]
        start, made, state = self._snapshots[-1]
        self._kill()
        self._decompressor = state.copy()
        self._in, self._made = start, made
        self.position, self._pending = offset, memoryview(b"")
        self._proc = self._spawn(start)

    def _kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()

    def close(self) -> None:
        self._stopped = True
        self._kill()


def find_member(stream: ResumableGzipStream, name: str, limit: int = 64 << 20) -> tuple[int, int]:
    """Uncompressed data offset and size of ``name`` in the tar stream, read from its start."""
    head = bytearray()
    while len(head) < limit:
        piece = stream.read(CHUNK)
        if not piece:
            break
        head += piece
        try:
            with tarfile.open(fileobj=io.BytesIO(bytes(head)), mode="r:") as tar:
                for member in tar:
                    if member.name == name:
                        return member.offset_data, member.size
        except tarfile.ReadError:  # a header cut off by the end of what has been read
            continue
    raise OSError(f"{name} not found in the archive's first {len(head)} bytes")


def row_group_spans(meta) -> list[tuple[int, int]]:
    """Byte range [start, end) of every row group in the file the footer describes."""
    spans = []
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        start, end = None, 0
        for j in range(group.num_columns):
            col = group.column(j)
            first = col.dictionary_page_offset if col.has_dictionary_page and col.dictionary_page_offset \
                else col.data_page_offset
            start = first if start is None else min(start, first)
            end = max(end, first + col.total_compressed_size)
        spans.append((start, end))
    return spans


class _Span(io.RawIOBase):
    """A file of ``size`` bytes of which only [start, start + len(data)) is known."""

    def __init__(self, data: bytes, start: int, size: int):
        self._data, self._start, self._size, self._pos = memoryview(data), start, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._pos = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence] + offset
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self._size - self._pos)
        lo = self._pos - self._start
        if lo < 0 or lo + n > len(self._data):
            raise OSError(f"read of {n} bytes at {self._pos} is outside the captured row group")
        buf[:n] = self._data[lo:lo + n]
        self._pos += n
        return n


def decode_row_group(meta, size: int, group: int, start: int, data: bytes, columns=None):
    """Decode one row group from its bytes alone, with the saved footer as metadata."""
    return pq.ParquetFile(_Span(data, start, size), metadata=meta).read_row_group(group, columns=columns)


def check_row_group(table) -> None:
    """Raise unless ``table`` is one whole sequence with bounded, finite features."""
    if table.num_rows != SEQUENCE_LENGTH:
        raise SchemaError(f"{table.num_rows} rows")
    ids, steps = table["seq_ix"].to_numpy(), table["step_in_seq"].to_numpy()
    if (ids != ids[0]).any() or (steps != np.arange(SEQUENCE_LENGTH)).any():
        raise SchemaError("not one whole sequence")
    for name in FEATURE_COLUMNS:
        values = table[name].to_numpy()
        if not (np.isfinite(values).all() and np.abs(values).max() <= FEATURE_BOUND):
            raise SchemaError(f"feature {name} is out of range")


class RemoteRowGroups:
    """Wanted row groups of the archive's train.parquet, read ahead into a bounded RAM buffer."""

    def __init__(self, footer: Path, groups, *, url: str = URL, buffer_bytes: int = 1200 << 20,
                 timeout: float = 1800.0, log=print, stream_options: dict | None = None,
                 meta=None, spans=None):
        self.meta = meta if meta is not None else pq.read_metadata(footer)
        self.spans = spans if spans is not None else row_group_spans(self.meta)
        footer_len = footer.stat().st_size - 12  # the footer file is PAR1 + footer + length + PAR1
        self.size = self.spans[-1][1] + footer_len + 8  # row groups are followed by footer, length, PAR1
        self.groups = sorted(groups)
        self.url, self.cap, self.timeout, self.log = url, buffer_bytes, timeout, log
        self.stream_options = stream_options or {}
        self.rereads = 0
        self._buffer: dict[int, bytes] = {}
        self._used = 0
        self._error: BaseException | None = None
        self._done = False
        self._closing = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            with ResumableGzipStream(self.url, **self.stream_options) as stream:
                offset, size = find_member(stream, MEMBER)
                if size != self.size:
                    raise OSError(f"archive train.parquet is {size} bytes, footer implies {self.size}")
                for group in self.groups:
                    data = self._fetch(stream, group, offset)
                    with self._cond:
                        if self._closing or data is None:
                            return
                        self._buffer[group] = data
                        self._used += len(data)
                        self._cond.notify_all()
                self.log(f"  remote: read {len(self.groups)} row groups ({stream.restarts} stream "
                         f"restarts, {self.rereads} row groups re-read)")
        except BaseException as exc:  # surfaced to the consumer by get()
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _fetch(self, stream: ResumableGzipStream, group: int, offset: int) -> bytes | None:
        """One row group's checked bytes, re-read from a snapshot while they come out corrupt."""
        start, end = self.spans[group]
        for attempt in range(FETCH_ATTEMPTS):
            try:
                if not self._skip_to(stream, offset + start):
                    return None
                data = self._read(stream, end - start)
                if data is None:
                    return None
                check_row_group(decode_row_group(self.meta, self.size, group, start, data))
                return data
            except StreamFailed:
                raise
            except Exception as exc:  # corrupt input either stops inflating or fails to decode
                self.rereads += 1
                self.log(f"  remote: row group {group} came out corrupt ({exc}); reading it again")
                stream.rewind(offset + start, older=attempt)
        raise OSError(f"row group {group} was still corrupt after {FETCH_ATTEMPTS} reads")

    def _skip_to(self, stream: ResumableGzipStream, target: int) -> bool:
        """Advance to ``target`` at full speed, nothing buffered; False when closing."""
        if stream.position > target:
            stream.rewind(target)
        while stream.position < target:
            if self._closing:
                return False
            if not stream.read(min(TAR_BUFFER, target - stream.position)):
                raise StreamFailed("archive ended before the wanted row groups")
        return True

    def _read(self, stream: ResumableGzipStream, length: int) -> bytes | None:
        """Read one row group, trickling while the buffer is full so the connection stays busy."""
        parts, remaining = [], length
        while remaining:
            if self._closing:
                return None
            full = self._used >= self.cap
            piece = stream.read(min(CHUNK if full else TAR_BUFFER, remaining))
            if not piece:
                raise StreamFailed("archive ended inside a row group")
            parts.append(piece)
            remaining -= len(piece)
            if full:
                time.sleep(0.5)
        return b"".join(parts)

    def get(self, group: int, columns=None):
        """Block until ``group`` has arrived, then decode it and free its buffer space."""
        deadline = time.monotonic() + self.timeout
        with self._cond:
            while group not in self._buffer:
                if self._error is not None:
                    raise OSError(f"remote row group {group} unavailable") from self._error
                if self._done:
                    raise OSError(f"remote reader finished without row group {group}")
                if not self._cond.wait(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"remote row group {group} did not arrive in {self.timeout:.0f}s")
            data = self._buffer.pop(group)
            self._used -= len(data)
        return decode_row_group(self.meta, self.size, group, self.spans[group][0], data, columns)

    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._thread.join(timeout=30)
