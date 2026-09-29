"""Stream training row groups straight from the starter pack archive, never touching disk.

The 29 GB ``train.parquet`` does not fit on this disk. The archive server
honours byte ranges over HTTP/1.1, but the archive is one gzip stream, so it
can only be read in order. ``RemoteRowGroups`` reads it once per epoch in a
background thread. It parses the tar headers, finds ``train.parquet``, and
cuts out the bytes of each wanted row group using the offsets in the saved
footer (``datasets/train.parquet.footer``). ``decode_row_group`` then turns
one row group's bytes into a table, with the footer as metadata.

Two facts about the server shape the reader:

* A stream left idle for 60 s is dropped (30 s is fine), so the reader never
  stops reading. When its buffer is full it trickles 1 MB every 0.5 s.
* Connections can still break. ``ResumableGzipStream`` snapshots the
  decompressor every 256 MB of input, and after an error it resumes from the
  last snapshot with a range request instead of starting over.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import threading
import time
import zlib
from pathlib import Path

import pyarrow.parquet as pq

URL = "https://files.wundernn.io/wnn_connectome_starterpack.tar.gz"
MEMBER = "wnn_connectome_starterpack/datasets/train.parquet"
CHUNK = 1 << 20
TAR_BUFFER = 8 << 20


class ResumableGzipStream(io.RawIOBase):
    """The uncompressed bytes of a remote .gz, read in order, surviving broken connections."""

    def __init__(self, url: str, *, snapshot_every: int = 256 << 20, retries: int = 8,
                 fail_after: int | None = None):
        self.url, self.snapshot_every, self.retries = url, snapshot_every, retries
        self._fail_after = fail_after  # test hook: kill the transfer once past this input offset
        self._decompressor = zlib.decompressobj(wbits=31)
        self._snapshot = (0, 0, self._decompressor.copy())  # (input offset, output offset, state)
        self._in = 0  # compressed bytes consumed
        self._out = 0  # uncompressed bytes produced so far (high-water mark)
        self._discard = 0  # re-produced bytes to drop after a resume
        self._pending = memoryview(b"")
        self.resumes = 0
        self._stopped = False
        self._proc = self._spawn(0)

    def _spawn(self, start: int) -> subprocess.Popen:
        command = ["curl", "-sS", "-fL", "--http1.1", "--retry", "3", "-r", f"{start}-", self.url]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        while not self._pending:
            if self._decompressor.eof or not self._fill():
                return 0
        n = min(len(buf), len(self._pending))
        buf[:n] = self._pending[:n]
        self._pending = self._pending[n:]
        return n

    def _fill(self) -> bool:
        """Decompress one more chunk into the pending buffer; False at the end of the stream."""
        chunk = self._proc.stdout.read(CHUNK)
        if self._fail_after is not None and self._in > self._fail_after:
            self._fail_after = None
            self._proc.kill()
            chunk = b""
        if not chunk:
            code = self._proc.wait()
            if self._decompressor.eof or self._stopped:
                return False
            self._resume(f"transfer ended early (curl exit {code})")
            return True
        self._in += len(chunk)
        out = self._decompressor.decompress(chunk)
        if self._discard:
            drop = min(self._discard, len(out))
            out, self._discard = out[drop:], self._discard - drop
        self._out += len(out)
        self._pending = memoryview(out)
        if self._in - self._snapshot[0] >= self.snapshot_every:
            self._snapshot = (self._in, self._out, self._decompressor.copy())
        return True

    def _resume(self, reason: str) -> None:
        if self.resumes >= self.retries:
            raise OSError(f"archive stream failed {self.resumes} times; last: {reason}")
        self.resumes += 1
        start, produced, state = self._snapshot
        self._proc.kill()
        self._decompressor = state.copy()
        self._in, self._discard = start, self._out - produced
        self._proc = self._spawn(start)

    def close(self) -> None:
        self._stopped = True
        if getattr(self, "_proc", None) is not None and self._proc.poll() is None:
            self._proc.kill()
        super().close()


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
            with ResumableGzipStream(self.url, **self.stream_options) as stream, \
                    tarfile.open(fileobj=stream, mode="r|", bufsize=TAR_BUFFER) as tar:
                for member in tar:
                    if member.name == MEMBER:
                        break
                else:
                    raise OSError(f"{MEMBER} not found in the archive")
                if member.size != self.size:
                    raise OSError(f"archive train.parquet is {member.size} bytes, footer implies {self.size}")
                source = tar.extractfile(member)
                position = 0
                for group in self.groups:
                    start, end = self.spans[group]
                    while position < start:  # skip at full speed: nothing is buffered
                        if self._closing:
                            return
                        skipped = source.read(min(TAR_BUFFER, start - position))
                        if not skipped:
                            raise OSError("archive ended before the wanted row groups")
                        position += len(skipped)
                    data = self._read(source, end - start)
                    position = end
                    with self._cond:
                        if self._closing:
                            return
                        self._buffer[group] = data
                        self._used += len(data)
                        self._cond.notify_all()
                self.log(f"  remote: read {len(self.groups)} row groups "
                         f"({stream.resumes} resumed connections)")
        except BaseException as exc:  # surfaced to the consumer by get()
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _read(self, source, length: int) -> bytes:
        """Read one row group, trickling while the buffer is full so the connection stays busy."""
        parts, remaining = [], length
        while remaining:
            full = self._used >= self.cap
            piece = source.read(min(CHUNK if full else TAR_BUFFER, remaining))
            if not piece:
                raise OSError("archive ended inside a row group")
            parts.append(piece)
            remaining -= len(piece)
            if full:
                time.sleep(0.5)
            if self._closing:
                break
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
