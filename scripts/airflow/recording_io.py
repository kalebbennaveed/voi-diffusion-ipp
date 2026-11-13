"""Atomic compressed recordings without a second full in-memory JSON copy."""
import gzip
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        # Map frames are numeric buffers. Check finiteness in NumPy instead
        # of revisiting every boxed scalar in Python after conversion.
        if value.dtype.kind in 'biuf' and np.isfinite(value).all():
            return value.tolist()
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _encode(value):
    return json.dumps(_clean(value), separators=(',', ':'), allow_nan=False,
                      default=lambda item: np.asarray(item).tolist()).encode()


def recording_chunks(record, chunk_bytes=262144):
    """Same JSON values/order as the original encoder, one recorded row at a time.

    Top-level frame/plan/sample lists dominate recording size. Encoding their
    individual rows avoids copying all accumulated frames and then allocating
    the entire JSON string and UTF-8 payload at every checkpoint. Numbers,
    precision, frame cadence and source data are unchanged.
    """
    pending = bytearray()

    def pieces():
        yield b'{'
        for index, (key, value) in enumerate(record.items()):
            if index:
                yield b','
            yield _encode(key) + b':'
            if isinstance(value, (list, tuple)):
                yield b'['
                for row, item in enumerate(value):
                    if row:
                        yield b','
                    yield _encode(item)
                yield b']'
            else:
                yield _encode(value)
        yield b'}'

    for piece in pieces():
        pending.extend(piece)
        if len(pending) >= chunk_bytes:
            yield bytes(pending)
            pending.clear()
    if pending:
        yield bytes(pending)


def write_recording(path, record):
    """Publish only a completely written gzip; preserve the last good checkpoint."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=3) as stream:
                for chunk in recording_chunks(record):
                    stream.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path, value):
    """Atomically write a JSON summary, including NumPy and JAX arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def json_value(item):
        if isinstance(item, (dict, list, tuple)):
            if isinstance(item, dict):
                return {key: json_value(val) for key, val in item.items()}
            return [json_value(val) for val in item]
        if hasattr(item, "__array__"):
            return json_value(np.asarray(item).tolist())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    temporary.write_text(json.dumps(json_value(value), separators=(',', ':'), allow_nan=False))
    temporary.replace(path)
