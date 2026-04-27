"""Newline-delimited JSON helpers for the bridge protocol.

Bridges spawned by the Electron main process exchange messages as
newline-delimited JSON over stdin/stdout. Stdout is reserved for protocol
messages — anything else (logging, library banners) must go to stderr or it
will break the parser on the TS side.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any, TextIO


def emit(message: dict[str, Any], *, stream: TextIO | None = None) -> None:
    """Write a single JSON message followed by a newline and flush."""
    out = stream if stream is not None else sys.stdout
    out.write(json.dumps(message, ensure_ascii=False))
    out.write("\n")
    out.flush()


def read_lines(stream: TextIO | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSON messages parsed from each non-empty line on the stream."""
    src = stream if stream is not None else sys.stdin
    for raw in src:
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
