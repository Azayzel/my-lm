"""Bridge I/O: newline-delimited JSON over stdin/stdout."""

from mylm.io.jsonl import emit, read_lines

__all__ = ["emit", "read_lines"]
