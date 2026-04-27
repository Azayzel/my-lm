"""Tests for the bridge protocol I/O helpers."""

from __future__ import annotations

import io
import json

from mylm.io import emit, read_lines


def test_emit_writes_compact_json_with_newline() -> None:
    buf = io.StringIO()
    emit({"type": "progress", "step": 3, "total": 10}, stream=buf)
    line = buf.getvalue()
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed == {"type": "progress", "step": 3, "total": 10}


def test_emit_preserves_unicode() -> None:
    buf = io.StringIO()
    emit({"text": "café — résumé 🚀"}, stream=buf)
    parsed = json.loads(buf.getvalue())
    assert parsed["text"] == "café — résumé 🚀"


def test_read_lines_skips_blank_and_invalid() -> None:
    src = io.StringIO('{"a": 1}\n\n  \nnot json\n{"b": 2}\n')
    messages = list(read_lines(src))
    assert messages == [{"a": 1}, {"b": 2}]


def test_read_lines_handles_empty_stream() -> None:
    assert list(read_lines(io.StringIO(""))) == []
