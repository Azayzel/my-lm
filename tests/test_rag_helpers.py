"""Tests for pure-function RAG helpers (no MongoDB / model needed)."""

from __future__ import annotations

from mylm.rag.db import (
    books_to_context,
    build_book_embedding_text,
    build_system_prompt,
    build_user_prompt,
)


def test_build_book_embedding_text_full_book() -> None:
    book = {
        "Title": "Project Hail Mary",
        "Authors": ["Andy Weir"],
        "Genres": ["Sci-Fi"],
        "Themes": ["First Contact"],
        "Moods": ["Hopeful"],
        "Description": "An astronaut wakes up alone.",
    }
    text = build_book_embedding_text(book)
    assert text.startswith("Project Hail Mary")
    assert "by Andy Weir" in text
    assert "genres: Sci-Fi" in text
    assert "themes: First Contact" in text
    assert "mood: Hopeful" in text
    assert "An astronaut wakes up alone." in text


def test_build_book_embedding_text_truncates_long_description() -> None:
    book = {"Title": "X", "Description": "x" * 500}
    text = build_book_embedding_text(book)
    assert len(text.split(". ", 1)[1]) == 300


def test_build_book_embedding_text_skips_empty_fields() -> None:
    text = build_book_embedding_text({"Title": "Solo"})
    assert text == "Solo"


def test_books_to_context_numbers_entries() -> None:
    books = [
        {"Title": "A", "Authors": ["Alpha"]},
        {"Title": "B", "Authors": ["Beta"]},
    ]
    ctx = books_to_context(books)
    assert "1. A" in ctx
    assert "2. B" in ctx


def test_build_user_prompt_includes_query_and_books() -> None:
    prompt = build_user_prompt("cozy mystery", [{"Title": "Test", "Authors": ["X"]}])
    assert "cozy mystery" in prompt
    assert "Test" in prompt


def test_system_prompt_mentions_constraints() -> None:
    sp = build_system_prompt()
    assert "ONLY recommend books that appear" in sp
