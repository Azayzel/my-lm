"""BookMind RAG: Atlas Vector Search + optional LLM-grounded explanation."""

from mylm.rag.db import (
    books_to_context,
    build_book_embedding_text,
    build_system_prompt,
    build_taste_vector_from_titles,
    build_user_prompt,
    embed_text,
    get_db,
    get_user_read_ids,
    get_user_taste_vector,
    load_embedder,
    load_env,
    vector_search_books,
)
from mylm.rag.goodreads import fetch_read_shelf
from mylm.rag.open_library import (
    fetch_genre_catalog,
    fetch_reading_log,
    fetch_subject_books,
    search_books_by_title_author,
    enrich_books_with_ol_metadata,
)

__all__ = [
    "books_to_context",
    "build_book_embedding_text",
    "build_system_prompt",
    "build_taste_vector_from_titles",
    "build_user_prompt",
    "embed_text",
    "enrich_books_with_ol_metadata",
    "fetch_genre_catalog",
    "fetch_read_shelf",
    "fetch_reading_log",
    "fetch_subject_books",
    "get_db",
    "get_user_read_ids",
    "get_user_taste_vector",
    "load_embedder",
    "load_env",
    "search_books_by_title_author",
    "vector_search_books",
]
