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

__all__ = [
    "books_to_context",
    "build_book_embedding_text",
    "build_system_prompt",
    "build_taste_vector_from_titles",
    "build_user_prompt",
    "embed_text",
    "fetch_read_shelf",
    "get_db",
    "get_user_read_ids",
    "get_user_taste_vector",
    "load_embedder",
    "load_env",
    "vector_search_books",
]
