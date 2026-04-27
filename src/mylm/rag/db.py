"""Core RAG helpers for the BookMind recommender.

Uses MongoDB Atlas Vector Search over the existing ``books`` collection, which
already has 384-dimensional embeddings on every document and a Vector Search
index called ``vs_books_embedding``.
"""

from __future__ import annotations

import contextlib
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _project_root() -> Path:
    # src/mylm/rag/db.py → src/mylm/rag → src/mylm → src → <repo>
    return Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Load .env from the project root. Safe to call multiple times."""
    load_dotenv(_project_root() / ".env")


def get_db() -> Any:
    """Return a pymongo Database handle pointing at the bookmind DB."""
    load_env()
    from pymongo import MongoClient

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set in .env")

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    db_name = os.environ.get("MONGODB_DB", "bookmind")
    return client[db_name]


@lru_cache(maxsize=1)
def load_embedder() -> Any:
    """Load the sentence-transformer used to create the existing embeddings."""
    load_env()
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get(
        "BOOKMIND_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return SentenceTransformer(model_name)


def embed_text(embedder: Any, text: str) -> list[float]:
    """Encode a free-text query to a unit-normalized 384-dim vector."""
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def build_book_embedding_text(book: dict[str, Any]) -> str:
    """Reproduce the exact text format used by BookMind.Api to embed books."""
    parts: list[str] = [str(book.get("Title") or "")]

    authors = book.get("Authors") or []
    if isinstance(authors, list) and authors:
        parts.append("by " + ", ".join(str(a) for a in authors))

    genres = book.get("Genres") or []
    if isinstance(genres, list) and genres:
        parts.append("genres: " + ", ".join(str(g) for g in genres))

    themes = book.get("Themes") or []
    if isinstance(themes, list) and themes:
        parts.append("themes: " + ", ".join(str(t) for t in themes))

    moods = book.get("Moods") or []
    if isinstance(moods, list) and moods:
        parts.append("mood: " + ", ".join(str(m) for m in moods))

    desc = book.get("Description") or ""
    if desc:
        parts.append(desc[:300] if len(desc) > 300 else desc)

    return ". ".join(parts)


def vector_search_books(
    db: Any,
    query_vector: list[float],
    *,
    limit: int = 20,
    num_candidates: int = 200,
    mongo_filter: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run an Atlas $vectorSearch over the books collection."""
    load_env()
    index_name = os.environ.get("BOOKMIND_VECTOR_INDEX", "vs_books_embedding")

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
                **({"filter": mongo_filter} if mongo_filter else {}),
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    default_projection = {
        "Title": 1,
        "Subtitle": 1,
        "Authors": 1,
        "Genres": 1,
        "Themes": 1,
        "Moods": 1,
        "Pacing": 1,
        "AgeGroup": 1,
        "Description": 1,
        "PublishedDate": 1,
        "AvgRating": 1,
        "RatingCount": 1,
        "CoverImageUrl": 1,
        "PageCount": 1,
        "Isbn13": 1,
        "score": 1,
    }
    if projection is None:
        projection = default_projection
    pipeline.append({"$project": projection})

    return list(db.books.aggregate(pipeline))


def get_user_taste_vector(db: Any, user_id: str) -> list[float] | None:
    """Return the user's stored 384-dim taste vector, or None if absent."""
    from bson import ObjectId

    query: dict[str, Any] = {"$or": [{"_id": user_id}, {"Email": user_id}]}
    with contextlib.suppress(Exception):
        query["$or"].append({"_id": ObjectId(user_id)})

    user = db.users.find_one(query, {"tasteVector": 1})
    if not user:
        return None
    tv = user.get("tasteVector")
    if not tv:
        return None
    return [float(x) for x in tv]


def get_user_read_ids(db: Any, user_id: str) -> set[Any]:
    """Return the set of BookIds the user has already read / DNFd / is reading."""
    from bson import ObjectId

    uid_variants: list[Any] = [user_id]
    with contextlib.suppress(Exception):
        uid_variants.append(ObjectId(user_id))

    cursor = db.userBooks.find(
        {"UserId": {"$in": uid_variants}},
        {"BookId": 1, "_id": 0},
    )
    ids: set[Any] = set()
    for doc in cursor:
        bid = doc.get("BookId")
        if bid is not None:
            ids.add(bid)
    return ids


def books_to_context(books: list[dict[str, Any]], max_chars_each: int = 500) -> str:
    """Format a list of books as a numbered context block for the LLM prompt."""
    lines: list[str] = []
    for i, b in enumerate(books, 1):
        authors = b.get("Authors") or []
        if isinstance(authors, list):
            author_str = ", ".join(a.strip() for a in authors if a)
        else:
            author_str = str(authors)

        title = b.get("Title") or "Untitled"
        subtitle = b.get("Subtitle")
        header = f"{title}"
        if subtitle:
            header += f": {subtitle}"
        header += f" — {author_str}" if author_str else ""

        meta_parts = []
        genres = b.get("Genres") or []
        if genres:
            meta_parts.append(f"Genres: {', '.join(genres[:5])}")
        themes = b.get("Themes") or []
        if themes:
            meta_parts.append(f"Themes: {', '.join(themes[:5])}")
        moods = b.get("Moods") or []
        if moods:
            meta_parts.append(f"Moods: {', '.join(moods[:5])}")
        pacing = b.get("Pacing")
        if pacing:
            meta_parts.append(f"Pacing: {pacing}")
        rating = b.get("AvgRating")
        if rating:
            meta_parts.append(f"Avg rating: {rating:.1f}")

        desc = (b.get("Description") or "").strip().replace("\n", " ")
        if len(desc) > max_chars_each:
            desc = desc[:max_chars_each].rstrip() + "…"

        block = f"{i}. {header}\n"
        if meta_parts:
            block += "   " + " | ".join(meta_parts) + "\n"
        if desc:
            block += f"   {desc}\n"
        lines.append(block)

    return "\n".join(lines)


def build_taste_vector_from_titles(
    db: Any,
    embedder: Any,
    books: list[dict[str, Any]],
    *,
    min_match_score: float = 0.6,
) -> tuple[list[float] | None, list[dict[str, Any]]]:
    """Resolve titles against the books collection and return a weighted-mean
    taste vector + the list of resolved books.
    """
    if not books:
        return None, []

    resolved: list[dict[str, Any]] = []
    acc: list[float] | None = None
    weight_sum = 0.0

    for b in books:
        title = (b.get("title") or "").strip()
        author = (b.get("author") or "").strip()
        if not title:
            continue

        query = title
        if author:
            query += f". by {author}"
        qv = embed_text(embedder, query)

        hits = vector_search_books(
            db,
            qv,
            limit=1,
            num_candidates=50,
            projection={
                "Title": 1,
                "Authors": 1,
                "embedding": 1,
                "score": 1,
            },
        )
        if not hits:
            continue

        hit = hits[0]
        score = float(hit.get("score") or 0)
        if score < min_match_score:
            continue

        emb = hit.get("embedding")
        if not emb:
            continue

        rating = int(b.get("rating") or 0)
        weight_table = {5: 1.0, 4: 0.6, 3: 0.2, 2: -0.3, 1: -0.7, 0: 0.3}
        weight = weight_table.get(rating, 0.3)

        if acc is None:
            acc = [0.0] * len(emb)
        for i, v in enumerate(emb):
            acc[i] += float(v) * weight
        weight_sum += abs(weight)

        resolved.append(
            {
                "input_title": title,
                "input_author": author,
                "rating": rating,
                "matched_title": hit.get("Title"),
                "matched_authors": hit.get("Authors") or [],
                "match_score": round(score, 3),
                "weight": round(weight, 2),
            }
        )

    if acc is None or weight_sum <= 0:
        return None, resolved

    norm = math.sqrt(sum(x * x for x in acc)) or 1.0
    taste_vec = [x / norm for x in acc]
    return taste_vec, resolved


def build_system_prompt() -> str:
    return (
        "You are BookMind, a knowledgeable and friendly book recommendation "
        "assistant. You will be given a user's request plus a shortlist of "
        "candidate books retrieved from a library. "
        "ONLY recommend books that appear in the provided candidate list — "
        "never invent titles. "
        "Pick the 3–5 best matches for the user's request, and for each one "
        "explain in 2–3 sentences why it fits — referencing specific genres, "
        "themes, moods, or plot elements from the candidate description. "
        "Use a numbered list with the book title in bold."
    )


def build_user_prompt(query: str, books: list[dict[str, Any]]) -> str:
    ctx = books_to_context(books)
    return (
        f"User request:\n{query}\n\n"
        f"Candidate books (only recommend from this list):\n\n{ctx}"
    )
