"""Open Library public API client for BookMind.

Open Library (openlibrary.org) provides free, no-auth access to:
  - User reading logs (already-read, want-to-read, currently-reading)
  - Rich book metadata (subjects, description, page count, cover, ISBNs)
  - Subject / genre catalogs for candidate generation

API docs: https://openlibrary.org/developers/api
"""

from __future__ import annotations

import time
from typing import Any

import requests

BASE = "https://openlibrary.org"
COVERS = "https://covers.openlibrary.org/b"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "BookMind/1.0 (+https://github.com/Azayzel/my-lm)"
)
_TIMEOUT = 15
_RATE_S = 0.5  # seconds between requests — be a good citizen


def _get(url: str, params: dict | None = None) -> dict | list | None:
    """GET JSON from Open Library, returning None on any failure."""
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
    finally:
        time.sleep(_RATE_S)


# ─── Reading logs ─────────────────────────────────────────────────────────────


def _parse_log_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a normalised book record from a reading log entry."""
    work = entry.get("work") or {}
    key = work.get("key") or ""  # e.g. "/works/OL45883W"
    title = work.get("title") or ""
    if not title:
        return None

    # Author names may be under entry or a nested list
    authors: list[str] = []
    for a in work.get("author_keys") or []:
        name = (a.get("author") or {}).get("key") or ""
        # We'll resolve names separately if needed — store the key for now
        if name:
            authors.append(name)
    author_names = work.get("author_names") or []

    return {
        "ol_key": key,
        "title": title,
        "author": ", ".join(author_names) if author_names else "",
        "cover_id": work.get("cover_id"),
        "shelf": entry.get("shelf") or "read",
        # rating not provided by OL reading logs
        "rating": 0,
        "source": "open_library",
    }


def fetch_reading_log(
    username: str,
    shelf: str = "already-read",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch a user's Open Library reading log.

    Args:
        username: Open Library username (not the display name — the URL slug).
        shelf: "already-read" | "want-to-read" | "currently-reading"
        limit: Max entries to return (OL pages at 10 per request).

    Returns:
        List of normalised book dicts with keys: ol_key, title, author,
        cover_id, shelf, rating, source.
    """
    books: list[dict[str, Any]] = []
    offset = 0
    page_size = 10

    while len(books) < limit:
        data = _get(
            f"{BASE}/people/{username}/books/{shelf}.json",
            params={"limit": page_size, "offset": offset},
        )
        if not data:
            break

        entries = data.get("reading_log_entries") or []
        if not entries:
            break

        for entry in entries:
            parsed = _parse_log_entry(entry)
            if parsed:
                books.append(parsed)

        total = data.get("numFound") or 0
        offset += page_size
        if offset >= total or offset >= limit:
            break

    return books[:limit]


# ─── Book metadata enrichment ─────────────────────────────────────────────────


def fetch_work_details(ol_key: str) -> dict[str, Any]:
    """Fetch rich metadata for a work key like '/works/OL45883W'.

    Returns a dict with: subjects, description, first_publish_year, covers.
    Empty dict on failure.
    """
    if not ol_key.startswith("/works/"):
        return {}
    data = _get(f"{BASE}{ol_key}.json")
    if not data:
        return {}

    desc = data.get("description") or ""
    if isinstance(desc, dict):
        desc = desc.get("value") or ""

    subjects = data.get("subjects") or []
    # OL subjects are free-text strings; normalise a bit
    subjects = [s.strip() for s in subjects if isinstance(s, str)][:20]

    covers = data.get("covers") or []
    cover_url = (
        f"{COVERS}/id/{covers[0]}-M.jpg" if covers else None
    )

    return {
        "subjects": subjects,
        "description": desc[:500] if desc else "",
        "first_publish_year": data.get("first_publish_date") or data.get("created", {}).get("value", "")[:4] if data.get("created") else "",
        "cover_url": cover_url,
    }


def enrich_books_with_ol_metadata(
    books: list[dict[str, Any]],
    *,
    max_enriched: int = 50,
) -> list[dict[str, Any]]:
    """For books that have an ol_key, fetch and merge work details.

    Modifies books in-place and returns the list. Limits API calls to
    ``max_enriched`` to stay fast.
    """
    enriched = 0
    for book in books:
        if enriched >= max_enriched:
            break
        key = book.get("ol_key") or ""
        if not key:
            continue
        details = fetch_work_details(key)
        if details:
            book.update(details)
            enriched += 1
    return books


def search_books_by_title_author(
    title: str,
    author: str = "",
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Search Open Library for a book by title/author; return work keys + metadata."""
    q = title
    if author:
        q += f" {author}"
    data = _get(f"{BASE}/search.json", params={"q": q, "limit": limit, "fields": "key,title,author_name,subject,first_publish_year,cover_i,number_of_pages_median"})
    if not data:
        return []

    results = []
    for doc in (data.get("docs") or []):
        results.append(
            {
                "ol_key": doc.get("key") or "",
                "title": doc.get("title") or "",
                "author": ", ".join((doc.get("author_name") or [])[:2]),
                "subjects": (doc.get("subject") or [])[:15],
                "first_publish_year": doc.get("first_publish_year"),
                "cover_id": doc.get("cover_i"),
                "page_count": doc.get("number_of_pages_median"),
                "source": "open_library",
            }
        )
    return results


# ─── Subject / genre catalogs ─────────────────────────────────────────────────

# Map genre labels (matching USERS profile) to OL subject slugs
GENRE_TO_OL_SUBJECT: dict[str, str] = {
    "Fantasy": "fantasy",
    "Paranormal": "paranormal",
    "Science Fiction": "science_fiction",
    "Speculative Fiction": "speculative_fiction",
    "Thriller": "thriller",
    "Mystery": "mystery",
    "Classics": "classic_literature",
    "Contemporary": "contemporary",
    "Fiction": "fiction",
    "Historical Fiction": "historical_fiction",
    "Horror": "horror",
    "Romance": "romance",
    "Literary Fiction": "literary_fiction",
    "Self-Help": "self-help",
    "Biography": "biography",
    "Memoir": "memoir",
    "Japan": "japan",
}


def fetch_subject_books(
    genre: str,
    limit: int = 20,
    min_edition_count: int = 5,
) -> list[dict[str, Any]]:
    """Fetch popular books for a genre/subject from Open Library.

    Args:
        genre: A genre label (matched against GENRE_TO_OL_SUBJECT) or a raw
               OL subject slug.
        limit: Max books to return.
        min_edition_count: Only include works with at least this many editions
                           (a rough proxy for popularity / quality signal).

    Returns:
        List of book dicts with keys: ol_key, title, author, subjects,
        edition_count, first_publish_year, source.
    """
    subject = GENRE_TO_OL_SUBJECT.get(genre, genre.lower().replace(" ", "_"))
    data = _get(
        f"{BASE}/subjects/{subject}.json",
        params={"limit": limit, "details": False},
    )
    if not data:
        return []

    books = []
    for work in data.get("works") or []:
        edition_count = work.get("edition_count") or 0
        if edition_count < min_edition_count:
            continue

        authors = [a.get("name") or "" for a in (work.get("authors") or [])]
        books.append(
            {
                "ol_key": work.get("key") or "",
                "title": work.get("title") or "",
                "author": ", ".join(a for a in authors[:2] if a),
                "subjects": (work.get("subject") or [])[:10],
                "edition_count": edition_count,
                "first_publish_year": work.get("first_publish_year"),
                "cover_id": work.get("cover_id"),
                "source": "open_library",
                # Normalise to lowercase for _fmt_book compatibility
                "rating": 0,
            }
        )

    return books


def fetch_genre_catalog(
    genres: list[str],
    per_genre: int = 20,
) -> list[dict[str, Any]]:
    """Fetch books across multiple genres; deduplicate by ol_key."""
    seen: set[str] = set()
    all_books: list[dict[str, Any]] = []
    for genre in genres:
        books = fetch_subject_books(genre, limit=per_genre)
        for b in books:
            key = b.get("ol_key") or b.get("title") or ""
            if key and key not in seen:
                seen.add(key)
                all_books.append(b)
    return all_books
