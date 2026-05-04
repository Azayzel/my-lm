"""Open Library → MongoDB Books ingest crawler.

Crawls Open Library subject catalogs, enriches each work with full metadata
(subjects, description, cover), embeds with sentence-transformers/all-MiniLM-
L6-v2, and upserts into the MongoDB ``books`` collection.

Crawl state (offset per subject) is tracked in a ``ol_ingest_state``
collection so the service can be stopped and resumed safely.

Usage:
    # One-shot pass over all subjects
    python scripts/ol_ingest.py

    # Only specific subjects
    python scripts/ol_ingest.py --subjects "fantasy,mystery,thriller"

    # Preview without writing to DB
    python scripts/ol_ingest.py --dry-run

    # Clear progress state and restart from scratch
    python scripts/ol_ingest.py --reset

    # Loop continuously (for use as a background service)
    python scripts/ol_ingest.py --daemon --sleep-hours 12

    # Cap books per subject per run (useful for rate-limiting)
    python scripts/ol_ingest.py --per-subject 50
"""

from __future__ import annotations

import argparse
import datetime
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mylm.rag.db import (
    build_book_embedding_text,
    embed_text,
    get_db,
    load_embedder,
)
from mylm.rag.open_library import (
    BASE,
    GENRE_TO_OL_SUBJECT,
    _get,
    fetch_work_details,
)

# ─── Configuration ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ol_ingest")

# All subjects we'll crawl (values from GENRE_TO_OL_SUBJECT)
DEFAULT_SUBJECTS: list[str] = sorted(set(GENRE_TO_OL_SUBJECT.values()))

# OL subjects API page size
_PAGE_SIZE = 20

# Rate limit between OL API calls (seconds)
_RATE_S = 0.5

# MongoDB collections
_BOOKS_COLLECTION = "books"
_STATE_COLLECTION = "ol_ingest_state"

# ─── Subject → genre/theme/mood classification ───────────────────────────────

_GENRE_KEYWORDS: frozenset[str] = frozenset(
    {
        "fantasy", "science fiction", "mystery", "thriller", "horror",
        "romance", "historical", "biography", "memoir", "classics",
        "literary fiction", "contemporary", "paranormal", "crime",
        "adventure", "dystopian", "steampunk", "urban fantasy",
        "magical realism", "graphic novel", "young adult",
    }
)

_MOOD_KEYWORDS: frozenset[str] = frozenset(
    {
        "dark", "atmospheric", "cozy", "humorous", "funny",
        "bleak", "uplifting", "whimsical", "suspenseful", "tense",
        "intense", "slow burn", "heartwarming", "melancholic", "gritty",
    }
)


def _classify_subjects(
    subjects: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split OL free-text subjects into Genres, Themes, and Moods.

    OL subjects look like "Fantasy fiction", "Magic -- Fiction",
    "England -- History", "Dark", etc.
    """
    genres: list[str] = []
    moods: list[str] = []
    themes: list[str] = []

    for raw in subjects:
        # Drop geographic qualifiers like "-- Fiction", "-- History"
        clean = raw.split("--")[0].strip()
        lower = clean.lower()

        if any(kw in lower for kw in _MOOD_KEYWORDS):
            moods.append(clean)
        elif any(kw in lower for kw in _GENRE_KEYWORDS):
            genres.append(clean.title())
        else:
            themes.append(clean)

    return genres[:6], themes[:8], moods[:4]


# ─── OL book → MongoDB document ──────────────────────────────────────────────


def _authors_list(author_str: str) -> list[str]:
    """Split "First Last, First2 Last2" author string into a list."""
    if not author_str:
        return []
    return [a.strip() for a in author_str.split(",") if a.strip()]


def map_ol_to_doc(
    ol_book: dict[str, Any],
    work_details: dict[str, Any],
) -> dict[str, Any] | None:
    """Map an OL book record + work details to the MongoDB books schema.

    Returns None if the record is too thin to be useful.
    """
    title = (ol_book.get("title") or "").strip()
    if not title:
        return None

    ol_key = (ol_book.get("ol_key") or "").strip()
    author_str = ol_book.get("author") or ""
    authors = _authors_list(author_str)

    # Merge subjects from both sources (catalog brief + work details)
    raw_subjects: list[str] = list(ol_book.get("subjects") or [])
    raw_subjects += work_details.get("subjects") or []
    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in raw_subjects:
        sl = s.lower()
        if sl not in seen:
            seen.add(sl)
            deduped.append(s)

    genres, themes, moods = _classify_subjects(deduped)

    description = (work_details.get("description") or ol_book.get("description") or "").strip()

    # Require at least title + some subjects or description
    if not genres and not themes and not description:
        return None

    cover_url = work_details.get("cover_url") or (
        f"https://covers.openlibrary.org/b/id/{ol_book['cover_id']}-M.jpg"
        if ol_book.get("cover_id")
        else None
    )

    pub_year = (
        work_details.get("first_publish_year")
        or ol_book.get("first_publish_year")
    )
    # Normalise year to int if possible
    if pub_year:
        try:
            pub_year = int(str(pub_year)[:4])
        except (ValueError, TypeError):
            pub_year = None

    doc: dict[str, Any] = {
        "Title": title,
        "Authors": authors,
        "Genres": genres,
        "Themes": themes,
        "Moods": moods,
        "Description": description[:500] if description else "",
        "Source": "open_library",
        "OlKey": ol_key,
        "EditionCount": ol_book.get("edition_count"),
        "FirstPublishYear": pub_year,
        "CoverUrl": cover_url,
        "UpdatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return doc


# ─── DB state helpers ─────────────────────────────────────────────────────────


def get_state(db: Any, subject: str) -> dict[str, Any]:
    """Fetch persisted crawl state for a subject (or return defaults)."""
    state = db[_STATE_COLLECTION].find_one({"_id": subject}) or {}
    return {
        "subject": subject,
        "offset": state.get("offset", 0),
        "total": state.get("total", 0),
        "completed": state.get("completed", False),
        "inserted": state.get("inserted", 0),
        "skipped": state.get("skipped", 0),
    }


def save_state(db: Any, state: dict[str, Any]) -> None:
    db[_STATE_COLLECTION].update_one(
        {"_id": state["subject"]},
        {
            "$set": {
                **state,
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        },
        upsert=True,
    )


def reset_state(db: Any) -> None:
    db[_STATE_COLLECTION].drop()
    log.info("Crawl state cleared.")


# ─── Core crawl logic ─────────────────────────────────────────────────────────


def _fetch_subject_page(
    subject_slug: str, offset: int, limit: int = _PAGE_SIZE
) -> dict[str, Any] | None:
    """Fetch one page of works for an OL subject slug."""
    return _get(
        f"{BASE}/subjects/{subject_slug}.json",
        params={"limit": limit, "offset": offset, "details": False},
    )


def crawl_subject(
    db: Any,
    embedder: Any,
    subject: str,
    *,
    per_subject: int = 200,
    dry_run: bool = False,
) -> dict[str, int]:
    """Crawl one OL subject, embed books, upsert to MongoDB.

    Returns {"inserted": N, "updated": N, "skipped": N, "errors": N}.
    """
    state = get_state(db, subject) if db is not None else {
        "subject": subject, "offset": 0, "total": 0,
        "completed": False, "inserted": 0, "skipped": 0,
    }

    if state["completed"]:
        log.info("  [%s] already complete (%d books). Skipping.", subject, state["inserted"])
        return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
    processed_this_run = 0

    log.info("  [%s] starting at offset %d", subject, state["offset"])

    while processed_this_run < per_subject:
        page = _fetch_subject_page(subject, offset=state["offset"])
        if not page:
            log.warning("  [%s] no data at offset %d — stopping.", subject, state["offset"])
            break

        # Update total on first page
        if state["offset"] == 0 or state["total"] == 0:
            state["total"] = page.get("work_count") or 0
            log.info("  [%s] total works: %d", subject, state["total"])

        works: list[dict] = page.get("works") or []
        if not works:
            state["completed"] = True
            save_state(db, state)
            log.info("  [%s] exhausted — marking complete.", subject)
            break

        for work in works:
            if processed_this_run >= per_subject:
                break

            ol_key = work.get("key") or ""
            if not ol_key:
                stats["skipped"] += 1
                continue

            # Skip if already in DB
            if db is not None:
                existing = db[_BOOKS_COLLECTION].find_one(
                    {"OlKey": ol_key}, projection={"_id": 1}
                )
                if existing:
                    stats["skipped"] += 1
                    continue

            # Enrich with full work details
            work_details = fetch_work_details(ol_key)

            # Build a basic OL record from catalog data
            authors = [a.get("name") or "" for a in (work.get("authors") or [])]
            ol_book: dict[str, Any] = {
                "ol_key": ol_key,
                "title": work.get("title") or "",
                "author": ", ".join(a for a in authors[:2] if a),
                "subjects": (work.get("subject") or [])[:15],
                "edition_count": work.get("edition_count"),
                "first_publish_year": work.get("first_publish_year"),
                "cover_id": work.get("cover_id"),
            }

            doc = map_ol_to_doc(ol_book, work_details)
            if not doc:
                stats["skipped"] += 1
                continue

            # Compute embedding
            embed_str = build_book_embedding_text(doc)
            try:
                doc["embedding"] = embed_text(embedder, embed_str)
            except Exception as exc:
                log.warning("  embed failed for '%s': %s", doc["Title"], exc)
                stats["errors"] += 1
                continue

            if dry_run:
                log.info(
                    "  [DRY-RUN] would upsert: %s by %s (%s)",
                    doc["Title"],
                    ", ".join(doc["Authors"]) or "unknown",
                    ", ".join(doc["Genres"][:2]) or "unclassified",
                )
                stats["inserted"] += 1
            else:
                result = db[_BOOKS_COLLECTION].update_one(
                    {"OlKey": ol_key},
                    {"$set": doc},
                    upsert=True,
                )
                if result.upserted_id:
                    stats["inserted"] += 1
                else:
                    stats["updated"] += 1

            processed_this_run += 1

        state["offset"] += len(works)
        state["inserted"] = state.get("inserted", 0) + stats["inserted"]

        # Check if we've exhausted this subject
        if state["total"] and state["offset"] >= state["total"]:
            state["completed"] = True
            log.info("  [%s] complete — %d total works.", subject, state["total"])

        if not dry_run and db is not None:
            save_state(db, state)

        if state["completed"]:
            break

        # Brief pause to be kind to OL servers
        time.sleep(_RATE_S)

    log.info(
        "  [%s] run done: inserted=%d updated=%d skipped=%d errors=%d",
        subject, stats["inserted"], stats["updated"], stats["skipped"], stats["errors"],
    )
    return stats


def run_pass(
    db: Any,
    embedder: Any,
    subjects: list[str],
    *,
    per_subject: int = 200,
    dry_run: bool = False,
) -> None:
    """Run one full crawl pass over all subjects."""
    totals = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
    for i, subject in enumerate(subjects):
        log.info("Subject %d/%d: %s", i + 1, len(subjects), subject)
        stats = crawl_subject(db, embedder, subject, per_subject=per_subject, dry_run=dry_run)
        for k in totals:
            totals[k] += stats.get(k, 0)

    log.info(
        "Pass complete — inserted=%d updated=%d skipped=%d errors=%d",
        totals["inserted"], totals["updated"], totals["skipped"], totals["errors"],
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated OL subject slugs to crawl (default: all in GENRE_TO_OL_SUBJECT)",
    )
    parser.add_argument(
        "--per-subject",
        type=int,
        default=200,
        help="Max new books to ingest per subject per run (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be inserted without writing to MongoDB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all crawl state and start from scratch",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Loop continuously, sleeping between passes",
    )
    parser.add_argument(
        "--sleep-hours",
        type=float,
        default=24.0,
        help="Hours to sleep between daemon passes (default: 24)",
    )
    args = parser.parse_args()

    # ── Parse subject list ────────────────────────────────────────────────────
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = DEFAULT_SUBJECTS

    log.info("Open Library ingest — %d subjects, %d books/subject/run", len(subjects), args.per_subject)
    log.info("Subjects: %s", ", ".join(subjects[:10]) + ("..." if len(subjects) > 10 else ""))

    # ── Connect to MongoDB + load embedder ────────────────────────────────────
    if not args.dry_run:
        log.info("Connecting to MongoDB Atlas...")
        db = get_db()
        log.info("Loading sentence-transformer embedder...")
        embedder = load_embedder()

        if args.reset:
            reset_state(db)

        # Ensure OlKey index for fast duplicate checks
        db[_BOOKS_COLLECTION].create_index("OlKey", unique=False, sparse=True)
        log.info("Ready.\n")
    else:
        log.info("DRY-RUN mode — MongoDB connection skipped")
        db = None  # type: ignore[assignment]
        log.info("Loading embedder for validation...")
        embedder = load_embedder()

    # ── Run ───────────────────────────────────────────────────────────────────
    pass_num = 0
    while True:
        pass_num += 1
        log.info("=== Pass %d started at %s ===", pass_num, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        run_pass(db, embedder, subjects, per_subject=args.per_subject, dry_run=args.dry_run)

        if not args.daemon:
            break

        sleep_secs = args.sleep_hours * 3600
        wake_at = datetime.datetime.now() + datetime.timedelta(seconds=sleep_secs)
        log.info("Daemon sleeping %.1fh — next run at %s", args.sleep_hours, wake_at.strftime("%Y-%m-%d %H:%M"))
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
