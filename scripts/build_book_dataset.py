"""Build a book-recommendation fine-tuning dataset.

Pulls both Goodreads profiles + Open Library reading logs, matches books
against the MongoDB Atlas ``bookmind`` library, then generates instruction-
tuning conversations that teach the LLM to reason about personal reading taste.

Outputs: datasets/books/train.jsonl  (and val.jsonl)

Usage:
    python scripts/build_book_dataset.py
    python scripts/build_book_dataset.py --no-mongo       # skip Atlas, scrape only
    python scripts/build_book_dataset.py --no-ol          # skip Open Library
    python scripts/build_book_dataset.py --out datasets/books/train.jsonl --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ─── User profiles ───────────────────────────────────────────────────────────

USERS: list[dict[str, Any]] = [
    {
        "name": "Josh",
        "goodreads_id": "64412103",
        "ol_username": "joshlavely",          # openlibrary.org/people/joshlavely
        "shelves": ["read", "to-read"],
        "favorite_genres": ["Fantasy", "Paranormal", "Science Fiction", "Thriller"],
        "avg_rating": 4.56,
        "notes": "Selective reader with high standards; prefers dark, speculative fiction.",
    },
    {
        "name": "Alli",
        "goodreads_id": "40285049",
        "ol_username": "allisandwich",        # openlibrary.org/people/allisandwich
        "shelves": ["read", "to-read"],
        "favorite_genres": [
            "Classics",
            "Contemporary",
            "Fantasy",
            "Fiction",
            "Historical Fiction",
            "Mystery",
        ],
        "avg_rating": 4.11,
        "notes": "Voracious reader across literary and genre fiction; appreciates strong prose and character work.",
    },
]

# ─── Conversation templates ───────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are BookMind, an expert personal book recommendation engine. "
    "You understand reading taste deeply — genre preferences, thematic resonance, "
    "prose style, pacing, and mood. Given a user's reading history with ratings, "
    "you identify patterns in what they love and use that to recommend books "
    "they haven't read yet, always explaining your reasoning clearly."
)


def _fmt_book(b: dict[str, Any]) -> str:
    """Format a book dict (Goodreads, Open Library, or Atlas) as a short string."""
    # Atlas format: {Title, Authors, Genres, ...}
    if "Title" in b:
        s = b.get("Title") or "Untitled"
        authors = b.get("Authors") or []
        if authors:
            s += " by " + ", ".join(authors[:2])
        genres = b.get("Genres") or []
        if genres:
            s += f" ({', '.join(genres[:3])})"
        return s
    # Goodreads / Open Library normalised format: {title, author, rating, ...}
    s = b.get("title") or "Untitled"
    if b.get("author"):
        s += f" by {b['author']}"
    if b.get("rating"):
        s += f" [{b['rating']}★]"
    return s


def _profile_summary(user: dict, books: list[dict]) -> str:
    """Describe a user's taste from their rated reading history."""
    rated = [b for b in books if b.get("rating", 0) >= 1]
    loved = [b for b in rated if b.get("rating", 0) >= 4]
    disliked = [b for b in rated if b.get("rating", 0) <= 2]

    genres_str = ", ".join(user["favorite_genres"])
    summary = (
        f"{user['name']} is a reader who gravitates toward {genres_str}. "
        f"They've rated {len(rated)} books with an average of {user['avg_rating']}★. "
    )
    if loved:
        top_loved = loved[:5]
        summary += (
            "Books they loved include: "
            + ", ".join(_fmt_book(b) for b in top_loved)
            + ". "
        )
    if disliked:
        summary += (
            "They were less enthusiastic about: "
            + ", ".join(_fmt_book(b) for b in disliked[:3])
            + ". "
        )
    summary += user["notes"]
    return summary


# ─── Training example generators ─────────────────────────────────────────────


def make_genre_queries(
    user: dict,
    read_books: list[dict],
    atlas_candidates: list[dict] | None,
) -> list[dict[str, Any]]:
    """'Recommend me a [genre] book' — genre-specific queries anchored to user taste."""
    examples = []
    rng = random.Random(17)
    rated = [b for b in read_books if b.get("rating", 0) >= 4]
    if not rated:
        return []

    for genre in user["favorite_genres"]:
        # pick a couple loved books as anchor context
        anchor = rng.sample(rated, min(3, len(rated)))
        anchor_str = ", ".join(b.get("title", "") for b in anchor)

        if atlas_candidates:
            picks = rng.sample(atlas_candidates, min(4, len(atlas_candidates)))
            rec_lines = "\n".join(f"  - {_fmt_book(p)}" for p in picks)
            answer = (
                f"Given that you love {anchor_str}, here are {genre} picks that should land well:\n"
                f"{rec_lines}\n\n"
                f"Each of these leans into what makes {genre} resonate for readers with your profile."
            )
        else:
            answer = (
                f"Based on your love of {anchor_str}, I'd look for {genre} titles that share "
                f"strong character work and the kind of immersive world-building you've rated highly before."
            )

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"I really love {anchor_str}. "
                            f"Can you recommend a great {genre} book I haven't read yet?"
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def make_similar_to_book(
    user: dict,
    read_books: list[dict],
    atlas_candidates: list[dict] | None,
) -> list[dict[str, Any]]:
    """'Find me something like [specific book I loved]'"""
    examples = []
    rng = random.Random(88)
    loved = [b for b in read_books if b.get("rating", 0) >= 4]
    if not loved:
        return []

    targets = rng.sample(loved, min(10, len(loved)))
    for book in targets:
        if atlas_candidates:
            picks = rng.sample(atlas_candidates, min(4, len(atlas_candidates)))
            rec_lines = "\n".join(f"  - {_fmt_book(p)}" for p in picks)
            answer = (
                f"Great choice — '{book.get('title')}' is a strong signal about your taste. "
                f"Books with similar DNA:\n{rec_lines}"
            )
        else:
            answer = (
                f"'{book.get('title')}' tells me a lot about what you're looking for. "
                f"I'd target books in the same genre with similar pacing and emotional stakes."
            )

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"I gave '{_fmt_book(book)}' five stars. Find me something just like it.",
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def make_rank_candidates(
    user: dict,
    read_books: list[dict],
    atlas_candidates: list[dict] | None,
) -> list[dict[str, Any]]:
    """'Here are 5 books — rank them for me based on my taste.'"""
    if not atlas_candidates or len(atlas_candidates) < 5:
        return []

    examples = []
    rng = random.Random(55)
    rated = [b for b in read_books if b.get("rating", 0) >= 4]
    if not rated:
        return []

    for _ in range(min(6, len(atlas_candidates) // 5)):
        pool = rng.sample(atlas_candidates, 5)
        history_sample = rng.sample(rated, min(5, len(rated)))

        history_str = "\n".join(f"  - {_fmt_book(b)}" for b in history_sample)
        candidates_str = "\n".join(f"  {i+1}. {_fmt_book(p)}" for i, p in enumerate(pool))

        # Rank: simple heuristic — genres matching user's favorites score higher
        def score_book(b: dict) -> int:
            genres = b.get("Genres") or []
            return sum(1 for g in genres if any(ug.lower() in g.lower() for ug in user["favorite_genres"]))

        ranked = sorted(pool, key=score_book, reverse=True)
        ranked_str = "\n".join(f"  {i+1}. {_fmt_book(b)}" for i, b in enumerate(ranked))

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"My reading history:\n{history_str}\n\n"
                            f"Rank these 5 books from most to least suited to my taste:\n{candidates_str}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Ranked for your taste profile:\n{ranked_str}\n\n"
                            f"The top picks align most closely with your preference for "
                            f"{', '.join(user['favorite_genres'][:2])}."
                        ),
                    },
                ]
            }
        )
    return examples


def make_recommend_from_history(
    user: dict,
    read_books: list[dict],
    to_read_books: list[dict],
    atlas_candidates: list[dict] | None,
) -> list[dict[str, Any]]:
    """'Given this reading history, what should I read next?'"""
    examples = []
    rng = random.Random(42)

    rated = [b for b in read_books if b.get("rating", 0) >= 1]
    if not rated:
        return []

    # Generate many varied history windows
    iterations = max(15, min(30, len(rated) // 2))
    for _ in range(iterations):
        sample_size = rng.randint(5, min(15, len(rated)))
        history_sample = rng.sample(rated, sample_size)
        history_str = "\n".join(f"  - {_fmt_book(b)}" for b in history_sample)

        # If we have Atlas candidates, pick a few to suggest from
        if atlas_candidates:
            picks = rng.sample(atlas_candidates, min(5, len(atlas_candidates)))
            rec_str = "\n".join(f"  - {_fmt_book(p)}" for p in picks)
            assistant_text = (
                f"Based on your reading history, I can see you're drawn to "
                f"{', '.join(user['favorite_genres'][:3])}. "
                f"Here are books I think you'd love:\n{rec_str}\n\n"
                "These match your taste because they share the themes, pacing, and "
                "emotional resonance of the books you've rated most highly."
            )
        elif to_read_books:
            want = rng.sample(to_read_books, min(5, len(to_read_books)))
            rec_str = "\n".join(f"  - {_fmt_book(w)}" for w in want)
            assistant_text = (
                f"Given what you've loved — especially the {', '.join(user['favorite_genres'][:2])} titles — "
                f"your instinct to add these to your to-read list looks excellent:\n{rec_str}"
            )
        else:
            continue

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here are some books I've read recently:\n{history_str}\n\n"
                            "What should I read next?"
                        ),
                    },
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    return examples


def make_taste_profile_conversation(
    user: dict,
    read_books: list[dict],
) -> list[dict[str, Any]]:
    """'What can you tell about my reading taste?'"""
    rated = [b for b in read_books if b.get("rating", 0) >= 1]
    if len(rated) < 5:
        return []

    loved = sorted(rated, key=lambda b: b.get("rating", 0), reverse=True)[:10]
    history_str = "\n".join(f"  - {_fmt_book(b)}" for b in loved)
    profile = _profile_summary(user, rated)

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Here are my highest-rated books:\n{history_str}\n\n"
                        "What does my reading taste tell you about me, and what kinds of books should I be looking for?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": profile,
                },
            ]
        }
    ]


def make_rating_explanation(
    user: dict,
    read_books: list[dict],
) -> list[dict[str, Any]]:
    """'Why did I [love/dislike] this book?' based on actual ratings."""
    examples = []
    rated = [b for b in read_books if b.get("rating", 0) >= 1]
    loved = [b for b in rated if b.get("rating", 0) >= 4]
    disliked = [b for b in rated if b.get("rating", 0) <= 2]

    for book in loved[:5]:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"I gave '{_fmt_book(book)}' {book.get('rating')} stars. "
                            f"My favorite genres are {', '.join(user['favorite_genres'][:3])}. "
                            "Why do you think I loved this book?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Your {book.get('rating')}★ rating for {book.get('title', 'this book')} "
                            f"makes sense given your love of {', '.join(user['favorite_genres'][:2])}. "
                            f"It likely resonated with your taste for "
                            f"{'dark, high-stakes narratives' if 'Thriller' in user['favorite_genres'] or 'Science Fiction' in user['favorite_genres'] else 'rich world-building and complex characters'}. "
                            "Books that connect emotionally with your existing favorites tend to land as five-star reads for you."
                        ),
                    },
                ]
            }
        )

    for book in disliked[:3]:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"I only gave '{_fmt_book(book)}' {book.get('rating')} stars. "
                            f"Why didn't it work for me?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"A {book.get('rating')}★ rating suggests this one didn't click. "
                            f"Given your profile — strong preference for {', '.join(user['favorite_genres'][:2])} "
                            f"and an average rating of {user['avg_rating']}★ — "
                            "it likely either felt too slow, strayed outside your thematic comfort zone, "
                            "or didn't deliver the emotional payoff you expect from highly-rated reads. "
                            "This is useful signal: I'll steer away from similar titles in future recommendations."
                        ),
                    },
                ]
            }
        )

    return examples


def make_cross_user_comparison(
    users: list[dict],
    all_books: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    """Generate examples comparing two users' taste overlap."""
    if len(users) < 2:
        return []
    u1, u2 = users[0], users[1]
    b1 = [b for b in all_books.get(u1["name"], []) if b.get("rating", 0) >= 4]
    b2 = [b for b in all_books.get(u2["name"], []) if b.get("rating", 0) >= 4]
    if not b1 or not b2:
        return []

    shared_genres = set(u1["favorite_genres"]) & set(u2["favorite_genres"])
    different_genres = (
        set(u1["favorite_genres"]) ^ set(u2["favorite_genres"])
    )

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{u1['name']} loves: {', '.join(g for g in u1['favorite_genres'])}.\n"
                        f"{u2['name']} loves: {', '.join(g for g in u2['favorite_genres'])}.\n"
                        f"What books would both of them enjoy?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"{u1['name']} and {u2['name']} share a love of "
                        f"{', '.join(shared_genres) if shared_genres else 'character-driven storytelling'}. "
                        f"The overlap in their taste points toward books that blend "
                        f"{'speculative imagination with literary depth' if 'Fantasy' in shared_genres else 'strong narrative and emotional resonance'}. "
                        f"Where they diverge — {u1['name']} leans toward {', '.join(list(set(u1['favorite_genres']) - set(u2['favorite_genres']))[:2] or ['darker fiction'])} "
                        f"while {u2['name']} gravitates toward {', '.join(list(set(u2['favorite_genres']) - set(u1['favorite_genres']))[:2] or ['literary prose'])} — "
                        "a book at the intersection would need to balance both sensibilities."
                    ),
                },
            ]
        }
    ]


def make_want_to_read_validation(
    user: dict,
    read_books: list[dict],
    to_read_books: list[dict],
) -> list[dict[str, Any]]:
    """'Should I actually read this book I saved?' — validates to-read shelf items."""
    if not to_read_books:
        return []

    examples = []
    rng = random.Random(99)
    sample = rng.sample(to_read_books, min(5, len(to_read_books)))
    rated = [b for b in read_books if b.get("rating", 0) >= 4]
    loved_str = ", ".join(_fmt_book(b) for b in rated[:3]) if rated else "your favorites"

    for book in sample:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"I have '{_fmt_book(book)}' on my to-read list. "
                            f"Given that I love {loved_str}, should I actually read it soon?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Yes — '{book.get('title', 'that book')}' is a strong match for your taste. "
                            f"Given your love of {', '.join(user['favorite_genres'][:2])}, "
                            "it should deliver the kind of reading experience you consistently rate highly. "
                            "I'd bump it toward the top of your queue."
                        ),
                    },
                ]
            }
        )
    return examples


# ─── Review-grounded generators ───────────────────────────────────────────────


def _extract_review_highlight(review: str) -> str:
    """Return the first meaningful sentence of a review (≤150 chars)."""
    for sent in review.split("."):
        s = sent.strip()
        if len(s) > 20:
            return s[:150]
    return review[:150]


def _infer_reading_values(rating: int, review: str, user: dict) -> str:
    """Infer taste values from rating + review text."""
    r = review.lower()
    values: list[str] = []
    if any(w in r for w in ["character", "protagonist", "narrator", "cast"]):
        values.append("strong characterization")
    if any(w in r for w in ["world", "setting", "atmosphere", "immersive", "world-building"]):
        values.append("immersive world-building")
    if any(w in r for w in ["prose", "writing", "style", "language", "beautifully written"]):
        values.append("high-quality prose")
    if any(w in r for w in ["plot", "pacing", "page-turner", "couldn't put", "couldn't stop"]):
        values.append("propulsive plotting")
    if any(w in r for w in ["emotional", "heart", "cry", "moved", "touching", "devastating"]):
        values.append("emotional resonance")
    if not values:
        values = [f"the qualities that make {user['favorite_genres'][0]} titles work for you"]
    return " and ".join(values[:3])


def make_review_text_training(
    user: dict,
    read_books: list[dict],
) -> list[dict[str, Any]]:
    """High-quality training examples grounded in the user's actual review text.

    Goodreads RSS includes ``user_review`` — real words the user wrote about
    why they loved or hated a book. This is the richest taste signal we have.
    """
    examples: list[dict[str, Any]] = []
    reviewed = [b for b in read_books if (b.get("review") or "").strip() and len(b.get("review", "")) > 40]
    if not reviewed:
        return []

    for book in reviewed:
        rating = book.get("rating", 0)
        review = book["review"][:400]
        title = book.get("title") or "this book"
        stars = f"{rating}★" if rating else "unrated"
        genres_str = ", ".join(user["favorite_genres"][:2])
        highlight = _extract_review_highlight(review)
        values = _infer_reading_values(rating, review, user)

        # Example 1: "What does this review tell you about my taste?"
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"I rated '{title}' {stars} and wrote: \"{review}\"\n\n"
                            f"My favourite genres are {genres_str}. "
                            "What does this review tell you about what I look for in books?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Your review of '{title}' ({stars}) gives me clear taste signals. "
                            f"You highlight: \"{highlight}\" — which tells me you prize {values}. "
                            f"Combined with your {genres_str} preferences and a {user['avg_rating']}★ average, "
                            f"I'll weight future recommendations heavily toward books that deliver on those fronts. "
                            f"{user['notes']}"
                        ),
                    },
                ]
            }
        )

        # Example 2: loved books → find something similar grounded in review
        if rating >= 4:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"I loved '{title}' ({stars}) and wrote: \"{review[:200]}\"\n\n"
                                "Find me something with the same feel."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"Based on what resonated with you in '{title}' — {values} — "
                                f"I'd look for {genres_str} titles that share those qualities. "
                                f"Your note about \"{highlight}\" is a particularly strong signal: "
                                "books that deliver on that same dimension are where I'd start."
                            ),
                        },
                    ]
                }
            )

        # Example 3: low-rated → what to avoid
        if rating <= 2:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"I only gave '{title}' {stars} and wrote: \"{review[:200]}\"\n\n"
                                "What should you avoid recommending to me?"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"Noted — '{title}' ({stars}) is useful negative signal. "
                                f"From your review: \"{highlight}\". "
                                "I'll steer away from books that share those characteristics. "
                                f"Given your love of {genres_str}, the issue seems to be "
                                f"{'pacing and execution' if 'slow' in (review.lower()) else 'a mismatch with your thematic preferences'} "
                                "rather than genre itself — so I'll stay within your preferred territory while "
                                "filtering for better alignment on the qualities that matter to you."
                            ),
                        },
                    ]
                }
            )

    return examples


# ─── OL-specific generators ────────────────────────────────────────────────────

# Module-level rng for OL generators (seeded deterministically)
rng_global = random.Random(123)


def make_ol_subject_enriched_recs(
    user: dict,
    read_books: list[dict],
    ol_catalog: list[dict],
) -> list[dict[str, Any]]:
    """Recommendations backed by Open Library subject metadata.

    These are richer than plain title/author because OL books carry subjects
    and descriptions — teaches the model to reason about thematic overlap.
    """
    if not ol_catalog:
        return []

    examples = []
    rng = random.Random(33)
    rated = [b for b in read_books if b.get("rating", 0) >= 4]
    if not rated:
        return []

    # Group OL catalog by genre for varied examples
    genre_buckets: dict[str, list[dict]] = {}
    for b in ol_catalog:
        subs = b.get("subjects") or []
        for genre in user["favorite_genres"]:
            if any(genre.lower() in s.lower() for s in subs):
                genre_buckets.setdefault(genre, []).append(b)
                break
        else:
            genre_buckets.setdefault("other", []).append(b)

    for genre, bucket in genre_buckets.items():
        if not bucket or genre == "other":
            continue
        picks = rng.sample(bucket, min(4, len(bucket)))
        history_sample = rng.sample(rated, min(5, len(rated)))
        history_str = "\n".join(f"  - {_fmt_book(b)}" for b in history_sample)

        rec_lines = []
        for p in picks:
            line = f"  - {_fmt_book(p)}"
            subs = (p.get("subjects") or [])[:4]
            if subs:
                line += f" — subjects: {', '.join(subs)}"
            rec_lines.append(line)

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"My reading history:\n{history_str}\n\n"
                            f"I'm in the mood for {genre}. What do you recommend?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Based on your history and your {genre} interests, here are some strong picks:\n"
                            + "\n".join(rec_lines)
                            + f"\n\nThese share thematic DNA with the {genre} books you've loved most."
                        ),
                    },
                ]
            }
        )
    return examples


def make_ol_enriched_taste_profile(
    user: dict,
    ol_read_books: list[dict],
) -> list[dict[str, Any]]:
    """Taste profile using Open Library subject metadata from a user's reading log."""
    if len(ol_read_books) < 5:
        return []

    subject_freq: dict[str, int] = {}
    for book in ol_read_books:
        for sub in (book.get("subjects") or []):
            subject_freq[sub] = subject_freq.get(sub, 0) + 1

    top_subjects = sorted(subject_freq, key=lambda s: subject_freq[s], reverse=True)[:10]
    sample_titles = rng_global.sample(ol_read_books, min(8, len(ol_read_books)))
    titles_str = "\n".join(f"  - {_fmt_book(b)}" for b in sample_titles)
    subjects_str = ", ".join(top_subjects[:6]) if top_subjects else "varied fiction"

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Here are some books I've tracked on Open Library:\n{titles_str}\n\n"
                        "What patterns do you see in my reading taste?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Looking at your reading log, the strongest recurring subjects are: {subjects_str}. "
                        f"This tells me you're drawn to {user['favorite_genres'][0]} with undertones of "
                        f"{'atmosphere and tension' if 'Thriller' in user['favorite_genres'] or 'Mystery' in user['favorite_genres'] else 'world-building and imaginative depth'}. "
                        f"{user['notes']}"
                    ),
                },
            ]
        }
    ]


# ─── Main pipeline ────────────────────────────────────────────────────────────


def fetch_user_books(
    user: dict,
) -> dict[str, list[dict]]:
    """Fetch read + to-read shelves from Goodreads."""
    from mylm.rag.goodreads import fetch_read_shelf

    result: dict[str, list[dict]] = {}
    for shelf in user["shelves"]:
        print(f"  [{user['name']}] fetching '{shelf}' shelf (id={user['goodreads_id']})...")
        books = fetch_read_shelf(user["goodreads_id"], shelf=shelf, max_books=400)
        print(f"  [{user['name']}] {shelf}: {len(books)} books")
        result[shelf] = books
    return result


def fetch_ol_user_books(user: dict) -> dict[str, list[dict]]:
    """Fetch reading logs from Open Library, with subject enrichment."""
    from mylm.rag.open_library import fetch_reading_log, enrich_books_with_ol_metadata

    ol_username = user.get("ol_username") or ""
    if not ol_username:
        return {}

    result: dict[str, list[dict]] = {}
    shelf_map = {
        "already-read": "read",
        "want-to-read": "to-read",
    }
    for ol_shelf, norm_shelf in shelf_map.items():
        print(f"  [{user['name']}] OL fetching '{ol_shelf}' (username={ol_username})...")
        books = fetch_reading_log(ol_username, shelf=ol_shelf, limit=200)
        # Enrich top 30 books with subject/description metadata
        enrich_books_with_ol_metadata(books, max_enriched=30)
        print(f"  [{user['name']}] OL {ol_shelf}: {len(books)} books")
        result[norm_shelf] = books
    return result


def fetch_atlas_candidates(db: Any, taste_vec: list[float], n: int = 30) -> list[dict]:
    """Get candidate books from Atlas vector search given a taste vector."""
    from mylm.rag.db import vector_search_books

    return vector_search_books(db, taste_vec, limit=n, num_candidates=n * 5)


def build_taste_vector(
    db: Any, embedder: Any, books: list[dict]
) -> list[float] | None:
    from mylm.rag.db import build_taste_vector_from_titles

    vec, resolved = build_taste_vector_from_titles(db, embedder, books)
    if resolved:
        print(f"    matched {len(resolved)} books to Atlas library")
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/books/train.jsonl"),
        help="Output training JSONL",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of examples held out for validation",
    )
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="Skip MongoDB Atlas (no taste vector or candidate lookup)",
    )
    parser.add_argument(
        "--no-ol",
        action="store_true",
        help="Skip Open Library (Goodreads-only dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Setup MongoDB (optional) ──────────────────────────────────────────────
    db = None
    embedder = None
    if not args.no_mongo:
        try:
            from mylm.rag.db import get_db, load_embedder

            print("Connecting to MongoDB Atlas...")
            db = get_db()
            print("Loading sentence-transformer embedder...")
            embedder = load_embedder()
            print("Ready.\n")
        except Exception as exc:
            print(f"[warn] MongoDB unavailable: {exc}. Running without Atlas candidates.")
            db = None
            embedder = None

    # ── Fetch Goodreads data ──────────────────────────────────────────────────
    all_shelf_data: dict[str, dict[str, list[dict]]] = {}
    for user in USERS:
        print(f"\nFetching Goodreads data for {user['name']}...")
        shelf_data = fetch_user_books(user)
        all_shelf_data[user["name"]] = shelf_data

    # ── Fetch Open Library data ───────────────────────────────────────────────
    ol_shelf_data: dict[str, dict[str, list[dict]]] = {}
    ol_genre_catalog: dict[str, list[dict]] = {}   # per-user OL subject catalog

    if not args.no_ol:
        from mylm.rag.open_library import fetch_genre_catalog

        for user in USERS:
            print(f"\nFetching Open Library data for {user['name']}...")
            ol_data = fetch_ol_user_books(user)
            ol_shelf_data[user["name"]] = ol_data

            print(f"  [{user['name']}] Building OL genre catalog for {user['favorite_genres']}...")
            catalog = fetch_genre_catalog(user["favorite_genres"], per_genre=15)
            print(f"  [{user['name']}] OL catalog: {len(catalog)} books across genres")
            ol_genre_catalog[user["name"]] = catalog

    # ── Build taste vectors + fetch Atlas candidates ───────────────────────────
    atlas_candidates_by_user: dict[str, list[dict]] = {}
    if db and embedder:
        for user in USERS:
            read_books = all_shelf_data[user["name"]].get("read", [])
            if read_books:
                print(f"\nBuilding taste vector for {user['name']} ({len(read_books)} books)...")
                taste_vec = build_taste_vector(db, embedder, read_books)
                if taste_vec:
                    candidates = fetch_atlas_candidates(db, taste_vec, n=40)
                    print(f"  → {len(candidates)} Atlas candidate books fetched")
                    atlas_candidates_by_user[user["name"]] = candidates

    # ── Generate training examples ────────────────────────────────────────────
    all_examples: list[dict] = []

    all_read_books: dict[str, list[dict]] = {}
    for user in USERS:
        read_books = all_shelf_data[user["name"]].get("read", [])
        to_read_books = all_shelf_data[user["name"]].get("to-read", [])
        atlas_cands = atlas_candidates_by_user.get(user["name"])
        all_read_books[user["name"]] = read_books

        print(f"\nGenerating examples for {user['name']}...")

        # 1. Recommend from history
        exs = make_recommend_from_history(user, read_books, to_read_books, atlas_cands)
        print(f"  recommend_from_history: {len(exs)}")
        all_examples.extend(exs)

        # 2. Taste profile analysis
        exs = make_taste_profile_conversation(user, read_books)
        print(f"  taste_profile: {len(exs)}")
        all_examples.extend(exs)

        # 3. Rating explanations
        exs = make_rating_explanation(user, read_books)
        print(f"  rating_explanation: {len(exs)}")
        all_examples.extend(exs)

        # 4. To-read validation
        exs = make_want_to_read_validation(user, read_books, to_read_books)
        print(f"  want_to_read_validation: {len(exs)}")
        all_examples.extend(exs)

        # 5. Genre-specific queries
        exs = make_genre_queries(user, read_books, atlas_cands)
        print(f"  genre_queries: {len(exs)}")
        all_examples.extend(exs)

        # 6. "Find me something like X"
        exs = make_similar_to_book(user, read_books, atlas_cands)
        print(f"  similar_to_book: {len(exs)}")
        all_examples.extend(exs)

        # 7. Rank candidate books
        exs = make_rank_candidates(user, read_books, atlas_cands)
        print(f"  rank_candidates: {len(exs)}")
        all_examples.extend(exs)

        # 8. Review-grounded training (uses actual review text from RSS)
        exs = make_review_text_training(user, read_books)
        print(f"  review_text_training: {len(exs)}")
        all_examples.extend(exs)

        # 9. Open Library subject-enriched recommendations
        ol_catalog = ol_genre_catalog.get(user["name"], [])
        exs = make_ol_subject_enriched_recs(user, read_books, ol_catalog)
        print(f"  ol_subject_enriched_recs: {len(exs)}")
        all_examples.extend(exs)

        # 10. Open Library taste profile (subject-aware)
        ol_read = (ol_shelf_data.get(user["name"]) or {}).get("read", [])
        exs = make_ol_enriched_taste_profile(user, ol_read)
        print(f"  ol_enriched_taste_profile: {len(exs)}")
        all_examples.extend(exs)

    # 11. Cross-user comparison
    exs = make_cross_user_comparison(USERS, all_read_books)
    print(f"\ncross_user_comparison: {len(exs)}")
    all_examples.extend(exs)

    # ── Shuffle and split ─────────────────────────────────────────────────────
    rng.shuffle(all_examples)
    val_n = max(1, int(len(all_examples) * args.val_split))
    val_examples = all_examples[:val_n]
    train_examples = all_examples[val_n:]

    # ── Write output ──────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    val_path = args.out.parent / ("val.jsonl")

    def _write(path: Path, examples: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} examples → {path}")

    _write(args.out, train_examples)
    _write(val_path, val_examples)

    print(f"\nTotal: {len(all_examples)} examples ({len(train_examples)} train / {len(val_examples)} val)")
    print("\nNext step:")
    print(f"  python scripts/train_qlora.py --data {args.out} --output-dir models/book-rec-lora")


if __name__ == "__main__":
    main()
