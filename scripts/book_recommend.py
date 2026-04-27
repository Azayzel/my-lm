"""Standalone BookMind recommender (CLI).

Usage:
    python scripts/book_recommend.py "a cozy fantasy with witches and tea"
    python scripts/book_recommend.py --user <user_id>
    python scripts/book_recommend.py "sci-fi first contact" --limit 15 --llm
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so arrows/emojis don't crash
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running without `pip install -e .` by adding src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mylm.rag import (  # noqa: E402
    build_system_prompt,
    build_taste_vector_from_titles,
    build_user_prompt,
    embed_text,
    fetch_read_shelf,
    get_db,
    get_user_read_ids,
    get_user_taste_vector,
    load_embedder,
    vector_search_books,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BookMind CLI recommender")
    parser.add_argument("query", nargs="?", help="Free-text query")
    parser.add_argument("--user", help="BookMind user _id or email (uses stored tasteVector)")
    parser.add_argument("--goodreads", help="Goodreads username or numeric id — builds a taste vector from their public read shelf")
    parser.add_argument("--limit", type=int, default=10, help="Top-K results")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Generate a natural-language recommendation using a local LLM",
    )
    parser.add_argument(
        "--llm-model",
        default="models/qwen2.5-3b",
        help="Local LLM directory for --llm",
    )
    args = parser.parse_args()

    if not args.query and not args.user and not args.goodreads:
        parser.error("provide a query, --user, or --goodreads")

    db = get_db()
    embedder = load_embedder()

    taste_vec = None
    gr_excluded: set[str] = set()

    if args.user:
        taste_vec = get_user_taste_vector(db, args.user)
        if taste_vec:
            print(f"[user] {args.user}: loaded stored taste vector ({len(taste_vec)}d)")
        else:
            print(f"[user] {args.user}: no taste vector found, falling back to query only")

    if args.goodreads and taste_vec is None:
        print(f"[goodreads] fetching public read shelf for '{args.goodreads}'...")
        gr_books = fetch_read_shelf(args.goodreads, max_books=100)
        print(f"[goodreads] fetched {len(gr_books)} books")
        if gr_books:
            taste_vec, resolved = build_taste_vector_from_titles(db, embedder, gr_books)
            print(f"[goodreads] matched {len(resolved)} against the library")
            for r in resolved[:5]:
                print(f"    [{r['rating']}*] {r['input_title']} -> {r['matched_title']} (score {r['match_score']})")
            if len(resolved) > 5:
                print(f"    ... and {len(resolved)-5} more")
            gr_excluded = {(r.get("matched_title") or "").strip().lower() for r in resolved}

    if args.query:
        q_vec = embed_text(embedder, args.query)
        if taste_vec:
            import math

            blended = [0.7 * a + 0.3 * b for a, b in zip(q_vec, taste_vec, strict=False)]
            n = math.sqrt(sum(x * x for x in blended)) or 1.0
            query_vec = [x / n for x in blended]
        else:
            query_vec = q_vec
    else:
        query_vec = taste_vec

    # Over-fetch for post-filtering (Atlas $vectorSearch only supports
    # filters on indexed filter fields, which we don't have).
    overfetch = args.limit * 3
    raw_books = vector_search_books(
        db,
        query_vec,
        limit=overfetch,
        num_candidates=max(overfetch * 10, 300),
    )

    excluded_ids: set = set()
    if args.user:
        excluded_ids = get_user_read_ids(db, args.user)
        if excluded_ids:
            print(f"[user] excluding {len(excluded_ids)} already-read books")

    books: list = []
    seen: set = set()
    for b in raw_books:
        if excluded_ids and b.get("_id") in excluded_ids:
            continue
        title_lower = (b.get("Title") or "").strip().lower()
        if gr_excluded and title_lower in gr_excluded:
            continue
        authors = b.get("Authors") or []
        first_author = (authors[0] if authors else "").strip().lower()
        key = (title_lower, first_author)
        if key in seen:
            continue
        seen.add(key)
        books.append(b)
        if len(books) >= args.limit:
            break

    print(f"\n=== Top {len(books)} vector-search candidates ===")
    for i, b in enumerate(books, 1):
        authors = b.get("Authors") or []
        author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        genres = (b.get("Genres") or [])[:4]
        print(f"{i:2d}. [{b.get('score', 0):.3f}] {b.get('Title')} — {author_str}")
        if genres:
            print(f"       genres: {', '.join(genres)}")

    if args.llm:
        print("\n=== LLM recommendation ===")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()

            messages = [
                {"role": "system", "content": build_system_prompt()},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        args.query or "Recommend books based on my taste.",
                        books,
                    ),
                },
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(text, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            response = tok.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            print(response)
        except Exception as e:
            print(f"[error] LLM generation failed: {e}")


if __name__ == "__main__":
    main()
