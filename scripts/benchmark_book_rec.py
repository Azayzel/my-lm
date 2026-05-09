"""Benchmark BookMind against HuggingFace book recommendation baselines.

Evaluates three systems on the same probe queries derived from Josh/Alli's
actual Goodreads data:

  1. AventIQ-AI/all-MiniLM-L6-v2-book-recommendation-system
     (embedding similarity — current HF SOTA, NDCG 0.82)
  2. Our RAG pipeline  (Atlas vector search, no LLM)
  3. Our fine-tuned model  (Stage 2 LoRA, personalised)

Metrics
-------
  Precision@K   — how many returned books are in the user's to-read/loved list
  NDCG@K        — normalised discounted cumulative gain (rank-sensitive)
  Novel@K       — fraction of recs NOT already in the user's read list
  PersonalScore — fraction of recs matching the user's stated favourite genres

Usage:
    python scripts/benchmark_book_rec.py
    python scripts/benchmark_book_rec.py --no-mongo --k 5
    python scripts/benchmark_book_rec.py --our-model models/book-rec-lora --k 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ─── User ground truth ────────────────────────────────────────────────────────

USERS = [
    {
        "name": "Josh",
        "goodreads_id": "64412103",
        "favorite_genres": {"Fantasy", "Paranormal", "Science Fiction", "Thriller"},
    },
    {
        "name": "Alli",
        "goodreads_id": "40285049",
        "favorite_genres": {
            "Classics", "Contemporary", "Fantasy",
            "Fiction", "Historical Fiction", "Mystery",
        },
    },
]

# Probes: free-text queries we'll fire at every system
PROBES = [
    "dark fantasy with morally complex characters",
    "cozy mystery set in historical England",
    "science fiction first contact with philosophical depth",
    "literary fiction about family secrets",
    "horror with dread and atmosphere, not gore",
    "a short, intense read under 300 pages",
    "epic fantasy with intricate world-building",
    "contemporary literary novel with great prose",
]


# ─── Metrics ──────────────────────────────────────────────────────────────────

import re as _re
_BY_AUTHOR_RE = _re.compile(r"\s+by\s+.+$", _re.IGNORECASE)


def _norm(title: str) -> str:
    """Lowercase, strip leading articles, strip ' by Author' suffix, series info, collapse whitespace."""
    t = title.lower().strip()
    t = _BY_AUTHOR_RE.sub("", t)                    # strip " by Author Name"
    t = _re.sub(r'\s*\([^)]*#\d[^)]*\)', '', t)    # strip "(Series Name, #1)"
    t = _re.sub(r'\s*\([^)]*book\s+\d[^)]*\)', '', t, flags=_re.IGNORECASE)  # strip "(Book 1)"
    t = _re.sub(r'^(the|a|an)\s+', '', t)           # strip leading articles
    t = _re.sub(r'[^\w\s]', '', t)                  # strip punctuation
    t = _re.sub(r'\s+', ' ', t).strip()
    return t


def _titles_set(books: list[dict]) -> set[str]:
    titles = set()
    for b in books:
        raw = (b.get("Title") or b.get("title") or "").strip()
        if raw:
            titles.add(_norm(raw))
            titles.add(raw.lower().strip())  # also keep exact for safety
    return titles


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for t in recommended[:k] if _norm(t) in relevant or t.lower().strip() in relevant)
    return hits / k if k > 0 else 0.0


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """NDCG@K where relevant books score 1, others 0."""
    def _hit(t: str) -> bool:
        return _norm(t) in relevant or t.lower().strip() in relevant
    dcg = sum(
        (1.0 if _hit(t) else 0.0) / math.log2(i + 2)
        for i, t in enumerate(recommended[:k])
    )
    # Ideal: all relevant at top
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def novel_at_k(recommended: list[str], already_read: set[str], k: int) -> float:
    novel = sum(1 for t in recommended[:k] if _norm(t) not in already_read and t.lower().strip() not in already_read)
    return novel / k if k > 0 else 0.0


def personal_score(recommended: list[str], all_books: list[dict], fav_genres: set[str], k: int) -> float:
    """Fraction of top-K recs that match the user's favourite genres."""
    book_genres: dict[str, set[str]] = {}
    for b in all_books:
        raw = (b.get("Title") or b.get("title") or "")
        genres = set(g.lower() for g in (b.get("Genres") or b.get("categories") or []))
        # Index by both normalized and raw-lower so LLM output variations still match
        book_genres[_norm(raw)] = genres
        book_genres[raw.lower()] = genres

    fav_lower = {g.lower() for g in fav_genres}
    matches = 0
    for title in recommended[:k]:
        book_g = book_genres.get(_norm(title)) or book_genres.get(title.lower(), set())
        if book_g & fav_lower:
            matches += 1
    return matches / k if k > 0 else 0.0


# ─── System: HF baseline (all-MiniLM-L6-v2-book-recommendation-system) ───────


def load_hf_baseline() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
        import torch
        model = SentenceTransformer(
            "AventIQ-AI/all-MiniLM-L6-v2-book-recommendation-system"
        )
        return model
    except Exception as e:
        print(f"[hf_baseline] load failed: {e}")
        return None


def hf_baseline_recommend(
    model: Any,
    query: str,
    candidate_books: list[dict],
    k: int,
) -> list[str]:
    """Embed query + candidate titles; return top-K by cosine similarity."""
    try:
        from sentence_transformers.util import cos_sim
        import torch

        titles = [
            (b.get("Title") or b.get("title") or "")
            for b in candidate_books
        ]
        query_emb = model.encode(query, convert_to_tensor=True)
        title_embs = model.encode(titles, convert_to_tensor=True, batch_size=64)
        sims = cos_sim(query_emb, title_embs).squeeze(0)
        top_idx = torch.topk(sims, k=min(k, len(titles))).indices.cpu().tolist()
        return [titles[i] for i in top_idx]
    except Exception:
        return []


# ─── System: Our RAG pipeline ─────────────────────────────────────────────────


def rag_recommend(
    db: Any,
    embedder: Any,
    query: str,
    taste_vec: list[float] | None,
    k: int,
) -> tuple[list[str], list[dict]]:
    """Vector search + optional taste vector blending."""
    from mylm.rag.db import embed_text, vector_search_books

    query_vec = embed_text(embedder, query)

    # Blend query vec with taste vec 50/50 if available
    if taste_vec:
        blended = [
            0.5 * q + 0.5 * t
            for q, t in zip(query_vec, taste_vec)
        ]
        # renormalise
        mag = sum(v ** 2 for v in blended) ** 0.5
        query_vec = [v / mag for v in blended] if mag > 0 else query_vec

    hits = vector_search_books(db, query_vec, limit=k, num_candidates=k * 5)
    return [b.get("Title") or "" for b in hits], hits


# ─── System: Our fine-tuned LLM ───────────────────────────────────────────────


def load_our_model(model_path: str) -> Any | None:
    import warnings
    warnings.filterwarnings("ignore", message="Passing `generation_config` together")
    warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModel
        import torch

        base = "models/qwen3.5-2b"
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.float16, device_map="auto"
        )
        from transformers import GenerationConfig, PreTrainedModel
        peft_model = PeftModel.from_pretrained(model, model_path)
        # Replace generation_config to avoid "both max_new_tokens and max_length set" warning
        gc = GenerationConfig(max_length=4096)
        peft_model.generation_config = gc  # type: ignore[assignment]
        pipe = pipeline(
            "text-generation",
            model=cast(PreTrainedModel, peft_model),
            tokenizer=tokenizer,
        )
        return pipe
    except Exception as e:
        print(f"[our_model] load failed: {e}")
        return None


def llm_recommend(pipe: Any, query: str, taste_summary: str, k: int) -> list[str]:
    """Ask the fine-tuned LLM for K recommendations and parse titles."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are BookMind. Return ONLY a numbered list of book titles, "
                "one per line, no extra commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User taste: {taste_summary}\n\n"
                f"Query: {query}\n\n"
                f"List {k} book recommendations:"
            ),
        },
    ]
    try:
        out = pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        # Extract last assistant turn
        if isinstance(out, list):
            asst = [m for m in out if m.get("role") == "assistant"]
            text = asst[-1]["content"] if asst else ""
        else:
            text = str(out)
        titles = []
        for line in text.splitlines():
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                titles.append(line)
        return titles[:k]
    except Exception:
        return []


def rag_llm_recommend(
    pipe: Any,
    db: Any,
    embedder: Any,
    query: str,
    taste_summary: str,
    taste_vec: list[float] | None,
    k: int,
) -> tuple[list[str], list[dict]]:
    """RAG-augmented LLM: retrieve 3×K candidates from Atlas, ask LLM to pick best K.

    This constrains the LLM to books that actually exist in the pool, making
    precision/personal_score metrics meaningful.
    """
    from mylm.rag.db import embed_text, vector_search_books

    # Retrieve candidate pool (10× k for diversity — wider net increases GT coverage)
    candidates_k = min(k * 10, 150)
    query_vec = embed_text(embedder, query)
    if taste_vec:
        blended = [0.5 * q + 0.5 * t for q, t in zip(query_vec, taste_vec)]
        mag = sum(v ** 2 for v in blended) ** 0.5
        query_vec = [v / mag for v in blended] if mag > 0 else query_vec
    hits = vector_search_books(db, query_vec, limit=candidates_k, num_candidates=candidates_k * 5)
    if not hits:
        return [], []

    candidate_list = "\n".join(
        f"{i+1}. {b.get('Title', '')} by {', '.join(b.get('Authors') or [b.get('Author', '')])}"
        for i, b in enumerate(hits)
    )

    prompt = [
        {
            "role": "system",
            "content": (
                "You are BookMind. You will be given a numbered list of candidate books. "
                f"Reply with ONLY {k} numbers (e.g. 3, 7, 12) separated by commas — "
                "the numbers of the best candidates for the user. No titles, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User taste: {taste_summary}\n\n"
                f"Query: {query}\n\n"
                f"Candidates:\n{candidate_list}\n\n"
                f"Pick the best {k} by their numbers only:"
            ),
        },
    ]
    try:
        out = pipe(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
        if isinstance(out, list):
            asst = [m for m in out if m.get("role") == "assistant"]
            text = asst[-1]["content"] if asst else ""
        else:
            text = str(out)
        # Parse comma/space/newline-separated integers and map back to hit titles
        import re as _re
        nums = [int(m) for m in _re.findall(r"\b(\d+)\b", text) if 1 <= int(m) <= len(hits)]
        # Deduplicate while preserving order
        seen: set[int] = set()
        picked = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                picked.append(hits[n - 1].get("Title", ""))
        return picked[:k], hits
    except Exception:
        return [], hits


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="Skip Atlas RAG system (only benchmark HF baseline)",
    )
    parser.add_argument(
        "--our-model",
        default=None,
        help="Path to our fine-tuned LoRA adapter (optional; skipped if not provided)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/book_rec_bench.json"),
    )
    args = parser.parse_args()

    results: dict[str, Any] = {"k": args.k, "users": {}}

    # ── Load Goodreads ground truth ───────────────────────────────────────────
    print("Fetching Goodreads ground truth...")
    from mylm.rag.goodreads import fetch_read_shelf

    user_data: dict[str, dict] = {}
    for user in USERS:
        read = fetch_read_shelf(user["goodreads_id"], shelf="read", max_books=400)
        to_read = fetch_read_shelf(user["goodreads_id"], shelf="to-read", max_books=200)
        # Ground truth = to-read shelf UNION books the user rated 4-5★
        loved = [b for b in read if b.get("rating", 0) >= 4]
        relevant_titles = _titles_set(to_read) | _titles_set(loved)
        user_data[user["name"]] = {
            "read_titles": _titles_set(read),
            "to_read_titles": relevant_titles,
            "taste_summary": (
                f"Loves: {', '.join(sorted(user['favorite_genres']))}. "
                f"Read {len(read)} books. "
                f"Recently loved: {', '.join(b.get('title','') for b in loved[:8])}"
            ),
        }
        print(f"  {user['name']}: {len(read)} read, {len(to_read)} to-read, {len(loved)} loved (4-5★)")

    # ── Load systems ──────────────────────────────────────────────────────────
    print("\nLoading systems...")
    hf_model = load_hf_baseline()
    print(f"  HF baseline: {'OK' if hf_model else 'FAILED'}")

    db = embedder = None
    taste_vecs: dict[str, list[float] | None] = {}
    all_atlas_books: list[dict] = []

    if not args.no_mongo:
        try:
            from mylm.rag.db import get_db, load_embedder, build_taste_vector_from_titles
            db = get_db()
            embedder = load_embedder()
            print("  RAG (Atlas): OK")
            for user in USERS:
                read = fetch_read_shelf(user["goodreads_id"], shelf="read", max_books=100)
                vec, _ = build_taste_vector_from_titles(db, embedder, read)
                taste_vecs[user["name"]] = vec
        except Exception as e:
            print(f"  RAG (Atlas): FAILED — {e}")

    our_pipe = None
    if args.our_model:
        our_pipe = load_our_model(args.our_model)
        print(f"  Our model: {'OK' if our_pipe else 'FAILED'}")

    # ── Candidate pool for HF baseline (needs a book list) ────────────────────
    # Use Atlas if available, else build a small pool from OL
    if db is not None and embedder is not None:
        from mylm.rag.db import embed_text, vector_search_books
        dummy_vec = embed_text(embedder, "book fiction novel")
        all_atlas_books = vector_search_books(db, dummy_vec, limit=2000, num_candidates=4000)
        print(f"  Candidate pool: {len(all_atlas_books)} Atlas books")
    else:
        from mylm.rag.open_library import fetch_genre_catalog
        all_genres = list({g for u in USERS for g in u["favorite_genres"]})
        all_atlas_books = fetch_genre_catalog(all_genres, per_genre=20)
        print(f"  Candidate pool: {len(all_atlas_books)} OL books (no Atlas)")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    print(f"\nRunning {len(PROBES)} probes × {len(USERS)} users × available systems...\n")

    for user in USERS:
        uname = user["name"]
        udata = user_data[uname]
        relevant = udata["to_read_titles"]  # ground truth: books user wants to read
        already_read = udata["read_titles"]
        taste_summary = udata["taste_summary"]
        taste_vec = taste_vecs.get(uname)

        sys_scores: dict[str, dict[str, list[float]]] = {
            "hf_baseline": {"precision": [], "ndcg": [], "novel": [], "personal": []},
            "rag": {"precision": [], "ndcg": [], "novel": [], "personal": []},
            "our_model": {"precision": [], "ndcg": [], "novel": [], "personal": []},
        }

        for probe in PROBES:
            print(f"  [{uname}] {probe[:55]}")

            # HF baseline
            if hf_model and all_atlas_books:
                recs = hf_baseline_recommend(hf_model, probe, all_atlas_books, args.k)
                sys_scores["hf_baseline"]["precision"].append(precision_at_k(recs, relevant, args.k))
                sys_scores["hf_baseline"]["ndcg"].append(ndcg_at_k(recs, relevant, args.k))
                sys_scores["hf_baseline"]["novel"].append(novel_at_k(recs, already_read, args.k))
                sys_scores["hf_baseline"]["personal"].append(personal_score(recs, all_atlas_books, user["favorite_genres"], args.k))

            # RAG
            if db is not None and embedder is not None:
                recs, hits = rag_recommend(db, embedder, probe, taste_vec, args.k)
                sys_scores["rag"]["precision"].append(precision_at_k(recs, relevant, args.k))
                sys_scores["rag"]["ndcg"].append(ndcg_at_k(recs, relevant, args.k))
                sys_scores["rag"]["novel"].append(novel_at_k(recs, already_read, args.k))
                sys_scores["rag"]["personal"].append(personal_score(recs, hits, user["favorite_genres"], args.k))

            # Our LLM — RAG-augmented when Atlas is available, free-form otherwise
            if our_pipe:
                if db is not None and embedder is not None:
                    recs, hits = rag_llm_recommend(
                        our_pipe, db, embedder, probe, taste_summary, taste_vec, args.k
                    )
                    sys_scores["our_model"]["precision"].append(precision_at_k(recs, relevant, args.k))
                    sys_scores["our_model"]["ndcg"].append(ndcg_at_k(recs, relevant, args.k))
                    sys_scores["our_model"]["novel"].append(novel_at_k(recs, already_read, args.k))
                    sys_scores["our_model"]["personal"].append(personal_score(recs, hits, user["favorite_genres"], args.k))
                else:
                    recs = llm_recommend(our_pipe, probe, taste_summary, args.k)
                    sys_scores["our_model"]["precision"].append(precision_at_k(recs, relevant, args.k))
                    sys_scores["our_model"]["ndcg"].append(ndcg_at_k(recs, relevant, args.k))
                    sys_scores["our_model"]["novel"].append(novel_at_k(recs, already_read, args.k))
                    sys_scores["our_model"]["personal"].append(0.0)

        # Average over probes
        def _avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        user_results = {}
        for sys_name, metrics in sys_scores.items():
            if not metrics["precision"]:  # skip if system never ran (empty list)
                continue
            user_results[sys_name] = {
                f"precision@{args.k}": _avg(metrics["precision"]),
                f"ndcg@{args.k}": _avg(metrics["ndcg"]),
                f"novel@{args.k}": _avg(metrics["novel"]),
                f"personal_score@{args.k}": _avg(metrics["personal"]),
            }

        results["users"][uname] = user_results

        # Pretty print
        print(f"\n  ─── {uname} ───")
        for sys_name, m in user_results.items():
            print(f"  {sys_name:30s}  " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        print()

    # ── Write JSON results ────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written → {args.out}")


if __name__ == "__main__":
    main()
