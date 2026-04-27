"""
book_bridge.py — BookMind RAG bridge for the MyLm UI.

Protocol:
  Reads newline-delimited JSON requests from stdin.
  Emits newline-delimited JSON events to stdout.

Request shape:
  {
    "query": "a mystery like Tana French set in Japan",
    "user_id": "<optional user _id or email>",
    "limit": 10,                 // top-K books from vector search
    "stream": true,              // stream LLM tokens (ignored — LLM is optional)
    "llm_model_dir": "models/qwen2.5-3b",   // optional, for generating explanations
    "filter": {"Genres": {"$in": ["mystery"]}}   // optional MongoDB filter
  }

Events:
  {"type": "status",     "message": "..."}
  {"type": "candidates", "books": [...]}        // raw vector-search hits
  {"type": "token",      "text": "..."}         // LLM token stream
  {"type": "done",       "text": "..."}         // final complete response
  {"type": "error",      "message": "..."}

Usage:
  python book_bridge.py
  then send JSON lines to stdin, read events from stdout.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from typing import Any

# Force UTF-8 on stdin/stdout so emojis/unicode don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Allow running without `pip install -e .` by adding src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


# Lazy module-level singletons — loaded on first request
_db = None
_embedder = None
_llm_state: dict[str, Any] = {"tokenizer": None, "model": None, "path": None}


def get_resources():
    global _db, _embedder
    if _db is None:
        from mylm.rag import get_db, load_embedder
        emit({"type": "status", "message": "Connecting to MongoDB..."})
        _db = get_db()
        emit({"type": "status", "message": "Loading sentence-transformer..."})
        _embedder = load_embedder()
        emit({"type": "status", "message": "Ready"})
    return _db, _embedder


def ensure_llm(model_path: str | None):
    """Load a causal LM for explanation generation. Lazy and cached."""
    if not model_path:
        return None
    if _llm_state["path"] == model_path and _llm_state["model"] is not None:
        return _llm_state
    emit({"type": "status", "message": f"Loading LLM from {model_path}..."})
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        _llm_state["tokenizer"] = tok
        _llm_state["model"] = model
        _llm_state["streamer_cls"] = TextIteratorStreamer
        _llm_state["path"] = model_path
    except Exception as e:
        emit({"type": "error", "message": f"LLM load failed: {e}"})
        return None
    return _llm_state


def stream_llm_response(
    llm: dict,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 600,
) -> str:
    """Generate a streaming response from the loaded LLM.

    Emits {"type": "token", ...} events and returns the full concatenated text.
    """

    tok = llm["tokenizer"]
    model = llm["model"]
    streamer_cls = llm["streamer_cls"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    streamer = streamer_cls(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    full = []
    for chunk in streamer:
        if chunk:
            emit({"type": "token", "text": chunk})
            full.append(chunk)
    thread.join()
    return "".join(full)


def simplify_book(book: dict) -> dict:
    """Strip MongoDB-heavy fields (embedding, _id ObjectId) for JSON serialization."""
    keep = {
        "_id",
        "Title",
        "Subtitle",
        "Authors",
        "Genres",
        "Themes",
        "Moods",
        "Pacing",
        "Description",
        "PublishedDate",
        "AvgRating",
        "RatingCount",
        "CoverImageUrl",
        "PageCount",
        "Isbn13",
        "score",
    }
    out = {k: book.get(k) for k in keep if k in book}
    # Convert ObjectId → str and published date → ISO string for JSON
    if "_id" in out and out["_id"] is not None:
        out["_id"] = str(out["_id"])
    if "PublishedDate" in out and out["PublishedDate"] is not None:
        out["PublishedDate"] = str(out["PublishedDate"])
    return out


def run_query(request: dict) -> None:
    from mylm.rag import (
        build_system_prompt,
        build_taste_vector_from_titles,
        build_user_prompt,
        embed_text,
        get_user_read_ids,
        get_user_taste_vector,
        vector_search_books,
    )

    query_text = (request.get("query") or "").strip()
    user_id = request.get("user_id")
    goodreads_user = (request.get("goodreads_user") or "").strip()
    limit = int(request.get("limit") or 10)
    num_candidates = max(limit * 10, 200)
    mongo_filter = request.get("filter") or None
    llm_path = request.get("llm_model_dir")

    if not query_text and not user_id and not goodreads_user:
        emit({"type": "error", "message": "query, user_id, or goodreads_user is required"})
        return

    db, embedder = get_resources()

    # Build the query vector — prefer the user's taste vector if provided
    taste_vec = None
    taste_source: str | None = None

    if user_id:
        taste_vec = get_user_taste_vector(db, user_id)
        if taste_vec:
            taste_source = f"stored tasteVector for {user_id}"
            emit({"type": "status", "message": f"Using {taste_source}"})

    gr_excluded_titles: set[str] = set()
    if goodreads_user and taste_vec is None:
        from mylm.rag import fetch_read_shelf

        emit({"type": "status", "message": f"Fetching Goodreads profile: {goodreads_user}..."})
        gr_books = fetch_read_shelf(goodreads_user, max_books=80)
        if not gr_books:
            emit({"type": "log", "text": f"No public books found for Goodreads user '{goodreads_user}' (profile may be private or username wrong)"})
        else:
            emit({
                "type": "status",
                "message": f"Fetched {len(gr_books)} books from Goodreads, building taste vector...",
            })
            taste_vec, resolved = build_taste_vector_from_titles(db, embedder, gr_books)
            emit({"type": "goodreads_matched", "books": resolved, "fetched": len(gr_books)})
            if taste_vec:
                taste_source = f"Goodreads profile '{goodreads_user}' ({len(resolved)} matched books)"
                gr_excluded_titles = {
                    (b.get("matched_title") or "").strip().lower()
                    for b in resolved
                    if b.get("matched_title")
                }

    if query_text:
        q_vec = embed_text(embedder, query_text)
        if taste_vec:
            # Blend free-text query (70%) with user taste (30%)
            blended = [0.7 * a + 0.3 * b for a, b in zip(q_vec, taste_vec, strict=False)]
            import math
            norm = math.sqrt(sum(x * x for x in blended)) or 1.0
            query_vec = [x / norm for x in blended]
        else:
            query_vec = q_vec
    else:
        query_vec = taste_vec

    if query_vec is None:
        emit({"type": "error", "message": "Could not build a query vector (empty query and no user data)"})
        return

    # Build exclusion sets for post-filtering. Atlas $vectorSearch only
    # supports filters on fields explicitly marked as filter fields in the
    # index definition, which we don't have — so we do post-filtering here.
    excluded_ids: set = set()
    if user_id:
        excluded_ids.update(get_user_read_ids(db, user_id))

    # Over-fetch so post-filtering still leaves `limit` results
    overfetch = limit
    if excluded_ids or gr_excluded_titles:
        overfetch = limit + max(len(gr_excluded_titles), 0) + 20

    emit({"type": "status", "message": f"Running vector search (limit={limit})..."})
    try:
        raw_books = vector_search_books(
            db,
            query_vec,
            limit=overfetch,
            num_candidates=num_candidates,
            mongo_filter=mongo_filter,
        )
    except Exception as e:
        emit({"type": "error", "message": f"Vector search failed: {e}"})
        return

    # Post-filter: drop books the user has already read and dedupe by
    # (title, first_author) so duplicate library entries don't flood results.
    books: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for b in raw_books:
        if excluded_ids and b.get("_id") in excluded_ids:
            continue
        title_lower = (b.get("Title") or "").strip().lower()
        if gr_excluded_titles and title_lower in gr_excluded_titles:
            continue
        authors = b.get("Authors") or []
        first_author = (authors[0] if authors else "").strip().lower()
        key = (title_lower, first_author)
        if key in seen:
            continue
        seen.add(key)
        books.append(b)
        if len(books) >= limit:
            break

    # Emit the raw candidates for UI display
    emit({"type": "candidates", "books": [simplify_book(b) for b in books]})

    # If an LLM path is provided, run a RAG generation pass
    if llm_path:
        llm = ensure_llm(llm_path)
        if llm:
            system = build_system_prompt()
            user = build_user_prompt(query_text or "Recommend books based on my taste.", books)
            emit({"type": "status", "message": "Generating recommendations..."})
            try:
                final = stream_llm_response(llm, system, user)
                emit({"type": "done", "text": final})
                return
            except Exception as e:
                emit({"type": "error", "message": f"LLM generation failed: {e}"})
                return

    emit({"type": "done", "text": ""})


def main() -> None:
    emit({"type": "status", "message": "BookMind bridge ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"Invalid JSON: {e}"})
            continue
        try:
            run_query(request)
        except Exception as e:
            import traceback

            emit(
                {
                    "type": "error",
                    "message": f"{e}\n{traceback.format_exc()}",
                }
            )


if __name__ == "__main__":
    main()
