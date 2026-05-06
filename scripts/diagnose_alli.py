"""Quick diagnostic: how many of Alli's ground-truth books are in the Atlas pool."""
from __future__ import annotations
import re
import os
from dotenv import load_dotenv

load_dotenv()

from mylm.rag.goodreads import fetch_read_shelf
from mylm.rag.db import get_db, load_embedder, embed_text, vector_search_books


def _norm(t: str) -> str:
    t = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
    t = re.sub(r"\s+by\s+.*$", "", t, flags=re.I).strip()
    for art in ("the ", "a ", "an "):
        if t.lower().startswith(art):
            t = t[len(art):]
    return t.lower().strip()


print("Fetching Alli shelves...")
read = fetch_read_shelf("40285049", shelf="read", max_books=400)
to_read = fetch_read_shelf("40285049", shelf="to-read", max_books=200)
loved = [b for b in read if b.get("rating", 0) >= 4]
ground_truth = {_norm(b["title"]) for b in to_read + loved if b.get("title")}
print(f"Read: {len(read)}  To-read: {len(to_read)}  Loved (4+): {len(loved)}")
print(f"Ground truth size: {len(ground_truth)}")

db = get_db()
embedder = load_embedder()
dummy_vec = embed_text(embedder, "book fiction novel")
pool = vector_search_books(db, dummy_vec, limit=2000, num_candidates=4000)
pool_norms = {_norm(b.get("Title") or b.get("title") or "") for b in pool if (b.get("Title") or b.get("title"))}

overlap = ground_truth & pool_norms
print(f"\nAtlas pool size: {len(pool)}")
print(f"Overlap (GT & pool): {len(overlap)}")
print("Matched:", sorted(overlap)[:20])

# ── Full pool coverage ────────────────────────────────────────────────────────
from pymongo import MongoClient
client = MongoClient(os.environ["MONGODB_URI"])
coll = client[os.environ.get("MONGODB_DB", "bookmind")]["books"]
total = coll.count_documents({})
print(f"\nFull Atlas collection: {total} books")
full_pool_norms = {_norm(d.get("Title") or d.get("title") or "") for d in coll.find({}, {"Title": 1, "title": 1})}
full_overlap = ground_truth & full_pool_norms
print(f"Full overlap (GT & full pool): {len(full_overlap)}")
print("Matched:", sorted(full_overlap)[:30])
print("\n── Per-probe LLM trace (first 3 probes) ──")
from mylm.rag.db import build_taste_vector_from_titles
taste_vec, _ = build_taste_vector_from_titles(db, embedder, read)

# Import rag_llm_recommend from benchmark script inline
import sys; sys.path.insert(0, "scripts")
from benchmark_book_rec import rag_llm_recommend, PROBES, _norm

# Try to load the model
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", "scripts/benchmark_book_rec.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    our_pipe = mod.load_our_model("models/book-rec-lora")
except Exception as e:
    print(f"Could not load model: {e}")
    our_pipe = None

taste_summary = (
    f"Loves: Fantasy, Science Fiction, Japan, Romance. "
    f"Read {len(read)} books. "
    f"Recently loved: {', '.join(b.get('title','') for b in read[:3] if b.get('rating',0)>=4)}"
)

for probe in PROBES[:3]:
    print(f"\nProbe: {probe[:60]}")
    recs, hits = rag_llm_recommend(our_pipe, db, embedder, probe, taste_summary, taste_vec, k=10)
    gt_in_candidates = [b.get("Title","") for b in hits if _norm(b.get("Title","")) in ground_truth]
    hits_in_gt = [r for r in recs if _norm(r) in ground_truth or r.lower().strip() in ground_truth]
    print(f"  Candidates with GT titles: {gt_in_candidates[:5]}")
    print(f"  LLM picked (hits in GT): {hits_in_gt}")
    print(f"  LLM picked: {recs[:5]}")
