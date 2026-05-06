# BookMind RAG

BookMind is the optional book-recommendation feature. It does semantic search over a MongoDB Atlas cluster using `$vectorSearch` on 384-dim embeddings, with optional LLM-grounded explanation.

## Requirements

- MongoDB Atlas cluster (free tier works) with the BookMind data loaded
- An Atlas Vector Search index on the `books` collection
- The same embedding model used to populate the embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)

## Configuration

Copy `.env.example` to `.env` and set:

```env
MONGODB_URI=mongodb+srv://<user>:<pw>@<cluster>.mongodb.net/?appName=BookMind
MONGODB_DB=bookmind
BOOKMIND_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
BOOKMIND_VECTOR_INDEX=vs_books_embedding
```

## Atlas Vector Search index definition

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

## Schema expectations

`books` collection:

```jsonc
{
  "_id": "...",
  "title": "...",
  "author": "...",
  "summary": "...",
  "embedding": [/* 384 floats */]
}
```

`users` collection (optional, for taste-blending and exclusions):

```jsonc
{
  "email": "you@example.com",
  "tasteVector": [/* 384 floats */],
  "userBooks": [{ "bookId": "..." }]
}
```

## Usage

From the UI: **Books** screen → enter a query → optionally select a user → click Recommend.

From the CLI:

```bash
python scripts/book_recommend.py "a cozy fantasy with witches" --user you@example.com --llm
```

## How LLM-grounded recommendations work

1. Vector search returns top-K candidates from `books`
2. The candidate set is passed verbatim to the local LLM as context
3. The LLM is instructed to recommend **only** from the provided set, with reasoning
4. Output streams to the renderer

This keeps the model from hallucinating books that don't exist.

---

## Fine-tuning the BookMind model

BookMind uses a two-stage QLoRA fine-tune on top of Qwen3.5-2B.

### Stage 1 — General book knowledge (one-time, ~7h)

Trains on 5,000 examples from `alhosseini/book-recommendation-chat` (HuggingFace) to teach the base model book recommendation conversation patterns.

```powershell
# Build the Stage 1 dataset (already done — datasets/books/hf_stage1.jsonl)
.venv\Scripts\python.exe scripts/blend_hf_dataset.py --n 5000 --stream-buffer 30000

# Run Stage 1 training
.venv\Scripts\python.exe scripts/train_qlora.py `
    --data datasets/books/hf_stage1.jsonl `
    --output-dir models/book-rec-stage1 `
    --epochs 1 `
    --lr 2e-4
```

Output: `models/book-rec-stage1/` (LoRA adapter). **Only needs to run once.**

### Stage 2 — Personal fine-tune (~2-3h)

Trains on your personal Goodreads reading history (ratings, reviews, taste profiles) exported via `build_book_dataset.py`. This is what makes recommendations personal.

```powershell
# Build personal dataset from Goodreads
.venv\Scripts\python.exe scripts/build_book_dataset.py --no-mongo

# Run Stage 2
.venv\Scripts\python.exe scripts/train_qlora.py `
    --model-id models/qwen3.5-2b `
    --data datasets/books/train.jsonl `
    --val-data datasets/books/val.jsonl `
    --output-dir models/book-rec-lora `
    --epochs 5 `
    --lr 5e-5 `
    --resume-from-lora models/book-rec-stage1
```

Output: `models/book-rec-lora/` (personal LoRA adapter).

### Updating after reading new books

When you've read 5-10+ new books, re-run Stage 2 only. Use a smaller LR and fewer epochs to avoid catastrophic forgetting:

```powershell
# 1. Rebuild dataset from latest Goodreads shelf
.venv\Scripts\python.exe scripts/build_book_dataset.py --no-mongo

# 2. Incremental Stage 2 — resumes from existing personal LoRA
.venv\Scripts\python.exe scripts/train_qlora.py `
    --model-id models/qwen3.5-2b `
    --data datasets/books/train.jsonl `
    --val-data datasets/books/val.jsonl `
    --output-dir models/book-rec-lora `
    --epochs 3 `
    --lr 2e-5 `
    --resume-from-lora models/book-rec-lora
```

Key differences from initial Stage 2: `--epochs 3` (down from 5), `--lr 2e-5` (down from `5e-5`), and `--resume-from-lora` points at `models/book-rec-lora` (the existing personal adapter) rather than the Stage 1 output.

### Summary

| Stage | Dataset | Time | Frequency |
|---|---|---|---|
| Stage 1 | `hf_stage1.jsonl` (5k HF examples) | ~7h | **Once ever** |
| Stage 2 (initial) | `train.jsonl` (personal, 204 examples) | ~2-3h | Once after Stage 1 |
| Stage 2 (update) | `train.jsonl` (rebuilt) | ~1-2h | After every 5-10 new books read |
