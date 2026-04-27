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
