# Architecture

## Process model

```text
┌──────────────────────────────────────────────────────────────────┐
│ Electron main process (Node)                                     │
│  ├─ window + IPC                                                 │
│  ├─ llmBridge   ──spawns──▶  python scripts/llm_bridge.py        │
│  ├─ imageBridge ──spawns──▶  python scripts/image_bridge.py      │
│  ├─ trainBridge ──spawns──▶  python scripts/train_bridge.py      │
│  ├─ bookBridge  ──spawns──▶  python scripts/book_bridge.py       │
│  ├─ modelManager (HF snapshot_download)                          │
│  ├─ historyStore / promptStore / configStore (userData/*.json)   │
│  └─ preload exposes contextBridge API as window.My               │
│                                                                  │
│ Renderer (sandboxed)                                             │
│  └─ vanilla TS + webpack, talks only via window.My               │
└──────────────────────────────────────────────────────────────────┘
```

All Python processes are long-lived subprocesses spawned by the Electron main process. Screens can switch freely without interrupting any running operation.

## Bridge protocol

Bridges read newline-delimited JSON from stdin and emit newline-delimited JSON events to stdout. **Never** print non-JSON to stdout — it breaks the parser. Logging goes to stderr.

### Request format

```json
{ "id": "req-42", "type": "generate", "params": { "prompt": "...", "steps": 30 } }
```

### Event format

```json
{ "id": "req-42", "type": "progress", "step": 12, "total": 30 }
{ "id": "req-42", "type": "preview", "data": "data:image/png;base64,..." }
{ "id": "req-42", "type": "done",   "result": { "path": "outputs/...png" } }
{ "id": "req-42", "type": "error",  "message": "CUDA OOM" }
```

The TS side correlates events by `id` and resolves the calling promise on `done`/`error`.

## Renderer ↔ Main IPC

Renderer-only world; `nodeIntegration: false`, `contextIsolation: true`. The preload exposes a typed surface as `window.My`:

```ts
window.My.chat.send(message, opts)        // request/response (invoke)
window.My.chat.onToken(cb)                // streaming subscription
window.My.image.generate(params)
window.My.image.onProgress(cb)
window.My.train.start(config)
window.My.models.list() / install(repoId) / catalog()
window.My.gpu.info() / nvidiaSmi()
window.My.history.* / prompts.* / config.*
```

Long-running operations use fire-and-forget `send()` + an `onEvent` subscription for streaming updates back. Promise-style `invoke()` is used for short request/response calls.

## Memory budget

See [features.md#memory-budget-on-6gb](features.md#memory-budget-on-6-gb) for the per-feature breakdown.

## Repository layout

```text
my-lm/
├── pyproject.toml              Python package metadata + ruff/black/mypy/pytest config
├── requirements.txt            Pinned runtime dependencies (with CUDA-specific notes)
├── src/mylm/                   Reusable Python library (pip-installable as `mylm`)
│   ├── imaging/                ESRGAN upscaler + ADetailer-style face detailer
│   ├── rag/                    BookMind RAG: Atlas Vector Search + Goodreads import
│   └── io/                     Newline-delimited JSON helpers for bridge protocol
├── scripts/                    Entry-point scripts (spawned by Electron main)
│   ├── llm_bridge.py           Streaming chat (stdin→stdout JSON)
│   ├── image_bridge.py         SDXL + compel + face detailer + upscaler
│   ├── train_bridge.py         QLoRA training with streaming metrics
│   ├── book_bridge.py          BookMind RAG pipeline bridge
│   ├── merge_lora.py           Fuse LoRA adapter into base model
│   ├── model_download.py       HF snapshot_download wrapper
│   ├── generate_image.py       Standalone CLI image generation
│   ├── train_qlora.py          Standalone CLI training
│   ├── qwen_inference.py       Standalone CLI chat
│   └── book_recommend.py       Standalone CLI recommender
├── tests/                      pytest suite (unit tests for pure library code)
└── ui/
    ├── src/
    │   ├── main/               Main process + IPC
    │   │   ├── main.ts
    │   │   ├── preload.ts
    │   │   ├── llmBridge.ts
    │   │   ├── imageBridge.ts
    │   │   ├── trainBridge.ts
    │   │   ├── bookBridge.ts
    │   │   ├── modelManager.ts
    │   │   ├── modelCatalog.ts
    │   │   ├── historyStore.ts
    │   │   ├── promptStore.ts
    │   │   └── configStore.ts
    │   └── renderer/           UI (vanilla TS + webpack)
    │       ├── index.html
    │       ├── app.ts
    │       └── global.d.ts
    └── package.json
```

`models/` and `outputs/` are runtime-only and gitignored.

### Why `src/mylm/` and not flat `mylm/`?

The `src/` layout prevents Python from accidentally importing the in-repo package when you're in the project root — you must `pip install -e .` (or set `PYTHONPATH`) to use it. That makes "did my install actually work?" failures loud rather than silent. Entry-point scripts in `scripts/` add `src/` to `sys.path` themselves so they keep working without an editable install.
