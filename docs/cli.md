# CLI tools

All scripts under `scripts/` run standalone — useful for quick smoke tests and headless use.

```bash
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
```

## Chat

```bash
python scripts/qwen_inference.py
```

One-shot interactive chat against the configured LLM.

## Image generation

```bash
python scripts/generate_image.py
```

Single SDXL generation with optional face fix and 4× upscale. Edit the script to change params, or use the UI.

## QLoRA training

```bash
python scripts/train_qlora.py
```

Trains from `datasets/train.jsonl`. Adapter saved under `models/`.

## Merge a LoRA adapter

```bash
python scripts/merge_lora.py <base_model_dir> <lora_adapter_dir> <output_dir>
```

## BookMind recommender

```bash
python scripts/book_recommend.py "a cozy fantasy with witches" --user you@example.com --llm
```

Flags:

- `--user <email>` — blend with user's `tasteVector`, exclude already-read books
- `--llm` — stream a grounded LLM explanation
- `--top-k <n>` — number of candidates (default 10)

Requires `.env` configured with Mongo + embedding settings — see [bookmind.md](bookmind.md).

## Model download

```bash
python scripts/model_download.py <hf_repo_id> [--dest models/<name>]
```

Wraps `huggingface_hub.snapshot_download`. Resumes on interruption.

## OpenLibrary ingest

```bash
# One-shot pass
python scripts/ol_ingest.py

# Loop continuously (or use the Windows service installer)
python scripts/ol_ingest.py --daemon --sleep-hours 1 --per-subject 100
```

Crawls Open Library subject catalogs, embeds each book with sentence-transformers, and upserts into the BookMind `books` collection. State is persisted in `ol_ingest_state` so it's resumable. See [ol_ingest.md](ol_ingest.md) for the full operational guide.

## Agent benchmark

```bash
# Validate task definitions, no Ollama required
python scripts/agent_bench.py --dry-run

# Quick run
python scripts/agent_bench.py --models llama3.2:3b \
    --tasks code_debugger,research_synth --trials 2
```

Multi-turn agent benchmark measuring TTFT, context drift, model reloads, and tool-call reliability. Results land in `benchmark_results/` as JSON + Markdown. Surfaced in the UI under the **Bench** screen.
