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
