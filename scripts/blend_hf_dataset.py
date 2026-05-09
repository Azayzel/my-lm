"""Sample and normalize HF book-recommendation datasets for two-stage training.

Pulls a quality-filtered sample from ``alhosseini/book-recommendation-chat``
(which is already in messages format), normalises it to match our JSONL schema,
and writes it ready for Stage 1 fine-tuning.

Two-stage training workflow
---------------------------
Stage 1  — large HF sample  → teaches book domain knowledge & rec format
Stage 2  — our personal data → stamps Josh/Alli taste onto the model

Usage:
    # Sample 5 000 rows from HF and write to datasets/books/hf_stage1.jsonl
    python scripts/blend_hf_dataset.py

    # Larger sample, stricter quality filter
    python scripts/blend_hf_dataset.py --n 15000 --min-history 5

    # Then run Stage 1
    python scripts/train_qlora.py \\
        --data datasets/books/hf_stage1.jsonl \\
        --output-dir models/book-rec-stage1 \\
        --epochs 1 --lr 2e-4

    # Then run Stage 2 (personal fine-tune from Stage 1 checkpoint)
    python scripts/train_qlora.py \\
        --model-id models/book-rec-stage1 \\
        --data datasets/books/train.jsonl \\
        --val-data datasets/books/val.jsonl \\
        --output-dir models/book-rec-lora \\
        --epochs 5 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are BookMind, an expert personal book recommendation engine. "
    "You understand reading taste deeply — genre preferences, thematic resonance, "
    "prose style, pacing, and mood. Given a user's reading history with ratings, "
    "you identify patterns in what they love and use that to recommend books "
    "they haven't read yet, always explaining your reasoning clearly."
)

# ─── Quality filters ──────────────────────────────────────────────────────────


def _count_history_books(user_content: str) -> int:
    """Count how many <book ...> tags are in the reading history."""
    return len(re.findall(r"<book\s+title=", user_content, re.IGNORECASE))


def _has_enough_assistant_content(assistant_content: str, min_chars: int = 100) -> bool:
    return len(assistant_content.strip()) >= min_chars


def _is_english(text: str) -> bool:
    """Very rough English check — rejects rows that are mostly non-ASCII."""
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return ascii_ratio > 0.85


def _normalise_system(messages: list[dict]) -> list[dict]:
    """Replace any existing system message with our BookMind system prompt.

    Keeps the conversation structure; just stamps our identity.
    """
    out = []
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            out.append({"role": "system", "content": SYSTEM_PROMPT})
            has_system = True
        else:
            out.append({"role": msg["role"], "content": msg["content"]})
    if not has_system:
        out.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return out


def _is_quality_row(
    row: dict[str, Any],
    min_history_books: int,
    min_assistant_chars: int,
) -> bool:
    """Return True if the row passes quality gates."""
    messages = row.get("messages") or []
    if not messages:
        return False

    user_msgs = [m for m in messages if m.get("role") == "user"]
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not user_msgs or not asst_msgs:
        return False

    user_content = user_msgs[0].get("content") or ""
    asst_content = asst_msgs[0].get("content") or ""

    if _count_history_books(user_content) < min_history_books:
        return False
    if not _has_enough_assistant_content(asst_content, min_assistant_chars):
        return False
    if not _is_english(user_content):
        return False

    return True


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/books/hf_stage1.jsonl"),
        help="Output path for Stage 1 training data",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Target number of quality-filtered examples to write",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=3,
        help="Minimum number of books in user reading history",
    )
    parser.add_argument(
        "--min-asst-chars",
        type=int,
        default=150,
        help="Minimum characters in assistant response",
    )
    parser.add_argument(
        "--stream-buffer",
        type=int,
        default=50000,
        help="How many rows to buffer from the stream before sampling (avoids downloading full dataset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    print("Streaming alhosseini/book-recommendation-chat from HuggingFace...")
    ds = load_dataset(
        "alhosseini/book-recommendation-chat",
        split="train",
        streaming=True,
    )

    # Buffer a chunk, quality-filter, then sample down to --n
    buffer: list[dict] = []
    kept = 0
    scanned = 0

    print(f"Scanning up to {args.stream_buffer:,} rows, targeting {args.n:,} quality examples...")
    for row in ds:
        scanned += 1
        if _is_quality_row(row, args.min_history, args.min_asst_chars):
            buffer.append(row)
            kept += 1
            if kept % 500 == 0:
                print(f"  scanned {scanned:,} | kept {kept:,}", end="\r", flush=True)
        if scanned >= args.stream_buffer:
            break

    print(f"\nScanned {scanned:,} rows → {kept:,} passed quality filter")

    if not buffer:
        sys.exit("No rows passed quality filter — try lowering --min-history or --min-asst-chars")

    # Sample down if we have more than needed
    sample = rng.sample(buffer, min(args.n, len(buffer)))
    print(f"Sampled {len(sample):,} examples for Stage 1")

    # Normalise system prompts
    normalised = []
    for row in sample:
        messages = _normalise_system(row["messages"])
        normalised.append({"messages": messages})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in normalised:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(normalised):,} examples → {args.out}")
    personal_data = Path("datasets/books/train.jsonl")
    personal_val = Path("datasets/books/val.jsonl")
    print(
        f"""
Two-stage training commands:

  # Stage 1 — book domain knowledge (run once, ~1 epoch on HF data)
  python scripts/train_qlora.py \\
      --data {args.out} \\
      --output-dir models/book-rec-stage1 \\
      --epochs 1 --lr 2e-4

  # Stage 2 — personal taste (fine-tune from Stage 1, low LR)
  python scripts/train_qlora.py \\
      --model-id models/book-rec-stage1 \\
      --data {personal_data} \\
      {"--val-data " + str(personal_val) + " \\\\" if personal_val.exists() else "\\\\"}
      --output-dir models/book-rec-lora \\
      --epochs 5 --lr 5e-5
"""
    )


if __name__ == "__main__":
    main()
