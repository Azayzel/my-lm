"""Merge LoRA adapters back into the base model.

Usage:
  python merge_lora.py <base_model_path> <adapter_path> <output_path>

Emits progress lines to stdout so the UI can show them.
"""

import sys
import json
from pathlib import Path


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> None:
    if len(sys.argv) != 4:
        emit({"type": "error", "message": "Usage: merge_lora.py <base> <adapter> <output>"})
        sys.exit(1)

    base_path = sys.argv[1]
    adapter_path = sys.argv[2]
    output_path = sys.argv[3]

    if not Path(base_path).exists():
        emit({"type": "error", "message": f"Base model not found: {base_path}"})
        sys.exit(1)
    if not Path(adapter_path).exists():
        emit({"type": "error", "message": f"Adapter not found: {adapter_path}"})
        sys.exit(1)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    emit({"type": "status", "message": "Loading base model..."})
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    emit({"type": "status", "message": "Loading adapter and merging..."})
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    emit({"type": "status", "message": f"Saving merged model to {output_path}..."})
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    emit({"type": "done", "output": output_path})


if __name__ == "__main__":
    main()
