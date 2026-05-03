"""Merge LoRA adapters back into the base model.

Usage:
  python merge_lora.py <base_model_path> <adapter_path> <output_path>

Emits progress lines to stdout so the UI can show them.
"""

import json
import shutil
import sys
import tempfile
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

    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    emit({"type": "status", "message": "Loading base model..."})
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    emit({"type": "status", "message": "Loading adapter and merging..."})
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    emit({"type": "status", "message": f"Saving merged model to {output_path}..."})

    # Write to a temp dir first, then atomically replace the final output dir.
    # This prevents stale/partial shard files from previous merges from corrupting loads.
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}-tmp-", dir=str(output_dir.parent))
    )

    try:
        merged.save_pretrained(str(temp_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(temp_dir))

        # Basic sanity check: merged model weights must exist before publishing.
        has_weights = (
            (temp_dir / "model.safetensors").exists()
            or (temp_dir / "pytorch_model.bin").exists()
            or (temp_dir / "model.safetensors.index.json").exists()
            or (temp_dir / "pytorch_model.bin.index.json").exists()
        )
        if not has_weights:
            raise RuntimeError("Merged output is missing model weight files")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    finally:
        # If something failed before replace(), clean up the temp dir.
        if temp_dir.exists() and temp_dir != output_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    emit({"type": "done", "output": output_path})


if __name__ == "__main__":
    main()
