"""
model_download.py - Downloads models from Hugging Face Hub.

Writes newline-delimited JSON progress to stdout.
Usage:
  python model_download.py <repo_id> <local_dir> [--type llm|image]
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def main():
    import argparse

    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. Qwen/Qwen2.5-1.5B")
    parser.add_argument("local_dir", help="Local directory to save the model")
    parser.add_argument("--type", default="llm", choices=["llm", "image"])
    args = parser.parse_args()

    emit({"type": "status", "message": f"Fetching model info for {args.repo_id}..."})

    tags = []
    pipeline_tag = None
    should_block_download = False
    try:
        api = HfApi()
        info = api.model_info(args.repo_id)
        tags = info.tags or []
        pipeline_tag = getattr(info, "pipeline_tag", None)
        emit({"type": "info", "id": args.repo_id, "tags": tags})

        image_hint = {
            "diffusers",
            "stable-diffusion",
            "text-to-image",
            "image-to-image",
        }
        llm_hint = {
            "text-generation",
            "causal-lm",
            "gguf",
            "llama",
            "qwen",
            "gemma",
        }
        tag_set = {str(t).lower() for t in tags}
        pipe = str(pipeline_tag or "").lower()

        if args.type == "image" and not (tag_set & image_hint or pipe in image_hint):
            if tag_set & llm_hint or pipe in llm_hint:
                should_block_download = True
                emit(
                    {
                        "type": "error",
                        "message": (
                            "Selected type is 'image', but this repository looks like an LLM "
                            "(text-generation/GGUF). Choose type 'llm' or use an image model repo."
                        ),
                    }
                )
    except Exception as e:
        emit({"type": "error", "message": f"Could not fetch model info: {e}"})

    if should_block_download:
        sys.exit(1)

    emit({"type": "status", "message": f"Downloading {args.repo_id} to {args.local_dir}..."})

    os.makedirs(args.local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
        )
        emit({"type": "done", "path": args.local_dir})
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
