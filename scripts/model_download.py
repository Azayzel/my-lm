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

    try:
        api = HfApi()
        info = api.model_info(args.repo_id)
        emit({"type": "info", "id": args.repo_id, "tags": info.tags or []})
    except Exception as e:
        emit({"type": "error", "message": f"Could not fetch model info: {e}"})

    emit({"type": "status", "message": f"Downloading {args.repo_id} to {args.local_dir}..."})

    os.makedirs(args.local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
        )
        emit({"type": "done", "path": args.local_dir})
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
