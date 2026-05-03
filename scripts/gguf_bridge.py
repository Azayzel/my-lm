"""
gguf_bridge.py - Streaming GGUF inference bridge for My UI.

Reads JSON requests from stdin and streams tokens as newline-delimited JSON.
Usage:
  python gguf_bridge.py <model_path.gguf>
"""

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def normalize_messages(messages):
    normalized = []
    for m in messages or []:
        role = str((m or {}).get("role", "user")).lower().strip()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = (m or {}).get("content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            content = "\n".join([p for p in parts if p])
        elif content is None:
            content = ""
        else:
            content = str(content)

        normalized.append({"role": role, "content": content})
    return normalized


def load_model(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError(
            "GGUF runtime requires llama-cpp-python. Install with: pip install llama-cpp-python"
        )

    n_ctx = int(os.environ.get("MYLM_GGUF_CTX", "4096"))
    n_threads = int(os.environ.get("MYLM_GGUF_THREADS", "0"))

    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    if n_threads > 0:
        kwargs["n_threads"] = n_threads

    # Prefer GPU offload when the local llama.cpp build supports it.
    kwargs["n_gpu_layers"] = -1

    try:
        llm = Llama(**kwargs)
    except TypeError:
        # Older llama-cpp-python versions may not accept some kwargs.
        kwargs.pop("n_gpu_layers", None)
        llm = Llama(**kwargs)

    return llm


def run_inference(llm, request: dict):
    messages = normalize_messages(request.get("messages", []))
    system_prompt = request.get("system", "You are My-LM, a helpful AI assistant.")
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    max_tokens = int(request.get("max_tokens", 512))
    temperature = float(request.get("temperature", 0.7))
    top_p = float(request.get("top_p", 0.9))

    emit({"type": "info", "message": f"max_new_tokens: {max_tokens}"})

    token_count = 0
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    for chunk in stream:
        try:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
        except Exception:
            text = ""

        if text:
            emit({"type": "token", "text": text})
            token_count += 1

    emit({"type": "info", "message": f"Generation complete: {token_count} chunks"})
    emit({"type": "done"})


def main():
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "Usage: gguf_bridge.py <model_path.gguf>"})
        sys.exit(1)

    model_path = os.path.abspath(sys.argv[1])

    if not os.path.isfile(model_path):
        emit({"type": "error", "message": f"GGUF model file not found: {model_path}"})
        sys.exit(1)

    if not model_path.lower().endswith(".gguf"):
        emit({"type": "error", "message": f"Expected a .gguf file path, got: {model_path}"})
        sys.exit(1)

    if os.path.basename(model_path).lower().startswith("mmproj"):
        emit(
            {
                "type": "error",
                "message": (
                    "Selected file looks like an mmproj projector file, not a runnable text model. "
                    "Choose a GGUF chat model file instead."
                ),
            }
        )
        sys.exit(1)

    emit({"type": "status", "message": "Loading GGUF model..."})

    try:
        llm = load_model(model_path)
        emit({"type": "ready"})
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load GGUF model: {e}"})
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"Invalid JSON: {e}"})
            continue

        try:
            run_inference(llm, request)
        except Exception as e:
            import traceback

            emit({"type": "error", "message": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
