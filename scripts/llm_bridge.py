"""
llm_bridge.py - Streaming LLM inference bridge for My UI.

Reads a JSON request from stdin, streams tokens to stdout as newline-delimited JSON.
Each line is one of:
  {"type": "token", "text": "..."}
  {"type": "done"}
  {"type": "error", "message": "..."}

Usage:
  python llm_bridge.py <model_path>
  then write JSON to stdin: {"messages": [...], "max_tokens": 512, "system": "..."}
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

        # Normalize various UI/content shapes to plain text for text-only models.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    # Common chat format: {"type": "text", "text": "..."}
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


def build_plain_prompt(messages):
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _is_peft_adapter(model_path: str) -> bool:
    """True if the path contains a PEFT adapter (not a full model)."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def _detect_device_map():
    """Pick the best device_map for loading: GPU-only if CUDA available, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return {"": 0}
    except ImportError:
        pass
    return "auto"


def load_model(model_path: str):
    import threading

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    dmap = _detect_device_map()
    dtype = torch.float16 if dmap != "auto" else "auto"

    if _is_peft_adapter(model_path):
        import json

        from peft import PeftModel

        with open(os.path.join(model_path, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        base_path = adapter_cfg.get("base_model_name_or_path", "")
        if not base_path or not os.path.isdir(base_path):
            raise FileNotFoundError(
                f"Adapter base model not found: {base_path!r} "
                f"(from {model_path}/adapter_config.json)"
            )
        emit({"type": "info", "message": f"Loading PEFT adapter on base {base_path}"})
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=dtype,
            device_map=dmap,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=dmap,
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, TextIteratorStreamer, threading


def run_inference(model, tokenizer, TextIteratorStreamer, threading, request: dict):
    messages = normalize_messages(request.get("messages", []))
    max_tokens = request.get("max_tokens", 512)
    system_prompt = request.get("system", "You are My-LM, a helpful AI assistant.")

    # Prepend system message if not already present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Build the prompt text using chat template, fall back to plain format.
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(text, str):
            text = str(text)
    except Exception:
        text = build_plain_prompt(messages)

    # Tokenize and move to model device
    tokenized = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in tokenized.items()}

    # Log input shape for diagnostics
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else "?"
    emit({"type": "info", "message": f"Input tokens: {input_len}, max_new_tokens: {max_tokens}"})

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "streamer": streamer,
        "do_sample": True,
        "temperature": request.get("temperature", 0.7),
        "top_p": request.get("top_p", 0.9),
    }

    gen_error = {"message": None}

    def _generate_wrapper():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            import traceback
            gen_error["message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            # Ensure the streamer loop can terminate even if generation fails.
            try:
                streamer.end()
            except Exception:
                pass

    thread = threading.Thread(target=_generate_wrapper, daemon=True)
    thread.start()

    token_count = 0
    for token_text in streamer:
        emit({"type": "token", "text": token_text})
        token_count += 1

    # Wait briefly for the generation thread to settle after streamer end.
    thread.join(timeout=1.0)
    if gen_error["message"]:
        emit({"type": "error", "message": gen_error["message"]})
    else:
        emit({"type": "info", "message": f"Generation complete: {token_count} tokens"})
        emit({"type": "done"})


def main():
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "Usage: llm_bridge.py <model_path>"})
        sys.exit(1)

    model_path = sys.argv[1]

    if os.path.isfile(model_path):
        emit(
            {
                "type": "error",
                "message": (
                    "Selected model path points to a file, but this LLM runtime expects "
                    "a Hugging Face model directory (with config/tokenizer/weights)."
                ),
            }
        )
        sys.exit(1)

    if not os.path.isdir(model_path):
        emit({"type": "error", "message": f"Model path not found: {model_path}"})
        sys.exit(1)

    emit({"type": "status", "message": "Loading model..."})

    try:
        model, tokenizer, TextIteratorStreamer, threading = load_model(model_path)
        emit({"type": "ready"})
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load model: {e}"})
        sys.exit(1)

    # Main request loop - read one JSON request per line from stdin
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
            run_inference(model, tokenizer, TextIteratorStreamer, threading, request)
        except Exception as e:
            emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    main()
