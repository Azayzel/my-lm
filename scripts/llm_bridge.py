"""
llm_bridge.py - Streaming LLM inference bridge for Lavely UI.

Reads a JSON request from stdin, streams tokens to stdout as newline-delimited JSON.
Each line is one of:
  {"type": "token", "text": "..."}
  {"type": "done"}
  {"type": "error", "message": "..."}

Usage:
  python llm_bridge.py <model_path>
  then write JSON to stdin: {"messages": [...], "max_tokens": 512, "system": "..."}
"""

import sys
import io
import json
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import threading

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer, TextIteratorStreamer, threading


def run_inference(model, tokenizer, TextIteratorStreamer, threading, request: dict):
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 512)
    system_prompt = request.get("system", "You are Lavely-LM, a helpful AI assistant.")

    # Prepend system message if not already present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

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

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        emit({"type": "token", "text": token_text})

    thread.join()
    emit({"type": "done"})


def main():
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "Usage: llm_bridge.py <model_path>"})
        sys.exit(1)

    model_path = sys.argv[1]

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
