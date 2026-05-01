"""
image_describe.py - One-shot image caption helper for My-LM UI.

Usage:
  python image_describe.py <image_path> [hint]

Outputs a single JSON line:
  {"ok": true, "caption": "...", "model": "..."}
or
  {"ok": false, "error": "..."}
"""

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def basic_visual_description(image, hint: str = "") -> str:
    width, height = image.size
    mode = image.mode
    orientation = "landscape" if width > height else "portrait" if height > width else "square"

    # Fast global tone estimate from a 1x1 downsample.
    r, g, b = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
    avg = (r + g + b) / 3.0
    if avg < 60:
        brightness = "very dark"
    elif avg < 110:
        brightness = "dark"
    elif avg < 170:
        brightness = "mid-tone"
    elif avg < 220:
        brightness = "bright"
    else:
        brightness = "very bright"

    dominant = "neutral"
    if r > g + 15 and r > b + 15:
        dominant = "red"
    elif g > r + 15 and g > b + 15:
        dominant = "green"
    elif b > r + 15 and b > g + 15:
        dominant = "blue"

    base = (
        f"A {orientation} image ({width}x{height}, mode {mode}) with "
        f"{brightness} lighting and mostly {dominant} tones."
    )
    if hint:
        return f"{base} User request context: {hint}"
    return base


def main() -> int:
    if len(sys.argv) < 2:
        emit({"ok": False, "error": "Usage: image_describe.py <image_path> [hint]"})
        return 1

    image_path = sys.argv[1]
    hint = " ".join(sys.argv[2:]).strip()

    if not os.path.isfile(image_path):
        emit({"ok": False, "error": f"Image not found: {image_path}"})
        return 1

    # Small, commonly available caption model. Downloads on first use.
    model_id = "Salesforce/blip-image-captioning-base"

    try:
        from PIL import Image
        import torch
        from transformers import AutoProcessor, BlipForConditionalGeneration

        image = Image.open(image_path).convert("RGB")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else None

            processor = AutoProcessor.from_pretrained(model_id)
            if torch_dtype is not None:
                model = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                )
            else:
                model = BlipForConditionalGeneration.from_pretrained(model_id)
            model.to(device)

            if hint:
                # Conditional captioning, guided by user prompt text.
                inputs = processor(images=image, text=hint, return_tensors="pt")
            else:
                inputs = processor(images=image, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=80)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if not text:
                raise RuntimeError("Caption model returned empty text")

            emit({"ok": True, "caption": text, "model": model_id})
            return 0
        except Exception as model_error:
            # Fallback keeps chat usable when model download/load is blocked.
            fallback = basic_visual_description(image, hint)
            emit(
                {
                    "ok": True,
                    "caption": fallback,
                    "model": "basic-vision-fallback",
                    "warning": str(model_error),
                }
            )
            return 0
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
