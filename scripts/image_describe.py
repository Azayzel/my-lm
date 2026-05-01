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
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def basic_visual_description(image, hint: str = "") -> str:
    width, height = image.size
    mode = image.mode
    orientation = (
        "landscape" if width > height else "portrait" if height > width else "square"
    )

    # PIL-only defaults in case optional dependencies are unavailable.
    rgb = image.convert("RGB")
    r, g, b = rgb.resize((1, 1)).getpixel((1 - 1, 1 - 1))
    luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    contrast = 0.0
    saturation = 0.0
    edge_density = 0.0
    sharpness = 0.0
    faces = []

    try:
        import cv2
        import numpy as np

        arr = np.array(rgb)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        luminance = float(gray.mean())
        contrast = float(gray.std())
        saturation = float(hsv[:, :, 1].mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        edges = cv2.Canny(gray, 80, 180)
        edge_density = float((edges > 0).mean())

        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml",
        )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            faces = list(detected) if detected is not None else []
    except Exception:
        # Keep fallback robust even if cv2/numpy are unavailable.
        pass

    if luminance < 60:
        brightness = "very dark"
    elif luminance < 105:
        brightness = "dark"
    elif luminance < 155:
        brightness = "balanced"
    elif luminance < 205:
        brightness = "bright"
    else:
        brightness = "very bright"

    if contrast < 35:
        contrast_label = "low contrast"
    elif contrast < 60:
        contrast_label = "moderate contrast"
    else:
        contrast_label = "high contrast"

    if sharpness < 80:
        focus_label = "soft focus"
    elif sharpness < 220:
        focus_label = "moderate focus"
    else:
        focus_label = "sharp focus"

    if edge_density < 0.05:
        complexity = "clean background"
    elif edge_density < 0.12:
        complexity = "moderately detailed background"
    else:
        complexity = "busy background"

    if saturation < 55:
        color_intensity = "muted color grading"
    elif saturation < 100:
        color_intensity = "natural color grading"
    else:
        color_intensity = "vivid color grading"

    dominant = "neutral"
    if r > g + 12 and r > b + 12:
        dominant = "warm red"
    elif g > r + 12 and g > b + 12:
        dominant = "green"
    elif b > r + 12 and b > g + 12:
        dominant = "blue"

    framing = "scene-oriented composition"
    subject_hint = ""
    face_ratio = 0.0
    x_center = 0.5
    if faces:
        face_areas = [int(wf) * int(hf) for (_x, _y, wf, hf) in faces]
        largest_idx = max(range(len(face_areas)), key=lambda i: face_areas[i])
        x, y, wf, hf = faces[largest_idx]
        face_ratio = (int(wf) * int(hf)) / float(width * height)
        x_center = (int(x) + (int(wf) / 2.0)) / float(width)

        if face_ratio > 0.22:
            framing = "tight portrait framing"
            subject_hint = "close head-and-shoulders emphasis"
        elif face_ratio > 0.10:
            framing = "medium portrait framing"
            subject_hint = "upper-body emphasis"
        else:
            framing = "wide portrait framing"
            subject_hint = "subject appears farther from camera"

        if x_center < 0.38:
            subject_hint = f"{subject_hint}, subject biased left"
        elif x_center > 0.62:
            subject_hint = f"{subject_hint}, subject biased right"
        else:
            subject_hint = f"{subject_hint}, subject near center"

        if len(faces) > 1:
            subject_hint = f"{subject_hint}, multiple faces detected ({len(faces)})"

    # Estimate body-position wording from framing and face/body cues.
    if face_ratio > 0.24:
        body_scale = "tight head-and-shoulders framing"
    elif face_ratio > 0.12:
        body_scale = "upper-body framing"
    elif face_ratio > 0.06:
        body_scale = "half-body framing"
    elif face_ratio > 0:
        body_scale = "full-body-or-wide portrait framing"
    else:
        body_scale = "subject-scale unclear"

    horizontal_position = "centered"
    if x_center < 0.38:
        horizontal_position = "left-weighted"
    elif x_center > 0.62:
        horizontal_position = "right-weighted"

    pose_tokens = []
    if "portrait" in framing or orientation == "portrait":
        pose_tokens.append("portrait stance")
    else:
        pose_tokens.append("scene stance")
    pose_tokens.append(horizontal_position)
    pose_tokens.append(body_scale)

    # If user intent asks for pose/body details, intensify wording.
    wants_pose_focus = bool(
        re.search(r"body|pose|position|posture|stance", hint, re.IGNORECASE)
    )
    pose_intensity = "high" if wants_pose_focus else "medium"
    body_position_line = (
        "Body position emphasis: "
        f"{', '.join(pose_tokens)} with {pose_intensity} pose focus."
    )
    weighted_pose_hint = (
        "Pose-weight hint: "
        f"({', '.join(pose_tokens)}:{'1.85' if wants_pose_focus else '1.65'})."
    )

    parts = [
        body_position_line,
        weighted_pose_hint,
        f"{orientation.capitalize()} composition ({width}x{height}, {mode}).",
        (
            f"{framing} with {brightness} lighting, {contrast_label}, "
            f"{focus_label}, and {complexity}."
        ),
        f"Color profile trends {dominant} with {color_intensity}.",
    ]
    if subject_hint:
        parts.append(f"Subject/framing cues: {subject_hint}.")
    if hint:
        parts.append(f"User intent: {hint}.")
    parts.append(
        "Prompt hint: prioritize explicit pose/action terms, limb placement, and camera angle details for stronger body positioning."
    )
    return " ".join(parts)


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
        except Exception:
            # Fallback keeps chat usable when model download/load is blocked.
            fallback = basic_visual_description(image, hint)
            emit(
                {
                    "ok": True,
                    "caption": fallback,
                    "model": "basic-vision-fallback",
                }
            )
            return 0
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
