"""ADetailer-style face refinement pass.

Detects faces via YOLOv8, crops each face with padding, runs SDXL img2img on
the crop at a low denoising strength, and composites the refined crop back
into the original image with a feathered mask.

Reuses the base pipeline's components (no extra VRAM for a second SDXL model).
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageFilter


def load_face_detector(model_path: str | Path):
    """Lazy-load the YOLOv8 face detector."""
    from ultralytics import YOLO

    return YOLO(str(model_path))


def detect_faces(
    detector,
    image: Image.Image,
    conf: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """Return a list of (x1, y1, x2, y2) bounding boxes for detected faces."""
    results = detector.predict(image, conf=conf, verbose=False)
    boxes: list[tuple[int, int, int, int]] = []
    for r in results:
        if r.boxes is None:
            continue
        for xyxy in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2, y2))
    return boxes


def _expand_box(
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.4,
    square: bool = True,
) -> tuple[int, int, int, int]:
    """Expand a face bbox with padding and (optionally) make it square."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    if square:
        side = max(w, h) * (1 + pad_ratio)
        w = h = side
    else:
        w *= 1 + pad_ratio
        h *= 1 + pad_ratio

    nx1 = int(max(0, cx - w / 2))
    ny1 = int(max(0, cy - h / 2))
    nx2 = int(min(img_w, cx + w / 2))
    ny2 = int(min(img_h, cy + h / 2))
    return nx1, ny1, nx2, ny2


def _feathered_mask(size: tuple[int, int], feather: int = 24) -> Image.Image:
    """White rectangle with feathered (blurred) edges, for alpha compositing."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    # Inner solid region, leaving `feather` px on each edge
    inset = feather
    if w > 2 * inset and h > 2 * inset:
        solid = Image.new("L", (w - 2 * inset, h - 2 * inset), 255)
        mask.paste(solid, (inset, inset))
    else:
        mask = Image.new("L", (w, h), 255)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))


def _build_img2img_pipeline(base_pipe):
    """Build an img2img pipeline that shares weights with the base SDXL pipe."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        unet=base_pipe.unet,
        scheduler=base_pipe.scheduler,
    )
    return img2img


def refine_face(
    img2img_pipe,
    image: Image.Image,
    box: tuple[int, int, int, int],
    prompt: str,
    negative_prompt: str,
    denoise: float = 0.45,
    steps: int = 25,
    guidance: float = 6.0,
    crop_size: int = 1024,
    pad_ratio: float = 0.4,
    feather: int = 24,
    compel=None,
) -> Image.Image:
    """Refine a single face region in `image` using img2img.

    Returns a new image with the face region replaced by the refined version.
    """
    img_w, img_h = image.size
    x1, y1, x2, y2 = _expand_box(box, img_w, img_h, pad_ratio=pad_ratio)
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    if cw < 16 or ch < 16:
        return image  # too small, skip

    # Resize crop up to crop_size for better detail in the refinement pass
    resized = crop.resize((crop_size, crop_size), Image.LANCZOS)

    # Build a face-focused prompt by prepending detail cues
    face_prompt = (
        "detailed face, symmetric eyes, natural skin texture, "
        "sharp focus, realistic facial features, " + prompt
    )

    kwargs = dict(
        image=resized,
        strength=denoise,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )

    # Use compel for long prompts if provided; otherwise pass raw strings
    if compel is not None:
        try:
            cond_batch, pool_batch = compel([face_prompt, negative_prompt])
            kwargs["prompt_embeds"] = cond_batch[0:1]
            kwargs["pooled_prompt_embeds"] = pool_batch[0:1]
            kwargs["negative_prompt_embeds"] = cond_batch[1:2]
            kwargs["negative_pooled_prompt_embeds"] = pool_batch[1:2]
        except Exception:
            kwargs["prompt"] = face_prompt
            kwargs["negative_prompt"] = negative_prompt
    else:
        kwargs["prompt"] = face_prompt
        kwargs["negative_prompt"] = negative_prompt

    refined = img2img_pipe(**kwargs).images[0]
    refined_resized = refined.resize((cw, ch), Image.LANCZOS)

    # Composite back with a feathered mask to hide seams
    mask = _feathered_mask((cw, ch), feather=feather)
    out = image.copy()
    out.paste(refined_resized, (x1, y1), mask)
    return out


def run_face_detailer(
    image: Image.Image,
    base_pipe,
    detector_path: str | Path,
    prompt: str,
    negative_prompt: str,
    denoise: float = 0.45,
    steps: int = 25,
    guidance: float = 6.0,
    max_faces: int = 4,
    compel=None,
    on_progress: Optional[callable] = None,
) -> Image.Image:
    """High-level entry: detect faces, refine each one, return the new image."""
    detector = load_face_detector(detector_path)
    boxes = detect_faces(detector, image)
    if not boxes:
        if on_progress:
            on_progress({"type": "log", "text": "No faces detected, skipping detailer"})
        return image

    # Sort boxes by area (largest first) and cap to max_faces
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    boxes = boxes[:max_faces]

    img2img = _build_img2img_pipeline(base_pipe)

    out = image
    for i, box in enumerate(boxes, 1):
        if on_progress:
            on_progress(
                {"type": "log", "text": f"Refining face {i}/{len(boxes)}..."}
            )
        out = refine_face(
            img2img,
            out,
            box,
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            steps=steps,
            guidance=guidance,
            compel=compel,
        )
        torch.cuda.empty_cache()

    # img2img pipe shares weights with base_pipe, no need to delete unet/vae
    del img2img
    torch.cuda.empty_cache()
    return out
