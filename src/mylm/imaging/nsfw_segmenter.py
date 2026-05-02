"""NSFW region segmentation using YOLOv8 models.

Loads YOLOv8 segmentation checkpoints from NSFW-API/NSFW_Segmentation to
detect and mask NSFW regions in images. Each checkpoint targets a specific
body region (breast, penis, vagina) in small (-s) or extra-large (-x) variants.
"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Model filename stems and the region they detect
REGION_MODELS = {
    "breast": "nsfw-seg-breast",
    "penis": "nsfw-seg-penis",
    "vagina": "nsfw-seg-vagina",
}


def load_segmenters(
    model_dir: str | Path,
    variant: str = "x",
) -> dict[str, Any]:
    """Load NSFW segmentation models from a directory.

    Args:
        model_dir: Directory containing the .pt checkpoint files.
        variant: ``"s"`` for small (fast, ~20 MB each) or ``"x"`` for
                 extra-large (accurate, ~120 MB each).

    Returns:
        Dict mapping region name to loaded YOLO model.
    """
    from ultralytics import YOLO

    model_dir = Path(model_dir)
    models: dict[str, Any] = {}
    for region, stem in REGION_MODELS.items():
        path = model_dir / f"{stem}-{variant}.pt"
        if path.exists():
            models[region] = YOLO(str(path))
    return models


def segment_nsfw(
    models: dict[str, Any],
    image: Image.Image,
    conf: float = 0.3,
) -> dict[str, Image.Image]:
    """Run all loaded segmentation models and return per-region binary masks.

    Args:
        models: Dict from :func:`load_segmenters`.
        image: Input PIL image.
        conf: Confidence threshold for detections.

    Returns:
        Dict mapping region name to a binary ``"L"`` mode PIL mask
        (255 = detected region, 0 = background). Only regions with
        at least one detection are included.
    """
    w, h = image.size
    masks: dict[str, Image.Image] = {}

    for region, model in models.items():
        results = model.predict(image, conf=conf, verbose=False)
        combined = np.zeros((h, w), dtype=np.uint8)

        for r in results:
            if r.masks is None:
                continue
            for mask_data in r.masks.data:
                mask_np = mask_data.cpu().numpy()
                # Resize mask to image dimensions (model may output different size)
                from PIL import Image as _Img

                mask_pil = _Img.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                combined = np.maximum(combined, np.array(mask_pil))

        if combined.max() > 0:
            masks[region] = Image.fromarray(combined, mode="L")

    return masks


def get_combined_mask(
    masks: dict[str, Image.Image],
    regions: list[str] | None = None,
) -> Image.Image | None:
    """Merge per-region masks into a single binary mask.

    Args:
        masks: Output of :func:`segment_nsfw`.
        regions: Region names to include. ``None`` means all.

    Returns:
        Combined ``"L"`` mode mask, or ``None`` if no detections.
    """
    selected = {k: v for k, v in masks.items() if regions is None or k in regions}
    if not selected:
        return None

    arrays = [np.array(m) for m in selected.values()]
    combined = arrays[0]
    for a in arrays[1:]:
        combined = np.maximum(combined, a)

    return Image.fromarray(combined, mode="L")
