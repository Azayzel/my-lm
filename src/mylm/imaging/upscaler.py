"""4x-UltraSharp upscaler using spandrel (ESRGAN-family loader)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader


def load_upscaler(model_path: str | Path, device: str = "cuda") -> ImageModelDescriptor:
    """Load an ESRGAN-style upscaler (e.g., 4x-UltraSharp)."""
    model = ModelLoader().load_from_file(str(model_path))
    assert isinstance(model, ImageModelDescriptor), "Expected a single-image model"
    model.to(device).eval()
    return model


@torch.inference_mode()
def upscale_image(
    model: ImageModelDescriptor,
    image: Image.Image,
    device: str = "cuda",
    tile: int = 512,
    tile_overlap: int = 32,
) -> Image.Image:
    """Upscale a PIL image using a spandrel model, tiled to fit VRAM.

    Tiling processes the image in patches with overlap, so a 1024x1024 base
    can be 4x'd to 4096x4096 without OOMing on 6 GB VRAM.
    """
    img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    scale = model.scale
    _, _, h, w = tensor.shape
    out = torch.zeros((1, 3, h * scale, w * scale), device=device, dtype=tensor.dtype)
    weight = torch.zeros_like(out)

    stride = tile - tile_overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            patch = tensor[:, :, y0:y1, x0:x1]
            up = model(patch).clamp(0, 1)
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += up
            weight[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += 1

    out = out / weight.clamp(min=1)
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)
