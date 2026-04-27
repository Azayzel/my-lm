"""Face swap using InsightFace + inswapper_128 ONNX model.

Given a source face image (the user's upload) and a target image (the generated
output), detects the face in the source and swaps it onto every face detected in
the target image.

Usage as a library:
    from face_swap import load_swapper, swap_face
    swapper, analyser = load_swapper("models/face_swap/inswapper_128.onnx")
    result = swap_face(swapper, analyser, source_pil, target_pil)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_swapper(
    model_path: str | Path,
    device: str = "cpu",
) -> tuple:
    """Load the InsightFace analyser and the inswapper model.

    Returns (swapper_model, face_analyser).
    Uses CPU by default since the ONNX model is small and fast.
    """
    import insightface
    from insightface.app import FaceAnalysis

    ctx_id = 0 if device == "cuda" else -1

    analyser = FaceAnalysis(name="buffalo_l")
    analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    return swapper, analyser


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGBA/RGB → OpenCV BGR."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR → PIL RGB."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def swap_face(
    swapper,
    analyser,
    source_image: Image.Image,
    target_image: Image.Image,
    max_faces: int = 4,
    on_progress: callable | None = None,
) -> Image.Image:
    """Swap the face from `source_image` onto every face in `target_image`.

    Args:
        swapper:       The inswapper ONNX model (from load_swapper)
        analyser:      InsightFace FaceAnalysis instance (from load_swapper)
        source_image:  PIL image containing the reference face
        target_image:  PIL image to swap faces into
        max_faces:     Maximum number of faces to swap in the target
        on_progress:   Optional callback for status messages

    Returns:
        A new PIL image with faces swapped.
    """
    src_cv = _pil_to_cv2(source_image)
    tgt_cv = _pil_to_cv2(target_image)

    # Detect the face in the source (reference) image
    if on_progress:
        on_progress({"type": "log", "text": "Detecting source face..."})
    src_faces = analyser.get(src_cv)
    if not src_faces:
        if on_progress:
            on_progress(
                {"type": "log", "text": "No face found in the uploaded source image"}
            )
        return target_image

    # Use the largest face in the source
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # Detect faces in the target (generated) image
    if on_progress:
        on_progress({"type": "log", "text": "Detecting target faces..."})
    tgt_faces = analyser.get(tgt_cv)
    if not tgt_faces:
        if on_progress:
            on_progress(
                {"type": "log", "text": "No faces found in the generated image"}
            )
        return target_image

    # Sort by face area (largest first) and cap
    tgt_faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    tgt_faces = tgt_faces[:max_faces]

    # Swap each target face with the source face
    result = tgt_cv.copy()
    for i, tgt_face in enumerate(tgt_faces, 1):
        if on_progress:
            on_progress(
                {"type": "log", "text": f"Swapping face {i}/{len(tgt_faces)}..."}
            )
        result = swapper.get(result, tgt_face, src_face, paste_back=True)

    return _cv2_to_pil(result)
