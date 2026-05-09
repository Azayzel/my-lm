"""
image_bridge.py - Image generation bridge for My UI.

Reads a JSON request from stdin, streams progress to stdout as newline-delimited JSON.
Each line is one of:
  {"type": "progress", "step": N, "total": N}
  {"type": "done", "path": "...absolute path to saved image..."}
  {"type": "error", "message": "..."}

Usage:
  python image_bridge.py <model_dir> <upscaler_path> <output_dir> [face_detector_path]
  then write JSON to stdin:
  {
    "prompt": "...",
    "negative_prompt": "...",
    "steps": 22,
    "guidance": 7.0,
    "width": 1024,
    "height": 1024,
    "upscale": true,
    "face_fix": true,
    "face_fix_strength": 0.45,
    "filename": "optional_name"
  }
"""

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Allow running without `pip install -e .` by adding src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def _load_taesd_preview(model_dir: str = "models/taesdxl"):
    """Load TAESDXL for fast latent->RGB previews during diffusion steps.

    Returns (tiny_vae, None) if available, or (None, None) if loading fails.
    """
    from pathlib import Path

    import torch
    from diffusers import AutoencoderTiny

    base = Path(__file__).resolve().parent.parent
    candidates = [
        Path(model_dir),
        base / model_dir,
        base / "models" / "taesdxl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    try:
        tiny = AutoencoderTiny.from_pretrained(str(path), torch_dtype=torch.float16)
        tiny.to("cuda")
        tiny.eval()
        return tiny
    except Exception:
        return None


def _latent_to_preview_b64(tiny_vae, latents) -> str | None:
    """Decode latents via TAESDXL and return a base64 PNG string."""
    import base64
    import io

    import torch
    from PIL import Image

    try:
        with torch.inference_mode():
            decoded = tiny_vae.decode(latents.to(tiny_vae.dtype)).sample
        img = decoded[0].float().clamp(-1, 1)
        img = (img + 1) / 2  # [0, 1]
        img = (img * 255).byte().cpu().numpy().transpose(1, 2, 0)
        pil = Image.fromarray(img)
        # Downscale for smaller payloads
        pil.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _patch_compel_multi_empty_z():
    """Compel 2.3.1 has a bug: EmbeddingsProviderMulti is missing `empty_z`,
    which is required for internal padding when positive/negative prompts
    differ in length. Patch by aggregating empty_z from the inner providers."""
    import torch
    from compel.embeddings_provider import EmbeddingsProviderMulti

    if hasattr(EmbeddingsProviderMulti, "empty_z"):
        return

    def empty_z(self):
        zs = [p.empty_z for p in self.embedding_providers]
        if self.concat_along_embedding_dim:
            return torch.cat(zs, dim=-1)
        return zs[0]

    EmbeddingsProviderMulti.empty_z = property(empty_z)


def load_pipeline(model_dir: str):
    import warnings

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    # Suppress compel's deprecation warning — it prints to stdout which
    # breaks our JSON-over-stdout protocol.
    warnings.filterwarnings("ignore", message=".*deprecated.*CompelFor.*")
    warnings.filterwarnings("ignore", message=".*multiple tokenizers.*")
    from compel import Compel, ReturnedEmbeddingsType

    _patch_compel_multi_empty_z()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")

    # Compel prints a deprecation warning directly to stdout when given
    # multiple tokenizers. Since we use stdout for JSON protocol, redirect
    # stdout to stderr during construction to avoid poisoning the stream.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
        device="cuda",
    )
    sys.stdout = real_stdout
    return pipe, compel


def make_progress_callback(total_steps: int, tiny_vae=None, preview_every: int = 2):
    def callback(pipe, step_index, timestep, callback_kwargs):
        step = step_index + 1
        payload = {"type": "progress", "step": step, "total": total_steps}
        # Decode and send a preview every `preview_every` steps (and on last step)
        if tiny_vae is not None and (step % preview_every == 0 or step == total_steps):
            latents = callback_kwargs.get("latents")
            if latents is not None:
                b64 = _latent_to_preview_b64(tiny_vae, latents)
                if b64:
                    payload["preview"] = b64
        emit(payload)
        return callback_kwargs
    return callback


def run_generation(
    pipe,
    compel,
    request: dict,
    upscaler_path: str,
    output_dir: str,
    face_detector_path: str = "",
    tiny_vae=None,
    nsfw_seg_model_dir: str = "",
):
    import torch

    from mylm.imaging import load_upscaler, upscale_image

    prompt = request.get("prompt", "")
    negative_prompt = request.get(
        "negative_prompt",
        "face asymmetry, eyes asymmetry, deformed eyes, open mouth, cartoon, painting, blurry, low quality",
    )
    steps = request.get("steps", 22)
    guidance = request.get("guidance", 7.0)
    width = request.get("width", 1024)
    height = request.get("height", 1024)
    do_upscale = request.get("upscale", True)
    do_face_fix = request.get("face_fix", False)
    face_fix_strength = float(request.get("face_fix_strength", 0.45))
    do_face_swap = request.get("face_swap", False)
    face_swap_source = request.get("face_swap_source", "")  # path to source face image
    do_nsfw_seg = request.get("nsfw_seg", False)
    filename = request.get("filename")

    # Pass both prompts as a list — compel batches them and auto-pads to the
    # same token length, which is required when truncate_long_prompts=False.
    conditioning_batch, pooled_batch = compel([prompt, negative_prompt])
    conditioning = conditioning_batch[0:1]
    negative_conditioning = conditioning_batch[1:2]
    pooled = pooled_batch[0:1]
    negative_pooled = pooled_batch[1:2]

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=negative_conditioning,
        negative_pooled_prompt_embeds=negative_pooled,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        callback_on_step_end=make_progress_callback(steps, tiny_vae=tiny_vae),
    ).images[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename if filename else f"image_{ts}"
    out_path = Path(output_dir) / f"{base_name}.png"
    image.save(out_path)

    result_path = str(out_path)

    # Face detailer pass — detect faces and refine each with img2img
    if do_face_fix and face_detector_path and os.path.isfile(face_detector_path):
        try:
            emit({"type": "status", "message": "Running face detailer..."})
            # Text encoders need to be on GPU for the img2img prompt encoding
            pipe.text_encoder.to("cuda")
            pipe.text_encoder_2.to("cuda")
            from mylm.imaging import run_face_detailer

            image = run_face_detailer(
                image=image,
                base_pipe=pipe,
                detector_path=face_detector_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                denoise=face_fix_strength,
                steps=max(20, steps),
                guidance=max(5.0, guidance - 1.5),
                compel=compel,
                on_progress=lambda m: emit(m),
            )
            # Save the face-fixed version
            fx_path = Path(output_dir) / f"{base_name}_face.png"
            image.save(fx_path)
            result_path = str(fx_path)
            torch.cuda.empty_cache()
        except Exception as e:
            emit({"type": "log", "text": f"Face detailer failed: {e}"})

    # Face swap pass — replace generated faces with the user's uploaded face
    if do_face_swap and face_swap_source and os.path.isfile(face_swap_source):
        try:
            emit({"type": "status", "message": "Swapping face..."})
            from PIL import Image as PILImage

            from face_swap import load_swapper, swap_face

            swap_model_path = Path(__file__).resolve().parent.parent / "models" / "face_swap" / "inswapper_128.onnx"
            if swap_model_path.exists():
                swapper, analyser = load_swapper(str(swap_model_path))
                source_face = PILImage.open(face_swap_source).convert("RGB")
                image = swap_face(
                    swapper,
                    analyser,
                    source_image=source_face,
                    target_image=image,
                    on_progress=lambda m: emit(m),
                )
                sw_path = Path(output_dir) / f"{base_name}_swap.png"
                image.save(sw_path)
                result_path = str(sw_path)
                del swapper, analyser, source_face
            else:
                emit({"type": "log", "text": f"inswapper_128.onnx not found at {swap_model_path}"})
        except Exception as e:
            emit({"type": "log", "text": f"Face swap failed: {e}"})

    if do_upscale and os.path.isfile(upscaler_path):
        try:
            emit({"type": "status", "message": "Upscaling with 4x-UltraSharp..."})
            torch.cuda.empty_cache()
            upscaler = load_upscaler(upscaler_path, device="cuda")
            upscaled = upscale_image(
                upscaler, image, device="cuda", tile=384, tile_overlap=32
            )
            up_path = Path(output_dir) / f"{base_name}_4x.png"
            upscaled.save(up_path)
            result_path = str(up_path)
            del upscaler, upscaled
            torch.cuda.empty_cache()
        except Exception as e:
            emit({"type": "log", "text": f"Upscale failed: {e}"})

    # NSFW segmentation pass — detect regions and save masks
    if do_nsfw_seg and nsfw_seg_model_dir and os.path.isdir(nsfw_seg_model_dir):
        try:
            emit({"type": "status", "message": "Running NSFW segmentation..."})
            from mylm.imaging import load_segmenters, segment_nsfw

            if not hasattr(run_generation, "_nsfw_models"):
                run_generation._nsfw_models = load_segmenters(nsfw_seg_model_dir, variant="x")
            masks = segment_nsfw(run_generation._nsfw_models, image)
            mask_paths = {}
            for region, mask in masks.items():
                mask_path = Path(output_dir) / f"{base_name}_mask_{region}.png"
                mask.save(mask_path)
                mask_paths[region] = str(mask_path)
            emit({"type": "masks", "regions": mask_paths})
            torch.cuda.empty_cache()
        except Exception as e:
            emit({"type": "log", "text": f"NSFW segmentation failed: {e}"})

    # Always emit `done` with whatever path we have (base / face-fixed /
    # upscaled), BEFORE doing cleanup that might fail. That way the UI
    # always hears about completion.
    emit({"type": "done", "path": result_path})

    # ── Cleanup for next generation ────────────────────────────────────
    try:
        del conditioning, pooled, negative_conditioning, negative_pooled
        torch.cuda.empty_cache()
    except Exception as e:
        emit({"type": "log", "text": f"Cleanup warning: {e}"})

    try:
        # enable_model_cpu_offload() attaches accelerate hooks that are
        # hard to remove. Moving components manually can fail on the second
        # generation, so we just swallow errors here.
        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
    except Exception as e:
        emit({"type": "log", "text": f"Post-gen text-encoder move warning: {e}"})


def main():
    if len(sys.argv) < 4:
        emit({"type": "error", "message": "Usage: image_bridge.py <model_dir> <upscaler_path> <output_dir> [face_detector_path]"})
        sys.exit(1)

    model_dir = sys.argv[1]
    upscaler_path = sys.argv[2]
    output_dir = sys.argv[3]
    face_detector_path = sys.argv[4] if len(sys.argv) > 4 else ""
    nsfw_seg_model_dir = sys.argv[5] if len(sys.argv) > 5 else ""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emit({"type": "status", "message": "Loading image model..."})

    try:
        pipe, compel = load_pipeline(model_dir)
        tiny_vae = _load_taesd_preview()
        if tiny_vae is None:
            emit({"type": "log", "text": "TAESDXL not found — streaming previews disabled"})
        emit({"type": "ready"})
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load pipeline: {e}"})
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
            run_generation(
                pipe,
                compel,
                request,
                upscaler_path,
                output_dir,
                face_detector_path,
                tiny_vae=tiny_vae,
                nsfw_seg_model_dir=nsfw_seg_model_dir,
            )
        except Exception as e:
            emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    main()
