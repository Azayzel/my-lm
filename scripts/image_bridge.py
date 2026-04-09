"""
image_bridge.py - Image generation bridge for Lavely UI.

Reads a JSON request from stdin, streams progress to stdout as newline-delimited JSON.
Each line is one of:
  {"type": "progress", "step": N, "total": N}
  {"type": "done", "path": "...absolute path to saved image..."}
  {"type": "error", "message": "..."}

Usage:
  python image_bridge.py <model_dir> <upscaler_path> <output_dir>
  then write JSON to stdin:
  {
    "prompt": "...",
    "negative_prompt": "...",
    "steps": 22,
    "guidance": 7.0,
    "width": 1024,
    "height": 1024,
    "upscale": true,
    "filename": "optional_name"
  }
"""

import sys
import io
import json
import os
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Allow importing upscaler from scripts dir
sys.path.insert(0, str(Path(__file__).resolve().parent))


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


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
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
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

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
        device="cuda",
    )
    return pipe, compel


def make_progress_callback(total_steps: int):
    def callback(pipe, step_index, timestep, callback_kwargs):
        emit({"type": "progress", "step": step_index + 1, "total": total_steps})
        return callback_kwargs
    return callback


def run_generation(pipe, compel, request: dict, upscaler_path: str, output_dir: str):
    import torch
    from upscaler import load_upscaler, upscale_image

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
    filename = request.get("filename", None)

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
        callback_on_step_end=make_progress_callback(steps),
    ).images[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename if filename else f"image_{ts}"
    out_path = Path(output_dir) / f"{base_name}.png"
    image.save(out_path)

    result_path = str(out_path)

    if do_upscale and os.path.isfile(upscaler_path):
        emit({"type": "progress", "step": steps, "total": steps, "message": "Upscaling..."})
        torch.cuda.empty_cache()
        # Re-enable text encoders on CPU for next generation
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")

        upscaler = load_upscaler(upscaler_path, device="cuda")
        upscaled = upscale_image(upscaler, image, device="cuda", tile=384, tile_overlap=32)
        up_path = Path(output_dir) / f"{base_name}_4x.png"
        upscaled.save(up_path)
        result_path = str(up_path)
        del upscaler, upscaled
        torch.cuda.empty_cache()

    del conditioning, pooled, negative_conditioning, negative_pooled
    torch.cuda.empty_cache()

    # Re-enable text encoders for next run
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    # Disable cpu offload for next run (will be re-enabled)
    pipe.reset_device_map() if hasattr(pipe, "reset_device_map") else None

    emit({"type": "done", "path": result_path})


def main():
    if len(sys.argv) < 4:
        emit({"type": "error", "message": "Usage: image_bridge.py <model_dir> <upscaler_path> <output_dir>"})
        sys.exit(1)

    model_dir = sys.argv[1]
    upscaler_path = sys.argv[2]
    output_dir = sys.argv[3]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emit({"type": "status", "message": "Loading image model..."})

    try:
        pipe, compel = load_pipeline(model_dir)
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
            run_generation(pipe, compel, request, upscaler_path, output_dir)
        except Exception as e:
            emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    main()
