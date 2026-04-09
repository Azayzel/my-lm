"""RealVisXL V4.0 image generation with 8-bit / CPU offload for 6 GB VRAM."""

import sys
from datetime import datetime
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from compel import Compel, ReturnedEmbeddingsType
from compel.embeddings_provider import EmbeddingsProviderMulti

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upscaler import load_upscaler, upscale_image

# Compel 2.3.1 bug: EmbeddingsProviderMulti (used for SDXL) is missing `empty_z`,
# which breaks internal padding when positive/negative prompts differ in length.
# Aggregate empty_z from the inner providers.
if not hasattr(EmbeddingsProviderMulti, "empty_z"):
    def _multi_empty_z(self):
        zs = [p.empty_z for p in self.embedding_providers]
        if self.concat_along_embedding_dim:
            return torch.cat(zs, dim=-1)
        return zs[0]
    EmbeddingsProviderMulti.empty_z = property(_multi_empty_z)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "realvisxl-v4"
UPSCALER_PATH = ROOT / "models" / "upscalers" / "4x-UltraSharp.pth"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=torch.float16,
    variant="fp16",  # use fp16 weights if available
    use_safetensors=True,
)

# Use recommended scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)

# Move text encoders to GPU for compel (small enough to fit alongside VAE, etc.)
pipe.text_encoder.to("cuda")
pipe.text_encoder_2.to("cuda")

# Compel: handles prompts longer than the 77-token CLIP limit by chunking
# and averaging embeddings. SDXL has two text encoders, so we pass both.
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    truncate_long_prompts=False,
    device="cuda",
)

prompt = "A portrait of an adorable cat in the Artemis II Passing around the moon, 8k, photorealistic, soft lighting, high detail"
negative_prompt = (
    "face asymmetry, eyes asymmetry, deformed eyes, open mouth, cartoon, painting"
)

# Build embeddings via compel.
# Passing both prompts as a list batches them and auto-pads to the same length,
# which is required when truncate_long_prompts=False.
conditioning_batch, pooled_batch = compel([prompt, negative_prompt])
conditioning = conditioning_batch[0:1]
negative_conditioning = conditioning_batch[1:2]
pooled = pooled_batch[0:1]
negative_pooled = pooled_batch[1:2]

# Move text encoders back to CPU, then enable CPU offload for the diffusion pass
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.empty_cache()
pipe.enable_model_cpu_offload()  # moves components to CPU when idle

image = pipe(
    prompt_embeds=conditioning,
    pooled_prompt_embeds=pooled,
    negative_prompt_embeds=negative_conditioning,
    negative_pooled_prompt_embeds=negative_pooled,
    num_inference_steps=22,
    guidance_scale=7.0,
    width=1024,
    height=1024,
).images[0]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUTPUT_DIR / f"image_{ts}.png"
image.save(out_path)
print(f"Base image saved to {out_path}")

# Free SDXL VRAM before upscaling
del pipe, compel, conditioning, pooled, negative_conditioning, negative_pooled
torch.cuda.empty_cache()

# 4x-UltraSharp upscale (1024 -> 4096)
print("Upscaling with 4x-UltraSharp...")
upscaler = load_upscaler(UPSCALER_PATH, device="cuda")
upscaled = upscale_image(upscaler, image, device="cuda", tile=384, tile_overlap=32)

up_path = OUTPUT_DIR / f"image_{ts}_4x.png"
upscaled.save(up_path)
print(f"Upscaled image saved to {up_path} ({upscaled.size[0]}x{upscaled.size[1]})")
