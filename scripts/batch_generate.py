"""Batch image generation with RealVisXL V4.0."""

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "../models/realvisxl-v4",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)

prompts = [
    "A serene mountain landscape at golden hour, photorealistic, 8K",
    "A futuristic robot in a workshop, cinematic, dramatic lighting",
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.0).images[0]
    image.save(f"../outputs/batch_{i:03d}.png")
    print(f"Saved batch_{i:03d}.png")
