// Curated catalog of models known to work well with My-LM.
// Each entry has a minimum VRAM recommendation (quantized where applicable)
// so we can filter by the user's actual GPU.

export interface CatalogEntry {
  id: string; // unique key
  name: string; // display name
  repoId: string; // HF repo id
  category: "llm" | "image" | "upscaler" | "vae" | "face";
  minVramGb: number; // realistic minimum for comfortable use
  sizeGb: number; // approximate download size
  description: string;
  tags: string[];
  targetDir: string; // suggested local dir under models/
}

export const MODEL_CATALOG: CatalogEntry[] = [
  // ───── LLMs ─────────────────────────────────────────────────────────────
  {
    id: "qwen25-05b",
    name: "Qwen 2.5 0.5B Instruct",
    repoId: "Qwen/Qwen2.5-0.5B-Instruct",
    category: "llm",
    minVramGb: 2,
    sizeGb: 1.2,
    description:
      "Tiny general-purpose LLM. Runs on almost anything. Good for quick testing.",
    tags: ["llm", "tiny", "fast"],
    targetDir: "qwen2.5-0.5b",
  },
  {
    id: "qwen25-15b",
    name: "Qwen 2.5 1.5B Instruct",
    repoId: "Qwen/Qwen2.5-1.5B-Instruct",
    category: "llm",
    minVramGb: 4,
    sizeGb: 3.1,
    description:
      "Small, fast chat LLM. Solid quality for its size, fits any modern GPU.",
    tags: ["llm", "fast"],
    targetDir: "qwen2.5-1.5b",
  },
  {
    id: "qwen25-3b",
    name: "Qwen 2.5 3B Instruct",
    repoId: "Qwen/Qwen2.5-3B-Instruct",
    category: "llm",
    minVramGb: 6,
    sizeGb: 6.2,
    description:
      "The sweet spot for 6 GB GPUs. Good reasoning, fast inference, QLoRA-trainable.",
    tags: ["llm", "recommended"],
    targetDir: "qwen2.5-3b",
  },
  {
    id: "qwen25-7b",
    name: "Qwen 2.5 7B Instruct",
    repoId: "Qwen/Qwen2.5-7B-Instruct",
    category: "llm",
    minVramGb: 10,
    sizeGb: 15.2,
    description:
      "Mid-size LLM with strong reasoning. Needs 4-bit quantization for 8 GB cards.",
    tags: ["llm"],
    targetDir: "qwen2.5-7b",
  },
  {
    id: "llama32-1b",
    name: "Llama 3.2 1B Instruct",
    repoId: "meta-llama/Llama-3.2-1B-Instruct",
    category: "llm",
    minVramGb: 3,
    sizeGb: 2.5,
    description:
      "Meta's tiny chat model. Requires HF gated access (accept license).",
    tags: ["llm", "gated"],
    targetDir: "llama3.2-1b",
  },
  {
    id: "llama32-3b",
    name: "Llama 3.2 3B Instruct",
    repoId: "meta-llama/Llama-3.2-3B-Instruct",
    category: "llm",
    minVramGb: 6,
    sizeGb: 6.4,
    description:
      "Meta's compact chat model. Strong instruction following. Gated on HF.",
    tags: ["llm", "gated"],
    targetDir: "llama3.2-3b",
  },
  {
    id: "phi35-mini",
    name: "Phi-3.5 Mini Instruct",
    repoId: "microsoft/Phi-3.5-mini-instruct",
    category: "llm",
    minVramGb: 6,
    sizeGb: 7.6,
    description:
      "Microsoft's reasoning-tuned 3.8B model. Punches way above its weight for reasoning.",
    tags: ["llm", "reasoning"],
    targetDir: "phi3.5-mini",
  },

  // ───── SDXL image models ────────────────────────────────────────────────
  {
    id: "realvis-v4",
    name: "RealVisXL V4.0",
    repoId: "SG161222/RealVisXL_V4.0",
    category: "image",
    minVramGb: 6,
    sizeGb: 6.5,
    description:
      "Photorealistic SDXL finetune. Great for portraits and realistic scenes.",
    tags: ["sdxl", "photoreal", "recommended"],
    targetDir: "realvisxl-v4",
  },
  {
    id: "juggernaut-xl",
    name: "Juggernaut XL v9",
    repoId: "RunDiffusion/Juggernaut-XL-v9",
    category: "image",
    minVramGb: 6,
    sizeGb: 6.6,
    description:
      "Versatile SDXL model. Good balance of photorealism and stylized output.",
    tags: ["sdxl", "versatile"],
    targetDir: "juggernaut-xl",
  },
  {
    id: "dreamshaper-xl",
    name: "DreamShaper XL v2 Turbo",
    repoId: "Lykon/dreamshaper-xl-v2-turbo",
    category: "image",
    minVramGb: 6,
    sizeGb: 6.5,
    description:
      "Fast SDXL variant — 4–8 steps for solid results. Great for low-VRAM cards.",
    tags: ["sdxl", "fast"],
    targetDir: "dreamshaper-xl",
  },
  {
    id: "sdxl-base",
    name: "SDXL Base 1.0",
    repoId: "stabilityai/stable-diffusion-xl-base-1.0",
    category: "image",
    minVramGb: 6,
    sizeGb: 6.9,
    description:
      "Official Stability AI SDXL base model. Good general-purpose baseline.",
    tags: ["sdxl", "official"],
    targetDir: "sdxl-base-1.0",
  },
  {
    id: "sdxl-refiner",
    name: "SDXL Refiner 1.0",
    repoId: "stabilityai/stable-diffusion-xl-refiner-1.0",
    category: "image",
    minVramGb: 6,
    sizeGb: 6.1,
    description:
      "Detail-refiner model for SDXL. Runs the final 20% of steps for extra polish.",
    tags: ["sdxl", "refiner"],
    targetDir: "sdxl-refiner-1.0",
  },

  // ───── Small / SD 1.5 for low-VRAM ─────────────────────────────────────
  {
    id: "rev-animated",
    name: "ReV Animated (SD 1.5)",
    repoId: "stablediffusionapi/rev-animated",
    category: "image",
    minVramGb: 4,
    sizeGb: 4.2,
    description:
      "Popular SD 1.5 model for stylized anime/semi-realistic output. 4 GB friendly.",
    tags: ["sd15", "anime"],
    targetDir: "rev-animated",
  },

  // ───── Upscalers ────────────────────────────────────────────────────────
  {
    id: "4x-ultrasharp",
    name: "4x-UltraSharp",
    repoId: "lokCX/4x-Ultrasharp",
    category: "upscaler",
    minVramGb: 2,
    sizeGb: 0.07,
    description:
      "High-quality ESRGAN 4× upscaler. The default detail-preserving upscaler.",
    tags: ["upscaler", "4x", "recommended"],
    targetDir: "upscalers",
  },
  {
    id: "real-esrgan",
    name: "RealESRGAN x4plus",
    repoId: "ai-forever/Real-ESRGAN",
    category: "upscaler",
    minVramGb: 2,
    sizeGb: 0.07,
    description:
      "General-purpose 4× upscaler. Slightly softer than UltraSharp.",
    tags: ["upscaler", "4x"],
    targetDir: "upscalers",
  },

  // ───── Face / detail ───────────────────────────────────────────────────
  {
    id: "yolov8-face",
    name: "YOLOv8n Face (ADetailer)",
    repoId: "Bingsu/adetailer",
    category: "face",
    minVramGb: 1,
    sizeGb: 0.01,
    description:
      "Fast face detector for the Face Fix pass. Required for ADetailer-style refinement.",
    tags: ["face", "detector", "recommended"],
    targetDir: "face_detector",
  },

  // ───── Preview VAEs ─────────────────────────────────────────────────────
  {
    id: "taesdxl",
    name: "TAESDXL (Tiny VAE)",
    repoId: "madebyollin/taesdxl",
    category: "vae",
    minVramGb: 1,
    sizeGb: 0.01,
    description:
      "Tiny decoder for live latent previews during SDXL generation. ~10 MB.",
    tags: ["vae", "preview", "recommended"],
    targetDir: "taesdxl",
  },
];

/**
 * Filter the catalog by available VRAM, returning entries the user can comfortably run.
 * If `vramGb` is 0 or unknown, returns the full catalog.
 */
export function filterByVram(vramGb: number): CatalogEntry[] {
  if (!vramGb || vramGb <= 0) return MODEL_CATALOG;
  return MODEL_CATALOG.filter((m) => m.minVramGb <= vramGb);
}
