# Features

Detailed per-screen tour. For a high-level overview see the [README](../README.md).

## Chat

- Local LLM inference via a persistent Python bridge (Qwen 2.5 / Llama 3.2 / Phi-3.5 supported)
- Streaming token rendering, system prompt, temperature / top-p / max tokens controls
- Conversation history saved per session

## Image generation

- **SDXL pipeline** (RealVisXL V4 by default) with `compel` for long-prompt support
- **Live latent previews** — TAESDXL tiny decoder streams a rough preview every 2 steps so you watch the image materialize
- **ETA** — moving-average estimator shows remaining time during generation
- **Prompt builder** — four structured fields (head / subject / position / style) compose the final prompt
- **Saved prompts** — name, save, load, and delete prompt presets (stored in `userData/prompts.json`)
- **4× Upscale** — 4x-UltraSharp ESRGAN, tiled to fit 6 GB VRAM
- **Face Fix** (ADetailer-style) — YOLOv8n face detection + SDXL img2img refinement per face with feathered alpha-compositing. Single biggest win for facial accuracy.
- Timestamped filenames, no overwrites

## BookMind (RAG book recommender)

- Semantic search over a MongoDB Atlas cluster using **`$vectorSearch`** on 384-dim embeddings
- Shares the same `all-MiniLM-L6-v2` model as the BookMind C# API that populated the database
- Optional **taste vector blending** — mixes a free-text query with a user's stored `tasteVector` (70/30)
- Auto-excludes books the user has already read (from `userBooks`)
- Optional LLM explanation pass — retrieves top-K candidates, then streams a grounded natural-language recommendation from your local Qwen model (hallucination-free: can only mention books in the retrieved set)
- New **Books** screen with candidate cards + live streaming LLM output
- CLI: `python scripts/book_recommend.py "a cozy fantasy with witches" --user <email> --llm`

See [docs/bookmind.md](bookmind.md) for setup.

## Training (QLoRA)

- 4-bit quantized base model + LoRA adapters, trainable on 6 GB VRAM
- Full hyperparameter grid: epochs, batch size, grad accum, LR, LoRA r/α/dropout, max seq len
- Live progress bar with loss / epoch / LR metrics streamed from a `TrainerCallback`
- **Merge Adapter** button fuses the LoRA back into the base model in one click

## Model catalog

- Curated list of 16+ models across LLMs, SDXL/SD1.5, upscalers, face detectors, and preview VAEs
- **GPU-aware** — detects your VRAM via `torch.cuda.get_device_properties` and hides entries that won't fit (toggle "Show all" to override)
- Category filter, size estimates, one-click install, "installed" state tracking
- Re-uses the same HF `snapshot_download` pipeline as manual downloads

## System

- **GPU dashboard** — `torch.cuda` info + `nvidia-smi` live output (memory, temp, utilization)
- **Diagnostics** — Python resolution, package versions, venv check
- Input / confirm modals that actually work in Electron (native `window.prompt` is disabled)

## Screens

| Screen   | Purpose                                                                                   |
| -------- | ----------------------------------------------------------------------------------------- |
| Chat     | Stream-rendered chat with the configured LLM. System prompt, sampling params, clear convo. |
| Generate | Prompt builder, SDXL with live latent preview + ETA, face fix, 4× upscale, saved prompts. |
| Books    | BookMind RAG recommender — Atlas Vector Search + optional LLM explanations.               |
| Media    | Gallery of generated images. Lightbox, open in viewer, show in folder, delete.            |
| Train    | QLoRA fine-tuning with live loss/epoch metrics, merge-adapter button.                     |
| Models   | Installed models, GPU-aware catalog, manual HF repo download.                             |
| GPU      | VRAM / utilization / temperature via nvidia-smi + torch.cuda.                             |

## Memory budget on 6 GB

The app is designed so no single step blows out VRAM:

- SDXL base pass: `enable_model_cpu_offload()` streams components on/off GPU
- Text encoders temporarily moved to GPU for compel, then back to CPU before diffusion
- Face detailer reuses base pipe's weights (no second SDXL load)
- 4× upscale runs tiled at 384 px with 32 px overlap
- Training uses 4-bit NF4 quantization + gradient checkpointing

Do **not** run chat + image + training simultaneously — they each want the whole GPU.

## How the face detailer works

1. Generate base image via SDXL
2. Run YOLOv8n face detector → list of face bounding boxes
3. For each face: expand bbox with 40% padding, crop, resize crop to 1024×1024
4. Run SDXL **img2img** on the crop (shares weights with base pipe, no extra VRAM) with the original prompt prefixed by face-specific detail cues, at denoise strength 0.35–0.55
5. Resize refined crop back to original size, alpha-composite into the full image with a feathered mask to hide seams

Cost: ~1 extra minute per face on a 6 GB RTX 2060.
