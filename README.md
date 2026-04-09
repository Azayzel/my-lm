# MyLm

**My LLM** — a local, all-in-one playground for running, training, and generating with open-weight models on a single GPU. Dark-mode Electron UI, Python backends, no cloud.

Designed around modest GPUs (built and tested on a **6 GB RTX 2060**) but scales up gracefully. Runs a chat LLM, an SDXL image generator, a QLoRA fine-tuning loop, and a curated model catalog — all from one window.

---

## ✨ Features

### Chat
- Local LLM inference via a persistent Python bridge (Qwen 2.5 / Llama 3.2 / Phi-3.5 supported)
- Streaming token rendering, system prompt, temperature / top-p / max tokens controls
- Conversation history saved per session

### Image generation
- **SDXL pipeline** (RealVisXL V4 by default) with `compel` for long-prompt support
- **Live latent previews** — TAESDXL tiny decoder streams a rough preview every 2 steps so you watch the image materialize
- **ETA** — moving-average estimator shows remaining time during generation
- **Prompt builder** — four structured fields (head / subject / position / style) compose the final prompt
- **Saved prompts** — name, save, load, and delete prompt presets (stored in `userData/prompts.json`)
- **4× Upscale** — 4x-UltraSharp ESRGAN, tiled to fit 6 GB VRAM
- **Face Fix** (ADetailer-style) — YOLOv8n face detection + SDXL img2img refinement per face with feathered alpha-compositing. Single biggest win for facial accuracy.
- Timestamped filenames, no overwrites

### Training (QLoRA)
- 4-bit quantized base model + LoRA adapters, trainable on 6 GB VRAM
- Full hyperparameter grid: epochs, batch size, grad accum, LR, LoRA r/α/dropout, max seq len
- Live progress bar with loss / epoch / LR metrics streamed from a `TrainerCallback`
- **Merge Adapter** button fuses the LoRA back into the base model in one click

### Model Catalog
- Curated list of 16+ models across LLMs, SDXL/SD1.5, upscalers, face detectors, and preview VAEs
- **GPU-aware** — detects your VRAM via `torch.cuda.get_device_properties` and hides entries that won't fit (toggle "Show all" to override)
- Category filter, size estimates, one-click install, "installed" state tracking
- Re-uses the same HF `snapshot_download` pipeline as manual downloads

### System
- **GPU dashboard** — `torch.cuda` info + `nvidia-smi` live output (memory, temp, utilization)
- **Diagnostics** — Python resolution, package versions, venv check
- Input / confirm modals that actually work in Electron (native `window.prompt` is disabled)

---

## 🧱 Architecture

```
MyLm/
├── .venv/                        Python virtual environment
├── models/                       Local model weights (gitignored)
│   ├── qwen2.5-*/                LLMs
│   ├── realvisxl-v4/             SDXL image model
│   ├── upscalers/                4x-UltraSharp.pth
│   ├── face_detector/            face_yolov8n.pt
│   └── taesdxl/                  Tiny VAE for streaming previews
├── outputs/                      Generated images (timestamped)
├── datasets/                     Training data (JSONL chat format)
├── scripts/                      Python bridges and tools
│   ├── llm_bridge.py             Streaming chat (stdin→stdout JSON)
│   ├── image_bridge.py           SDXL + compel + face detailer + upscaler
│   ├── train_bridge.py           QLoRA training with streaming metrics
│   ├── face_detailer.py          YOLOv8 + SDXL img2img refinement
│   ├── upscaler.py               Spandrel ESRGAN loader with tiling
│   ├── merge_lora.py             Fuse LoRA adapter into base model
│   ├── model_download.py         HF snapshot_download wrapper
│   ├── generate_image.py         Standalone CLI image generation
│   ├── train_qlora.py            Standalone CLI training
│   └── qwen_inference.py         Standalone CLI chat
└── ui/                           Electron app (TypeScript)
    ├── src/
    │   ├── main/                 Main process + IPC
    │   │   ├── main.ts
    │   │   ├── preload.ts
    │   │   ├── llmBridge.ts
    │   │   ├── imageBridge.ts
    │   │   ├── trainBridge.ts
    │   │   ├── modelManager.ts
    │   │   ├── modelCatalog.ts   Curated catalog + VRAM filter
    │   │   ├── historyStore.ts   Chat/image history
    │   │   └── promptStore.ts    Saved prompts
    │   └── renderer/             UI (vanilla TS + webpack)
    │       ├── index.html
    │       ├── app.ts
    │       └── global.d.ts
    ├── package.json
    ├── tsconfig.main.json
    ├── tsconfig.renderer.json
    └── tsconfig.json             Root (for IDE)
```

All Python processes are long-lived subprocesses spawned by the Electron main process; they communicate via newline-delimited JSON over stdin/stdout. Screens can switch freely without interrupting any running operation.

---

## 🚀 Quick start

**Prerequisites:** Python 3.10+, Node.js 18+, NVIDIA GPU with CUDA drivers, `nvidia-smi` on PATH.

### One-shot setup

**Windows:**

```powershell
git clone https://github.com/yourname/mylm.git
cd mylm
setup.bat
cd ui
npm start
```

**Linux/macOS:**

```bash
git clone https://github.com/yourname/mylm.git
cd mylm
./setup.sh
cd ui
npm start
```

The setup script:

1. Creates `.venv` (Python virtual environment)
2. Installs PyTorch 2.5.1 + CUDA 12.1 wheels (with a post-install CUDA sanity check)
3. Installs all pinned dependencies from [requirements.txt](requirements.txt)
4. Uninstalls `xformers` if a transitive dep pulled it in (incompatible with torch 2.5.x)
5. Runs `npm install` and `npm run build` in `ui/`

### First run — model download wizard

On first launch, MyLm checks for the essential models. If any are missing, a welcome modal appears with an **Install All** button that downloads:

| Model                | Size    | Purpose                             |
| -------------------- | ------- | ----------------------------------- |
| Qwen 2.5 3B Instruct | ~6 GB   | Default chat LLM                    |
| RealVisXL V4.0       | ~6.5 GB | Photorealistic SDXL image model     |
| 4x-UltraSharp        | ~70 MB  | 4× image upscaler                   |
| YOLOv8n Face         | ~6 MB   | Face detector for Face Fix          |
| TAESDXL              | ~10 MB  | Tiny VAE for streaming previews     |

**Total: ~13 GB** on a typical broadband connection (~10–20 minutes).

You can also install them individually, skip the modal, or open the full **Model Catalog** under the Models screen for more options (Llama 3.2, Phi-3.5, Juggernaut XL, DreamShaper, SDXL Base/Refiner, etc.). The catalog is GPU-aware and hides models that won't fit your VRAM by default.

### Development mode (hot reload)

```bash
cd ui
npm run dev
```

---

## 🎛 Screens

| Screen | Purpose |
|---|---|
| **Chat** | Stream-rendered chat with Qwen (or any installed LLM). System prompt, sampling params, clear-conversation. |
| **Generate** | Prompt builder, SDXL with live latent preview + ETA, face fix, 4× upscale, saved prompts. |
| **Media** | Gallery of generated images. Lightbox, open in viewer, show in folder, delete. |
| **Train** | QLoRA fine-tuning with live loss/epoch metrics, merge-adapter button. |
| **Models** | Installed models, GPU-aware catalog, manual HF repo download. |
| **GPU** | VRAM / utilization / temperature via nvidia-smi + torch.cuda. |

---

## 🧪 CLI tools

All scripts under `scripts/` also run standalone:

```bash
python scripts/qwen_inference.py                   # one-shot chat
python scripts/generate_image.py                   # single SDXL generation + face fix + upscale
python scripts/train_qlora.py                      # QLoRA training from datasets/train.jsonl
python scripts/merge_lora.py <base> <lora> <out>   # fuse adapter into base
```

---

## 🧠 How the face detailer works

1. Generate base image via SDXL
2. Run YOLOv8n face detector → list of face bounding boxes
3. For each face: expand bbox with 40% padding, crop, resize crop to 1024×1024
4. Run SDXL **img2img** on the crop (shares weights with base pipe, no extra VRAM) with the original prompt prefixed by face-specific detail cues, at denoise strength 0.35–0.55
5. Resize refined crop back to original size, alpha-composite into the full image with a feathered mask to hide seams

Cost: ~1 extra minute per face on a 6 GB RTX 2060.

---

## 📏 Memory budget on 6 GB

The app is designed so no single step blows out VRAM:

- SDXL base pass: `enable_model_cpu_offload()` streams components on/off GPU
- Text encoders temporarily moved to GPU for compel, then back to CPU before diffusion
- Face detailer reuses base pipe's weights (no second SDXL load)
- 4x upscale runs tiled at 384 px with 32 px overlap
- Training uses 4-bit NF4 quantization + gradient checkpointing

Do **not** run chat + image + training simultaneously — they each want the whole GPU.

---

## 📦 Packaging

```bash
cd ui
npm run package    # produces release/ with installer (NSIS on Windows, AppImage on Linux)
```

The `extraResources` config in `ui/package.json` bundles `scripts/`, `models/`, and `outputs/` alongside the app.

---

## 📝 License

MIT
