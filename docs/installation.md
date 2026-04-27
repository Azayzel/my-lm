# Installation

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Node.js 18+**
- **NVIDIA GPU with CUDA drivers** — `nvidia-smi` must be on PATH
- ~20 GB free disk for the default model bundle
- Windows 10/11, Linux, or macOS (macOS runs CPU/MPS — slow for SDXL)

## One-shot setup

**Linux / macOS:**

```bash
git clone https://github.com/lavely/my-lm.git
cd my-lm
./setup.sh
cd ui
npm start
```

**Windows:**

```powershell
git clone https://github.com/lavely/my-lm.git
cd my-lm
setup.bat
cd ui
npm start
```

The setup script:

1. Creates `.venv` (Python virtual environment)
2. Installs PyTorch 2.5.1 + CUDA 12.1 wheels (with a post-install CUDA sanity check)
3. Installs all pinned dependencies from [requirements.txt](../requirements.txt)
4. Uninstalls `xformers` if a transitive dep pulled it in (incompatible with torch 2.5.x)
5. Runs `npm install` and `npm run build` in `ui/`

## First run — model download wizard

On first launch, My-LM checks for the essential models. If any are missing, a welcome modal appears with an **Install All** button that downloads:

| Model                | Size    | Purpose                         |
| -------------------- | ------- | ------------------------------- |
| Qwen 2.5 3B Instruct | ~6 GB   | Default chat LLM                |
| RealVisXL V4.0       | ~6.5 GB | Photorealistic SDXL image model |
| 4x-UltraSharp        | ~70 MB  | 4× image upscaler               |
| YOLOv8n Face         | ~6 MB   | Face detector for Face Fix      |
| TAESDXL              | ~10 MB  | Tiny VAE for streaming previews |

**Total: ~13 GB** on a typical broadband connection (~10–20 minutes).

You can also install them individually, skip the modal, or open the full **Model Catalog** under the Models screen for more options (Llama 3.2, Phi-3.5, Juggernaut XL, DreamShaper, SDXL Base/Refiner, etc.). The catalog is GPU-aware and hides models that won't fit your VRAM by default.

## Configuration

Copy `.env.example` to `.env` and fill in values for any feature you use:

```bash
cp .env.example .env
```

| Variable                | Required for | Notes                                                       |
| ----------------------- | ------------ | ----------------------------------------------------------- |
| `MONGODB_URI`           | BookMind     | Atlas cluster connection string                             |
| `MONGODB_DB`            | BookMind     | Database name                                               |
| `BOOKMIND_EMBED_MODEL`  | BookMind     | Must match the model used to populate embeddings            |
| `BOOKMIND_VECTOR_INDEX` | BookMind     | Atlas Vector Search index name on the books collection      |

`.env` is gitignored — never commit it.

## Development mode (hot reload)

```bash
cd ui
npm run dev
```

This runs the renderer webpack in watch mode, the main TS compiler in watch mode, and Electron — all concurrently.

## Troubleshooting

### `torch.cuda.is_available()` returns False after setup

Pip's resolver sometimes pulls a CPU-only torch as a transitive dep. Re-run setup — it auto-detects and force-reinstalls with `--no-deps` from the CUDA index.

### `xformers` errors during SDXL

`xformers` wheels are pinned to torch 2.10+ and break SDXL on 2.5.x. The setup script uninstalls it; if you re-installed it manually, run:

```bash
pip uninstall -y xformers
```

Diffusers falls back to PyTorch's built-in scaled-dot-product attention.

### Model download fails midway

`huggingface_hub` resumes interrupted downloads. Click Install again — it'll pick up where it left off.

### Out of VRAM on 6 GB cards

- Disable Face Fix and Upscale for first pass
- Reduce SDXL steps to 25–30
- Don't run chat + image + training simultaneously
