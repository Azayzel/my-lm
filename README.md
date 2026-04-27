# My-LM

> **My LM** — a play on "LLM". A local, all-in-one playground for running, training, and generating with open-weight models on a single GPU. Dark-mode Electron UI, Python backends, no cloud.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Designed around modest GPUs (built and tested on a **6 GB RTX 2060**) but scales up gracefully. Runs a chat LLM, an SDXL image generator, a QLoRA fine-tuning loop, and a curated model catalog — all from one window.

---

## Features

- **Chat** — local LLM inference (Qwen 2.5 / Llama 3.2 / Phi-3.5) with streaming, sampling controls, and per-session history.
- **Image generation** — SDXL with `compel` long-prompt support, live latent previews via TAESDXL, ETA, prompt builder, saved presets, 4× ESRGAN upscale, ADetailer-style face fix.
- **BookMind RAG** — semantic book recommendations via MongoDB Atlas `$vectorSearch` with optional grounded LLM explanations.
- **Training** — QLoRA fine-tuning (4-bit NF4) with live loss/epoch metrics and one-click adapter merge.
- **Model catalog** — curated, GPU-aware model picker that hides anything that won't fit your VRAM.
- **System** — GPU dashboard (`nvidia-smi` + `torch.cuda`), diagnostics, sane defaults.

See [docs/features.md](docs/features.md) for the full breakdown.

---

## Quick start

**Prerequisites:** Python 3.10+, Node.js 18+, NVIDIA GPU with CUDA drivers, `nvidia-smi` on PATH.

```bash
git clone https://github.com/lavely/my-lm.git
cd my-lm

# Linux / macOS
./setup.sh

# Windows
setup.bat

# Launch
cd ui && npm start
```

The setup script creates a `.venv`, installs CUDA-matched PyTorch wheels, installs Python dependencies, and builds the Electron UI. On first launch the app prompts you to download essential models (~13 GB).

For configuration (BookMind / MongoDB), copy `.env.example` to `.env` and fill in your values. Full setup details and troubleshooting live in [docs/installation.md](docs/installation.md).

---

## Documentation

| Doc                                          | What's in it                                       |
| -------------------------------------------- | -------------------------------------------------- |
| [docs/features.md](docs/features.md)         | Per-screen feature tour                            |
| [docs/installation.md](docs/installation.md) | Setup, model downloads, troubleshooting            |
| [docs/architecture.md](docs/architecture.md) | Process model, IPC, bridge protocol, memory budget |
| [docs/cli.md](docs/cli.md)                   | Standalone CLI tools (`scripts/`)                  |
| [docs/bookmind.md](docs/bookmind.md)         | BookMind RAG configuration                         |
| [docs/packaging.md](docs/packaging.md)       | Building installers                                |
| [CONTRIBUTING.md](CONTRIBUTING.md)           | How to contribute                                  |
| [SECURITY.md](SECURITY.md)                   | Reporting vulnerabilities                          |

---

## Project layout

```text
my-lm/
├── src/mylm/        Reusable Python library (pip-installable)
├── scripts/         Bridge + CLI entry points (spawned by Electron)
├── ui/              Electron app (TypeScript main + renderer)
├── tests/           pytest suite
├── datasets/        Example training data
├── docs/            Long-form documentation
└── .github/         Issue/PR templates and CI workflows
```

The `models/` and `outputs/` directories are created on first run and are gitignored — they hold multi-GB model weights and generated images. See [docs/architecture.md](docs/architecture.md) for the full breakdown.

---

## Contributing

PRs welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — it covers the dev loop, coding conventions, and how to run linters/tests locally. Be excellent to each other ([Code of Conduct](CODE_OF_CONDUCT.md)).

Found a bug or have an idea? [Open an issue](https://github.com/lavely/my-lm/issues/new/choose).

---

## License

[MIT](LICENSE) © 2026 Josh Lavely
