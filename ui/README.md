# Lavely UI

A dark-mode Electron application for chatting with a local LLM and generating images — all in one window.

## Architecture

```
ui/                         ← Electron app (TypeScript)
├── src/
│   ├── main/
│   │   ├── main.ts         ← Electron main process, IPC handlers
│   │   ├── preload.ts      ← contextBridge (safe renderer↔main API)
│   │   ├── llmBridge.ts    ← Spawns llm_bridge.py, streams tokens
│   │   ├── imageBridge.ts  ← Spawns image_bridge.py, streams progress
│   │   ├── historyStore.ts ← Persists chat+image history to JSON
│   │   └── modelManager.ts ← Lists and downloads HuggingFace models
│   └── renderer/
│       ├── index.html      ← App shell (dark theme, all screens)
│       ├── app.ts          ← All UI logic (chat, image gen, media, models)
│       └── global.d.ts     ← TypeScript type declarations
scripts/
├── llm_bridge.py           ← Streaming LLM inference (stdin→stdout JSON)
├── image_bridge.py         ← SDXL image gen (stdin→stdout JSON + progress)
└── model_download.py       ← HuggingFace snapshot_download wrapper
```

## Screens

| Screen | What it does |
|---|---|
| **Chat** | Stream-rendered chat with Qwen3.5-2B. System prompt, temperature, top-p, max tokens controls. |
| **Generate** | Prompt builder (head / name / position / weights → full prompt), SDXL generation with live progress bar, 4× upscale. |
| **Media** | Gallery of all generated images. Lightbox, open, show in folder, delete. |
| **Models** | Lists local models, download any HuggingFace repo by ID. |

## Quick Start

```powershell
cd ui
npm install
npm run build
npm start          # builds and launches Electron
```

## Development (hot-reload renderer)

```powershell
cd ui
npm run dev        # watches main TS + renderer webpack + runs Electron
```

## How the Python bridges work

**LLM Bridge** (`scripts/llm_bridge.py`):
- Launched once by main process, stays alive
- Receives one JSON request per line on `stdin`
- Streams `{"type":"token","text":"..."}` lines to `stdout`
- Sends `{"type":"done"}` when generation finishes

**Image Bridge** (`scripts/image_bridge.py`):
- Same persistent process model
- Sends `{"type":"progress","step":N,"total":N}` on each diffusion step
- Sends `{"type":"done","path":"..."}` with saved file path

## Prompt Builder

The image screen has four fields that compose the final prompt:

```
head     = "masterpiece, best quality, photorealistic, ultra-detailed, 8k raw photo"
name     = "Superman"
position = "taking a mirror selfie while wearing the classic suit, smiling with a smirk"
weights  = "Soft natural indoor lighting, realistic skin texture, flushed cheeks..."

prompt   = "$head $name. $position. $weights"
```

## Packaging

```powershell
cd ui
npm run package    # produces release/ with installer
```
