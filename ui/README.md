# MyLm UI

The Electron frontend for **MyLm** — see the [top-level README](../README.md) for the full feature list and setup.

This doc covers UI internals only.

## Architecture

```
ui/
├── src/
│   ├── main/                   Main process (Node)
│   │   ├── main.ts             Window, IPC handlers, process lifecycle
│   │   ├── preload.ts          contextBridge API exposed as window.lavely
│   │   ├── llmBridge.ts        Spawns llm_bridge.py, streams tokens
│   │   ├── imageBridge.ts      Spawns image_bridge.py, streams progress + previews
│   │   ├── trainBridge.ts      Spawns train_bridge.py, streams training metrics
│   │   ├── modelManager.ts     Lists installed models, downloads HF repos
│   │   ├── modelCatalog.ts     Curated catalog with VRAM filter
│   │   ├── historyStore.ts     Chat + image history (userData/history.json)
│   │   └── promptStore.ts      Saved prompts (userData/prompts.json)
│   └── renderer/               UI (vanilla TS + webpack)
│       ├── index.html          All screens in one file
│       ├── app.ts              Screen logic
│       └── global.d.ts         TypeScript globals
├── package.json
├── tsconfig.json               Root (IDE language service)
├── tsconfig.main.json          Main build (CommonJS for Node)
├── tsconfig.renderer.json      Renderer build (ES2020 for webpack)
└── webpack.renderer.js
```

## Screens

| Screen | What it does |
|---|---|
| **Chat** | Streaming chat with the configured LLM |
| **Generate** | SDXL prompt builder, live latent previews (TAESDXL), ETA, face fix, upscale, saved prompts |
| **Media** | Image gallery with lightbox + file operations |
| **Train** | QLoRA fine-tuning with live metrics + merge-adapter action |
| **Models** | Installed list + GPU-aware catalog + manual HF download |
| **GPU** | Live VRAM / temp / utilization |

## Run

```powershell
cd ui
npm install
npm run build
npm start          # build + launch
npm run dev        # watch mode
npm run package    # produce installer
```

## IPC pattern

All renderer → main calls go through `window.lavely.*` (defined in `preload.ts`) which wraps `ipcRenderer.invoke`. Long-running operations (chat, image gen, training, downloads) use fire-and-forget `send()` + an `onEvent` subscription for streaming updates back to the renderer.

Each Python bridge is a long-lived subprocess owned by the main process. Bridges read newline-delimited JSON from stdin and emit newline-delimited JSON events to stdout. Screen navigation never kills or restarts bridges — you can freely leave and return while anything is running.

## Prompt builder

The image screen composes four fields into the final SDXL prompt:

```
head     = "masterpiece, best quality, photorealistic, ultra-detailed, 8k raw photo"
subject  = "a snow leopard on a cliff at dawn"
position = "looking toward the camera with piercing green eyes"
style    = "Soft golden-hour lighting, cinematic depth of field, high resolution"

prompt   = "$head. $subject. $position. $style"
```

Saved prompts store all four fields plus all generation params (steps, CFG, size, face fix, etc.) so you can fully restore a setup with one click.
