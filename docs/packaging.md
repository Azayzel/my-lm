# Packaging

## Build an installer

```bash
cd ui
npm run package
```

Outputs to `ui/release/`:

- **Windows:** NSIS installer (`.exe`)
- **Linux:** AppImage
- **macOS:** DMG

## What's bundled

`electron-builder` (configured in [ui/package.json](../ui/package.json)) bundles:

- The compiled main and renderer (`dist/`)
- `node_modules`
- The Python `scripts/` directory (as `extraResources`)

## What's NOT bundled

- The `.venv` — end users need their own Python install with deps
- Model weights (`models/`) — too large; downloaded on first run
- Generated outputs (`outputs/`) — runtime-only

## End-user requirements

A packaged build still needs:

- Python 3.10+ on PATH with the deps from `requirements.txt` installed
- An NVIDIA GPU + CUDA drivers
- ~13 GB free disk for the default model bundle (downloaded by the app)

For a fully self-contained installer that ships its own Python, see the open issue [#TODO] for the embedded-Python plan.

## Code signing

Not configured. To ship signed builds:

- Windows: set `CSC_LINK` and `CSC_KEY_PASSWORD` env vars before `npm run package`
- macOS: set `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` and add `mac.notarize: true` to the build config

See the [electron-builder docs](https://www.electron.build/code-signing) for details.
