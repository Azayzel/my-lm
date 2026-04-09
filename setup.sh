#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  MyLm — one-shot setup for Linux/macOS
#  Creates .venv, installs Python deps with correct CUDA wheels (Linux), and
#  builds the Electron UI. Safe to re-run.
# ─────────────────────────────────────────────────────────────────────────
set -e

echo
echo "=== MyLm setup ==="
echo

# --- 1. Check Python -----------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[x] python3 not found. Install Python 3.10+ and try again."
    exit 1
fi

PYVER=$(python3 --version | awk '{print $2}')
echo "[1/5] Found Python $PYVER"

# --- 2. Create venv if missing -------------------------------------------
if [ -x ".venv/bin/python" ]; then
    echo "[2/5] .venv already exists — reusing"
else
    echo "[2/5] Creating .venv..."
    python3 -m venv .venv
fi

VENV_PY=".venv/bin/python"

# --- 3. Install PyTorch --------------------------------------------------
$VENV_PY -m pip install --upgrade pip >/dev/null

OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
    echo "[3/5] Installing PyTorch 2.5.1 + CUDA 12.1..."
    $VENV_PY -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

    if ! $VENV_PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[!] CUDA check failed — forcing reinstall with --no-deps"
        $VENV_PY -m pip install torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
    fi
    $VENV_PY -c "import torch; print('    torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
else
    echo "[3/5] Installing PyTorch (CPU/MPS build for macOS)..."
    $VENV_PY -m pip install torch==2.5.1 torchvision==0.20.1
fi

# --- 4. Install the rest of the Python deps -----------------------------
echo "[4/5] Installing Python dependencies from requirements.txt..."
$VENV_PY -m pip install -r requirements.txt

# Remove xformers if pulled in transitively — breaks SDXL on torch 2.5
$VENV_PY -m pip uninstall -y xformers >/dev/null 2>&1 || true

# --- 5. Build the Electron UI -------------------------------------------
if ! command -v npm >/dev/null 2>&1; then
    echo "[!] npm not found — skipping UI build."
    echo "    Install Node.js 18+, then run:"
    echo "        cd ui && npm install && npm run build"
else
    echo "[5/5] Installing and building UI..."
    (cd ui && npm install && npm run build)
fi

echo
echo "=== Setup complete! ==="
echo
echo "Next steps:"
echo "  1. Launch the app:     cd ui && npm start"
echo "  2. On first run the app will prompt you to download essential models."
echo
echo "CLI usage:"
echo "  source .venv/bin/activate"
echo "  python scripts/qwen_inference.py"
echo "  python scripts/generate_image.py"
echo
