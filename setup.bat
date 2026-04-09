@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  MyLm — one-shot setup for Windows
REM  Creates .venv, installs Python deps with correct CUDA wheels, and builds
REM  the Electron UI. Safe to re-run.
REM ─────────────────────────────────────────────────────────────────────────
setlocal

echo.
echo === MyLm setup ===
echo.

REM --- 1. Check Python ------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [x] Python not found on PATH. Install Python 3.10 or newer from python.org.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/5] Found Python %PYVER%

REM --- 2. Create venv if missing --------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] .venv already exists — reusing
) else (
    echo [2/5] Creating .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [x] Failed to create venv
        exit /b 1
    )
)

set VENV_PY=.venv\Scripts\python.exe

REM --- 3. Install PyTorch with CUDA 12.1 ------------------------------------
echo [3/5] Installing PyTorch 2.5.1 + CUDA 12.1...
%VENV_PY% -m pip install --upgrade pip >nul
%VENV_PY% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [x] PyTorch install failed
    exit /b 1
)

REM Verify CUDA actually works (dep resolver sometimes pulls CPU-only torch)
%VENV_PY% -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after install'; print('    torch', torch.__version__, 'CUDA OK -', torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo [!] PyTorch CUDA check failed — forcing reinstall with --no-deps
    %VENV_PY% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
)

REM --- 4. Install the rest of the Python deps ------------------------------
echo [4/5] Installing Python dependencies from requirements.txt...
%VENV_PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [x] pip install -r requirements.txt failed
    exit /b 1
)

REM Remove xformers if some other package pulled it in — it breaks SDXL on torch 2.5
%VENV_PY% -m pip uninstall -y xformers >nul 2>&1

REM --- 5. Build the Electron UI ---------------------------------------------
where npm >nul 2>&1
if errorlevel 1 (
    echo [!] npm not found — skipping UI build.
    echo     Install Node.js 18+ from nodejs.org, then run:
    echo         cd ui ^&^& npm install ^&^& npm run build
    goto done
)

echo [5/5] Installing and building UI...
pushd ui
call npm install
if errorlevel 1 (
    echo [x] npm install failed
    popd
    exit /b 1
)
call npm run build
if errorlevel 1 (
    echo [x] npm run build failed
    popd
    exit /b 1
)
popd

:done
echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. Launch the app:     cd ui ^&^& npm start
echo   2. On first run the app will prompt you to download essential models.
echo.
echo CLI usage:
echo   .venv\Scripts\activate
echo   python scripts\qwen_inference.py
echo   python scripts\generate_image.py
echo.
endlocal
