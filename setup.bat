@echo off
echo === Lavely-LM Setup ===

:: Create virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

:: Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

:: Enable faster HF downloads
set HF_HUB_ENABLE_HF_TRANSFER=1

echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. Activate venv:  .venv\Scripts\activate
echo   2. Download Qwen3.5-2B:
echo      huggingface-cli download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
echo   3. Download RealVisXL V4.0:
echo      huggingface-cli download SG161222/RealVisXL_V4.0 --local-dir models/realvisxl-v4
echo   4. Run inference:  python scripts/qwen_inference.py
