@echo off
setlocal enabledelayedexpansion
title Narrate Studio — Setup
color 0B

echo.
echo  ===========================================
echo    Narrate Studio — First-time Setup
echo  ===========================================
echo.

:: ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python was not found.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Found Python %PY_VER%
echo.

:: ── Create virtual environment ────────────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    echo  [OK] Virtual environment already exists, skipping creation.
) else (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)
echo.

:: ── Upgrade pip ───────────────────────────────────────────────────────────────
echo  Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo  [OK] pip up to date.
echo.

:: ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
echo  Installing PyTorch with CUDA 12.4 support...
echo  (Downloading ~2.5 GB — this will take a few minutes)
echo.
.venv\Scripts\pip install ^
    torch==2.6.0+cu124 ^
    torchvision==0.21.0+cu124 ^
    torchaudio==2.6.0+cu124 ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    --quiet

if errorlevel 1 (
    echo.
    echo  [WARN] CUDA build failed. Falling back to CPU-only PyTorch.
    echo  The app will work but generation will be very slow without a GPU.
    echo.
    .venv\Scripts\pip install torch torchvision torchaudio --quiet
)
echo  [OK] PyTorch installed.
echo.

:: ── Core dependencies ─────────────────────────────────────────────────────────
echo  Installing core dependencies...
.venv\Scripts\pip install ^
    transformers ^
    accelerate ^
    snac ^
    soundfile ^
    PyQt6 ^
    numpy ^
    --quiet
echo  [OK] Core dependencies installed.
echo.

:: ── TTS models ────────────────────────────────────────────────────────────────
echo  Installing TTS model packages...
.venv\Scripts\pip install kokoro --quiet
echo  [OK] Kokoro installed.

.venv\Scripts\pip install chatterbox-tts --quiet
echo  [OK] Chatterbox installed.
echo.

:: ── spaCy (binary wheel only — avoids build errors) ──────────────────────────
echo  Installing spaCy (language processing for Kokoro)...
.venv\Scripts\pip install spacy --only-binary :all: --quiet
echo  [OK] spaCy installed.
echo.

:: ── HuggingFace token (for Orpheus) ──────────────────────────────────────────
echo  ===========================================
echo    HuggingFace Token (Orpheus model)
echo  ===========================================
echo.
echo  The Orpheus TTS model is hosted on HuggingFace and requires
echo  a free account + access approval.
echo.
echo  Steps:
echo    1. Create a free account at  https://huggingface.co
echo    2. Visit this model page and click "Request access":
echo         https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod
echo    3. Once approved, go to  https://huggingface.co/settings/tokens
echo       and create a Read token.
echo    4. Paste it below.
echo.
echo  (You can skip this now and add your token to hf_token.txt later.)
echo.

if exist "hf_token.txt" (
    echo  [OK] hf_token.txt already exists, skipping.
) else (
    set /p "HF_TOKEN=  Paste HuggingFace token (or press Enter to skip): "
    if not "!HF_TOKEN!"=="" (
        echo !HF_TOKEN!> hf_token.txt
        echo  [OK] Token saved to hf_token.txt
    ) else (
        echo.> hf_token.txt
        echo  [SKIP] You can add your token to hf_token.txt later.
    )
)
echo.

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ===========================================
echo    Setup complete!
echo  ===========================================
echo.
echo  To launch Narrate Studio, double-click  run.bat
echo.
echo  First launch notes:
echo    - Orpheus model:    ~6 GB download on first use
echo    - Kokoro model:     ~400 MB download on first use
echo    - Chatterbox model: ~1 GB download on first use
echo    - Models are cached locally after the first download.
echo.
pause
