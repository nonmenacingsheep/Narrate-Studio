@echo off
setlocal enabledelayedexpansion
title Narrate Studio — Setup
color 0B

echo.
echo  ===========================================
echo    Narrate Studio — First-time Setup
echo  ===========================================
echo.

:: ── Find a compatible Python (3.10–3.13) ──────────────────────────────────────
set PYTHON=

:: Try the Python Launcher (py) with specific versions first
for %%V in (3.13 3.12 3.11 3.10) do (
    if "!PYTHON!"=="" (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON=py -%%V
            for /f "tokens=2" %%v in ('py -%%V --version 2^>^&1') do set PY_VER=%%v
        )
    )
)

:: Fall back to plain "python" if py launcher didn't find anything
if "!PYTHON!"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
        for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
            set PY_MAJOR=%%a
            set PY_MINOR=%%b
        )
        if !PY_MINOR! GEQ 10 if !PY_MINOR! LEQ 13 set PYTHON=python
    )
)

if "!PYTHON!"=="" (
    echo  [ERROR] No compatible Python version found (need 3.10-3.13^).
    echo.
    echo  Install Python 3.13 with:
    echo    winget install Python.Python.3.13
    echo.
    echo  Or download from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  [OK] Found Python %PY_VER% (using: !PYTHON!)
echo.

:: ── Create virtual environment ────────────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    echo  [OK] Virtual environment already exists, skipping creation.
) else (
    echo  Creating virtual environment...
    !PYTHON! -m venv .venv
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

:: ── PyTorch (try CUDA versions in order, fall back to CPU) ───────────────────
echo  Installing PyTorch with GPU (CUDA) support...
echo  (Downloading ~2.5 GB — this will take a few minutes)
echo.

set TORCH_OK=0

echo  Trying CUDA 12.8...
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet --no-cache-dir
if not errorlevel 1 set TORCH_OK=1

if "%TORCH_OK%"=="0" (
    echo  Trying CUDA 12.4...
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124 --quiet --no-cache-dir
    if not errorlevel 1 set TORCH_OK=1
)

if "%TORCH_OK%"=="0" (
    echo  Trying CUDA 12.1...
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet --no-cache-dir
    if not errorlevel 1 set TORCH_OK=1
)

if "%TORCH_OK%"=="0" (
    echo.
    echo  [WARN] No CUDA build found. Installing CPU-only PyTorch.
    echo  The app will work but generation will be very slow without a GPU.
    echo.
    .venv\Scripts\pip install torch --quiet --no-cache-dir
)

echo  [OK] PyTorch installed.
echo.

:: ── TTS models (installed first so their numpy/etc pins take priority) ────────
:: Use --only-binary for spacy/thinc/blis so pip never tries to compile them from source.
echo  Installing Kokoro...
.venv\Scripts\pip install kokoro --only-binary spacy,thinc,blis,cymem,murmurhash,preshed,srsly,catalogue --quiet --no-cache-dir
if errorlevel 1 (
    echo  [WARN] Kokoro install had issues. Trying without spaCy...
    .venv\Scripts\pip install kokoro --no-deps --quiet --no-cache-dir
    .venv\Scripts\pip install "misaki[en]" --only-binary spacy,thinc,blis,cymem,murmurhash,preshed,srsly,catalogue --quiet --no-cache-dir 2>nul
    .venv\Scripts\pip install loguru einops piper-phonemize --quiet --no-cache-dir 2>nul
)
echo  [OK] Kokoro installed.

echo  Installing Chatterbox...
.venv\Scripts\pip install chatterbox-tts --quiet --no-cache-dir
echo  [OK] Chatterbox installed.
echo.

:: ── Core dependencies (installed after TTS packages to respect their version pins) ──
echo  Installing core dependencies...
.venv\Scripts\pip install transformers accelerate snac soundfile PyQt6 --quiet --no-cache-dir
echo  [OK] Core dependencies installed.
echo.

:: ── HuggingFace token ────────────────────────────────────────────────────────
echo  ===========================================
echo    HuggingFace Token (Orpheus model)
echo  ===========================================
echo.
echo  The Orpheus TTS model requires a free HuggingFace account.
echo.
echo  Steps:
echo    1. Create a free account at  https://huggingface.co
echo    2. Request access at:
echo         https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod
echo    3. Once approved, create a Read token at:
echo         https://huggingface.co/settings/tokens
echo    4. Paste it below.
echo.
echo  (You can skip this and add your token to hf_token.txt later.)
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
        echo  [SKIP] Add your token to hf_token.txt later.
    )
)
echo.

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo  ===========================================
echo    Setup complete!
echo  ===========================================
echo.
echo  Double-click run.bat to launch Narrate Studio.
echo.
echo  First launch will download TTS models automatically:
echo    Orpheus    ~6 GB  (requires HuggingFace token)
echo    Kokoro     ~400 MB
echo    Chatterbox ~1 GB
echo.
pause
