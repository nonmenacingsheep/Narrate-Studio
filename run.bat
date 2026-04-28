@echo off
set HF_HOME=%~dp0.hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
for /f "usebackq delims=" %%T in ("%~dp0hf_token.txt") do set HF_TOKEN=%%T
"%~dp0.venv\Scripts\python.exe" "%~dp0tts_app.py"
