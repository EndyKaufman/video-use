# dot-source this in PowerShell:  . D:\Dev\video-use\env.ps1
# Puts the local ffmpeg build and the repo venv on PATH for the current shell.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$env:PATH = "D:\Dev\ffmpeg\bin;$root\.venv\Scripts;$env:PATH"

# Keep every cache off C: — that disk has under 2 GB free.
if (-not $env:VIDEO_USE_WHISPER_MODEL_DIR) { $env:VIDEO_USE_WHISPER_MODEL_DIR = "$root\.model_cache" }
if (-not $env:HF_HOME)                     { $env:HF_HOME = "$root\.model_cache\hf" }
if (-not $env:VIDEO_USE_ASR)               { $env:VIDEO_USE_ASR = "whisper" }

Write-Host "video-use env: ffmpeg + venv on PATH, ASR=$($env:VIDEO_USE_ASR), models=$($env:VIDEO_USE_WHISPER_MODEL_DIR)"
