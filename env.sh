# shellcheck shell=bash
# source this in Git Bash / WSL-ish shells:  source /d/Dev/video-use/env.sh
# Puts the local ffmpeg build and the repo venv on PATH for the current shell.

VIDEO_USE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Git Bash wants POSIX-style paths; ffmpeg.exe itself accepts either.
export PATH="$VIDEO_USE_ROOT/.venv/Scripts:/d/Dev/ffmpeg/bin:$PATH"

# Keep every cache off C: — that disk has under 2 GB free.
export VIDEO_USE_WHISPER_MODEL_DIR="${VIDEO_USE_WHISPER_MODEL_DIR:-$VIDEO_USE_ROOT/.model_cache}"
export HF_HOME="${HF_HOME:-$VIDEO_USE_ROOT/.model_cache/hf}"
export VIDEO_USE_ASR="${VIDEO_USE_ASR:-whisper}"

echo "video-use env: ffmpeg + venv on PATH, ASR=$VIDEO_USE_ASR, models=$VIDEO_USE_WHISPER_MODEL_DIR"
