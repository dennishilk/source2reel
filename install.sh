#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOURCE2REEL_ROOT="$ROOT"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$ROOT/cache/torch"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/runtime/python"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$ROOT/models" "$ROOT/runtime/pids" "$ROOT/runtime/logs" "$ROOT/output"
"$ROOT/tools/bootstrap-arch.sh"
"$ROOT/tools/setup-voice.sh"
echo
echo "Source2Reel installation checks:"
"$ROOT/s2r" doctor
echo "Ready. Run ./s2r, ./s2r voice-test or ./s2r create <source>."
