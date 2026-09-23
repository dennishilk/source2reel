#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-$ROOT/vendor/llama.cpp/build-vulkan/bin/llama-server}"
MODEL="${DENNIS_LLM_MODEL:-}"
MMPROJ="${DENNIS_LLM_MMPROJ:-}"
[[ -x "$SERVER" ]] || { echo "llama-server not found: $SERVER" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "Set DENNIS_LLM_MODEL to the selected GGUF." >&2; exit 2; }
args=("$SERVER" -m "$MODEL" -a qwen3-vl-8b-instruct -ngl 999 -c 32768 --host 127.0.0.1 --port 8080)
if [[ -n "$MMPROJ" ]]; then
  [[ -f "$MMPROJ" ]] || { echo "mmproj not found: $MMPROJ" >&2; exit 2; }
  args+=(--mmproj "$MMPROJ")
fi
exec "${args[@]}"
