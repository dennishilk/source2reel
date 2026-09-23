#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/llama.cpp"
REF="${LLAMA_CPP_REF:-master}"
mkdir -p "$ROOT/vendor"
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
git -C "$DEST" fetch --tags origin
git -C "$DEST" checkout "$REF"
cmake -S "$DEST" -B "$DEST/build-vulkan" -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build-vulkan" --config Release -j"$(nproc)"
printf '\nllama.cpp ref: '; git -C "$DEST" rev-parse HEAD
printf 'server: %s\n' "$DEST/build-vulkan/bin/llama-server"
