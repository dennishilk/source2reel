#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

sudo pacman -S --needed \
  ffmpeg espeak-ng uv git cmake ninja base-devel \
  vulkan-radeon vulkan-icd-loader

cd "$ROOT"
uv python install 3.12
uv sync --python 3.12
"$ROOT/tools/install-cli.sh"

cat <<EOF

Source2Reel bootstrap complete.

Next:
  1. Build/start a local AI backend (see docs/LOCAL_AI.md).
  2. Put the selected GGUF/mmproj paths in environment variables or your service.
  3. Run: source2reel create <PROJECT_URL> --review

Canonical narration uses Kokoro. The first run may populate the local model cache.
EOF
