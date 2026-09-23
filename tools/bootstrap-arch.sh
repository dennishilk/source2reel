#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

sudo pacman -S --needed   ffmpeg espeak-ng uv git   llama-cpp ggml-vulkan   vulkan-radeon vulkan-icd-loader vulkan-tools   libva-utils

cd "$ROOT"
uv python install 3.12
uv sync --python 3.12
"$ROOT/tools/install-cli.sh"

cat <<EOF

Source2Reel baseline bootstrap complete.

Primary local-AI path:
  llama.cpp + Vulkan/RADV

Before downloading a production GGUF:
  vulkaninfo --summary
  vainfo --display drm --device /dev/dri/renderD128
  source2reel doctor

See docs/CTHULHU_AUDIT.md and docs/LOCAL_AI.md.
EOF
