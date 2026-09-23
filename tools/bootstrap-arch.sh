#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Install only vendor-neutral baseline packages first. No CUDA/ROCm stack is
# part of the normal Source2Reel baseline.
sudo pacman -S --needed \
  ffmpeg espeak-ng uv git pciutils \
  llama-cpp ggml-vulkan \
  vulkan-icd-loader vulkan-tools \
  libva-utils

printf '\nInitial hardware detection:\n'
python3 "$ROOT/source2reel/hardware.py" --format text || true

GPU_VENDOR="$(python3 "$ROOT/source2reel/hardware.py" --format env | sed -n 's/^SOURCE2REEL_GPU_VENDOR=//p')"
case "$GPU_VENDOR" in
  amd)
    echo "AMD GPU detected: installing Mesa RADV Vulkan driver."
    sudo pacman -S --needed vulkan-radeon
    ;;
  intel)
    echo "Intel GPU detected: installing Mesa Intel Vulkan driver."
    sudo pacman -S --needed vulkan-intel
    ;;
  nvidia)
    cat <<'MSG'
NVIDIA GPU detected.
Source2Reel does not auto-install CUDA, cuDNN, NCCL, or proprietary NVIDIA
packages. The baseline remains llama.cpp + Vulkan when a working NVIDIA
Vulkan ICD is already configured; otherwise the detector selects CPU mode.
MSG
    ;;
  *)
    echo "No supported GPU vendor detected; Source2Reel will retain a CPU fallback."
    ;;
esac

printf '\nFinal hardware detection:\n'
python3 "$ROOT/source2reel/hardware.py" --format text || true

cd "$ROOT"
uv python install 3.12
uv lock --python 3.12

# Safety gate: the normal core lock must not contain a GPU-vendor Python stack.
# Voice/Torch is installed only by tools/setup-voice.sh in a separate environment.
if grep -Eqi 'name = "torch"|nvidia[-_]|cuda|cudnn|nccl' uv.lock; then
  echo "ERROR: core uv.lock contains Torch/NVIDIA/CUDA runtime entries; refusing baseline sync." >&2
  exit 3
fi
uv sync --python 3.12 --locked
"$ROOT/tools/install-cli.sh"

cat <<EOF

Source2Reel core bootstrap complete.

The core environment intentionally contains no Kokoro/Torch stack.
Install the optional permanent-voice candidate separately with:
  ./tools/setup-voice.sh

Before downloading a production GGUF:
  vulkaninfo --summary
  vainfo --display drm --device /dev/dri/renderD128
  source2reel doctor

See docs/CTHULHU_AUDIT.md and docs/LOCAL_AI.md.

EOF
