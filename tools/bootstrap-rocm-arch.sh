#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
This installs the optional ROCm/HIP benchmark path.
It is NOT required for the Source2Reel Vulkan baseline.

The RX 9060 XT (gfx1200) is supported by current ROCm, but Arch Linux is
not one of AMD's certified ROCm operating systems. Arch packages this stack
natively; use it as a measured alternative, not as an assumption.
EOF

sudo pacman -S --needed   rocm-core hip-runtime-amd hipblas rocblas rocminfo   ggml-hip python-pytorch-rocm

printf '
ROCm device summary:
'
rocminfo | grep -E 'Name:|gfx1200' | head -40 || true

printf '
PyTorch ROCm summary:
'
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
