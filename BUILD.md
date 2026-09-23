# BUILD — Source2Reel on Arch Linux / Cthulhu

The current target machine is Cthulhu: Arch Linux x86_64, Ryzen 7 5800X3D, Radeon RX 9060 XT 16 GiB and about 31 GiB system RAM. Hardware-specific choices remain configurable and must not leak into episode files.

See \`docs/CTHULHU_AUDIT.md\` for the dated capability decision and support qualifications.

## 1. Hardware-aware core bootstrap

The primary local-AI path is llama.cpp + Vulkan. ROCm is optional.

\`\`\`bash
./tools/bootstrap-arch.sh
\`\`\`

The bootstrap installs vendor-neutral packages first, detects the GPU through PCI data, checks Vulkan, and then chooses the vendor-specific Vulkan package where appropriate:

- AMD → \`vulkan-radeon\`, preferred backend \`llama.cpp+vulkan/radv\`
- Intel → \`vulkan-intel\`, preferred backend \`llama.cpp+vulkan\`
- NVIDIA → no CUDA/NVIDIA packages are installed automatically; an existing working Vulkan ICD may use \`llama.cpp+vulkan\`
- no usable Vulkan path → \`llama.cpp+cpu\`

The baseline never installs CUDA, cuDNN, NCCL, ROCm or PyTorch implicitly.

Core packages are intentionally small. Python core dependencies currently exclude Kokoro and Torch.

The bootstrap runs:

\`\`\`bash
uv python install 3.12
uv lock --python 3.12
uv sync --python 3.12 --locked
\`\`\`

Before the sync, it rejects a generated core \`uv.lock\` containing Torch/NVIDIA/CUDA runtime entries.

## 2. Optional permanent voice stack

Kokoro is not a Source2Reel core dependency. Install the Dennis Explainer voice candidate explicitly:

\`\`\`bash
./tools/setup-voice.sh
\`\`\`

This creates:

\`\`\`text
.venv-voice-kokoro/
\`\`\`

separately from the core \`.venv\`.

The voice installer deliberately uses the official PyTorch CPU wheel index for Torch, then installs Kokoro's non-Torch dependencies and finally \`kokoro==0.9.4 --no-deps\`. The initial production design therefore remains Kokoro-on-CPU even while llama.cpp uses the AMD GPU.

eSpeak remains a smoke-test fallback only and is never promoted silently to the permanent Dennis Explainer Voice.

## 3. Hardware sanity checks

\`\`\`bash
python3 source2reel/hardware.py
vulkaninfo --summary
vainfo --display drm --device /dev/dri/renderD128
ffmpeg -hide_banner -encoders | grep -E 'h264_vaapi|hevc_vaapi|av1_vaapi'
\`\`\`

## 4. Doctor

\`\`\`bash
source2reel doctor
\`\`\`

Doctor reports separate sections for:

- required core tools/profile files
- detected GPU vendor/device, Vulkan state and selected inference backend
- local AI endpoint
- optional voice runtime and eSpeak preview fallback

A missing Kokoro runtime is reported as optional and does not by itself make the core installation unhealthy.

## 5. Local AI service

The engine expects a localhost API and remains independent of runtime.

Primary baseline: packaged llama.cpp + Vulkan/RADV on Cthulhu.

A production GGUF is not committed. After choosing a model, store it outside Git, calculate SHA-256 and record exact model/revision/quantization in \`PROJECT_STATE.md\` and \`LICENSES.md\`.

See \`docs/LOCAL_AI.md\`.

## 6. Optional ROCm/HIP benchmark

\`\`\`bash
./tools/bootstrap-rocm-arch.sh
\`\`\`

Do this only after the Vulkan baseline works. ROCm is not required for normal Source2Reel installation.

## 7. Create/review an episode

\`\`\`bash
source2reel create https://github.com/dennishilk/cisco9951-doom --review
source2reel build cisco9951-doom
\`\`\`

## 8. Revise without manual scene programming

\`\`\`bash
source2reel revise cisco9951-doom \
  "Make the hook stronger but keep all factual claims evidence-grounded." \
  --build
\`\`\`

Generated \`work/\`, \`output/\`, local voice environments, models and rendered media are intentionally excluded from Git.
