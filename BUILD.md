# BUILD — Source2Reel on Arch Linux / Cthulhu

The current target machine is Cthulhu: Arch Linux x86_64, Ryzen 7 5800X3D, Radeon RX 9060 XT 16 GiB and about 31 GiB system RAM. Hardware-specific AI/encoding choices remain configurable and must not leak into episode files.

See `docs/CTHULHU_AUDIT.md` for the dated capability decision and support qualifications.

## 1. Baseline packages

The primary local-AI path is llama.cpp + Vulkan/RADV. ROCm is optional.

```bash
./tools/bootstrap-arch.sh
```

Equivalent core packages include:

```bash
sudo pacman -S --needed \
  ffmpeg espeak-ng uv git \
  llama-cpp ggml-vulkan \
  vulkan-radeon vulkan-icd-loader vulkan-tools \
  libva-utils
```

Keep the system Python untouched for the voice environment. The pinned Kokoro environment uses an isolated Python 3.12 interpreter:

```bash
uv python install 3.12
uv sync --python 3.12
```

## 2. Hardware sanity checks

```bash
vulkaninfo --summary
vainfo --display drm --device /dev/dri/renderD128
ffmpeg -hide_banner -encoders | grep -E 'h264_vaapi|hevc_vaapi|av1_vaapi'
```

## 3. CLI

```bash
uv run source2reel --help
./tools/install-cli.sh
source2reel doctor
```

## 4. Local AI service

The engine expects a localhost API and remains independent of runtime.

Primary baseline: packaged llama.cpp + Vulkan/RADV.

A production GGUF is not committed. After choosing a model, store it outside Git, calculate SHA-256 and record exact model/revision/quantization in `PROJECT_STATE.md` and `LICENSES.md`.

If the packaged server cannot provide a required multimodal feature, `tools/build-llama-vulkan.sh` remains available for a pinned upstream build. Do not leave it tracking an unfrozen moving `master` in a production freeze.

See `docs/LOCAL_AI.md`.

## 5. Optional ROCm/HIP benchmark

```bash
./tools/bootstrap-rocm-arch.sh
```

Do this only after the Vulkan baseline works. Compare the same model and context on both paths before freezing a backend.

## 6. Create/review an episode

```bash
source2reel create https://github.com/dennishilk/cisco9951-doom --review
```

Then inspect the generated evidence/research/storyboard and build:

```bash
source2reel build cisco9951-doom
```

## 7. Revise without manual scene programming

```bash
source2reel revise cisco9951-doom \
  "Make the hook stronger but keep all factual claims evidence-grounded." \
  --build
```

## 8. Outputs

Each episode output directory contains at least:

- final MP4
- `transcript.txt`
- `storyboard.json`
- `evidence-manifest.json`
- `research.json` when available
- `render-manifest.json`
- `provenance.json`
- `ENGINE_LICENSES.md`

Generated `work/` and `output/` directories are intentionally ignored by Git.
