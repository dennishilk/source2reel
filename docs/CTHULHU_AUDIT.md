# Cthulhu capability audit — 2026-09-23

This document records the first physical-deployment decision set for Source2Reel.

## Target

- Arch Linux x86_64
- Linux 7.2.6-arch2-1 at audit time
- Ryzen 7 5800X3D
- Radeon RX 9060 XT 16 GiB
- approximately 31 GiB system RAM
- Wayland / Sway

Hardware-specific choices remain configuration. They are not episode-schema requirements.


## Hardware-aware bootstrap correction — 2026-09-23

The first physical bootstrap exposed a dependency-architecture bug: Kokoro was a mandatory core Python dependency, and its unqualified \`torch\` dependency allowed the Linux PyPI resolver to select a CUDA/NVIDIA Torch stack on an AMD-only machine.

The baseline is now explicitly hardware-aware:

- \`source2reel/hardware.py\` inspects PCI display devices and Vulkan availability.
- AMD + working Vulkan selects \`llama.cpp+vulkan/radv\`.
- NVIDIA + working Vulkan selects \`llama.cpp+vulkan\`; Source2Reel does not infer CUDA merely from Linux/x86_64 or NVIDIA presence.
- no usable Vulkan path selects the CPU fallback.
- \`tools/bootstrap-arch.sh\` installs vendor-neutral packages first, then installs \`vulkan-radeon\` only when AMD is detected and \`vulkan-intel\` only when Intel is detected.
- the baseline does not auto-install CUDA, cuDNN, NCCL, ROCm or PyTorch.
- the generated core \`uv.lock\` is scanned before sync; Torch/NVIDIA/CUDA runtime entries abort the baseline install.
- Kokoro now lives in a separate optional \`.venv-voice-kokoro\` created by \`tools/setup-voice.sh\`.
- the voice setup installs Torch explicitly from the official CPU wheel index and installs Kokoro with \`--no-deps\` after its non-Torch dependencies are present.

This is a Source2Reel capability, not a Cthulhu special case. The same detection layer reports AMD, NVIDIA, Intel/other and CPU-only situations. Cthulhu remains the first physical acceptance target.


## GPU identity and ROCm status

AMD's current ROCm documentation lists the Radeon RX 9060 XT as:

- architecture: RDNA4
- LLVM target: `gfx1200`
- ROCm compute support: supported

Important qualification: AMD's official ROCm OS matrices list specific Ubuntu/RHEL/SLES/etc. releases; Arch Linux is not an AMD-certified ROCm target even though Arch packages ROCm directly. Therefore:

- GPU support: official upstream AMD support
- Arch distro combination: usable and well packaged, but not an AMD-certified OS combination

Arch currently provides ROCm 7.2.x components including `hip-runtime-amd`, `hipblas`, `rocblas`, and a ROCm-enabled PyTorch package.

## Primary local LLM runtime decision

### Baseline: llama.cpp + Vulkan/RADV

Use the Arch packages first:

```bash
sudo pacman -S --needed llama-cpp ggml-vulkan vulkan-radeon vulkan-tools
```

Reasons:

1. It avoids making the entire production system depend on the ROCm user-space stack.
2. Mesa/RADV is native to current Arch and already the normal graphics stack on Cthulhu.
3. Arch documents `llama-cpp` + `ggml-vulkan` as the Vulkan inference route.
4. Source2Reel already talks to an OpenAI-compatible localhost service, so the backend remains replaceable.
5. GGUF files are easy to pin by exact filename and SHA-256.
6. One-user batch research/storyboard generation does not need a multi-user serving stack.

Start with the packaged `llama-server`. Only build upstream llama.cpp manually if a required multimodal/model feature is missing from the packaged release. If a source build becomes necessary, pin the exact commit in `PROJECT_STATE.md`.

### Secondary benchmark: llama.cpp + HIP/ROCm

Arch also packages `ggml-hip`. Benchmark it after the Vulkan baseline works:

```bash
sudo pacman -S --needed rocm-core hip-runtime-amd hipblas rocblas ggml-hip rocminfo
```

Do not assume HIP is faster merely because ROCm officially supports gfx1200. Measure prompt processing, token generation, VRAM, stability and cold-start behavior with the same model/context.

### Optional convenience provider: Ollama

Arch packages both `ollama-vulkan` and `ollama-rocm`. Source2Reel has an Ollama adapter, but Ollama is not the first canonical baseline because direct llama.cpp + GGUF provides simpler artifact provenance and launch control.

### Not baseline: vLLM / SGLang

These frameworks can be valuable for high-throughput serving, but Source2Reel is a single-workstation batch production pipeline. Their larger Python/ROCm dependency surface is unnecessary for the first reproducible baseline on 16 GiB VRAM.

## Model sizing on 16 GiB VRAM

The goal is quality without turning every storyboard revision into a memory-management exercise.

### First multimodal model

`Qwen/Qwen3-VL-8B-Instruct`

Why:
- image + text input in one model
- Apache-2.0 upstream
- enough room at 4-bit GGUF for model, vision projector and practical KV cache inside 16 GiB
- suitable for project screenshots, terminal images and evidence selection

Freeze the exact GGUF/mmproj only after a physical quality test. Record source revision, quantization, filenames and SHA-256.

### Text-planning quality candidate

`Qwen/Qwen3-14B`

Upstream model card: 14.8B parameters, Apache-2.0.

A 4-bit GGUF is realistic on Cthulhu, but 32k FP16 KV cache plus weights can approach the 16 GiB VRAM ceiling. Start with a smaller context (for example 12k–16k) and/or quantized KV cache. Source2Reel already batches evidence, so it does not need to stuff an entire repository into one context.

### 24B class

A 24B model such as Mistral Small 3.2 can run quantized with partial CPU offload, but it is not the first baseline. At 4-bit its weights alone consume most of 16 GiB VRAM; system RAM allows it, but latency rises and the benefit must be demonstrated against the 14B path.

## PyTorch ROCm

AMD's current PyTorch/ROCm selector includes the RX 9060 XT (`gfx1200`). Arch currently ships `python-pytorch-rocm`.

This makes PyTorch ROCm a valid experimental tool on Cthulhu, with the same OS qualification noted above: the GPU is supported upstream, Arch itself is not an AMD-certified ROCm OS.

Verification after optional install:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("HIP:", torch.version.hip)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
PY
```

PyTorch intentionally exposes HIP devices through much of the `torch.cuda` API.

## Kokoro TTS device decision

Keep the permanent voice on CPU for the first frozen baseline. The Kokoro/Torch stack is installed in the separate optional `.venv-voice-kokoro` environment and is not part of the Source2Reel core dependency graph.

Reasons:
- Kokoro-82M is small relative to the LLM.
- TTS is generated scene-by-scene and does not justify coupling the voice path to ROCm.
- CPU TTS keeps the permanent voice operational while the GPU is occupied by the planner/model.
- It avoids mixing a Python ROCm stack into the isolated Python-3.12 voice environment.

Physical gate: measure real-time factor on the 5800X3D. Move Kokoro to GPU only if CPU generation is demonstrably a production bottleneck.

## FFmpeg AMD hardware encoding

Use Mesa VA-API, not AMF, as the Linux hardware-encoding path.

AMD's Linux guidance moved toward Mesa multimedia/VA-API and away from AMF for ordinary Linux encoding. Arch FFmpeg exposes VA-API and the open AMD driver uses `radeonsi`.

Verification:

```bash
sudo pacman -S --needed libva-utils
vainfo --display drm --device /dev/dri/renderD128
ffmpeg -hide_banner -encoders | grep -E 'h264_vaapi|hevc_vaapi|av1_vaapi'
```

Smoke encode:

```bash
ffmpeg -y \
  -vaapi_device /dev/dri/renderD128 \
  -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=5 \
  -vf 'format=nv12,hwupload' \
  -c:v h264_vaapi -qp 18 \
  /tmp/source2reel-vaapi-test.mp4
```

For the canonical YouTube master, keep `libx264` as the initial default because quality/reproducibility matters more than encode speed at 1080p30. VA-API is an excellent preview/fast-build option and can become the default later if physical quality comparisons justify it.

## Likely bottlenecks

1. LLM quality and prompt processing, not final video encoding.
2. 16 GiB VRAM when moving from 8B to 14B+ models with long contexts.
3. KV-cache growth; evidence batching is therefore an architectural feature, not only a prompt convenience.
4. Multimodal vision tokenization for many screenshots/video keyframes.
5. Large repository ingestion if unbounded; Source2Reel deliberately snapshots/batches sources.
6. CPU/GPU contention if TTS and LLM are unnecessarily placed on the same accelerator.

## Physical verification order

Run these on Cthulhu before downloading a large production model:

```bash
uname -a
lspci -nnk | grep -A3 -E 'VGA|Display'
vulkaninfo --summary
vainfo --display drm --device /dev/dri/renderD128
ffmpeg -hide_banner -encoders | grep -E '264|265|av1|vaapi'
```

For optional ROCm verification:

```bash
rocminfo | grep -E 'Name:|gfx1200' | head -40
hipconfig --full
```

Then run the same small GGUF benchmark through Vulkan and HIP before choosing a frozen backend.

## Baseline decision

For Source2Reel v0.2 physical deployment:

- LLM runtime: llama.cpp
- GPU backend: Vulkan/RADV first
- LLM API: localhost OpenAI-compatible `llama-server`
- first multimodal candidate: Qwen3-VL-8B-Instruct, 4-bit GGUF class
- second text-quality candidate: Qwen3-14B, 4-bit GGUF class
- TTS: Kokoro-82M / Dennis Explainer profile on CPU
- final encode: libx264 initially
- fast preview encode: VA-API candidate
- ROCm/HIP: supported-GPU benchmark path, not a hard dependency

## 2026-09-23 — clean reinstall handoff

Physical evidence: system `/usr/bin/llama-server` 0.4.1-dev (`b29c606e28`)
detected AMD Radeon RX 9060 XT (RADV GFX1200); Qwen3-VL-8B-Instruct
Q4_K_M and a matching projector served localhost HTTP 200. The isolated
voice had Torch 2.14.0+cpu and Kokoro 0.9.4. First narration exposed
Misaki installing `en_core_web_sm` in the wrong (core) interpreter.
The new installer pins the English model in voice and keeps generated
runtime state inside the checkout. Its physical fresh-install test remains
outstanding.
