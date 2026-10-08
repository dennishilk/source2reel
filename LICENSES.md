# LICENSES

Verified 2026-09-23. This file documents the reusable engine dependencies; it is not a blanket legal guarantee for every future episode asset.

## Kokoro-82M v1.0
- upstream: `hexgrad/Kokoro-82M`
- model release: v1.0, published 2025-01-27
- model/weights license: Apache-2.0 per upstream model card
- selected stock voice: `am_michael`
- do not describe the voice as a clone or imitation of a real person

## kokoro Python inference library
- upstream: `hexgrad/kokoro`
- pinned package: 0.9.4
- installation: optional isolated voice environment via `tools/setup-voice.sh`; not a core dependency
- license: Apache-2.0 per upstream/PyPI metadata

## Misaki G2P
- upstream: `hexgrad/misaki`
- license: Apache-2.0 per upstream package metadata

## eSpeak NG
- system runtime dependency used by English G2P fallback on the canonical Arch build
- Arch package: `espeak-ng`
- GPL-3.0-or-later project; the engine does not redistribute its binary

## FFmpeg
- system package only; license depends on the exact Arch build configuration
- record `ffmpeg -version` in release provenance before freezing a production baseline

## Fonts
- MVP references fonts installed by the operating system and does not redistribute font files.

## Cisco DOOM project media
- documentary project media comes from the user's own Cisco CP-9951 project/repository.
- Freedoom, Cisco firmware, commercial DOOM IWADs and other third-party materials must retain their own provenance and licenses; do not assume engine licensing covers episode assets.

## Local planning model candidates
The local LLM/VLM is replaceable and not redistributed by this repository.

Current first candidate:
- `Qwen/Qwen3-VL-8B-Instruct`
- upstream model card license: Apache-2.0
- purpose: local media inspection + factual research + storyboard generation
- status: candidate, not yet frozen

Alternative text-planning candidates currently documented:
- `Qwen/Qwen3-14B` — Apache-2.0 upstream
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — Apache-2.0 upstream

Before freezing any downloaded GGUF/quantization, record the exact source repository, filename, SHA-256, quantizer/converter, model revision and license. Do not assume a third-party quantization inherits provenance correctly without checking it.

## llama.cpp
- local inference/runtime candidate
- supports Linux, AMD HIP/ROCm and Vulkan backends upstream
- document the exact commit/version used on Cthulhu once frozen

## Ollama
- optional alternative local provider
- not required by episode files or renderer
- document exact version if selected for the frozen baseline


## PyTorch CPU runtime for optional voice stack
- installed only by `tools/setup-voice.sh`, not by the Source2Reel core resolver
- initial policy: official PyTorch CPU wheel index; no CUDA/NVIDIA runtime required
- exact Torch version is not frozen until physical Cthulhu voice validation; record it in provenance when the voice baseline is accepted
