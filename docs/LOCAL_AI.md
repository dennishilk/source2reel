# Local AI backend on Cthulhu

Source2Reel does not hard-code a model runtime or one immutable LLM. Providers and model names live in configuration.

## Current architecture

The engine can talk to:

- a localhost OpenAI-compatible endpoint, intended initially for `llama-server`
- a localhost Ollama endpoint

This HTTP boundary is intentional. Switching Vulkan ↔ HIP/ROCm, changing quantization, or replacing the model does not alter episode data or renderer code.

## First physical milestone

Before a runtime is frozen, verify the actual Cthulhu hardware/software combination:

- Arch Linux x86_64
- Ryzen 7 5800X3D
- Radeon RX 9060 XT 16 GiB
- ~31 GiB system RAM

The current conservative baseline candidate is llama.cpp with Vulkan. ROCm/HIP, Ollama and PyTorch ROCm must be judged from current support for this exact GPU rather than assumed from generic AMD support.

## Model role

The selected local model must reliably perform:

- documentation/source-code analysis
- development-vs-final architecture distinction
- evidence inventory reasoning
- English narration generation
- machine-readable storyboard output
- revision of an existing episode
- optionally authentic image/screenshot inspection

The initial multimodal candidate remains Qwen3-VL-class 8B, but no model/runtime is frozen until physical tests on Cthulhu confirm compatibility, quality, VRAM behavior and reproducibility.

## Provenance freeze rule

When a model is accepted, record exact upstream repository/revision, model filename, quantization, SHA-256, runtime version/commit, backend and relevant launch arguments. Runtime/model payloads are downloaded locally and are never committed to this repository.
