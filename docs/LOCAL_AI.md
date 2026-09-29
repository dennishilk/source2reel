# Local AI backend on Cthulhu

## Repository-local session

Physical Cthulhu evidence: system `/usr/bin/llama-server` 0.4.1-dev
(commit `b29c606e28`) detected Radeon RX 9060 XT RADV GFX1200. A
Qwen3-VL-8B-Instruct Q4_K_M GGUF with projector served HTTP 200 on localhost.
The consolidated session launcher still needs physical validation.

Put model files under ignored `models/` and configure paths in ignored
`config/local.toml` under `[local_ai]`, or use `DENNIS_LLM_MODEL` and
`DENNIS_LLM_MMPROJ`. `LLAMA_SERVER` can override system executable discovery.
`./s2r ai start` writes a PID record and log under `runtime/`;
`./s2r stop all` checks process identity before signaling. External
llama-server processes are untouched. `./s2r` provides the interactive
starter. Vulkan/RADV is the current default; CPU is available without Vulkan.
ROCm is not automatically installed or selected.

The managed llama.cpp server defaults to one parallel slot
(`[local_ai] parallel_slots = 1`). Source2Reel's production pipeline issues
model requests serially, so extra idle server slots add no throughput benefit
and can retain multiple large prompt contexts. Increase this only for a
separate workload that intentionally sends concurrent requests.


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


## Hardware detection and Python dependency boundary

Source2Reel now performs local hardware detection independently of the model provider. \`source2reel/hardware.py\` inspects PCI display controllers and \`vulkaninfo --summary\` and reports a selected baseline:

- AMD + Vulkan → \`llama.cpp+vulkan/radv\`
- other GPU + Vulkan → \`llama.cpp+vulkan\`
- no usable Vulkan → \`llama.cpp+cpu\`

The detector does not infer NVIDIA/CUDA from Linux or x86_64. The normal Python core environment contains no Kokoro or Torch dependency. This keeps local LLM runtime choice, Python rendering dependencies and optional TTS acceleration separate.

Kokoro is configured as an explicit CPU voice stack under \`.venv-voice-kokoro\`; ROCm remains an optional benchmark path rather than a core installation requirement.
