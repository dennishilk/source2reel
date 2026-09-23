# PROJECT_STATE

## 2026-09-23 — Source2Reel v0.2 canonical baseline

Canonical repository: `dennishilk/source2reel` (private during initial development).

Source2Reel is the reusable local, evidence-first AI-assisted technical explainer production engine. The Dennis Explainer visual identity and voice remain the default production profile rather than the generic product name.

Implemented reusable infrastructure:

- `source2reel` CLI with `create`, `inventory`, `build`, `revise`, and `doctor`
- multiple URL/local-directory sources per episode
- bounded same-origin website snapshotting
- GitHub repository clone ingestion
- deterministic evidence inventory with SHA-256, excerpts and media metadata
- optional local multimodal media inspection
- replaceable local LLM provider interface
- OpenAI-compatible local server adapter
- Ollama adapter
- evidence-grounded research pass with development/final/background/unknown phase labels
- machine-readable `episode.json` storyboard contract
- scene schema validation and evidence-ref validation
- permanent scene-type registry
- generic static evidence renderer with position/scale/crop lock
- generic code, title, summary, graph and diagram-family renderers
- configurable profile selecting theme, voice and series label
- Kokoro narration path and normalization
- narration-derived scene timing
- per-scene assembly and final H.264/AAC MP4
- transcript, storyboard, evidence, render and provenance manifests
- revision workflow that edits episode JSON through the local model and preserves previous versions

Cisco reference status:

- Cisco-specific data lives under `projects/cisco-doom/`
- the reference evidence image is copied byte-for-byte from the authoritative Cisco project repository
- the one-scene MVP builds through the same generic `source2reel build` path future episodes use
- no rendered MP4/WAV/work directories belong in Git
- eSpeak remains smoke-test-only; the canonical voice candidate is Kokoro

### Cthulhu capability audit — 2026-09-23

Verified from current upstream/distro documentation:

- Radeon RX 9060 XT: RDNA4 / `gfx1200`
- the GPU is currently supported by ROCm
- Arch Linux itself is not an AMD-certified ROCm OS, despite having current native ROCm packages
- Arch currently packages `llama-cpp`, `ggml-vulkan`, `ggml-hip`, `ollama-vulkan`, `ollama-rocm`, ROCm 7.2.x components and `python-pytorch-rocm`
- Linux hardware video encoding should use Mesa VA-API rather than making AMF a dependency

Physical baseline decision:

- primary LLM runtime: llama.cpp
- primary GPU backend: Vulkan/RADV
- ROCm/HIP: optional measured comparison path
- first multimodal model class: Qwen3-VL-8B-Instruct 4-bit GGUF
- second text-planning candidate: Qwen3-14B 4-bit GGUF
- Kokoro TTS: CPU for the first freeze
- canonical final encode: libx264 initially
- VA-API: preview/fast-build candidate

See `docs/CTHULHU_AUDIT.md`.

Not frozen yet:

- exact GGUF and mmproj filenames/hashes
- local AI model quality acceptance
- Vulkan-vs-HIP performance on physical Cthulhu
- final Dennis voice approval after a real Kokoro render
- final hardware-encode quality decision
- diagram/graph template polish

Next production gates:

1. Pull the canonical repository on Cthulhu.
2. Run the hardware sanity checks in `docs/CTHULHU_AUDIT.md`.
3. Install the baseline packages with `tools/bootstrap-arch.sh`.
4. Start a small local Vulkan model and run `source2reel doctor`.
5. Run `source2reel create` against the Cisco sources and review evidence-grounded research/storyboard quality.
6. Render canonical Kokoro narration and freeze the voice if approved.
7. Expand Cisco into the complete reference episode.
8. Run Apollo DSKY or a World Observer through the same engine without modifying reusable core code.
