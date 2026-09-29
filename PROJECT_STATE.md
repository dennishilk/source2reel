# PROJECT_STATE

## 2026-09-29 — planner narrative source context

The real SPDIF-Fix production exposed a reusable information-loss boundary:
research correctly retained grounded atomic facts, but the storyboard planner
could lose the original README framing and conditional workflow context and
therefore produce a technically grounded fact dump instead of a coherent
explanation.

Permanent fix:

- the planner deterministically selects at most four topically relevant,
  verbatim passages from scoped original documents, bounded to 1400 characters
  per passage and at most 4200 characters total; the total budget scales down
  automatically for smaller configured local-model contexts;
- these `source_context` passages are narrative-only and cannot authorize new
  factual claims; storyboard facts remain the sole factual authority;
- the same source context is injected into direct planning, planner
  map/reduce compaction, multipart outlines and scene generation;
- source-context refs stay in the planner evidence index for provenance without
  becoming valid scene citations by themselves;
- conditional fact metadata such as `operation_guard` survives planner
  records, compaction capsules and compact research, and is attached to both
  naturalized and raw exact code facts;
- facts from mutually exclusive values of the same operation guard cannot be
  merged into one scene; exhausted outline recovery deterministically keeps one
  branch and existing requested-coverage recovery restores the omitted branch
  separately;
- an explicit causal "why" request is represented as a relationship topic
  rather than generic purpose when the request itself names a causal relation;
  a missed relationship can be recovered only from an explicit bounded
  primary-source passage (for example `because`, `causes`, `leads to` or
  `→`), including a short adjacent passage when one documented relationship
  is split across neighboring source lines; never from inferred mechanism;
- planner/compaction prompts explicitly use context for framing and grouping
  while forbidding facts invented from context alone.

The implementation is source-agnostic; README receives only a modest generic
overview-source preference and every selected passage must still overlap the
requested subject, instructions or grounded research.

## 2026-09-24 — interactive producer (feature branch, pending Cthulhu acceptance)

The bare `./s2r` entry point offers New, Continue, and Build episode menus,
plus the existing local AI controls, Doctor, and Voice test. New episode accepts
one or more sources, keeps blank instructions empty, and defaults to stopping
after the storyboard. Review reads the saved episode locally; Build delegates to
the existing pipeline. The wizard starts or reuses managed llama.cpp for the
built-in OpenAI-compatible provider. Explicit CLI behavior is unchanged.

## 2026-09-23 — post-acceptance visual product changes

Physical acceptance freeze: `889d772919d713a178f4a7bc5bf0ce4ef9b143ad`.
The twelve committed Episode 001 JSON fixtures remain immutable. Cthulhu
proved 12 scenes, about 283.6 seconds, 1920×1080, video seek offsets, diagram
nodes, narration and final assembly. The permanent Kokoro voice is approved:
`am_michael`, speed 0.94, isolated CPU runtime.

This descendant adds deterministic narration captions, a large metadata-led
endcard, explicit schema/prompt support for video offsets, and diagram-node
validation. Episode 001 endcard copy lives in a new `presentation.json`.
Synthetic FFmpeg tests verify caption pixels; the final full render with
original large media remains a Cthulhu gate. Build with
`./s2r build cisco-doom-episode-001` after pulling the changes.

Final polish: authentic video scenes now use the same burned-in caption
pipeline as static scenes. The episode presentation sidecar supplies an ASCII
SSH title correction while preserving the frozen episode JSON and endcard.

## 2026-09-23 — installer and runtime consolidation (pending Cthulhu)

- `./install.sh` joins hardware-aware bootstrap, local core, Kokoro CPU
  voice, explicit English spaCy model, pipeline validation and doctor.
- `./s2r` is the repository-local CLI and terminal session starter.
  The system llama-server is preferred; Vulkan/RADV is selected when detected.
- Hugging Face, uv and Torch caches, model storage, logs, PID records and
  output resolve under the checkout.
- `ai start/status` and `stop all` track PID and kernel start tick; cleanup
  previews generated state and requires `--yes` for removal.
- Voice candidate `am_michael` remains pending audible approval.
- Fresh install, local model placement, actual server start and narration
  listening on Cthulhu remain outstanding.


## 2026-09-23 — hardware-aware bootstrap / dependency fix

Physical Cthulhu deployment exposed a core dependency bug: \`kokoro==0.9.4\` was mandatory in \`pyproject.toml\`; Kokoro requires unqualified \`torch\`, and Linux PyPI resolution attempted to install a large NVIDIA/CUDA runtime on the AMD-only Cthulhu machine.

Fixed architecture:

- Source2Reel core no longer depends on Kokoro or Torch.
- core Python dependency resolution is limited to reusable engine requirements; Pillow remains the current Python core dependency.
- new generic hardware detection reports PCI GPU vendor/device, Vulkan availability and selected inference backend.
- AMD + Vulkan selects \`llama.cpp+vulkan/radv\`; NVIDIA is never inferred from Linux/x86_64 and does not trigger an automatic CUDA install; CPU-only fallback is explicit.
- \`tools/bootstrap-arch.sh\` performs hardware detection before vendor-specific Vulkan setup.
- baseline bootstrap generates and scans the core \`uv.lock\` before syncing; Torch/NVIDIA/CUDA runtime entries are rejected.
- Kokoro is installed explicitly through \`tools/setup-voice.sh\` into \`.venv-voice-kokoro\`.
- the voice installer selects CPU Torch from the official CPU wheel index and keeps the TTS stack isolated from core \`uv sync\`.
- \`source2reel doctor\` separates required core checks from optional voice checks.
- eSpeak remains preview/smoke-test only.

Sandbox verification for this fix:

- 11 tests passed
- \`compileall\` passed
- shell syntax checks passed
- Cisco eSpeak preview still rendered H.264 1920x1080 + AAC 48 kHz
- a resolver smoke lock for the core dependency graph contained only Source2Reel + Pillow and no Torch/Kokoro/NVIDIA/CUDA/cuDNN/NCCL entries

This is not a claim that the physical AMD install is proven after the fix. Physical validation resumes on Cthulhu after this commit.

## Canonical baseline

Canonical repository: \`dennishilk/source2reel\` (private during initial development).

Source2Reel is the reusable local, evidence-first AI-assisted technical explainer production engine. The Dennis Explainer visual identity and voice remain the default production profile rather than the generic product name.

Reusable architecture remains:

- \`source2reel\` CLI with \`create\`, \`inventory\`, \`build\`, \`revise\`, and \`doctor\`
- source ingestion and deterministic evidence/provenance manifests
- replaceable local LLM provider interface
- OpenAI-compatible and Ollama adapters
- evidence-grounded research and machine-readable storyboard generation
- fixed scene-template registry and static authentic-evidence rendering
- configurable theme/voice/series profile
- narration-derived timing and FFmpeg H.264/AAC assembly
- revision workflow over the same episode specification

Cthulhu baseline remains:

- LLM runtime: llama.cpp
- primary GPU backend: Vulkan/RADV
- ROCm/HIP: optional measured comparison path
- first multimodal model class: Qwen3-VL-8B-Instruct 4-bit GGUF
- Kokoro TTS: CPU, optional voice environment, pending audible approval
- canonical final encode: libx264 initially
- VA-API: preview/fast-build candidate

Not frozen yet:

- exact GGUF/mmproj filenames and hashes
- local model quality acceptance
- Vulkan-vs-HIP physical performance
- final Dennis voice approval
- final hardware-encode quality decision

Next gate: repeat only the physical Source2Reel baseline bootstrap on Cthulhu and inspect the generated core lock before continuing.
