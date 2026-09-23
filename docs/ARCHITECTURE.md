# Source2Reel — Permanent Architecture

Source2Reel is the reusable production engine. Cisco DOOM is Episode 001 and a reference acceptance test; Cisco-specific facts and assets live under `projects/cisco-doom/` only.

```text
project URL(s) / local directory
            │
            ▼
        INGESTION
    repo / site / files
            │
            ▼
   EVIDENCE INVENTORY ─────────────┐
 hashes, excerpts, media metadata  │
            │                      │
            ▼                      │
   LOCAL LLM RESEARCH              │
 supported facts + phase labels    │
            │                      │
            ▼                      │
   LOCAL LLM STORYBOARD            │
 episode.json + evidence refs      │
            │                      │
            ▼                      │
    TEMPLATE REGISTRY ◄────────────┘
 fixed profile layouts
            │
            ├────► configured fixed voice ─► audio duration
            │
            ▼
 scene frames / controlled diagrams
            │
            ▼
         FFmpeg
   H.264 / AAC / 1920×1080
            │
            ▼
 MP4 + transcript + storyboard + evidence/provenance manifests
```

## Non-negotiable separation

`source2reel/`, `themes/`, `voices/`, `config/`, `prompts/` and `schemas/` are reusable infrastructure.

`projects/<episode>/` contains episode data: source/evidence records, research, episode JSON, reference assets and revision history. Generated `work/` and `output/` directories are ignored by Git.

No project name, Cisco path, Cisco fact or Cisco layout is required by the reusable renderer.

## Production profiles

The core reads a profile from `config/engine.toml`. A profile selects:

- theme
- voice
- series label

The repository default is the Dennis Explainer profile. This preserves one permanent series identity for Dennis's productions without hard-coding that branding into Source2Reel itself.

## Local AI provider abstraction

The planner calls a small `LLMProvider` interface. Two local adapters ship initially:

- `openai_compat`: intended for `llama-server` from llama.cpp and other local OpenAI-compatible servers.
- `ollama`: direct local Ollama API.

Provider/runtime/model selection is configuration, not renderer code.

## Evidence contract

Every evidence item receives a stable ref such as `E0007` plus path, SHA-256, type and metadata/excerpt. Research facts and factual scenes must cite valid evidence refs. This does not automatically prove a generated claim true, but makes unsupported output detectable and reviewable.

The research pass labels facts as `development`, `final`, `background` or `unknown`. The storyboard pass is instructed not to merge prototype architecture into final architecture.

## Revision contract

`source2reel revise <project> "instruction"` sends the current episode JSON, research file and valid evidence refs to the local model. The model returns a complete revised episode JSON. Previous versions are stored under `revisions/`; scene code is not rebuilt manually.

## Static evidence invariant

For HERO/PROJECT_EVIDENCE/TERMINAL_EVIDENCE/HARDWARE_EVIDENCE, the renderer creates one fixed contained image for the whole scene. Duration changes do not alter X/Y/scale/crop. There is no Ken Burns path in these templates.
