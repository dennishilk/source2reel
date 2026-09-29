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

The planner also receives a small deterministic set of verbatim `source_context`
passages from scoped original documents. These passages preserve framing,
terminology, grouping, menu/flow context and relationships that can disappear
when research is reduced to atomic facts. They are narrative context only:
they never authorize a factual proposition. Titles, summaries, narration,
diagrams and annotations still require selected evidence-grounded research
facts. The same bounded context is carried through direct planning and every
map/reduce level, while trusted fact metadata such as conditional
`operation_guard` is preserved through compaction. Mutually exclusive guards
for the same selector cannot share one storyboard scene; exhausted outline
recovery keeps one branch and lets grounded requested-coverage recovery restore
the other branch separately.

For an explicit causal "why" request, research distinguishes a documented
relationship from generic project purpose. If model research omits that
relationship, a bounded exact primary-source line containing an explicit
relation signal such as `because`, `causes`, `leads to` or `→` can be
recovered as a quoted research fact. This never licenses an inferred mechanism.

## Revision contract

`source2reel revise <project> "instruction"` sends the current episode JSON, research file and valid evidence refs to the local model. The model returns a complete revised episode JSON. Previous versions are stored under `revisions/`; scene code is not rebuilt manually.

## Static evidence invariant

For HERO/PROJECT_EVIDENCE/TERMINAL_EVIDENCE/HARDWARE_EVIDENCE, the renderer creates one fixed contained image for the whole scene. Duration changes do not alter X/Y/scale/crop. There is no Ken Burns path in these templates.
