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
- the one-scene MVP builds through the same generic `source2reel build` path future episodes use
- no rendered MP4/WAV/work directories belong in Git
- eSpeak remains smoke-test-only; the canonical voice candidate is Kokoro

Not frozen yet:

- final Dennis voice approval awaits an audible Kokoro render on Cthulhu
- local AI runtime/model baseline awaits physical Cthulhu benchmarking
- AMD inference and hardware-encoding paths await capability verification
- diagram/graph templates are functional early implementations, not a final polish freeze

Next production gates:

1. Complete the physical Cthulhu AMD/runtime capability audit.
2. Install and verify the selected local AI runtime from this repository.
3. Run `source2reel create` against the Cisco sources and review evidence-grounded research/storyboard quality.
4. Render canonical Kokoro narration and freeze the voice if approved.
5. Expand Cisco into the complete reference episode.
6. Run Apollo DSKY or a World Observer through the same engine without modifying reusable core code.
