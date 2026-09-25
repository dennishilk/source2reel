# Source2Reel

**Source2Reel is a local, evidence-first AI-assisted technical explainer production engine.**

It ingests authoritative project sources, builds a hashed evidence inventory, uses a replaceable local AI provider for research and storyboard generation, renders through a fixed template/profile system, generates local narration, derives timing from audio, and produces a deterministic 1920×1080 H.264/AAC explainer package.

Cisco CP-9951 DOOM is the first reference episode and acceptance test. It is not a one-off renderer.

## Normal workflow

```bash
./s2r
```

Choose **New episode**, paste a GitHub URL, website URL or local directory,
then press Enter for automatic story discovery and the default storyboard
review. The producer starts or reuses Source2Reel-managed local AI when its
built-in llama.cpp provider is configured. At the storyboard prompt, choose
Review, Build, or Quit to continue later. **Continue episode** lists saved
storyboards, and **Build episode** builds a selected storyboard directly.

The explicit CLI remains available for automation and advanced use:

```bash
./s2r create https://github.com/OWNER/PROJECT
```

Multiple authoritative sources can be combined:

```bash
./s2r create \
  https://www.example.com/project/ \
  https://github.com/OWNER/PROJECT \
  /path/to/original-media \
  --instructions "Focus on the hardware architecture and final result."
```

The pipeline performs:

```text
source ingestion
→ evidence inventory + hashes
→ optional local vision inspection of authentic media
→ local LLM research with evidence refs
→ local LLM episode/storyboard JSON
→ configured fixed scene templates
→ configured permanent local voice
→ narration-derived timing
→ FFmpeg H.264/AAC build
→ transcript + storyboard + evidence + provenance artifacts
```

Research receives the requested project subject and marks likely embedded
examples and generated artifacts as supporting evidence. These files remain
available to cite. When primary research exists, a deterministic budget limits
supporting facts and refs passed to the planner so repeated generated text
cannot win by volume. An explicit request to focus on an embedded example
removes that default limit.
This is a research hint, not a guarantee about an AI-generated storyboard.

## Context-safe local AI passes

Source2Reel budgets requests against `local_ai.context_size` instead of assuming
one large prompt will fit. Oversized research input is split automatically into
checkpointed parts, and oversized planner input uses evidence-grounded map/reduce
compaction before the final storyboard request. Completed parts are input-hashed
under `projects/<episode>/manifests/`, so reruns reuse valid work and retry only
the missing or changed part.

`create` and `build` report their active stage on stderr. Research batches,
planner compaction parts, and render scenes show known counters. On an interactive
terminal, a blocking step also updates its elapsed time about every seven seconds.
Redirected stderr receives only start and completion lines, without terminal
control sequences or periodic heartbeats. The final storyboard (`--review`) or
video path remains the only output on stdout.

## Core commands

```bash
./s2r doctor
./s2r create <PROJECT_URL_OR_DIRECTORY>
./s2r create <SOURCE> --review
./s2r build <project-slug>
./s2r revise <project-slug> "Make the hook stronger." --build
```

## Engine vs. production profile

**Source2Reel** is the reusable engine. The repository ships with the **Dennis Explainer** profile as the current default production configuration:

- `themes/dennis-dark.toml`
- `voices/dennis-explainer.toml`
- profile selection in `config/engine.toml`

Dennis-specific series branding is configuration, not a renderer dependency. A different profile can select another theme, voice and series label without modifying the reusable core.

Reusable functionality belongs in `source2reel/`, `themes/`, `voices/`, `schemas/`, `prompts/` and `config/`. Episode-specific information belongs in `projects/<episode>/`.

## Cisco reference MVP

```bash
./s2r build cisco-doom
```

The canonical command requires Kokoro. In a restricted smoke-test environment only:

```bash
./s2r build cisco-doom --preview-espeak
```

eSpeak is explicitly non-canonical and must never become the Dennis Explainer production voice.

See `docs/ARCHITECTURE.md`, `docs/LOCAL_AI.md`, `BUILD.md`, `VOICE.md`, `STYLE_GUIDE.md`, `LICENSES.md` and `PROJECT_STATE.md`.


## Installation and local state

```bash
git clone https://github.com/dennishilk/source2reel.git
cd source2reel
./install.sh
./s2r doctor
./s2r voice-test
./s2r ai start
./s2r ai status
./s2r stop all
```

`./s2r` opens the interactive producer when called without arguments. No global CLI or
Fish PATH change is necessary. The installer creates separate core and Kokoro
CPU voice environments, explicitly installs the English spaCy model into voice,
and validates the pipeline. `am_michael` remains pending audible approval.

Put GGUF models in ignored `models/`. Set `[local_ai] model_path` and
`mmproj_path` in ignored `config/local.toml`, or set `DENNIS_LLM_MODEL` and
`DENNIS_LLM_MMPROJ`. The system `llama-server` is preferred; set
`LLAMA_SERVER` to override it. Logs and process records are under `runtime/`;
Hugging Face, uv and Torch caches are under `cache/`; voice samples are under
`output/`. `./s2r clean` previews generated-state removal; `--yes` confirms,
while models require an additional `--models` flag. See `BUILD.md`.
