# Source2Reel

**Source2Reel is a local, evidence-first AI-assisted technical explainer production engine.**

It ingests authoritative project sources, builds a hashed evidence inventory, uses a replaceable local AI provider for research and storyboard generation, renders through a fixed template/profile system, generates local narration, derives timing from audio, and produces a deterministic 1920×1080 H.264/AAC explainer package.

Cisco CP-9951 DOOM is the first reference episode and acceptance test. It is not a one-off renderer.

## Normal workflow

```bash
source2reel create https://github.com/OWNER/PROJECT
```

Multiple authoritative sources can be combined:

```bash
source2reel create \
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

## Core commands

```bash
source2reel doctor
source2reel create <PROJECT_URL_OR_DIRECTORY>
source2reel create <SOURCE> --review
source2reel build <project-slug>
source2reel revise <project-slug> "Make the hook stronger." --build
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
source2reel build cisco-doom
```

The canonical command requires Kokoro. In a restricted smoke-test environment only:

```bash
source2reel build cisco-doom --preview-espeak
```

eSpeak is explicitly non-canonical and must never become the Dennis Explainer production voice.

See `docs/ARCHITECTURE.md`, `docs/LOCAL_AI.md`, `BUILD.md`, `VOICE.md`, `STYLE_GUIDE.md`, `LICENSES.md` and `PROJECT_STATE.md`.


## Installation split

The reusable core and permanent voice are deliberately separate:

\`\`\`bash
./tools/bootstrap-arch.sh   # core + hardware detection + llama.cpp/Vulkan baseline
./tools/setup-voice.sh      # optional Kokoro CPU voice environment
\`\`\`

The normal core install does not depend on Kokoro or Torch and does not auto-install CUDA/NVIDIA runtime packages. \`source2reel doctor\` reports the optional voice stack separately.
