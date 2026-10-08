# Source2Reel

[![Tests](https://github.com/dennishilk/source2reel/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/dennishilk/source2reel/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Local AI](https://img.shields.io/badge/AI-local-93dfc1)](#privacy-and-local-processing)
[![1080p H.264 / AAC](https://img.shields.io/badge/Output-1080p%20H.264%20%2F%20AAC-76d2fa)](#what-you-get)
[![Watch the demo on YouTube](https://img.shields.io/badge/Demo-YouTube-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=94koql02-tM)

**Turn project sources into technical explainer videos, with local AI and traceable evidence.**

Source2Reel takes GitHub repositories, websites and local project files through
research, a reviewable storyboard, local narration and FFmpeg rendering. The
result is a 1920 × 1080 explainer with captions, authentic project media and
the source records behind its claims.

Built for developers, hardware tinkerers and people who want to explain how
their projects actually work.

[Demo](#watch-a-real-episode) · [Quick start](#quick-start) ·
[Commands](#commands) · [Privacy](#privacy-and-local-processing) ·
[Documentation](#documentation) · [Development](#development)

## Watch a real episode

[![Watch “I Made a €35 Cisco Office Phone Run DOOM — Here’s How” on Dennis Hilk's YouTube channel](https://i.ytimg.com/vi/94koql02-tM/hqdefault.jpg)](https://www.youtube.com/watch?v=94koql02-tM)

**[I Made a €35 Cisco Office Phone Run DOOM — Here’s How](https://www.youtube.com/watch?v=94koql02-tM)**

A real Source2Reel episode about my Cisco CP-9951 DOOM project, published on
[Dennis Hilk's channel](https://www.youtube.com/@dennis_hilk). Click the preview
to watch the result.

The accepted episode's [storyboard](projects/cisco-doom-episode-001/episode.json)
and [presentation settings](projects/cisco-doom-episode-001/presentation.json)
are in this repository. Original large media and local AI models are not
included in a fresh clone; rebuilding the reference episode needs those assets.

## Why Source2Reel?

| Capability | What it gives you |
| --- | --- |
| Source evidence | An inventory with SHA-256 hashes, evidence references and provenance records. |
| Local production | Configurable local AI, a separate CPU voice environment and FFmpeg rendering. |
| Storyboard review | Read and revise the episode before spending time on the final render. |
| Authentic visuals | Use project screenshots and video, with explicit evidence links. |
| Long-source handling | Context budgets, split research, planner compaction and reusable checkpoints. |
| Reusable presentation | Select a theme, voice and series label through configuration. |

Evidence checks help keep narration tied to documented facts. AI output still
needs editorial review; the pipeline does not guarantee that every generated
storyboard is correct.

## Quick start

The automated installer currently targets **Arch Linux x86_64**. It installs
system packages through `pacman` and creates separate Python 3.12 environments
for the core and Kokoro voice. See [BUILD.md](BUILD.md) for the full procedure.

### 1. Install

```bash
git clone https://github.com/dennishilk/source2reel.git
cd source2reel
./install.sh
./s2r doctor
```

Doctor checks the core separately from optional voice and AI readiness. A
successful core check does not mean that a model is configured or running.

### 2. Configure your local model

Put your GGUF model and matching multimodal projector under `models/`.
Create `config/local.toml` with your actual filenames:

```toml
[local_ai]
model_path = "models/YOUR-MODEL.gguf"
mmproj_path = "models/YOUR-MMPROJ.gguf"
```

These are placeholders, not download instructions. Choose a model/runtime
combination appropriate for your hardware and record its provenance.
`models/` and `config/local.toml` are ignored by Git.

The default provider uses a localhost OpenAI-compatible endpoint, with
llama.cpp as the managed runtime. A single model directly under `models/`
can also be discovered automatically. `DENNIS_LLM_MODEL`,
`DENNIS_LLM_MMPROJ` and `LLAMA_SERVER` provide explicit overrides.
See [local AI setup](docs/LOCAL_AI.md).

### 3. Start the producer

```bash
./s2r
```

Choose **New episode**, paste a GitHub URL, website URL or local directory,
and press Enter for automatic story discovery. The producer starts or reuses
the managed local AI when it is configured, then stops at storyboard review.

Choose **Review**, **Build** or **Quit**. **Continue episode** opens a saved
storyboard, and **Build episode** renders an existing one. The repository-local
launcher needs no global CLI installation or shell PATH changes.

For a first explicit run:

```bash
./s2r create https://github.com/dennishilk/cisco9951-doom --review
```

Inspect the saved storyboard and use its project slug with `./s2r build`.

## How it works

1. **Collect sources.** Clone a GitHub project, fetch website pages or inspect
   a local directory.
2. **Record evidence.** Inventory documents and media, keeping hashes and
   references to the original material.
3. **Research and plan.** Ask the configured local model for supported facts
   and a structured storyboard; inspect original media when vision is enabled.
4. **Review the story.** Read or revise the episode before building it.
5. **Produce narration and visuals.** Render configured scene templates,
   project media, Kokoro speech and captions timed from the generated audio.
6. **Assemble the video.** Use FFmpeg for the full-HD H.264/AAC output and
   preserve the accompanying transcript and provenance artifacts.

Large requests are split to fit `local_ai.context_size`. Input-hashed
checkpoints under `projects/<episode>/manifests/` let reruns reuse valid
completed work. Supporting examples and generated artifacts are scoped so
their volume does not automatically displace the project's primary sources.

## What you get

- A **1920 × 1080 H.264/AAC MP4** under `projects/<episode>/output/`.
- The reviewable **`episode.json` storyboard**.
- Narration-derived captions and a transcript.
- Evidence, research and production provenance artifacts.

Timing comes from the generated audio. The default **Dennis Explainer**
profile uses Kokoro's stock American English voice `am_michael`, at speed
0.94, in a separate CPU environment. See [VOICE.md](VOICE.md).

## Commands

| Task | Command |
| --- | --- |
| Open the interactive producer | `./s2r` |
| Check the installation | `./s2r doctor` |
| Start or inspect the managed AI | `./s2r ai start` / `./s2r ai status` |
| Create a storyboard for review | `./s2r create <SOURCE> --review` |
| Build a saved episode | `./s2r build <project-slug>` |
| Revise and render | `./s2r revise <project-slug> "Make the hook stronger." --build` |
| Test the configured voice | `./s2r voice-test` |
| Stop managed processes | `./s2r stop all` |
| Preview generated-state cleanup | `./s2r clean` |

Combine multiple sources when a repository and original media tell different
parts of the story:

```bash
./s2r create \
  https://www.example.com/project/ \
  https://github.com/OWNER/PROJECT \
  /path/to/original-media \
  --instructions "Focus on the hardware architecture and final result." \
  --review
```

`create` without `--review` proceeds to rendering. Stage progress goes to stderr;
the saved storyboard or video path is the command's final stdout output.
`--config` selects an additional config override for create, build and revise.

## Privacy and local processing

**The default AI endpoint is localhost.** Research requests and optional media
inspection go to the configured provider; local Kokoro narration and FFmpeg
rendering run on your machine.

Network access is used when fetching GitHub or website sources, installing
dependencies, or downloading model and voice assets. Changing the provider
URL can send requests to another machine, so keep it local when that is your
intended privacy boundary. You control that setting.

The core has no Kokoro, Torch or CUDA dependency. The optional voice environment
uses CPU Torch. Local model inference can use Vulkan when available, with an
explicit CPU fallback. The Arch installer does not automatically install
ROCm or a CUDA stack.

Generated state stays under the checkout: `models/`, `cache/`, `runtime/`,
`output/` and per-episode work directories are ignored. Cleanup previews its
targets first; removal requires `--yes`, and model removal also requires
`--models`. Stop managed processes before cleaning.
See [runtime and cleanup details](BUILD.md#runtime-state-and-reset).

## Profiles and project layout

The default series identity lives in [config/engine.toml](config/engine.toml),
[themes/dennis-dark.toml](themes/dennis-dark.toml) and
[voices/dennis-explainer.toml](voices/dennis-explainer.toml). Choose a different
theme, voice or series label in configuration.

Reusable engine code belongs in `source2reel/`; scene schemas, prompts and
presentation profiles live alongside it. Episode-specific sources, manifests
and storyboards belong in `projects/<episode>/`.

## Current status

Source2Reel is an **experimental Linux project under active development**.
The Cisco reference episode was rendered and accepted on Cthulhu on
23 September 2026: 12 scenes, about 283.6 seconds, at 1920 × 1080.

The consolidated fresh installer and later rendering changes have separate
physical acceptance gates. The published reference episode does not imply that
every model, GPU or fresh installation has been validated.
[PROJECT_STATE.md](PROJECT_STATE.md) records the individual milestones.

## Documentation

| Guide | Contents |
| --- | --- |
| [Build and installation](BUILD.md) | Arch setup, model paths, rendering and cleanup. |
| [Local AI](docs/LOCAL_AI.md) | Providers, hardware backends and managed runtime. |
| [Architecture](docs/ARCHITECTURE.md) | Evidence, planning, templates and production boundaries. |
| [Voice](VOICE.md) | Kokoro runtime, accepted voice and pronunciation. |
| [Style guide](STYLE_GUIDE.md) | The default series presentation. |
| [Dependency and media licenses](LICENSES.md) | Third-party licensing and provenance notes. |
| [Project state](PROJECT_STATE.md) | Historical validation and pending acceptance work. |
| [Contributing](CONTRIBUTING.md) | A small core-only developer setup and test commands. |

A practical walkthrough is also available on my website:
[English](https://www.dennishilk.com/blog/source2reel-github-local-explainer-video/) ·
[Deutsch](https://www.dennishilk.com/de/blog/source2reel-github-erklaervideo-lokal/).

## Development

Install the `dev` extra in a Python 3.12 virtual environment, then run the suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

GitHub Actions runs the tests, including synthetic FFmpeg caption checks, on
Linux. The checks do not need GGUF downloads, a GPU or the Kokoro runtime;
full physical voice and reference-episode renders are separate.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing

[LICENSES.md](LICENSES.md) documents third-party dependencies, model weights
and episode media. The repository currently does not declare a standalone
license for the engine's own code.
