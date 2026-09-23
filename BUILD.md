# BUILD — Arch Linux / Cthulhu

Target: Arch Linux x86_64, Ryzen 7 5800X3D and Radeon RX 9060 XT 16 GiB
with Mesa RADV/Vulkan. The system `/usr/bin/llama-server` was physically
proven on Cthulhu; this consolidated installer has not yet run there.

## Fresh installation

```bash
git clone https://github.com/dennishilk/source2reel.git
cd source2reel
./install.sh
./s2r doctor
./s2r voice-test
```

`install.sh` creates local runtime directories, calls the hardware-aware
`tools/bootstrap-arch.sh` and isolated `tools/setup-voice.sh`, then runs doctor.
It is safe to rerun. Baseline Arch packages include FFmpeg, uv, Vulkan tools,
llama.cpp and the detected vendor's Vulkan driver. AMD gets `vulkan-radeon`.
ROCm and CUDA are never automatically installed. Core `uv.lock` is scanned
for Torch/NVIDIA/CUDA before syncing. Voice uses CPU Torch, Kokoro and a
spaCy 3.8 English model inside `.venv-voice-kokoro`; setup verifies
`spacy.load("en_core_web_sm")` and initializes `KPipeline(lang_code="a")`.

`./s2r` runs the local `.venv/bin/python`; it does not use global PATH setup.

## Configure and start local AI

Place authorized GGUF and matching projector files in ignored `models/`.
Set exact paths in the ignored `config/local.toml`:

```toml
[local_ai]
model_path = "models/YOUR-MODEL.gguf"
mmproj_path = "models/YOUR-MMPROJ.gguf"
```

Alternatively set `DENNIS_LLM_MODEL` and `DENNIS_LLM_MMPROJ`. One non-mmproj
GGUF directly under `models/` is discovered automatically. Record chosen
model/revision/hash in `PROJECT_STATE.md` and `LICENSES.md` before freezing
production. `LLAMA_SERVER` overrides binary discovery, which prefers the
system executable over a vendored build. The API binds only to localhost.
Vulkan is chosen when detected; CPU is the fallback. ROCm is not offered
without a validated integration.

```bash
./s2r                 # terminal session interface
./s2r ai start        # tracked local server
./s2r ai status
./s2r stop all        # stops only recorded, identity-matched processes
./s2r create https://github.com/dennishilk/cisco9951-doom --review
./s2r build cisco-doom
```

## Rebuild accepted Episode 001

After pulling the post-acceptance changes on Cthulhu, keep the existing
untracked source media in `projects/cisco-doom-episode-001/` and run:

```bash
./s2r build cisco-doom-episode-001
```

This uses the frozen `episode.json` and the new, separate `presentation.json`.
The output is named from the frozen episode slug in
`projects/cisco-doom-episode-001/output/`. Captions are generated from each
scene's narration and final WAV duration; the ASS intermediates remain under
ignored `work/captions/`. They are burned into static segments by FFmpeg's
libass filter. A local FFmpeg build with `ass` support and a DejaVu Sans font
is required. The original media is required only for the physical render.

Static scenes caption by default; moving evidence and the outro do not.
Set `"captions": {"enabled": true}` or `false` on a scene to override. Set
`[captions] enabled = false` in the engine config to disable the default for
the whole build (explicit scene overrides still apply). `media.start_seconds`
is the non-negative seek time for video evidence. Diagram scenes require
explicit `diagram.nodes` or `diagram.steps` with at least two labels; empty
diagrams fail validation. New episodes may put `presentation.outro` in the
episode spec; a project's `presentation.json` can supply or override it when
the episode JSON is immutable. An outro takes `eyebrow`, `headline` (1–3
lines), `links` (label and URL or URL lines), and optional `footer`.

Doctor reports missing models and a stopped API as optional; `ai start`
explains which model path is missing. The physical fresh-install and sound
check remain acceptance gates.

## Runtime state and reset

The ignored `.venv/`, `.venv-voice-kokoro/`, `models/`, `cache/` (Hugging
Face, Torch, uv), `runtime/` (Python, PID records, logs), and `output/` stay
under the checkout. Per-episode generated `work/` and `output/` are ignored.

```bash
./s2r clean                 # preview only
./s2r stop all
./s2r clean --yes           # environments/cache/runtime/root output
./s2r clean --models --yes  # also remove downloaded models
./install.sh
```

Cleanup does not remove Arch packages, global caches, Git credentials or
project source. The legacy `tools/install-cli.sh` is optional only; an
existing `~/.local/bin/source2reel` can be removed manually.
