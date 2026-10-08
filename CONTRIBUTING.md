# Contributing to Source2Reel

Source2Reel combines source evidence, local AI planning and a fixed video
production pipeline. A useful change keeps the engine reusable and explains
how its behavior was checked.

## Core developer setup

Use **Python 3.12** in a virtual environment. This setup installs the engine,
Pillow and the test runner; it does not install Arch system packages or the
optional Kokoro/Torch voice environment.

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

```bash
.venv/bin/python -m pytest -q
```

FFmpeg and ffprobe enable the synthetic caption/render tests. On Linux, install
them through your distribution together with a DejaVu Sans font and FFmpeg's
libass support. Tests requiring FFmpeg are skipped when it is absent; GitHub
Actions installs it and runs those checks.

The process-ownership integration test needs Linux with `/proc` mounted for
the process's PID namespace. It is skipped when that is unavailable; GitHub
Actions checks that prerequisite before running the suite.

The tests use fake model responses and temporary files. They do not require a
live local model, downloaded GGUF weights or Kokoro. Physical GPU, voice and
full reference-episode validation are distinct from these automated checks.

## What to preserve

- Keep reusable changes in the engine, prompts, schemas and profiles.
  Episode-specific content belongs under `projects/<episode>/`.
- Keep factual claims tied to supported research and their original evidence.
  Narrative source context must not authorize additional facts.
- Preserve the accepted Episode 001 fixtures; use a presentation sidecar for
  documented presentation-only changes.
- Keep model payloads, generated output and private local configuration out
  of commits.
- Keep Kokoro/Torch separate from the core Python dependency graph.
- Add a focused regression check when fixing meaningful runtime behavior.

For a proposed change, describe the concrete trigger, the resulting behavior
and the relevant validation. The [architecture](docs/ARCHITECTURE.md) and
[project state](PROJECT_STATE.md) explain the current boundaries.
