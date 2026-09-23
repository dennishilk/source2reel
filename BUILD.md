# BUILD — Source2Reel on Arch Linux / Cthulhu

The current target machine is Cthulhu: Arch Linux x86_64, Ryzen 7 5800X3D, Radeon RX 9060 XT 16 GiB and about 31 GiB system RAM. Hardware-specific AI/encoding choices remain configurable and must not leak into episode files.

## 1. Base packages

Keep the system Python untouched. The current TTS environment uses Python 3.12 because the pinned `kokoro==0.9.4` package declares a Python constraint that is safest to satisfy with an isolated interpreter.

```bash
sudo pacman -S --needed ffmpeg espeak-ng git cmake ninja base-devel vulkan-radeon vulkan-icd-loader
```

Install `uv` using its current official instructions if needed, then:

```bash
uv python install 3.12
uv sync --python 3.12
```

## 2. CLI

Inside the repository:

```bash
uv run source2reel --help
```

Optional local wrapper:

```bash
./tools/install-cli.sh
```

## 3. Local AI service

The engine expects a localhost API and remains independent of runtime. The first conservative candidate is current llama.cpp with Vulkan on AMD Linux; ROCm/HIP is evaluated separately in the Cthulhu capability audit before being frozen.

See `docs/LOCAL_AI.md`.

## 4. Create or review an episode

```bash
uv run source2reel create https://github.com/dennishilk/cisco9951-doom --review
```

Then inspect the generated evidence/research/storyboard and build:

```bash
uv run source2reel build cisco9951-doom
```

## 5. Revise without manual scene programming

```bash
uv run source2reel revise cisco9951-doom \
  "Make the hook stronger but keep all factual claims evidence-grounded." \
  --build
```

## 6. Outputs

Each episode output directory contains at least:

- final MP4
- `transcript.txt`
- `storyboard.json`
- `evidence-manifest.json`
- `research.json` when available
- `render-manifest.json`
- `provenance.json`
- `ENGINE_LICENSES.md`

Generated `work/` and `output/` directories are intentionally ignored by Git.
