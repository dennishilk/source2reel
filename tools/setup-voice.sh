#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOICE_ENV="${SOURCE2REEL_VOICE_ENV:-$ROOT/.venv-voice-kokoro}"
VOICE_PY="$VOICE_ENV/bin/python"
TORCH_SPEC="${SOURCE2REEL_TORCH_CPU_SPEC:-torch}"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/runtime/python"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HUB_CACHE" "$TORCH_HOME"

command -v uv >/dev/null || { echo "uv is required. Run tools/bootstrap-arch.sh first." >&2; exit 2; }

printf 'Source2Reel optional voice setup\n'
python3 "$ROOT/source2reel/hardware.py" --format text || true
cat <<'MSG'

Voice backend: Kokoro CPU
This is intentionally isolated from the Source2Reel core environment.
No CUDA/NVIDIA runtime is required for this voice path.
MSG

uv python install 3.12
uv venv --python 3.12 "$VOICE_ENV"

# Resolve torch from the official CPU wheel index. The additional index is
# package-scoped by availability: torch is found there, while ordinary
# dependencies continue to resolve from PyPI.
uv pip install --python "$VOICE_PY" \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy first-index \
  "$TORCH_SPEC"

uv pip install --python "$VOICE_PY" -r "$ROOT/requirements/voice-kokoro-cpu.txt"
uv pip install --python "$VOICE_PY" --no-deps kokoro==0.9.4
# Misaki otherwise invokes pip at first narration and can target the wrong
# interpreter. Install the exact English spaCy pipeline in this uv environment.
uv pip install --python "$VOICE_PY" \
  'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
uv pip check --python "$VOICE_PY"

"$VOICE_PY" - <<'PY'
import torch
import kokoro
import spacy
from kokoro import KPipeline
assert not torch.cuda.is_available(), "Voice runtime must use CPU Torch"
assert spacy.load("en_core_web_sm") is not None
KPipeline(lang_code="a")
print("torch:", torch.__version__)
print("torch CUDA available:", torch.cuda.is_available())
print("kokoro:", getattr(kokoro, "__version__", "0.9.4"))
print("spaCy English model and Kokoro American English pipeline: OK")
PY

printf '\nInstalled optional Kokoro CPU voice environment: %s\n' "$VOICE_ENV"
printf 'The Dennis Explainer voice remains pending audible approval on Cthulhu.\n'
