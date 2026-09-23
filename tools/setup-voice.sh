#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOICE_ENV="${SOURCE2REEL_VOICE_ENV:-$ROOT/.venv-voice-kokoro}"
VOICE_PY="$VOICE_ENV/bin/python"
TORCH_SPEC="${SOURCE2REEL_TORCH_CPU_SPEC:-torch}"

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
uv pip check --python "$VOICE_PY"

"$VOICE_PY" - <<'PY'
import torch
import kokoro
print("torch:", torch.__version__)
print("torch CUDA available:", torch.cuda.is_available())
print("kokoro:", getattr(kokoro, "__version__", "0.9.4"))
PY

printf '\nInstalled optional Kokoro CPU voice environment: %s\n' "$VOICE_ENV"
printf 'The Dennis Explainer voice remains pending audible approval on Cthulhu.\n'
