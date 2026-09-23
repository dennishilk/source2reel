#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT" exec python -m source2reel.cli build cisco-doom --preview-espeak
