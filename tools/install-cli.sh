#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/source2reel" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
export SOURCE2REEL_ROOT="$ROOT"
exec uv run --project "$ROOT" source2reel "\$@"
SCRIPT
chmod +x "$HOME/.local/bin/source2reel"
printf 'Installed %s\n' "$HOME/.local/bin/source2reel"
