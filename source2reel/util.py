from __future__ import annotations
import hashlib, json, os, re, subprocess, sys
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:80] or "episode"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text_lossy(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if limit is not None:
        data = data[:limit]
    return data.decode("utf-8", errors="replace")


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    strcmd = [str(x) for x in cmd]
    print("+", " ".join(strcmd), file=sys.stderr, flush=True)
    return subprocess.run(strcmd, cwd=cwd, check=True, text=True, capture_output=capture)


def ffprobe_duration(path: Path) -> float:
    p = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path)
    ], capture=True)
    return float(p.stdout.strip())
