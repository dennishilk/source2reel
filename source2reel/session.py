"""Local llama.cpp starter with exact PID ownership records."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from urllib.parse import urlsplit

from .config import load_engine_config
from .hardware import detect_hardware
from .paths import local_environment, runtime_paths

_children: dict[int, subprocess.Popen] = {}


def server_binary(root: Path) -> Path | None:
    override = os.environ.get("LLAMA_SERVER")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            resolved = shutil.which(override)
            candidate = Path(resolved) if resolved else root / candidate
        return candidate.resolve() if candidate.is_file() and os.access(candidate, os.X_OK) else None
    system = shutil.which("llama-server")
    if system:
        return Path(system).resolve()
    vendor = root / "vendor/llama.cpp/build-vulkan/bin/llama-server"
    return vendor.resolve() if vendor.is_file() and os.access(vendor, os.X_OK) else None


def _model_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def server_command(root: Path, backend: str | None = None) -> list[str]:
    root = root.resolve()
    cfg = load_engine_config(root)
    ai = cfg.get("local_ai", {})
    server = server_binary(root)
    if server is None:
        raise RuntimeError("llama-server missing; install the Arch llama-cpp package or set LLAMA_SERVER")
    info = detect_hardware()
    available = ["cpu"]
    if info.vulkan_available:
        available.insert(0, "vulkan")
    # An AMD PCI ID alone does not prove a usable HIP runtime.
    selected = backend or ("vulkan" if "vulkan" in available else "cpu")
    if selected not in available:
        raise RuntimeError(f"backend {selected!r} unavailable; detected: {', '.join(available)}")
    model_raw = os.environ.get("DENNIS_LLM_MODEL") or str(ai.get("model_path", ""))
    if not model_raw:
        candidates = [p for p in (root / "models").glob("*.gguf") if "mmproj" not in p.name.lower()]
        if len(candidates) == 1:
            model_raw = str(candidates[0])
    model = _model_path(root, model_raw) if model_raw else None
    if model is None or not model.is_file():
        raise RuntimeError("Model GGUF missing. Place it under models/ and set [local_ai] model_path in config/local.toml (or DENNIS_LLM_MODEL).")
    mmproj_raw = os.environ.get("DENNIS_LLM_MMPROJ") or str(ai.get("mmproj_path", ""))
    mmproj = _model_path(root, mmproj_raw) if mmproj_raw else None
    if mmproj and not mmproj.is_file():
        raise RuntimeError(f"mmproj missing: {mmproj}")
    url = urlsplit(cfg["llm"]["base_url"])
    if url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Local AI endpoint must bind to localhost")
    host = "127.0.0.1" if url.hostname == "localhost" else url.hostname
    command = [str(server), "-m", str(model), "-a", str(cfg["llm"]["model"]),
               "-ngl", "0" if selected == "cpu" else "999",
               "-c", str(int(ai.get("context_size", 32768))),
               "--host", host, "--port", str(url.port or 8080)]
    if mmproj:
        command += ["--mmproj", str(mmproj)]
    return command


def _proc_identity(pid: int) -> str | None:
    """Kernel start tick prevents a stale PID file from targeting a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def _registry(root: Path) -> Path:
    return runtime_paths(root)["pids"] / "managed.json"


def _read(root: Path) -> list[dict]:
    try:
        data = json.loads(_registry(root).read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _write(root: Path, entries: list[dict]) -> None:
    path = _registry(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        os.chmod(tmp, 0o600)
        json.dump(entries, stream, indent=2)
    os.replace(tmp, path)


def _owned(entry: dict) -> bool:
    pid = entry.get("pid")
    return (type(pid) is int and pid > 1 and
            _proc_identity(pid) == entry.get("start_tick") and
            os.getpgid(pid) == pid)


def status(root: Path) -> list[dict]:
    entries = []
    for entry in _read(root):
        try:
            if _owned(entry):
                entries.append(entry)
        except ProcessLookupError:
            pass
    _write(root, entries)
    return entries


def start(root: Path, backend: str | None = None) -> dict:
    if status(root):
        raise RuntimeError("A Source2Reel-managed server is already running; use ./s2r ai status")
    command = server_command(root, backend)
    paths = runtime_paths(root)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    log = paths["logs"] / "llama-server.log"
    with log.open("a") as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, env=local_environment(root),
                                   start_new_session=True)
    time.sleep(0.2)
    if process.poll() is not None:
        raise RuntimeError(f"llama-server exited with status {process.returncode}; inspect {log}")
    tick = _proc_identity(process.pid)
    if tick is None:
        raise RuntimeError("llama-server disappeared before PID registration")
    entry = {"pid": process.pid, "start_tick": tick, "session": uuid.uuid4().hex,
             "kind": "llama-server", "log": str(log)}
    _children[process.pid] = process
    _write(root, [entry])
    return entry


def stop_all(root: Path) -> int:
    entries = status(root)
    for entry in entries:
        try:
            if _owned(entry):
                os.killpg(entry["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(50):
        if not any(_proc_identity(entry["pid"]) == entry["start_tick"] for entry in entries):
            break
        time.sleep(0.1)
    for entry in entries:
        try:
            if _owned(entry):
                os.killpg(entry["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        process = _children.pop(entry["pid"], None)
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    _write(root, [])
    return len(entries)


def interactive(root: Path) -> None:
    info = detect_hardware()
    choices = "Vulkan, CPU" if info.vulkan_available else "CPU"
    print(f"Source2Reel session | GPU: {info.vendor} | available: {choices}")
    while True:
        print("\n1 Start local AI  2 Doctor  3 Voice test  4 Status  5 Stop all  0 Exit")
        try:
            choice = input("s2r> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        try:
            if choice in ("0", "exit", "quit"):
                return
            if choice == "1":
                print(start(root))
            elif choice == "2":
                from .doctor import run_doctor
                run_doctor(root)
            elif choice == "3":
                from .cli import voice_test
                print(voice_test(root))
            elif choice == "4":
                print(status(root) or "No managed processes")
            elif choice in ("5", "stop all"):
                print(f"Stopped {stop_all(root)} managed process(es)")
            else:
                print("Unknown choice")
        except (RuntimeError, OSError) as exc:
            print(f"Error: {exc}")
