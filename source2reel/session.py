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
from .schema import validate_episode
from .util import json_load

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


def _source(raw: str) -> str | None:
    source = raw.strip()
    if source.startswith(("http://", "https://")):
        return source if urlsplit(source).netloc else None
    if source:
        try:
            if Path(source).expanduser().is_dir():
                return source
        except (OSError, ValueError):
            pass
    return None


def _projects(root: Path) -> list[str]:
    """List only local project directories with a readable, usable storyboard."""
    projects = root / "projects"
    if not projects.is_dir():
        return []
    names = []
    for directory in sorted(projects.iterdir(), key=lambda p: p.name):
        episode = directory / "episode.json"
        if not directory.is_dir() or directory.is_symlink() or not episode.is_file() or episode.is_symlink():
            continue
        try:
            data = json_load(episode)
            if not isinstance(data, dict):
                continue
            validate_episode(data)
        except (OSError, ValueError, TypeError, AttributeError, KeyError, UnicodeError):
            continue
        names.append(directory.name)
    return names


def _select_project(root: Path) -> str | None:
    projects = _projects(root)
    if not projects:
        print("No usable episodes found in projects/.")
        return None
    print("\nEpisodes:")
    for number, project in enumerate(projects, 1):
        print(f"{number}  {project}")
    while True:
        choice = input("Project number [0 = back]: ").strip()
        if choice == "0":
            return None
        if choice.isdecimal() and 1 <= int(choice) <= len(projects):
            return projects[int(choice) - 1]
        print("Invalid project number.")


def _review(root: Path, project: str) -> None:
    episode = json_load(root / "projects" / project / "episode.json")
    print(f"\nStoryboard: {episode['title']}")
    for scene in episode["scenes"]:
        title = f" | {scene['title']}" if scene.get("title") else ""
        refs = ", ".join(scene.get("evidence_refs", [])) or "none"
        preview = " ".join(scene["narration"].split())
        if len(preview) > 140:
            preview = preview[:139].rstrip() + "…"
        print(f"\n{scene['id']} | {scene['type']}{title}\n  Evidence: {refs}\n  Narration: {preview}")


def _review_or_build(root: Path, project: str, *, show_first: bool, new: bool) -> None:
    from .pipeline import build_existing

    print(f"Storyboard: {root / 'projects' / project / 'episode.json'}")
    if show_first:
        _review(root, project)
    while True:
        choice = input("[R] Review  [B] Build  [Q] " + ("Quit / continue later: " if new else "Back: ")).strip().lower()
        if choice == "r":
            _review(root, project)
        elif choice == "b":
            print(f"Build output: {build_existing(root, project)}")
            return
        elif choice == "q":
            return
        else:
            print("Choose R, B or Q.")


def _new_episode(root: Path) -> None:
    from .pipeline import create

    while True:
        first = input("Source URL, repository or local directory:\n> ").strip()
        if _source(first):
            break
        print("Enter a valid http(s) URL or existing local directory.")
    sources = [first]
    while True:
        additional = input("Additional source? [Enter = none]\n> ").strip()
        if not additional:
            break
        if _source(additional):
            sources.append(additional)
        else:
            print("Enter a valid http(s) URL or existing local directory.")
    instructions = input("Episode focus/instructions? [Enter = discover the story automatically]\n> ").strip()
    while True:
        answer = input("Stop for storyboard review first? [Y/n]\n> ").strip().lower()
        if answer in ("", "y", "yes", "n", "no"):
            review = answer not in ("n", "no")
            break
        print("Enter Y or N.")
    print("\nProject:")
    for source in sources:
        print(f"  source: {source}")
    print(f"  mode: {'custom focus' if instructions else 'automatic story discovery'}")
    print(f"  review: {'yes' if review else 'no'}")

    # Only the built-in OpenAI-compatible provider uses the managed llama-server.
    # Another configured provider, such as Ollama, manages its own lifecycle.
    if load_engine_config(root)["llm"]["provider"] == "openai_compat":
        if status(root):
            print("Reusing Source2Reel-managed local AI.")
        else:
            print("Starting Source2Reel local AI...")
            start(root)
    output = create(root, sources, None, instructions, False, review)
    if review:
        _review_or_build(root, output.parent.name, show_first=False, new=True)
    else:
        print(f"Build output: {output}")


def _local_ai(root: Path) -> None:
    while True:
        print("\nLocal AI\n1  Start\n2  Status\n3  Stop all\n0  Back")
        choice = input("ai> ").strip().lower()
        if choice == "0":
            return
        if choice == "1":
            print(start(root))
        elif choice == "2":
            print(status(root) or "No managed processes")
        elif choice == "3":
            print(f"Stopped {stop_all(root)} managed process(es)")
        else:
            print("Unknown choice")


def interactive(root: Path) -> None:
    print("SOURCE2REEL")
    while True:
        print("\n1  New episode\n2  Continue episode\n3  Build episode\n4  Local AI\n5  Doctor\n6  Voice test\n0  Exit")
        try:
            choice = input("s2r> ").strip().lower()
            if choice in ("0", "exit", "quit"):
                return
            if choice == "1":
                _new_episode(root)
            elif choice == "2":
                project = _select_project(root)
                if project:
                    _review_or_build(root, project, show_first=True, new=False)
            elif choice == "3":
                project = _select_project(root)
                if project:
                    from .pipeline import build_existing
                    print(f"Build output: {build_existing(root, project)}")
            elif choice == "4":
                _local_ai(root)
            elif choice == "5":
                from .doctor import run_doctor
                run_doctor(root)
            elif choice == "6":
                from .cli import voice_test
                print(voice_test(root))
            else:
                print("Unknown choice")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Error: {exc}")
