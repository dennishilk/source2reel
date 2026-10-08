from __future__ import annotations
import os, shutil, subprocess, tomllib
from pathlib import Path
from typing import Any
from .config import profile_paths
from .paths import local_environment


def _voice_cfg(root: Path, cfg: dict[str,Any]):
    _, voice_path = profile_paths(root,cfg)
    return tomllib.loads(voice_path.read_text())


def apply_pronunciations(text: str, cfg: dict[str,Any]) -> str:
    for source,target in cfg.get("pronunciation",{}).items():
        text=text.replace(source,target)
    return text


def voice_runtime_python(root: Path, voice_cfg: dict[str,Any]) -> Path:
    raw = str(voice_cfg.get("voice", {}).get("runtime_python", ".venv-voice-kokoro/bin/python"))
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    return expanded if expanded.is_absolute() else root / expanded


def kokoro_scene(text: str, out: Path, root: Path, engine_cfg: dict[str,Any]):
    cfg=_voice_cfg(root,engine_cfg)
    v=cfg["voice"]
    text=apply_pronunciations(text,cfg)
    runtime=voice_runtime_python(root,cfg)
    if not runtime.exists():
        raise RuntimeError(
            f"Kokoro voice runtime not installed: {runtime}. "
            "Run tools/setup-voice.sh for the optional CPU voice stack."
        )
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        str(runtime), str(root/"source2reel"/"kokoro_worker.py"),
        "--out", str(out),
        "--voice", str(v["voice"]),
        "--language-code", str(v["language_code"]),
        "--speed", str(float(v["speed"])),
        "--sample-rate", str(int(v["sample_rate"])),
        "--pause-ms", str(int(cfg.get("delivery",{}).get("pause_between_chunks_ms",110))),
    ], input=text, text=True, check=True, env=local_environment(root))


def espeak_preview(text: str, out: Path):
    binary = shutil.which("espeak") or shutil.which("espeak-ng")
    if binary is None:
        raise RuntimeError("eSpeak preview requires espeak or espeak-ng on PATH")
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([binary,"-v","en-us+m3","-s","150","-p","32","-a","180","-w",str(out),text],check=True)


def render_scene_audio(text: str, raw: Path, normalized: Path, root: Path, engine_cfg: dict[str,Any], preview_espeak: bool=False):
    try:
        kokoro_scene(text,raw,root,engine_cfg)
        backend="kokoro"
    except Exception:
        if not preview_espeak:
            raise
        espeak_preview(text,raw)
        backend="espeak-preview"
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-i",str(raw),
        "-af","loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000",str(normalized)
    ],check=True)
    return backend
