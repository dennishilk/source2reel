from __future__ import annotations
import subprocess, tomllib
from pathlib import Path
from typing import Any
from .config import profile_paths

_PIPELINES={}


def _voice_cfg(root: Path, cfg: dict[str,Any]):
    _, voice_path = profile_paths(root,cfg)
    return tomllib.loads(voice_path.read_text())


def apply_pronunciations(text: str, cfg: dict[str,Any]) -> str:
    for source,target in cfg.get("pronunciation",{}).items():
        text=text.replace(source,target)
    return text


def _kokoro_pipeline(lang_code: str):
    if lang_code not in _PIPELINES:
        from kokoro import KPipeline
        _PIPELINES[lang_code]=KPipeline(lang_code=lang_code)
    return _PIPELINES[lang_code]


def kokoro_scene(text: str, out: Path, root: Path, engine_cfg: dict[str,Any]):
    import numpy as np, soundfile as sf
    cfg=_voice_cfg(root,engine_cfg)
    v=cfg["voice"]
    text=apply_pronunciations(text,cfg)
    pipeline=_kokoro_pipeline(v["language_code"])
    chunks=[]
    generated=list(pipeline(text,voice=v["voice"],speed=float(v["speed"])))
    for i,(_,_,audio) in enumerate(generated):
        chunks.append(audio)
        if i != len(generated)-1:
            chunks.append(np.zeros(int(v["sample_rate"]*0.11),dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    out.parent.mkdir(parents=True,exist_ok=True)
    sf.write(out,np.concatenate(chunks),int(v["sample_rate"]))


def espeak_preview(text: str, out: Path):
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["espeak","-v","en-us+m3","-s","150","-p","32","-a","180","-w",str(out),text],check=True)


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
