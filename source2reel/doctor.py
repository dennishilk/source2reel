from __future__ import annotations
import importlib.util, shutil, urllib.request
from pathlib import Path
from .config import load_engine_config, profile_paths


def _http_ok(url: str, timeout=3):
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r: return True, f"HTTP {r.status}"
    except Exception as e: return False, str(e)


def run_doctor(root: Path, cfg_path: Path|None=None) -> int:
    cfg=load_engine_config(root,cfg_path); checks=[]
    for cmd in ("ffmpeg","ffprobe","git"):
        ok=bool(shutil.which(cmd)); checks.append((cmd,ok,shutil.which(cmd) or "not found"))
    for mod in ("PIL","numpy","soundfile","kokoro"):
        ok=importlib.util.find_spec(mod) is not None; checks.append((f"python:{mod}",ok,"importable" if ok else "missing"))
    theme_path,voice_path=profile_paths(root,cfg)
    checks.append(("theme",theme_path.exists(),str(theme_path)))
    checks.append(("voice",voice_path.exists(),str(voice_path)))
    p=cfg["llm"]
    if p["provider"]=="openai_compat": url=p["base_url"].rstrip("/")+"/models"
    else: url=p["base_url"].rstrip("/")+"/api/tags"
    ok,msg=_http_ok(url); checks.append(("local-ai",ok,msg))
    width=max(len(x[0]) for x in checks)
    for name,ok,msg in checks: print(f"{name:<{width}}  {'OK' if ok else 'FAIL':<4}  {msg}")
    critical={"ffmpeg","ffprobe","git","python:PIL","python:numpy","python:soundfile","python:kokoro","theme","voice","local-ai"}
    return 0 if all(ok for name,ok,_ in checks if name in critical) else 1
