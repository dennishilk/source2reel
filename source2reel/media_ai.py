from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
from typing import Any
from PIL import Image
from .providers import LLMProvider
from .progress import Progress, step
from .inventory import resolve_evidence_path
from .util import json_dump



def _video_keyframe(path: Path, out: Path):
    subprocess.run(["ffmpeg","-y","-loglevel","error","-ss","1","-i",str(path),"-frames:v","1",str(out)],check=True)


def _bounded_image(path: Path, out: Path, max_dim: int=1600):
    with Image.open(path) as im:
        im=im.convert("RGB"); scale=min(1.0,max_dim/max(im.width,im.height)); size=(max(1,round(im.width*scale)),max(1,round(im.height*scale)))
        if size!=(im.width,im.height): im=im.resize(size,Image.Resampling.LANCZOS)
        im.save(out,"JPEG",quality=90,optimize=True)


def _priority(e: dict[str,Any]):
    meta=e.get("media") or {}; area=int(meta.get("width",0) or 0)*int(meta.get("height",0) or 0); name=e.get("relative_path","").lower()
    bonus=2_000_000 if any(k in name for k in ("final","hero","proof","doom","screen","terminal","hardware","photo")) else 0
    return bonus+area


def enrich_media(provider: LLMProvider, inventory: dict[str,Any], project_dir: Path, max_items: int=40, progress: Progress|None=None) -> dict[str,Any]:
    media=sorted((e for e in inventory["evidence"] if e.get("kind")=="media"),key=_priority,reverse=True)[:max_items]
    for index, e in enumerate(media, 1):
        with step(progress, f"Visual evidence {index}/{len(media)}"):
            p=resolve_evidence_path(e,project_dir); tmp=tempfile.TemporaryDirectory(); img=Path(tmp.name)/"inspection.jpg"
            try:
                if p.suffix.lower() in {".mp4",".mov",".mkv",".webm"}: _video_keyframe(p,img)
                elif p.suffix.lower()==".svg": continue
                else: _bounded_image(p,img)
                system=(project_dir.parents[1]/"prompts"/"media-inspection.txt").read_text()
                r=provider.complete_json_with_image(system,"Inspect this project asset for later storyboard selection. Do not identify people. Return keys: category, caption, visible_text, evidence_value (high|medium|low), reasons.",img)
                e["ai_media"]=r
            except Exception as ex:
                e["ai_media_error"]=str(ex)
            finally: tmp.cleanup()
    json_dump(project_dir/"manifests"/"evidence.json",inventory)
    return inventory
