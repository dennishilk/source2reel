from __future__ import annotations
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from PIL import Image
from .providers import LLMProvider
from .progress import Progress, step
from .inventory import resolve_evidence_path
from .util import json_dump, json_load, sha256_file


MEDIA_INSPECTION_CONTRACT = "media-inspection-v1"
_INSPECTION_USER = (
    "Inspect this project asset for later storyboard selection. Do not identify people. "
    "Return keys: category, caption, visible_text, evidence_value (high|medium|low), reasons."
)
_CHECKPOINT_VERSION = 1



def _video_keyframe(path: Path, out: Path):
    subprocess.run(["ffmpeg","-y","-loglevel","error","-ss","1","-i",str(path),"-frames:v","1",str(out)],check=True)


def _bounded_image(path: Path, out: Path, max_dim: int=1600):
    with Image.open(path) as im:
        im=im.convert("RGB"); im.info.clear()
        scale=min(1.0,max_dim/max(im.width,im.height)); size=(max(1,round(im.width*scale)),max(1,round(im.height*scale)))
        if size!=(im.width,im.height): im=im.resize(size,Image.Resampling.LANCZOS)
        im.save(out,"JPEG",quality=90,optimize=True)


def _priority(e: dict[str,Any]):
    meta=e.get("media") or {}; area=int(meta.get("width",0) or 0)*int(meta.get("height",0) or 0); name=e.get("relative_path","").lower()
    bonus=2_000_000 if any(k in name for k in ("final","hero","proof","doom","screen","terminal","hardware","photo")) else 0
    return bonus+area


def _valid_result(result: Any) -> bool:
    """Only complete, JSON-safe inspections can become reusable evidence."""
    if not isinstance(result, dict) or not all(
        isinstance(result.get(key), str) and result[key].strip()
        for key in ("category", "caption")
    ) or not isinstance(result.get("evidence_value"), str) or (
        result["evidence_value"] not in {"high", "medium", "low"}
    ) or "visible_text" not in result or "reasons" not in result:
        return False
    visible, reasons = result.get("visible_text"), result.get("reasons")
    if not (visible is None or isinstance(visible, str) or
            isinstance(visible, list) and all(isinstance(item, str) for item in visible)):
        return False
    if not (isinstance(reasons, str) and bool(reasons.strip()) or
            isinstance(reasons, list) and all(isinstance(item, str) for item in reasons)):
        return False
    try:
        encoded = json.dumps(result, allow_nan=False)
        # A provider may return Python objects that serialize differently. The
        # first run and a later JSON cache hit must expose the same structure.
        return json.loads(encoded) == result
    except (TypeError, ValueError, RecursionError):
        return False


def _inspection_digest(image: Path, system: str, signature: dict[str, Any]) -> str:
    request = {"contract": MEDIA_INSPECTION_CONTRACT,
               "inspection_image_sha256": sha256_file(image),
               "system": system, "user": _INSPECTION_USER,
               "provider": signature}
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _cached_result(checkpoint: Path, digest: str) -> dict[str, Any] | None:
    try:
        cached = json_load(checkpoint)
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None
    if (not isinstance(cached, dict) or type(cached.get("version")) is not int or
            cached["version"] != _CHECKPOINT_VERSION or
            cached.get("input_sha256") != digest or not _valid_result(cached.get("result"))):
        return None
    return cached["result"]


def _save_result(checkpoint: Path, digest: str, result: dict[str, Any]) -> None:
    """Replace only this content key after a complete result is on disk."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkpoint.parent) as tmp:
        staged = Path(tmp) / "inspection.json"
        json_dump(staged, {"version": _CHECKPOINT_VERSION,
                           "input_sha256": digest, "result": result})
        staged.replace(checkpoint)


def enrich_media(provider: LLMProvider, inventory: dict[str,Any], project_dir: Path, max_items: int=40, progress: Progress|None=None) -> dict[str,Any]:
    media=sorted((e for e in inventory["evidence"] if e.get("kind")=="media"),key=_priority,reverse=True)[:max_items]
    for index, e in enumerate(media, 1):
        with step(progress, f"Visual evidence {index}/{len(media)}"):
            p=resolve_evidence_path(e,project_dir); tmp=tempfile.TemporaryDirectory(); img=Path(tmp.name)/"inspection.jpg"
            try:
                suffix = p.suffix.lower()
                if suffix == ".svg": continue
                e.pop("ai_media", None)
                e.pop("ai_media_error", None)
                if suffix in {".mp4",".mov",".mkv",".webm"}: _video_keyframe(p,img)
                else: _bounded_image(p,img)
                system=(project_dir.parents[1]/"prompts"/"media-inspection.txt").read_text()
                signature = getattr(provider, "visual_cache_signature", lambda: None)()
                if isinstance(signature, dict):
                    try:
                        json.dumps(signature, allow_nan=False)
                    except (TypeError, ValueError, RecursionError):
                        signature = None
                else:
                    signature = None
                digest = _inspection_digest(img, system, signature) if signature is not None else None
                checkpoint = (project_dir/"manifests"/"media-inspection"/f"{digest}.json"
                              if digest is not None else None)
                cached = _cached_result(checkpoint, digest) if checkpoint is not None else None
                if cached is not None:
                    e["ai_media"] = cached
                    continue
                r=provider.complete_json_with_image(system,_INSPECTION_USER,img)
                if not _valid_result(r):
                    raise ValueError("Visual inspection returned incomplete structured output")
                if checkpoint is not None:
                    _save_result(checkpoint, digest, r)
                e["ai_media"]=r
            except Exception as ex:
                e["ai_media_error"]=str(ex)
            finally: tmp.cleanup()
    json_dump(project_dir/"manifests"/"evidence.json",inventory)
    return inventory
