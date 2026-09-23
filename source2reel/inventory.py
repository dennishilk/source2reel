from __future__ import annotations
import json, mimetypes, subprocess
from pathlib import Path
from typing import Any, Iterable
from PIL import Image
from .ingest import DOC_EXTS, MEDIA_EXTS, SKIP_DIRS
from .util import json_dump, read_text_lossy, sha256_file


def resolve_evidence_path(item: dict[str, Any], project_dir: Path) -> Path:
    raw=Path(item["path"])
    return (project_dir/raw).resolve() if item.get("path_base")=="project" else raw.expanduser().resolve()


def _stored_path(p: Path, project_dir: Path) -> tuple[str,str]:
    try:
        return str(p.resolve().relative_to(project_dir.resolve())), "project"
    except ValueError:
        return str(p.resolve()), "absolute"


def _media_meta(path: Path) -> dict[str, Any]:
    ext=path.suffix.lower(); meta={}
    try:
        if ext in {".png",".jpg",".jpeg",".webp",".gif"}:
            with Image.open(path) as im: meta.update(width=im.width,height=im.height,format=im.format)
        elif ext in {".mp4",".mov",".mkv",".webm"}:
            p=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration:stream=width,height,codec_name","-of","json",str(path)],capture_output=True,text=True,check=True)
            meta.update(json.loads(p.stdout))
    except Exception as e: meta["probe_error"]=str(e)
    return meta


def _text_chunks(text: str, target_chars: int=8000, overlap_lines: int=8) -> Iterable[tuple[int,int,str]]:
    lines=text.splitlines()
    if not lines:
        yield 1,1,""
        return
    start=0
    while start < len(lines):
        size=0; end=start
        while end < len(lines) and (size < target_chars or end==start):
            size += len(lines[end])+1; end += 1
        yield start+1,end,"\n".join(lines[start:end])
        if end >= len(lines): break
        start=max(start+1,end-overlap_lines)


def build_inventory(source_root: Path | list[Path], project_dir: Path, max_file_bytes: int = 250000, chunk_chars: int=8000) -> dict[str, Any]:
    entries=[]; candidates=[]; license_candidates=[]
    roots=source_root if isinstance(source_root,list) else [source_root]
    for root_index, root in enumerate(roots,1):
        paths=sorted(root.rglob("*") if root.is_dir() else [root])
        prefix=f"source-{root_index:02d}" if len(roots)>1 else ""
        for p in paths:
            if not p.is_file(): continue
            if any(part in SKIP_DIRS for part in p.parts): continue
            if p.name in {"source.json","website-manifest.json","local-source-manifest.json"}: continue
            ext=p.suffix.lower()
            if ext not in DOC_EXTS|MEDIA_EXTS: continue
            try: rel=str(p.relative_to(root))
            except ValueError: rel=p.name
            if prefix: rel=f"{prefix}/{rel}"
            stored,path_base=_stored_path(p,project_dir); digest=sha256_file(p); mime=mimetypes.guess_type(p.name)[0]
            if ext in DOC_EXTS:
                text=read_text_lossy(p,max_file_bytes)
                for line_start,line_end,excerpt in _text_chunks(text,chunk_chars):
                    ref=f"E{len(entries)+1:04d}"
                    entries.append({
                        "ref":ref,"path":stored,"path_base":path_base,"relative_path":rel,
                        "size":p.stat().st_size,"sha256":digest,"mime":mime,"kind":"document",
                        "line_start":line_start,"line_end":line_end,"excerpt":excerpt,
                        "truncated_file":p.stat().st_size>max_file_bytes
                    })
                    if p.name.upper().startswith(("LICENSE","COPYING","NOTICE")):
                        license_candidates.append(ref)
            else:
                ref=f"E{len(entries)+1:04d}"
                entries.append({
                    "ref":ref,"path":stored,"path_base":path_base,"relative_path":rel,
                    "size":p.stat().st_size,"sha256":digest,"mime":mime,"kind":"media","media":_media_meta(p)
                })
                candidates.append(ref)
    out={"version":1,"source_roots":[str(x) for x in roots],"evidence":entries,"media_candidates":candidates,"license_candidates":license_candidates}
    json_dump(project_dir/"manifests"/"evidence.json",out)
    return out
