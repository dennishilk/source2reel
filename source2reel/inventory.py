from __future__ import annotations
import json, mimetypes, subprocess
from pathlib import Path
from typing import Any, Iterable
from PIL import Image
from .ingest import DOC_EXTS, MEDIA_EXTS, SKIP_DIRS
from .util import json_dump, read_text_lossy, sha256_file

# These paths often contain demonstrations of a source project. They remain
# citable evidence; the role only tells research how they relate to the root.
_EMBEDDED_ROOTS = {
    "example", "examples", "sample", "samples", "demo", "demos",
    "fixture", "fixtures", "reference", "references", "test-projects", "projects",
    "test", "tests",
}
_GENERATED_FILES = {
    "episode.json", "research.json", "research-compact.json",
    "planner-evidence.json", "evidence.json",
}
_GENERATED_DIRS = {"manifests", "research-parts", "research-compact-parts", "planner-compact-parts"}


def evidence_role(relative: Path, has_primary_siblings: bool) -> str:
    """Conservative source-relative hint, not an exclusion or truth ranking."""
    parts = tuple(part.lower() for part in relative.parts)
    if not has_primary_siblings or len(parts) < 2 or parts[0] not in _EMBEDDED_ROOTS:
        return "primary"
    name = parts[-1]
    if (any(part in _GENERATED_DIRS for part in parts[1:-1])
            or name in _GENERATED_FILES or (name.startswith("episode-review") and name.endswith(".json"))):
        return "generated_artifact"
    return "embedded_reference"


def _has_primary_siblings(root: Path, paths: list[Path]) -> bool:
    """A source consisting only of examples/projects is itself the subject."""
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in DOC_EXTS | MEDIA_EXTS:
            continue
        relative = path.relative_to(root) if root.is_dir() else Path(path.name)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative.parts[0].lower() not in _EMBEDDED_ROOTS:
            return True
    return False


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
        # A minified JSON artifact can be a single enormous line. Keep its
        # pieces citeable at the original line instead of exceeding context.
        if target_chars > 0 and len(lines[start]) > target_chars:
            for offset in range(0, len(lines[start]), target_chars):
                yield start+1, start+1, lines[start][offset:offset+target_chars]
            start += 1
            continue
        size=0; end=start
        while end < len(lines) and (size < target_chars or end==start):
            if end > start and target_chars > 0 and len(lines[end]) > target_chars:
                break
            size += len(lines[end])+1; end += 1
        yield start+1,end,"\n".join(lines[start:end])
        if end >= len(lines): break
        start=max(start+1,end-overlap_lines)


def build_inventory(source_root: Path | list[Path], project_dir: Path, max_file_bytes: int = 250000, chunk_chars: int=8000) -> dict[str, Any]:
    entries=[]; candidates=[]; license_candidates=[]
    roots=source_root if isinstance(source_root,list) else [source_root]
    for root_index, root in enumerate(roots,1):
        paths=sorted(root.rglob("*") if root.is_dir() else [root])
        has_primary_siblings = _has_primary_siblings(root, paths)
        prefix=f"source-{root_index:02d}" if len(roots)>1 else ""
        for p in paths:
            if not p.is_file(): continue
            if any(part in SKIP_DIRS for part in p.parts): continue
            if p.name in {"source.json","website-manifest.json","local-source-manifest.json"}: continue
            ext=p.suffix.lower()
            if ext not in DOC_EXTS|MEDIA_EXTS: continue
            try: source_relative=p.relative_to(root)
            except ValueError: source_relative=Path(p.name)
            role=evidence_role(source_relative,has_primary_siblings)
            rel=str(source_relative)
            if prefix: rel=f"{prefix}/{rel}"
            stored,path_base=_stored_path(p,project_dir); digest=sha256_file(p); mime=mimetypes.guess_type(p.name)[0]
            if ext in DOC_EXTS:
                text=read_text_lossy(p,max_file_bytes)
                for line_start,line_end,excerpt in _text_chunks(text,chunk_chars):
                    ref=f"E{len(entries)+1:04d}"
                    entries.append({
                        "ref":ref,"path":stored,"path_base":path_base,"relative_path":rel,"evidence_role":role,
                        "size":p.stat().st_size,"sha256":digest,"mime":mime,"kind":"document",
                        "line_start":line_start,"line_end":line_end,"excerpt":excerpt,
                        "truncated_file":p.stat().st_size>max_file_bytes
                    })
                    if p.name.upper().startswith(("LICENSE","COPYING","NOTICE")):
                        license_candidates.append(ref)
            else:
                ref=f"E{len(entries)+1:04d}"
                entries.append({
                    "ref":ref,"path":stored,"path_base":path_base,"relative_path":rel,"evidence_role":role,
                    "size":p.stat().st_size,"sha256":digest,"mime":mime,"kind":"media","media":_media_meta(p)
                })
                candidates.append(ref)
    out={"version":1,"source_roots":[str(x) for x in roots],"evidence":entries,"media_candidates":candidates,"license_candidates":license_candidates}
    json_dump(project_dir/"manifests"/"evidence.json",out)
    return out
