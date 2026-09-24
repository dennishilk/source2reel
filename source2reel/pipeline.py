from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
from .config import load_engine_config
from .ingest import ingest
from .inventory import build_inventory
from .media_ai import enrich_media
from .planner import plan
from .progress import Progress, step
from .providers import provider_from_config
from .renderer import build_episode
from .research import research
from .util import json_dump, json_load, slugify


def project_title_from_source(source: str) -> str:
    if source.startswith(("http://","https://")):
        p=urlparse(source); return Path(p.path.rstrip("/")).name or p.netloc
    return Path(source).name


def _ingest_many(sources: list[str], pdir: Path, max_pages: int, progress: Progress|None=None) -> list[Path]:
    roots=[]; records=[]
    for i,source in enumerate(sources,1):
        ns=f"source-{i:02d}"
        with step(progress, f"Ingesting source {i}/{len(sources)}"):
            root=ingest(source,pdir,max_pages=max_pages,namespace=ns)
        roots.append(root)
        detail=pdir/"sources"/ns/"source.json"
        rec=json_load(detail) if detail.exists() else {"source":source,"local_path":str(root)}
        rec["namespace"]=ns; records.append(rec)
    json_dump(pdir/"sources"/"source.json",{"version":1,"sources":records})
    return roots


def _context_options(cfg: dict) -> dict:
    chunking = cfg.get("chunking", {})
    return {
        "context_size": int(cfg.get("local_ai", {}).get("context_size", 32768)),
        "output_reserve_tokens": int(chunking.get("output_reserve_tokens", 4096)),
        "safety_tokens": int(chunking.get("safety_tokens", 1024)),
        "max_retries": int(chunking.get("max_retries", 2)),
    }


def create(root: Path, sources: list[str], slug: str|None, instructions: str, preview_espeak: bool, stop_after_storyboard: bool, cfg_path: Path|None=None, progress: Progress|None=None) -> Path:
    if not sources: raise ValueError("At least one source is required")
    slug=slug or slugify(project_title_from_source(sources[0])); pdir=root/"projects"/slug; pdir.mkdir(parents=True,exist_ok=True)
    title_hint=project_title_from_source(sources[0])
    cfg=load_engine_config(root,cfg_path)
    progress=progress or Progress()
    source_roots=_ingest_many(sources,pdir,int(cfg.get("ingest",{}).get("max_web_pages",12)),progress)
    with progress.step("Building evidence inventory"):
        inv=build_inventory(source_roots,pdir)
    provider=provider_from_config(cfg)
    if cfg.get("vision",{}).get("enabled",False):
        with progress.step("Inspecting visual evidence", heartbeat=False):
            inv=enrich_media(provider,inv,pdir,int(cfg.get("vision",{}).get("max_items",40)),progress=progress)
    context = _context_options(cfg)
    with progress.step("Researching evidence", heartbeat=False):
        res=research(provider,inv,pdir,int(cfg["research"].get("batch_chars",45000)),
                     title_hint=title_hint,instructions=instructions,progress=progress,**context)
    with progress.step("Planning storyboard", heartbeat=False):
        ep=plan(
            provider,res,inv,pdir,title_hint,instructions,
            **context,progress=progress,
            max_reduce_levels=int(cfg.get("chunking",{}).get("max_reduce_levels",4)),
        )
    if stop_after_storyboard:
        out=pdir/"episode.json"
        progress.ready("Storyboard ready for review",out)
        return out
    with progress.step("Rendering episode", heartbeat=False):
        out=build_episode(root,pdir,ep,inv,preview_espeak,cfg,progress=progress)
    progress.ready("Episode ready",out)
    return out


def build_existing(root: Path, project: str, preview_espeak: bool=False, cfg_path: Path|None=None, progress: Progress|None=None) -> Path:
    progress=progress or Progress()
    pdir=root/"projects"/project; ep=json_load(pdir/"episode.json"); inv=json_load(pdir/"manifests"/"evidence.json")
    cfg=load_engine_config(root,cfg_path)
    with progress.step("Rendering episode", heartbeat=False):
        out=build_episode(root,pdir,ep,inv,preview_espeak,cfg,progress=progress)
    progress.ready("Episode ready",out)
    return out
