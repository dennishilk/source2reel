from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
from .config import load_engine_config
from .ingest import ingest
from .inventory import build_inventory
from .media_ai import enrich_media
from .planner import plan
from .providers import provider_from_config
from .renderer import build_episode
from .research import research
from .util import json_dump, json_load, slugify


def project_title_from_source(source: str) -> str:
    if source.startswith(("http://","https://")):
        p=urlparse(source); return Path(p.path.rstrip("/")).name or p.netloc
    return Path(source).name


def _ingest_many(sources: list[str], pdir: Path, max_pages: int) -> list[Path]:
    roots=[]; records=[]
    for i,source in enumerate(sources,1):
        ns=f"source-{i:02d}"
        root=ingest(source,pdir,max_pages=max_pages,namespace=ns); roots.append(root)
        detail=pdir/"sources"/ns/"source.json"
        rec=json_load(detail) if detail.exists() else {"source":source,"local_path":str(root)}
        rec["namespace"]=ns; records.append(rec)
    json_dump(pdir/"sources"/"source.json",{"version":1,"sources":records})
    return roots


def create(root: Path, sources: list[str], slug: str|None, instructions: str, preview_espeak: bool, stop_after_storyboard: bool, cfg_path: Path|None=None) -> Path:
    if not sources: raise ValueError("At least one source is required")
    slug=slug or slugify(project_title_from_source(sources[0])); pdir=root/"projects"/slug; pdir.mkdir(parents=True,exist_ok=True)
    cfg=load_engine_config(root,cfg_path)
    source_roots=_ingest_many(sources,pdir,int(cfg.get("ingest",{}).get("max_web_pages",12)))
    inv=build_inventory(source_roots,pdir); provider=provider_from_config(cfg)
    if cfg.get("vision",{}).get("enabled",False): inv=enrich_media(provider,inv,pdir,int(cfg.get("vision",{}).get("max_items",40)))
    res=research(provider,inv,pdir,int(cfg["research"].get("batch_chars",45000)))
    ep=plan(provider,res,inv,pdir,project_title_from_source(sources[0]),instructions)
    if stop_after_storyboard: return pdir/"episode.json"
    return build_episode(root,pdir,ep,inv,preview_espeak,cfg)


def build_existing(root: Path, project: str, preview_espeak: bool=False, cfg_path: Path|None=None) -> Path:
    pdir=root/"projects"/project; ep=json_load(pdir/"episode.json"); inv=json_load(pdir/"manifests"/"evidence.json")
    cfg=load_engine_config(root,cfg_path)
    return build_episode(root,pdir,ep,inv,preview_espeak,cfg)
