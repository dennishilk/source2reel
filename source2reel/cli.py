from __future__ import annotations
import argparse, os
from pathlib import Path
from .config import load_engine_config
from .doctor import run_doctor
from .ingest import ingest
from .inventory import build_inventory
from .pipeline import build_existing, create
from .providers import provider_from_config
from .revise import revise
from .util import json_load


def root_from_here() -> Path:
    env=os.environ.get("SOURCE2REEL_ROOT")
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[1]


def main(argv=None):
    root=root_from_here()
    ap=argparse.ArgumentParser(
        prog="source2reel",
        description="Local, evidence-first AI-assisted technical explainer production engine",
    )
    sub=ap.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("create")
    c.add_argument("sources", nargs="+")
    c.add_argument("--slug")
    c.add_argument("--instructions",default="")
    c.add_argument("--review",action="store_true",help="stop after storyboard for human review")
    c.add_argument("--preview-espeak",action="store_true")
    c.add_argument("--config",type=Path)
    b=sub.add_parser("build")
    b.add_argument("project")
    b.add_argument("--preview-espeak",action="store_true")
    b.add_argument("--config",type=Path)
    r=sub.add_parser("revise")
    r.add_argument("project")
    r.add_argument("instruction")
    r.add_argument("--build",action="store_true")
    r.add_argument("--preview-espeak",action="store_true")
    r.add_argument("--config",type=Path)
    i=sub.add_parser("inventory")
    i.add_argument("project")
    i.add_argument("source")
    d=sub.add_parser("doctor")
    d.add_argument("--config",type=Path)
    a=ap.parse_args(argv)
    if a.cmd=="doctor": raise SystemExit(run_doctor(root,a.config))
    if a.cmd=="create":
        out=create(root,a.sources,a.slug,a.instructions,a.preview_espeak,a.review,a.config)
        print(out)
        return
    if a.cmd=="build":
        print(build_existing(root,a.project,a.preview_espeak,a.config))
        return
    if a.cmd=="inventory":
        pdir=root/"projects"/a.project
        src=ingest(a.source,pdir)
        build_inventory(src,pdir)
        print(pdir/"manifests"/"evidence.json")
        return
    if a.cmd=="revise":
        pdir=root/"projects"/a.project
        cfg=load_engine_config(root,a.config)
        provider=provider_from_config(cfg)
        ep=json_load(pdir/"episode.json")
        inv=json_load(pdir/"manifests"/"evidence.json")
        research=json_load(pdir/"manifests"/"research.json")
        revise(provider,ep,research,inv,a.instruction,pdir)
        print(build_existing(root,a.project,a.preview_espeak,a.config) if a.build else pdir/"episode.json")
        return


if __name__=="__main__":
    main()
