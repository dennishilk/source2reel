from __future__ import annotations
import argparse, os, shutil
from pathlib import Path
from .config import load_engine_config
from .doctor import run_doctor
from .ingest import ingest
from .inventory import build_inventory
from .pipeline import build_existing, create
from .providers import provider_from_config
from .revise import revise
from .util import json_load
from .paths import local_environment, runtime_paths
from .session import interactive, start, status, stop_all
from .tts_engine import render_scene_audio


def root_from_here() -> Path:
    env=os.environ.get("SOURCE2REEL_ROOT")
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[1]


VOICE_TEST_PASSAGE = (
    "Cisco CP-9951 runs DOOM locally on ARMv6 and MontaVista Linux. "
    "The framebuffer at /dev/fb1 shows the game. "
    "GitHub preserves the evidence. Apollo DSKY is another engineering story."
)


def voice_test(root: Path, preview_espeak: bool = False) -> Path:
    cfg = load_engine_config(root)
    out = runtime_paths(root)["output"] / "tests"
    raw = out / "voice-test.raw.wav"
    normalized = out / "voice-test.wav"
    render_scene_audio(VOICE_TEST_PASSAGE, raw, normalized, root, cfg, preview_espeak)
    raw.unlink(missing_ok=True)
    return normalized


def clean(root: Path, yes: bool = False, models: bool = False) -> None:
    if status(root):
        raise RuntimeError("Stop Source2Reel-managed services with ./s2r stop all first")
    paths = runtime_paths(root)
    targets = [root / ".venv", root / ".venv-voice-kokoro",
               paths["cache"], paths["runtime"], paths["output"]]
    if models:
        targets.append(paths["models"])
    for path in targets:
        if not path.exists() and not path.is_symlink():
            continue
        if path.parent.resolve() != root.resolve():
            raise RuntimeError(f"Refusing to clean outside Source2Reel root: {path}")
        print(("REMOVE " if yes else "WOULD REMOVE ") + str(path))
        if yes:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
    if not yes:
        print("Dry run only. Add --yes to remove these generated paths; --models also removes local model files.")


def main(argv=None):
    root=root_from_here()
    ap=argparse.ArgumentParser(
        prog="source2reel",
        description="Local, evidence-first AI-assisted technical explainer production engine",
    )
    sub=ap.add_subparsers(dest="cmd")
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
    v=sub.add_parser("voice-test")
    v.add_argument("--preview-espeak",action="store_true")
    ai=sub.add_parser("ai")
    ai.add_argument("action",choices=("start","status","stop"))
    ai.add_argument("--backend",choices=("vulkan","cpu","rocm"))
    stop=sub.add_parser("stop")
    stop.add_argument("target",choices=("all",))
    cleanup=sub.add_parser("clean")
    cleanup.add_argument("--yes",action="store_true",help="confirm removal of generated state")
    cleanup.add_argument("--models",action="store_true",help="also remove local model artifacts")
    a=ap.parse_args(argv)
    if a.cmd is None:
        interactive(root)
        return
    if a.cmd=="doctor": raise SystemExit(run_doctor(root,a.config))
    if a.cmd=="voice-test":
        print(voice_test(root,a.preview_espeak))
        return
    if a.cmd=="clean":
        clean(root,a.yes,a.models)
        return
    if a.cmd=="stop" or (a.cmd=="ai" and a.action=="stop"):
        print(f"Stopped {stop_all(root)} Source2Reel-managed process(es)")
        return
    if a.cmd=="ai":
        if a.action=="status":
            print(status(root) or "No Source2Reel-managed processes")
        elif a.action=="start":
            print(start(root,a.backend))
        return
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
