from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any
from . import __version__
from .config import load_engine_config, profile_paths
from .schema import validate_episode, validate_presentation, EVIDENCE_TYPES
from .inventory import resolve_evidence_path
from .templates import render_scene, render_video_shell
from .captions import ass_filter, caption_enabled, estimate_events, write_ass
from .tts_engine import render_scene_audio
from .util import ffprobe_duration, json_dump, json_load, run, sha256_file

VIDEO_EXTS={".mp4",".mov",".mkv",".webm"}


def _evidence_map(inv): return {e["ref"]:e for e in inv["evidence"]}


def _visual_scene(scene: dict[str,Any], presentation: dict[str,Any]) -> dict[str,Any]:
    title=presentation.get("scene_titles",{}).get(scene["id"])
    return {**scene,"title":title} if title else scene


def _encode_static_segment(frame: Path, audio: Path, seg: Path, dur: float, pad: float, captions: Path|None=None):
    video_filter=f"[0:v]{ass_filter(captions)}[vo];" if captions else "[0:v]null[vo];"
    run(["ffmpeg","-y","-loglevel","error","-loop","1","-framerate","30","-i",frame,"-i",audio,
         "-filter_complex",video_filter+f"[1:a]apad=pad_dur={pad}[a]","-map","[vo]","-map","[a]","-t",f"{dur:.3f}",
         "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p","-r","30",
         "-c:a","aac","-b:a","192k","-ar","48000","-movflags","+faststart",seg])


def _encode_video_evidence(shell: Path, source_video: Path, audio: Path, seg: Path, dur: float, pad: float, start_seconds: float=0.0, captions: Path|None=None):
    cmd=["ffmpeg","-y","-loglevel","error","-loop","1","-framerate","30","-i",shell]
    if start_seconds>0: cmd += ["-ss",f"{start_seconds:.3f}"]
    height=630 if captions else 800
    video_filter=f"[1:v]setpts=PTS-STARTPTS,scale=1776:{height}:force_original_aspect_ratio=decrease[v];"
    video_filter+=f"[0:v][v]overlay=x=(W-w)/2:y=154+({height}-h)/2:shortest=0[base];"
    video_filter+=f"[base]{ass_filter(captions)}[vo];" if captions else "[base]null[vo];"
    cmd += ["-stream_loop","-1","-i",source_video,"-i",audio,
            "-filter_complex",
            video_filter+f"[2:a]apad=pad_dur={pad}[a]",
            "-map","[vo]","-map","[a]","-t",f"{dur:.3f}",
            "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p","-r","30",
            "-c:a","aac","-b:a","192k","-ar","48000","-movflags","+faststart",seg]
    run(cmd)


def build_episode(root: Path, project_dir: Path, episode: dict[str,Any], inventory: dict[str,Any], preview_espeak=False, cfg: dict[str,Any]|None=None) -> Path:
    cfg=cfg or load_engine_config(root)
    emap=_evidence_map(inventory); validate_episode(episode,set(emap))
    sidecar=project_dir/"presentation.json"
    presentation={**episode.get("presentation",{}), **(json_load(sidecar) if sidecar.exists() else {})}
    validate_presentation(presentation,any(s["type"]=="OUTRO" for s in episode["scenes"]))
    work=project_dir/"work"; outdir=project_dir/"output"; frames=work/"frames"; audio=work/"audio"; segs=work/"segments"
    for d in (frames,audio,segs,outdir): d.mkdir(parents=True,exist_ok=True)
    manifest=[]; transcript=[]; segment_paths=[]; backends=set()
    for idx,scene in enumerate(episode["scenes"],1):
        sid=scene["id"]; aref=scene.get("asset_ref"); evidence_item=emap.get(aref) if aref else None; asset=resolve_evidence_path(evidence_item,project_dir) if evidence_item else None
        visual_scene=_visual_scene(scene,presentation)
        frame=frames/f"{idx:03d}-{sid}.png"
        is_video=bool(asset and asset.suffix.lower() in VIDEO_EXTS and scene["type"] in EVIDENCE_TYPES)
        captions_on=caption_enabled(scene,is_video,cfg)
        if is_video: render_video_shell(visual_scene,frame,root,cfg,captions_on)
        else: render_scene(visual_scene,asset,frame,root,cfg,evidence_item,captions_on,presentation)
        raw=audio/f"{idx:03d}-{sid}-raw.wav"; norm=audio/f"{idx:03d}-{sid}.wav"
        backend=render_scene_audio(scene["narration"],raw,norm,root,cfg,preview_espeak); backends.add(backend)
        pad=float(scene.get("pad_after_seconds",0.5)); voice_duration=ffprobe_duration(norm); dur=voice_duration+pad; seg=segs/f"{idx:03d}-{sid}.mp4"
        events=estimate_events(scene["narration"],voice_duration) if captions_on else []
        caption_path=work/"captions"/f"{idx:03d}-{sid}.ass" if events else None
        if caption_path: write_ass(caption_path,events)
        if is_video:
            start=float((scene.get("media") or {}).get("start_seconds",0.0)); _encode_video_evidence(frame,asset,norm,seg,dur,pad,start,caption_path)
        else:
            _encode_static_segment(frame,norm,seg,dur,pad,caption_path)
        segment_paths.append(seg); transcript.append({"scene":sid,"narration":scene["narration"],"duration":dur})
        manifest.append({
            "scene":sid,"type":scene["type"],"asset_ref":aref,
            "asset_sha256":evidence_item.get("sha256") if evidence_item else None,
            "presentation":"video-fixed-frame" if is_video else "static-frame",
            "frame":str(frame.relative_to(project_dir)),"duration":dur,
            "captions":len(events),"media_start_seconds":start if is_video else None
        })
    concat=work/"concat.txt"; concat.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))
    final=outdir/f"{episode.get('slug') or project_dir.name}.mp4"
    run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",concat,"-c","copy","-movflags","+faststart",final])
    (outdir/"transcript.txt").write_text("\n\n".join(f"[{x['scene']}]\n{x['narration']}" for x in transcript)+"\n")
    final_sha=sha256_file(final)
    json_dump(outdir/"render-manifest.json",{"voice_backends":sorted(backends),"scenes":manifest,"final":final.name,"sha256":final_sha})
    shutil.copy2(project_dir/"episode.json",outdir/"storyboard.json")
    if sidecar.exists(): shutil.copy2(sidecar,outdir/"presentation.json")
    shutil.copy2(project_dir/"manifests"/"evidence.json",outdir/"evidence-manifest.json")
    research=project_dir/"manifests"/"research.json"
    if research.exists(): shutil.copy2(research,outdir/"research.json")
    source_manifest=project_dir/"sources"/"source.json"
    if source_manifest.exists(): shutil.copy2(source_manifest,outdir/"source-manifest.json")
    shutil.copy2(root/"LICENSES.md",outdir/"ENGINE_LICENSES.md")
    theme_path,voice_path=profile_paths(root,cfg)
    voice_cfg=voice_path.read_text()
    provenance={
        "engine":"Source2Reel","engine_version":__version__,"episode":episode.get("slug") or project_dir.name,"final_sha256":final_sha,
        "profile":cfg.get("profile",{}).get("name"),
        "theme_config":str(theme_path.relative_to(root)),
        "voice_config":str(voice_path.relative_to(root)),"voice_config_text":voice_cfg,
        "evidence_manifest":"evidence-manifest.json","storyboard":"storyboard.json","engine_licenses":"ENGINE_LICENSES.md",
        "license_candidate_refs":inventory.get("license_candidates",[]),
        "presentation":"presentation.json" if sidecar.exists() else "storyboard.json"
    }
    json_dump(outdir/"provenance.json",provenance)
    return final
