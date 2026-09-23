from __future__ import annotations
from typing import Any
import math
import re

SCENE_TYPES={"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE","ARCHITECTURE_DIAGRAM","DATA_FLOW","TIMELINE","CODE","GRAPH","SECTION_TITLE","SUMMARY","OUTRO"}
EVIDENCE_TYPES={"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE"}
DIAGRAM_TYPES={"ARCHITECTURE_DIAGRAM","DATA_FLOW","TIMELINE"}
_TIMED_MEDIA_NOTE=re.compile(r"\b(?:start|seek|begin|footage|gameplay)\b[^.\n]{0,80}?\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b",re.I)


def validate_presentation(presentation: dict, has_outro: bool) -> None:
    if not isinstance(presentation,dict): raise ValueError("presentation must be an object")
    if not has_outro: return
    outro=presentation.get("outro")
    if not isinstance(outro,dict): raise ValueError("OUTRO requires presentation.outro")
    headline=outro.get("headline")
    if not isinstance(headline,list) or not 1<=len(headline)<=3 or any(not isinstance(x,str) or not x.strip() for x in headline):
        raise ValueError("OUTRO headline requires 1–3 nonempty lines")
    links=outro.get("links")
    if not isinstance(links,list) or not 1<=len(links)<=3:
        raise ValueError("OUTRO requires 1–3 links")
    for link in links:
        if not isinstance(link,dict) or not isinstance(link.get("label"),str) or not link["label"].strip():
            raise ValueError("OUTRO link requires a label")
        url=link.get("url")
        if isinstance(url,str): url=[url]
        if not isinstance(url,list) or not 1<=len(url)<=3 or any(not isinstance(x,str) or not x.strip() for x in url):
            raise ValueError("OUTRO link requires 1–3 URL lines")


def validate_episode(ep: dict[str, Any], evidence_refs: set[str] | None = None) -> None:
    if ep.get("version") != 1: raise ValueError("episode.version must be 1")
    if not isinstance(ep.get("title"),str) or not ep["title"].strip(): raise ValueError("episode.title missing")
    scenes=ep.get("scenes")
    if not isinstance(scenes,list) or not scenes: raise ValueError("episode.scenes must be non-empty")
    if "presentation" in ep and not isinstance(ep["presentation"],dict): raise ValueError("episode.presentation must be an object")
    ids=set()
    for s in scenes:
        sid=s.get("id")
        if not isinstance(sid,str) or not sid: raise ValueError("scene.id missing")
        if sid in ids: raise ValueError(f"duplicate scene id {sid}")
        ids.add(sid)
        if s.get("type") not in SCENE_TYPES: raise ValueError(f"unsupported scene type {s.get('type')}")
        if not isinstance(s.get("narration"),str) or not s["narration"].strip(): raise ValueError(f"{sid}: narration missing")
        refs=s.get("evidence_refs",[])
        if not isinstance(refs,list): raise ValueError(f"{sid}: evidence_refs must be list")
        if evidence_refs is not None:
            bad=[r for r in refs if r not in evidence_refs]
            if bad: raise ValueError(f"{sid}: unknown evidence refs {bad}")
        if s["type"] not in {"SECTION_TITLE","OUTRO"} and not refs:
            raise ValueError(f"{sid}: factual/content scene requires evidence_refs")
        if s["type"] in EVIDENCE_TYPES:
            aref=s.get("asset_ref")
            if not aref: raise ValueError(f"{sid}: evidence scene requires asset_ref")
            if aref not in refs: raise ValueError(f"{sid}: asset_ref must also appear in evidence_refs")
        media=s.get("media",{})
        if not isinstance(media,dict): raise ValueError(f"{sid}: media must be an object")
        if "start_seconds" in media:
            start=media["start_seconds"]
            if type(start) not in (int,float) or not math.isfinite(start) or start<0:
                raise ValueError(f"{sid}: media.start_seconds must be finite and non-negative")
            if s["type"] not in EVIDENCE_TYPES:
                raise ValueError(f"{sid}: media.start_seconds requires an evidence scene")
        elif s["type"] in EVIDENCE_TYPES and _TIMED_MEDIA_NOTE.search(s.get("notes", "")):
            raise ValueError(f"{sid}: video offset in notes must be media.start_seconds")
        captions=s.get("captions",{})
        if not isinstance(captions,dict) or ("enabled" in captions and type(captions["enabled"]) is not bool):
            raise ValueError(f"{sid}: captions.enabled must be boolean")
        if s["type"] in DIAGRAM_TYPES:
            diagram=s.get("diagram")
            if not isinstance(diagram,dict): raise ValueError(f"{sid}: diagram must be an object")
            nodes=diagram.get("nodes") or diagram.get("steps")
            if not isinstance(nodes,list) or not 2<=len(nodes)<=8 or any(
                not isinstance(n,(str,dict)) or not str(n if isinstance(n,str) else n.get("label") or n.get("name") or "").strip()
                for n in nodes
            ):
                raise ValueError(f"{sid}: {s['type']} requires 2–8 explicit, labeled diagram nodes")
