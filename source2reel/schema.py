from __future__ import annotations
from typing import Any

SCENE_TYPES={"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE","ARCHITECTURE_DIAGRAM","DATA_FLOW","TIMELINE","CODE","GRAPH","SECTION_TITLE","SUMMARY","OUTRO"}
EVIDENCE_TYPES={"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE"}


def validate_episode(ep: dict[str, Any], evidence_refs: set[str] | None = None) -> None:
    if ep.get("version") != 1: raise ValueError("episode.version must be 1")
    if not isinstance(ep.get("title"),str) or not ep["title"].strip(): raise ValueError("episode.title missing")
    scenes=ep.get("scenes")
    if not isinstance(scenes,list) or not scenes: raise ValueError("episode.scenes must be non-empty")
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
