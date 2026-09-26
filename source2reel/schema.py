from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import math
import re

@dataclass(frozen=True)
class SceneContract:
    requires_asset_ref: bool = False
    allows_asset_ref: bool = False
    requires_facts: bool = True
    allows_facts: bool = True
    requires_diagram: bool = False
    allows_media_timing: bool = False
    canonical_optional_fields: frozenset[str] = frozenset({
        "annotations", "notes", "pad_after_seconds", "captions",
    })


_COMMON_OPTIONAL = frozenset({"annotations", "notes", "pad_after_seconds", "captions"})
_VISUAL_OPTIONAL = _COMMON_OPTIONAL | {"media"}
_DIAGRAM_OPTIONAL = _COMMON_OPTIONAL | {"diagram"}
SCENE_CONTRACTS = {
    **{name: SceneContract(requires_asset_ref=True, allows_asset_ref=True,
                           allows_media_timing=True,
                           canonical_optional_fields=_VISUAL_OPTIONAL)
       for name in ("HERO", "PROJECT_EVIDENCE", "TERMINAL_EVIDENCE", "HARDWARE_EVIDENCE")},
    **{name: SceneContract(requires_diagram=True, canonical_optional_fields=_DIAGRAM_OPTIONAL)
       for name in ("ARCHITECTURE_DIAGRAM", "DATA_FLOW", "TIMELINE")},
    **{name: SceneContract(canonical_optional_fields=_DIAGRAM_OPTIONAL)
       for name in ("CODE", "GRAPH")},
    **{name: SceneContract(requires_facts=False, canonical_optional_fields=_COMMON_OPTIONAL)
       for name in ("SECTION_TITLE", "OUTRO")},
    "SUMMARY": SceneContract(),
}
SCENE_TYPES = set(SCENE_CONTRACTS)
EVIDENCE_TYPES = {name for name, contract in SCENE_CONTRACTS.items()
                  if contract.requires_asset_ref}
DIAGRAM_TYPES = {name for name, contract in SCENE_CONTRACTS.items()
                 if contract.requires_diagram}
_TIMED_MEDIA_NOTE=re.compile(r"\b(?:start|seek|begin|footage|gameplay)\b[^.\n]{0,80}?\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b",re.I)


def validate_presentation(presentation: dict, has_outro: bool) -> None:
    if not isinstance(presentation,dict): raise ValueError("presentation must be an object")
    titles=presentation.get("scene_titles",{})
    if not isinstance(titles,dict) or any(not isinstance(k,str) or not isinstance(v,str) or not v.strip() or "\ufffd" in v for k,v in titles.items()):
        raise ValueError("presentation.scene_titles requires nonempty titles without replacement characters")
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


def validate_episode(
    ep: dict[str, Any], evidence_refs: set[str] | None = None,
    *, require_integrated_presentation: bool = False,
) -> None:
    if ep.get("version") != 1: raise ValueError("episode.version must be 1")
    if not isinstance(ep.get("title"),str) or not ep["title"].strip(): raise ValueError("episode.title missing")
    scenes=ep.get("scenes")
    if not isinstance(scenes,list) or not scenes: raise ValueError("episode.scenes must be non-empty")
    if "presentation" in ep and not isinstance(ep["presentation"],dict): raise ValueError("episode.presentation must be an object")
    ids=set()
    for s in scenes:
        if not isinstance(s, dict): raise ValueError("scene must be an object")
        sid=s.get("id")
        if not isinstance(sid,str) or not sid: raise ValueError("scene.id missing")
        if sid in ids: raise ValueError(f"duplicate scene id {sid}")
        ids.add(sid)
        if not isinstance(s.get("type"),str) or s["type"] not in SCENE_TYPES:
            raise ValueError(f"unsupported scene type {s.get('type')}")
        contract=SCENE_CONTRACTS[s["type"]]
        if not isinstance(s.get("narration"),str) or not s["narration"].strip(): raise ValueError(f"{sid}: narration missing")
        refs=s.get("evidence_refs",[])
        if not isinstance(refs,list) or any(not isinstance(ref,str) for ref in refs):
            raise ValueError(f"{sid}: evidence_refs must be a list of strings")
        if evidence_refs is not None:
            bad=[r for r in refs if r not in evidence_refs]
            if bad: raise ValueError(f"{sid}: unknown evidence refs {bad}")
        if contract.requires_facts and not refs:
            raise ValueError(f"{sid}: factual/content scene requires evidence_refs")
        if contract.requires_asset_ref:
            aref=s.get("asset_ref")
            if not aref: raise ValueError(f"{sid}: evidence scene requires asset_ref")
            if aref not in refs: raise ValueError(f"{sid}: asset_ref must also appear in evidence_refs")
        media=s.get("media",{})
        if not isinstance(media,dict): raise ValueError(f"{sid}: media must be an object")
        if "start_seconds" in media:
            start=media["start_seconds"]
            if type(start) not in (int,float) or not math.isfinite(start) or start<0:
                raise ValueError(f"{sid}: media.start_seconds must be finite and non-negative")
            if not contract.allows_media_timing:
                raise ValueError(f"{sid}: media.start_seconds requires an evidence scene")
        elif contract.allows_media_timing and _TIMED_MEDIA_NOTE.search(s.get("notes", "")):
            raise ValueError(f"{sid}: video offset in notes must be media.start_seconds")
        captions=s.get("captions",{})
        if not isinstance(captions,dict) or ("enabled" in captions and type(captions["enabled"]) is not bool):
            raise ValueError(f"{sid}: captions.enabled must be boolean")
        if contract.requires_diagram:
            diagram=s.get("diagram")
            if not isinstance(diagram,dict): raise ValueError(f"{sid}: diagram must be an object")
            nodes=diagram.get("nodes") or diagram.get("steps")
            if not isinstance(nodes,list) or not 2<=len(nodes)<=8 or any(
                not isinstance(n,(str,dict)) or not str(n if isinstance(n,str) else n.get("label") or n.get("name") or "").strip()
                for n in nodes
            ):
                raise ValueError(f"{sid}: {s['type']} requires 2–8 explicit, labeled diagram nodes")
    # Newly planned episodes must carry their own OUTRO metadata. Existing
    # frozen episodes may instead supply it through presentation.json at build
    # time; the renderer validates the merged presentation separately.
    if require_integrated_presentation:
        outro_indices = [index for index, scene in enumerate(scenes) if scene["type"] == "OUTRO"]
        if len(outro_indices) > 1: raise ValueError("episode may have only one OUTRO scene")
        if outro_indices and outro_indices[0] != len(scenes)-1:
            raise ValueError("OUTRO must be the final scene")
        presentation = ep.get("presentation", {})
        validate_presentation(presentation, bool(outro_indices))
        if "outro" in presentation and not outro_indices:
            raise ValueError("presentation.outro requires a final OUTRO scene")
