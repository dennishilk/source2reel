from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .providers import LLMProvider
from .schema import SCENE_TYPES, validate_episode
from .util import json_dump



def plan(provider: LLMProvider, research: dict[str,Any], inventory: dict[str,Any], project_dir: Path, title_hint: str, instructions: str="") -> dict[str,Any]:
    media=[{k:e.get(k) for k in ("ref","relative_path","kind","mime","media","ai_media")} for e in inventory["evidence"] if e["kind"]=="media"]
    evidence_index=[{k:e.get(k) for k in ("ref","relative_path","kind","line_start","line_end","mime")} for e in inventory["evidence"]]
    ask={
        "project_title_hint":title_hint,
        "optional_instructions":instructions,
        "research":research,
        "media_inventory":media,
        "evidence_index":evidence_index,
        "output_contract":{
            "version":1,"title":"English title","slug":"short-slug","summary":"English summary",
            "scenes":[{
                "id":"s001","type":"HERO","title":"English on-screen title","narration":"English narration",
                "evidence_refs":["E0001"],"asset_ref":"E0001","annotations":[],"pad_after_seconds":0.5,
                "diagram":{},"notes":""
            }]
        },
        "optional_structured_scene_fields":{
            "media.start_seconds":"non-negative seconds into an authentic video, when evidence calls for an offset",
            "captions.enabled":"boolean scene override; static scenes default on, video defaults off",
            "diagram.nodes":"at least two explicit evidence-supported labels for diagram scenes"
        }
    }
    system=(project_dir.parents[1]/"prompts"/"storyboard.txt").read_text()
    ep=provider.complete_json(system,json.dumps(ask,ensure_ascii=False))
    valid={e["ref"] for e in inventory["evidence"]}
    validate_episode(ep,valid)
    json_dump(project_dir/"episode.json",ep)
    return ep
