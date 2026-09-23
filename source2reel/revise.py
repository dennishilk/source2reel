from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .providers import LLMProvider
from .schema import validate_episode
from .util import json_dump


def revise(provider: LLMProvider, episode: dict[str,Any], research: dict[str,Any], inventory: dict[str,Any], instruction: str, project_dir: Path) -> dict[str,Any]:
    payload={"instruction":instruction,"current_episode":episode,"research":research,"evidence_refs":[e["ref"] for e in inventory["evidence"]]}
    system=(project_dir.parents[1]/"prompts"/"revision.txt").read_text()
    out=provider.complete_json(system,json.dumps(payload,ensure_ascii=False))
    validate_episode(out,set(payload["evidence_refs"]))
    backup=project_dir/"revisions"; backup.mkdir(exist_ok=True)
    existing=sorted(backup.glob("episode-*.json")); json_dump(backup/f"episode-{len(existing)+1:03d}.json",episode)
    json_dump(project_dir/"episode.json",out)
    return out
