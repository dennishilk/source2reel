from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chunking import checkpointed_complete_json, fits_context, split_for_context
from .progress import Progress, step
from .providers import LLMProvider
from .schema import SCENE_TYPES, validate_episode
from .util import json_dump


def _media_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {k: e.get(k) for k in ("ref", "relative_path", "kind", "mime", "media", "ai_media", "evidence_role")}
        for e in inventory["evidence"]
        if e["kind"] == "media"
    ]


def _evidence_index(inventory: dict[str, Any], refs: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        {k: e.get(k) for k in ("ref", "relative_path", "kind", "line_start", "line_end", "mime", "evidence_role")}
        for e in inventory["evidence"]
        if refs is None or e["ref"] in refs
    ]


def _research_refs(research: dict[str, Any]) -> set[str]:
    refs = {
        ref
        for fact in research.get("facts", [])
        for ref in fact.get("evidence_refs", [])
        if isinstance(ref, str)
    }
    refs.update(
        asset.get("evidence_ref")
        for asset in research.get("assets", [])
        if isinstance(asset, dict) and isinstance(asset.get("evidence_ref"), str)
    )
    return refs


def _make_ask(
    research: dict[str, Any],
    media: list[dict[str, Any]],
    evidence_index: list[dict[str, Any]],
    title_hint: str,
    instructions: str,
) -> dict[str, Any]:
    return {
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "allowed_scene_types": sorted(SCENE_TYPES),
        "research": research,
        "media_inventory": media,
        "evidence_index": evidence_index,
        "output_contract": {
            "version": 1,
            "title": "English title",
            "slug": "short-slug",
            "summary": "English summary",
            "scenes": [{
                "id": "s001",
                "type": "HERO",
                "title": "English on-screen title",
                "narration": "English narration",
                "evidence_refs": ["E0001"],
                "asset_ref": "E0001",
                "annotations": [],
                "pad_after_seconds": 0.5,
                "diagram": {},
                "notes": "",
            }],
        },
        "optional_structured_scene_fields": {
            "media.start_seconds": "non-negative seconds into an authentic video, when evidence calls for an offset",
            "captions.enabled": "boolean scene override; static and video evidence scenes default on",
            "diagram.nodes": "at least two explicit evidence-supported labels for diagram scenes",
        },
    }


def _planner_records(
    research: dict[str, Any],
    media: list[dict[str, Any]],
    evidence_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_ref = {e["ref"]: e for e in evidence_index}
    media_by_ref = {m["ref"]: m for m in media}
    records: list[dict[str, Any]] = []
    referenced_media: set[str] = set()

    for fact in research.get("facts", []):
        refs = [r for r in fact.get("evidence_refs", []) if r in evidence_by_ref]
        fact_media = [media_by_ref[r] for r in refs if r in media_by_ref]
        referenced_media.update(m["ref"] for m in fact_media)
        records.append({
            "kind": "fact",
            "claim": fact.get("claim", ""),
            "phase": fact.get("phase", "unknown"),
            "confidence": fact.get("confidence", "low"),
            "evidence_refs": refs,
            "evidence": [evidence_by_ref[r] for r in refs],
            "media": fact_media,
        })

    for asset in research.get("assets", []):
        ref = asset.get("evidence_ref")
        if ref not in evidence_by_ref:
            continue
        records.append({
            "kind": "asset_hint",
            "asset": asset,
            "evidence": [evidence_by_ref[ref]],
            "media": [media_by_ref[ref]] if ref in media_by_ref else [],
        })
        if ref in media_by_ref:
            referenced_media.add(ref)

    for item in media:
        if item["ref"] in referenced_media:
            continue
        records.append({
            "kind": "media",
            "media": item,
            "evidence": [evidence_by_ref[item["ref"]]] if item["ref"] in evidence_by_ref else [],
        })
    return records


def _compact_payload(level: int, part: int, records: list[dict[str, Any]], title_hint: str, instructions: str) -> dict[str, Any]:
    return {
        "level": level,
        "part": part,
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "records": records,
        "required_output": {
            "capsules": [{
                "claim": "one concise evidence-grounded planning fact or visual observation",
                "evidence_refs": ["E0001"],
                "media_refs": ["E0002"],
                "phase": "development|final|background|unknown",
                "confidence": "high|medium|low",
                "visual_purpose": "why this evidence or media matters to the storyboard",
            }]
        },
    }


def _part_refs(records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Refs actually supplied in one compaction request, including carried capsules."""
    evidence: set[str] = set()
    media: set[str] = set()
    for record in records:
        if record.get("kind") == "capsule":
            capsule = record["capsule"]
            evidence.update(capsule.get("evidence_refs", []))
            media.update(capsule.get("media_refs", []))
            continue
        evidence.update(record.get("evidence_refs", []))
        evidence.update(item["ref"] for item in record.get("evidence", []))
        if record.get("asset"):
            evidence.add(record["asset"]["evidence_ref"])
        provided_media = record.get("media") or []
        if isinstance(provided_media, dict):
            provided_media = [provided_media]
        media.update(item["ref"] for item in provided_media)
    evidence.update(media)
    return evidence, media


def _normalize_capsules(
    result: dict[str, Any],
    valid_refs: set[str],
    media_refs: set[str],
) -> dict[str, Any]:
    capsules = []
    for capsule in result.get("capsules", []):
        if not isinstance(capsule, dict):
            continue
        claim = str(capsule.get("claim", "")).strip()
        if not claim:
            continue
        mrefs = [r for r in capsule.get("media_refs", []) if r in media_refs]
        refs = [r for r in capsule.get("evidence_refs", []) if r in valid_refs]
        refs = list(dict.fromkeys(refs + mrefs))
        if not refs:
            continue
        phase = capsule.get("phase", "unknown")
        if phase not in {"development", "final", "background", "unknown"}:
            phase = "unknown"
        confidence = capsule.get("confidence", "low")
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        capsules.append({
            "claim": claim,
            "evidence_refs": refs,
            "media_refs": list(dict.fromkeys(mrefs)),
            "phase": phase,
            "confidence": confidence,
            "visual_purpose": str(capsule.get("visual_purpose", "")).strip(),
        })
    return {"capsules": capsules}


def _dedupe_capsules(capsules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for capsule in capsules:
        marker = (capsule["claim"].casefold(), tuple(capsule["evidence_refs"]))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(capsule)
    return out


def _capsules_to_research(capsules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "facts": [{
            "claim": c["claim"],
            "evidence_refs": c["evidence_refs"],
            "phase": c["phase"],
            "confidence": c["confidence"],
        } for c in capsules],
        "assets": [{
            "evidence_ref": ref,
            "purpose": c["visual_purpose"],
            "authentic_project_media": True,
        } for c in capsules for ref in c["media_refs"]],
    }


def _repair_episode_shape(
    value: dict[str, Any],
    valid_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Apply only deterministic schema repairs; never invent editorial content."""
    if not isinstance(value, dict):
        raise ValueError("planner output must be a JSON object")
    if isinstance(value.get("episode"), dict) and "scenes" not in value:
        value = value["episode"]
    ep = dict(value)
    if ep.get("version") == "1" or ("version" not in ep and isinstance(ep.get("scenes"), list)):
        ep["version"] = 1
    scenes = ep.get("scenes")
    if isinstance(scenes, list):
        repaired = []
        for index, raw_scene in enumerate(scenes, 1):
            if not isinstance(raw_scene, dict):
                repaired.append(raw_scene)
                continue
            scene = dict(raw_scene)
            if not scene.get("id"):
                scene["id"] = f"s{index:03d}"
            refs = scene.get("evidence_refs")
            evidence_scene = scene.get("type") in {
                "HERO", "PROJECT_EVIDENCE", "TERMINAL_EVIDENCE", "HARDWARE_EVIDENCE"
            }
            if evidence_scene and isinstance(refs, list):
                asset_ref = scene.get("asset_ref")
                if not asset_ref and len(refs) == 1:
                    scene["asset_ref"] = refs[0]
                elif asset_ref and asset_ref not in refs:
                    if valid_refs is None or asset_ref in valid_refs:
                        scene["evidence_refs"] = refs + [asset_ref]
                    elif valid_refs is not None:
                        valid_scene_refs = [ref for ref in refs if ref in valid_refs]
                        if len(valid_scene_refs) == 1:
                            scene["asset_ref"] = valid_scene_refs[0]
                if not refs and scene.get("asset_ref") and (
                    valid_refs is None or scene["asset_ref"] in valid_refs
                ):
                    scene["evidence_refs"] = [scene["asset_ref"]]
            repaired.append(scene)
        ep["scenes"] = repaired
    return ep


def _complete_episode(
    provider: LLMProvider,
    system: str,
    ask: dict[str, Any],
    valid_refs: set[str],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        payload = dict(ask)
        if last_error is not None:
            payload["validation_feedback"] = (
                "The previous storyboard response failed schema validation: "
                f"{last_error}. Return the requested top-level episode object exactly."
            )
        raw = provider.complete_json(system, json.dumps(payload, ensure_ascii=False))
        try:
            episode = _repair_episode_shape(raw, valid_refs)
            validate_episode(episode, valid_refs)
            return episode
        except (ValueError, TypeError) as exc:
            last_error = exc
    assert last_error is not None
    raise ValueError(f"planner output invalid after {attempts} attempt(s): {last_error}")


def plan(
    provider: LLMProvider,
    research: dict[str, Any],
    inventory: dict[str, Any],
    project_dir: Path,
    title_hint: str,
    instructions: str = "",
    context_size: int = 32768,
    output_reserve_tokens: int = 4096,
    safety_tokens: int = 1024,
    max_retries: int = 2,
    max_reduce_levels: int = 4,
    progress: Progress | None = None,
) -> dict[str, Any]:
    valid = {e["ref"] for e in inventory["evidence"]}
    media = _media_inventory(inventory)

    candidate_refs = _research_refs(research) | {m["ref"] for m in media}
    evidence_index = _evidence_index(inventory, candidate_refs)
    system = (project_dir.parents[1] / "prompts" / "storyboard.txt").read_text()

    direct_ask = _make_ask(research, media, evidence_index, title_hint, instructions)
    direct_user = json.dumps(direct_ask, ensure_ascii=False)
    if fits_context(system, direct_user, context_size, output_reserve_tokens, safety_tokens):
        json_dump(project_dir / "manifests" / "planner-evidence.json", {
            "version": 1,
            "strategy": "direct",
            "research": research,
            "media_inventory": media,
            "evidence_index": evidence_index,
        })
        with step(progress, "Generating storyboard"):
            ep = _complete_episode(provider, system, direct_ask, valid, max_retries)
        json_dump(project_dir / "episode.json", ep)
        return ep

    compact_system = (project_dir.parents[1] / "prompts" / "planner_compact.txt").read_text()
    records = _planner_records(research, media, evidence_index)
    media_ref_set = {m["ref"] for m in media}
    make_compact_payload = lambda level, part, batch: _compact_payload(level, part, batch, title_hint, instructions)

    for level in range(1, max(1, max_reduce_levels) + 1):
        chunks = split_for_context(
            records,
            compact_system,
            lambda part, batch: make_compact_payload(level, part, batch),
            context_size,
            output_reserve_tokens,
            safety_tokens,
        )
        capsules: list[dict[str, Any]] = []
        for part, batch in enumerate(chunks, 1):
            checkpoint = (
                project_dir / "manifests" / "planner-compact-parts"
                / f"level-{level:02d}-part-{part:03d}.json"
            )
            provided_refs, provided_media = _part_refs(batch)
            with step(progress, f"Planning evidence — level {level}, part {part}/{len(chunks)}"):
                result = checkpointed_complete_json(
                    provider,
                    compact_system,
                    make_compact_payload(level, part, batch),
                    checkpoint,
                    lambda value: _normalize_capsules(
                        value, valid & provided_refs, media_ref_set & provided_media
                    ),
                    max_retries=max_retries,
                )
            capsules.extend(result["capsules"])

        capsules = _dedupe_capsules(capsules)
        if not capsules:
            raise RuntimeError("Planner compaction returned no evidence-grounded capsules")

        compact_research = _capsules_to_research(capsules)
        selected_refs = _research_refs(compact_research)
        selected_media_refs = {
            ref
            for capsule in capsules
            for ref in capsule.get("media_refs", [])
        }
        compact_media = [m for m in media if m["ref"] in selected_media_refs]
        compact_index = _evidence_index(inventory, selected_refs | selected_media_refs)
        ask = _make_ask(compact_research, compact_media, compact_index, title_hint, instructions)
        user = json.dumps(ask, ensure_ascii=False)

        if fits_context(system, user, context_size, output_reserve_tokens, safety_tokens):
            json_dump(project_dir / "manifests" / "planner-evidence.json", {
                "version": 1,
                "strategy": "map-reduce",
                "levels": level,
                "research": compact_research,
                "media_inventory": compact_media,
                "evidence_index": compact_index,
            })
            with step(progress, "Generating storyboard"):
                ep = _complete_episode(provider, system, ask, valid, max_retries)
            json_dump(project_dir / "episode.json", ep)
            return ep

        records = [{"kind": "capsule", "capsule": capsule} for capsule in capsules]

    raise RuntimeError(
        "Planner payload still exceeds the configured context budget after "
        f"{max_reduce_levels} compaction level(s)"
    )
