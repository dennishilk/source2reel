from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .chunking import checkpointed_complete_json, checkpointed_split_json, fits_context, split_for_context
from .progress import Progress, step
from .providers import LLMProvider, StructuredOutputError
from .research import _consolidate, _reference_focus
from .schema import SCENE_TYPES, validate_episode, validate_presentation
from .util import json_dump, json_load


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


def _primary_anchors(
    research: dict[str, Any], inventory: dict[str, Any], media_refs: set[str],
) -> list[dict[str, Any]]:
    """Carry a small, diverse set of original primary facts across reductions."""
    entries = {entry["ref"]: entry for entry in inventory["evidence"]}
    primary_refs = {
        ref for ref in (_research_refs(research) | media_refs)
        if ref in entries and (entries[ref].get("evidence_role") or "primary") == "primary"
    }
    candidates = []
    for fact in research.get("facts", []):
        refs = [r for r in fact.get("evidence_refs", []) if r in entries]
        claim = str(fact.get("claim", "")).strip()
        if not claim or not refs or any(r not in primary_refs for r in refs):
            continue
        candidates.append({
            "claim": claim, "evidence_refs": list(dict.fromkeys(refs)),
            "media_refs": [r for r in refs if r in media_refs],
            "phase": fact.get("phase", "unknown"),
            "confidence": fact.get("confidence", "low"), "visual_purpose": "",
        })

    # If research supplied no primary-only claim, the existence and path of a
    # cited primary source remain safe, modest fallback facts; do not promote a
    # mixed embedded claim to a primary fact.
    if not candidates:
        candidates = [{
            "claim": f"Primary source: {entry.get('relative_path', entry['ref'])}",
            "evidence_refs": [entry["ref"]],
            "media_refs": [entry["ref"]] if entry["ref"] in media_refs else [],
            "phase": "unknown", "confidence": "high", "visual_purpose": "",
        } for entry in inventory["evidence"] if entry["ref"] in primary_refs]

    candidates = _dedupe_capsules(candidates)
    if len(candidates) <= 4:
        return candidates

    def spread(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        return [items[i * (len(items) - 1) // (count - 1)] for i in range(count)]

    # Diversify by source file first; spread across the input when many files
    # exist, so later top-level sources are also represented.
    by_path: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        ref = candidate["evidence_refs"][0]
        by_path.setdefault(str(entries[ref].get("relative_path", ref)), candidate)
    unique = list(by_path.values())
    selected = spread(unique, 4) if len(unique) >= 4 else unique + spread(candidates, 4)
    for candidate in candidates:
        if len(_dedupe_capsules(selected)) >= 4:
            break
        selected.append(candidate)
    return _dedupe_capsules(selected)[:4]


def _scope_capsules(
    capsules: list[dict[str, Any]], anchors: list[dict[str, Any]],
    inventory: dict[str, Any], title_hint: str, instructions: str,
) -> list[dict[str, Any]]:
    """Apply the existing research fact/ref quota at each planner level."""
    candidates = _dedupe_capsules(capsules + anchors)
    if not anchors:
        return candidates
    bounded, _ = _consolidate(candidates, [], inventory, title_hint, instructions)
    return [{**capsule, "media_refs": [r for r in capsule["media_refs"]
                                       if r in capsule["evidence_refs"]]}
            for capsule in bounded]


def _evidence_scope(index: list[dict[str, Any]]) -> dict[str, list[str]]:
    scope = {"primary": [], "embedded_reference": [], "generated_artifact": []}
    for item in index:
        role = item.get("evidence_role") or "primary"
        scope[role if role in scope else "primary"].append(item["ref"])
    return scope


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


class _EpisodeValidationExhausted(ValueError):
    """Full episode output stayed schema-invalid after bounded retries."""


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
    raise _EpisodeValidationExhausted(f"planner output invalid after {attempts} attempt(s): {last_error}")


_MAX_STORYBOARD_SCENES = 32
_SCENES_PER_PART = 2
_OUTLINE_SYSTEM = (
    "\n\nMultipart storyboard outline: return one compact JSON object containing "
    "version, title, slug, summary, optional presentation, and an ordered "
    "scene_intents list. Choose the episode length editorially (1 to 32 scenes). "
    "Each intent needs type, purpose (at most 160 characters), and at most six "
    "evidence_refs from the supplied planner evidence. Do not write scene narration yet. "
    "If you plan an OUTRO, supply presentation.outro with grounded headline and "
    "links; if no links are supported, finish with SUMMARY instead. Keep any "
    "requested final sentence for narration in the last scene."
)
_SCENES_SYSTEM = (
    "\n\nMultipart storyboard scenes: return only a JSON object with a scenes list, "
    "one full scene object per requested intent, in the exact requested order "
    "and with its assigned id and type. Cite only the supplied part evidence. "
    "Preserve structured scene fields including media.start_seconds, captions, "
    "diagram nodes and annotations when appropriate. Keep narration concise. "
    "Only if this part contains the episode's last scene, satisfy any requested "
    "final sentence in that scene's narration."
)


def _outline_payload(ask: dict[str, Any]) -> dict[str, Any]:
    return {
        "storyboard_mode": "outline",
        "project_title_hint": ask["project_title_hint"],
        "optional_instructions": ask["optional_instructions"],
        "allowed_scene_types": ask["allowed_scene_types"],
        "research": ask["research"],
        "media_inventory": ask["media_inventory"],
        "evidence_index": ask["evidence_index"],
        "required_output": {
            "version": 1, "title": "English title", "slug": "short-slug",
            "summary": "English summary",
            "presentation": "optional episode presentation; OUTRO requires presentation.outro",
            "scene_intents": [{
                "type": "one allowed scene type", "purpose": "brief editorial aim",
                "evidence_refs": ["a supplied evidence ref"],
            }],
        },
    }


def _normalize_outline(value: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") not in (1, "1"):
        raise StructuredOutputError("Storyboard outline must have version 1")
    outline = {"version": 1}
    for key, limit in (("title", 240), ("slug", 100), ("summary", 1600)):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip() or len(item) > limit:
            raise StructuredOutputError(f"Storyboard outline requires a concise {key}")
        outline[key] = item.strip()
    raw_intents = value.get("scene_intents")
    if not isinstance(raw_intents, list) or not 1 <= len(raw_intents) <= _MAX_STORYBOARD_SCENES:
        raise StructuredOutputError(
            f"Storyboard outline requires 1–{_MAX_STORYBOARD_SCENES} scene intents"
        )
    intents = []
    for index, raw in enumerate(raw_intents, 1):
        if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or raw["type"] not in SCENE_TYPES:
            raise StructuredOutputError(f"Storyboard intent {index} has an invalid scene type")
        purpose, refs = raw.get("purpose"), raw.get("evidence_refs", [])
        if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 160:
            raise StructuredOutputError(f"Storyboard intent {index} needs a short purpose")
        if not isinstance(refs, list) or len(refs) > 6 or any(
            not isinstance(ref, str) or ref not in allowed for ref in refs
        ):
            raise StructuredOutputError(f"Storyboard intent {index} cites evidence outside planner scope")
        refs = list(dict.fromkeys(refs))
        if raw["type"] not in {"SECTION_TITLE", "OUTRO"} and not refs:
            raise StructuredOutputError(f"Storyboard intent {index} needs evidence refs")
        intents.append({"id": f"s{index:03d}", "type": raw["type"],
                        "purpose": purpose.strip(), "evidence_refs": refs})
    outline["scene_intents"] = intents
    if "presentation" in value:
        outline["presentation"] = value["presentation"]
    try:
        validate_presentation(outline.get("presentation", {}),
                              any(intent["type"] == "OUTRO" for intent in intents))
        titles = outline.get("presentation", {}).get("scene_titles", {})
        if any(scene_id not in {intent["id"] for intent in intents} for scene_id in titles):
            raise ValueError("presentation.scene_titles references an unplanned scene")
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"Storyboard outline presentation invalid: {exc}") from exc
    return outline


def _scene_part_payload(
    ask: dict[str, Any], outline: dict[str, Any], intents: list[dict[str, Any]],
    part_number: int, part_count: int, scope_id: str,
) -> dict[str, Any]:
    scoped = {item["ref"] for item in ask["evidence_index"]}
    requested = {ref for intent in intents for ref in intent["evidence_refs"]}
    facts = [dict(fact) for fact in ask["research"].get("facts", [])
             if any(ref in requested for ref in fact.get("evidence_refs", []))]
    for fact in facts:
        fact["evidence_refs"] = [ref for ref in fact.get("evidence_refs", []) if ref in scoped]
    # A supporting fact may need two citations. Include its whole cited scope,
    # while never expanding beyond the planner context used by the fast path.
    supplied = requested | {ref for fact in facts for ref in fact["evidence_refs"]}
    media = [item for item in ask["media_inventory"] if item["ref"] in supplied]
    assets = [asset for asset in ask["research"].get("assets", [])
              if asset.get("evidence_ref") in supplied]
    index = [item for item in ask["evidence_index"] if item["ref"] in supplied]
    all_intents = outline["scene_intents"]
    first = int(intents[0]["id"][1:]) - 1
    last = int(intents[-1]["id"][1:])
    neighbors = {}
    for key, offset in (("previous_intent", first - 1), ("next_intent", last)):
        if 0 <= offset < len(all_intents):
            neighbor = all_intents[offset]
            neighbors[key] = {name: neighbor[name] for name in ("id", "type", "purpose")}
    metadata = {key: outline[key] for key in ("version", "title", "slug", "summary", "presentation")
                if key in outline}
    return {
        "storyboard_mode": "scenes", "part_number": part_number, "part_count": part_count,
        "scope_sha256": scope_id,
        "project_title_hint": ask["project_title_hint"],
        "optional_instructions": ask["optional_instructions"],
        "episode_metadata": metadata,
        "total_scenes": len(all_intents),
        "contains_final_scene": intents[-1]["id"] == all_intents[-1]["id"],
        "scene_intents": intents,
        **neighbors,
        "research": {"version": ask["research"].get("version", 1),
                     "facts": facts, "assets": assets},
        "media_inventory": media, "evidence_index": index,
        "optional_structured_scene_fields": ask["optional_structured_scene_fields"],
        "required_output": {"scenes": [
            {"id": intent["id"], "type": intent["type"], "title": "on-screen title",
             "narration": "concise evidence-grounded narration",
             "evidence_refs": intent["evidence_refs"]}
            for intent in intents
        ]},
    }


def _normalize_scene_part(
    value: dict[str, Any], intents: list[dict[str, Any]],
    allowed: set[str], outline: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"scenes"}:
        raise StructuredOutputError("Storyboard part must contain only a scenes list")
    scenes = value["scenes"]
    if not isinstance(scenes, list) or len(scenes) != len(intents):
        raise StructuredOutputError(f"Storyboard part requires exactly {len(intents)} scenes")
    for scene, intent in zip(scenes, intents):
        if not isinstance(scene, dict) or scene.get("id") != intent["id"] or scene.get("type") != intent["type"]:
            raise StructuredOutputError(f"Storyboard part scene order/id/type differs from {intent['id']}")
        refs = scene.get("evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in allowed for ref in refs):
            raise StructuredOutputError(f"{intent['id']}: evidence refs outside this part's planner scope: {refs}")
    partial = {"version": 1, "title": outline["title"], "scenes": scenes}
    if "presentation" in outline:
        partial["presentation"] = outline["presentation"]
    try:
        partial = _repair_episode_shape(partial, allowed)
        validate_episode(partial, allowed)
        validate_presentation(partial.get("presentation", {}),
                              any(scene["type"] == "OUTRO" for scene in partial["scenes"]))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise StructuredOutputError(f"Storyboard part invalid: {exc}") from exc
    return {"scenes": partial["scenes"]}


def _clean_storyboard_parts(part_dir: Path, used: set[Path]) -> None:
    if not part_dir.exists():
        return
    for stale in part_dir.glob("part-*.json"):
        if stale not in used:
            stale.unlink()


def _multipart_episode(
    provider: LLMProvider, system: str, ask: dict[str, Any],
    project_dir: Path, context_size: int, output_reserve_tokens: int,
    safety_tokens: int, max_retries: int, progress: Progress | None,
    scope_id: str,
) -> dict[str, Any]:
    part_dir = project_dir / "manifests" / "storyboard-parts"
    allowed = {item["ref"] for item in ask["evidence_index"]}
    outline_system = system + _OUTLINE_SYSTEM
    outline_payload = _outline_payload(ask)
    if not fits_context(outline_system, json.dumps(outline_payload, ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens):
        raise ValueError("Storyboard outline input exceeds the configured context budget")
    with step(progress, "Planning storyboard parts"):
        outline = checkpointed_complete_json(
            provider, outline_system, outline_payload, part_dir / "outline.json",
            lambda value: _normalize_outline(value, allowed), max_retries=max_retries,
        )

    scene_system = system + _SCENES_SYSTEM
    intents = outline["scene_intents"]
    groups = []
    for offset in range(0, len(intents), _SCENES_PER_PART):
        candidate = intents[offset:offset + _SCENES_PER_PART]
        preview = _scene_part_payload(ask, outline, candidate,
                                      _MAX_STORYBOARD_SCENES, _MAX_STORYBOARD_SCENES, scope_id)
        if fits_context(scene_system, json.dumps(preview, ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens):
            groups.append(candidate)
            continue
        for intent in candidate:
            single = _scene_part_payload(ask, outline, [intent],
                                         _MAX_STORYBOARD_SCENES, _MAX_STORYBOARD_SCENES, scope_id)
            if not fits_context(scene_system, json.dumps(single, ensure_ascii=False),
                                context_size, output_reserve_tokens, safety_tokens):
                raise ValueError(f"Storyboard scene {intent['id']} exceeds the configured input context budget")
            groups.append([intent])

    scenes = []
    used: set[Path] = set()
    for number, group in enumerate(groups, 1):
        checkpoint = part_dir / f"part-{number:03d}.json"

        def payload_for(items: list[dict[str, Any]]) -> dict[str, Any]:
            return _scene_part_payload(ask, outline, items, number, len(groups), scope_id)

        def normalize_for(value: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
            part_allowed = {entry["ref"] for entry in payload_for(items)["evidence_index"]}
            return _normalize_scene_part(value, items, part_allowed, outline)

        results = checkpointed_split_json(
            provider, scene_system, group, payload_for, checkpoint, normalize_for,
            max_retries=max_retries, progress=progress,
            label=f"Storyboard part {number}/{len(groups)}", used=used,
        )
        for result in results:
            scenes.extend(result["scenes"])

    if [scene["id"] for scene in scenes] != [intent["id"] for intent in intents]:
        raise ValueError("Storyboard parts are missing or out of order")
    episode = {key: outline[key] for key in ("version", "title", "slug", "summary", "presentation")
               if key in outline}
    episode["scenes"] = scenes
    episode = _repair_episode_shape(episode, allowed)
    validate_episode(episode, allowed)
    validate_presentation(episode.get("presentation", {}),
                          any(scene["type"] == "OUTRO" for scene in scenes))
    _clean_storyboard_parts(part_dir, used)
    return episode


def _generate_episode(
    provider: LLMProvider, system: str, ask: dict[str, Any], project_dir: Path,
    context_size: int, output_reserve_tokens: int, safety_tokens: int,
    max_retries: int, progress: Progress | None,
) -> dict[str, Any]:
    # The manifest's final evidence index, not every ref in the repository,
    # bounds both the full-response fast path and all fallback scene requests.
    allowed = {entry["ref"] for entry in ask["evidence_index"]}
    scope_id = hashlib.sha256(json.dumps({
        "system": system, "ask": ask, "output_reserve_tokens": output_reserve_tokens,
        "provider_type": type(provider).__name__, "model": getattr(provider, "model", None),
    }, ensure_ascii=False).encode("utf-8")).hexdigest()
    part_dir = project_dir / "manifests" / "storyboard-parts"
    marker = part_dir / "recovery.json"
    recovering = False
    if marker.exists():
        try:
            cached = json_load(marker)
            recovering = isinstance(cached, dict) and cached.get("input_sha256") == scope_id \
                and cached.get("mode") == "multipart"
        except (OSError, ValueError, TypeError):
            pass
    if not recovering:
        try:
            with step(progress, "Generating storyboard"):
                episode = _complete_episode(provider, system, ask, allowed, max_retries)
        except (StructuredOutputError, json.JSONDecodeError, _EpisodeValidationExhausted) as exc:
            json_dump(marker, {"version": 1, "input_sha256": scope_id,
                               "mode": "multipart", "reason": str(exc)})
            if progress is not None:
                progress.note("Storyboard output incomplete; continuing in bounded parts")
        else:
            _clean_storyboard_parts(part_dir, set())
            for stale in (part_dir / "outline.json", marker):
                if stale.exists():
                    stale.unlink()
            return episode
    elif progress is not None:
        progress.note("Resuming storyboard in bounded parts")
    return _multipart_episode(
        provider, system, ask, project_dir, context_size, output_reserve_tokens,
        safety_tokens, max_retries, progress, scope_id,
    )


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
            "evidence_scope": _evidence_scope(evidence_index),
        })
        ep = _generate_episode(
            provider, system, direct_ask, project_dir, context_size,
            output_reserve_tokens, safety_tokens, max_retries, progress,
        )
        json_dump(project_dir / "episode.json", ep)
        return ep

    compact_system = (project_dir.parents[1] / "prompts" / "planner_compact.txt").read_text()
    records = _planner_records(research, media, evidence_index)
    media_ref_set = {m["ref"] for m in media}
    roles = {e["ref"]: e.get("evidence_role") or "primary" for e in inventory["evidence"]}
    anchors = (
        [] if _reference_focus(title_hint, instructions, inventory)
        else _primary_anchors(research, inventory, media_ref_set)
    )
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
            def normalize_part(value: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
                provided_refs, provided_media = _part_refs(items)
                return _normalize_capsules(value, valid & provided_refs, media_ref_set & provided_media)

            results = checkpointed_split_json(
                provider, compact_system, batch,
                lambda items: make_compact_payload(level, part, items),
                checkpoint, normalize_part,
                max_retries=max_retries, progress=progress,
                label=f"Planning evidence — level {level}, part {part}/{len(chunks)}",
            )
            for result in results:
                capsules.extend(result["capsules"])

        capsules = _scope_capsules(capsules, anchors, inventory, title_hint, instructions)
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
                "evidence_scope": _evidence_scope(compact_index),
            })
            ep = _generate_episode(
                provider, system, ask, project_dir, context_size,
                output_reserve_tokens, safety_tokens, max_retries, progress,
            )
            json_dump(project_dir / "episode.json", ep)
            return ep

        records = [{
            "kind": "capsule", "capsule": capsule,
            "evidence_roles": {ref: roles[ref] for ref in capsule["evidence_refs"]},
        } for capsule in capsules]

    raise RuntimeError(
        "Planner payload still exceeds the configured context budget after "
        f"{max_reduce_levels} compaction level(s)"
    )
