from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .chunking import checkpointed_complete_json, checkpointed_split_json, fits_context, split_for_context
from .progress import Progress, step
from .providers import LLMProvider, StructuredOutputError
from .research import _consolidate, _fact_scope, _reference_focus
from .schema import DIAGRAM_TYPES, EVIDENCE_TYPES, SCENE_TYPES, validate_episode, validate_presentation
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


def _authoritative_resource_urls(project_dir: Path) -> list[str]:
    """Use only HTTP(S) URLs recorded by ingestion, never model prose."""
    sources_dir = project_dir / "sources"
    source_file = sources_dir / "source.json"
    if not source_file.exists():
        return []
    source_data = json_load(source_file)
    records = source_data.get("sources", [source_data])
    urls: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username:
            urls.append(value)

    for record in records:
        if not isinstance(record, dict):
            continue
        add(record.get("source"))
        if record.get("kind") == "github":
            repo_url = record.get("repo_url")
            add(repo_url)
            if isinstance(repo_url, str) and repo_url.endswith(".git"):
                parsed = urlsplit(repo_url)
                add(urlunsplit(parsed._replace(path=parsed.path[:-4])))
        if record.get("kind") == "website":
            namespace = record.get("namespace")
            base = sources_dir / namespace if isinstance(namespace, str) and re.fullmatch(r"source-\d+", namespace) else sources_dir
            manifest = base / "website-manifest.json"
            if manifest.exists():
                for page in json_load(manifest).get("pages", []):
                    if isinstance(page, dict) and "html" in page and "text" in page:
                        add(page.get("url"))
    return list(dict.fromkeys(urls))


_FINAL_SUFFIX = re.compile(
    r'\bend\s+with\s*:\s*(?:“(?P<curly>[^”\n]+)”|"(?P<plain>[^"\n]+)")',
    re.I,
)


def _final_narration_suffix(instructions: str) -> str | None:
    matches = list(_FINAL_SUFFIX.finditer(instructions))
    if len(matches) != 1 or re.search(
        r"(?:\bnot|\bnever|\bdon't)\s*$", instructions[max(0, matches[0].start()-24):matches[0].start()], re.I,
    ):
        return None
    return (matches[0].group("curly") or matches[0].group("plain")).strip()


def _identified_research(research: dict[str, Any], evidence_index: list[dict[str, Any]]) -> dict[str, Any]:
    roles = {e["ref"]: e.get("evidence_role") or "primary" for e in evidence_index}
    return {**research, "facts": [
        {**fact, "fact_id": f"F{index:04d}",
         "subject_scope": _fact_scope(fact.get("evidence_refs", []), roles)}
        for index, fact in enumerate(research.get("facts", []), 1)
    ]}


def _make_ask(
    research: dict[str, Any],
    media: list[dict[str, Any]],
    evidence_index: list[dict[str, Any]],
    title_hint: str,
    instructions: str,
    resource_urls: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "allowed_scene_types": sorted(SCENE_TYPES),
        "research": _identified_research(research, evidence_index),
        "media_inventory": media,
        "evidence_index": evidence_index,
        "authoritative_resource_urls": resource_urls or [],
        "required_narration_suffix": _final_narration_suffix(instructions),
        "fact_selection_requirement": (
            "Every factual scene must select up to six fact_ids from research.facts. Base its actual "
            "claims only on those selected facts; respect each fact's derived subject_scope: "
            "supporting_only facts describe examples, and only primary-backed facts can "
            "define the main project's properties; cite only their evidence_refs (plus a "
            "selected research asset for an evidence scene). A setup, maintenance, revision "
            "or optional fact alone cannot describe the mandatory normal workflow. Do not "
            "expand a profile's documented responsibilities. Quote exact commands, config "
            "assignments, file paths, APIs or code only when the selected facts support "
            "those exact literals. Evidence refs alone do not prove narration."
        ),
        "output_contract": {
            "version": 1,
            "title": "English title",
            "slug": "short-slug",
            "summary": "English summary",
            "presentation": {"outro": {
                "headline": ["1–3 evidence-supported lines"],
                "links": [{"label": "supported resource", "url": ["supported URL"]}],
            }},
            "scenes": [{
                "id": "s001",
                "type": "HERO",
                "title": "English on-screen title",
                "narration": "English narration",
                "fact_ids": ["F0001"],
                "evidence_refs": ["E0001"],
                "asset_ref": "E0001",
                "annotations": [],
                "pad_after_seconds": 0.5,
                "diagram": {},
                "notes": "",
            }],
        },
        "presentation_requirement": (
            "For an OUTRO, include episode.presentation.outro with 1–3 grounded "
            "headline lines and 1–3 links whose URLs are exactly in "
            "authoritative_resource_urls. An OUTRO must be the single last scene. "
            "If no supported links exist, finish with SUMMARY and omit presentation.outro."
        ),
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
            "subject_scope": _fact_scope(refs, {ref: evidence_by_ref[ref].get("evidence_role") or "primary"
                                                for ref in refs}),
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


_PRIMARY_ANCHOR_LIMIT = 10
_PRIMARY_ANCHOR_BYTE_BUDGET = 2600
_PRIMARY_ANCHOR_REF_BUDGET = 20
_TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "their", "this", "to",
    "what", "when", "where", "which", "why", "with", "your", "explain",
    "describe", "show", "project", "subject", "episode", "video",
}


def _topic_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[^\W_]+", text.casefold())
            if (len(word) > 2 or word.isdigit()) and word not in _TOPIC_STOPWORDS}


def _primary_anchors(
    research: dict[str, Any], inventory: dict[str, Any], media_refs: set[str],
    title_hint: str = "", instructions: str = "",
) -> list[dict[str, Any]]:
    """Carry a bounded set of distinct, subject-relevant original primary facts."""
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
    subject = _topic_words(title_hint)
    subject_key = "".join(re.findall(r"[^\W_]+", title_hint.casefold()))
    requested = _topic_words(instructions) - subject
    profiles = []
    for index, candidate in enumerate(candidates):
        claim = candidate["claim"]
        path = str(entries[candidate["evidence_refs"][0]].get("relative_path", ""))
        path_parts = Path(path).parts
        words = _topic_words(claim)
        claim_key = "".join(re.findall(r"[^\W_]+", claim.casefold()))
        title_score = 5 if len(subject_key) >= 4 and subject_key in claim_key else 1.5 * len(subject & words)
        confidence_score = {"high": 2, "medium": 1}.get(candidate["confidence"], 0)
        phase_score = 2 if candidate["phase"] == "final" else 0
        # Shallow original documents often state what the project does before
        # implementation files name it repeatedly. This is a modest source
        # signal, never a fixed filename or a replacement for claim relevance.
        overview_score = max(0, 2 - (len(path_parts) - 1)) * 2.5
        if Path(path).suffix.casefold() in {".md", ".rst", ".txt", ".adoc"}:
            overview_score += 1.5
        base = (title_score + 0.8 * min(4, len(requested & words)) +
                confidence_score + phase_score + overview_score)
        profiles.append((index, candidate, path, path_parts[0] if path_parts else "", words - subject, base))

    selected: list[tuple[int, dict[str, Any], str, str, set[str], float]] = []
    remaining = profiles[:]
    used_bytes = 0
    used_refs: set[str] = set()
    while remaining and len(selected) < _PRIMARY_ANCHOR_LIMIT:
        ranked = []
        for profile in remaining:
            index, candidate, path, root, words, base = profile
            refs = set(candidate["evidence_refs"])
            if (used_bytes + len(candidate["claim"].encode("utf-8")) > _PRIMARY_ANCHOR_BYTE_BUDGET or
                    len(used_refs | refs) > _PRIMARY_ANCHOR_REF_BUDGET):
                continue
            similarities = [len(words & prior[4]) / max(1, len(words | prior[4]))
                            for prior in selected]
            similarity = max(similarities, default=0)
            # Preserve different measured values even when the wording is
            # similar; a repeated paraphrase alone adds little information.
            if any(sim >= 0.7 and {w for w in words if any(ch.isdigit() for ch in w)} ==
                   {w for w in prior[4] if any(ch.isdigit() for ch in w)}
                   for sim, prior in zip(similarities, selected)):
                continue
            root_bonus = 2 if root and all(root != prior[3] for prior in selected) else 0
            path_bonus = 0.5 if path and all(path != prior[2] for prior in selected) else 0
            ranked.append((base + root_bonus + path_bonus - 7 * similarity, -index, profile))
        if not ranked:
            break
        best = max(ranked)
        profile = best[2]
        selected.append(profile)
        used_bytes += len(profile[1]["claim"].encode("utf-8"))
        used_refs.update(profile[1]["evidence_refs"])
        remaining.remove(profile)
    if selected:
        return [profile[1] for profile in selected]
    # A huge research claim can exceed the anchor budget. Keep at least one
    # modest, provable primary-source pointer so role scoping stays active.
    for entry in inventory["evidence"]:
        if entry["ref"] in primary_refs:
            return [{"claim": f"Primary source: {entry.get('relative_path', entry['ref'])}",
                     "evidence_refs": [entry["ref"]], "media_refs": [],
                     "phase": "unknown", "confidence": "high", "visual_purpose": ""}]
    return []


def _scope_capsules(
    capsules: list[dict[str, Any]], anchors: list[dict[str, Any]],
    inventory: dict[str, Any], title_hint: str, instructions: str,
) -> list[dict[str, Any]]:
    """Apply the existing research fact/ref quota at each planner level."""
    # Put original, source-grounded primary coverage first in the next level
    # and final storyboard request, even if local compaction omitted it.
    candidates = _dedupe_capsules(anchors + capsules)
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


def _validate_scene_facts(scene: dict[str, Any], ask: dict[str, Any],
                          fixed_ids: list[str] | None = None) -> None:
    """Bind cited refs to selected claims and an explicitly selected asset."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    refs = scene.get("evidence_refs", [])
    ids = scene.get("fact_ids", [])
    needs_facts = scene.get("type") not in {"SECTION_TITLE", "OUTRO"} or bool(refs)
    if not isinstance(ids, list) or len(ids) > 6 or (needs_facts and not ids) or any(
        not isinstance(fact_id, str) or fact_id not in facts for fact_id in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError(f"{scene.get('id', 'Scene')}: select valid fact_ids from supplied planner facts")
    if fixed_ids is not None and ids != fixed_ids:
        raise ValueError(f"{scene.get('id', 'Scene')}: scene fact_ids differ from fixed outline selection")
    supported = {ref for fact_id in ids for ref in facts[fact_id]["evidence_refs"]}
    asset_ref = scene.get("asset_ref")
    if scene.get("type") in EVIDENCE_TYPES and asset_ref in {
        asset.get("evidence_ref") for asset in ask["research"].get("assets", [])
    }:
        supported.add(asset_ref)
    if not isinstance(refs, list) or any(ref not in supported for ref in refs):
        raise ValueError(f"{scene.get('id', 'Scene')}: evidence_refs must come from selected fact_ids or selected asset_ref")


def _validate_resource_links(presentation: Any, ask: dict[str, Any]) -> None:
    outro = presentation.get("outro") if isinstance(presentation, dict) else None
    if not isinstance(outro, dict):
        return
    allowed = set(ask["authoritative_resource_urls"])
    for link in outro.get("links", []):
        lines = link["url"] if isinstance(link["url"], list) else [link["url"]]
        if not all(isinstance(line, str) for line in lines) or not (
            all(line.strip() in allowed for line in lines) or
            "".join(line.strip() for line in lines) in allowed
        ):
            raise ValueError(f"OUTRO link URL is not an authoritative source URL: {lines}")


def _validate_final_narration(episode: dict[str, Any], ask: dict[str, Any]) -> None:
    suffix = ask["required_narration_suffix"]
    if suffix and not episode["scenes"][-1]["narration"].endswith(suffix):
        raise ValueError(f"Last scene narration must end exactly with: {suffix}")


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
            validate_episode(episode, valid_refs, require_integrated_presentation=True)
            for scene in episode["scenes"]:
                _validate_scene_facts(scene, ask)
            _validate_resource_links(episode.get("presentation", {}), ask)
            _validate_final_narration(episode, ask)
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
    "Each factual intent must select fact_ids from the supplied research facts; "
    "its purpose and evidence_refs must follow only those selected claims. "
    "Normal workflow scenes must select normal workflow facts, not just setup "
    "or optional maintenance facts. Each intent needs type, purpose (at most "
    "160 characters), and at most six evidence_refs. For each evidence scene "
    "type, also choose one fixed asset_ref from that intent's evidence_refs; "
    "other scene types do not need an asset_ref. Do not write scene narration yet. "
    "If you plan an OUTRO, it must be the single final scene, with grounded "
    "presentation.outro links selected only from authoritative_resource_urls. "
    "Otherwise finish with SUMMARY and omit presentation.outro. Reserve the "
    "required_narration_suffix for the last scene's narration, not its headline."
)
_SCENES_SYSTEM = (
    "\n\nMultipart storyboard scenes: return only a JSON object with a scenes list, "
    "one full scene object per requested intent, in the exact requested order "
    "and with its assigned id and type. Follow each scene's type-specific "
    "required_output: preserve its fixed asset_ref for evidence scenes and "
    "supply 2–8 labeled, evidence-grounded diagram nodes for diagram scenes. "
    "Use only the selected fact claims for each scene; never add factual "
    "workflow steps or responsibilities absent from those claims. Do not "
    "invent exact commands, config assignments, file paths, APIs or code. "
    "Cite only the supplied part evidence. "
    "Preserve structured scene fields including media.start_seconds, captions, "
    "diagram nodes and annotations when appropriate. Keep narration concise. "
    "If this part contains the last scene, its narration must end exactly "
    "with required_narration_suffix when provided."
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
        "authoritative_resource_urls": ask["authoritative_resource_urls"],
        "required_narration_suffix": ask["required_narration_suffix"],
        "fact_selection_requirement": ask["fact_selection_requirement"],
        "scene_type_requirements": {
            "asset_ref_required_types": sorted(EVIDENCE_TYPES),
            "asset_ref": "For these types, choose one evidence_ref from the same intent as the fixed visual asset; omit for other types.",
        },
        "required_output": {
            "version": 1, "title": "English title", "slug": "short-slug",
            "summary": "English summary",
            "presentation": ask["output_contract"]["presentation"],
            "presentation_requirement": ask["presentation_requirement"],
            "scene_intents": [{
                "type": "one allowed scene type", "purpose": "brief editorial aim",
                "fact_ids": ["F0001"],
                "evidence_refs": ["a supplied evidence ref"],
                "asset_ref": "one of this intent's evidence_refs if type requires an asset_ref; omit otherwise",
            }],
        },
    }


def _final_requests_fit(
    system: str, ask: dict[str, Any], context_size: int,
    output_reserve_tokens: int, safety_tokens: int,
) -> bool:
    """Leave enough input room for recovery even if the full output overflows."""
    if not fits_context(system, json.dumps(ask, ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens):
        return False
    return fits_context(system + _OUTLINE_SYSTEM,
                        json.dumps(_outline_payload(ask), ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens)


def _normalize_outline(value: dict[str, Any], allowed: set[str], ask: dict[str, Any]) -> dict[str, Any]:
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
        intent = {"id": f"s{index:03d}", "type": raw["type"],
                  "purpose": purpose.strip(), "fact_ids": raw.get("fact_ids", []),
                  "evidence_refs": refs}
        if raw["type"] in EVIDENCE_TYPES:
            asset_ref = raw.get("asset_ref")
            if not isinstance(asset_ref, str) or asset_ref not in refs or asset_ref not in allowed:
                raise StructuredOutputError(
                    f"Storyboard intent {index} asset_ref must be one of its scoped evidence_refs"
                )
            intent["asset_ref"] = asset_ref
        try:
            _validate_scene_facts(intent, ask)
        except ValueError as exc:
            raise StructuredOutputError(f"Storyboard intent {index} invalid: {exc}") from exc
        intents.append(intent)
    outline["scene_intents"] = intents
    if "presentation" in value:
        outline["presentation"] = value["presentation"]
    try:
        validate_presentation(outline.get("presentation", {}),
                              any(intent["type"] == "OUTRO" for intent in intents))
        outro_indices = [i for i, intent in enumerate(intents) if intent["type"] == "OUTRO"]
        if len(outro_indices) > 1 or (outro_indices and outro_indices[0] != len(intents)-1):
            raise ValueError("OUTRO must be the single final scene")
        if "outro" in outline.get("presentation", {}) and not outro_indices:
            raise ValueError("presentation.outro requires a final OUTRO scene")
        _validate_resource_links(outline.get("presentation", {}), ask)
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
    selected_ids = {fact_id for intent in intents for fact_id in intent["fact_ids"]}
    facts = [fact for fact in ask["research"].get("facts", [])
             if fact["fact_id"] in selected_ids]
    requested = {ref for intent in intents for ref in intent["evidence_refs"]}
    supplied = (requested | {ref for fact in facts for ref in fact["evidence_refs"]}) & scoped
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

    def required_scene(intent: dict[str, Any]) -> dict[str, Any]:
        scene = {"id": intent["id"], "type": intent["type"], "title": "on-screen title",
                 "narration": "concise evidence-grounded narration",
                 "fact_ids": intent["fact_ids"],
                 "evidence_refs": intent["evidence_refs"]}
        if intent["type"] in EVIDENCE_TYPES:
            scene["asset_ref"] = intent["asset_ref"]
        if intent["type"] in DIAGRAM_TYPES:
            scene["diagram"] = {"nodes": [
                "first labeled relationship or step grounded in supplied evidence",
                "second labeled relationship or step grounded in supplied evidence",
            ]}
        return scene

    return {
        "storyboard_mode": "scenes", "part_number": part_number, "part_count": part_count,
        "scope_sha256": scope_id,
        "project_title_hint": ask["project_title_hint"],
        "optional_instructions": ask["optional_instructions"],
        "episode_metadata": metadata,
        "total_scenes": len(all_intents),
        "contains_final_scene": intents[-1]["id"] == all_intents[-1]["id"],
        "required_narration_suffix": (
            ask["required_narration_suffix"]
            if intents[-1]["id"] == all_intents[-1]["id"] else None
        ),
        "fact_selection_requirement": ask["fact_selection_requirement"],
        "scene_intents": intents,
        **neighbors,
        "research": {"version": ask["research"].get("version", 1),
                     "facts": facts, "assets": assets},
        "media_inventory": media, "evidence_index": index,
        "optional_structured_scene_fields": ask["optional_structured_scene_fields"],
        "optional_scene_fields": {
            "annotations": "optional evidence-grounded labels",
            "notes": "optional production note",
            "pad_after_seconds": "optional non-negative pause",
        },
        "scene_type_requirements": {
            "fixed_asset_ref": "Evidence scenes must return the exact asset_ref selected in the outline and include it in evidence_refs.",
            "diagram_nodes": "Diagram scenes require 2–8 explicit labeled nodes or steps grounded in the supplied part evidence.",
        },
        "required_output": {"scenes": [required_scene(intent) for intent in intents]},
    }


def _normalize_scene_part(
    value: dict[str, Any], intents: list[dict[str, Any]],
    allowed: set[str], outline: dict[str, Any], ask: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"scenes"}:
        raise StructuredOutputError("Storyboard part must contain only a scenes list")
    scenes = value["scenes"]
    if not isinstance(scenes, list) or len(scenes) != len(intents):
        raise StructuredOutputError(f"Storyboard part requires exactly {len(intents)} scenes")
    normalized_scenes = []
    for scene, intent in zip(scenes, intents):
        if not isinstance(scene, dict) or scene.get("id") != intent["id"] or scene.get("type") != intent["type"]:
            raise StructuredOutputError(f"Storyboard part scene order/id/type differs from {intent['id']}")
        refs = scene.get("evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in allowed for ref in refs):
            raise StructuredOutputError(f"{intent['id']}: evidence refs outside this part's planner scope: {refs}")
        if intent["type"] in EVIDENCE_TYPES:
            fixed = intent["asset_ref"]
            if fixed not in allowed:
                raise StructuredOutputError(f"{intent['id']}: fixed asset_ref outside this part's planner scope")
            returned = scene.get("asset_ref")
            if returned not in (None, ""):
                if not isinstance(returned, str) or returned not in allowed:
                    raise StructuredOutputError(f"{intent['id']}: asset_ref outside this part's planner scope")
                if returned != fixed:
                    raise StructuredOutputError(f"{intent['id']}: asset_ref differs from fixed outline choice {fixed}")
            if fixed not in refs:
                raise StructuredOutputError(f"{intent['id']}: asset_ref must appear in scene evidence_refs")
            if returned in (None, ""):
                scene = {**scene, "asset_ref": fixed}
        if intent["type"] in DIAGRAM_TYPES:
            diagram = scene.get("diagram")
            nodes = (diagram.get("nodes") or diagram.get("steps")) if isinstance(diagram, dict) else None
            if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8:
                raise StructuredOutputError(f"{intent['id']}: diagram nodes/steps require 2–8 labeled entries")
        try:
            _validate_scene_facts(scene, ask, intent["fact_ids"])
        except ValueError as exc:
            raise StructuredOutputError(f"Storyboard part invalid: {exc}") from exc
        normalized_scenes.append(scene)
    partial = {"version": 1, "title": outline["title"], "scenes": normalized_scenes}
    if "presentation" in outline:
        partial["presentation"] = (
            outline["presentation"] if any(intent["type"] == "OUTRO" for intent in intents)
            else {key: value for key, value in outline["presentation"].items() if key != "outro"}
        )
    try:
        partial = _repair_episode_shape(partial, allowed)
        validate_episode(partial, allowed, require_integrated_presentation=True)
        if intents[-1]["id"] == outline["scene_intents"][-1]["id"]:
            _validate_final_narration(partial, ask)
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
            lambda value: _normalize_outline(value, allowed, ask), max_retries=max_retries,
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
            return _normalize_scene_part(value, items, part_allowed, outline, ask)

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
    validate_episode(episode, allowed, require_integrated_presentation=True)
    _validate_resource_links(episode.get("presentation", {}), ask)
    _validate_final_narration(episode, ask)
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
    resource_urls = _authoritative_resource_urls(project_dir)

    direct_ask = _make_ask(research, media, evidence_index, title_hint, instructions, resource_urls)
    if _final_requests_fit(system, direct_ask, context_size,
                           output_reserve_tokens, safety_tokens):
        json_dump(project_dir / "manifests" / "planner-evidence.json", {
            "version": 1,
            "strategy": "direct",
            "research": direct_ask["research"],
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
        else _primary_anchors(research, inventory, media_ref_set, title_hint, instructions)
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
        ask = _make_ask(compact_research, compact_media, compact_index,
                        title_hint, instructions, resource_urls)
        if _final_requests_fit(system, ask, context_size,
                               output_reserve_tokens, safety_tokens):
            json_dump(project_dir / "manifests" / "planner-evidence.json", {
                "version": 1,
                "strategy": "map-reduce",
                "levels": level,
                "research": ask["research"],
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
