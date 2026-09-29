from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .chunking import checkpointed_complete_json, checkpointed_split_json, fits_context, split_for_context
from .editorial import (EDITORIAL_CONTRACT, SPECIALIZED_TYPES, SceneTypeUnsuitable,
                        deterministic_field_guard, prune_redundant_content_scenes,
                        selected_claims, structured_fields, validate_novelty,
                        validate_scene_type)
from .grounding import (GROUNDING_CONTRACT, GroundingError,
                        _editorial_entailment_candidate, deterministic_decision,
                        verify_claims)
from .progress import Progress, step
from .providers import LLMProvider, StructuredOutputError
from .research import (_consolidate, _fact_scope, _reference_focus,
                       _requested_concepts, _requested_fact_groups,
                       _verified_code_paraphrase)
from .schema import (DIAGRAM_TYPES, EVIDENCE_TYPES, SCENE_CONTRACTS, SCENE_TYPES,
                     validate_episode, validate_presentation)
from .util import json_dump, json_load


def _media_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {k: e.get(k) for k in ("ref", "relative_path", "kind", "mime", "media", "ai_media", "evidence_role")}
        for e in inventory["evidence"]
        if e["kind"] == "media"
    ]


def _compact_media_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep inspected visual cues without expanding every reduce-level ask."""
    ai = item.get("ai_media")
    if not isinstance(ai, dict):
        return item
    compact = {key: value[:240] for key in ("category", "caption", "visible_text")
               if isinstance(value := ai.get(key), str) and value.strip()}
    if isinstance(ai.get("visible_text"), list):
        compact["visible_text"] = [str(value)[:120] for value in ai["visible_text"][:4]]
    if ai.get("evidence_value") in {"high", "medium", "low"}:
        compact["evidence_value"] = ai["evidence_value"]
    return {**item, "ai_media": compact}


_RENDERABLE_VISUAL_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif",
                           ".mp4", ".mov", ".mkv", ".webm"}
_PLANNER_MEDIA_CONTRACT = "planner-authentic-media-v1"
_PLANNER_MEDIA_LIMIT = 5
_PLANNER_SOURCE_CONTEXT_CONTRACT = "planner-source-context-v1"
_PLANNER_SOURCE_CONTEXT_LIMIT = 4
_PLANNER_SOURCE_CONTEXT_TOTAL_CHARS = 4200
_PLANNER_SOURCE_CONTEXT_PASSAGE_CHARS = 1400


def _visual_asset_refs(evidence_index: list[dict[str, Any]]) -> list[str]:
    """Only inventory media that the current renderer can display is visual."""
    return list(dict.fromkeys(
        entry["ref"] for entry in evidence_index
        if entry.get("kind") == "media" and (
            not entry.get("relative_path") or
            Path(entry["relative_path"]).suffix.lower() in _RENDERABLE_VISUAL_EXTS
        )
    ))


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


def _supported_research(research: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Planner facts must still point to the research stage's exact quotations."""
    entries = {entry["ref"]: entry for entry in inventory["evidence"]}
    kept = []
    for fact in research.get("facts", []):
        if not isinstance(fact, dict):
            continue
        refs = fact.get("evidence_refs")
        spans = fact.get("support")
        if (not isinstance(refs, list) or not refs or not all(
            isinstance(ref, str) and ref in entries for ref in refs
        ) or not isinstance(spans, list) or not spans):
            continue
        if any(not isinstance(span, dict) or span.get("evidence_ref") not in refs or
               not isinstance(span.get("text"), str) or not span["text"].strip() or
               (isinstance(entries[span["evidence_ref"]].get("excerpt"), str) and
                span["text"] not in entries[span["evidence_ref"]]["excerpt"])
               for span in spans):
            continue
        if any(not any(span["evidence_ref"] == ref for span in spans) for ref in refs):
            continue
        kept.append(fact)
    if not kept:
        raise RuntimeError("Planner has no research facts with exact supporting spans")
    return {**research, "facts": kept}


_GERMAN_FUNCTION_WORDS = {
    "dass", "dafür", "dieses", "einem", "einen", "einer", "erwarteten",
    "nicht", "schaltet", "statt", "unterstützt", "wird", "werden",
}


def _obviously_german_claim(claim: str) -> bool:
    """Avoid verbatim German narration when English research is available."""
    if not re.search(r"[äöüßÄÖÜ]", claim):
        return False
    words = set(re.findall(r"[a-zäöüß]+", claim.casefold()))
    return len(words & _GERMAN_FUNCTION_WORDS) >= 3


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
         "source_fact_id": fact.get("source_fact_id", f"R{index:04d}"),
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
    identified = _identified_research(research, evidence_index)
    visual_refs = [ref for ref in _visual_asset_refs(evidence_index)
                   if ref in _research_refs(identified)]
    roles = {entry["ref"]: entry.get("evidence_role") or "primary" for entry in evidence_index}
    paths = {entry["ref"]: str(entry.get("relative_path") or "") for entry in evidence_index}
    scoped_refs = set(roles)
    requested_fact_groups = _requested_fact_groups(
        [fact for fact in identified["facts"]
         if all(ref in scoped_refs for ref in fact.get("evidence_refs", [])) and
         all(isinstance(span, dict) and
             span.get("evidence_ref") in fact["evidence_refs"] and
             isinstance(span.get("text"), str) and span["text"].strip()
             for span in fact.get("support", []))],
        instructions, title_hint, roles, paths,
    )
    requested_topic_specs = {
        key: spec for key, spec in _requested_concepts(instructions, title_hint).items()
        if key in requested_fact_groups
    }
    requested = _explicit_topic_words(instructions) - _topic_words(title_hint)
    ranked = sorted((
        (len(requested & _topic_words(fact["claim"])), index, fact["fact_id"])
        for index, fact in enumerate(identified["facts"])
        if fact.get("phase") == "final" and fact.get("confidence") == "high" and
        _fact_scope(fact["evidence_refs"], roles) == "main_subject"
    ), reverse=True)
    priorities = [fact_id for score, _index, fact_id in ranked if score >= 1][:12]
    first = identified["facts"][0] if identified["facts"] else None
    example_asset = next((ref for ref in visual_refs if first and (
        ref in first["evidence_refs"] or any(
            asset.get("evidence_ref") == ref for asset in identified.get("assets", [])
        )
    )), None)
    example_scene = {
        "id": "s001", "type": "HERO" if example_asset else "SUMMARY",
        "title": "English on-screen title", "narration": "English narration",
        "fact_ids": [first["fact_id"]] if first else [],
        "evidence_refs": list(dict.fromkeys((first["evidence_refs"][:1] if first else []) +
                                            ([example_asset] if example_asset else []))),
    }
    if example_asset:
        example_scene["asset_ref"] = example_asset
    return {
        "grounding_contract": GROUNDING_CONTRACT,
        "editorial_contract": EDITORIAL_CONTRACT,
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "allowed_scene_types": sorted(
            SCENE_TYPES if visual_refs else SCENE_TYPES - EVIDENCE_TYPES
        ),
        "visual_asset_refs": visual_refs,
        "visual_asset_requirement": (
            "Evidence scenes need a visual_asset_refs member selected by a fact or "
            "research.assets. If none, use a non-evidence type; documents are not visuals."
        ),
        "research": identified,
        "requested_topic_fact_ids": requested_fact_groups,
        "requested_topic_specs": requested_topic_specs,
        "requested_topic_requirement": (
            "For each listed topic, select one listed fact ID in a content scene; "
            "SECTION_TITLE/OUTRO do not count."
        ),
        "priority_fact_ids": priorities,
        "priority_requirement": (
            "Ranking hints only; requested_topic_fact_ids govern coverage."
        ),
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
            "or optional fact alone cannot describe the mandatory normal workflow. Facts with "
            "operation_guard are conditional on that exact source branch; preserve that condition "
            "and never present them as unconditional behavior. Do not "
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
            "scenes": [example_scene],
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

    for number, fact in enumerate(research.get("facts", []), 1):
        refs = [r for r in fact.get("evidence_refs", []) if r in evidence_by_ref]
        fact_media = [media_by_ref[r] for r in refs if r in media_by_ref]
        referenced_media.update(m["ref"] for m in fact_media)
        records.append({
            "kind": "fact",
            "source_fact_id": fact.get("source_fact_id", f"R{number:04d}"),
            "claim": fact.get("claim", ""),
            "support": fact.get("support", []),
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
        "grounding_contract": GROUNDING_CONTRACT,
        "planner_media_contract": _PLANNER_MEDIA_CONTRACT,
        "level": level,
        "part": part,
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "records": records,
        "required_output": {
            "capsules": [{
                "source_fact_id": "R0001: select exactly one supplied source fact; do not rewrite its claim",
                "claim": "optional label; the original source fact claim is authoritative",
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
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_facts = []
    for record in records or []:
        source = record.get("capsule", {}) if record.get("kind") == "capsule" else record
        if source.get("kind", record.get("kind")) == "asset":
            continue
        if source.get("source_fact_id") and source.get("claim") and source.get("evidence_refs"):
            source_facts.append(source)
    by_id = {item["source_fact_id"]: item for item in source_facts}
    capsules = []
    returned = result.get("capsules", []) if isinstance(result, dict) else []
    for capsule in returned if isinstance(returned, list) else []:
        if not isinstance(capsule, dict):
            continue
        raw_media = capsule.get("media_refs", [])
        raw_refs = capsule.get("evidence_refs", [])
        mrefs = [r for r in raw_media if isinstance(r, str) and r in media_refs] \
            if isinstance(raw_media, list) else []
        refs = [r for r in raw_refs if isinstance(r, str) and r in valid_refs] \
            if isinstance(raw_refs, list) else []
        refs = list(dict.fromkeys(refs))
        if not refs and not mrefs:
            continue
        source_id = capsule.get("source_fact_id")
        source = by_id.get(source_id) if isinstance(source_id, str) else None
        if source is not None and not set(source["evidence_refs"]) & set(refs):
            source = None
        if source is None:
            # Legacy compaction responses can still select an unambiguous
            # source by ref. The model's own claim is NEVER promoted.
            matches = [item for item in source_facts if set(item["evidence_refs"]) & set(refs)]
            exact = [item for item in matches if item["claim"].casefold() == str(
                capsule.get("claim", "")
            ).strip().casefold()]
            options = exact or matches
            source = options[0] if len(options) == 1 else None
        if source is None:
            if mrefs:
                capsules.append({"kind": "asset", "claim": "", "evidence_refs": list(dict.fromkeys(mrefs)),
                                 "media_refs": list(dict.fromkeys(mrefs)), "visual_purpose": str(
                                     capsule.get("visual_purpose", "")
                                 ).strip()[:160], "phase": "unknown", "confidence": "low"})
            continue
        source_refs = [ref for ref in source["evidence_refs"] if ref in valid_refs]
        if len(source_refs) != len(source["evidence_refs"]):
            continue
        capsules.append({
            "kind": "fact", "claim": source["claim"],
            "source_fact_id": source["source_fact_id"],
            "support": source.get("support", []),
            "evidence_refs": source_refs,
            "media_refs": list(dict.fromkeys(mrefs)),
            "phase": source.get("phase", "unknown"),
            "confidence": source.get("confidence", "low"),
            "visual_purpose": str(capsule.get("visual_purpose", "")).strip()[:160],
        })
    return {"capsules": capsules}


def _dedupe_capsules(capsules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for capsule in capsules:
        marker = (capsule.get("source_fact_id"), capsule["claim"].casefold(),
                  tuple(capsule["evidence_refs"]))
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


def _explicit_topic_words(instructions: str) -> set[str]:
    """Extract user focus, ignoring citation rules and a quoted closing line."""
    subjects = []
    for clause in re.split(r"[.!?;\n]", instructions):
        if re.search(r"\b(?:explain|describe|cover|focus|explore|showcase|"
                     r"distinguish|compare|how|why|what)\b", clause, re.I):
            subjects.append(clause)
    return _topic_words(" ".join(subjects))


def _bounded_source_context_text(text: str) -> str:
    """Bound one exact source block without inventing or rewriting it."""
    text = text.strip()
    if len(text) <= _PLANNER_SOURCE_CONTEXT_PASSAGE_CHARS:
        return text
    limit = _PLANNER_SOURCE_CONTEXT_PASSAGE_CHARS
    candidates = [
        text.rfind("\n", 0, limit),
        text.rfind(". ", 0, limit) + 1,
        text.rfind("; ", 0, limit) + 1,
    ]
    cut = max(candidates)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip()


def _source_context_blocks(excerpt: str) -> list[str]:
    """Return prose-oriented exact blocks; code-heavy blocks stay out of narrative context."""
    blocks = []
    for raw in re.split(r"\n[ \t]*\n+", excerpt):
        block = raw.strip()
        if not block or block.startswith(chr(96) * 3):
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        prose_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", block)
        if len(prose_words) < 4:
            continue
        executable = sum(bool(re.match(
            r"^(?:sudo|git|cd|chmod|python|pip|uv|cargo|make|cmake|sed|cp|mv|rm|"
            r"systemctl|pactl|play|sleep|\./)[ \t]",
            line,
        )) for line in lines)
        if executable * 2 > len(lines):
            continue
        bounded = _bounded_source_context_text(block)
        if bounded:
            blocks.append(bounded)
    return blocks


def _source_context_passages(
    research: dict[str, Any], inventory: dict[str, Any],
    title_hint: str, instructions: str,
    limit: int = _PLANNER_SOURCE_CONTEXT_LIMIT,
) -> list[dict[str, Any]]:
    """Select bounded verbatim primary-source passages for narrative understanding only."""
    requested = _explicit_topic_words(instructions) - _topic_words(title_hint)
    subject = _topic_words(title_hint)
    fact_words: set[str] = set()
    for fact in research.get("facts", []):
        fact_words.update(_topic_words(str(fact.get("claim", ""))))
    fact_refs = _research_refs(research)
    reference_focus = _reference_focus(title_hint, instructions, inventory)
    allowed_roles = {"primary", "embedded_reference"} if reference_focus else {"primary"}

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for entry_index, entry in enumerate(inventory.get("evidence", [])):
        if (entry.get("kind") != "document" or
                (entry.get("evidence_role") or "primary") not in allowed_roles or
                not isinstance(entry.get("excerpt"), str)):
            continue
        excerpt = entry["excerpt"]
        path = str(entry.get("relative_path") or "")
        basename = Path(path).name.casefold()
        depth = len(Path(path).parts)
        for block_index, block in enumerate(_source_context_blocks(excerpt)):
            words = _topic_words(block)
            if not words:
                continue
            request_overlap = len(requested & words)
            subject_overlap = len(subject & words)
            fact_overlap = len(fact_words & words)
            readme_bonus = 10 if re.fullmatch(
                r"readme(?:\.[a-z]{2})?\.(?:md|rst|txt|adoc)", basename, re.I
            ) else 0
            linked_bonus = 5 if entry.get("ref") in fact_refs else 0
            shallow_bonus = max(0, 3 - depth)
            if not (request_overlap or subject_overlap or fact_overlap or readme_bonus):
                continue
            score = (
                10 * min(request_overlap, 8) +
                3 * min(subject_overlap, 5) +
                min(fact_overlap, 10) +
                readme_bonus + linked_bonus + shallow_bonus -
                0.05 * block_index
            )
            candidates.append((score, entry_index, block_index, {
                "evidence_ref": entry["ref"],
                "relative_path": path,
                "text": block,
                "context_role": "narrative_only",
            }))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    used_chars = 0
    for _score, entry_index, block_index, passage in candidates:
        marker = (passage["evidence_ref"], re.sub(r"\s+", " ", passage["text"]).casefold())
        if marker in seen:
            continue
        size = len(passage["text"])
        if used_chars + size > _PLANNER_SOURCE_CONTEXT_TOTAL_CHARS:
            continue
        selected.append((entry_index, block_index, passage))
        seen.add(marker)
        used_chars += size
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (item[0], item[1]))
    return [passage for _entry, _block, passage in selected]


def _planner_visuals(
    research: dict[str, Any], media: list[dict[str, Any]],
    title_hint: str, instructions: str, inventory: dict[str, Any],
    limit: int = _PLANNER_MEDIA_LIMIT, selected_hints: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Choose a bounded set of project visuals from original, scoped evidence."""
    # A title's descriptive subtitle is not a project name. Inspect only the
    # basename, caption and visible text: a website's shared directory name
    # must not make every unrelated gallery image look like project evidence.
    name = re.split(r"[:|\u2014\u2013]", title_hint, maxsplit=1)[0].strip()
    name_words = re.findall(r"[^\W_]+", name.casefold())
    name_key = "".join(name_words)
    stem = (name_key[:-2] if len(name_words) == 1 and name_key.endswith("os")
            and len(name_key) >= 8 else "")
    roles = {entry["ref"]: entry.get("evidence_role") or "primary"
             for entry in inventory["evidence"]}
    fact_links: dict[str, list[str]] = {}
    for number, fact in enumerate(research.get("facts", []), 1):
        refs = fact.get("evidence_refs", [])
        if _fact_scope(refs, roles) != "main_subject":
            continue
        for ref in refs:
            fact_links.setdefault(ref, []).append(
                fact.get("source_fact_id", f"R{number:04d}")
            )

    focus = _reference_focus(title_hint, instructions, inventory)
    ranked = []
    for index, item in enumerate(media):
        ref = item["ref"]
        if (ref not in roles or (roles[ref] != "primary" and not focus) or
                ref not in _visual_asset_refs([item])):
            continue
        ai = item.get("ai_media") or {}
        ai = ai if isinstance(ai, dict) else {}
        caption = ai.get("caption") if isinstance(ai.get("caption"), str) else ""
        visible = ai.get("visible_text") or ""
        if isinstance(visible, list):
            visible = " ".join(str(value) for value in visible)
        visible = visible if isinstance(visible, str) else ""
        basename = Path(item.get("relative_path") or "").stem
        description = " ".join((caption, visible, basename))
        words = re.findall(r"[^\W_]+", description.casefold())
        joined = "".join(words)
        software = bool(re.search(
            r"\b(?:screenshot|screen|desktop|terminal|console|interface|"
            r"editor|file\s*manager|file\s*browser|software|application|"
            r"preview|output|logo|ui)\b",
            re.sub(r"[_-]", " ", " ".join((str(ai.get("category") or ""), description))).casefold(),
        ))
        identity = bool(name_key and (name_key in joined or (
            stem and any(word.startswith(stem) for word in words)
        )))
        hint = (selected_hints or {}).get(ref, "")
        hint_words = re.findall(r"[^\W_]+", hint.casefold())
        hinted_identity = bool(software and name_key and (
            name_key in "".join(hint_words) or
            (stem and any(word.startswith(stem) for word in hint_words))
        ))
        linked = ref in fact_links and bool(
            software or caption or visible or identity
        )
        # Project identity by itself does not qualify an unrelated photo.
        # A lower-level model hint can identify a real software screenshot,
        # but cannot turn an inspected gallery photo into software evidence.
        if not linked and not (identity and software) and not hinted_identity:
            continue
        tier = (3 if (identity or hinted_identity) and software
                else 2 if linked and software else 1)
        evidence_value = {"high": 2, "medium": 1}.get(ai.get("evidence_value"), 0)
        ranked.append((-tier, -evidence_value, index, ref,
                       hint if hinted_identity and not identity else caption or hint))

    capsules = []
    for _tier, _value, _index, ref, caption in sorted(ranked)[:limit]:
        capsules.append({
            "kind": "asset", "claim": "", "evidence_refs": [ref],
            "media_refs": [ref], "visual_purpose": (caption or f"Authentic visual for {name}")[:160],
            "phase": "unknown", "confidence": "low",
        })
    return capsules


def _carry_planner_visuals(
    capsules: list[dict[str, Any]], selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reattach the same authorized media after every model reduction level."""
    return [{**capsule, "media_refs": []} for capsule in capsules
            if capsule.get("kind") != "asset"] + selected


class _RequestedCoverageError(StructuredOutputError):
    """A valid scene selection omitted supported explicit requested topics."""

    def __init__(self, missing: dict[str, list[str]]):
        self.missing = missing
        detail = "; ".join(f"{topic} -> choose one of {', '.join(ids)}"
                           for topic, ids in missing.items())
        super().__init__(f"Missing requested storyboard coverage: {detail}")


def _missing_story_topics(scenes: list[dict[str, Any]],
                          ask: dict[str, Any]) -> dict[str, list[str]]:
    selected = {fact_id for scene in scenes
                if scene["type"] not in {"SECTION_TITLE", "OUTRO"}
                for fact_id in scene.get("fact_ids", [])}
    return {topic: ids for topic, ids in ask.get("requested_topic_fact_ids", {}).items()
            if ids and selected.isdisjoint(ids)}


def _validate_focus_coverage(scenes: list[dict[str, Any]], ask: dict[str, Any]) -> None:
    missing = _missing_story_topics(scenes, ask)
    if missing:
        raise _RequestedCoverageError(missing)


def _prune_editorial_repetition(
    scenes: list[dict[str, Any]], presentation: Any, ask: dict[str, Any],
    *, enforce_coverage: bool = True,
) -> tuple[list[dict[str, Any]], Any]:
    """Drop exhausted content while preserving coverage and presentation references."""
    kept = prune_redundant_content_scenes(scenes)
    removed = {scene["id"] for scene in scenes} - {scene["id"] for scene in kept}
    if not removed:
        return kept, presentation
    missing_before = _missing_story_topics(scenes, ask)
    missing_after = _missing_story_topics(kept, ask)
    newly_missing = {topic: ids for topic, ids in missing_after.items()
                     if topic not in missing_before}
    if newly_missing and enforce_coverage:
        raise _RequestedCoverageError(newly_missing)
    if isinstance(presentation, dict) and isinstance(presentation.get("scene_titles"), dict):
        presentation = {**presentation, "scene_titles": {
            scene_id: title for scene_id, title in presentation["scene_titles"].items()
            if scene_id not in removed
        }}
    return kept, presentation


def _canonical_story_composition(
    scenes: list[dict[str, Any]], presentation: Any, ask: dict[str, Any],
    allowed: set[str], *, full: bool = False, enforce_coverage: bool = True,
    provider: LLMProvider | None = None,
    project_dir: Path | None = None, progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Turn selected, authorized facts into distinct content and a grounded ending."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    preferred = {raw_id: natural_id for raw_id, raw in facts.items()
                 for natural_id, natural in facts.items()
                 if _verified_code_paraphrase(natural, raw)}
    retained: list[dict[str, Any]] = []
    exhausted: set[str] = set()
    intro_facts: set[str] = set()
    prior_ids: list[str] = []
    used_assets: set[str] = set()
    pruned_intro_section = False
    for original in scenes:
        scene = original
        if scene["type"] not in {"CODE", "SECTION_TITLE", "OUTRO"}:
            selected_ids = list(dict.fromkeys(
                preferred.get(fact_id, fact_id) for fact_id in scene["fact_ids"]
            ))
            if selected_ids != scene["fact_ids"]:
                scene = {**scene, "fact_ids": selected_ids}
                scene["evidence_refs"] = _canonical_evidence_refs(scene, ask, allowed)
                _validate_scene_facts(scene, ask)
                if full:
                    kind = (scene["type"] if scene["type"] in EVIDENCE_TYPES | {"HERO"}
                            else "SUMMARY")
                    scene = _canonical_scene_fields({**scene, "type": kind,
                                                     "title": "Documented observation",
                                                     "annotations": None, "notes": None})
                    if kind == "SUMMARY":
                        scene.pop("asset_ref", None)
                    scene["narration"] = _grounded_narration_fallback(scene, ask, is_final=False)
                else:
                    scene["purpose"] = _safe_intent_purpose(scene, facts)
        if scene["type"] == "SECTION_TITLE" and scene.get("fact_ids"):
            scene = {**scene, "type": "SUMMARY"}
            scene["evidence_refs"] = _canonical_evidence_refs(scene, ask, allowed)
            _validate_scene_facts(scene, ask)
            if not full and scene["purpose"] == _EDITORIAL_SCENE_PURPOSES["SECTION_TITLE"]:
                scene["purpose"] = _EDITORIAL_SCENE_PURPOSES["SUMMARY"]
            if full:
                scene = _canonical_scene_fields(scene)
        if scene["type"] == "SECTION_TITLE":
            if scene.get("purpose") == _EDITORIAL_SCENE_PURPOSES["SECTION_TITLE"]:
                pruned_intro_section = bool(retained and retained[-1]["type"] == "HERO")
                continue
            if not retained or retained[-1]["type"] != "SECTION_TITLE":
                retained.append(scene)
            pruned_intro_section = False
            continue
        if (pruned_intro_section and scene["type"] == "SUMMARY" and retained and
                retained[-1]["type"] == "HERO" and scene["fact_ids"] and
                set(scene["fact_ids"]) == set(retained[-1]["fact_ids"])):
            # A removed, empty chapter must not turn an exact hero reprise
            # into a second content scene.
            continue
        pruned_intro_section = False
        if scene["type"] in {"HERO", "OUTRO"}:
            retained.append(scene)
            if scene["type"] == "HERO":
                intro_facts.update(scene["fact_ids"])
                if isinstance(scene.get("asset_ref"), str):
                    used_assets.add(scene["asset_ref"])
            continue

        asset = scene.get("asset_ref") if scene["type"] in EVIDENCE_TYPES else None
        if asset in used_assets:
            # A second use of the same image is no longer distinct evidence.
            # Keep the new facts as a summary without implying another visual.
            scene = {**scene, "type": "SUMMARY"}
            scene.pop("asset_ref", None)
            if full:
                scene = _canonical_scene_fields(scene)
            scene["evidence_refs"] = _canonical_evidence_refs(scene, ask, allowed)
            _validate_scene_facts(scene, ask)
            asset = None
        selected = list(scene["fact_ids"])
        if (selected and set(selected) <= intro_facts and
                not any(prior["type"] not in {"HERO", "OUTRO", "SECTION_TITLE"}
                        for prior in retained)):
            continue
        if scene["type"] != "CODE":
            selected = [fact_id for fact_id in selected if fact_id not in exhausted]
            if selected and prior_ids and not (asset and asset not in used_assets):
                comparisons = []
                entailed: set[str] = set()
                for fact_id in selected:
                    claim = facts[fact_id]["claim"]
                    for earlier_id in prior_ids:
                        earlier = facts[earlier_id]["claim"]
                        comparison_id = f"{scene['id']}:{fact_id}:{earlier_id}"
                        if deterministic_decision(claim, [earlier]) == "accept":
                            entailed.add(comparison_id)
                        elif provider is not None and _editorial_entailment_candidate(
                            claim, [earlier],
                        ):
                            comparisons.append({"id": comparison_id, "claim": claim,
                                                "support": [earlier]})
                if comparisons:
                    if project_dir is None:
                        with tempfile.TemporaryDirectory() as tmp:
                            entailed.update(verify_claims(
                                provider, comparisons, Path(tmp), "editorial-novelty-v3",
                                progress, editorial_omission=True,
                            ))
                    else:
                        entailed.update(verify_claims(
                            provider, comparisons, project_dir, "editorial-novelty-v3",
                            progress, editorial_omission=True,
                        ))
                missing_topics = set(_missing_story_topics(retained, ask))
                selected = [fact_id for fact_id in selected if (
                    not any(f"{scene['id']}:{fact_id}:{earlier_id}" in entailed
                            for earlier_id in prior_ids) or
                    any(fact_id in topic_ids and topic in missing_topics
                        for topic, topic_ids in ask.get("requested_topic_fact_ids", {}).items())
                )]
        if not selected:
            continue
        if selected != scene["fact_ids"]:
            scene = {**scene, "fact_ids": selected}
            scene["evidence_refs"] = _canonical_evidence_refs(scene, ask, allowed)
            if full:
                # Prior diagram labels and narration may refer to the removed facts.
                scene = _canonical_scene_fields({**scene, "type": "SUMMARY",
                                                 "title": "Documented observation",
                                                 "annotations": None, "notes": None})
                scene["narration"] = _grounded_narration_fallback(scene, ask, is_final=False)
            else:
                scene["purpose"] = _safe_intent_purpose(scene, facts)
                try:
                    validate_scene_type(scene, facts, ask["evidence_index"])
                except SceneTypeUnsuitable:
                    scene = {**scene, "type": "SUMMARY",
                             "purpose": _safe_intent_purpose({**scene, "type": "SUMMARY"}, facts)}
            _validate_scene_facts(scene, ask)
        retained.append(scene)
        for fact_id in selected:
            if fact_id not in exhausted:
                prior_ids.append(fact_id)
                exhausted.add(fact_id)
        if asset:
            used_assets.add(asset)

    if retained and retained[-1]["type"] == "OUTRO" and not retained[-1]["fact_ids"]:
        selected_content = list(dict.fromkeys(
            fact_id for scene in retained if scene["type"] not in {"SECTION_TITLE", "HERO", "OUTRO"}
            for fact_id in scene["fact_ids"]
        ))
        overview = ask.get("requested_topic_fact_ids", {}).get("overview", [])
        candidates = [fact_id for fact_id in overview if fact_id in selected_content]
        candidates += [fact_id for fact_id in selected_content if fact_id not in candidates]
        for fact_id in candidates:
            draft = {**retained[-1], "fact_ids": [fact_id],
                     "evidence_refs": [ref for ref in facts[fact_id]["evidence_refs"]
                                       if ref in allowed][:1]}
            draft["evidence_refs"] = _canonical_evidence_refs(draft, ask, allowed)
            _validate_scene_facts(draft, ask)
            try:
                narration = _grounded_narration_fallback(draft, ask, is_final=True)
            except GroundingError:
                continue
            if full:
                draft["narration"] = narration
            retained[-1] = draft
            break
        if selected_content and not retained[-1]["fact_ids"]:
            raise StructuredOutputError("OUTRO has no supported selected fact that fits its narration")

    kept, presentation = _prune_editorial_repetition(
        retained, presentation, ask, enforce_coverage=enforce_coverage,
    )
    removed = {scene["id"] for scene in scenes} - {scene["id"] for scene in kept}
    missing_before = _missing_story_topics(scenes, ask)
    newly_missing = {topic: ids for topic, ids in _missing_story_topics(kept, ask).items()
                     if topic not in missing_before}
    if newly_missing and enforce_coverage:
        raise _RequestedCoverageError(newly_missing)
    if removed and isinstance(presentation, dict) and isinstance(presentation.get("scene_titles"), dict):
        presentation = {**presentation, "scene_titles": {
            scene_id: title for scene_id, title in presentation["scene_titles"].items()
            if scene_id not in removed
        }}
    return kept, presentation


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
    for number, fact in enumerate(research.get("facts", []), 1):
        refs = [r for r in fact.get("evidence_refs", []) if r in entries]
        claim = str(fact.get("claim", "")).strip()
        if not claim or not refs or any(r not in primary_refs for r in refs):
            continue
        candidates.append({
            "kind": "fact", "claim": claim, "evidence_refs": list(dict.fromkeys(refs)),
            "source_fact_id": fact.get("source_fact_id", f"R{number:04d}"),
            "support": fact.get("support", []),
            "media_refs": [r for r in refs if r in media_refs],
            "phase": fact.get("phase", "unknown"),
            "confidence": fact.get("confidence", "low"), "visual_purpose": "",
            "current_overview": fact.get("current_overview") is True,
        })

    if not candidates:
        return []

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
        if (candidate["current_overview"] and candidate["phase"] == "development" and
                re.search(r"\b(?:planned|future|later)\b", instructions, re.I)):
            phase_score += 12
        # Shallow original documents often state what the project does before
        # implementation files name it repeatedly. This is a modest source
        # signal, never a fixed filename or a replacement for claim relevance.
        overview_score = max(0, 2 - (len(path_parts) - 1)) * 2.5
        if Path(path).suffix.casefold() in {".md", ".rst", ".txt", ".adoc"}:
            overview_score += 1.5
        # Explicitly requested concepts outweigh filesystem depth and the
        # number of interesting but tangential implementation facts.
        base = (title_score + 7 * min(5, len(requested & words)) +
                confidence_score + phase_score + overview_score +
                12 * candidate["current_overview"])
        profiles.append((index, candidate, path, path_parts[0] if path_parts else "", words - subject, base))

    selected: list[tuple[int, dict[str, Any], str, str, set[str], float]] = []
    remaining = profiles[:]
    used_bytes = 0
    used_refs: set[str] = set()
    named = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
    while remaining and len(selected) < _PRIMARY_ANCHOR_LIMIT:
        ranked = []
        for profile in remaining:
            index, candidate, path, root, words, base = profile
            refs = set(candidate["evidence_refs"])
            size = len(candidate["claim"].encode("utf-8")) + sum(
                len(item.get("text", "").encode("utf-8")) for item in candidate["support"]
            )
            if (used_bytes + size > _PRIMARY_ANCHOR_BYTE_BUDGET or
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
            current_names = {name.casefold() for prior in selected if prior[1]["current_overview"]
                             for name in named.findall(prior[1]["claim"])}
            linked_names = {name.casefold() for name in named.findall(candidate["claim"])}
            link_bonus = 6 * min(3, len(linked_names & current_names))
            ranked.append((base + root_bonus + path_bonus + link_bonus - 7 * similarity,
                           -index, profile))
        if not ranked:
            break
        best = max(ranked)
        profile = best[2]
        selected.append(profile)
        used_bytes += len(profile[1]["claim"].encode("utf-8")) + sum(
            len(item.get("text", "").encode("utf-8")) for item in profile[1]["support"]
        )
        used_refs.update(profile[1]["evidence_refs"])
        remaining.remove(profile)
    if selected:
        return [profile[1] for profile in selected]
    # Do not synthesize a source-path claim without a supporting quotation.
    return []


def _scope_capsules(
    capsules: list[dict[str, Any]], anchors: list[dict[str, Any]],
    inventory: dict[str, Any], title_hint: str, instructions: str,
) -> list[dict[str, Any]]:
    """Apply the existing research fact/ref quota at each planner level."""
    # Put original, source-grounded primary coverage first in the next level
    # and final storyboard request, even if local compaction omitted it.
    assets = [c for c in capsules if c.get("kind") == "asset"]
    candidates = _dedupe_capsules(anchors + [c for c in capsules if c.get("kind") != "asset"])
    if not anchors:
        return candidates + assets
    bounded, _ = _consolidate(candidates, [], inventory, title_hint, instructions)
    allowed = {ref for c in bounded for ref in c["evidence_refs"]}
    roles = {e["ref"]: e.get("evidence_role") or "primary" for e in inventory["evidence"]}
    return [{**capsule, "media_refs": [r for r in capsule["media_refs"]
                                       if r in capsule["evidence_refs"]]}
            for capsule in bounded] + [asset for asset in assets if any(
                r in allowed or roles.get(r) == "primary" for r in asset["media_refs"]
            )]


def _evidence_scope(index: list[dict[str, Any]]) -> dict[str, list[str]]:
    scope = {"primary": [], "embedded_reference": [], "generated_artifact": []}
    for item in index:
        role = item.get("evidence_role") or "primary"
        scope[role if role in scope else "primary"].append(item["ref"])
    return scope


def _capsules_to_research(capsules: list[dict[str, Any]]) -> dict[str, Any]:
    fact_links: dict[str, list[str]] = {}
    for capsule in capsules:
        if capsule.get("kind") != "asset":
            for ref in capsule["evidence_refs"]:
                fact_links.setdefault(ref, []).append(capsule["source_fact_id"])
    return {
        "version": 1,
        "facts": [{
            "claim": c["claim"],
            "evidence_refs": c["evidence_refs"],
            "source_fact_id": c["source_fact_id"],
            "support": c["support"],
            "phase": c["phase"],
            "confidence": c["confidence"],
        } for c in capsules if c.get("kind") != "asset"],
        "assets": [{
            "evidence_ref": ref,
            "purpose": c["visual_purpose"],
            "authentic_project_media": True,
            **({"source_fact_ids": fact_links[ref]} if ref in fact_links else {}),
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
            scene = _canonical_scene_fields(raw_scene)
            if not scene.get("id"):
                scene["id"] = f"s{index:03d}"
            refs = scene.get("evidence_refs")
            if isinstance(refs, list):
                seen_refs: set[str] = set()
                unique_refs = []
                for ref in refs:
                    if isinstance(ref, str):
                        if ref in seen_refs:
                            continue
                        seen_refs.add(ref)
                    unique_refs.append(ref)
                refs = unique_refs
                scene["evidence_refs"] = refs
            if scene.get("type") in EVIDENCE_TYPES:
                asset_ref = scene.get("asset_ref")
                if refs is None:
                    refs = []
                    scene["evidence_refs"] = refs
                if isinstance(refs, list):
                    if not asset_ref and len(refs) == 1 and isinstance(refs[0], str) and (
                        valid_refs is None or refs[0] in valid_refs
                    ):
                        scene["asset_ref"] = refs[0]
                    elif isinstance(asset_ref, str) and asset_ref and (
                        valid_refs is None or asset_ref in valid_refs
                    ) and asset_ref not in refs:
                        scene["evidence_refs"] = refs + [asset_ref]
            repaired.append(scene)
        ep["scenes"] = repaired
    return ep


def _canonical_scene_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Discard fields with no meaning for this scene type before validation."""
    kind = raw.get("type")
    contract = SCENE_CONTRACTS.get(kind) if isinstance(kind, str) else None
    if contract is None:
        return dict(raw)  # The schema reports the invalid type.
    fields = {"id", "type", "title", "narration", "fact_ids", "evidence_refs"}
    fields.update(contract.canonical_optional_fields)
    if contract.allows_asset_ref:
        fields.add("asset_ref")
    scene = {key: value for key, value in raw.items() if key in fields}
    if (kind == "SECTION_TITLE" and scene.get("fact_ids") == [] and
            isinstance(scene.get("narration"), str) and
            re.fullmatch(r"closing\W*", scene["narration"].strip(), re.I)):
        scene["narration"] = "Next, a closer look."
    for optional in contract.canonical_optional_fields:
        if scene.get(optional) is None:
            scene.pop(optional, None)
    return scene


def _draft_evidence_refs(raw: dict[str, Any]) -> list[str]:
    """Treat a missing/null list as omitted refs, with a bounded raw draft."""
    refs = raw.get("evidence_refs")
    if refs is None:
        return []
    if not isinstance(refs, list) or len(refs) > 64 or any(
        not isinstance(ref, str) for ref in refs
    ):
        raise ValueError("evidence_refs must be a bounded list of strings")
    return refs


def _validate_scene_facts(scene: dict[str, Any], ask: dict[str, Any],
                          fixed_ids: list[str] | None = None) -> None:
    """Bind cited refs to selected claims and an explicitly selected asset."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    refs = scene.get("evidence_refs", [])
    ids = scene.get("fact_ids", [])
    needs_facts = SCENE_CONTRACTS[scene["type"]].requires_facts or bool(refs)
    if not isinstance(ids, list) or len(ids) > 6 or (needs_facts and not ids) or any(
        not isinstance(fact_id, str) or fact_id not in facts for fact_id in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError(f"{scene.get('id', 'Scene')}: select valid fact_ids from supplied planner facts")
    if fixed_ids is not None and ids != fixed_ids:
        raise ValueError(f"{scene.get('id', 'Scene')}: scene fact_ids differ from fixed outline selection")
    supported = {ref for fact_id in ids for ref in facts[fact_id]["evidence_refs"]}
    asset_ref = scene.get("asset_ref")
    if SCENE_CONTRACTS[scene["type"]].allows_asset_ref and asset_ref in {
        asset.get("evidence_ref") for asset in ask["research"].get("assets", [])
    }:
        supported.add(asset_ref)
    if not isinstance(refs, list) or any(ref not in supported for ref in refs):
        raise ValueError(f"{scene.get('id', 'Scene')}: evidence_refs must come from selected fact_ids or selected asset_ref")


_MAX_NARRATION_CHARS = 1200


class _NarrationGroundingRejected(StructuredOutputError):
    """A structurally valid scene has narration that cannot pass grounding."""

    failure_category = "narration grounding"

    def __init__(self, scene_id: str, reason: str):
        self.scene_id = scene_id
        super().__init__(f"{scene_id}: {reason}")


class _NarrationFallbackUnavailable(GroundingError):
    """No complete selected claim can safely satisfy the narration contract."""


def _grounded_narration_fallback(
    scene: dict[str, Any], ask: dict[str, Any], *, is_final: bool,
) -> str:
    """Use complete, selected, supported claims; never derive facts from prose or refs."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    selected_ids = list(dict.fromkeys(scene["fact_ids"]))
    suffix = ask["required_narration_suffix"] if is_final else None
    available = _MAX_NARRATION_CHARS - (len(suffix) + 1 if suffix else 0)
    claims: list[str] = []
    length = 0
    for fact_id in selected_ids:
        fact = facts.get(fact_id)
        if fact is None or not isinstance(fact.get("claim"), str) or not fact["claim"].strip():
            raise _NarrationFallbackUnavailable(f"{scene['id']}: selected grounded fact claim is missing")
        spans = fact.get("support")
        if not isinstance(spans, list) or not spans or any(
            not isinstance(span, dict) or span.get("evidence_ref") not in fact["evidence_refs"] or
            not isinstance(span.get("text"), str) or not span["text"].strip()
            for span in spans
        ):
            raise _NarrationFallbackUnavailable(f"{scene['id']}: selected fact lacks exact evidence support")
        claim = fact["claim"]
        extra = len(claim) + (1 if claims else 0)
        if length + extra <= available:
            claims.append(claim)
            length += extra
    if selected_ids and not claims:
        raise _NarrationFallbackUnavailable(
            f"{scene['id']}: no complete selected grounded fact claim fits "
            f"the {_MAX_NARRATION_CHARS}-character narration bound"
        )
    if claims:
        body = " ".join(claims)
    elif scene["type"] == "SECTION_TITLE" and not scene["evidence_refs"]:
        body = "Next, a closer look."
    elif scene["type"] == "OUTRO" and not scene["evidence_refs"]:
        body = "Closing."
    else:
        raise _NarrationFallbackUnavailable(f"{scene['id']}: no selected facts for factual narration")
    if suffix:
        if suffix in body:
            if body.endswith(suffix) and body.count(suffix) == 1:
                return body
            raise _NarrationFallbackUnavailable(
                f"{scene['id']}: selected claim overlaps the required suffix outside its final position"
            )
        body += " " + suffix
    if len(body) > _MAX_NARRATION_CHARS:
        raise _NarrationFallbackUnavailable(
            f"{scene['id']}: narration fallback exceeds the bounded grounding check"
        )
    return body


def _recognized_grounded_narrations(
    scenes: list[dict[str, Any]], ask: dict[str, Any], final_scene_id: str,
) -> dict[str, str]:
    """Exact selected-claim narrations remain safe when a checkpoint is reused."""
    recognized: dict[str, str] = {}
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    for scene in scenes:
        if scene["fact_ids"]:
            if not any(
                isinstance(facts[fact_id].get("claim"), str) and
                scene["narration"].startswith(facts[fact_id]["claim"])
                for fact_id in scene["fact_ids"]
            ):
                continue
        elif scene["type"] == "SECTION_TITLE":
            if not scene["narration"].startswith("Next, a closer look."):
                continue
        elif not scene["narration"].startswith("Closing."):
            continue
        try:
            safe = _grounded_narration_fallback(
                scene, ask, is_final=scene["id"] == final_scene_id,
            )
        except _NarrationFallbackUnavailable:
            continue
        if scene["narration"] == safe:
            recognized[scene["id"]] = safe
    return recognized


def _validate_narration_grounding(
    provider: LLMProvider, scenes: list[dict[str, Any]], ask: dict[str, Any],
    project_dir: Path | None, progress: Progress | None = None,
    trusted_fallbacks: dict[str, str] | None = None,
    final_scene_id: str | None = None,
) -> None:
    """Check actual narration against just the scene's selected source facts."""
    known = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    future_ids = set(ask.get("requested_topic_fact_ids", {}).get("future-work", []))
    final_scene_id = final_scene_id or scenes[-1]["id"]
    checks = []
    for scene in scenes:
        selected = [known[ref] for ref in scene.get("fact_ids", [])]
        for fact in selected:
            if not fact.get("support") or any(
                span.get("evidence_ref") not in fact["evidence_refs"] or
                not isinstance(span.get("text"), str) or not span["text"].strip()
                for span in fact["support"]
            ):
                raise ValueError(f"{scene['id']}: selected fact lacks exact evidence support")
        narration = scene["narration"].strip()
        if len(narration) > _MAX_NARRATION_CHARS:
            raise _NarrationGroundingRejected(scene["id"], "narration exceeds the bounded grounding check")
        if scene["id"] in (trusted_fallbacks or {}):
            if scene["narration"] != trusted_fallbacks[scene["id"]]:
                raise ValueError(f"{scene['id']}: grounded fallback narration was modified")
            continue
        if (scene["id"] == final_scene_id and ask["required_narration_suffix"] and
                narration.endswith(ask["required_narration_suffix"])):
            narration = narration[:-len(ask["required_narration_suffix"])].strip()
        if not narration:
            continue
        if future_ids & set(scene.get("fact_ids", [])) and not (
            re.search(r"\b(?:planned|future|later|yet|next|roadmap)\b", narration, re.I) and
            any(len(_topic_words(narration) & _topic_words(known[fact_id]["claim"])) >= 2
                for fact_id in future_ids & set(scene["fact_ids"]))
        ):
            raise _NarrationGroundingRejected(
                scene["id"], "narration omits the selected future-work boundary",
            )
        if not selected and re.fullmatch(
            r"(?:closing|thank you(?: for watching)?|thanks(?: for watching)?|the end)\W*",
            narration, re.I,
        ):
            continue
        if not selected and scene["type"] == "SECTION_TITLE" and narration == "Next, a closer look.":
            continue
        claims = [fact["claim"] for fact in selected]
        if narration != " ".join(claims) and not deterministic_field_guard(narration, claims):
            raise _NarrationGroundingRejected(
                scene["id"], "narration introduces an unsupported factual proposition",
            )
        checks.append({"id": scene["id"], "claim": narration, "facts": [{
            "claim": fact["claim"], "support": fact["support"]
        } for fact in selected]})
    if not checks:
        return
    if project_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            verified = verify_claims(provider, checks, Path(tmp), "scene", progress)
    else:
        verified = verify_claims(provider, checks, project_dir, "scene", progress)
    for item in checks:
        if item["id"] not in verified:
            raise _NarrationGroundingRejected(
                item["id"], "narration introduces an unsupported factual proposition",
            )


def _validate_structured_grounding(
    provider: LLMProvider | None, scenes: list[dict[str, Any]], ask: dict[str, Any],
    project_dir: Path | None, progress: Progress | None = None,
) -> None:
    """Reject unsupported structured fields; omit unsupported optional annotations."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    checks = []
    annotation_checks = []
    annotation_candidates = []
    for scene in scenes:
        claims = selected_claims(scene, facts)
        for field, label in structured_fields(scene, facts):
            if not deterministic_field_guard(label, claims):
                raise ValueError(f"{scene['id']}: {field} has unsupported factual content from selected fact_ids")
            if provider is not None:
                checks.append({"id": f"{scene['id']}:{field}", "claim": label,
                               "facts": [{"claim": facts[fact_id]["claim"],
                                          "support": facts[fact_id]["support"]}
                                         for fact_id in scene["fact_ids"]]})
        if "annotations" in scene:
            candidates = []
            raw = scene["annotations"]
            for index, annotation in enumerate(raw if isinstance(raw, list) else []):
                label = annotation.get("text") if isinstance(annotation, dict) else annotation
                if not isinstance(label, str) or not label.strip():
                    continue
                if not deterministic_field_guard(label, claims):
                    continue
                check_id = f"{scene['id']}:annotations[{index}]"
                candidates.append((check_id, annotation))
                annotation_checks.append({"id": check_id, "claim": label,
                                          "facts": [{"claim": facts[fact_id]["claim"],
                                                     "support": facts[fact_id]["support"]}
                                                    for fact_id in scene["fact_ids"]]})
            annotation_candidates.append((scene, candidates))
    if checks:
        if project_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                verified = verify_claims(provider, checks, Path(tmp), "scene-fields", progress)
        else:
            verified = verify_claims(provider, checks, project_dir, "scene-fields", progress)
        for item in checks:
            if item["id"] not in verified:
                raise ValueError(f"{item['id']} has unsupported factual content from selected fact_ids")
    if provider is None:
        # No semantic verifier: retain only quotes accepted deterministically.
        accepted_annotations = {item["id"] for item in annotation_checks
                                if deterministic_decision(item["claim"],
                                    [fact["claim"] for fact in item["facts"]]) == "accept"}
    elif annotation_checks:
        try:
            if project_dir is None:
                with tempfile.TemporaryDirectory() as tmp:
                    accepted_annotations = verify_claims(provider, annotation_checks,
                                                         Path(tmp), "scene-annotations", progress,
                                                         max_retries=0, max_split_depth=0)
            else:
                accepted_annotations = verify_claims(provider, annotation_checks,
                                                     project_dir, "scene-annotations", progress,
                                                     max_retries=0, max_split_depth=0)
        except Exception:
            # An unavailable verifier cannot authorize optional model-authored text.
            accepted_annotations = set()
    else:
        accepted_annotations = set()
    for scene, candidates in annotation_candidates:
        scene["annotations"] = [annotation for check_id, annotation in candidates
                                if check_id in accepted_annotations]


def _recover_narration_after_retries(
    provider: LLMProvider, scenes: list[dict[str, Any]], ask: dict[str, Any],
    project_dir: Path | None, progress: Progress | None,
    first_rejected: str, final_scene_id: str,
) -> list[dict[str, Any]]:
    """Repair rejected full-response narrations, then check untouched natural scenes."""
    canonical = [{**scene} for scene in scenes]
    by_id = {scene["id"]: scene for scene in canonical}
    trusted: dict[str, str] = {}
    rejected = first_rejected
    for _ in canonical:
        if rejected not in by_id or rejected in trusted:
            raise StructuredOutputError("Narration recovery selected an invalid scene")
        scene = by_id[rejected]
        narration = _grounded_narration_fallback(
            scene, ask, is_final=rejected == final_scene_id,
        )
        scene["narration"] = narration
        trusted[rejected] = narration
        try:
            _validate_narration_grounding(provider, canonical, ask, project_dir, progress,
                                          trusted_fallbacks=trusted)
            return canonical
        except _NarrationGroundingRejected as exc:
            rejected = exc.scene_id
    raise StructuredOutputError("Narration recovery could not ground all scenes")


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


def _canonical_full_episode(
    episode: dict[str, Any], ask: dict[str, Any], allowed: set[str],
    *, provider: LLMProvider | None = None, project_dir: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Give full responses the same fact and asset authority as outline parts."""
    scenes = episode.get("scenes")
    if not isinstance(scenes, list):
        return episode  # The schema reports the missing or malformed scenes.
    canonical = []
    for index, raw in enumerate(scenes, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Scene {index} must be an object")
        scene = _canonical_scene_fields(raw)
        kind = scene.get("type")
        if not isinstance(kind, str) or kind not in SCENE_CONTRACTS:
            raise ValueError(f"Scene {index} has an invalid scene type")
        try:
            refs = _draft_evidence_refs(scene)
        except ValueError as exc:
            raise ValueError(f"{scene['id']}: {exc}") from exc
        scene["evidence_refs"] = refs
        if not _valid_outline_fact_ids(scene.get("fact_ids"), scene, ask) and any(
            ref not in allowed for ref in refs
        ):
            raise ValueError(f"{scene['id']}: evidence outside planner scope with invalid fact_ids")
        scene["fact_ids"] = _outline_fact_ids(scene.get("fact_ids"), scene, ask)
        scene["evidence_refs"] = _canonical_evidence_refs(scene, ask, allowed)
        if _placeholder_scene_title(scene.get("title")):
            facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
            scene["title"] = _suggested_scene_title(scene, facts, episode.get("title", ""))
        if len(scene["evidence_refs"]) > 6:
            raise ValueError(f"{scene['id']}: evidence_refs exceeds six selected refs")
        _validate_scene_facts(scene, ask)
        try:
            validate_scene_type(scene, {fact["fact_id"]: fact for fact in ask["research"]["facts"]},
                                ask["evidence_index"])
        except SceneTypeUnsuitable:
            # The selected facts and refs have already passed provenance checks.
            # Retain them and discard structured fields tied to the unsuitable type.
            scene = _canonical_scene_fields({**scene, "type": "SUMMARY"})
        canonical.append(scene)
    canonical, presentation = _canonical_story_composition(
        canonical, episode.get("presentation"), ask, allowed,
        full=True, provider=provider, project_dir=project_dir, progress=progress,
    )
    validate_novelty(canonical)
    result = {**episode, "scenes": canonical}
    if presentation is not None:
        result["presentation"] = presentation
    return result


def _retry_feedback(error: Exception) -> str:
    """Keep retries focused on the contract the previous answer actually broke."""
    if isinstance(error, _RequestedCoverageError):
        return (f"{error}. Add content coverage for these supported topics; preserve "
                "valid scene fact selections and unrelated metadata.")
    message = str(error)[:260]
    if "allowed_fact_ids:" in message:
        return f"{message} Copy only those exact existing IDs; preserve their cited evidence."
    if "no planner-scoped visual" in message:
        return (f"Previous storyboard rejected: {message}. This source has no usable "
                "visual asset for that scene. Keep its selected facts and choose a "
                "compatible non-evidence scene type.")
    if "asset_ref" in message or "visual asset" in message:
        if "allowed asset_ref: " in message:
            options = message.split("allowed asset_ref: ", 1)[1]
            return (f"Allowed asset_ref: {options}. Copy one exactly for this evidence "
                    "scene, or choose a compatible non-evidence type.")
        return (f"Previous storyboard rejected: {message}. For evidence scenes, choose a "
                "renderable visual in visual_asset_refs selected by a fact or research.assets; "
                "otherwise choose a compatible non-evidence type. Preserve fixed outline assets.")
    if "evidence" in message or "planner scope" in message:
        return (f"Previous storyboard rejected: {message}. Select fact_ids first; cite only "
                "their scoped evidence refs and the authorized visual asset, at most six.")
    if "unsupported factual" in message or "grounding" in message:
        return f"Previous storyboard rejected: {message}. Rewrite only unsupported prose using the selected facts."
    if "presentation" in message or "OUTRO" in message or "link" in message:
        return f"Previous storyboard rejected: {message}. Repair the OUTRO order and supplied presentation URLs."
    return f"Previous storyboard rejected: {message}. Repair the stated field and return the required JSON object."


def _complete_episode(
    provider: LLMProvider,
    system: str,
    ask: dict[str, Any],
    valid_refs: set[str],
    max_retries: int,
    *, project_dir: Path | None = None, progress: Progress | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    coverage_candidate: dict[str, Any] | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        coverage_candidate = None
        payload = dict(ask)
        if last_error is not None:
            payload["validation_feedback"] = _retry_feedback(last_error)
        raw = provider.complete_json(system, json.dumps(payload, ensure_ascii=False))
        try:
            episode = _repair_episode_shape(raw, valid_refs)
            episode = _canonical_full_episode(
                episode, ask, valid_refs, provider=provider,
                project_dir=project_dir, progress=progress,
            )
            validate_episode(episode, valid_refs, require_integrated_presentation=True)
            _validate_structured_grounding(provider, episode["scenes"], ask,
                                           project_dir, progress)
            episode = _canonical_full_summary(provider, episode, ask, project_dir, progress)
            _validate_resource_links(episode.get("presentation", {}), ask)
            _validate_final_narration(episode, ask)
            _validate_narration_grounding(
                provider, episode["scenes"], ask, project_dir, progress,
                trusted_fallbacks=_recognized_grounded_narrations(
                    episode["scenes"], ask, episode["scenes"][-1]["id"],
                ),
            )
            coverage_candidate = episode
            _validate_focus_coverage(episode["scenes"], ask)
            return episode
        except _NarrationGroundingRejected as exc:
            if attempt == attempts - 1:
                canonical = _recover_narration_after_retries(
                    provider, episode["scenes"], ask, project_dir, progress,
                    exc.scene_id, episode["scenes"][-1]["id"],
                )
                episode = {**episode, "scenes": canonical}
                validate_episode(episode, valid_refs, require_integrated_presentation=True)
                _validate_final_narration(episode, ask)
                try:
                    _validate_focus_coverage(episode["scenes"], ask)
                except _RequestedCoverageError:
                    return _recover_full_coverage(provider, system, episode, ask, valid_refs,
                                                  max_retries, project_dir, progress)
                return episode
            last_error = exc
        except (ValueError, TypeError) as exc:
            last_error = exc
    assert last_error is not None
    if isinstance(last_error, _RequestedCoverageError) and coverage_candidate is not None:
        return _recover_full_coverage(provider, system, coverage_candidate, ask, valid_refs,
                                      max_retries, project_dir, progress)
    raise _EpisodeValidationExhausted(f"planner output invalid after {attempts} attempt(s): {last_error}")


def _recover_full_coverage(
    provider: LLMProvider, system: str, episode: dict[str, Any], ask: dict[str, Any],
    allowed: set[str], max_retries: int, project_dir: Path | None,
    progress: Progress | None,
) -> dict[str, Any]:
    """Generate missing grounded content naturally, with exact-claim scene fallback."""
    additions = _coverage_additions(episode["scenes"], ask, allowed)
    scenes, presentation = _insert_coverage_scenes(
        episode["scenes"], additions, episode.get("presentation"),
        preserve_existing_ids=True, keep_last=bool(ask["required_narration_suffix"]),
    )
    recovered = {**episode, "scenes": scenes}
    if presentation is not None:
        recovered["presentation"] = presentation
    by_id = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    outline = {**recovered, "scene_intents": [
        {"id": scene["id"], "type": scene["type"],
         "purpose": scene.get("purpose") or _safe_intent_purpose(scene, by_id),
         "fact_ids": scene["fact_ids"], "evidence_refs": scene["evidence_refs"]}
        for scene in scenes
    ]}
    insertion = len(episode["scenes"]) - int(
        episode["scenes"][-1]["type"] == "OUTRO" or bool(ask["required_narration_suffix"])
    )
    for number, intent in enumerate(
        outline["scene_intents"][insertion:insertion + len(additions)], 1,
    ):
        request = _scene_part_payload(ask, outline, [intent], number,
                                      len(additions), "coverage-recovery")
        scoped = {entry["ref"] for entry in request["evidence_index"]}
        template = request["required_output"]["scenes"][0]
        result = None
        for _ in range(max(1, max_retries + 1)):
            try:
                proposed = provider.complete_json(system + _SCENES_SYSTEM,
                                                  json.dumps(request, ensure_ascii=False))
                result = _normalize_scene_part(proposed, [intent], scoped, outline, ask,
                                               provider, project_dir, progress)["scenes"][0]
                break
            except (StructuredOutputError, ValueError, TypeError, KeyError, AttributeError):
                continue
        if result is None:
            template["narration"] = _grounded_narration_fallback(
                intent, ask, is_final=intent["id"] == scenes[-1]["id"],
            )
            result = _normalize_scene_part({"scenes": [template]}, [intent], scoped,
                                           outline, ask)["scenes"][0]
        scenes[next(index for index, scene in enumerate(scenes)
                    if scene["id"] == intent["id"])] = result
    _validate_focus_coverage(scenes, ask)
    recovered["scenes"] = scenes
    validate_episode(recovered, allowed, require_integrated_presentation=True)
    _validate_resource_links(recovered.get("presentation", {}), ask)
    _validate_final_narration(recovered, ask)
    _validate_narration_grounding(
        provider, scenes, ask, project_dir, progress,
        trusted_fallbacks=_recognized_grounded_narrations(scenes, ask, scenes[-1]["id"]),
    )
    if progress is not None:
        progress.note("Storyboard coverage retries exhausted; added grounded summary scenes")
    return recovered


_MAX_STORYBOARD_SCENES = 32
_MAX_OUTLINE_SUMMARY_CHARS = 1600
_MAX_OUTLINE_PURPOSE_CHARS = 160
_SCENES_PER_PART = 2
_OUTLINE_SYSTEM = (
    "\n\nMultipart storyboard outline: return one compact JSON object containing "
    "version, title, slug, summary, optional presentation, and an ordered "
    "scene_intents list. Choose the episode length editorially (1 to 32 scenes). "
    "Each factual intent must select fact_ids from the supplied research facts; "
    "its purpose and evidence_refs must follow only those selected claims. "
    "Every factual proposition in summary must follow supplied planner facts, "
    "and every factual proposition in an intent purpose must follow that "
    "intent's selected fact_ids. Unsupported planning labels are replaced "
    "with exact selected claims or editorial scene labels before scene generation. "
    "Select every fact_id verbatim from allowed_fact_ids; never continue the "
    "numeric sequence or use an ID from a previous request. "
    "Choose evidence_refs from selected fact_ids in fact_evidence_map, plus a "
    "selected asset_ref for an evidence scene. "
    "Normal workflow scenes must select normal workflow facts, not just setup "
    "or optional maintenance facts. Each intent needs type, purpose (at most "
    "160 characters), and at most six evidence_refs. For each evidence scene "
    "type, choose one fixed visual asset_ref from selected fact refs or "
    "research assets within planner scope, and cite it in evidence_refs; "
    "omit asset_ref for other types. Do not write scene narration yet. "
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
    "The outline summary and scene purposes are planning labels, not evidence; "
    "only selected fact claims authorize factual narration. "
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
        "allowed_fact_ids": [fact["fact_id"] for fact in ask["research"]["facts"]],
        "fact_id_requirement": "Every intent fact_id must be copied verbatim from allowed_fact_ids.",
        "fact_evidence_map": {fact["fact_id"]: fact["evidence_refs"]
                              for fact in ask["research"]["facts"]},
        "evidence_ref_requirement": (
            "Use selected fact_ids' refs in fact_evidence_map; an evidence scene "
            "may also cite its fixed asset_ref when it is a selected research asset."
        ),
        "project_title_hint": ask["project_title_hint"],
        "optional_instructions": ask["optional_instructions"],
        "allowed_scene_types": ask["allowed_scene_types"],
        "visual_asset_refs": ask.get("visual_asset_refs", _visual_asset_refs(ask["evidence_index"])),
        "research": ask["research"],
        "media_inventory": ask["media_inventory"],
        "evidence_index": ask["evidence_index"],
        "authoritative_resource_urls": ask["authoritative_resource_urls"],
        "required_narration_suffix": ask["required_narration_suffix"],
        "fact_selection_requirement": ask["fact_selection_requirement"],
        "editorial_contract": ask["editorial_contract"],
        "priority_fact_ids": ask["priority_fact_ids"],
        "priority_requirement": ask["priority_requirement"],
        "requested_topic_fact_ids": ask.get("requested_topic_fact_ids", {}),
        "requested_topic_specs": ask.get("requested_topic_specs", {}),
        "requested_topic_requirement": ask.get("requested_topic_requirement", ""),
        "scene_type_requirements": {
            "asset_ref_required_types": sorted(EVIDENCE_TYPES),
            "asset_ref": "Choose one visual_asset_refs member from selected fact_ids' fact_evidence_map refs or research.assets evidence_ref; cite it in evidence_refs. Omit for other types.",
        },
        "required_output": {
            "version": 1, "title": "English title", "slug": "short-slug",
            "summary": "English summary",
            "presentation": ask["output_contract"]["presentation"],
            "presentation_requirement": ask["presentation_requirement"],
            "scene_intents": [{
                "type": "one allowed scene type", "purpose": "brief editorial aim",
                "fact_ids": ([ask["research"]["facts"][0]["fact_id"]]
                             if ask["research"]["facts"] else []),
                "evidence_refs": ["a supplied evidence ref"],
                **({"asset_ref": "renderable selected fact ref or selected research asset, for evidence types only"}
                   if ask.get("visual_asset_refs", _visual_asset_refs(ask["evidence_index"])) else {}),
            }],
        },
    }


_OUTLINE_RETRY_FEEDBACK_RESERVE = 48


def _brief_outline_feedback(error: Exception) -> str:
    """A bounded corrective hint when the full validation message will not fit."""
    message = str(error)
    match = re.search(r"allowed asset_ref:\s*(E\d+)", message)
    if match:
        return f"Use asset_ref {match.group(1)} or non-evidence type."
    if "allowed asset_ref: none" in message or "no planner-scoped visual" in message:
        return "Use a compatible non-evidence scene type."
    if "fact_ids" in message:
        return "Copy only supplied fact_ids for each intent."
    if isinstance(error, _RequestedCoverageError):
        return "Cover the required topic facts in the outline."
    return "Repair the rejected outline field."


def _final_requests_fit(
    system: str, ask: dict[str, Any], context_size: int,
    output_reserve_tokens: int, safety_tokens: int,
) -> bool:
    """Leave enough input room for recovery even if the full output overflows."""
    if not fits_context(system, json.dumps(ask, ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens):
        return False
    # A retry must fit at least a short, concrete validation hint. Large
    # feedback uses the normal path only when the context has spare room.
    outline = {**_outline_payload(ask),
               "validation_feedback": "x" * _OUTLINE_RETRY_FEEDBACK_RESERVE}
    return fits_context(system + _OUTLINE_SYSTEM,
                        json.dumps(outline, ensure_ascii=False),
                        context_size, output_reserve_tokens, safety_tokens)


def _valid_outline_fact_ids(raw: Any, intent: dict[str, Any], ask: dict[str, Any]) -> bool:
    needs_facts = SCENE_CONTRACTS[intent["type"]].requires_facts or bool(intent["evidence_refs"])
    if raw is None:
        return not needs_facts
    allowed = {fact["fact_id"] for fact in ask["research"]["facts"]}
    return (isinstance(raw, list) and len(raw) <= 6 and (raw or not needs_facts) and
            all(isinstance(fact_id, str) and fact_id in allowed for fact_id in raw) and
            len(set(raw)) == len(raw))


def _outline_fact_ids(raw: Any, intent: dict[str, Any], ask: dict[str, Any]) -> list[str]:
    """Repair only a unique ref-to-existing-fact mapping, never editorial meaning."""
    facts = ask["research"]["facts"]
    allowed = [fact["fact_id"] for fact in facts]
    allowed_set = set(allowed)
    ids = raw if isinstance(raw, list) else []
    if _valid_outline_fact_ids(raw, intent, ask):
        return ids

    # Repeating an already valid fact ID does not introduce a new selection or
    # require inference. Normalize only exact duplicates before attempting any
    # evidence-ref-based repair; unknown IDs and oversized outputs still fail.
    if (isinstance(raw, list) and len(raw) <= 6 and raw and
            all(isinstance(fact_id, str) and fact_id in allowed_set for fact_id in raw)):
        deduped = list(dict.fromkeys(raw))
        if len(deduped) < len(raw):
            return deduped

    invalid = ([str(fact_id)[:60] for fact_id in ids[:6] if not isinstance(fact_id, str)
                or fact_id not in allowed_set] if isinstance(raw, list) else
               [str(raw)[:60] if raw is not None else "<missing>"])
    label = ", ".join(invalid) if invalid else "<missing or duplicate>"
    if isinstance(raw, list) and len(raw) > 6:
        label += f" ({len(raw)} returned IDs; maximum 6)"
    error = StructuredOutputError(
        f"{intent['id']}: invalid fact_ids [{label}]; allowed_fact_ids: [{', '.join(allowed)}]"
    )
    if (not isinstance(raw, (list, type(None))) or len(ids) > 6 or
            len(set(map(str, ids))) != len(ids) or not intent["evidence_refs"]):
        raise error

    asset = intent.get("asset_ref")
    asset_refs = {item.get("evidence_ref") for item in ask["research"].get("assets", [])}
    selected: set[str] = set()
    for ref in intent["evidence_refs"]:
        if ref == asset and ref in asset_refs:
            continue
        compatible = [fact["fact_id"] for fact in facts if ref in fact["evidence_refs"]]
        if len(compatible) != 1:
            raise error
        selected.add(compatible[0])
    if (not selected or len(selected) > 6 or
            any(fact_id in allowed_set and fact_id not in selected for fact_id in ids)):
        raise error
    return [fact_id for fact_id in allowed if fact_id in selected]


def _canonical_evidence_refs(
    intent: dict[str, Any], ask: dict[str, Any], allowed: set[str],
) -> list[str]:
    """Derive factual and visual authorization before using raw refs for order."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    fact_refs = list(dict.fromkeys(
        ref for fact_id in intent["fact_ids"]
        for ref in facts[fact_id]["evidence_refs"] if ref in allowed
    ))
    selected_asset = intent.get("asset_ref")
    asset_refs = {asset.get("evidence_ref") for asset in ask["research"].get("assets", [])}
    contract = SCENE_CONTRACTS[intent["type"]]
    visual_refs = set(_visual_asset_refs(ask["evidence_index"])) & _research_refs(ask["research"])
    # An opening hero can use the sole authentic project visual when the model
    # leaves asset_ref null. Do not replace an explicit choice or guess among
    # multiple images; the ordinary authorization below still applies.
    if contract.requires_asset_ref and intent["type"] == "HERO" and selected_asset is None:
        scoped_visuals = visual_refs & allowed
        if len(scoped_visuals) == 1:
            only = next(iter(scoped_visuals))
            primary = any(entry["ref"] == only and
                          (entry.get("evidence_role") or "primary") == "primary"
                          for entry in ask["evidence_index"])
            authentic = any(asset.get("evidence_ref") == only and
                            asset.get("authentic_project_media") is True
                            for asset in ask["research"].get("assets", []))
            if primary and authentic:
                selected_asset = only
                intent["asset_ref"] = only
    if contract.requires_asset_ref and not visual_refs:
        raise ValueError(f"{intent['id']}: no planner-scoped visual asset_ref supports this evidence scene; choose a compatible non-evidence scene type")
    if contract.requires_asset_ref and (
        not isinstance(selected_asset, str) or not selected_asset.strip() or
        selected_asset not in allowed or selected_asset not in visual_refs or
        (selected_asset not in fact_refs and selected_asset not in asset_refs)
    ):
        eligible = [ref for ref in ask.get("visual_asset_refs", _visual_asset_refs(ask["evidence_index"]))
                    if ref in allowed and ref in visual_refs and
                    (ref in fact_refs or ref in asset_refs)]
        options = ", ".join(eligible[:8]) or "none (choose a non-evidence scene type)"
        raise ValueError(
            f"{intent['id']}: asset_ref must be a renderable, scoped selected fact or "
            f"research asset; allowed asset_ref: {options}"
        )
    authorized = set(fact_refs)
    if contract.requires_asset_ref and selected_asset in asset_refs:
        authorized.add(selected_asset)
    kept = list(dict.fromkeys(ref for ref in intent["evidence_refs"] if ref in authorized))
    if selected_asset is not None and selected_asset not in kept:
        if selected_asset not in authorized or len(kept) >= 6:
            raise ValueError(f"{intent['id']}: no room for selected asset_ref in evidence_refs")
        kept.append(selected_asset)
    if contract.requires_facts and not any(
        ref in fact_refs for ref in kept
    ):
        if not fact_refs or len(kept) >= 6:
            raise ValueError(f"{intent['id']}: selected facts have no room for scoped evidence")
        kept.append(fact_refs[0])
    if len(kept) > 6:
        raise ValueError(f"{intent['id']}: evidence_refs exceeds six selected refs")
    return kept


def _canonical_outline_intent(
    raw: Any, index: int, allowed: set[str], ask: dict[str, Any],
    *, scene_id: str | None = None,
) -> dict[str, Any]:
    """Turn one raw intent into the only form accepted by downstream validators."""
    if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or raw["type"] not in SCENE_TYPES:
        raise StructuredOutputError(f"Storyboard intent {index} has an invalid scene type")
    purpose = raw.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        raise StructuredOutputError(f"Storyboard intent {index} needs a purpose")
    try:
        refs = _draft_evidence_refs(raw)
    except ValueError as exc:
        raise StructuredOutputError(f"Storyboard intent {index} has invalid {exc}") from exc
    intent = {"id": scene_id or f"s{index:03d}", "type": raw["type"],
              "purpose": purpose.strip(), "evidence_refs": list(dict.fromkeys(refs))}
    if SCENE_CONTRACTS[raw["type"]].allows_asset_ref:
        # Carry the choice for unique fact-ID recovery, but authorize it only
        # after the selected fact IDs have been fixed.
        intent["asset_ref"] = raw.get("asset_ref")
    valid_fact_ids = _valid_outline_fact_ids(raw.get("fact_ids"), intent, ask)
    if not valid_fact_ids and any(ref not in allowed for ref in intent["evidence_refs"]):
        raise StructuredOutputError(f"Storyboard intent {index} cites evidence outside planner scope")
    try:
        intent["fact_ids"] = _outline_fact_ids(raw.get("fact_ids"), intent, ask)
        intent["evidence_refs"] = _canonical_evidence_refs(intent, ask, allowed)
        _validate_scene_facts(intent, ask)
    except StructuredOutputError:
        raise
    except ValueError as exc:
        raise StructuredOutputError(f"Storyboard intent {index} invalid: {exc}") from exc
    return intent


def _normalize_outline(value: dict[str, Any], allowed: set[str], ask: dict[str, Any],
                       *, check_coverage: bool = True,
                       provider: LLMProvider | None = None,
                       project_dir: Path | None = None,
                       progress: Progress | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") not in (1, "1"):
        raise StructuredOutputError("Storyboard outline must have version 1")
    outline = {"version": 1}
    for key, limit in (("title", 240), ("slug", 100), ("summary", None)):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip() or (
            limit is not None and len(item) > limit
        ):
            raise StructuredOutputError(f"Storyboard outline requires a concise {key}")
        outline[key] = item.strip()
    raw_intents = value.get("scene_intents")
    if not isinstance(raw_intents, list) or not 1 <= len(raw_intents) <= _MAX_STORYBOARD_SCENES:
        raise StructuredOutputError(
            f"Storyboard outline requires 1–{_MAX_STORYBOARD_SCENES} scene intents"
        )
    # A canonical checkpoint can have gaps after pruning. Keep its ordered IDs
    # on re-normalization so its presentation and scene parts remain stable.
    supplied_ids = [raw.get("id") if isinstance(raw, dict) else None
                    for raw in raw_intents]
    preserve_ids = all(isinstance(scene_id, str) and
                       re.fullmatch(r"s\d{3}", scene_id) and
                       1 <= int(scene_id[1:]) <= _MAX_STORYBOARD_SCENES
                       for scene_id in supplied_ids) and all(
                           int(left[1:]) < int(right[1:])
                           for left, right in zip(supplied_ids, supplied_ids[1:])
                       )
    intents = []
    for index, raw in enumerate(raw_intents, 1):
        intents.append(_canonical_outline_intent(
            raw, index, allowed, ask,
            scene_id=supplied_ids[index - 1] if preserve_ids else None,
        ))
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
    try:
        facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
        canonical = []
        for intent in intents:
            try:
                validate_scene_type(intent, facts, ask["evidence_index"])
            except SceneTypeUnsuitable:
                # No facts or refs change. Only a specialized presentation that
                # those facts cannot express is reduced to a factual summary.
                if intent["type"] not in SPECIALIZED_TYPES:
                    raise
                purpose = intent["purpose"]
                if purpose == _EDITORIAL_SCENE_PURPOSES[intent["type"]]:
                    purpose = _EDITORIAL_SCENE_PURPOSES["SUMMARY"]
                intent = {**intent, "type": "SUMMARY", "purpose": purpose}
            canonical.append(intent)
        intents, presentation = _canonical_story_composition(
            canonical, outline.get("presentation"), ask, allowed,
            enforce_coverage=check_coverage,
            provider=provider, project_dir=project_dir, progress=progress,
        )
        if presentation is not None:
            outline["presentation"] = presentation
    except _RequestedCoverageError:
        raise
    except ValueError as exc:
        raise StructuredOutputError(f"Storyboard outline invalid: {exc}") from exc
    outline["scene_intents"] = intents
    try:
        validate_novelty(intents)
    except ValueError as exc:
        raise StructuredOutputError(f"Storyboard outline invalid: {exc}") from exc
    if check_coverage:
        _validate_focus_coverage(intents, ask)
    return outline


def _recover_outline_assets(
    value: Any, allowed: set[str], ask: dict[str, Any],
    *, require_explicit_cited_asset: bool = False,
) -> dict[str, Any] | None:
    """Retain grounded facts when an exhausted outline chose an unusable visual.

    A rejected asset never becomes evidence. After normal retries are exhausted,
    strip only that rejected visual choice, recover the selected facts under a
    non-visual contract, then either use one unambiguous authorized visual or
    downgrade the scene to SUMMARY. Malformed fact selections still fail closed.
    """
    if not isinstance(value, dict) or not isinstance(value.get("scene_intents"), list):
        return None
    repaired = []
    changed = False
    visual_refs = set(_visual_asset_refs(ask["evidence_index"])) & _research_refs(ask["research"])
    asset_refs = {asset.get("evidence_ref") for asset in ask["research"].get("assets", [])}
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}

    for index, raw in enumerate(value["scene_intents"], 1):
        try:
            _canonical_outline_intent(raw, index, allowed, ask)
        except StructuredOutputError as exc:
            if ("asset_ref" not in str(exc) or not isinstance(raw, dict) or
                    raw.get("type") not in EVIDENCE_TYPES):
                return None
            try:
                refs = _draft_evidence_refs(raw)
            except ValueError:
                return None
            selected_asset = raw.get("asset_ref")
            if selected_asset is not None and not isinstance(selected_asset, str):
                return None
            if require_explicit_cited_asset and (
                not isinstance(selected_asset, str) or not selected_asset.strip() or
                selected_asset not in refs or selected_asset not in allowed
            ):
                return None

            # The rejected visual cannot participate in fact-ID recovery. With
            # valid fact IDs, other noisy refs are handled by the ordinary
            # canonical ref filter; with missing/invalid IDs, unique ref repair
            # still requires every remaining cited ref to be in scope.
            repair_refs = [
                ref for ref in refs
                if not (selected_asset and ref == selected_asset and ref not in allowed)
            ]
            repair_intent = {
                "id": raw.get("id") or f"s{index:03d}",
                "type": "SUMMARY",
                "evidence_refs": list(dict.fromkeys(repair_refs)),
            }
            valid_fact_ids = _valid_outline_fact_ids(raw.get("fact_ids"), repair_intent, ask)
            if not valid_fact_ids and any(ref not in allowed for ref in repair_refs):
                return None
            try:
                fact_ids = _outline_fact_ids(raw.get("fact_ids"), repair_intent, ask)
            except StructuredOutputError:
                return None

            fact_refs = {
                ref for fact_id in fact_ids for ref in facts[fact_id]["evidence_refs"]
                if ref in allowed
            }
            eligible = [
                ref for ref in ask.get(
                    "visual_asset_refs", _visual_asset_refs(ask["evidence_index"])
                )
                if ref in allowed and ref in visual_refs and
                (ref in fact_refs or ref in asset_refs)
            ]
            if len(eligible) == 1:
                replacement_asset = eligible[0]
                visual = {**raw, "fact_ids": fact_ids, "asset_ref": replacement_asset}
                visual["evidence_refs"] = [
                    ref for ref in refs
                    if ref != selected_asset or ref == replacement_asset
                ]
                if replacement_asset not in visual["evidence_refs"]:
                    visual["evidence_refs"].append(replacement_asset)
                try:
                    _canonical_outline_intent(visual, index, allowed, ask)
                except StructuredOutputError:
                    pass
                else:
                    repaired.append(visual)
                    changed = True
                    continue

            try:
                downgraded = {**raw, "type": "SUMMARY", "fact_ids": fact_ids}
                downgraded.pop("asset_ref", None)
                downgraded["evidence_refs"] = _canonical_evidence_refs(
                    {
                        "id": repair_intent["id"],
                        "type": "SUMMARY",
                        "fact_ids": fact_ids,
                        "evidence_refs": repair_refs,
                    },
                    ask, allowed,
                )
            except ValueError:
                return None
            repaired.append(downgraded)
            changed = True
        else:
            repaired.append(raw)
    return {**value, "scene_intents": repaired} if changed else None


def _outline_error_intent_index(error: Exception, value: Any) -> int | None:
    """Locate only the intent named by a deterministic outline validation error."""
    if not isinstance(value, dict) or not isinstance(value.get("scene_intents"), list):
        return None
    intents = value["scene_intents"]
    message = str(error)
    match = re.search(r"\b(s\d{3})\b", message)
    if match:
        scene_id = match.group(1)
        for index, raw in enumerate(intents):
            if isinstance(raw, dict) and raw.get("id") == scene_id:
                return index
        numeric = int(scene_id[1:]) - 1
        if 0 <= numeric < len(intents):
            return numeric
    match = re.search(r"Storyboard intent (\d+)", message)
    if match:
        numeric = int(match.group(1)) - 1
        if 0 <= numeric < len(intents):
            return numeric
    return None


def _recover_outline_missing_fact_ids(
    value: Any, error: Exception, ask: dict[str, Any],
) -> dict[str, Any] | None:
    """Discard only the currently rejected intent when its fact selection is absent.

    A missing selection provides no factual authority to preserve. Other raw
    intents are intentionally left untouched so a later deterministic recovery
    step can handle an independent asset/schema error from the same LLM answer.
    Unknown or contradictory fact IDs are never discarded by this repair.
    """
    index = _outline_error_intent_index(error, value)
    if index is None:
        return None
    intents = value["scene_intents"]
    raw = intents[index]
    if not isinstance(raw, dict) or raw.get("fact_ids") not in (None, []):
        return None
    kind = raw.get("type")
    contract = SCENE_CONTRACTS.get(kind) if isinstance(kind, str) else None
    refs = raw.get("evidence_refs")
    needs_facts = bool(contract and contract.requires_facts) or bool(refs)
    if not needs_facts or len(intents) <= 1:
        return None

    repaired = {**value, "scene_intents": [*intents[:index], *intents[index + 1:]]}
    presentation = repaired.get("presentation")
    if isinstance(presentation, dict):
        presentation = {**presentation}
        presentation.pop("scene_titles", None)
        if not any(isinstance(item, dict) and item.get("type") == "OUTRO"
                   for item in repaired["scene_intents"]):
            presentation.pop("outro", None)
        repaired["presentation"] = presentation
    return repaired


def _recover_outline_asset_error(
    value: Any, error: Exception, allowed: set[str], ask: dict[str, Any],
    *, require_explicit_cited_asset: bool = False,
) -> dict[str, Any] | None:
    """Repair only the asset-invalid intent and leave other draft errors pending."""
    index = _outline_error_intent_index(error, value)
    if index is None:
        return None
    intents = value["scene_intents"]
    raw = intents[index]
    if not isinstance(raw, dict):
        return None
    repaired = _recover_outline_assets(
        {"scene_intents": [raw]}, allowed, ask,
        require_explicit_cited_asset=require_explicit_cited_asset,
    )
    if repaired is None:
        return None
    replacement = repaired["scene_intents"][0]
    return {**value, "scene_intents": [
        *intents[:index], replacement, *intents[index + 1:]
    ]}


_EDITORIAL_PREFIX = re.compile(
    r"^(?:introduce|show|explain|compare|conclude with|summarize|present|describe)\s+", re.I
)
_EDITORIAL_ONLY = re.compile(
    r"(?:(?:the|a|an)\s+)?(?:introduction|opening|closing|conclusion|outro|"
    r"summary|transition|section|point\s+\d+)", re.I
)
_EDITORIAL_SCENE_PURPOSES = {
    kind: f"Plan a {kind.lower().replace('_', ' ')} scene" for kind in SCENE_TYPES
}


def _outline_factual_text(purpose: str) -> str:
    """Discard an editorial verb, never the factual object it introduces."""
    if purpose in _EDITORIAL_SCENE_PURPOSES.values():
        return ""
    content = _EDITORIAL_PREFIX.sub("", purpose.strip()).strip().rstrip(". ")
    if _EDITORIAL_ONLY.fullmatch(content):
        return ""
    # "Introduce X as Y" asserts "X is Y". Keep that assertion for checking.
    content = re.sub(r"^(.+?)\s+as\s+((?:an?|the)\s+.+)$", r"\1 is \2", content, flags=re.I)
    return content


def _selected_fact_summary(
    intents: list[dict[str, Any]], ask: dict[str, Any], *, label: str,
) -> str:
    """Concatenate only complete, supported claims in episode order."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    ids = dict.fromkeys(fact_id for intent in intents for fact_id in intent["fact_ids"])
    claims: list[str] = []
    length = 0
    for fact_id in ids:
        fact = facts[fact_id]
        claim = fact.get("claim")
        if not isinstance(claim, str) or not claim.strip() or not fact.get("support"):
            continue
        extra = len(claim) + (1 if claims else 0)
        if length + extra <= _MAX_OUTLINE_SUMMARY_CHARS:
            claims.append(claim)
            length += extra
    if not claims:
        raise StructuredOutputError(
            f"{label}: no selected grounded fact claim fits the {_MAX_OUTLINE_SUMMARY_CHARS}-character summary bound"
        )
    return " ".join(claims)


def _safe_intent_purpose(intent: dict[str, Any], facts: dict[str, dict[str, Any]]) -> str:
    for fact_id in intent["fact_ids"]:
        fact = facts[fact_id]
        claim = fact.get("claim")
        if (isinstance(claim, str) and claim.strip() and
                len(claim) <= _MAX_OUTLINE_PURPOSE_CHARS and fact.get("support")):
            return claim
    return _EDITORIAL_SCENE_PURPOSES[intent["type"]]


def _placeholder_scene_title(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip() or bool(re.fullmatch(
        r"(?:english\s+)?(?:on-screen\s+)?title", value.strip(), re.I,
    ))


def _suggested_scene_title(
    intent: dict[str, Any], facts: dict[str, dict[str, Any]], episode_title: str,
) -> str:
    """Ground a short template title in the selected intent or its first fact."""
    purpose = str(intent.get("purpose") or "").strip()
    if not purpose or purpose in _EDITORIAL_SCENE_PURPOSES.values():
        purpose = next((facts[fact_id]["claim"] for fact_id in intent.get("fact_ids", [])
                        if fact_id in facts), episode_title)
    else:
        purpose = _EDITORIAL_PREFIX.sub("", purpose)
    first = re.split(r"(?<=[.!?])\s+", purpose.strip(), maxsplit=1)[0].rstrip(". ")
    if len(first) > 96:
        first = first[:93].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return first or episode_title or "Documented observation"


def _coverage_additions(
    scenes: list[dict[str, Any]], ask: dict[str, Any], allowed: set[str],
) -> list[dict[str, Any]]:
    """Cover missing topics with specific facts before broad multi-topic facts."""
    missing = set(_missing_story_topics(scenes, ask))
    groups = ask.get("requested_topic_fact_ids", {})
    facts = ask["research"]["facts"]
    additions = []
    while missing:
        if len(scenes) + len(additions) >= _MAX_STORYBOARD_SCENES:
            raise StructuredOutputError(
                f"Requested storyboard coverage has no room within {_MAX_STORYBOARD_SCENES} scenes: "
                f"{', '.join(sorted(missing))}"
            )
        detail_topics = [topic for topic in groups if topic in missing and
                         topic.startswith("detail-")]
        topic = (detail_topics or [key for key in groups if key in missing])[0]
        candidates = {fact["fact_id"]: fact for fact in facts
                      if fact["fact_id"] in groups[topic]}
        if not candidates:
            raise StructuredOutputError("Requested storyboard coverage has no supported fact candidate")
        detail_groups = [ids for key, ids in groups.items() if key.startswith("detail-")]
        chosen = min((candidates[fact_id] for fact_id in groups[topic] if fact_id in candidates),
                     key=lambda fact: sum(fact["fact_id"] in ids for ids in detail_groups))
        draft = {"type": "SUMMARY", "fact_ids": [chosen["fact_id"]],
                 "evidence_refs": chosen["evidence_refs"][:1],
                 "purpose": (chosen["claim"] if len(chosen["claim"]) <= _MAX_OUTLINE_PURPOSE_CHARS
                             else _EDITORIAL_SCENE_PURPOSES["SUMMARY"])}
        position = len(scenes) - int(bool(scenes and (
            scenes[-1]["type"] == "OUTRO" or ask["required_narration_suffix"]
        )))
        additions.append(_canonical_outline_intent(draft, position + len(additions) + 1,
                                                   allowed, ask))
        missing -= {topic for topic in missing if chosen["fact_id"] in groups[topic]}
    return additions


def _insert_coverage_scenes(
    scenes: list[dict[str, Any]], additions: list[dict[str, Any]],
    presentation: dict[str, Any] | None,
    *, preserve_existing_ids: bool = False, keep_last: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Insert before OUTRO and keep positional IDs and scene titles aligned."""
    if not additions:
        return scenes, presentation
    at = len(scenes) - int(scenes[-1]["type"] == "OUTRO" or keep_last)
    if preserve_existing_ids:
        used = {scene["id"] for scene in scenes}
        serial = max([int(scene_id[1:]) for scene_id in used
                      if re.fullmatch(r"s\d+", scene_id)] + [len(scenes)])
        appended = []
        for addition in additions:
            serial += 1
            while f"s{serial:03d}" in used:
                serial += 1
            scene_id = f"s{serial:03d}"
            used.add(scene_id)
            appended.append({**addition, "id": scene_id})
        return [*scenes[:at], *appended, *scenes[at:]], presentation
    combined = [*scenes[:at], *additions, *scenes[at:]]
    renamed = {scene["id"]: f"s{index:03d}"
               for index, scene in enumerate(combined, 1)
               if not at < index <= at + len(additions)}
    combined = [{**scene, "id": f"s{index:03d}"}
                for index, scene in enumerate(combined, 1)]
    if presentation is not None and "scene_titles" in presentation:
        presentation = {**presentation, "scene_titles": {
            renamed.get(scene_id, scene_id): title
            for scene_id, title in presentation["scene_titles"].items()
        }}
    return combined, presentation


def _recover_outline_coverage(
    outline: dict[str, Any], ask: dict[str, Any], allowed: set[str],
) -> dict[str, Any]:
    additions = _coverage_additions(outline["scene_intents"], ask, allowed)
    intents, presentation = _insert_coverage_scenes(
        outline["scene_intents"], additions, outline.get("presentation"),
        keep_last=bool(ask["required_narration_suffix"]),
    )
    recovered = {**outline, "scene_intents": intents}
    if presentation is not None:
        recovered["presentation"] = presentation
    canonical = _normalize_outline(recovered, allowed, ask)
    _validate_focus_coverage(canonical["scene_intents"], ask)
    return canonical


def _canonicalize_outline_metadata(
    provider: LLMProvider, outline: dict[str, Any], ask: dict[str, Any],
    project_dir: Path, progress: Progress | None = None,
) -> dict[str, Any]:
    """Verify model labels once, then replace rejected prose without new claims."""
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    try:
        safe_summary = _selected_fact_summary(
            outline["scene_intents"], ask, label="outline-summary",
        )
    except StructuredOutputError:
        safe_summary = None
    accepted = set()
    checks = []
    if outline["summary"] == safe_summary:
        accepted.add("outline-summary")
    elif len(outline["summary"]) <= _MAX_OUTLINE_SUMMARY_CHARS:
        checks.append({"id": "outline-summary", "claim": outline["summary"],
                       "facts": list(facts.values())})
    for intent in outline["scene_intents"]:
        purpose = intent["purpose"]
        claim = _outline_factual_text(purpose)
        if not claim:
            continue
        if len(purpose) <= _MAX_OUTLINE_PURPOSE_CHARS and any(
            purpose == facts[fact_id]["claim"] and facts[fact_id].get("support")
            for fact_id in intent["fact_ids"]
        ):
            accepted.add(f"outline-{intent['id']}")
        elif len(purpose) <= _MAX_OUTLINE_PURPOSE_CHARS:
            checks.append({"id": f"outline-{intent['id']}", "claim": claim,
                           "facts": [facts[fact_id] for fact_id in intent["fact_ids"]]})
    if checks:
        accepted.update(verify_claims(provider, checks, project_dir, "outline", progress))
    canonical = dict(outline)
    if "outline-summary" not in accepted:
        if safe_summary is None:
            raise StructuredOutputError(
                f"outline-summary: no selected grounded fact claim fits the "
                f"{_MAX_OUTLINE_SUMMARY_CHARS}-character summary bound"
            )
        canonical["summary"] = safe_summary
    intents = []
    for intent in outline["scene_intents"]:
        if (_outline_factual_text(intent["purpose"]) and
                f"outline-{intent['id']}" not in accepted):
            intents.append({**intent, "purpose": _safe_intent_purpose(intent, facts)})
        else:
            intents.append(intent)
    canonical["scene_intents"] = intents
    return canonical


def _canonical_full_summary(
    provider: LLMProvider, episode: dict[str, Any], ask: dict[str, Any],
    project_dir: Path | None, progress: Progress | None,
) -> dict[str, Any]:
    """Full responses use the same non-authoritative summary policy."""
    if "summary" not in episode:
        return episode
    summary = episode["summary"]
    if isinstance(summary, str) and summary.strip() and len(summary) <= _MAX_OUTLINE_SUMMARY_CHARS:
        facts = ask["research"]["facts"]
        check = [{"id": "episode-summary", "claim": summary, "facts": facts}]
        if project_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                accepted = verify_claims(provider, check, Path(tmp), "episode-summary", progress)
        else:
            accepted = verify_claims(provider, check, project_dir, "episode-summary", progress)
        if "episode-summary" in accepted:
            return episode
    fallback = _selected_fact_summary(episode["scenes"], ask, label="episode-summary")
    return {**episode, "summary": fallback}


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
    first = next(index for index, item in enumerate(all_intents)
                 if item["id"] == intents[0]["id"])
    last = first + len(intents)
    neighbors = {}
    for key, offset in (("previous_intent", first - 1), ("next_intent", last)):
        if 0 <= offset < len(all_intents):
            neighbor = all_intents[offset]
            neighbors[key] = {name: neighbor[name] for name in ("id", "type", "purpose")}
    metadata = {key: outline[key] for key in ("version", "title", "slug", "summary", "presentation")
                if key in outline}

    def required_scene(intent: dict[str, Any]) -> dict[str, Any]:
        selected_facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
        scene = {"id": intent["id"], "type": intent["type"],
                 "title": _suggested_scene_title(intent, selected_facts, outline["title"]),
                 "narration": "concise evidence-grounded narration",
                 "fact_ids": intent["fact_ids"],
                 "evidence_refs": intent["evidence_refs"]}
        contract = SCENE_CONTRACTS[intent["type"]]
        if contract.requires_asset_ref:
            scene["asset_ref"] = intent["asset_ref"]
        if contract.requires_diagram:
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
        "editorial_contract": ask["editorial_contract"],
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
            "fixed_asset_ref": "Evidence scenes must keep the exact outline asset_ref and cite it in evidence_refs; selected facts and the fixed asset bound final refs.",
            "diagram_nodes": "Diagram scenes require 2–8 explicit labeled nodes or steps grounded in the supplied part evidence.",
        },
        "required_output": {"scenes": [required_scene(intent) for intent in intents]},
    }


def _normalize_scene_part(
    value: dict[str, Any], intents: list[dict[str, Any]],
    allowed: set[str], outline: dict[str, Any], ask: dict[str, Any],
    provider: LLMProvider | None = None, project_dir: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"scenes"}:
        raise StructuredOutputError("Storyboard part must contain only a scenes list")
    scenes = value["scenes"]
    if not isinstance(scenes, list) or len(scenes) != len(intents):
        raise StructuredOutputError(f"Storyboard part requires exactly {len(intents)} scenes")
    normalized_scenes = []
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    for raw, intent in zip(scenes, intents):
        if not isinstance(raw, dict) or raw.get("id") != intent["id"] or raw.get("type") != intent["type"]:
            raise StructuredOutputError(f"Storyboard part scene order/id/type differs from {intent['id']}")
        scene = _canonical_scene_fields(raw)
        if _placeholder_scene_title(scene.get("title")):
            selected_facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
            scene["title"] = _suggested_scene_title(intent, selected_facts, outline["title"])
        if (intent["type"] == "SECTION_TITLE" and not intent["fact_ids"] and
                isinstance(scene.get("narration"), str) and
                re.fullmatch(r"closing\W*", scene["narration"].strip(), re.I)):
            scene["narration"] = "Next, a closer look."
        try:
            refs = _draft_evidence_refs(scene)
        except ValueError as exc:
            raise StructuredOutputError(f"{intent['id']}: {exc}") from exc
        contract = SCENE_CONTRACTS[intent["type"]]
        if contract.requires_asset_ref:
            fixed = intent["asset_ref"]
            if fixed not in allowed:
                raise StructuredOutputError(f"{intent['id']}: fixed asset_ref outside this part's planner scope")
            returned = scene.get("asset_ref")
            if returned not in (None, ""):
                if not isinstance(returned, str) or returned not in allowed:
                    raise StructuredOutputError(f"{intent['id']}: asset_ref outside this part's planner scope")
                if returned != fixed:
                    raise StructuredOutputError(f"{intent['id']}: asset_ref differs from fixed outline choice {fixed}")
            scene["asset_ref"] = fixed
        if contract.requires_diagram:
            diagram = scene.get("diagram")
            nodes = (diagram.get("nodes") or diagram.get("steps")) if isinstance(diagram, dict) else None
            if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8:
                raise StructuredOutputError(f"{intent['id']}: diagram nodes/steps require 2–8 labeled entries")
        try:
            if "fact_ids" not in scene:
                scene["fact_ids"] = intent["fact_ids"]
            elif scene["fact_ids"] != intent["fact_ids"]:
                # Fix a stale echo of the IDs only when the returned refs
                # already belong to the outline's selected facts and asset.
                # A response citing a different fact still fails validation.
                facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
                selected_refs = {ref for fact_id in intent["fact_ids"]
                                 for ref in facts[fact_id]["evidence_refs"]}
                if contract.requires_asset_ref:
                    selected_refs.add(intent["asset_ref"])
                if any(ref not in selected_refs for ref in refs):
                    raise ValueError(f"{intent['id']}: scene fact_ids differ from fixed outline selection")
                scene["fact_ids"] = intent["fact_ids"]
            scene["evidence_refs"] = _canonical_evidence_refs(
                {**intent, "evidence_refs": refs}, ask, allowed,
            )
            if (scene["type"] == "SUMMARY" and len(scene["fact_ids"]) > 1 and
                    isinstance(scene.get("narration"), str)):
                claims = [facts[fact_id]["claim"] for fact_id in scene["fact_ids"]]
                title_words = _topic_words(str(scene.get("title", "")))
                if (scene["narration"].startswith(claims[0]) and
                        len(title_words & _topic_words(claims[0])) < 2 and
                        any(len(title_words & _topic_words(claim)) >= 2
                            for claim in claims[1:])):
                    scene["title"] = _suggested_scene_title(
                        {"fact_ids": scene["fact_ids"][:1], "purpose": ""},
                        facts, outline["title"],
                    )
            _validate_scene_facts(scene, ask, intent["fact_ids"])
            validate_scene_type(scene, {fact["fact_id"]: fact for fact in ask["research"]["facts"]},
                                ask["evidence_index"])
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
        _validate_structured_grounding(provider, partial["scenes"], ask, project_dir, progress)
        if intents[-1]["id"] == outline["scene_intents"][-1]["id"]:
            _validate_final_narration(partial, ask)
        if provider is not None:
            _validate_narration_grounding(
                provider, partial["scenes"], ask, project_dir, progress,
                trusted_fallbacks=_recognized_grounded_narrations(
                    partial["scenes"], ask, outline["scene_intents"][-1]["id"],
                ),
                final_scene_id=outline["scene_intents"][-1]["id"],
            )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        if isinstance(exc, _NarrationGroundingRejected):
            raise
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
        feedback: str | None = None
        brief_feedback = ""
        coverage_candidate: dict[str, Any] | None = None
        last_outline_candidate: dict[str, Any] | None = None

        class OutlineRetryProvider:
            def complete_json(self, request_system: str, user: str) -> dict[str, Any]:
                if feedback:
                    for hint in (feedback, brief_feedback):
                        revised_user = json.dumps({**json.loads(user),
                                                   "validation_feedback": hint}, ensure_ascii=False)
                        if fits_context(request_system, revised_user, context_size,
                                        output_reserve_tokens, safety_tokens):
                            user = revised_user
                            break
                    else:
                        raise StructuredOutputError(
                            "Storyboard outline retry feedback exceeds the configured context budget"
                        )
                return provider.complete_json(request_system, user)

        def normalize_outline(value: dict[str, Any]) -> dict[str, Any]:
            nonlocal feedback, brief_feedback, coverage_candidate, last_outline_candidate
            last_outline_candidate = value
            coverage_candidate = None
            try:
                normalized = _normalize_outline(
                    value, allowed, ask, check_coverage=False,
                    provider=provider, project_dir=project_dir, progress=progress,
                )
                canonical = _canonicalize_outline_metadata(provider, normalized, ask,
                                                           project_dir, progress)
                # Metadata grounding can replace an unsupported section purpose
                # with a generic label. Apply composition once more so the
                # recovered label cannot leave an empty chapter in the outline.
                canonical = _normalize_outline(canonical, allowed, ask,
                                               check_coverage=False)
                coverage_candidate = canonical
                _validate_focus_coverage(canonical["scene_intents"], ask)
                return canonical
            except (StructuredOutputError, _RequestedCoverageError) as exc:
                feedback = _retry_feedback(exc)
                brief_feedback = _brief_outline_feedback(exc)[:_OUTLINE_RETRY_FEEDBACK_RESERVE]
                raise

        def recover_outline(error: Exception) -> dict[str, Any] | None:
            if isinstance(error, _RequestedCoverageError) and coverage_candidate is not None:
                recovered = _recover_outline_coverage(coverage_candidate, ask, allowed)
                if progress is not None:
                    progress.note("Outline coverage retries exhausted; adding grounded summary scenes")
                return recovered
            if not isinstance(error, StructuredOutputError) or last_outline_candidate is None:
                return None

            draft = last_outline_candidate
            current_error: Exception = error
            intents = draft.get("scene_intents") if isinstance(draft, dict) else None
            max_steps = min(_MAX_STORYBOARD_SCENES + 4,
                            (len(intents) if isinstance(intents, list) else 0) + 6)
            for _ in range(max(1, max_steps)):
                message = str(current_error)
                repaired = None
                if "invalid fact_ids" in message:
                    repaired = _recover_outline_missing_fact_ids(draft, current_error, ask)
                elif "asset_ref" in message or "visual asset" in message:
                    repaired = _recover_outline_asset_error(
                        draft, current_error, allowed, ask,
                        require_explicit_cited_asset=max_retries < 1,
                    )
                    if repaired is None:
                        repaired = _recover_outline_missing_fact_ids(
                            draft, current_error, ask,
                        )
                if repaired is None or repaired == draft:
                    return None
                draft = repaired

                normalized = None
                try:
                    normalized = _normalize_outline(
                        draft, allowed, ask, check_coverage=False,
                        provider=provider, project_dir=project_dir, progress=progress,
                    )
                    recovered = _canonicalize_outline_metadata(
                        provider, normalized, ask, project_dir, progress,
                    )
                    recovered = _normalize_outline(
                        recovered, allowed, ask, check_coverage=False,
                    )
                    try:
                        _validate_focus_coverage(recovered["scene_intents"], ask)
                    except _RequestedCoverageError:
                        recovered = _recover_outline_coverage(recovered, ask, allowed)
                    if progress is not None:
                        progress.note(
                            "Outline retries exhausted; applied bounded deterministic "
                            "repairs and restored grounded requested coverage"
                        )
                    return recovered
                except _RequestedCoverageError:
                    try:
                        if normalized is None:
                            return None
                        return _recover_outline_coverage(normalized, ask, allowed)
                    except (StructuredOutputError, ValueError):
                        return None
                except StructuredOutputError as exc:
                    current_error = exc
                    continue
                except ValueError:
                    return None
            return None

        outline_checkpoint = part_dir / "outline.json"
        outline = checkpointed_complete_json(
            OutlineRetryProvider(), outline_system, outline_payload, outline_checkpoint,
            normalize_outline, max_retries=max_retries, recover_exhausted=recover_outline,
        )
        # A checkpoint from an older planner may contain prose that now needs
        # safe canonicalization. Replace its result before scene generation.
        cached = json_load(outline_checkpoint)
        if cached["result"] != outline:
            json_dump(outline_checkpoint, {**cached, "result": outline})

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
        part_feedback: dict[tuple[str, ...], str] = {}
        rejected_candidates: dict[tuple[str, ...], dict[str, Any]] = {}

        class SceneRetryProvider:
            def complete_json(self, request_system: str, user: str) -> dict[str, Any]:
                request = json.loads(user)
                key = tuple(item["id"] for item in request["scene_intents"])
                if key in part_feedback:
                    revised = json.dumps({**request, "validation_feedback": part_feedback[key]},
                                         ensure_ascii=False)
                    if fits_context(request_system, revised, context_size,
                                    output_reserve_tokens, safety_tokens):
                        user = revised
                return provider.complete_json(request_system, user)

        def payload_for(items: list[dict[str, Any]]) -> dict[str, Any]:
            return _scene_part_payload(ask, outline, items, number, len(groups), scope_id)

        def normalize_for(value: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
            part_allowed = {entry["ref"] for entry in payload_for(items)["evidence_index"]}
            try:
                return _normalize_scene_part(value, items, part_allowed, outline, ask,
                                             provider, project_dir, progress)
            except _NarrationGroundingRejected as exc:
                if len(items) == 1:
                    rejected_candidates[(items[0]["id"],)] = copy.deepcopy(value)
                part_feedback[tuple(item["id"] for item in items)] = _retry_feedback(exc)
                raise
            except StructuredOutputError as exc:
                part_feedback[tuple(item["id"] for item in items)] = _retry_feedback(exc)
                raise

        def recover_single(items: list[dict[str, Any]], error: Exception) -> dict[str, Any] | None:
            intent = items[0]
            facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}

            def exact_outline_scene(note: str) -> dict[str, Any] | None:
                """Realize one already-validated intent without trusting rejected scene output."""
                scene = {
                    "id": intent["id"], "type": intent["type"],
                    "title": _safe_intent_purpose(intent, facts),
                    "fact_ids": intent["fact_ids"],
                    "evidence_refs": intent["evidence_refs"],
                }
                contract = SCENE_CONTRACTS[intent["type"]]
                if contract.requires_asset_ref:
                    scene["asset_ref"] = intent["asset_ref"]
                if contract.requires_diagram:
                    labels = [facts[fact_id]["claim"] for fact_id in intent["fact_ids"]]
                    if len(labels) < 2:
                        return None
                    scene["diagram"] = {"nodes": labels[:8]}
                if intent["type"] == "GRAPH":
                    return None  # Numerical points cannot be inferred from fact IDs.
                fallback = _grounded_narration_fallback(
                    intent, ask, is_final=intent["id"] == outline["scene_intents"][-1]["id"],
                )
                scene["narration"] = fallback
                part_allowed = {entry["ref"] for entry in payload_for(items)["evidence_index"]}
                canonical = _normalize_scene_part(
                    {"scenes": [scene]}, items, part_allowed, outline, ask,
                )
                if canonical["scenes"][0]["narration"] != fallback:
                    raise StructuredOutputError(
                        f"{intent['id']}: grounded fallback narration was modified"
                    )
                if progress is not None:
                    progress.note(f"{intent['id']}: {note}; using exact outline claims")
                return canonical

            message = str(error)
            if (max_retries > 0 and isinstance(error, StructuredOutputError) and
                    "scene fact_ids differ from fixed outline selection" in message):
                # The final scene response still cites a different fact. Do
                # not keep any of its prose, refs, or structured labels.
                return exact_outline_scene("fact selection retries exhausted")

            if isinstance(error, StructuredOutputError) and any(fragment in message for fragment in (
                "Storyboard part requires exactly 1 scenes",
                "Storyboard part must contain only a scenes list",
                "Storyboard part scene order/id/type differs from",
            )):
                # The outline already fixed scene identity, type, selected
                # facts and any authentic asset. A malformed one-scene model
                # response cannot add authority, so reconstruct only from that
                # validated intent instead of failing an unsplittable record.
                return exact_outline_scene("single-scene structure retries exhausted")

            if not isinstance(error, _NarrationGroundingRejected):
                return None
            candidate = rejected_candidates.get((intent["id"],))
            if candidate is None:
                return None  # No structurally valid, rejected narration was ever returned.
            fallback = _grounded_narration_fallback(
                intent, ask, is_final=intent["id"] == outline["scene_intents"][-1]["id"],
            )
            repaired = {**candidate, "scenes": [
                {**candidate["scenes"][0], "narration": fallback}
            ]}
            part_allowed = {entry["ref"] for entry in payload_for(items)["evidence_index"]}
            canonical = _normalize_scene_part(repaired, items, part_allowed, outline, ask)
            if canonical["scenes"][0]["narration"] != fallback:
                raise StructuredOutputError(f"{intent['id']}: grounded fallback narration was modified")
            if progress is not None:
                progress.note(f"{intent['id']}: narration grounding retries exhausted; using exact selected claims")
            return canonical


        try:
            results = checkpointed_split_json(
                SceneRetryProvider(), scene_system, group, payload_for, checkpoint, normalize_for,
                max_retries=max_retries, progress=progress,
                label=f"Storyboard part {number}/{len(groups)}", used=used,
                recover_single=recover_single,
            )
        except RuntimeError as exc:
            if "narration introduces an unsupported factual proposition" in str(exc):
                raise GroundingError(
                    f"Storyboard grounding failed in part {number}: "
                    "a scene narration adds an unsupported factual proposition"
                ) from None
            raise
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
                episode = _complete_episode(provider, system, ask, allowed, max_retries,
                                            project_dir=project_dir, progress=progress)
        except (StructuredOutputError, json.JSONDecodeError, _EpisodeValidationExhausted) as exc:
            if isinstance(exc, _EpisodeValidationExhausted) and (
                "unsupported factual proposition" in str(exc) or
                "selected fact lacks exact evidence support" in str(exc)
            ):
                raise GroundingError(
                    "Storyboard grounding failed: narration adds an unsupported "
                    "factual proposition after bounded retries"
                ) from None
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
    research = _supported_research(research, inventory)
    english_facts = [fact for fact in research["facts"]
                     if not _obviously_german_claim(fact["claim"])]
    if english_facts:
        research = {**research, "facts": english_facts}
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
    # Keep two distinct project visuals in small local contexts when available.
    # They are compact; final request fitting still governs the reduce levels.
    media_limit = min(_PLANNER_MEDIA_LIMIT, max(2, context_size // 8192)) \
        if context_size >= 8192 else 1
    selected_visuals: list[dict[str, Any]] = []
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
                return _normalize_capsules(value, valid & provided_refs, media_ref_set & provided_media, items)

            results = checkpointed_split_json(
                provider, compact_system, batch,
                lambda items: make_compact_payload(level, part, items),
                checkpoint, normalize_part,
                max_retries=max_retries, progress=progress,
                label=f"Planning evidence — level {level}, part {part}/{len(chunks)}",
            )
            for result in results:
                capsules.extend(result["capsules"])

        if level == 1:
            selected_hints: dict[str, str] = {}
            for capsule in capsules:
                for ref in capsule.get("media_refs", []):
                    selected_hints[ref] = " ".join(filter(None, (
                        selected_hints.get(ref, ""), capsule.get("visual_purpose", ""),
                    )))
            selected_visuals = _planner_visuals(
                research, media, title_hint, instructions, inventory,
                media_limit, selected_hints,
            )
        capsules = _carry_planner_visuals(
            _scope_capsules(capsules, anchors, inventory, title_hint, instructions),
            selected_visuals,
        )
        if not capsules:
            raise RuntimeError("Planner compaction returned no evidence-grounded capsules")

        compact_research = _capsules_to_research(capsules)
        selected_refs = _research_refs(compact_research)
        selected_media_refs = {
            ref
            for capsule in capsules
            for ref in capsule.get("media_refs", [])
        }
        compact_media = [_compact_media_item(m) for m in media
                         if m["ref"] in selected_media_refs]
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
