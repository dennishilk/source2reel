"""Source-agnostic editorial constraints for planner-selected facts."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .flow_language import distinct_flow_actions
from .grounding import deterministic_decision


EDITORIAL_CONTRACT = "storyboard-editorial-grounding-v5"
FRAMING_TYPES = {"SECTION_TITLE", "HERO", "OUTRO"}
EVIDENCE_TYPES = {"PROJECT_EVIDENCE", "TERMINAL_EVIDENCE", "HARDWARE_EVIDENCE"}
SPECIALIZED_TYPES = {"ARCHITECTURE_DIAGRAM", "DATA_FLOW", "TIMELINE", "GRAPH", "CODE"}


class SceneTypeUnsuitable(ValueError):
    """The selected facts are valid, but cannot carry this presentation type."""
_CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".c", ".h", ".cpp",
    ".cs", ".java", ".kt", ".swift", ".rb", ".sh", ".bash", ".ps1", ".sql",
    ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".xml", ".ini",
}
_CODE_LITERAL = re.compile(
    r"(?:```\w*\s*\n|^\s*(?:\$\s+|(?:async\s+)?def\s+|class\s+|import\s+|"
    r"from\s+\S+\s+import\s+|[\w.]+\s*\([^\n)]*\)\s*[=:;]))", re.M,
)
_TEMPORAL = re.compile(
    r"\b(?:before|after|then|next|first|last|followed by|preced(?:es|ed|ing)|"
    r"subsequent(?:ly)?|prior to|later|earlier|chronolog(?:y|ical)|"
    r"timestamp|time-ordered|sequence|at \d{1,2}:\d{2})\b", re.I,
)
_TEMPORAL_DATE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|"
    r"(?:on|in|by|during|as of)\s+(?:19|20)\d{2}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b(?:\s+\d{1,2})?(?:,?\s+(?:19|20)\d{2})?|"
    r"\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)(?:\s+(?:19|20)\d{2})?|"
    r"\d{1,2}:\d{2}(?::\d{2})?)\b", re.I,
)
_TEMPORAL_BRIDGE = re.compile(
    r"\b(?:before|after|followed\s+by|follows?|followed|"
    r"precedes?|preceded|prior\s+to|subsequent\s+to|then)\b", re.I,
)
_TEMPORAL_ID = re.compile(
    r"\b(?:M\d+|v\d+(?:\.\d+)*|"
    r"(?:version|milestone|release|phase|stage)\s+[A-Za-z0-9]+(?:\.\d+)*)\b", re.I,
)
_TEMPORAL_WORD = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", re.I)
_TEMPORAL_GENERIC = {
    "a", "an", "the", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "to", "was", "were", "is", "are", "it", "its", "this", "that",
    "version", "milestone", "project", "phase", "stage", "process", "system",
    "event", "events", "physical", "runtime", "image", "commit", "data",
    "first", "last", "next", "then", "later", "earlier", "shipped",
    "launched", "started", "completed", "happened", "occurred", "again",
}
# Match complete relationship clauses, rather than a word found anywhere in
# the concatenation of otherwise unrelated selected facts.
_ARCHITECTURE_RELATIONS = (
    re.compile(r"^(?P<source>.+?)\s+(?:sends?|routes?|passes?)\s+.+?\s+(?:to|into)\s+(?P<target>.+)$", re.I),
    re.compile(r"^(?P<source>.+?)\s+(?:feeds?|calls?|depends?\s+on|connects?\s+(?:to|with)|is\s+connected\s+to)\s+(?P<target>.+)$", re.I),
    re.compile(r"^(?P<target>.+?)\s+receives?\s+.+?\s+from\s+(?P<source>.+)$", re.I),
    re.compile(r"^.+?\s+flows?\s+from\s+(?P<source>.+?)\s+(?:to|into|through)\s+(?P<target>.+)$", re.I),
)
_ARCHITECTURE_PARTS = re.compile(
    r"^(?P<source>.+?)\s+(?:consists?\s+of|comprises?|is\s+composed\s+of|includes?|contains?)\s+(?P<members>.+)$", re.I,
)
_ARCHITECTURE_MEMBER = re.compile(r",\s*|\s+and\s+", re.I)
_ARCHITECTURE_WORD = re.compile(r"[\w-]+", re.U)
_ARCHITECTURE_GENERIC = {
    "component", "components", "subsystem", "subsystems", "part", "parts",
    "module", "modules", "service", "services", "stage", "stages",
    "event", "events", "data", "state", "system", "project",
}
_GRAPHABLE = re.compile(
    r"\b(?:measurements?|data points?|series|samples?|charts?|graphs?|"
    r"values?|counts? (?:at|for|by|over)|rates? (?:at|for|by|over))\b|"
    r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)", re.I,
)
_FLOW_RELATION = re.compile(
    r"\b(?:into|through|from .+? to|then|followed by|passes? .+? to|"
    r"transforms? .+? into|routes? .+? to)\b", re.I,
)
_PURPOSE = re.compile(
    r"\b(?:for\s+(?!example\b|instance\b)|in order to\s+|so that\s+|"
    r"to\s+(?:analy[sz]e|diagnose|improve|enable|provide|show|ensure|prevent|"
    r"support|allow|help|produce|achieve|make)\b)", re.I,
)
_CAUSAL = re.compile(
    r"\b(?:because|therefore|thus|as a result|caus(?:es|ed|ing)|result(?:s|ed|ing) in|"
    r"leads? to|ensures?|guarantees?|prevents?|enables?|allows?)\b", re.I,
)
_DIRECTION = re.compile(r"(?:→|->|⇒)")
_TOKEN = re.compile(r"[a-z][a-z0-9]+", re.I)
_STOP = {"the", "and", "with", "into", "from", "that", "this", "they", "them",
         "for", "via", "are", "its", "not", "only", "over", "then", "which",
         "there", "their", "into", "can", "does", "was", "were", "without"}


def selected_claims(scene: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    return [facts[fact_id]["claim"] for fact_id in scene.get("fact_ids", [])]


def _code_evidence(scene: dict[str, Any], facts: dict[str, dict[str, Any]],
                   evidence_index: list[dict[str, Any]]) -> bool:
    entries = {entry["ref"]: entry for entry in evidence_index}
    for fact_id in scene.get("fact_ids", []):
        fact = facts[fact_id]
        for ref in fact["evidence_refs"]:
            if ref not in scene.get("evidence_refs", []):
                continue
            entry = entries.get(ref, {})
            if (entry.get("kind") in {"code", "source"} or
                    Path(entry.get("relative_path") or "").suffix.lower() in _CODE_EXTENSIONS):
                return True
            if any(span.get("evidence_ref") == ref and
                   _CODE_LITERAL.search(span.get("text", "")) for span in fact.get("support", [])):
                return True
    return False


def _architecture_entity(phrase: str) -> str | None:
    """Keep literal endpoint names; generic component nouns alone are not nodes."""
    words = _ARCHITECTURE_WORD.findall(phrase.strip().strip("`'\" "))
    if len(words) > 1 and words[0].casefold() in {"a", "an", "the"}:
        words.pop(0)
    if not words or all(word.casefold() in _ARCHITECTURE_GENERIC for word in words):
        return None
    return " ".join(word.casefold() for word in words)


def _architecture_suitable(claims: list[str]) -> bool:
    """Every selected fact must contribute to one connected, explicit topology.

    Endpoint matching is deliberately literal: when two facts use different
    names, the planner can present them as a summary without guessing an edge.
    """
    if not claims:
        return False
    edges: set[tuple[str, str]] = set()
    for claim in claims:
        claim_edges: set[tuple[str, str]] = set()
        for clause in re.split(r"[.;]\s*", claim):
            clause = clause.strip().rstrip(".!? ")
            if not clause:
                continue
            parts = _ARCHITECTURE_PARTS.fullmatch(clause)
            if parts:
                source = _architecture_entity(parts["source"])
                members = [_architecture_entity(member) for member in
                           _ARCHITECTURE_MEMBER.split(parts["members"])]
                # Named enumeration establishes topology; generic membership
                # ("Project includes component X") does not.
                if source and len(members) >= 2 and all(members) and len(set(members)) >= 2:
                    claim_edges.update((source, member) for member in members if member != source)
                continue
            for relation in _ARCHITECTURE_RELATIONS:
                match = relation.fullmatch(clause)
                if match:
                    source = _architecture_entity(match["source"])
                    target = _architecture_entity(match["target"])
                    if source and target and source != target:
                        claim_edges.add((source, target))
                    break
        if not claim_edges:
            return False
        edges.update(claim_edges)
    connected = {next(iter(edges))[0]}
    while True:
        expanded = connected | {right for left, right in edges if left in connected}
        expanded |= {left for left, right in edges if right in connected}
        if expanded == connected:
            break
        connected = expanded
    return all(left in connected and right in connected for left, right in edges)


def _timeline_terms(phrase: str) -> set[str]:
    terms = {word.casefold() for word in _TEMPORAL_WORD.findall(phrase)
             if word.casefold() not in _TEMPORAL_GENERIC}
    # Preserve the identifier in "Version A", while a sentence's article
    # "A" remains too generic to connect two otherwise unrelated claims.
    if phrase.strip() == "A" or re.search(
        r"\b(?:version|milestone|release|model|stage|phase)\s+A\b", phrase, re.I
    ):
        terms.add("a")
    return terms


def _timeline_endpoint_names_fact(endpoint: str, claim: str) -> bool:
    """Require a named event/version, not an incidental shared project noun."""
    # A shared action such as "writes files" cannot connect M63 to M66.
    endpoint_ids = {match.group().split()[-1].casefold()
                    for match in _TEMPORAL_ID.finditer(endpoint)}
    if endpoint_ids and not endpoint_ids & {
        match.group().split()[-1].casefold() for match in _TEMPORAL_ID.finditer(claim)
    }:
        return False
    names = _timeline_terms(endpoint)
    overlap = names & _timeline_terms(claim)
    if len(overlap) >= 2:
        return True
    if any(len(name) == 1 or (any(char.isdigit() for char in name) and
                              any(char.isalpha() for char in name))
           for name in overlap):
        return True
    if len(names) != 1 or not overlap:
        return False
    name = next(iter(names))
    return len(name) >= 4


def _timeline_dated(claim: str) -> bool:
    for match in _TEMPORAL_DATE.finditer(claim):
        # A date-shaped release or version identifier is not a dated event.
        if not re.search(r"\b(?:version|milestone|build|model|revision|release)\s*$",
                         claim[:match.start()], re.I):
            return True
    return False


def _timeline_suitable(claims: list[str]) -> bool:
    """Every selected fact needs a shared clock or an explicitly named order edge.

    An internal sequence in one claim cannot place another claim on that
    sequence. Milestone and version numbers are identities, never ordering.
    """
    if not claims:
        return False
    if len(claims) == 1:
        return bool(_TEMPORAL.search(claims[0]) or _timeline_dated(claims[0]))

    dates = {index for index, claim in enumerate(claims) if _timeline_dated(claim)}
    if len(dates) == len(claims):
        return True
    edges: dict[int, set[int]] = {index: set() for index in range(len(claims))}
    for index in dates:
        edges[index].update(dates - {index})
    for index, claim in enumerate(claims):
        for marker in _TEMPORAL_BRIDGE.finditer(claim):
            # A "then" near the end of a long, comma-separated list does not
            # relate an earlier named milestone to a second selected fact.
            left = re.split(r"[.!?;,]", claim[:marker.start()].rstrip(" ,"))[-1]
            right = re.split(r"[.!?;,]", claim[marker.end():])[0]
            if not _timeline_terms(left) or not _timeline_terms(right):
                continue
            for other, selected in enumerate(claims):
                if other != index and (
                    _timeline_endpoint_names_fact(left, selected) or
                    _timeline_endpoint_names_fact(right, selected)
                ):
                    edges[index].add(other)
                    edges[other].add(index)
    reached = {0}
    while True:
        expanded = reached | {neighbor for index in reached for neighbor in edges[index]}
        if expanded == reached:
            return len(reached) == len(claims)
        reached = expanded


def validate_scene_type(scene: dict[str, Any], facts: dict[str, dict[str, Any]],
                        evidence_index: list[dict[str, Any]]) -> None:
    """Reject special templates whose selected facts cannot supply their semantics."""
    kind = scene["type"]
    claims = selected_claims(scene, facts)
    text = " ".join(claims)
    if kind == "DATA_FLOW" and not (
        len(distinct_flow_actions(text)) >= 2 or
        distinct_flow_actions(text) and _FLOW_RELATION.search(text) or
        re.search(r"\bworkflow\b.+\bthen\b", text, re.I)
    ):
        raise SceneTypeUnsuitable(f"{scene['id']}: DATA_FLOW requires selected facts describing an actual flow")
    if kind == "TIMELINE" and not _timeline_suitable(claims):
        raise SceneTypeUnsuitable(f"{scene['id']}: TIMELINE requires explicit temporal or order evidence")
    if kind == "ARCHITECTURE_DIAGRAM" and not _architecture_suitable(claims):
        raise SceneTypeUnsuitable(f"{scene['id']}: ARCHITECTURE_DIAGRAM requires a coherent selected structural relationship")
    if kind == "CODE" and not _code_evidence(scene, facts, evidence_index):
        raise SceneTypeUnsuitable(f"{scene['id']}: CODE requires selected code or literal source evidence")
    if kind == "GRAPH" and not (
        _GRAPHABLE.search(text) and len(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", text)) >= 2
    ):
        raise SceneTypeUnsuitable(f"{scene['id']}: GRAPH requires selected graphable numerical data")


def _novelty_scan(
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify repetitions using only selected facts and distinct evidence assets."""
    exhausted: set[str] = set()
    used_assets: set[str] = set()
    kept = []
    redundant = []
    for scene in scenes:
        if scene["type"] in FRAMING_TYPES:
            kept.append(scene)
            continue
        selected = set(scene.get("fact_ids", []))
        asset = scene.get("asset_ref") if scene["type"] in EVIDENCE_TYPES else None
        new_asset = isinstance(asset, str) and asset not in used_assets
        if selected and selected <= exhausted and not new_asset:
            redundant.append(scene)
            continue
        kept.append(scene)
        exhausted.update(selected)
        if asset:
            used_assets.add(asset)
    return kept, redundant


def prune_redundant_content_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove later content that adds neither a selected fact nor an authentic asset.

    Call only after fact IDs and evidence assets have been canonicalized and checked.
    """
    kept, _ = _novelty_scan(scenes)
    return kept


def validate_novelty(scenes: list[dict[str, Any]]) -> None:
    """An authentic new visual may explain a prior fact; a new template alone cannot."""
    _, redundant = _novelty_scan(scenes)
    if redundant:
        raise ValueError(
            f"{redundant[0]['id']}: content repeats already covered fact_ids "
            "without distinct authentic evidence"
        )


def _words(value: str) -> set[str]:
    return {word for word in _TOKEN.findall(value.casefold()) if len(word) > 2 and word not in _STOP}


def deterministic_field_guard(value: str, claims: list[str]) -> bool:
    """Block expansions even if a semantic verifier would approve a loose paraphrase."""
    if not isinstance(value, str) or not value.strip() or not claims:
        return False
    if any(value.strip().casefold() == claim.strip().casefold() for claim in claims):
        return True
    selected = " ".join(claims)
    if _PURPOSE.search(value) and not _PURPOSE.search(selected):
        return False
    if _CAUSAL.search(value) and not _CAUSAL.search(selected):
        return False
    for marker in (_PURPOSE, _CAUSAL):
        for added in marker.finditer(value):
            tail = _words(re.split(r"[.;,!?]|\b(?:and|but)\b", value[added.end():], 1)[0])
            supported = set().union(*(
                _words(re.split(r"[.;,!?]|\b(?:and|but)\b", claim[match.end():], 1)[0])
                for claim in claims for match in marker.finditer(claim)
            ))
            if tail and not tail <= supported:
                return False
    if _DIRECTION.search(value) and not (
        _DIRECTION.search(selected) or _FLOW_RELATION.search(selected) or _TEMPORAL.search(selected)
    ):
        return False
    if _TEMPORAL.search(value) and not _TEMPORAL.search(selected):
        return False
    words = _words(value)
    common = words & _words(selected)
    if len(words) >= 3 and len(common) < max(2, (len(words) + 1) // 2):
        return False
    return deterministic_decision(value, claims) != "reject"


def _number_in_claim(number: float, claim: str) -> bool:
    return any(float(match.group()) == number for match in re.finditer(
        r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", claim
    ))


def structured_fields(scene: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """Return strictly validated structured labels, excluding optional annotations."""
    fields: list[tuple[str, str]] = []
    diagram = scene.get("diagram") or {}
    if not isinstance(diagram, dict):
        raise ValueError(f"{scene['id']}: diagram must be an object")
    claims = selected_claims(scene, facts)
    for key in ("nodes", "steps"):
        for index, node in enumerate(diagram.get(key) or []):
            if isinstance(node, dict):
                label = node.get("label") or node.get("name")
            else:
                label = node
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{scene['id']}: diagram.{key}[{index}] needs a factual label")
            fields.append((f"diagram.{key}[{index}]", label))
    for key in ("title", "x_label", "y_label", "series", "caption"):
        if key in diagram:
            label = diagram[key]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{scene['id']}: diagram.{key} needs text")
            fields.append((f"diagram.{key}", label))
    if scene["type"] == "CODE" and diagram.get("code"):
        code = diagram["code"]
        spans = [span["text"] for fact_id in scene["fact_ids"]
                 for span in facts[fact_id].get("support", [])]
        if not isinstance(code, str) or not any(code.strip() in span for span in spans):
            raise ValueError(f"{scene['id']}: diagram.code requires an exact selected source excerpt")
    if scene["type"] == "GRAPH":
        points = diagram.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"{scene['id']}: GRAPH requires at least two grounded numerical points")
        for index, point in enumerate(points):
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
                label = point.get("label")
                if label is not None:
                    if not isinstance(label, str):
                        raise ValueError(f"{scene['id']}: diagram.points[{index}].label needs text")
                    if label.strip():
                        fields.append((f"diagram.points[{index}].label", label))
            elif isinstance(point, (tuple, list)) and len(point) == 2:
                x, y = point
            else:
                raise ValueError(f"{scene['id']}: diagram.points[{index}] needs x and y")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (x, y)):
                raise ValueError(f"{scene['id']}: diagram.points[{index}] needs finite numbers")
            if not any(_number_in_claim(x, claim) and _number_in_claim(y, claim) for claim in claims):
                raise ValueError(f"{scene['id']}: diagram.points[{index}] has numbers absent from selected facts")
    return fields
