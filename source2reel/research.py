from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .chunking import checkpointed_complete_json, checkpointed_split_json, fits_context, split_for_context
from .flow_language import distinct_flow_actions
from .grounding import GROUNDING_CONTRACT, _declarative_text, verify_claims
from .inventory import _EMBEDDED_ROOTS
from .progress import Progress, step
from .providers import LLMProvider
from .util import json_dump


MAX_FACTS_PER_REF = 6
MAX_FACTS_PER_REQUEST = 12
MAX_ASSETS_PER_REF = 2
MAX_ASSETS_PER_REQUEST = 6
MAX_RANKED_CANDIDATES = 128
MAX_COVERAGE_RECORDS = 4
MAX_COVERAGE_CHARS = 16000
MAX_SUPPORT_CHARS = 1024
RESEARCH_SEMANTICS_CONTRACT = "requested-topic-semantics-v6"
COVERAGE_CONTRACT = "requested-primary-coverage-v5"


def _payload(batch_number: int, evidence: list[dict[str, Any]], title_hint: str, instructions: str) -> dict[str, Any]:
    return {
        "batch": batch_number,
        "project_title_hint": title_hint,
        "optional_instructions": instructions,
        "evidence": evidence,
        "research_limits": {
            "facts_per_ref": MAX_FACTS_PER_REF,
            "facts_per_request": MAX_FACTS_PER_REQUEST,
            "assets_per_ref": MAX_ASSETS_PER_REF,
            "assets_per_request": MAX_ASSETS_PER_REQUEST,
            "support_chars_per_entry": MAX_SUPPORT_CHARS,
        },
        "grounding_contract": GROUNDING_CONTRACT,
        "research_semantics_contract": RESEARCH_SEMANTICS_CONTRACT,
        "required_output": {
            "facts": [{
                "claim": "...",
                "evidence_refs": ["E0001"],
                "support": [{"evidence_ref": "E0001", "text": "short exact excerpt supporting the claim"}],
                "phase": "development|final|background|unknown",
                "confidence": "high|medium|low",
            }],
            "assets": [{
                "evidence_ref": "E0002",
                "purpose": "...",
                "authentic_project_media": True,
            }],
        },
    }


def _dedupe(items: list[dict[str, Any]], key) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        marker = key(item)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def _split_research_document(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Split a cited excerpt at an existing paragraph, line, or sentence boundary."""
    excerpt = entry.get("excerpt")
    if entry.get("kind") != "document" or not isinstance(excerpt, str):
        return None

    def usable(positions: list[int]) -> list[int]:
        return [position for position in positions
                if excerpt[:position].strip() and excerpt[position:].strip()]

    line_breaks = usable([match.end() for match in re.finditer("\n", excerpt)])
    paragraph_breaks = [position for position in line_breaks
                        if excerpt[:position].endswith("\n\n") and
                        len(excerpt) // 4 <= position <= 3 * len(excerpt) // 4]
    # Fall back to whole lines if paragraphs are too uneven. A long single
    # line may still contain complete sentences, which retain the source text.
    sentence_breaks = usable([match.end() for match in
                              re.finditer(r"(?<=[.!?;])\s+(?=\S)", excerpt)])
    boundaries = paragraph_breaks or line_breaks or sentence_breaks
    if not boundaries:
        return None
    position = min(boundaries, key=lambda value: (abs(2 * value - len(excerpt)), value))
    left = {**entry, "excerpt": excerpt[:position]}
    right = {**entry, "excerpt": excerpt[position:]}
    start = entry.get("line_start")
    if isinstance(start, int):
        right_start = start + left["excerpt"].count("\n")
        left["line_end"] = right_start - int(left["excerpt"].endswith("\n"))
        right["line_start"] = right_start
        right["line_end"] = right_start + right["excerpt"].count("\n") - int(
            right["excerpt"].endswith("\n"))
    return left, right


def _limit_by_ref(items: list[dict[str, Any]], refs, limit: int) -> list[dict[str, Any]]:
    """Bound one original ref across all successful split children and batches."""
    count: dict[str, int] = {}
    kept = []
    for item in items:
        cited = set(refs(item))
        if any(count.get(ref, 0) >= limit for ref in cited):
            continue
        kept.append(item)
        for ref in cited:
            count[ref] = count.get(ref, 0) + 1
    return kept


_REQUEST_STOP = set((
    "explain describe show cover focus clearly distinguish compare make use only do not "
    "invent what why how it is and the a an its to from into with about through "
    "end video episode project source sources evidence technical finished"
).split())
_PURPOSE = re.compile(
    r"\b(?:purpose|goal|motivation|because|so that|designed to|created to|"
    r"built to|aims? to|exists? to|in order to)\b", re.I,
)
_WORKFLOW = re.compile(
    r"\b(?:pipeline|workflow|stages?|transform(?:s|ed|ing)?|turn(?:s|ed|ing)?)\b", re.I,
)
_OVERVIEW = re.compile(
    r"\b(?:is|are)\s+(?:an?|the)\b[^.!?;]{0,100}"
    r"\b(?:engine|tool|system|framework|application|platform|library|service|"
    r"program|analyzer|inspector)\b", re.I,
)


def _rank_requested_facts(
    facts: list[dict[str, Any]], instructions: str, title_hint: str,
) -> list[dict[str, Any]]:
    """Reserve bounded output for distinct, explicitly requested grounded topics."""
    if not instructions.strip():
        return facts
    instruction = re.sub(r"\bend with:.*", "", instructions, flags=re.I | re.S)
    terms = {word for word in re.findall(r"[a-z]{4,}", instruction.casefold())
             if word not in _REQUEST_STOP and word not in title_hint.casefold()}
    concepts = _requested_concepts(instructions, title_hint)
    def matched(claim: str) -> set[str]:
        return {key for key, spec in concepts.items() if _covers_request(claim, spec)}

    remaining = list(enumerate(facts))
    ranked = []
    covered: set[str] = set()
    while remaining:
        def score(pair: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
            index, fact = pair
            claim = fact["claim"]
            matches = matched(claim)
            lexical = len(terms & set(re.findall(r"[a-z]{4,}", claim.casefold())))
            primary = fact.get("subject_scope") == "main_subject"
            quality = (fact.get("phase") == "final") + (fact.get("confidence") == "high")
            return (int(primary), 5 * len(matches - covered) + 2 * len(matches) +
                    min(lexical, 5), quality, -index)
        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        claim = chosen[1]["claim"]
        covered.update(matched(claim))
        ranked.append(chosen[1])
    return ranked


_REQUEST_SPECIFIC_STOP = set((
    "explain describe show cover focus clearly distinguish compare differentiate "
    "what why how it is and the a an its to from into with about through "
    "end video episode project projects technical finished first local "
    "turn turns works work pipeline workflow"
).split())
_COVERAGE_SYSTEM = (
    "\n\nFocused requested-topic recovery. Return only facts explicitly supported "
    "by these supplied primary evidence excerpts for missing_requested_topics. "
    "Use the same exact quotation, subject-scope, and grounding rules as normal "
    "research. If a requested fact is absent, omit it. Do not infer capabilities "
    "from code identifiers or fill gaps from the user's instruction."
)


def _request_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {word[:-1] if word.endswith("s") and len(word) >= 6 else word
            for word in words if len(word) >= 3 and word not in _REQUEST_SPECIFIC_STOP}


def _requested_concepts(instructions: str, title_hint: str) -> dict[str, dict[str, Any]]:
    """Describe explicit topics without treating a request as evidence."""
    instruction = re.sub(r"\bend with:.*", "", instructions, flags=re.I | re.S)
    concepts: dict[str, dict[str, Any]] = {}
    if re.search(r"\b(?:what|overview|define|definition)\b", instruction, re.I):
        concepts["overview"] = {"kind": "overview"}
    if re.search(r"\b(?:why|purpose|motivation|reason|goal)\b", instruction, re.I):
        concepts["purpose"] = {"kind": "purpose"}
    if re.search(r"\b(?:how|workflow|pipeline|process|stages?)\b", instruction, re.I):
        how = re.search(r"\bhow\b([^.!?;\n]+)", instruction, re.I)
        focus = _request_words(how.group(1)) if how else set()
        focus -= _request_words(title_hint)
        concepts["workflow"] = {"kind": "workflow", "terms": sorted(focus)}
    distinction_count = 0
    for clause in re.split(r"[.!?;\n]", instruction):
        match = re.search(
            r"\b(?:distinguish|differentiate|contrast|compare)\s+(.+?)\s+"
            r"(?:from|with|versus|vs)\s+(.+)$", clause, re.I,
        )
        if match and distinction_count < 2:
            left, right = _request_words(match.group(1)), _request_words(match.group(2))
            if left and right:
                distinction_count += 1
                concepts[f"distinction-{distinction_count}"] = {
                    "kind": "distinction", "left": sorted(left), "right": sorted(right),
                }
    return concepts


def _covers_request(claim: str, spec: dict[str, Any]) -> bool:
    kind = spec["kind"]
    if kind == "purpose":
        return bool(_PURPOSE.search(claim))
    if kind == "overview":
        return bool(_OVERVIEW.search(claim))
    words = _request_words(claim)
    if kind == "distinction":
        # A generic overlap (e.g. "explainer production") must not imply
        # that a named production profile was actually covered. The compared
        # right-hand concept must be present in full.
        right = set(spec["right"]) - set(spec["left"])
        return bool(words & set(spec["left"]) and right and right <= words)
    terms = set(spec["terms"])
    overlap = words & terms
    return bool((_workflow_signal(claim) and
                 len(overlap) >= min(2, len(terms))) or
                (overlap and len(distinct_flow_actions(claim)) >= 2))


def _missing_requested_concepts(
    facts: list[dict[str, Any]], instructions: str, title_hint: str,
    roles: dict[str, str],
) -> dict[str, dict[str, Any]]:
    concepts = _requested_concepts(instructions, title_hint)
    primary = [fact for fact in facts if fact.get("evidence_refs") and
               all(roles.get(ref) == "primary" for ref in fact["evidence_refs"])]
    return {key: spec for key, spec in concepts.items() if not any(
        _covers_request(fact["claim"], spec) for fact in primary
    )}


def _requested_fact_groups(
    facts: list[dict[str, Any]], instructions: str, title_hint: str,
    roles: dict[str, str],
) -> dict[str, list[str]]:
    """Map supported requested topics to original main-subject facts in stable order.

    The request identifies topics, never evidence. Planner callers supply facts
    already checked against their exact source quotations and allowed inventory.
    """
    concepts = _requested_concepts(instructions, title_hint)
    groups: dict[str, list[str]] = {}
    for key, spec in concepts.items():
        ids = [fact["fact_id"] for fact in facts
               if isinstance(fact.get("fact_id"), str) and
               isinstance(fact.get("claim"), str) and
               fact.get("support") and fact.get("evidence_refs") and
               all(roles.get(ref) == "primary" for ref in fact["evidence_refs"]) and
               _covers_request(fact["claim"], spec)]
        if ids:
            groups[key] = list(dict.fromkeys(ids))
    if re.search(r"\b(?:planned|future|later|roadmap)\b", instructions, re.I):
        future = [fact["fact_id"] for fact in facts
                  if isinstance(fact.get("fact_id"), str) and fact.get("support") and
                  fact.get("phase") == "development" and
                  fact.get("evidence_refs") and
                  all(roles.get(ref) == "primary" for ref in fact["evidence_refs"]) and
                  re.search(r"\b(?:planned|future|later|yet|next|roadmap)\b",
                            str(fact.get("claim", "")), re.I)]
        if future:
            groups["future-work"] = list(dict.fromkeys(future))
    return groups


_MAX_WORKFLOW_SOURCE_SENTENCE = 480
_MAX_WORKFLOW_SOURCE_WINDOW = 720
_SOURCE_SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])\s+(?=\S)|\n(?=[A-Z]|[-*+] )",
)
_FOLLOWING_FLOW_SENTENCE = re.compile(
    r"^(?:these|those|this|they|it|their|then|next|the resulting)\b", re.I,
)


def _workflow_source_passage(
    excerpt: str, spec: dict[str, Any], *, strong_only: bool = False,
    min_length: int = 0, max_length: int | None = None,
) -> str | None:
    """Find the shortest exact local prose passage proving workflow relevance."""
    terms = set(spec["terms"])
    needed = min(2, len(terms))
    singles: list[str] = []
    windows: list[str] = []

    def eligible(length: int) -> bool:
        return length >= min_length and (max_length is None or length <= max_length)

    for paragraph in re.split(r"\n[ \t]*\n+", excerpt):
        prose = _declarative_text(paragraph)
        if not prose:
            continue
        sentences = [part.strip() for part in _SOURCE_SENTENCE_BREAK.split(prose)
                     if part.strip()]
        positions: list[tuple[int, int] | None] = []
        cursor = 0
        for sentence in sentences:
            start = paragraph.find(sentence, cursor)
            positions.append((start, start + len(sentence)) if start >= 0 else None)
            if start >= 0:
                cursor = start + len(sentence)
        for index, sentence in enumerate(sentences):
            if len(sentence) > _MAX_WORKFLOW_SOURCE_SENTENCE:
                continue
            overlap = _request_words(sentence) & terms
            signal = _workflow_signal(sentence)
            actions = distinct_flow_actions(sentence)
            strong = bool(overlap and len(actions) >= 2)
            ordinary = bool(signal and len(overlap) >= needed)
            if positions[index] and eligible(len(sentence)) and (
                strong or (ordinary and not strong_only)
            ):
                singles.append(sentence)
            if index + 1 == len(sentences):
                continue
            following = sentences[index + 1]
            if (len(sentence) + len(following) > _MAX_WORKFLOW_SOURCE_WINDOW or
                    not _FOLLOWING_FLOW_SENTENCE.match(following)):
                continue
            following_overlap = _request_words(following) & terms
            following_actions = distinct_flow_actions(following)
            strong_window = bool(len(actions | following_actions) >= 2 and
                                 ((actions and overlap) or
                                  (following_actions and following_overlap)))
            ordinary_window = bool(len(overlap | following_overlap) >= needed and
                                   ((signal and overlap) or
                                    (_workflow_signal(following) and following_overlap)))
            if not (strong_window or (ordinary_window and not strong_only)):
                continue
            first, second = positions[index], positions[index + 1]
            if first and second and not paragraph[first[1]:second[0]].strip():
                window = paragraph[first[0]:second[1]]
                if len(window) <= _MAX_WORKFLOW_SOURCE_WINDOW and eligible(len(window)):
                    windows.append(window)
    return min(singles, key=len) if singles else min(windows, key=len) if windows else None


def _workflow_source_match(excerpt: str, spec: dict[str, Any]) -> bool:
    """Match a topical workflow or strong operational flow in local prose."""
    return _workflow_source_passage(excerpt, spec) is not None


def _coverage_candidates(
    inventory: dict[str, Any], missing: dict[str, dict[str, Any]],
    title_hint: str, system: str, context_size: int,
    output_reserve_tokens: int, safety_tokens: int, instructions: str,
) -> list[dict[str, Any]]:
    """Choose at most one useful primary record per missing topic."""
    ranked = []
    title_key = re.sub(r"[^a-z0-9]", "", title_hint.casefold())
    doc_exts = {".md", ".rst", ".txt", ".adoc", ".html", ".htm"}
    for index, entry in enumerate(inventory["evidence"]):
        if (entry.get("evidence_role", "primary") != "primary" or
                entry.get("kind") != "document" or
                not isinstance(entry.get("excerpt"), str)):
            continue
        excerpt = entry["excerpt"]
        if not excerpt.strip() or len(excerpt) > MAX_COVERAGE_CHARS:
            continue
        prose = _declarative_text(excerpt)
        if not prose:
            continue
        named = bool(title_key and title_key in re.sub(r"[^a-z0-9]", "", prose.casefold()))
        matched = set()
        for key, spec in missing.items():
            if spec["kind"] in {"purpose", "overview"} and not named:
                continue
            if spec["kind"] == "workflow":
                if _workflow_source_match(excerpt, spec):
                    matched.add(key)
            elif _covers_request(prose, spec):
                matched.add(key)
        if not matched:
            continue
        path = Path(entry.get("relative_path") or "")
        score = (8 * len(matched) + 6 * (path.suffix.casefold() in doc_exts) +
                 4 * named + 2 * (path.name.casefold() == "readme.md") -
                 min(4, len(path.parts)))
        ranked.append((score, -index, entry, matched))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_refs: set[str] = set()
    covered: set[str] = set()
    for topic in missing:
        if topic in covered:
            continue
        for _score, _order, entry, matched in ranked:
            if topic not in matched or entry["ref"] in selected_refs:
                continue
            batch = selected + [entry]
            if (len(batch) > MAX_COVERAGE_RECORDS or
                    sum(len(item["excerpt"]) for item in batch) > MAX_COVERAGE_CHARS):
                continue
            payload = _coverage_payload(batch, missing, title_hint, instructions)
            if not fits_context(system, json.dumps(payload, ensure_ascii=False), context_size,
                                output_reserve_tokens, safety_tokens):
                continue
            selected.append(entry)
            selected_refs.add(entry["ref"])
            covered.update(matched)
            break
    return selected


def _coverage_payload(
    batch: list[dict[str, Any]], missing: dict[str, dict[str, Any]],
    title_hint: str, instructions: str,
) -> dict[str, Any]:
    return {**_payload(1, batch, title_hint, instructions),
            "research_mode": "requested_coverage", "coverage_contract": COVERAGE_CONTRACT,
            "missing_requested_topics": missing}


def _exact_workflow_fact(
    batch: list[dict[str, Any]], spec: dict[str, Any], roles: dict[str, str],
) -> dict[str, Any] | None:
    """Build one bounded verbatim workflow fact from selected primary prose."""
    for entry in batch:
        ref, excerpt = entry.get("ref"), entry.get("excerpt")
        if (entry.get("kind") != "document" or not isinstance(ref, str) or
                roles.get(ref) != "primary" or
                entry.get("evidence_role", "primary") != "primary" or
                not isinstance(excerpt, str)):
            continue
        passage = _workflow_source_passage(
            excerpt, spec, strong_only=True, min_length=12, max_length=320,
        )
        if passage is None or len(passage) > 360 or passage not in excerpt:
            continue
        return {
            "claim": passage,
            "evidence_refs": [ref],
            "subject_scope": "main_subject",
            "support": [{"evidence_ref": ref, "text": passage}],
            "phase": "unknown", "confidence": "high",
        }
    return None


def _fact_scope(refs: list[str], roles: dict[str, str]) -> str:
    """Derive subject scope from citations rather than a model-supplied label."""
    primary = any(roles.get(ref, "primary") == "primary" for ref in refs)
    supporting = any(roles.get(ref, "primary") != "primary" for ref in refs)
    return "mixed" if primary and supporting else "main_subject" if primary else "supporting_only"


def _starts_with_subject(claim: str, title_hint: str) -> bool:
    """Conservatively catch claims that explicitly assign a property to the main title."""
    title = re.sub(r"[^a-z0-9]", "", title_hint.casefold())
    if len(title) < 4:
        return False
    words = re.findall(r"[a-z0-9]+", claim.casefold())
    while words and words[0] in {"the", "a", "an", "this", "our"}:
        words.pop(0)
    prefix = ""
    for word in words[:5]:
        prefix += word
        if prefix == title:
            return True
        if len(prefix) >= len(title):
            break
    return False


_ORDER_CLAIM = re.compile(
    r"\b(?:then|before|after|followed\s+by|sequence|sequential|staged|"
    r"(?:normal|standard|mandatory)(?:\s+\w+){0,2}\s+(?:workflow|pipeline))\b|->|→",
    re.I,
)
_ORDER_SOURCE = re.compile(
    r"\b(?:then|before|after|followed\s+by|sequence|sequential|staged|"
    r"(?:normal|standard|mandatory)(?:\s+\w+){0,2}\s+(?:workflow|pipeline)|"
    r"ordered\s+steps|workflow\s+(?:runs|follows|begins|starts))\b|->|→",
    re.I,
)
def _workflow_signal(text: str) -> bool:
    """Recognize an operation sequence, never an unqualified process entity."""
    if _WORKFLOW.search(text) or _ORDER_SOURCE.search(text):
        return True
    return (len(distinct_flow_actions(text)) >= 2 and
            bool(re.search(r"\b(?:and|then|into)\b|,", text, re.I)))


_PREDICATE_STOPWORDS = {
    "about", "after", "also", "before", "from", "into", "their", "them",
    "there", "these", "this", "those", "under", "using", "when", "where",
    "which", "while", "with", "without", "would",
}


def _primary_supports_main_predicate(
    claim: str, support: list[dict[str, str]], roles: dict[str, str], title_hint: str,
) -> bool:
    title = re.sub(r"[^a-z0-9]", "", title_hint.casefold())
    predicate_words = {
        word for word in re.findall(r"[a-z0-9]+", claim.casefold())
        if len(word) >= 4 and word != title and word not in _PREDICATE_STOPWORDS
    }
    for item in support:
        if roles.get(item["evidence_ref"], "primary") != "primary":
            continue
        primary_words = set(re.findall(r"[a-z0-9]+", item["text"].casefold()))
        if predicate_words & primary_words:
            return True
    return False


def _reference_focus(title_hint: str, instructions: str, inventory: dict[str, Any]) -> bool:
    """Recognize an explicit request to make an embedded example the subject."""
    if not instructions.strip():
        return False
    title_key = re.sub(r"[^a-z0-9]", "", title_hint.casefold())
    targets = set()
    for entry in inventory["evidence"]:
        if entry.get("evidence_role", "primary") == "primary":
            continue
        parts = Path(entry.get("relative_path", "")).parts
        # Only the project immediately under an embedded root names a subject.
        # Deeper generic directories (assets/evidence, manifests, etc.) do not.
        for index, part in enumerate(parts[:-2]):
            if part.casefold() in _EMBEDDED_ROOTS:
                key = re.sub(r"[^a-z0-9]", "", parts[index + 1].casefold())
                if len(key) >= 4 and key != title_key:
                    targets.add(key)
                break

    focus_patterns = (
        r"\b(?:focus|center|centre)\s+(?:(?:the|this)\s+(?:episode|video|story)\s+)?"
        r"(?:(?:primarily|mainly)\s+)?(?:on|around)\s+(.+)",
        r"\b(?:primarily|mainly)\s+(?:explain|explore|cover)\s+(.+)",
        r"\bmake\s+(.+?)\s+the\s+(?:main|primary|central)\s+(?:subject|focus)\b",
        r"\bshowcase\s+(.+?)\s+as\s+(?:the\s+)?(?:main\s+)?subject\b",
    )
    for clause in re.split(r"[.!?;\n]", instructions.casefold()):
        for pattern in focus_patterns:
            for match in re.finditer(pattern, clause):
                # A later comparison or mention in the same sentence is not
                # the object of the focus request.
                subject = re.split(r",|\b(?:and|while|but|versus|vs|rather\s+than)\b",
                                   match.group(1), maxsplit=1)[0].strip()
                subject = re.sub(r"^(?:(?:the|a|an|our|your|this|that|its|included)\s+)+",
                                 "", subject)
                if re.match(
                    r"^(?:(?:embedded\s+)?(?:example|demo|sample|fixture|case study)\b|"
                    r"(?:embedded\s+)?reference\s+(?:project|case|example)\b|"
                    r"embedded\s+project\b)", subject
                ):
                    return True
                subject = re.sub(r"^(?:(?:embedded|project)\s+)+", "", subject)
                words = re.findall(r"[a-z0-9]+", subject)
                for target in targets:
                    prefix = ""
                    for word in words:
                        prefix += word
                        if prefix == target:
                            return True
                        if len(prefix) >= len(target):
                            break
    return False


def _current_overview_facts(
    facts: list[dict[str, Any]], inventory: dict[str, Any],
    title_hint: str, instructions: str, provider: LLMProvider,
    project_dir: Path, progress: Progress | None,
) -> list[dict[str, Any]]:
    """Recover current, verbatim primary claims missed by broad local research.

    A completed milestone is not necessarily the current state. For requests
    that explicitly ask for the present implementation, use only prose in the
    main project's top-level overview and keep exact source quotations.
    """
    if not re.search(r"\b(?:current|today|present|physically proven)\b", instructions, re.I):
        return facts
    overviews = []
    for entry in inventory["evidence"]:
        parts = Path(entry.get("relative_path") or "").parts
        if parts and re.fullmatch(r"source-\d+", parts[0]):
            parts = parts[1:]
        if (entry.get("evidence_role", "primary") == "primary" and
                entry.get("kind") == "document" and len(parts) == 1 and
                re.fullmatch(r"README(?:\.[a-z]{2})?\.(?:md|rst|txt|adoc)",
                             parts[0], re.I) and
                isinstance(entry.get("excerpt"), str) and
                title_hint.casefold().split()[0] in entry["excerpt"].casefold()):
            overviews.append(entry)
    if not overviews:
        return facts

    focus = re.split(r"\b(?:use only|clearly distinguish|use authentic)\b",
                     instructions, maxsplit=1, flags=re.I)[0]
    requested = _request_words(focus) - _request_words(title_hint)
    subject = _request_words(title_hint)
    name_key = re.sub(r"[^a-z0-9]", "", title_hint.casefold())
    name_stem = name_key[:-2] if name_key.endswith("os") and len(name_key) >= 8 else name_key
    existing = [_request_words(fact["claim"]) for fact in facts]
    candidates = []
    for entry in overviews[:4]:
        section = "overview"
        in_code = False
        paragraph: list[str] = []

        def finish() -> None:
            if not paragraph:
                return
            prose = "\n".join(paragraph).strip()
            paragraph.clear()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", prose)
            # Short adjacent sentences often form one self-contained source
            # statement (for example three native applications and their
            # separate roles). Keep that exact passage as another candidate.
            passages = ([prose] if 2 <= len(sentences) <= 3 else []) + sentences
            for sentence in passages:
                sentence = sentence.strip()
                if (not 35 <= len(sentence) <= 360 or
                        sentence[-1] not in ".!?" or
                        re.match(r"^(?:These|Those|This|It|They)\b", sentence) or
                        any(marker in sentence for marker in ("↓", "✅", "┌", "│")) or
                        not _declarative_text(sentence) or
                        sentence not in entry["excerpt"]):
                    continue
                # Keep the source and the proposed claim identical. This also
                # avoids converting a Markdown heading or code fragment into
                # an ungrounded statement.
                if any(marker in sentence for marker in ("**", "[", "](")):
                    continue
                words = _request_words(sentence) - subject
                if (not words or any(sentence == fact["claim"] or
                                     sentence.startswith(fact["claim"] + " ")
                                     for fact in facts) or
                        any(len(words & prior) / max(1, len(words | prior)) >= 0.8
                            for prior in existing)):
                    continue
                planned = bool(re.search(r"\b(?:later|future|planned|next|yet)\b", sentence, re.I))
                candidates.append((section, entry["ref"], sentence, words, planned))

        for line in entry["excerpt"].splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                finish()
                in_code = not in_code
            elif in_code:
                continue
            elif stripped.startswith("## "):
                finish()
                section = stripped[3:].strip().casefold()
            elif not stripped:
                finish()
            elif stripped.startswith(("#", "- ", "* ", ">", "|")):
                finish()
            else:
                paragraph.append(line)
        finish()

    selected = []
    covered: set[str] = set()
    sections: set[str] = set()
    seen = {fact["claim"].casefold() for fact in facts}
    while candidates and len(selected) < 8:
        ranked = []
        for index, (section, ref, claim, words, planned) in enumerate(candidates):
            overlap = requested & words
            identity = bool(name_stem and name_stem in re.sub(r"[^a-z0-9]", "", claim.casefold()))
            section_relevance = len(requested & _request_words(section))
            if not overlap and not identity and not planned and not section_relevance:
                continue
            similarity = max((len(words & prior) / max(1, len(words | prior))
                              for prior in (*existing, *(item[3] for item in selected))), default=0)
            score = (6 * len(overlap - covered) + 2 * len(overlap) +
                     4 * section_relevance + 5 * identity +
                     5 * int(section not in sections) +
                     2 * bool(re.search(r"\b(?:current|today|now|physical|proven)\b", claim, re.I)) +
                     6 * planned + 3 * int(claim.count(". ") >= 1) -
                     5 * int(len(claim) < 70 and not identity) - 12 * similarity)
            ranked.append((score, -index, index))
        if not ranked:
            break
        _score, _order, index = max(ranked)
        candidate = candidates.pop(index)
        section, ref, claim, words, planned = candidate
        if claim.casefold() in seen or any(
            claim in prior[2] or prior[2] in claim for prior in selected
        ):
            continue
        selected.append(candidate)
        seen.add(claim.casefold())
        covered.update(requested & words)
        sections.add(section)

    if not selected:
        return facts
    checks = [{"id": f"O{index:04d}", "claim": claim, "support": [claim]}
              for index, (_section, _ref, claim, _words, _planned) in enumerate(selected, 1)]
    verified = verify_claims(provider, checks, project_dir, "current-overview", progress)
    additions = [{"claim": claim, "evidence_refs": [ref],
                  "support": [{"evidence_ref": ref, "text": claim}],
                  "subject_scope": "main_subject", "phase": "development" if planned else "unknown",
                  "current_overview": True,
                  "confidence": "high"}
                 for index, (_section, ref, claim, _words, planned) in enumerate(selected, 1)
                 if f"O{index:04d}" in verified]
    if not additions:
        return facts

    current_milestone = max((int(value) for entry in overviews[:4]
                             for value in re.findall(r"\bM(\d+)\b", entry["excerpt"])),
                            default=0)
    affirmative = [_request_words(sentence) for entry in overviews[:4]
                   for sentence in re.split(r"(?<=[.!?])\s+", entry["excerpt"])
                   if re.search(r"\b(?:implemented|implements|supports|includes|runs)\b", sentence, re.I)
                   and not re.search(r"\b(?:does not|not yet)\b", sentence, re.I)]
    overview_text = "\n".join(entry["excerpt"] for entry in overviews[:4])
    overview_hashes = set(re.findall(r"\b[0-9a-f]{40}\b", overview_text, re.I))
    kept = []
    for fact in facts:
        claim = fact["claim"]
        stage = re.search(r"\b(?:M|Milestone\s+)(\d+)\b", claim, re.I)
        historical = (stage and int(stage.group(1)) < current_milestone and
                      not re.search(r"\b(?:physical|physically|proven|acceptance)\b", claim, re.I) and
                      not re.search(rf"\b(?:M|Milestone\s+){stage.group(1)}\b", instructions, re.I))
        negative = re.search(r"\bdoes not support\b", claim, re.I)
        contradicted = negative and any(
            len((_request_words(claim[negative.end():]) - subject) & words) >= 2
            for words in affirmative
        )
        claimed_hashes = set(re.findall(r"\b[0-9a-f]{40}\b", claim, re.I))
        superseded_freeze = (claimed_hashes and not claimed_hashes & overview_hashes and
                             overview_hashes and re.search(r"\b(?:frozen|freeze)\b", claim, re.I) and
                             re.search(r"\b(?:frozen|freeze)\b", overview_text, re.I))
        if not historical and not contradicted and not superseded_freeze:
            kept.append(fact)
    return additions + kept


def _consolidate(
    facts: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    inventory: dict[str, Any],
    title_hint: str,
    instructions: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap supporting facts and distinct refs at half the primary counts (min 1)."""
    roles = {e["ref"]: e.get("evidence_role", "primary") for e in inventory["evidence"]}
    rank = {"primary": 0, "embedded_reference": 1, "generated_artifact": 2}
    def role(ref: str) -> str:
        value = roles.get(ref, "primary")
        return value if value in rank else "primary"
    fact_role = lambda fact: max((role(ref) for ref in fact["evidence_refs"]), key=lambda r: rank.get(r, 0))
    primary = [fact for fact in facts if fact_role(fact) == "primary"]
    # When the request really targets an embedded project, or there is no
    # primary research to form a backbone, do not impose the default quota.
    if not primary or _reference_focus(title_hint, instructions, inventory):
        return facts, assets

    primary_refs = {ref for fact in primary for ref in fact["evidence_refs"]}
    fact_limit = max(1, len(primary) // 2)
    ref_limit = max(1, len(primary_refs) // 2)
    groups = {
        secondary_role: [f for f in facts if fact_role(f) == secondary_role]
        for secondary_role in ("embedded_reference", "generated_artifact")
    }
    selected: list[dict[str, Any]] = []
    supporting_refs: set[str] = set()
    seen_claims = {f["claim"].casefold() for f in primary}

    def select(fact: dict[str, Any], secondary_role: str) -> None:
        if len(selected) >= fact_limit or fact["claim"].casefold() in seen_claims:
            return
        # A compound fact can need every cited source. Never discard one of
        # its refs while leaving a claim or quoted support behind.
        secondary = {ref for ref in fact["evidence_refs"] if role(ref) != "primary"}
        if not any(role(ref) == secondary_role for ref in secondary) or len(
            supporting_refs | secondary
        ) > ref_limit:
            return
        selected.append(fact)
        supporting_refs.update(secondary)
        seen_claims.add(fact["claim"].casefold())

    embedded = groups["embedded_reference"]
    generated = groups["generated_artifact"]
    # With room for two examples, reserve one place for each kind of proof.
    if embedded and generated and fact_limit >= 2 and ref_limit >= 2:
        select(embedded[0], "embedded_reference")
        select(generated[0], "generated_artifact")
    for secondary_role, candidates in groups.items():
        for fact in candidates:
            select(fact, secondary_role)

    kept_assets = [asset for asset in assets if role(asset["evidence_ref"]) == "primary"]
    seen_asset_refs: set[str] = set()
    for secondary_role in ("embedded_reference", "generated_artifact"):
        for asset in assets:
            ref = asset["evidence_ref"]
            if role(ref) != secondary_role or ref in seen_asset_refs:
                continue
            if ref not in supporting_refs and len(supporting_refs) >= ref_limit:
                continue
            kept_assets.append(asset)
            supporting_refs.add(ref)
            seen_asset_refs.add(ref)
    return primary + selected, kept_assets


def research(
    provider: LLMProvider,
    inventory: dict[str, Any],
    project_dir: Path,
    max_chars: int = 45000,
    context_size: int = 32768,
    output_reserve_tokens: int = 4096,
    safety_tokens: int = 1024,
    max_retries: int = 2,
    title_hint: str = "",
    instructions: str = "",
    progress: Progress | None = None,
) -> dict[str, Any]:
    system = (project_dir.parents[1] / "prompts" / "research.txt").read_text()
    valid = {e["ref"] for e in inventory["evidence"]}
    roles = {e["ref"]: e.get("evidence_role", "primary") for e in inventory["evidence"]}
    reference_focus = _reference_focus(title_hint, instructions, inventory)

    def normalize(result: dict[str, Any], batch: list[dict[str, Any]]) -> dict[str, Any]:
        # A ref must be in this exact request, including after a failed part
        # splits into children. Global validity alone does not prove provenance.
        supplied = valid & {entry["ref"] for entry in batch}
        by_ref = {entry["ref"]: entry for entry in batch if entry["ref"] in supplied}
        candidates = []
        candidate_count: dict[str, int] = {}
        returned_facts = result.get("facts", [])
        if instructions.strip() and isinstance(returned_facts, list):
            # Rank a finite candidate pool before the existing 24-candidate
            # and per-ref limits can discard a requested purpose or workflow.
            scoped = []
            for fact in returned_facts[:MAX_RANKED_CANDIDATES]:
                if not isinstance(fact, dict) or not isinstance(fact.get("claim"), str):
                    continue
                refs = fact.get("evidence_refs")
                scoped.append({**fact, "subject_scope": _fact_scope(
                    refs if isinstance(refs, list) else [], roles)})
            returned_facts = _rank_requested_facts(scoped, instructions, title_hint)
        for fact in returned_facts if isinstance(returned_facts, list) else []:
            if len(candidates) >= 2 * MAX_FACTS_PER_REQUEST:
                break
            if (not isinstance(fact, dict) or not isinstance(fact.get("claim"), str) or
                    not fact["claim"].strip() or len(fact["claim"]) > 360):
                continue
            raw_refs = fact.get("evidence_refs")
            if not isinstance(raw_refs, list):
                continue
            refs = [r for r in raw_refs if isinstance(r, str) and r in supplied]
            if not refs:
                continue
            if any(candidate_count.get(ref, 0) >= 2 * MAX_FACTS_PER_REF for ref in set(refs)):
                continue
            claim = str(fact["claim"]).strip()
            scope = _fact_scope(refs, roles)
            # A generated example cannot by itself confer a capability or
            # guarantee on the requested project. Keep claims about the
            # example itself as supporting evidence.
            if (scope == "supporting_only" and
                    not reference_focus and
                    _starts_with_subject(claim, title_hint)):
                continue

            support = []
            invalid_support = False
            raw_support = fact.get("support", [])
            if not isinstance(raw_support, list):
                continue
            for item in raw_support:
                if not isinstance(item, dict):
                    invalid_support = True
                    break
                ref, span = item.get("evidence_ref"), item.get("text")
                if (ref not in refs or not isinstance(span, str) or
                        not 12 <= len(span) <= MAX_SUPPORT_CHARS or
                        span not in (by_ref[ref].get("excerpt") or "")):
                    invalid_support = True
                    break
                support.append({"evidence_ref": ref, "text": span})
            if invalid_support or not support or any(
                not any(item["evidence_ref"] == ref for item in support) for ref in refs
            ):
                continue
            if (scope == "mixed" and _starts_with_subject(claim, title_hint) and
                    not _primary_supports_main_predicate(claim, support, roles, title_hint)):
                continue
            # An enumeration of commands is not evidence of their order.
            # Ordered/normal-flow claims need a verbatim batch-local passage
            # explicitly describing that relationship.
            if _ORDER_CLAIM.search(claim) and not any(
                _ORDER_SOURCE.search(item["text"]) for item in support
            ):
                continue
            clean = {
                "claim": claim,
                "evidence_refs": list(dict.fromkeys(refs)),
                "subject_scope": scope,
                "support": _dedupe(support, lambda item: (item["evidence_ref"], item["text"])),
                "phase": (fact["phase"] if isinstance(fact.get("phase"), str) and fact["phase"] in
                          {"development", "final", "background", "unknown"} else "unknown"),
                "confidence": (fact["confidence"] if isinstance(fact.get("confidence"), str) and fact["confidence"] in
                               {"high", "medium", "low"} else "low"),
            }
            candidates.append(clean)
            for ref in set(refs):
                candidate_count[ref] = candidate_count.get(ref, 0) + 1

        verdicts = verify_claims(provider, [{
            "id": f"C{index:04d}", "claim": candidate["claim"],
            "support": [item["text"] for item in candidate["support"]],
        } for index, candidate in enumerate(candidates, 1)], project_dir, "research", progress)
        facts = []
        facts_per_ref: dict[str, int] = {}
        verified = [candidate for index, candidate in enumerate(candidates, 1)
                    if f"C{index:04d}" in verdicts]
        for candidate in _rank_requested_facts(verified, instructions, title_hint):
            refs = set(candidate["evidence_refs"])
            if (len(facts) >= MAX_FACTS_PER_REQUEST or any(
                facts_per_ref.get(ref, 0) >= MAX_FACTS_PER_REF for ref in refs
            )):
                continue
            facts.append(candidate)
            for ref in refs:
                facts_per_ref[ref] = facts_per_ref.get(ref, 0) + 1

        assets = []
        assets_per_ref: dict[str, int] = {}
        returned_assets = result.get("assets", [])
        for asset in returned_assets if isinstance(returned_assets, list) else []:
            if len(assets) >= MAX_ASSETS_PER_REQUEST:
                break
            if (not isinstance(asset, dict) or not isinstance(asset.get("evidence_ref"), str) or
                    asset["evidence_ref"] not in supplied or
                    assets_per_ref.get(asset["evidence_ref"], 0) >= MAX_ASSETS_PER_REF):
                continue
            clean = dict(asset)
            clean["purpose"] = str(clean.get("purpose", "")).strip()
            assets.append(clean)
            ref = asset["evidence_ref"]
            assets_per_ref[ref] = assets_per_ref.get(ref, 0) + 1
        return {"facts": facts, "assets": assets}

    make_payload = lambda part, batch: _payload(part, batch, title_hint, instructions)
    # Keep inventory and refs intact; process core evidence before supporting
    # examples and repetitive generated records when creating research facts.
    role_order = {"primary": 0, "embedded_reference": 1, "generated_artifact": 2}
    ordered_evidence = sorted(
        inventory["evidence"],
        key=lambda entry: role_order.get(entry.get("evidence_role", "primary"), 0),
    )
    chunks = split_for_context(
        ordered_evidence,
        system,
        make_payload,
        context_size,
        output_reserve_tokens,
        safety_tokens,
        max_chars=max_chars,
    )

    part_dir = project_dir / "manifests" / "research-parts"
    allfacts: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    used: set[Path] = set()

    for i, batch in enumerate(chunks, 1):
        checkpoint = part_dir / f"part-{i:03d}.json"
        results = checkpointed_split_json(
            provider, system, batch, lambda items: make_payload(i, items),
            checkpoint, normalize,
            max_retries=max_retries, progress=progress,
            label=f"Research batch {i}/{len(chunks)}", used=used,
            split_single=_split_research_document,
        )
        for result in results:
            allfacts.extend(result["facts"])
            assets.extend(result["assets"])

    if part_dir.exists():
        for stale in part_dir.glob("part-*.json"):
            if stale not in used:
                stale.unlink()

    allfacts = _dedupe(
        allfacts,
        lambda f: (
            f["claim"].casefold(),
            tuple(f["evidence_refs"]),
            f.get("phase", "unknown"),
        ),
    )
    assets = _dedupe(
        assets,
        lambda a: (
            a.get("evidence_ref"),
            str(a.get("purpose", "")).casefold(),
        ),
    )

    allfacts = _limit_by_ref(_rank_requested_facts(allfacts, instructions, title_hint),
                             lambda fact: fact["evidence_refs"], MAX_FACTS_PER_REF)
    assets = _limit_by_ref(assets, lambda asset: [asset["evidence_ref"]], MAX_ASSETS_PER_REF)

    allfacts, assets = _consolidate(allfacts, assets, inventory, title_hint, instructions)

    missing = _missing_requested_concepts(allfacts, instructions, title_hint, roles)
    if missing:
        coverage_system = system + _COVERAGE_SYSTEM
        batch = _coverage_candidates(
            inventory, missing, title_hint, coverage_system, context_size,
            output_reserve_tokens, safety_tokens, instructions,
        )
        if batch:
            payload = _coverage_payload(batch, missing, title_hint, instructions)
            checkpoint = project_dir / "manifests" / "research-coverage" / "part-001.json"

            def normalize_coverage(value: dict[str, Any]) -> dict[str, Any]:
                canonical = normalize(value, batch)
                remaining = _missing_requested_concepts(
                    allfacts + canonical["facts"], instructions, title_hint, roles,
                )
                if "workflow" not in remaining:
                    return canonical
                exact = _exact_workflow_fact(batch, remaining["workflow"], roles)
                if exact is None or "C0001" not in verify_claims(provider, [{
                    "id": "C0001", "claim": exact["claim"],
                    "support": [exact["support"][0]["text"]],
                }], project_dir, "research", progress):
                    return canonical
                ranked = _rank_requested_facts(
                    canonical["facts"] + [exact], instructions, title_hint,
                )
                canonical["facts"] = _limit_by_ref(
                    ranked, lambda fact: fact["evidence_refs"], MAX_FACTS_PER_REF,
                )[:MAX_FACTS_PER_REQUEST]
                return canonical

            with step(progress, "Recovering requested coverage"):
                recovered = checkpointed_complete_json(
                    provider, coverage_system, payload, checkpoint,
                    normalize_coverage, max_retries=max_retries,
                )["facts"]
            if recovered:
                allfacts = _dedupe(allfacts + recovered, lambda fact: (
                    fact["claim"].casefold(), tuple(fact["evidence_refs"]),
                    fact.get("phase", "unknown"),
                ))
                allfacts = _limit_by_ref(
                    _rank_requested_facts(allfacts, instructions, title_hint),
                    lambda fact: fact["evidence_refs"], MAX_FACTS_PER_REF,
                )
                allfacts, assets = _consolidate(allfacts, assets, inventory,
                                                title_hint, instructions)

    allfacts = _current_overview_facts(allfacts, inventory, title_hint, instructions,
                                       provider, project_dir, progress)
    allfacts, assets = _consolidate(allfacts, assets, inventory, title_hint, instructions)
    out = {"version": 1, "facts": allfacts, "assets": assets}
    json_dump(project_dir / "manifests" / "research.json", out)
    return out
