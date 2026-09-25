from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .chunking import checkpointed_split_json, split_for_context
from .grounding import GROUNDING_CONTRACT, verify_claims
from .inventory import _EMBEDDED_ROOTS
from .progress import Progress
from .providers import LLMProvider
from .util import json_dump


MAX_FACTS_PER_REF = 6
MAX_FACTS_PER_REQUEST = 12
MAX_ASSETS_PER_REF = 2
MAX_ASSETS_PER_REQUEST = 6
MAX_RANKED_CANDIDATES = 128


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
        },
        "grounding_contract": GROUNDING_CONTRACT,
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
_WORKFLOW = re.compile(r"\b(?:pipeline|workflow|stages?|process|transform|turns?)\b", re.I)
_OVERVIEW = re.compile(r"\b(?:is an?|engine|tool|system|framework|application)\b", re.I)


def _rank_requested_facts(
    facts: list[dict[str, Any]], instructions: str, title_hint: str,
) -> list[dict[str, Any]]:
    """Reserve bounded output for distinct, explicitly requested grounded topics."""
    if not instructions.strip():
        return facts
    instruction = re.sub(r"\bend with:.*", "", instructions, flags=re.I | re.S)
    terms = {word for word in re.findall(r"[a-z]{4,}", instruction.casefold())
             if word not in _REQUEST_STOP and word not in title_hint.casefold()}
    requested = set()
    if re.search(r"\b(?:why|purpose|motivation|reason|goal)\b", instruction, re.I):
        requested.add("purpose")
    if re.search(r"\b(?:how|workflow|pipeline|process|stages?)\b", instruction, re.I):
        requested.add("workflow")
    if re.search(r"\b(?:what|overview|define|definition)\b", instruction, re.I):
        requested.add("overview")
    remaining = list(enumerate(facts))
    ranked = []
    covered: set[str] = set()
    while remaining:
        def score(pair: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
            index, fact = pair
            claim = fact["claim"]
            matches = ({"purpose"} if _PURPOSE.search(claim) else set()) | (
                {"workflow"} if _WORKFLOW.search(claim) else set()) | (
                {"overview"} if _OVERVIEW.search(claim) else set())
            matches &= requested
            lexical = len(terms & set(re.findall(r"[a-z]{4,}", claim.casefold())))
            primary = fact.get("subject_scope") == "main_subject"
            quality = (fact.get("phase") == "final") + (fact.get("confidence") == "high")
            return (int(primary), 5 * len(matches - covered) + 2 * len(matches) +
                    min(lexical, 5), quality, -index)
        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        claim = chosen[1]["claim"]
        covered.update(({"purpose"} if _PURPOSE.search(claim) else set()) |
                       ({"workflow"} if _WORKFLOW.search(claim) else set()) |
                       ({"overview"} if _OVERVIEW.search(claim) else set()))
        ranked.append(chosen[1])
    return ranked


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
                        not 12 <= len(span) <= 320 or
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

    out = {"version": 1, "facts": allfacts, "assets": assets}
    json_dump(project_dir / "manifests" / "research.json", out)
    return out
