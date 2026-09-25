"""Bounded, fail-closed checks between quoted evidence and generated claims."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .chunking import checkpointed_split_json
from .progress import Progress
from .providers import LLMProvider, StructuredOutputError


# Included in research and storyboard request hashes: old normalized checkpoints
# must not bypass a newly strengthened provenance contract.
GROUNDING_CONTRACT = "mapped-support-kind-v2"
VERIFIER_BATCH_SIZE = 12
VERIFIER_REQUEST_MAX_CHARS = 12000
MAX_PROPOSITIONS = 16


class GroundingError(RuntimeError):
    """A storyboard remained unsupported after bounded generation attempts."""

_LITERAL = re.compile(
    r"https?://[^\s`\"'<>]+|(?:[\w.-]+/)+[\w.-]+|"
    r"\b[\w-]+\.(?:[a-zA-Z][a-zA-Z0-9]{1,7})\b|"
    r"\b\d+(?:\.\d+)*\b|\b[A-Z]{2,}[A-Z0-9]*\b|"
    r"\b(?:[A-Z][a-z0-9]+){2,}\b"
)
_BACKTICK = re.compile(r"`([^`]+)`")
_NAMED = re.compile(r"\b[A-Z][a-z][a-z0-9]+\b")
_INITIAL_WORDS = {"The", "This", "That", "These", "Those", "An", "And", "But",
                  "In", "On", "For", "When", "If", "As", "At", "By", "From",
                  "With", "Without", "No", "Not", "Only", "Most", "Each", "Some",
                  "Then", "First", "After", "Before", "New", "Existing"}
_CAUSE = re.compile(
    r"\b(?:ensures?|because|therefore|thus|enables?|allows?|avoids?|"
    r"prevents?|guarantees?|keeps?|leads? to|as a result|in order to|so that)\b", re.I
)
_EXCLUSIVE = re.compile(r"\b(?:only|solely|exclusively|always|never|every)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:not|no|never|without|excludes?|doesn't|isn't|cannot)\b", re.I)
_STOP = set((
    "a an and are as at be been by can do does for from has have in into is it "
    "its of on or our that the their there these this those to was were what when "
    "which while who with within you your it s they them project source document "
    "then one first second says includes via using use" 
).split())

_CODE_LINE = re.compile(
    r"^(?:from\s+\S+\s+import\s+|import\s+|(?:async\s+)?def\s+|class\s+|"
    r"(?:if|elif|else|for|while|return|raise|assert|try|except|with)\b|"
    r"@\w|\$\s|[\w.\[\]]+\s*(?:\([^)]*\)\s*)?=\s*|"
    r"[\w.]+\s*\(|[{}\[\]]\s*$)", re.I,
)
_SYNTAX_CLAIM = re.compile(
    r"^(?:(?:the|this|a)\s+)?(?:file|module|code|script|function\s+[\w.]+|"
    r"class\s+[\w.]+|command|config(?:uration)?)\s+"
    r"(imports?|calls?|invokes?|passes?|assigns?|sets?|returns?)\s+(.+)$", re.I,
)
_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|does|do|uses?|creates?|produces?|"
    r"calls?|imports?|runs?|installs?|supports?|includes?|maps?|defines?|"
    r"provides?|allows?|enables?|turns?|transforms?|builds?|generates?|"
    r"describes?|records?|documents?|applies?|produces?|remains?|exists?|depends?|validates?)\b", re.I,
)
_LIST_START = re.compile(
    r"\b(?:by|with|using|including|include|includes|supports|contains|"
    r"offers|provides|features|uses|consists\s+of|"
    r"configured\s+(?:by|with|for)|configurable\s+(?:by|with|for)|"
    r"such\s+as|excludes|requires|"
    r"(?:stages?|backends?|packages?|dependencies|exclusions?|capabilities|"
    r"versions?|paths?|commands?|guarantees?|options?|properties)\s+"
    r"(?:are|include|includes|:))\s+(.+)$", re.I,
)


def _declarative_text(span: str) -> str:
    """Keep explicit prose, comments and docstrings; never promote code identifiers."""
    lines = []
    for line in span.splitlines():
        stripped = line.strip().strip('`')
        if not stripped:
            continue
        if stripped.startswith(('"""', "'''")):
            stripped = stripped.strip('"\' ')
        elif stripped.startswith(('# ', '// ')):
            stripped = stripped.lstrip('#/ ')
        elif stripped.startswith('assert '):
            # An explicit assertion message is prose; the assertion expression is code.
            message = re.search(r",\s*(['\"])([^'\"]{12,})\1\s*$", stripped)
            stripped = message.group(2) if message else ''
        elif _CODE_LINE.match(stripped) or re.match(r"^(?:```|[\w.]+\s*:\s*\S)", stripped):
            continue
        if len(re.findall(r"[A-Za-z]+", stripped)) >= 3 and (
            _VERB.search(stripped) or re.match(r"^[A-Z][^\n]{2,80}:\s*\S", stripped)
        ):
            lines.append(stripped)
    return "\n".join(lines)


def _syntactic_claim(claim: str, spans: list[str]) -> bool:
    """Only directly visible operations may be inferred from executable syntax."""
    match = _SYNTAX_CLAIM.fullmatch(claim.strip().rstrip('.'))
    if not match:
        return False
    operation, object_text = match.groups()
    if operation.lower().startswith(('import', 'call', 'invoke')) and not re.fullmatch(
        r"[\w.]+(?:\([^()]*\))?(?:,\s*[\w.]+)*", object_text
    ):
        return False
    code = "\n".join(spans)
    if not _CODE_LINE.search(code) and not any(_CODE_LINE.match(line.strip()) for line in code.splitlines()):
        return False
    if operation.lower().startswith('import'):
        return bool(re.search(r"\bimport\s+", code) and
                    any(word in code for word in re.findall(r"[\w.]+", object_text)))
    if operation.lower().startswith(('call', 'invoke')):
        return any(re.search(r"\b" + re.escape(word) + r"\s*\(", code)
                   for word in re.findall(r"[\w.]+", object_text))
    if operation.lower().startswith('return'):
        return bool(re.search(r"\breturn\s+", code))
    if operation.lower().startswith(('assign', 'set')):
        return bool('=' in code and any(word in code for word in
                    re.findall(r"[\w.]+", object_text)))
    if operation.lower().startswith('pass'):
        return bool(re.search(r"--[\w-]+", object_text) and
                    any(flag in code for flag in re.findall(r"--[\w-]+", object_text)))
    return False


def _enumeration_members(claim: str) -> list[str]:
    """Extract explicitly listed items so no list member can be supplied by inference."""
    members = []
    for sentence in re.split(r"[.;!?]", claim):
        match = _LIST_START.search(sentence)
        if not match or ',' not in match.group(1):
            continue
        listed = re.split(r",\s*|\s+\b(?:and|or)\b\s+", match.group(1))
        for fragment in listed:
            item = re.sub(r"^(?:and|or|the|a|an)\s+", "", fragment.strip(), flags=re.I)
            if item and len(item.split()) <= 5 and not _VERB.search(item):
                members.append(item)
    return list(dict.fromkeys(members))


def _member_present(member: str, spans: list[str]) -> bool:
    normalized = re.sub(r"[_-]+", " ", " ".join(spans)).casefold()
    phrase = re.sub(r"[_-]+", " ", member).casefold()
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized))


def _propositions(claim: str) -> list[str]:
    """Provide a deterministic, bounded list of sentences and enumerated members."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", claim.strip())
                 if part.strip()]
    units = []
    for sentence in sentences:
        units.append(sentence)
        # Two independently asserted clauses require two separate mappings.
        clauses = re.split(r"\s+\b(?:and|but|while)\b\s+", sentence)
        if len(clauses) > 1 and all(_VERB.search(clause) for clause in clauses):
            units.extend(clause.strip() for clause in clauses)
    units.extend(_enumeration_members(claim))
    return list(dict.fromkeys(units))


def _support_texts(item: dict[str, Any]) -> list[str]:
    if "facts" in item:
        return [fact["claim"] for fact in item["facts"]]
    return item.get("support", [])


def _words(value: str) -> set[str]:
    def stem(word: str) -> str:
        if word.endswith("ing") and len(word) >= 7:
            return word[:-3]
        if word.endswith("ed") and len(word) >= 6:
            return word[:-2]
        if word.endswith("s") and len(word) >= 6:
            return word[:-1]
        return word
    return {stem(word) for word in re.findall(r"[a-z0-9]+", value.casefold())
            if len(word) > 2 and word not in _STOP}


def deterministic_decision(claim: str, support: list[str]) -> str:
    """Reject obvious expansions; accept exact quotes; verify other paraphrases."""
    if not isinstance(claim, str) or not claim.strip() or not support or any(
        not isinstance(span, str) or not span.strip() for span in support
    ):
        return "reject"
    if any(not _member_present(member, support) for member in _enumeration_members(claim)):
        return "reject"
    if any(claim.strip() == span.strip() for span in support):
        return "accept"  # Verbatim source syntax or prose, with no inferred meaning.
    prose = [_declarative_text(span) for span in support]
    if not any(prose) and not _syntactic_claim(claim, support) and not any(
        claim.strip().rstrip(" .;:!?") == span.strip().rstrip(" .;:!?") for span in support
    ):
        return "reject"
    if any(prose) and not _syntactic_claim(claim, support):
        # A code fragment in a mixed quotation cannot silently provide the
        # meaning missing from the explicit prose in that quotation.
        support = [span for span in prose if span]
    quoted = "\n".join(support)
    folded = quoted.casefold()
    for literal in [*_BACKTICK.findall(claim), *_LITERAL.findall(claim)]:
        if literal.casefold() not in folded:
            return "reject"
    for name in _NAMED.finditer(claim):
        if name.group() not in _INITIAL_WORDS and name.group().casefold() not in folded:
            return "reject"
    if _CAUSE.search(claim) and not _CAUSE.search(quoted):
        return "reject"
    if _EXCLUSIVE.search(claim) and not _EXCLUSIVE.search(quoted):
        return "reject"
    if _NEGATIVE.search(claim) and not _NEGATIVE.search(quoted):
        return "reject"
    claim_words, support_words = _words(claim), _words(quoted)
    if not _syntactic_claim(claim, support) and len(claim_words) >= 4 and len(
        claim_words & support_words
    ) < max(
        2, (2 * len(claim_words) + 2) // 3
    ):
        return "reject"
    stripped = claim.strip().rstrip(" .;:!?")
    at = folded.find(stripped.casefold())
    if at >= 0:
        before, after = folded[:at], folded[at + len(stripped):]
        prefix = re.split(r"[.!?;\n]", before)[-1]
        suffix = re.split(r"[.!?;\n]", after)[0]
        if re.search(r"\b(?:if|when|unless|only|except|during)\b", prefix) or re.search(
            r"\b(?:if|when|unless|only|except|during|under|from|in|for|via)\b", suffix
        ):
            return "verify"
        return "accept"
    return "verify"


def verify_claims(
    provider: LLMProvider, items: list[dict[str, Any]], project_dir: Path,
    stage: str, progress: Progress | None = None,
) -> set[str]:
    """Accept deterministic quotes or exact-input checkpointed semantic verdicts.

    The verifier receives only the proposition and the specific attached
    support. Missing, malformed or unavailable decisions never authorize facts.
    """
    accepted: set[str] = set()
    pending: list[dict[str, Any]] = []
    for item in items:
        if "facts" in item and item["claim"].strip() == " ".join(
            fact["claim"] for fact in item["facts"]
        ) and all(fact.get("support") for fact in item["facts"]):
            accepted.add(item["id"])
            continue
        texts = _support_texts(item)
        if len(_propositions(item["claim"])) > MAX_PROPOSITIONS:
            continue
        decision = deterministic_decision(item["claim"], texts)
        if decision == "accept":
            accepted.add(item["id"])
        elif decision == "verify":
            pending.append({**item, "required_propositions": _propositions(item["claim"])})
    if not pending:
        return accepted

    system = (Path(__file__).resolve().parents[1] / "prompts" / "grounding.txt").read_text()
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    size = 0
    for item in pending:
        item_size = len(json.dumps(item, ensure_ascii=False))
        if item_size > VERIFIER_REQUEST_MAX_CHARS:
            continue  # Too much context to verify safely: fail closed.
        if batch and (len(batch) >= VERIFIER_BATCH_SIZE or
                      size + item_size > VERIFIER_REQUEST_MAX_CHARS):
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += item_size
    if batch:
        batches.append(batch)
    for batch in batches:
        digest = hashlib.sha256((system + json.dumps(batch, ensure_ascii=False,
                                                        sort_keys=True)).encode()).hexdigest()
        checkpoint = (project_dir / "manifests" / "grounding-parts" /
                      f"{stage}-{digest[:20]}.json")

        def normalize(value: dict[str, Any], part: list[dict[str, Any]]) -> dict[str, Any]:
            decisions = value.get("decisions") if isinstance(value, dict) else None
            expected = {item["id"]: item for item in part}
            if not isinstance(decisions, list) or len(decisions) != len(part) or any(
                not isinstance(row, dict) or row.get("id") not in expected or
                type(row.get("supported")) is not bool for row in decisions
            ) or {row["id"] for row in decisions} != set(expected):
                raise StructuredOutputError("Grounding verifier returned incomplete decisions")
            normalized = []
            for row in decisions:
                item = expected[row["id"]]
                required = item["required_propositions"]
                sources = _support_texts(item)
                mapping = row.get("propositions")
                valid = isinstance(mapping, list) and len(mapping) == len(required)
                if valid:
                    observed = []
                    for mapped in mapping:
                        if not isinstance(mapped, dict) or not isinstance(mapped.get("text"), str):
                            valid = False
                            break
                        indices = mapped.get("support_indices")
                        if (not isinstance(indices, list) or not indices or
                            any(type(index) is not int or index < 0 or index >= len(sources)
                                for index in indices) or len(set(indices)) != len(indices)):
                            valid = False
                            break
                        proposition = mapped["text"]
                        observed.append(proposition)
                        selected = [sources[index] for index in indices]
                        if deterministic_decision(proposition, selected) == "reject":
                            valid = False
                            break
                        if "facts" in item and any(
                            not any(_declarative_text(span["text"])
                                    for span in item["facts"][index]["support"]) and
                            deterministic_decision(
                                proposition,
                                [span["text"] for span in item["facts"][index]["support"]],
                            ) == "reject" for index in indices
                        ):
                            valid = False
                            break
                    if observed != required:
                        valid = False
                normalized.append({"id": row["id"],
                                   "supported": row["supported"] is True and valid,
                                   "propositions": mapping if valid else []})
            return {"decisions": normalized}

        try:
            results = checkpointed_split_json(
                provider, system, batch, lambda part: {"checks": part}, checkpoint,
                normalize, max_retries=1, progress=progress,
                label=f"Checking {stage} grounding",
            )
        except Exception:
            # Never interpret a transport, parsing or output-limit failure as
            # affirmative evidence. A later identical run can retry the check.
            if progress is not None:
                progress.note(f"Grounding check unavailable; omitting unverified {stage} claims")
            continue
        accepted.update(row["id"] for result in results for row in result["decisions"]
                        if row["supported"] is True)
    return accepted
