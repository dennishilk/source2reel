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
GROUNDING_CONTRACT = "quoted-proposition-v1"
VERIFIER_BATCH_SIZE = 12
VERIFIER_REQUEST_MAX_CHARS = 12000


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
    if len(claim_words) >= 4 and len(claim_words & support_words) < max(2, len(claim_words) // 3):
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
        ):
            accepted.add(item["id"])
            continue
        texts = ([fact["claim"] for fact in item["facts"]]
                 if "facts" in item else item.get("support", []))
        decision = deterministic_decision(item["claim"], texts)
        if decision == "accept":
            accepted.add(item["id"])
        elif decision == "verify":
            pending.append(item)
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
            expected = {item["id"] for item in part}
            if not isinstance(decisions, list) or len(decisions) != len(part) or any(
                not isinstance(row, dict) or row.get("id") not in expected or
                type(row.get("supported")) is not bool for row in decisions
            ) or {row["id"] for row in decisions} != expected:
                raise StructuredOutputError("Grounding verifier returned incomplete decisions")
            return {"decisions": [{"id": row["id"], "supported": row["supported"]}
                                  for row in decisions]}

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
