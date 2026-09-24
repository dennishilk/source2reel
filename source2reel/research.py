from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import checkpointed_complete_json, split_for_context
from .providers import LLMProvider
from .util import json_dump


def _payload(batch_number: int, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch": batch_number,
        "evidence": evidence,
        "required_output": {
            "facts": [{
                "claim": "...",
                "evidence_refs": ["E0001"],
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


def research(
    provider: LLMProvider,
    inventory: dict[str, Any],
    project_dir: Path,
    max_chars: int = 45000,
    context_size: int = 32768,
    output_reserve_tokens: int = 4096,
    safety_tokens: int = 1024,
    max_retries: int = 2,
) -> dict[str, Any]:
    system = (project_dir.parents[1] / "prompts" / "research.txt").read_text()
    valid = {e["ref"] for e in inventory["evidence"]}

    def normalize(result: dict[str, Any]) -> dict[str, Any]:
        facts = []
        for fact in result.get("facts", []):
            if not isinstance(fact, dict) or not str(fact.get("claim", "")).strip():
                continue
            refs = [r for r in fact.get("evidence_refs", []) if r in valid]
            if not refs:
                continue
            clean = dict(fact)
            clean["claim"] = str(fact["claim"]).strip()
            clean["evidence_refs"] = list(dict.fromkeys(refs))
            if clean.get("phase") not in {"development", "final", "background", "unknown"}:
                clean["phase"] = "unknown"
            if clean.get("confidence") not in {"high", "medium", "low"}:
                clean["confidence"] = "low"
            facts.append(clean)

        assets = []
        for asset in result.get("assets", []):
            if not isinstance(asset, dict) or asset.get("evidence_ref") not in valid:
                continue
            clean = dict(asset)
            clean["purpose"] = str(clean.get("purpose", "")).strip()
            assets.append(clean)
        return {"facts": facts, "assets": assets}

    chunks = split_for_context(
        inventory["evidence"],
        system,
        _payload,
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
        used.add(checkpoint)
        result = checkpointed_complete_json(
            provider,
            system,
            _payload(i, batch),
            checkpoint,
            normalize,
            max_retries=max_retries,
        )
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

    out = {"version": 1, "facts": allfacts, "assets": assets}
    json_dump(project_dir / "manifests" / "research.json", out)
    return out
