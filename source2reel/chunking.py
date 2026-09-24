from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .providers import LLMProvider
from .util import json_dump, json_load


Normalizer = Callable[[dict[str, Any]], dict[str, Any]]
PayloadBuilder = Callable[[int, list[Any]], dict[str, Any]]


def estimate_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate suitable for local context budgeting."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(text.encode("utf-8")) / 3.5))


def input_token_budget(context_size: int, output_reserve_tokens: int = 4096, safety_tokens: int = 1024) -> int:
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    budget = context_size - max(0, output_reserve_tokens) - max(0, safety_tokens)
    if budget < 1024:
        raise ValueError(
            f"context budget too small: context_size={context_size}, "
            f"output_reserve_tokens={output_reserve_tokens}, safety_tokens={safety_tokens}"
        )
    return budget


def fits_context(
    system: str,
    user: str,
    context_size: int,
    output_reserve_tokens: int = 4096,
    safety_tokens: int = 1024,
) -> bool:
    return estimate_tokens(system) + estimate_tokens(user) <= input_token_budget(
        context_size, output_reserve_tokens, safety_tokens
    )


def split_for_context(
    items: Iterable[Any],
    system: str,
    make_payload: PayloadBuilder,
    context_size: int,
    output_reserve_tokens: int = 4096,
    safety_tokens: int = 1024,
    max_chars: int | None = None,
) -> list[list[Any]]:
    """Split records so every complete request fits the configured local context."""
    chunks: list[list[Any]] = []
    current: list[Any] = []
    current_chars = 0

    def item_chars(item: Any) -> int:
        return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))

    for item in items:
        size = item_chars(item)
        candidate = current + [item]
        candidate_chars = current_chars + size
        part_number = len(chunks) + 1
        payload = json.dumps(make_payload(part_number, candidate), ensure_ascii=False)
        too_many_chars = max_chars is not None and candidate_chars > max_chars

        if current and (too_many_chars or not fits_context(
            system, payload, context_size, output_reserve_tokens, safety_tokens
        )):
            chunks.append(current)
            current = [item]
            current_chars = size
            part_number = len(chunks) + 1
            payload = json.dumps(make_payload(part_number, current), ensure_ascii=False)
            if not fits_context(
                system, payload, context_size, output_reserve_tokens, safety_tokens
            ):
                raise ValueError(
                    "A single evidence record exceeds the configured local-AI context budget; "
                    "split the source evidence into smaller inventory records."
                )
        else:
            current = candidate
            current_chars = candidate_chars

    if current:
        chunks.append(current)
    return chunks


def checkpointed_complete_json(
    provider: LLMProvider,
    system: str,
    payload: dict[str, Any],
    checkpoint: Path,
    normalize: Normalizer,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run one JSON request with an input-hashed checkpoint and bounded retry."""
    user = json.dumps(payload, ensure_ascii=False)
    digest = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()

    if checkpoint.exists():
        try:
            cached = json_load(checkpoint)
            if cached.get("input_sha256") == digest and isinstance(cached.get("result"), dict):
                return normalize(cached["result"])
        except (OSError, ValueError, TypeError):
            pass

    last_error: Exception | None = None
    for _attempt in range(max(1, max_retries + 1)):
        try:
            result = normalize(provider.complete_json(system, user))
            json_dump(checkpoint, {
                "version": 1,
                "input_sha256": digest,
                "result": result,
            })
            return result
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise last_error
