from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .providers import LLMProvider
from .providers import OutputLimitExceeded, StructuredOutputError
from .progress import Progress, step
from .util import json_dump, json_load


Normalizer = Callable[[dict[str, Any]], dict[str, Any]]
PayloadBuilder = Callable[[int, list[Any]], dict[str, Any]]
MAX_SPLIT_DEPTH = 6


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
            if not fits_context(system, payload, context_size, output_reserve_tokens, safety_tokens):
                raise ValueError(
                    "Research request metadata or a single evidence record exceeds the "
                    "configured local-AI context budget."
                )
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


def _record_refs(record: Any) -> str:
    """Identify an unsplittable record without requiring a particular payload shape."""
    if not isinstance(record, dict):
        return repr(record)[:120]
    refs = record.get("ref") or record.get("evidence_refs")
    if refs is None and isinstance(record.get("asset"), dict):
        refs = record["asset"].get("evidence_ref")
    if refs is None and isinstance(record.get("capsule"), dict):
        refs = record["capsule"].get("evidence_refs")
    if refs is None and isinstance(record.get("media"), dict):
        refs = record["media"].get("ref")
    return str(refs if refs is not None else record.get("kind", "unknown record"))


def checkpointed_split_json(
    provider: LLMProvider,
    system: str,
    items: list[Any],
    make_payload: Callable[[list[Any]], dict[str, Any]],
    checkpoint: Path,
    normalize_batch: Callable[[dict[str, Any], list[Any]], dict[str, Any]],
    *,
    max_retries: int = 2,
    progress: Progress | None = None,
    label: str = "Structured request",
    used: set[Path] | None = None,
    max_split_depth: int = MAX_SPLIT_DEPTH,
    split_single: Callable[[Any], tuple[Any, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Retry a part, then split recoverable output; optionally divide one item."""
    if max_split_depth < 0:
        raise ValueError("max_split_depth must be non-negative")
    if not items:
        raise ValueError("Cannot split an empty structured request")
    used = used if used is not None else set()

    def run(batch: list[Any], path: Path, depth: int, suffix: str) -> list[dict[str, Any]]:
        used.add(path)
        payload = make_payload(batch)
        user = json.dumps(payload, ensure_ascii=False)
        digest = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()
        marker = path.with_name(f"{path.stem}-split{path.suffix}")
        current_label = label if not suffix else f"{label} — split {suffix} ({'1' if suffix.endswith('a') else '2'}/2)"

        # A split marker is routing metadata, not a successful model result.
        # Reuse it only for identical input, and always prefer a valid parent.
        cached_parent = False
        if path.exists():
            try:
                cached = json_load(path)
                cached_parent = cached.get("input_sha256") == digest and isinstance(cached.get("result"), dict)
            except (OSError, ValueError, TypeError):
                pass
        split_known = False
        saved_reason = ""
        saved_output_limit = False
        if not cached_parent and marker.exists():
            try:
                saved = json_load(marker)
                split_known = saved.get("input_sha256") == digest and saved.get("split") is True
                if split_known:
                    saved_reason = str(saved.get("reason") or "")
                    saved_output_limit = saved.get("output_limit") is True
            except (OSError, ValueError, TypeError):
                pass

        if not split_known:
            try:
                with step(progress, current_label):
                    return [checkpointed_complete_json(
                        provider, system, payload, path,
                        lambda value: normalize_batch(value, batch), max_retries=max_retries,
                    )]
            except (StructuredOutputError, json.JSONDecodeError) as exc:
                failure = exc
        else:
            failure = None
        reason = str(failure) if failure is not None else saved_reason or "model output was incomplete"

        if len(batch) == 1:
            if split_single is None or not (isinstance(failure, OutputLimitExceeded) or
                                            (split_known and saved_output_limit)):
                raise RuntimeError(
                    f"{current_label}: invalid structured output for single record "
                    f"{_record_refs(batch[0])} ({reason}); check the model output budget "
                    "or split the source record"
                ) from failure
            if depth >= max_split_depth:
                raise RuntimeError(
                    f"{label}: record {_record_refs(batch[0])} hit the output limit "
                    f"({reason}); maximum split depth {max_split_depth} reached; "
                    "source record cannot be subdivided further"
                ) from failure
            divided = split_single(batch[0])
            if divided is None:
                raise RuntimeError(
                    f"{label}: record {_record_refs(batch[0])} hit the output limit "
                    f"({reason}); source record cannot be subdivided further"
                ) from failure
            left, right = [divided[0]], [divided[1]]
        elif depth >= max_split_depth:
            raise RuntimeError(
                f"{current_label}: structured output still invalid at maximum split depth "
                f"{max_split_depth}; refs: {', '.join(_record_refs(item) for item in batch)}; "
                f"reason: {reason}"
            ) from failure
        else:
            midpoint = len(batch) // 2
            left, right = batch[:midpoint], batch[midpoint:]

        used.add(marker)
        if not split_known:
            json_dump(marker, {"version": 1, "input_sha256": digest, "split": True,
                               "reason": reason,
                               "output_limit": isinstance(failure, OutputLimitExceeded)})
        if progress is not None:
            progress.note(f"{current_label}: malformed or truncated structured output; processing smaller parts"
                          if not split_known else f"{current_label}: resuming smaller parts")
        first = path.with_name(f"{path.stem}-a{path.suffix}")
        second = path.with_name(f"{path.stem}-b{path.suffix}")
        return run(left, first, depth + 1, suffix + "a") + run(
            right, second, depth + 1, suffix + "b"
        )

    return run(items, checkpoint, 0, "")
