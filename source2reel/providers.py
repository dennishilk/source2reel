from __future__ import annotations
import base64, json, math, mimetypes, urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from .config import DEFAULT_OUTPUT_RESERVE_TOKENS, output_reserve_tokens


class StructuredOutputError(ValueError):
    """The model's structured content is incomplete or malformed."""


class OutputLimitExceeded(StructuredOutputError):
    """The provider explicitly stopped at its generation limit."""


class ProviderTransportError(RuntimeError):
    """The HTTP response envelope was not valid JSON."""


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        raise NotImplementedError("This provider has no vision adapter")

    def visual_cache_signature(self) -> dict[str, Any] | None:
        """Identify vision request semantics; unknown providers do not reuse results.

        Subclasses with additional semantic options should override this method.
        Transport location and timeout do not change the model request.
        """
        model = getattr(self, "model", None)
        temperature = getattr(self, "temperature", None)
        limit = getattr(self, "max_output_tokens", None)
        if (not isinstance(model, str) or not model or
                type(temperature) not in (int, float) or not math.isfinite(temperature) or
                type(limit) is not int or limit <= 0):
            return None
        return {"provider": f"{type(self).__module__}.{type(self).__qualname__}",
                "model": model, "temperature": temperature,
                "max_output_tokens": limit}


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        try:
            return json.loads(r.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderTransportError("Local AI server returned a malformed HTTP JSON response") from exc


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        if "\n" not in text:
            raise StructuredOutputError("Local LLM returned an incomplete JSON code fence")
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise StructuredOutputError("Local LLM did not return a JSON object")
    try:
        result = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Local LLM returned malformed JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise StructuredOutputError("Local LLM did not return a JSON object")
    return result


def _openai_result(data: dict[str, Any]) -> dict[str, Any]:
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise OutputLimitExceeded("Local AI response reached the configured output token limit")
    return _extract_json(choice["message"]["content"])


def _ollama_result(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("done_reason") == "length":
        raise OutputLimitExceeded("Local AI response reached the configured output token limit")
    return _extract_json(data["message"]["content"])


class OpenAICompatProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float = 0.1, timeout: int = 600,
                 max_output_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        data = _post_json(self.url, {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }, self.timeout)
        return _openai_result(data)

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        data = _post_json(self.url, {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":[
                    {"type":"text","text":user},
                    {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}}
                ]}
            ],
            "response_format":{"type":"json_object"}
        }, self.timeout)
        return _openai_result(data)


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float = 0.1, timeout: int = 600,
                 max_output_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        data = _post_json(self.url, {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_output_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }, self.timeout)
        return _ollama_result(data)

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        data = _post_json(self.url, {
            "model":self.model,"stream":False,"format":"json",
            "options":{"temperature":self.temperature,"num_predict":self.max_output_tokens},
            "messages":[
                {"role":"system","content":system},
                {"role":"user","content":user,"images":[b64]}
            ]
        }, self.timeout)
        return _ollama_result(data)


def provider_from_config(cfg: dict[str, Any]) -> LLMProvider:
    p = cfg["llm"]
    kind = p["provider"]
    limit = output_reserve_tokens(cfg)
    if kind == "openai_compat":
        return OpenAICompatProvider(p["base_url"], p["model"], float(p.get("temperature", 0.1)), int(p.get("timeout", 600)), limit)
    if kind == "ollama":
        return OllamaProvider(p["base_url"], p["model"], float(p.get("temperature", 0.1)), int(p.get("timeout", 600)), limit)
    raise ValueError(f"Unsupported local LLM provider: {kind}")
