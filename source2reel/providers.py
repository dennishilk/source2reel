from __future__ import annotations
import base64, json, mimetypes, urllib.request
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        raise NotImplementedError("This provider has no vision adapter")


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Local LLM did not return a JSON object")
    return json.loads(text[start:end + 1])


class OpenAICompatProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float = 0.1, timeout: int = 600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        data = _post_json(self.url, {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }, self.timeout)
        return _extract_json(data["choices"][0]["message"]["content"])

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        data = _post_json(self.url, {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":[
                    {"type":"text","text":user},
                    {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}}
                ]}
            ],
            "response_format":{"type":"json_object"}
        }, self.timeout)
        return _extract_json(data["choices"][0]["message"]["content"])


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float = 0.1, timeout: int = 600):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        data = _post_json(self.url, {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }, self.timeout)
        return _extract_json(data["message"]["content"])

    def complete_json_with_image(self, system: str, user: str, image_path) -> dict[str, Any]:
        b64 = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        data = _post_json(self.url, {
            "model":self.model,"stream":False,"format":"json",
            "options":{"temperature":self.temperature},
            "messages":[
                {"role":"system","content":system},
                {"role":"user","content":user,"images":[b64]}
            ]
        }, self.timeout)
        return _extract_json(data["message"]["content"])


def provider_from_config(cfg: dict[str, Any]) -> LLMProvider:
    p = cfg["llm"]
    kind = p["provider"]
    if kind == "openai_compat":
        return OpenAICompatProvider(p["base_url"], p["model"], float(p.get("temperature", 0.1)), int(p.get("timeout", 600)))
    if kind == "ollama":
        return OllamaProvider(p["base_url"], p["model"], float(p.get("temperature", 0.1)), int(p.get("timeout", 600)))
    raise ValueError(f"Unsupported local LLM provider: {kind}")
