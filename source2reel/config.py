from __future__ import annotations
import os, tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def deep_env_expand(value):
    if isinstance(value, dict):
        return {k: deep_env_expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_env_expand(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_engine_config(root: Path, override: Path | None = None) -> dict[str, Any]:
    cfg = load_toml(root / "config" / "engine.toml")
    for path in (root / "config" / "local.toml", override):
        if path is None or not path.exists():
            continue
        user = load_toml(path)
        for section, values in user.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return deep_env_expand(cfg)


def profile_paths(root: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    profile = cfg.get("profile", {})
    theme = str(profile.get("theme", "dennis-dark"))
    voice = str(profile.get("voice", "dennis-explainer"))
    return root / "themes" / f"{theme}.toml", root / "voices" / f"{voice}.toml"


def series_label(cfg: dict[str, Any]) -> str:
    return str(cfg.get("profile", {}).get("series_label", "SOURCE2REEL"))
