"""Repository-owned runtime paths and subprocess environment."""
from __future__ import annotations

import os
from pathlib import Path


def runtime_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    return {
        "cache": root / "cache",
        "huggingface": root / "cache" / "huggingface",
        "uv": root / "cache" / "uv",
        "torch": root / "cache" / "torch",
        "models": root / "models",
        "runtime": root / "runtime",
        "pids": root / "runtime" / "pids",
        "logs": root / "runtime" / "logs",
        "output": root / "output",
    }


def local_environment(root: Path) -> dict[str, str]:
    paths = runtime_paths(root)
    env = os.environ.copy()
    env.update({
        "SOURCE2REEL_ROOT": str(root.resolve()),
        "HF_HOME": str(paths["huggingface"]),
        "HF_HUB_CACHE": str(paths["huggingface"] / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(paths["huggingface"] / "hub"),
        "TRANSFORMERS_CACHE": str(paths["huggingface"] / "transformers"),
        "TORCH_HOME": str(paths["torch"]),
        "UV_CACHE_DIR": str(paths["uv"]),
        "UV_PYTHON_INSTALL_DIR": str(root.resolve() / "runtime" / "python"),
        "XDG_CACHE_HOME": str(paths["cache"] / "xdg"),
    })
    return env
