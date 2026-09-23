from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tomllib
import urllib.request

from .config import load_engine_config, profile_paths
from .hardware import HardwareInfo, detect_hardware
from .tts_engine import voice_runtime_python


@dataclass(frozen=True)
class Check:
    section: str
    name: str
    ok: bool
    message: str
    required: bool = False


def _http_ok(url: str, timeout: int = 3) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


def _voice_checks(root: Path, voice_path: Path) -> list[Check]:
    checks: list[Check] = []
    if not voice_path.exists():
        return checks

    voice_cfg = tomllib.loads(voice_path.read_text())
    engine = str(voice_cfg.get("voice", {}).get("engine", "")).lower()
    if engine != "kokoro":
        checks.append(Check("VOICE (optional)", "voice-runtime", True, f"engine={engine or 'unspecified'}"))
        return checks

    runtime = voice_runtime_python(root, voice_cfg)
    if not runtime.exists():
        checks.append(Check(
            "VOICE (optional)",
            "kokoro-runtime",
            False,
            f"not installed ({runtime}); run ./tools/setup-voice.sh",
            required=False,
        ))
        return checks

    probe = subprocess.run(
        [str(runtime), "-c", "import kokoro, torch; print(torch.__version__)"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks.append(Check(
        "VOICE (optional)",
        "kokoro-runtime",
        probe.returncode == 0,
        probe.stdout.strip() or f"probe exit {probe.returncode}",
        required=False,
    ))
    return checks


def collect_checks(
    root: Path,
    cfg_path: Path | None = None,
    hardware: HardwareInfo | None = None,
) -> list[Check]:
    cfg = load_engine_config(root, cfg_path)
    checks: list[Check] = []

    for cmd in ("ffmpeg", "ffprobe", "git"):
        path = shutil.which(cmd)
        checks.append(Check("CORE", cmd, bool(path), path or "not found", required=True))

    pillow_ok = importlib.util.find_spec("PIL") is not None
    checks.append(Check("CORE", "python:PIL", pillow_ok, "importable" if pillow_ok else "missing", required=True))

    theme_path, voice_path = profile_paths(root, cfg)
    checks.append(Check("CORE", "theme", theme_path.exists(), str(theme_path), required=True))
    checks.append(Check("CORE", "voice-profile", voice_path.exists(), str(voice_path), required=True))
    checks.extend(_voice_checks(root, voice_path))

    info = hardware or detect_hardware()
    checks.append(Check("HARDWARE", "gpu-vendor", True, info.vendor))
    if info.devices:
        for index, device in enumerate(info.devices, start=1):
            checks.append(Check("HARDWARE", f"gpu-{index}", True, device))
    else:
        checks.append(Check("HARDWARE", "gpu", True, "no discrete/display GPU detected"))
    checks.append(Check("HARDWARE", "vulkan", info.vulkan_available, info.vulkan_summary, required=False))
    checks.append(Check("HARDWARE", "inference-backend", True, info.inference_backend))

    provider = cfg["llm"]
    if provider["provider"] == "openai_compat":
        url = provider["base_url"].rstrip("/") + "/models"
    else:
        url = provider["base_url"].rstrip("/") + "/api/tags"
    ok, message = _http_ok(url)
    checks.append(Check("LOCAL AI", "local-ai", ok, message, required=True))

    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    checks.append(Check(
        "VOICE (optional)",
        "espeak-preview",
        bool(espeak),
        (espeak or "not installed") + "; smoke-test fallback only",
        required=False,
    ))
    return checks


def exit_code(checks: list[Check]) -> int:
    return 0 if all(check.ok for check in checks if check.required) else 1


def run_doctor(root: Path, cfg_path: Path | None = None) -> int:
    checks = collect_checks(root, cfg_path)
    width = max(len(check.name) for check in checks)
    current = None
    for check in checks:
        if check.section != current:
            if current is not None:
                print()
            current = check.section
            print(f"[{current}]")
        status = "OK" if check.ok else ("FAIL" if check.required else "OPTIONAL")
        print(f"{check.name:<{width}}  {status:<8}  {check.message}")
    return exit_code(checks)
