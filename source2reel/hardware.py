from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import re
import shutil
import subprocess


@dataclass(frozen=True)
class HardwareInfo:
    vendor: str
    devices: tuple[str, ...]
    vulkan_available: bool
    vulkan_summary: str
    inference_backend: str


def parse_gpu_devices(lspci_text: str) -> tuple[str, ...]:
    devices: list[str] = []
    for raw in lspci_text.splitlines():
        line = raw.strip()
        if re.search(r"\b(VGA compatible controller|3D controller|Display controller)\b", line, re.I):
            devices.append(line)
    return tuple(devices)


def classify_gpu_vendor(devices: tuple[str, ...]) -> str:
    joined = "\n".join(devices)
    if re.search(r"Advanced Micro Devices|AMD/ATI|\bAMD\b|\bATI\b", joined, re.I):
        return "amd"
    if re.search(r"\bNVIDIA\b", joined, re.I):
        return "nvidia"
    if re.search(r"\bIntel\b", joined, re.I):
        return "intel"
    return "cpu-only" if not devices else "other"


def select_inference_backend(vendor: str, vulkan_available: bool) -> str:
    if vulkan_available:
        if vendor == "amd":
            return "llama.cpp+vulkan/radv"
        return "llama.cpp+vulkan"
    return "llama.cpp+cpu"


def _run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    if not shutil.which(args[0]):
        return None
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def detect_hardware() -> HardwareInfo:
    lspci = _run(["lspci", "-nn"])
    devices = parse_gpu_devices(lspci.stdout if lspci else "")
    vendor = classify_gpu_vendor(devices)

    vulkan = _run(["vulkaninfo", "--summary"])
    vulkan_ok = bool(vulkan and vulkan.returncode == 0)
    vulkan_summary = "unavailable"
    if vulkan:
        lines = [line.strip() for line in vulkan.stdout.splitlines() if line.strip()]
        interesting = [line for line in lines if "GPU" in line or "deviceName" in line or "driverName" in line]
        vulkan_summary = "; ".join(interesting[:6]) or ("available" if vulkan_ok else "command failed")

    return HardwareInfo(
        vendor=vendor,
        devices=devices,
        vulkan_available=vulkan_ok,
        vulkan_summary=vulkan_summary,
        inference_backend=select_inference_backend(vendor, vulkan_ok),
    )


def _as_dict(info: HardwareInfo) -> dict[str, object]:
    return {
        "vendor": info.vendor,
        "devices": list(info.devices),
        "vulkan_available": info.vulkan_available,
        "vulkan_summary": info.vulkan_summary,
        "inference_backend": info.inference_backend,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect Source2Reel local inference hardware")
    parser.add_argument("--format", choices=("text", "json", "env"), default="text")
    args = parser.parse_args()
    info = detect_hardware()

    if args.format == "json":
        print(json.dumps(_as_dict(info), indent=2))
    elif args.format == "env":
        print(f"SOURCE2REEL_GPU_VENDOR={info.vendor}")
        print(f"SOURCE2REEL_VULKAN_AVAILABLE={'1' if info.vulkan_available else '0'}")
        print(f"SOURCE2REEL_INFERENCE_BACKEND={info.inference_backend}")
    else:
        print(f"GPU vendor: {info.vendor}")
        if info.devices:
            for device in info.devices:
                print(f"GPU: {device}")
        else:
            print("GPU: none detected")
        print(f"Vulkan: {'available' if info.vulkan_available else 'unavailable'}")
        print(f"Vulkan detail: {info.vulkan_summary}")
        print(f"Source2Reel inference backend: {info.inference_backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
