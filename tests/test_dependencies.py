from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_core_dependencies_exclude_kokoro_and_torch():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    deps = "\n".join(project.get("dependencies", [])).lower()
    assert "kokoro" not in deps
    assert "torch" not in deps


def test_arch_baseline_pacman_install_excludes_nvidia_cuda_runtime_packages():
    script = (ROOT / "tools" / "bootstrap-arch.sh").read_text().lower()
    pacman_commands = "\n".join(
        line for line in script.splitlines() if "pacman" in line or line.lstrip().startswith(("ffmpeg ", "llama-cpp ", "vulkan-", "libva-"))
    )
    for token in ("nvidia", "cuda", "cudnn", "nccl", "rocm", "hip-runtime"):
        assert token not in pacman_commands, token
    assert "vulkan-radeon" in script
    assert "uv lock --python 3.12" in script
    assert "refusing baseline sync" in script


def test_voice_requirements_use_explicit_cpu_torch_path():
    requirements = (ROOT / "requirements" / "voice-kokoro-cpu.txt").read_text().lower()
    assert not re.search(r"^torch(?:[<>=~!].*)?$", requirements, re.M)
    setup = (ROOT / "tools" / "setup-voice.sh").read_text().lower()
    assert "download.pytorch.org/whl/cpu" in setup
    assert "--no-deps kokoro==0.9.4" in setup
