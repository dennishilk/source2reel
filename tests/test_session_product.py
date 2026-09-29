"""Exercise real launcher/process behavior without Arch package installation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from source2reel.cli import clean
from source2reel.hardware import HardwareInfo
from source2reel.paths import local_environment
from source2reel.session import server_command, start, status, stop_all


SOURCE = Path(__file__).resolve().parents[1]


class SessionProductTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        (self.root / "models").mkdir()
        (self.root / "models" / "test.gguf").write_text("fake model")
        (self.root / "config" / "engine.toml").write_text(
            '[llm]\nbase_url = "http://127.0.0.1:8080/v1"\nmodel = "test"\n'
        )
        self.server = self.root / "llama-server"
        self.server.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
        self.server.chmod(0o755)

    def test_managed_llama_server_defaults_to_one_slot(self):
        hardware = HardwareInfo("amd", (), True, "RADV", "llama.cpp+vulkan/radv")
        with patch.dict(os.environ, {"LLAMA_SERVER": str(self.server)}), patch(
            "source2reel.session.detect_hardware", return_value=hardware
        ):
            command = server_command(self.root)
        slot = command.index("-np")
        self.assertEqual(command[slot + 1], "1")

    def test_owned_stop_does_not_touch_unrelated_server_or_reused_pid_record(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()))
        hardware = HardwareInfo("amd", (), True, "RADV", "llama.cpp+vulkan/radv")
        try:
            with patch.dict(os.environ, {"LLAMA_SERVER": str(self.server)}), patch(
                "source2reel.session.detect_hardware", return_value=hardware
            ):
                entry = start(self.root)
            self.assertEqual(entry["pid"], status(self.root)[0]["pid"])
            self.assertFalse(unrelated.poll() is not None)
            # Inject an unrelated PID with the wrong start tick: it must be ignored.
            registry = self.root / "runtime/pids/managed.json"
            registry.write_text(json.dumps([entry, {"pid": unrelated.pid, "start_tick": "0"}]))
            self.assertEqual(stop_all(self.root), 1)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(status(self.root), [])
        finally:
            stop_all(self.root)

    def test_local_cache_and_safe_cleanup(self):
        env = local_environment(self.root)
        for key in ("HF_HOME", "HF_HUB_CACHE", "UV_CACHE_DIR", "TORCH_HOME",
                    "XDG_CACHE_HOME", "UV_PYTHON_INSTALL_DIR"):
            self.assertTrue(Path(env[key]).is_relative_to(self.root), key)
        (self.root / "cache").mkdir()
        (self.root / "cache" / "example").write_text("generated")
        (self.root / "projects").mkdir()
        (self.root / "projects" / "keep.txt").write_text("user project")
        clean(self.root, yes=False)
        self.assertTrue((self.root / "cache/example").exists())
        clean(self.root, yes=True)
        self.assertFalse((self.root / "cache").exists())
        self.assertTrue((self.root / "projects/keep.txt").exists())

    def test_repository_launcher_uses_local_interpreter(self):
        # A clean directory with just the launcher and a local Python is
        # sufficient to prove PATH-independent command dispatch.
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/python").symlink_to(sys.executable)
        (self.root / "s2r").write_bytes((SOURCE / "s2r").read_bytes())
        (self.root / "s2r").chmod(0o755)
        result = subprocess.run([str(self.root / "s2r"), "--help"], cwd=SOURCE,
                                text=True, capture_output=True,
                                env={"PATH": "/usr/bin:/bin", **{
                                    k: v for k, v in os.environ.items() if k != "PATH"
                                }})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("voice-test", result.stdout)


if __name__ == "__main__":
    unittest.main()
