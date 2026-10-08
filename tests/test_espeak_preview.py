"""Preview speech works with either installed executable, without real TTS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from source2reel.tts_engine import espeak_preview


class EspeakPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.output = self.root / "audio" / "preview.wav"

    def install_stub(self, name):
        executable = self.bin / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, wave\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "Path(__file__).with_suffix('.json').write_text(json.dumps(args))\n"
            "with wave.open(args[args.index('-w') + 1], 'wb') as output:\n"
            "    output.setnchannels(1)\n"
            "    output.setsampwidth(2)\n"
            "    output.setframerate(24000)\n"
            "    output.writeframes(bytes(480))\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable.with_suffix(".json")

    def check_output(self, log, text):
        arguments = json.loads(log.read_text())
        self.assertEqual(arguments[-1], text)
        self.assertEqual(arguments[arguments.index("-w") + 1], str(self.output))
        self.assertEqual(arguments[arguments.index("-v") + 1], "en-us+m3")
        with wave.open(str(self.output)) as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getframerate(), 24000)
            self.assertGreater(audio.getnframes(), 0)

    def test_ng_only_installation_generates_preview_audio(self):
        log = self.install_stub("espeak-ng")
        text = "Cisco CP-9951 — local preview."
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            espeak_preview(text, self.output)
        self.check_output(log, text)

    def test_classic_only_installation_still_generates_audio(self):
        log = self.install_stub("espeak")
        text = "Classic eSpeak preview."
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            espeak_preview(text, self.output)
        self.check_output(log, text)

    def test_existing_classic_selection_is_preserved_when_both_are_installed(self):
        classic_log = self.install_stub("espeak")
        ng_log = self.install_stub("espeak-ng")
        text = "Keep the existing preview selection."
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            espeak_preview(text, self.output)
        self.check_output(classic_log, text)
        self.assertFalse(ng_log.exists())

    def test_missing_executables_report_an_actionable_error(self):
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            with self.assertRaisesRegex(RuntimeError, "espeak or espeak-ng"):
                espeak_preview("Preview.", self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
