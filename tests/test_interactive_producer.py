"""Interactive producer tests with all production boundaries mocked."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import cli, session


EPISODE = {
    "version": 1,
    "title": "Example",
    "scenes": [
        {"id": "s001", "type": "PROJECT_EVIDENCE", "title": "Proof",
         "narration": "The original device runs the software.",
         "evidence_refs": ["E0001"], "asset_ref": "E0001"},
    ],
}


class InteractiveProducerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        (self.root / "config/engine.toml").write_text('[llm]\nprovider = "openai_compat"\n')
        self.stdout = StringIO()

    def episode(self, name="example"):
        directory = self.root / "projects" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "episode.json"
        path.write_text(json.dumps(EPISODE))
        return path

    def run_new(self, answers, managed=()):
        path = self.episode()
        with patch("builtins.input", side_effect=answers), \
             patch("source2reel.session.status", return_value=list(managed)) as status, \
             patch("source2reel.session.start") as start, \
             patch("source2reel.pipeline.create", return_value=path) as create, \
             redirect_stdout(self.stdout):
            session._new_episode(self.root)
        return create, start, status

    def test_one_source_blank_instructions_default_review_and_managed_reuse(self):
        create, start, status = self.run_new(
            ["https://github.com/dennishilk/boringos", "", "", "", "q"], managed=[{"pid": 123}]
        )
        create.assert_called_once_with(
            self.root, ["https://github.com/dennishilk/boringos"], None, "", False, True
        )
        status.assert_called_once_with(self.root)
        start.assert_not_called()
        self.assertIn("automatic story discovery", self.stdout.getvalue())
        self.assertIn("projects/example/episode.json", self.stdout.getvalue())

    def test_multiple_sources_custom_focus_and_absent_ai_starts(self):
        create, start, _ = self.run_new(
            ["https://example.com", "https://github.com/example/repo", "", "Focus on proof", "y", "q"]
        )
        create.assert_called_once_with(
            self.root, ["https://example.com", "https://github.com/example/repo"],
            None, "Focus on proof", False, True
        )
        start.assert_called_once_with(self.root)

    def test_explicit_no_review_builds_in_existing_create(self):
        with patch("builtins.input", side_effect=["", "missing-directory", str(self.root), "", "", "n"]), \
             patch("source2reel.session.status", return_value=[]), \
             patch("source2reel.session.start"), \
             patch("source2reel.pipeline.create", return_value=self.root / "output.mp4") as create, \
             patch("source2reel.pipeline.build_existing") as build, \
             redirect_stdout(self.stdout):
            session._new_episode(self.root)
        create.assert_called_once_with(self.root, [str(self.root)], None, "", False, False)
        build.assert_not_called()
        self.assertIn("Enter a valid", self.stdout.getvalue())

    def test_ollama_config_does_not_start_llama_server(self):
        (self.root / "config/engine.toml").write_text('[llm]\nprovider = "ollama"\n')
        _, start, status = self.run_new(["https://example.com", "", "", "", "q"])
        start.assert_not_called()
        status.assert_not_called()

    def test_project_list_is_sorted_and_ignores_unusable_episodes(self):
        self.episode("z-good")
        self.episode("a-good")
        broken = self.root / "projects/broken"
        broken.mkdir()
        (broken / "episode.json").write_text("not JSON")
        empty = self.root / "projects/empty"
        empty.mkdir()
        (empty / "episode.json").write_text(json.dumps({"version": 1, "scenes": []}))
        (self.root / "projects/no-episode").mkdir()
        (self.root / "projects/link").symlink_to(self.root / "projects/a-good", target_is_directory=True)
        self.assertEqual(session._projects(self.root), ["a-good", "z-good"])
        with patch("builtins.input", side_effect=["bad", "3", "2"]), redirect_stdout(self.stdout):
            self.assertEqual(session._select_project(self.root), "z-good")
        self.assertEqual(self.stdout.getvalue().count("Invalid project number."), 2)

    def test_empty_project_list(self):
        with redirect_stdout(self.stdout):
            self.assertIsNone(session._select_project(self.root))
        self.assertIn("No usable episodes", self.stdout.getvalue())

    def test_review_is_deterministic_and_build_dispatches(self):
        self.episode()
        with redirect_stdout(self.stdout):
            session._review(self.root, "example")
        first = self.stdout.getvalue()
        self.stdout = StringIO()
        with redirect_stdout(self.stdout):
            session._review(self.root, "example")
        self.assertEqual(first, self.stdout.getvalue())
        self.assertIn("s001 | PROJECT_EVIDENCE | Proof", first)
        self.assertIn("Evidence: E0001", first)
        with patch("builtins.input", side_effect=["r", "b"]), \
             patch("source2reel.pipeline.build_existing", return_value=self.root / "video.mp4") as build, \
             redirect_stdout(self.stdout):
            session._review_or_build(self.root, "example", show_first=False, new=False)
        build.assert_called_once_with(self.root, "example")

    def test_build_menu_and_continue_quit(self):
        self.episode()
        with patch("builtins.input", side_effect=["2", "1", "q", "3", "1", "0"]), \
             patch("source2reel.pipeline.build_existing", return_value=self.root / "video.mp4") as build, \
             redirect_stdout(self.stdout):
            session.interactive(self.root)
        build.assert_called_once_with(self.root, "example")
        self.assertIn("Storyboard: Example", self.stdout.getvalue())

    def test_invalid_menu_eof_and_ctrl_c(self):
        for signal in (EOFError, KeyboardInterrupt):
            self.stdout = StringIO()
            with patch("builtins.input", side_effect=["invalid", signal]), redirect_stdout(self.stdout):
                session.interactive(self.root)
            self.assertIn("Unknown choice", self.stdout.getvalue())
        with patch("builtins.input", side_effect=["1", EOFError]), redirect_stdout(StringIO()):
            session.interactive(self.root)

    def test_failed_ai_start_returns_to_menu_without_creating(self):
        with patch("builtins.input", side_effect=["1", "https://example.com", "", "", "", "0"]), \
             patch("source2reel.session.status", return_value=[]), \
             patch("source2reel.session.start", side_effect=RuntimeError("model missing")), \
             patch("source2reel.pipeline.create") as create, redirect_stdout(self.stdout):
            session.interactive(self.root)
        create.assert_not_called()
        self.assertIn("Error: model missing", self.stdout.getvalue())

    def test_failed_create_and_build_return_to_menu_and_keep_project(self):
        path = self.episode()
        with patch("builtins.input", side_effect=[
            "1", "https://example.com", "", "", "", "3", "1", "0"
        ]), patch("source2reel.session.status", return_value=[{"pid": 123}]), \
             patch("source2reel.pipeline.create", side_effect=RuntimeError("research failed")), \
             patch("source2reel.pipeline.build_existing", side_effect=RuntimeError("render failed")), \
             redirect_stdout(self.stdout):
            session.interactive(self.root)
        self.assertTrue(path.is_file())
        self.assertIn("Error: research failed", self.stdout.getvalue())
        self.assertIn("Error: render failed", self.stdout.getvalue())

    def test_explicit_cli_arguments_remain_unchanged(self):
        with patch("source2reel.cli.root_from_here", return_value=self.root), \
             patch("source2reel.cli.create", return_value=self.root / "episode.json") as create, \
             redirect_stdout(self.stdout):
            cli.main(["create", "https://example.com", "--review"])
            cli.main(["create", "https://example.com"])
        self.assertEqual(create.call_args_list[0].args, (
            self.root, ["https://example.com"], None, "", False, True, None
        ))
        self.assertEqual(create.call_args_list[1].args, (
            self.root, ["https://example.com"], None, "", False, False, None
        ))


if __name__ == "__main__":
    unittest.main()
