from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from source2reel import cli, pipeline
from source2reel.progress import Progress
from source2reel.util import run


class RecordingStream(io.StringIO):
    def __init__(self, tty: bool = False):
        super().__init__()
        self.tty = tty
        self.flushes = 0
        self.heartbeat = threading.Event()

    def isatty(self) -> bool:
        return self.tty

    def write(self, value: str) -> int:
        if "Still working" in value:
            self.heartbeat.set()
        return super().write(value)

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class ProgressTests(unittest.TestCase):
    def test_redirected_output_has_only_flushed_start_and_finish_lines(self):
        stream = RecordingStream()
        with Progress(stream, interval=0.001).step("Research batch 2/3"):
            self.assertEqual(stream.getvalue(), "● Research batch 2/3\n")
        self.assertEqual(stream.getvalue(), "● Research batch 2/3\nComplete: Research batch 2/3\n")
        self.assertEqual(stream.flushes, 2)
        self.assertNotIn("\x1b", stream.getvalue())

    def test_tty_heartbeat_cleans_up_on_success_and_error(self):
        for error in (False, True):
            with self.subTest(error=error):
                stream = RecordingStream(tty=True)
                display = Progress(stream, interval=0.001)
                clock = lambda: 120.0 if threading.current_thread() is threading.main_thread() else 128.0
                with patch("source2reel.progress.time.monotonic", side_effect=clock):
                    if error:
                        with self.assertRaisesRegex(ValueError, "provider failed"):
                            with display.step("Generating storyboard"):
                                self.assertTrue(stream.heartbeat.wait(1))
                                raise ValueError("provider failed")
                    else:
                        with display.step("Generating storyboard"):
                            self.assertTrue(stream.heartbeat.wait(1))
                output = stream.getvalue()
                self.assertIn("Still working — elapsed 00m 08s", output)
                self.assertIn("\r\033[K", output)
                self.assertTrue(output.endswith(("Failed: Generating storyboard\n" if error else "Complete: Generating storyboard\n")))
                after = stream.getvalue()
                self.assertEqual(after, stream.getvalue())

    def test_keyboard_interrupt_cleans_tty_line(self):
        stream = RecordingStream(tty=True)
        with self.assertRaises(KeyboardInterrupt):
            with Progress(stream, interval=0.001).step("Research batch 1/1"):
                self.assertTrue(stream.heartbeat.wait(1))
                raise KeyboardInterrupt
        self.assertTrue(stream.getvalue().endswith("Interrupted: Research batch 1/1\n"))

    def test_create_pipeline_reports_stages_and_plain_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "projects" / "demo" / "episode.json"
            stream = RecordingStream()
            cfg = {"research": {}, "vision": {"enabled": False}}
            with patch.object(pipeline, "load_engine_config", return_value=cfg), \
                 patch.object(pipeline, "ingest", return_value=root / "input"), \
                 patch.object(pipeline, "build_inventory", return_value={"evidence": []}), \
                 patch.object(pipeline, "provider_from_config", return_value=object()), \
                 patch.object(pipeline, "research", return_value={"facts": []}), \
                 patch.object(pipeline, "plan", return_value={"scenes": []}), \
                 patch.object(pipeline, "build_episode", return_value=path.with_suffix(".mp4")):
                self.assertEqual(
                    pipeline.create(root, ["demo"], "demo", "", False, True, progress=Progress(stream)), path
                )
                self.assertEqual(
                    pipeline.create(root, ["demo"], "demo", "", False, False, progress=Progress(stream)),
                    path.with_suffix(".mp4"),
                )
            messages = stream.getvalue()
            for label in ("Ingesting source 1/1", "Building evidence inventory", "Researching evidence", "Planning storyboard", "Rendering episode"):
                self.assertIn(f"● {label}\n", messages)
                self.assertIn(f"Complete: {label}\n", messages)
            self.assertIn(f"Storyboard ready for review: {path}\n", messages)
            self.assertIn(f"Episode ready: {path.with_suffix('.mp4')}\n", messages)
            self.assertNotIn("\x1b", messages)

    def test_failure_reports_failed_stage_and_does_not_report_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = RecordingStream()
            with patch.object(pipeline, "load_engine_config", return_value={"research": {}}), \
                 patch.object(pipeline, "ingest", side_effect=ValueError("bad source")):
                with self.assertRaisesRegex(ValueError, "bad source"):
                    pipeline.create(Path(tmp), ["bad"], "demo", "", False, True, progress=Progress(stream))
            self.assertIn("Failed: Ingesting source 1/1\n", stream.getvalue())
            self.assertNotIn("ready", stream.getvalue())

    def test_optional_vision_and_existing_build_report_real_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "demo"
            (project / "manifests").mkdir(parents=True)
            (project / "episode.json").write_text("{}")
            (project / "manifests" / "evidence.json").write_text('{"evidence": []}')
            stream = RecordingStream()
            cfg = {"research": {}, "vision": {"enabled": True}}
            output = project / "output" / "demo.mp4"
            with patch.object(pipeline, "load_engine_config", return_value=cfg), \
                 patch.object(pipeline, "ingest", return_value=root / "input"), \
                 patch.object(pipeline, "build_inventory", return_value={"evidence": []}), \
                 patch.object(pipeline, "provider_from_config", return_value=object()), \
                 patch.object(pipeline, "enrich_media", return_value={"evidence": []}) as vision, \
                 patch.object(pipeline, "research", return_value={"facts": []}), \
                 patch.object(pipeline, "plan", return_value={"scenes": []}), \
                 patch.object(pipeline, "build_episode", return_value=output):
                pipeline.create(root, ["demo"], "demo", "", False, True, progress=Progress(stream))
                self.assertEqual(pipeline.build_existing(root, "demo", progress=Progress(stream)), output)
            self.assertEqual(vision.call_count, 1)
            self.assertIn("● Inspecting visual evidence\n", stream.getvalue())
            self.assertIn("Complete: Inspecting visual evidence\n", stream.getvalue())
            self.assertIn(f"Episode ready: {output}\n", stream.getvalue())

    def test_explicit_cli_create_and_build_print_only_path_to_stdout(self):
        for command in (["create", "source", "--review"], ["build", "demo"]):
            with self.subTest(command=command):
                out, err = RecordingStream(), RecordingStream()
                expected = Path("/tmp/demo/episode.json")
                with patch.object(cli, "root_from_here", return_value=Path("/tmp")), \
                     patch.object(cli, "create", return_value=expected), \
                     patch.object(cli, "build_existing", return_value=expected), \
                     redirect_stdout(out), redirect_stderr(err):
                    cli.main(command)
                self.assertEqual(out.getvalue(), f"{expected}\n")
                self.assertGreater(out.flushes, 0)

    def test_cli_ctrl_c_exits_130_with_newline(self):
        out, err = RecordingStream(), RecordingStream()
        with patch.object(cli, "root_from_here", return_value=Path("/tmp")), \
             patch.object(cli, "create", side_effect=KeyboardInterrupt), \
             redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["create", "source"])
        self.assertEqual(raised.exception.code, 130)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "Interrupted.\n")

    def test_process_command_diagnostic_goes_to_stderr(self):
        out, err = RecordingStream(), RecordingStream()
        with patch("source2reel.util.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")), \
             redirect_stdout(out), redirect_stderr(err):
            run(["ffprobe", "file.wav"], capture=True)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "+ ffprobe file.wav\n")
        self.assertGreater(err.flushes, 0)


if __name__ == "__main__":
    unittest.main()
