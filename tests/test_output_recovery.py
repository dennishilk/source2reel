from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.chunking import checkpointed_split_json
from source2reel.config import DEFAULT_OUTPUT_RESERVE_TOKENS
from source2reel.pipeline import _context_options
from source2reel.planner import plan
from source2reel.progress import Progress
from source2reel.providers import (
    OllamaProvider, OpenAICompatProvider, OutputLimitExceeded,
    ProviderTransportError, StructuredOutputError, provider_from_config,
)
from source2reel.research import research
from source2reel.util import json_load


def _refs(count: int) -> list[dict[str, str]]:
    return [{"ref": f"E{i:04d}", "kind": "document", "relative_path": f"doc-{i}.txt",
             "excerpt": f"Grounded E{i:04d} is established. Fact E{i:04d} is recorded."}
            for i in range(1, count + 1)]


def _run_split(provider, items, checkpoint, *, retries=1, depth=6, progress=None):
    return checkpointed_split_json(
        provider, "research", items, lambda part: {"evidence": part}, checkpoint,
        lambda value, part: {"facts": [fact for fact in value["facts"]
                                               if fact["ref"] in {item["ref"] for item in part}]},
        max_retries=retries, max_split_depth=depth, progress=progress,
        label="Research batch 2/93",
    )


class ProviderBudgetTests(unittest.TestCase):
    def test_openai_cap_applies_to_text_and_vision(self):
        provider = OpenAICompatProvider("http://local/v1", "local", max_output_tokens=777)
        with tempfile.TemporaryDirectory() as tmp:
            picture = Path(tmp) / "photo.png"
            picture.write_bytes(b"image-bytes")
            with patch("source2reel.providers._post_json", return_value={
                "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]
            }) as post:
                self.assertEqual(provider.complete_json("s", "u"), {"ok": True})
                self.assertEqual(provider.complete_json_with_image("s", "u", picture), {"ok": True})
            self.assertEqual([call.args[1]["max_tokens"] for call in post.call_args_list], [777, 777])

    def test_ollama_cap_applies_to_text_and_vision(self):
        provider = OllamaProvider("http://local", "local", max_output_tokens=811)
        with tempfile.TemporaryDirectory() as tmp:
            picture = Path(tmp) / "photo.png"
            picture.write_bytes(b"image-bytes")
            with patch("source2reel.providers._post_json", return_value={
                "done_reason": "stop", "message": {"content": '{"ok":true}'}
            }) as post:
                self.assertEqual(provider.complete_json("s", "u"), {"ok": True})
                self.assertEqual(provider.complete_json_with_image("s", "u", picture), {"ok": True})
            self.assertEqual([call.args[1]["options"]["num_predict"]
                              for call in post.call_args_list], [811, 811])

    def test_default_and_override_share_one_source_for_context_and_providers(self):
        for kind in ("openai_compat", "ollama"):
            for chunking, expected in (({}, DEFAULT_OUTPUT_RESERVE_TOKENS),
                                       ({"output_reserve_tokens": 777}, 777)):
                with self.subTest(kind=kind, chunking=chunking):
                    cfg = {"llm": {"provider": kind, "base_url": "http://local", "model": "m"},
                           "chunking": chunking}
                    self.assertEqual(provider_from_config(cfg).max_output_tokens, expected)
                    self.assertEqual(_context_options(cfg)["output_reserve_tokens"], expected)
        cfg["chunking"]["output_reserve_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "output_reserve_tokens must be positive"):
            provider_from_config(cfg)

    def test_provider_reports_length_and_model_parse_failure_as_recoverable(self):
        with patch("source2reel.providers._post_json", return_value={
            "choices": [{"finish_reason": "length", "message": {"content": '{"ok":true}'}}]
        }):
            with self.assertRaises(OutputLimitExceeded):
                OpenAICompatProvider("http://local/v1", "m").complete_json("s", "u")
        with patch("source2reel.providers._post_json", return_value={
            "done_reason": "length", "message": {"content": "{"}
        }):
            with self.assertRaises(OutputLimitExceeded):
                OllamaProvider("http://local", "m").complete_json("s", "u")
        with patch("source2reel.providers._post_json", return_value={
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":'}}]
        }):
            with self.assertRaises(StructuredOutputError):
                OpenAICompatProvider("http://local/v1", "m").complete_json("s", "u")


class SplitRecoveryTests(unittest.TestCase):
    def test_research_recovers_only_failed_batch_and_recurses_only_failed_child(self):
        class SelectiveProvider:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                refs = tuple(entry["ref"] for entry in json.loads(user)["evidence"])
                self.requests.append(refs)
                if refs in (("E0003", "E0004", "E0005", "E0006"), ("E0005", "E0006")):
                    raise StructuredOutputError("response truncated")
                return {"facts": [{"claim": f"Grounded {ref}", "evidence_refs": [ref],
                                   "support": [{"evidence_ref": ref,
                                                "text": f"Grounded {ref} is established."}]}
                                  for ref in refs], "assets": []}

        provider = SelectiveProvider()
        evidence = _refs(6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "demo"
            (root / "prompts").mkdir()
            (root / "prompts" / "research.txt").write_text("research")
            output = io.StringIO()
            with patch("source2reel.research.split_for_context",
                       return_value=[evidence[:2], evidence[2:]]):
                first = research(provider, {"evidence": evidence}, project, progress=Progress(output),
                                 max_retries=1)
                part_dir = project / "manifests" / "research-parts"
                parent = part_dir / "part-001.json"
                saved_parent = parent.read_bytes()
                for name in ("part-002-split.json", "part-002-a.json", "part-002-b-split.json",
                             "part-002-b-a.json", "part-002-b-b.json"):
                    self.assertTrue((part_dir / name).exists(), name)
                self.assertFalse((part_dir / "part-002.json").exists())
                self.assertFalse((part_dir / "part-002-a-a.json").exists())
                self.assertEqual([fact["evidence_refs"][0] for fact in first["facts"]],
                                 [entry["ref"] for entry in evidence])
                self.assertEqual(provider.requests.count(("E0001", "E0002")), 1)
                self.assertEqual(provider.requests.count(("E0003", "E0004")), 1)
                self.assertEqual(provider.requests.count(("E0005", "E0006")), 2)
                total_calls = len(provider.requests)
                # Old failed children are reused on resume; no successful parent is invalidated.
                (part_dir / "part-999.json").write_text("{}")
                second = research(provider, {"evidence": evidence}, project, max_retries=1)
            self.assertEqual(first, second)
            self.assertEqual(len(provider.requests), total_calls)
            self.assertEqual(parent.read_bytes(), saved_parent)
            self.assertFalse((part_dir / "part-999.json").exists())
            self.assertIn("malformed or truncated structured output; processing smaller parts", output.getvalue())
            self.assertIn("Research batch 2/2 — split a", output.getvalue())
            self.assertNotIn("%", output.getvalue())
            self.assertNotIn("\x1b", output.getvalue())

    def test_json_decode_error_is_recoverable_and_results_are_normalized(self):
        class DecodeProvider:
            def complete_json(self, system, user):
                batch = json.loads(user)["evidence"]
                if len(batch) > 1:
                    raise json.JSONDecodeError("incomplete", "{", 1)
                return {"facts": [{"ref": batch[0]["ref"]}, {"ref": "E9999"}]}

        with tempfile.TemporaryDirectory() as tmp:
            results = _run_split(DecodeProvider(), _refs(2), Path(tmp) / "part.json", retries=1)
            self.assertEqual([item["facts"] for item in results],
                             [[{"ref": "E0001"}], [{"ref": "E0002"}]])

    def test_success_keeps_original_parent_checkpoint_and_behavior(self):
        class Success:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                return {"facts": [{"ref": item["ref"]} for item in json.loads(user)["evidence"]]}

        provider = Success()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "part.json"
            first = _run_split(provider, _refs(2), checkpoint)
            self.assertEqual(first, _run_split(provider, _refs(2), checkpoint))
            self.assertEqual(provider.calls, 1)
            self.assertEqual(json_load(checkpoint)["result"], first[0])
            self.assertFalse((Path(tmp) / "part-split.json").exists())

    def test_resume_after_child_infrastructure_failure_reuses_finished_child(self):
        class InterruptedProvider:
            unavailable = True

            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                refs = tuple(item["ref"] for item in json.loads(user)["evidence"])
                self.requests.append(refs)
                if len(refs) > 1:
                    raise StructuredOutputError("truncated")
                if refs == ("E0002",) and self.unavailable:
                    raise ConnectionRefusedError("server stopped")
                return {"facts": [{"ref": refs[0]}]}

        provider = InterruptedProvider()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "part.json"
            with self.assertRaises(ConnectionRefusedError):
                _run_split(provider, _refs(2), checkpoint)
            self.assertEqual(provider.requests,
                             [("E0001", "E0002"), ("E0001", "E0002"),
                              ("E0001",), ("E0002",), ("E0002",)])
            self.assertTrue((Path(tmp) / "part-a.json").exists())
            provider.unavailable = False
            self.assertEqual(_run_split(provider, _refs(2), checkpoint),
                             [{"facts": [{"ref": "E0001"}]}, {"facts": [{"ref": "E0002"}]}])
            self.assertEqual(provider.requests[-1], ("E0002",))
            self.assertEqual(len(provider.requests), 6)

    def test_changed_input_invalidates_split_marker(self):
        class Provider:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                refs = tuple(item["ref"] for item in json.loads(user)["evidence"])
                self.requests.append(refs)
                if refs == ("E0001", "E0002"):
                    raise StructuredOutputError("truncated")
                return {"facts": [{"ref": ref} for ref in refs]}

        provider = Provider()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "part.json"
            _run_split(provider, _refs(2), checkpoint)
            changed = [_refs(2)[0], _refs(3)[2]]
            self.assertEqual(_run_split(provider, changed, checkpoint),
                             [{"facts": [{"ref": "E0001"}, {"ref": "E0003"}]}])
            self.assertEqual(provider.requests[-1], ("E0001", "E0003"))

    def test_maximum_depth_is_finite_and_single_record_error_names_ref(self):
        class AlwaysMalformed:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                raise StructuredOutputError("too much output")

        with tempfile.TemporaryDirectory() as tmp:
            provider = AlwaysMalformed()
            with self.assertRaisesRegex(RuntimeError, "maximum split depth 1.*E0001.*E0002"):
                _run_split(provider, _refs(4), Path(tmp) / "part.json", retries=1, depth=1)
            self.assertEqual(provider.calls, 4)
            with self.assertRaisesRegex(RuntimeError, r"single record E0001 \(too much output\)"):
                _run_split(provider, _refs(1), Path(tmp) / "single.json", retries=0)
            self.assertEqual(provider.calls, 5)

    def test_infrastructure_errors_and_interrupt_never_split(self):
        for error in (ConnectionRefusedError("offline"), TimeoutError("timed out"),
                      PermissionError("denied"), ProviderTransportError("bad HTTP"),
                      KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                class Failed:
                    calls = 0

                    def complete_json(self, system, user):
                        self.calls += 1
                        raise error

                provider = Failed()
                checkpoint = Path(tmp) / "part.json"
                output = io.StringIO()
                with self.assertRaises(type(error)):
                    _run_split(provider, _refs(4), checkpoint, progress=Progress(output))
                self.assertEqual(provider.calls, 1 if isinstance(error, KeyboardInterrupt) else 2)
                self.assertEqual(list(Path(tmp).iterdir()), [])
                self.assertTrue(output.getvalue().endswith("Interrupted: Research batch 2/93\n"
                                                         if isinstance(error, KeyboardInterrupt)
                                                         else "Failed: Research batch 2/93\n"))

    def test_planner_split_children_only_accept_refs_given_to_that_child(self):
        class LeakyProvider:
            def complete_json(self, system, user):
                payload = json.loads(user)
                if system == "compact":
                    if len(payload["records"]) > 1:
                        raise OutputLimitExceeded("length")
                    return {"capsules": [{"claim": "Grounded",
                                          "evidence_refs": ["E0001", "E0002"],
                                          "media_refs": [], "phase": "final", "confidence": "high"}]}
                fact_id = next(fact["fact_id"] for fact in payload["research"]["facts"]
                               if "E0001" in fact["evidence_refs"])
                return {"version": 1, "title": "Title", "slug": "title", "summary": "Summary",
                        "scenes": [{"id": "s001", "type": "SUMMARY", "title": "Proof",
                                    "narration": next(f["claim"] for f in payload["research"]["facts"]
                                                      if f["fact_id"] == fact_id),
                                    "evidence_refs": ["E0001"],
                                    "fact_ids": [fact_id], "annotations": [],
                                    "pad_after_seconds": 0.5, "diagram": {}, "notes": ""}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "demo"
            (root / "prompts").mkdir()
            (root / "prompts" / "storyboard.txt").write_text("storyboard")
            (root / "prompts" / "planner_compact.txt").write_text("compact")
            research_input = {"facts": [{"claim": f"Grounded {e['ref']}",
                                         "evidence_refs": [e["ref"]], "support": [{
                                             "evidence_ref": e["ref"],
                                             "text": f"Grounded {e['ref']} is established."}]}
                                        for e in _refs(2)], "assets": []}
            with patch("source2reel.planner.fits_context", side_effect=[False, True, True]), \
                 patch("source2reel.planner.split_for_context", side_effect=lambda records, *args, **kwargs: [records]):
                episode = plan(LeakyProvider(), research_input, {"evidence": _refs(2)}, project,
                               "Demo", max_retries=0)
            self.assertEqual(episode["version"], 1)
            parts = project / "manifests" / "planner-compact-parts"
            self.assertEqual(json_load(parts / "level-01-part-001-a.json")["result"]["capsules"][0]["evidence_refs"],
                             ["E0001"])
            self.assertEqual(json_load(parts / "level-01-part-001-b.json")["result"]["capsules"][0]["evidence_refs"],
                             ["E0002"])


if __name__ == "__main__":
    unittest.main()
