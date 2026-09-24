"""Final storyboard output can outgrow one structured model response."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.planner import plan
from source2reel.progress import Progress
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.schema import validate_episode
from source2reel.util import json_load


INSTRUCTIONS = (
    "Explain what Source2Reel is, why it exists, and how its evidence-first local "
    "pipeline turns project sources into a finished technical explainer. Use only "
    "repository evidence and do not invent capabilities. Clearly distinguish the "
    "reusable Source2Reel engine from the Dennis Explainer production profile. "
    "End with: “And yes — this video was made with the tool it just explained.”"
)


def _source():
    inventory = {"evidence": [
        {"ref": f"E{i:04d}", "kind": "media" if i == 2 else "document",
         "relative_path": f"file-{i}.mp4" if i == 2 else f"file-{i}.md",
         "evidence_role": "primary" if i <= 4 else "generated_artifact"}
        for i in range(1, 7)
    ] + [{"ref": "E0099", "kind": "document", "relative_path": "not-researched.md",
          "evidence_role": "primary"}]}
    research = {"version": 1, "facts": [
        {"claim": f"Grounded fact {i}", "evidence_refs": [f"E{i:04d}"],
         "phase": "final", "confidence": "high"}
        for i in range(1, 7)
    ], "assets": []}
    return inventory, research


def _project(root):
    project = root / "projects" / "demo"
    (root / "prompts").mkdir()
    (root / "prompts" / "storyboard.txt").write_text("storyboard")
    return project


class StoryboardProvider:
    def __init__(self, count=7):
        self.count = count
        self.calls = []
        self.fail_part = None
        self.bad_part = None
        self.oversize_part = None
        self.full_error = OutputLimitExceeded

    def complete_json(self, system, user):
        payload = json.loads(user)
        self.calls.append(payload)
        mode = payload.get("storyboard_mode")
        if mode is None:
            raise self.full_error("complete output exceeds limit")
        if mode == "outline":
            kinds = ["HERO", "CODE", "DATA_FLOW", "PROJECT_EVIDENCE", "TIMELINE", "SUMMARY", "OUTRO"]
            refs = [["E0001"], ["E0003"], ["E0001", "E0003"], ["E0002"],
                    ["E0004", "E0005"], ["E0001"], []]
            intents = [{"type": kinds[i % len(kinds)], "purpose": f"Point {i+1}",
                        "evidence_refs": refs[i % len(refs)]}
                       for i in range(self.count)]
            return {"version": 1, "title": "Grounded project", "slug": "grounded-project",
                    "summary": "An evidence-first explanation.",
                    "presentation": {"outro": {
                        "headline": ["The project"],
                        "links": [{"label": "Project", "url": ["https://example.test/project"]}]
                    }, "scene_titles": {"s001": "Project begins"}},
                    "scene_intents": intents}
        if mode == "scenes":
            part = payload["part_number"]
            if self.fail_part == part:
                raise ConnectionRefusedError("server stopped")
            intents = payload["scene_intents"]
            if self.oversize_part == part and len(intents) > 1:
                raise OutputLimitExceeded("scene group too long")
            scenes = []
            for intent in intents:
                refs = intent["evidence_refs"]
                scene = {"id": intent["id"], "type": intent["type"],
                         "title": intent["purpose"], "narration": f"Narration {intent['id']}.",
                         "evidence_refs": refs, "notes": "", "annotations": [],
                         "pad_after_seconds": 0.5}
                if scene["type"] in {"HERO", "PROJECT_EVIDENCE", "TERMINAL_EVIDENCE", "HARDWARE_EVIDENCE"}:
                    scene["asset_ref"] = refs[0]
                if scene["type"] in {"DATA_FLOW", "TIMELINE", "ARCHITECTURE_DIAGRAM"}:
                    scene["diagram"] = {"nodes": ["Input", "Output"]}
                if scene["type"] == "PROJECT_EVIDENCE":
                    scene["media"] = {"start_seconds": 2.5}
                    scene["captions"] = {"enabled": False}
                if self.bad_part == part:
                    scene["evidence_refs"] = ["E0099"]  # Valid inventory ref, absent from this planner scope.
                scenes.append(scene)
            return {"scenes": scenes}
        raise AssertionError("Unexpected storyboard request")


class StoryboardRecoveryTests(unittest.TestCase):
    def test_map_reduce_output_limit_reuses_scoped_compaction_on_resume(self):
        class CompactThenRecover(StoryboardProvider):
            def complete_json(self, system, user):
                if system != "compact":
                    return super().complete_json(system, user)
                payload = json.loads(user)
                self.calls.append(payload)
                refs = [ref for record in payload["records"]
                        for ref in record.get("evidence_refs", [])]
                return {"capsules": [
                    {"claim": f"Scoped {ref}", "evidence_refs": [ref], "media_refs": [],
                     "phase": "final", "confidence": "high", "visual_purpose": ""}
                    for ref in refs
                ]}

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            (Path(tmp) / "prompts" / "planner_compact.txt").write_text("compact")
            provider = CompactThenRecover()
            inventory, research = _source()
            real_fits = planner.fits_context
            checks = 0

            def force_one_compaction(system, user, *args, **kwargs):
                nonlocal checks
                checks += 1
                if checks in (1, 2):
                    return checks == 2
                return real_fits(system, user, *args, **kwargs)

            with patch("source2reel.planner.fits_context", side_effect=force_one_compaction), \
                 patch("source2reel.planner.split_for_context",
                       side_effect=lambda records, *args, **kwargs: [records]):
                episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                               max_retries=0)
                compact_calls = len([call for call in provider.calls if "records" in call])
                checks = 0
                plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                     max_retries=0)
            self.assertEqual(len([call for call in provider.calls if "records" in call]), compact_calls)
            manifest = json_load(project / "manifests" / "planner-evidence.json")
            self.assertEqual(manifest["strategy"], "map-reduce")
            selected = {entry["ref"] for entry in manifest["evidence_index"]}
            self.assertNotIn("E0099", selected)
            outline_request = next(call for call in provider.calls
                                   if call.get("storyboard_mode") == "outline")
            self.assertEqual(outline_request["research"], manifest["research"])
            self.assertEqual(outline_request["evidence_index"], manifest["evidence_index"])
            for request in provider.calls:
                if request.get("storyboard_mode") == "scenes":
                    self.assertLessEqual({entry["ref"] for entry in request["evidence_index"]}, selected)
                    self.assertNotIn("E0099", {entry["ref"] for entry in request["evidence_index"]})
            validate_episode(episode, selected)

    def test_complete_episode_fast_path_stays_one_request(self):
        class FastProvider:
            def __init__(self):
                self.calls = []

            def complete_json(self, system, user):
                self.calls.append(json.loads(user))
                return {"version": 1, "title": "Fast", "slug": "fast", "summary": "Fast",
                        "scenes": [{"id": "s001", "type": "HERO", "title": "Proof",
                                    "narration": "Proven.", "evidence_refs": ["E0001"],
                                    "asset_ref": "E0001"}]}

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = FastProvider()
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS)
            self.assertEqual(episode["title"], "Fast")
            self.assertEqual(len(provider.calls), 1)
            self.assertFalse((project / "manifests" / "storyboard-parts").exists())
            validate_episode(episode, {"E0001"})

    def test_output_limit_recovers_with_ordered_scene_groups_and_preserved_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider()
            inventory, research = _source()
            progress = io.StringIO()
            episode = plan(provider, research, inventory, project, "DemoEngine",
                           INSTRUCTIONS, max_retries=0, progress=Progress(progress))
            self.assertEqual(episode["version"], 1)
            self.assertEqual((episode["title"], episode["slug"], episode["summary"]),
                             ("Grounded project", "grounded-project", "An evidence-first explanation."))
            self.assertEqual([s["id"] for s in episode["scenes"]],
                             [f"s{i:03d}" for i in range(1, 8)])
            self.assertEqual([s["title"] for s in episode["scenes"]],
                             [f"Point {i}" for i in range(1, 8)])
            self.assertEqual(episode["scenes"][3]["media"]["start_seconds"], 2.5)
            self.assertIs(episode["scenes"][3]["captions"]["enabled"], False)
            self.assertEqual(episode["scenes"][2]["diagram"]["nodes"], ["Input", "Output"])
            self.assertEqual(episode["presentation"]["outro"]["headline"], ["The project"])
            self.assertEqual(episode["presentation"]["scene_titles"]["s001"], "Project begins")
            self.assertEqual(json_load(project / "episode.json"), episode)
            validate_episode(episode, {f"E{i:04d}" for i in range(1, 7)})
            parts = project / "manifests" / "storyboard-parts"
            self.assertGreaterEqual(len(list(parts.glob("part-[0-9][0-9][0-9].json"))), 3)
            self.assertIn("Failed: Generating storyboard", progress.getvalue())
            self.assertIn("Storyboard part 1/", progress.getvalue())
            self.assertNotIn("%", progress.getvalue())
            self.assertNotIn("\x1b", progress.getvalue())
            multipart = [p for p in provider.calls if p.get("storyboard_mode")]
            self.assertTrue(all(p["project_title_hint"] == "DemoEngine" and
                                p["optional_instructions"] == INSTRUCTIONS for p in multipart))
            scene_requests = [p for p in multipart if p["storyboard_mode"] == "scenes"]
            self.assertEqual([p["contains_final_scene"] for p in scene_requests],
                             [False, False, False, True])
            self.assertTrue(all("media.start_seconds" in p["optional_structured_scene_fields"]
                                for p in scene_requests))

    def test_resumes_finished_parts_without_retrying_full_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider()
            provider.fail_part = 3
            inventory, research = _source()
            with self.assertRaises(ConnectionRefusedError):
                plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                     max_retries=0)
            self.assertFalse((project / "episode.json").exists())
            saved = project / "manifests" / "storyboard-parts"
            self.assertTrue((saved / "outline.json").exists())
            for n in (1, 2):
                self.assertTrue((saved / f"part-{n:03d}.json").exists())
            counts = lambda: (
                len([p for p in provider.calls if "storyboard_mode" not in p]),
                len([p for p in provider.calls if p.get("storyboard_mode") == "outline"]),
                len([p for p in provider.calls if p.get("part_number") in (1, 2)]),
            )
            self.assertEqual(counts(), (1, 1, 2))
            provider.fail_part = None
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertEqual(len(episode["scenes"]), 7)
            self.assertEqual(counts(), (1, 1, 2))

    def test_changed_instructions_and_scope_invalidate_parts_and_remove_stale_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=7)
            inventory, research = _source()
            plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS, max_retries=0)
            first = len([p for p in provider.calls if p.get("storyboard_mode") == "scenes"])
            provider.count = 3
            changed = INSTRUCTIONS + " Add a short ending."
            plan(provider, research, inventory, project, "DemoEngine", changed, max_retries=0)
            second = [p for p in provider.calls if p.get("storyboard_mode") == "scenes"]
            self.assertGreater(len(second), first)
            self.assertTrue(all(p["optional_instructions"] == changed for p in second[first:]))
            part_dir = project / "manifests" / "storyboard-parts"
            self.assertEqual(sorted(p.name for p in part_dir.glob("part-[0-9][0-9][0-9].json")),
                             ["part-001.json", "part-002.json"])
            # Even if the outline model chooses the same intents, a changed
            # final planner scope must invalidate existing scene checkpoints.
            provider.count = 3
            prior = len(second)
            inventory["evidence"].append({"ref": "E0007", "kind": "document",
                                          "relative_path": "new-primary.md", "evidence_role": "primary"})
            research["facts"].append({"claim": "New source", "evidence_refs": ["E0007"]})
            plan(provider, research, inventory, project, "DemoEngine", changed, max_retries=0)
            self.assertGreater(len([p for p in provider.calls if p.get("storyboard_mode") == "scenes"]), prior)

    def test_truncated_scene_part_splits_only_that_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=6)
            provider.oversize_part = 1
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertEqual([s["id"] for s in episode["scenes"]],
                             [f"s{i:03d}" for i in range(1, 7)])
            part_dir = project / "manifests" / "storyboard-parts"
            self.assertTrue((part_dir / "part-001-a.json").exists())
            self.assertTrue((part_dir / "part-001-b.json").exists())
            self.assertTrue((part_dir / "part-002.json").exists())

    def test_invalid_part_cannot_cite_globally_valid_out_of_scope_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=6)
            provider.bad_part = 2
            inventory, research = _source()
            with self.assertRaisesRegex((ValueError, RuntimeError), "E0099|invalid structured output"):
                plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                     max_retries=0)
            self.assertFalse((project / "episode.json").exists())

    def test_missing_scene_and_misordered_ids_cannot_silently_assemble(self):
        for broken in ("missing", "reordered"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as tmp:
                project = _project(Path(tmp))

                class BrokenPart(StoryboardProvider):
                    def complete_json(self, system, user):
                        payload = json.loads(user)
                        result = super().complete_json(system, user)
                        if payload.get("storyboard_mode") == "scenes" and payload["part_number"] == 2:
                            if broken == "missing":
                                result["scenes"] = result["scenes"][:-1]
                            else:
                                result["scenes"][0]["id"] = "s999"
                        return result

                provider = BrokenPart(count=6)
                inventory, research = _source()
                with self.assertRaisesRegex(RuntimeError, "invalid structured output"):
                    plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                         max_retries=0)
                self.assertFalse((project / "episode.json").exists())

    def test_malformed_full_model_output_recovers_but_fast_path_ref_scope_remains_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=3)
            provider.full_error = StructuredOutputError
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertEqual(len(episode["scenes"]), 3)

        class LeakyFull(StoryboardProvider):
            def complete_json(self, system, user):
                payload = json.loads(user)
                if payload.get("storyboard_mode") is None:
                    self.calls.append(payload)
                    return {"version": 1, "title": "Leaky", "scenes": [{
                        "id": "s001", "type": "PROJECT_EVIDENCE", "narration": "Wrong proof.",
                        "evidence_refs": ["E0099"], "asset_ref": "E0099",
                    }]}
                return super().complete_json(system, user)

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = LeakyFull(count=3)
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertNotIn("E0099", {r for scene in episode["scenes"]
                                        for r in scene["evidence_refs"]})
            self.assertTrue((project / "manifests" / "storyboard-parts" / "outline.json").exists())

    def test_network_failure_does_not_start_multipart_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider()
            provider.full_error = ConnectionRefusedError
            inventory, research = _source()
            with self.assertRaises(ConnectionRefusedError):
                plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                     max_retries=0)
            self.assertEqual(len(provider.calls), 1)
            self.assertFalse((project / "manifests" / "storyboard-parts").exists())


if __name__ == "__main__":
    unittest.main()
