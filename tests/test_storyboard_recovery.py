"""Final storyboard output can outgrow one structured model response."""
from __future__ import annotations

import copy
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
        {"ref": f"E{i:04d}", "kind": "media" if i in (1, 2) else "document",
         "relative_path": f"file-{i}.mp4" if i == 2 else "file-1.png" if i == 1 else f"file-{i}.md",
         "evidence_role": "primary" if i <= 4 else "generated_artifact"}
        for i in range(1, 7)
    ] + [{"ref": "E0099", "kind": "document", "relative_path": "not-researched.md",
          "evidence_role": "primary"}]}
    research = {"version": 1, "facts": [
        {"claim": f"Grounded fact {i}", "evidence_refs": [f"E{i:04d}"],
         "support": [{"evidence_ref": f"E{i:04d}", "text": f"Grounded fact {i}"}],
         "phase": "final", "confidence": "high"}
        for i in range(1, 7)
    ], "assets": []}
    return inventory, research


def _source_with_two_ref_hero():
    inventory, research = _source()
    inventory["evidence"].append({"ref": "E0020", "kind": "media",
                                  "relative_path": "second-proof.png", "evidence_role": "primary"})
    research["facts"].append({"claim": "Grounded fact 20", "evidence_refs": ["E0020"],
                              "support": [{"evidence_ref": "E0020", "text": "Grounded fact 20"}],
                              "phase": "final", "confidence": "high"})
    return inventory, research


def _project(root):
    project = root / "projects" / "demo"
    (root / "prompts").mkdir()
    (root / "prompts" / "storyboard.txt").write_text("storyboard")
    (project / "sources").mkdir(parents=True)
    (project / "sources" / "source.json").write_text(json.dumps({
        "kind": "website", "source": "https://example.test/project"
    }))
    return project


class StoryboardProvider:
    def __init__(self, count=7):
        self.count = count
        self.calls = []
        self.fail_part = None
        self.bad_part = None
        self.oversize_part = None
        self.full_error = OutputLimitExceeded
        self.first_refs = None

    def complete_json(self, system, user):
        payload = json.loads(user)
        self.calls.append(payload)
        mode = payload.get("storyboard_mode")
        if mode is None:
            raise self.full_error("complete output exceeds limit")
        if mode == "outline":
            kinds = ["HERO", "SUMMARY", "SUMMARY", "PROJECT_EVIDENCE", "SUMMARY",
                     "SUMMARY", "OUTRO"]
            refs = [self.first_refs or ["E0001"], ["E0003"], ["E0001", "E0003"],
                    ["E0002"], ["E0004", "E0005"], ["E0006"], []]
            requested_intent = payload["required_output"]["scene_intents"][0]
            evidence_types = payload.get("scene_type_requirements", {}).get(
                "asset_ref_required_types", ())
            supplied_facts = payload["research"]["facts"]
            intents = []
            for i in range(self.count):
                chosen_refs = refs[i % len(refs)]
                purpose = next((fact["claim"] for fact in supplied_facts
                                if set(fact["evidence_refs"]) & set(chosen_refs)), "Closing")
                intent = {"type": kinds[i % len(kinds)], "purpose": purpose,
                          "evidence_refs": chosen_refs}
                if "fact_ids" in requested_intent:
                    intent["fact_ids"] = [fact["fact_id"] for fact in supplied_facts
                                          if set(fact["evidence_refs"]) & set(intent["evidence_refs"])]
                if intent["type"] in evidence_types and "asset_ref" in requested_intent:
                    intent["asset_ref"] = intent["evidence_refs"][-1]
                intents.append(intent)
            presentation = {"scene_titles": {"s001": "Project begins"}}
            if intents[-1]["type"] == "OUTRO":
                presentation["outro"] = {
                    "headline": ["The project"],
                    "links": [{"label": "Project", "url": [payload["authoritative_resource_urls"][0]]}]
                }
            return {"version": 1, "title": "Grounded project", "slug": "grounded-project",
                    "summary": supplied_facts[0]["claim"], "presentation": presentation,
                    "scene_intents": intents}
        if mode == "scenes":
            part = payload["part_number"]
            if self.fail_part == part:
                raise ConnectionRefusedError("server stopped")
            intents = payload["scene_intents"]
            if self.oversize_part == part and len(intents) > 1:
                raise OutputLimitExceeded("scene group too long")
            scenes = []
            for intent, requested in zip(intents, payload["required_output"]["scenes"]):
                scene = copy.deepcopy(requested)
                scene["title"] = f"Point {int(intent['id'][1:])}"
                chosen = {fact["fact_id"]: fact["claim"] for fact in payload["research"]["facts"]}
                scene["narration"] = (" ".join(chosen[fact_id] for fact_id in intent["fact_ids"])
                                      or "Closing.")
                if payload["contains_final_scene"] and intent == intents[-1] and payload["required_narration_suffix"]:
                    scene["narration"] += " " + payload["required_narration_suffix"]
                if "diagram" in scene:
                    claims = [fact["claim"] for fact in payload["research"]["facts"]
                              if set(fact["evidence_refs"]) & set(intent["evidence_refs"])]
                    scene["diagram"]["nodes"] = claims[:2]
                optional = payload.get("optional_scene_fields", {})
                if "annotations" in optional:
                    scene["annotations"] = []
                if "notes" in optional:
                    scene["notes"] = ""
                if "pad_after_seconds" in optional:
                    scene["pad_after_seconds"] = 0.5
                if (scene.get("asset_ref") in {m["ref"] for m in payload["media_inventory"]} and
                        "media.start_seconds" in payload["optional_structured_scene_fields"]):
                    scene["media"] = {"start_seconds": 2.5}
                    if "captions.enabled" in payload["optional_structured_scene_fields"]:
                        scene["captions"] = {"enabled": False}
                if self.bad_part == part:
                    scene["evidence_refs"] = ["E0099"]  # Valid inventory ref, absent from this planner scope.
                scenes.append(scene)
            return {"scenes": scenes}
        raise AssertionError("Unexpected storyboard request")


class StoryboardRecoveryTests(unittest.TestCase):
    def test_scene_part_uses_fixed_outline_facts_when_model_changes_fact_ids(self):
        inventory, research = _source()
        ask = planner._make_ask(research, planner._media_inventory(inventory),
                                inventory["evidence"], "DemoEngine", "")

        class WrongSceneFacts:
            def __init__(self, leak_refs):
                self.scene_calls = 0
                self.leak_refs = leak_refs

            def complete_json(self, _system, user):
                request = json.loads(user)
                if request.get("storyboard_mode") == "outline":
                    return {"version": 1, "title": "Grounded", "slug": "grounded",
                            "summary": "Grounded fact 3 Grounded fact 4",
                            "scene_intents": [
                                {"type": "SUMMARY", "purpose": f"Grounded fact {number}",
                                 "fact_ids": [f"F{number:04d}"],
                                 "evidence_refs": [f"E{number:04d}"]}
                                for number in (3, 4)]}
                if request.get("storyboard_mode") == "scenes":
                    self.scene_calls += 1
                    scenes = []
                    for template in request["required_output"]["scenes"]:
                        scene = dict(template)
                        scene.update(title="Grounded", narration="Grounded fact 3")
                        if scene["id"] == "s002":
                            scene["fact_ids"] = ["F0003"]
                            if self.leak_refs:
                                scene["evidence_refs"] = ["E0003"]
                        scenes.append(scene)
                    return {"scenes": scenes}
                raise AssertionError("Unexpected request")

        for leak_refs in (False, True):
            with self.subTest(leak_refs=leak_refs), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(planner, "fits_context", return_value=True):
                project = Path(tmp) / "projects" / "demo"
                project.mkdir(parents=True)
                provider = WrongSceneFacts(leak_refs)
                episode = planner._multipart_episode(
                    provider, "storyboard", ask, project, 8192, 4096, 1024, 1, None, "scope")
                self.assertEqual(episode["scenes"][1]["fact_ids"], ["F0004"])
                self.assertEqual(episode["scenes"][1]["evidence_refs"], ["E0004"])
                self.assertEqual(episode["scenes"][1]["narration"], "Grounded fact 4")
                self.assertEqual(episode["scenes"][1]["title"],
                                 "Grounded fact 4" if leak_refs else "Grounded")
                first_calls = provider.scene_calls
                self.assertEqual(planner._multipart_episode(
                    provider, "storyboard", ask, project, 8192, 4096, 1024, 1, None, "scope"),
                    episode)
                self.assertEqual(provider.scene_calls, first_calls)

    def test_full_episode_repairs_only_selected_asset_ref_relationship(self):
        inventory, research = _source()
        ask = planner._make_ask(research, [], inventory["evidence"], "DemoEngine", "")

        class FullProvider:
            def __init__(self, refs, asset="E0001", fact_ids=None):
                self.refs, self.asset, self.fact_ids = refs, asset, fact_ids or ["F0001"]
                self.calls = 0

            def complete_json(self, _system, _user):
                self.calls += 1
                scene = {"id": "s001", "type": "HERO", "narration": "Grounded fact 1",
                         "asset_ref": self.asset, "fact_ids": self.fact_ids}
                if self.refs is not ...:
                    scene["evidence_refs"] = self.refs
                return {"version": 1, "title": "Grounded", "scenes": [scene]}

        for refs in (..., None, []):
            with self.subTest(refs=refs):
                provider = FullProvider(refs)
                episode = planner._complete_episode(provider, "storyboard", ask,
                                                    {"E0001", "E0002"}, 0)
                self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0001"])
                self.assertEqual(provider.calls, 1)
        episode = planner._complete_episode(FullProvider(["E0002", "E0002"],
                                                          fact_ids=["F0001", "F0002"]),
                                            "storyboard", ask, {"E0001", "E0002"}, 0)
        self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0002", "E0001"])
        for refs in (..., ["E0001"]):
            with self.subTest(invalid_refs=refs):
                with self.assertRaisesRegex(ValueError, "evidence_refs|asset_ref"):
                    planner._complete_episode(FullProvider(refs, "E0099"), "storyboard", ask,
                                              {"E0001", "E0002"}, 0)
        with self.assertRaisesRegex(ValueError, "asset_ref"):
            planner._complete_episode(FullProvider(..., "E0002"), "storyboard", ask,
                                      {"E0001", "E0002"}, 0)
        with self.assertRaisesRegex(ValueError, "evidence_refs"):
            planner._complete_episode(FullProvider("E0001"), "storyboard", ask,
                                      {"E0001", "E0002"}, 0)

    def test_unsupported_outline_purpose_is_replaced_without_outline_retry(self):
        class BadFirst(StoryboardProvider):
            bad = True

            def complete_json(self, system, user):
                request = json.loads(user)
                result = super().complete_json(system, user)
                if request.get("storyboard_mode") == "outline" and self.bad:
                    result["scene_intents"][0]["purpose"] = (
                        "Turn project sources into finished technical documentaries."
                    )
                    self.bad = False
                return result

        for retry in (False, True):
            with self.subTest(retry=retry), tempfile.TemporaryDirectory() as tmp:
                provider = BadFirst(count=3)
                project = _project(Path(tmp))
                inventory, research = _source()
                episode = plan(provider, research, inventory, project, "DemoEngine",
                               INSTRUCTIONS, max_retries=int(retry))
                self.assertEqual(episode["scenes"][0]["id"], "s001")
                modes = [call.get("storyboard_mode") for call in provider.calls]
                self.assertEqual(modes.count("outline"), 1)
                self.assertIn("scenes", modes)
                self.assertEqual(json_load(project / "manifests" / "storyboard-parts" /
                                           "outline.json")["result"]["scene_intents"][0]["purpose"],
                                 "Grounded fact 1")

    def test_cached_outline_metadata_is_recanonicalized_even_with_current_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=3)
            inventory, research = _source()
            plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                 max_retries=0)
            path = project / "manifests" / "storyboard-parts" / "outline.json"
            saved = json_load(path)
            saved["result"]["summary"] = "AI enrichment provides guaranteed results."
            saved["result"]["scene_intents"][0]["purpose"] = (
                "The project predicts every future result."
            )
            path.write_text(json.dumps(saved))
            before = len([call for call in provider.calls if call.get("storyboard_mode") == "outline"])
            plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                 max_retries=0)
            after = len([call for call in provider.calls if call.get("storyboard_mode") == "outline"])
            self.assertEqual(after, before)
            self.assertEqual(json_load(path)["result"]["summary"],
                             "Grounded fact 1 Grounded fact 3")
            self.assertEqual(json_load(path)["result"]["scene_intents"][0]["purpose"],
                             "Grounded fact 1")

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
                request = json.loads(user)
                self.calls.append(request)
                return {"version": 1, "title": "Fast", "slug": "fast", "summary": "Fast",
                        "scenes": [{"id": "s001", "type": "HERO", "title": "Proof",
                                    "narration": request["research"]["facts"][0]["claim"] + " " +
                                                 request["required_narration_suffix"],
                                    "fact_ids": [request["research"]["facts"][0]["fact_id"]],
                                    "evidence_refs": ["E0001"],
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
                             ("Grounded project", "grounded-project", "Grounded fact 1"))
            self.assertEqual([s["id"] for s in episode["scenes"]],
                             [f"s{i:03d}" for i in range(1, 8)])
            self.assertEqual([s["title"] for s in episode["scenes"]],
                             [f"Point {i}" for i in range(1, 8)])
            self.assertEqual(episode["scenes"][3]["media"]["start_seconds"], 2.5)
            self.assertIs(episode["scenes"][3]["captions"]["enabled"], False)
            self.assertNotIn("diagram", episode["scenes"][2])
            self.assertEqual(episode["scenes"][0]["annotations"], [])
            self.assertEqual(episode["scenes"][0]["notes"], "")
            self.assertEqual(episode["scenes"][0]["pad_after_seconds"], 0.5)
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

    def test_contract_only_provider_recovers_multi_ref_hero_split_and_all_diagram_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider()
            provider.first_refs = ["E0001", "E0020"]
            provider.oversize_part = 1
            inventory, research = _source_with_two_ref_hero()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertEqual([s["id"] for s in episode["scenes"]],
                             [f"s{i:03d}" for i in range(1, 8)])
            self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0001", "E0020"])
            self.assertEqual(episode["scenes"][0]["asset_ref"], "E0020")
            self.assertTrue(episode["scenes"][-1]["narration"].endswith(
                "And yes — this video was made with the tool it just explained."
            ))
            self.assertEqual(episode["presentation"]["outro"]["headline"], ["The project"])
            for scene in episode["scenes"]:
                if "diagram" in scene:
                    self.assertEqual(len(scene["diagram"]["nodes"]), 2)
            validate_episode(episode, {f"E{i:04d}" for i in range(1, 7)} | {"E0020"},
                             require_integrated_presentation=True)
            requests = [call for call in provider.calls if call.get("storyboard_mode") == "scenes"]
            self.assertGreaterEqual(len({r["part_number"] for r in requests}), 3)
            parent, child = [r for r in requests if r["part_number"] == 1][:2]
            self.assertEqual(len(parent["scene_intents"]), 2)
            self.assertEqual(len(child["scene_intents"]), 1)
            for request in (parent, child):
                self.assertEqual(request["scene_intents"][0]["asset_ref"], "E0020")
                self.assertEqual(request["required_output"]["scenes"][0]["asset_ref"], "E0020")
                self.assertIn("E0020", {e["ref"] for e in request["evidence_index"]})
            for request in requests:
                for template in request["required_output"]["scenes"]:
                    if template["type"] in {"DATA_FLOW", "TIMELINE", "ARCHITECTURE_DIAGRAM"}:
                        self.assertEqual(len(template["diagram"]["nodes"]), 2)
                        self.assertIn("2–8", request["scene_type_requirements"]["diagram_nodes"])
                    elif template["type"] in {"CODE", "OUTRO"}:
                        self.assertNotIn("diagram", template)
                        self.assertNotIn("asset_ref", template)
            part_dir = project / "manifests" / "storyboard-parts"
            for filename in ("outline.json", "part-001-split.json", "part-001-a.json",
                             "part-001-b.json", "part-002.json", "part-003.json"):
                self.assertTrue((part_dir / filename).exists(), filename)
            first_calls = len(provider.calls)
            self.assertEqual(plan(provider, research, inventory, project, "DemoEngine",
                                  INSTRUCTIONS, max_retries=0), episode)
            self.assertEqual(len(provider.calls), first_calls)

    def test_invalid_outline_asset_and_returned_asset_fail_precisely(self):
        inventory, research = _source_with_two_ref_hero()

        for bad in ("missing", "out_of_planner_scope", "not_in_intent_refs"):
            with self.subTest(bad_outline=bad), tempfile.TemporaryDirectory() as tmp:
                class BadOutline(StoryboardProvider):
                    def complete_json(self, system, user):
                        request = json.loads(user)
                        result = super().complete_json(system, user)
                        if request.get("storyboard_mode") == "outline":
                            first = result["scene_intents"][0]
                            if bad == "missing":
                                first.pop("asset_ref", None)
                            else:
                                first["asset_ref"] = "E0099" if bad == "out_of_planner_scope" else "E0003"
                        return result

                provider = BadOutline(count=3)
                provider.first_refs = ["E0001", "E0020"]
                project = _project(Path(tmp))
                with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                    plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                         max_retries=0)
                self.assertFalse((project / "episode.json").exists())

        for bad in ("outside_part", "uncited", "different", "missing"):
            with self.subTest(bad_scene=bad), tempfile.TemporaryDirectory() as tmp:
                class BadScene(StoryboardProvider):
                    def complete_json(self, system, user):
                        request = json.loads(user)
                        result = super().complete_json(system, user)
                        if request.get("storyboard_mode") == "scenes" and request["part_number"] == 1:
                            scene = result["scenes"][0]
                            if bad == "outside_part":
                                scene["asset_ref"] = "E0099"
                            elif bad == "uncited":
                                scene["evidence_refs"] = ["E0001"]
                            elif bad == "different":
                                scene["asset_ref"] = "E0001"
                            else:
                                scene.pop("asset_ref", None)
                        return result

                provider = BadScene(count=3)
                provider.first_refs = ["E0001", "E0020"]
                project = _project(Path(tmp))
                if bad in {"missing", "uncited"}:
                    episode = plan(provider, research, inventory, project, "DemoEngine",
                                   INSTRUCTIONS, max_retries=0)
                    self.assertEqual(episode["scenes"][0]["asset_ref"], "E0020")
                    self.assertIn("E0020", episode["scenes"][0]["evidence_refs"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "asset_ref"):
                        plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                             max_retries=0)
                    self.assertFalse((project / "episode.json").exists())

    def test_malformed_diagram_is_rejected_before_assembly(self):
        from test_storyboard_editorial import physical_ask, intent, outline
        ask, allowed = physical_ask()
        planned = outline([intent("DATA_FLOW", 2)])
        planned.pop("presentation")
        normalized = planner._normalize_outline(planned, allowed, ask, check_coverage=False)
        for bad in ("missing", "too_few_nodes", "unlabeled"):
            with self.subTest(bad=bad):
                scene = {"id": "s001", "type": "DATA_FLOW", "title": "ETW flow",
                         "narration": ask["research"]["facts"][1]["claim"],
                         "fact_ids": ["F0002"], "evidence_refs": ["E0003"]}
                if bad == "too_few_nodes":
                    scene["diagram"] = {"nodes": ["one label"]}
                elif bad == "unlabeled":
                    scene["diagram"] = {"nodes": ["", "label"]}
                with self.assertRaisesRegex(StructuredOutputError, "diagram nodes|labeled diagram"):
                    planner._normalize_scene_part({"scenes": [scene]},
                        normalized["scene_intents"], allowed, normalized, ask)

    def test_stale_multipart_contract_checkpoints_refresh_without_full_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=3)
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine",
                           INSTRUCTIONS, max_retries=0)
            part_dir = project / "manifests" / "storyboard-parts"
            for name in ("outline.json", "part-001.json"):
                path = part_dir / name
                checkpoint = json_load(path)
                checkpoint["input_sha256"] = "previous multipart contract"
                path.write_text(json.dumps(checkpoint))
            counts = lambda: (
                len([p for p in provider.calls if "storyboard_mode" not in p]),
                len([p for p in provider.calls if p.get("storyboard_mode") == "outline"]),
                len([p for p in provider.calls if p.get("storyboard_mode") == "scenes"]),
            )
            self.assertEqual(counts(), (1, 1, 2))
            self.assertEqual(plan(provider, research, inventory, project, "DemoEngine",
                                  INSTRUCTIONS, max_retries=0), episode)
            self.assertEqual(counts(), (1, 2, 3))
            self.assertTrue((part_dir / "recovery.json").exists())

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
            research["facts"].append({"claim": "New source", "evidence_refs": ["E0007"],
                                      "support": [{"evidence_ref": "E0007", "text": "New source"}]})
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

    def test_out_of_scope_scene_ref_is_discarded_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            provider = StoryboardProvider(count=6)
            provider.bad_part = 2
            inventory, research = _source()
            episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                           max_retries=0)
            self.assertNotIn("E0099", {ref for scene in episode["scenes"]
                                        for ref in scene["evidence_refs"]})
            part = json_load(project / "manifests" / "storyboard-parts" / "part-002.json")
            self.assertNotIn("E0099", str(part["result"]))

    def test_unsplittable_single_scene_structure_recovers_from_fixed_outline(self):
        for broken in ("missing", "wrong_id", "wrong_type", "duplicate", "wrapper"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as tmp:
                project = _project(Path(tmp))

                class BrokenPart(StoryboardProvider):
                    def complete_json(self, system, user):
                        payload = json.loads(user)
                        result = super().complete_json(system, user)
                        if payload.get("storyboard_mode") == "scenes" and payload["part_number"] == 2:
                            if broken == "missing":
                                result["scenes"] = result["scenes"][:-1]
                            elif broken == "wrong_id":
                                result["scenes"][0]["id"] = "s999"
                            elif broken == "wrong_type":
                                result["scenes"][0]["type"] = "OUTRO"
                            elif broken == "wrapper":
                                return {"scenes": result["scenes"],
                                        "untrusted_prose": "Invented verification guarantee"}
                            elif len(payload["scene_intents"]) == 1:
                                result["scenes"].append(copy.deepcopy(result["scenes"][0]))
                            if result["scenes"]:
                                result["scenes"][0]["narration"] = "Invented verification guarantee"
                                result["scenes"][0]["evidence_refs"] = ["E0099"]
                        return result

                provider = BrokenPart(count=6)
                inventory, research = _source()
                episode = plan(provider, research, inventory, project, "DemoEngine", INSTRUCTIONS,
                               max_retries=0)
                self.assertEqual([scene["id"] for scene in episode["scenes"]],
                                 [f"s{i:03d}" for i in range(1, 7)])
                self.assertTrue((project / "episode.json").exists())
                saved = project / "manifests" / "storyboard-parts"
                serialized = " ".join(
                    path.read_text() for path in saved.glob("part-*.json") if path.is_file()
                )
                self.assertNotIn("s999", serialized)
                self.assertNotIn("Invented verification guarantee", serialized)
                self.assertNotIn("E0099", serialized)
                intents = json_load(saved / "outline.json")["result"]["scene_intents"]
                fixed = {intent["id"]: intent for intent in intents}
                for scene in episode["scenes"]:
                    intent = fixed[scene["id"]]
                    self.assertEqual(scene["fact_ids"], intent["fact_ids"])
                    self.assertEqual(scene["evidence_refs"], intent["evidence_refs"])
                    self.assertEqual(scene.get("asset_ref"), intent.get("asset_ref"))
                    self.assertNotIn("Invented verification guarantee", scene["narration"])

    def test_structural_recovery_uses_grounded_diagram_labels_and_refuses_graph_points(self):
        cases = [
            ("GRAPH", ["The measured series records x 1 with y 10 and x 2 with y 20."]),
            ("DATA_FLOW", ["The collector reads records and then normalizes the records.",
                           "The renderer receives normalized records and displays the results."]),
        ]
        for kind, claims in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(planner, "fits_context", return_value=True):
                facts = [{"claim": claim, "evidence_refs": ["E0003"],
                          "support": [{"evidence_ref": "E0003", "text": claim}],
                          "phase": "final", "confidence": "high"} for claim in claims]
                inventory = [{"ref": "E0003", "kind": "document", "evidence_role": "primary",
                              "relative_path": "readme.md"}]
                ask = planner._make_ask({"version": 1, "facts": facts, "assets": []},
                                        [], inventory, "Demo", "")

                class BrokenDiagram:
                    def complete_json(self, _system, user):
                        payload = json.loads(user)
                        if payload.get("storyboard_mode") == "outline":
                            return {"version": 1, "title": "Demo", "slug": "demo",
                                    "summary": claims[0], "scene_intents": [{
                                        "type": kind, "purpose": claims[0],
                                        "fact_ids": [f"F{i:04d}" for i in range(1, len(claims) + 1)],
                                        "evidence_refs": ["E0003"],
                                    }]}
                        if payload.get("storyboard_mode") == "scenes":
                            return {"scenes": []}
                        raise AssertionError("Unexpected model call")

                run = lambda: planner._multipart_episode(
                    BrokenDiagram(), "storyboard", ask, Path(tmp), 8192, 4096, 1024,
                    0, None, "scope")
                if kind == "GRAPH":
                    with self.assertRaisesRegex(RuntimeError, "invalid structured output"):
                        run()
                else:
                    episode = run()
                    self.assertEqual(episode["scenes"][0]["type"], kind)
                    self.assertEqual(episode["scenes"][0]["diagram"]["nodes"], claims)
                    self.assertEqual(episode["scenes"][0]["fact_ids"], ["F0001", "F0002"])

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
