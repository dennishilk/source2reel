"""Exhausted narration retries recover with selected, exact research claims."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.chunking import checkpointed_split_json
from source2reel.grounding import GroundingError
from source2reel.progress import Progress
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.schema import validate_episode
from source2reel.util import json_load

from test_planner_source_agnostic import ProfileProvider, project, source


def ask_for(claims, *, visual=(), instructions=""):
    inventory, research = source(claims, visual=visual)
    ask = planner._make_ask(research, planner._media_inventory(inventory),
                            inventory["evidence"], "Grounded tool", instructions)
    return inventory, research, ask


class GroundedNarrationFallbackTests(unittest.TestCase):
    def test_multiple_facts_deduplicate_keep_order_and_skip_only_complete_claims(self):
        claims = ["A" * 1100, "B" * 200, "C" * 80, "Unselected capability."]
        _, _, ask = ask_for(claims, visual=(90,))
        scene = {"id": "s001", "type": "PROJECT_EVIDENCE",
                 "fact_ids": ["F0001", "F0002", "F0003", "F0001"],
                 "evidence_refs": ["E0001", "E0090"], "asset_ref": "E0090",
                 "purpose": "Project predicts the future.",
                 "summary": "Generated prose cannot authorize facts."}
        before = copy.deepcopy(scene)
        first = planner._grounded_narration_fallback(scene, ask, is_final=False)
        self.assertEqual(first, claims[0] + " " + claims[2])
        self.assertEqual(first, planner._grounded_narration_fallback(
            scene, ask, is_final=False,
        ))
        self.assertLessEqual(len(first), 1200)
        self.assertNotIn(claims[1], first)
        self.assertNotIn(claims[3], first)
        self.assertEqual(scene, before)

    def test_no_complete_selected_claim_can_fit_and_invalid_support_fail(self):
        _, _, ask = ask_for(["L" * 1201])
        scene = {"id": "s001", "type": "SUMMARY", "fact_ids": ["F0001"],
                 "evidence_refs": ["E0001"]}
        with self.assertRaisesRegex(GroundingError, "no complete selected grounded fact claim fits"):
            planner._grounded_narration_fallback(scene, ask, is_final=False)
        ask["research"]["facts"][0]["support"] = []
        with self.assertRaisesRegex(GroundingError, "lacks exact evidence support"):
            planner._grounded_narration_fallback(scene, ask, is_final=False)

    def test_pipeline_fails_explicitly_when_a_selected_fact_cannot_fit(self):
        inventory, research = source(["L" * 1201, "The next fact is short."])
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            with self.assertRaisesRegex(GroundingError,
                                        "no complete selected grounded fact claim fits"):
                planner.plan(ProfileProvider(), research, inventory, path, "Tool", "",
                             max_retries=0)
            self.assertFalse((path / "episode.json").exists())

    def test_direct_unsplittable_fact_does_not_restart_an_outline(self):
        inventory, research = source(["L" * 1201])

        class Direct:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                self.assert_mode = request.get("storyboard_mode")
                return {"version": 1, "title": "Tool", "scenes": [{
                    "id": "s001", "type": "SUMMARY",
                    "narration": "The tool guarantees predictions.",
                    "fact_ids": ["F0001"], "evidence_refs": ["E0001"],
                }]}

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = Direct()
            with self.assertRaisesRegex(GroundingError,
                                        "no complete selected grounded fact claim fits"):
                planner.plan(provider, research, inventory, path, "Tool", "",
                             max_retries=0)
            self.assertEqual(len(provider.calls), 1)
            self.assertIsNone(provider.assert_mode)
            self.assertFalse((path / "manifests" / "storyboard-parts" / "recovery.json").exists())

    def test_suffix_is_reserved_within_limit_and_factless_outro_is_neutral(self):
        suffix = "Exactly this ending."
        _, _, ask = ask_for(["Documented observations are local."])
        ask["required_narration_suffix"] = suffix
        scene = {"id": "s001", "type": "SUMMARY", "fact_ids": ["F0001"],
                 "evidence_refs": ["E0001"]}
        narration = planner._grounded_narration_fallback(scene, ask, is_final=True)
        self.assertEqual(narration, "Documented observations are local. " + suffix)
        self.assertEqual(narration.count(suffix), 1)
        outro = {"id": "s002", "type": "OUTRO", "fact_ids": [], "evidence_refs": []}
        self.assertEqual(planner._grounded_narration_fallback(outro, ask, is_final=True),
                         "Closing. " + suffix)
        self.assertEqual(planner._grounded_narration_fallback(outro, ask, is_final=False),
                         "Closing.")
        _, _, large = ask_for(["L" * 1190])
        large["required_narration_suffix"] = suffix
        with self.assertRaisesRegex(GroundingError, "no complete selected grounded fact claim fits"):
            planner._grounded_narration_fallback(scene, large, is_final=True)

    def test_required_suffix_is_exempt_only_for_actual_final_scene(self):
        _, _, ask = ask_for(["The tool records local observations."])
        ask["required_narration_suffix"] = "The tool guarantees future outcomes."
        scene = {"id": "s001", "type": "SUMMARY", "fact_ids": ["F0001"],
                 "evidence_refs": ["E0001"],
                 "narration": ask["required_narration_suffix"]}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            StructuredOutputError, "unsupported factual proposition",
        ):
            planner._validate_narration_grounding(
                object(), [scene], ask, Path(tmp), final_scene_id="s002",
            )

    def test_supported_natural_narration_is_kept_without_fallback(self):
        claim = "The tool records local observations."
        inventory, _, ask = ask_for([claim])
        natural = "The tool records local observations"  # Supported, distinct from exact claim.

        class Direct:
            calls = 0

            def complete_json(self, _system, _user):
                self.calls += 1
                return {"version": 1, "title": "Tool", "scenes": [{
                    "id": "s001", "type": "SUMMARY", "narration": natural,
                    "fact_ids": ["F0001"], "evidence_refs": ["E0001"],
                }]}

        provider = Direct()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "source2reel.planner._grounded_narration_fallback",
            side_effect=AssertionError("Fallback must not run for grounded model narration"),
        ):
            episode = planner._complete_episode(
                provider, "storyboard", ask,
                {entry["ref"] for entry in inventory["evidence"]}, 1,
                project_dir=Path(tmp),
            )
        self.assertEqual(episode["scenes"][0]["narration"], natural)
        self.assertEqual(provider.calls, 1)

    def test_direct_path_repairs_only_rejected_scene_after_bounded_retries(self):
        claims = ["The tool records local observations.",
                  "The tool groups those observations by process."]
        inventory, _, ask = ask_for(claims)
        natural = claims[0][:-1]

        class Direct:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                return {"version": 1, "title": "Tool", "scenes": [
                    {"id": "s001", "type": "SUMMARY", "narration": natural,
                     "fact_ids": ["F0001"], "evidence_refs": ["E0001"]},
                    {"id": "s002", "type": "SUMMARY",
                     "narration": "The tool guarantees every future process outcome.",
                     "fact_ids": ["F0002"], "evidence_refs": ["E0002"]},
                ]}

        with tempfile.TemporaryDirectory() as tmp:
            provider = Direct()
            episode = planner._complete_episode(
                provider, "storyboard", ask,
                {entry["ref"] for entry in inventory["evidence"]}, 1,
                project_dir=Path(tmp),
            )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual([scene["narration"] for scene in episode["scenes"]],
                         [natural, claims[1]])
        self.assertEqual([scene["fact_ids"] for scene in episode["scenes"]],
                         [["F0001"], ["F0002"]])

    def test_split_b_retries_then_falls_back_without_regenerating_split_a(self):
        claims = [
            "Windows Telemetry Inspector is a passive Windows diagnostics tool.",
            "ETW supplies process-scoped network observations.",
            "DNS and task links are best-effort correlation, not proven causality.",
        ]
        inventory, research = source(claims)
        hallucination = "Windows Telemetry Inspector proves the cause of every DNS event."

        class SplitProvider:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                if "checks" in request:
                    return {"decisions": [
                        {"id": item["id"], "supported": False, "propositions": []}
                        for item in request["checks"]
                    ]}
                mode = request.get("storyboard_mode")
                if mode is None:
                    raise OutputLimitExceeded("full storyboard output limit")
                if mode == "outline":
                    return {"version": 1, "title": "Windows Telemetry Inspector",
                            "slug": "windows-telemetry", "summary": claims[0],
                            "scene_intents": [
                                {"type": "SUMMARY", "purpose": claims[0],
                                 "fact_ids": ["F0001"],
                                 "evidence_refs": ["E0001"]},
                                {"type": "SUMMARY", "purpose": claims[2],
                                 "fact_ids": ["F0003"], "evidence_refs": ["E0003"]},
                                {"type": "SUMMARY", "purpose": claims[1],
                                 "fact_ids": ["F0002"], "evidence_refs": ["E0002"]},
                            ]}
                ids = tuple(item["id"] for item in request["scene_intents"])
                if ids == ("s001", "s002"):
                    raise StructuredOutputError("model response truncated")
                scenes = []
                for template in request["required_output"]["scenes"]:
                    scene = copy.deepcopy(template)
                    scene["title"] = "Documented observation"
                    scene["narration"] = hallucination if scene["id"] == "s002" else (
                        claims[0] if scene["id"] == "s001" else claims[1]
                    )
                    if scene["id"] == "s001":
                        scene["annotations"] = [claims[0]]
                    scenes.append(scene)
                return {"scenes": scenes}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = project(root)
            provider = SplitProvider()
            output = io.StringIO()
            with patch("source2reel.planner.verify_claims",
                       wraps=planner.verify_claims) as verified:
                episode = planner.plan(provider, research, inventory, path,
                                       "Windows Telemetry Inspector", "",
                                       max_retries=1, progress=Progress(output))
            self.assertFalse(any(
                item["id"] == "s002" and item["claim"] == claims[2]
                for call in verified.call_args_list for item in call.args[1]
            ))
            scenes = episode["scenes"]
            self.assertEqual([s["id"] for s in scenes], ["s001", "s002", "s003"])
            self.assertEqual([s["narration"] for s in scenes],
                             [claims[0], claims[2], claims[1]])
            self.assertEqual([s["fact_ids"] for s in scenes],
                             [["F0001"], ["F0003"], ["F0002"]])
            self.assertNotIn(hallucination, json.dumps(episode))
            requests = [tuple(item["id"] for item in call["scene_intents"])
                        for call in provider.calls if call.get("storyboard_mode") == "scenes"]
            self.assertEqual(requests.count(("s001", "s002")), 2)
            self.assertEqual(requests.count(("s001",)), 1)
            self.assertEqual(requests.count(("s002",)), 2)
            self.assertEqual(requests.count(("s003",)), 1)
            self.assertEqual(sum(call.get("storyboard_mode") == "outline"
                                 for call in provider.calls), 1)
            self.assertIn("malformed or truncated structured output; processing smaller parts",
                          output.getvalue())
            self.assertIn("narration grounding retries exhausted", output.getvalue())
            parts = path / "manifests" / "storyboard-parts"
            self.assertTrue((parts / "part-001-split.json").exists())
            self.assertEqual(json_load(parts / "part-001-a.json")["result"]["scenes"][0]["narration"],
                             claims[0])
            self.assertEqual(json_load(parts / "part-001-b.json")["result"]["scenes"][0]["narration"],
                             claims[2])
            self.assertEqual(json_load(parts / "part-002.json")["result"]["scenes"][0]["narration"],
                             claims[1])
            validate_episode(episode, {"E0001", "E0002", "E0003"},
                             require_integrated_presentation=True)
            before = len(provider.calls)
            (path / "episode.json").unlink()  # Resume from part checkpoints, not final output.
            self.assertEqual(planner.plan(provider, research, inventory, path,
                                          "Windows Telemetry Inspector", "",
                                          max_retries=1), episode)
            self.assertEqual(len(provider.calls), before)

    def test_final_factless_outro_recovers_and_keeps_exact_suffix(self):
        claim = "The tool records local events."
        inventory, research = source([claim])
        suffix = "And this is the exact end."
        rejected = "The tool guarantees perfect predictions. " + suffix

        class OutroProvider:
            def complete_json(self, _system, user):
                request = json.loads(user)
                if "checks" in request:
                    return {"decisions": [
                        {"id": item["id"], "supported": False, "propositions": []}
                        for item in request["checks"]
                    ]}
                mode = request.get("storyboard_mode")
                if mode is None:
                    raise OutputLimitExceeded("full output limit")
                if mode == "outline":
                    return {"version": 1, "title": "Tool", "slug": "tool",
                            "summary": claim, "scene_intents": [
                                {"type": "SUMMARY", "purpose": claim, "fact_ids": ["F0001"],
                                 "evidence_refs": ["E0001"]},
                                {"type": "OUTRO", "purpose": "Closing", "fact_ids": [],
                                 "evidence_refs": []},
                            ], "presentation": {"outro": {
                                "headline": ["The tool"],
                                "links": [{"label": "Project", "url": ["https://example.test/fixture"]}],
                            }}}
                scenes = []
                for template in request["required_output"]["scenes"]:
                    scene = copy.deepcopy(template)
                    scene["title"] = "Documented"
                    scene["narration"] = rejected if scene["type"] == "OUTRO" else claim
                    scenes.append(scene)
                return {"scenes": scenes}

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            episode = planner.plan(OutroProvider(), research, inventory, path, "Tool",
                                   f"End with: “{suffix}”", max_retries=1)
            final = episode["scenes"][-1]
            self.assertEqual(final["type"], "OUTRO")
            self.assertEqual(final["fact_ids"], [])
            self.assertEqual(final["narration"], "Closing. " + suffix)
            self.assertEqual(final["narration"].count(suffix), 1)
            self.assertNotIn("perfect predictions", json.dumps(episode))
            validate_episode(episode, {"E0001"}, require_integrated_presentation=True)

    def test_progress_identifies_semantic_and_structural_split_causes(self):
        class Provider:
            def complete_json(self, _system, _user):
                return {"ok": True}

        for cause, message in (
            (planner._NarrationGroundingRejected("s001", "unsupported narration"),
             "narration grounding failure; processing smaller parts"),
            (StructuredOutputError("diagram nodes invalid"),
             "schema or structural validation failure; processing smaller parts"),
        ):
            with self.subTest(cause=str(cause)), tempfile.TemporaryDirectory() as tmp:
                output = io.StringIO()
                def normalize(_value, batch):
                    if len(batch) > 1:
                        raise cause
                    return {"scenes": [batch[0]]}
                result = checkpointed_split_json(
                    Provider(), "storyboard", [{"id": "s001"}, {"id": "s002"}],
                    lambda batch: {"items": batch}, Path(tmp) / "part.json",
                    normalize, max_retries=0, progress=Progress(output),
                )
                self.assertEqual(len(result), 2)
                self.assertIn(message, output.getvalue())
                self.assertNotIn("malformed or truncated structured output",
                                 output.getvalue())
