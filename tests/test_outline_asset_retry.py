"""Outline retries must name the actual scoped visual choice."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.providers import StructuredOutputError


def _ask():
    index = [
        {"ref": "E0001", "kind": "document", "relative_path": "README.md",
         "evidence_role": "primary"},
        {"ref": "E0982", "kind": "media", "relative_path": "boringfetch.webp",
         "evidence_role": "primary"},
    ]
    research = {"version": 1, "facts": [{
        "claim": "BoringOS has a kernel.", "evidence_refs": ["E0001"],
        "support": [{"evidence_ref": "E0001", "text": "BoringOS has a kernel."}],
        "phase": "final", "confidence": "high",
    }], "assets": [{"evidence_ref": "E0982", "purpose": "BoringOS terminal preview",
                   "authentic_project_media": True}]}
    media = [{"ref": "E0982", "kind": "media", "relative_path": "boringfetch.webp",
              "evidence_role": "primary"}]
    return planner._make_ask(research, media, index, "BoringOS", "Explain BoringOS.")


class OutlineAssetRetryTests(unittest.TestCase):
    def test_null_hero_asset_selects_only_unique_authentic_primary_visual(self):
        ask = _ask()
        raw = {"type": "HERO", "purpose": "Introduce BoringOS",
               "fact_ids": ["F0001"], "evidence_refs": ["E0001"], "asset_ref": None}
        intent = planner._canonical_outline_intent(raw, 1, {"E0001", "E0982"}, ask)
        self.assertEqual(intent["asset_ref"], "E0982")
        self.assertEqual(intent["evidence_refs"], ["E0001", "E0982"])

        for name, change in (
            ("not-authentic", lambda value: value["research"]["assets"][0].pop("authentic_project_media")),
            ("non-primary", lambda value: value["evidence_index"][1].update(evidence_role="supporting")),
            ("ambiguous", lambda value: value["research"]["assets"].append({
                "evidence_ref": "E0984", "authentic_project_media": True})),
        ):
            with self.subTest(case=name):
                other = copy.deepcopy(ask)
                change(other)
                if name == "ambiguous":
                    other["evidence_index"].append({"ref": "E0984", "kind": "media",
                                                    "relative_path": "files.webp", "evidence_role": "primary"})
                    other["visual_asset_refs"].append("E0984")
                with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                    planner._canonical_outline_intent(raw, 1,
                        {item["ref"] for item in other["evidence_index"]}, other)

        with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
            planner._canonical_outline_intent({**raw, "asset_ref": "E0983"}, 1,
                                              {"E0001", "E0982"}, ask)

    def test_invalid_asset_feedback_names_only_the_eligible_visual(self):
        ask = _ask()
        raw = {"type": "HERO", "purpose": "Introduce BoringOS with its preview",
               "fact_ids": ["F0001"], "evidence_refs": ["E0001", "E0983"],
               "asset_ref": "E0983"}
        with self.assertRaises(StructuredOutputError) as caught:
            planner._canonical_outline_intent(raw, 1, {"E0001", "E0982"}, ask)
        feedback = planner._retry_feedback(caught.exception)
        self.assertIn("E0982", feedback)
        self.assertNotIn("E0983", feedback)
        # The rejected ref is never added to the actual scope.
        self.assertEqual(ask["visual_asset_refs"], ["E0982"])

    def test_final_fit_reserves_outline_retry_feedback_space(self):
        ask = _ask()
        seen = []

        def fits(_system, user, *_budget):
            payload = json.loads(user)
            seen.append(payload)
            return payload.get("storyboard_mode") != "outline" or bool(
                payload.get("validation_feedback"))

        with patch.object(planner, "fits_context", side_effect=fits):
            self.assertTrue(planner._final_requests_fit("storyboard", ask, 8192, 4096, 1024))
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(seen[1]["validation_feedback"]), 48)

    def test_single_research_asset_is_named_in_bounded_outline_retry(self):
        ask = _ask()

        class Provider:
            def __init__(self, always_null=False):
                self.outline_requests = []
                self.always_null = always_null

            def complete_json(self, _system, user):
                payload = json.loads(user)
                if payload.get("storyboard_mode") == "outline":
                    self.outline_requests.append(payload)
                    selected = (None if self.always_null else
                                "E0982" if "E0982" in payload.get("validation_feedback", "")
                                else "E0983")
                    return {"version": 1, "title": "BoringOS", "slug": "boringos",
                            "summary": "BoringOS has a kernel.", "scene_intents": [{
                                "type": "HERO", "purpose": "BoringOS has a kernel.",
                                "fact_ids": ["F0001"],
                                "evidence_refs": ["E0001"] + ([selected] if selected else []),
                                "asset_ref": selected,
                            }]}
                if payload.get("storyboard_mode") == "scenes":
                    scene = dict(payload["required_output"]["scenes"][0])
                    scene.update(title="BoringOS", narration="BoringOS has a kernel.")
                    return {"scenes": [scene]}
                raise AssertionError("Unexpected request")

        def narrow_context(_system, user, *_budget):
            payload = json.loads(user)
            return (payload.get("storyboard_mode") != "outline" or
                    len(payload.get("validation_feedback", "")) <= 48)

        provider = Provider()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(planner, "fits_context", side_effect=narrow_context), \
             patch.object(planner, "_canonicalize_outline_metadata", side_effect=lambda _p, value, *_args: value), \
             patch.object(planner, "_validate_narration_grounding"), \
             patch.object(planner, "_validate_structured_grounding"):
            project = Path(tmp) / "projects" / "boringos"
            project.mkdir(parents=True)
            episode = planner._multipart_episode(
                provider, "storyboard", ask, project, 8192, 4096, 1024, 1, None,
                "scope",
            )
        self.assertEqual(episode["scenes"][0]["asset_ref"], "E0982")
        self.assertEqual(len(provider.outline_requests), 2)
        self.assertNotIn("validation_feedback", provider.outline_requests[0])
        self.assertIn("E0982", provider.outline_requests[1]["validation_feedback"])
        self.assertLessEqual(len(provider.outline_requests[1]["validation_feedback"]), 48)

        stubborn = Provider(always_null=True)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(planner, "fits_context", return_value=True), \
             patch.object(planner, "_canonicalize_outline_metadata", side_effect=lambda _p, value, *_args: value), \
             patch.object(planner, "_validate_narration_grounding"), \
             patch.object(planner, "_validate_structured_grounding"):
            project = Path(tmp) / "projects" / "boringos"
            project.mkdir(parents=True)
            episode = planner._multipart_episode(
                stubborn, "storyboard", ask, project, 8192, 4096, 1024, 2, None,
                "scope",
            )
        self.assertEqual(episode["scenes"][0]["asset_ref"], "E0982")
        self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0001", "E0982"])
        self.assertEqual(len(stubborn.outline_requests), 1)

    def test_exhausted_outline_downgrades_unusable_evidence_scenes_without_losing_facts(self):
        index = [{"ref": f"E{i:04d}", "kind": "document",
                  "relative_path": f"source-{i}.md", "evidence_role": "primary"}
                 for i in range(1, 4)]
        index.append({"ref": "E0982", "kind": "media", "relative_path": "boringfetch.webp",
                      "evidence_role": "primary"})
        research = {"version": 1, "facts": [{
            "claim": f"BoringOS fact {i}.", "evidence_refs": [f"E{i:04d}"],
            "support": [{"evidence_ref": f"E{i:04d}", "text": f"BoringOS fact {i}."}],
            "phase": "final", "confidence": "high",
        } for i in range(1, 4)], "assets": [{"evidence_ref": "E0982",
                    "purpose": "BoringOS terminal preview", "authentic_project_media": True}]}
        ask = planner._make_ask(research, [index[-1]], index, "BoringOS", "")

        class Provider:
            def __init__(self, out_of_scope=False):
                self.outline_calls = 0
                self.out_of_scope = out_of_scope

            def complete_json(self, _system, user):
                request = json.loads(user)
                if request.get("storyboard_mode") == "outline":
                    self.outline_calls += 1
                    bad_ref = "E0999" if self.out_of_scope else None
                    return {"version": 1, "title": "BoringOS", "slug": "boringos",
                            "summary": "BoringOS fact 1.", "scene_intents": [
                                {"type": kind, "purpose": f"BoringOS fact {i}.",
                                 "fact_ids": [f"F{i:04d}"],
                                 "evidence_refs": [f"E{i:04d}"] + ([bad_ref] if bad_ref else []),
                                 "asset_ref": bad_ref}
                                for i, kind in enumerate(("HERO", "HARDWARE_EVIDENCE",
                                                          "TERMINAL_EVIDENCE"), 1)
                            ]}
                if request.get("storyboard_mode") == "scenes":
                    scenes = []
                    for template in request["required_output"]["scenes"]:
                        scene = dict(template)
                        fact = next(f for f in request["research"]["facts"]
                                    if f["fact_id"] == scene["fact_ids"][0])
                        scene.update(title="BoringOS", narration=fact["claim"])
                        scenes.append(scene)
                    return {"scenes": scenes}
                raise AssertionError("Unexpected request")

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(planner, "fits_context", return_value=True), \
             patch.object(planner, "_canonicalize_outline_metadata", side_effect=lambda _p, value, *_args: value), \
             patch.object(planner, "_validate_narration_grounding"), \
             patch.object(planner, "_validate_structured_grounding"):
            project = Path(tmp) / "projects" / "boringos"
            project.mkdir(parents=True)
            provider = Provider()
            episode = planner._multipart_episode(
                provider, "storyboard", ask, project, 8192, 4096, 1024, 1, None, "scope")
            self.assertEqual([s["type"] for s in episode["scenes"]],
                             ["HERO", "SUMMARY", "SUMMARY"])
            self.assertEqual(episode["scenes"][0]["asset_ref"], "E0982")
            self.assertEqual([s["fact_ids"] for s in episode["scenes"]],
                             [["F0001"], ["F0002"], ["F0003"]])
            self.assertEqual(provider.outline_calls, 2)
            cached = planner._multipart_episode(
                provider, "storyboard", ask, project, 8192, 4096, 1024, 1, None, "scope")
            self.assertEqual(cached, episode)
            self.assertEqual(provider.outline_calls, 2)

            class NonVisualAsset(Provider):
                def complete_json(self, system, user):
                    request = json.loads(user)
                    result = super().complete_json(system, user)
                    if request.get("storyboard_mode") == "outline":
                        for intent in result["scene_intents"]:
                            intent["asset_ref"] = intent["evidence_refs"][0]
                    return result

            zero_retry = project.parent / "recover-zero-retry"
            zero_retry.mkdir()
            non_visual = NonVisualAsset()
            recovered = planner._multipart_episode(
                non_visual, "storyboard", ask,
                zero_retry, 8192, 4096, 1024, 0, None, "scope",
            )
            self.assertEqual([s["type"] for s in recovered["scenes"]],
                             ["SUMMARY", "SUMMARY", "SUMMARY"])
            self.assertEqual([s["fact_ids"] for s in recovered["scenes"]],
                             [["F0001"], ["F0002"], ["F0003"]])
            self.assertEqual([s["evidence_refs"] for s in recovered["scenes"]],
                             [["E0001"], ["E0002"], ["E0003"]])
            self.assertFalse(any("asset_ref" in s for s in recovered["scenes"]))
            self.assertEqual(non_visual.outline_calls, 1)

            class NonVisualAssetWithRepairableFactIds(Provider):
                def complete_json(self, system, user):
                    request = json.loads(user)
                    result = super().complete_json(system, user)
                    if request.get("storyboard_mode") == "outline":
                        for intent in result["scene_intents"]:
                            intent["fact_ids"] = ["F9999"]
                            intent["asset_ref"] = intent["evidence_refs"][0]
                    return result

            repairable = project.parent / "recover-fact-id-and-asset"
            repairable.mkdir()
            combined = NonVisualAssetWithRepairableFactIds()
            repaired = planner._multipart_episode(
                combined, "storyboard", ask,
                repairable, 8192, 4096, 1024, 0, None, "scope",
            )
            self.assertEqual([s["type"] for s in repaired["scenes"]],
                             ["SUMMARY", "SUMMARY", "SUMMARY"])
            self.assertEqual([s["fact_ids"] for s in repaired["scenes"]],
                             [["F0001"], ["F0002"], ["F0003"]])
            self.assertEqual([s["evidence_refs"] for s in repaired["scenes"]],
                             [["E0001"], ["E0002"], ["E0003"]])
            self.assertFalse(any("asset_ref" in s for s in repaired["scenes"]))
            self.assertEqual(combined.outline_calls, 1)

            other = project.parent / "reject"
            other.mkdir()
            with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                planner._multipart_episode(Provider(out_of_scope=True), "storyboard", ask,
                                           other, 8192, 4096, 1024, 1, None, "scope")


if __name__ == "__main__":
    unittest.main()
