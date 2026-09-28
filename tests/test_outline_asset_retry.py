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


if __name__ == "__main__":
    unittest.main()
