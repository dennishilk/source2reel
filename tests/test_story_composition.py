"""Final composition of factual framing, partial repetition and grounded endings."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.schema import validate_episode
from source2reel.util import json_load

from test_storyboard_editorial import (CLAIMS, PhysicalOutlineProvider, intent,
                                       outline, physical_ask)


def physical_outline():
    raw = outline([
        intent("SECTION_TITLE", 1), intent("SECTION_TITLE", 3),
        intent("SECTION_TITLE", 4), intent("SUMMARY", 2),
        intent("SUMMARY", 1, 4), intent("SUMMARY", 5),
        {"type": "OUTRO", "purpose": "Closing", "fact_ids": [], "evidence_refs": []},
    ])
    ids = ["s001", "s002", "s003", "s004", "s006", "s008", "s009"]
    for scene, scene_id in zip(raw["scene_intents"], ids):
        scene["id"] = scene_id
    raw["scene_intents"][0]["purpose"] = "Introduce the tool and its passive diagnostic nature"
    raw["scene_intents"][3]["purpose"] = "Show how ETW callbacks normalize and enqueue network events"
    raw["presentation"]["scene_titles"] = {
        scene_id: f"Documented scene {scene_id}" for scene_id in ids
    }
    return raw


def selected_content(scenes):
    return [scene for scene in scenes
            if scene["type"] not in {"SECTION_TITLE", "HERO", "OUTRO"}]


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.ask, self.allowed = physical_ask()

    def assert_physical_properties(self, scenes):
        self.assertFalse(any(scene["type"] == "SECTION_TITLE" and scene["fact_ids"]
                             for scene in scenes))
        self.assertEqual([scene["id"] for scene in scenes if scene["type"] == "SECTION_TITLE"], [])
        self.assertTrue(all(scene["id"] not in {"s006", "s008"} for scene in scenes))
        contents = selected_content(scenes)
        self.assertEqual(sum("F0001" in scene["fact_ids"] for scene in contents), 1)
        self.assertEqual(sum("F0004" in scene["fact_ids"] for scene in contents), 1)
        self.assertEqual(sum("F0005" in scene["fact_ids"] for scene in contents), 0)
        self.assertTrue(any("F0002" in scene["fact_ids"] for scene in contents))
        self.assertTrue(any("F0003" in scene["fact_ids"] for scene in contents))
        self.assertEqual(scenes[-1]["type"], "OUTRO")
        self.assertEqual(scenes[-1]["fact_ids"], ["F0001"])
        self.assertEqual(planner._missing_story_topics(scenes, self.ask), {})
        planner.validate_novelty(scenes)

    def test_exact_physical_outline_never_improving_provider_and_checkpoint(self):
        raw = physical_outline()
        provider = PhysicalOutlineProvider(raw)
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)

            def run():
                return planner._multipart_episode(provider, "storyboard", self.ask,
                    project_dir, 32768, 4096, 1024, 0, None, "physical-composition")

            episode = run()
            checkpoint_path = project_dir / "manifests/storyboard-parts/outline.json"
            saved = json_load(checkpoint_path)
            canonical = saved["result"]
            self.assert_physical_properties(canonical["scene_intents"])
            self.assert_physical_properties(episode["scenes"])
            self.assertEqual(episode["scenes"][-1]["narration"], CLAIMS[0])
            self.assertNotEqual(episode["scenes"][-1]["narration"], "Closing.")
            self.assertEqual(set(canonical["presentation"]["scene_titles"]),
                             {scene["id"] for scene in canonical["scene_intents"]})
            self.assertEqual(canonical["presentation"]["outro"], raw["presentation"]["outro"])
            self.assertEqual(planner._normalize_outline(
                canonical, self.allowed, self.ask, provider=provider,
                project_dir=project_dir,
            ), canonical)
            self.assertEqual(self.ask["research"]["facts"][1]["evidence_refs"], ["E0003"])
            self.assertIn("F0002", self.ask["requested_topic_fact_ids"]["workflow"])
            self.assertNotIn("purpose", self.ask["requested_topic_fact_ids"])
            scene_requests = [request for request in provider.calls
                              if request.get("storyboard_mode") == "scenes"]
            self.assertEqual([scene["id"] for request in scene_requests
                              for scene in request["required_output"]["scenes"]],
                             [scene["id"] for scene in episode["scenes"]])
            self.assertEqual(sum(request.get("storyboard_mode") == "outline"
                                 for request in provider.calls), 1)
            before = len(provider.calls)
            self.assertEqual(run(), episode)
            self.assertEqual(len(provider.calls), before)
            self.assertEqual(json_load(checkpoint_path), saved)

    def test_full_response_has_the_same_composition(self):
        raw = physical_outline()
        full = {"version": 1, "title": raw["title"],
                "presentation": raw["presentation"],
                "scenes": [{**{key: value for key, value in item.items() if key != "purpose"},
                            "title": "Documented scene",
                            "narration": " ".join(CLAIMS[int(fact_id[1:]) - 1]
                                                  for fact_id in item["fact_ids"]) or "Closing."}
                           for item in raw["scene_intents"]]}

        class FullProvider(PhysicalOutlineProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "storyboard_mode" not in request and "checks" not in request:
                    self.calls.append(request)
                    return copy.deepcopy(full)
                return super().complete_json(system, user)

        provider = FullProvider(raw)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._complete_episode(provider, "storyboard", self.ask,
                self.allowed, 0, project_dir=Path(tmp))
        self.assert_physical_properties(episode["scenes"])
        self.assertEqual(episode["scenes"][-1]["narration"], CLAIMS[0])
        self.assertEqual(set(episode["presentation"]["scene_titles"]),
                         {scene["id"] for scene in episode["scenes"]})
        self.assertEqual(sum("storyboard_mode" not in call and "checks" not in call
                             for call in provider.calls), 1)
        validate_episode(episode, self.allowed, require_integrated_presentation=True)

    def test_partial_fact_trim_recomputes_refs_without_new_claims(self):
        raw = outline([intent("SUMMARY", 4), intent("SUMMARY", 2, 4)])
        raw.pop("presentation")
        result = planner._normalize_outline(raw, self.allowed, self.ask,
                                            check_coverage=False)
        self.assertEqual([item["fact_ids"] for item in result["scene_intents"]],
                         [["F0004"], ["F0002"]])
        self.assertEqual(result["scene_intents"][1]["evidence_refs"], ["E0003"])
        self.assertEqual(result["scene_intents"][1]["purpose"], CLAIMS[1])

        full = {"version": 1, "title": "Documented facts", "scenes": [
            {"id": "s001", "type": "SUMMARY", "title": "Correlation",
             "narration": CLAIMS[3], "fact_ids": ["F0004"], "evidence_refs": ["E0002"]},
            {"id": "s002", "type": "DATA_FLOW", "title": "Correlation",
             "narration": CLAIMS[1] + " " + CLAIMS[3],
             "fact_ids": ["F0002", "F0004"], "evidence_refs": ["E0003", "E0002"],
             "diagram": {"nodes": ["ETW callbacks normalize raw provider data",
                                   "DNS and scheduled-task relationships"]},
             "annotations": [CLAIMS[3]]},
        ]}
        canonical = planner._canonical_full_episode(full, self.ask, self.allowed)
        changed = canonical["scenes"][1]
        self.assertEqual(changed["fact_ids"], ["F0002"])
        self.assertEqual(changed["evidence_refs"], ["E0003"])
        self.assertEqual(changed["narration"], CLAIMS[1])
        self.assertEqual(changed["type"], "SUMMARY")
        self.assertNotIn("diagram", changed)
        self.assertNotIn("annotations", changed)

    def test_semantic_verification_keeps_uncertain_and_distinct_claims(self):
        raw = outline([intent("SUMMARY", 1), intent("SUMMARY", 3),
                       intent("SUMMARY", 4), intent("SUMMARY", 5),
                       intent("SUMMARY", 2)])
        raw.pop("presentation")
        with tempfile.TemporaryDirectory() as tmp:
            approving = PhysicalOutlineProvider(raw)
            canonical = planner._normalize_outline(raw, self.allowed, self.ask,
                provider=approving, project_dir=Path(tmp))
            self.assertEqual([item["fact_ids"] for item in canonical["scene_intents"]],
                             [["F0001"], ["F0003"], ["F0004"], ["F0002"]])
            checks = [check for request in approving.calls if "checks" in request
                      for check in request["checks"]]
            self.assertTrue(any(check["claim"] == CLAIMS[4] and
                                check["support"] == [CLAIMS[3]] for check in checks))
            self.assertFalse(any(check["claim"] == CLAIMS[2] for check in checks))

        class Uncertain(PhysicalOutlineProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "checks" in request:
                    self.calls.append(request)
                    return {"decisions": [{"id": check["id"], "supported": False,
                                           "propositions": []} for check in request["checks"]]}
                return super().complete_json(system, user)

        with tempfile.TemporaryDirectory() as tmp:
            uncertain = planner._normalize_outline(raw, self.allowed, self.ask,
                provider=Uncertain(raw), project_dir=Path(tmp))
        self.assertIn(["F0005"], [item["fact_ids"] for item in uncertain["scene_intents"]])

    def test_stronger_fact_is_not_removed_for_word_overlap(self):
        ask = copy.deepcopy(self.ask)
        prior = "The tool records local network activity."
        stronger = "The tool records all local network activity."
        for number, claim in ((7, prior), (8, stronger)):
            ask["research"]["facts"].append({
                "fact_id": f"F{number:04d}", "claim": claim,
                "evidence_refs": ["E0002"],
                "support": [{"evidence_ref": "E0002", "text": claim}],
            })
        raw = outline([
            {"type": "SUMMARY", "purpose": prior, "fact_ids": ["F0007"],
             "evidence_refs": ["E0002"]},
            {"type": "SUMMARY", "purpose": stronger, "fact_ids": ["F0008"],
             "evidence_refs": ["E0002"]},
        ])
        raw.pop("presentation")
        with tempfile.TemporaryDirectory() as tmp:
            provider = PhysicalOutlineProvider(raw)  # Would approve any check offered.
            result = planner._normalize_outline(raw, self.allowed, ask,
                check_coverage=False, provider=provider, project_dir=Path(tmp))
        self.assertEqual([item["fact_ids"] for item in result["scene_intents"]],
                         [["F0007"], ["F0008"]])
        self.assertFalse(any("checks" in call for call in provider.calls))

    def test_semantic_omission_cannot_lose_requested_topic(self):
        ask = copy.deepcopy(self.ask)
        ask["requested_topic_fact_ids"]["correlation"] = ["F0005"]
        raw = outline([intent("SUMMARY", 1), intent("SUMMARY", 2),
                       intent("SUMMARY", 4), intent("SUMMARY", 5)])
        raw.pop("presentation")
        with tempfile.TemporaryDirectory() as tmp:
            result = planner._normalize_outline(raw, self.allowed, ask,
                provider=PhysicalOutlineProvider(raw), project_dir=Path(tmp))
        self.assertIn(["F0005"], [item["fact_ids"] for item in result["scene_intents"]])
        self.assertEqual(planner._missing_story_topics(result["scene_intents"], ask), {})

    def test_identical_claim_with_a_different_id_needs_no_verifier(self):
        ask = copy.deepcopy(self.ask)
        ask["research"]["facts"].append({
            "fact_id": "F0007", "claim": CLAIMS[3], "evidence_refs": ["E0002"],
            "support": [{"evidence_ref": "E0002", "text": CLAIMS[3]}],
        })
        raw = outline([intent("SUMMARY", 4), {"type": "SUMMARY",
            "purpose": CLAIMS[3], "fact_ids": ["F0007"], "evidence_refs": ["E0002"]}])
        raw.pop("presentation")
        result = planner._normalize_outline(raw, self.allowed, ask,
                                            check_coverage=False)
        self.assertEqual([item["fact_ids"] for item in result["scene_intents"]],
                         [["F0004"]])

    def test_pure_title_run_collapses_and_no_outro_is_added(self):
        pure = {"type": "SECTION_TITLE", "purpose": "Section break",
                "fact_ids": [], "evidence_refs": []}
        raw = outline([copy.deepcopy(pure), copy.deepcopy(pure),
                       intent("SUMMARY", 1), copy.deepcopy(pure), copy.deepcopy(pure)])
        raw.pop("presentation")
        result = planner._normalize_outline(raw, self.allowed, self.ask,
                                            check_coverage=False)
        self.assertEqual([item["id"] for item in result["scene_intents"]],
                         ["s001", "s003", "s004"])
        self.assertEqual([item["type"] for item in result["scene_intents"]],
                         ["SECTION_TITLE", "SUMMARY", "SECTION_TITLE"])
        self.assertNotIn("OUTRO", [item["type"] for item in result["scene_intents"]])

    def test_distinct_authentic_asset_and_hero_remain_valid(self):
        visual = copy.deepcopy(self.ask)
        visual["evidence_index"].append({"ref": "E0090", "kind": "media",
                                         "relative_path": "authentic.png"})
        visual["evidence_index"].append({"ref": "E0091", "kind": "media",
                                         "relative_path": "opening.png"})
        visual["research"]["assets"] = [
            {"evidence_ref": "E0090", "purpose": "Capture"},
            {"evidence_ref": "E0091", "purpose": "Opening"},
        ]
        proof = {**intent("PROJECT_EVIDENCE", 1), "asset_ref": "E0090",
                 "evidence_refs": ["E0002", "E0090"]}
        hero = {**intent("HERO", 1), "asset_ref": "E0091",
                "evidence_refs": ["E0002", "E0091"]}
        raw = outline([hero, intent("SUMMARY", 1), proof])
        raw.pop("presentation")
        result = planner._normalize_outline(raw, self.allowed | {"E0090", "E0091"},
                                            visual, check_coverage=False)
        self.assertEqual([item["type"] for item in result["scene_intents"]],
                         ["HERO", "SUMMARY", "PROJECT_EVIDENCE"])
        self.assertEqual(result["scene_intents"][-1]["fact_ids"], ["F0001"])
        planner.validate_novelty(result["scene_intents"])

    def test_metadata_fallback_removes_empty_chapter_and_repeated_intro(self):
        visual = copy.deepcopy(self.ask)
        visual["evidence_index"].append({"ref": "E0091", "kind": "media",
                                         "relative_path": "opening.png"})
        visual["research"]["assets"] = [{"evidence_ref": "E0091", "purpose": "Opening"}]
        raw = outline([
            {**intent("HERO", 1), "asset_ref": "E0091",
             "evidence_refs": ["E0002", "E0091"]},
            {"type": "SECTION_TITLE", "purpose": "An unsupported invented chapter",
             "fact_ids": [], "evidence_refs": []},
            intent("SUMMARY", 1), intent("SUMMARY", 2),
        ])
        raw.pop("presentation")
        allowed = self.allowed | {"E0091"}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("source2reel.planner.verify_claims", return_value=set()):
            initial = planner._normalize_outline(raw, allowed, visual,
                                                 check_coverage=False)
            self.assertEqual([scene["id"] for scene in initial["scene_intents"]],
                             ["s001", "s002", "s003", "s004"])
            metadata = planner._canonicalize_outline_metadata(
                PhysicalOutlineProvider(raw), initial, visual, Path(tmp))
            result = planner._normalize_outline(metadata, allowed, visual,
                                                check_coverage=False)
        self.assertEqual([scene["id"] for scene in result["scene_intents"]],
                         ["s001", "s004"])
        self.assertEqual(result["scene_intents"][-1]["fact_ids"], ["F0002"])

    def test_mixed_summary_title_follows_its_first_spoken_claim(self):
        raw = outline([intent("SUMMARY", 1, 2)])
        raw.pop("presentation")
        canonical = planner._normalize_outline(raw, self.allowed, self.ask,
                                               check_coverage=False)
        draft = {"id": "s001", "type": "SUMMARY", "title": "ETW callbacks normalize events",
                 "narration": CLAIMS[0] + " " + CLAIMS[1],
                 "fact_ids": ["F0001", "F0002"],
                 "evidence_refs": ["E0002", "E0003"]}
        part = planner._normalize_scene_part({"scenes": [draft]},
                                             canonical["scene_intents"], self.allowed,
                                             canonical, self.ask)
        self.assertTrue(part["scenes"][0]["title"].startswith("Windows Telemetry Inspector"))

    def test_empty_outro_recovers_exact_final_suffix_with_zero_retries(self):
        ask = copy.deepcopy(self.ask)
        suffix = "Documented only."
        ask["required_narration_suffix"] = suffix
        raw = outline([intent("SUMMARY", 1), {"type": "OUTRO", "purpose": "Closing",
                                             "fact_ids": [], "evidence_refs": []}])

        class StubbornOutro(PhysicalOutlineProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if request.get("storyboard_mode") == "scenes":
                    result = super().complete_json(system, user)
                    for scene in result["scenes"]:
                        if scene["type"] == "OUTRO":
                            scene["narration"] = "Closing. " + suffix
                    return result
                return super().complete_json(system, user)

        provider = StubbornOutro(raw)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._multipart_episode(provider, "storyboard", ask,
                Path(tmp), 32768, 4096, 1024, 0, None, "grounded-outro")
        self.assertEqual(episode["scenes"][-1]["fact_ids"], ["F0001"])
        self.assertEqual(episode["scenes"][-1]["narration"], CLAIMS[0] + " " + suffix)
        self.assertEqual(episode["scenes"][-1]["narration"].count(suffix), 1)


if __name__ == "__main__":
    unittest.main()
