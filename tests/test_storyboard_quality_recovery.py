"""Current-state evidence must survive into content scenes without template artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner
from source2reel.util import json_load
from test_planner_source_agnostic import project, source


class StoryboardQualityRecoveryTests(unittest.TestCase):
    def test_selected_future_fact_must_be_spoken(self):
        inventory, research = source([
            "WidgetOS boots from writable storage.",
            "There is no native GPU driver yet. Better graphics are planned for later work.",
        ])
        research["facts"][1]["phase"] = "development"
        ask = planner._make_ask(
            research, [], inventory["evidence"], "WidgetOS",
            "Explain the current implementation and planned future work.",
        )
        scene = {"id": "s001", "type": "SUMMARY", "fact_ids": ["F0001", "F0002"],
                 "evidence_refs": ["E0001", "E0002"],
                 "narration": research["facts"][0]["claim"]}
        with self.assertRaisesRegex(ValueError, "omits the selected future-work boundary"):
            planner._validate_narration_grounding(None, [scene], ask, None)

    def test_future_boundary_requires_content_and_duplicate_hero_visual_becomes_summary(self):
        inventory, research = source([
            "WidgetOS boots a native desktop.",
            "WidgetEdit loads and saves through WidgetFS.",
            "There is no native GPU driver yet. A better framebuffer mode is planned for later work.",
        ], visual=(90,))
        research["facts"][2]["phase"] = "development"
        ask = planner._make_ask(
            research, planner._media_inventory(inventory), inventory["evidence"],
            "WidgetOS", "Explain the current desktop and planned future work.",
        )
        self.assertEqual(ask["requested_topic_fact_ids"]["future-work"], ["F0003"])
        intents = [
            {"type": "HERO", "purpose": research["facts"][0]["claim"],
             "fact_ids": ["F0001"], "evidence_refs": ["E0001", "E0090"],
             "asset_ref": "E0090"},
            {"type": "SECTION_TITLE", "purpose": "Plan a section title scene",
             "fact_ids": [], "evidence_refs": []},
            {"type": "PROJECT_EVIDENCE", "purpose": research["facts"][1]["claim"],
             "fact_ids": ["F0002"], "evidence_refs": ["E0002", "E0090"],
             "asset_ref": "E0090"},
        ]
        outline = {"version": 1, "title": "WidgetOS", "slug": "widgetos",
                   "summary": research["facts"][0]["claim"],
                   "scene_intents": intents}
        with self.assertRaisesRegex(ValueError, "future-work"):
            planner._normalize_outline(outline, set(ask["visual_asset_refs"]) |
                                       {"E0001", "E0002", "E0003"}, ask)
        selected = planner._normalize_outline(
            {**outline, "scene_intents": [*intents, {
                "type": "SUMMARY", "purpose": research["facts"][2]["claim"],
                "fact_ids": ["F0003"], "evidence_refs": ["E0003"],
            }]}, set(ask["visual_asset_refs"]) | {"E0001", "E0002", "E0003"}, ask,
        )
        scenes = selected["scene_intents"]
        self.assertEqual([scene["id"] for scene in scenes], ["s001", "s003", "s004"])
        self.assertEqual([scene["type"] for scene in scenes],
                         ["HERO", "SUMMARY", "SUMMARY"])
        self.assertNotIn("asset_ref", scenes[1])
        self.assertEqual(scenes[1]["evidence_refs"], ["E0002"])

        payload = planner._scene_part_payload(ask, selected, [scenes[1]], 1, 1, "scope")
        self.assertNotEqual(payload["required_output"]["scenes"][0]["title"],
                            "on-screen title")
        scene = {**payload["required_output"]["scenes"][0],
                 "title": "on-screen title",
                 "narration": research["facts"][1]["claim"]}
        normalized = planner._normalize_scene_part(
            {"scenes": [scene]}, [scenes[1]], {"E0002"}, selected, ask,
        )
        self.assertEqual(normalized["scenes"][0]["title"],
                         "WidgetEdit loads and saves through WidgetFS")

    def test_english_storyboard_excludes_obvious_german_research_claim(self):
        german = ("Meldet ein Gerät den erwarteten Nachweis dafür, dass dieses "
                  "Kommando nicht unterstützt wird, schaltet WidgetOS auf FUA um.")
        english = "WidgetOS keeps the writable USB root across a reboot."
        inventory, research = source([german, english])

        class ExactProvider:
            def complete_json(self, _system, user):
                payload = json.loads(user)
                fact = payload["research"]["facts"][0]
                return {"version": 1, "title": "WidgetOS", "scenes": [{
                    "id": "s001", "type": "SUMMARY", "title": "Writable USB root",
                    "narration": fact["claim"], "fact_ids": [fact["fact_id"]],
                    "evidence_refs": fact["evidence_refs"],
                }]}

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            planner.plan(ExactProvider(), research, inventory, path, "WidgetOS")
            claims = [fact["claim"] for fact in json_load(
                path / "manifests" / "planner-evidence.json",
            )["research"]["facts"]]
        self.assertEqual(claims, [english])


if __name__ == "__main__":
    unittest.main()
