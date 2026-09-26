"""Multipart outline IDs must belong to the exact planner scope."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.providers import StructuredOutputError
from source2reel.util import json_load


CLAIMS = [
    "WidgetEngine ingests original project sources.",
    "WidgetEngine indexes grounded evidence records.",
    "WidgetEngine renders a planned technical explainer.",
]


def ask(shared_ref: bool = False):
    entries = [{"ref": f"E{i:04d}", "kind": "media" if i <= 2 else "document",
                "relative_path": f"media/{i}.png" if i <= 2 else f"docs/{i}.md",
                "evidence_role": "primary", "excerpt": claim}
               for i, claim in enumerate(CLAIMS, 1)]
    if shared_ref:
        entries[0]["excerpt"] += " " + CLAIMS[1]
    facts = [{"claim": claim, "evidence_refs": ["E0001" if shared_ref and i == 2
                                                else f"E{i:04d}"],
              "support": [{"evidence_ref": "E0001" if shared_ref and i == 2
                           else f"E{i:04d}", "text": claim}],
              "phase": "final", "confidence": "high"}
             for i, claim in enumerate(CLAIMS, 1)]
    return planner._make_ask({"version": 1, "facts": facts, "assets": []}, [], entries,
                             "WidgetEngine", "")


def outline(third_id: str = "F0003") -> dict:
    return {"version": 1, "title": "Grounded technical explainer", "slug": "grounded",
            "summary": CLAIMS[0], "scene_intents": [
                {"type": "CODE", "purpose": claim, "fact_ids": [f"F{i:04d}" if i < 3 else third_id],
                 "evidence_refs": [f"E{i:04d}"]}
                for i, claim in enumerate(CLAIMS, 1)
            ]}


class OutlineProvider:
    def __init__(self, outputs):
        self.outputs = outputs
        self.outline_calls = []
        self.scene_calls = []
        self.verifier_calls = []

    def complete_json(self, system, user):
        payload = json.loads(user)
        if "checks" in payload:
            self.verifier_calls.append(payload)
            return {"decisions": [{
                "id": check["id"], "supported": True,
                "propositions": [{"text": proposition, "support_indices": [0]}
                                 for proposition in check["required_propositions"]],
            } for check in payload["checks"]]}
        if payload.get("storyboard_mode") == "outline":
            self.outline_calls.append(payload)
            index = min(len(self.outline_calls)-1, len(self.outputs)-1)
            return copy.deepcopy(self.outputs[index])
        if payload.get("storyboard_mode") == "scenes":
            self.scene_calls.append(payload)
            facts = {fact["fact_id"]: fact["claim"] for fact in payload["research"]["facts"]}
            return {"scenes": [
                {**copy.deepcopy(requested), "title": "Supported scene",
                 "narration": facts[requested["fact_ids"][0]]}
                for requested in payload["required_output"]["scenes"]
            ]}
        raise AssertionError("Unexpected provider request")


class OutlineFactIdIntegrityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.project = Path(tmp.name) / "projects" / "demo"
        self.ask = ask()
        self.allowed = {entry["ref"] for entry in self.ask["evidence_index"]}

    def run_parts(self, provider, retries=0):
        return planner._multipart_episode(
            provider, "Storyboard from supplied facts.", self.ask, self.project,
            context_size=32768, output_reserve_tokens=4096, safety_tokens=1024,
            max_retries=retries, progress=None, scope_id="three-facts",
        )

    def make_e3_ambiguous(self):
        second = self.ask["research"]["facts"][1]
        second["evidence_refs"].append("E0003")
        second["support"].append({"evidence_ref": "E0003", "text": CLAIMS[1]})
        next(item for item in self.ask["evidence_index"]
             if item["ref"] == "E0003")["excerpt"] = CLAIMS[1] + " " + CLAIMS[2]

    def add_visual_asset(self):
        self.ask["evidence_index"].append({"ref": "E0090", "kind": "media",
                                           "evidence_role": "primary"})
        self.ask["research"]["assets"].append({"evidence_ref": "E0090",
                                                 "purpose": "Visual only"})
        self.allowed.add("E0090")

    def test_omitted_research_asset_is_added_to_first_evidence_intent(self):
        self.add_visual_asset()
        value = outline()
        value["scene_intents"][0].update({"type": "PROJECT_EVIDENCE", "asset_ref": "E0090"})
        provider = OutlineProvider([value])
        episode = self.run_parts(provider)
        first = episode["scenes"][0]
        self.assertEqual(first["fact_ids"], ["F0001"])
        self.assertEqual(first["evidence_refs"], ["E0001", "E0090"])
        self.assertEqual(first["asset_ref"], "E0090")
        saved = json_load(self.project / "manifests" / "storyboard-parts" / "outline.json")
        self.assertEqual(saved["result"]["scene_intents"][0]["evidence_refs"],
                         ["E0001", "E0090"])
        self.assertEqual(provider.scene_calls[0]["scene_intents"][0]["asset_ref"], "E0090")

    def test_present_asset_stays_in_returned_order_without_duplication(self):
        self.add_visual_asset()
        value = outline()
        value["scene_intents"][0].update({
            "type": "HERO", "evidence_refs": ["E0001", "E0090"], "asset_ref": "E0090",
        })
        first = planner._normalize_outline(value, self.allowed, self.ask)["scene_intents"][0]
        self.assertEqual(first["evidence_refs"], ["E0001", "E0090"])

    def test_globally_known_stray_ref_drops_before_missing_asset_is_added(self):
        self.add_visual_asset()
        self.ask["evidence_index"].append({"ref": "E9990", "kind": "document",
                                           "evidence_role": "primary"})
        self.allowed.add("E9990")
        value = outline()
        value["scene_intents"][0].update({
            "type": "HERO", "evidence_refs": ["E9990", "E0001"], "asset_ref": "E0090",
        })
        first = planner._normalize_outline(value, self.allowed, self.ask)["scene_intents"][0]
        self.assertEqual(first["fact_ids"], ["F0001"])
        self.assertEqual(first["evidence_refs"], ["E0001", "E0090"])

    def test_six_raw_slots_are_filtered_before_asset_inclusion(self):
        self.add_visual_asset()
        extras = [f"E{i:04d}" for i in range(70, 75)]
        self.ask["evidence_index"].extend({"ref": ref, "kind": "document",
                                           "evidence_role": "primary"} for ref in extras)
        self.allowed.update(extras)
        value = outline()
        value["scene_intents"][0].update({
            "type": "HERO", "evidence_refs": extras + ["E0001"], "asset_ref": "E0090",
        })
        first = planner._normalize_outline(value, self.allowed, self.ask)["scene_intents"][0]
        self.assertEqual(first["evidence_refs"], ["E0001", "E0090"])
        self.assertLessEqual(len(first["evidence_refs"]), 6)

        # Six legitimate fact refs plus a visual asset cannot fit; never drop the asset.
        source = self.ask["research"]["facts"][0]
        source["evidence_refs"].extend(extras)
        source["support"].extend({"evidence_ref": ref, "text": CLAIMS[0]}
                                 for ref in extras)
        with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
            planner._normalize_outline(value, self.allowed, self.ask)

    def test_missing_invalid_or_unselected_asset_is_not_repaired(self):
        for asset in ("E9999", "E0002", 90, ""):
            with self.subTest(asset=asset):
                value = outline()
                value["scene_intents"][0].update({"type": "HERO", "asset_ref": asset})
                with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                    planner._normalize_outline(value, self.allowed, self.ask)

    def test_non_asset_scene_discards_visual_asset(self):
        self.add_visual_asset()
        value = outline()
        value["scene_intents"][0]["asset_ref"] = "E0090"
        canonical = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertNotIn("asset_ref", canonical["scene_intents"][0])

    def test_omitted_asset_never_supplies_narration_facts(self):
        self.add_visual_asset()
        value = outline()
        value["scene_intents"][0].update({"type": "HERO", "asset_ref": "E0090"})
        first = planner._normalize_outline(value, self.allowed, self.ask)["scene_intents"][0]
        with self.assertRaisesRegex(ValueError, "unsupported factual proposition"):
            planner._validate_narration_grounding(
                OutlineProvider([]), [{**first,
                                       "narration": "WidgetEngine streams unsupported live footage."}],
                self.ask, self.project,
            )

    def test_omitted_asset_with_uniquely_recovered_fact_id(self):
        self.add_visual_asset()
        value = outline()
        value["scene_intents"][0].update({
            "type": "HERO", "fact_ids": ["F0004"], "asset_ref": "E0090",
        })
        first = planner._normalize_outline(value, self.allowed, self.ask)["scene_intents"][0]
        self.assertEqual(first["fact_ids"], ["F0001"])
        self.assertEqual(first["evidence_refs"], ["E0001", "E0090"])

    def test_payload_lists_exact_allowed_fact_ids_and_a_real_example(self):
        request = planner._outline_payload(self.ask)
        self.assertEqual(request["allowed_fact_ids"], ["F0001", "F0002", "F0003"])
        self.assertEqual(request["fact_evidence_map"], {
            "F0001": ["E0001"], "F0002": ["E0002"], "F0003": ["E0003"],
        })
        self.assertLess(len(json.dumps(request["fact_evidence_map"])), 100)
        self.assertEqual(request["required_output"]["scene_intents"][0]["fact_ids"], ["F0001"])
        self.assertIn("verbatim", request["fact_id_requirement"])
        self.assertIn("allowed_fact_ids", planner._OUTLINE_SYSTEM)
        self.assertIn("fact_evidence_map", planner._OUTLINE_SYSTEM)
        self.assertIn("selected fact_ids", request["evidence_ref_requirement"])

    def test_valid_fact_keeps_its_selection_when_global_ref_is_stray(self):
        value = outline()
        value["scene_intents"][0]["evidence_refs"] = ["E0002"]
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][0]["fact_ids"], ["F0001"])
        self.assertEqual(result["scene_intents"][0]["evidence_refs"], ["E0001"])
        planner._validate_scene_facts(result["scene_intents"][0], self.ask)

    def test_mixed_refs_keep_authorized_order_and_deduplicate(self):
        first = self.ask["research"]["facts"][0]
        first["evidence_refs"].append("E0003")
        first["support"].append({"evidence_ref": "E0003", "text": CLAIMS[0]})
        next(item for item in self.ask["evidence_index"]
             if item["ref"] == "E0003")["excerpt"] = CLAIMS[0] + " " + CLAIMS[2]
        value = outline()
        value["scene_intents"][0]["evidence_refs"] = ["E0001", "E0002", "E0001"]
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][0]["evidence_refs"], ["E0001"])
        self.assertEqual(result["scene_intents"][0]["fact_ids"], ["F0001"])

    def test_multiple_selected_facts_do_not_adopt_a_third_fact(self):
        value = outline()
        value["scene_intents"][1]["fact_ids"] = ["F0001", "F0002"]
        value["scene_intents"][1]["evidence_refs"] = ["E0002", "E0003"]
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][1]["fact_ids"], ["F0001", "F0002"])
        self.assertEqual(result["scene_intents"][1]["evidence_refs"], ["E0002"])

    def test_empty_refs_are_completed_from_valid_selected_fact(self):
        value = outline()
        value["scene_intents"][0]["evidence_refs"] = []
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][0]["evidence_refs"], ["E0001"])

    def test_out_of_scope_stray_ref_is_discarded_only_with_valid_fact_ids(self):
        value = outline()
        value["scene_intents"][0]["evidence_refs"] = ["E9999"]
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][0]["evidence_refs"], ["E0001"])
        self.assertEqual(result["scene_intents"][0]["fact_ids"], ["F0001"])

    def test_selected_visual_asset_survives_without_authorizing_narration(self):
        self.ask["evidence_index"].append({"ref": "E0090", "kind": "media",
                                           "evidence_role": "primary"})
        self.ask["research"]["assets"].append({"evidence_ref": "E0090",
                                                 "purpose": "Visual only"})
        self.allowed.add("E0090")
        value = outline()
        value["scene_intents"][0].update({
            "type": "HERO", "evidence_refs": ["E0090", "E9999"], "asset_ref": "E0090",
        })
        result = planner._normalize_outline(value, self.allowed, self.ask)
        first = result["scene_intents"][0]
        self.assertEqual(first["fact_ids"], ["F0001"])
        self.assertEqual(first["evidence_refs"], ["E0090", "E0001"])
        self.assertEqual(first["asset_ref"], "E0090")
        self.assertNotIn("E0090", self.ask["research"]["facts"][0]["evidence_refs"])
        with self.assertRaisesRegex(ValueError, "selected fact_ids or selected asset_ref"):
            planner._validate_scene_facts({**first, "evidence_refs": ["E0090", "E0002"]},
                                          self.ask)
        with self.assertRaisesRegex(ValueError, "unsupported factual proposition"):
            planner._validate_narration_grounding(
                OutlineProvider([]), [{**first,
                                       "narration": "WidgetEngine streams unsupported live footage."}],
                self.ask, self.project,
            )

    def test_invalid_or_contradictory_asset_fails_instead_of_normalizing(self):
        for asset in ("E9999", "E0002"):
            with self.subTest(asset=asset):
                value = outline()
                value["scene_intents"][0].update({
                    "type": "HERO", "asset_ref": asset, "evidence_refs": [asset],
                })
                with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                    planner._normalize_outline(value, self.allowed, self.ask)

    def test_sixth_intent_stray_ref_normalizes_and_checkpoint_is_reused(self):
        value = outline()
        value["scene_intents"].extend(copy.deepcopy(value["scene_intents"][:3]))
        value["scene_intents"][5].update({
            "type": "SUMMARY", "fact_ids": ["F0001"],
            "evidence_refs": ["E0002"], "purpose": CLAIMS[0],
        })
        provider = OutlineProvider([value])
        with patch("source2reel.planner._canonicalize_outline_metadata",
                   wraps=planner._canonicalize_outline_metadata) as grounding:
            episode = self.run_parts(provider)
        grounding.assert_called_once()
        self.assertEqual(len(episode["scenes"]), 6)
        self.assertEqual(episode["scenes"][5]["fact_ids"], ["F0001"])
        self.assertEqual(episode["scenes"][5]["evidence_refs"], ["E0001"])
        path = self.project / "manifests" / "storyboard-parts" / "outline.json"
        self.assertEqual(json_load(path)["result"]["scene_intents"][5]["evidence_refs"],
                         ["E0001"])
        self.run_parts(provider)
        self.assertEqual(len(provider.outline_calls), 1)

    def test_valid_ids_pass_unchanged(self):
        value = outline()
        value["scene_intents"][1]["fact_ids"] = ["F0002"]
        result = planner._normalize_outline(value, self.allowed, self.ask)
        self.assertEqual(result["scene_intents"][1]["fact_ids"], ["F0002"])

    def test_unique_ref_repairs_only_id_then_normal_grounding_runs(self):
        provider = OutlineProvider([outline("F0004")])
        with patch("source2reel.planner._canonicalize_outline_metadata",
                   wraps=planner._canonicalize_outline_metadata) as grounding:
            episode = self.run_parts(provider)
        grounding.assert_called_once()
        self.assertEqual(episode["scenes"][2]["fact_ids"], ["F0003"])
        self.assertEqual(episode["scenes"][2]["narration"], CLAIMS[2])
        checkpoint = json_load(self.project / "manifests" / "storyboard-parts" / "outline.json")
        self.assertEqual(checkpoint["result"]["scene_intents"][2]["fact_ids"], ["F0003"])

    def test_unique_id_repair_keeps_fact_authority_when_purpose_is_replaced(self):
        value = outline("F0004")
        value["scene_intents"][2]["purpose"] = "WidgetEngine guarantees automatic success."
        provider = OutlineProvider([value])
        episode = self.run_parts(provider)
        checkpoint = json_load(self.project / "manifests" / "storyboard-parts" / "outline.json")
        self.assertEqual(checkpoint["result"]["scene_intents"][2]["purpose"], CLAIMS[2])
        self.assertEqual(checkpoint["result"]["scene_intents"][2]["fact_ids"], ["F0003"])
        self.assertEqual(episode["scenes"][2]["narration"], CLAIMS[2])

    def test_ambiguous_reference_requires_retry_and_fails_closed(self):
        scope = ask(shared_ref=True)
        value = outline()
        value["scene_intents"] = [{"type": "CODE", "purpose": CLAIMS[0],
                                   "fact_ids": ["F9999"], "evidence_refs": ["E0001"]}]
        with self.assertRaisesRegex(StructuredOutputError, "F9999.*allowed_fact_ids"):
            planner._normalize_outline(value, {"E0001", "E0002", "E0003"}, scope)
        self.ask = scope
        provider = OutlineProvider([value])
        with self.assertRaisesRegex(StructuredOutputError, "F9999.*allowed_fact_ids"):
            self.run_parts(provider, retries=1)
        self.assertEqual(len(provider.outline_calls), 2)
        self.assertEqual(provider.scene_calls, [])

    def test_out_of_scope_ref_never_repairs(self):
        value = outline("F0004")
        value["scene_intents"][2]["evidence_refs"] = ["E0099"]
        with self.assertRaisesRegex(StructuredOutputError, "outside planner scope"):
            planner._normalize_outline(value, self.allowed, self.ask)

    def test_retry_feedback_names_bad_id_and_complete_allowed_set(self):
        value = outline("F9999")
        self.make_e3_ambiguous()
        provider = OutlineProvider([value, outline()])
        episode = self.run_parts(provider, retries=1)
        self.assertEqual(episode["scenes"][2]["fact_ids"], ["F0003"])
        feedback = provider.outline_calls[1]["validation_feedback"]
        self.assertIn("F9999", feedback)
        self.assertIn("allowed_fact_ids: [F0001, F0002, F0003]", feedback)
        self.assertLess(len(feedback), 400)
        stored = json_load(self.project / "manifests" / "storyboard-parts" / "outline.json")
        self.assertEqual(stored["result"]["scene_intents"][2]["fact_ids"], ["F0003"])
        self.assertEqual(len(provider.outline_calls), 2)
        self.assertEqual(len(provider.scene_calls), 2)

    def test_invalid_outline_is_not_checkpointed_before_valid_retry(self):
        value = outline("F9999")
        self.make_e3_ambiguous()
        provider = OutlineProvider([value, outline()])
        with self.assertRaisesRegex(StructuredOutputError, "F9999"):
            self.run_parts(provider, retries=0)
        path = self.project / "manifests" / "storyboard-parts" / "outline.json"
        self.assertFalse(path.exists())
        self.assertEqual(provider.scene_calls, [])
        episode = self.run_parts(provider, retries=1)
        self.assertEqual(episode["scenes"][2]["fact_ids"], ["F0003"])
        self.assertEqual(json_load(path)["result"]["scene_intents"][2]["fact_ids"], ["F0003"])


if __name__ == "__main__":
    unittest.main()
