"""The multipart boundary derives authority before interpreting raw model refs."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.schema import EVIDENCE_TYPES, validate_episode
from source2reel.util import json_load
from tests.test_outline_fact_id_integrity import CLAIMS, ask


MISSING = object()


class MultipartContractMatrixTests(unittest.TestCase):
    def setUp(self):
        self.ask = ask()
        self.extras = [f"E{i:04d}" for i in range(70, 75)]
        for ref in ["E0090", "E9990", *self.extras]:
            self.ask["evidence_index"].append({
                "ref": ref, "kind": "media" if ref == "E0090" else "document",
                "evidence_role": "primary",
            })
        self.ask["research"]["assets"].append({
            "evidence_ref": "E0090", "purpose": "Visual only",
        })
        self.allowed = {entry["ref"] for entry in self.ask["evidence_index"]}

    def canonical(self, refs, asset=MISSING, ids=("F0001",), kind="HERO"):
        raw = {"type": kind, "purpose": CLAIMS[0], "evidence_refs": refs}
        if ids is not MISSING:
            raw["fact_ids"] = list(ids)
        if asset is not MISSING:
            raw["asset_ref"] = asset
        return planner._canonical_outline_intent(raw, 1, self.allowed, self.ask)

    def test_fact_id_contract_matrix(self):
        cases = [
            ("valid IDs authoritative despite stray", ("F0001",), ["E0002", "E0001"], "E0001"),
            ("missing IDs uniquely recovered", MISSING, ["E0001"], "E0001"),
            ("invalid ID uniquely recovered", ("F9999",), ["E0001"], "E0001"),
            ("research asset does not infer a fact", MISSING, ["E0090"], "E0090"),
        ]
        for label, ids, refs, asset in cases:
            with self.subTest(label=label):
                if label == "research asset does not infer a fact":
                    with self.assertRaisesRegex(StructuredOutputError, "allowed_fact_ids"):
                        self.canonical(refs, asset, ids)
                else:
                    result = self.canonical(refs, asset, ids)
                    self.assertEqual(result["fact_ids"], ["F0001"])
                    self.assertEqual(result["evidence_refs"], ["E0001"])

        ambiguous = ask(shared_ref=True)
        raw = {"type": "HERO", "purpose": CLAIMS[0],
               "fact_ids": ["F9999"], "evidence_refs": ["E0001"],
               "asset_ref": "E0001"}
        with self.assertRaisesRegex(StructuredOutputError, "allowed_fact_ids"):
            planner._canonical_outline_intent(
                raw, 1, {entry["ref"] for entry in ambiguous["evidence_index"]}, ambiguous,
            )

    def test_evidence_ref_contract_matrix(self):
        cases = [
            ("exact", ["E0001"], "E0001", ["E0001"]),
            ("empty", [], "E0001", ["E0001"]),
            ("duplicate", ["E0001", "E0001"], "E0001", ["E0001"]),
            ("global stray", ["E9990", "E0001"], "E0001", ["E0001"]),
            ("outside scope with valid IDs", ["E9999", "E0001"], "E0001", ["E0001"]),
            ("six raw slots with junk", self.extras + ["E0001"], "E0090",
             ["E0001", "E0090"]),
            ("empty with research asset", [], "E0090", ["E0090", "E0001"]),
        ]
        for label, refs, asset, expected in cases:
            with self.subTest(label=label):
                result = self.canonical(refs, asset)
                self.assertEqual(result["fact_ids"], ["F0001"])
                self.assertEqual(result["evidence_refs"], expected)

        with self.assertRaisesRegex(StructuredOutputError, "outside planner scope"):
            self.canonical(["E9999", "E0001"], "E0001", ("F9999",))

        # Six legitimate factual refs plus a distinct asset cannot fit.
        fact = self.ask["research"]["facts"][0]
        fact["evidence_refs"].extend(self.extras)
        fact["support"].extend({"evidence_ref": ref, "text": CLAIMS[0]}
                               for ref in self.extras)
        with self.assertRaisesRegex(StructuredOutputError, "no room for selected asset_ref"):
            self.canonical(["E0001", *self.extras], "E0090")

    def test_asset_and_scene_type_contract_matrix(self):
        cases = [
            ("selected fact present", ["E0001"], "E0001", True),
            ("selected fact omitted", [], "E0001", True),
            ("research asset present", ["E0001", "E0090"], "E0090", True),
            ("research asset omitted", ["E0001"], "E0090", True),
            ("global stray omitted", ["E0001"], "E9990", False),
            ("global stray present cannot launder", ["E0001", "E0002"], "E0002", False),
            ("outside scope", ["E0001"], "E9999", False),
            ("malformed number", ["E0001"], 90, False),
            ("malformed list", ["E0001"], ["E0001"], False),
            ("missing asset", ["E0001"], MISSING, False),
        ]
        for kind, (label, refs, asset, valid) in itertools.product(
            sorted(EVIDENCE_TYPES), cases
        ):
            with self.subTest(kind=kind, label=label):
                if valid:
                    result = self.canonical(refs, asset, kind=kind)
                    self.assertEqual(result["fact_ids"], ["F0001"])
                    self.assertEqual(result["asset_ref"], asset)
                    self.assertEqual(result["evidence_refs"].count(asset), 1)
                    self.assertIn("E0001", result["evidence_refs"])
                else:
                    with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                        self.canonical(refs, asset, kind=kind)

        self.assertNotIn("asset_ref", self.canonical(["E0001"], "E0001", kind="CODE"))
        for kind in ("SECTION_TITLE", "OUTRO"):
            with self.subTest(kind=kind):
                raw = {"type": kind, "purpose": "Closing",
                       "fact_ids": [], "evidence_refs": []}
                result = planner._canonical_outline_intent(raw, 1, self.allowed, self.ask)
                self.assertEqual(result["fact_ids"], [])
                self.assertEqual(result["evidence_refs"], [])
                self.assertNotIn("asset_ref", result)

    def test_outline_prompt_names_conditional_asset_sources(self):
        payload = planner._outline_payload(self.ask)
        rule = payload["scene_type_requirements"]["asset_ref"]
        self.assertIn("selected fact_ids' fact_evidence_map refs", rule)
        self.assertIn("research.assets evidence_ref", rule)
        self.assertNotIn("allowed_asset_refs", payload)

    def test_deterministic_authority_stress_matrix(self):
        patterns = ([], ["fact"], ["stray", "fact"], ["outside", "fact"],
                    ["stray", "stray", "fact"])
        for kind, selected, asset_kind, present, pattern in itertools.product(
            sorted(EVIDENCE_TYPES), ("F0001", "F0002"),
            ("fact", "research"), (False, True), patterns,
        ):
            fact_ref = "E0001" if selected == "F0001" else "E0002"
            asset = fact_ref if asset_kind == "fact" else "E0090"
            refs = [fact_ref if item == "fact" else
                    "E9990" if item == "stray" else "E9999" for item in pattern]
            if not present:
                refs = [ref for ref in refs if ref != asset]
            elif asset not in refs:
                refs.append(asset)
            raw = {"type": kind, "purpose": CLAIMS[0], "fact_ids": [selected],
                   "evidence_refs": refs, "asset_ref": asset}
            with self.subTest(kind=kind, selected=selected,
                              asset_kind=asset_kind, present=present, pattern=pattern):
                result = planner._canonical_outline_intent(raw, 1, self.allowed, self.ask)
                self.assertEqual(result["fact_ids"], [selected])
                self.assertEqual(set(result["evidence_refs"]), {fact_ref, asset})
                self.assertEqual(len(result["evidence_refs"]),
                                 len(set(result["evidence_refs"])))
                self.assertLessEqual(len(result["evidence_refs"]), 6)
                planner._validate_scene_facts(result, self.ask)
                self.assertEqual(
                    planner._canonical_outline_intent(result, 1, self.allowed, self.ask),
                    result,
                )


class MultipartContractProvider:
    def __init__(self, *, unsupported_narration=False):
        self.calls = []
        self.unsupported_narration = unsupported_narration

    def complete_json(self, _system, user):
        payload = json.loads(user)
        self.calls.append(payload)
        if "checks" in payload:
            return {"decisions": [{
                "id": check["id"], "supported": True,
                "propositions": [{"text": proposition, "support_indices": [0]}
                                 for proposition in check["required_propositions"]],
            } for check in payload["checks"]]}
        mode = payload.get("storyboard_mode")
        if mode is None:
            raise OutputLimitExceeded("full storyboard intentionally overflows")
        if mode == "outline":
            facts = payload["research"]["facts"]
            return {
                "version": 1, "title": "Grounded WidgetEngine", "slug": "widget-engine",
                "summary": facts[0]["claim"], "scene_intents": [
                    {"type": "HERO", "purpose": facts[0]["claim"],
                     "fact_ids": [facts[0]["fact_id"]], "evidence_refs": [],
                     "asset_ref": "E0090"},
                    {"type": "CODE", "purpose": facts[1]["claim"],
                     "fact_ids": ["F9999"], "evidence_refs": ["E0002", "E0002"]},
                    {"type": "PROJECT_EVIDENCE", "purpose": facts[2]["claim"],
                     "fact_ids": [facts[2]["fact_id"]],
                     "evidence_refs": ["E0001", "E0003"], "asset_ref": "E0090"},
                    {"type": "HARDWARE_EVIDENCE", "purpose": facts[0]["claim"],
                     "fact_ids": [facts[0]["fact_id"]],
                     "evidence_refs": ["E0001"], "asset_ref": "E0090"},
                ],
            }
        if mode == "scenes":
            claims = {fact["fact_id"]: fact["claim"] for fact in payload["research"]["facts"]}
            scenes = []
            for template in payload["required_output"]["scenes"]:
                scene = copy.deepcopy(template)
                scene["title"] = "Supported scene"
                scene["narration"] = claims[scene["fact_ids"][0]]
                if scene["id"] == "s001":
                    scene["evidence_refs"] = []
                if scene["id"] == "s003":
                    scene["evidence_refs"] = ["E0003"]
                    scene.pop("asset_ref")
                if scene["id"] == "s001" and self.unsupported_narration:
                    scene["narration"] = "WidgetEngine guarantees unsupported predictions."
                scenes.append(scene)
            return {"scenes": scenes}
        raise AssertionError(f"Unexpected mode: {mode}")


class MultipartContractIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "prompts").mkdir()
        (root / "prompts" / "storyboard.txt").write_text("Plan grounded scenes.")
        self.project = root / "projects" / "demo"
        (self.project / "sources").mkdir(parents=True)
        (self.project / "sources" / "source.json").write_text(
            json.dumps({"kind": "website", "source": "https://example.test/widget"})
        )
        self.inventory = {"evidence": [
            {"ref": f"E{i:04d}", "kind": "document", "evidence_role": "primary",
             "relative_path": f"docs/{i}.md", "excerpt": claim}
            for i, claim in enumerate(CLAIMS, 1)
        ] + [{"ref": "E0090", "kind": "media", "evidence_role": "primary",
              "relative_path": "media/visual.png"}]}
        self.research = {"version": 1, "facts": [
            {"claim": claim, "evidence_refs": [f"E{i:04d}"],
             "support": [{"evidence_ref": f"E{i:04d}", "text": claim}],
             "phase": "final", "confidence": "high"}
            for i, claim in enumerate(CLAIMS, 1)
        ], "assets": [{"evidence_ref": "E0090", "purpose": "Visual only"}]}

    def run_plan(self, provider):
        return planner.plan(provider, self.research, self.inventory, self.project,
                            "WidgetEngine", "Storyboard from supplied facts.",
                            max_retries=0)

    def test_full_overflow_enters_multipart_and_checkpoints_canonical_intents(self):
        provider = MultipartContractProvider()
        with patch("source2reel.planner._canonicalize_outline_metadata",
                   wraps=planner._canonicalize_outline_metadata) as outline_grounding, patch(
                   "source2reel.planner._validate_narration_grounding",
                   wraps=planner._validate_narration_grounding) as narration_grounding:
            episode = self.run_plan(provider)
        self.assertEqual(outline_grounding.call_count, 1)
        self.assertGreaterEqual(narration_grounding.call_count, 2)
        self.assertEqual([scene["fact_ids"] for scene in episode["scenes"]],
                         [["F0001"], ["F0002"], ["F0003"], ["F0001"]])
        self.assertEqual([scene["evidence_refs"] for scene in episode["scenes"]],
                         [["E0090", "E0001"], ["E0002"], ["E0003", "E0090"],
                          ["E0001", "E0090"]])
        self.assertEqual(episode["scenes"][2]["asset_ref"], "E0090")
        validate_episode(episode, {"E0001", "E0002", "E0003", "E0090"},
                         require_integrated_presentation=True)

        parts = self.project / "manifests" / "storyboard-parts"
        self.assertTrue((parts / "recovery.json").exists())
        saved = json_load(parts / "outline.json")["result"]["scene_intents"]
        self.assertEqual([intent["evidence_refs"] for intent in saved],
                         [["E0090", "E0001"], ["E0002"], ["E0003", "E0090"],
                          ["E0001", "E0090"]])
        part_scenes = [scene for path in sorted(parts.glob("part-[0-9][0-9][0-9].json"))
                       for scene in json_load(path)["result"]["scenes"]]
        self.assertEqual(part_scenes, episode["scenes"])
        self.assertEqual(json_load(self.project / "episode.json"), episode)
        before = len(provider.calls)
        self.assertEqual(self.run_plan(provider), episode)
        self.assertEqual(len(provider.calls), before)

    def test_canonical_refs_do_not_authorize_unsupported_narration(self):
        provider = MultipartContractProvider(unsupported_narration=True)
        episode = self.run_plan(provider)
        self.assertEqual(episode["scenes"][0]["narration"], CLAIMS[0])
        self.assertEqual(episode["scenes"][0]["fact_ids"], ["F0001"])
        self.assertNotIn("unsupported predictions", json.dumps(episode))
        split = self.project / "manifests" / "storyboard-parts"
        self.assertEqual(json_load(split / "part-001-a.json")["result"]["scenes"][0]["narration"],
                         CLAIMS[0])

    def test_scene_part_discards_asset_on_non_evidence_type(self):
        class NonEvidenceAsset(MultipartContractProvider):
            def complete_json(self, system, user):
                result = super().complete_json(system, user)
                if json.loads(user).get("storyboard_mode") == "scenes":
                    for scene in result["scenes"]:
                        if scene["id"] == "s002":
                            scene["asset_ref"] = "E0002"
                return result

        episode = self.run_plan(NonEvidenceAsset())
        self.assertNotIn("asset_ref", episode["scenes"][1])
        part = json_load(self.project / "manifests" / "storyboard-parts" / "part-001.json")
        self.assertNotIn("asset_ref", part["result"]["scenes"][1])


if __name__ == "__main__":
    unittest.main()
