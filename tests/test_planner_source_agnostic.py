"""Deterministic planner acceptance across source classes and raw draft shapes."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.schema import SCENE_CONTRACTS, validate_episode
from source2reel.util import json_load


def source(claims, *, visual=(), roles=None):
    entries = []
    facts = []
    roles = roles or {}
    for number, claim in enumerate(claims, 1):
        ref = f"E{number:04d}"
        entries.append({"ref": ref, "kind": "document", "relative_path": f"docs/{number}.md",
                        "evidence_role": roles.get(number, "primary"), "excerpt": claim})
        facts.append({"claim": claim, "evidence_refs": [ref],
                      "support": [{"evidence_ref": ref, "text": claim}],
                      "phase": "final", "confidence": "high"})
    for number in visual:
        entries.append({"ref": f"E{number:04d}", "kind": "media",
                        "relative_path": f"media/{number}.png", "evidence_role": "primary"})
    return {"evidence": entries}, {"version": 1, "facts": facts,
                                    "assets": [{"evidence_ref": f"E{number:04d}",
                                                "purpose": "Authentic visual"} for number in visual]}


def project(root):
    (root / "prompts").mkdir()
    (root / "prompts" / "storyboard.txt").write_text("Plan grounded scenes.")
    result = root / "projects" / "fixture"
    (result / "sources").mkdir(parents=True)
    (result / "sources" / "source.json").write_text(json.dumps({
        "kind": "website", "source": "https://example.test/fixture",
    }))
    return result


class ProfileProvider:
    """Overflow a full answer, then realize selected facts without model inference."""
    def __init__(self, *, visual=False, retry_impossible=False, bad_purpose=False):
        self.visual = visual
        self.retry_impossible = retry_impossible
        self.bad_purpose = bad_purpose
        self.calls = []

    def complete_json(self, _system, user):
        request = json.loads(user)
        self.calls.append(request)
        if "checks" in request:
            return {"decisions": [{"id": check["id"], "supported": True,
                    "propositions": [{"text": text, "support_indices": [0]}
                                     for text in check["required_propositions"]]}
                    for check in request["checks"]]}
        if request.get("storyboard_mode") is None:
            raise OutputLimitExceeded("force multipart recovery")
        if request["storyboard_mode"] == "outline":
            facts = request["research"]["facts"]
            kind = "PROJECT_EVIDENCE" if self.visual else "DATA_FLOW"
            if self.retry_impossible and not any("validation_feedback" in call for call in self.calls):
                kind = "HERO"
            first = {"type": kind, "purpose": facts[0]["claim"],
                     "fact_ids": [facts[0]["fact_id"]],
                     "evidence_refs": [facts[-1]["evidence_refs"][0],
                                       facts[0]["evidence_refs"][0],
                                       facts[0]["evidence_refs"][0]],
                     "asset_ref": "E0090" if self.visual else {"malformed": ["irrelevant"]}}
            if self.retry_impossible and kind == "HERO":
                first["asset_ref"] = "E0001"  # A document cannot render as visual evidence.
            if self.bad_purpose:
                first["purpose"] = "The machine reads private thoughts without sensors."
            second = {"type": "CODE", "purpose": facts[1]["claim"],
                      "fact_ids": [facts[1]["fact_id"]], "evidence_refs": []}
            return {"version": 1, "title": "Grounded fixture", "slug": "fixture",
                    "summary": facts[0]["claim"], "scene_intents": [first, second]}
        facts = {fact["fact_id"]: fact for fact in request["research"]["facts"]}
        scenes = []
        for template in request["required_output"]["scenes"]:
            scene = copy.deepcopy(template)
            scene["narration"] = facts[scene["fact_ids"][0]]["claim"]
            scene["evidence_refs"] = ["E9999", *scene["evidence_refs"],
                                      *scene["evidence_refs"]]
            scene["annotations"] = []
            if scene["type"] == "DATA_FLOW":
                scene["asset_ref"] = ["uninterpreted", "noise"]
                scene["media"] = {"start_seconds": "wrong type"}
                scene["diagram"] = {"nodes": ["Original input", "Documented output"]}
            if scene["type"] == "PROJECT_EVIDENCE":
                scene["evidence_refs"] = []  # The selected fact and fixed asset recover.
            scenes.append(scene)
        return {"scenes": scenes}


class SourceProfileTests(unittest.TestCase):
    def test_full_response_repairs_noise_and_no_media_retry(self):
        inventory, research = source(["PaperTool reads the local manifest.",
                                      "PaperTool produces a local summary."])

        class DirectProvider:
            def __init__(self):
                self.requests = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.requests.append(request)
                if "checks" in request:
                    raise AssertionError("Exact source claims need no model verifier")
                fact = request["research"]["facts"][0]
                if "validation_feedback" not in request:
                    return {"version": 1, "title": "PaperTool", "scenes": [{
                        "type": "HERO", "narration": fact["claim"],
                        "fact_ids": [fact["fact_id"]], "asset_ref": "E0001",
                        "evidence_refs": ["E0001"],
                    }]}
                return {"version": "1", "title": "PaperTool", "scenes": [{
                    "type": "DATA_FLOW", "narration": fact["claim"],
                    "fact_ids": [fact["fact_id"]], "evidence_refs": ["E0002", "E0001", "E0001"],
                    "asset_ref": {"malformed": "irrelevant"},
                    "media": {"start_seconds": "invalid"},
                    "diagram": {"nodes": ["Local manifest", "Local summary"]},
                }]}

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = DirectProvider()
            episode = planner.plan(provider, research, inventory, path, "PaperTool", "",
                                   max_retries=1)
            self.assertEqual(len(provider.requests), 2)
            self.assertIn("no usable visual asset", provider.requests[1]["validation_feedback"])
            self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0001"])
            self.assertNotIn("asset_ref", episode["scenes"][0])
            self.assertNotIn("media", episode["scenes"][0])
            self.assertEqual(json_load(path / "episode.json"), episode)
            self.assertFalse((path / "manifests" / "storyboard-parts" / "outline.json").exists())

    def test_small_documented_software_project_overflow_discards_data_flow_asset(self):
        claims = ["TraceKit sends a captured trace to a local parser.",
                  "TraceKit does not upload traces to a cloud service.",
                  "TraceKit's tests cover parser errors.",
                  "The source module stores parsed records."]
        inventory, research = source(claims)
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = ProfileProvider()
            episode = planner.plan(provider, research, inventory, path, "TraceKit",
                                   "Explain trace parsing and its offline behavior.", max_retries=0)
            first = episode["scenes"][0]
            self.assertEqual(first["type"], "DATA_FLOW")
            self.assertNotIn("asset_ref", first)
            self.assertNotIn("media", first)
            self.assertEqual(first["fact_ids"], ["F0001"])
            self.assertEqual(first["evidence_refs"], ["E0001"])
            self.assertEqual(episode["scenes"][1]["narration"], claims[1])
            self.assertNotIn("asset_ref", json_load(path / "manifests" / "storyboard-parts" /
                                                    "outline.json")["result"]["scene_intents"][0])
            part = json_load(path / "manifests" / "storyboard-parts" / "part-001.json")
            self.assertNotIn("asset_ref", part["result"]["scenes"][0])
            prior_calls = len(provider.calls)
            self.assertEqual(planner.plan(provider, research, inventory, path, "TraceKit",
                                          "Explain trace parsing and its offline behavior.",
                                          max_retries=0), episode)
            self.assertEqual(len(provider.calls), prior_calls)

    def test_media_rich_hardware_uses_only_selected_visual(self):
        inventory, research = source([
            "BoardLab first streamed measurements to a workstation.",
            "BoardLab's final device displays measurements locally.",
            "A service manual documents the older serial connection.",
        ], visual=(90, 91))
        research["assets"] = research["assets"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            episode = planner.plan(ProfileProvider(visual=True), research, inventory, path,
                                   "BoardLab", "Explain the final display.", max_retries=0)
            first = episode["scenes"][0]
            self.assertEqual(first["asset_ref"], "E0090")
            self.assertEqual(first["evidence_refs"], ["E0090", "E0001"])
            ask = planner._make_ask(research, planner._media_inventory(inventory),
                                    inventory["evidence"], "BoardLab", "")
            bad = {"type": "HERO", "purpose": research["facts"][0]["claim"],
                   "fact_ids": ["F0001"], "evidence_refs": ["E0001"], "asset_ref": "E0091"}
            with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
                planner._canonical_outline_intent(bad, 1,
                                                  {entry["ref"] for entry in inventory["evidence"]}, ask)
            self.assertNotIn("E0090", ask["research"]["facts"][0]["evidence_refs"])

    def test_scene_part_retry_reports_fixed_asset_not_outline_prose(self):
        inventory, research = source(["BoardLab reads a sensor.",
                                      "BoardLab writes measured values."], visual=(90, 91))
        research["assets"] = research["assets"][:1]

        class WrongOnce(ProfileProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                result = super().complete_json(system, user)
                if request.get("storyboard_mode") == "scenes" and (
                    not request.get("validation_feedback")
                ):
                    for scene in result["scenes"]:
                        if scene["type"] == "PROJECT_EVIDENCE":
                            scene["asset_ref"] = "E0091"
                return result

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = WrongOnce(visual=True)
            episode = planner.plan(provider, research, inventory, path, "BoardLab", "",
                                   max_retries=1)
            requests = [call for call in provider.calls if call.get("storyboard_mode") == "scenes"]
            self.assertEqual(len(requests), 2)
            self.assertIn("fixed outline assets", requests[1]["validation_feedback"])
            self.assertNotIn("unsupported summary", requests[1]["validation_feedback"])
            self.assertEqual(episode["scenes"][0]["asset_ref"], "E0090")
            part = json_load(path / "manifests" / "storyboard-parts" / "part-001.json")
            self.assertEqual(part["result"]["scenes"][0]["asset_ref"], "E0090")

    def test_document_only_retry_chooses_non_evidence_type(self):
        inventory, research = source([
            "A public guide describes the service request flow.",
            "The reference manual limits requests to local files.",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = ProfileProvider(retry_impossible=True)
            episode = planner.plan(provider, research, inventory, path, "GuideService",
                                   "Explain the documented request flow.", max_retries=1)
            requests = [call for call in provider.calls if call.get("storyboard_mode") == "outline"]
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["visual_asset_refs"], [])
            self.assertNotIn("HERO", requests[0]["allowed_scene_types"])
            self.assertIn("no usable visual asset", requests[1]["validation_feedback"])
            self.assertEqual(episode["scenes"][0]["type"], "DATA_FLOW")
            self.assertNotIn("asset_ref", episode["scenes"][0])
            self.assertFalse(any("asset_ref" in scene for scene in episode["scenes"]))

    def test_low_media_inventory_does_not_offer_unsupported_visuals(self):
        inventory, research = source(["ArchiveTool reads signed documents.",
                                      "ArchiveTool writes a local index."])
        inventory["evidence"].extend([
            {"ref": "E0090", "kind": "media", "relative_path": "decorative.png",
             "evidence_role": "primary"},
            {"ref": "E0091", "kind": "media", "relative_path": "diagram.svg",
             "evidence_role": "primary"},
        ])
        research["assets"].append({"evidence_ref": "E0091", "purpose": "Vector diagram"})
        ask = planner._make_ask(research, planner._media_inventory(inventory),
                                inventory["evidence"], "ArchiveTool", "")
        self.assertEqual(ask["visual_asset_refs"], [])
        self.assertFalse(set(planner.EVIDENCE_TYPES) & set(ask["allowed_scene_types"]))
        with self.assertRaisesRegex(StructuredOutputError, "no planner-scoped visual asset_ref"):
            planner._canonical_outline_intent({"type": "HERO", "purpose": research["facts"][0]["claim"],
                "fact_ids": ["F0001"], "evidence_refs": ["E0001"], "asset_ref": "E0091"},
                1, {entry["ref"] for entry in inventory["evidence"]}, ask)

    def test_large_self_referential_material_does_not_take_subject_authority(self):
        primary = ["ForgeEngine indexes original project sources.",
                   "ForgeEngine plans a cited storyboard from verified facts.",
                   "ForgeEngine renders the planned output locally."]
        secondary = [f"Example test fixture {i} describes an embedded sample."
                     for i in range(4, 18)]
        generated = [f"Generated report {i} records an earlier run." for i in range(18, 29)]
        roles = {i: "embedded_reference" for i in range(4, 18)}
        roles.update({i: "generated_artifact" for i in range(18, 29)})
        inventory, research = source(primary + secondary + generated, roles=roles)
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            episode = planner.plan(ProfileProvider(), research, inventory, path,
                                   "ForgeEngine", "Explain sources and storyboard planning.",
                                   max_retries=0)
            self.assertEqual([scene["fact_ids"] for scene in episode["scenes"]],
                             [["F0001"], ["F0002"]])
            self.assertEqual([scene["evidence_refs"] for scene in episode["scenes"]],
                             [["E0001"], ["E0002"]])
            manifest = json_load(path / "manifests" / "planner-evidence.json")
            self.assertEqual(manifest["research"]["facts"][3]["subject_scope"], "supporting_only")
            ask = planner._make_ask(research, [], inventory["evidence"], "ForgeEngine",
                                    "Explain sources and storyboard planning.")
            self.assertGreaterEqual(len(set(ask["priority_fact_ids"]) & {"F0001", "F0002"}), 2)
            self.assertNotIn("E0028", {ref for scene in episode["scenes"]
                                       for ref in scene["evidence_refs"]})

    def test_bad_outline_purpose_is_replaced_before_scene_checkpoint(self):
        inventory, research = source(["SourceDoc records local input.",
                                      "SourceDoc validates output fields."])
        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = ProfileProvider(bad_purpose=True)
            episode = planner.plan(provider, research, inventory, path, "SourceDoc", "",
                                   max_retries=0)
            saved = json_load(path / "manifests" / "storyboard-parts" / "outline.json")
            self.assertEqual(saved["result"]["scene_intents"][0]["purpose"], research["facts"][0]["claim"])
            self.assertEqual(saved["result"]["scene_intents"][0]["fact_ids"], ["F0001"])
            self.assertEqual(episode["scenes"][0]["narration"], research["facts"][0]["claim"])
            self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 1)

    def test_unsupported_narration_fails_even_with_canonical_refs(self):
        inventory, research = source(["SourceDoc records local input.",
                                      "SourceDoc validates output fields."])

        class UnsupportedNarration(ProfileProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "checks" in request:
                    return {"decisions": [{"id": check["id"], "supported": False,
                            "propositions": []} for check in request["checks"]]}
                result = super().complete_json(system, user)
                if request.get("storyboard_mode") == "scenes":
                    result["scenes"][0]["narration"] = "SourceDoc guarantees impossible predictions."
                return result

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            with self.assertRaisesRegex((RuntimeError, StructuredOutputError),
                                        "unsupported factual proposition|grounding failed"):
                planner.plan(UnsupportedNarration(), research, inventory, path,
                             "SourceDoc", "", max_retries=0)
            self.assertFalse((path / "episode.json").exists())

    def test_legacy_accepted_episode_still_validates(self):
        path = Path(__file__).resolve().parents[1] / "projects" / "cisco-doom-episode-001" / "episode.json"
        validate_episode(json_load(path))


class CanonicalPropertyMatrix(unittest.TestCase):
    def test_12_types_fact_ref_and_noise_combinations(self):
        inventory, research = source(["FlowCore reads documented input.",
                                      "FlowCore writes documented output."], visual=(90,))
        ask = planner._make_ask(research, planner._media_inventory(inventory),
                                inventory["evidence"], "FlowCore", "",
                                ["https://example.test/source"])
        allowed = {entry["ref"] for entry in inventory["evidence"]}
        patterns = ([], ["E0001"], ["E0001", "E0001"], ["E0002", "E0001"],
                    ["E9999", "E0001", "E0001"], ["E0002"])
        noise = ({"asset_ref": {"junk": 1}, "media": {"start_seconds": "invalid"}},
                 {"asset_ref": None, "diagram": {"broken": True}},
                 {"asset_ref": "E0090", "unknown_field": {"junk": 1}},
                 {"asset_ref": ["invalid"], "captions": {"enabled": False}})
        checked = 0
        for kind, pattern, extra, fact_state in itertools.product(
            sorted(SCENE_CONTRACTS), patterns, noise, ("selected", "unique recovery"),
        ):
            if fact_state == "unique recovery" and pattern not in (["E0001"], ["E0001", "E0001"]):
                continue
            contract = SCENE_CONTRACTS[kind]
            raw = {"type": kind, "purpose": research["facts"][0]["claim"],
                   "evidence_refs": pattern + ["E0090"]
                   if contract.requires_asset_ref else pattern, **extra}
            if fact_state == "selected":
                raw["fact_ids"] = ["F0001"]
            if contract.requires_asset_ref:
                raw["asset_ref"] = "E0090"
            with self.subTest(kind=kind, pattern=pattern, extra=extra, fact_state=fact_state):
                intent = planner._canonical_outline_intent(raw, 1, allowed, ask)
                self.assertEqual(intent["fact_ids"], ["F0001"])
                self.assertEqual(planner._canonical_outline_intent(intent, 1, allowed, ask), intent)
                self.assertLessEqual(len(intent["evidence_refs"]), 6)
                expected_refs = ({"E0001"} if contract.requires_facts or "E0001" in pattern
                                 else set())
                self.assertEqual(set(intent["evidence_refs"]) - {"E0090"}, expected_refs)
                scene = {**raw, "id": "s001", "title": "Flow", "narration": research["facts"][0]["claim"],
                         "annotations": []}
                if contract.requires_diagram:
                    scene["diagram"] = {"nodes": ["Documented input", "Documented output"]}
                if contract.allows_media_timing and "media" in scene:
                    scene["media"] = {"start_seconds": 0.5}
                outline = {"title": "FlowCore", "scene_intents": [intent]}
                if kind == "OUTRO":
                    outline["presentation"] = {"outro": {"headline": ["FlowCore"],
                        "links": [{"label": "Source", "url": ["https://example.test/source"]}]}}
                full = planner._canonical_full_episode(planner._repair_episode_shape({
                    "version": 1, "title": "FlowCore", "scenes": [scene],
                    **({"presentation": outline["presentation"]} if kind == "OUTRO" else {}),
                }, allowed), ask, allowed)
                part = planner._normalize_scene_part({"scenes": [scene]}, [intent], allowed,
                                                     outline, ask)["scenes"][0]
                self.assertEqual(full["scenes"][0], part)
                self.assertEqual(planner._canonical_full_episode(full, ask, allowed), full)
                validate_episode(full, allowed, require_integrated_presentation=True)
                if not contract.allows_asset_ref:
                    self.assertNotIn("asset_ref", part)
                if not contract.allows_media_timing:
                    self.assertNotIn("media", part)
                self.assertNotIn("unknown_field", part)
                checked += 1
        self.assertEqual(checked, 12 * (len(patterns) + 2) * len(noise))

    def test_ambiguous_recovery_and_unauthorized_assets_fail_closed(self):
        inventory, research = source(["FlowCore reads input.", "FlowCore validates input."],
                                     visual=(90, 91))
        research["facts"][1]["evidence_refs"] = ["E0001"]
        research["facts"][1]["support"] = [{"evidence_ref": "E0001",
                                             "text": research["facts"][1]["claim"]}]
        inventory["evidence"][0]["excerpt"] += " " + research["facts"][1]["claim"]
        research["assets"] = research["assets"][:1]
        ask = planner._make_ask(research, planner._media_inventory(inventory),
                                inventory["evidence"], "FlowCore", "")
        allowed = {entry["ref"] for entry in inventory["evidence"]}
        raw = {"type": "HERO", "purpose": research["facts"][0]["claim"],
               "evidence_refs": ["E0001"], "asset_ref": "E0090"}
        with self.assertRaisesRegex(StructuredOutputError, "allowed_fact_ids"):
            planner._canonical_outline_intent(raw, 1, allowed, ask)
        with self.assertRaisesRegex(StructuredOutputError, "asset_ref"):
            planner._canonical_outline_intent({**raw, "fact_ids": ["F0001"],
                                              "asset_ref": "E0091"}, 1, allowed, ask)
        with self.assertRaisesRegex(StructuredOutputError, "allowed_fact_ids"):
            planner._canonical_outline_intent({**raw, "fact_ids": ["F9999"]}, 1, allowed, ask)
        with self.assertRaisesRegex(StructuredOutputError, "allowed_fact_ids"):
            planner._canonical_full_episode({"version": 1, "title": "FlowCore", "scenes": [{
                **raw, "id": "s001", "narration": research["facts"][0]["claim"],
            }]}, ask, allowed)

    def test_null_refs_are_derived_only_from_selected_facts(self):
        inventory, research = source(["FlowCore reads input.", "FlowCore writes output."])
        ask = planner._make_ask(research, [], inventory["evidence"], "FlowCore", "")
        allowed = {"E0001", "E0002"}
        raw = {"type": "CODE", "purpose": research["facts"][0]["claim"],
               "fact_ids": ["F0001"], "evidence_refs": None, "asset_ref": {"ignored": True}}
        intent = planner._canonical_outline_intent(raw, 1, allowed, ask)
        self.assertEqual(intent["evidence_refs"], ["E0001"])
        self.assertNotIn("asset_ref", intent)
        full = planner._canonical_full_episode({"version": 1, "title": "FlowCore",
            "scenes": [{**raw, "id": "s001", "narration": research["facts"][0]["claim"]}]},
            ask, allowed)
        self.assertEqual(full["scenes"][0]["evidence_refs"], ["E0001"])
        validate_episode(full, allowed)


if __name__ == "__main__":
    unittest.main()
