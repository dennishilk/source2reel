"""Selected-fact temporal suitability at the physical BoringOS boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner, research
from source2reel.editorial import EDITORIAL_CONTRACT, SceneTypeUnsuitable
from source2reel.grounding import GROUNDING_CONTRACT
from source2reel.media_ai import MEDIA_INSPECTION_CONTRACT
from source2reel.providers import StructuredOutputError
from source2reel.util import json_dump, json_load


F0004 = (
    "The M63 physical acceptance proved a real writable USB-root lifecycle on "
    "Cthulhu: create and write files, reboot, boot again, read the persisted "
    "data, and then cleanly power off through ACPI S5."
)
F0012 = (
    "The M66 physical runtime is frozen at commit "
    "8ccd618dfc4e8163821de552a3c912bab4e6f36a with a 96 MiB USB image."
)


def physical_ask():
    facts = []
    for index in range(1, 13):
        claim = F0004 if index == 4 else F0012 if index == 12 else f"Documented fact {index}."
        ref = "E0897" if index == 12 else "E0036"
        facts.append({"claim": claim, "evidence_refs": [ref],
                      "support": [{"evidence_ref": ref, "text": claim}],
                      "phase": "final", "confidence": "high"})
    evidence = [{"ref": ref, "kind": "document", "relative_path": f"docs/{ref}.md",
                 "evidence_role": "primary"} for ref in ("E0036", "E0897")]
    ask = planner._make_ask({"version": 1, "facts": facts, "assets": []}, [],
                            evidence, "BoringOS", "Explain the documented project.")
    return ask, {item["ref"] for item in evidence}


def physical_outline():
    return {"version": 1, "title": "BoringOS", "slug": "boringos",
            "summary": F0012, "scene_intents": [{
                "id": "s003", "type": "TIMELINE", "purpose": F0012,
                "fact_ids": ["F0004", "F0012"],
                "evidence_refs": ["E0036", "E0897"],
            }]}


class NoOutlineProvider:
    def __init__(self):
        self.outline_calls = 0
        self.scene_requests = []

    def complete_json(self, _system, user):
        payload = json.loads(user)
        if payload.get("storyboard_mode") == "outline":
            self.outline_calls += 1
            raise AssertionError("cached outline should be re-normalized")
        if payload.get("storyboard_mode") == "scenes":
            self.scene_requests.append(payload)
            return {"scenes": [
                {**template, "title": "Physical runtime",
                 "narration": " ".join(fact["claim"] for fact in payload["research"]["facts"]
                                         if fact["fact_id"] in template["fact_ids"])}
                for template in payload["required_output"]["scenes"]
            ]}
        if "checks" in payload:
            return {"decisions": [
                {"id": check["id"], "supported": True,
                 "propositions": [{"text": text, "support_indices": [0]}
                                  for text in check["required_propositions"]]}
                for check in payload["checks"]
            ]}
        raise AssertionError("Unexpected model request")


class TimelineSuitabilityTests(unittest.TestCase):
    def test_physical_selection_becomes_summary_before_scene_request(self):
        ask, allowed = physical_ask()
        selected = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
        raw = physical_outline()
        self.assertEqual(selected["F0004"]["claim"], F0004)
        self.assertEqual(selected["F0012"]["claim"], F0012)
        self.assertEqual(selected["F0004"]["evidence_refs"], ["E0036"])
        self.assertEqual(selected["F0012"]["evidence_refs"], ["E0897"])
        with self.assertRaisesRegex(SceneTypeUnsuitable, "temporal or order"):
            planner.validate_scene_type(raw["scene_intents"][0], selected,
                                        ask["evidence_index"])
        canonical = planner._normalize_outline(raw, allowed, ask, check_coverage=False)
        scene = canonical["scene_intents"][0]
        self.assertEqual(scene["id"], "s003")
        self.assertEqual(scene["type"], "SUMMARY")
        self.assertEqual(scene["fact_ids"], ["F0004", "F0012"])
        self.assertEqual(scene["evidence_refs"], ["E0036", "E0897"])
        request = planner._scene_part_payload(ask, canonical, [scene], 1, 1, "scope")
        self.assertEqual(request["required_output"]["scenes"][0]["type"], "SUMMARY")
        self.assertNotIn("diagram", request["required_output"]["scenes"][0])
        self.assertNotIn("diagram.nodes", json.dumps(request["required_output"]))
        self.assertNotIn("diagram.steps", json.dumps(request["required_output"]))

    def test_selected_facts_need_a_shared_clock_or_explicit_named_order(self):
        cases = [
            ([F0004], True),
            (["The process writes the file, then reboots, then reads the data."], True),
            ([F0004, F0012], False),
            (["The process writes a file, then reboots.",
              "The display service paints the desktop."], False),
            (["Milestone M63 includes a writable USB root.",
              "Milestone M66 includes a 96 MiB image."], False),
            (["The M63 runtime writes files, then reboots.",
              "The M66 runtime writes files on USB."], False),
            (["Version 1.2 has a new runtime.", "Version 1.3 adds the GUI."], False),
            (["Version 2026-01-01 has a CLI.",
              "Version 2026-03-01 has a GUI."], False),
            (["Version A shipped in January 2026.",
              "Version B shipped in March 2026."], True),
            (["Version A shipped.", "Version B followed Version A."], True),
            (["Version A shipped.",
              "Version A shipped, then Version B followed."], True),
            (["Version A shipped.", "Version B shipped.",
              "Version B followed Version A."], True),
            (["Capture completed before aggregation.",
              "Aggregation finished after capture."], True),
            (["The initial capture records events.",
              "The final report followed the initial capture."], True),
            (["Version A shipped in January 2026.",
              "The desktop service paints a blue background."], False),
        ]
        for claims, suitable in cases:
            with self.subTest(claims=claims):
                facts = {f"F{i:04d}": {"claim": claim}
                         for i, claim in enumerate(claims, 1)}
                scene = {"id": "s001", "type": "TIMELINE", "fact_ids": list(facts)}
                if suitable:
                    planner.validate_scene_type(scene, facts, [])
                else:
                    with self.assertRaises(SceneTypeUnsuitable):
                        planner.validate_scene_type(scene, facts, [])

    def test_genuinely_temporal_scene_still_rejects_unsupported_required_node(self):
        claim = "The process writes the file, then reboots, then reads the persisted data."
        fact = {"fact_id": "F0001", "claim": claim, "evidence_refs": ["E0001"],
                "support": [{"evidence_ref": "E0001", "text": claim}]}
        scene = {"id": "s001", "type": "TIMELINE", "fact_ids": ["F0001"],
                 "evidence_refs": ["E0001"], "diagram": {"nodes": [
                     "The process writes the file", "An unsupported cloud upload occurs",
                 ]}}
        planner.validate_scene_type(scene, {"F0001": fact}, [])
        with self.assertRaisesRegex(ValueError,
                                    r"diagram\.nodes\[1\] has unsupported factual content"):
            planner._validate_structured_grounding(None, [scene],
                {"research": {"facts": [fact]}}, None)
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "events.md",
                     "evidence_role": "primary"}]
        ask = planner._make_ask({"version": 1, "facts": [{key: value for key, value
            in fact.items() if key != "fact_id"}], "assets": []}, [], evidence,
            "Event recorder", "Explain the documented sequence.")
        outline = planner._normalize_outline({"version": 1, "title": "Events",
            "slug": "events", "summary": claim, "scene_intents": [{
                "id": "s001", "type": "TIMELINE", "purpose": claim,
                "fact_ids": ["F0001"], "evidence_refs": ["E0001"],
            }]}, {"E0001"}, ask, check_coverage=False)
        self.assertEqual(outline["scene_intents"][0]["type"], "TIMELINE")
        request = planner._scene_part_payload(ask, outline, outline["scene_intents"],
                                              1, 1, "scope")
        self.assertIn("diagram", request["required_output"]["scenes"][0])
        returned = {**scene, "title": "Observed sequence", "narration": claim}
        with self.assertRaisesRegex(StructuredOutputError,
                                    r"diagram\.nodes\[1\] has unsupported factual content"):
            planner._normalize_scene_part({"scenes": [returned]}, outline["scene_intents"],
                                          {"E0001"}, outline, ask)

    def test_current_checkpoint_is_recanonicalized_without_an_outline_model_call(self):
        ask, _ = physical_ask()
        self.assertEqual(EDITORIAL_CONTRACT, "storyboard-editorial-grounding-v6")
        self.assertEqual(ask["editorial_contract"], EDITORIAL_CONTRACT)
        self.assertEqual(research.RESEARCH_SEMANTICS_CONTRACT, "requested-topic-semantics-v21")
        self.assertEqual(GROUNDING_CONTRACT, "mapped-support-kind-v5")
        self.assertEqual(MEDIA_INSPECTION_CONTRACT, "media-inspection-v1")
        provider = NoOutlineProvider()
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            path = project / "manifests/storyboard-parts/outline.json"
            system = "storyboard" + planner._OUTLINE_SYSTEM
            request = json.dumps(planner._outline_payload(ask), ensure_ascii=False)
            digest = hashlib.sha256((system + "\0" + request).encode("utf-8")).hexdigest()
            json_dump(path, {"version": 1, "input_sha256": digest,
                             "result": physical_outline()})
            episode = planner._multipart_episode(provider, "storyboard", ask, project,
                32768, 4096, 1024, 0, None, "physical-scope")
            self.assertEqual(provider.outline_calls, 0)
            self.assertEqual(episode["scenes"][0]["type"], "SUMMARY")
            self.assertEqual(episode["scenes"][0]["fact_ids"], ["F0004", "F0012"])
            self.assertEqual(episode["scenes"][0]["evidence_refs"], ["E0036", "E0897"])
            self.assertEqual(json_load(path)["input_sha256"], digest)
            self.assertEqual(json_load(path)["result"]["scene_intents"][0]["type"],
                             "SUMMARY")
            self.assertEqual(provider.scene_requests[0]["required_output"]["scenes"][0]
                             ["type"], "SUMMARY")
            self.assertNotIn("diagram", provider.scene_requests[0]["required_output"]
                             ["scenes"][0])
            scene_calls = len(provider.scene_requests)
            self.assertEqual(planner._multipart_episode(provider, "storyboard", ask, project,
                32768, 4096, 1024, 0, None, "physical-scope"), episode)
            self.assertEqual(len(provider.scene_requests), scene_calls)
            self.assertEqual(provider.outline_calls, 0)
