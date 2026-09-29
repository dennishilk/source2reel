"""Physical editorial failure and compact, grounded alternatives."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner, research
from source2reel.editorial import SceneTypeUnsuitable, prune_redundant_content_scenes
from source2reel.providers import StructuredOutputError
from source2reel.schema import validate_episode
from source2reel.util import json_load


CLAIMS = [
    "Windows Telemetry Inspector is a passive Windows 11 network diagnostics tool that uses ETW to show network activity by process and service.",
    "ETW callbacks normalize raw provider data into small internal events and enqueue them.",
    "Windows Telemetry Inspector is a passive diagnostics tool that observes Windows network traffic via ETW without modifying or intercepting it.",
    "DNS and scheduled-task relationships are presented as best-effort correlation, not proven causality.",
    "Correlation of DNS and scheduled tasks is best-effort and never implies proven causality.",
    "The tool does not modify traffic, install drivers, disable telemetry, or intercept TLS.",
]
INSTRUCTIONS = (
    "Explain what Windows Telemetry Inspector is, why it exists, and how its architecture "
    "turns Windows ETW network observations into useful local diagnostics. Use only repository "
    "evidence and do not invent capabilities. Clearly distinguish passive observation and "
    "correlation from blocking, interception, or proven causality."
)
BORINGOS_CLAIMS = {
    3: "BoringOS does not use Linux, BSD, or any other kernel; its components are written primarily in C with isolated x86_64 assembly.",
    5: "RAMFS stores file data in the BoringKernel heap and does not add a separate allocator.",
    8: "BoringOS includes a native Ring-3 display service called `boring-display`, installed as `/bin/boring-display`.",
    9: "BoringOS Milestone 22 implements a modern VirtIO block device path using a single PCI device under QEMU.",
}


def boringos_ask():
    refs = {3: "E0036", 5: "E0097", 8: "E0049", 9: "E0049"}
    filler = "The repository documents this project milestone."
    facts = []
    for number in range(1, 10):
        claim = BORINGOS_CLAIMS.get(number, filler)
        ref = refs.get(number, "E0036")
        facts.append({"claim": claim, "evidence_refs": [ref],
                      "support": [{"evidence_ref": ref, "text": claim}],
                      "phase": "final", "confidence": "high"})
    evidence = [{"ref": ref, "kind": "document", "relative_path": f"docs/{ref}.md",
                 "evidence_role": "primary"} for ref in ("E0036", "E0097", "E0049")]
    ask = planner._make_ask({"version": 1, "facts": facts, "assets": []}, [],
                            evidence, "BoringOS", "Explain the documented project.")
    return ask, {entry["ref"] for entry in evidence}


def boringos_outline():
    return {"version": 1, "title": "BoringOS", "slug": "boringos",
            "summary": BORINGOS_CLAIMS[3], "scene_intents": [
                {"id": "s001", "type": "SUMMARY", "purpose": "Project context",
                 "fact_ids": ["F0001"], "evidence_refs": ["E0036"]},
                {"id": "s002", "type": "ARCHITECTURE_DIAGRAM",
                 "purpose": BORINGOS_CLAIMS[3],
                 "fact_ids": ["F0003", "F0005", "F0008", "F0009"],
                 "evidence_refs": ["E0036", "E0097", "E0049"]},
            ]}


def physical_ask():
    refs = [["E0002"], ["E0003"], ["E0002"], ["E0002"], ["E0002", "E0003"], ["E0002"]]
    evidence = [{"ref": ref, "kind": "document", "relative_path": f"docs/{ref}.md",
                 "evidence_role": "primary"} for ref in ("E0002", "E0003")]
    research = {"version": 1, "facts": [
        {"claim": claim, "evidence_refs": selected,
         "support": [{"evidence_ref": ref, "text": claim} for ref in selected],
         "phase": "final", "confidence": "high"}
        for claim, selected in zip(CLAIMS, refs)
    ], "assets": []}
    ask = planner._make_ask(research, [], evidence, "Windows Telemetry Inspector",
                            INSTRUCTIONS, ["https://example.test/source"])
    return ask, {item["ref"] for item in evidence}


def intent(kind, *numbers):
    return {"type": kind, "purpose": CLAIMS[numbers[0] - 1],
            "fact_ids": [f"F{number:04d}" for number in numbers],
            "evidence_refs": list(dict.fromkeys(
                ref for number in numbers for ref in
                (["E0003"] if number == 2 else ["E0002", "E0003"] if number == 5 else ["E0002"])
            ))}


def outline(intents):
    return {"version": 1, "title": "Windows Telemetry Inspector", "slug": "telemetry",
            "summary": CLAIMS[0], "scene_intents": intents,
            "presentation": {"outro": {"headline": ["Local diagnostics"],
                         "links": [{"label": "Source", "url": ["https://example.test/source"]}]}}}


class ApprovingVerifier:
    def complete_json(self, _system, _user):
        raise AssertionError("Deterministic rejection must precede the verifier")


class PhysicalOutlineProvider:
    """Always propose the same unsupported architecture type when asked for an outline."""

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def complete_json(self, _system, user):
        request = json.loads(user)
        self.calls.append(request)
        if request.get("storyboard_mode") == "outline":
            return copy.deepcopy(self.raw)
        if request.get("storyboard_mode") == "scenes":
            facts = {fact["fact_id"]: fact["claim"] for fact in request["research"]["facts"]}
            scenes = []
            for template in request["required_output"]["scenes"]:
                scene = copy.deepcopy(template)
                scene["narration"] = " ".join(facts[fact_id] for fact_id in scene["fact_ids"])
                scene["title"] = "Documented observation"
                if scene["type"] == "DATA_FLOW":
                    scene["diagram"] = {"nodes": [
                        "ETW callbacks normalize raw provider data into small internal events",
                        "small internal events",
                    ]}
                scenes.append(scene)
            return {"scenes": scenes}
        if "checks" in request:
            return {"decisions": [
                {"id": check["id"], "supported": True,
                 "propositions": [{"text": text, "support_indices": [0]}
                                  for text in check["required_propositions"]]}
                for check in request["checks"]
            ]}
        raise AssertionError("Multipart fixture must not request full output")


class EditorialTests(unittest.TestCase):
    def setUp(self):
        self.ask, self.allowed = physical_ask()

    def test_physical_seven_scene_outline_prunes_only_exhausted_content(self):
        bad = outline([
            intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
            intent("ARCHITECTURE_DIAGRAM", 3), intent("TIMELINE", 4, 5),
            intent("CODE", 2), intent("GRAPH", 4, 5), intent("OUTRO", 3, 4, 5),
        ])
        bad["presentation"]["scene_titles"] = {
            f"s{index:03d}": f"Original title {index}" for index in range(1, 8)
        }
        result = planner._normalize_outline(bad, self.allowed, self.ask)
        scenes = result["scene_intents"]
        ids = [scene["id"] for scene in scenes]
        self.assertNotIn("s005", ids)
        self.assertNotIn("s006", ids)
        self.assertIn("s002", ids)
        self.assertIn("s003", ids)
        self.assertIn("s004", ids)
        self.assertEqual(scenes[-1]["id"], "s007")
        self.assertEqual(scenes[-1]["type"], "OUTRO")
        self.assertEqual(next(scene["type"] for scene in scenes if scene["id"] == "s002"),
                         "DATA_FLOW")
        self.assertEqual([scene["type"] for scene in scenes if scene["id"] in {"s003", "s004"}],
                         ["SUMMARY", "SUMMARY"])
        self.assertEqual(sum(scene["type"] not in {"SECTION_TITLE", "HERO", "OUTRO"}
                             and scene["fact_ids"] == ["F0002"] for scene in scenes), 1)
        self.assertEqual(sum(scene["type"] not in {"SECTION_TITLE", "HERO", "OUTRO"}
                             and scene["fact_ids"] == ["F0004", "F0005"] for scene in scenes), 1)
        self.assertEqual(set(result["presentation"]["scene_titles"]), set(ids))
        self.assertEqual(result["presentation"]["scene_titles"]["s007"],
                         "Original title 7")
        self.assertEqual(result["presentation"]["outro"], bad["presentation"]["outro"])
        self.assertEqual(planner._missing_story_topics(scenes, self.ask), {})
        planner.validate_novelty(scenes)
        self.assertEqual(planner._normalize_outline(result, self.allowed, self.ask), result)

        short = outline([intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
                         intent("SUMMARY", 3, 6), intent("SUMMARY", 4, 5),
                         intent("OUTRO", 3, 4, 5)])
        concise = planner._normalize_outline(short, self.allowed, self.ask)
        self.assertEqual(len(concise["scene_intents"]), 5)
        self.assertIn("F0002", self.ask["requested_topic_fact_ids"]["workflow"])

    def test_editorial_contract_is_in_both_checkpoint_inputs(self):
        contract = self.ask["editorial_contract"]
        self.assertEqual(contract, "storyboard-editorial-grounding-v6")
        self.assertEqual(planner._outline_payload(self.ask)["editorial_contract"], contract)
        selected = planner._normalize_outline({"version": 1, "title": "ETW", "slug": "etw",
            "summary": CLAIMS[1], "scene_intents": [intent("DATA_FLOW", 2)]},
            self.allowed, self.ask, check_coverage=False)
        part = planner._scene_part_payload(self.ask, selected, selected["scene_intents"],
                                           1, 1, "scope")
        self.assertEqual(part["editorial_contract"], contract)
        old = {**self.ask, "editorial_contract": "storyboard-editorial-grounding-v3"}
        digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()
        self.assertNotEqual(digest(planner._outline_payload(old)),
                            digest(planner._outline_payload(self.ask)))
        old_part = planner._scene_part_payload(old, selected, selected["scene_intents"],
                                               1, 1, "scope")
        self.assertNotEqual(digest(old_part), digest(part))
        research_request = research._payload(1, [], "BoringOS", "Explain the project.")
        self.assertEqual(research_request["research_semantics_contract"],
                         "requested-topic-semantics-v18")
        self.assertNotIn("editorial_contract", research_request)
        self.assertNotIn("editorial_contract",
                         planner._compact_payload(1, 1, [], "BoringOS", "Explain the project."))

    def test_physical_bad_architecture_recovers_through_multipart_and_checkpoint(self):
        raw = outline([intent("SECTION_TITLE", 1), intent("ARCHITECTURE_DIAGRAM", 3),
                       intent("DATA_FLOW", 2), intent("SUMMARY", 4, 5),
                       intent("OUTRO", 3, 4, 5)])
        provider = PhysicalOutlineProvider(raw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def run(ask):
                return planner._multipart_episode(provider, "storyboard", ask, path,
                    32768, 4096, 1024, 0, None, "physical-scope")

            # Even zero permitted retries work: only the presentation is unsuitable.
            episode = run(self.ask)
            checkpoint = path / "manifests/storyboard-parts/outline.json"
            canonical = json_load(checkpoint)["result"]
            self.assertEqual([item["type"] for item in canonical["scene_intents"]],
                             ["SUMMARY", "SUMMARY", "DATA_FLOW", "SUMMARY", "OUTRO"])
            self.assertEqual(canonical["scene_intents"][1]["fact_ids"], ["F0003"])
            self.assertEqual(canonical["scene_intents"][1]["evidence_refs"], ["E0002"])
            self.assertEqual([scene["type"] for scene in episode["scenes"]],
                             [item["type"] for item in canonical["scene_intents"]])
            self.assertEqual(episode["scenes"][2]["fact_ids"], ["F0002"])
            self.assertNotIn("diagram", episode["scenes"][1])
            self.assertIn("diagram", episode["scenes"][2])
            self.assertEqual(episode["scenes"][-1]["type"], "OUTRO")
            self.assertEqual(episode["presentation"]["outro"], raw["presentation"]["outro"])
            self.assertEqual(planner._missing_story_topics(canonical["scene_intents"], self.ask), {})
            self.assertNotIn("purpose", self.ask["requested_topic_fact_ids"])
            self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 1)
            scene_requests = [call for call in provider.calls if call.get("storyboard_mode") == "scenes"]
            self.assertTrue(any(item["type"] == "SUMMARY" and item["id"] == "s002"
                                and "diagram" not in item
                                for call in scene_requests for item in call["required_output"]["scenes"]))
            calls = len(provider.calls)
            self.assertEqual(run(self.ask), episode)
            self.assertEqual(len(provider.calls), calls)
            v4_digest = json_load(checkpoint)["input_sha256"]

            # A v3 input hash cannot authorize the v4 canonical checkpoint.
            older = {**self.ask, "editorial_contract": "storyboard-editorial-grounding-v3"}
            self.assertEqual(run(older), episode)
            v3_digest = json_load(checkpoint)["input_sha256"]
            self.assertNotEqual(v3_digest, v4_digest)
            self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 2)
            self.assertEqual(run(self.ask), episode)
            self.assertEqual(json_load(checkpoint)["input_sha256"], v4_digest)
            self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 3)
            calls = len(provider.calls)
            self.assertEqual(run(self.ask), episode)
            self.assertEqual(len(provider.calls), calls)

    def test_physical_boringos_architecture_downgrades_without_losing_authority(self):
        ask, allowed = boringos_ask()
        raw = boringos_outline()
        selected = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
        self.assertEqual({number: selected[f"F{number:04d}"]["claim"]
                          for number in BORINGOS_CLAIMS}, BORINGOS_CLAIMS)
        with self.assertRaisesRegex(SceneTypeUnsuitable, "coherent selected structural"):
            planner.validate_scene_type({"id": "s002", **raw["scene_intents"][1]},
                                        selected, ask["evidence_index"])
        invented = {"id": "s002", **raw["scene_intents"][1],
                    "diagram": {"nodes": ["BoringKernel → RAMFS → boring-display",
                                          "boring-display sends frames to VirtIO block"]}}
        with self.assertRaisesRegex(ValueError,
                                    r"s002: diagram\.nodes\[0\] has unsupported factual content"):
            planner._validate_structured_grounding(None, [invented], ask, None)
        canonical = planner._normalize_outline(raw, allowed, ask, check_coverage=False)
        second = canonical["scene_intents"][1]
        self.assertEqual(second["id"], "s002")
        self.assertEqual(second["type"], "SUMMARY")
        self.assertEqual(second["fact_ids"], ["F0003", "F0005", "F0008", "F0009"])
        self.assertEqual(second["evidence_refs"], ["E0036", "E0097", "E0049"])
        payload = planner._scene_part_payload(ask, canonical, [second], 2, 2, "scope")
        self.assertEqual(payload["scene_intents"], [second])
        self.assertEqual(payload["required_output"]["scenes"][0]["fact_ids"],
                         second["fact_ids"])
        self.assertEqual(payload["required_output"]["scenes"][0]["evidence_refs"],
                         second["evidence_refs"])
        self.assertNotIn("diagram", payload["required_output"]["scenes"][0])

        provider = PhysicalOutlineProvider(raw)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._multipart_episode(provider, "storyboard", ask,
                Path(tmp), 32768, 4096, 1024, 0, None, "boringos-scope")
        self.assertEqual(episode["scenes"][1]["type"], "SUMMARY")
        self.assertEqual(episode["scenes"][1]["fact_ids"], second["fact_ids"])
        self.assertNotIn("diagram", episode["scenes"][1])
        self.assertTrue(all("diagram" not in item
                            for call in provider.calls if call.get("storyboard_mode") == "scenes"
                            for item in call["required_output"]["scenes"]))

    def test_architecture_requires_explicit_connected_entities(self):
        def check(claims, suitable):
            facts = {f"F{index:04d}": {"claim": claim}
                     for index, claim in enumerate(claims, 1)}
            scene = {"id": "s001", "type": "ARCHITECTURE_DIAGRAM",
                     "fact_ids": list(facts)}
            if suitable:
                planner.validate_scene_type(scene, facts, [])
            else:
                with self.assertRaises(SceneTypeUnsuitable):
                    planner.validate_scene_type(scene, facts, [])

        for claims in (["Project includes component X."],
                       ["Subsystem Y exists."],
                       ["RAMFS uses the kernel heap."],
                       ["Component A sends events to Component B.",
                        "Service X depends on subsystem Y."],
                       ["The pipeline consists of Capture, Aggregation, and GUI stages.",
                        "A separate sensor exists."]):
            with self.subTest(claims=claims):
                check(claims, False)
        for claims in (["Component A sends events to Component B."],
                       ["Service X depends on subsystem Y."],
                       ["Events flow from capture into aggregation."],
                       ["The display service receives buffers from the compositor."],
                       ["Project X includes Component A and Component B."],
                       ["The pipeline consists of Capture, Aggregation, and GUI stages."],
                       ["ETW providers send normalized events to the core capture worker.",
                        "The core capture worker feeds bounded application state used by the GUI."]):
            with self.subTest(claims=claims):
                check(claims, True)

        valid = {"F0001": {"claim": "Component A sends events to Component B."}}
        scene = {"id": "s001", "type": "ARCHITECTURE_DIAGRAM",
                 "fact_ids": ["F0001"], "diagram": {"nodes": [
                     "Component A sends events to Component B",
                     "Component C performs undocumented dispatch"]}}
        planner.validate_scene_type(scene, valid, [])
        with self.assertRaisesRegex(ValueError,
                                    r"diagram\.nodes\[1\] has unsupported factual content"):
            planner._validate_structured_grounding(None, [scene],
                {"research": {"facts": [{"fact_id": "F0001", **valid["F0001"]}]}}, None)

    def test_downgraded_duplicate_is_removed_with_no_retry(self):
        bad = outline([intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
                       intent("CODE", 2), intent("SUMMARY", 3), intent("OUTRO", 3)])
        canonical = planner._normalize_outline(bad, self.allowed, self.ask,
                                               check_coverage=False)
        self.assertEqual([scene["id"] for scene in canonical["scene_intents"]],
                         ["s001", "s002", "s004", "s005"])
        provider = PhysicalOutlineProvider(bad)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._multipart_episode(provider, "storyboard", self.ask,
                Path(tmp), 32768, 4096, 1024, 0, None, "novelty-scope")
        self.assertEqual([scene["type"] for scene in episode["scenes"]],
                         ["SUMMARY", "DATA_FLOW", "SUMMARY", "OUTRO"])
        self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 1)

    def test_never_improving_provider_prunes_checkpoint_and_part_requests(self):
        raw = outline([intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
                       intent("SUMMARY", 3), intent("SUMMARY", 4, 5),
                       intent("CODE", 2), intent("OUTRO", 3, 4, 5)])
        raw["presentation"]["scene_titles"] = {
            f"s{index:03d}": f"Scene title {index}" for index in range(1, 7)
        }
        provider = PhysicalOutlineProvider(raw)
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)

            def run():
                return planner._multipart_episode(provider, "storyboard", self.ask,
                    project_dir, 32768, 4096, 1024, 0, None, "never-improves")

            episode = run()
            checkpoint = json_load(project_dir / "manifests/storyboard-parts/outline.json")
            canonical = checkpoint["result"]
            ids = [scene["id"] for scene in canonical["scene_intents"]]
            self.assertEqual(ids, ["s001", "s002", "s003", "s004", "s006"])
            self.assertEqual([scene["id"] for scene in episode["scenes"]], ids)
            self.assertEqual(episode["scenes"][1]["type"], "DATA_FLOW")
            self.assertEqual(episode["scenes"][1]["fact_ids"], ["F0002"])
            self.assertEqual(episode["scenes"][-1]["type"], "OUTRO")
            self.assertEqual(planner._missing_story_topics(canonical["scene_intents"], self.ask), {})
            planner.validate_novelty(canonical["scene_intents"])
            self.assertNotIn("s005", canonical["presentation"]["scene_titles"])
            self.assertEqual(canonical["presentation"]["scene_titles"]["s006"],
                             "Scene title 6")
            self.assertEqual(canonical["presentation"]["outro"], raw["presentation"]["outro"])
            scene_requests = [call for call in provider.calls
                              if call.get("storyboard_mode") == "scenes"]
            self.assertTrue(scene_requests)
            self.assertEqual([item["id"] for call in scene_requests
                              for item in call["required_output"]["scenes"]], ids)
            self.assertTrue(all(item["type"] != "CODE" and item["type"] != "GRAPH"
                                for call in scene_requests
                                for item in call["required_output"]["scenes"]))
            self.assertEqual(sum(call.get("storyboard_mode") == "outline"
                                 for call in provider.calls), 1)
            calls = len(provider.calls)
            self.assertEqual(run(), episode)
            self.assertEqual(len(provider.calls), calls)
            self.assertEqual(json_load(project_dir / "manifests/storyboard-parts/outline.json"),
                             checkpoint)

    def test_full_response_downgrade_discards_stale_diagram_but_keeps_facts(self):
        raw = {"version": 1, "title": "Passive observation", "scenes": [{
            "id": "s001", "type": "ARCHITECTURE_DIAGRAM", "title": "Observation",
            "narration": CLAIMS[2], "fact_ids": ["F0003"], "evidence_refs": ["E0002"],
            "diagram": {"nodes": ["Unsupported step A", "Unsupported step B"]},
        }]}
        fixed = planner._canonical_full_episode(raw, self.ask, self.allowed)
        self.assertEqual(fixed["scenes"][0]["type"], "SUMMARY")
        self.assertEqual(fixed["scenes"][0]["fact_ids"], ["F0003"])
        self.assertNotIn("diagram", fixed["scenes"][0])

    def test_full_response_prunes_same_duplicate_and_title(self):
        selected = [intent("DATA_FLOW", 2), intent("SUMMARY", 3),
                    intent("CODE", 2), intent("OUTRO", 3)]
        scenes = []
        for index, item in enumerate(selected, 1):
            scene = {"id": f"s{index:03d}", "type": item["type"],
                     "title": f"Scene {index}", "narration": item["purpose"],
                     "fact_ids": item["fact_ids"], "evidence_refs": item["evidence_refs"]}
            if item["type"] == "DATA_FLOW":
                scene["diagram"] = {"nodes": ["ETW callbacks normalize raw provider data",
                                              "small internal events"]}
            if item["type"] == "CODE":
                scene["diagram"] = {"code": "unsupported code"}
            scenes.append(scene)
        raw = {"version": 1, "title": "Documented observations", "scenes": scenes,
               "presentation": {"scene_titles": {
                   f"s{index:03d}": f"Title {index}" for index in range(1, 5)
               }, "outro": outline([])["presentation"]["outro"]}}
        canonical = planner._canonical_full_episode(raw, self.ask, self.allowed)
        self.assertEqual([scene["id"] for scene in canonical["scenes"]],
                         ["s001", "s002", "s004"])
        self.assertEqual(list(canonical["presentation"]["scene_titles"]),
                         ["s001", "s002", "s004"])
        self.assertEqual(canonical["presentation"]["outro"], raw["presentation"]["outro"])
        self.assertEqual(planner._missing_story_topics(canonical["scenes"], self.ask), {})
        planner.validate_novelty(canonical["scenes"])
        validate_episode(canonical, self.allowed, require_integrated_presentation=True)

        class FixedFullProvider(PhysicalOutlineProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "storyboard_mode" not in request and "checks" not in request:
                    self.calls.append(request)
                    return copy.deepcopy(raw)
                return super().complete_json(system, user)

        provider = FixedFullProvider(outline([]))
        with tempfile.TemporaryDirectory() as tmp:
            completed = planner._complete_episode(provider, "storyboard", self.ask,
                self.allowed, 0, project_dir=Path(tmp))
        self.assertEqual([scene["id"] for scene in completed["scenes"]],
                         ["s001", "s002", "s004"])
        self.assertEqual(sum("storyboard_mode" not in call and "checks" not in call
                             for call in provider.calls), 1)

    def test_partial_novelty_and_framing_are_not_pruned(self):
        visual = copy.deepcopy(self.ask)
        visual["evidence_index"].append({"ref": "E0090", "kind": "media",
                                         "relative_path": "authentic.png"})
        visual["research"]["assets"] = [{"evidence_ref": "E0090", "purpose": "Capture"}]
        raw = outline([intent("HERO", 1), intent("SUMMARY", 1),
                       intent("SUMMARY", 1, 2), intent("SUMMARY", 1),
                       intent("OUTRO", 1)])
        raw["scene_intents"][0]["asset_ref"] = "E0090"
        raw["scene_intents"][0]["evidence_refs"].append("E0090")
        raw_scenes = [{"id": f"s{index:03d}", **scene}
                      for index, scene in enumerate(raw["scene_intents"], 1)]
        with self.assertRaisesRegex(ValueError, "s004: content repeats"):
            planner.validate_novelty(raw_scenes)
        self.assertEqual([scene["id"] for scene in prune_redundant_content_scenes(raw_scenes)],
                         ["s001", "s002", "s003", "s005"])
        canonical = planner._normalize_outline(raw, self.allowed | {"E0090"}, visual,
                                               check_coverage=False)
        self.assertEqual([scene["id"] for scene in canonical["scene_intents"]],
                         ["s001", "s003", "s005"])
        self.assertIn("F0002", canonical["scene_intents"][1]["fact_ids"])
        self.assertEqual(canonical["scene_intents"][-1]["type"], "OUTRO")
        planner.validate_novelty(canonical["scene_intents"])

    def test_type_fallback_does_not_repair_bad_authority(self):
        raw = outline([intent("ARCHITECTURE_DIAGRAM", 3)])
        raw.pop("presentation")
        for mutation, message in (({"fact_ids": ["F9999"]}, "allowed_fact_ids"),
                                  ({"fact_ids": [], "evidence_refs": []}, "fact_ids")):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(raw)
                candidate["scene_intents"][0].update(mutation)
                with self.assertRaisesRegex(StructuredOutputError, message):
                    planner._normalize_outline(candidate, self.allowed, self.ask,
                                               check_coverage=False)

    def test_selected_fact_isolation_purpose_and_temporal_invention(self):
        scenes = [
            (2, "DATA_FLOW", {"nodes": ["ETW callbacks normalize raw provider data",
                                              "Enqueue small internal events for analysis"]}),
            (3, "SUMMARY", {"nodes": ["Callbacks Normalize Data into Internal Events"]},
             ),
            (4, "SUMMARY", {"nodes": ["DNS Query Timestamp → Correlated with Scheduled Task Start Time",
                                            "Scheduled Task Execution → Correlated with DNS Query Domain"]}),
        ]
        for number, kind, diagram in scenes:
            with self.subTest(fact=number):
                scene = {"id": "s001", "type": kind, "title": "Grounded",
                         "narration": CLAIMS[number - 1], "fact_ids": [f"F{number:04d}"],
                         "evidence_refs": ["E0003" if number == 2 else "E0002"],
                         "diagram": diagram}
                with self.assertRaisesRegex(ValueError, "unsupported factual content from selected fact_ids"):
                    planner._validate_structured_grounding(ApprovingVerifier(), [scene],
                                                           self.ask, None)
        clean = {"id": "s001", "type": "DATA_FLOW", "narration": CLAIMS[1],
                 "fact_ids": ["F0002"], "evidence_refs": ["E0003"],
                 "diagram": {"nodes": ["ETW callbacks normalize raw provider data",
                                       "small internal events"]}}
        planner._validate_structured_grounding(None, [clean], self.ask, None)
        clean["narration"] = CLAIMS[1][:-1] + " for analysis."
        with self.assertRaisesRegex(StructuredOutputError, "unsupported factual proposition"):
            planner._validate_narration_grounding(ApprovingVerifier(), [clean], self.ask, None)
        purpose = copy.deepcopy(clean)
        purpose["fact_ids"] = ["F0001"]
        purpose["evidence_refs"] = ["E0002"]
        purpose["narration"] = CLAIMS[0][:-1] + " for analysis."
        with self.assertRaisesRegex(StructuredOutputError, "unsupported factual proposition"):
            planner._validate_narration_grounding(ApprovingVerifier(), [purpose], self.ask, None)
        correlation = copy.deepcopy(clean)
        correlation["fact_ids"] = ["F0004"]
        correlation["evidence_refs"] = ["E0002"]
        correlation["narration"] = CLAIMS[3][:-1] + ", therefore DNS causes task starts."
        with self.assertRaisesRegex(StructuredOutputError, "unsupported factual proposition"):
            planner._validate_narration_grounding(ApprovingVerifier(), [correlation], self.ask, None)

    def test_graph_code_timeline_and_distinct_asset_rules(self):
        for kind, numbers, message in (("GRAPH", (4, 5), "numerical"),
                                       ("CODE", (2,), "code"),
                                       ("TIMELINE", (4, 5), "temporal"),
                                       ("ARCHITECTURE_DIAGRAM", (3,), "structural")):
            with self.subTest(kind=kind):
                raw = outline([intent(kind, *numbers)])
                raw.pop("presentation")
                selected = {fact["fact_id"]: fact for fact in self.ask["research"]["facts"]}
                with self.assertRaisesRegex(SceneTypeUnsuitable, message):
                    planner.validate_scene_type({"id": "s001", **raw["scene_intents"][0]},
                                                selected, self.ask["evidence_index"])
                canonical = planner._normalize_outline(raw, self.allowed, self.ask,
                                                       check_coverage=False)
                self.assertEqual(canonical["scene_intents"][0]["type"], "SUMMARY")
        scene = {"id": "s001", "type": "GRAPH", "narration": CLAIMS[3],
                 "fact_ids": ["F0004"], "evidence_refs": ["E0002"]}
        with self.assertRaisesRegex(ValueError, "GRAPH requires at least two"):
            planner._validate_structured_grounding(None, [scene], self.ask, None)
        with self.assertRaisesRegex(ValueError, "GRAPH requires at least two"):
            validate_episode({"version": 1, "title": "Graph", "scenes": [scene]}, self.allowed)

        extra = copy.deepcopy(self.ask)
        extra["research"]["facts"].append({"fact_id": "F0007",
            "claim": "The collector calls enqueue(event).", "evidence_refs": ["E0007"],
            "support": [{"evidence_ref": "E0007", "text": "enqueue(event)"}]})
        extra["evidence_index"].append({"ref": "E0007", "kind": "document",
                                        "relative_path": "src/collector.py"})
        code = {"id": "s001", "type": "CODE", "fact_ids": ["F0007"],
                "evidence_refs": ["E0007"]}
        planner.validate_scene_type(code, {f["fact_id"]: f for f in extra["research"]["facts"]},
                                    extra["evidence_index"])
        ordered = copy.deepcopy(extra)
        ordered["research"]["facts"].append({"fact_id": "F0008",
            "claim": "The initial capture happens before the final report.",
            "evidence_refs": ["E0007"],
            "support": [{"evidence_ref": "E0007", "text": "The initial capture happens before the final report."}]})
        timeline = {"id": "s002", "type": "TIMELINE", "fact_ids": ["F0008"],
                    "evidence_refs": ["E0007"]}
        planner.validate_scene_type(timeline, {f["fact_id"]: f for f in ordered["research"]["facts"]},
                                    ordered["evidence_index"])
        chart = copy.deepcopy(extra)
        chart["research"]["facts"].append({"fact_id": "F0009",
            "claim": "The measured series records x 1 with y 10 and x 2 with y 20.",
            "evidence_refs": ["E0007"],
            "support": [{"evidence_ref": "E0007", "text": "The measured series records x 1 with y 10 and x 2 with y 20."}]})
        graph = {"id": "s001", "type": "GRAPH", "fact_ids": ["F0009"],
                 "evidence_refs": ["E0007"], "diagram": {"points": [[1, 10], [2, 20]]}}
        planner.validate_scene_type(graph, {f["fact_id"]: f for f in chart["research"]["facts"]},
                                    chart["evidence_index"])
        planner._validate_structured_grounding(None, [graph], chart, None)
        graph["diagram"]["points"][1] = [2, 25]
        with self.assertRaisesRegex(ValueError, "numbers absent from selected facts"):
            planner._validate_structured_grounding(None, [graph], chart, None)

        visual = copy.deepcopy(self.ask)
        visual["evidence_index"].append({"ref": "E0090", "kind": "media",
                                         "relative_path": "authentic.png"})
        visual["research"]["assets"] = [{"evidence_ref": "E0090", "purpose": "Capture"}]
        reused = {"id": "s002", "type": "PROJECT_EVIDENCE", "fact_ids": ["F0001"],
                  "purpose": CLAIMS[0], "evidence_refs": ["E0002", "E0090"],
                  "asset_ref": "E0090"}
        retained = planner._normalize_outline({"version": 1, "title": "Proof", "slug": "proof",
            "summary": CLAIMS[0], "scene_intents": [
                intent("SUMMARY", 1), {key: value for key, value in reused.items() if key != "id"},
            ]}, {"E0002", "E0003", "E0090"}, visual, check_coverage=False)
        # The new screenshot adds no new claim to this second scene.
        self.assertEqual([scene["id"] for scene in retained["scene_intents"]], ["s001"])
        repeated_asset = planner._normalize_outline({"version": 1, "title": "Proof",
            "slug": "proof", "summary": CLAIMS[0], "scene_intents": [
                intent("SUMMARY", 1), {key: value for key, value in reused.items() if key != "id"},
                {key: value for key, value in reused.items() if key != "id"},
            ]}, {"E0002", "E0003", "E0090"}, visual, check_coverage=False)
        self.assertEqual([scene["id"] for scene in repeated_asset["scene_intents"]],
                         ["s001"])
        framed = [{"id": f"s{i:03d}", **intent(kind, 1)} for i, kind in
                  enumerate(("HERO", "SUMMARY", "OUTRO"), 1)]
        planner.validate_novelty(framed)
        with self.assertRaisesRegex(ValueError, "repeats"):
            planner.validate_novelty([{"id": "s001", **intent("SUMMARY", 1)},
                                      {"id": "s002", **intent("DATA_FLOW", 1)}])


if __name__ == "__main__":
    unittest.main()
