"""Physical editorial failure and compact, grounded alternatives."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

from source2reel import planner
from source2reel.providers import StructuredOutputError
from source2reel.schema import validate_episode


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


class EditorialTests(unittest.TestCase):
    def setUp(self):
        self.ask, self.allowed = physical_ask()

    def test_physical_seven_scene_outline_fails_and_short_story_is_valid(self):
        bad = outline([
            intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
            intent("ARCHITECTURE_DIAGRAM", 3), intent("TIMELINE", 4, 5),
            intent("CODE", 2), intent("GRAPH", 4, 5), intent("OUTRO", 3, 4, 5),
        ])
        with self.assertRaisesRegex(StructuredOutputError, "TIMELINE|ARCHITECTURE_DIAGRAM|CODE|repeats"):
            planner._normalize_outline(bad, self.allowed, self.ask)
        # Even when the unsuited templates are changed, the duplicated content remains invalid.
        for row in bad["scene_intents"]:
            if row["type"] in {"ARCHITECTURE_DIAGRAM", "TIMELINE", "CODE", "GRAPH"}:
                row["type"] = "SUMMARY"
        with self.assertRaisesRegex(StructuredOutputError, "s005: content repeats"):
            planner._normalize_outline(bad, self.allowed, self.ask, check_coverage=False)
        short = outline([intent("SECTION_TITLE", 1), intent("DATA_FLOW", 2),
                         intent("SUMMARY", 3, 6), intent("SUMMARY", 4, 5),
                         intent("OUTRO", 3, 4, 5)])
        result = planner._normalize_outline(short, self.allowed, self.ask)
        self.assertEqual(len(result["scene_intents"]), 5)
        self.assertIn("F0002", self.ask["requested_topic_fact_ids"]["workflow"])

    def test_editorial_contract_is_in_both_checkpoint_inputs(self):
        contract = self.ask["editorial_contract"]
        self.assertEqual(planner._outline_payload(self.ask)["editorial_contract"], contract)
        selected = planner._normalize_outline({"version": 1, "title": "ETW", "slug": "etw",
            "summary": CLAIMS[1], "scene_intents": [intent("DATA_FLOW", 2)]},
            self.allowed, self.ask, check_coverage=False)
        part = planner._scene_part_payload(self.ask, selected, selected["scene_intents"],
                                           1, 1, "scope")
        self.assertEqual(part["editorial_contract"], contract)
        old = {key: value for key, value in self.ask.items() if key != "editorial_contract"}
        digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()
        self.assertNotEqual(digest(old), digest(self.ask))

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
            with self.subTest(kind=kind), self.assertRaisesRegex(StructuredOutputError, message):
                raw = outline([intent(kind, *numbers)])
                raw.pop("presentation")
                planner._normalize_outline(raw, self.allowed,
                                           self.ask, check_coverage=False)
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
        planner._normalize_outline({"version": 1, "title": "Proof", "slug": "proof",
            "summary": CLAIMS[0], "scene_intents": [
                intent("SUMMARY", 1), {key: value for key, value in reused.items() if key != "id"},
            ]}, {"E0002", "E0003", "E0090"}, visual, check_coverage=False)
        framed = [{"id": f"s{i:03d}", **intent(kind, 1)} for i, kind in
                  enumerate(("HERO", "SUMMARY", "OUTRO"), 1)]
        planner.validate_novelty(framed)
        with self.assertRaisesRegex(ValueError, "repeats"):
            planner.validate_novelty([{"id": "s001", **intent("SUMMARY", 1)},
                                      {"id": "s002", **intent("DATA_FLOW", 1)}])


if __name__ == "__main__":
    unittest.main()
