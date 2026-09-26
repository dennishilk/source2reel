"""Process entities do not fulfill a requested account of how data flows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from source2reel.chunking import checkpointed_complete_json, checkpointed_split_json
from source2reel.planner import _make_ask
from source2reel.research import (
    COVERAGE_CONTRACT, RESEARCH_SEMANTICS_CONTRACT, _coverage_candidates,
    _coverage_payload, _covers_request, _missing_requested_concepts, _payload,
    _requested_concepts,
)
from source2reel.util import json_dump, json_load


TITLE = "Windows Telemetry Inspector"
INSTRUCTIONS = (
    "Explain what Windows Telemetry Inspector is, why it exists, and how its "
    "architecture turns Windows ETW network observations into useful local diagnostics. "
    "Use only repository evidence and do not invent capabilities. Clearly distinguish "
    "passive observation and correlation from blocking, interception, or proven causality."
)
OBSERVED = [
    "Windows Telemetry Inspector is a passive Windows 11 network diagnostics "
    "tool that uses ETW to show network activity by process and service.",
    "The tool correlates DNS and scheduled-task relationships as best-effort, "
    "not proven causality.",
    "It does not modify traffic, install drivers, disable telemetry, or intercept TLS.",
    "It does not block, intercept, or decrypt traffic, nor install drivers or "
    "disable telemetry.",
]
ARCHITECTURE = (
    "ETW callbacks normalize raw provider data into small internal events and "
    "enqueue them. A single aggregation worker updates flow counters, resolves "
    "cached metadata, correlates DNS/tasks, classifies the event, and writes to "
    "sinks. ETW network observations turn into bounded application state and "
    "local diagnostics through the capture workflow."
)


def fact(claim: str, ref: str = "E0002") -> dict:
    return {"claim": claim, "evidence_refs": [ref],
            "support": [{"evidence_ref": ref, "text": claim}],
            "phase": "final", "confidence": "high"}


def digest(system: str, payload: dict) -> str:
    user = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()


class WorkflowSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.spec = _requested_concepts(INSTRUCTIONS, TITLE)["workflow"]

    def test_exact_physical_overview_is_not_workflow(self):
        concepts = _requested_concepts(INSTRUCTIONS, TITLE)
        self.assertTrue(_covers_request(OBSERVED[0], concepts["overview"]))
        self.assertFalse(_covers_request(OBSERVED[0], self.spec))
        self.assertIn("process and service", OBSERVED[0])

    def test_process_entity_usages_do_not_supply_workflow_signal(self):
        cases = [
            ("The tool groups traffic by process and service.", ["traffic", "process"]),
            ("Process metadata includes PID and executable name.", ["process", "metadata"]),
            ("The UI displays process activity.", ["process", "activity"]),
            ("A process can own multiple network connections.", ["process", "network"]),
            ("The selected row shows process and service details.", ["process", "service"]),
        ]
        for claim, terms in cases:
            with self.subTest(claim=claim):
                self.assertFalse(_covers_request(claim, {"kind": "workflow", "terms": terms}))

    def test_real_data_flows_and_ordered_process_language_still_match(self):
        cases = [
            ("ETW callbacks normalize raw provider data into small internal events "
             "and enqueue them.", ["etw", "event"]),
            ("A worker updates counters, resolves metadata, correlates DNS, "
             "classifies the event, and writes to sinks.", ["counter", "metadata"]),
            ("The pipeline transforms captured events into bounded application state.",
             ["bounded", "event"]),
            ("The workflow begins with capture and then writes JSONL output.",
             ["capture", "jsonl"]),
            ("The process begins with event capture, then normalizes and stores "
             "the records.", ["event", "record"]),
            ("The stage turns inputs into records.", ["input", "record"]),
        ]
        for claim, terms in cases:
            with self.subTest(claim=claim):
                self.assertTrue(_covers_request(claim, {"kind": "workflow", "terms": terms}))

    def test_physical_four_facts_leave_workflow_missing(self):
        missing = _missing_requested_concepts(
            [fact(claim) for claim in OBSERVED], INSTRUCTIONS, TITLE,
            {"E0002": "primary"},
        )
        self.assertIn("workflow", missing)
        self.assertIn("purpose", missing)
        self.assertNotIn("overview", missing)

    def test_architecture_record_becomes_scoped_coverage_candidate(self):
        missing = _missing_requested_concepts(
            [fact(claim) for claim in OBSERVED], INSTRUCTIONS, TITLE,
            {"E0002": "primary"},
        )
        inventory = {"evidence": [
            {"ref": "E0002", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": "\n".join(OBSERVED)},
            {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
             "evidence_role": "primary", "excerpt": ARCHITECTURE},
        ]}
        selected = _coverage_candidates(inventory, missing, TITLE, "Research", 32768,
                                        4096, 1024, INSTRUCTIONS)
        self.assertIn("E0003", [entry["ref"] for entry in selected])

    def test_planner_mapping_uses_same_corrected_semantics(self):
        entries = [
            {"ref": "E0002", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": "\n".join(OBSERVED)},
            {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
             "evidence_role": "primary", "excerpt": ARCHITECTURE},
        ]
        research = {"facts": [fact(claim) for claim in OBSERVED] +
                    [fact(ARCHITECTURE, "E0003")], "assets": []}
        ask = _make_ask(research, [], entries, TITLE, INSTRUCTIONS)
        groups = ask["requested_topic_fact_ids"]
        self.assertIn("F0001", groups["overview"])
        self.assertNotIn("F0001", groups["workflow"])
        self.assertIn("F0005", groups["workflow"])
        self.assertNotIn("purpose", groups)  # A requested why is not source evidence.

    def test_semantic_contracts_invalidate_old_checkpoint_digests(self):
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "README.md",
                     "evidence_role": "primary", "excerpt": "WidgetEngine is a tool."}]
        current = _payload(1, evidence, "WidgetEngine", "Explain what it is.")
        previous = {key: value for key, value in current.items()
                    if key != "research_semantics_contract"}
        self.assertEqual(current["research_semantics_contract"],
                         RESEARCH_SEMANTICS_CONTRACT)
        self.assertNotEqual(digest("Research", previous), digest("Research", current))

        missing = {"workflow": {"kind": "workflow", "terms": ["event"]}}
        coverage = _coverage_payload(evidence, missing, "WidgetEngine", "How do events flow?")
        old_coverage = {**coverage, "coverage_contract": "requested-primary-coverage-v1"}
        self.assertEqual(coverage["coverage_contract"], COVERAGE_CONTRACT)
        self.assertEqual(COVERAGE_CONTRACT, "requested-primary-coverage-v2")
        self.assertNotEqual(digest("Research", old_coverage), digest("Research", coverage))

        class Provider:
            calls = 0

            def complete_json(self, _system, _user):
                self.calls += 1
                return {"facts": [{"claim": evidence[0]["excerpt"]}], "assets": []}

        provider = Provider()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "part-001.json"
            json_dump(checkpoint, {"version": 1, "input_sha256": digest("Research", previous),
                                   "result": {"facts": [{"claim": "old semantics"}], "assets": []}})
            def read_part():
                return checkpointed_split_json(
                    provider, "Research", evidence,
                    lambda batch: _payload(1, batch, "WidgetEngine", "Explain what it is."),
                    checkpoint, lambda result, _items: result, max_retries=0,
                )
            fresh = read_part()
            self.assertEqual(fresh[0]["facts"][0]["claim"], evidence[0]["excerpt"])
            self.assertEqual(provider.calls, 1)
            self.assertEqual(json_load(checkpoint)["input_sha256"], digest("Research", current))
            self.assertEqual(read_part(), fresh)
            self.assertEqual(provider.calls, 1)

            coverage_checkpoint = Path(tmp) / "coverage.json"
            json_dump(coverage_checkpoint, {
                "version": 1, "input_sha256": digest("Research", old_coverage),
                "result": {"facts": [{"claim": "old coverage semantics"}], "assets": []},
            })
            focused = checkpointed_complete_json(provider, "Research", coverage,
                                                 coverage_checkpoint, lambda result: result,
                                                 max_retries=0)
            self.assertEqual(focused["facts"][0]["claim"], evidence[0]["excerpt"])
            self.assertEqual(provider.calls, 2)
            self.assertEqual(json_load(coverage_checkpoint)["input_sha256"],
                             digest("Research", coverage))


if __name__ == "__main__":
    unittest.main()
