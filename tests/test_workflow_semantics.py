"""Requested workflow recovery stays local and recognizes explicit source flow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from source2reel.chunking import checkpointed_complete_json, checkpointed_split_json
from source2reel.flow_language import distinct_flow_actions
from source2reel.grounding import _declarative_text
from source2reel.planner import _make_ask
from source2reel.research import (
    COVERAGE_CONTRACT, RESEARCH_SEMANTICS_CONTRACT, _coverage_candidates,
    _coverage_payload, _covers_request, _missing_requested_concepts, _payload,
    _requested_concepts, _workflow_signal, _workflow_source_match,
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
CALLBACK = "ETW callbacks normalize raw provider data into small internal events and enqueue them."
WORKER = ("A single aggregation worker updates flow counters, resolves cached metadata, "
          "correlates DNS/tasks, classifies the event, and writes to sinks.")
GUI_SINK = ("The GUI sink enqueues enriched events for batched dispatcher updates; "
            "an optional recording sink appends JSONL independently of UI retention.")
ARCHITECTURE = f"{CALLBACK} {WORKER} {GUI_SINK}"
README = (
    "Windows Telemetry Inspector is a passive Windows 11 network diagnostics "
    "tool. It uses ETW to show process/service traffic.\n\n"
    "Select Start Capture. If Windows denies the ETW "
    "provider, restart elevated.\n\n"
    "Generate normal network traffic.\n\n"
    "Record Capture writes retained events.\n\n"
    "The privacy workflow keeps processing local."
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

    def test_exact_event_pipeline_becomes_scoped_coverage_candidate(self):
        missing = _missing_requested_concepts(
            [fact(claim) for claim in OBSERVED], INSTRUCTIONS, TITLE,
            {"E0002": "primary"},
        )
        inventory = {"evidence": [
            {"ref": "E0002", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": README},
            {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
             "evidence_role": "primary", "excerpt": ARCHITECTURE},
        ]}
        selected = _coverage_candidates(inventory, {"workflow": missing["workflow"]},
                                        TITLE, "Research", 32768,
                                        4096, 1024, INSTRUCTIONS)
        self.assertEqual([entry["ref"] for entry in selected], ["E0003"])
        # Other missing topics can still independently choose README; its title
        # and path bonus cannot make it eligible for the missing workflow.
        selected_all = _coverage_candidates(inventory, missing, TITLE, "Research", 32768,
                                            4096, 1024, INSTRUCTIONS)
        self.assertIn("E0003", [entry["ref"] for entry in selected_all])

    def test_readme_document_wide_terms_do_not_compose_workflow(self):
        prose = _declarative_text(README)
        self.assertTrue(_workflow_signal(prose))
        self.assertFalse(_workflow_source_match(README, self.spec))
        self.assertFalse(_workflow_source_match(
            "Select Start Capture. If Windows denies the ETW provider, restart elevated.",
            self.spec))
        self.assertFalse(_workflow_source_match("Record Capture writes retained events.",
                                               self.spec))
        self.assertFalse(_workflow_source_match(
            "The privacy workflow uses local processing by default.", self.spec))
        self.assertFalse(_workflow_source_match(
            "ETW network observations are local diagnostics.\n\n"
            "A worker updates counters and resolves metadata.", self.spec))
        self.assertFalse(_workflow_source_match(
            "ETW network observations are local diagnostics. "
            "Start Capture writes retained records.", self.spec))
        self.assertFalse(_workflow_source_match(
            "Windows Telemetry Inspector is a passive network diagnostics tool\n"
            "Select Start Capture to start ETW collection.", self.spec))

    def test_real_event_pipeline_matches_physical_workflow_request(self):
        self.assertEqual(self.spec["terms"],
                         ["architecture", "diagnostic", "etw", "network",
                          "observation", "useful"])
        prose = _declarative_text(ARCHITECTURE)
        self.assertIn(CALLBACK, prose)
        self.assertIn(WORKER, prose)
        self.assertIn(GUI_SINK, prose)
        self.assertTrue(_workflow_source_match(ARCHITECTURE, self.spec))
        self.assertTrue(_workflow_source_match(CALLBACK, self.spec))
        self.assertFalse(_workflow_source_match(WORKER, self.spec))
        self.assertFalse(_covers_request(CALLBACK, self.spec))
        self.assertTrue(_workflow_source_match(
            CALLBACK, {"kind": "workflow", "terms": ["etw", "event"]}))
        self.assertTrue(_workflow_source_match(
            "ETW callbacks normalize raw provider data into\n"
            "small internal events and enqueue them.",
            {"kind": "workflow", "terms": ["etw", "event"]}))
        self.assertFalse(_workflow_source_match(
            "ETW network observations are local diagnostics.\n\n" + WORKER,
            self.spec))
        self.assertFalse(_workflow_source_match(
            "ETW callbacks " + "padding " * 200 +
            "normalize raw data and enqueue events.", self.spec))

    def test_strong_flow_window_requires_adjacent_bounded_continuation(self):
        callback = "ETW callbacks normalize raw provider data."
        continuation = "These events are enqueued for processing."
        self.assertTrue(_workflow_source_match(callback + " " + continuation, self.spec))
        self.assertFalse(_workflow_source_match(callback + "\n\n" + continuation,
                                                self.spec))
        self.assertFalse(_workflow_source_match(
            callback + " These " + "intermediate " * 100 +
            "events are enqueued for processing.", self.spec))

    def test_distinct_actions_and_topical_overlap_are_both_required(self):
        self.assertEqual(distinct_flow_actions(
            "ETW callbacks normalize and normalizes data, then normalized events."),
            {"normalize"})
        self.assertEqual(distinct_flow_actions(CALLBACK), {"normalize", "enqueue"})
        self.assertFalse(_workflow_source_match("ETW callbacks normalize raw events.",
                                                self.spec))
        self.assertFalse(_workflow_source_match(
            "ETW callbacks normalize, then normalizes events.", self.spec))
        self.assertTrue(_workflow_source_match(
            "ETW callbacks normalize raw data and enqueue events.", self.spec))
        self.assertFalse(_workflow_source_match(
            "A worker updates counters, resolves metadata, and writes sinks.", self.spec))
        self.assertTrue(_workflow_source_match(
            "ETW events are read, transformed and written to sinks.", self.spec))

    def test_flow_verbs_are_prose_but_code_stays_code(self):
        prose_lines = [
            "ETW callbacks normalize raw data and enqueue events.",
            "The worker updates counters, resolves metadata, correlates DNS and classifies events.",
            "The sink writes, stores and captures records.",
            "The stage aggregates data, converts records, produces summaries and emits results.",
            "The reader reads local records.",
        ]
        self.assertEqual(_declarative_text("\n".join(prose_lines)), "\n".join(prose_lines))
        code_lines = [
            "def normalize_event(event):",
            "from app.flow import capture_event",
            "events = enqueue(raw_data)",
            "worker.update(counters)",
            "if capture_event:",
        ]
        self.assertEqual(_declarative_text("\n".join(code_lines)), "")

    def test_planner_mapping_uses_same_corrected_semantics(self):
        entries = [
            {"ref": "E0002", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": "\n".join(OBSERVED)},
            {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
             "evidence_role": "primary", "excerpt": ARCHITECTURE},
        ]
        research = {"facts": [fact(claim) for claim in OBSERVED] +
                    [fact(CALLBACK, "E0003"), fact(WORKER, "E0003")], "assets": []}
        ask = _make_ask(research, [], entries, TITLE, INSTRUCTIONS)
        groups = ask["requested_topic_fact_ids"]
        self.assertIn("F0001", groups["overview"])
        # A selected source does not change conservative fact-level matching.
        self.assertNotIn("workflow", groups)
        self.assertNotIn("purpose", groups)  # A requested why is not source evidence.

    def test_semantic_contracts_invalidate_old_checkpoint_digests(self):
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "README.md",
                     "evidence_role": "primary", "excerpt": "WidgetEngine is a tool."}]
        current = _payload(1, evidence, "WidgetEngine", "Explain what it is.")
        previous = {**current, "research_semantics_contract": "requested-topic-semantics-v3"}
        self.assertEqual(current["research_semantics_contract"],
                         RESEARCH_SEMANTICS_CONTRACT)
        self.assertEqual(RESEARCH_SEMANTICS_CONTRACT, "requested-topic-semantics-v4")
        self.assertNotEqual(digest("Research", previous), digest("Research", current))

        missing = {"workflow": {"kind": "workflow", "terms": ["event"]}}
        coverage = _coverage_payload(evidence, missing, "WidgetEngine", "How do events flow?")
        old_coverage = {**coverage, "coverage_contract": "requested-primary-coverage-v3",
                        "research_semantics_contract": "requested-topic-semantics-v3"}
        self.assertEqual(coverage["coverage_contract"], COVERAGE_CONTRACT)
        self.assertEqual(COVERAGE_CONTRACT, "requested-primary-coverage-v4")
        self.assertNotEqual(digest("Research", {**coverage,
                                                "coverage_contract": "requested-primary-coverage-v3"}),
                            digest("Research", coverage))
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
