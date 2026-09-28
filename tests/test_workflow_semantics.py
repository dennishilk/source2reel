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
    _coverage_payload, _covers_request, _exact_workflow_fact,
    _missing_requested_concepts, _payload, _requested_concepts,
    _workflow_signal, _workflow_source_match, _workflow_source_passage, research,
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
    "The privacy workflow keeps processing local.\n\n"
    "Passive observation and correlation are distinct from blocking, "
    "interception, and proven causality."
)


def fact(claim: str, ref: str = "E0002") -> dict:
    return {"claim": claim, "evidence_refs": [ref],
            "support": [{"evidence_ref": ref, "text": claim}],
            "phase": "final", "confidence": "high"}


def digest(system: str, payload: dict) -> str:
    user = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()


MALFORMED_WORKFLOW_CLAIM = (
    "ETW events are ingested, normalized, and correlated into structured events "
    "with metadata including process, service, DNS, and task context."
)
MALFORMED_SUPPORT = (
    "The platform-facing engine owns: ... PID/process metadata ... DNS ..."
)


class PhysicalCoverageProvider:
    def __init__(self, recovered: list[dict]):
        self.recovered = recovered
        self.research_calls = 0
        self.coverage_calls: list[dict] = []
        self.verifier_calls = 0

    def complete_json(self, _system: str, user: str) -> dict:
        payload = json.loads(user)
        if "checks" in payload:
            self.verifier_calls += 1
            return {"decisions": []}
        if payload.get("research_mode") == "requested_coverage":
            self.coverage_calls.append(payload)
            return {"facts": self.recovered, "assets": []}
        self.research_calls += 1
        return {"facts": [fact(claim) for claim in OBSERVED[:3]], "assets": []}


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
        self.assertIsNone(_workflow_source_passage(README, self.spec, strong_only=True,
                                                  min_length=12, max_length=320))
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
        passage = _workflow_source_passage(ARCHITECTURE, self.spec, strong_only=True,
                                           min_length=12, max_length=320)
        self.assertEqual(passage, CALLBACK)
        self.assertIn(passage, ARCHITECTURE)
        self.assertTrue(12 <= len(passage) <= 320)
        self.assertLessEqual(len(passage), 360)
        self.assertTrue(_workflow_source_match(CALLBACK, self.spec))
        self.assertFalse(_workflow_source_match(WORKER, self.spec))
        self.assertTrue(_covers_request(CALLBACK, self.spec))
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
        joined = callback + " " + continuation
        self.assertEqual(_workflow_source_passage(joined, self.spec, strong_only=True),
                         joined)
        self.assertTrue(_workflow_source_match(joined, self.spec))
        single = "ETW callbacks normalize and enqueue raw events."
        self.assertEqual(_workflow_source_passage(joined + " " + single, self.spec,
                                                  strong_only=True), single)
        self.assertFalse(_workflow_source_match(callback + "\n\n" + continuation,
                                                self.spec))
        self.assertFalse(_workflow_source_match(
            callback + " These " + "intermediate " * 100 +
            "events are enqueued for processing.", self.spec))
        self.assertIsNone(_workflow_source_passage(
            callback + "\nraw = enqueue(event)\n" + continuation, self.spec,
            strong_only=True))
        ordinary = "The ETW network workflow starts with a single capture."
        self.assertEqual(_workflow_source_passage(ordinary, self.spec), ordinary)
        self.assertIsNone(_workflow_source_passage(ordinary, self.spec, strong_only=True))

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
        self.assertFalse(_covers_request(
            "Select Start Capture. If Windows denies the ETW provider, restart elevated.",
            self.spec))
        self.assertFalse(_covers_request("Record Capture writes retained events.", self.spec))
        self.assertFalse(_covers_request(
            "A worker updates counters, resolves metadata, and writes sinks.", self.spec))
        self.assertFalse(_covers_request(
            "ETW callbacks normalize and normalized events.", self.spec))

    def test_exact_passage_bounds_and_primary_authority(self):
        primary = {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
                   "evidence_role": "primary", "excerpt": ARCHITECTURE}
        exact = _exact_workflow_fact([primary], self.spec, {"E0003": "primary"})
        self.assertEqual(exact["claim"], CALLBACK)
        self.assertEqual(exact["support"], [{"evidence_ref": "E0003", "text": CALLBACK}])
        self.assertEqual(exact["phase"], "unknown")
        self.assertEqual(exact["subject_scope"], "main_subject")
        for role in ("embedded_reference", "generated_artifact"):
            with self.subTest(role=role):
                self.assertIsNone(_exact_workflow_fact(
                    [{**primary, "evidence_role": role}], self.spec, {"E0003": role}))
        long_sentence = "ETW callbacks normalize " + "raw provider data " * 17 + "and enqueue events."
        self.assertGreater(len(long_sentence), 320)
        self.assertLessEqual(len(long_sentence), 480)
        self.assertIsNone(_workflow_source_passage(long_sentence, self.spec,
                                                  strong_only=True, min_length=12,
                                                  max_length=320))
        self.assertIsNone(_exact_workflow_fact(
            [{**primary, "excerpt": long_sentence}], self.spec, {"E0003": "primary"}))
        self.assertIsNone(_workflow_source_passage(
            "events = normalize(etw_data)\nevents = enqueue(events)", self.spec,
            strong_only=True, min_length=12, max_length=320))

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
        self.assertNotIn("F0001", groups["workflow"])
        self.assertIn("F0005", groups["workflow"])
        self.assertNotIn("F0006", groups["workflow"])
        self.assertNotIn("purpose", groups)  # A requested why is not source evidence.

    def test_semantic_contracts_invalidate_old_checkpoint_digests(self):
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "README.md",
                     "evidence_role": "primary", "excerpt": "WidgetEngine is a tool."}]
        current = _payload(1, evidence, "WidgetEngine", "Explain what it is.")
        previous = {**current, "research_semantics_contract": "requested-topic-semantics-v5"}
        self.assertEqual(current["research_semantics_contract"],
                         RESEARCH_SEMANTICS_CONTRACT)
        self.assertEqual(RESEARCH_SEMANTICS_CONTRACT, "requested-topic-semantics-v6")
        self.assertNotEqual(digest("Research", previous), digest("Research", current))

        missing = {"workflow": {"kind": "workflow", "terms": ["event"]}}
        coverage = _coverage_payload(evidence, missing, "WidgetEngine", "How do events flow?")
        old_coverage = {**coverage, "coverage_contract": "requested-primary-coverage-v4",
                        "research_semantics_contract": "requested-topic-semantics-v4"}
        self.assertEqual(coverage["coverage_contract"], COVERAGE_CONTRACT)
        self.assertEqual(COVERAGE_CONTRACT, "requested-primary-coverage-v5")
        self.assertNotEqual(digest("Research", {**coverage,
                                                "coverage_contract": "requested-primary-coverage-v4"}),
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


class PhysicalCoverageChainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "prompts").mkdir()
        (root / "prompts" / "research.txt").write_text("Use exact supplied evidence.")
        self.project = root / "projects" / "windows-telemetry-inspector"
        self.entries = [
            {"ref": "E0002", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": README + "\n\n" + "\n\n".join(OBSERVED[:3])},
            {"ref": "E0003", "kind": "document", "relative_path": "docs/ARCHITECTURE.md",
             "evidence_role": "primary", "excerpt": ARCHITECTURE},
        ]
        self.roles = {entry["ref"]: "primary" for entry in self.entries}

    def run_research(self, provider):
        return research(provider, {"evidence": self.entries}, self.project,
                        title_hint="windows-telemetry-inspector",
                        instructions=INSTRUCTIONS, max_retries=0)

    def test_malformed_model_support_gets_exact_fallback_and_checkpoint_reuse(self):
        invalid = {"claim": MALFORMED_WORKFLOW_CLAIM, "evidence_refs": ["E0003"],
                   "support": [{"evidence_ref": "E0003", "text": MALFORMED_SUPPORT}]}
        self.assertNotIn(MALFORMED_SUPPORT, ARCHITECTURE)
        provider = PhysicalCoverageProvider([fact(OBSERVED[1]), invalid])
        result = self.run_research(provider)
        claims = [item["claim"] for item in result["facts"]]
        for expected in (OBSERVED[0], OBSERVED[1], OBSERVED[2], CALLBACK):
            self.assertIn(expected, claims)
        self.assertNotIn(MALFORMED_WORKFLOW_CLAIM, claims)
        callback_facts = [item for item in result["facts"] if item["claim"] == CALLBACK]
        self.assertEqual(len(callback_facts), 1)
        self.assertEqual(callback_facts[0]["evidence_refs"], ["E0003"])
        self.assertEqual(callback_facts[0]["support"],
                         [{"evidence_ref": "E0003", "text": CALLBACK}])
        self.assertEqual(callback_facts[0]["subject_scope"], "main_subject")
        self.assertEqual(callback_facts[0]["phase"], "unknown")
        missing = _missing_requested_concepts(result["facts"], INSTRUCTIONS,
                                              "windows-telemetry-inspector", self.roles)
        self.assertNotIn("workflow", missing)
        self.assertIn("purpose", missing)
        ask = _make_ask(result, [], self.entries, "windows-telemetry-inspector",
                        INSTRUCTIONS)
        by_id = {item["fact_id"]: item["claim"] for item in ask["research"]["facts"]}
        self.assertIn(CALLBACK, [by_id[fid] for fid in
                                 ask["requested_topic_fact_ids"]["workflow"]])
        self.assertNotIn("purpose", ask["requested_topic_fact_ids"])

        self.assertEqual(len(provider.coverage_calls), 1)
        self.assertEqual({entry["ref"] for entry in
                          provider.coverage_calls[0]["evidence"]}, {"E0002", "E0003"})
        coverage_checkpoint = self.project / "manifests" / "research-coverage" / "part-001.json"
        canonical = json_load(coverage_checkpoint)["result"]
        self.assertIn(OBSERVED[1], [item["claim"] for item in canonical["facts"]])
        self.assertEqual([item["claim"] for item in canonical["facts"]].count(CALLBACK), 1)
        self.assertNotIn(MALFORMED_WORKFLOW_CLAIM,
                         [item["claim"] for item in canonical["facts"]])

        self.assertEqual(self.run_research(provider), result)
        self.assertEqual(provider.research_calls, 1)
        self.assertEqual(len(provider.coverage_calls), 1)
        self.assertEqual(provider.verifier_calls, 0)
        self.assertEqual(json_load(coverage_checkpoint)["result"], canonical)
        self.assertEqual(len(callback_facts[0]["support"]), 1)

    def test_valid_natural_workflow_fact_prevents_redundant_fallback(self):
        provider = PhysicalCoverageProvider([fact(OBSERVED[1]), fact(CALLBACK, "E0003")])
        result = self.run_research(provider)
        callback_facts = [item for item in result["facts"] if item["claim"] == CALLBACK]
        self.assertEqual(len(callback_facts), 1)
        self.assertEqual(callback_facts[0]["phase"], "final")
        checkpoint = json_load(self.project / "manifests" / "research-coverage" / "part-001.json")
        coverage_facts = [item for item in checkpoint["result"]["facts"]
                          if item["claim"] == CALLBACK]
        self.assertEqual(len(coverage_facts), 1)
        self.assertEqual(coverage_facts[0]["phase"], "final")
        self.assertEqual(len(provider.coverage_calls), 1)


if __name__ == "__main__":
    unittest.main()
