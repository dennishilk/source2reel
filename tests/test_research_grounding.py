"""Research claims must keep source relationships and subjects intact."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel.planner import _make_ask, _planner_records
from source2reel.research import _fact_scope, research


class FixedProvider:
    def __init__(self, facts):
        self.facts = facts
        self.requests = []

    def complete_json(self, system, user):
        request = json.loads(user)
        self.requests.append((system, request))
        if "checks" in request:
            return {"decisions": [{"id": check["id"], "supported": True}
                                  for check in request["checks"]]}
        return {"facts": self.facts, "assets": []}


class ResearchGroundingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "prompts").mkdir()
        (root / "prompts" / "research.txt").write_text("Research the evidence.")
        self.project = root / "projects" / "widget"

    def run_research(self, evidence, facts, instructions=""):
        by_ref = {entry["ref"]: entry.get("excerpt", "") for entry in evidence}
        facts = [{**fact, "support": fact.get("support", [
            {"evidence_ref": ref, "text": by_ref[ref][:320]}
            for ref in fact.get("evidence_refs", []) if len(by_ref.get(ref, "")) >= 12
        ])} for fact in facts]
        provider = FixedProvider(facts)
        result = research(provider, {"evidence": evidence}, self.project,
                          title_hint="WidgetEngine", instructions=instructions)
        return result, provider

    def test_independent_setup_interface_and_build_do_not_establish_a_normal_sequence(self):
        excerpt = ("Setup command A installs dependencies.\n"
                   "Command B opens an interactive interface.\n"
                   "Command C builds an existing episode.")
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "SETUP.md",
                     "evidence_role": "primary", "excerpt": excerpt}]
        facts = [{"claim": "A, then B, then C are the normal production pipeline.",
                  "evidence_refs": ["E0001"], "support": [
                      {"evidence_ref": "E0001", "text": excerpt}]}]
        result, provider = self.run_research(evidence, facts)
        self.assertEqual(result["facts"], [])
        self.assertIn("support", provider.requests[0][1]["required_output"]["facts"][0])

    def test_explicit_documented_order_can_be_retained(self):
        excerpt = "The normal production pipeline runs A, then B, then C."
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "WORKFLOW.md",
                     "evidence_role": "primary", "excerpt": excerpt}]
        facts = [{"claim": "The normal pipeline runs A, then B, then C.",
                  "evidence_refs": ["E0001"], "support": [
                      {"evidence_ref": "E0001", "text": excerpt}]}]
        result, _ = self.run_research(evidence, facts)
        self.assertEqual(len(result["facts"]), 1)

    def test_supporting_subject_cannot_become_engine_offline_guarantee(self):
        evidence = [
            {"ref": "E0001", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": "WidgetEngine produces diagrams."},
            {"ref": "E0002", "kind": "document",
             "relative_path": "projects/demo-phone/manifests/evidence.json",
             "evidence_role": "generated_artifact",
             "excerpt": "DemoPhone runs offline without a host service."},
        ]
        facts = [
            {"claim": "WidgetEngine produces diagrams.", "evidence_refs": ["E0001"]},
            {"claim": "WidgetEngine runs offline without external services.",
             "evidence_refs": ["E0002"], "support": [
                 {"evidence_ref": "E0002", "text": evidence[1]["excerpt"]}]},
            {"claim": "DemoPhone runs offline without a host service.",
             "evidence_refs": ["E0002"], "support": [
                 {"evidence_ref": "E0002", "text": evidence[1]["excerpt"]}]},
        ]
        result, _ = self.run_research(evidence, facts)
        self.assertEqual([f["claim"] for f in result["facts"]],
                         [facts[0]["claim"], facts[2]["claim"]])
        self.assertEqual([f["subject_scope"] for f in result["facts"]],
                         ["main_subject", "supporting_only"])

    def test_primary_profile_summary_and_derived_mixed_scope(self):
        evidence = [
            {"ref": "E0001", "kind": "document", "relative_path": "STYLE.md",
             "evidence_role": "primary", "excerpt": (
                 "WidgetProfile defines visual style.\n"
                 "Permanent scene vocabulary: HERO, SUMMARY.\n"
                 "Permanent evidence rules: keep photos static.")},
            {"ref": "E0002", "kind": "document",
             "relative_path": "examples/demo/README.md", "evidence_role": "embedded_reference",
             "excerpt": "Demo uses WidgetProfile style."},
        ]
        claim = "WidgetProfile defines visual style, scene vocabulary and evidence rules."
        facts = [{"claim": claim, "evidence_refs": ["E0001"],
                  "subject_scope": "supporting_only"},
                 {"claim": "Demo uses WidgetProfile style.", "evidence_refs": ["E0001", "E0002"],
                  "subject_scope": "main_subject"}]
        result, _ = self.run_research(evidence, facts)
        self.assertEqual(result["facts"][0]["claim"], claim)
        self.assertEqual(result["facts"][0]["subject_scope"], "main_subject")
        self.assertEqual(result["facts"][1]["subject_scope"], "mixed")
        ask = _make_ask(result, [], evidence, "WidgetEngine", "")
        self.assertEqual([f["subject_scope"] for f in ask["research"]["facts"]],
                         ["main_subject", "mixed"])
        self.assertEqual([r["subject_scope"] for r in _planner_records(result, [], evidence)],
                         ["main_subject", "mixed"])
        self.assertEqual(_fact_scope(["E0002"], {"E0002": "embedded_reference"}),
                         "supporting_only")

    def test_mixed_main_claim_requires_matching_primary_support(self):
        evidence = [
            {"ref": "E0001", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary", "excerpt": "WidgetEngine builds diagrams."},
            {"ref": "E0002", "kind": "document",
             "relative_path": "projects/demo-phone/manifests/evidence.json",
             "evidence_role": "generated_artifact", "excerpt": "DemoPhone runs offline."},
        ]
        facts = [
            {"claim": "WidgetEngine runs offline without external services.",
             "evidence_refs": ["E0001", "E0002"], "support": [
                 {"evidence_ref": "E0001", "text": evidence[0]["excerpt"]},
                 {"evidence_ref": "E0002", "text": evidence[1]["excerpt"]}]},
            {"claim": "WidgetEngine builds diagrams, as DemoPhone demonstrates.",
             "evidence_refs": ["E0001", "E0002"], "support": [
                 {"evidence_ref": "E0001", "text": evidence[0]["excerpt"]},
                 {"evidence_ref": "E0002", "text": evidence[1]["excerpt"]}]},
        ]
        result, _ = self.run_research(evidence, facts)
        self.assertEqual([f["claim"] for f in result["facts"]], [facts[1]["claim"]])
        self.assertEqual(result["facts"][0]["subject_scope"], "mixed")

    def test_invented_support_text_is_rejected(self):
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "README.md",
                     "evidence_role": "primary", "excerpt": "WidgetEngine builds diagrams."}]
        facts = [{"claim": "WidgetEngine can do anything.", "evidence_refs": ["E0001"],
                  "support": [{"evidence_ref": "E0001", "text": "WidgetEngine can do anything."}]}]
        result, _ = self.run_research(evidence, facts)
        self.assertEqual(result["facts"], [])

    def test_explicit_embedded_focus_allows_embedded_subject_without_quota(self):
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "README.md",
                     "evidence_role": "primary", "excerpt": "WidgetEngine builds diagrams."},
                    {"ref": "E0002", "kind": "document",
                     "relative_path": "examples/demo-phone/README.md",
                     "evidence_role": "embedded_reference", "excerpt": "DemoPhone runs offline."}]
        facts = [{"claim": "WidgetEngine builds diagrams.", "evidence_refs": ["E0001"]},
                 {"claim": "DemoPhone runs offline.", "evidence_refs": ["E0002"]}]
        result, _ = self.run_research(evidence, facts,
                                      "Focus primarily on DemoPhone as the reference project.")
        self.assertEqual(len(result["facts"]), 2)
        self.assertEqual(result["facts"][1]["subject_scope"], "supporting_only")


if __name__ == "__main__":
    unittest.main()
