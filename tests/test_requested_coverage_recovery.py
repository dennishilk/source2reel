"""Requested topics omitted by broad research receive one scoped primary pass."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel.planner import _make_ask
from source2reel.research import MAX_COVERAGE_CHARS, MAX_COVERAGE_RECORDS, research
from source2reel.util import json_load


OVERVIEW = "WidgetEngine is a reusable evidence-first explainer engine."
PURPOSE = ("WidgetEngine exists to turn authoritative project sources into "
           "evidence-grounded explainers through a local production pipeline.")
DISTINCTION = ("WidgetEngine is the reusable engine. The repository ships with "
               "the Production Profile as the default production configuration.")


def record(number: int, prose: str, path: str = "README.md", role: str = "primary") -> dict:
    return {"ref": f"E{number:04d}", "kind": "document", "relative_path": path,
            "evidence_role": role, "excerpt": prose}


def fact(claim: str, source: dict) -> dict:
    return {"claim": claim, "evidence_refs": [source["ref"]],
            "support": [{"evidence_ref": source["ref"], "text": claim}],
            "phase": "final", "confidence": "high"}


class OmittingProvider:
    def __init__(self, broad: list[dict], recovered: list[dict] | None = None):
        self.broad = broad
        self.recovered = recovered or []
        self.research_calls: list[dict] = []
        self.coverage_calls: list[dict] = []

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        if "checks" in payload:
            return {"decisions": [{
                "id": check["id"], "supported": True,
                "propositions": [{"text": proposition,
                                  "support_indices": list(range(len(check["support"])))}
                                 for proposition in check["required_propositions"]],
            } for check in payload["checks"]]}
        if payload.get("research_mode") == "requested_coverage":
            self.coverage_calls.append(payload)
            return {"facts": self.recovered, "assets": []}
        self.research_calls.append(payload)
        return {"facts": self.broad, "assets": []}


class RequestedCoverageRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "prompts").mkdir()
        (root / "prompts" / "research.txt").write_text("Research only supplied evidence.")
        self.project = root / "projects" / "widget"

    def run_research(self, provider, entries, instruction):
        return research(provider, {"evidence": entries}, self.project,
                        title_hint="WidgetEngine", instructions=instruction,
                        max_retries=0)

    def test_omitted_primary_purpose_recovers_and_reaches_planner_with_lineage(self):
        entries = [record(1, OVERVIEW), record(2, PURPOSE, "docs/purpose.md")]
        entries += [record(i, f"WidgetEngine module {i} configures local buffers.",
                           f"implementation/module-{i}.py") for i in range(3, 53)]
        provider = OmittingProvider([fact(OVERVIEW, entries[0])], [fact(PURPOSE, entries[1])])
        result = self.run_research(provider, entries,
                                   "Explain what WidgetEngine is and why it exists.")
        self.assertIn(PURPOSE, [item["claim"] for item in result["facts"]])
        recovered = next(item for item in result["facts"] if item["claim"] == PURPOSE)
        self.assertEqual(recovered["subject_scope"], "main_subject")
        self.assertEqual(recovered["support"], fact(PURPOSE, entries[1])["support"])
        ask = _make_ask(result, [], entries, "WidgetEngine", "Explain why it exists.")
        visible = next(item for item in ask["research"]["facts"] if item["claim"] == PURPOSE)
        self.assertEqual(visible["evidence_refs"], ["E0002"])
        self.assertEqual(visible["source_fact_id"], f"R{result['facts'].index(recovered)+1:04d}")
        self.assertEqual(len(provider.coverage_calls), 1)
        self.assertEqual([item["ref"] for item in provider.coverage_calls[0]["evidence"]],
                         ["E0002"])
        self.assertLess(len(provider.coverage_calls[0]["evidence"]), len(entries))

        ordinary = json_load(self.project / "manifests" / "research-parts" / "part-001.json")
        output = self.run_research(provider, entries,
                                   "Explain what WidgetEngine is and why it exists.")
        self.assertEqual(output, result)
        self.assertEqual(len(provider.coverage_calls), 1)
        self.assertEqual(json_load(self.project / "manifests" / "research-parts" /
                                   "part-001.json")["input_sha256"], ordinary["input_sha256"])
        self.assertTrue((self.project / "manifests" / "research-coverage" /
                         "part-001.json").exists())

    def test_no_primary_purpose_evidence_makes_no_recovery_request(self):
        overview = record(1, OVERVIEW)
        entries = [overview, record(2, "WidgetEngine writes scene data.", "docs/output.md")]
        provider = OmittingProvider([fact(OVERVIEW, overview)])
        result = self.run_research(provider, entries, "Explain what WidgetEngine is and why it exists.")
        self.assertEqual([item["claim"] for item in result["facts"]], [OVERVIEW])
        self.assertEqual(provider.coverage_calls, [])

    def test_generic_named_distinction_recovers_from_primary_prose(self):
        overview, comparison = record(1, OVERVIEW), record(2, DISTINCTION, "docs/roles.md")
        provider = OmittingProvider([fact(OVERVIEW, overview)], [fact(DISTINCTION, comparison)])
        result = self.run_research(provider, [overview, comparison],
                                   "Clearly distinguish WidgetEngine from the Production Profile.")
        self.assertIn(DISTINCTION, [item["claim"] for item in result["facts"]])
        self.assertEqual(provider.coverage_calls[0]["evidence"][0]["ref"], "E0002")
        self.assertEqual(provider.coverage_calls[0]["missing_requested_topics"][
            "distinction-1"]["kind"], "distinction")

    def test_generic_engine_words_do_not_count_as_named_profile_coverage(self):
        overview_text = ("WidgetEngine is a reusable local evidence-first technical "
                         "explainer production engine.")
        overview, comparison = record(1, overview_text), record(2,
            "WidgetEngine is the reusable engine. The repository ships with the "
            "Ada Explainer production profile as its default configuration.", "docs/roles.md")
        provider = OmittingProvider([fact(overview_text, overview)],
                                   [fact(comparison["excerpt"], comparison)])
        result = self.run_research(provider, [overview, comparison],
            "Clearly distinguish the reusable WidgetEngine engine from the "
            "Ada Explainer production profile.")
        self.assertEqual(len(provider.coverage_calls), 1)
        self.assertIn("distinction-1", provider.coverage_calls[0]["missing_requested_topics"])
        self.assertIn(comparison["excerpt"], [item["claim"] for item in result["facts"]])

    def test_supporting_test_fixture_never_supplies_primary_purpose(self):
        overview = record(1, OVERVIEW)
        test = record(2, "WidgetEngine exists to automate everything.",
                      "tests/test_widget.py", "embedded_reference")
        provider = OmittingProvider([fact(OVERVIEW, overview)], [fact(test["excerpt"], test)])
        result = self.run_research(provider, [overview, test], "Why does WidgetEngine exist?")
        self.assertNotIn(test["excerpt"], [item["claim"] for item in result["facts"]])
        self.assertEqual(provider.coverage_calls, [])

    def test_existing_requested_coverage_skips_targeted_pass(self):
        entries = [record(1, OVERVIEW), record(2, PURPOSE), record(3, DISTINCTION)]
        provider = OmittingProvider([fact(entry["excerpt"], entry) for entry in entries])
        result = self.run_research(provider, entries,
                                   "Explain what WidgetEngine is and why it exists. "
                                   "Distinguish WidgetEngine from the Production Profile.")
        self.assertEqual(len(result["facts"]), 3)
        self.assertEqual(provider.coverage_calls, [])

    def test_large_inventory_has_one_small_primary_recovery_request(self):
        overview = record(1, OVERVIEW)
        candidates = [record(i, f"WidgetEngine exists to turn source {i} into cited output.",
                             f"docs/purpose-{i}.md") for i in range(2, 13)]
        entries = [overview] + candidates + [
            record(i, f"WidgetEngine component {i} configures local storage.",
                   f"implementation/component-{i}.py") for i in range(13, 233)]
        provider = OmittingProvider([fact(OVERVIEW, overview)])
        self.run_research(provider, entries, "Explain what WidgetEngine is and why it exists.")
        self.assertEqual(len(provider.coverage_calls), 1)
        selected = provider.coverage_calls[0]["evidence"]
        self.assertLessEqual(len(selected), MAX_COVERAGE_RECORDS)
        self.assertLessEqual(sum(len(item["excerpt"]) for item in selected), MAX_COVERAGE_CHARS)
        self.assertEqual(len(selected), 1)  # One best record for the one missing concept.
        self.assertTrue(all(item["evidence_role"] == "primary" for item in selected))
        self.assertLess(len(selected), len(entries))

    def test_unverified_or_fabricated_targeted_fact_stays_omitted(self):
        overview, purpose = record(1, OVERVIEW), record(2, PURPOSE)
        forged = {**fact("WidgetEngine exists to replace human editors.", purpose),
                  "support": [{"evidence_ref": "E0002", "text": "invented quotation"}]}
        provider = OmittingProvider([fact(OVERVIEW, overview)], [forged])
        result = self.run_research(provider, [overview, purpose], "Why does WidgetEngine exist?")
        self.assertEqual([item["claim"] for item in result["facts"]], [OVERVIEW])
        self.assertEqual(len(provider.coverage_calls), 1)


if __name__ == "__main__":
    unittest.main()
