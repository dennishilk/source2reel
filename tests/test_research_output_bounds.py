"""Dense single sources recover without changing global evidence provenance."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.research import research
from source2reel.util import json_load


class ResearchOutputBoundsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "prompts").mkdir()
        prompt = Path(__file__).resolve().parents[1] / "prompts" / "research.txt"
        (self.root / "prompts" / "research.txt").write_text(prompt.read_text())
        self.project = self.root / "projects" / "widget"

    def record(self, lines=16):
        return {
            "ref": "E0103", "kind": "document", "relative_path": "docs/STATE.md",
            "path": "sources/source-01/repo/docs/STATE.md", "path_base": "project",
            "sha256": "source-digest", "evidence_role": "primary",
            "line_start": 25, "line_end": 24 + lines,
            "excerpt": "\n".join(
                f"WidgetEngine documented detail {i:02d} remains available to users."
                for i in range(1, lines + 1)
            ),
        }

    def run_research(self, provider, records, **kwargs):
        return research(provider, {"evidence": records}, self.project,
                        title_hint="WidgetEngine", max_retries=0, **kwargs)

    def test_single_record_output_limit_splits_without_changing_ref_or_source(self):
        record = self.record()

        class Provider:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                item = json.loads(user)["evidence"][0]
                self.requests.append(item)
                if len(item["excerpt"].splitlines()) > 8:
                    raise OutputLimitExceeded("Local AI response reached the configured output token limit")
                span = item["excerpt"].splitlines()[0]
                return {"facts": [
                    {"claim": span, "evidence_refs": ["E0103"], "phase": "final",
                     "confidence": "high", "support": [{"evidence_ref": "E0103", "text": span}]},
                    {"claim": "Foreign ref", "evidence_refs": ["E0999"]},
                    {"claim": "Wrong segment quotation", "evidence_refs": ["E0103"],
                     "support": [{"evidence_ref": "E0103", "text": (
                         record["excerpt"].splitlines()[-1] if item["line_start"] == 25
                         else record["excerpt"].splitlines()[0])}]},
                ], "assets": []}

        provider = Provider()
        output = self.run_research(provider, [record])
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(len(output["facts"]), 2)
        self.assertEqual({tuple(f["evidence_refs"]) for f in output["facts"]}, {("E0103",)})
        self.assertEqual({f["subject_scope"] for f in output["facts"]}, {"main_subject"})
        self.assertTrue(all(f["support"][0]["text"] in provider.requests[index]["excerpt"]
                            for f, index in zip(output["facts"], (1, 2))))
        children = provider.requests[1:]
        self.assertEqual("".join(e["excerpt"] for e in children), record["excerpt"])
        self.assertEqual([(e["line_start"], e["line_end"]) for e in children],
                         [(25, 32), (33, 40)])
        for child in children:
            for key in ("ref", "path", "path_base", "relative_path", "sha256", "evidence_role"):
                self.assertEqual(child[key], record[key])
        parts = self.project / "manifests" / "research-parts"
        self.assertTrue((parts / "part-001-split.json").exists())
        self.assertTrue((parts / "part-001-a.json").exists())
        self.assertTrue((parts / "part-001-b.json").exists())
        manifest = json_load(self.project / "manifests" / "research.json")
        self.assertEqual({ref for fact in manifest["facts"] for ref in fact["evidence_refs"]},
                         {"E0103"})
        # Completed child checkpoints and the split route are reused on resume.
        self.assertEqual(self.run_research(provider, [record]), output)
        self.assertEqual(len(provider.requests), 3)

    def test_one_record_can_split_recursively_and_reuse_child_checkpoints(self):
        record = self.record(8)

        class Provider:
            calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                item = json.loads(user)["evidence"][0]
                if len(item["excerpt"].splitlines()) > 2:
                    raise OutputLimitExceeded("output token limit")
                span = item["excerpt"].splitlines()[0]
                return {"facts": [{"claim": span, "evidence_refs": ["E0103"],
                                   "support": [{"evidence_ref": "E0103", "text": span}]}],
                        "assets": []}

        provider = Provider()
        result = self.run_research(provider, [record])
        self.assertEqual(provider.calls, 7)
        self.assertEqual(len(result["facts"]), 4)
        self.assertEqual([fact["evidence_refs"] for fact in result["facts"]], [["E0103"]] * 4)
        parts = self.project / "manifests" / "research-parts"
        self.assertTrue((parts / "part-001-a-split.json").exists())
        self.assertTrue((parts / "part-001-b-split.json").exists())
        self.assertEqual(self.run_research(provider, [record]), result)
        self.assertEqual(provider.calls, 7)

    def test_single_line_uses_sentence_boundaries_without_inventing_new_lines(self):
        record = self.record(1)
        record["excerpt"] = (
            "WidgetEngine documents its source index. "
            "WidgetEngine documents its local build artifacts."
        )

        class Provider:
            def __init__(self):
                self.items = []

            def complete_json(self, system, user):
                item = json.loads(user)["evidence"][0]
                self.items.append(item)
                if len(item["excerpt"]) > 55:
                    raise OutputLimitExceeded("output token limit")
                span = item["excerpt"].strip()
                return {"facts": [{"claim": span, "evidence_refs": ["E0103"],
                                   "support": [{"evidence_ref": "E0103", "text": span}]}],
                        "assets": []}

        provider = Provider()
        result = self.run_research(provider, [record])
        self.assertEqual(len(result["facts"]), 2)
        self.assertEqual("".join(item["excerpt"] for item in provider.items[1:]),
                         record["excerpt"])
        self.assertEqual([(item["line_start"], item["line_end"])
                          for item in provider.items[1:]], [(25, 25), (25, 25)])

    def test_overall_per_ref_limit_applies_across_recursive_children(self):
        record = self.record(16)

        class Provider:
            def complete_json(self, system, user):
                item = json.loads(user)["evidence"][0]
                if len(item["excerpt"].splitlines()) > 2:
                    raise OutputLimitExceeded("output token limit")
                span = item["excerpt"].splitlines()[0]
                return {"facts": [{"claim": span, "evidence_refs": ["E0103"],
                                   "support": [{"evidence_ref": "E0103", "text": span}]}],
                        "assets": [{"evidence_ref": "E0103", "purpose": span}]}

        result = self.run_research(Provider(), [record])
        self.assertEqual(len(result["facts"]), 6)
        self.assertEqual(len(result["assets"]), 2)

    def test_split_supporting_example_retains_scope_and_rejects_main_claim(self):
        record = {**self.record(8), "evidence_role": "generated_artifact",
                  "relative_path": "projects/demo-phone/manifests/research.json"}
        record["excerpt"] = record["excerpt"].replace("WidgetEngine", "DemoPhone")

        class Provider:
            def complete_json(self, system, user):
                item = json.loads(user)["evidence"][0]
                if len(item["excerpt"].splitlines()) > 4:
                    raise OutputLimitExceeded("output token limit")
                span = item["excerpt"].splitlines()[0]
                return {"facts": [
                    {"claim": "WidgetEngine owns DemoPhone's properties.",
                     "evidence_refs": ["E0103"],
                     "support": [{"evidence_ref": "E0103", "text": span}]},
                    {"claim": span, "evidence_refs": ["E0103"],
                     "support": [{"evidence_ref": "E0103", "text": span}]},
                ], "assets": []}

        result = self.run_research(Provider(), [record])
        self.assertEqual(len(result["facts"]), 2)
        self.assertTrue(all(fact["claim"].startswith("DemoPhone") for fact in result["facts"]))
        self.assertEqual({fact["subject_scope"] for fact in result["facts"]}, {"supporting_only"})

    def test_indivisible_source_reports_original_ref_and_output_limit(self):
        record = self.record(1)

        class Provider:
            calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                raise OutputLimitExceeded("Local AI response reached the configured output token limit")

        provider = Provider()
        with self.assertRaisesRegex(RuntimeError,
                                    r"E0103.*output limit.*cannot be subdivided further") as caught:
            self.run_research(provider, [record])
        self.assertNotIn("invalid structured output", str(caught.exception))
        self.assertEqual(provider.calls, 1)
        self.assertEqual(list((self.project / "manifests" / "research-parts").glob("*split*")), [])

    def test_other_malformed_single_record_is_not_text_split(self):
        record = self.record(16)

        class Provider:
            calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                raise StructuredOutputError("malformed JSON")

        provider = Provider()
        with self.assertRaisesRegex(RuntimeError, "single record E0103.*malformed JSON"):
            self.run_research(provider, [record])
        self.assertEqual(provider.calls, 1)

    def test_recursive_split_has_a_finite_depth(self):
        record = self.record(128)

        class Provider:
            calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                raise OutputLimitExceeded("output token limit")

        provider = Provider()
        with self.assertRaisesRegex(RuntimeError,
                                    r"E0103.*output limit.*maximum split depth 6"):
            self.run_research(provider, [record])
        self.assertEqual(provider.calls, 7)

    def test_output_caps_apply_even_when_provider_ignores_contract(self):
        record = self.record(4)

        class Provider:
            def complete_json(self, system, user):
                request = json.loads(user)
                self.system = system
                self.request = request
                facts = [{"claim": f"WidgetEngine documented detail {i:02d} remains available.",
                          "evidence_refs": ["E0103"], "support": [
                              {"evidence_ref": "E0103", "text": request["evidence"][0]["excerpt"]
                               .splitlines()[0]}]} for i in range(25)]
                assets = [{"evidence_ref": "E0103", "purpose": f"Visual {i}"}
                          for i in range(25)]
                return {"facts": facts, "assets": assets}

        provider = Provider()
        result = self.run_research(provider, [record])
        self.assertEqual(len(result["facts"]), 6)
        self.assertEqual(len(result["assets"]), 2)
        self.assertEqual(provider.request["research_limits"], {
            "facts_per_ref": 6, "facts_per_request": 12,
            "assets_per_ref": 2, "assets_per_request": 6,
        })
        self.assertIn("6 facts per evidence ref", provider.system)
        self.assertIn("12 facts per request", provider.system)
        self.assertIn("2 assets per evidence ref", provider.system)
        self.assertIn("6 assets per request", provider.system)

    def test_distinct_records_share_request_cap_and_keep_batch_local_refs(self):
        records = [self.record(2), {**self.record(2), "ref": "E0104",
                                     "relative_path": "docs/SECOND.md"}]

        class Provider:
            def complete_json(self, system, user):
                supplied = json.loads(user)["evidence"]
                return {"facts": [{"claim": f"Claim {entry['ref']} {i}",
                                   "evidence_refs": [entry["ref"]]}
                                  for entry in supplied for i in range(20)],
                        "assets": [{"evidence_ref": entry["ref"], "purpose": f"Asset {i}"}
                                   for entry in supplied for i in range(20)]}

        result = self.run_research(Provider(), records)
        self.assertEqual(len(result["facts"]), 12)
        self.assertEqual([sum(e["ref"] in f["evidence_refs"] for f in result["facts"])
                          for e in records], [6, 6])
        self.assertEqual(len(result["assets"]), 4)

    def test_successful_earlier_batch_remains_checkpointed_during_later_split(self):
        first = {**self.record(1), "ref": "E0101", "excerpt": "WidgetEngine first reference stays valid."}
        second = self.record(8)

        class Provider:
            def __init__(self):
                self.seen = []

            def complete_json(self, system, user):
                item = json.loads(user)["evidence"][0]
                self.seen.append(item["ref"])
                if item["ref"] == "E0103" and len(item["excerpt"].splitlines()) > 4:
                    raise OutputLimitExceeded("output token limit")
                return {"facts": [{"claim": item["excerpt"].splitlines()[0],
                                   "evidence_refs": [item["ref"]]}], "assets": []}

        provider = Provider()
        with patch("source2reel.research.split_for_context", return_value=[[first], [second]]):
            self.run_research(provider, [first, second])
            self.assertEqual(provider.seen, ["E0101", "E0103", "E0103", "E0103"])
            self.run_research(provider, [first, second])
        self.assertEqual(provider.seen, ["E0101", "E0103", "E0103", "E0103"])


if __name__ == "__main__":
    unittest.main()
