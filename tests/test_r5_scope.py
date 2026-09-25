"""Provenance and role balance through research and planner compaction."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.planner import plan
from source2reel.providers import StructuredOutputError
from source2reel.research import research
from source2reel.util import json_dump, json_load


def _project(root: Path) -> Path:
    project = root / "projects" / "demo"
    (root / "prompts").mkdir()
    for name in ("research", "storyboard", "planner_compact"):
        (root / "prompts" / f"{name}.txt").write_text("compact" if name == "planner_compact" else name)
    return project


def _evidence() -> list[dict]:
    return [
        {"ref": f"E{i:04d}", "kind": "document", "relative_path": f"core-{i}.md",
         "evidence_role": "primary", "excerpt": f"Core E{i:04d}, Own E{i:04d}, Fact E{i:04d} are documented."}
        for i in range(1, 5)
    ] + [
        {"ref": "E0005", "kind": "document", "relative_path": "examples/space-game/README.md",
         "evidence_role": "embedded_reference", "excerpt": "SpaceGame is documented."}
    ] + [
        {"ref": f"E{i:04d}", "kind": "document",
         "relative_path": "examples/space-game/manifests/research.json",
         "evidence_role": "generated_artifact", "excerpt": f"Embedded claim E{i:04d} is documented."}
        for i in range(6, 36)
    ]


class ResearchProvenanceTests(unittest.TestCase):
    def test_old_valid_checkpoint_is_rescoped_without_rerunning_provider(self):
        entries = _evidence()[:4]

        class Provider:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, user):
                self.calls += 1
                ref = json.loads(user)["evidence"][0]["ref"]
                return {"facts": [{"claim": f"Own {ref}", "evidence_refs": [ref],
                                   "support": [{"evidence_ref": ref,
                                                "text": f"Core {ref}, Own {ref}, Fact {ref} are documented."}]}], "assets": []}

        provider = Provider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _project(root)
            with patch("source2reel.research.split_for_context",
                       return_value=[entries[:2], entries[2:]]):
                research(provider, {"evidence": entries}, project)
                checkpoint = project / "manifests" / "research-parts" / "part-001.json"
                cached = json_load(checkpoint)
                cached["result"]["facts"].append({
                    "claim": "Previously accepted foreign fact", "evidence_refs": ["E0003"]
                })
                cached["result"]["assets"].append({"evidence_ref": "E0003"})
                json_dump(checkpoint, cached)
                output = research(provider, {"evidence": entries}, project)
            self.assertEqual(provider.calls, 2)
            self.assertNotIn("Previously accepted foreign fact", [f["claim"] for f in output["facts"]])
            self.assertEqual(output["assets"], [])

    def test_generated_batch_cannot_promote_its_claim_with_foreign_primary_ref(self):
        primary = _evidence()[:4]
        generated = _evidence()[5:10]

        class MasqueradingProvider:
            def complete_json(self, system, user):
                supplied = json.loads(user)["evidence"]
                if supplied[0]["evidence_role"] == "primary":
                    return {"facts": [{"claim": f"Core {e['ref']}",
                                       "evidence_refs": [e["ref"]], "support": [{
                                           "evidence_ref": e["ref"], "text": e["excerpt"]}]}
                                      for e in supplied], "assets": []}
                return {"facts": [{"claim": f"Embedded claim {e['ref']}",
                                   "evidence_refs": ["E0001"]} for e in supplied],
                        "assets": [{"evidence_ref": "E0001", "purpose": "foreign"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _project(root)
            with patch("source2reel.research.split_for_context",
                       return_value=[primary, generated]):
                output = research(MasqueradingProvider(), {"evidence": primary + generated}, project)
            self.assertEqual([fact["claim"] for fact in output["facts"]],
                             [f"Core {item['ref']}" for item in primary])
            self.assertEqual(output["assets"], [])

    def test_each_research_batch_rejects_other_globally_valid_fact_and_asset_refs(self):
        entries = _evidence()[:4]

        class CrossBatchProvider:
            def complete_json(self, system, user):
                supplied = json.loads(user)["evidence"]
                own = supplied[0]["ref"]
                foreign = "E0003" if own == "E0001" else "E0001"
                return {"facts": [{"claim": f"Own {own}", "evidence_refs": [own, foreign],
                                   "support": [{"evidence_ref": own, "text": supplied[0]["excerpt"]}]},
                                  {"claim": "Foreign claim", "evidence_refs": [foreign]}],
                        "assets": [{"evidence_ref": own, "purpose": "own"},
                                   {"evidence_ref": foreign, "purpose": "foreign"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _project(root)
            # Both batches contain refs that are globally valid; the provider
            # also cites a ref from the other batch in each request.
            with patch("source2reel.research.split_for_context",
                       return_value=[entries[:2], entries[2:]]):
                result = research(CrossBatchProvider(), {"evidence": entries}, project)
            parts = project / "manifests" / "research-parts"
            for index, supplied in ((1, {"E0001", "E0002"}), (2, {"E0003", "E0004"})):
                saved = json_load(parts / f"part-{index:03d}.json")["result"]
                self.assertTrue(saved["facts"])
                self.assertEqual(saved["facts"][0]["evidence_refs"], [f"E{1 if index == 1 else 3:04d}"])
                self.assertTrue(all(set(f["evidence_refs"]) <= supplied for f in saved["facts"]))
                self.assertTrue(all(a["evidence_ref"] in supplied for a in saved["assets"]))
            self.assertFalse(any(f["claim"] == "Foreign claim" for f in result["facts"]))

    def test_recursive_research_children_cannot_cite_siblings(self):
        class CrossChildProvider:
            def complete_json(self, system, user):
                supplied = json.loads(user)["evidence"]
                if len(supplied) > 1:
                    raise StructuredOutputError("too long")
                own = supplied[0]["ref"]
                other = "E0002" if own == "E0001" else "E0001"
                return {"facts": [{"claim": f"Fact {own}", "evidence_refs": [own, other],
                                   "support": [{"evidence_ref": own, "text": supplied[0]["excerpt"]}]}],
                        "assets": [{"evidence_ref": own}, {"evidence_ref": other}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _project(root)
            inventory = {"evidence": _evidence()[:2]}
            with patch("source2reel.research.split_for_context",
                       return_value=[inventory["evidence"]]):
                result = research(CrossChildProvider(), inventory, project, max_retries=0)
            parts = project / "manifests" / "research-parts"
            for suffix, ref in (("a", "E0001"), ("b", "E0002")):
                saved = json_load(parts / f"part-001-{suffix}.json")["result"]
                self.assertEqual(saved["facts"][0]["evidence_refs"], [ref])
                self.assertEqual([a["evidence_ref"] for a in saved["assets"]], [ref])
            self.assertEqual({r for fact in result["facts"] for r in fact["evidence_refs"]},
                             {"E0001", "E0002"})


class PlannerRoleTests(unittest.TestCase):
    def _run_plan(self, root: Path, instructions: str, entries=None):
        entries = [{**item, "excerpt": f"Grounded point from {item['relative_path']} ({item['ref']})"}
                   for item in (entries if entries is not None else _evidence())]
        roles = {item["ref"]: item["evidence_role"] for item in entries}
        research_input = {"facts": [
            {"claim": f"Grounded point from {item['relative_path']} ({item['ref']})",
             "evidence_refs": [item["ref"]], "phase": "final", "confidence": "high",
             "support": [{"evidence_ref": item["ref"],
                          "text": f"Grounded point from {item['relative_path']} ({item['ref']})"}]}
            for item in entries
        ], "assets": []}

        class LosingProvider:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                payload = json.loads(user)
                self.requests.append((system, payload))
                if system == "compact":
                    refs = [r for record in payload["records"] for r in (
                        record.get("evidence_refs") or record.get("capsule", {}).get("evidence_refs", []))]
                    # At level one the model omits all primary facts. At level
                    # two it collapses every supplied ref into one mixed fact.
                    if payload["level"] == 1:
                        refs = [r for r in refs if roles[r] != "primary"]
                        return {"capsules": [self.capsule([r], f"Reference {r}") for r in refs]}
                    # Reject the model's invented mixed claim, while retaining
                    # original facts explicitly selected from this level.
                    selected = [record.get("capsule", record) for record in payload["records"]]
                    return {"capsules": [self.capsule(refs, "One broad mixed capsule"), *[
                        {**self.capsule(fact["evidence_refs"], fact["claim"]),
                         "source_fact_id": fact["source_fact_id"]}
                        for fact in selected if fact.get("source_fact_id")
                    ]]}
                first = payload["evidence_index"][0]["ref"]
                selected = next(fact for fact in payload["research"]["facts"]
                                if first in fact["evidence_refs"])
                return {"version": 1, "title": "Demo", "slug": "demo", "summary": "Demo",
                        "scenes": [{"id": "s001", "type": "PROJECT_EVIDENCE", "title": "Proof",
                                    "narration": selected["claim"], "evidence_refs": [first],
                                    "fact_ids": [selected["fact_id"]], "asset_ref": first,
                                    "annotations": [], "pad_after_seconds": 0.5,
                                    "diagram": {}, "notes": ""}]}

            @staticmethod
            def capsule(refs, claim):
                return {"claim": claim, "evidence_refs": refs, "media_refs": [],
                        "phase": "final", "confidence": "high", "visual_purpose": ""}

        provider = LosingProvider()
        project = _project(root)
        with patch("source2reel.planner.fits_context", side_effect=[False, False, True, True]), \
             patch("source2reel.planner.split_for_context",
                   side_effect=lambda records, *args, **kwargs: [records]):
            plan(provider, research_input, {"evidence": entries}, project,
                 "WidgetEngine", instructions, max_retries=0)
        return json_load(project / "manifests" / "planner-evidence.json"), provider

    def test_two_levels_preserve_primary_backbone_and_bound_generated_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, provider = self._run_plan(Path(tmp),
                                                 "Explain the reusable WidgetEngine and its evidence pipeline.")
            self.assertEqual(manifest["levels"], 2)
            index = manifest["evidence_index"]
            primary = [e for e in index if e["evidence_role"] == "primary"]
            supporting = [e for e in index if e["evidence_role"] != "primary"]
            self.assertGreaterEqual(len(primary), 2)
            self.assertLessEqual(len(supporting), max(1, len(primary) // 2))
            self.assertLess(len(supporting), 10)
            self.assertTrue(supporting)
            for role in ("primary", "embedded_reference", "generated_artifact"):
                self.assertEqual(manifest["evidence_scope"][role],
                                 [entry["ref"] for entry in index if entry["evidence_role"] == role])
            self.assertTrue(any(set(f["evidence_refs"]) <= {p["ref"] for p in primary}
                                for f in manifest["research"]["facts"]))
            level_two = [req for system, req in provider.requests if system == "compact"
                         and req["level"] == 2]
            self.assertEqual(len(level_two), 1)
            carried = level_two[0]["records"]
            self.assertTrue(any(record["kind"] == "capsule" and
                                "primary" in record["evidence_roles"].values()
                                for record in carried))
            self.assertTrue(all(set(record["evidence_roles"]) ==
                                set(record["capsule"]["evidence_refs"]) for record in carried))
            carried_primary = {ref for record in carried for ref, role in
                               record["evidence_roles"].items() if role == "primary"}
            carried_support = {ref for record in carried for ref, role in
                               record["evidence_roles"].items() if role != "primary"}
            self.assertGreaterEqual(len(carried_primary), 2 * len(carried_support))

    def test_explicit_embedded_focus_can_keep_many_supporting_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self._run_plan(Path(tmp), "Focus on the reference project.")
            index = manifest["evidence_index"]
            self.assertEqual(manifest["levels"], 2)
            self.assertGreater(len([e for e in index if e["evidence_role"] != "primary"]), 10)

    def test_primary_anchors_cover_multiple_authoritative_sources(self):
        entries = _evidence()
        for i, entry in enumerate(entries[:4]):
            entry["relative_path"] = f"source-{1 + i // 2:02d}/core-{i}.md"
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self._run_plan(Path(tmp), "Explain the two top-level sources.", entries)
        primary_paths = [e["relative_path"] for e in manifest["evidence_index"]
                         if e["evidence_role"] == "primary"]
        self.assertTrue(any(path.startswith("source-01/") for path in primary_paths))
        self.assertTrue(any(path.startswith("source-02/") for path in primary_paths))

    def test_embedded_only_source_classified_primary_stays_available(self):
        entries = [{"ref": f"E{i:04d}", "kind": "document",
                    "relative_path": f"examples/space-game/proof-{i}.md", "evidence_role": "primary"}
                   for i in range(1, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self._run_plan(Path(tmp), "Explain the SpaceGame project.", entries)
        self.assertTrue(manifest["evidence_index"])
        self.assertTrue(all(e["evidence_role"] == "primary" for e in manifest["evidence_index"]))


if __name__ == "__main__":
    unittest.main()
