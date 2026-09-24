"""Subject-scoped research without an LLM or production media runtime."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.chunking import fits_context
from source2reel.inventory import build_inventory
from source2reel.pipeline import create
from source2reel.planner import _compact_payload, _evidence_index, _media_inventory
from source2reel.research import _consolidate, research


class RecordingProvider:
    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    def complete_json(self, system: str, user: str) -> dict:
        request = json.loads(user)
        self.requests.append((system, request))
        return {
            "facts": [
                {"claim": f"Statement from {e['ref']}", "evidence_refs": [e["ref"]],
                 "phase": "final", "confidence": "high"}
                for e in request["evidence"]
            ],
            "assets": [],
        }


class EvidenceSubjectScopingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "projects" / "widget-engine"
        (self.root / "prompts").mkdir()
        (self.root / "prompts" / "research.txt").write_text("Research the requested subject with cited evidence.")

    def synthetic_inventory(self):
        source = self.root / "source"
        (source / "examples" / "space-game" / "manifests").mkdir(parents=True)
        (source / "README.md").write_text("WidgetEngine is a reusable rendering engine.\n")
        (source / "engine.py").write_text("def draw(): pass\n")
        (source / "examples" / "space-game" / "README.md").write_text(
            ("SpaceGame runs on a toaster.\n" * 480)
        )
        (source / "examples" / "space-game" / "manifests" / "research.json").write_text(
            json.dumps({"facts": ["SpaceGame runs on a toaster"] * 1200}, indent=2)
        )
        return build_inventory(source, self.project, max_file_bytes=50000, chunk_chars=1400)

    def flooded_inventory(self):
        source = self.root / "flooded-source"
        embedded = source / "examples" / "space-game"
        (embedded / "manifests").mkdir(parents=True)
        for filename in ("README.md", "ARCHITECTURE.md", "API.md", "DESIGN.md"):
            (source / filename).write_text(f"WidgetEngine core: {filename} describes its rendering pipeline.\n")
        (embedded / "README.md").write_text("SpaceGame demonstrates WidgetEngine in use.\n")
        (embedded / "manifests" / "research.json").write_text(
            json.dumps({"facts": ["SpaceGame runs on a toaster"] * 1500}, indent=2)
        )
        return build_inventory(source, self.project, max_file_bytes=50000, chunk_chars=500)

    @staticmethod
    def selected_roles(result, inventory):
        roles = {e["ref"]: e["evidence_role"] for e in inventory["evidence"]}
        return [roles[ref] for fact in result["facts"] for ref in fact["evidence_refs"]]

    def test_primary_reference_and_generated_roles_remain_in_inventory(self):
        inventory = self.synthetic_inventory()
        by_path: dict[str, set[str]] = {}
        for entry in inventory["evidence"]:
            by_path.setdefault(entry["relative_path"], set()).add(entry["evidence_role"])
        self.assertEqual(by_path["README.md"], {"primary"})
        self.assertEqual(by_path["engine.py"], {"primary"})
        self.assertEqual(by_path["examples/space-game/README.md"], {"embedded_reference"})
        self.assertEqual(by_path["examples/space-game/manifests/research.json"], {"generated_artifact"})
        self.assertTrue(all(entry["ref"] for entry in inventory["evidence"]))

    def test_embedded_only_root_is_not_reclassified(self):
        source = self.root / "only-examples"
        (source / "examples" / "one").mkdir(parents=True)
        (source / "examples" / "one" / "README.md").write_text("This is the requested source.\n")
        inventory = build_inventory(source, self.project)
        self.assertEqual(inventory["evidence"][0]["evidence_role"], "primary")

    def test_single_line_generated_artifact_stays_context_safe(self):
        source = self.root / "minified-source"
        artifact = source / "examples" / "sample" / "manifests" / "research.json"
        artifact.parent.mkdir(parents=True)
        (source / "README.md").write_text("WidgetEngine is the subject.\n")
        artifact.write_text(json.dumps({"repeated": "x" * 12000}))
        inventory = build_inventory(source, self.project, chunk_chars=800)
        generated = [e for e in inventory["evidence"] if e["evidence_role"] == "generated_artifact"]
        self.assertGreater(len(generated), 1)
        self.assertTrue(all(len(e["excerpt"]) <= 800 and e["line_start"] == 1 for e in generated))
        provider = RecordingProvider()
        research(provider, inventory, self.project, max_chars=8000, context_size=1900,
                 output_reserve_tokens=400, safety_tokens=300, title_hint="WidgetEngine")
        self.assertTrue(provider.requests)

    def test_multiple_sources_are_classified_independently(self):
        first = self.root / "source-one"
        second = self.root / "source-two"
        (first / "examples" / "nested").mkdir(parents=True)
        (first / "README.md").write_text("First source.\n")
        (first / "examples/nested/README.md").write_text("Example.\n")
        (second / "examples" / "nested").mkdir(parents=True)
        (second / "examples/nested/README.md").write_text("Entire second source.\n")
        entries = build_inventory([first, second], self.project)["evidence"]
        roles = {entry["relative_path"]: entry["evidence_role"] for entry in entries}
        self.assertEqual(roles["source-01/examples/nested/README.md"], "embedded_reference")
        self.assertEqual(roles["source-02/examples/nested/README.md"], "primary")

    def test_research_subject_instructions_budget_and_reference_citation(self):
        inventory = self.synthetic_inventory()
        provider = RecordingProvider()
        result = research(
            provider, inventory, self.project, max_chars=6000, context_size=3000,
            output_reserve_tokens=600, safety_tokens=400,
            title_hint="WidgetEngine", instructions="Show its example briefly.",
        )
        self.assertGreater(len(provider.requests), 1)
        first_role = provider.requests[0][1]["evidence"][0]["evidence_role"]
        self.assertEqual(first_role, "primary")
        roles = set()
        for system, request in provider.requests:
            self.assertEqual(request["project_title_hint"], "WidgetEngine")
            self.assertEqual(request["optional_instructions"], "Show its example briefly.")
            self.assertTrue(fits_context(system, json.dumps(request, ensure_ascii=False), 3000, 600, 400))
            roles.update(e["evidence_role"] for e in request["evidence"])
        self.assertEqual(roles, {"primary", "embedded_reference", "generated_artifact"})
        reference_refs = {e["ref"] for e in inventory["evidence"] if e["evidence_role"] != "primary"}
        cited_refs = {r for fact in result["facts"] for r in fact["evidence_refs"]}
        self.assertTrue(reference_refs & cited_refs)

    def test_repetitive_generated_facts_cannot_outvote_primary_evidence(self):
        inventory = self.flooded_inventory()
        self.assertGreater(sum(e["evidence_role"] == "generated_artifact"
                               for e in inventory["evidence"]), 50)
        provider = RecordingProvider()
        result = research(provider, inventory, self.project, max_chars=6000,
                          context_size=3000, output_reserve_tokens=600, safety_tokens=400,
                          title_hint="WidgetEngine", instructions="")
        roles = self.selected_roles(result, inventory)
        self.assertEqual(roles.count("primary"), 4)
        self.assertEqual(roles.count("embedded_reference"), 1)
        self.assertEqual(roles.count("generated_artifact"), 1)
        valid_refs = {e["ref"] for e in inventory["evidence"]}
        self.assertTrue(all(ref in valid_refs for fact in result["facts"] for ref in fact["evidence_refs"]))

    def test_explicit_example_focus_preserves_secondary_facts(self):
        inventory = self.flooded_inventory()
        provider = RecordingProvider()
        instruction = "Focus on SpaceGame as the reference project."
        result = research(provider, inventory, self.project, max_chars=6000,
                          context_size=3000, output_reserve_tokens=600, safety_tokens=400,
                          title_hint="WidgetEngine", instructions=instruction)
        roles = self.selected_roles(result, inventory)
        self.assertGreater(roles.count("generated_artifact"), 10)
        self.assertTrue(all(req["optional_instructions"] == instruction for _, req in provider.requests))

    def test_supporting_assets_respect_same_reference_budget(self):
        inventory = {"evidence": [
            *({"ref": f"P{i}", "evidence_role": "primary"} for i in range(4)),
            *({"ref": f"G{i}", "evidence_role": "generated_artifact"} for i in range(50)),
        ]}
        facts = [{"claim": f"Primary {i}", "evidence_refs": [f"P{i}"]} for i in range(4)]
        assets = [{"evidence_ref": f"G{i}", "purpose": f"Visual {i}"} for i in range(50)]
        selected_facts, selected_assets = _consolidate(facts, assets, inventory, "WidgetEngine", "")
        self.assertEqual(selected_facts, facts)
        self.assertEqual(len(selected_assets), 2)

    def test_primary_facts_from_multiple_sources_are_all_retained(self):
        inventory = {"evidence": [
            *({"ref": f"P{i}", "evidence_role": "primary",
               "relative_path": f"source-{1 + i // 2:02d}/README-{i}.md"} for i in range(4)),
            {"ref": "E1", "evidence_role": "embedded_reference",
             "relative_path": "source-02/examples/demo/README.md"},
            {"ref": "G1", "evidence_role": "generated_artifact",
             "relative_path": "source-01/examples/demo/manifests/research.json"},
        ]}
        facts = ([{"claim": f"Core {i}", "evidence_refs": [f"P{i}"]} for i in range(4)]
                 + [{"claim": "Example", "evidence_refs": ["E1"]},
                    {"claim": "Derived example", "evidence_refs": ["G1"]}])
        selected, _ = _consolidate(facts, [], inventory, "WidgetEngine", "")
        self.assertEqual({f["evidence_refs"][0] for f in selected},
                         {"P0", "P1", "P2", "P3", "E1", "G1"})

    def test_generic_explicit_reference_focus_lifts_default_quota(self):
        inventory = {"evidence": [
            {"ref": "P0", "evidence_role": "primary"},
            *({"ref": f"G{i}", "evidence_role": "generated_artifact",
               "relative_path": "examples/demo/manifests/evidence.json"} for i in range(10)),
        ]}
        facts = ([{"claim": "Core", "evidence_refs": ["P0"]}]
                 + [{"claim": f"Example {i}", "evidence_refs": [f"G{i}"]} for i in range(10)])
        selected, _ = _consolidate(facts, [], inventory, "WidgetEngine",
                                   "Focus on the reference project.")
        self.assertEqual(selected, facts)

    def test_without_primary_facts_supporting_evidence_is_not_silenced(self):
        inventory = {"evidence": [{"ref": f"G{i}", "evidence_role": "generated_artifact"}
                                  for i in range(5)]}
        facts = [{"claim": f"Example {i}", "evidence_refs": [f"G{i}"]} for i in range(5)]
        selected, _ = _consolidate(facts, [], inventory, "WidgetEngine", "")
        self.assertEqual(selected, facts)

    def test_changed_subject_or_instructions_invalidates_research_checkpoints(self):
        inventory = {"evidence": [{"ref": "E0001", "kind": "document", "excerpt": "WidgetEngine."}]}
        provider = RecordingProvider()
        kwargs = {"title_hint": "WidgetEngine", "instructions": ""}
        research(provider, inventory, self.project, **kwargs)
        self.assertEqual(provider.requests[-1][1]["optional_instructions"], "")
        research(provider, inventory, self.project, **kwargs)
        self.assertEqual(len(provider.requests), 1)
        research(provider, inventory, self.project, title_hint="AnotherEngine", instructions="")
        self.assertEqual(len(provider.requests), 2)
        research(provider, inventory, self.project, title_hint="AnotherEngine", instructions="Focus on API")
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(provider.requests[-1][1]["optional_instructions"], "Focus on API")

    def test_long_subject_instructions_are_included_in_context_budget(self):
        inventory = {"evidence": [{"ref": "E0001", "kind": "document", "excerpt": "Short fact."}]}
        provider = RecordingProvider()
        with self.assertRaisesRegex(ValueError, "context budget"):
            research(provider, inventory, self.project, context_size=2200,
                     output_reserve_tokens=500, safety_tokens=300,
                     title_hint="WidgetEngine", instructions="Focus " * 1200)
        self.assertEqual(provider.requests, [])

    def test_planner_compaction_can_see_subject_and_reference_role(self):
        inventory = {"evidence": [
            {"ref": "E0001", "kind": "media", "relative_path": "examples/demo/screen.png",
             "evidence_role": "embedded_reference"},
        ]}
        media = _media_inventory(inventory)
        index = _evidence_index(inventory)
        self.assertEqual(media[0]["evidence_role"], "embedded_reference")
        self.assertEqual(index[0]["evidence_role"], "embedded_reference")
        request = _compact_payload(1, 1, [{"kind": "media", "media": media[0]}],
                                   "WidgetEngine", "Focus on its demo.")
        self.assertEqual(request["project_title_hint"], "WidgetEngine")
        self.assertEqual(request["optional_instructions"], "Focus on its demo.")

    def test_create_forwards_same_title_and_exact_instructions_to_both_stages(self):
        cfg = {"ingest": {}, "vision": {"enabled": False}, "research": {}, "local_ai": {}, "chunking": {}}
        inventory = {"evidence": []}
        with patch("source2reel.pipeline.load_engine_config", return_value=cfg), \
             patch("source2reel.pipeline._ingest_many", return_value=[self.root / "source"]), \
             patch("source2reel.pipeline.build_inventory", return_value=inventory), \
             patch("source2reel.pipeline.provider_from_config", return_value=object()), \
             patch("source2reel.pipeline.research", return_value={"facts": [], "assets": []}) as do_research, \
             patch("source2reel.pipeline.plan", return_value={}) as do_plan:
            create(self.root, ["https://github.com/org/widget-engine", "https://example.com/docs"],
                   None, "", False, True)
        self.assertEqual(do_research.call_args.kwargs["title_hint"], "widget-engine")
        self.assertEqual(do_research.call_args.kwargs["instructions"], "")
        self.assertEqual(do_plan.call_args.args[4:6], ("widget-engine", ""))


if __name__ == "__main__":
    unittest.main()
