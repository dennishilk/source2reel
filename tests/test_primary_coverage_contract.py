"""Planner primary coverage and integrated OUTRO contract regressions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.planner import _make_ask, _outline_payload, _primary_anchors, plan
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.renderer import build_episode
from source2reel.schema import validate_episode, validate_presentation
from source2reel.util import json_load


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = "Explain WidgetEngine's normal source-to-output pipeline and optional maintenance path."
DEFINITION = (
    "WidgetEngine ingests source packages and produces indexed evidence records "
    "with stable references and hashes."
)
NORMAL_FLOW = (
    "The normal WidgetEngine workflow researches evidence, plans a storyboard "
    "and renders configured output."
)
OPTIONAL_FLOW = "WidgetEngine optionally clears old output caches during maintenance."
PRESENTATION = {
    "outro": {"headline": ["The documented project"],
              "links": [{"label": "Project documentation", "url": ["https://example.test/project"]}]},
    "scene_titles": {"s001": "The project"},
}


def _project(root: Path) -> Path:
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "storyboard.txt").write_text("storyboard")
    (prompts / "planner_compact.txt").write_text("compact")
    project = root / "projects" / "demo"
    (project / "sources").mkdir(parents=True)
    (project / "sources" / "source.json").write_text(json.dumps({
        "kind": "website", "source": "https://example.test/project"
    }))
    return project


def _primary_input():
    entries, facts = [], []
    for i in range(1, 51):
        ref = f"E{i:04d}"
        path = f"implementation/module-{i:02d}.py"
        claim = f"WidgetEngine worker module {i} configures buffer allocation step {i}."
        if i == 12:
            path, claim = "OVERVIEW.md", DEFINITION
        if i == 29:
            path, claim = "docs/normal-flow.md", NORMAL_FLOW
        if i == 43:
            path, claim = "other-source/overview.md", (
                "WidgetEngine accepts another original media source and includes it "
                "in the evidence inventory."
            )
        entries.append({"ref": ref, "kind": "document", "relative_path": path,
                        "evidence_role": "primary"})
        facts.append({"claim": claim, "evidence_refs": [ref],
                      "support": [{"evidence_ref": ref, "text": claim}],
                      "phase": "final", "confidence": "high"})
    ref = "E0051"
    entries.append({"ref": ref, "kind": "document", "relative_path": "ops/maintenance.md",
                    "evidence_role": "primary"})
    facts.append({"claim": OPTIONAL_FLOW, "evidence_refs": [ref],
                  "support": [{"evidence_ref": ref, "text": OPTIONAL_FLOW}],
                  "phase": "final", "confidence": "high"})
    for i in range(101, 141):
        ref = f"E{i:04d}"
        entries.append({"ref": ref, "kind": "document",
                        "relative_path": f"artifacts/report-{i}.json",
                        "evidence_role": "generated_artifact"})
        facts.append({"claim": f"Generated report {i} records an implementation detail.",
                      "support": [{"evidence_ref": ref,
                                   "text": f"Generated report {i} records an implementation detail."}],
                      "evidence_refs": [ref], "phase": "unknown", "confidence": "medium"})
    return {"evidence": entries}, {"version": 1, "facts": facts, "assets": []}


class PrimaryCoverageTests(unittest.TestCase):
    def test_requested_purpose_survives_bounded_primary_selection_only_when_sourced(self):
        inventory, research = _primary_input()
        purpose = ("WidgetEngine exists to turn authoritative project sources into "
                   "evidence-grounded technical explainers through a local production pipeline.")
        instructions = "Explain what WidgetEngine is, why it exists, and how it works."
        inventory["evidence"].extend([
            {"ref": "E0052", "kind": "document", "relative_path": "README.md",
             "evidence_role": "primary"},
            {"ref": "E0200", "kind": "document", "relative_path": "tests/test_config.py",
             "evidence_role": "embedded_reference"},
        ])
        research["facts"].extend([
            {"claim": purpose, "evidence_refs": ["E0052"],
             "support": [{"evidence_ref": "E0052", "text": purpose}],
             "phase": "final", "confidence": "high"},
            {"claim": "The configuration uses [profile] with name = widget-series.",
             "evidence_refs": ["E0200"], "support": [{"evidence_ref": "E0200",
             "text": "The configuration uses [profile] with name = widget-series."}],
             "phase": "final", "confidence": "high"},
        ])
        anchors = _primary_anchors(research, inventory, set(), "WidgetEngine", instructions)
        self.assertLessEqual(len(anchors), 10)
        self.assertIn(purpose, [item["claim"] for item in anchors])
        self.assertNotIn("widget-series", " ".join(item["claim"] for item in anchors))
        ask = _make_ask(research, [], inventory["evidence"], "WidgetEngine", instructions)
        purpose_id = next(item["fact_id"] for item in ask["research"]["facts"]
                          if item["claim"] == purpose)
        self.assertIn(purpose_id, ask["priority_fact_ids"])
        without_purpose = {**research, "facts": research["facts"][:-2] + research["facts"][-1:]}
        self.assertNotIn(purpose, [item["claim"] for item in _primary_anchors(
            without_purpose, inventory, set(), "WidgetEngine", instructions)])

    def test_canonical_primary_claims_survive_two_compaction_levels_with_secondary_quota(self):
        inventory, research = _primary_input()
        roles = {entry["ref"]: entry["evidence_role"] for entry in inventory["evidence"]}

        class OmittingProvider:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, user):
                request = json.loads(user)
                self.requests.append(request)
                if system == "compact":
                    refs = [ref for record in request["records"]
                            for ref in record.get("evidence_refs",
                                                  record.get("capsule", {}).get("evidence_refs", []))]
                    # The model loses primary detail at both levels. Original
                    # source facts must still reach the final planner intact.
                    return {"capsules": [
                        {"claim": f"Supporting {ref}", "evidence_refs": [ref],
                         "media_refs": [], "phase": "unknown", "confidence": "low",
                         "visual_purpose": ""}
                        for ref in refs if roles[ref] != "primary"
                    ]}
                ref = request["evidence_index"][0]["ref"]
                selected = next(fact for fact in request["research"]["facts"]
                                if ref in fact["evidence_refs"])
                return {"version": 1, "title": "WidgetEngine", "slug": "widget-engine",
                        "summary": selected["claim"], "scenes": [{
                            "id": "s001", "type": "SUMMARY", "narration": selected["claim"],
                            "fact_ids": [selected["fact_id"]], "evidence_refs": [ref],
                        }]}

        provider = OmittingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            with patch("source2reel.planner.fits_context", side_effect=[False, False, True, True]), \
                 patch("source2reel.planner.split_for_context",
                       side_effect=lambda records, *args, **kwargs: [records]):
                plan(provider, research, inventory, project, "WidgetEngine", INSTRUCTIONS,
                     max_retries=0)
            manifest = json_load(project / "manifests" / "planner-evidence.json")
        self.assertEqual(manifest["levels"], 2)
        claims = {fact["claim"] for fact in manifest["research"]["facts"]}
        self.assertIn(DEFINITION, claims)
        self.assertIn(NORMAL_FLOW, claims)
        self.assertIn(OPTIONAL_FLOW, claims)
        first_claims = [fact["claim"] for fact in manifest["research"]["facts"][:4]]
        self.assertIn(DEFINITION, first_claims)
        self.assertIn(NORMAL_FLOW, first_claims)
        refs = manifest["evidence_scope"]
        self.assertLessEqual(len(refs["generated_artifact"]),
                             max(1, len(refs["primary"]) // 2))
        self.assertLess(len(refs["generated_artifact"]), 10)
        self.assertTrue(all(request["optional_instructions"] == INSTRUCTIONS
                            for request in provider.requests))

    def test_distinct_information_beats_repeated_primary_claims_from_many_files(self):
        inventory = {"evidence": []}
        research = {"facts": [], "assets": []}
        claims = [
            "WidgetEngine accepts source packages as original inputs.",
            "WidgetEngine records stable evidence identifiers and content hashes.",
            "WidgetEngine plans configured output using cited evidence.",
        ]
        for i in range(30):
            ref = f"E{i+1:04d}"
            inventory["evidence"].append({"ref": ref, "relative_path": f"copy-{i}.md",
                                          "evidence_role": "primary"})
            research["facts"].append({
                "claim": "WidgetEngine performs local, offline source processing in one engine.",
                "evidence_refs": [ref], "phase": "final", "confidence": "high"})
        for i, claim in enumerate(claims):
            ref = f"E{i+31:04d}"
            inventory["evidence"].append({"ref": ref, "relative_path": f"docs/topic-{i}.md",
                                          "evidence_role": "primary"})
            research["facts"].append({"claim": claim, "evidence_refs": [ref],
                                       "phase": "final", "confidence": "high"})
        selected = _primary_anchors(research, inventory, set(), "WidgetEngine",
                                    "Explain its source inputs, evidence, and output.")
        selected_claims = [entry["claim"] for entry in selected]
        self.assertTrue(all(claim in selected_claims for claim in claims))
        self.assertLessEqual(selected_claims.count(
            "WidgetEngine performs local, offline source processing in one engine."), 1)
        self.assertLessEqual(len(selected), 10)
        self.assertLessEqual(sum(len(entry["claim"].encode()) for entry in selected), 2600)

    def test_multiple_authoritative_sources_and_normal_flow_beat_optional_only(self):
        inventory, research = _primary_input()
        selected = _primary_anchors(research, inventory, set(), "WidgetEngine", INSTRUCTIONS)
        claims = {item["claim"] for item in selected}
        self.assertIn(NORMAL_FLOW, claims)
        self.assertIn(DEFINITION, claims)
        self.assertIn(OPTIONAL_FLOW, claims)
        refs = {ref for item in selected for ref in item["evidence_refs"]}
        self.assertIn("E0043", refs)  # Independent later authoritative source.

    def test_top_level_primary_workflow_survives_without_repeating_subject_name(self):
        inventory = {"evidence": []}
        research = {"facts": [], "assets": []}
        details = ["network packet frames", "archive bundles", "storage layout pages",
                   "configuration keys", "plugin markers", "logging records",
                   "bitmap sprites", "audio waveforms", "hardware switches",
                   "error envelopes", "color palettes", "license headers"]
        for i, detail in enumerate(details, 1):
            ref = f"E{i:04d}"
            inventory["evidence"].append({"ref": ref,
                "relative_path": f"implementation/component-{i}.py", "evidence_role": "primary"})
            research["facts"].append({"claim": f"WidgetEngine processes {detail}.",
                                      "evidence_refs": [ref], "phase": "final",
                                      "confidence": "high"})
        overview = "Original sources become hashed evidence before planning and output."
        inventory["evidence"].append({"ref": "E0013", "relative_path": "overview.md",
                                      "evidence_role": "primary"})
        research["facts"].append({"claim": overview, "evidence_refs": ["E0013"],
                                  "phase": "final", "confidence": "high"})
        anchors = _primary_anchors(research, inventory, set(), "WidgetEngine",
                                   "Explain what WidgetEngine does.")
        self.assertIn(overview, [item["claim"] for item in anchors])


class IntegratedOutroTests(unittest.TestCase):
    @staticmethod
    def _episode(presentation=None, kind="OUTRO"):
        episode = {"version": 1, "title": "Grounded result", "slug": "grounded-result",
                   "summary": "Only supported claims", "scenes": [{
                   "id": "s001", "type": kind, "narration": "Closing.",
                       "evidence_refs": [] if kind == "OUTRO" else ["E0001"],
                   }]}
        if presentation is not None:
            episode["presentation"] = presentation
        return episode

    def test_integrated_outro_contract_and_external_presentation_override(self):
        with self.assertRaisesRegex(ValueError, "OUTRO requires presentation.outro"):
            validate_episode(self._episode(), set(), require_integrated_presentation=True)
        validate_episode(self._episode(PRESENTATION), set(),
                         require_integrated_presentation=True)
        for broken in ({"outro": {"links": PRESENTATION["outro"]["links"]}},
                       {"outro": {"headline": ["Heading"]}},
                       {"outro": {"headline": ["Heading"], "links": []}}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_episode(self._episode(broken), set(),
                                 require_integrated_presentation=True)
        validate_episode(self._episode(kind="SUMMARY"), {"E0001"},
                         require_integrated_presentation=True)
        # The frozen-episode build path deliberately checks the sidecar AFTER
        # the ordinary schema, as renderer.build_episode does today.
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "presentation.json"
            sidecar.write_text(json.dumps(PRESENTATION))
            frozen = self._episode()
            validate_episode(frozen, set())
            merged = {**frozen.get("presentation", {}), **json_load(sidecar)}
            validate_presentation(merged, True)

            class StopBeforeRendering(Exception):
                pass

            with patch("source2reel.renderer.render_scene", side_effect=StopBeforeRendering) as render:
                with self.assertRaises(StopBeforeRendering):
                    build_episode(Path(tmp), Path(tmp), frozen, {"evidence": []},
                                  cfg={"profile": {"name": "test"}})
            self.assertEqual(render.call_args.args[-1], PRESENTATION)

    def test_full_and_multipart_storyboard_both_require_integrated_outro(self):
        inventory = {"evidence": [{"ref": "E0001", "kind": "document",
                                  "relative_path": "overview.md", "evidence_role": "primary"}]}
        research = {"version": 1, "facts": [{"claim": "Documented subject",
                                                 "evidence_refs": ["E0001"],
                                                 "support": [{"evidence_ref": "E0001",
                                                              "text": "Documented subject"}]}], "assets": []}

        class EndingProvider:
            def __init__(self, *, full_error=None, bad_outline=False):
                self.full_error = full_error
                self.bad_outline = bad_outline
                self.requests = []

            def complete_json(self, system, user):
                request = json.loads(user)
                self.requests.append(request)
                mode = request.get("storyboard_mode")
                if mode is None:
                    if self.full_error:
                        raise self.full_error("oversized complete episode")
                    return {**IntegratedOutroTests._episode(PRESENTATION),
                            "summary": "Documented subject"}
                if mode == "outline":
                    outline = {"version": 1, "title": "Grounded result", "slug": "grounded-result",
                               "summary": "Documented subject", "scene_intents": [{
                                   "type": "OUTRO", "purpose": "Closing",
                                   "evidence_refs": [],
                               }]}
                    if not self.bad_outline:
                        outline["presentation"] = PRESENTATION
                    return outline
                return {"scenes": [IntegratedOutroTests._episode()["scenes"][0]]}

        for error in (None, OutputLimitExceeded, StructuredOutputError):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                project = _project(Path(tmp))
                provider = EndingProvider(full_error=error)
                ep = plan(provider, research, inventory, project, "WidgetEngine", INSTRUCTIONS,
                          max_retries=0)
                self.assertEqual(ep["presentation"], PRESENTATION)
                self.assertEqual([req.get("storyboard_mode") for req in provider.requests],
                                 [None] if error is None else [None, "outline", "scenes"])
                self.assertTrue(all(req["optional_instructions"] == INSTRUCTIONS
                                    for req in provider.requests))

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))

            class MissingFullPresentation(EndingProvider):
                def complete_json(self, system, user):
                    if "storyboard_mode" not in json.loads(user):
                        self.requests.append(json.loads(user))
                        return IntegratedOutroTests._episode()
                    return super().complete_json(system, user)

            provider = MissingFullPresentation()
            recovered = plan(provider, research, inventory, project,
                             "WidgetEngine", INSTRUCTIONS, max_retries=0)
            self.assertEqual(recovered["presentation"], PRESENTATION)
            self.assertEqual([req.get("storyboard_mode") for req in provider.requests],
                             [None, "outline", "scenes"])

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            with self.assertRaisesRegex(StructuredOutputError, "OUTRO requires"):
                plan(EndingProvider(full_error=OutputLimitExceeded, bad_outline=True),
                     research, inventory, project, "WidgetEngine", INSTRUCTIONS,
                     max_retries=0)
            self.assertFalse((project / "episode.json").exists())

    def test_factual_prompt_and_full_outline_contracts_explicitly_bound_claims(self):
        prompt = (ROOT / "prompts" / "storyboard.txt").read_text().casefold()
        for term in ("target audiences", "motivations", "module responsibilities",
                     "normal workflow stages", "causal relationships", "external-service claims",
                     "optional", "mandatory", "episode.presentation.outro"):
            with self.subTest(term=term):
                self.assertIn(term, prompt)
        ask = _make_ask({"facts": [], "assets": []}, [], [], "WidgetEngine", INSTRUCTIONS)
        self.assertEqual(ask["optional_instructions"], INSTRUCTIONS)
        self.assertIn("links", ask["output_contract"]["presentation"]["outro"])
        outline = _outline_payload(ask)
        self.assertEqual(outline["optional_instructions"], INSTRUCTIONS)
        self.assertEqual(outline["required_output"]["presentation"],
                         ask["output_contract"]["presentation"])


if __name__ == "__main__":
    unittest.main()
