"""Planning labels recover from rejected prose without changing fact authority."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel import planner
from source2reel.providers import StructuredOutputError
from source2reel.util import json_load

from test_planner_source_agnostic import ProfileProvider, project, source


class NoVerifier:
    def complete_json(self, _system, _user):
        raise AssertionError("Exact claims and editorial labels need no semantic verifier")


def ask_for(claims, *, visual=(), roles=None):
    inventory, research = source(claims, visual=visual, roles=roles)
    ask = planner._make_ask(research, planner._media_inventory(inventory),
                            inventory["evidence"], "Fixture", "")
    return inventory, research, ask


def outline_for(ask, summary, selections):
    facts = {fact["fact_id"]: fact for fact in ask["research"]["facts"]}
    intents = []
    for kind, purpose, selected in selections:
        refs = list(dict.fromkeys(ref for fact_id in selected
                                  for ref in facts[fact_id]["evidence_refs"]))
        intent = {"type": kind, "purpose": purpose,
                  "fact_ids": selected, "evidence_refs": refs}
        if kind == "PROJECT_EVIDENCE":
            intent["asset_ref"] = "E0090"
            intent["evidence_refs"].append("E0090")
        intents.append(intent)
    allowed = {entry["ref"] for entry in ask["evidence_index"]}
    return planner._normalize_outline(
        {"version": 1, "title": "Fixture", "slug": "fixture",
         "summary": summary, "scene_intents": intents}, allowed, ask,
    )


class MetadataCanonicalizationTests(unittest.TestCase):
    def test_supported_summary_and_purpose_are_preserved_without_model(self):
        claim = "A documented tool processes local observations."
        _, _, ask = ask_for([claim])
        original = outline_for(ask, claim, [("SUMMARY", claim, ["F0001"])])
        with tempfile.TemporaryDirectory() as tmp:
            fixed = planner._canonicalize_outline_metadata(
                NoVerifier(), original, ask, Path(tmp),
            )
        self.assertEqual(fixed, original)

    def test_semantically_supported_model_labels_remain_verbatim(self):
        _, _, ask = ask_for(["A tool processes local observations."])
        summary = "The tool processes observations locally."
        purpose = "Present the tool processing local observations."
        original = outline_for(ask, summary, [("SUMMARY", purpose, ["F0001"])])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "source2reel.planner.verify_claims", return_value={
                "outline-summary", "outline-s001",
            },
        ) as verify:
            fixed = planner._canonicalize_outline_metadata(
                NoVerifier(), original, ask, Path(tmp),
            )
        self.assertEqual(fixed, original)
        verify.assert_called_once()

    def test_rejection_only_changes_labels_and_uses_selected_claim_order(self):
        claims = [
            "The tool reads a local event stream.",
            "The tool groups events by recorded process.",
            "A later optional export saves a report.",
        ]
        _, _, ask = ask_for(claims, visual=(90,))
        original = outline_for(ask, "The tool proves every network cause.", [
            ("SUMMARY", "Infer all unseen behavior.", ["F0002", "F0001"]),
            ("PROJECT_EVIDENCE", "Identify hidden users.", ["F0001"]),
            ("SECTION_TITLE", "The tool guarantees all outcomes.", ["F0002"]),
        ])
        snapshot = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "source2reel.planner.verify_claims", return_value=set(),
        ) as verify:
            fixed = planner._canonicalize_outline_metadata(
                NoVerifier(), original, ask, Path(tmp),
            )
        verify.assert_called_once()  # The exact fallback never returns to the verifier.
        self.assertEqual(fixed["summary"], claims[1] + " " + claims[0])
        self.assertEqual([item["purpose"] for item in fixed["scene_intents"]],
                         [claims[1], claims[0]])
        self.assertEqual([item["id"] for item in fixed["scene_intents"]], ["s001", "s002"])
        self.assertNotIn(claims[2], fixed["summary"])
        self.assertEqual(original, snapshot)
        for returned, raw in zip(fixed["scene_intents"], original["scene_intents"]):
            self.assertEqual({key: value for key, value in returned.items()
                              if key != "purpose"},
                             {key: value for key, value in raw.items()
                              if key != "purpose"})
        with tempfile.TemporaryDirectory() as tmp, patch(
            "source2reel.planner.verify_claims",
            side_effect=AssertionError("canonical fallback must not be reverified"),
        ):
            self.assertEqual(planner._canonicalize_outline_metadata(
                NoVerifier(), fixed, ask, Path(tmp)), fixed)

    def test_complete_claims_fit_bound_and_long_purpose_becomes_editorial(self):
        claims = ["A" * 1499 + ".", "B" * 199 + ".", "C" * 79 + "."]
        _, _, ask = ask_for(claims)
        raw = outline_for(ask, "Unsafe summary", [
            ("SUMMARY", "Unsupported purpose", ["F0001"]),
            ("SUMMARY", "Unsupported purpose", ["F0002"]),
            ("SUMMARY", "Unsupported purpose", ["F0003", "F0001"]),
        ])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "source2reel.planner.verify_claims", return_value=set(),
        ) as verify:
            fixed = planner._canonicalize_outline_metadata(NoVerifier(), raw, ask, Path(tmp))
        self.assertEqual(fixed["summary"], claims[0] + " " + claims[2])
        self.assertLessEqual(len(fixed["summary"]), 1600)
        self.assertEqual(fixed["scene_intents"][0]["purpose"], "Plan a summary scene")
        self.assertEqual(fixed["scene_intents"][1]["purpose"], "Plan a summary scene")
        self.assertEqual(fixed["scene_intents"][2]["purpose"], claims[2])
        self.assertEqual(planner._outline_factual_text("Plan a summary scene"), "")
        self.assertEqual(planner._outline_factual_text("Plan a summary scene"), "")
        verify.assert_called_once()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(planner._canonicalize_outline_metadata(
                NoVerifier(), fixed, ask, Path(tmp)), fixed)

    def test_overlong_labels_recover_and_no_complete_claim_fails_clearly(self):
        claim = "A documented source records events."
        _, _, ask = ask_for([claim])
        raw = outline_for(ask, "U" * 1601, [
            ("SUMMARY", "Unsupported " * 18, ["F0001"]),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            fixed = planner._canonicalize_outline_metadata(NoVerifier(), raw, ask, Path(tmp))
        self.assertEqual(fixed["summary"], claim)
        self.assertEqual(fixed["scene_intents"][0]["purpose"], claim)

        _, _, too_long_ask = ask_for(["L" * 1601])
        too_long = outline_for(too_long_ask, "Unsupported summary", [
            ("SUMMARY", "Closing", ["F0001"]),
        ])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            StructuredOutputError, "no selected grounded fact claim fits",
        ):
            planner._canonicalize_outline_metadata(NoVerifier(), too_long, too_long_ask,
                                                   Path(tmp))

    def test_full_path_summary_uses_selected_facts_without_narration_repair(self):
        inventory, research, ask = ask_for([
            "The tool reads a local event stream.",
            "An unrelated example shows a different input.",
        ])
        selected = ask["research"]["facts"][0]
        episode = {"version": 1, "title": "Tool", "summary": "The tool predicts the future.",
                   "scenes": [{"id": "s001", "type": "SUMMARY", "title": "Events",
                               "narration": selected["claim"],
                               "fact_ids": [selected["fact_id"]],
                               "evidence_refs": selected["evidence_refs"]}]}

        class Direct:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                if "checks" in request:
                    return {"decisions": [{"id": item["id"], "supported": False,
                                           "propositions": []} for item in request["checks"]]}
                return copy.deepcopy(episode)

        with tempfile.TemporaryDirectory() as tmp:
            provider = Direct()
            fixed = planner._complete_episode(
                provider, "storyboard", ask,
                {entry["ref"] for entry in inventory["evidence"]}, 0,
                project_dir=Path(tmp),
            )
        self.assertEqual(fixed["summary"], selected["claim"])
        self.assertEqual(fixed["scenes"][0]["narration"], selected["claim"])
        self.assertEqual(len([call for call in provider.calls if "checks" not in call]), 1)


class PhysicalShapedRecoveryTests(unittest.TestCase):
    def test_windows_telemetry_multipart_canonicalizes_checkpoint_and_continues(self):
        claims = [
            "Windows Telemetry Inspector is a passive Windows diagnostics tool.",
            "ETW supplies process-scoped network observations to Windows Telemetry Inspector.",
            "DNS and task relationships are best-effort correlations, not proven causality.",
        ]
        inventory, research = source(claims)
        rejected = (
            "Windows Telemetry Inspector proves why every process opened its network "
            "connection using ETW and DNS."
        )

        class WindowsProvider(ProfileProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "checks" in request:
                    self.calls.append(request)
                    return {"decisions": [
                        {"id": check["id"], "supported": False, "propositions": []}
                        for check in request["checks"]
                    ]}
                if request.get("storyboard_mode") == "outline":
                    self.calls.append(request)
                    facts = request["research"]["facts"]
                    return {
                        "version": 1, "title": "Windows Telemetry Inspector",
                        "slug": "windows-telemetry-inspector", "summary": rejected,
                        "scene_intents": [
                            {"type": "SUMMARY", "purpose": "It secretly uploads all captured data.",
                             "fact_ids": ["F0001"], "evidence_refs": ["E0001"],
                             "asset_ref": {"irrelevant": "invalid"}},
                            {"type": "SUMMARY", "purpose": facts[1]["claim"],
                             "fact_ids": ["F0002"], "evidence_refs": ["E0002"]},
                            {"type": "SUMMARY", "purpose": facts[2]["claim"],
                             "fact_ids": ["F0003"], "evidence_refs": ["E0003"]},
                        ],
                    }
                return super().complete_json(system, user)

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            provider = WindowsProvider()
            episode = planner.plan(provider, research, inventory, path,
                                   "Windows Telemetry Inspector", "", max_retries=2)
            saved = json_load(path / "manifests" / "storyboard-parts" / "outline.json")
            canonical = saved["result"]
            self.assertEqual(canonical["summary"], " ".join(claims))
            self.assertNotIn(rejected, json.dumps(canonical))
            self.assertEqual(canonical["scene_intents"][0]["purpose"], claims[0])
            self.assertEqual([item["fact_ids"] for item in canonical["scene_intents"]],
                             [["F0001"], ["F0002"], ["F0003"]])
            self.assertEqual([item["evidence_refs"] for item in canonical["scene_intents"]],
                             [["E0001"], ["E0002"], ["E0003"]])
            self.assertTrue(all("asset_ref" not in item for item in canonical["scene_intents"]))
            self.assertEqual([scene["narration"] for scene in episode["scenes"]], claims)
            self.assertTrue(any(call.get("storyboard_mode") == "scenes" for call in provider.calls))
            self.assertEqual(sum(call.get("storyboard_mode") == "outline" for call in provider.calls), 1)
            before = len(provider.calls)
            self.assertEqual(planner.plan(provider, research, inventory, path,
                                          "Windows Telemetry Inspector", "", max_retries=2), episode)
            self.assertEqual(len(provider.calls), before)

    def test_large_self_referential_source_selects_original_facts_for_fallback(self):
        primary = [
            "The engine indexes original project sources.",
            "The engine plans scenes from grounded research facts.",
            "The engine renders planned scenes locally.",
        ]
        examples = [f"Embedded example {i} describes a different project." for i in range(4, 19)]
        generated = [f"Generated output {i} records an older run." for i in range(19, 31)]
        roles = {i: "embedded_reference" for i in range(4, 19)}
        roles.update({i: "generated_artifact" for i in range(19, 31)})
        inventory, research = source(primary + examples + generated, roles=roles)

        class RejectedSummary(ProfileProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                result = super().complete_json(system, user)
                if request.get("storyboard_mode") == "outline":
                    result["summary"] = "The engine proves what its generated examples do."
                return result

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            episode = planner.plan(RejectedSummary(), research, inventory, path,
                                   "The engine", "", max_retries=0)
            outline = json_load(path / "manifests" / "storyboard-parts" / "outline.json")["result"]
            self.assertEqual(outline["summary"], primary[0] + " " + primary[1])
            self.assertEqual([scene["fact_ids"] for scene in episode["scenes"]],
                             [["F0001"], ["F0002"]])

    def test_unsupported_narration_is_replaced_after_metadata_repair(self):
        inventory, research = source(["The tool records local events.",
                                      "The tool reports observed processes."])

        class BadNarration(ProfileProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if "checks" in request:
                    self.calls.append(request)
                    return {"decisions": [
                        {"id": check["id"], "supported": False, "propositions": []}
                        for check in request["checks"]
                    ]}
                result = super().complete_json(system, user)
                if request.get("storyboard_mode") == "outline":
                    result["summary"] = "The tool predicts every future event."
                    result["scene_intents"][0]["purpose"] = "The tool can read thoughts."
                if request.get("storyboard_mode") == "scenes":
                    result["scenes"][0]["narration"] = "The tool predicts every future event."
                return result

        with tempfile.TemporaryDirectory() as tmp:
            path = project(Path(tmp))
            episode = planner.plan(BadNarration(), research, inventory, path, "The tool", "",
                                   max_retries=0)
            outline = json_load(path / "manifests" / "storyboard-parts" / "outline.json")["result"]
            self.assertEqual(outline["summary"], " ".join(f["claim"] for f in research["facts"]))
            self.assertEqual(outline["scene_intents"][0]["purpose"], research["facts"][0]["claim"])
            self.assertEqual(episode["scenes"][0]["narration"], research["facts"][0]["claim"])
            self.assertNotIn("predicts every future event", json.dumps(episode))
