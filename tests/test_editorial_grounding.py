"""Final storyboard claims, links and explicit closing text stay scoped."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from source2reel.planner import (
    _authoritative_resource_urls, _complete_episode, _final_narration_suffix,
    _final_requests_fit, _make_ask, _normalize_outline, _normalize_scene_part,
    _validate_outline_grounding,
    _scene_part_payload, plan,
)
from source2reel.chunking import fits_context
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.schema import validate_episode
from source2reel.util import json_load


SUFFIX = "Final exact sentence."
SOURCE_URL = "https://github.com/example/widget"


def _evidence():
    claims = [
        "The normal workflow ingests sources, researches evidence, then plans output.",
        "Optional repair clears outdated caches during maintenance.",
        "The production profile selects a theme and voice.",
        "The normal workflow renders the planned output with local tools.",
        "The configuration uses [profile] with name = widget-series.",
    ]
    inventory = {"evidence": [
        {"ref": f"E{i:04d}", "kind": "document", "relative_path": f"original-{i}.md",
         "evidence_role": "primary"} for i in range(1, 6)
    ]}
    research = {"version": 1, "facts": [
        {"claim": claim, "evidence_refs": [f"E{i:04d}"],
         "support": [{"evidence_ref": f"E{i:04d}", "text": claim}],
         "phase": "final", "confidence": "high"}
        for i, claim in enumerate(claims, 1)
    ], "assets": []}
    return inventory, research


def _project(root: Path, source: str | None = SOURCE_URL) -> Path:
    project = root / "projects" / "widget"
    (root / "prompts").mkdir()
    (root / "prompts" / "storyboard.txt").write_text("storyboard")
    (project / "sources").mkdir(parents=True)
    record = {"kind": "github", "source": source,
              "repo_url": SOURCE_URL + ".git"} if source else {"kind": "local", "source": "/tmp/widget"}
    (project / "sources" / "source.json").write_text(json.dumps({
        "version": 1, "sources": [{**record, "namespace": "source-01"}]
    }))
    return project


def _ask(instructions: str = "Explain the normal workflow.", urls=None):
    inventory, research = _evidence()
    return _make_ask(research, [], inventory["evidence"], "WidgetEngine", instructions,
                     [SOURCE_URL] if urls is None else urls)


class GroundedProvider:
    """Realize only fields and facts present in each request contract."""

    def __init__(self, multipart=True):
        self.multipart = multipart
        self.calls = []

    def complete_json(self, system, user):
        request = json.loads(user)
        self.calls.append(request)
        mode = request.get("storyboard_mode")
        if mode is None:
            if self.multipart:
                raise OutputLimitExceeded("storyboard exceeds one response")
            fact = request["research"]["facts"][0]
            narration = fact["claim"]
            if request["required_narration_suffix"]:
                narration += " " + request["required_narration_suffix"]
            return {"version": 1, "title": "Widget", "scenes": [{
                "id": "s001", "type": "SUMMARY", "narration": narration,
                "fact_ids": [fact["fact_id"]], "evidence_refs": fact["evidence_refs"],
            }]}
        if mode == "outline":
            facts = request["research"]["facts"]
            ids = [fact["fact_id"] for fact in facts]
            refs = [fact["evidence_refs"] for fact in facts]
            intents = [
                {"type": "HERO", "purpose": facts[0]["claim"], "fact_ids": [ids[0]],
                 "evidence_refs": refs[0]},
                {"type": "DATA_FLOW", "purpose": facts[0]["claim"], "fact_ids": [ids[0], ids[3]],
                 "evidence_refs": refs[0] + refs[3]},
                {"type": "CODE", "purpose": facts[2]["claim"], "fact_ids": [ids[2]],
                 "evidence_refs": refs[2]},
                {"type": "SUMMARY", "purpose": facts[1]["claim"], "fact_ids": [ids[1]],
                 "evidence_refs": refs[1]},
            ]
            if ("asset_ref" in request["required_output"]["scene_intents"][0] and
                    intents[0]["type"] in request["scene_type_requirements"]["asset_ref_required_types"]):
                intents[0]["asset_ref"] = refs[0][0]
            output = {"version": 1, "title": "Widget", "slug": "widget",
                      "summary": facts[0]["claim"], "scene_intents": intents}
            if request["authoritative_resource_urls"]:
                intents.append({"type": "OUTRO", "purpose": "Closing", "fact_ids": [],
                                "evidence_refs": []})
                output["presentation"] = {"outro": {
                    "headline": ["Documented project"],
                    "links": [{"label": "Source", "url": [request["authoritative_resource_urls"][0]]}],
                }}
            return output
        if mode == "scenes":
            facts = {fact["fact_id"]: fact for fact in request["research"]["facts"]}
            scenes = []
            for template in request["required_output"]["scenes"]:
                scene = copy.deepcopy(template)
                scene["narration"] = " ".join(facts[fact_id]["claim"] for fact_id in scene["fact_ids"]) or "Closing."
                scene["title"] = scene["narration"].split(".")[0]
                if "diagram" in scene:
                    scene["diagram"]["nodes"] = [facts[fact_id]["claim"] for fact_id in scene["fact_ids"]]
                scenes.append(scene)
            if request["contains_final_scene"] and request["required_narration_suffix"]:
                scenes[-1]["narration"] += " " + request["required_narration_suffix"]
            return {"scenes": scenes}
        raise AssertionError("Unknown storyboard contract")


class EditorialGroundingTests(unittest.TestCase):
    def test_outline_summary_and_purpose_must_follow_their_scoped_facts(self):
        definition = "WidgetEngine is a reusable evidence-first explainer engine."
        rendering = "The normal workflow renders planned output with local tools."
        facts = [definition, rendering]
        evidence = [{"ref": f"E{i:04d}", "evidence_role": "primary"}
                    for i in (1, 2)]
        research = {"facts": [{"claim": claim, "evidence_refs": [f"E{i:04d}"],
                               "support": [{"evidence_ref": f"E{i:04d}", "text": claim}]}
                              for i, claim in enumerate(facts, 1)], "assets": []}
        ask = _make_ask(research, [], evidence, "WidgetEngine", "Explain what it does.")

        class NoVerifier:
            def complete_json(self, _system, _user):
                raise AssertionError("These exact or obviously unsupported checks need no model")

        def outline(summary, purpose, ids):
            return _normalize_outline({"version": 1, "title": "WidgetEngine", "slug": "widget",
                                       "summary": summary, "scene_intents": [{
                                           "type": "SUMMARY", "purpose": purpose,
                                           "fact_ids": ids, "evidence_refs": ["E0001"],
                                       }]}, {"E0001", "E0002"}, ask)

        with tempfile.TemporaryDirectory() as tmp:
            provider, root = NoVerifier(), Path(tmp)
            clean = outline(definition,
                            "Introduce WidgetEngine as a reusable evidence-first explainer engine.",
                            ["F0001"])
            _validate_outline_grounding(provider, clean, ask, root)
            for summary, purpose, ids, expected in (
                (definition, "Turn project sources into finished technical documentaries.",
                 ["F0001"], "outline-s001"),
                (definition, "Show source → evidence → AI enrichment → rendering → output.",
                 ["F0001", "F0002"], "outline-s001"),
                ("WidgetEngine uses AI enrichment.", definition, ["F0001"],
                 "outline-summary"),
                (definition, rendering, ["F0001"], "outline-s001"),
            ):
                with self.subTest(expected=expected, purpose=purpose, summary=summary):
                    with self.assertRaisesRegex(StructuredOutputError, expected):
                        _validate_outline_grounding(provider, outline(summary, purpose, ids),
                                                    ask, root)

            scene = {"id": "s001", "type": "SUMMARY", "narration":
                     "WidgetEngine guarantees fully autonomous video production.",
                     "fact_ids": ["F0001"], "evidence_refs": ["E0001"]}
            with self.assertRaisesRegex(StructuredOutputError,
                                        "narration introduces an unsupported factual proposition"):
                _normalize_scene_part({"scenes": [scene]}, clean["scene_intents"],
                                      {"E0001"}, clean, ask, provider, root)

    def test_fact_ids_follow_final_research_order_and_are_stable(self):
        inventory, research = _evidence()
        first = _make_ask(research, [], inventory["evidence"], "Widget", "")
        second = _make_ask(research, [], inventory["evidence"], "Widget", "")
        self.assertEqual(first["research"], second["research"])
        self.assertEqual([fact["fact_id"] for fact in first["research"]["facts"]],
                         [f"F{i:04d}" for i in range(1, 6)])
        self.assertNotIn("fact_id", research["facts"][0])

    def test_final_request_budget_includes_possible_multipart_outline(self):
        ask = _ask()
        self.assertTrue(fits_context("storyboard", json.dumps(ask, ensure_ascii=False),
                                     1500, 0, 0))
        self.assertFalse(_final_requests_fit("storyboard", ask, 1500, 0, 0))
        self.assertTrue(_final_requests_fit("storyboard", ask, 1900, 0, 0))

    def test_selected_fact_scope_excludes_unrelated_setup_profile_and_config(self):
        ask = _ask()
        outline = _normalize_outline({"version": 1, "title": "Widget", "slug": "widget",
                                      "summary": "Normal flow", "scene_intents": [{
                                          "type": "CODE", "purpose": "Normal workflow",
                                          "fact_ids": ["F0001"], "evidence_refs": ["E0001"],
                                      }]}, {f"E{i:04d}" for i in range(1, 6)}, ask)
        part = _scene_part_payload(ask, outline, outline["scene_intents"], 1, 1, "scope")
        self.assertEqual([fact["fact_id"] for fact in part["research"]["facts"]], ["F0001"])
        self.assertEqual([e["ref"] for e in part["evidence_index"]], ["E0001"])
        self.assertEqual(part["required_output"]["scenes"][0]["fact_ids"], ["F0001"])
        rule = part["fact_selection_requirement"]
        for literal in ("setup", "maintenance", "profile", "commands", "config"):
            self.assertIn(literal, rule)
        with self.assertRaisesRegex(StructuredOutputError, "fact_ids"):
            _normalize_outline({"version": 1, "title": "Widget", "slug": "widget",
                                "summary": "Normal flow", "scene_intents": [{
                                    "type": "CODE", "purpose": "Normal workflow",
                                    "fact_ids": ["F9999"], "evidence_refs": ["E0001"],
                                }]}, {f"E{i:04d}" for i in range(1, 6)}, ask)
        with self.assertRaisesRegex(StructuredOutputError, "fact_ids"):
            _normalize_outline({"version": 1, "title": "Widget", "slug": "widget",
                                "summary": "Normal flow", "scene_intents": [{
                                    "type": "CODE", "purpose": "Normal workflow",
                                    "evidence_refs": ["E0001"],
                                }]}, {f"E{i:04d}" for i in range(1, 6)}, ask)
        for leak in ("E0002", "E0003", "E0005"):
            with self.subTest(leak=leak), self.assertRaisesRegex(StructuredOutputError, "evidence_refs"):
                _normalize_scene_part({"scenes": [{
                    "id": "s001", "type": "CODE", "title": "Workflow", "narration": "Unsupported.",
                    "fact_ids": ["F0001"], "evidence_refs": ["E0001", leak],
                }]}, outline["scene_intents"], {f"E{i:04d}" for i in range(1, 6)}, outline, ask)
        with self.assertRaisesRegex(StructuredOutputError, "differ from fixed outline"):
            _normalize_scene_part({"scenes": [{
                "id": "s001", "type": "CODE", "title": "Workflow", "narration": "Optional repair.",
                "fact_ids": ["F0002"], "evidence_refs": ["E0002"],
            }]}, outline["scene_intents"], {f"E{i:04d}" for i in range(1, 6)}, outline, ask)

    def test_both_quote_styles_and_ambiguous_or_absent_suffix(self):
        for quoted in ('“Final exact sentence.”', '"Final exact sentence."'):
            self.assertEqual(_final_narration_suffix("End with: " + quoted), SUFFIX)
        for instructions in ("Explain a widget.", "End with: no quotes",
                             'Do not end with: "Unwanted."',
                             'End with: "First." End with: "Second."'):
            self.assertIsNone(_final_narration_suffix(instructions))

    def test_fast_path_retries_when_suffix_is_only_in_headline(self):
        ask = _ask('End with: "Final exact sentence."', [SOURCE_URL])
        class RepairOnFeedback:
            calls = 0
            def complete_json(self, system, user):
                self.calls += 1
                payload = json.loads(user)
                return {"version": 1, "title": "Widget", "presentation": {"outro": {
                    "headline": [SUFFIX], "links": [{"label": "Project", "url": [SOURCE_URL]}]}},
                    "scenes": [{"id": "s001", "type": "OUTRO", "narration": (
                        "Closing. " + SUFFIX if "validation_feedback" in payload else "Closing."
                    ), "evidence_refs": []}]}
        provider = RepairOnFeedback()
        episode = _complete_episode(provider, "storyboard", ask, {"E0001"}, 1)
        self.assertEqual(provider.calls, 2)
        self.assertTrue(episode["scenes"][-1]["narration"].endswith(SUFFIX))

    def test_no_suffix_directive_keeps_fast_path_one_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp), None)
            inventory, research = _evidence()
            provider = GroundedProvider(multipart=False)
            ep = plan(provider, research, inventory, project, "WidgetEngine", "Explain the documented pipeline.")
            self.assertEqual(ep["scenes"][-1]["narration"], research["facts"][0]["claim"])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(provider.calls[0]["authoritative_resource_urls"], [])

    def test_integrated_outro_requires_single_final_scene_and_own_presentation(self):
        outro = {"id": "s001", "type": "OUTRO", "narration": "Close.", "evidence_refs": []}
        summary = {"id": "s002", "type": "SUMMARY", "narration": "Grounded.", "evidence_refs": ["E0001"]}
        presentation = {"outro": {"headline": ["End"],
                                  "links": [{"label": "Source", "url": [SOURCE_URL]}]}}
        cases = [
            ({"scenes": [outro]}, "OUTRO requires presentation.outro"),
            ({"scenes": [summary], "presentation": presentation}, "presentation.outro requires"),
            ({"scenes": [outro, summary], "presentation": presentation}, "OUTRO must be the final"),
            ({"scenes": [outro, {**outro, "id": "s002"}], "presentation": presentation}, "only one OUTRO"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_episode({"version": 1, "title": "Widget", **data}, {"E0001"},
                                 require_integrated_presentation=True)
        validate_episode({"version": 1, "title": "Widget", "scenes": [summary, {**outro, "id": "s003"}],
                          "presentation": presentation}, {"E0001"}, require_integrated_presentation=True)
        validate_episode({"version": 1, "title": "Widget", "scenes": [summary]}, {"E0001"},
                         require_integrated_presentation=True)
        validate_episode({"version": 1, "title": "Frozen", "scenes": [outro]}, set())

    def test_authoritative_github_urls_reject_invented_owner_and_deep_link(self):
        from source2reel.planner import _validate_resource_links
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            urls = _authoritative_resource_urls(project)
            self.assertIn(SOURCE_URL, urls)
            ask = _ask(urls=urls)
            for bad in ("https://github.com/fake/widget",
                        SOURCE_URL + "/blob/main/fake.md"):
                with self.subTest(url=bad), self.assertRaisesRegex(ValueError, "authoritative"):
                    _validate_resource_links({"outro": {"links": [{"url": [bad]}]}}, ask)
            _validate_resource_links({"outro": {"links": [{"url": [SOURCE_URL]}]}}, ask)

            class InventedLink:
                def complete_json(self, system, user):
                    return {"version": 1, "title": "Widget", "presentation": {"outro": {
                        "headline": ["Source"],
                        "links": [{"label": "Source", "url": ["https://github.com/fake/widget"]}]}},
                        "scenes": [{"id": "s001", "type": "OUTRO", "narration": "Closing.",
                                    "evidence_refs": []}]}
            with self.assertRaisesRegex(ValueError, "authoritative"):
                _complete_episode(InventedLink(), "storyboard", ask, {"E0001"}, 0)
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp), SOURCE_URL + ".git")
            self.assertIn(SOURCE_URL, _authoritative_resource_urls(project))

    def test_explicit_website_pages_are_allowed_but_failed_pages_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp), None)
            source = project / "sources" / "source.json"
            source.write_text(json.dumps({"version": 1, "sources": [{
                "kind": "website", "source": "https://example.test/widget",
                "namespace": "source-01",
            }]}))
            website = project / "sources" / "source-01"
            website.mkdir()
            (website / "website-manifest.json").write_text(json.dumps({
                "root_url": "https://example.test/widget",
                "pages": [{"url": "https://example.test/widget/docs", "html": "web/page-001.html",
                           "text": "web/page-001.txt"},
                          {"url": "https://example.test/widget/missing", "error": "timeout"}],
            }))
            urls = _authoritative_resource_urls(project)
            self.assertEqual(urls, ["https://example.test/widget", "https://example.test/widget/docs"])

    def test_contract_following_multipart_pipeline_validates_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            inventory, research = _evidence()
            provider = GroundedProvider()
            instructions = 'Explain normal flow, optional repair and the profile. End with: “Final exact sentence.”'
            episode = plan(provider, research, inventory, project, "WidgetEngine", instructions,
                           max_retries=0)
            self.assertEqual([s["type"] for s in episode["scenes"]],
                             ["HERO", "DATA_FLOW", "CODE", "SUMMARY", "OUTRO"])
            self.assertEqual(episode["scenes"][-1]["narration"], "Closing. " + SUFFIX)
            self.assertEqual(episode["presentation"]["outro"]["links"][0]["url"], [SOURCE_URL])
            self.assertEqual(episode["scenes"][1]["fact_ids"], ["F0001", "F0004"])
            self.assertEqual(episode["scenes"][2]["narration"], research["facts"][2]["claim"])
            self.assertEqual(episode["scenes"][3]["narration"], research["facts"][1]["claim"])
            self.assertNotIn("name = widget-series", episode["scenes"][2]["narration"])
            validate_episode(episode, {e["ref"] for e in inventory["evidence"]},
                             require_integrated_presentation=True)
            requests = [call for call in provider.calls if call.get("storyboard_mode") == "scenes"]
            self.assertGreaterEqual(len(requests), 3)
            self.assertEqual([f["fact_id"] for f in requests[1]["research"]["facts"]],
                             ["F0002", "F0003"])
            self.assertIsNone(requests[0]["required_narration_suffix"])
            self.assertEqual(requests[-1]["required_narration_suffix"], SUFFIX)
            self.assertEqual(json_load(project / "episode.json"), episode)
            calls = len(provider.calls)
            self.assertEqual(plan(provider, research, inventory, project, "WidgetEngine",
                                  instructions, max_retries=0), episode)
            self.assertEqual(len(provider.calls), calls)

    def test_local_multipart_source_can_finish_with_summary_without_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp), None)
            inventory, research = _evidence()
            provider = GroundedProvider()
            episode = plan(provider, research, inventory, project, "WidgetEngine",
                           "Explain the normal workflow and optional maintenance.", max_retries=0)
            self.assertEqual(episode["scenes"][-1]["type"], "SUMMARY")
            self.assertNotIn("outro", episode.get("presentation", {}))
            outline = next(call for call in provider.calls if call.get("storyboard_mode") == "outline")
            self.assertEqual(outline["authoritative_resource_urls"], [])

    def test_multipart_rejects_missing_final_narration_and_reuses_prior_parts(self):
        class DropsSuffix(GroundedProvider):
            drop = True

            def complete_json(self, system, user):
                request = json.loads(user)
                result = super().complete_json(system, user)
                if self.drop and request.get("storyboard_mode") == "scenes" and request["contains_final_scene"]:
                    result["scenes"][-1]["narration"] = "Closing without the required suffix."
                    self.assert_suffix_in_request = request["required_narration_suffix"]
                return result

        with tempfile.TemporaryDirectory() as tmp:
            project = _project(Path(tmp))
            inventory, research = _evidence()
            provider = DropsSuffix()
            instructions = 'Explain WidgetEngine. End with: "Final exact sentence."'
            with self.assertRaisesRegex(RuntimeError, "Last scene narration must end exactly"):
                plan(provider, research, inventory, project, "WidgetEngine", instructions,
                     max_retries=0)
            self.assertEqual(provider.assert_suffix_in_request, SUFFIX)
            self.assertFalse((project / "episode.json").exists())
            prior_calls = len(provider.calls)
            provider.drop = False
            episode = plan(provider, research, inventory, project, "WidgetEngine",
                           instructions, max_retries=0)
            self.assertTrue(episode["scenes"][-1]["narration"].endswith(SUFFIX))
            self.assertEqual(len(provider.calls), prior_calls + 1)


if __name__ == "__main__":
    unittest.main()
