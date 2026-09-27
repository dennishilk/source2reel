"""Requested story concepts need grounded, nonframing scene selections."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner
from source2reel.providers import OutputLimitExceeded
from source2reel.util import json_dump, json_load

from test_planner_source_agnostic import source


INSTRUCTIONS = (
    "Explain what Windows Telemetry Inspector is, why it exists, and how its "
    "architecture turns Windows ETW network observations into useful local diagnostics. "
    "Use only repository evidence and do not invent capabilities. Clearly distinguish "
    "passive observation and correlation from blocking, interception, or proven causality."
)
CLAIMS = [
    "Windows Telemetry Inspector is a passive Windows network diagnostics tool.",
    "Windows Telemetry Inspector exists to make process and service network activity "
    "inspectable locally without changing traffic.",
    "ETW network observations enter the core capture workflow, which turns them into "
    "bounded process state and local GUI diagnostics.",
    "Passive observation and correlation are not blocking, interception, or proven "
    "causality. DNS and task relationships are best-effort correlation.",
]
TITLE = "Windows Telemetry Inspector"


def scope(claims=CLAIMS, instructions=INSTRUCTIONS, roles=None):
    inventory, research = source(claims, roles=roles)
    ask = planner._make_ask(research, [], inventory["evidence"], TITLE, instructions,
                            ["https://example.test/docs"])
    return ask, {item["ref"] for item in inventory["evidence"]}


def intent(ask, number, kind="SUMMARY"):
    fact = ask["research"]["facts"][number - 1]
    return {"type": kind, "purpose": fact["claim"],
            "fact_ids": [fact["fact_id"]], "evidence_refs": fact["evidence_refs"]}


def outline(ask, numbers, kinds=None):
    kinds = kinds or ["SUMMARY"] * len(numbers)
    return {"version": 1, "title": TITLE, "slug": "windows-telemetry-inspector",
            "summary": ask["research"]["facts"][numbers[0] - 1]["claim"],
            "scene_intents": [intent(ask, number, kind) for number, kind in zip(numbers, kinds)],
            **({"presentation": {"outro": {"headline": ["Documented project"],
                                   "links": [{"label": "Docs", "url":
                                              ["https://example.test/docs"]}]},
                                  "scene_titles": {f"s{len(numbers):03d}": "The source"}}}
               if "OUTRO" in kinds else {})}


def episode_from_outline(raw):
    scenes = []
    for index, original in enumerate(raw["scene_intents"], 1):
        scene = {"id": f"s{index:03d}", "type": original["type"],
                 "title": "Source fact", "narration": original["purpose"],
                 "fact_ids": original["fact_ids"],
                 "evidence_refs": original["evidence_refs"]}
        if original["type"] in {"DATA_FLOW", "ARCHITECTURE_DIAGRAM"}:
            scene["diagram"] = {"nodes": ["Observation", "Correlation"]}
        scenes.append(scene)
    return {**{key: value for key, value in raw.items() if key != "scene_intents"},
            "scenes": scenes}


class RepeatingProvider:
    def __init__(self, ask, *, full=False):
        self.ask = ask
        self.full = full
        self.requests = []
        self.bad = outline(ask, [4, 4, 4],
                           ["SECTION_TITLE", "SUMMARY", "OUTRO"])

    def complete_json(self, _system, user):
        request = json.loads(user)
        self.requests.append(request)
        if "checks" in request:
            return {"decisions": [{"id": check["id"], "supported": True,
                                   "propositions": [{"text": text, "support_indices": [0]}
                                                    for text in check["required_propositions"]]}
                                  for check in request["checks"]]}
        if request.get("storyboard_mode") == "outline":
            return copy.deepcopy(self.bad)
        if request.get("storyboard_mode") == "scenes":
            facts = {f["fact_id"]: f for f in request["research"]["facts"]}
            scenes = copy.deepcopy(request["required_output"]["scenes"])
            for scene in scenes:
                scene["narration"] = facts[scene["fact_ids"][0]]["claim"]
                if scene["type"] in {"DATA_FLOW", "ARCHITECTURE_DIAGRAM"}:
                    scene["diagram"] = {"nodes": ["Observation", "Correlation"]}
            return {"scenes": scenes}
        if self.full:
            return episode_from_outline(self.bad)
        raise OutputLimitExceeded("exercise multipart")


class RequestedStoryCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ask, self.allowed = scope()
        self.assertEqual(self.ask["requested_topic_fact_ids"], {
            "overview": ["F0001"], "purpose": ["F0002"],
            "workflow": ["F0003"], "distinction-1": ["F0004"],
        })

    def test_physical_repetition_misses_three_topics_and_feedback_is_targeted(self):
        provider = RepeatingProvider(self.ask)
        with self.assertRaises(planner._RequestedCoverageError) as caught:
            planner._normalize_outline(provider.bad, self.allowed, self.ask)
        self.assertEqual(list(caught.exception.missing),
                         ["overview", "purpose", "workflow"])
        feedback = planner._retry_feedback(caught.exception)
        for topic, fact_id in (("overview", "F0001"), ("purpose", "F0002"),
                               ("workflow", "F0003")):
            self.assertIn(f"{topic} -> choose one of {fact_id}", feedback)
        self.assertNotIn("distinction-1 ->", feedback)
        self.assertNotIn("rewrite summary", feedback.lower())
        self.assertEqual(planner._outline_payload(self.ask)["requested_topic_fact_ids"],
                         self.ask["requested_topic_fact_ids"])

    def test_one_two_and_many_content_scenes_repeating_a_fact_fail(self):
        for count in (1, 2, 5):
            with self.subTest(count=count):
                with self.assertRaises(planner._RequestedCoverageError) as caught:
                    planner._normalize_outline(outline(self.ask, [4] * count),
                                               self.allowed, self.ask)
                self.assertEqual(set(caught.exception.missing),
                                 {"overview", "purpose", "workflow"})

    def test_factual_title_counts_as_content_and_disjoint_content_can_cover(self):
        bad = outline(self.ask, [1, 4, 2], ["SECTION_TITLE", "SUMMARY", "OUTRO"])
        with self.assertRaises(planner._RequestedCoverageError) as caught:
            planner._normalize_outline(bad, self.allowed, self.ask)
        self.assertEqual(set(caught.exception.missing), {"purpose", "workflow"})
        bad["scene_intents"][0].update({"fact_ids": [], "evidence_refs": [],
                                        "purpose": "Opening"})
        with self.assertRaises(planner._RequestedCoverageError) as caught:
            planner._normalize_outline(bad, self.allowed, self.ask)
        self.assertEqual(set(caught.exception.missing), {"overview", "purpose", "workflow"})
        good = outline(self.ask, [1, 2, 3, 4])
        self.assertEqual(planner._missing_story_topics(
            planner._normalize_outline(good, self.allowed, self.ask)["scene_intents"],
            self.ask), {})

    def test_two_content_scenes_can_cover_four_topics_with_selected_facts(self):
        raw = outline(self.ask, [1, 3])
        raw["scene_intents"][0]["fact_ids"].append("F0002")
        raw["scene_intents"][0]["evidence_refs"].append("E0002")
        raw["scene_intents"][1]["fact_ids"].append("F0004")
        raw["scene_intents"][1]["evidence_refs"].append("E0004")
        self.assertEqual(len(planner._normalize_outline(raw, self.allowed, self.ask)
                             ["scene_intents"]), 2)

    def test_one_genuinely_multi_concept_fact_counts_for_both(self):
        joint = ("Windows Telemetry Inspector is a passive Windows network diagnostics "
                 "tool designed to make local process activity inspectable.")
        ask, allowed = scope([joint], "Explain what it is and why it exists.")
        self.assertEqual(ask["requested_topic_fact_ids"],
                         {"overview": ["F0001"], "purpose": ["F0001"]})
        normalized = planner._normalize_outline(outline(ask, [1]), allowed, ask)
        self.assertEqual(len(normalized["scene_intents"]), 1)

    def test_unsupported_topic_and_supporting_only_fact_are_not_required(self):
        instruction = ("Explain what it is, why it exists, and how its architecture turns "
                       "Windows ETW network observations into useful local diagnostics.")
        ask, allowed = scope([CLAIMS[0], CLAIMS[2]], instruction)
        self.assertEqual(set(ask["requested_topic_fact_ids"]), {"overview", "workflow"})
        planner._normalize_outline(outline(ask, [1, 2]), allowed, ask)
        scoped, _ = scope([CLAIMS[0], CLAIMS[1], CLAIMS[2]], instruction,
                          roles={2: "embedded_reference"})
        self.assertNotIn("purpose", scoped["requested_topic_fact_ids"])

    def test_distinction_requires_complete_other_side_not_generic_overlap(self):
        generic = ("Windows Telemetry Inspector is a passive observation tool "
                   "that reports correlation, without blocking.")
        ask, allowed = scope([CLAIMS[0], CLAIMS[1], CLAIMS[2], generic], INSTRUCTIONS)
        self.assertNotIn("distinction-1", ask["requested_topic_fact_ids"])
        planner._normalize_outline(outline(ask, [1, 2, 3]), allowed, ask)

    def test_generic_tool_mention_does_not_satisfy_overview(self):
        correlation = ("The tool correlates passive observations without blocking, "
                       "interception, or proven causality.")
        ask, _ = scope([CLAIMS[0], CLAIMS[1], CLAIMS[2], correlation])
        self.assertEqual(ask["requested_topic_fact_ids"]["overview"], ["F0001"])

    def test_greedy_recovery_tie_break_and_idempotence_before_outro(self):
        joint = ("Windows Telemetry Inspector is a passive diagnostics tool designed to "
                 "make process activity inspectable locally.")
        ask, allowed = scope([joint, CLAIMS[2], CLAIMS[3]], INSTRUCTIONS)
        bad = outline(ask, [3, 3], ["SUMMARY", "OUTRO"])
        normalized = planner._normalize_outline(bad, allowed, ask, check_coverage=False)
        recovered = planner._recover_outline_coverage(normalized, ask, allowed)
        self.assertEqual([item["fact_ids"] for item in recovered["scene_intents"]],
                         [["F0003"], ["F0001"], ["F0002"], ["F0003"]])
        self.assertEqual(recovered["presentation"]["scene_titles"], {"s004": "The source"})
        self.assertEqual(planner._recover_outline_coverage(recovered, ask, allowed), recovered)

    def test_scene_limit_fails_clearly(self):
        bad = outline(self.ask, [4] * 32)
        canonical = {**bad, "scene_intents": [
            {**row, "id": f"s{index:03d}"}
            for index, row in enumerate(bad["scene_intents"], 1)
        ]}
        with self.assertRaisesRegex(planner.StructuredOutputError, "no room within 32 scenes"):
            planner._recover_outline_coverage(canonical, self.ask, self.allowed)

    def test_multipart_retries_recover_checkpoint_and_generate_diverse_scenes(self):
        provider = RepeatingProvider(self.ask)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def run():
                return planner._multipart_episode(provider, "storyboard", self.ask, path,
                                                  32768, 4096, 1024, 1, None, "scope")
            episode = run()
            outline_calls = [r for r in provider.requests if r.get("storyboard_mode") == "outline"]
            self.assertEqual(len(outline_calls), 2)
            self.assertIn("overview -> choose one of F0001",
                          outline_calls[1]["validation_feedback"])
            checkpoint = json_load(path / "manifests/storyboard-parts/outline.json")
            self.assertEqual([s["fact_ids"] for s in checkpoint["result"]["scene_intents"]],
                             [["F0004"], ["F0001"], ["F0002"], ["F0003"], ["F0004"]])
            self.assertEqual(episode["scenes"][-1]["type"], "OUTRO")
            self.assertEqual({s["fact_ids"][0] for s in episode["scenes"]
                              if s["type"] not in {"SECTION_TITLE", "OUTRO"}},
                             {"F0001", "F0002", "F0003", "F0004"})
            before = len(provider.requests)
            self.assertEqual(run(), episode)
            self.assertEqual(len(provider.requests), before)

            # An older cached outline with repeated fact selection is never trusted.
            checkpoint["result"] = copy.deepcopy(checkpoint["result"])
            checkpoint["result"]["scene_intents"][2]["fact_ids"] = ["F0004"]
            json_dump(path / "manifests/storyboard-parts/outline.json", checkpoint)
            self.assertEqual(run(), episode)
            self.assertEqual(len([r for r in provider.requests
                                  if r.get("storyboard_mode") == "outline"]), 4)
            self.assertEqual(json_load(path / "manifests/storyboard-parts/outline.json")
                             ["result"]["scene_intents"],
                             planner._normalize_outline(
                                 json_load(path / "manifests/storyboard-parts/outline.json")
                                 ["result"], self.allowed, self.ask,
                             )["scene_intents"])

    def test_full_path_retries_then_generates_missing_summary_scenes(self):
        provider = RepeatingProvider(self.ask, full=True)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._complete_episode(provider, "storyboard", self.ask,
                                                self.allowed, 1, project_dir=Path(tmp))
        full = [r for r in provider.requests if r.get("storyboard_mode") is None and
                "checks" not in r]
        self.assertEqual(len(full), 2)
        self.assertIn("purpose -> choose one of F0002", full[1]["validation_feedback"])
        self.assertEqual([s["fact_ids"] for s in episode["scenes"]],
                         [["F0004"], ["F0001"], ["F0002"], ["F0003"], ["F0004"]])
        self.assertEqual(episode["presentation"]["scene_titles"], {"s003": "The source"})
        self.assertEqual(episode["scenes"][-1]["id"], "s003")
        self.assertEqual({s["narration"] for s in episode["scenes"]
                          if s["type"] not in {"SECTION_TITLE", "OUTRO"}}, set(CLAIMS))

    def test_natural_outline_retry_wins_without_recovery(self):
        class ImprovingProvider(RepeatingProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if request.get("storyboard_mode") == "outline" and "validation_feedback" in request:
                    self.requests.append(request)
                    return outline(self.ask, [1, 2, 3, 4])
                return super().complete_json(system, user)

        provider = ImprovingProvider(self.ask)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._multipart_episode(provider, "storyboard", self.ask,
                                                 Path(tmp), 32768, 4096, 1024, 1,
                                                 None, "scope")
        self.assertEqual(len(episode["scenes"]), 4)
        self.assertEqual([scene["fact_ids"] for scene in episode["scenes"]],
                         [["F0001"], ["F0002"], ["F0003"], ["F0004"]])

    def test_full_recovery_exact_narration_fallback_after_natural_rejection(self):
        class RejectedNarration(RepeatingProvider):
            def complete_json(self, system, user):
                request = json.loads(user)
                if request.get("storyboard_mode") == "scenes":
                    result = super().complete_json(system, user)
                    for scene in result["scenes"]:
                        scene["narration"] = "The tool secretly blocks all traffic."
                    return result
                if "checks" in request:
                    self.requests.append(request)
                    return {"decisions": [{"id": check["id"],
                                           "supported": check["id"] == "episode-summary",
                                           "propositions": ([{"text": text, "support_indices": [0]}
                                                             for text in check["required_propositions"]]
                                                            if check["id"] == "episode-summary" else [])}
                                          for check in request["checks"]]}
                return super().complete_json(system, user)

        provider = RejectedNarration(self.ask, full=True)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._complete_episode(provider, "storyboard", self.ask,
                                                self.allowed, 1, project_dir=Path(tmp))
        added = episode["scenes"][1:4]
        self.assertEqual([scene["narration"] for scene in added], CLAIMS[:3])
        self.assertEqual(len([request for request in provider.requests
                              if request.get("storyboard_mode") == "scenes"]), 6)

    def test_full_recovery_preserves_final_suffix_without_outro(self):
        ask, allowed = scope(CLAIMS, INSTRUCTIONS + ' End with: “Documented only.”')

        class NoOutro(RepeatingProvider):
            def __init__(self, ask):
                super().__init__(ask, full=True)
                self.bad = outline(ask, [4])
                self.bad["scene_intents"][0]["purpose"] += " Documented only."

        provider = NoOutro(ask)
        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._complete_episode(provider, "storyboard", ask,
                                                allowed, 0, project_dir=Path(tmp))
        self.assertEqual(episode["scenes"][-1]["id"], "s001")
        self.assertTrue(episode["scenes"][-1]["narration"].endswith("Documented only."))
        self.assertEqual({scene["fact_ids"][0] for scene in episode["scenes"][:-1]},
                         {"F0001", "F0002", "F0003"})

    def test_full_recovery_prunes_duplicates_before_scene_limit(self):
        class TooMany(RepeatingProvider):
            def __init__(self, ask):
                super().__init__(ask, full=True)
                self.bad = outline(ask, [4] * 32)

        with tempfile.TemporaryDirectory() as tmp:
            episode = planner._complete_episode(TooMany(self.ask), "storyboard", self.ask,
                                                self.allowed, 0, project_dir=Path(tmp))
        self.assertLessEqual(len(episode["scenes"]), planner._MAX_STORYBOARD_SCENES)
        self.assertEqual(planner._missing_story_topics(episode["scenes"], self.ask), {})
        self.assertEqual(sum(scene["fact_ids"] == ["F0004"]
                             for scene in episode["scenes"]), 1)
        planner.validate_novelty(episode["scenes"])


if __name__ == "__main__":
    unittest.main()
