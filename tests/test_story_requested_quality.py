"""Grounded story choices when source code and authentic visuals overlap."""
from __future__ import annotations

import unittest

from source2reel import planner
from source2reel.research import _rank_requested_facts


COMMAND = "sed -i '/idle-timeout/s/^/#/' \"$CONFIG_FILE\""
NATURAL = "The service branch comments matching `idle-timeout` lines in `$CONFIG_FILE`."
INSTRUCTIONS = "Explain how the tool handles idle-timeout and how rollback works."


def fact(claim, ref, support, **extra):
    return {"claim": claim, "evidence_refs": [ref],
            "support": [{"evidence_ref": ref, "text": support}],
            "phase": "final", "confidence": "high", **extra}


def outline(*intents):
    return {"version": 1, "title": "Configuration tool", "slug": "config-tool",
            "summary": NATURAL, "scene_intents": list(intents)}


class RequestedStoryQualityTests(unittest.TestCase):
    def test_verified_code_paraphrase_wins_but_raw_command_remains_for_code(self):
        raw = fact(COMMAND, "E0004", COMMAND, direct_code_evidence=True)
        natural = fact(NATURAL, "E0004", COMMAND, direct_code_evidence=True,
                       naturalized_code_evidence=True)
        inventory = [{"ref": "E0004", "kind": "document", "evidence_role": "primary",
                      "relative_path": "config.sh"}]
        ranked = _rank_requested_facts([raw, natural], INSTRUCTIONS, "Config tool")
        self.assertEqual(ranked[0]["claim"], NATURAL)
        ask = planner._make_ask({"version": 1, "facts": [raw, natural], "assets": []},
                                [], inventory, "Config tool", INSTRUCTIONS)
        self.assertEqual(ask["requested_topic_fact_ids"]["detail-1"], ["F0002"])
        self.assertEqual(ask["research"]["facts"][0]["claim"], COMMAND)

        summary = lambda fact_id: {"type": "SUMMARY", "purpose": NATURAL,
                                   "fact_ids": [fact_id], "evidence_refs": ["E0004"]}
        canonical = planner._normalize_outline(outline(summary("F0001"), summary("F0002")),
                                               {"E0004"}, ask, check_coverage=False)
        self.assertEqual([(scene["id"], scene["fact_ids"])
                          for scene in canonical["scene_intents"]], [("s001", ["F0002"])])
        code = planner._normalize_outline(outline({**summary("F0001"), "type": "CODE"}),
                                          {"E0004"}, ask, check_coverage=False)
        self.assertEqual(code["scene_intents"][0]["fact_ids"], ["F0001"])
        self.assertEqual(code["scene_intents"][0]["type"], "CODE")


    def test_asset_recovery_defers_coverage_until_specific_repair_can_run(self):
        raw = fact(COMMAND, "E0004", COMMAND, direct_code_evidence=True)
        natural = fact(NATURAL, "E0004", COMMAND, direct_code_evidence=True,
                       naturalized_code_evidence=True)
        rollback = fact("The tool restores the previous service configuration.",
                        "E0004", "The tool restores the previous service configuration.")
        media = [("E0005", "fix.jpeg"), ("E0006", "menu.jpeg"),
                 ("E0007", "undo.jpeg")]
        inventory = [
            {"ref": "E0004", "kind": "document", "evidence_role": "primary",
             "relative_path": "config.sh"},
            *[{"ref": ref, "kind": "media", "evidence_role": "primary",
               "relative_path": f"screens/{name}"} for ref, name in media],
        ]
        research = {"version": 1, "facts": [raw, natural, rollback], "assets": [
            {"evidence_ref": ref, "purpose": name, "authentic_project_media": True}
            for ref, name in media
        ]}
        ask = planner._make_ask(research, [], inventory, "Configuration tool", INSTRUCTIONS)
        # Before canonical story composition the raw command covers this topic.
        # Composition correctly replaces it with the verified natural paraphrase,
        # so coverage must be checked afterwards, where F0003 can repair it.
        ask["requested_topic_fact_ids"] = {"detail-1": ["F0003", "F0001"]}
        allowed = {entry["ref"] for entry in inventory}
        broken = outline({
            "type": "HERO", "purpose": NATURAL, "fact_ids": ["F0001"],
            "evidence_refs": ["E0004"], "asset_ref": "E0004",
        })

        asset_repaired = planner._recover_outline_assets(broken, allowed, ask)
        self.assertIsNotNone(asset_repaired)
        self.assertEqual(asset_repaired["scene_intents"][0]["type"], "SUMMARY")

        canonical = planner._normalize_outline(
            asset_repaired, allowed, ask, check_coverage=False,
        )
        self.assertEqual(canonical["scene_intents"][0]["fact_ids"], ["F0002"])
        with self.assertRaises(planner._RequestedCoverageError):
            planner._validate_focus_coverage(canonical["scene_intents"], ask)

        recovered = planner._recover_outline_coverage(canonical, ask, allowed)
        self.assertEqual([scene["fact_ids"] for scene in recovered["scene_intents"]],
                         [["F0002"], ["F0003"]])
        planner._validate_focus_coverage(recovered["scene_intents"], ask)

    def test_two_screenshots_cannot_repeat_an_introductory_fact(self):
        purpose = "The configuration tool is designed to address idle playback delays."
        mechanism = "The tool comments idle-timeout in the service configuration."
        facts = [fact(purpose, "E0002", purpose), fact(mechanism, "E0004", mechanism)]
        media = [("E0005", "fix.jpeg"), ("E0006", "menu.jpeg"),
                 ("E0007", "undo.jpeg")]
        inventory = ([{"ref": ref, "kind": "document", "evidence_role": "primary",
                       "relative_path": f"doc-{ref}.txt"} for ref in ("E0002", "E0004")] +
                     [{"ref": ref, "kind": "media", "evidence_role": "primary",
                       "relative_path": f"screens/{name}"} for ref, name in media])
        research = {"version": 1, "facts": facts, "assets": [
            {"evidence_ref": ref, "purpose": name, "authentic_project_media": True}
            for ref, name in media]}
        ask = planner._make_ask(research, [], inventory, "Configuration tool", "")
        raw = outline(
            {"type": "HERO", "purpose": purpose, "fact_ids": ["F0001"],
             "evidence_refs": ["E0002", "E0006"], "asset_ref": "E0006"},
            {"type": "SUMMARY", "purpose": purpose, "fact_ids": ["F0001"],
             "evidence_refs": ["E0002"]},
            {"type": "PROJECT_EVIDENCE", "purpose": purpose, "fact_ids": ["F0001"],
             "evidence_refs": ["E0002", "E0005"], "asset_ref": "E0005"},
            {"type": "PROJECT_EVIDENCE", "purpose": mechanism, "fact_ids": ["F0002"],
             "evidence_refs": ["E0004", "E0007"], "asset_ref": "E0007"},
        )
        canonical = planner._normalize_outline(raw, {entry["ref"] for entry in inventory},
                                               ask, check_coverage=False)
        self.assertEqual([(scene["id"], scene.get("asset_ref"))
                          for scene in canonical["scene_intents"]],
                         [("s001", "E0006"), ("s004", "E0007")])

    def test_coverage_recovery_chooses_narrow_mechanism_facts(self):
        names = ("backend alpha", "backend beta", "storage", "rollback", "helper")
        broad = fact("The project configures backend alpha, backend beta, storage, "
                     "rollback, and helper.", "E0001", "A combined overview.")
        specific = [fact(f"The project configures {name} independently.", f"E{i:04d}",
                         f"The project configures {name} independently.")
                    for i, name in enumerate(names, 2)]
        inventory = [{"ref": f"E{i:04d}", "kind": "document", "evidence_role": "primary",
                      "relative_path": f"doc-{i}.txt"} for i in range(1, 7)]
        ask = planner._make_ask({"version": 1, "facts": [broad, *specific], "assets": []},
                                [], inventory, "Configuration tool", "")
        ask["requested_topic_fact_ids"] = {
            f"detail-{i}": ["F0001", f"F{i+1:04d}"] for i in range(1, 6)
        }
        added = planner._coverage_additions([], ask, {entry["ref"] for entry in inventory})
        self.assertEqual([scene["fact_ids"] for scene in added],
                         [[f"F{i:04d}"] for i in range(2, 7)])


if __name__ == "__main__":
    unittest.main()
