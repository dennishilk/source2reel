"""Regression coverage for bounded original-source context in planning."""
from __future__ import annotations

import unittest

from source2reel.planner import (
    _PLANNER_SOURCE_CONTEXT_CONTRACT,
    _PLANNER_SOURCE_CONTEXT_LIMIT,
    _PLANNER_SOURCE_CONTEXT_TOTAL_CHARS,
    _capsules_to_research,
    _compact_payload,
    _make_ask,
    _normalize_capsules,
    _outline_payload,
    _planner_records,
    _scene_part_payload,
    _source_context_passages,
)


README = """# SPDIF Fix Tool

A simple interactive Bash tool to fix SPDIF sound delay and standby issues on Linux.
Supports PulseAudio, PipeWire, and ALSA.

## Features

- Keeps SPDIF output active → no more sound delay
- Prevents SPDIF from going into standby
- PulseAudio removes module-suspend-on-idle
- PipeWire disables suspend in pipewire.conf
- ALSA creates .asoundrc for SPDIF keepalive
- Backup and restore of original configuration
- The fallback autostart helper uses an effectively silent sine wave.

## Installation

~~~bash
git clone https://example.invalid/spdif-fix
cd spdif-fix
chmod +x spdif-fix.sh
~~~

1 - fix
2 - undo
3 - close
"""

SCRIPT = """if [ "$CHOICE" = "1" ]; then
    sudo sed -i '/module-suspend-on-idle/s/^/#/' /etc/pulse/default.pa
elif [ "$CHOICE" = "2" ]; then
    sudo sed -i 's/^#\(.*module-suspend-on-idle.*\)/\1/' /etc/pulse/default.pa
fi
"""

INSTRUCTIONS = (
    "Explain what this project solves, why SPDIF standby can cause delayed audio, "
    "how PulseAudio, PipeWire and ALSA are handled, how backup and reset work, "
    "and what the fallback autostart helper does."
)


def evidence(ref: str, path: str, excerpt: str) -> dict:
    return {
        "ref": ref,
        "kind": "document",
        "relative_path": path,
        "evidence_role": "primary",
        "excerpt": excerpt,
    }


class PlannerSourceContextTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {"evidence": [
            evidence("E0001", "README.md", README),
            evidence("E0002", "spdif-fix.sh", SCRIPT),
        ]}
        self.research = {"version": 1, "facts": [{
            "claim": "The SPDIF Fix Tool addresses SPDIF audio delay and standby issues.",
            "evidence_refs": ["E0001"],
            "support": [{"evidence_ref": "E0001",
                         "text": "A simple interactive Bash tool to fix SPDIF sound delay and standby issues on Linux."}],
            "phase": "final",
            "confidence": "high",
        }], "assets": []}

    def test_readme_context_is_bounded_verbatim_and_narrative_only(self):
        passages = _source_context_passages(
            self.research, self.inventory, "spdif-fix", INSTRUCTIONS,
        )
        self.assertTrue(passages)
        self.assertLessEqual(len(passages), _PLANNER_SOURCE_CONTEXT_LIMIT)
        self.assertLessEqual(
            sum(len(item["text"]) for item in passages),
            _PLANNER_SOURCE_CONTEXT_TOTAL_CHARS,
        )
        readme = [item for item in passages if item["evidence_ref"] == "E0001"]
        self.assertTrue(readme)
        joined = "\n".join(item["text"] for item in readme)
        self.assertIn("Keeps SPDIF output active", joined)
        self.assertIn("Prevents SPDIF from going into standby", joined)
        for item in passages:
            source = next(entry["excerpt"] for entry in self.inventory["evidence"]
                          if entry["ref"] == item["evidence_ref"])
            self.assertIn(item["text"], source)
            self.assertEqual(item["context_role"], "narrative_only")

    def test_context_reaches_direct_compact_outline_and_scene_payloads(self):
        context = _source_context_passages(
            self.research, self.inventory, "spdif-fix", INSTRUCTIONS,
        )
        index = [{key: entry.get(key) for key in
                  ("ref", "relative_path", "kind", "evidence_role")}
                 for entry in self.inventory["evidence"]]
        ask = _make_ask(
            self.research, [], index, "spdif-fix", INSTRUCTIONS, [], context,
        )
        self.assertEqual(ask["source_context_contract"], _PLANNER_SOURCE_CONTEXT_CONTRACT)
        self.assertEqual(ask["source_context"], context)
        self.assertEqual(_outline_payload(ask)["source_context"], context)
        self.assertEqual(
            _compact_payload(1, 1, [], "spdif-fix", INSTRUCTIONS, context)["source_context"],
            context,
        )

        fact = ask["research"]["facts"][0]
        intent = {
            "id": "s001",
            "type": "SUMMARY",
            "purpose": "Introduce the documented problem.",
            "fact_ids": [fact["fact_id"]],
            "evidence_refs": fact["evidence_refs"],
        }
        outline = {
            "version": 1,
            "title": "SPDIF Fix",
            "slug": "spdif-fix",
            "summary": fact["claim"],
            "scene_intents": [intent],
        }
        scene_payload = _scene_part_payload(ask, outline, [intent], 1, 1, "scope")
        self.assertEqual(scene_payload["source_context"], context)
        self.assertIn("not factual authority", ask["source_context_requirement"])

    def test_operation_guard_survives_planner_compaction(self):
        guarded = {
            "version": 1,
            "facts": [{
                "claim": "The script removes the leading comment marker from matching lines.",
                "evidence_refs": ["E0002"],
                "support": [{"evidence_ref": "E0002",
                             "text": "sudo sed -i 's/^#\\(.*module-suspend-on-idle.*\\)/\\1/' /etc/pulse/default.pa"}],
                "phase": "unknown",
                "confidence": "high",
                "operation_guard": {
                    "kind": "shell_equals",
                    "variable": "CHOICE",
                    "equals": "2",
                    "source": 'elif [ "$CHOICE" = "2" ]; then',
                },
            }],
            "assets": [],
        }
        records = _planner_records(guarded, [], self.inventory["evidence"])
        self.assertEqual(records[0]["operation_guard"]["equals"], "2")
        normalized = _normalize_capsules(
            {"capsules": [{
                "source_fact_id": "R0001",
                "claim": "model label is not authoritative",
                "evidence_refs": ["E0002"],
                "media_refs": [],
                "phase": "unknown",
                "confidence": "high",
                "visual_purpose": "",
            }]},
            {"E0002"}, set(), records,
        )["capsules"]
        self.assertEqual(normalized[0]["operation_guard"]["equals"], "2")
        compact_research = _capsules_to_research(normalized)
        self.assertEqual(compact_research["facts"][0]["operation_guard"]["equals"], "2")


if __name__ == "__main__":
    unittest.main()
