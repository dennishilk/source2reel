"""Current overview recovery must remain source-exact and avoid stale milestones."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from source2reel.planner import _primary_anchors
from source2reel.research import _current_overview_facts


class NoModel:
    def complete_json(self, *_args):
        raise AssertionError("An exact quotation should not need a model call")


class CurrentOverviewTests(unittest.TestCase):
    def test_current_readme_recovers_proven_path_and_future_boundary(self):
        overview = """# WidgetOS

## Current status

```text
M66 physical USB baseline
WidgetOS does not support the native desktop.
```

The current runtime is frozen at:

```text
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

WidgetOS boots a writable USB root, enumerates the real Genesys hub, accepts the native mouse and runs WidgetWM with WidgetEdit and WidgetFiles.

Implemented foundations include Ring 3, native syscalls and a display service.

## Native desktop

WidgetEdit loads and saves through WidgetFS. WidgetFiles uses VFS to enumerate real directories.

## Graphics today

There is no native GPU driver yet. Native modesetting is planned for later work.
"""
        inventory = {"evidence": [
            {"ref": "E0001", "kind": "document", "relative_path": "source-01/README.md",
             "evidence_role": "primary", "excerpt": overview},
            {"ref": "E0002", "kind": "document", "relative_path": "source-01/docs/M12.md",
             "evidence_role": "primary", "excerpt": "Historical development notes."},
        ]}
        originals = [
            {"claim": "WidgetOS is a desktop operating system.", "evidence_refs": ["E0001"],
             "support": [{"evidence_ref": "E0001", "text": "WidgetOS is a desktop operating system."}],
             "phase": "final", "confidence": "high"},
            {"claim": "WidgetOS Milestone 12 configures the first USB endpoint.",
             "evidence_refs": ["E0002"],
             "support": [{"evidence_ref": "E0002", "text": "Historical development notes."}],
             "phase": "final", "confidence": "high"},
            {"claim": "WidgetOS does not support ring 3 or native syscalls.",
             "evidence_refs": ["E0002"],
             "support": [{"evidence_ref": "E0002", "text": "Historical development notes."}],
             "phase": "final", "confidence": "high"},
            {"claim": "The M66 physical runtime is frozen at commit " + "a" * 40 + ".",
             "evidence_refs": ["E0002"],
             "support": [{"evidence_ref": "E0002", "text": "Historical development notes."}],
             "phase": "final", "confidence": "high"},
        ]
        request = ("Explain the current proven WidgetOS implementation: its USB path, "
                   "native desktop, and what remains planned for future work.")
        with tempfile.TemporaryDirectory() as tmp:
            facts = _current_overview_facts(
                originals, inventory, "WidgetOS", request, NoModel(), Path(tmp), None,
            )
        claims = [fact["claim"] for fact in facts]
        self.assertTrue(any("Genesys hub" in claim and "WidgetWM" in claim
                            for claim in claims))
        self.assertTrue(any("WidgetFiles uses VFS to enumerate real directories" in claim
                            for claim in claims))
        self.assertTrue(any("native GPU driver yet" in claim for claim in claims))
        self.assertFalse(any("does not support ring 3" in claim for claim in claims))
        self.assertFalse(any("Milestone 12" in claim for claim in claims))
        self.assertFalse(any("a" * 40 in claim for claim in claims))
        self.assertFalse(any("does not support the native desktop" in claim
                             for claim in claims))
        self.assertEqual(originals, _current_overview_facts(
            originals, inventory, "WidgetOS", "Explain the historical milestones.",
            NoModel(), Path(tmp), None,
        ))
        for fact in facts:
            if fact in originals:
                continue
            self.assertEqual(fact["evidence_refs"], ["E0001"])
            self.assertEqual(fact["support"], [{"evidence_ref": "E0001",
                                                "text": fact["claim"]}])
            self.assertIs(fact["current_overview"], True)
        anchors = _primary_anchors(
            {"facts": facts, "assets": []}, inventory, set(), "WidgetOS", request,
        )
        self.assertTrue(any("Genesys hub" in item["claim"] for item in anchors))

    def test_bounded_anchors_carry_current_apps_and_planned_graphics(self):
        current = [
            "WidgetOS boots its native writable USB root, enumerates a physical hub, "
            "accepts the mouse through HID and runs WidgetWM with WidgetTerminal, "
            "WidgetEdit and WidgetFiles.",
            "The physical hub, mouse, cursor focus and low-latency path are also "
            "proven on real hardware.",
            "WidgetOS supports multiple USB controllers, downstream hub topology, "
            "HID interrupt transfers and mass storage.",
            "WidgetTerminal runs a separately scheduled shell. WidgetEdit loads and "
            "saves through WidgetFS. WidgetFiles uses VFS to enumerate directories.",
            "There is no native GPU driver yet. Native modesetting is planned for "
            "later work.",
        ]
        old = [
            "WidgetOS is an independently built desktop operating system around its own kernel.",
            "WidgetOS implements memory management and carefully bounded page table mappings.",
            "WidgetOS uses a heap allocator to store temporary filesystem pages.",
            "WidgetOS includes an independent display service in userspace.",
            "WidgetOS supports an older virtual block device for testing storage.",
            "WidgetOS has a carefully designed system call dispatch boundary.",
        ]
        inventory = {"evidence": [{
            "ref": "E0001", "kind": "document", "relative_path": "source-01/README.md",
            "evidence_role": "primary", "excerpt": "# WidgetOS\n\n## Current status\n\n" +
            "\n\n".join(current),
        }] + [{
            "ref": f"E{i:04d}", "kind": "document",
            "relative_path": f"source-01/docs/archive-{i}.md",
            "evidence_role": "primary", "excerpt": claim,
        } for i, claim in enumerate(old, 2)]}
        originals = [{"claim": claim, "evidence_refs": [f"E{i:04d}"],
                      "support": [{"evidence_ref": f"E{i:04d}", "text": claim}],
                      "phase": "final", "confidence": "high"}
                     for i, claim in enumerate(old, 2)]
        request = ("Explain the current WidgetOS implementation: native desktop, "
                   "USB/HID path and physical proof. Distinguish proven work from "
                   "planned future graphics.")
        with tempfile.TemporaryDirectory() as tmp:
            facts = _current_overview_facts(
                originals, inventory, "WidgetOS", request, NoModel(), Path(tmp), None,
            )
        anchors = _primary_anchors(
            {"facts": facts, "assets": []}, inventory, set(), "WidgetOS", request,
        )
        claims = [item["claim"] for item in anchors]
        self.assertTrue(any("WidgetFiles uses VFS" in claim for claim in claims))
        self.assertTrue(any("native GPU driver yet" in claim for claim in claims))


if __name__ == "__main__":
    unittest.main()
