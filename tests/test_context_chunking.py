from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel.chunking import checkpointed_complete_json, fits_context, split_for_context
from source2reel.planner import _repair_episode_shape, plan
from source2reel.research import research
from source2reel.util import json_load


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system: str, user: str):
        self.calls += 1
        payload = json.loads(user)
        if system == "research":
            ref = payload["evidence"][0]["ref"]
            return {
                "facts": [{
                    "claim": f"Supported fact for {ref}",
                    "evidence_refs": [ref],
                    "phase": "final",
                    "confidence": "high",
                }],
                "assets": [],
            }
        if system == "compact":
            refs = []
            media_refs = []
            for record in payload["records"]:
                if record.get("kind") == "capsule":
                    capsule = record["capsule"]
                    refs.extend(capsule.get("evidence_refs", []))
                    media_refs.extend(capsule.get("media_refs", []))
                else:
                    refs.extend(record.get("evidence_refs", []))
                    media = record.get("media")
                    if isinstance(media, dict):
                        media_refs.append(media.get("ref"))
                    elif isinstance(media, list):
                        media_refs.extend(m.get("ref") for m in media)
                    asset = record.get("asset")
                    if isinstance(asset, dict):
                        refs.append(asset.get("evidence_ref"))
            refs = [r for r in dict.fromkeys(refs) if r]
            media_refs = [r for r in dict.fromkeys(media_refs) if r]
            chosen = refs[:2] or media_refs[:1] or ["E0001"]
            return {
                "capsules": [{
                    "claim": "Compact supported capsule.",
                    "evidence_refs": chosen,
                    "media_refs": media_refs[:1],
                    "phase": "final",
                    "confidence": "high",
                    "visual_purpose": "Compact visual evidence.",
                }]
            }
        if system == "storyboard":
            return {
                "version": 1,
                "title": "Context-safe test",
                "slug": "context-safe-test",
                "summary": "Test",
                "scenes": [{
                    "id": "s001",
                    "type": "PROJECT_EVIDENCE",
                    "title": "Proof",
                    "narration": "Evidence-grounded narration.",
                    "evidence_refs": ["E0001"],
                    "asset_ref": "E0001",
                    "annotations": [],
                    "pad_after_seconds": 0.5,
                    "diagram": {},
                    "notes": "",
                }],
            }
        return {"ok": True}


class ContextChunkingTests(unittest.TestCase):
    def test_split_for_context_keeps_each_request_inside_budget(self):
        items = [{"id": i, "text": "x" * 1200} for i in range(8)]
        make_payload = lambda n, batch: {"part": n, "records": batch}
        chunks = split_for_context(
            items, "system", make_payload,
            context_size=2200, output_reserve_tokens=500, safety_tokens=300,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(c) for c in chunks), len(items))
        for i, chunk in enumerate(chunks, 1):
            self.assertTrue(fits_context(
                "system",
                json.dumps(make_payload(i, chunk)),
                2200, 500, 300,
            ))

    def test_deterministic_episode_shape_repair(self):
        repaired = _repair_episode_shape({
            "episode": {
                "title": "Demo",
                "scenes": [{
                    "type": "PROJECT_EVIDENCE",
                    "narration": "Supported narration.",
                    "evidence_refs": ["E0001"],
                }],
            }
        })
        self.assertEqual(repaired["version"], 1)
        self.assertEqual(repaired["scenes"][0]["id"], "s001")
        self.assertEqual(repaired["scenes"][0]["asset_ref"], "E0001")

    def test_checkpoint_reuses_identical_input_and_invalidates_changed_input(self):
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.json"
            normalize = lambda value: value
            first = checkpointed_complete_json(provider, "other", {"n": 1}, path, normalize)
            second = checkpointed_complete_json(provider, "other", {"n": 1}, path, normalize)
            self.assertEqual(first, second)
            self.assertEqual(provider.calls, 1)
            checkpointed_complete_json(provider, "other", {"n": 2}, path, normalize)
            self.assertEqual(provider.calls, 2)

    def test_research_auto_splits_and_resumes_saved_parts(self):
        provider = FakeProvider()
        evidence = [
            {"ref": f"E{i:04d}", "kind": "document", "text": "x" * 1400}
            for i in range(1, 9)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "demo"
            (root / "prompts").mkdir(parents=True)
            (root / "prompts" / "research.txt").write_text("research")
            out = research(
                provider, {"evidence": evidence}, project,
                max_chars=999999, context_size=2400,
                output_reserve_tokens=500, safety_tokens=300,
            )
            part_count = len(list((project / "manifests" / "research-parts").glob("part-*.json")))
            self.assertGreater(part_count, 1)
            calls_after_first = provider.calls
            self.assertEqual(len(out["facts"]), part_count)

            research(
                provider, {"evidence": evidence}, project,
                max_chars=999999, context_size=2400,
                output_reserve_tokens=500, safety_tokens=300,
            )
            self.assertEqual(provider.calls, calls_after_first)

    def test_planner_uses_map_reduce_when_direct_payload_is_too_large(self):
        provider = FakeProvider()
        evidence = []
        facts = []
        for i in range(1, 25):
            ref = f"E{i:04d}"
            kind = "media" if i <= 4 else "document"
            entry = {
                "ref": ref,
                "kind": kind,
                "relative_path": f"asset-{i}.png" if kind == "media" else f"doc-{i}.txt",
                "mime": "image/png" if kind == "media" else "text/plain",
            }
            if kind == "media":
                entry["ai_media"] = {"description": "authentic UI " + ("x" * 200)}
            evidence.append(entry)
            facts.append({
                "claim": f"Supported fact {i}: " + ("detail " * 90),
                "evidence_refs": [ref],
                "phase": "final",
                "confidence": "high",
            })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "demo"
            (root / "prompts").mkdir(parents=True)
            (root / "prompts" / "storyboard.txt").write_text("storyboard")
            (root / "prompts" / "planner_compact.txt").write_text("compact")
            ep = plan(
                provider,
                {"version": 1, "facts": facts, "assets": []},
                {"evidence": evidence},
                project,
                "Demo",
                "Keep it grounded.",
                context_size=4200,
                output_reserve_tokens=1000,
                safety_tokens=500,
            )
            self.assertEqual(ep["version"], 1)
            manifest = json_load(project / "manifests" / "planner-evidence.json")
            self.assertEqual(manifest["strategy"], "map-reduce")
            self.assertTrue(list((project / "manifests" / "planner-compact-parts").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
