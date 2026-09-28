"""Authentic project visuals survive planner map-reduce compaction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.planner import _compact_payload, _planner_visuals, plan
from source2reel.util import json_dump, json_load


def _fixture():
    def media(ref, filename, category, caption, value="high"):
        return {"ref": ref, "kind": "media", "relative_path": f"source-02/gallery/{filename}",
                "evidence_role": "primary", "mime": "image/png", "ai_media": {
                    "category": category, "caption": caption, "visible_text": [],
                    "evidence_value": value, "reasons": ["Visible in the supplied image"],
                }}

    entries = [
        {"ref": "E0036", "kind": "document", "relative_path": "source-01/README.md",
         "evidence_role": "primary", "excerpt": "BoringOS has its own kernel."},
        {"ref": "E0983", "kind": "media", "relative_path": "source-02/gallery/edit-preview.png",
         "evidence_role": "primary", "mime": "image/png", "excerpt": "BoringOS editor preview.",
         "ai_media": {"category": "screenshot", "caption": "BoringOS editor window",
                      "visible_text": [], "evidence_value": "medium", "reasons": ["Visible"]}},
        media("E0970", "rack.png", "photo", "Personal server rack"),
        media("E0971", "retro-room.png", "photo", "Retro computer room"),
        media("E0972", "pokemon-pc.png", "photo", "Pokémon themed PC"),
        media("E0973", "motorcycle.png", "photo", "Motorcycle"),
        media("E0974", "cat.png", "photo", "Cat"),
        media("E0975", "cisco.png", "photo", "Cisco switch"),
        media("E0982", "boringfetch-preview.png", "terminal screenshot",
              "BoringOS boringfetch terminal output"),
        media("E0984", "files-preview.png", "software screenshot",
              "File manager window"),
    ]
    facts = [
        {"claim": "BoringOS has its own kernel.", "evidence_refs": ["E0036"],
         "support": [{"evidence_ref": "E0036", "text": "BoringOS has its own kernel."}],
         "phase": "final", "confidence": "high"},
        {"claim": "BoringOS editor preview.", "evidence_refs": ["E0983"],
         "support": [{"evidence_ref": "E0983", "text": "BoringOS editor preview."}],
         "phase": "final", "confidence": "high"},
    ]
    return {"version": 1, "facts": facts, "assets": []}, {"evidence": entries}


class DroppingMediaProvider:
    def __init__(self):
        self.compact_requests = []
        self.storyboard_requests = []

    def complete_json(self, system, user):
        payload = json.loads(user)
        if system == "compact":
            self.compact_requests.append(payload)
            first_fact = next(record.get("capsule", record) for record in payload["records"]
                              if record.get("source_fact_id") or
                              record.get("capsule", {}).get("source_fact_id"))
            selected = [{"source_fact_id": first_fact["source_fact_id"],
                         "claim": first_fact["claim"],
                         "evidence_refs": first_fact["evidence_refs"], "media_refs": []}]
            if payload["level"] == 1:
                # The first-level response contains real media and a generic
                # gallery photo, but omits the relevant boringfetch screenshot.
                selected.extend({"evidence_refs": [ref], "media_refs": [ref],
                                 "visual_purpose": purpose}
                                for ref, purpose in (("E0983", "BoringOS edit preview"),
                                                     ("E0984", "BoringOS files preview"),
                                                     ("E0970", "Authentic rack image")))
            # The later model response omits every asset capsule.
            return {"capsules": selected}
        self.storyboard_requests.append(payload)
        fact = payload["research"]["facts"][0]
        return {"version": 1, "title": "BoringOS", "slug": "boringos",
                "summary": fact["claim"], "scenes": [{
                    "id": "s001", "type": "PROJECT_EVIDENCE", "title": "BoringOS preview",
                    "narration": fact["claim"], "fact_ids": [fact["fact_id"]],
                    "evidence_refs": [*fact["evidence_refs"], "E0982"],
                    "asset_ref": "E0982",
                }]}


class PlannerMediaCarryTests(unittest.TestCase):
    def test_small_context_carries_two_distinct_project_visuals(self):
        research, inventory = _fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "boringos"
            (root / "prompts").mkdir()
            (root / "prompts" / "storyboard.txt").write_text("storyboard")
            (root / "prompts" / "planner_compact.txt").write_text("compact")
            with patch("source2reel.planner._final_requests_fit",
                       side_effect=[False, True]), \
                 patch("source2reel.planner.split_for_context",
                       side_effect=lambda records, *args, **kwargs: [records]):
                plan(DroppingMediaProvider(), research, inventory, project,
                     "BoringOS", "Explain the actual OS.",
                     context_size=8192, max_retries=0)
            refs = {item["ref"] for item in json_load(
                project / "manifests" / "planner-evidence.json",
            )["media_inventory"]}
            self.assertEqual(len(refs), 2)
            self.assertIn("E0982", refs)
            self.assertFalse(refs & {f"E{ref}" for ref in range(970, 976)})

    def test_boringos_visuals_outlive_two_reduce_levels_and_reach_storyboard(self):
        research, inventory = _fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "projects" / "boringos"
            (root / "prompts").mkdir()
            (root / "prompts" / "storyboard.txt").write_text("storyboard")
            (root / "prompts" / "planner_compact.txt").write_text("compact")
            provider = DroppingMediaProvider()

            def run():
                with patch("source2reel.planner._final_requests_fit",
                           side_effect=[False, False, True]), \
                     patch("source2reel.planner.split_for_context",
                           side_effect=lambda records, *args, **kwargs: [records]):
                    return plan(provider, research, inventory, project, "BoringOS",
                                "Explain the actual OS.", max_retries=0)

            episode = run()
            self.assertEqual(episode["scenes"][0]["asset_ref"], "E0982")
            part = json_load(project / "manifests/planner-compact-parts/level-01-part-001.json")
            first_media = {r for c in part["result"]["capsules"] for r in c["media_refs"]}
            self.assertEqual(first_media, {"E0983", "E0984", "E0970"})
            second = provider.compact_requests[1]
            self.assertEqual({r for item in second["records"]
                              for r in item["capsule"]["media_refs"]},
                             {"E0982", "E0983", "E0984"})
            manifest = json_load(project / "manifests/planner-evidence.json")
            self.assertEqual((manifest["strategy"], manifest["levels"]), ("map-reduce", 2))
            expected = ["E0982", "E0984", "E0983"]
            self.assertEqual([m["ref"] for m in manifest["media_inventory"]],
                             ["E0983", "E0982", "E0984"])
            self.assertEqual({a["evidence_ref"] for a in manifest["research"]["assets"]},
                             set(expected))
            self.assertEqual(next(asset for asset in manifest["research"]["assets"]
                                  if asset["evidence_ref"] == "E0983")["source_fact_ids"],
                             ["R0002"])
            ask = provider.storyboard_requests[-1]
            self.assertEqual(set(ask["visual_asset_refs"]), set(expected))
            self.assertIn("PROJECT_EVIDENCE", ask["allowed_scene_types"])

            compact_calls = len(provider.compact_requests)
            run()
            self.assertEqual(len(provider.compact_requests), compact_calls)
            self.assertEqual(json_load(project / "manifests/planner-evidence.json")["media_inventory"],
                             manifest["media_inventory"])

            # An old, pre-fix checkpoint must be regenerated once; the new
            # checkpoint then remains reusable on identical subsequent runs.
            old = dict(provider.compact_requests[0])
            del old["planner_media_contract"]
            part_path = project / "manifests/planner-compact-parts/level-01-part-001.json"
            cached = json_load(part_path)
            cached["input_sha256"] = hashlib.sha256(
                ("compact\0" + json.dumps(old, ensure_ascii=False)).encode("utf-8")
            ).hexdigest()
            json_dump(part_path, cached)
            run()
            self.assertEqual(len(provider.compact_requests), compact_calls + 1)
            self.assertEqual(json_load(project / "manifests/planner-evidence.json")["media_inventory"],
                             manifest["media_inventory"])

    def test_gallery_never_beats_software_and_selection_is_bounded(self):
        research, inventory = _fixture()
        for i in range(20):
            inventory["evidence"].append({
                "ref": f"E{1100 + i}", "kind": "media",
                "relative_path": f"source-02/gallery/boringos-screen-{i}.png",
                "evidence_role": "primary", "ai_media": {
                    "category": "screenshot", "caption": f"BoringOS screen {i}",
                    "evidence_value": "low", "visible_text": [], "reasons": ["Visible"],
                },
            })
        media = [e for e in inventory["evidence"] if e["kind"] == "media"]
        selected = _planner_visuals(research, media, "BoringOS", "Explain the OS",
                                    inventory, selected_hints={"E0984": "BoringOS files preview",
                                                               "E0970": "BoringOS software output"})
        refs = [c["media_refs"][0] for c in selected]
        self.assertEqual(refs[:3], ["E0982", "E0984", "E0983"])
        self.assertEqual(len(refs), 5)
        self.assertFalse({f"E{r}" for r in range(970, 976)} & set(refs))
        self.assertEqual(selected, _planner_visuals(research, media, "BoringOS",
                                                    "Explain the OS", inventory,
                                                    selected_hints={"E0984": "BoringOS files preview",
                                                                    "E0970": "BoringOS software output"}))

    def test_compaction_contract_is_part_of_checkpoint_input(self):
        payload = _compact_payload(1, 1, [], "BoringOS", "Explain the OS")
        self.assertEqual(payload["planner_media_contract"], "planner-authentic-media-v1")
        self.assertNotEqual(json.dumps(payload, sort_keys=True), json.dumps(
            {k: v for k, v in payload.items() if k != "planner_media_contract"},
            sort_keys=True))


if __name__ == "__main__":
    unittest.main()
