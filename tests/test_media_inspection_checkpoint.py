"""Content-addressed reuse of successful visual evidence inspections."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from source2reel import media_ai, planner, research
from source2reel.editorial import EDITORIAL_CONTRACT
from source2reel.grounding import GROUNDING_CONTRACT
from source2reel.media_ai import MEDIA_INSPECTION_CONTRACT, enrich_media
from source2reel.providers import LLMProvider, OllamaProvider, OpenAICompatProvider
from source2reel.util import json_load, sha256_file


class Vision(LLMProvider):
    def __init__(self, *, model="local-model", temperature=0.1, max_output_tokens=512,
                 url="http://127.0.0.1:8080", failure=False, result=None):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.url = url
        self.failure = failure
        self.result = result
        self.calls = []

    def complete_json(self, _system, _user):
        raise AssertionError("Only the synthetic image adapter should be called")

    def complete_json_with_image(self, system, user, image_path):
        self.calls.append((system, user, sha256_file(Path(image_path))))
        if self.failure:
            raise ConnectionError("local AI unavailable")
        if self.result is not None:
            return copy.deepcopy(self.result)
        return {"category": "photo", "caption": f"Visible variant {len(self.calls)}",
                "visible_text": [], "evidence_value": "high",
                "reasons": ["Visible project media"], "detail": {"nested": [1, 2]}}


class OtherVision(Vision):
    pass


def setup(root: Path):
    project = root / "projects" / "demo"
    project.mkdir(parents=True)
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "media-inspection.txt").write_text("Inspect only visible project media.")
    image = project / "still.png"
    Image.new("RGB", (32, 24), (125, 20, 40)).save(image)
    return project, image


def inventory(image: Path, *, ref="E0001", relative="still.png", width=32, height=24):
    return {"version": 1, "evidence": [{"ref": ref, "kind": "media",
             "path": str(image), "relative_path": relative,
             "media": {"width": width, "height": height},
             "sha256": sha256_file(image), "evidence_role": "primary"}]}


def checkpoints(project: Path):
    return sorted((project / "manifests" / "media-inspection").glob("*.json"))


class VisualCheckpointTests(unittest.TestCase):
    def test_successful_image_is_reused_with_identical_planner_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            first_provider = Vision()
            first = enrich_media(first_provider, inventory(image), project)
            result = copy.deepcopy(first["evidence"][0]["ai_media"])
            planner_input = planner._media_inventory(first)
            saved = checkpoints(project)
            self.assertEqual(len(first_provider.calls), 1)
            self.assertEqual(len(saved), 1)
            record = json_load(saved[0])
            self.assertEqual(record, {"version": 1, "input_sha256": saved[0].stem,
                                      "result": result})

            unavailable = Vision(failure=True, url="http://127.0.0.1:9900")
            second = enrich_media(unavailable, inventory(image), project)
            self.assertEqual(unavailable.calls, [])
            self.assertEqual(second["evidence"][0]["ai_media"], result)
            self.assertEqual(planner._media_inventory(second), planner_input)
            self.assertNotIn("ai_media_error", second["evidence"][0])
            self.assertEqual(json_load(project / "manifests/evidence.json"), second)
            self.assertEqual(set(planner_input[0]), {"ref", "relative_path", "kind",
                                                     "mime", "media", "ai_media", "evidence_role"})
            self.assertEqual(len(checkpoints(project)), 1)

            # The evidence ordinal is not part of what the vision model sees.
            other_ref = enrich_media(Vision(failure=True), inventory(image, ref="E0993"), project)
            self.assertEqual(other_ref["evidence"][0]["ai_media"], result)
            self.assertEqual(len(checkpoints(project)), 1)

    def test_provider_signature_changes_semantics_but_not_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            baseline = Vision()
            enrich_media(baseline, inventory(image), project)
            self.assertEqual(len(baseline.calls), 1)
            endpoint = Vision(url="http://127.0.0.1:9999", failure=True)
            self.assertEqual(endpoint.visual_cache_signature(), baseline.visual_cache_signature())
            self.assertIn("ai_media", enrich_media(endpoint, inventory(image), project)["evidence"][0])
            self.assertEqual(endpoint.calls, [])
            for candidate in (Vision(model="another-model"), Vision(temperature=0.25),
                              Vision(max_output_tokens=513), OtherVision()):
                with self.subTest(signature=candidate.visual_cache_signature()):
                    self.assertNotEqual(candidate.visual_cache_signature(),
                                        baseline.visual_cache_signature())
                    enrich_media(candidate, inventory(image), project)
                    self.assertEqual(len(candidate.calls), 1)
            self.assertEqual(len(checkpoints(project)), 5)
            first = OpenAICompatProvider("http://localhost:8080/v1", "model")
            moved = OpenAICompatProvider("http://localhost:9999/v1", "model")
            self.assertEqual(first.visual_cache_signature(), moved.visual_cache_signature())
            self.assertNotEqual(first.visual_cache_signature(),
                                OllamaProvider("http://localhost:8080", "model").visual_cache_signature())
            self.assertNotIn("url", first.visual_cache_signature())
            self.assertNotIn("timeout", first.visual_cache_signature())

    def test_prompt_instruction_and_contract_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, image = setup(root)
            enrich_media(Vision(), inventory(image), project)
            prompt = root / "prompts/media-inspection.txt"
            prompt.write_text("Describe only the new visual evidence.")
            changed_prompt = Vision()
            enrich_media(changed_prompt, inventory(image), project)
            self.assertEqual(len(changed_prompt.calls), 1)
            with patch.object(media_ai, "_INSPECTION_USER", "A different visual inspection request"):
                changed_user = Vision()
                enrich_media(changed_user, inventory(image), project)
                self.assertEqual(len(changed_user.calls), 1)
            with patch.object(media_ai, "MEDIA_INSPECTION_CONTRACT", "media-inspection-v2"):
                changed_contract = Vision()
                enrich_media(changed_contract, inventory(image), project)
                self.assertEqual(len(changed_contract.calls), 1)
            self.assertEqual(len(checkpoints(project)), 4)

    def test_bounded_image_bytes_not_source_metadata_or_filename_are_keyed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            first = Vision()
            original = enrich_media(first, inventory(image), project)["evidence"][0]["ai_media"]
            original_source_sha = sha256_file(image)
            png = PngImagePlugin.PngInfo()
            png.add_text("comment", "Different source bytes, same visible pixels")
            Image.new("RGB", (32, 24), (125, 20, 40)).save(image, pnginfo=png)
            self.assertNotEqual(sha256_file(image), original_source_sha)
            reused = Vision(failure=True)
            second = enrich_media(reused, inventory(image), project)
            self.assertEqual(reused.calls, [])
            self.assertEqual(second["evidence"][0]["ai_media"], original)
            Image.new("RGB", (32, 24), (20, 125, 40)).save(image)
            changed = Vision()
            third = enrich_media(changed, inventory(image), project)
            self.assertEqual(len(changed.calls), 1)
            self.assertNotEqual(changed.calls[0][2], first.calls[0][2])
            self.assertEqual(len(checkpoints(project)), 2)
            self.assertIn("ai_media", third["evidence"][0])

    def test_failure_never_authorizes_stale_success_or_overwrites_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            previous = enrich_media(Vision(), inventory(image), project)["evidence"][0]
            old_result = copy.deepcopy(previous["ai_media"])
            old_path = checkpoints(project)[0]
            old_bytes = old_path.read_bytes()
            Image.new("RGB", (32, 24), "blue").save(image)
            failing = Vision(failure=True)
            # Even a reused in-memory inventory must not carry stale authority.
            previous["path"] = str(image)
            result = enrich_media(failing, {"version": 1, "evidence": [previous]}, project)
            self.assertEqual(len(failing.calls), 1)
            self.assertNotIn("ai_media", result["evidence"][0])
            self.assertIn("local AI unavailable", result["evidence"][0]["ai_media_error"])
            self.assertEqual(checkpoints(project), [old_path])
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(json_load(old_path)["result"], old_result)

    def test_first_failure_or_partial_result_retries_after_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            failed = enrich_media(Vision(failure=True), inventory(image), project)
            self.assertNotIn("ai_media", failed["evidence"][0])
            self.assertIn("ai_media_error", failed["evidence"][0])
            self.assertEqual(checkpoints(project), [])
            partial = Vision(result={"category": "photo", "caption": "Incomplete"})
            incomplete = enrich_media(partial, inventory(image), project)
            self.assertNotIn("ai_media", incomplete["evidence"][0])
            self.assertEqual(len(partial.calls), 1)
            self.assertEqual(checkpoints(project), [])
            restored = Vision()
            success = enrich_media(restored, inventory(image), project)
            self.assertEqual(len(restored.calls), 1)
            self.assertEqual(len(checkpoints(project)), 1)
            cached = enrich_media(Vision(failure=True), inventory(image), project)
            self.assertEqual(cached["evidence"][0]["ai_media"],
                             success["evidence"][0]["ai_media"])

    def test_non_json_provider_result_cannot_become_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            invalid = Vision(result={"category": "photo", "caption": "Visible asset",
                                     "visible_text": [], "evidence_value": "high",
                                     "reasons": ["Visible project media"],
                                     "detail": {1: "non-JSON object key"}})
            result = enrich_media(invalid, inventory(image), project)
            self.assertNotIn("ai_media", result["evidence"][0])
            self.assertIn("ai_media_error", result["evidence"][0])
            self.assertEqual(checkpoints(project), [])

    def test_duplicate_inspection_content_shares_success_within_one_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            duplicate = project / "duplicate.png"
            shutil.copyfile(image, duplicate)
            entries = [inventory(image)["evidence"][0],
                       inventory(duplicate, ref="E0993", relative="duplicate.png")["evidence"][0]]
            provider = Vision()
            result = enrich_media(provider, {"version": 1, "evidence": entries}, project)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(result["evidence"][0]["ai_media"],
                             result["evidence"][1]["ai_media"])
            self.assertEqual(len(checkpoints(project)), 1)

    def test_corrupt_incomplete_wrong_version_or_digest_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, image = setup(Path(tmp))
            enrich_media(Vision(), inventory(image), project)
            path = checkpoints(project)[0]
            valid = json_load(path)
            corrupt = ("{bad json", json.dumps({**valid, "version": 2}),
                       json.dumps({**valid, "input_sha256": "wrong"}),
                       json.dumps({**valid, "result": {"caption": "partial"}}),
                       json.dumps({"version": 1, "input_sha256": path.stem}))
            for invalid in corrupt:
                with self.subTest(invalid=invalid[:30]):
                    path.write_text(invalid)
                    replacement = Vision()
                    result = enrich_media(replacement, inventory(image), project)
                    self.assertEqual(len(replacement.calls), 1)
                    self.assertEqual(json_load(path)["result"], result["evidence"][0]["ai_media"])
                    self.assertEqual(json_load(path)["input_sha256"], path.stem)
                    self.assertEqual(len(checkpoints(project)), 1)

    def test_svg_skip_and_priority_limit_preserve_inventory_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hero = setup(Path(tmp))
            hero.rename(project / "hero.png")
            hero = project / "hero.png"
            large = project / "large.png"
            small = project / "small.png"
            svg = project / "diagram.svg"
            Image.new("RGB", (80, 80), "green").save(large)
            Image.new("RGB", (4, 4), "blue").save(small)
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
            entries = [inventory(small, ref="E0001", relative="small.png", width=4, height=4)["evidence"][0],
                       inventory(large, ref="E0002", relative="large.png", width=80, height=80)["evidence"][0],
                       inventory(hero, ref="E0003", relative="hero.png")["evidence"][0],
                       inventory(svg, ref="E0004", relative="diagram.svg")["evidence"][0]]
            inv = {"version": 1, "evidence": entries}
            provider = Vision()
            result = enrich_media(provider, inv, project, max_items=2)
            self.assertEqual([entry["ref"] for entry in result["evidence"]],
                             ["E0001", "E0002", "E0003", "E0004"])
            self.assertEqual(len(provider.calls), 2)
            self.assertNotIn("ai_media", result["evidence"][0])
            self.assertIn("ai_media", result["evidence"][1])
            self.assertIn("ai_media", result["evidence"][2])
            self.assertNotIn("ai_media", result["evidence"][3])
            later = Vision()
            result = enrich_media(later, inv, project, max_items=4)
            self.assertEqual(len(later.calls), 1)  # Only the previously unselected small PNG.
            self.assertNotIn("ai_media", result["evidence"][3])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_video_keyframe_content_survives_path_change_and_detects_new_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, frame = setup(Path(tmp))
            video = project / "clip.mp4"
            def encode(color):
                Image.new("RGB", (32, 24), color).save(frame)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                                "-framerate", "2", "-i", str(frame), "-t", "2",
                                "-c:v", "mpeg4", str(video)], check=True)

            encode("red")
            first_provider = Vision()
            first = enrich_media(first_provider, inventory(video, relative="clip.mp4"), project)
            self.assertIn("ai_media", first["evidence"][0])
            self.assertEqual(len(first_provider.calls), 1)
            copy_path = project / "renamed.mp4"
            shutil.copyfile(video, copy_path)
            reused = Vision(failure=True)
            second = enrich_media(reused, inventory(copy_path, ref="E0993",
                                                    relative="renamed.mp4"), project)
            self.assertEqual(reused.calls, [])
            self.assertEqual(second["evidence"][0]["ai_media"], first["evidence"][0]["ai_media"])
            encode("blue")
            changed = Vision()
            third = enrich_media(changed, inventory(video, relative="clip.mp4"), project)
            self.assertEqual(len(changed.calls), 1)
            self.assertNotEqual(changed.calls[0][2], first_provider.calls[0][2])
            self.assertIn("ai_media", third["evidence"][0])
            self.assertEqual(len(checkpoints(project)), 2)

    def test_contracts_remain_scoped_and_asset_authority_is_preserved(self):
        self.assertEqual(MEDIA_INSPECTION_CONTRACT, "media-inspection-v1")
        self.assertEqual(research.RESEARCH_SEMANTICS_CONTRACT, "requested-topic-semantics-v7")
        self.assertEqual(EDITORIAL_CONTRACT, "storyboard-editorial-grounding-v5")
        self.assertEqual(GROUNDING_CONTRACT, "mapped-support-kind-v4")
        self.assertNotIn("media_inspection_contract",
                         research._payload(1, [], "BoringOS", "Explain the project"))
        self.assertNotIn("media_inspection_contract",
                         planner._compact_payload(1, 1, [], "BoringOS", "Explain the project"))
        ask = {
            "research": {
                "facts": [{"fact_id": "F0001", "evidence_refs": ["E0001"]}],
                "assets": [{"evidence_ref": "E0090"}],
            },
            "evidence_index": [
                {"ref": "E0001", "kind": "document", "relative_path": "source.md"},
                {"ref": "E0090", "kind": "media", "relative_path": "selected.png"},
                {"ref": "E0091", "kind": "media", "relative_path": "unrelated.png"},
            ],
        }
        intent = {"id": "s001", "type": "PROJECT_EVIDENCE", "fact_ids": ["F0001"],
                  "evidence_refs": ["E0001", "E0091"], "asset_ref": "E0091"}
        with self.assertRaisesRegex(ValueError,
                                    "asset_ref must be a renderable, scoped selected fact or research asset"):
            planner._canonical_evidence_refs(intent, ask, {"E0001", "E0090", "E0091"})


if __name__ == "__main__":
    unittest.main()
