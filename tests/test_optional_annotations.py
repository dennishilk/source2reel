"""Optional scene annotations can be omitted without weakening required grounding."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from source2reel import planner, research
from source2reel.editorial import EDITORIAL_CONTRACT, deterministic_field_guard
from source2reel.grounding import GROUNDING_CONTRACT, deterministic_decision
from source2reel.util import json_dump, json_load


PHYSICAL_CLAIM = (
    "The M63 physical acceptance proved a real writable USB-root lifecycle on "
    "Cthulhu: create and write files, reboot, boot again, read the persisted "
    "data, and then cleanly power off through ACPI S5."
)
PARAPHRASE = "M63 proved a writable USB-root lifecycle on Cthulhu."
UNSUPPORTED = "NVMe-powered persistence layer"


def physical_scope():
    claims = ["The first milestone is documented.",
              "The second milestone is documented.",
              "The third milestone is documented.", PHYSICAL_CLAIM]
    facts = [{"claim": claim, "evidence_refs": ["E0036"],
              "support": [{"evidence_ref": "E0036", "text": claim}],
              "phase": "final", "confidence": "high"} for claim in claims]
    evidence = [
        {"ref": "E0036", "kind": "document", "relative_path": "docs/m63.md",
         "evidence_role": "primary"},
        {"ref": "E0993", "kind": "media", "relative_path": "images/m63.png",
         "evidence_role": "primary"},
    ]
    media = [{"ref": "E0993", "kind": "media", "relative_path": "images/m63.png"}]
    assets = [{"evidence_ref": "E0993", "purpose": "Authentic hardware photo"}]
    ask = planner._make_ask({"version": 1, "facts": facts, "assets": assets},
                            media, evidence, "BoringOS", "Explain the physical acceptance.")
    return ask, {"E0036", "E0993"}


def physical_outline():
    return {"version": 1, "title": "BoringOS", "slug": "boringos",
            "summary": PHYSICAL_CLAIM, "scene_intents": [{
                "id": "s003", "type": "HARDWARE_EVIDENCE",
                "purpose": "Plan a hardware evidence scene",
                "fact_ids": ["F0004"], "evidence_refs": ["E0036", "E0993"],
                "asset_ref": "E0993",
            }]}


def physical_scene(annotations):
    return {"id": "s003", "type": "HARDWARE_EVIDENCE", "title": "Writable USB root",
            "narration": PHYSICAL_CLAIM, "fact_ids": ["F0004"],
            "evidence_refs": ["E0036", "E0993"], "asset_ref": "E0993",
            "annotations": annotations}


class AnnotationVerifier:
    def __init__(self, result="approve"):
        self.result = result
        self.calls = []

    def complete_json(self, _system, user):
        request = json.loads(user)
        self.calls.append(request)
        if self.result == "unavailable":
            raise RuntimeError("verifier unavailable")
        if self.result == "malformed":
            return {"decisions": []}
        return {"decisions": [{"id": check["id"],
                               "supported": self.result == "approve",
                               "propositions": [{"text": proposition, "support_indices": [0]}
                                                for proposition in check["required_propositions"]]}
                              for check in request["checks"]]}


class OptionalAnnotationTests(unittest.TestCase):
    def test_exact_physical_scene_prunes_only_unsupported_annotation(self):
        ask, allowed = physical_scope()
        outline = planner._normalize_outline(physical_outline(), allowed, ask,
                                             check_coverage=False)
        original = physical_scene([UNSUPPORTED])
        intent = outline["scene_intents"][0]
        self.assertEqual(intent["type"], "HARDWARE_EVIDENCE")
        self.assertEqual(intent["fact_ids"], ["F0004"])
        self.assertEqual(intent["evidence_refs"], ["E0036", "E0993"])
        self.assertEqual(intent["asset_ref"], "E0993")
        self.assertFalse(deterministic_field_guard(UNSUPPORTED, [PHYSICAL_CLAIM]))

        result = planner._normalize_scene_part({"scenes": [original]}, [intent],
                                                allowed, outline, ask)
        scene = result["scenes"][0]
        self.assertEqual(scene, {**original, "annotations": []})
        planner._validate_structured_grounding(None, result["scenes"], ask, None)
        self.assertEqual(original["annotations"], [UNSUPPORTED])

    def test_mixed_annotations_keep_exact_and_verified_paraphrase_in_order(self):
        ask, allowed = physical_scope()
        outline = planner._normalize_outline(physical_outline(), allowed, ask,
                                             check_coverage=False)
        annotation = {"text": PARAPHRASE, "kind": "label"}
        scene = physical_scene([PHYSICAL_CLAIM, None, UNSUPPORTED, annotation,
                                {}, {"text": ""}, {"text": 5}, False])
        self.assertTrue(deterministic_field_guard(PARAPHRASE, [PHYSICAL_CLAIM]))
        self.assertEqual(deterministic_decision(PARAPHRASE, [PHYSICAL_CLAIM]), "verify")
        provider = AnnotationVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            canonical = planner._normalize_scene_part({"scenes": [scene]},
                outline["scene_intents"], allowed, outline, ask, provider, Path(tmp))
        self.assertEqual(canonical["scenes"][0]["annotations"], [PHYSICAL_CLAIM, annotation])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual([check["claim"] for check in provider.calls[0]["checks"]],
                         [PARAPHRASE])

    def test_optional_verifier_rejection_or_unavailability_omits_paraphrase(self):
        ask, _ = physical_scope()
        for result in ("reject", "malformed", "unavailable"):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as tmp:
                scene = physical_scene([PHYSICAL_CLAIM, PARAPHRASE])
                provider = AnnotationVerifier(result)
                planner._validate_structured_grounding(provider, [scene], ask, Path(tmp))
                self.assertEqual(scene["annotations"], [PHYSICAL_CLAIM])
                self.assertEqual(len(provider.calls), 1)

        scene = physical_scene([PHYSICAL_CLAIM, PARAPHRASE])
        planner._validate_structured_grounding(None, [scene], ask, None)
        self.assertEqual(scene["annotations"], [PHYSICAL_CLAIM])
        scene["annotations"] = {"text": PHYSICAL_CLAIM}
        planner._validate_structured_grounding(None, [scene], ask, None)
        self.assertEqual(scene["annotations"], [])

        other = "The M63 physical acceptance proved a writable USB-root lifecycle."
        self.assertEqual(deterministic_decision(other, [PHYSICAL_CLAIM]), "verify")
        with tempfile.TemporaryDirectory() as tmp:
            scene = physical_scene([PHYSICAL_CLAIM, PARAPHRASE, other])
            provider = AnnotationVerifier("malformed")
            planner._validate_structured_grounding(provider, [scene], ask, Path(tmp))
            self.assertEqual(scene["annotations"], [PHYSICAL_CLAIM])
            self.assertEqual(len(provider.calls), 1)  # No verifier split/retry for decoration.

    def test_required_structured_fields_remain_fatal(self):
        def reject(claim, kind, diagram, message, support=None):
            fact = {"fact_id": "F0001", "claim": claim,
                    "support": [{"evidence_ref": "E0001", "text": support or claim}]}
            scene = {"id": "s001", "type": kind, "fact_ids": ["F0001"],
                     "annotations": [UNSUPPORTED], "diagram": diagram}
            with self.assertRaisesRegex(ValueError, message):
                planner._validate_structured_grounding(None, [scene],
                    {"research": {"facts": [fact]}}, None)

        reject("Component A sends events to Component B.", "ARCHITECTURE_DIAGRAM",
               {"nodes": ["Component A sends events to Component B.",
                          "Undocumented network controller"]},
               r"diagram\.nodes\[1\] has unsupported")
        reject("ETW callbacks normalize raw provider data into small internal events and enqueue them.",
               "DATA_FLOW", {"steps": [
                   "ETW callbacks normalize raw provider data into small internal events and enqueue them.",
                   "Uploads to cloud analytics"]},
               r"diagram\.steps\[1\] has unsupported")
        reject("The measured series records x 1 with y 10 and x 2 with y 20.",
               "GRAPH", {}, "GRAPH requires at least two grounded numerical points")
        reject("The measured series records x 1 with y 10 and x 2 with y 20.",
               "GRAPH", {"points": [[1, 10], [2, 25]]}, "numbers absent from selected facts")
        reject("The collector calls enqueue(event).", "CODE",
               {"code": "delete_all()"}, "diagram.code requires an exact selected source excerpt",
               support="enqueue(event)")

        claim = "ETW providers send normalized events to the core capture worker."
        paraphrase = "ETW providers send events to the core capture worker."
        self.assertTrue(deterministic_field_guard(paraphrase, [claim]))
        self.assertEqual(deterministic_decision(paraphrase, [claim]), "verify")
        scene = {"id": "s001", "type": "ARCHITECTURE_DIAGRAM",
                 "fact_ids": ["F0001"], "annotations": [UNSUPPORTED],
                 "diagram": {"nodes": [claim, paraphrase]}}
        fact = {"fact_id": "F0001", "claim": claim,
                "support": [{"evidence_ref": "E0001", "text": claim}]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, r"diagram\.nodes\[1\] has unsupported"):
                planner._validate_structured_grounding(
                    AnnotationVerifier("unavailable"), [scene],
                    {"research": {"facts": [fact]}}, Path(tmp))

    def test_multipart_single_scene_and_checkpoint_reuse(self):
        ask, allowed = physical_scope()
        raw = physical_outline()

        class Provider:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                if request.get("storyboard_mode") == "outline":
                    return copy.deepcopy(raw)
                if request.get("storyboard_mode") == "scenes":
                    return {"scenes": [physical_scene([UNSUPPORTED])]}
                if "checks" in request:
                    return AnnotationVerifier().complete_json(_system, user)
                raise AssertionError("Unexpected full episode request")

        provider = Provider()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def run():
                return planner._multipart_episode(provider, "storyboard", ask, path,
                    32768, 4096, 1024, 0, None, "physical-scope")

            episode = run()
            self.assertEqual(episode["scenes"], [physical_scene([])])
            self.assertEqual([call.get("storyboard_mode") for call in provider.calls],
                             ["outline", "scenes"])
            part = path / "manifests/storyboard-parts/part-001.json"
            cached = json_load(part)
            self.assertEqual(cached["result"]["scenes"], [physical_scene([])])
            # A v4 cached part is re-normalized under the new code even when
            # its input hash and both existing contracts remain unchanged.
            cached["result"]["scenes"][0]["annotations"] = [UNSUPPORTED]
            json_dump(part, cached)
            self.assertEqual(run(), episode)
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(json_load(part)["input_sha256"], cached["input_sha256"])

    def test_full_response_uses_the_same_omission(self):
        ask, allowed = physical_scope()
        raw = {"version": 1, "title": "BoringOS",
               "scenes": [physical_scene([UNSUPPORTED])]}

        class FullProvider:
            def __init__(self):
                self.calls = []

            def complete_json(self, _system, user):
                request = json.loads(user)
                self.calls.append(request)
                if "checks" in request:
                    return AnnotationVerifier().complete_json(_system, user)
                return copy.deepcopy(raw)

        provider = FullProvider()
        episode = planner._complete_episode(provider, "storyboard", ask, allowed, 0)
        self.assertEqual(episode["scenes"], [physical_scene([])])
        self.assertEqual(len(provider.calls), 1)

    def test_checkpoint_contracts_remain_scoped(self):
        ask, _ = physical_scope()
        self.assertEqual(ask["editorial_contract"], EDITORIAL_CONTRACT)
        self.assertEqual(EDITORIAL_CONTRACT, "storyboard-editorial-grounding-v6")
        self.assertEqual(GROUNDING_CONTRACT, "mapped-support-kind-v5")
        research_request = research._payload(1, [], "BoringOS", "Physical acceptance")
        self.assertEqual(research_request["research_semantics_contract"],
                         "requested-topic-semantics-v18")
        self.assertNotIn("editorial_contract", research_request)
        self.assertNotIn("editorial_contract",
                         planner._compact_payload(1, 1, [], "BoringOS", "Physical acceptance"))


if __name__ == "__main__":
    unittest.main()
