"""Physical regression for code-heavy requested research recovery."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel.research import _repair_support_span, research
from source2reel.util import json_load


README = """# SPDIF Fix Tool

A simple interactive Bash tool to fix SPDIF sound delay and standby issues on Linux.
Supports PulseAudio, PipeWire, and ALSA.
It was designed to address delayed audio after SPDIF playback resumes.
"""

SCRIPT = """#!/usr/bin/env bash
# SPDIF Sound Delay Fix Tool
""" + "\n".join(f"# implementation note {i}: keep exact source context" for i in range(30)) + """
restart_soundserver() {
    if pgrep -x "pulseaudio" > /dev/null; then
        pulseaudio -k
        pulseaudio --start
    fi
    if pgrep -x "pipewire" > /dev/null; then
        systemctl --user restart pipewire pipewire-pulse || true
    fi
}
if [ "$CHOICE" = "1" ]; then
    pactl unload-module module-suspend-on-idle 2>/dev/null || true
    sudo cp /etc/pulse/default.pa /etc/pulse/default.pa.bak
    sudo sed -i '/module-suspend-on-idle/s/^/#/' /etc/pulse/default.pa
    CONFIG_DIR="$HOME/.config/pipewire"
    CONFIG_FILE="$CONFIG_DIR/pipewire.conf"
    mkdir -p "$CONFIG_DIR"
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
    sed -i '/suspend-on-idle/s/^/#/' "$CONFIG_FILE"
    ALSA_FILE="$HOME/.asoundrc"
    cat << 'EOF' > "$ALSA_FILE"
pcm.spdif_keepalive {
    type plug
    slave.pcm "spdif"
}
EOF
elif [ "$CHOICE" = "2" ]; then
    sudo mv /etc/pulse/default.pa.bak /etc/pulse/default.pa
    pactl load-module module-suspend-on-idle 2>/dev/null || true
    CONFIG_FILE="$HOME/.config/pipewire/pipewire.conf"
    mv "$CONFIG_FILE.bak" "$CONFIG_FILE"
    ALSA_FILE="$HOME/.asoundrc"
    rm "$ALSA_FILE"
fi
"""

HELPER = """#!/usr/bin/env bash
# wait for pipewire
sleep 3
play -n -c2 synth sin gain -100
"""

INSTRUCTIONS = (
    "Explain what this project solves, why SPDIF standby or suspend can cause delayed "
    "audio after playback resumes, how the tool detects and handles PulseAudio, PipeWire, "
    "and ALSA, what changes it makes, how backup and reset work, and what the fallback "
    "autostart helper does. Stay strictly grounded in the repository."
)


def entry(ref: str, path: str, excerpt: str) -> dict:
    return {"ref": ref, "kind": "document", "relative_path": path,
            "evidence_role": "primary", "excerpt": excerpt}


def fact(claim: str, ref: str, support: str) -> dict:
    return {"claim": claim, "evidence_refs": [ref],
            "support": [{"evidence_ref": ref, "text": support}],
            "phase": "final", "confidence": "high"}


class SpdifProvider:
    def __init__(self, *, omit_coverage_code: bool = False):
        self.omit_coverage_code = omit_coverage_code
        self.research_calls = 0
        self.coverage_calls: list[dict] = []
        self.verifier_calls = 0

    def complete_json(self, _system: str, user: str) -> dict:
        payload = json.loads(user)
        if "checks" in payload:
            self.verifier_calls += 1
            return {"decisions": [{
                "id": check["id"], "supported": True,
                "propositions": [
                    {"text": proposition, "support_indices": [0]}
                    for proposition in check["required_propositions"]
                ],
            } for check in payload["checks"]]}
        if payload.get("research_mode") == "requested_coverage":
            self.coverage_calls.append(payload)
            facts = []
            by_path = {item["relative_path"]: item for item in payload["evidence"]}
            if "README.md" in by_path:
                facts.append(fact(
                    "The tool was designed to address delayed audio after SPDIF playback resumes.",
                    by_path["README.md"]["ref"],
                    "It was designed to address delayed audio after SPDIF playback resumes.",
                ))
            if not self.omit_coverage_code:
                script = next((item for item in payload["evidence"]
                               if item["relative_path"] in {"run.sh", "spdif-fix.sh"}), None)
                helper = by_path.get("soundfixforautostart.sh")
                if script is not None:
                    facts.extend([
                        fact("The script restarts PulseAudio and PipeWire.",
                             script["ref"], script["excerpt"]),
                        fact("The script comments PipeWire suspend-on-idle in its configuration.",
                             script["ref"], script["excerpt"]),
                    ])
                if helper is not None:
                    facts.append(fact(
                        "The fallback autostart helper runs the play command.",
                        helper["ref"], "play -n -c2 synth sin gain -100",
                    ))
            return {"facts": facts, "assets": []}
        self.research_calls += 1
        return {"facts": [
            fact("The SPDIF Fix Tool is an interactive Bash tool for SPDIF delay and standby issues.",
                 "E0002",
                 "A simple interactive Bash tool to fix SPDIF sound delay and standby issues on Linux."),
            fact("The tool supports PulseAudio, PipeWire, and ALSA.",
                 "E0002", "Supports PulseAudio, PipeWire, and ALSA."),
        ], "assets": []}


class SpdifResearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "prompts").mkdir()
        (root / "prompts" / "research.txt").write_text("Use only supplied evidence.")
        self.project = root / "projects" / "spdif-fix-guided"
        self.inventory = {"evidence": [
            entry("E0002", "README.md", README),
            entry("E0004", "run.sh", SCRIPT),
            entry("E0008", "soundfixforautostart.sh", HELPER),
            entry("E0009", "spdif-fix.sh", SCRIPT),
        ]}

    def run(self, provider):
        return research(provider, self.inventory, self.project,
                        title_hint="spdif-fix", instructions=INSTRUCTIONS,
                        max_retries=0)

    def test_code_coverage_recovers_natural_facts_and_deduplicates_duplicate_script(self):
        self.assertGreater(len(SCRIPT), 1024)
        provider = SpdifProvider()
        result = self.run(provider)
        claims = [item["claim"] for item in result["facts"]]
        self.assertIn("The script restarts PulseAudio and PipeWire.", claims)
        self.assertIn("The script comments PipeWire suspend-on-idle in its configuration.", claims)
        self.assertIn("The fallback autostart helper runs the play command.", claims)
        restart = next(item for item in result["facts"]
                       if item["claim"] == "The script restarts PulseAudio and PipeWire.")
        self.assertLessEqual(len(restart["support"][0]["text"]), 1024)
        source = next(item["excerpt"] for item in self.inventory["evidence"]
                      if item["ref"] == restart["evidence_refs"][0])
        self.assertIn(restart["support"][0]["text"], source)

        self.assertEqual(len(provider.coverage_calls), 1)
        refs = [item["ref"] for item in provider.coverage_calls[0]["evidence"]]
        self.assertIn("E0002", refs)
        self.assertIn("E0008", refs)
        self.assertEqual(len({"E0004", "E0009"} & set(refs)), 1)

        saved = json_load(self.project / "manifests" / "research.json")
        self.assertEqual(saved, result)

    def test_exact_code_floor_survives_even_when_focused_model_omits_code_facts(self):
        provider = SpdifProvider(omit_coverage_code=True)
        result = self.run(provider)
        claims = [item["claim"] for item in result["facts"]]
        self.assertIn("systemctl --user restart pipewire pipewire-pulse || true", claims)
        self.assertIn("play -n -c2 synth sin gain -100", claims)
        self.assertTrue(any(".bak" in claim for claim in claims))
        exact = [item for item in result["facts"] if item.get("direct_code_evidence")]
        self.assertGreaterEqual(len(exact), 3)
        for item in exact:
            self.assertEqual(item["claim"], item["support"][0]["text"])
            source = next(entry["excerpt"] for entry in self.inventory["evidence"]
                          if entry["ref"] == item["evidence_refs"][0])
            self.assertIn(item["claim"], source)

    def test_support_repair_never_redirects_fabricated_text(self):
        claim = "The script restarts PulseAudio and PipeWire."
        self.assertIsNone(_repair_support_span(
            claim, "invented cloud restart operation", SCRIPT,
        ))


if __name__ == "__main__":
    unittest.main()
