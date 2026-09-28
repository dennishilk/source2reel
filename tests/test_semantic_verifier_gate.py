"""Physical-shape regressions for claim/support syntax and verifier mappings."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from source2reel.grounding import (GROUNDING_CONTRACT, _propositions,
                                   deterministic_decision, verify_claims)
from source2reel.planner import _repair_episode_shape
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.research import research


class AgreeingVerifier:
    """Simulate the physically observed false-positive verifier verdict."""
    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        checks = json.loads(user)['checks']
        return {'decisions': [{
            'id': check['id'], 'supported': True,
            'propositions': [{'text': unit, 'support_indices': [0]}
                             for unit in check['required_propositions']],
        } for check in checks]}


class SupportKindRegressions(unittest.TestCase):
    def verify(self, claim, support, provider=None):
        provider = provider or AgreeingVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            accepted = verify_claims(provider, [{'id': 'C0001', 'claim': claim,
                                                   'support': support}], Path(tmp), 'research')
        return accepted, provider

    def test_imports_cannot_establish_behavior_or_architecture(self):
        examples = [
            ('WidgetEngine validates episode structure using its schema module.',
             ['from widget.schema import validate_episode']),
            ("WidgetEngine's planner maps evidence to scene intents.",
             ['from widget.planner import plan']),
            ('WidgetEngine includes a pronunciation engine for text-to-speech conversion.',
             ['from widget.tts_engine import apply_pronunciations']),
            ('WidgetEngine is a local evidence-first research pipeline that turns project '
             'sources into technical explainer videos.',
             ['from widget.research import research',
              'research(provider, {"evidence": evidence}, project)']),
            ("WidgetEngine's pipeline includes research, inventory building and output "
             'generation stages.',
             ['from widget.research import research', 'from widget.inventory import build_inventory',
              'from widget.output import create']),
        ]
        for claim, spans in examples:
            with self.subTest(claim=claim):
                accepted, provider = self.verify(claim, spans)
                self.assertEqual(accepted, set())
                self.assertEqual(provider.calls, 0)

    def test_true_verifier_cannot_override_code_only_gate(self):
        claim = 'WidgetEngine validates episode structure using its schema module.'
        accepted, provider = self.verify(claim, ['from widget.schema import validate_episode'],
                                         AgreeingVerifier())
        self.assertEqual(accepted, set())
        self.assertEqual(provider.calls, 0)

    def test_scene_cannot_expand_syntactic_source_fact_into_behavior(self):
        claim = 'The module validates episode structure using validate_episode.'
        item = {'id': 's001', 'claim': claim, 'facts': [{
            'claim': 'The module imports validate_episode.',
            'support': [{'evidence_ref': 'E0001',
                         'text': 'from widget.schema import validate_episode'}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            accepted = verify_claims(AgreeingVerifier(), [item], Path(tmp), 'scene')
        self.assertEqual(accepted, set())

    def test_visible_code_operation_is_eligible_but_not_a_system_guarantee(self):
        code = 'result = validate_episode(ep)'
        self.assertIn(deterministic_decision('This code calls validate_episode.', [code]),
                      {'verify', 'accept'})
        self.assertEqual(deterministic_decision(
            'The system guarantees structural consistency across all outputs.', [code]), 'reject')
        self.assertIn(deterministic_decision('The file imports validate_episode.',
                                            ['from widget.schema import validate_episode']),
                      {'verify', 'accept'})

    def test_explicit_comment_can_support_purpose_and_prose_paraphrases_survive(self):
        comment = '# The planner maps selected evidence to scene intents.'
        self.assertEqual(deterministic_decision('The planner maps selected evidence to scene intents.',
                                                [comment]), 'accept')
        claim = 'CUDA is not auto-installed by the baseline.'
        accepted, provider = self.verify(claim,
            ['The baseline does not automatically install CUDA.'])
        self.assertEqual(accepted, {'C0001'})
        self.assertEqual(provider.calls, 1)

    def test_missing_enumeration_member_rejects_whole_claim(self):
        source = ('The voice settings are configurable by language, voice and speed. '
                  'The worker accepts those three options.')
        bad = 'The voice settings are configurable by language, voice, speed, and sample rate.'
        good = 'The voice settings are configurable by language, voice, and speed.'
        self.assertEqual(deterministic_decision(bad, [source]), 'reject')
        self.assertIn(deterministic_decision(good, [source]), {'accept', 'verify'})
        self.assertEqual(self.verify(bad, [source])[0], set())
        self.assertEqual(self.verify(good, [source])[0], {'C0001'})
        code = 'KPipeline(lang_code=args.language_code)\npipeline(text, voice=args.voice, speed=args.speed)'
        self.assertEqual(deterministic_decision(bad, [code]), 'reject')

    def test_short_declarative_support_reaches_semantic_verifier(self):
        claim = 'The tool supports PulseAudio, PipeWire, and ALSA audio systems.'
        support = ['Supports **PulseAudio**, **PipeWire**, and **ALSA**.']
        self.assertEqual(deterministic_decision(claim, support), 'verify')
        accepted, provider = self.verify(claim, support)
        self.assertEqual(accepted, {'C0001'})
        self.assertEqual(provider.calls, 1)

    def test_spdif_readme_paraphrase_reaches_semantic_verifier(self):
        claim = ('The SPDIF Fix Tool is an interactive Bash script designed to fix SPDIF '
                 'audio delay and standby issues on Linux systems.')
        support = ['A simple interactive Bash tool to **fix SPDIF sound delay & standby issues on Linux**.']
        self.assertEqual(deterministic_decision(claim, support), 'verify')
        accepted, provider = self.verify(claim, support)
        self.assertEqual(accepted, {'C0001'})
        self.assertEqual(provider.calls, 1)

    def test_direct_shell_operations_are_verifiable_but_purpose_is_not(self):
        pipewire = (
            'if pgrep -x "pipewire" > /dev/null; then\n'
            'CONFIG_FILE="$CONFIG_DIR/pipewire.conf"\n'
            'sed -i \'/suspend-on-idle/s/^/#/\' "$CONFIG_FILE"'
        )
        restart = (
            'pulseaudio -k\n'
            'pulseaudio --start\n'
            'systemctl --user restart pipewire pipewire-pulse || true'
        )
        cases = [
            ('The tool disables PipeWire suspend-on-idle by commenting the matching '
             'line in pipewire.conf.', [pipewire]),
            ('The tool restarts PulseAudio and PipeWire.', [restart]),
        ]
        for claim, support in cases:
            with self.subTest(claim=claim):
                self.assertEqual(deterministic_decision(claim, support), 'verify')
                accepted, provider = self.verify(claim, support)
                self.assertEqual(accepted, {'C0001'})
                self.assertEqual(provider.calls, 1)

        unsupported = 'The tool restarts PulseAudio and PipeWire to prevent audio dropouts.'
        self.assertEqual(deterministic_decision(unsupported, [restart]), 'reject')
        accepted, provider = self.verify(unsupported, [restart])
        self.assertEqual(accepted, set())
        self.assertEqual(provider.calls, 0)

    def test_mapping_must_cover_each_proposition_and_use_real_local_index(self):
        claim = ('The baseline does not automatically install CUDA. '
                 'The separate voice environment installs Torch.')
        spans = ['The baseline does not automatically install CUDA.',
                 'The separate voice environment installs Torch.']
        self.assertEqual(len(_propositions(claim)), 2)

        class BadMapping:
            def __init__(self, mode):
                self.mode = mode
            def complete_json(self, system, user):
                check = json.loads(user)['checks'][0]
                propositions = [{'text': unit, 'support_indices': [index]}
                                for index, unit in enumerate(check['required_propositions'])]
                if self.mode == 'missing':
                    propositions.pop()
                elif self.mode == 'invalid':
                    propositions[-1]['support_indices'] = [99]
                elif self.mode == 'wrong-span':
                    propositions[-1]['support_indices'] = [0]
                return {'decisions': [{'id': check['id'], 'supported': True,
                                       'propositions': propositions}]}

        for mode in ('missing', 'invalid', 'wrong-span'):
            with self.subTest(mode=mode):
                self.assertEqual(self.verify(claim, spans, BadMapping(mode))[0], set())
        self.assertEqual(self.verify(claim, spans, BadMapping('valid'))[0], {'C0001'})

    def test_boolean_only_malformed_timeout_and_output_limit_fail_closed(self):
        claim = 'CUDA is not auto-installed by the baseline.'
        support = ['The baseline does not automatically install CUDA.']

        class Faulty:
            def __init__(self, mode):
                self.mode = mode
            def complete_json(self, system, user):
                if self.mode == 'boolean':
                    return {'decisions': [{'id': 'C0001', 'supported': True}]}
                if self.mode == 'timeout':
                    raise TimeoutError('synthetic timeout')
                if self.mode == 'limit':
                    raise OutputLimitExceeded('output token limit')
                raise StructuredOutputError('malformed response')

        for mode in ('boolean', 'timeout', 'limit', 'malformed'):
            with self.subTest(mode=mode):
                self.assertEqual(self.verify(claim, support, Faulty(mode))[0], set())
        self.assertNotEqual(GROUNDING_CONTRACT, 'quoted-proposition-v1')

    def test_order_preserving_evidence_ref_deduplication(self):
        episode = {'scenes': [{'type': 'SUMMARY', 'evidence_refs':
                               ['E0001', 'E0001', 'E0002', 'E0001']}]}
        repaired = _repair_episode_shape(episode)
        self.assertEqual(repaired['scenes'][0]['evidence_refs'], ['E0001', 'E0002'])


class ResearchPriorityRegressions(unittest.TestCase):
    def test_requested_topics_survive_six_fact_per_ref_cap_without_invention(self):
        noisy = [f'WidgetEngine uses a voice setting named preset {index}.' for index in range(20)]
        overview = 'WidgetEngine is a tool for turning project sources into explanations.'
        purpose = 'WidgetEngine was designed to help teams explain decisions using source evidence.'
        workflow = 'The pipeline transforms project sources into a narrated episode.'

        def run(include_purpose=True):
            claims = [*noisy, overview, workflow]
            if include_purpose:
                claims.append(purpose)
            inventory = {'evidence': [{'ref': 'E0001', 'kind': 'document',
                                        'relative_path': 'OVERVIEW.md', 'evidence_role': 'primary',
                                        'excerpt': '\n'.join(claims)}]}

            class Provider:
                def complete_json(self, system, user):
                    data = json.loads(user)
                    if 'checks' in data:
                        return {'decisions': []}
                    return {'facts': [{'claim': c, 'evidence_refs': ['E0001'],
                                      'support': [{'evidence_ref': 'E0001', 'text': c}],
                                      'phase': 'final', 'confidence': 'high'} for c in claims],
                            'assets': []}

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'prompts').mkdir()
                (root / 'prompts' / 'research.txt').write_text('Select grounded episode facts.')
                return research(Provider(), inventory, root / 'projects' / 'widget',
                                title_hint='WidgetEngine', instructions=(
                                    'Explain what WidgetEngine is, why it exists, and how its '
                                    'pipeline turns project sources into an episode.'))

        selected = [f['claim'] for f in run()['facts']]
        self.assertEqual(len(selected), 6)
        self.assertTrue({overview, purpose, workflow} <= set(selected))
        no_purpose = [f['claim'] for f in run(False)['facts']]
        self.assertNotIn(purpose, no_purpose)
        self.assertEqual(len(no_purpose), 6)


if __name__ == '__main__':
    unittest.main()
