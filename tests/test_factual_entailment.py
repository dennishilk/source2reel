"""Provenance must survive research, compaction, and both storyboard paths."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source2reel.grounding import deterministic_decision, verify_claims
from source2reel.planner import (_complete_episode, _make_ask, _normalize_capsules,
                                 _normalize_outline, _primary_anchors, plan)
from source2reel.providers import OutputLimitExceeded, StructuredOutputError
from source2reel.research import research
from source2reel.util import json_load


class GroundingRegressions(unittest.TestCase):
    def test_exact_physical_bad_merged_claim_and_atomic_facts(self):
        lock = ('if grep -Eqi \'name = "torch"|nvidia[-_]|cuda|cudnn|nccl\' uv.lock; '
                'then echo "ERROR: core uv.lock contains Torch/NVIDIA/CUDA runtime '
                'entries; refusing baseline sync." >&2; exit 3; fi')
        voice = 'Kokoro/Torch is installed in a separate optional voice environment by tools/setup-voice.sh.'
        merged = ('The baseline does not include Torch, CUDA, or NVIDIA proprietary '
                  'packages; these are installed only via setup-voice.sh.')
        self.assertEqual(deterministic_decision(merged, [lock]), "reject")
        self.assertEqual(deterministic_decision('The core lock rejects Torch/NVIDIA/CUDA runtime entries.', [lock]), "reject")
        self.assertEqual(deterministic_decision('Kokoro/Torch is installed in a separate optional voice environment.', [voice]), "accept")

    def test_exact_but_unsupported_and_technical_injections(self):
        span = 'This document records the first physical-deployment decision set for WidgetEngine.'
        claim = 'WidgetEngine is a local evidence-first system that turns project sources into finished videos.'
        self.assertEqual(deterministic_decision(claim, [span]), "reject")
        source = 'The baseline installs its core packages locally.'
        for added in ('setup-voice.sh', 'CUDA', 'Torch', 'Kokoro', 'API endpoint', 'version 7.14',
                      'https://example.test/api'):
            with self.subTest(added=added):
                self.assertEqual(deterministic_decision(source + ' It uses ' + added + '.', [source]), "reject")

    def test_clean_paraphrase_is_eligible_for_verification(self):
        support = 'The baseline does not install CUDA automatically.'
        self.assertNotEqual(deterministic_decision('CUDA is not installed automatically by the baseline.', [support]), "reject")

    def test_research_rejects_missing_propositions_in_real_normalizer(self):
        lock = ('The baseline does not auto-install CUDA, cuDNN, NCCL, ROCm or PyTorch. '
                'The core lock rejects Torch, CUDA and NVIDIA packages.')
        voice = 'Kokoro/Torch is installed separately in an optional voice environment by tools/setup-voice.sh.'
        evidence = [{"ref": "E0001", "kind": "document", "relative_path": "docs/setup.md",
                     "evidence_role": "primary", "excerpt": lock + '\n' + voice}]
        bad = {'claim': ('Source2Reel’s baseline does not include Torch, CUDA, or NVIDIA '
                         'proprietary packages; these are installed only via setup-voice.sh.'),
               'evidence_refs': ['E0001'], 'support': [{'evidence_ref': 'E0001', 'text': lock}]}
        good = [{'claim': lock, 'evidence_refs': ['E0001'],
                 'support': [{'evidence_ref': 'E0001', 'text': lock}]},
                {'claim': voice, 'evidence_refs': ['E0001'],
                 'support': [{'evidence_ref': 'E0001', 'text': voice}]}]

        class Provider:
            def complete_json(self, system, user):
                return {'facts': [bad, *good], 'assets': []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'prompts' / 'research.txt').write_text('Research evidence.')
            result = research(Provider(), {'evidence': evidence}, root / 'projects' / 'sample',
                              title_hint='WidgetEngine')
        self.assertEqual([fact['claim'] for fact in result['facts']], [fact['claim'] for fact in good])

    def test_irrelevant_exact_research_quote_cannot_support_an_engine_definition(self):
        span = 'This document records the first deployment decision set for WidgetEngine.'
        inventory = {'evidence': [{'ref': 'E0001', 'kind': 'document',
                                   'relative_path': 'project-state.md',
                                   'evidence_role': 'primary', 'excerpt': span}]}

        class Provider:
            def complete_json(self, system, user):
                return {'facts': [{'claim': ('WidgetEngine is a local evidence-first system '
                                            'that turns project sources into finished videos.'),
                                   'evidence_refs': ['E0001'],
                                   'support': [{'evidence_ref': 'E0001', 'text': span}]}],
                        'assets': []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'prompts' / 'research.txt').write_text('research')
            result = research(Provider(), inventory, root / 'projects' / 'sample',
                              title_hint='WidgetEngine')
        self.assertEqual(result['facts'], [])

    def test_planner_refuses_input_facts_without_attached_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'exact supporting spans'):
                plan(None, {'facts': [{'claim': 'A plausible fact.',
                                       'evidence_refs': ['E0001']}], 'assets': []},
                     {'evidence': [{'ref': 'E0001', 'kind': 'document',
                                    'relative_path': 'docs/a.txt',
                                    'excerpt': 'An unrelated fact.'}]},
                     Path(tmp) / 'projects' / 'sample', 'Example')

    def test_compaction_selects_original_claim_with_exact_support(self):
        source = {'kind': 'fact', 'source_fact_id': 'R0001', 'claim': 'Kokoro runs on CPU.',
                  'evidence_refs': ['E0001'], 'support': [{'evidence_ref': 'E0001',
                                                         'text': 'Kokoro runs on CPU.'}],
                  'phase': 'final', 'confidence': 'high'}
        invented = {'claim': 'Kokoro runs on CPU, which ensures all hardware is compatible.',
                    'source_fact_id': 'R0001', 'evidence_refs': ['E0001'], 'media_refs': []}
        value = _normalize_capsules({'capsules': [invented]}, {'E0001'}, set(), [source])
        self.assertEqual(value['capsules'][0]['claim'], source['claim'])
        self.assertEqual(value['capsules'][0]['support'], source['support'])
        self.assertEqual(value['capsules'][0]['source_fact_id'], 'R0001')

    def test_storyboard_fast_path_retries_causal_expansion_and_keeps_paraphrase(self):
        source = 'The baseline does not install CUDA automatically.'
        research_input = {'version': 1, 'facts': [{'claim': source, 'evidence_refs': ['E0001'],
                         'support': [{'evidence_ref': 'E0001', 'text': source}],
                         'phase': 'final', 'confidence': 'high'}], 'assets': []}
        index = [{'ref': 'E0001', 'relative_path': 'docs/setup.md', 'evidence_role': 'primary'}]
        ask = _make_ask(research_input, [], index, 'WidgetEngine', 'Explain the baseline.')

        class Provider:
            calls = 0
            def complete_json(self, system, user):
                self.calls += 1
                payload = json.loads(user)
                narration = (source if 'validation_feedback' in payload else
                             source + ' This ensures compatibility with standard hardware.')
                return {'version': 1, 'title': 'Baseline', 'scenes': [{
                    'id': 's001', 'type': 'SUMMARY', 'title': 'Baseline',
                    'narration': narration, 'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}

        with tempfile.TemporaryDirectory() as tmp:
            provider = Provider()
            episode = _complete_episode(provider, 'storyboard', ask, {'E0001'}, 1,
                                        project_dir=Path(tmp))
        self.assertEqual(provider.calls, 2)
        self.assertEqual(episode['scenes'][0]['narration'], source)

    def test_verifier_accepts_conservative_paraphrase_and_checkpoints_only_its_sources(self):
        span = 'The baseline does not install CUDA automatically.'
        item = {'id': 'C0001', 'claim': 'CUDA is not installed automatically by the baseline.',
                'support': [span]}

        class Provider:
            calls = []
            def complete_json(self, system, user):
                payload = json.loads(user)
                self.calls.append(payload)
                return {'decisions': [{'id': check['id'], 'supported': True,
                                      'propositions': [{'text': proposition, 'support_indices': [0]}
                                                       for proposition in check['required_propositions']]}
                                      for check in payload['checks']]}

        with tempfile.TemporaryDirectory() as tmp:
            provider = Provider()
            self.assertEqual(verify_claims(provider, [item], Path(tmp), 'research'), {'C0001'})
            self.assertEqual(verify_claims(provider, [item], Path(tmp), 'research'), {'C0001'})
            self.assertEqual(provider.calls, [{'checks': [{**item,
                'required_propositions': ['CUDA is not installed automatically by the baseline.']}]}])
            self.assertEqual(len(list((Path(tmp) / 'manifests' / 'grounding-parts').glob('*.json'))), 1)

    def test_verifier_malformed_or_incomplete_decisions_fail_closed(self):
        item = {'id': 'C0001', 'claim': 'CUDA is not installed automatically by the baseline.',
                'support': ['The baseline does not install CUDA automatically.']}

        class Malformed:
            def complete_json(self, system, user):
                return {'decisions': [{'id': 'C9999', 'supported': 'yes'}]}

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(verify_claims(Malformed(), [item], Path(tmp), 'research'), set())
            self.assertFalse(list((Path(tmp) / 'manifests' / 'grounding-parts').glob('*.json')))

    def test_forced_map_reduce_cannot_promote_invented_compaction_consequence(self):
        facts = [f'Engine maps source group {i} to recorded evidence.' for i in range(1, 4)]
        inventory = {'evidence': [{'ref': f'E{i:04d}', 'kind': 'document',
                                   'relative_path': f'docs/{i}.md',
                                   'evidence_role': 'primary', 'excerpt': fact}
                                  for i, fact in enumerate(facts, 1)]}
        research_data = {'version': 1, 'facts': [
            {'claim': claim, 'evidence_refs': [f'E{i:04d}'],
             'support': [{'evidence_ref': f'E{i:04d}', 'text': claim}],
             'phase': 'final', 'confidence': 'high'}
            for i, claim in enumerate(facts, 1)], 'assets': []}

        class Provider:
            def complete_json(self, system, user):
                payload = json.loads(user)
                if 'records' in payload:
                    return {'capsules': [
                        {'source_fact_id': record['source_fact_id'],
                         'claim': record['claim'] + ' This guarantees every project succeeds.',
                         'evidence_refs': record['evidence_refs'], 'media_refs': []}
                        for record in payload['records'] if record['kind'] == 'fact'
                    ]}
                fact = payload['research']['facts'][0]
                return {'version': 1, 'title': 'Engine', 'scenes': [{
                    'id': 's001', 'type': 'SUMMARY', 'narration': fact['claim'],
                    'fact_ids': [fact['fact_id']], 'evidence_refs': fact['evidence_refs']}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'prompts' / 'storyboard.txt').write_text('storyboard')
            (root / 'prompts' / 'planner_compact.txt').write_text('compact')
            project = root / 'projects' / 'sample'
            with patch('source2reel.planner.fits_context', side_effect=[False, True, True]), \
                 patch('source2reel.planner.split_for_context', side_effect=lambda records, *a, **kw: [records]):
                plan(Provider(), research_data, inventory, project, 'Engine', max_retries=0)
            manifest = json_load(project / 'manifests' / 'planner-evidence.json')
        self.assertEqual(manifest['strategy'], 'map-reduce')
        for fact in manifest['research']['facts']:
            self.assertIn(fact['claim'], facts)
            self.assertEqual(fact['support'][0]['text'], fact['claim'])
            self.assertRegex(fact['source_fact_id'], r'^R\d{4}$')

    def test_fast_path_replaces_unsupported_narration_after_retries(self):
        claim = 'The core dependency graph excludes Torch.'
        ask = _make_ask({'facts': [{'claim': claim, 'evidence_refs': ['E0001'],
                        'support': [{'evidence_ref': 'E0001', 'text': claim}]}], 'assets': []},
                        [], [{'ref': 'E0001', 'evidence_role': 'primary'}], 'Engine', '')

        class Provider:
            def complete_json(self, system, user):
                return {'version': 1, 'title': 'Engine', 'scenes': [{
                    'id': 's001', 'type': 'SUMMARY', 'narration': claim +
                    ' This keeps the engine lightweight and accessible.',
                    'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}

        episode = _complete_episode(Provider(), 'storyboard', ask, {'E0001'}, 0)
        self.assertEqual(episode['scenes'][0]['narration'], claim)
        self.assertEqual(episode['scenes'][0]['fact_ids'], ['F0001'])

    def test_scene_paraphrase_verifier_sees_only_selected_fact_and_exact_support(self):
        claim = 'The baseline does not install CUDA automatically.'
        hidden = 'Separate builds can use custom transports.'
        facts = [{'claim': statement, 'evidence_refs': [f'E{i:04d}'],
                  'support': [{'evidence_ref': f'E{i:04d}', 'text': statement}]}
                 for i, statement in enumerate((claim, hidden), 1)]
        ask = _make_ask({'facts': facts, 'assets': []}, [], [
            {'ref': 'E0001', 'evidence_role': 'primary'},
            {'ref': 'E0002', 'evidence_role': 'primary'},
        ], 'Engine', '')
        narration = 'CUDA is not installed automatically by the baseline.'

        class Provider:
            checked = []
            def complete_json(self, system, user):
                payload = json.loads(user)
                if 'checks' in payload:
                    self.checked.append(payload)
                    return {'decisions': [{'id': 's001', 'supported': True,
                                           'propositions': [{'text': proposition, 'support_indices': [0]}
                                                            for proposition in payload['checks'][0]['required_propositions']]}]}
                return {'version': 1, 'title': 'Engine', 'scenes': [{
                    'id': 's001', 'type': 'SUMMARY', 'narration': narration,
                    'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}

        with tempfile.TemporaryDirectory() as tmp:
            provider = Provider()
            episode = _complete_episode(provider, 'storyboard', ask, {'E0001', 'E0002'}, 0,
                                        project_dir=Path(tmp))
        self.assertEqual(episode['scenes'][0]['narration'], narration)
        self.assertEqual(provider.checked[0]['checks'][0]['facts'], [
            {'claim': claim, 'support': [{'evidence_ref': 'E0001', 'text': claim}]}])
        self.assertNotIn(hidden, json.dumps(provider.checked))

    def test_multipart_scene_replaces_expansion_and_reuses_grounded_checkpoint(self):
        claim = 'Kokoro runs on CPU for the baseline.'
        inventory = {'evidence': [{'ref': 'E0001', 'kind': 'document',
                                   'relative_path': 'overview.md', 'evidence_role': 'primary',
                                   'excerpt': claim}]}
        data = {'version': 1, 'facts': [{'claim': claim, 'evidence_refs': ['E0001'],
                'support': [{'evidence_ref': 'E0001', 'text': claim}],
                'phase': 'final', 'confidence': 'high'}], 'assets': []}

        class Provider:
            bad = True
            def complete_json(self, system, user):
                request = json.loads(user)
                mode = request.get('storyboard_mode')
                if mode is None:
                    raise OutputLimitExceeded('single storyboard output limit')
                if mode == 'outline':
                    return {'version': 1, 'title': 'Engine', 'slug': 'engine',
                            'summary': claim, 'scene_intents': [{
                                'type': 'SUMMARY', 'purpose': claim,
                                'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}
                return {'scenes': [{'id': 's001', 'type': 'SUMMARY', 'title': 'CPU',
                                    'narration': claim + (
                                        ' This ensures compatibility with standard hardware.'
                                        if self.bad else ''),
                                    'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'prompts' / 'storyboard.txt').write_text('storyboard')
            project = root / 'projects' / 'widget'
            provider = Provider()
            episode = plan(provider, data, inventory, project, 'Engine', max_retries=0)
            self.assertEqual(episode['scenes'][0]['narration'], claim)
            checkpoint = json_load(project / 'manifests' / 'storyboard-parts' / 'part-001.json')
            self.assertEqual(checkpoint['result']['scenes'][0]['narration'], claim)
            provider.bad = False
            self.assertEqual(plan(provider, data, inventory, project, 'Engine',
                                  max_retries=0), episode)

    def test_fast_path_fallback_keeps_hallucination_out_without_multipart(self):
        fact = 'The core dependency graph excludes Torch.'
        inventory = {'evidence': [{'ref': 'E0001', 'kind': 'document',
                                   'relative_path': 'deps.md', 'evidence_role': 'primary',
                                   'excerpt': fact}]}
        data = {'facts': [{'claim': fact, 'evidence_refs': ['E0001'],
                'support': [{'evidence_ref': 'E0001', 'text': fact}]}], 'assets': []}

        class Provider:
            calls = 0
            def complete_json(self, system, user):
                self.calls += 1
                request = json.loads(user)
                self.assert_full = request.get('storyboard_mode')
                return {'version': 1, 'title': 'Engine', 'scenes': [{
                    'id': 's001', 'type': 'SUMMARY', 'narration': fact +
                    ' This keeps the engine lightweight and accessible.',
                    'fact_ids': ['F0001'], 'evidence_refs': ['E0001']}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'prompts' / 'storyboard.txt').write_text('storyboard')
            project = root / 'projects' / 'sample'
            provider = Provider()
            episode = plan(provider, data, inventory, project, 'Engine', max_retries=0)
            self.assertEqual(provider.calls, 1)
            self.assertIsNone(provider.assert_full)
            self.assertEqual(episode['scenes'][0]['narration'], fact)
            self.assertEqual(json_load(project / 'episode.json'), episode)
            self.assertFalse((project / 'manifests' / 'storyboard-parts' / 'recovery.json').exists())

    def test_explicit_requested_architecture_and_workflow_outweigh_details(self):
        entries, facts = [], []
        for i in range(15):
            claim = f'WidgetEngine configures local hardware switch {i}.'
            entries.append({'ref': f'E{i+1:04d}', 'relative_path': f'implementation/hw{i}.py',
                            'evidence_role': 'primary'})
            facts.append({'claim': claim, 'evidence_refs': [f'E{i+1:04d}'],
                          'support': [{'evidence_ref': f'E{i+1:04d}', 'text': claim}],
                          'phase': 'final', 'confidence': 'high'})
        for i, claim in enumerate(('The architecture maps original inputs to indexed evidence.',
                                   'The workflow selects documented evidence before planning.'), 16):
            entries.append({'ref': f'E{i:04d}', 'relative_path': f'docs/{i}.md',
                            'evidence_role': 'primary'})
            facts.append({'claim': claim, 'evidence_refs': [f'E{i:04d}'],
                          'support': [{'evidence_ref': f'E{i:04d}', 'text': claim}],
                          'phase': 'final', 'confidence': 'high'})
        instructions = 'Explain the architecture and workflow.'
        anchors = _primary_anchors({'facts': facts, 'assets': []}, {'evidence': entries},
                                   set(), 'WidgetEngine', instructions)
        self.assertEqual({fact['claim'] for fact in anchors[:2]},
                         {facts[-2]['claim'], facts[-1]['claim']})
        ask = _make_ask({'facts': facts, 'assets': []}, [], entries, 'WidgetEngine', instructions)
        self.assertEqual(len(ask['priority_fact_ids']), 2)
        with self.assertRaisesRegex(StructuredOutputError,
                                    'Missing requested storyboard coverage: workflow -> choose one of F0017'):
            _normalize_outline({'version': 1, 'title': 'Engine', 'slug': 'engine',
                                'summary': 'Story', 'scene_intents': [{
                                    'type': 'CODE', 'purpose': f'Detail {i}',
                                    'fact_ids': [f'F{i+1:04d}'],
                                    'evidence_refs': [f'E{i+1:04d}']}
                                    for i in range(3)] + [{
                                    'type': 'CODE', 'purpose': 'Requested architecture',
                                    'fact_ids': ['F0016'], 'evidence_refs': ['E0016']}]},
                               {entry['ref'] for entry in entries}, ask)


if __name__ == '__main__':
    unittest.main()
