"""CPU metadata checks only; synthetic fixtures are not performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Loquetier_project.scripts.run_loquetier_trace_gate import (
    _adapter_footprint, _static_capacity_shards, _tensor_header,
    _ownership_prompt, _ownership_forward, _capacity_inputs, _capacity_forward,
    _capacity_bound, _capacity_mixed_forward, _capacity_score_target, _capacity_score_targets, _parse_args)


class LoquetierCapacityTests(unittest.TestCase):
    def fixture(self, aid='a', rank=8):
        config = dict(peft_type='LORA', bias='none', r=rank, base_model_name_or_path='/model')
        tensors = {
            'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':
                dict(shape=[rank, 16], dtype='F16'),
            'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':
                dict(shape=[16, rank], dtype='F16')}
        return config, tensors, _adapter_footprint(aid, config, tensors)

    def test_actual_shape_rank_not_config_rank_only(self):
        config, tensors, _ = self.fixture(rank=16)
        config['r'] = 8
        row = _adapter_footprint('a', config, tensors)
        self.assertEqual(row['native_max_rank'], 16)
        self.assertEqual(row['actual_ranks'], [16])
        self.assertEqual(row['peft_original_fp32_bytes'], 2048)
        self.assertEqual(row['prepared_fp16_bytes_per_rank'], 64)
        self.assertEqual(row['pointer_bytes'], 16)

    def test_rank_pair_and_unsupported_weights_rejected(self):
        config, tensors, _ = self.fixture()
        for modified in ({next(iter(tensors)): tensors[next(iter(tensors))]},
                         dict(tensors, **{'embed_tokens.weight': dict(shape=[8, 16])})):
            with self.assertRaises(ValueError):
                _adapter_footprint('a', config, modified)
        invalid = copy.deepcopy(tensors)
        invalid[list(invalid)[1]]['shape'][1] = 16
        with self.assertRaises(ValueError):
            _adapter_footprint('a', config, invalid)
        with self.assertRaises(ValueError):
            _adapter_footprint('a', dict(config, use_dora=True), tensors)

    def test_deterministic_full_disjoint_mapping_without_request_popularity(self):
        adapters = [self.fixture(str(i), 8 if i % 2 else 16)[2] for i in range(20)]
        result = _static_capacity_shards(adapters, 4)
        self.assertEqual(result, _static_capacity_shards(list(reversed(adapters)), 4))
        self.assertEqual(sorted(a for r in result for a in r['adapter_ids']), sorted(a['id'] for a in adapters))
        self.assertEqual(sum(r['adapter_count'] for r in result), 20)
        for row in result:
            self.assertEqual(row['prepared_fp16_bytes'], row['adapter_count'] * row['native_max_rank'] * 64)

    def test_duplicate_or_heterogeneous_module_mapping_rejected(self):
        a, b = self.fixture('a')[2], self.fixture('b')[2]
        with self.assertRaises(ValueError):
            _static_capacity_shards([a, a], 2)
        b['module_signature'][0][1] = 32
        with self.assertRaises(ValueError):
            _static_capacity_shards([a, b], 2)
        for count in (0, 5, True):
            with self.assertRaises(ValueError):
                _static_capacity_shards([a], count)

    def test_header_reads_metadata_and_checks_actual_file_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.safetensors'
            data = json.dumps({'a': dict(shape=[2, 3], dtype='F16', data_offsets=[0, 12])}).encode()
            payload = struct.pack('<Q', len(data)) + data + bytes(12)
            path.write_bytes(payload)
            rows, digest = _tensor_header(path)
            self.assertEqual(rows['a']['shape'], [2, 3])
            self.assertEqual(len(digest), 64)
            for invalid in (b'bad', payload[:-1], payload + b'bad', struct.pack('<Q', 2**63)):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    _tensor_header(path)

    def test_owner_prompt_reuses_first_native_input_and_verifies_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'native.jsonl'
            first = dict(event='loquetier_native_completed', request_id='first',
                observation=dict(native_terminal_observed=True, native_prompt_token_ids=list(range(64))))
            second = dict(first, request_id='later')
            path.write_text('\n'.join(json.dumps(x) for x in (dict(event='start'), first, second))+'\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = _ownership_prompt(path, digest)
            self.assertEqual(result['source_request_id'], 'first')
            self.assertEqual(result['prompt_ids'], list(range(32)))
            self.assertEqual(result['source_prompt_tokens'], 64)
            self.assertFalse(result['performance_input'])
            with self.assertRaisesRegex(ValueError, 'SHA'):
                _ownership_prompt(path, '0'*64)
            first['observation']['native_terminal_observed'] = False
            path.write_text(json.dumps(first)+'\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                _ownership_prompt(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_owner_cli_requires_explicit_optin_and_source_identity(self):
        base = ['gate', '--base-model', '/base', '--adapter-root', '/pool', '--output', '/new',
                '--native-capacity-probe', '--static-capacity-plan', '/plan',
                '--measurement-view', '/view', '--measurement-view-sha256', 'a'*64,
                '--physical-gpu-id', '0', '--main-repo', '/main']
        with patch('sys.argv', base):
            self.assertFalse(_parse_args().inference_owner_probe)
        inputs = ['--ownership-input-journal', '/journal', '--ownership-input-sha256', 'b'*64]
        for extra in (['--inference-owner-probe'], inputs,
                      ['--inference-owner-probe', *inputs[:-1], 'bad']):
            with patch('sys.argv', base+extra), self.assertRaises(SystemExit):
                _parse_args()
        with patch('sys.argv', base+['--inference-owner-probe', *inputs]):
            self.assertTrue(_parse_args().inference_owner_probe)

    def test_owner_forward_covers_every_id_with_same_grouping_and_rejects_wrong_binding(self):
        class Bridge:
            batch_size = 2
            def __init__(self, wrong=False):
                self.generator = SimpleNamespace(requests_len=0)
                self.objects, self.groups, self.mark, self.steps = {}, [], 0, 0
                self.wrong = wrong
            def add(self, rid, aid, ids, target):
                mark = self.mark
                self.mark += 1
                self.objects[mark] = (aid, ids)
                self.generator.requests_len += 1
                return mark
            def step(self):
                self.steps += 1
                if self.steps % 2:
                    self.groups.append([aid for aid, _ in self.objects.values()])
                    return {}
                result = {mark: dict(native_adapter_id='bad' if self.wrong else aid,
                    native_prompt_token_ids=ids, native_output_tokens=2, native_forward_count=2,
                    native_adapter_index=mark, completion_token_ids=[3, 4])
                    for mark, (aid, ids) in self.objects.items()}
                self.objects.clear()
                self.generator.requests_len = 0
                return result
        bridge = Bridge()
        rows = _ownership_forward(bridge, ['a', 'b', 'c'], [1, 2], 'before')
        self.assertEqual([row['adapter_id'] for row in rows], ['a', 'b', 'c'])
        self.assertEqual(bridge.groups, [['a', 'b'], ['c']])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            _ownership_forward(Bridge(wrong=True), ['a'], [1, 2], 'after')

    def test_capacity_inputs_select_geometry_not_latency_and_keep_full_input(self):
        def row(rid, length, target):
            return dict(event='loquetier_native_completed', request_id=rid,
                client_cancelled=False, observation=dict(native_terminal_observed=True,
                native_adapter_id='a', native_prompt_token_ids=list(range(length)),
                native_output_tokens=target, native_forward_count=target,
                completion_token_ids=[7]*target))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'native.jsonl'
            rows = [row('z', 760, 64), row('a', 760, 64), row('b', 759, 256)]
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value = _capacity_inputs(path, digest)
            self.assertEqual([r['source_request_id'] for r in value['cases']], ['a', 'b'])
            self.assertEqual([len(r['prompt_ids']) for r in value['cases']], [760, 759])
            self.assertEqual([r['target_tokens'] for r in value['cases']], [64, 256])
            self.assertEqual(value['source_native_completions'], 3)
            self.assertFalse(value['performance_input'])
            path.write_text(json.dumps(row('both', 760, 256))+'\n')
            same = _capacity_inputs(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(same['cases']), 1)
            self.assertEqual(same['cases'][0]['selection'], 'max_prefill+max_context')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                _capacity_inputs(path, '0'*64)
            for invalid in (row('a', 760, 265), dict(rows[0], client_cancelled=True), rows[1]):
                path.write_text('\n'.join(json.dumps(r) for r in [rows[1], invalid])+'\n')
                with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
                    _capacity_inputs(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_capacity_cli_is_explicit_and_exclusive(self):
        args = ['gate', '--base-model', '/base', '--adapter-root', '/pool', '--output', '/new',
                '--native-capacity-probe', '--static-capacity-plan', '/plan',
                '--measurement-view', '/view', '--measurement-view-sha256', 'a'*64,
                '--physical-gpu-id', '0', '--main-repo', '/main',
                '--ownership-input-journal', '/journal', '--ownership-input-sha256', 'b'*64]
        with patch('sys.argv', args+['--forward-capacity-probe']):
            value = _parse_args()
            self.assertTrue(value.forward_capacity_probe)
            self.assertFalse(value.inference_owner_probe)
        with patch('sys.argv', args+['--forward-capacity-probe', '--inference-owner-probe']), self.assertRaises(SystemExit):
            _parse_args()
        for extra in (['--capacity-evidence', '/old'],
                      ['--forward-capacity-probe', '--capacity-evidence', '/old'],
                      ['--forward-capacity-probe', '--capacity-evidence', '/old', '--capacity-evidence-sha256', 'bad']):
            with patch('sys.argv', args+extra), self.assertRaises(SystemExit):
                _parse_args()
        with patch('sys.argv', args+['--forward-capacity-probe', '--capacity-evidence', '/old',
                                    '--capacity-evidence-sha256', 'c'*64]):
            self.assertEqual(_parse_args().capacity_evidence, Path('/old'))

    def test_capacity_bound_rounds_kv_workspace_and_largest_allocation_without_latency(self):
        common = dict(device_total_bytes=1000, device_free_bytes=700,
                      reserved_bytes=280, inactive_split_bytes=10)
        samples = [dict(common, phase='registered', step=0, live_kv_bytes=40),
            dict(common, phase='prefill', step=1, live_kv_bytes=40,
                 peak_allocated_bytes=250),
            dict(common, phase='decode', step=2, live_kv_bytes=0, allocated_bytes=110)]
        evidence = dict(passed=True, verified_adapter_count=125, forward_capacity=dict(
            passed=True, batch_size=2, kv_page_bytes=10, kv_page_tokens=16,
            cases=[dict(samples=samples, outputs=[dict(native_output_tokens=2)]*2,
                        target_tokens=2, prompt_tokens=2, idle_allocated_bytes=100)]))
        bound = _capacity_bound(evidence, dict(vocab_size=2), 32)
        self.assertEqual(bound['candidate_batch_size'], 8)
        self.assertEqual(bound['per_request_accounted_bytes'], 100)
        self.assertEqual(bound['per_request_scratch_bytes'], 60)
        self.assertEqual(bound['predicted_total_bytes'], 940)
        self.assertTrue(bound['estimated_only'])
        self.assertFalse(bound['runtime_qualified'])
        with self.assertRaisesRegex(ValueError, 'complete prior'):
            _capacity_bound(dict(evidence, passed=False), dict(vocab_size=2), 32)
        with self.assertRaisesRegex(ValueError, 'larger distinct'):
            _capacity_bound(dict(evidence, verified_adapter_count=4), dict(vocab_size=2), 32)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            _capacity_bound(evidence, dict(vocab_size=0), 32)

    def test_distinct_capacity_inputs_preserve_original_adapter_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.jsonl'
            def row(rid, aid, length):
                return dict(event='loquetier_native_completed', request_id=rid,
                    observation=dict(native_terminal_observed=True, native_adapter_id=aid,
                        native_prompt_token_ids=list(range(length)), native_output_tokens=2,
                        native_forward_count=2, completion_token_ids=[5, 6]))
            path.write_text('\n'.join(json.dumps(r) for r in [row('c','a',15),
                row('b','a',20), row('d','b',18), row('a','c',18)])+'\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value = _capacity_inputs(path, digest, 3)
            self.assertEqual([(c['source_request_id'], c['adapter_id']) for c in value['cases']],
                             [('b', 'a'), ('a', 'c'), ('d', 'b')])
            self.assertFalse(value['duplicate_batch_copies'])
            for count in (0, True, 4):
                with self.assertRaises(ValueError):
                    _capacity_inputs(path, digest, count)

    def test_mixed_capacity_has_actual_staggered_registration_and_exact_drain(self):
        class Bridge:
            batch_size = 4
            def __init__(self):
                self.objects, self.generator = {}, SimpleNamespace(requests_len=0)
                self.mark, self.steps, self.added_at = 0, 0, []
            def add(self, rid, aid, ids, target):
                mark = self.mark
                self.mark += 1
                self.objects[mark] = [aid, ids, target, 0]
                self.generator.requests_len += 1
                self.added_at.append(self.steps)
                return mark
            def step(self):
                self.steps += 1
                result = {}
                for mark, item in list(self.objects.items()):
                    item[3] += 1
                    aid, ids, target, count = item
                    if count == target:
                        result[mark] = dict(native_adapter_id=aid, native_adapter_index=mark,
                            native_prompt_token_ids=ids, native_output_tokens=target,
                            native_forward_count=count, completion_token_ids=[8]*target)
                        del self.objects[mark]
                        self.generator.requests_len -= 1
                return result
        cases = [dict(source_request_id=str(i), adapter_id=str(i), prompt_ids=[1, 2],
                      target_tokens=4, retained_output_ids=[8]*4) for i in range(4)]
        for staggered in (False, True):
            bridge, samples = Bridge(), []
            rows = _capacity_mixed_forward(bridge, cases, lambda p,s:samples.append((p,s)), staggered)
            self.assertEqual(bridge.steps, 4+int(staggered))
            self.assertEqual(bridge.added_at, [0, 0, 1, 1] if staggered else [0]*4)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['matches_retained_output'] for r in rows))
            self.assertEqual(('mixed', 2) in samples, staggered)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            _capacity_mixed_forward(Bridge(), [cases[0]]*4, lambda *_:None, False)

    def test_score_target_binds_exact_retained_inputs_and_first_difference(self):
        cases = [dict(source_request_id=str(i), adapter_id=str(i), prompt_ids=[1, 2],
                      target_tokens=3, retained_output_ids=[4, 5, 6]) for i in range(2)]
        inputs = dict(cases=cases, journal_sha256='b'*64)
        outputs = [dict(source_request_id=str(i), native_adapter_id=str(i),
            native_output_tokens=3, native_forward_count=3,
            completion_token_ids=[4, 7, 8] if i == 0 else [4, 5, 6],
            matches_retained_output=i == 1) for i in range(2)]
        prior = dict(forward_capacity=dict(input=inputs,
            cases=[dict(selection='all_prefill', outputs=outputs)]))
        target = _capacity_score_target(prior, inputs)
        self.assertEqual(target, dict(case=cases[0], token_index=1, retained_token=5, observed_token=7))
        with self.assertRaisesRegex(ValueError, 'input contract'):
            _capacity_score_target(prior, dict(inputs, journal_sha256='c'*64))
        changed = copy.deepcopy(prior)
        changed['forward_capacity']['cases'][0]['outputs'][0]['matches_retained_output'] = True
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            _capacity_score_target(changed, inputs)
        changed = copy.deepcopy(prior)
        changed['forward_capacity']['cases'][0]['outputs'][1] = outputs[0]
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            _capacity_score_target(changed, inputs)
        changed = copy.deepcopy(prior)
        changed['forward_capacity']['cases'][0]['outputs'][0]['completion_token_ids'] = [4, 5, 6]
        changed['forward_capacity']['cases'][0]['outputs'][0]['matches_retained_output'] = True
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            _capacity_score_target(changed, inputs)

    def test_score_cli_requires_derived_capacity_and_source_sha(self):
        common = ['gate', '--base-model', '/m', '--adapter-root', '/a', '--output', '/o',
            '--native-capacity-probe', '--forward-capacity-probe',
            '--static-capacity-plan', '/plan', '--measurement-view', '/view',
            '--measurement-view-sha256', 'a'*64, '--physical-gpu-id', '0', '--main-repo', '/main',
            '--ownership-input-journal', '/j', '--ownership-input-sha256', 'b'*64]
        for extra in (['--capacity-score-evidence', '/scores'],
                      ['--capacity-score-evidence', '/scores', '--capacity-score-evidence-sha256', 'c'*64]):
            with patch('sys.argv', common+extra), self.assertRaises(SystemExit):
                _parse_args()
        extra = ['--capacity-evidence', '/capacity', '--capacity-evidence-sha256', 'd'*64,
                 '--capacity-score-evidence', '/scores', '--capacity-score-evidence-sha256', 'c'*64]
        with patch('sys.argv', common+extra):
            self.assertEqual(_parse_args().capacity_score_evidence, Path('/scores'))
        with patch('sys.argv', common+extra+['--capacity-score-mode', 'completed_phases_v2']):
            self.assertEqual(_parse_args().capacity_score_mode, 'completed_phases_v2')
        with patch('sys.argv', common+['--capacity-score-mode', 'completed_phases_v2']), self.assertRaises(SystemExit):
            _parse_args()

    def score_phases_fixture(self):
        cases = [dict(source_request_id=str(i), adapter_id=str(i), prompt_ids=[1, 2],
                      target_tokens=3, retained_output_ids=[4, 5, 6]) for i in range(2)]
        inputs = dict(cases=cases, journal_sha256='b'*64)
        phases = []
        for phase, changed_id in [('all_prefill', 0), ('mixed_prefill_decode', 1)]:
            outputs = [dict(source_request_id=str(i), native_adapter_id=str(i),
                native_output_tokens=3, native_forward_count=3,
                completion_token_ids=[4, 7, 8] if i == changed_id else [4, 5, 6],
                matches_retained_output=i != changed_id) for i in range(2)]
            phases.append(dict(selection=phase, outputs=outputs))
        return dict(passed=True, forward_capacity=dict(passed=True, input=inputs,
            cases=[dict(selection='logit_reference_batch2')]+phases)), inputs

    def test_score_targets_include_all_completed_phase_discrepancies(self):
        prior, inputs = self.score_phases_fixture()
        original = copy.deepcopy(prior)
        targets = _capacity_score_targets(prior, inputs)
        self.assertEqual([t['case']['source_request_id'] for t in targets], ['0', '1'])
        self.assertEqual([t['prior_phases'] for t in targets], [['all_prefill'], ['mixed_prefill_decode']])
        self.assertTrue(all((t['token_index'], t['retained_token'], t['observed_token']) == (1, 5, 7)
                            for t in targets))
        self.assertEqual(prior, original)
        prior['forward_capacity']['cases'].reverse()
        self.assertEqual(_capacity_score_targets(prior, inputs), targets)

    def test_score_targets_reject_missing_duplicate_conflicting_or_incomplete_evidence(self):
        prior, inputs = self.score_phases_fixture()
        changes = []
        bad = copy.deepcopy(prior); bad['passed'] = False; changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'].pop(); changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'].append(bad['forward_capacity']['cases'][-1]); changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'][1]['outputs'].pop(); changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'][1]['outputs'][0]['native_adapter_id'] = 'wrong'; changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'][1]['outputs'][0]['completion_token_ids'] = [4, 7]; changes.append(bad)
        bad = copy.deepcopy(prior); bad['forward_capacity']['cases'][1]['outputs'][0]['matches_retained_output'] = True; changes.append(bad)
        bad = copy.deepcopy(prior)
        bad['forward_capacity']['cases'][2]['outputs'][0].update(completion_token_ids=[4, 5, 7], matches_retained_output=False)
        changes.append(bad)
        for bad in changes:
            with self.subTest(evidence=bad), self.assertRaises(ValueError):
                _capacity_score_targets(bad, inputs)
        with self.assertRaisesRegex(ValueError, 'input contract'):
            _capacity_score_targets(prior, dict(inputs, journal_sha256='c'*64))

    def test_capacity_forward_uses_full_target_and_rejects_malformed_native_result(self):
        class Bridge:
            batch_size = 2
            def __init__(self, wrong=False):
                self.objects, self.generator = {}, SimpleNamespace(requests_len=0)
                self.calls, self.wrong = 0, wrong
            def add(self, rid, aid, ids, target):
                mark = len(self.objects)
                self.objects[mark] = (aid, ids, target)
                self.generator.requests_len += 1
                return mark
            def step(self):
                self.calls += 1
                if self.calls != 5:
                    return {}
                rows = {m: dict(native_adapter_id=aid, native_adapter_index=0,
                    native_prompt_token_ids=ids, native_output_tokens=target,
                    native_forward_count=target, completion_token_ids=[8]*(target-int(self.wrong)))
                    for m, (aid, ids, target) in self.objects.items()}
                self.objects.clear()
                self.generator.requests_len = 0
                return rows
        case = dict(selection='max_prefill', adapter_id='a', prompt_ids=list(range(760)),
            target_tokens=5, retained_output_ids=[8]*5)
        samples = []
        bridge = Bridge()
        result = _capacity_forward(bridge, case, lambda p, n: samples.append((p, n)))
        self.assertEqual(bridge.calls, 5)
        self.assertEqual(samples, [('registered', 0), ('prefill', 1),
            ('decode', 2), ('decode', 3), ('decode', 4), ('decode', 5)])
        self.assertTrue(all(r['matches_retained_output'] for r in result))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            _capacity_forward(Bridge(True), case, lambda *_: None)


if __name__ == '__main__':
    unittest.main()
