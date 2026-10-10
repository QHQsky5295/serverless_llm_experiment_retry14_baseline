"""CPU-only population validation, not synthetic performance evidence."""
import copy
import hashlib
import json
from types import SimpleNamespace as NS
import unittest

from scripts.ieee_tc.audit_d402_serverlessllm_path import validate_population, compare_reference_row


class QualificationPopulationTests(unittest.TestCase):
    def fixture(self, coverage):
        n = 500 if coverage else 100
        entries, contracts = [], {}
        for i in range(n):
            row = dict(adapter_id=f'a{i}' if coverage else 'a', expected_output_tokens=300)
            if coverage:
                row['tc_adapter_coverage'] = dict(contract='adapter_coverage_v1', ordinal=i,
                    original_source_item_sha256='b'*64, original_adapter_id='source',
                    pool_index_sha256='c'*64)
            rid = f'r{i}'
            entries.append(NS(request_id=rid, source_json=json.dumps(row), source_sha256=f'sha{i}'))
            contracts[rid] = dict(source_item_sha256=f'sha{i}', adapter_id=row['adapter_id'],
                                 target_tokens=256)
            if coverage:
                contracts[rid]['tc_adapter_coverage'] = row['tc_adapter_coverage']
        plan = NS(entries=entries, profile='ADAPTER_COVERAGE_V1' if coverage else 'W0',
                  identity=lambda: dict(count=n, view_sha256='unit-view'))
        native = NS(load_http_replay_plan=lambda cfg: plan)
        return dict(request_count=n), dict(plan=plan.identity()), contracts, native, plan

    def test_both_explicit_populations_pass(self):
        for coverage, name in ((False, 'prefix100'), (True, 'adapter_coverage_v1')):
            cfg, ready, contracts, native, _ = self.fixture(coverage)
            self.assertEqual(validate_population(cfg, ready, contracts, native, name), len(contracts))

    def test_count_view_and_offered_set_must_match(self):
        cfg, ready, contracts, native, _ = self.fixture(True)
        mutations = [(dict(request_count=100), ready, contracts),
                     (cfg, dict(plan=dict(count=500, view_sha256='other')), contracts),
                     (cfg, ready, {k: v for k, v in contracts.items() if k != 'r0'})]
        for c, r, rows in mutations:
            with self.subTest(c=c, ready=r), self.assertRaises(ValueError):
                validate_population(c, r, rows, native, 'adapter_coverage_v1')

    def test_missing_or_changed_derivation_binding_and_target_rejected(self):
        cfg, ready, contracts, native, _ = self.fixture(True)
        for key, value in (('tc_adapter_coverage', None), ('adapter_id', 'wrong'),
                           ('source_item_sha256', 'wrong'), ('target_tokens', 255)):
            rows = copy.deepcopy(contracts)
            rows['r0'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_population(cfg, ready, rows, native, 'adapter_coverage_v1')

    def test_repeated_binding_cannot_count_as_whole_pool(self):
        cfg, ready, contracts, native, plan = self.fixture(True)
        contracts['r1']['adapter_id'] = 'a0'
        row = json.loads(plan.entries[1].source_json)
        row['adapter_id'] = 'a0'
        plan.entries[1].source_json = json.dumps(row)
        with self.assertRaisesRegex(ValueError, 'incomplete unique'):
            validate_population(cfg, ready, contracts, native, 'adapter_coverage_v1')

    def test_ordinary_prefix_cannot_contain_coverage_provenance(self):
        cfg, ready, contracts, native, plan = self.fixture(False)
        contracts['r0']['tc_adapter_coverage'] = {}
        with self.assertRaisesRegex(ValueError, 'ordinary prefix'):
            validate_population(cfg, ready, contracts, native, 'prefix100')
        del contracts['r0']['tc_adapter_coverage']
        plan.profile = 'ADAPTER_COVERAGE_V1'
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'prefix100')

    def test_coverage_profile_required_and_unknown_population_rejected(self):
        cfg, ready, contracts, native, plan = self.fixture(True)
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'automatic')
        plan.profile = 'W0'
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'adapter_coverage_v1')

    def test_thousand_original_requests_require_thousand_rows(self):
        cfg, ready, contracts, native, plan = self.fixture(False)
        with self.assertRaises(ValueError):
            validate_population(dict(request_count=1000), ready, contracts, native, 'prefix1000')
        for i in range(100, 1000):
            plan.entries.append(NS(request_id=f'r{i}', source_json=plan.entries[0].source_json,
                                   source_sha256=f'sha{i}'))
            contracts[f'r{i}'] = dict(source_item_sha256=f'sha{i}', adapter_id='a', target_tokens=256)
        self.assertEqual(validate_population(dict(request_count=1000), ready, contracts,
                                            native, 'prefix1000'), 1000)


class ReferenceReuseTests(unittest.TestCase):
    def fixture(self):
        sha = hashlib.sha256(b'[1,2]').hexdigest()
        observation = dict(completion_token_ids=[1, 2], completion_token_ids_sha256=sha,
                           native_lora_name='a', native_prompt_token_ids_sha256='prompt')
        contract = dict(request_id='r', adapter_id='a', canonical_prompt_sha256='text', target_tokens=2)
        source = dict(success=True, request_id='r', adapter_id='a', canonical_prompt_sha256='text',
                      requested_completion_tokens=2, native_token_timing=dict(
                          completion_token_ids=[1, 2], completion_token_ids_sha256=sha,
                          native_prompt_token_ids_sha256='prompt'))
        reference = dict(request_id='r', adapter_id='a', target_tokens=2, reference_tokens=2,
                         native_ids_sha256=sha, reference_ids_sha256=sha,
                         exact_sequence_match=True, first_divergent_position=None)
        return observation, contract, source, reference

    def test_exact_reuse_is_bounded_sequence_fact(self):
        row = compare_reference_row(*self.fixture())
        self.assertTrue(row['exact_sequence_match'])
        self.assertFalse(row['independently_observed_margin_reusable'])

    def test_identity_failures_are_not_sequence_disagreements(self):
        for field, value in [('adapter_id', 'wrong'), ('target_tokens', 1),
                             ('canonical_prompt_sha256', 'wrong')]:
            o, c, s, r = self.fixture()
            c[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_reference_row(o, c, s, r)

    def test_unobserved_margin_stays_unknown(self):
        o, c, s, r = self.fixture()
        o['completion_token_ids'] = [1, 3]
        o['completion_token_ids_sha256'] = hashlib.sha256(b'[1,3]').hexdigest()
        row = compare_reference_row(o, c, s, r)
        self.assertFalse(row['exact_sequence_match'])
        self.assertEqual(row['known_first_divergent_position'], 1)
        self.assertIsNone(row['hf_top1_minus_top2'])

    def test_matching_first_divergence_can_reuse_actual_margin(self):
        o, c, s, r = self.fixture()
        r.update(exact_sequence_match=False, reference_ids_sha256=hashlib.sha256(b'[1,3]').hexdigest(),
                 reference_prefix=[1, 3], native_prefix=[1, 2], first_divergent_position=1,
                 hf_top1_minus_top2=0.015625, native_token_in_hf_top2=True)
        row = compare_reference_row(o, c, s, r)
        self.assertFalse(row['exact_sequence_match'])
        self.assertTrue(row['independently_observed_margin_reusable'])
        self.assertEqual(row['hf_top1_minus_top2'], 0.015625)


if __name__ == '__main__':
    unittest.main()
