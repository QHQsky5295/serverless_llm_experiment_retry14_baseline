"""CPU-only population validation, not synthetic performance evidence."""
import copy
import json
from types import SimpleNamespace as NS
import unittest

from scripts.ieee_tc.audit_d402_serverlessllm_path import validate_population


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


if __name__ == '__main__':
    unittest.main()
