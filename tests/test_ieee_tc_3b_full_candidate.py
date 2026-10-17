"""CPU-only checks for the measured 3B Full assembly, no inference."""
import copy
import json
from pathlib import Path
import unittest
import yaml

from scripts.ieee_tc.prepare_d298_capacity import assemble_3b_current_full
from scripts.run_all_experiments import ScenarioRunner, _prepare_scenario_runtime_model_config

ROOT = Path(__file__).resolve().parents[1]


class Current3BFullCandidateTest(unittest.TestCase):
    def setUp(self):
        self.previous = yaml.safe_load((ROOT/'results/ieee_tc/p2_backend_qualification/d138_20261001/3b_main_config_full1.yaml').read_text())
        self.contract = json.loads((ROOT/'paper_results/ieee_tc/p2_backend/d643_3b_initializer/development_contract.json').read_text())
        self.profiles = json.loads((ROOT/'paper_results/ieee_tc/p2_backend/d645_3b_profiles/initialization/manifest.json').read_text())

    def assemble(self, count=100):
        return assemble_3b_current_full(self.previous, self.contract, self.profiles,
            output_dir='/never-created/results', nvme_dir='/never-created/nvme',
            host_dir='/never-created/host', count=count)

    def test_child_matches_measurement_and_input_is_not_mutated(self):
        before = copy.deepcopy((self.previous, self.contract, self.profiles))
        cfg, child, env = self.assemble()
        self.assertEqual(child, self.contract['model_config'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(cfg['model']['visible_device_ids'], [0, 1, 2, 3])
        self.assertEqual((self.previous, self.contract, self.profiles), before)

    def test_smoke_and_full_only_differ_in_population(self):
        smoke, _, _ = self.assemble(100)
        full, _, _ = self.assemble(4000)
        smoke['workload']['total_requests'] = 4000
        self.assertEqual(smoke, full)
        coord = full['resource_coordination']
        self.assertEqual(coord['min_instances'], 1)
        self.assertEqual(coord['max_instances'], 4)
        self.assertEqual(coord['ieee_scaling'], self.contract['coordination']['ieee_scaling'])

    def test_foreign_model_rejected(self):
        self.contract['model'] = '7b'
        with self.assertRaises(ValueError):
            self.assemble()

    def test_unmeasured_capacity_rejected(self):
        self.contract['model_config']['runtime_concurrency_cap'] = 14
        with self.assertRaises(ValueError):
            self.assemble()

    def test_foreign_minimum_rejected(self):
        self.contract['coordination']['min_instances'] = 2
        with self.assertRaises(ValueError):
            self.assemble()

    def test_actual_profile_loaders_and_wrong_identity_rejection(self):
        cfg, _, _ = self.assemble()
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.coord_cfg = cfg['resource_coordination']
        runner._initial_runtime_pending = True
        runner.model_cfg = _prepare_scenario_runtime_model_config(cfg['model'], cfg['workload'], runner.coord_cfg)
        runner._service_profiles = runner._load_ieee_service_profiles()
        self.assertIsNotNone(runner._load_ieee_preparation_profiles())
        runner.model_cfg['max_lora_rank'] = 64
        with self.assertRaises(ValueError):
            runner._load_ieee_service_profiles()
        with self.assertRaises(ValueError):
            runner._load_ieee_preparation_profiles()


if __name__ == '__main__':
    unittest.main()
