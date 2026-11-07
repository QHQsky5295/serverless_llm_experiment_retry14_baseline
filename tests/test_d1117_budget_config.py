"""Budget working point changes capacity, not trace or paper mechanisms."""
import copy
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class BudgetConfigTests(unittest.TestCase):
    def test_only_replica_cap_and_new_output_roots_change(self):
        original = yaml.safe_load((ROOT/'configs/ieee_tc/20261107_d1100_7b_busy_service_rate_w2.yaml').read_text())
        candidate = yaml.safe_load((ROOT/'configs/ieee_tc/20261107_d1117_7b_budget_cap3.yaml').read_text())
        self.assertEqual(candidate['resource_coordination']['max_instances'], 3)
        self.assertEqual(candidate['model']['visible_device_ids'], [0, 1, 2, 3])
        self.assertTrue(candidate['resource_coordination']['ieee_predictive_scaling']['allow_forecast_scale_down'])
        expected = copy.deepcopy(original)
        expected['resource_coordination']['max_instances'] = 3
        for section, key in [('experiment','output_dir'), ('storage','nvme_cache_dir'), ('storage','host_cache_dir')]:
            self.assertNotEqual(candidate[section][key], original[section][key])
            self.assertIn('d1117', candidate[section][key])
            expected[section][key] = candidate[section][key]
        self.assertEqual(candidate, expected)


if __name__ == '__main__':
    unittest.main()
