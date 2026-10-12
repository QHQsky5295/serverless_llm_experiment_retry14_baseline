"""D456 is an observation-only path clone of the sealed D371 configuration."""
from pathlib import Path
import re
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class HostQueueValidationContract(unittest.TestCase):
    def test_d464_only_exclusive_paths_differ_from_d458(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d458_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d464_7b_host_queue_validation_100.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d464', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_d464_reuses_first100_without_observer_or_new_parameters(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d464\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=100\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d464', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class HostQueueFullContract(unittest.TestCase):
    def test_d465_only_exclusive_paths_differ_from_d458(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d458_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d465_7b_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d465', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_d465_full_source_without_observer_or_new_parameters(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d465\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d465', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class CPUPreparationFullContract(unittest.TestCase):
    def test_d472_only_cpu_policy_and_exclusive_paths_differ(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d465_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d472_7b_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d472', after)
        self.assertEqual(new['resource_coordination'].pop('ieee_cpu_frontend_policy'),
                         'fixed_uuid_cpu_frontend_v1')
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_d472_original_full_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d472\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d472', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class HostExecutionValidationContract(unittest.TestCase):
    def test_d475_only_exclusive_paths_differ_from_d471(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d471_7b_cpu_frontend_validation_100.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d475_7b_cpu_frontend_validation_100.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d475', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)
        self.assertEqual(new['resource_coordination']['ieee_cpu_frontend_policy'],
                         'fixed_uuid_cpu_frontend_v1')

    def test_d475_reuses_original100_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d475\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=100\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d475', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class HostExecutionFullContract(unittest.TestCase):
    def test_d476_only_exclusive_paths_differ_from_d472(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d472_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d476_7b_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d476', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_d476_full_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d476\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d476', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D456DiagnosticContract(unittest.TestCase):
    def test_only_exclusive_paths_differ_from_d371(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261010_d371_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d456_7b_full_w0_prefix1.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d456', after)
            self.assertIn('prefix1', after)
        self.assertEqual(old, new)
        self.assertEqual(new['model']['runtime_concurrency_cap'], 14)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_launcher_registers_only_original_prefix(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d456\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertIn('"${1:-d293}" == d456', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)

    def test_wrapper_retains_qualification_and_frames_boundary(self):
        shell = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertIn('d456_7b_full_w0_prefix1)', shell)
        self.assertIn('test -f "$FAASLORA_TC_LAUNCH_RECEIPT"', shell)
        self.assertIn('test "${FAASLORA_FORMAL_RUN:-}" = 0', shell)
        branch = re.search(r'if \[\[ (.*?)\]\]; then\n(.*?)\nelse', shell, re.S)
        self.assertIn('d456_7b_full_w0_prefix1', branch.group(1))
        self.assertIn('FAASLORA_TC_STACK_SAMPLING=python_frames_v1', branch.group(2))
        self.assertNotIn('CORE_TIMING', branch.group(2))
