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


class GPUExecutionValidationContract(unittest.TestCase):
    def test_d515_only_exclusive_paths_differ_from_d475(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d475_7b_cpu_frontend_validation_100.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261014_d515_7b_cpu_frontend_validation_100.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d515', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)
        self.assertEqual(new['resource_coordination']['ieee_cpu_frontend_policy'],
                         'fixed_uuid_cpu_frontend_v1')

    def test_d515_reuses_original100_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d515\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=100\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d515', shell)
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


class GPUExecutionFullContract(unittest.TestCase):
    def test_d516_only_exclusive_paths_differ_from_d476(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d476_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261014_d516_7b_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d516', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_d516_full_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d516\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertNotIn('"${1:-d293}" == d516', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D519DiagnosticContract(unittest.TestCase):
    def test_restored_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d476_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261014_d519_7b_full_w0_prefix1.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertNotEqual(old[section].pop(field), new[section].pop(field))
        self.assertEqual(old, new)

    def test_explicit_prefix_and_components_not_production(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d519\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        branch = re.search(r'if \[\[ "\$FAASLORA_RESULTS_TAG" == d519_7b_full_w0_prefix1 \|\| "\$FAASLORA_RESULTS_TAG" == d723_3b_step_components100 \]\]; then\n(.*?)\nelse\n(.*?)\nfi', wrapper, re.S)
        self.assertIsNotNone(branch)
        self.assertIn('export FAASLORA_TC_CORE_STEP_TIMING=v2_components_v1', branch.group(1))
        self.assertIn('unset FAASLORA_TC_CORE_STEP_TIMING', branch.group(2))
        self.assertIn('FAASLORA_TC_CORE_TIMING=host_spans_v1', wrapper)


class D863Attempt2Contract(unittest.TestCase):
    def test_failed_prelaunch_preserved_and_same_policy_in_new_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d863_3b_native_prefix1000.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d863_attempt2_3b_native_prefix1000.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('attempt2', after)
        self.assertEqual(old, new)
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d863_attempt2\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000\n', block)
        self.assertIn('d863_20261026_attempt2/3b_native_prefix1000', block)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d863_attempt2' in condition]
        self.assertEqual(len(selected), 3)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[1])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[2])
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertEqual(wrapper.count('d863_attempt2_3b_native_prefix1000'), 1)


class D863DiagnosticContract(unittest.TestCase):
    def test_d858_policy_only_three_exclusive_paths_change(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d858_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d863_3b_native_prefix1000.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d863', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_original_w2_prefix1000_uses_shared_remote_and_decoder(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d863\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d863' in condition]
        self.assertEqual(len(selected), 3)
        self.assertIn('profile_d328_controller_python.sh', selected[0])
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[1])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[1])
        self.assertIn('http://192.168.4.174:18080', selected[1])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[2])
        self.assertIn('d838_20261025/dependency:$ROOT', selected[2])
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)

    def test_native_spans_only_no_components_inventory_or_frames(self):
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertEqual(wrapper.count('d863_3b_native_prefix1000'), 1)
        self.assertIn('d863_3b_native_prefix1000)', wrapper)
        self.assertIn('export FAASLORA_TC_CORE_TIMING=host_spans_v1', wrapper)
        self.assertIn('unset FAASLORA_TC_STACK_SAMPLING', wrapper)
        self.assertIn('unset FAASLORA_TC_CORE_STEP_TIMING', wrapper)
        self.assertIn('unset FAASLORA_TC_HOST_INVENTORY_TIMING', wrapper)
        source = (ROOT/'faaslora/utils/core_timing.py').read_text()
        self.assertIn('MAX_BYTES = 64 * 1024 * 1024', source)
        self.assertIn('MAX_SPANS = 200000', source)


class D861FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d846(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d846_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d861_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d846', 'd861').replace('20261025', '20261026'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_reuses_decoder_and_does_not_enable_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d846\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d861\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d846', 'd861').replace('20261025', '20261026'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d861' in condition]
        self.assertEqual(len(selected), 2)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertIn('d838_20261025/dependency:$ROOT', selected[1])
        self.assertNotIn('d861', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D866FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d858(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d858_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d866_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d858', 'd866'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_uses_qualified_graph_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d858\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d866\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d858', 'd866'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d866' in condition]
        self.assertEqual(len(selected), 3)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertIn('FAASLORA_IEEE_HOST_INVENTORY_MODE=graph', selected[2])
        self.assertIn('d864_20261026/attempt3/', selected[2])
        self.assertIn('30455d8f678483eb1adf3f7e594ffb6bef90a7aad462fbc03efbd98b969edf80', selected[2])
        self.assertNotIn('d866', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D870FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d858(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d858_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d870_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d858', 'd870'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_keeps_default_descriptor_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d858\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d870\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d858', 'd870'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d870' in condition]
        self.assertEqual(len(selected), 2)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('d747_20261023/', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertNotIn('FAASLORA_IEEE_HOST_INVENTORY_MODE=graph', ''.join(selected))
        self.assertIn('unset FAASLORA_IEEE_HOST_INVENTORY_MODE', shell)
        self.assertNotIn('d870', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D883FullContract(unittest.TestCase):
    def test_only_three_output_paths_change_from_d870(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d870_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d883_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d870', 'd883'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_host_construction', new['model'])
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_same_full_environment_without_graph_or_cpu_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d870\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d883\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d870', 'd883'), new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        previous = [body for condition, body in blocks if '== d870' in condition]
        selected = [body for condition, body in blocks if '== d883' in condition]
        self.assertEqual(len(selected), 2)
        self.assertEqual(previous, selected)
        self.assertNotIn('d883', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('PREFIX_COUNT=\n', new)


class D876FullContract(unittest.TestCase):
    def test_only_private_host_policy_and_three_exclusive_paths_change(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d870_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d876_3b_max3_full_w2.yaml').read_text())
        self.assertEqual(new['model'].pop('ieee_host_construction'), 'private_cpu_core_commit_v1')
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d870', 'd876'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_same_full_runner_environment_without_profiling(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d870\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d876\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d870', 'd876'), new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        previous = [body for condition, body in blocks if '== d870' in condition]
        selected = [body for condition, body in blocks if '== d876' in condition]
        self.assertEqual(len(selected), 2)
        self.assertEqual(previous, selected)
        self.assertNotIn('d876', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D858FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d846(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d846_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261026_d858_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d846', 'd858').replace('20261025', '20261026'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_reuses_decoder_and_does_not_enable_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d846\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d858\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d846', 'd858').replace('20261025', '20261026'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d858' in condition]
        self.assertEqual(len(selected), 2)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertIn('d838_20261025/dependency:$ROOT', selected[1])
        self.assertNotIn('d858', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D854FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d846(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d846_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d854_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d846', 'd854'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_reuses_decoder_and_does_not_enable_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d846\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d854\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d846', 'd854'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d854' in condition]
        self.assertEqual(len(selected), 2)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertIn('d838_20261025/dependency:$ROOT', selected[1])
        self.assertNotIn('d854', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D846FullContract(unittest.TestCase):
    def test_only_three_exclusive_paths_change_from_d840(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d840_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d846_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d840', 'd846'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_w2_reuses_decoder_and_does_not_enable_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d840\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d846\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d840', 'd846'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        blocks = re.findall(r'^if \[\[ ([^\n]+)\]\]; then\n(.*?)^fi', shell, re.M | re.S)
        selected = [body for condition, body in blocks if '== d846' in condition]
        self.assertEqual(len(selected), 2)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', selected[0])
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', selected[0])
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal', selected[1])
        self.assertIn('d838_20261025/dependency:$ROOT', selected[1])
        self.assertNotIn('d846', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D843DiagnosticContract(unittest.TestCase):
    def test_d840_configuration_only_changes_three_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d840_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d843_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d840', 'd843'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_complete_current_w2_reuses_environment_and_decoder(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d840\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d843\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d840', 'd843'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertEqual(shell.count(' || "${1:-d293}" == d843'), 3)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)
        w2 = re.search(r'^if \[\[ .*?== d814 .*?== d843 .*?== d846 \]\]; then\n(.*?)^fi', shell, re.M | re.S).group(1)
        self.assertIn('FAASLORA_TC_REPLAY_PROFILE=W2', w2)
        self.assertIn('20261018_3b_w2_adapter_map_v1.json', w2)

    def test_existing_observer_full_scope_without_extra_native_profiling(self):
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        tag = 'd843_3b_max3_full_w2'
        self.assertEqual(wrapper.count(tag), 3)
        self.assertIn('RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d843_20261025', wrapper)
        self.assertIn('export FAASLORA_TC_STACK_SAMPLING=python_frames_v1', wrapper)
        full = re.search(r'  if \[\[ .*?d603_7b_controller_full_w2.*?d843_3b_max3_full_w2.*?\]\]; then\n(.*?)\n  fi', wrapper, re.S).group(1)
        self.assertEqual(full, '    export FAASLORA_TC_STACK_REPLAY_SCOPE=full_source_v1')
        self.assertIn('unset FAASLORA_TC_CORE_TIMING', wrapper)
        for flag in ('FAASLORA_TC_CORE_STEP_TIMING', 'FAASLORA_TC_HOST_INVENTORY_TIMING'):
            self.assertIn('unset '+flag, wrapper)
        self.assertIn('test "${FAASLORA_FORMAL_RUN:-}" = 0', wrapper)


class D840FullContract(unittest.TestCase):
    def test_same_d835_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d835_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d840_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d835', 'd840'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_file_pressure_transport', new['resource_coordination'])

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d835\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d840\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d835', 'd840'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('"${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d840', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())
    def test_only_candidate_selects_frozen_single_thread_decoder(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^if \[\[ "\$\{1:-d293\}" == d840 \|\| "\$\{1:-d293\}" == d843 .*?== d846 \]\]; then\n(.*?)^fi', shell, re.M | re.S).group(1)
        self.assertEqual(block,
            '  export FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal\n'
            '  export PYTHONPATH="/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d838_20261025/dependency:$ROOT"\n')
        self.assertEqual(shell.count('FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER='), 1)


class D835FullContract(unittest.TestCase):
    def test_same_d832_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d832_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d835_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d832', 'd835'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_file_pressure_transport', new['resource_coordination'])

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d832\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d835\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d832', 'd835'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d835', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D832FullContract(unittest.TestCase):
    def test_same_d829_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d829_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d832_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d829', 'd832'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_file_pressure_transport', new['resource_coordination'])

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d829\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d832\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d829', 'd832'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d832', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D829FullContract(unittest.TestCase):
    def test_same_d826_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d826_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d829_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d826', 'd829'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_file_pressure_transport', new['resource_coordination'])

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d826\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d829\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d826', 'd829'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d829', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D826FullContract(unittest.TestCase):
    def test_same_d817_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d817_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d826_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d817', 'd826'), new[section].pop(field))
        self.assertEqual(old, new)
        self.assertNotIn('ieee_file_pressure_transport', new['resource_coordination'])

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d817\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d826\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d817', 'd826'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d826', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D822FullContract(unittest.TestCase):
    def test_only_coordination_transport_and_exclusive_paths_change(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d817_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d822_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d817', 'd822'), new[section].pop(field))
        self.assertEqual(new['resource_coordination'].pop('ieee_file_pressure_transport'),
                         'same_host_memfd_flock_v1')
        self.assertEqual(old, new)

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d817\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d822\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d817', 'd822'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d822', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D817FullContract(unittest.TestCase):
    def test_same_d814_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d814_3b_max3_full_w2.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d817_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertEqual(old[section].pop(field).replace('d814', 'd817'), new[section].pop(field))
        self.assertEqual(old, new)

    def test_shared_w2_environment_and_full_request_set(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        old = re.search(r'^  d814\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        new = re.search(r'^  d817\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertEqual(old.replace('d814', 'd817'), new)
        self.assertIn('PREFIX_COUNT=\n', new)
        self.assertIn('if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d846 ]]; then', shell)
        self.assertNotIn('d817', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D814FullContract(unittest.TestCase):
    def test_same_frozen_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261024_d812_3b_max3_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d814_3b_max3_full_w2.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            self.assertNotEqual(old[section].pop(field), new[section][field])
            self.assertIn('d814', new[section].pop(field))
        self.assertEqual(old, new)

    def test_only_frozen_adapter_view_changes_in_qualified_launch(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        case = re.search(r'^  d814\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', case)
        self.assertIn('SCENARIO=tc_ieee_full', case)
        pattern = r'^if \[\[ "\$\{1:-d293\}" == %s \]\]; then\n(.*?)^fi'
        old = re.search(pattern % 'd812', shell, re.M | re.S).group(1)
        new = re.search(pattern % r'd814 \|\| "\$\{1:-d293\}" == d817 \|\| "\$\{1:-d293\}" == d822 \|\| "\$\{1:-d293\}" == d826 \|\| "\$\{1:-d293\}" == d829 \|\| "\$\{1:-d293\}" == d832 \|\| "\$\{1:-d293\}" == d835 \|\| "\$\{1:-d293\}" == d840 \|\| "\$\{1:-d293\}" == d843 \|\| "\$\{1:-d293\}" == d846', shell, re.M | re.S).group(1)
        prefix = ('  export FAASLORA_TC_REPLAY_PROFILE=W2\n'
                  '  export FAASLORA_TC_ADAPTER_MAP="$ROOT/paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json"\n'
                  '  export FAASLORA_TC_ADAPTER_MAP_SHA256=28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23\n')
        self.assertEqual(new, prefix+old)
        self.assertNotIn('d814', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())

    def test_existing_w2_view_preserves_arrivals_prompts_and_targets(self):
        import json
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        mapping = ROOT/'paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json'
        source = json.loads(mapping.read_text())['source_path']
        w0 = FrozenReplayPlan.load(source)
        w2 = FrozenReplayPlan.load(source, profile='W2', adapter_map=mapping,
            adapter_map_sha256='28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23')
        self.assertEqual(len(w2.entries), 4000)
        self.assertEqual(w2.identity()['view_sha256'], '5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174')
        self.assertEqual(w0.source_sha256, w2.source_sha256)
        for before, after in zip(w0.entries, w2.entries):
            self.assertEqual((before.request_id, before.offset_s), (after.request_id, after.offset_s))
            original, mapped = json.loads(before.source_json), json.loads(after.source_json)
            provenance = mapped.pop('tc_adapter_rotation')
            self.assertEqual(provenance['original_source_item_sha256'], before.source_sha256)
            mapped['adapter_id'] = original['adapter_id']
            mapped['body']['lora_adapter_name'] = original['body']['lora_adapter_name']
            mapped['adapter_domain'] = original['adapter_domain']
            self.assertEqual(original, mapped)


class D813FullContract(unittest.TestCase):
    def test_same_frozen_configuration_except_exclusive_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261024_d812_3b_max3_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261025_d813_3b_max3_full_w1.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d813', after)
        self.assertEqual(old, new)
        self.assertEqual(new['resource_coordination']['max_instances'], 3)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_only_arrival_view_changes_in_qualified_launch(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        case = re.search(r'^  d813\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', case)
        self.assertIn('SCENARIO=tc_ieee_full', case)
        pattern = r'^if \[\[ "\$\{1:-d293\}" == %s \]\]; then\n(.*?)^fi'
        old = re.search(pattern % 'd812', shell, re.M | re.S).group(1)
        new = re.search(pattern % 'd813', shell, re.M | re.S).group(1)
        self.assertEqual(new, '  export FAASLORA_TC_REPLAY_PROFILE=W1\n'+old)
        self.assertNotIn('d813', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())

    def test_existing_w1_view_preserves_all_work_and_registered_timing(self):
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        source = Path('/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/'
            '20260509_134259_llama32_3b_main_s8/shared_artifacts/'
            'llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json')
        w0 = FrozenReplayPlan.load(source)
        w1 = FrozenReplayPlan.load(source, profile='W1')
        self.assertEqual(len(w1.entries), 4000)
        self.assertEqual(w1.source_sha256, '4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d')
        self.assertEqual(w1.identity()['view_sha256'], '8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f')
        self.assertEqual([(e.request_id, e.source_sha256) for e in w0.entries],
                         [(e.request_id, e.source_sha256) for e in w1.entries])
        for i, entry in enumerate(w1.entries):
            if i % 500 == 0:
                phase_base = 0. if i == 0 else w1.entries[i-1].offset_s+30.
            expected = phase_base+(w0.entries[i].offset_s-w0.entries[i//500*500].offset_s)/2.
            self.assertAlmostEqual(entry.offset_s, expected, places=6)


class D812FullContract(unittest.TestCase):
    def test_actual_controller_respects_three_including_pending(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl, ScalingAction
        cfg = yaml.safe_load((ROOT/'configs/ieee_tc/20261024_d812_3b_max3_full_w0.yaml').read_text())['resource_coordination']
        for ready, pending in ((1, 2), (2, 1), (3, 0)):
            control = IEEEReplicaControl(cfg['ieee_scaling'],
                interval_s=cfg['scale_eval_interval_s'], min_instances=cfg['min_instances'],
                max_instances=cfg['max_instances'])
            event = control.evaluate(now=0., queue_depth=100, active_requests=ready*8,
                ready_capacity=ready*8, ready_instances=ready, pending_instances=pending)
            self.assertEqual(event['action'], ScalingAction.NO_ACTION)
            self.assertEqual(event['target_instances'], 3)
            self.assertEqual(event['active_saturation'], 1.)

    def test_only_model_replica_bound_and_paths_change(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d752_3b_dense_alias_validation_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261024_d812_3b_max3_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d812', after)
        self.assertEqual(old['resource_coordination'].pop('max_instances'), 4)
        self.assertEqual(new['resource_coordination'].pop('max_instances'), 3)
        self.assertEqual(old, new)
        self.assertEqual(new['resource_coordination']['min_instances'], 1)
        self.assertEqual(new['model']['visible_device_ids'], [0, 1, 2, 3])
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_trace_and_qualified_3b_path(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d812\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        candidate = re.search(r'^if \[\[ "\$\{1:-d293\}" == d812 \]\]; then\n(.*?)^fi', shell, re.M | re.S).group(1)
        self.assertIn('FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080', candidate)
        self.assertIn('llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json', candidate)
        self.assertIn('FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044', candidate)
        self.assertIn('FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c', candidate)
        self.assertNotIn('FAASLORA_TC_REPLAY_PROFILE=', candidate)
        self.assertNotIn('d812', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D752FullContract(unittest.TestCase):
    def test_parent_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d749_3b_fresh_host_descriptor_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d752_3b_dense_alias_validation_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d752', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_trace_and_exact_qualified_descriptor(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d752\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d752' in line]
        self.assertEqual(len(switches), 3)
        self.assertEqual(sum('"${1:-d293}" == d659' in line for line in switches), 2)
        candidate = re.search(r'^if \[\[ "\$\{1:-d293\}" == d752 \]\]; then\n(.*?)^fi', shell, re.M | re.S).group(1)
        self.assertIn('FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=', candidate)
        self.assertIn('_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so', candidate)
        self.assertIn('FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044', candidate)
        self.assertEqual(candidate.count('export '), 2)
        self.assertNotIn('d752', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D749FullContract(unittest.TestCase):
    def test_parent_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d742_3b_execution_view_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d749_3b_fresh_host_descriptor_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d749', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_trace_and_exact_qualified_descriptor(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d749\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d749' in line]
        self.assertEqual(len(switches), 3)
        self.assertEqual(sum('"${1:-d293}" == d659' in line for line in switches), 2)
        candidate = re.search(r'^if \[\[ "\$\{1:-d293\}" == d749 \]\]; then\n(.*?)^fi', shell, re.M | re.S).group(1)
        self.assertIn('FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=', candidate)
        self.assertIn('_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so', candidate)
        self.assertIn('FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044', candidate)
        self.assertEqual(candidate.count('export '), 2)
        self.assertNotIn('d749', (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text())


class D745FullContract(unittest.TestCase):
    def test_current_d742_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d742_3b_execution_view_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d745_3b_observe_after_turn_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d745', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_original_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d745\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d745' in line]
        self.assertEqual(len(switches), 2)
        self.assertTrue(all('"${1:-d293}" == d659' in line for line in switches))
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertNotIn('d745', wrapper)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D742FullContract(unittest.TestCase):
    def test_current_d736_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d736_3b_validated_publication_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d742_3b_execution_view_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d742', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_original_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d742\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d742' in line]
        self.assertEqual(len(switches), 2)
        self.assertTrue(all('"${1:-d293}" == d659' in line for line in switches))
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertNotIn('d742', wrapper)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D738FullContract(unittest.TestCase):
    def test_current_d736_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d736_3b_validated_publication_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261023_d738_3b_planning_wire_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d738', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_original_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d738\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d738' in line]
        self.assertEqual(len(switches), 2)
        self.assertTrue(all('"${1:-d293}" == d659' in line for line in switches))
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertNotIn('d738', wrapper)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)



class D736FullContract(unittest.TestCase):
    def test_current_d732_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d732_3b_round_file_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d736_3b_validated_publication_full_w0.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d736', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_full_original_source_without_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d736\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        switches = [line for line in shell.splitlines() if '"${1:-d293}" == d736' in line]
        self.assertEqual(len(switches), 2)
        self.assertTrue(all('"${1:-d293}" == d659' in line for line in switches))
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertNotIn('d736', wrapper)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)


class D734DiagnosticContract(unittest.TestCase):
    def test_current_d732_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d732_3b_round_file_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d734_3b_controller_prefix1000.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d734', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_original_prefix_and_only_existing_stack_observer(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d734\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertEqual(shell.count(' || "${1:-d293}" == d734'), 3)
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        tag = 'd734_3b_controller_prefix1000'
        self.assertEqual(wrapper.count(tag), 2)
        branch = re.search(r'if \[\[ (.*?)\]\]; then\n(.*?)\nelse', wrapper, re.S)
        self.assertIn(tag, branch.group(1))
        self.assertIn('export FAASLORA_TC_STACK_SAMPLING=python_frames_v1', branch.group(2))
        self.assertIn('unset FAASLORA_TC_CORE_TIMING', branch.group(2))
        self.assertIn('test "${FAASLORA_FORMAL_RUN:-}" = 0', wrapper)


class D723DiagnosticContract(unittest.TestCase):
    def test_current_configuration_only_changes_three_paths(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d719_3b_identity_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261022_d723_3b_step_components100.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d723', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_registered_first100_components_without_inventory(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d723\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=100\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertEqual(shell.count(' || "${1:-d293}" == d723'), 3)
        wrapper = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        tag = 'd723_3b_step_components100'
        self.assertEqual(wrapper.count(tag), 2)  # case registration + component allowlist only
        self.assertIn('test "${FAASLORA_FORMAL_RUN:-}" = 0', wrapper)
        self.assertIn('unset FAASLORA_TC_CORE_STEP_TIMING', wrapper)


class D512DiagnosticContract(unittest.TestCase):
    def test_only_exclusive_paths_differ_from_current_d476(self):
        old = yaml.safe_load((ROOT/'configs/ieee_tc/20261012_d476_7b_full_w0.yaml').read_text())
        new = yaml.safe_load((ROOT/'configs/ieee_tc/20261014_d512_7b_full_w0_prefix1.yaml').read_text())
        for section, field in [('experiment', 'output_dir'), ('storage', 'nvme_cache_dir'),
                               ('storage', 'host_cache_dir')]:
            before, after = old[section].pop(field), new[section].pop(field)
            self.assertNotEqual(before, after)
            self.assertIn('d512', after)
            self.assertIn('prefix1', after)
        self.assertEqual(old, new)
        self.assertEqual(new['workload']['total_requests'], 4000)

    def test_existing_external_replay_selects_original1000(self):
        shell = (ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh').read_text()
        block = re.search(r'^  d512\)(.*?)^    ;;', shell, re.M | re.S).group(1)
        self.assertIn('PREFIX_COUNT=1000\n', block)
        self.assertIn('SCENARIO=tc_ieee_full', block)
        self.assertNotIn('export', block)
        self.assertIn('"${1:-d293}" == d512', shell)
        self.assertIn('export FAASLORA_FORMAL_RUN=0', shell)
        self.assertIn('export FAASLORA_TC_EXTERNAL_REPLAY=1', shell)

    def test_wrapper_uses_existing_core_not_frame_observer(self):
        shell = (ROOT/'scripts/ieee_tc/profile_d328_controller_python.sh').read_text()
        self.assertIn('d512_7b_full_w0_prefix1)', shell)
        self.assertIn('test -f "$FAASLORA_TC_LAUNCH_RECEIPT"', shell)
        self.assertIn('test "${FAASLORA_FORMAL_RUN:-}" = 0', shell)
        branch = re.search(r'if \[\[ (.*?)\]\]; then\n(.*?)\nelse\n(.*?)\nfi', shell, re.S)
        self.assertNotIn('d512', branch.group(1))
        self.assertIn('unset FAASLORA_TC_STACK_SAMPLING', branch.group(3))
        self.assertIn('FAASLORA_TC_CORE_TIMING=host_spans_v1', branch.group(3))


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
        # D603 explicitly unsets the core observer in the stack-only branch.
        # Disabling it is required isolation, not an enabled second observer.
        self.assertIn('unset FAASLORA_TC_CORE_TIMING', branch.group(2))
        self.assertNotIn('export FAASLORA_TC_CORE_TIMING', branch.group(2))
