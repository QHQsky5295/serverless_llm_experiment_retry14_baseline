"""CPU-only dLoRA receipt/lease rejection tests; not performance evidence."""
import copy
from pathlib import Path
import unittest

from scripts.ieee_tc.audit_d402_serverlessllm_path import (
    audit_layout, validate_dlora_lifetime, validate_slora_preparation, validate_dlora_worker_configuration)
from tests.test_tc_serverless_qualification_population import SloraEvidenceTests


class DloraEvidenceTests(unittest.TestCase):
    def test_exact_service_to_worker_derivation_preserves_native_parameters(self):
        config = dict(schema='ieee_tc_dlora_native_service_v1', engine_args={'swap_space':8}, gpu_ids=[0,1])
        cfg = dict(config, schema='ieee_tc_dlora_native_worker_v1', remote_config={'cache_dir':'/cache'},
            engine_args=dict(swap_space=8,lora_adapter_subset='/model/native_adapter_subset.json',
                             lora_adapter_root='/cache/objects',lora_adapter_limit=500))
        validate_dlora_worker_configuration(cfg, config, Path('/model'))
        for key, value in [('swap_space',4),('lora_adapter_limit',100),('lora_adapter_root','/foreign')]:
            bad = copy.deepcopy(cfg); bad['engine_args'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dlora_worker_configuration(bad, config, Path('/model'))

    def fixture(self):
        service = dict(cgroup='/owned', affinity=[4, 5])
        identity = dict(service, pid=10, start_ticks=1)
        workers = [dict(engine_id=i, cuda_visible=str(i),
                        identity=dict(service, pid=11+i, start_ticks=2+i)) for i in range(2)]
        row = dict(identity=identity, workers=workers, at=4.)
        cfg = dict(gpu_ids=[0, 1], deployment_notice_s=1.)
        events = []
        for kind, at in [('acquire',2.), ('worker_spawn',3.), ('native_workers',5.),
                         ('native_workers_exited',7.), ('release',8.)]:
            e = dict(event=kind, at=at, clock_id='local', lease_id='lease',
                     owner_id='lease', gpu_uuids=['g0', 'g1'])
            if kind in ('acquire', 'native_workers', 'release'):
                devices = []
                for i, w in enumerate(workers):
                    processes = [dict(identity=w['identity'], pid=w['identity']['pid'],
                        service_member=True, previously_owned=False, kind='compute')] if kind == 'native_workers' else []
                    devices.append(dict(gpu_uuid=f'g{i}', index=i, processes=processes))
                e['census'] = dict(source='nvml_v3_compute_and_graphics', clock_id='local', devices=devices)
            if kind == 'worker_spawn': e['worker'] = identity
            if kind == 'native_workers':
                e['workers'] = [dict(device_uuid=f'g{i}', pid=11+i, worker_rank=i) for i in range(2)]
            if kind == 'native_workers_exited': e['worker_pids'] = [11, 12]
            events.append(e)
        allocation = dict(released=True, gpu_uuids=['g0','g1'], lease_id='lease', events=events)
        return row, cfg, allocation, service

    def test_two_gpu_native_lease_not_one_static_rr_worker(self):
        row, cfg, allocation, service = self.fixture()
        self.assertEqual(len(validate_dlora_lifetime(row, cfg, allocation, service, 'local', 6.)), 2)
        self.assertEqual(audit_layout(Path('/run'), '3b', 'dlora_native_manager_v1', 'prefix100')[0],
                         Path('/run/launch3b.launch'))

    def test_worker_identity_and_bounded_scope_cannot_be_substituted(self):
        for key, value in [('pid',10), ('start_ticks',0), ('cgroup','/foreign'), ('affinity',[0])]:
            row, cfg, allocation, service = self.fixture()
            row = copy.deepcopy(row)
            row['workers'][0]['identity'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dlora_lifetime(row, cfg, allocation, service, 'local', 6.)

    def test_release_and_two_native_exits_required(self):
        for field, value in [('worker_pids',[11]), ('worker_pids',[11,11]), ('at',5.5)]:
            row, cfg, allocation, service = self.fixture()
            allocation['events'][3][field] = value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate_dlora_lifetime(row, cfg, allocation, service, 'local', 6.)

    def test_foreign_or_missing_gpu_census_is_not_release(self):
        for change in ('missing','compute','clock'):
            row, cfg, allocation, service = self.fixture()
            census = allocation['events'][-1]['census']
            if change == 'missing': census['devices'].pop()
            if change == 'compute':
                census['devices'][0]['processes'] = copy.deepcopy(allocation['events'][2]['census']['devices'][0]['processes'])
            if change == 'clock': census['clock_id'] = 'other'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_dlora_lifetime(row, cfg, allocation, service, 'local', 6.)

    def test_full_pool_preparation_keeps_dlora_identity(self):
        prep, remote = SloraEvidenceTests().preparation()
        for aid, r in prep.items(): r['tc_artifact_request_id'] = 'dlora-full-pool:'+aid
        self.assertEqual(validate_slora_preparation(prep, set(prep), 1., 501., remote, backend='dlora'), 5000)
        with self.assertRaises(ValueError):
            validate_slora_preparation(prep, set(prep), 1., 501., remote, backend='slora')


if __name__ == '__main__':
    unittest.main()
