"""CPU-only population validation, not synthetic performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.ieee_tc.audit_d402_serverlessllm_path import (
    validate_population, compare_reference_row, frozen_slo_summary, validate_terminal_outcome,
    validate_vllm_ready_row, validate_slora_ready_row, validate_slora_allocation,
    validate_slora_preparation, validate_slora_preparation_reference, validate_artifact_transfer,
    audit_slora_receipts, digest, validate_slora_notice, validate_slora_resolver_journal,
    MODEL_IDENTITIES, audit_layout, validate_metric_model, validate_request_resolver_ledger)


class RequestDownloadLedgerTests(unittest.TestCase):
    def setUp(self):
        from faaslora.storage.http_artifact_store import preparation_content_sha256
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.contracts = {'r0': {'adapter_id': 'a'}, 'r1': {'adapter_id': 'a'},
                          'r2': {'adapter_id': 'b'}}
        self.rows, self.controls, self.receipts, self.pool, self.remote = [], {}, {}, {}, {}
        for aid in ('a', 'b'):
            files = {'adapter.safetensors': (100, 'a'*64)}
            content = preparation_content_sha256(files)
            tid = aid*32
            evidence = dict(remote_delivery_mode='prepublished_gzip_v1', http_transfer_id=tid,
                remote_archive_sha256='e'*64, transferred_bytes=10, published_archive_verified=True,
                content_verified=True, remote_pack_performed=False, loading_clock_id='local')
            receipt = dict(adapter_id=aid, content_sha256=content, content_manifest_sha256='c'*64,
                           evidence=evidence)
            path = Path(self.tmp.name)/(aid+'.json')
            path.write_text(json.dumps(receipt))
            self.receipts[aid] = (path, receipt)
            self.pool[aid] = {'files': [dict(path=k, size_bytes=v[0], sha256=v[1]) for k, v in files.items()]}
            self.remote[tid] = dict(artifact_id=aid, outcome='sent', pack_performed=False,
                temporary_created=False, delivery_mode='prepublished_gzip_v1', clock_id='remote',
                transfer_id=tid, archive_sha256='e'*64, archive_bytes=10, bytes_read=10, bytes_written=10)
        for i, (rid, contract) in enumerate(self.contracts.items()):
            aid = contract['adapter_id']
            path, receipt = self.receipts[aid]
            control = dict(tc_artifact_request_id=rid, tc_artifact_id=aid,
                tc_artifact_content_sha256=receipt['content_sha256'], tc_artifact_manifest_sha256='c'*64,
                tc_artifact_receipt_sha256=digest(path), tc_artifact_transfer_id=aid*32,
                tc_artifact_cache_hit=rid == 'r1', tc_artifact_transferred_bytes=0 if rid == 'r1' else 10,
                tc_artifact_delivery_mode='prepublished_gzip_v1', tc_artifact_resolve_started_s=float(i),
                tc_artifact_resolve_completed_s=i+.5)
            self.rows.extend([dict(event='resolve_started', pid=12, request_id=rid, adapter_id=aid,
                                   monotonic_s=i+.1),
                              dict(control, event='resolve_completed', pid=12, monotonic_s=i+.6)])
            if rid != 'r0':
                self.controls[rid] = control

    def check(self):
        return validate_request_resolver_ledger(self.rows, self.contracts, self.controls,
            self.receipts, self.pool, self.remote, 'remote', 'local')

    def test_failed_download_owner_is_charged_even_if_success_only_hits_cache(self):
        report = self.check()
        self.assertEqual(report['remote_wire_bytes'], 20)
        self.assertEqual(report['successful_download_owner_wire_bytes'], 10)
        self.assertEqual(report['failed_download_owner_wire_bytes'], 10)
        self.assertEqual(report['resolves_without_successful_response'], 1)

    def test_all_download_owners_can_have_failed(self):
        self.controls = {}
        report = self.check()
        self.assertEqual(report['failed_download_owner_wire_bytes'], 20)
        self.assertEqual(report['successful_download_owner_wire_bytes'], 0)

    def test_duplicate_completion_is_not_a_second_request_or_download(self):
        self.rows.append(copy.deepcopy(self.rows[1]))
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_download_owner_does_not_pass_on_a_later_cache_hit(self):
        self.rows = self.rows[2:]
        with self.assertRaises(ValueError):
            self.check()

    def test_unjoined_remote_transfer_is_rejected(self):
        self.remote['orphan'] = dict(self.remote['a'*32], transfer_id='orphan')
        with self.assertRaises(ValueError):
            self.check()

    def test_unknown_request_is_not_allowed_to_explain_extra_bytes(self):
        del self.contracts['r0']
        with self.assertRaises(ValueError):
            self.check()

    def test_successful_response_must_match_journal(self):
        self.controls['r1']['tc_artifact_transferred_bytes'] = 10
        with self.assertRaises(ValueError):
            self.check()

    def test_remote_clock_or_wire_bytes_cannot_be_ignored_for_failed_owner(self):
        for key, value in [('clock_id', 'wrong'), ('bytes_written', 9), ('pack_performed', True)]:
            old = self.remote['a'*32][key]
            self.remote['a'*32][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            self.remote['a'*32][key] = old

    def test_incomplete_or_unknown_resolver_event_is_not_silently_dropped(self):
        self.rows.append(dict(event='resolve_failed', request_id='r0'))
        with self.assertRaises(ValueError):
            self.check()

    def test_resolver_timestamps_are_checked_in_actual_emit_order(self):
        self.rows[0]['monotonic_s'] = -1
        with self.assertRaises(ValueError):
            self.check()


class ModelIdentityTests(unittest.TestCase):
    def test_explicit_layout_preserves_each_population_and_scheduler(self):
        root = Path('/owned/run')
        for model in ('7b', '3b'):
            for scheduler in ('original', 'request_identity_v1', 'resident_rr_v1',
                              'slora_resident_rr_v1', 'loquetier_static_shards_v1'):
                suffix = '_repaired' if scheduler in ('original', 'request_identity_v1') else ''
                for population, count in [('prefix100', 100), ('prefix1000', 1000),
                                          ('adapter_coverage_v1', 500), ('full4000', 4000),
                                          ('full4000_w1', 4000), ('full4000_w2', 4000)]:
                    with self.subTest(model=model, scheduler=scheduler, population=population):
                        self.assertEqual(audit_layout(root, model, scheduler, population),
                            (root/f'launch{model}{suffix}.launch', root/f'model{model}{suffix}',
                             root/f'http_{model}_{count}.json'))

    def test_historical_entry_points_keep_explicit_7b_default(self):
        import inspect
        from scripts.ieee_tc.audit_d402_serverlessllm_path import audit, reuse_numerical_reference
        for function in (audit, reuse_numerical_reference):
            self.assertEqual(inspect.signature(function).parameters['model_key'].default, '7b')

    def test_unknown_model_or_population_is_not_a_7b_fallback(self):
        with self.assertRaises(ValueError):
            audit_layout(Path('/run'), '13b', 'resident_rr_v1', 'prefix100')
        with self.assertRaises(ValueError):
            audit_layout(Path('/run'), '3b', 'resident_rr_v1', 'unknown')
        with self.assertRaises(ValueError):
            audit_layout(Path('/run'), '3b', 'unknown', 'prefix100')
        with self.assertRaises(ValueError):
            validate_metric_model({}, 'unknown', Path('/main'))

    def test_model_slo_identity_and_symlink_alias_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp)
            for profile, relative in MODEL_IDENTITIES.values():
                (main/relative).mkdir(parents=True)
            for key, (profile, relative) in MODEL_IDENTITIES.items():
                with self.subTest(model=key):
                    http = {'backbone': str(main/relative)}
                    self.assertEqual(validate_metric_model(http, key, main), profile)
                    self.assertEqual(validate_metric_model(http, key, main, {'model': profile}), profile)
                    alias = main/f'alias_{key}'
                    alias.symlink_to(main/relative, target_is_directory=True)
                    self.assertEqual(validate_metric_model({'backbone': str(alias)}, key, main), profile)
                    other = '3b' if key == '7b' else '7b'
                    with self.assertRaisesRegex(ValueError, 'backbone'):
                        validate_metric_model(http, other, main)
                    with self.assertRaisesRegex(ValueError, 'frozen SLO model'):
                        validate_metric_model(http, key, main, {'model': MODEL_IDENTITIES[other][0]})
                    with self.assertRaises(KeyError):
                        validate_metric_model(http, key, main, {})


class SloraEvidenceTests(unittest.TestCase):
    def fixture(self):
        cfg = dict(gpu_ids=[2, 3], deployment_notice_s=1., native_input_params={'tp': 1},
            native_environment={'PATH': '/native/bin'}, host_layout='packed_views_v1',
            source_view_manifest_sha256='a'*64, native_args={'model_dir': '/model'},
            adapters={f'a{i:03}': [i, f'/cache/objects/a{i:03}'] for i in range(500)})
        svc = dict(cgroup='/owned', affinity=[4, 5])
        identity = dict(svc, pid=123, start_ticks=9)
        parent = dict(svc, pid=120, start_ticks=8)
        row = dict(schema='ieee_tc_slora_backend_ready_v1', index=1, clock_id='local',
            cuda_visible='3', cuda_device_count=1, tp=1, identity=identity, api_parent_identity=parent,
            native_input_params=cfg['native_input_params'], native_environment=cfg['native_environment'],
            host_layout=cfg['host_layout'], source_view_manifest_sha256='a'*64, model_dir='/model',
            ready_boundary='native_full_pool_and_model_initialized',
            adapter_ids={p: i for i, p in cfg['adapters'].values()}, adapter_count=500, at=10.)
        process = dict(pid=123, start_ticks=9, clock_id='local')
        c = dict(tc_route_ordinal=3, instance_id='slora-1', tc_clock_id='local',
            tc_backend_entry_s=11., tc_worker_ready_s=10., tc_deployment_notice_s=1., tc_worker_native_process=process)
        o = dict(timing_contract='ieee_tc_slora_native_v1', native_process=process,
            native_adapter_path='/cache/objects/a000', native_adapter_index=0)
        return row, cfg, c, svc, o

    def test_native_gpu_child_not_http_parent_or_vllm_identity(self):
        row, cfg, c, svc, o = self.fixture()
        self.assertEqual(validate_slora_ready_row(row, cfg, c, 3, svc, 'local', o), 1)
        for field, value in [('native_process', dict(pid=120, start_ticks=8, clock_id='local')),
                             ('timing_contract', 'ieee_tc_native_v1'),
                             ('native_adapter_index', 1), ('native_adapter_path', '/other/a000')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_slora_ready_row(row, cfg, c, 3, svc, 'local', dict(o, **{field: value}))

    def test_ready_requires_full_registry_source_config_and_actual_containment(self):
        row, cfg, c, svc, o = self.fixture()
        mutations = [('adapter_count', 499), ('adapter_ids', {}), ('host_layout', 'native_v1'),
                     ('native_input_params', {}), ('native_environment', {}),
                     ('source_view_manifest_sha256', 'b'*64), ('model_dir', '/other'),
                     ('ready_boundary', 'http_healthy'), ('index', 0), ('cuda_visible', '2'),
                     ('cuda_device_count', 2), ('tp', 2), ('clock_id', 'foreign'),
                     ('at', 12.), ('at', float('nan')), ('api_parent_identity', row['identity'])]
        mutations += [('identity', dict(row['identity'], **{k: v})) for k, v in
                      [('cgroup', '/other'), ('affinity', [0]), ('start_ticks', None)]]
        for field, value in mutations:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_slora_ready_row(dict(row, **{field: value}), cfg, c, 3, svc, 'local', o)

    def test_last_request_cannot_silently_change_route_or_notice(self):
        row, cfg, c, svc, o = self.fixture()
        for field, value in [('tc_route_ordinal', 2), ('instance_id', 'slora-0'),
                             ('tc_worker_ready_s', 9.), ('tc_deployment_notice_s', 0.)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_slora_ready_row(row, cfg, dict(c, **{field: value}), 3, svc, 'local', o)

    def allocation(self, row):
        graphics = dict(kind='graphics', pid=200, identity={'pid': 200}, service_member=False, previously_owned=False)
        worker = dict(kind='compute', pid=123, identity=row['identity'], service_member=True, previously_owned=True)
        def census(processes):
            return dict(source='nvml_v3_compute_and_graphics', clock_id='local',
                devices=[dict(index=3, gpu_uuid='GPU3', processes=processes)])
        def event(kind, at, **fields):
            return dict(event=kind, at=at, clock_id='local', lease_id='lease', owner_id='lease',
                        gpu_uuids=['GPU3'], **fields)
        events = [event('acquire', 2., census=census([graphics])),
                  event('worker_spawn', 3., worker=row['api_parent_identity']),
                  event('native_workers', 10.5, census=census([worker, graphics]),
                        workers=[dict(device_uuid='GPU3', pid=123, worker_rank=1)]),
                  event('native_workers_exited', 15., worker_pids=[123]),
                  event('release', 16., census=census([graphics]))]
        return dict(released=True, lease_id='lease', gpu_uuids=['GPU3'], events=events)

    def test_physical_native_child_parent_and_release_join(self):
        row, cfg, *_ = self.fixture()
        a = self.allocation(row)
        validate_slora_allocation(a, row, cfg, 1, 'local', 14.)
        # Unrelated, identified graphics does not mean the experiment still owns a GPU.
        for index, field, value in [(1, 'worker', row['identity']),
                                    (2, 'workers', [dict(device_uuid='GPU3', pid=120, worker_rank=1)]),
                                    (3, 'worker_pids', [120]), (4, 'at', 13.), (0, 'lease_id', 'other')]:
            bad = copy.deepcopy(a)
            bad['events'][index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                validate_slora_allocation(bad, row, cfg, 1, 'local', 14.)

    def test_physical_release_cannot_ignore_compute_owned_or_unknown_graphics(self):
        row, cfg, *_ = self.fixture()
        for field, value in [('kind', 'compute'), ('service_member', True),
                             ('previously_owned', True), ('identity', None)]:
            a = self.allocation(row)
            a['events'][-1]['census']['devices'][0]['processes'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_slora_allocation(a, row, cfg, 1, 'local', 14.)

    def preparation(self):
        prep, remote = {}, {}
        for i in range(500):
            aid, tid = f'a{i:03}', f'{i:032x}'
            prep[aid] = dict(tc_artifact_id=aid, tc_artifact_request_id='slora-full-pool:'+aid,
                tc_artifact_cache_hit=False, tc_artifact_delivery_mode='prepublished_gzip_v1',
                tc_artifact_transfer_id=tid, tc_artifact_transferred_bytes=10,
                tc_artifact_resolve_started_s=1.+i, tc_artifact_resolve_completed_s=2.+i)
            remote[tid] = dict(artifact_id=aid, bytes_written=10)
        return prep, remote

    def test_full_pool_transfer_count_is_independent_of_offered_unique_ids(self):
        prep, remote = self.preparation()
        self.assertEqual(validate_slora_preparation(prep, set(prep), 1., 501., remote), 5000)
        # Repeated requests reference the same deployment record without adding wire bytes.
        result = dict(artifact_attribution='deployment_preparation_reference_not_request_transfer',
                      artifact_transferred_bytes=0, preparation_transferred_bytes=10)
        for _ in range(3):
            validate_slora_preparation_reference(dict(prep['a000'], unrelated='kept'), prep['a000'], result)
        with self.assertRaises(ValueError):
            validate_slora_preparation_reference(prep['a000'], prep['a000'], dict(result, artifact_transferred_bytes=10))
        with self.assertRaises(ValueError):
            validate_slora_preparation_reference(dict(prep['a000'], tc_artifact_cache_hit=True), prep['a000'], result)

    def test_preparation_cannot_use_hot_subset_or_pre_notice_or_duplicate_transfer(self):
        prep, remote = self.preparation()
        with self.assertRaises(ValueError):
            validate_slora_preparation({k: prep[k] for k in list(prep)[:132]}, set(prep), 1., 501., remote)
        for key, value in [('tc_artifact_resolve_started_s', .9), ('tc_artifact_resolve_completed_s', 502.),
                           ('tc_artifact_cache_hit', True), ('tc_artifact_request_id', 'request0'),
                           ('tc_artifact_transferred_bytes', 0), ('tc_artifact_transfer_id', f'{1:032x}')]:
            bad = copy.deepcopy(prep)
            bad['a000'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_slora_preparation(bad, set(prep), 1., 501., remote)
        with self.assertRaises(ValueError):
            validate_slora_preparation(prep, set(prep), 1., 500., remote)

    def test_transfer_join_requires_server_content_and_actual_wire_counts(self):
        from faaslora.storage.http_artifact_store import preparation_content_sha256
        files = {'adapter.safetensors': (100, 'a'*64)}
        content = preparation_content_sha256(files)
        c = dict(tc_artifact_receipt_sha256='b'*64, tc_artifact_content_sha256=content,
                 tc_artifact_manifest_sha256='c'*64, tc_artifact_transfer_id='d'*32)
        evidence = dict(remote_delivery_mode='prepublished_gzip_v1', http_transfer_id='d'*32,
            remote_archive_sha256='e'*64, transferred_bytes=10, published_archive_verified=True,
            content_verified=True, remote_pack_performed=False, loading_clock_id='local')
        receipt = dict(adapter_id='a', content_sha256=content, content_manifest_sha256='c'*64, evidence=evidence)
        server = dict(artifact_id='a', outcome='sent', pack_performed=False, temporary_created=False,
            delivery_mode='prepublished_gzip_v1', clock_id='remote', transfer_id='d'*32,
            archive_sha256='e'*64, archive_bytes=10, bytes_read=10, bytes_written=10)
        validate_artifact_transfer('a', c, receipt, 'b'*64, files, server, 'remote', 'local')
        for field, value in [('bytes_written', 9), ('clock_id', 'local'), ('pack_performed', True),
                             ('archive_sha256', 'wrong'), ('artifact_id', 'other')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_artifact_transfer('a', c, receipt, 'b'*64, files,
                                          dict(server, **{field: value}), 'remote', 'local')

    def test_common_notice_does_not_move_arrivals_to_native_ready(self):
        ready = dict(clock_id='local', plan={'view': 'frozen'}, config_sha256='a'*64)
        launch = dict(ready, deployment_notice_s=10., replay_t0_s=70.)
        cfg = dict(deployment_notice_s=10.)
        entries = [NS(request_id='r0', offset_s=.5), NS(request_id='r1', offset_s=2.)]
        rows = {'r0': {'planned_arrival_s': 70.5}, 'r1': {'planned_arrival_s': 72.}}
        validate_slora_notice(launch, cfg, ready, entries, rows)
        for field, value in [('deployment_notice_s', 11.), ('replay_t0_s', 80.),
                             ('clock_id', 'other'), ('config_sha256', 'b'*64), ('plan', {})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_slora_notice(dict(launch, **{field: value}), cfg, ready, entries, rows)
        for arrival in (80.5, float('nan')):
            with self.assertRaises(ValueError):
                validate_slora_notice(launch, cfg, ready, entries, dict(rows, r0={'planned_arrival_s': arrival}))

    def test_preparation_must_match_unique_actual_resolver_completions(self):
        prep, _ = self.preparation()
        rows = [dict(event='resolve_completed', **r) for r in prep.values()]
        validate_slora_resolver_journal(rows, prep)
        for bad in (rows[:-1], rows+[rows[0]], rows+[{'event': 'resolve_failed'}],
                    [dict(rows[0], tc_artifact_transfer_id='wrong')]+rows[1:]):
            with self.assertRaises(ValueError):
                validate_slora_resolver_journal(bad, prep)

    def test_slora_reference_reuses_path_identity_without_requiring_cross_backend_exactness(self):
        o, c, s, r = ReferenceReuseTests().fixture()
        o.pop('native_lora_name')
        o.update(timing_contract='ieee_tc_slora_native_v1', native_adapter_path='/objects/a')
        self.assertTrue(compare_reference_row(o, c, s, r)['exact_sequence_match'])
        o['completion_token_ids'] = [1, 3]
        o['completion_token_ids_sha256'] = hashlib.sha256(b'[1,3]').hexdigest()
        self.assertFalse(compare_reference_row(o, c, s, r)['exact_sequence_match'])

    def test_retained_ready_worker_and_physical_files_join(self):
        # Source/config validators have independent native tests. This fixture
        # exercises the actual offline file joins, not a claimed GPU receipt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'model7b'
            (model/'worker_receipts').mkdir(parents=True)
            rpc = root/'view/slora/server/router/model_infer/model_rpc.py'
            rpc.parent.mkdir(parents=True)
            rpc.write_text('# CPU fixture\n')
            runner = root/'runner.py'
            runner.write_text('# CPU fixture\n')
            row, cfg, c, svc, o = self.fixture()
            cfg.update(source_view=str(root/'view'), remote_config={'cache_dir': '/cache'},
                       source_sha256={str(runner): digest(runner)})
            row['native_input_params'] = cfg['native_input_params'] = {'model_dir': '/model'}
            configuration = copy.deepcopy(cfg)
            worker = model/'worker_config.json'
            worker.write_text(json.dumps(cfg))
            http = root/'http.json'
            http.write_text('{}')
            requests, allocations = [], []
            for i in range(2):
                r, control, observation = copy.deepcopy(row), copy.deepcopy(c), copy.deepcopy(o)
                rid = f'{i:032x}'
                r.update(index=i, cuda_visible=str(cfg['gpu_ids'][i]), receipt_id=rid,
                    worker_config_sha256=digest(worker), runner_sha256=digest(runner),
                    rpc_path=str(rpc), rpc_sha256=digest(rpc))
                r['identity']['pid'] += i
                r['api_parent_identity']['pid'] += i
                process = dict(pid=r['identity']['pid'], start_ticks=9, clock_id='local')
                control.update(tc_route_ordinal=i, instance_id=f'slora-{i}', tc_worker_receipt_id=rid,
                               tc_worker_native_process=process)
                observation.update(native_process=process, worker_completed_monotonic_s=14.)
                path = model/'worker_receipts'/(rid+'.json')
                path.write_text(json.dumps(r))
                control['tc_worker_receipt_sha256'] = digest(path)
                observation['control_observation'] = control
                requests.append(dict(request_id=f'r{i}', response={'metrics': {'ieee_tc': observation}}))
                a = self.allocation(r)
                a.update(lease_id=f'lease{i}', gpu_uuids=[f'GPU{i}'])
                for event in a['events']:
                    event.update(lease_id=f'lease{i}', owner_id=f'lease{i}', gpu_uuids=[f'GPU{i}'])
                    if 'census' in event:
                        event['census']['devices'][0].update(index=cfg['gpu_ids'][i], gpu_uuid=f'GPU{i}')
                        for p in event['census']['devices'][0]['processes']:
                            if p['kind'] == 'compute':
                                p.update(pid=r['identity']['pid'], identity=r['identity'])
                    if event['event'] == 'native_workers':
                        event['workers'][0].update(pid=r['identity']['pid'], worker_rank=i, device_uuid=f'GPU{i}')
                    if event['event'] == 'native_workers_exited':
                        event['worker_pids'] = [r['identity']['pid']]
                journal = root/f'lease{i}.jsonl'
                journal.write_text(''.join(json.dumps(e)+'\n' for e in a['events']))
                a['journal'] = str(journal)
                allocations.append(a)
            q = dict(schema='ieee_tc_slora_native_model_qualification_v1', configuration=configuration,
                physical_lifecycle=dict(mode='slora_resident_pool_v1', measurement_complete=True),
                source_sha256=cfg['source_sha256'], physical_allocations=allocations, requests=requests, service=svc)
            native = NS(load_http_replay_plan=lambda cfg: NS(entries=[NS(request_id='r0'), NS(request_id='r1')]))
            module = NS(__file__=str(runner), validate_slora_service_config=lambda *a: None,
                        verify_slora_source_view=lambda *a: None, SLORA_INPUT_KEYS=['model_dir'])
            with patch.dict(sys.modules, replay_vllm_engine_trace=module):
                self.assertEqual(len(audit_slora_receipts(root, cfg, q, {'clock_id': 'local'}, http, native)), 2)
                Path(allocations[0]['journal']).write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'journal differs'):
                    audit_slora_receipts(root, cfg, q, {'clock_id': 'local'}, http, native)


class ResidentReadyTests(unittest.TestCase):
    def fixture(self):
        cfg = dict(gpu_ids=[2, 3], engine_args={'model':'frozen'}, native_toolchain={'sha':'frozen'})
        service = dict(cgroup='/owned', affinity=[4, 5])
        row = dict(schema='ieee_tc_vllm_backend_ready_v1', index=1, cuda_visible='3',
            clock_id='local', identity=dict(cgroup='/owned',affinity=[4,5],pid=123,start_ticks=9),
            engine_args=cfg['engine_args'],native_toolchain=cfg['native_toolchain'],at=10.)
        control = dict(tc_route_ordinal=3,instance_id='vllm-1',tc_clock_id='local',tc_backend_entry_s=11.)
        return row, cfg, control, service

    def test_actual_ordinal_mapping_not_arrival_order(self):
        row,cfg,c,svc = self.fixture()
        self.assertEqual(validate_vllm_ready_row(row,cfg,c,3,svc,'local'),1)
        for field,value in [('index',0),('cuda_visible','2'),('clock_id','different'),
                            ('schema','ieee_tc_native_backend_ready_v1')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_vllm_ready_row(dict(row,**{field:value}),cfg,c,3,svc,'local')

    def test_changed_policy_or_toolchain_rejected(self):
        row,cfg,c,svc = self.fixture()
        for field,value in [('engine_args',{}),('native_toolchain',{})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_vllm_ready_row(dict(row,**{field:value}),cfg,c,3,svc,'local')
        for field,value in [('tc_route_ordinal',2),('instance_id','vllm-0')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_vllm_ready_row(row,cfg,dict(c,**{field:value}),3,svc,'local')

    def test_actual_process_resource_and_birth_required(self):
        row,cfg,c,svc = self.fixture()
        for field,value in [('cgroup','/other'),('affinity',[0]),('pid',0),('start_ticks',None)]:
            bad = dict(row,identity=dict(row['identity'],**{field:value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_vllm_ready_row(bad,cfg,c,3,svc,'local')

    def test_ready_must_precede_service_in_same_finite_clock(self):
        row,cfg,c,svc = self.fixture()
        for at in (12.,float('nan'),None):
            with self.subTest(at=at), self.assertRaises(ValueError):
                validate_vllm_ready_row(dict(row,at=at),cfg,c,3,svc,'local')


class QualificationPopulationTests(unittest.TestCase):
    def fixture(self, coverage, n=None):
        n = n if n is not None else 500 if coverage else 100
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
                  identity=lambda: dict(count=n, source_count=4000, profile='W0',
                                        rate_scale=1., view_sha256='unit-view'))
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

    def test_complete_unscaled_w0_population_passes(self):
        cfg, ready, contracts, native, _ = self.fixture(False, n=4000)
        self.assertEqual(validate_population(cfg, ready, contracts, native, 'full4000'), 4000)

    def test_complete_w1_requires_explicit_population_and_matched_identity(self):
        cfg, ready, contracts, native, plan = self.fixture(False, n=4000)
        with self.assertRaisesRegex(ValueError, 'complete unscaled W1'):
            validate_population(cfg, ready, contracts, native, 'full4000_w1')
        identity = dict(plan.identity(), profile='W1', view_sha256='w1-view')
        plan.profile = 'W1'
        plan.identity = lambda: identity
        self.assertEqual(validate_population(cfg, dict(plan=identity), contracts, native, 'full4000_w1'), 4000)
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'full4000_w1')
        with self.assertRaises(ValueError):
            validate_population(cfg, dict(plan=identity), contracts, native, 'full4000')
        for key, value in (('source_count', 5000), ('count', 1000), ('rate_scale', 2.)):
            changed = dict(identity, **{key:value})
            plan.identity = lambda: changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_population(cfg, dict(plan=changed), contracts, native, 'full4000_w1')

    def test_full4000_rejects_short_runs_and_unknown_aliases(self):
        cfg, ready, contracts, native, _ = self.fixture(False, n=1000)
        for name in ('full4000', 'full', 'prefix4000'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_population(cfg, ready, contracts, native, name)

    def test_full_w2_requires_map_and_per_request_rotation_provenance(self):
        cfg, ready, contracts, native, plan = self.fixture(False,n=4000)
        identity = dict(plan.identity(),profile='W2',view_sha256='w2-view',
                        adapter_map='/frozen/map.json',adapter_map_sha256='map-sha')
        cfg.update(adapter_map=identity['adapter_map'],adapter_map_sha256='map-sha')
        plan.profile,plan.identity = 'W2',lambda: identity
        for entry in plan.entries:
            row = json.loads(entry.source_json)
            row['tc_adapter_rotation'] = dict(adapter_map_sha256='map-sha',request=entry.request_id)
            entry.source_json = json.dumps(row)
            contracts[entry.request_id]['tc_adapter_rotation'] = row['tc_adapter_rotation']
        ready = dict(plan=identity)
        self.assertEqual(validate_population(cfg,ready,contracts,native,'full4000_w2'),4000)
        for population in ('full4000','full4000_w1'):
            with self.assertRaises(ValueError):
                validate_population(cfg,ready,contracts,native,population)
        for change in (dict(adapter_map=None),dict(adapter_map_sha256='wrong')):
            with self.assertRaises(ValueError):
                validate_population(dict(cfg,**change),ready,contracts,native,'full4000_w2')
        contracts['r0'].pop('tc_adapter_rotation')
        with self.assertRaisesRegex(ValueError,'rotation derivation'):
            validate_population(cfg,ready,contracts,native,'full4000_w2')

    def test_full4000_cannot_be_prefix_scaled_or_different_profile(self):
        cfg, ready, contracts, native, plan = self.fixture(False, n=4000)
        original = plan.identity()
        for key, value in (('source_count', 5000), ('source_count', None),
                           ('count', 1000), ('rate_scale', 2.), ('profile', 'W1')):
            changed = dict(original, **{key: value})
            plan.identity = lambda: changed
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'complete unscaled'):
                validate_population(cfg, dict(plan=changed), contracts, native, 'full4000')

    def test_full4000_rejects_coverage_remapping_and_missing_offered_row(self):
        cfg, ready, contracts, native, _ = self.fixture(False, n=4000)
        contracts['r0']['tc_adapter_coverage'] = {}
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'full4000')
        del contracts['r0']
        with self.assertRaises(ValueError):
            validate_population(cfg, ready, contracts, native, 'full4000')


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


class TerminalOutcomeTests(unittest.TestCase):
    def fixture(self, failed=0):
        summary = dict(counts=dict(N_plan=20, N_arrived=20, N_submitted=20,
            N_terminal=20, N_response=20-failed, N_failed=failed),
            measurement_complete=True, workload_passed=failed == 0)
        qualifier = dict(passed=failed == 0)
        if failed:
            qualifier['error'] = 'ValueError: external HTTP qualification failed; do not label binding as correctness'
        return summary, qualifier

    def test_success_default_cannot_accept_failures(self):
        self.assertEqual(validate_terminal_outcome(*self.fixture(), 20, 'success'), 20)
        with self.assertRaisesRegex(ValueError, 'success outcome'):
            validate_terminal_outcome(*self.fixture(1), 20, 'success')

    def test_failure_diagnostic_is_explicit_and_does_not_rewrite_inputs(self):
        summary, qualifier = self.fixture(3)
        before = copy.deepcopy((summary, qualifier))
        self.assertEqual(validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures'), 17)
        self.assertEqual((summary, qualifier), before)
        self.assertFalse(summary['workload_passed'])
        with self.assertRaisesRegex(ValueError, 'explicit complete'):
            validate_terminal_outcome(*self.fixture(), 20, 'complete_with_failures')

    def test_incomplete_population_and_other_errors_cannot_be_relabeled(self):
        for key in ('N_plan', 'N_arrived', 'N_submitted', 'N_terminal', 'N_response', 'N_failed'):
            summary, qualifier = self.fixture(3)
            summary['counts'][key] -= 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures')
        for key in ('physical_confirmation_error', 'worker_audit_error', 'delete_error',
                    'physical_release_error', 'physical_measurement_error', 'error'):
            summary, qualifier = self.fixture(3)
            qualifier[key] = 'unrelated failure'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures')
        summary, qualifier = self.fixture(3)
        summary['measurement_complete'] = False
        with self.assertRaises(ValueError):
            validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures')

    def test_zero_response_population_does_not_invent_success(self):
        summary, qualifier = self.fixture(20)
        self.assertEqual(validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures'), 0)
        with self.assertRaises(ValueError):
            validate_terminal_outcome(summary, qualifier, 20, 'success')
        rows = [dict(request_id=f'r{i}', status='http_request_failed') for i in range(20)]
        result = frozen_slo_summary(rows, {}, [])
        self.assertEqual(result['offered'], 20)
        self.assertEqual(result['binding_valid'], 0)
        self.assertEqual(result['joint_attainment'], 0.)
        self.assertIsNone(result['conditional_tpot_attainment'])
        self.assertFalse(result['observed_feasible'])


class ResidentFailureOutcomeTests(unittest.TestCase):
    def fixture(self, backend='loquetier', mode='loquetier_static_shards_v1'):
        summary, qualifier = TerminalOutcomeTests().fixture(3)
        qualifier.update(schema=f'ieee_tc_{backend}_native_model_qualification_v1',
            error='ValueError: incomplete/failed native HTTP workload; retained for diagnosis',
            physical_terminal_error='physical or native workload measurement incomplete',
            physical_lifecycle=dict(mode=mode, measurement_complete=True),
            http_completion=dict(event='http_replay_complete',
                **{k: v for k, v in summary['counts'].items() if k != 'N_submitted'}))
        physical = dict(contract='physical_gpu_resource_summary_v1', measurement_complete=True,
            eligible_physical_measurement=True, open_lease_ids=[], n_plan=20, n_terminal=20,
            n_native_contract_complete=17, n_interrupted=0)
        return summary, qualifier, physical

    def validate(self, summary, qualifier, physical, contract='loquetier_static_shards_v1'):
        return validate_terminal_outcome(summary, qualifier, 20, 'complete_with_failures', contract, physical)

    def test_complete_failure_diagnostic_preserves_inputs_and_success_rejection(self):
        for backend, contract, mode in (
                ('loquetier', 'loquetier_static_shards_v1', 'loquetier_static_shards_v1'),
                ('slora', 'slora_resident_rr_v1', 'slora_resident_pool_v1'),
                ('vllm', 'resident_rr_v1', 'vllm_resident_pool_v1')):
            summary, qualifier, physical = self.fixture(backend, mode)
            before = copy.deepcopy((summary, qualifier, physical))
            self.assertEqual(self.validate(summary, qualifier, physical, contract), 17)
            self.assertEqual((summary, qualifier, physical), before)
            with self.assertRaisesRegex(ValueError, 'success outcome'):
                validate_terminal_outcome(summary, qualifier, 20, 'success', contract, physical)

    def test_resource_or_population_failure_cannot_be_relabeled(self):
        changes = dict(contract='other', measurement_complete=False,
            eligible_physical_measurement=False, open_lease_ids=['still-live'],
            n_plan=19, n_terminal=19, n_native_contract_complete=16, n_interrupted=1)
        for key, value in changes.items():
            summary, qualifier, physical = self.fixture()
            physical[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'closed physical evidence'):
                self.validate(summary, qualifier, physical)
        with self.assertRaises(ValueError):
            self.validate(*self.fixture()[:2], None)

    def test_foreign_error_schema_release_and_lifecycle_are_rejected(self):
        for key, value in dict(schema='other', error='another failure',
                physical_terminal_error='another failure', physical_release_errors=['missing release'],
                physical_lifecycle=dict(mode='loquetier_static_shards_v1', measurement_complete=False)).items():
            summary, qualifier, physical = self.fixture()
            qualifier[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(summary, qualifier, physical)
        summary, qualifier, physical = self.fixture()
        with self.assertRaises(ValueError):
            self.validate(summary, qualifier, physical, 'original')

    def test_terminal_receipt_must_match_every_count(self):
        for key in ('N_plan', 'N_arrived', 'N_terminal', 'N_response', 'N_failed', 'event'):
            summary, qualifier, physical = self.fixture()
            qualifier['http_completion'][key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(summary, qualifier, physical)


class FrozenSloTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(request_id=f'r{i}', status='http_response', actual_tokens=2,
                     target_tokens=2, ttft_ms=10., tpot_ms=2.) for i in range(20)]
        observations = {r['request_id']: dict(native_prompt_token_ids=[1], native_output_tokens=2)
                        for r in rows}
        groups = [dict(lower_exclusive=None, upper_inclusive=None, ttft_ms=10., tpot_ms=2.)]
        return rows, observations, groups

    def test_joint_intersection_and_exact_95_percent_boundary(self):
        rows, obs, groups = self.fixture()
        rows[0]['tpot_ms'] = 3.
        value = frozen_slo_summary(rows, obs, groups)
        self.assertEqual(value['joint_attainment'], .95)
        self.assertTrue(value['observed_feasible'])
        rows[1]['ttft_ms'] = 11.
        value = frozen_slo_summary(rows, obs, groups)
        self.assertEqual(value['joint_attainment'], .9)
        self.assertFalse(value['observed_feasible'])

    def test_failed_request_keeps_offered_denominator(self):
        rows, obs, groups = self.fixture()
        rows[0] = dict(request_id='r0', status='http_request_failed')
        del obs['r0']
        value = frozen_slo_summary(rows, obs, groups)
        self.assertEqual(value['joint_good'], 19)
        self.assertEqual(value['offered'], 20)
        self.assertFalse(value['observed_feasible'])

    def test_single_token_tpot_is_not_imputed(self):
        rows, obs, groups = self.fixture()
        rows[0].update(actual_tokens=1, target_tokens=1, tpot_ms=None)
        obs['r0']['native_output_tokens'] = 1
        value = frozen_slo_summary(rows, obs, groups)
        self.assertEqual(value['tpot_eligible'], 19)
        self.assertEqual(value['joint_good'], 20)
        rows[0]['tpot_ms'] = 0.
        with self.assertRaises(ValueError):
            frozen_slo_summary(rows, obs, groups)

    def test_ambiguous_group_or_missing_native_timing_rejected(self):
        rows, obs, groups = self.fixture()
        with self.assertRaises(ValueError):
            frozen_slo_summary(rows, obs, groups * 2)
        rows[0]['tpot_ms'] = float('nan')
        with self.assertRaises(ValueError):
            frozen_slo_summary(rows, obs, groups)


class MatchedW0Tests(unittest.TestCase):
    def loquetier_fixture(self):
        contract = dict(adapter_id='a', native_backend='loquetier', request_id='r', target_tokens=2)
        native = dict(timing_contract='ieee_tc_loquetier_native_v1', native_adapter_path='/objects/a',
            native_adapter_id='a', native_request_id='r', native_adapter_index=0,
            native_adapter_binding_source='loquetier_actual_forward_segments_and_startup_pointer_tables',
            native_registered_monotonic_s=7., native_scheduled_monotonic_s=8.,
            native_first_token_monotonic_s=9., native_output_tokens=2,
            native_forward_count=2, native_terminal_observed=True)
        return contract, native

    def test_loquetier_registration_not_schedule_or_fabricated_vllm_entry(self):
        from scripts.ieee_tc.check_d299_full_context import matched_native_entry
        contract, native = self.loquetier_fixture()
        original = copy.deepcopy(native)
        self.assertEqual(matched_native_entry('Loquetier', contract, native), 7.)
        self.assertEqual(native, original)
        for key, value in [('native_registered_monotonic_s', None),
                           ('native_registered_monotonic_s', float('nan')),
                           ('native_registered_monotonic_s', True),
                           ('native_scheduled_monotonic_s', 6.),
                           ('native_scheduled_monotonic_s', 10.),
                           ('native_first_token_monotonic_s', None)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                matched_native_entry('Loquetier', contract, dict(native, **{key: value}))

    def test_loquetier_join_requires_actual_adapter_request_forward_binding(self):
        from scripts.ieee_tc.check_d299_full_context import matched_native_entry
        contract, native = self.loquetier_fixture()
        for key, value in [('timing_contract', 'ieee_tc_native_v1'),
                           ('native_adapter_id', 'wrong'), ('native_request_id', 'wrong'),
                           ('native_adapter_path', '/objects/b'), ('native_adapter_path', 'a'),
                           ('native_adapter_index', -1), ('native_adapter_index', True),
                           ('native_adapter_binding_source', 'request_name'),
                           ('native_forward_count', 1), ('native_forward_count', True),
                           ('native_output_tokens', 0), ('native_output_tokens', 3),
                           ('native_terminal_observed', False)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                matched_native_entry('Loquetier', contract, dict(native, **{key: value}))
        with self.assertRaises(ValueError):
            matched_native_entry('Loquetier', dict(contract, native_backend='vllm'), native)
        with self.assertRaises(ValueError):
            matched_native_entry('vLLM', contract, native)

    def test_slora_entry_uses_native_rpc_without_fabricated_vllm_fields(self):
        from scripts.ieee_tc.check_d299_full_context import matched_native_entry
        contract = dict(adapter_id='a', native_backend='slora')
        native = dict(timing_contract='ieee_tc_slora_native_v1', native_adapter_path='/objects/a',
            native_adapter_index=0, native_adapter_binding_source='slora_model_rpc_actual_batch_adapter',
            native_rpc_registered_monotonic_s=7.)
        original = copy.deepcopy(native)
        self.assertEqual(matched_native_entry('S-LoRA', contract, native), 7.)
        self.assertEqual(native, original)
        for key, value in [('timing_contract', 'ieee_tc_native_v1'),
                           ('native_adapter_path', '/objects/b'), ('native_adapter_path', 'a'),
                           ('native_adapter_index', True), ('native_adapter_index', -1),
                           ('native_adapter_binding_source', 'request_name'),
                           ('native_rpc_registered_monotonic_s', None),
                           ('native_rpc_registered_monotonic_s', float('nan'))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                matched_native_entry('S-LoRA', contract, dict(native, **{key: value}))
        with self.assertRaises(ValueError):
            matched_native_entry('vLLM', contract, native)
        self.assertEqual(matched_native_entry('vLLM', contract,
            dict(native_lora_name='a', native_dispatch_monotonic_s=9.)), 9.)

    def test_nonfeasible_baseline_retains_raw_difference_not_g1_claim(self):
        from scripts.ieee_tc.check_d299_full_context import matched_relative_observation
        prime = dict(physical_gpu_seconds=90., joint_slo=.977, observed_protocol_feasible=True,
                     metrics={'ttft_ms': {'p95_type1': 3.}})
        baseline = dict(physical_gpu_seconds=100., joint_slo=.94425, observed_protocol_feasible=False,
                        metrics={'ttft_ms': {'p95_type1': 5.}})
        result = matched_relative_observation(prime, baseline)
        self.assertEqual(result['gpu_seconds_relative_reduction'], .1)
        self.assertEqual(result['p95_ttft_relative_reduction'], .4)
        self.assertAlmostEqual(result['joint_slo_difference_percentage_points'], 3.275)
        self.assertFalse(result['both_observed_protocol_feasible'])
        self.assertTrue(matched_relative_observation(prime,
            dict(baseline, observed_protocol_feasible=True))['both_observed_protocol_feasible'])

    def test_existing_resident_reference_cannot_be_refrozen(self):
        from scripts.ieee_tc.check_d299_full_context import require_unchanged_reference
        frozen = dict(U_ref_gpu_seconds=16000., Q_ref_p95_ttft_ms=1200., B_gpu_seconds=12000.,
                      reference_id='original', source_audit_sha256='source')
        self.assertIs(require_unchanged_reference(dict(frozen), frozen), frozen)
        for key in frozen:
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_unchanged_reference(dict(frozen, **{key: 'changed'}), frozen)

    def fixture(self):
        return dict(request_id='r', source_item_sha256='source', adapter_id='a',
            target_tokens=256, canonical_prompt_sha256='prompt',
            native_prompt_token_ids_sha256='native', native_prompt_tokens=759,
            artifact_content_sha256='content', generation_contract='fixed_length_greedy_v1',
            arrival_offset_s=1., output_digest='original')

    def test_each_work_field_and_arrival_must_match(self):
        from scripts.ieee_tc.check_d299_full_context import matched_request_contract
        left=self.fixture()
        matched_request_contract(left,dict(left))
        for key in left.keys()-{'output_digest'}:
            bad=dict(left);bad[key]='changed' if key!='arrival_offset_s' else 1.01
            with self.subTest(key=key), self.assertRaises(ValueError):
                matched_request_contract(left,bad)
        for value in (None, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                matched_request_contract(left,dict(left,arrival_offset_s=value))

    def test_output_changes_do_not_change_input_eligibility(self):
        from scripts.ieee_tc.check_d299_full_context import matched_request_contract
        left=self.fixture()
        matched_request_contract(left,dict(left,output_digest='changed',arrival_offset_s=1.+1e-8))
        bad=dict(left);del bad['artifact_content_sha256']
        with self.assertRaises(ValueError):
            matched_request_contract(left,bad)

    def physical(self):
        return dict(measurement_complete=True,open_lease_ids=[],n_plan=4000,n_terminal=4000,
            n_native_contract_complete=4000,device_intervals={'GPU-a':[[1.,11.]],'GPU-b':[[2.,12.]]},
            gpu_seconds_observed=20.,window_gpu_seconds={'pre_arrival':2.,'arrival':15.,'drain':2.,'cleanup':1.})

    def test_resource_union_and_partition(self):
        from scripts.ieee_tc.check_d299_full_context import closed_physical_integral
        self.assertEqual(closed_physical_integral(self.physical()),20.)
        for k,v in [('measurement_complete',False),('open_lease_ids',['open']),('n_terminal',3999),
                    ('n_native_contract_complete',3999),('device_intervals',{}),
                    ('gpu_seconds_observed',19.),('window_gpu_seconds',{'arrival':19.})]:
            with self.subTest(key=k), self.assertRaises(ValueError):
                closed_physical_integral(dict(self.physical(),**{k:v}))

    def test_duplicate_ownership_and_nonfinite_intervals_rejected(self):
        from scripts.ieee_tc.check_d299_full_context import closed_physical_integral
        for intervals in ([[1.,11.],[5.,6.]], [[1.,float('nan')]], [[11.,1.]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                closed_physical_integral(dict(self.physical(),device_intervals={'GPU':intervals}))

    def test_diagnostic_population_is_explicit_and_not_inferred_from_result(self):
        from scripts.ieee_tc.check_d299_full_context import closed_physical_integral
        prefix = dict(self.physical(),n_plan=100,n_terminal=100,n_native_contract_complete=100)
        self.assertEqual(closed_physical_integral(prefix,expected_count=100),20.)
        with self.assertRaises(ValueError):
            closed_physical_integral(prefix)
        for count in (True,0,-1,100.,None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                closed_physical_integral(prefix,expected_count=count)
        for field in ('n_plan','n_terminal','n_native_contract_complete'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                closed_physical_integral(dict(prefix,**{field:99}),expected_count=100)
        for changed in (dict(measurement_complete=False),dict(open_lease_ids=['open']),
                        dict(gpu_seconds_observed=19.),dict(device_intervals={}),
                        dict(window_gpu_seconds={'arrival':19.}),
                        dict(device_intervals={'GPU':[[1.,11.],[5.,6.]]})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                closed_physical_integral(dict(prefix,**changed),expected_count=100)


class ElasticLocalityMatchedTests(unittest.TestCase):
    def row(self, rid='r'):
        return dict(request_id=rid, source_item_sha256='source', adapter_id='a',
            target_tokens=2, canonical_prompt_sha256='prompt',
            native_prompt_token_ids_sha256='input', native_prompt_tokens=3,
            artifact_content_sha256='content', generation_contract='fixed_length_greedy_v1',
            arrival_offset_s=1., native=dict(completion_token_ids_sha256='a'*64,
                                           native_output_tokens=2))

    def test_same_backend_differences_retained_without_automatic_explanation(self):
        from scripts.ieee_tc.check_d299_full_context import matched_output_digest_audit
        left = {rid:self.row(rid) for rid in ('r1','r2')}
        right = copy.deepcopy(left)
        right['r2']['native']['completion_token_ids_sha256'] = 'b'*64
        audit = matched_output_digest_audit(left, right, ['r1','r2'])
        self.assertEqual((audit['equal'], audit['changed']), (1, 1))
        self.assertEqual(audit['changed_request_ids'], ['r2'])
        self.assertEqual(audit['by_adapter'], {'a':dict(total=2, changed=1)})
        self.assertFalse(audit['independent_numerical_proof'])
        self.assertFalse(audit['first_divergence_and_logits_available'])

    def test_missing_invalid_or_short_outputs_cannot_be_treated_as_equal(self):
        from scripts.ieee_tc.check_d299_full_context import matched_output_digest_audit
        left = {'r':self.row()}
        for field, value in [('completion_token_ids_sha256', None),
                             ('completion_token_ids_sha256', ''),
                             ('completion_token_ids_sha256', 'x'*64),
                             ('native_output_tokens', 1)]:
            right = copy.deepcopy(left)
            right['r']['native'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                matched_output_digest_audit(left, right, ['r'])

    def test_output_audit_rejects_missing_duplicate_or_mismatched_input(self):
        from scripts.ieee_tc.check_d299_full_context import matched_output_digest_audit
        left = {'r':self.row()}
        for right, order in [({}, ['r']), (left, ['r','r']),
                             ({'r':dict(self.row(), adapter_id='wrong')}, ['r'])]:
            with self.assertRaises(ValueError):
                matched_output_digest_audit(left, right, order)

    def test_extension_cannot_drop_prior_rows(self):
        from scripts.ieee_tc.check_d299_full_context import compare_d424_sealed_w0
        with tempfile.TemporaryDirectory() as directory:
            for slora, loquetier in [(False, False), (True, False), (False, True)]:
                with self.subTest(slora=slora, loquetier=loquetier), self.assertRaises(ValueError):
                    compare_d424_sealed_w0(Path(directory)/'new', slora, loquetier, True)

    def test_streamed_controller_loader_requires_explicit_complete_scenario(self):
        from scripts.ieee_tc.check_d299_full_context import matched_controller_rows
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.json'
            source.write_text(json.dumps(dict(detailed_results=dict(tc_ieee_full=dict(requests=[])))))
            for scenario in ('tc_ieee_full','tc_ieee_elastic_locality','wrong'):
                with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                    matched_controller_rows(source, scenario, {})


if __name__ == '__main__':
    unittest.main()
