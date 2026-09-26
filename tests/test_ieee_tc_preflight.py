import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import ieee_tc_preflight as p


class ProtocolGates(unittest.TestCase):
    def test_host_allocator_controls_selected_before_measurement(self):
        rows = [dict(adapter_id=a, weight_bytes=b, weight_sha256=s,
            configured_rank=r, target_modules=['v_proj','q_proj'], inspected=True,
            all_finite=True) for a,b,s,r in [('b',8,'same',8),('a',8,'same',8),('c',16,'large',16)]]
        audit=dict(kind='existing_artifact_tensor_audit_v1', audit_complete=True,
            pools=[dict(root='/frozen',complete=True,rows=rows)])
        self.assertEqual([r['adapter_id'] for r in p.select_host_allocator_controls(audit)], ['a','c'])
        for replacement in (dict(audit_complete=False),dict(pools=[])):
            with self.assertRaises(ValueError):
                p.select_host_allocator_controls({**audit,**replacement})
        rows[0]['all_finite']=False
        with self.assertRaisesRegex(ValueError,'finite'):
            p.select_host_allocator_controls(audit)

    def test_host_allocator_check_requires_guard_before_inputs_or_cuda(self):
        with patch.object(p,'verify_current_service',side_effect=RuntimeError('no guard')):
            with self.assertRaisesRegex(RuntimeError,'no guard'):
                p.backend_host_allocator_check(Path('/missing'),Path('/missing'))

    def test_numeric_controls_use_existing_nonzero_and_same_rank_zero_content(self):
        from types import SimpleNamespace
        rows = [dict(adapter_id=aid, configured_rank=8, all_finite=True,
                     all_tensors_zero=zero, all_ab_updates_provably_zero=zero,
                     weight_sha256=sha) for aid, zero, sha in
                [('a', False, 'sha-a'), ('alias', False, 'sha-a'),
                 ('zero', True, 'sha-zero'), ('b', False, 'sha-b')]]
        audit = dict(audit_complete=True, pools=[dict(root='/frozen', complete=True, rows=rows)])
        entries = [SimpleNamespace(source_json=json.dumps({'adapter_id': aid}))
                   for aid in ('a', 'alias', 'zero', 'a', 'b')]
        controls = p.select_numeric_controls(audit, Path('/frozen'), entries)
        self.assertEqual({k: v['adapter_id'] for k,v in controls.items()},
                         dict(nonzero_a='a', nonzero_b='b', zero='zero'))
        rows[-1]['configured_rank'] = 16
        with self.assertRaisesRegex(ValueError, 'same-rank'):
            p.select_numeric_controls(audit, Path('/frozen'), entries)
        rows[0]['all_tensors_zero'] = True
        with self.assertRaisesRegex(ValueError, 'nonzero operands'):
            p.select_numeric_controls(audit, Path('/frozen'), entries)

    def test_numeric_control_compares_probabilities_not_just_argmax(self):
        a = dict(prompt_sha256='p', native_prompt_ids_sha256='t', output_token_ids=[7, 8],
                 first_token_logprobs={'7': -1., '8': -2.})
        b = {**a, 'first_token_logprobs': {'7': -1.25, '8': -2.5}}
        result = p.compare_first_token_probabilities(a, b)
        self.assertTrue(result['all_output_tokens_equal'])
        self.assertEqual(result['max_abs_logprob_difference'], .5)
        self.assertEqual(result['common_token_count'], 2)
        self.assertEqual(p.compare_first_token_probabilities(a,a)['max_abs_logprob_difference'], 0.)
        self.assertIsNone(p.compare_first_token_probabilities(a,
            {**b, 'first_token_logprobs': {'9': -2.}})['max_abs_logprob_difference'])
        for bad in ({**b, 'prompt_sha256': 'wrong'},
                    {**b, 'first_token_logprobs': {'7': float('nan')}}):
            with self.assertRaises(ValueError):
                p.compare_first_token_probabilities(a, bad)

    def test_concurrent_qualification_copies_exact_native_mapping(self):
        mapping = {'request-a': ['native-a-random'], 'request-b': ['native-b-random']}
        result = p.qualification_request_mapping(['request-a', 'request-b'], mapping)
        self.assertEqual(result, {'request-a': 'native-a-random', 'request-b': 'native-b-random'})
        mapping['request-a'].clear()
        self.assertEqual(result['request-a'], 'native-a-random')
        self.assertIsNone(p.qualification_request_mapping(['request-a'], mapping))
        self.assertIsNone(p.qualification_request_mapping(['missing'], mapping))

    def test_concurrent_qualification_rejects_ambiguous_native_mapping(self):
        for mapping in ({'a':['one','two']}, {'a':[None]}, {'a':['']},
                        {'a':['same'], 'b':['same']}):
            with self.subTest(mapping=mapping), self.assertRaises(RuntimeError):
                p.qualification_request_mapping(list(mapping), mapping)

    def test_qualification_eviction_distinguishes_native_lru_absence_from_failure(self):
        p.validate_qualification_eviction({'evicted': True, 'reason': 'removed'}, present_before=True)
        p.validate_qualification_eviction({'evicted': False, 'reason': 'absent'}, present_before=False)
        for receipt, present in (({'evicted': False, 'reason': 'absent'}, True),
                                 ({'evicted': True, 'reason': 'removed'}, False),
                                 ({'evicted': False, 'reason': 'referenced'}, False),
                                 ({'evicted': False, 'reason': 'externally_pinned'}, False),
                                 ({'evicted': 1, 'reason': 'removed'}, True), ({}, False)):
            with self.subTest(receipt=receipt, present=present), self.assertRaises(RuntimeError):
                p.validate_qualification_eviction(receipt, present_before=present)

    def test_model_check_requires_guard_before_reading_inputs(self):
        import asyncio
        with patch.object(p, 'verify_current_service', side_effect=RuntimeError('no guard')):
            with self.assertRaisesRegex(RuntimeError, 'no guard'):
                asyncio.run(p.backend_model_check(Path('/missing'), Path('/missing'),
                                                  'profile', Path('/missing'), 4))

    def test_model_check_rejects_unqualified_runtime_before_engine_import(self):
        import asyncio
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'runtime.json'
            receipt.write_text(json.dumps({'kind': 'backend_cuda_import_qualification_v1',
                                          'pass': False}))
            with patch.object(p, 'verify_current_service', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'completed CUDA check'):
                    asyncio.run(p.backend_model_check(receipt, Path('/missing'),
                                                     'profile', Path('/missing'), 4))

    def test_model_worker_requires_actual_owner_affinity_clock_and_device(self):
        pid = p.os.getpid()
        actual = dict(pid=pid, uid=p.os.getuid(), cgroup='/service', affinity=[4, 28])
        observation = dict(kind='ieee_native_worker_qualification_observation',
                           pid=pid, uid=p.os.getuid(), cgroup=Path(f'/proc/{pid}/cgroup').read_text().strip(),
                           affinity=[4, 28], clock_id='clock', visible_gpu_count=1, backend_version='0.30.0')
        service = {'service_identity': {'path': '/service'}}
        with patch.object(p, 'gpu_process_identity', return_value=actual):
            p.validate_model_worker(observation, service, 'clock')
            for change in ({'clock_id':'wrong'}, {'visible_gpu_count':4}, {'backend_version':'0.10.2'},
                           {'affinity':[0]}, {'cgroup':'wrong'}):
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'actual model worker'):
                    p.validate_model_worker({**observation, **change}, service, 'clock')
        for identity in (None, {**actual, 'cgroup':'/escaped'}, {**actual, 'affinity':[0]}):
            with patch.object(p, 'gpu_process_identity', return_value=identity):
                with self.assertRaisesRegex(RuntimeError, 'actual model worker'):
                    p.validate_model_worker(observation, service, 'clock')

    def test_backend_check_requires_guard_before_reading_install_or_importing_cuda(self):
        with patch.object(p, 'verify_current_service', side_effect=RuntimeError('no guard')):
            with self.assertRaisesRegex(RuntimeError, 'no guard'):
                p.backend_runtime_check(Path('/missing/install'), Path('/missing/requirements'))

    def test_backend_check_rejects_incomplete_or_different_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements, receipt = root / 'requirements.txt', root / 'install.json'
            requirements.write_text('tiny-fixture-not-installable\n')
            setup = dict(kind='isolated_backend_dependency_install', **{'pass': True},
                         plan_sha256='plan', requirements_sha256=p.digest(requirements),
                         environment=p.sys.prefix, steps=[dict(returncode=0)] * 3)
            for changes in ({'pass': False}, {'environment': '/wrong/venv'},
                            {'requirements_sha256': 'wrong'}, {'steps': [dict(returncode=0)]}):
                receipt.write_text(json.dumps({**setup, **changes}))
                with patch.object(p, 'verify_current_service', return_value={}), \
                     patch.object(p, 'check_plan', return_value='plan'):
                    with self.assertRaisesRegex(RuntimeError, 'completed hash-locked'):
                        p.backend_runtime_check(receipt, requirements)

    def test_backend_import_failure_is_preserved_without_another_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements, receipt = root / 'requirements.txt', root / 'install.json'
            requirements.write_text('tiny-fixture-not-installable\n')
            receipt.write_text(json.dumps(dict(kind='isolated_backend_dependency_install',
                **{'pass': True}, plan_sha256='plan', requirements_sha256=p.digest(requirements),
                environment=p.sys.prefix, steps=[dict(returncode=0)] * 3)))
            with patch.object(p, 'verify_current_service', return_value={}), \
                 patch.object(p, 'check_plan', return_value='plan'), \
                 patch('importlib.import_module', side_effect=ImportError('native library missing')) as load:
                result = p.backend_runtime_check(receipt, requirements)
            self.assertFalse(result['pass'])
            self.assertFalse(result['model_qualification'])
            self.assertEqual(result['stage'], 'import:torch')
            self.assertEqual(result['error_type'], 'ImportError')
            load.assert_called_once_with('torch')

    def test_backend_check_uses_candidate_native_extension_not_legacy_module(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements, receipt = root / 'requirements.txt', root / 'install.json'
            requirements.write_text('tiny-fixture-not-installable\n')
            receipt.write_text(json.dumps(dict(kind='isolated_backend_dependency_install',
                **{'pass': True}, plan_sha256='plan', requirements_sha256=p.digest(requirements),
                environment=p.sys.prefix, steps=[dict(returncode=0)] * 3)))
            module = SimpleNamespace(__file__='/fixture/module',
                                     cuda=SimpleNamespace(is_available=lambda: False))
            with patch.object(p, 'verify_current_service', return_value={}), \
                 patch.object(p, 'check_plan', return_value='plan'), \
                 patch('importlib.metadata.version', return_value='0.30.0'), \
                 patch('importlib.import_module', return_value=module) as load:
                result = p.backend_runtime_check(receipt, requirements)
            names = [call.args[0] for call in load.call_args_list]
            self.assertIn('vllm._C_stable_libtorch', names)
            self.assertNotIn('vllm._C', names)
            self.assertEqual(result['stage'], 'cuda_device')
            self.assertFalse(result['pass'])
            self.assertIn('usable CUDA GPU', result['error'])

    def test_startup_margin_is_consistent(self):
        self.assertEqual(p.memory_required(), 102 * p.GIB)
        self.assertEqual(p.memory_required(10*p.GIB, p.GIB), 91*p.GIB)
        with self.assertRaises(ValueError):
            p.memory_required(81*p.GIB)

    def test_disk_threshold_is_incremental(self):
        self.assertEqual(p.disk_required(0), 150*p.GIB)
        self.assertEqual(p.disk_required(100*p.GIB), 250*p.GIB)
        with self.assertRaises(ValueError):
            p.disk_required(-1)

    def test_cpu_sets_preserve_smt_pairs(self):
        svc = set(p.POLICY['service_cpus'])
        aux = set(p.POLICY['aux_cpus'])
        reserve = set(p.POLICY['reserved_cpus'])
        self.assertEqual(len(svc), 40)
        self.assertFalse(svc & aux or svc & reserve or aux & reserve)
        self.assertEqual(svc | aux | reserve, set(range(48)))
        for group in (svc, aux, reserve):
            self.assertTrue(all((c % 24) in group and c % 24 + 24 in group for c in group))

    def test_source_plan_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'plan', Path(d)/'snapshot'
            a.write_text('approved\n'); b.write_text('approved\n')
            with patch.object(p, 'PLAN', a), patch.object(p, 'SNAPSHOT', b):
                self.assertEqual(p.check_plan(), p.digest(a))
                a.write_text('changed\n')
                with self.assertRaises(RuntimeError):
                    p.check_plan()

    def test_protected_deleted_added_modified_and_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'protected'; root.mkdir()
            f = root/'result'; f.write_text('old')
            (root/'link').symlink_to(f)
            before = p.protected_entries([root])
            manifest = Path(d)/'manifest.json'
            manifest.write_text(json.dumps({'roots':[str(root)], 'entries':before}))
            with patch.object(p, 'check_plan', return_value='test'):
                self.assertTrue(p.verify_seal(manifest)['pass'])
                f.write_text('different')
                self.assertEqual(len(p.verify_seal(manifest)['changed']), 2)
                f.unlink()
                self.assertFalse(p.verify_seal(manifest)['pass'])

    def test_cannot_kill_unowned_unit(self):
        with self.assertRaises(ValueError):
            p.stop_own_unit('user@1001.service')

    def test_hard_limit_witness_isolates_throttling(self):
        self.assertIn('MemoryHigh=128M', p.scope_command('test', 'oom'))
        self.assertIn('MemoryHigh=64M', p.scope_command('test', 'inspect'))
        self.assertEqual(p.POLICY['service_high_bytes'], 72*p.GIB)
        self.assertEqual(p.test_limits('ray')['memory.max'], 3*p.GIB)
        self.assertIn('MemoryMax=3072M', p.scope_command('test', 'ray'))
        self.assertEqual(p.test_limits('replay'), {'memory.high':192*p.MIB,
                         'memory.max':256*p.MIB, 'memory.swap.max':0})

    def test_cleanup_stops_empty_owned_scope(self):
        with patch.object(p.subprocess, 'run') as run:
            p.stop_own_unit('primelora-tc-test-unit.scope')
            self.assertEqual(run.call_count, 2)
            self.assertIn('stop', run.call_args.args[0])

    def test_watchdog_warning_does_not_classify_service_failure(self):
        watch = p.WatchdogDecision()
        outcome = watch.observe(23*p.GIB, 1, [])
        self.assertTrue(outcome['warning'])
        self.assertEqual(outcome['abort_reasons'], [])
        self.assertIsNone(outcome['classification'])
        self.assertFalse(watch.observe(24*p.GIB, 1, [])['warning'])

    def test_watchdog_stops_below_not_at_16_GiB(self):
        watch = p.WatchdogDecision()
        self.assertEqual(watch.observe(16*p.GIB, 0, [])['abort_reasons'], [])
        outcome = watch.observe(16*p.GIB-1, 0, [])
        self.assertIn('host_memory_below_stop', outcome['abort_reasons'])
        self.assertEqual(outcome['classification'], 'safety_abort_unattributed')

    def test_pressure_requires_ten_consecutive_joint_samples(self):
        watch = p.WatchdogDecision()
        for _ in range(9):
            self.assertFalse(watch.observe(23*p.GIB, 10, [])['abort_reasons'])
        self.assertEqual(watch.observe(24*p.GIB, 10, [])['pressure_streak'], 0)
        for _ in range(9):
            self.assertFalse(watch.observe(23*p.GIB, 10, [])['abort_reasons'])
        self.assertIn('sustained_host_memory_pressure',
                      watch.observe(23*p.GIB, 10, [])['abort_reasons'])

    def test_disk_stop_and_inode_failure(self):
        watch = p.WatchdogDecision()
        disk = {'path':'/test', 'free_bytes':100*p.GIB, 'free_inodes':1}
        self.assertFalse(watch.observe(100*p.GIB, 0, [disk])['abort_reasons'])
        for bad in ({**disk, 'free_bytes':100*p.GIB-1}, {**disk, 'free_inodes':0}):
            self.assertIn('filesystem_below_stop:/test',
                          watch.observe(100*p.GIB, 0, [bad])['abort_reasons'])

    def test_scope_requires_unambiguous_owned_uuid(self):
        for unit in ('user@1001.service', 'ray.scope', 'primelora-tc-svc-name.scope'):
            with self.assertRaises(ValueError):
                p.scope_identity(unit)

    def test_scope_identity_change_never_signals(self):
        with tempfile.TemporaryDirectory() as d:
            identity = {'path':d, 'unit':'test', 'inode':1, 'invocation_id':'a'}
            with patch.object(p, 'scope_identity', return_value={**identity,'invocation_id':'b'}), \
                 patch.object(p.signal, 'pidfd_send_signal') as send:
                with self.assertRaises(RuntimeError):
                    p.stop_scope_identity(identity)
                send.assert_not_called()

    def test_watchdog_cannot_share_service_ancestry(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/service/worker')):
            with self.assertRaisesRegex(RuntimeError, 'outside service ancestry'):
                p.watch_scope({'path':'/test/service'}, paths=[], emit=lambda _:None)

    def test_sensor_invalidity_not_silent_zero(self):
        watch = p.WatchdogDecision()
        for available, psi in ((-1, 0), (100*p.GIB, float('nan')), (100*p.GIB, 101)):
            with self.assertRaises(ValueError):
                watch.observe(available, psi, [])

    def test_unsupported_interpreter_fails_before_launch_or_ready(self):
        with patch.object(p.signal, 'pidfd_send_signal', None, create=True), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'PID-handle signaling'):
                p.watchdog_test()
            launch.assert_not_called()

    def test_installation_cannot_run_outside_bounded_build_scope(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/ordinary.scope')), \
             patch.object(p, 'cgroup_snapshot', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'bounded build scope'):
                p.install_candidate(Path('/not-created'), Path('/not-read'), Path('/not-written'))

    def test_installation_limits_checked_before_creating_environment(self):
        group = Path('/test/primelora-tc-build-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.high':3*p.GIB,
                                                            'memory.max':'max', 'memory.swap.max':0}):
            with self.assertRaisesRegex(RuntimeError, 'effective before environment'):
                p.install_candidate(Path('/not-created'), Path('/not-read'), Path('/not-written'))

    def test_launch_cannot_start_model_before_auxiliary_limits_exist(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/unbounded.scope')), \
             patch.object(p, 'cgroup_snapshot', return_value={}), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'shared auxiliary scope'):
                p.gated_launch(['/usr/bin/python3'], Path('/not-written'))
            launch.assert_not_called()

    def test_tiny_gate_cannot_be_used_to_launch_an_unbounded_model(self):
        group = Path('/test/primelora-tc-aux-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.max':4*p.GIB, 'memory.swap.max':0}), \
             patch.object(p.os, 'sched_getaffinity', return_value=set(p.POLICY['aux_cpus'])), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'fixed no-GPU inheritance witness'):
                p.gated_launch(['/usr/bin/python3', 'model.py'], Path('/not-written'), tiny=True)
            launch.assert_not_called()

    def test_live_install_prevents_overlapping_model_launch(self):
        group = Path('/test/primelora-tc-aux-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.max':4*p.GIB, 'memory.swap.max':0}), \
             patch.object(p.os, 'sched_getaffinity', return_value=set(p.POLICY['aux_cpus'])), \
             patch.object(p.subprocess, 'check_output', return_value='primelora-tc-build-'+'b'*32+'.scope active'), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'another heavy setup'):
                p.gated_launch(['/usr/bin/python3'], Path('/not-written'))
            launch.assert_not_called()

    def test_ready_receipt_must_match_actual_watcher_birth_and_domain(self):
        identity = {'unit':'test', 'path':'/service', 'invocation_id':'one', 'inode':1}
        auxiliary = Path('/aux')
        proc = {'pid':123, 'start_ticks':5, 'affinity':p.POLICY['aux_cpus'], 'cgroup':'/aux'}
        event = {'event':'watchdog_ready', 'service_identity':identity, 'watchdog_pid':123,
                 'watchdog_process':proc, 'aux':{'path':'/aux'}}
        with patch.object(p, 'owned_pids', return_value=[proc]), \
             patch.object(p, 'scope_still_owned', return_value=True):
            self.assertEqual(p.verify_watchdog_attachment(event, identity, auxiliary), proc)
            with self.assertRaisesRegex(RuntimeError, 'another resource domain'):
                p.verify_watchdog_attachment(event, dict(identity, invocation_id='two'), auxiliary)
        with patch.object(p, 'owned_pids', return_value=[dict(proc, start_ticks=6)]):
            with self.assertRaisesRegex(RuntimeError, 'birth identity'):
                p.verify_watchdog_attachment(event, identity, auxiliary)

    def test_missing_launch_receipt_is_not_authorization(self):
        with patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'missing guarded'):
                p.verify_current_service()


if __name__ == '__main__':
    unittest.main()
