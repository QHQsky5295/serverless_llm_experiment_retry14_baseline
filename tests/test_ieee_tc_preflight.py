import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import ieee_tc_preflight as p


class ApprovedPlanIdentity(unittest.TestCase):
    def test_current_binding_matches_explicitly_approved_prime_first_plan(self):
        self.assertEqual(p.SNAPSHOT.name, 'PLAN_APPROVED_20261002_PRIME_FIRST.md')
        self.assertEqual(p.check_plan(),
            '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7')

    def test_plan_mismatch_still_refuses_instead_of_selecting_a_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, approved = Path(tmp)/'plan', Path(tmp)/'approved'
            source.write_text('changed')
            approved.write_text('original')
            with patch.object(p, 'PLAN', source), patch.object(p, 'SNAPSHOT', approved):
                with self.assertRaisesRegex(RuntimeError, 'Plan changed'):
                    p.check_plan()


class ForwardedCommandCLI(unittest.TestCase):
    def test_publisher_records_prefix_scope_and_rejects_whole_source_prefix(self):
        import io
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'trace.json'
            source.write_text(json.dumps(dict(requests=[dict(request_id=str(i), arrival_time_s=i)
                                                       for i in range(3)])))
            args = SimpleNamespace(replay_trace=source, replay_profile='W0', tiny_witness=False,
                diagnostic_prefix_count=2, output=root/'out.jsonl', gate_socket=str(root/'socket'),
                gate_nonce='fixture')
            observed = []
            async def publisher(plan, address, nonce, start, emit):
                observed.extend(e.request_id for e in plan.entries)
                self.assertEqual(await start(), {'fixture':'origin'})
                emit(dict(event='replay_ready', plan=plan.identity()))
            with patch('faaslora.datasets.workload_generator.publish_frozen_replay', publisher), \
                    patch.object(p, 'cg_path', return_value=Path('/fixture')), \
                    patch.object(p, 'cgroup_snapshot', return_value={'memory.max':p.POLICY['aux_max_bytes']}), \
                    patch('sys.stdin', io.StringIO('{"fixture":"origin"}\n')), \
                    patch('builtins.print'):
                p.replay_publisher(args)
                args.diagnostic_prefix_count = 3
                args.output = root/'rejected.jsonl'
                with self.assertRaisesRegex(ValueError, 'proper prefix'):
                    p.replay_publisher(args)
            self.assertEqual(observed, ['0','1'])
            event = json.loads((root/'out.jsonl').read_text())
            self.assertEqual(event['replay_scope'], 'diagnostic_prefix_v1')
            self.assertEqual(event['plan']['source_count'], 3)
            self.assertEqual(event['diagnostic_prefix_count'], 2)
            self.assertFalse(args.output.exists())

    def test_invalid_prefix_is_rejected_before_resources_or_workers(self):
        import os
        for options in (dict(diagnostic_prefix_count=0, replay_trace='trace'),
                        dict(diagnostic_prefix_count=True, replay_trace='trace'),
                        dict(diagnostic_prefix_count=2),
                        dict(diagnostic_prefix_count=2, replay_trace='trace', tiny=True),
                        dict(diagnostic_prefix_count=2, replay_trace='trace', http_replay_config='http')):
            with self.subTest(options=options), patch.object(p, 'require_watchdog_primitives') as guard:
                with self.assertRaisesRegex(ValueError, 'diagnostic prefix'):
                    p.gated_launch(['/bin/true'], Path('/unused'), **options)
                guard.assert_not_called()
        with patch.dict(os.environ, FAASLORA_FORMAL_RUN='1'), \
                patch.object(p, 'require_watchdog_primitives') as guard:
            with self.assertRaisesRegex(ValueError, 'diagnostic prefix'):
                p.gated_launch(['/bin/true'], Path('/unused'),
                    diagnostic_prefix_count=2, replay_trace='trace')
            guard.assert_not_called()

    def test_prefix_cli_is_explicit_and_forwards_the_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'launch.json'
            argv = ['preflight', 'gated-launch', '--output', str(output),
                    '--replay-trace', '/existing/trace.json', '--diagnostic-prefix-count', '1000',
                    '--exec', '/bin/echo', '--config', 'child.json']
            with patch('sys.argv', argv), patch.object(p, 'check_plan'), \
                    patch.object(p, 'gated_launch', return_value={'pass':True}) as launch, \
                    patch('builtins.print'):
                p.main()
            self.assertEqual(launch.call_args.kwargs['diagnostic_prefix_count'], 1000)
            self.assertEqual(launch.call_args.kwargs['replay_trace'], Path('/existing/trace.json'))
            self.assertEqual(launch.call_args.args[0], ['/bin/echo', '--config', 'child.json'])

    def test_complete_failed_http_work_is_not_a_broken_measurement_or_a_pass(self):
        ready = dict(event='replay_ready', plan=dict(count=2, view_sha256='frozen'))
        rows = [ready,
                dict(event='request_contract', request_id='a'),
                dict(event='request_contract', request_id='b'),
                dict(event='request_created', request_id='a'),
                dict(event='request_created', request_id='b'),
                dict(event='http_response', request_id='a', response=dict(protocol_valid=True)),
                dict(event='http_request_failed', request_id='b', error='HTTP500'),
                dict(event='http_replay_complete', N_plan=2, N_arrived=2,
                     N_terminal=2, N_response=1, N_failed=1)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'replay.jsonl'
            def write(items):
                path.write_text(''.join(json.dumps(row)+'\n' for row in items))
            write(rows)
            result = p.completed_http_failure(path, ready)
            self.assertTrue(result['measurement_complete'])
            self.assertFalse(result['workload_passed'])
            self.assertEqual(result['N_failed'], 1)
            self.assertEqual(result['journal_sha256'], p.digest(path))
            bad_cases = [rows[:-1], rows+[rows[-1]], rows[:3]+[rows[2]]+rows[3:],
                         rows[:3]+rows[4:],
                         rows[:-1]+[{**rows[-1], 'N_failed':0}],
                         rows[:-1]+[{**rows[-1], 'event':'http_replay_incomplete'}],
                         [{**ready, 'plan':dict(count=2, view_sha256='other')}]+rows[1:],
                         rows[:5]+[{**rows[5], 'response':dict(protocol_valid=False)}]+rows[6:],
                         rows[:6]+[{**rows[6], 'request_id':'a'}]+rows[7:]]
            for i, bad in enumerate(bad_cases):
                with self.subTest(case=i):
                    write(bad)
                    with self.assertRaises(ValueError):
                        p.completed_http_failure(path, ready)
            write(rows)
            path.write_text(path.read_text().rstrip('\n'))
            with self.assertRaises(ValueError):
                p.completed_http_failure(path, ready)

    def test_http_publisher_does_not_inherit_serving_startup_hooks(self):
        parent = dict(PYTHONPATH='/serving', PYTHONHOME='/other-python',
                      USE_TORCH='1', HF_HUB_OFFLINE='0', PATH='/usr/bin')
        env = p.tokenizer_publisher_environment(parent)
        self.assertEqual(parent['PYTHONPATH'], '/serving')
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertEqual(env['PATH'], parent['PATH'])
        for key in ('PYTHONNOUSERSITE', 'PYTHONSAFEPATH', 'HF_HUB_OFFLINE'):
            self.assertEqual(env[key], '1')
        self.assertEqual(env['USE_TORCH'], '0')

    def test_child_options_are_opaque_for_supervisor_and_nested_gate(self):
        child = ['/bin/echo', '--host', '192.168.4.178', '--config', 'child.json']
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'launch.json'
            argv = ['preflight', 'gated-launch', '--output', str(output), '--exec', *child]
            with patch('sys.argv', argv), patch.object(p, 'check_plan'), \
                    patch.object(p, 'gated_launch', return_value={'pass': True}) as launch, \
                    patch('builtins.print'):
                p.main()
            self.assertEqual(launch.call_args.args[0], child)
            self.assertEqual(json.loads(output.read_text()), {'pass': True})
        argv = ['preflight', '_launch-gate', '--gate-socket', '/tmp/example.sock',
                '--gate-nonce', 'example', '--exec', *child]
        with patch('sys.argv', argv), patch.object(p, 'launch_gate_worker') as gate:
            p.main()
        gate.assert_called_once_with('/tmp/example.sock', 'example', child, False)


class IndependentNumericReference(unittest.TestCase):
    def test_closeness_is_not_discrimination_and_nonfinite_or_missing_reject(self):
        native={'1':-1.,'2':-2.}
        same=p.reference_probability_comparison(native,{'1':-1.001,'2':-2.001})
        wrong=p.reference_probability_comparison(native,{'1':-1.01,'2':-2.02})
        self.assertTrue(same['close'])
        self.assertTrue(wrong['close'])  # Both passing cannot identify the correct adapter.
        self.assertFalse(p.reference_probability_comparison(native,{'1':-1.1,'2':-2.})['close'])
        for bad in ({'1':-1.},{'1':float('nan'),'2':-2.}):
            with self.assertRaises(ValueError): p.reference_probability_comparison(native,bad)

    def test_backbone_snapshot_reuses_existing_files_and_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('config.json','tokenizer_config.json','tokenizer.json','tokenizer.model',
                         'special_tokens_map.json','generation_config.json','model-1.safetensors'):
                (root/name).write_bytes(b'existing-test-data')
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'weight':'model-1.safetensors'}}))
            observation=p.snapshot_reference_backbone(root)
            self.assertEqual(len(observation['files']),8)
            p.verify_reference_backbone(observation)
            (root/'model-1.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed during execution'):
                p.verify_reference_backbone(observation)
            index.write_text(json.dumps({'weight_map':{'weight':'../outside.safetensors'}}))
            with self.assertRaises(ValueError): p.snapshot_reference_backbone(root)


class ExistingContentIndex(unittest.TestCase):
    def make(self):
        import os
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)/'pool'
        root.mkdir()
        rows = []
        for aid in ('a', 'b'):
            directory = root/aid
            directory.mkdir()
            for name, data in (('adapter_model.safetensors', b'existing-weight-fixture'),
                               ('adapter_config.json', b'{"r":8}')):
                if aid == 'a':
                    (directory/name).write_bytes(data)
                else:
                    os.link(root/'a'/name, directory/name)
            rows.append(dict(adapter_id=aid, inspected=True, padding=None,
                weight_sha256=p.digest(directory/'adapter_model.safetensors'),
                config_sha256=p.digest(directory/'adapter_config.json'),
                logical_file_bytes=sum(f.stat().st_size for f in directory.iterdir())))
        manifest = root/'.publicmix_generation_manifest.json'
        manifest.write_text(json.dumps({'adapters':[{'id':r['adapter_id']} for r in rows]}))
        audit = Path(temporary.name)/'audit.json'
        audit.write_text(json.dumps(dict(kind='existing_artifact_tensor_audit_v1',audit_complete=True,
            pools=[dict(root=str(root),complete=True,rows=rows,manifest_sha256=p.digest(manifest))])))
        return root,audit

    def test_reuses_tensor_audit_but_hashes_current_files_and_consumes_existing_http_schema(self):
        # This client is stdlib-only. Load its actual module without the package
        # facade, whose legacy remote storage imports NumPy in system Python.
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location('content_index_http_client',
            p.ROOT/'faaslora/storage/http_artifact_store.py')
        module = importlib.util.module_from_spec(spec)
        # Dataclasses with postponed annotations resolve their owning module.
        # Match normal import semantics without importing the package facade.
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
        root,audit = self.make()
        result = p.index_existing_artifact_pool(root,audit,2)
        proof = result['provenance']
        self.assertEqual((proof['file_entries'],proof['unique_hashed_inodes'],
                          proof['exact_file_tree_classes']),(4,2,1))
        self.assertEqual(proof['logical_payload_bytes'],2*proof['unique_hashed_bytes'])
        self.assertFalse(proof['tensor_audit_repeated'])
        self.assertFalse(proof['remote_content_verified'])
        self.assertFalse(proof['serving_qualified'])
        client = module.HttpArtifactStoreClient(endpoint='http://127.0.0.1:1',token='')
        client.configure_content_manifest(result)
        for aid in ('a','b'):
            identity = client.routing_identity(aid,(root/aid/'adapter_config.json').read_bytes())
            self.assertEqual(proof['content_identity_groups'][identity['content_sha256']],['a','b'])
        # Equal bytes in distinct inodes are not guessed to be the same file.
        weight=root/'b'/'adapter_model.safetensors'
        data=weight.read_bytes(); weight.unlink(); weight.write_bytes(data)
        result=p.index_existing_artifact_pool(root,audit,2)
        self.assertEqual(result['provenance']['unique_hashed_inodes'],3)

    def test_changed_audit_or_payload_and_wrong_pool_are_rejected(self):
        root,audit = self.make()
        original=json.loads(audit.read_text())
        for change in ({'audit_complete':False},{'pools':[]}):
            audit.write_text(json.dumps({**original,**change}))
            with self.assertRaises(ValueError): p.index_existing_artifact_pool(root,audit,2)
        audit.write_text(json.dumps(original))
        with self.assertRaises(ValueError): p.index_existing_artifact_pool(root,audit,3)
        (root/'b'/'adapter_model.safetensors').write_bytes(b'changed-weight-fixture!')
        with self.assertRaisesRegex(ValueError,'payload differs'):
            p.index_existing_artifact_pool(root,audit,2)

    def test_symlink_and_new_directory_do_not_enter_verified_index(self):
        root,audit = self.make()
        path=root/'b'/'adapter_model.safetensors'
        path.unlink(); path.symlink_to(root/'a'/'adapter_model.safetensors')
        with self.assertRaisesRegex(ValueError,'payload differs'):
            p.index_existing_artifact_pool(root,audit,2)
        (root/'extra').mkdir()
        with self.assertRaisesRegex(ValueError,'directory IDs'):
            p.index_existing_artifact_pool(root,audit,2)

    def test_outside_support_links_match_actual_existing_server_payload(self):
        import io,tarfile,threading,urllib.request
        from remote_artifact_node.server import ArtifactHandler,ArtifactServer
        root,audit = self.make()
        support=root.parent/'support.json'
        support.write_bytes(b'local-backbone-metadata')
        data=json.loads(audit.read_text())
        for row in data['pools'][0]['rows']:
            (root/row['adapter_id']/'tokenizer.json').symlink_to(support)
            row['logical_file_bytes']+=support.stat().st_size
        audit.write_text(json.dumps(data))
        result=p.index_existing_artifact_pool(root,audit,2)
        self.assertEqual(len(result['provenance']['excluded_links']),2)
        self.assertEqual(result['provenance']['excluded_readable_bytes'],2*support.stat().st_size)
        server=ArtifactServer(('127.0.0.1',0),ArtifactHandler,root=root)
        thread=threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for artifact in result['artifacts']:
                with opener.open(f'http://127.0.0.1:{server.server_port}/artifacts/{artifact["id"]}.tar.gz',timeout=2) as response:
                    payload=response.read()
                with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
                    actual={m.name:(m.size,p.hashlib.sha256(archive.extractfile(m).read()).hexdigest())
                            for m in archive.getmembers() if m.isfile()}
                self.assertEqual(actual,{f['path']:(f['size_bytes'],f['sha256']) for f in artifact['files']})
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
        materialized=p.index_existing_artifact_pool(root,audit,2,
            materialized_support_roots=(support.parent,))
        self.assertEqual(len(materialized['provenance']['materialized_links']),2)
        self.assertFalse(materialized['provenance']['excluded_links'])
        self.assertEqual(materialized['provenance']['file_entries'],6)
        self.assertEqual(materialized['provenance']['unique_hashed_inodes'],3)
        self.assertEqual(materialized['provenance']['logical_payload_bytes'],
                         materialized['provenance']['source_logical_readable_bytes'])
        for artifact in materialized['artifacts']:
            entry=next(f for f in artifact['files'] if f['path']=='tokenizer.json')
            self.assertEqual(entry['sha256'],p.digest(support))
        self.assertTrue((root/'a'/'tokenizer.json').is_symlink())  # no pool rewrite
        with self.assertRaisesRegex(ValueError,'unapproved'):
            p.index_existing_artifact_pool(root,audit,2,materialized_support_roots=(root,))
        internal=root/'a'/'internal'
        internal.symlink_to(root/'a'/'adapter_config.json')
        with self.assertRaisesRegex(ValueError,'internal symlink'):
            p.index_existing_artifact_pool(root,audit,2)

    def test_shared_support_mutation_between_adapters_is_rejected(self):
        root,audit = self.make()
        support=root.parent/'support.json'
        support.write_bytes(b'original')
        data=json.loads(audit.read_text())
        for row in data['pools'][0]['rows']:
            (root/row['adapter_id']/'tokenizer.json').symlink_to(support)
            row['logical_file_bytes']+=support.stat().st_size
        trigger=root/'b'/'0-trigger'
        trigger.write_bytes(b'trigger')
        data['pools'][0]['rows'][1]['logical_file_bytes']+=trigger.stat().st_size
        audit.write_text(json.dumps(data))
        original=p.digest
        def mutate(path):
            if path==trigger:
                support.write_bytes(b'modified')
            return original(path)
        with patch.object(p,'digest',side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError,'shared support target changed'):
                p.index_existing_artifact_pool(root,audit,2,
                    materialized_support_roots=(support.parent,))

    def test_mutation_after_hash_and_existing_output_are_rejected(self):
        root,audit = self.make()
        original=p.digest
        def change(path):
            value=original(path)
            if path.name=='adapter_model.safetensors':
                path.write_bytes(b'changed-during-hash')
            return value
        with patch.object(p,'digest',side_effect=change), self.assertRaisesRegex(RuntimeError,'changed'):
            p.index_existing_artifact_pool(root,audit,2)
        with patch('sys.argv',['ieee_tc_preflight.py','artifact-index','--output',str(audit)]), \
                self.assertRaises(SystemExit) as stopped:
            p.main()
        self.assertEqual(stopped.exception.code,2)


class ProtocolGates(unittest.TestCase):
    def test_workspace_bounds_reuse_complete_matching_classes_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_path, observation_path = Path(directory)/'audit.json', Path(directory)/'observation.json'
            audit = dict(kind='existing_artifact_tensor_audit_v1', audit_complete=True,
                pools=[dict(root='/existing/pool', complete=True, inspected_adapters=1, expected_adapters=1,
                    rows=[dict(adapter_id='a', weight_bytes=120, weight_sha256='a'*64, configured_rank=8,
                               target_modules=['q_proj'], inspected=True, all_finite=True)])])
            audit_path.write_text(json.dumps(audit))
            observation = dict(kind='native_host_allocator_observation_v1', stage='complete',
                **{'pass':True}, artifact_audit_sha256=p.digest(audit_path),
                torch_version='2.13.0+cu130', backend_version='0.30.0',
                allocator_settings={'max_cached_size':0}, controls=p.select_host_allocator_controls(audit),
                cases=[dict(pool_root='/existing/pool', adapter_id='a', weight_sha256='a'*64, rank=8,
                    contract=dict(kind='dense_safetensors_native_host_loading_v1', dtype='torch.float16',
                                  source_file_bytes=120, converted_pageable_bytes=100, tensor_count=2))])
            observation_path.write_text(json.dumps(observation))
            result = p.derive_host_workspace_contracts(observation_path, audit_path)
            contract = result['pools'][0]['workspace_contract']
            self.assertEqual(contract['max_resident_pinned_bytes'], 100)
            self.assertEqual(contract['max_transient_tensor_bytes'], 220)
            self.assertEqual(contract['source_audit_sha256'], p.digest(audit_path))
            self.assertFalse(result['new_execution'])
            for change in ({'cases':[]}, {'allocator_settings':{'max_cached_size':False}},
                           {'artifact_audit_sha256':'b'*64}, {'controls':[]}):
                observation_path.write_text(json.dumps({**observation,**change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    p.derive_host_workspace_contracts(observation_path, audit_path)

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
            with self.assertRaisesRegex(RuntimeError,'no guard'):
                p.backend_host_allocator_check(Path('/missing'),Path('/missing'), copy_lifecycle=True)
            with self.assertRaisesRegex(RuntimeError,'no guard'):
                p.backend_host_allocator_check(Path('/missing'),Path('/missing'),
                                              copy_lifecycle=True, copy_background=True)

    def test_background_copy_requires_explicit_diagnostic_and_native_readback(self):
        config = 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True'
        settings = dict(max_cached_size=0, PYTORCH_CUDA_ALLOC_CONF=config)
        environment = dict(PYTORCH_ALLOC_CONF=config)
        result = p._host_copy_background_policy(settings, environment, '2.13.0+cu130')
        self.assertTrue(result['verified'])
        self.assertFalse(result['production_launch_authorized'])
        self.assertFalse(result['immediate_release_guaranteed'])
        self.assertEqual(result['background_readback'], 'parsed_configuration_string')
        for invalid in (None, {}, {**settings,'max_cached_size':False},
                        {**settings,'max_cached_size':1},
                        {**settings,'PYTORCH_CUDA_ALLOC_CONF':'pinned_max_cached_size_mb:0'}):
            with self.subTest(settings=invalid), self.assertRaises(RuntimeError):
                p._host_copy_background_policy(invalid, environment, '2.13.0+cu130')
        for extra in ({'PYTORCH_ALLOC_CONF':'pinned_max_cached_size_mb:0'},
                      {'PYTORCH_CUDA_ALLOC_CONF':config}, {'PYTORCH_HIP_ALLOC_CONF':config},
                      {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY':'uncached_v1'}):
            with self.subTest(environment=extra), self.assertRaises(RuntimeError):
                p._host_copy_background_policy(settings, {**environment,**extra}, '2.13.0+cu130')
        with self.assertRaises(RuntimeError):
            p._host_copy_background_policy(settings, environment, '2.12.0')
        with patch.object(p, 'verify_current_service', return_value={}):
            with self.assertRaisesRegex(ValueError, 'requires the copy'):
                p.backend_host_allocator_check(Path('/missing'), Path('/missing'), copy_background=True)

    def test_background_copy_flag_rejects_other_experiments(self):
        for args in (['preflight'], ['backend-host-check'], ['backend-copy-check']):
            with self.subTest(args=args), patch('sys.argv', ['ieee_tc_preflight.py', *args, '--host-copy-background']), \
                 patch.object(p, 'check_plan', side_effect=AssertionError('invalid flag reached execution')):
                with self.assertRaises(SystemExit) as raised:
                    p.main()
                self.assertEqual(raised.exception.code, 2)

    def test_host_copy_lifetime_flag_cannot_silently_apply_to_another_action(self):
        with patch('sys.argv', ['ieee_tc_preflight.py', 'preflight', '--host-copy-lifecycle']), \
             patch.object(p, 'check_plan', side_effect=AssertionError('invalid flag reached execution')):
            with self.assertRaises(SystemExit) as raised:
                p.main()
            self.assertEqual(raised.exception.code, 2)

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

    def test_artifact_disk_rule_sums_concurrent_growth_without_inference_floor(self):
        rows = [dict(max_concurrent=4, remaining_archive_bytes=5),
                dict(max_concurrent=2, remaining_archive_bytes=7)]
        result = p.artifact_disk_required(concurrent_packs=rows,
            log_growth_bytes=3, safety_reserve_bytes=100)
        self.assertEqual(result['packing_peak_bytes'], 34)
        self.assertEqual(result['required_bytes'], 156)  # ceil(1.5 * 37) + 100
        self.assertFalse(result['production_launch_authorized'])
        self.assertEqual(p.disk_required(0), 150*p.GIB)
        self.assertEqual(p.POLICY['disk_stop_bytes'], 100*p.GIB)
        self.assertEqual(p.memory_required(), 102*p.GIB)

    def test_artifact_disk_rule_rejects_unknown_or_invalid_bounds(self):
        args = dict(concurrent_packs=[dict(max_concurrent=2, remaining_archive_bytes=7)],
                    log_growth_bytes=0, safety_reserve_bytes=100)
        bad = [dict(concurrent_packs=[]), dict(concurrent_packs=None),
               dict(concurrent_packs=[dict(max_concurrent=0, remaining_archive_bytes=7)]),
               dict(concurrent_packs=[dict(max_concurrent=True, remaining_archive_bytes=7)]),
               dict(concurrent_packs=[dict(max_concurrent=2, remaining_archive_bytes=-1)]),
               dict(log_growth_bytes=-1), dict(log_growth_bytes=1.5),
               dict(safety_reserve_bytes=0), dict(safety_reserve_bytes=None)]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(ValueError):
                p.artifact_disk_required(**{**args, **override})

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
        with patch.object(p, 'gpu_process_identity', return_value=dict(proc,uid=p.os.getuid())), \
             patch.object(p, 'scope_still_owned', return_value=True):
            self.assertEqual(p.verify_watchdog_attachment(event, identity, auxiliary), proc)
            with self.assertRaisesRegex(RuntimeError, 'another resource domain'):
                p.verify_watchdog_attachment(event, dict(identity, invocation_id='two'), auxiliary)
        with patch.object(p, 'gpu_process_identity', return_value=dict(proc, start_ticks=6,uid=p.os.getuid())):
            with self.assertRaisesRegex(RuntimeError, 'birth identity'):
                p.verify_watchdog_attachment(event, identity, auxiliary)

    def test_watcher_check_is_independent_of_other_auxiliary_process_churn(self):
        identity = {'unit':'test', 'path':'/service', 'invocation_id':'one', 'inode':1}
        proc = {'pid':123, 'start_ticks':5, 'affinity':p.POLICY['aux_cpus'], 'cgroup':'/aux'}
        event = {'event':'watchdog_ready', 'service_identity':identity, 'watchdog_pid':123,
                 'watchdog_process':proc, 'aux':{'path':'/aux'}}
        with patch.object(p, 'owned_pids', side_effect=RuntimeError('unrelated helper moved')) as scan, \
             patch.object(p, 'gpu_process_identity', return_value=dict(proc,uid=p.os.getuid())) as observe, \
             patch.object(p, 'scope_still_owned', return_value=True):
            self.assertEqual(p.verify_watchdog_attachment(event,identity,Path('/aux')),proc)
            observe.assert_called_once_with(123)
            scan.assert_not_called()

    def test_target_watcher_missing_foreign_owner_domain_or_affinity_still_rejects(self):
        identity = {'unit':'test', 'path':'/service', 'invocation_id':'one', 'inode':1}
        proc = {'pid':123, 'start_ticks':5, 'affinity':p.POLICY['aux_cpus'], 'cgroup':'/aux'}
        event = {'event':'watchdog_ready', 'service_identity':identity, 'watchdog_pid':123,
                 'watchdog_process':proc, 'aux':{'path':'/aux'}}
        observed=dict(proc,uid=p.os.getuid())
        for value in (None,dict(observed,uid=p.os.getuid()+1),dict(observed,cgroup='/other'),
                      dict(observed,start_ticks=6),dict(observed,affinity=[0])):
            with self.subTest(value=value), \
                 patch.object(p,'gpu_process_identity',return_value=value), \
                 patch.object(p,'scope_still_owned',return_value=True):
                with self.assertRaises(RuntimeError):p.verify_watchdog_attachment(event,identity,Path('/aux'))
        with patch.object(p,'gpu_process_identity',return_value=observed), \
             patch.object(p,'scope_still_owned',return_value=False):
            with self.assertRaises(RuntimeError):p.verify_watchdog_attachment(event,identity,Path('/aux'))

    def test_direct_watcher_read_matches_actual_current_process_without_cuda(self):
        identity=p.watchdog_process_identity(p.os.getpid(),p.cg_path())
        self.assertEqual(identity['pid'],p.os.getpid())
        self.assertEqual(identity['cgroup'],str(p.cg_path()))
        self.assertEqual(identity['affinity'],sorted(p.os.sched_getaffinity(0)))

    def test_missing_launch_receipt_is_not_authorization(self):
        with patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'missing guarded'):
                p.verify_current_service()


class MeasuredAdmissionInitializer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(name='existing-model', backend='vllm', device_id=0,
                        visible_device_ids=[0], ieee_input_upper_bounds=[759],
                        generation_contract='fixed_length_greedy_v1')
        samples = [dict(source_request_id=f'r{i}', native_prompt_tokens=prompt,
                        completed_output_tokens=count, prompt_sha256=str(i)*64,
                        content_prompt_tokens=prompt-1)
                   for i, (prompt, count) in enumerate(((100, 10), (759, 30), (760, 40)))]
        # One warmup, three original requests and one legitimate repeated source.
        rows = []
        for wave, selected in enumerate(([samples[2]], samples, [samples[0]])):
            for lane, s in enumerate(selected):
                rows.append(dict(request_id=f'source-profile/w{wave}/l{lane}',
                    source_request_id=s['source_request_id'], **{'pass': True},
                    actual_tokens=s['completed_output_tokens'], target_tokens=s['completed_output_tokens'],
                    input_content_tokens=s['content_prompt_tokens'], timing=dict(
                        native_terminal_observed=True, native_output_tokens=float(s['completed_output_tokens']),
                        actual_prompt_tokens=s['native_prompt_tokens'],
                        native_prompt_token_ids_sha256=s['prompt_sha256'])))
        self.run = dict(kind='backend_native_native_source_matrix_qualification_v1',
            **{'pass': True}, stage='complete', model_config=copy.deepcopy(self.cfg),
            runtime_receipt_sha256='runtime', trace=dict(source_sha256='trace'), requests=rows,
            source_profile_spec=dict(waves=[dict(role=role, requests=[dict(source_request_id=s['source_request_id']) for s in selected])
                for role, selected in [('kernel_warmup_retained', [samples[2]]),
                                       ('representative_measurement', samples),
                                       ('representative_measurement', [samples[0]])]]))
        self.entry = dict(model_config=copy.deepcopy(self.cfg), same_backend='0.30.0',
            input_upper_bounds=[759], samples=samples, unique_original_requests=3,
            unobserved_bucket_ids=[], observed_buckets=[
                dict(bucket=0, unique_requests=2, completed_output_mean=20.0),
                dict(bucket=1, unique_requests=1, completed_output_mean=40.0)])
        self.audit = dict(kind='native_completed_output_length_initialization_audit_v1',
            development_only=True, production_profile_frozen=False, models={'3b': self.entry})
        self.binding = dict(kind='native_completed_length_binding_v1', model='3b', window_s=5)

    def bind(self):
        raw = self.root/'raw.json'
        raw.write_text(json.dumps(self.run))
        self.entry['source_run'] = dict(path=str(raw), sha256=p.digest(raw))
        audit = self.root/'audit.json'
        audit.write_text(json.dumps(self.audit))
        self.binding['audit'] = dict(path=str(audit), sha256=p.digest(audit))

    def derive(self, **overrides):
        kwargs = dict(model_config=self.cfg, backend_version='0.30.0',
                      runtime_receipt_sha256='runtime', source_trace_sha256='trace',
                      movement_concurrency=3)
        kwargs.update(overrides)
        return p.measured_admission_initializer(self.binding, **kwargs)

    def test_raw_native_counts_and_boundary_deduplicated_without_mutation(self):
        self.bind()
        cfg_before = copy.deepcopy(self.cfg)
        profile, evidence = self.derive()
        self.assertEqual(profile['profile_means'], [20.0, 40.0])
        self.assertEqual(profile['transfer_limit'], 3)
        self.assertEqual(profile['window_s'], 5.0)
        self.assertEqual(evidence['unique_original_requests'], 3)
        self.assertFalse(evidence['performance_samples_relabelled'])
        self.assertFalse(evidence['production_profile_frozen'])
        self.assertEqual(self.cfg, cfg_before)
        self.assertEqual(self.derive(), (profile, evidence))
        placed = dict(self.cfg, device_id=2, visible_device_ids=[2])
        self.assertEqual(self.derive(model_config=placed), (profile, evidence))
        self.assertNotEqual(self.derive(movement_concurrency=4)[0]['profile_id'], profile['profile_id'])

    def test_wrong_identity_sha_or_existing_profile_rejected(self):
        self.bind()
        for kwargs in (dict(backend_version='different'), dict(runtime_receipt_sha256='other'),
                       dict(source_trace_sha256='other'), dict(movement_concurrency=0),
                       dict(movement_concurrency=True), dict(model_config=dict(self.cfg, max_loras=99)),
                       dict(model_config=dict(self.cfg, ieee_admission_profile={})),
                       dict(model_config=dict(self.cfg, ieee_input_upper_bounds=[760]))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.derive(**kwargs)
        self.binding['audit']['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.derive()
        self.bind()
        (self.root/'raw.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.derive()

    def test_only_existing_facade_capacity_default_can_be_resolved_for_length_reuse(self):
        self.bind()
        profile, evidence = self.derive()
        self.assertEqual(self.derive(model_config=dict(self.cfg, max_cpu_loras=24)), (profile, evidence))
        self.assertEqual(evidence['base_model_config']['max_cpu_loras'], 24)
        self.assertNotIn('max_cpu_loras', evidence['source_recorded_model_config'])
        self.assertFalse(evidence['performance_samples_relabelled'])
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.derive(model_config=dict(self.cfg, max_cpu_loras=32))

    def test_bad_raw_completion_coverage_or_curated_means_rejected(self):
        original_run, original_entry = copy.deepcopy(self.run), copy.deepcopy(self.entry)
        mutations = [
            lambda: self.run.update(stage='incomplete'),
            lambda: self.run.update(**{'pass': False}),
            lambda: self.run['requests'].pop(),
            lambda: self.run['requests'].append(copy.deepcopy(self.run['requests'][1])),
            lambda: self.run['requests'][1].update(target_tokens=11),
            lambda: self.run['requests'][1]['timing'].update(native_terminal_observed=False),
            lambda: self.run['requests'][1]['timing'].update(native_output_tokens=10.5),
            lambda: self.run['requests'][1]['timing'].update(native_output_tokens=True),
            lambda: self.run['requests'][-1]['timing'].update(actual_prompt_tokens=101),
            lambda: self.run['requests'][1]['timing'].pop('native_output_tokens'),
            lambda: self.run['requests'][1]['timing'].update(native_prompt_token_ids_sha256='missing'),
            lambda: self.entry['samples'].append(copy.deepcopy(self.entry['samples'][0])),
            lambda: self.entry['observed_buckets'][0].update(completed_output_mean=21),
            lambda: self.entry['unobserved_bucket_ids'].append(1),
            lambda: self.entry.update(input_upper_bounds=[760]),
        ]
        for number, mutate in enumerate(mutations):
            self.run, self.entry = copy.deepcopy(original_run), copy.deepcopy(original_entry)
            self.audit['models']['3b'] = self.entry
            mutate()
            self.bind()
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self.derive()

    def test_explicit_window_and_development_scope_required(self):
        self.bind()
        for window in (0, -1, float('nan'), float('inf'), True, 'auto'):
            self.binding['window_s'] = window
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.derive()
        self.binding['window_s'] = 5
        self.audit['production_profile_frozen'] = True
        self.bind()
        with self.assertRaisesRegex(ValueError, 'development'):
            self.derive()


class SlotContentQualification(unittest.TestCase):
    def engine(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        return SimpleNamespace(
            ieee_scheduler_observation=AsyncMock(return_value={
                'admitted': [], 'unretired_iterations': [], 'native_deferred_free_batches': []}),
            ieee_worker_observation=AsyncMock(return_value={'workers': [{
                'device_barrier_used': True,
                'slot_content_audit': {'kind': 'native_registered_to_gpu_slot_content_v1',
                    'exact_content_pass': True, 'adapters': [{'adapter_int_id': 7,
                        'exact_content_pass': True, 'tensor_count': 2}]}}]}))

    def test_isolated_snapshot_is_not_performance_or_full_semantic_qualification(self):
        import asyncio
        engine = self.engine()
        result = asyncio.run(p.qualify_slot_content_snapshot(engine, 7))
        self.assertTrue(result['pass'])
        self.assertFalse(result['performance_sample'])
        self.assertFalse(result['semantic_full_pool_qualification'])
        engine.ieee_worker_observation.assert_awaited_once_with(synchronize=True, audit_adapter_ids=[7])

    def test_busy_native_scheduler_is_rejected_before_barrier(self):
        import asyncio
        for field in ('admitted', 'unretired_iterations', 'native_deferred_free_batches'):
            engine = self.engine()
            engine.ieee_scheduler_observation.return_value[field] = ['live']
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'drained'):
                asyncio.run(p.qualify_slot_content_snapshot(engine, 7))
            engine.ieee_worker_observation.assert_not_called()

    def test_mismatch_absent_empty_and_wrong_identity_never_pass(self):
        import asyncio
        mutations = (
            lambda worker: worker.pop('slot_content_audit'),
            lambda worker: worker.update(device_barrier_used=False),
            lambda worker: worker['slot_content_audit'].update(exact_content_pass=False),
            lambda worker: worker['slot_content_audit']['adapters'][0].update(adapter_int_id=8),
            lambda worker: worker['slot_content_audit']['adapters'][0].update(exact_content_pass=False),
            lambda worker: worker['slot_content_audit']['adapters'][0].update(tensor_count=0),
        )
        for number, mutate in enumerate(mutations):
            engine = self.engine()
            mutate(engine.ieee_worker_observation.return_value['workers'][0])
            with self.subTest(number=number):
                self.assertFalse(asyncio.run(p.qualify_slot_content_snapshot(engine, 7))['pass'])

    def test_multiple_or_missing_workers_are_not_tp1_evidence(self):
        import asyncio
        for count in (0, 2):
            engine = self.engine()
            workers = engine.ieee_worker_observation.return_value['workers']
            engine.ieee_worker_observation.return_value['workers'] = workers * count
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, 'one actual'):
                asyncio.run(p.qualify_slot_content_snapshot(engine, 7))


if __name__ == '__main__':
    unittest.main()
