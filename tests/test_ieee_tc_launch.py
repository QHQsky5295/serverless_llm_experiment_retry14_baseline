"""TC launcher integration, with no model/driver operation in these unit tests."""
import asyncio
import os
import time
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import run_all_experiments as runner
from faaslora.clock import local_monotonic_clock_id
from faaslora.datasets.workload_generator import FrozenReplayPlan, publish_frozen_replay


def ieee_control_fixture():
    # Deliberately synthetic: correctness constants, not a frozen serving profile.
    return dict(queue_upper=2, queue_lower=1, active_upper=.75, active_lower=.25,
        ttft_upper_ms=1000, ttft_lower_ms=500, ttft_window_s=100, scale_down_cooldown_s=3)


class IEEEOnlineObservationBinding(unittest.TestCase):
    def setUp(self):
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.preloading.preloading_manager import OwnedMovementQueue
        self.model = dict(ieee_admission_profile=dict(window_s=5.0, transfer_limit=3,
                         profile_id='measured-fixture', model_backend_id='fixture-model'))
        self.coord = dict(online_hotness_window_s=5.0, max_concurrent_loads=3)
        self.stack = SimpleNamespace(hotness_tracker=HotnessTracker(None, window_seconds=5),
            preloading_manager=SimpleNamespace(ieee_movements=OwnedMovementQueue(3)))

    def check(self, **changes):
        values = dict(model_cfg=self.model, coord_cfg=self.coord, preload_cfg={},
                      stack=self.stack, runner_window_s=5.0)
        values.update(changes)
        return runner._validate_ieee_online_observation_binding(**values)

    def test_actual_owners_match_and_no_runtime_state_is_created(self):
        result = self.check(preload_cfg=dict(online_hotness_window_s=5))
        self.assertEqual(result['window_s'], 5.0)
        self.assertEqual(result['transfer_limit'], 3)
        self.assertEqual(result['demand_windows']['actual_demand'], 5)
        self.assertEqual(result['movement_limits']['actual_movement'], 3)
        self.assertFalse(self.stack.preloading_manager.ieee_movements.bound)
        self.assertEqual(self.stack.hotness_tracker.snapshot().total_arrivals, 0)

    def test_explicit_window_required_instead_of_legacy_default_or_precedence(self):
        for value in (None, 'auto', True, 0, 2, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'same explicit W'):
                self.check(coord_cfg=dict(self.coord, online_hotness_window_s=value))
        with self.assertRaisesRegex(ValueError, 'same explicit W'):
            self.check(preload_cfg=dict(online_hotness_window_s=10))
        with self.assertRaisesRegex(ValueError, 'same explicit W'):
            self.check(runner_window_s=10)
        self.stack.hotness_tracker.window_seconds = 10
        with self.assertRaisesRegex(ValueError, 'same explicit W'):
            self.check()

    def test_observation_limit_cannot_silently_differ_from_movement_capacity(self):
        for value in (None, 2, 5, True, 3.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'actual shared movement'):
                self.check(coord_cfg=dict(self.coord, max_concurrent_loads=value))
        self.stack.preloading_manager.ieee_movements.max_concurrent = 5
        with self.assertRaisesRegex(ValueError, 'actual shared movement'):
            self.check()

    def test_missing_or_invalid_admission_owner_is_not_zero_pressure(self):
        with self.assertRaisesRegex(ValueError, 'native admission'):
            self.check(model_cfg={})
        for window in (None, 0, True, float('nan')):
            model = dict(ieee_admission_profile={**self.model['ieee_admission_profile'], 'window_s':window})
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, 'positive W'):
                self.check(model_cfg=model)
        with self.assertRaisesRegex(ValueError, 'same explicit W'):
            self.check(stack=None)

    def test_actual_runner_records_binding_without_changing_the_guard(self):
        service = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        service._ieee_online_observation_binding = self.check()
        service._retired_coord_metrics = []
        service.instance_pool = service.coordinator = None
        result = service._current_coord_metrics()
        self.assertEqual(result['ieee_online_observation_binding'], self.check())
        result['ieee_online_observation_binding']['window_s'] = 99
        self.assertEqual(service._ieee_online_observation_binding['window_s'], 5)
        with self.assertRaisesRegex(RuntimeError, 'not qualified'):
            service._require_ieee_full_qualification()


class ScenarioRuntimeConfiguration(unittest.TestCase):
    def test_main_and_source_common_generation_capacity_boundary_is_pure(self):
        source = dict(backend='vllm', max_num_seqs=4, runtime_concurrency_cap=8,
                      max_input_len=1024, max_output_tokens_cap=1024)
        workload = dict(generation_contract='fixed_length_greedy_v1',
                        fixed_prompt_max_tokens=759, fixed_output_max_tokens=256)
        direct = runner._prepare_scenario_runtime_model_config(source, workload, {})
        twice = runner._prepare_scenario_runtime_model_config(
            runner._normalize_runtime_concurrency_cap(source), workload, {})
        self.assertEqual(direct, twice)
        self.assertEqual(direct['runtime_concurrency_cap'], 4)
        self.assertEqual(direct['requested_runtime_concurrency_cap'], 8)
        self.assertEqual((direct['max_input_len'], direct['max_output_tokens_cap']), (759,256))
        self.assertEqual(source['max_input_len'], 1024)
        self.assertNotIn('requested_runtime_concurrency_cap', source)

    def test_explicit_scenario_override_is_a_new_capacity_request(self):
        source = runner._normalize_runtime_concurrency_cap(dict(
            backend='vllm', max_num_seqs=4, runtime_concurrency_cap=8))
        coord = dict(instance_model_overrides=dict(runtime_concurrency_cap=2, max_input_len=128))
        cfg = runner._prepare_scenario_runtime_model_config(source,
            dict(generation_contract='fixed_length_greedy_v1'), coord)
        self.assertEqual((cfg['requested_runtime_concurrency_cap'], cfg['runtime_concurrency_cap']), (2,2))
        self.assertEqual(cfg['max_input_len'],128)
        self.assertEqual(source['requested_runtime_concurrency_cap'],8)
        self.assertEqual(coord['instance_model_overrides']['runtime_concurrency_cap'],2)

    def test_legacy_generation_does_not_gain_fixed_length_caps(self):
        cfg = runner._prepare_scenario_runtime_model_config(dict(max_input_len=100), {}, {})
        self.assertEqual(cfg['generation_contract'], 'legacy')
        self.assertEqual(cfg['max_input_len'], 100)
        self.assertNotIn('max_output_tokens_cap',cfg)

    def test_source_assembly_derives_prior_before_full_fields_and_matches_main(self):
        from scripts import ieee_tc_preflight as p
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp)/'index.json'
            index.write_text('{}')
            spec = dict(content_index=dict(path=str(index),sha256=p.digest(index)),
                admission_initialization={'fixture':True}, movement_concurrency=3)
            cfg = dict(backend='vllm', generation_contract='fixed_length_greedy_v1',
                max_input_len=759, max_output_tokens_cap=256, max_num_seqs=4, runtime_concurrency_cap=8)
            admission = dict(window_s=5., transfer_limit=3, profile_means=[20.,40.],
                model_backend_id='fixture-model',profile_id='fixture-profile')
            with patch.object(p,'measured_admission_initializer', return_value=(admission, {'fixture':True})) as derive:
                assembled, _, evidence = p.prepare_admission_source_runtime(cfg, spec,
                    backend_version='0.30.0',runtime_receipt_sha256='runtime',source_trace_sha256='trace')
            prior_cfg=derive.call_args.kwargs['model_config']
            self.assertNotIn('artifact_content_manifest_path',prior_cfg)
            self.assertNotIn('requested_runtime_concurrency_cap',prior_cfg)
            self.assertNotIn('ieee_admission_profile',prior_cfg)
            main = runner._prepare_scenario_runtime_model_config(dict(cfg,
                ieee_admission_profile=admission,artifact_content_manifest_path=str(index)), evidence['workload'], {})
            self.assertEqual(main,assembled)
            child,_=runner._prepare_dedicated_subprocess_model_cfg(main,device_id=0,runtime_gpu_ids=[0])
            self.assertEqual(child,evidence['planned_child_model_config'])
            self.assertFalse(evidence['performance_samples_relabelled'])
            self.assertNotIn('ieee_admission_profile',cfg)
            contract_path=Path(tmp)/'contract.json'
            contract=dict(kind='ieee_development_control_runtime_contract_v1',model_config=child,
                runtime_receipt_sha256='runtime',coordination=dict(online_hotness_window_s=5.,max_concurrent_loads=3))
            contract_path.write_text(json.dumps(contract))
            spec['full_development_contract']=dict(path=str(contract_path),sha256=p.digest(contract_path))
            with patch.object(p,'measured_admission_initializer',return_value=(admission,{})):
                _,_,bound=p.prepare_admission_source_runtime(cfg,spec,backend_version='0.30.0',
                    runtime_receipt_sha256='runtime',source_trace_sha256='trace')
                self.assertFalse(bound['control_settings_applied_to_source_collector'])
                self.assertEqual(bound['full_development_contract'],spec['full_development_contract'])
                contract['model_config']['runtime_concurrency_cap']=2
                contract_path.write_text(json.dumps(contract))
                spec['full_development_contract']['sha256']=p.digest(contract_path)
                with self.assertRaisesRegex(ValueError,'binding differs'):
                    p.prepare_admission_source_runtime(cfg,spec,backend_version='0.30.0',
                        runtime_receipt_sha256='runtime',source_trace_sha256='trace')
            spec['content_index']['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'SHA256'):
                p.prepare_admission_source_runtime(cfg,spec,backend_version='0.30.0',
                    runtime_receipt_sha256='runtime',source_trace_sha256='trace')


class SourceProfileDomainCoverage(unittest.TestCase):
    def fixture(self):
        from dataclasses import asdict
        from faaslora.experiment.instance_pool import ServiceClassBins
        identities = {a: dict(adapter_id=a, rank=8, content_sha256='a'*64) for a in ('a','b')}
        edges = dict(prompt_tokens=[10], declared_output_tokens=[2], adapter_rank=[8],
                     footprint_bytes=[], admitted_requests=[1,2])
        bins = ServiceClassBins(**{k:tuple(v) for k,v in edges.items()})
        payload = dict(kind='backend_native_native_source_matrix_qualification_v1',
            stage='complete', shutdown_called=True, profile_workspaces_removed=True,
            artifact_mode='prepublished_gzip_v1_real_remote_no_fallback',
            model_config=dict(generation_contract='fixed_length_greedy_v1',
                runtime_concurrency_cap=2,max_input_len=10,max_output_tokens_cap=2),
            source_profile_spec=dict(bins=edges,waves=[]),profile_waves=[],requests=[],**{'pass':True})
        for i,s in enumerate(('remote','nvme','file_host','native_host','gpu')):
            tier = 'host' if s.endswith('host') else s
            native = s in ('native_host','gpu')
            rep = {'remote':'compressed','nvme':'file','file_host':'file','native_host':'tensor','gpu':'slot'}[s]
            wave = dict(source=s,role='representative_measurement',requests=[])
            payload['source_profile_spec']['waves'].append(wave)
            payload['profile_waves'].append(dict(complete=True))
            for lane,aid in enumerate(identities):
                selected = dict(adapter_id=aid,source_request_id='original-'+aid)
                wave['requests'].append(selected)
                features = dict(tier=tier,representation=rep,footprint_bytes=32,adapter_rank=8,
                    prompt_tokens=10,declared_output_tokens=2,admitted_after_accept=lane+1)
                key = asdict(bins.classify(**features))
                source = dict(tier=tier,representation=rep,footprint_bytes=32,native=native)
                source['expected_content_sha256' if native else 'content_sha256']='a'*64
                payload['requests'].append(dict(selected,request_id=f'source-profile/w{i}/l{lane}',
                    requested_source=s,reservation_released=True,actual_tokens=2,target_tokens=2,
                    class_features=features,service_class=key,
                    source_evidence=dict(source_admission=dict(source=source,service_class=key,
                        admitted_after_accept=lane+1)),**{'pass':True}))
        return payload,identities

    def test_observed_full_class_domain_is_not_production_qualification(self):
        from scripts.ieee_tc_preflight import source_profile_class_coverage
        result=source_profile_class_coverage(*self.fixture())
        self.assertTrue(result['class_domain_covered'])
        self.assertEqual(result['measured_service_classes'],10)
        self.assertEqual(result['measured_preparation_classes'],4)
        self.assertFalse(result['full_qualified'])
        self.assertFalse(result['timing_samples_exported'])

    def test_unmeasured_static_content_and_warmup_only_source_are_missing(self):
        from scripts.ieee_tc_preflight import source_profile_class_coverage
        payload,identities=self.fixture()
        identities['c']=dict(adapter_id='c',rank=16,content_sha256='c'*64)
        payload['source_profile_spec']['waves'][-1]['role']='kernel_warmup_retained'
        result=source_profile_class_coverage(payload,identities)
        self.assertFalse(result['class_domain_covered'])
        self.assertEqual(len(result['missing_content_sources']),6)

    def test_tail_prompt_and_post_accept_count_are_not_extrapolated(self):
        from scripts.ieee_tc_preflight import source_profile_class_coverage
        payload,identities=self.fixture()
        payload['model_config'].update(max_input_len=11,runtime_concurrency_cap=3)
        result=source_profile_class_coverage(payload,identities)
        self.assertFalse(result['class_domain_covered'])
        self.assertEqual(result['required_service_classes'],30)
        self.assertEqual(len(result['missing_service_classes']),20)

    def test_incomplete_or_inconsistent_evidence_is_rejected(self):
        from scripts.ieee_tc_preflight import source_profile_class_coverage
        for change in ('missing','rank','source','count'):
            payload,identities=self.fixture()
            if change=='missing': payload['requests'].pop()
            elif change=='rank': payload['requests'][0]['class_features']['adapter_rank']=16
            elif change=='source': payload['requests'][0]['source_evidence']['source_admission']['source']['content_sha256']='b'*64
            else: payload['requests'][0]['class_features']['admitted_after_accept']=3
            with self.subTest(change=change),self.assertRaises(ValueError):
                source_profile_class_coverage(payload,identities)

    def test_identical_content_cannot_imply_different_rank(self):
        from scripts.ieee_tc_preflight import source_profile_class_coverage
        payload,identities=self.fixture()
        identities['b']['rank']=16
        with self.assertRaisesRegex(ValueError,'different PEFT ranks'):
            source_profile_class_coverage(payload,identities)


class MeasuredSourceProfileExport(unittest.TestCase):
    """Synthetic timestamp fixtures only; never model profiling measurements."""
    def setUp(self):
        import copy
        from tests.test_ieee_tc_preparation_cost import PreparationIntervals
        from tests.test_ieee_tc_transfer_pressure import ActivationPreparation
        self.payload, self.identities = SourceProfileDomainCoverage().fixture()
        fixture = ActivationPreparation()
        self.addCleanup(fixture.doCleanups)
        *_, native = fixture.make()
        p = self.payload
        p.update(runtime_boundary='dedicated_subprocess',
                 model_config_source='initialized_model_config_v1',
                 physical_allocation=dict(released=True), sources_before=native,
                 remote_transfers=[])
        p['model_config'].update(timing_contract='ieee_tc_native_v1',
            ieee_physical_allocation=True, ieee_admission_profile=dict(window_s=5.))
        self.context = dict(backend_environment_sha256='b'*64,
            resource_envelope_sha256='c'*64, input_contract_sha256='d'*64)
        for q in p['requests']:
            aid, e = q['adapter_id'], q['source_evidence']
            old = e['source_admission']
            tier, is_native = old['source']['tier'], old['source']['native']
            fixture = PreparationIntervals().inputs('host' if tier == 'gpu' else tier, is_native)
            a, n, r = fixture['admission'], fixture['native'], fixture['remote']
            old['source'].update(owner_id=a['source']['owner_id'],path=a['source']['path'])
            a.update(source=old['source'],service_class=old['service_class'],
                     admitted_after_accept=old['admitted_after_accept'],clock_id=n['clock_id'])
            n.update(lora_name=aid,lease_id=q['request_id'])
            if r:
                r.update(artifact_id=aid, transfer_id=q['request_id'],
                         remote_pack_performed=False,published_archive_verified=True)
                p['remote_transfers'].append(copy.deepcopy(r))
                e['remote_preparation'] = r
            e.update(state='released',source_admission=a,receipt=n,
                pending_kv_admission=dict(state='closed',close_receipt=dict(closed=True)))
            first, last = 154., 155.
            q.update(admitted_monotonic_s=100.,acquired_monotonic_s=100. if tier=='gpu' else 153.,
                first_token_monotonic_s=first,last_token_monotonic_s=last,
                admission_clock_id=n['clock_id'],native_clock_id=n['clock_id'],
                protected_at_admission=True,native_events=[{},dict(token_count=2)],
                timing=dict(native_output_tokens=2,native_terminal_observed=True,
                    native_clock_id=n['clock_id'],native_first_token_monotonic_s=first,
                    native_last_token_monotonic_s=last,native_decode_ms=1000.,native_tpot_ms=1000.,
                    native_dispatch_monotonic_s=153.,worker_wall_e2e_ms=2001.,
                    worker_completion_notification_ms=1.))

    def export(self, **changes):
        from scripts.ieee_tc_preflight import measured_source_profile_payloads
        args = dict(source_run_sha256='e'*64, identities=self.identities,
                    context=self.context,model_config=self.payload['model_config'])
        return measured_source_profile_payloads(self.payload, **(args | changes))

    def test_native_loaders_recompute_distinct_service_and_preparation_intervals(self):
        from faaslora.experiment.instance_pool import FrozenServiceProfiles
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
        from scripts.ieee_tc_preflight import digest
        service, preparation, coverage = self.export()
        self.assertTrue(coverage['class_domain_covered'])
        self.assertEqual((len(service['samples']),len(preparation['samples'])),(10,8))
        self.assertFalse(service['full_qualified'])
        self.assertFalse(preparation['numerical_adapter_correctness_qualified'])
        with tempfile.TemporaryDirectory() as tmp:
            loaded=[]
            for i,(payload,cls) in enumerate(((service,FrozenServiceProfiles),
                                           (preparation,FrozenPreparationProfiles))):
                path=Path(tmp)/f'{i}.json'
                path.write_text(json.dumps(payload))
                loaded.append(cls.load(path,expected_sha256=digest(path),
                    model_config=self.payload['model_config'],expected_context=self.context,beta=.5))
            self.assertEqual({v.d_ms for k,v in loaded[0].profiles.items() if k.tier=='remote'},{53000.})
            self.assertEqual({v for k,v in loaded[1].profiles.items() if k.tier=='remote'},{42000.})
            self.assertEqual({v for k,v in loaded[1].profiles.items() if k.tier=='host'},{3000.})
            self.assertTrue(loaded[1].identity()['activation_layout_available'])

    def test_missing_sha_context_and_changed_runtime_reject(self):
        for changes in (dict(source_run_sha256='not-a-sha'),dict(context={}),
                        dict(model_config=self.payload['model_config'] | {'dtype':'different'})):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.export(**changes)
        self.payload['model_config'].pop('ieee_admission_profile')
        with self.assertRaisesRegex(ValueError,'admission runtime'): self.export()

    def test_missing_native_completion_or_open_ownership_is_not_exported(self):
        import copy
        original=copy.deepcopy(self.payload)
        for mutate in (
            lambda q:q['source_evidence']['pending_kv_admission'].update(state='pending'),
            lambda q:q['timing'].update(native_output_tokens=1),
            lambda q:q['timing'].update(native_first_token_monotonic_s=153.5),
            lambda q:q['timing'].update(native_tpot_ms=900.),
            lambda q:q.update(native_clock_id='other'),
            lambda q:q['source_evidence']['receipt'].update(native_load_invoked=False)):
            self.payload=copy.deepcopy(original)
            mutate(self.payload['requests'][0])
            with self.assertRaises(ValueError): self.export()

    def test_remote_evidence_must_be_the_same_original_transfer(self):
        self.payload['requests'][0]['source_evidence']['remote_preparation']['transfer_id']='missing'
        with self.assertRaisesRegex(ValueError,'published transfer'): self.export()

    def test_only_warmup_cannot_fill_missing_classes(self):
        self.payload['source_profile_spec']['waves'][0]['role']='kernel_warmup_retained'
        with self.assertRaisesRegex(ValueError,'missing source/service'): self.export()

    def test_extra_warmup_retained_in_raw_but_not_exported(self):
        import copy
        p=self.payload
        p['source_profile_spec']['waves'].append(copy.deepcopy(p['source_profile_spec']['waves'][0]))
        p['source_profile_spec']['waves'][-1]['role']='kernel_warmup_retained'
        p['profile_waves'].append(dict(complete=True))
        for lane,q in enumerate(copy.deepcopy(p['requests'][:2])):
            q['request_id']=f'source-profile/w5/l{lane}'
            p['requests'].append(q)
        s,d,_=self.export()
        self.assertEqual((len(p['requests']),len(s['samples']),len(d['samples'])),(12,10,8))


class DevelopmentControlDerivation(unittest.TestCase):
    def fixture(self):
        waves,requests=[],[]
        for i in range(3):
            selected=dict(adapter_id='a',source_request_id='original')
            waves.append(dict(source='gpu',role='representative_measurement',round=i,requests=[selected]))
            a,b,f,z=10.+i,10.+i,10.+i+.1*(i+1),10.+i+.1*(i+1)+.02
            requests.append(dict(selected,request_id=f'source-profile/w{i}/l0',requested_source='gpu',
                reservation_released=True,actual_tokens=3,target_tokens=3,protected_at_admission=True,
                service_class=dict(tier='gpu'),admitted_monotonic_s=a,acquired_monotonic_s=b,
                first_token_monotonic_s=f,last_token_monotonic_s=z,admission_clock_id='fixture',
                native_clock_id='fixture',timing=dict(native_output_tokens=3,native_terminal_observed=True,
                    native_clock_id='fixture',native_first_token_monotonic_s=f,
                    native_last_token_monotonic_s=z,native_tpot_ms=10.),**{'pass':True}))
        return dict(kind='backend_native_native_source_matrix_qualification_v1',stage='complete',
            shutdown_called=True,profile_workspaces_removed=True,startup_latency_ms=7500.,
            model_config=dict(runtime_concurrency_cap=2,generation_contract='fixed_length_greedy_v1'),
            source_profile_spec=dict(waves=waves),requests=requests,**{'pass':True})

    def derive(self,payload,**updates):
        from scripts.ieee_tc_preflight import derive_ieee_development_controls
        values=dict(window_s=5.,interval_s=2.,historical_ttft_ms=5000.,min_instances=1,max_instances=4)
        values.update(updates)
        return derive_ieee_development_controls(payload,**values)

    def test_observed_spacing_capacity_startup_and_rounds_determine_explicit_values(self):
        result=self.derive(self.fixture())
        cc=result['coordination']
        self.assertAlmostEqual(cc['service_bin_ms'],10.)
        self.assertAlmostEqual(cc['ieee_scaling']['ttft_lower_ms'],300.)
        self.assertEqual(cc['ieee_scaling']['scale_down_cooldown_s'],8.)
        self.assertEqual(cc['ieee_scaling']['ttft_window_s'],15.)
        self.assertEqual(cc['ieee_scaling']['active_upper'],.5)
        self.assertEqual(cc['ieee_scaling']['active_lower'],.25)
        self.assertEqual(result['service_ewma_beta'],.5)
        self.assertFalse(result['formal_common_slo'])
        self.assertFalse(result['optimality_qualified'])

    def test_real_controller_can_keep_known_low_reference_through_cooldown(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl,ScalingAction
        cc=self.derive(self.fixture())['coordination']
        control=IEEEReplicaControl(cc['ieee_scaling'],interval_s=cc['scale_eval_interval_s'],
                                   min_instances=cc['min_instances'],max_instances=cc['max_instances'])
        control.observe_ttft('completed',100.,observed_at=10.)
        for now in (10.,12.,14.,16.,18.):
            result=control.evaluate(now=now,queue_depth=0,active_requests=0,ready_capacity=4,
                                    ready_instances=2,pending_instances=0)
        self.assertEqual(result['action'],ScalingAction.SCALE_DOWN)

    def test_low_capacity_threshold_preserves_upper_target_after_one_idle_replica_retirement(self):
        for cap in (2,8):
            raw=self.fixture();raw['model_config']['runtime_concurrency_cap']=cap
            cfg=self.derive(raw)['coordination']['ieee_scaling']
            for replicas in range(2,5):
                for active in range(replicas*cap+1):
                    if active/(replicas*cap)<cfg['active_lower']:
                        self.assertLess(active/((replicas-1)*cap),cfg['active_upper'])

    def test_unsupported_missing_or_invalid_evidence_does_not_get_a_default(self):
        for problem in ('cap_one','incomplete','failed','native_clock','no_tpot','wrong_generation'):
            raw=self.fixture()
            if problem=='cap_one':raw['model_config']['runtime_concurrency_cap']=1
            elif problem=='incomplete':raw['requests'].pop()
            elif problem=='failed':raw['pass']=False
            elif problem=='native_clock':raw['requests'][0]['native_clock_id']='other'
            elif problem=='no_tpot':raw['requests'][0]['timing']['native_tpot_ms']=None
            else:raw['model_config']['generation_contract']='legacy'
            with self.subTest(problem=problem),self.assertRaises(ValueError):self.derive(raw)
        with self.assertRaisesRegex(ValueError,'does not fit'):self.derive(self.fixture(),historical_ttft_ms=100.)
        with self.assertRaises(ValueError):self.derive(self.fixture(),interval_s=True)


class NativeAllocatorLaunchContract(unittest.TestCase):
    def test_worker_configuration_receipt_published_as_one_complete_document(self):
        from scripts.dedicated_engine_worker import _write_ready
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ready.json'
            value = dict(status='ready', model_config={'large_fixture': 'x'*100000})
            replace = Path.replace
            def inspect(pending, target):
                self.assertFalse(target.exists())
                self.assertEqual(json.loads(pending.read_text()), value)
                return replace(pending, target)
            with patch.object(Path, 'replace', inspect):
                _write_ready(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_existing_facade_capacity_resolution_is_pure_and_not_upstream_default(self):
        from faaslora.runtime_configuration import resolve_facade_lora_capacity
        for source, expected in (({}, 24), ({'max_loras':32}, 32),
                                 ({'max_loras':4, 'max_cpu_loras':32}, 32)):
            before = dict(source)
            result = resolve_facade_lora_capacity(source)
            self.assertEqual(result['max_cpu_loras'], expected)
            self.assertEqual(source, before)
            child, _ = runner._prepare_dedicated_subprocess_model_cfg(source, device_id=0)
            self.assertEqual(child['max_cpu_loras'], expected)
        for backend in ('transformers', 'sglang'):
            self.assertEqual(resolve_facade_lora_capacity({'backend':backend}), {'backend':backend})

    def test_initialized_worker_receipt_is_required_and_all_fields_remain_checked(self):
        from faaslora.runtime_configuration import initialized_worker_configuration, resolve_facade_lora_capacity
        requested = dict(backend='vllm', max_loras=4, device_id=0, timing_contract='ieee_tc_native_v1')
        actual = resolve_facade_lora_capacity(requested)
        receipt = dict(configuration_contract='initialized_model_config_v1', model_config=actual)
        result = initialized_worker_configuration(receipt, requested)
        self.assertEqual(result, actual)
        result['max_cpu_loras'] = 99
        self.assertEqual(actual['max_cpu_loras'], 24)
        for bad in ({}, {'model_config':actual}, dict(receipt, model_config=None)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'receipt'):
                initialized_worker_configuration(bad, requested)
        for field, value in (('max_cpu_loras',32), ('device_id',1), ('extra',None),
                             ('timing_contract','other'), ('max_loras',8)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'differs'):
                initialized_worker_configuration(dict(receipt, model_config=dict(actual, **{field:value})), requested)

    def test_spawn_rejects_wrong_initialized_configuration_and_retires_its_process(self):
        from faaslora.runtime_configuration import resolve_facade_lora_capacity
        process = SimpleNamespace(pid=987654, returncode=None)
        process.poll = lambda: process.returncode
        def wait(timeout):
            process.returncode = 0
            return 0
        process.wait = Mock(side_effect=wait)
        with tempfile.TemporaryDirectory() as tmp:
            def start(command, **kwargs):
                payload = json.loads(Path(command[command.index('--payload')+1]).read_text())
                actual = resolve_facade_lora_capacity(payload['model_cfg'])
                actual['max_cpu_loras'] += 1
                Path(command[command.index('--ready-file')+1]).write_text(json.dumps(dict(
                    status='ready', host='127.0.0.1', port=18080,
                    configuration_contract='initialized_model_config_v1', model_config=actual)))
                return process
            with patch.object(runner.SubprocessInferenceEngineProxy, '_startup_cleanup_done', True), \
                 patch.object(runner.SubprocessInferenceEngineProxy, '_write_process_meta'), \
                 patch.object(runner.tempfile, 'mkdtemp', return_value=tmp), \
                 patch.object(runner.subprocess, 'Popen', side_effect=start), \
                 patch.object(runner.os, 'getpgid', return_value=987654), \
                 patch.object(runner.os, 'killpg') as terminate:
                with self.assertRaisesRegex(ValueError, 'max_cpu_loras'):
                    asyncio.run(runner.SubprocessInferenceEngineProxy.spawn(
                        model_cfg=dict(backend='vllm', max_loras=4), cost_model={}, device_id=0))
                terminate.assert_called_once_with(987654, runner.signal.SIGTERM)
                process.wait.assert_called_once_with(5)

    def test_pending_profile_binding_matches_factory_without_mutating_descriptor(self):
        model = dict(backend='vllm', tensor_parallel_size=1, enforce_eager=True,
                     ieee_physical_allocation=True, visible_device_ids=[0, 1, 2, 3])
        service = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        service.model_cfg = dict(model)
        service.engine = SimpleNamespace(model_cfg=dict(model))
        service._initial_runtime_pending = True
        expected, _ = runner._prepare_dedicated_subprocess_model_cfg(
            model, device_id=0, runtime_gpu_ids=[0])
        bound = service._ieee_profile_binding_model_config()
        self.assertEqual(bound, expected)
        self.assertTrue(bound['skip_stale_gpu_cleanup'])
        self.assertTrue(bound['ieee_physical_allocation'])
        self.assertEqual(service.model_cfg, model)
        self.assertEqual(service.engine.model_cfg, model)
        service._initial_runtime_pending = False
        actual = dict(expected, max_loras=8)
        service.engine.model_cfg = actual
        self.assertIs(service._ieee_profile_binding_model_config(), actual)

    def test_source_profiles_use_actual_full_subprocess_boundary(self):
        from scripts.ieee_tc_preflight import initialize_qualification_runtime
        model = dict(backend='vllm', ieee_physical_allocation=True)
        actual = SimpleNamespace(model_cfg=dict(model, skip_stale_gpu_cleanup=True))
        with patch.object(runner.SubprocessInferenceEngineProxy, 'spawn',
                          new=AsyncMock(return_value=actual)) as spawn, \
             patch.object(runner, 'InferenceEngine') as direct:
            result = asyncio.run(initialize_qualification_runtime(model, 'native_source_matrix'))
            self.assertIs(result, actual)
            spawn.assert_awaited_once_with(model_cfg=model, cost_model={}, device_id=0, runtime_gpu_ids=[0])
            direct.assert_not_called()

    def test_source_profiles_cannot_bypass_physical_allocation(self):
        from scripts.ieee_tc_preflight import initialize_qualification_runtime
        with patch.object(runner.SubprocessInferenceEngineProxy, 'spawn', new=AsyncMock()) as spawn:
            for value in (None, False, 1):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'actual GPU allocation'):
                    asyncio.run(initialize_qualification_runtime(
                        dict(ieee_physical_allocation=value), 'native_source_matrix'))
            spawn.assert_not_awaited()

    def test_import_defaults_preserve_explicit_settings_and_conflicts(self):
        default = {}
        runner._apply_allocator_import_defaults(default)
        self.assertEqual(default, {'PYTORCH_ALLOC_CONF': 'expandable_segments:False',
                                  'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False'})
        candidates = [
            {'PYTORCH_ALLOC_CONF': 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True',
             'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY': 'uncached_background_v1'},
            {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            {'PYTORCH_ALLOC_CONF': 'one', 'PYTORCH_CUDA_ALLOC_CONF': 'another'},
            {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY': 'uncached_background_v1'},
        ]
        for env in candidates:
            before = dict(env)
            runner._apply_allocator_import_defaults(env)
            self.assertEqual(env, before)

    def test_actual_fresh_import_preserves_preconfigured_allocator(self):
        import subprocess
        import sys
        env = dict(os.environ)
        for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF'):
            env.pop(key, None)
        env['PYTORCH_ALLOC_CONF'] = 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True'
        env['FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY'] = 'uncached_background_v1'
        # No model, CUDA operation or live policy change: exercise the exact
        # fresh-process import which the previous unit-only checks missed.
        code = '''
import os
keys = ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_HIP_ALLOC_CONF',
        'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY')
before = {key: os.environ.get(key) for key in keys}
from scripts import run_all_experiments as runner
assert {key: os.environ.get(key) for key in keys} == before
runner._ieee_native_allocator_environment(
    {'backend':'vllm','ieee_native_host_allocator_policy':'uncached_background_v1'}, os.environ)
'''
        subprocess.run([sys.executable, '-c', code], env=env,
                       cwd=Path(runner.__file__).resolve().parent.parent,
                       check=True, capture_output=True, text=True, timeout=120)

    def test_background_candidate_has_distinct_frozen_identity_and_no_live_upgrade(self):
        cfg = {'backend':'vllm', 'ieee_native_host_allocator_policy':'uncached_background_v1'}
        setting = 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True'
        result = runner._ieee_native_allocator_environment(cfg, {})
        self.assertEqual(result['PYTORCH_ALLOC_CONF'], setting)
        self.assertEqual(result['FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY'], cfg['ieee_native_host_allocator_policy'])
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            runner._ieee_native_allocator_environment(cfg, {'PYTORCH_ALLOC_CONF':'pinned_max_cached_size_mb:0'})
        engine = runner.InferenceEngine.__new__(runner.InferenceEngine)
        engine.model_cfg = cfg
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'fresh process'):
            asyncio.run(engine.initialize())
        from scripts.dedicated_engine_worker import _run_worker
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory)/'payload.json'
            payload.write_text(json.dumps(dict(repo_root=str(Path.cwd()), model_cfg=cfg)))
            with patch.dict(os.environ, result, clear=True), self.assertRaises(KeyError) as raised:
                # Passing the pre-import check reaches the deliberately absent
                # next required payload field, without importing a model.
                asyncio.run(_run_worker(payload, Path(directory)/'ready.json'))
            self.assertEqual(raised.exception.args, ('cost_model',))
            with patch.dict(os.environ, {**result,'PYTORCH_ALLOC_CONF':'pinned_max_cached_size_mb:0'}, clear=True), \
                 self.assertRaisesRegex(ValueError, 'before worker imports'):
                asyncio.run(_run_worker(payload, Path(directory)/'ready.json'))

    def test_default_preserves_inherited_environment_without_claiming_policy(self):
        env = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OTHER': 'value'}
        self.assertEqual(runner._ieee_native_allocator_environment({}, env), env)

    def test_candidate_is_explicit_preimport_and_removes_alias_precedence(self):
        env = {'OTHER': 'value', 'PYTORCH_CUDA_ALLOC_CONF': 'pinned_max_cached_size_mb:0',
               'PYTORCH_HIP_ALLOC_CONF': ''}
        cfg = {'backend': 'vllm', 'ieee_native_host_allocator_policy': 'uncached_v1'}
        result = runner._ieee_native_allocator_environment(cfg, env)
        self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', result)
        self.assertNotIn('PYTORCH_HIP_ALLOC_CONF', result)
        self.assertEqual(result['PYTORCH_ALLOC_CONF'], 'pinned_max_cached_size_mb:0')
        self.assertEqual(result['FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY'], 'uncached_v1')
        self.assertEqual(result['OTHER'], 'value')
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF', env)

    def test_unknown_candidate_and_conflicting_tuning_are_rejected(self):
        for cfg in ({'ieee_native_host_allocator_policy': False},
                    {'ieee_native_host_allocator_policy': 'uncached_v1', 'backend': 'sglang'}):
            with self.assertRaises(ValueError):
                runner._ieee_native_allocator_environment(cfg, {})
        for alias in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF'):
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, 'conflicts'):
                runner._ieee_native_allocator_environment(
                    {'ieee_native_host_allocator_policy': 'uncached_v1'}, {alias: 'expandable_segments:True'})
        with self.assertRaisesRegex(ValueError, 'frozen model'):
            runner._ieee_native_allocator_environment({},
                {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY': 'uncached_v1'})

    def test_inprocess_initialization_cannot_apply_policy_after_import(self):
        engine = runner.InferenceEngine.__new__(runner.InferenceEngine)
        engine.model_cfg = {'ieee_native_host_allocator_policy': 'uncached_v1'}
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'fresh process'):
            asyncio.run(engine.initialize())

    def test_worker_rejects_environment_payload_mismatch_before_backend_import(self):
        from scripts.dedicated_engine_worker import _run_worker
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / 'payload.json'
            payload.write_text(json.dumps(dict(repo_root=str(Path.cwd()),
                model_cfg={'ieee_native_host_allocator_policy': 'uncached_v1'})))
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'before worker imports'):
                asyncio.run(_run_worker(payload, Path(directory) / 'ready.json'))


class IEEEControlContract(unittest.TestCase):
    def make(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl
        return IEEEReplicaControl(ieee_control_fixture(), interval_s=1, min_instances=1, max_instances=4)

    def evaluate(self, control, now, **overrides):
        args = dict(now=now, queue_depth=0, active_requests=0, ready_capacity=8,
                    ready_instances=2, pending_instances=0)
        args.update(overrides)
        return control.evaluate(**args)

    def test_each_signal_uses_max_not_averages_and_strict_upper_boundary(self):
        from faaslora.coordination.autoscaler import ScalingAction
        for signal in ('queue', 'active', 'ttft'):
            control = self.make()
            if signal == 'ttft':
                control.observe_ttft('slow', 1001., observed_at=0.)
            args = dict(queue_depth=3) if signal == 'queue' else dict(active_requests=7) if signal == 'active' else {}
            observed = self.evaluate(control, 0., **args)
            self.assertEqual(observed['action'], ScalingAction.SCALE_UP)
            self.assertEqual(observed['target_instances'], 3)
        control = self.make()
        observed = self.evaluate(control, 0., queue_depth=2, active_requests=6)
        self.assertEqual(observed['score'], 1.)
        self.assertEqual(observed['action'], ScalingAction.NO_ACTION)

    def test_p95_type1_deduplicated_window_unknown_not_free_scalein(self):
        control = self.make()
        for index in range(20):
            control.observe_ttft(str(index), 100 if index < 18 else 1100, observed_at=0.)
        self.assertFalse(control.observe_ttft('0', 100, observed_at=0.))
        sample = self.evaluate(control, 0.)
        self.assertEqual((sample['p95_ttft_ms'], sample['ttft_sample_count']), (1100,20))
        self.assertIsNone(self.evaluate(control, .1))
        sample = self.evaluate(control, 100.)
        self.assertIsNone(sample['p95_ttft_ms'])
        self.assertFalse(sample['all_low'])
        self.assertIsNone(control.low_since)

    def test_low_cooldown_resets_on_pressure_or_pending_and_respects_minimum(self):
        from faaslora.coordination.autoscaler import ScalingAction
        control = self.make()
        control.observe_ttft('fast', 100, observed_at=0.)
        self.evaluate(control, 0.)
        self.evaluate(control, 1., queue_depth=1)  # At lower boundary is not below.
        self.assertIsNone(control.low_since)
        self.evaluate(control, 2.)
        self.evaluate(control, 3., pending_instances=1)
        self.assertIsNone(control.low_since)
        self.evaluate(control, 4.)
        self.assertEqual(self.evaluate(control, 6.)['action'], ScalingAction.NO_ACTION)
        self.assertEqual(self.evaluate(control, 7.)['action'], ScalingAction.SCALE_DOWN)
        self.assertEqual(self.evaluate(control, 10., ready_instances=1, ready_capacity=4)['action'],
                         ScalingAction.NO_ACTION)

    def test_pending_counts_toward_replica_limit_not_ready_saturation(self):
        from faaslora.coordination.autoscaler import ScalingAction
        control = self.make()
        observed = self.evaluate(control, 0., queue_depth=100, active_requests=8, pending_instances=2)
        self.assertEqual(observed['active_saturation'], 1.)
        self.assertEqual(observed['action'], ScalingAction.NO_ACTION)
        with self.assertRaises(ValueError): self.evaluate(control, 1., active_requests=9)
        with self.assertRaises(ValueError): self.evaluate(control, -1.)

    def test_no_implicit_thresholds_or_nonfinite_state(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl
        for config in ({}, ieee_control_fixture() | dict(active_upper=float('nan')),
                       ieee_control_fixture() | dict(queue_lower=2), ieee_control_fixture() | dict(extra=1)):
            with self.assertRaises(ValueError):
                IEEEReplicaControl(config, interval_s=1, min_instances=1, max_instances=4)
        control = self.make()
        with self.assertRaises(ValueError): control.observe_ttft('bad', None, observed_at=0.)
        control.observe_ttft('good', 10, observed_at=2.)
        with self.assertRaises(ValueError): self.evaluate(control, 1.)


class IEEEActualControl(unittest.TestCase):
    def test_host_capacity_refresh_uses_existing_frozen_control_cadence(self):
        _,_,service,queue,engine = self.make()
        async def run():
            service.engine_factory = AsyncMock(return_value=(engine,None))
            service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            service._refresh_ieee_deferred_host_capacity = AsyncMock()
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            with patch('scripts.run_all_experiments.time.monotonic', return_value=1000.):
                await self.evaluate(service,result,backlog=0,active_requests=0)
                await self.evaluate(service,result,backlog=0,active_requests=0)
            service._refresh_ieee_deferred_host_capacity.assert_awaited_once()
            await queue.close()
        asyncio.run(run())

    def make(self, policy='full'):
        from tests.test_ieee_tc_transfer_pressure import ActivationPreparation
        fixture = ActivationPreparation()
        self.addCleanup(fixture.doCleanups)
        files, service, queue, engine, native = fixture.make(policy)
        service.coord_cfg = dict(ieee_scaling=ieee_control_fixture())
        service._scale_eval_interval_s = 1.
        service._instance_mode = 'dedicated'
        service._coordination_enabled = True
        service._hierarchical_residency_enabled = False
        service._primary_instance_id = None
        service._scaleup_runtime_instance_ids = set()
        service._runtime_forward_capacity_limit = Mock(return_value=4)
        service._select_dedicated_device_id = Mock(return_value=0)
        service._arrived_request_count = Mock(return_value=3)
        service._live_scale_up_preferred_gpu_adapters = Mock(side_effect=AssertionError('legacy forecast'))
        service._update_dynamic_scaling_live_state = Mock(side_effect=AssertionError('legacy votes'))
        service._stack.trigger_scaling_preload = AsyncMock(side_effect=AssertionError('legacy preload'))
        service._last_scale_up_handoff_plan = dict(planned_adapters=['stale-legacy'])
        service._last_scale_up_preload_budget = dict(mode='stale-legacy')
        return fixture, files, service, queue, engine

    async def evaluate(self, service, result, **overrides):
        args = dict(result=result, coord_enabled=True, replay_t0=0., results_view=[],
                    backlog=3, active_requests=0, busy_ratio=0., completed_count=0)
        args.update(overrides)
        return await service._maybe_run_live_scale_control_evaluation(**args)

    def test_actual_control_reaches_owned_activation_no_legacy_preparation_or_metadata(self):
        fixture, files, service, queue, engine = self.make()
        async def run():
            service.engine_factory = AsyncMock(return_value=(engine,None))
            result = SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            self.assertTrue(await self.evaluate(service,result))
            await service._wait_for_pending_scale_up_tasks()
            await fixture.finish_preparation(service)
            self.assertEqual(service.instance_pool.count(),1)
            self.assertEqual((files.host/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(len(result.scale_up_events),1)
            event = result.scale_up_events[0]
            self.assertEqual(event['activation_kind'],'natural_scaleout')
            self.assertIsNotNone(event['handoff_plan_sha256'])
            self.assertNotIn('planned_adapters',event)
            self.assertNotIn('budget_mode',event)
            self.assertEqual(service._ieee_control_events[0]['queue_depth'],3)
            service._stack.trigger_scaling_preload.assert_not_awaited()
            await queue.close()
        asyncio.run(run())

    def test_no_handoff_still_runs_actual_ready_residency_without_duplicate_epoch(self):
        from tests.test_ieee_tc_transfer_pressure import MixedOwnedPreparation
        fixture=MixedOwnedPreparation()
        self.addCleanup(fixture.doCleanups)
        files,service,queue,slot,owner,_,loads=fixture.make(remote_gpu=True)
        engine=slot.engine
        service.instance_pool=SimpleNamespace(get_slots=lambda:[slot])
        service._ieee_handoff_policy='no_handoff'
        service._coordination_enabled=True
        async def run():
            self.assertFalse((files.nvme/'a').exists())
            service._hierarchical_residency_enabled=True
            service._schedule_ieee_residency_epochs()
            task=service._ieee_residency_tasks[id(engine)]
            service._schedule_ieee_residency_epochs()
            self.assertIs(service._ieee_residency_tasks[id(engine)],task)
            await task
            service._reap_ieee_residency_tasks()
            self.assertEqual((files.nvme/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'completed')
            fixture.check_clean(files,service,owner)
            await queue.close()
        asyncio.run(run())

    def test_admitted_work_is_not_counted_twice_and_missing_config_never_uses_legacy(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.engine_factory=AsyncMock(return_value=(engine,None))
            service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            self.assertFalse(await self.evaluate(service,result,backlog=3,active_requests=3))
            record=service._ieee_control_events[0]
            self.assertEqual((record['queue_depth'],record['active_saturation']),(0,.75))
            service._ieee_scale_controller=None
            service.coord_cfg={}
            with self.assertRaises(ValueError): await self.evaluate(service,result)
            service._update_dynamic_scaling_live_state.assert_not_called()
            await queue.close()
        asyncio.run(run())

    def test_failed_residency_surfaces_once_not_automatic_retry(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            service._hierarchical_residency_enabled=True
            service._run_ieee_owned_preparation_plan=AsyncMock(side_effect=ValueError('missing class'))
            service._schedule_ieee_residency_epochs()
            await asyncio.gather(*service._ieee_residency_tasks.values(),return_exceptions=True)
            with self.assertRaisesRegex(ValueError,'missing class'): service._reap_ieee_residency_tasks()
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'failed')
            self.assertEqual(service._run_ieee_owned_preparation_plan.await_count,1)
            await queue.close()
        asyncio.run(run())

    def test_all_low_scalein_withdraws_before_cleanup_and_keeps_minimum(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.engine_factory=AsyncMock()
            primary=service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            second=SimpleNamespace(model_cfg=engine.model_cfg)
            victim=service.instance_pool.add_instance(second,None,owns_engine=True,device_id=1)
            service._primary_instance_id=primary
            async def cleanup(slot, **kw):
                self.assertEqual(slot.instance_id,victim)
                self.assertIsNone(service.instance_pool.get_slot(victim))
                self.assertEqual(service.instance_pool.count(),1)
            service._cleanup_removed_slot=AsyncMock(side_effect=cleanup)
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            completed=[SimpleNamespace(success=True,request_id='fast',overall_ttft_ms=100.)]
            for when in (10.,11.,12.,13.,14.,17.):
                with patch('scripts.run_all_experiments.time.monotonic',return_value=when):
                    await self.evaluate(service,result,backlog=0,active_requests=0,
                                        completed_count=1,results_view=completed)
            service._cleanup_removed_slot.assert_awaited_once()
            self.assertEqual(result.scale_down_events,1)
            self.assertEqual(service._ieee_control_events[3]['outcome'],'drained_replica_retired')
            self.assertEqual(service._ieee_control_events[-1]['ttft_sample_count'],1)
            await queue.close()
        asyncio.run(run())

    def test_failed_activation_is_not_retried_on_next_control_interval(self):
        _,_,service,queue,_=self.make()
        async def run():
            service.engine_factory=AsyncMock(side_effect=RuntimeError('startup ownership unknown'))
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            await self.evaluate(service,result)
            with self.assertRaisesRegex(RuntimeError,'no blind control-loop retry'):
                await service._wait_for_pending_scale_up_tasks()
            with self.assertRaisesRegex(RuntimeError,'no blind control-loop retry'):
                await self.evaluate(service,result)
            service.engine_factory.assert_awaited_once()
            self.assertEqual(service._ieee_activations[0]['state'],'startup_ownership_unresolved')
            await queue.close()
        asyncio.run(run())

    def test_retirement_joins_a_planning_epoch_before_runtime_shutdown(self):
        _,_,service,queue,engine=self.make()
        async def run():
            sid=service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            service._hierarchical_residency_enabled=True
            entered,release=asyncio.Event(),asyncio.Event()
            async def plan(**_):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await release.wait()  # An actual reader/observation must join first.
            service._run_ieee_owned_preparation_plan=AsyncMock(side_effect=plan)
            service._mark_instance_lifecycle_removed=Mock()
            service._scaleup_runtime_handoff_plans={}
            service._scaleup_runtime_lora_request_ordinals={}
            service._cancel_runtime_gpu_forward_tasks=AsyncMock()
            service._retire_ieee_host_budget=Mock()
            service._sync_stack_gpu_accounting=Mock()
            service._schedule_ieee_residency_epochs()
            await entered.wait()
            removed=service.instance_pool.remove_instance(sid)
            retiring=asyncio.create_task(service._cleanup_removed_slot(removed))
            await asyncio.sleep(0)
            engine.shutdown.assert_not_awaited()
            self.assertFalse(retiring.done())
            release.set()
            await retiring
            engine.shutdown.assert_awaited_once()
            self.assertFalse(service._ieee_residency_tasks)
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'cancelled')
            await queue.close()
        asyncio.run(run())


class IEEEInitialDeployment(unittest.TestCase):
    def test_main_deployment_contract_rejects_reuse_and_nonowned_runtime(self):
        model = dict(backend='vllm', ieee_gpu_references=True)
        coord = dict(routing_policy='ieee_confirmed', instance_mode='dedicated')
        scenarios = [dict(name='tc', resource_coordination=coord)]
        check = runner._ieee_initial_deployment_requested
        self.assertTrue(check(model, scenarios, {}, num_runs=1))
        for changes in (dict(num_runs=2), dict(scenarios=scenarios+[dict(name='legacy')]),
                        dict(model_cfg=dict(model, backend='transformers')),
                        dict(model_cfg=dict(model, ieee_gpu_references=False)),
                        dict(scenarios=[dict(name='tc',resource_coordination=dict(coord,instance_mode='shared'))])):
            args = dict(model_cfg=model, scenarios=scenarios, coord_cfg={}, num_runs=1)
            args.update(changes)
            with self.assertRaises(ValueError): check(**args)
        self.assertTrue(check(model, scenarios+[dict(name='legacy')], {}, num_runs=1, only_scenario='tc'))
        self.assertFalse(check({}, [dict(name='legacy')], {}, num_runs=2))

    def test_actual_constructor_keeps_uninitialized_descriptor_out_of_ready_pool(self):
        from faaslora.experiment.instance_pool import FrozenServiceProfiles, ServiceClassBins
        model = dict(backend='vllm', ieee_gpu_references=True)
        descriptor = runner.InferenceEngine(model, {})
        # Empty because this test must never publish a runtime or estimate work.
        profiles = FrozenServiceProfiles(ServiceClassBins((),(),(),(),()), {}, {},
                                         'constructor-fixture', (), .5, '{}')
        with tempfile.TemporaryDirectory() as root, \
             patch.object(runner.ScenarioRunner, '_load_ieee_service_profiles', return_value=profiles), \
             patch.object(runner.ScenarioRunner, '_load_ieee_preparation_profiles', return_value=None), \
             patch.object(runner, '_remote_artifact_from_env', return_value=None), \
             patch.object(runner, 'ResourceCoordinator'):
            args = dict(name='tc', baseline_type='faaslora_full', adapter_info={}, traces=[],
                remote_dir=Path(root)/'remote', nvme_dir=Path(root)/'nvme', bandwidth_mbps=0,
                hardware_cfg={}, cost_model={}, engine=descriptor, preload_cfg={}, workload_cfg={},
                coord_cfg=dict(routing_policy='ieee_confirmed', instance_mode='dedicated',
                               service_bin_ms=1., min_instances=2, max_instances=2),
                engine_factory=AsyncMock(), initial_runtime_pending=True)
            service = runner.ScenarioRunner(**args)
            self.assertEqual(service.instance_pool.count(), 0)
            self.assertIsNone(service._primary_instance_id)
            self.assertIsNone(descriptor.engine)
            service.engine_factory.assert_not_awaited()
            descriptor.engine = object()
            with self.assertRaisesRegex(ValueError, 'uninitialized descriptor'):
                runner.ScenarioRunner(**args)

    def make(self, policy='full'):
        from tests.test_ieee_tc_transfer_pressure import ActivationPreparation
        fixture = ActivationPreparation()
        self.addCleanup(fixture.doCleanups)
        files, service, queue, engine, native = fixture.make(policy)
        service.engine = runner.InferenceEngine(service.model_cfg, {})
        service.engine_factory = AsyncMock(return_value=(engine, None))
        service._initial_runtime_pending = True
        service._instance_mode = 'dedicated'
        service._coordination_enabled = True
        service._primary_instance_id = None
        service._pending_scale_up_tasks = set()
        service._pending_scale_up_device_ids = set()
        service._failed_runtime_device_ids = set()
        service._available_device_ids = Mock(return_value=[0, 1])
        service._scaleup_runtime_instance_ids = set()
        service._scaleup_runtime_handoff_plans = {}
        service._scaleup_runtime_lora_request_ordinals = {}
        service._mark_instance_lifecycle_removed = Mock()
        service._cancel_runtime_gpu_forward_tasks = AsyncMock()
        service._sync_stack_gpu_accounting = Mock()
        # CPU fixture uses this test process, not an exited native worker.
        service._retire_ieee_host_budget = Mock()
        return fixture, files, service, queue, engine

    def test_actual_initial_preparation_overlaps_init_without_changing_arrival_origin(self):
        fixture, files, service, queue, engine = self.make()
        async def run():
            started, release = asyncio.Event(), asyncio.Event()
            async def initialize(**_):
                started.set()
                await release.wait()
                return engine, None
            service.engine_factory = AsyncMock(side_effect=initialize)
            service._active_replay_t0 = 1234.5
            deployment = asyncio.create_task(service._start_ieee_initial_deployment())
            await started.wait()
            await fixture.finish_preparation(service)
            self.assertEqual((files.host/'a'/'weights').read_bytes(), b'a'*12288)
            self.assertEqual(service.instance_pool.count(), 0)
            self.assertFalse(deployment.done())
            release.set()
            await deployment
            self.assertIs(service.engine, engine)
            self.assertFalse(service._initial_runtime_pending)
            self.assertEqual(service._active_replay_t0, 1234.5)
            self.assertEqual(service._ieee_activations[0]['category'], 'initial')
            self.assertEqual(service._ieee_initial_deployment['state'], 'serving')
            self.assertEqual(service._primary_instance_id, service._ieee_activations[0]['activation_id'])
            self.assertFalse(service._pending_scale_up_device_ids)
            await service._shutdown_instance_pool()
            engine.shutdown.assert_awaited_once()
        asyncio.run(run())

    def test_no_handoff_still_activates_when_preload_disabled(self):
        _, files, service, _, engine = self.make('no_handoff')
        async def run():
            service.preload_cfg = dict(enabled=False)
            # This fixture tests entry wiring, not full-campaign qualification.
            with patch.object(service, '_require_ieee_full_qualification') as qualification:
                await service.preload()
                qualification.assert_called_once_with()
            self.assertIs(service.engine, engine)
            self.assertFalse((files.host/'a').exists())
            self.assertEqual(service._ieee_activations[0]['preparation_state'], 'disabled')
            await service._shutdown_instance_pool()
        asyncio.run(run())

    def test_unqualified_pending_preload_cannot_skip_guard(self):
        _, _, service, queue, _ = self.make('no_handoff')
        async def run():
            service.preload_cfg = dict(enabled=False)
            with self.assertRaisesRegex(RuntimeError, 'not qualified'):
                await service.preload()
            service.engine_factory.assert_not_awaited()
            self.assertEqual(service.instance_pool.count(), 0)
            await queue.close()
        asyncio.run(run())

    def test_insufficient_devices_reject_before_any_factory_or_reservation_leak(self):
        _, _, service, queue, _ = self.make()
        async def run():
            service.instance_pool.min_instances = 2
            service._available_device_ids.return_value = [0]
            with self.assertRaisesRegex(ValueError, 'insufficient physical'):
                await service._start_ieee_initial_deployment()
            service.engine_factory.assert_not_awaited()
            self.assertFalse(service._pending_scale_up_device_ids)
            await queue.close()
        asyncio.run(run())

    def test_first_ready_does_not_wait_for_min_pool_and_pending_keeps_initial_category(self):
        _, _, service, _, engine = self.make('no_handoff')
        async def run():
            service.instance_pool.min_instances = 2
            second_started, release = asyncio.Event(), asyncio.Event()
            async def activate(*_, reserved_device_id, activation_kind):
                if reserved_device_id == 1:
                    second_started.set()
                    await release.wait()
                distinct_engine = SimpleNamespace(model_cfg=engine.model_cfg)
                sid = service.instance_pool.add_instance(distinct_engine, None, owns_engine=False,
                    device_id=reserved_device_id)
                return dict(instance_id=sid, activation_kind=activation_kind)
            service._add_dedicated_instance_slot = AsyncMock(side_effect=activate)
            await service._start_ieee_initial_deployment()
            await second_started.wait()
            self.assertEqual(service.instance_pool.count(), 1)
            self.assertEqual(service._pending_scale_up_device_ids, {1})
            await service._ensure_min_instances(True)
            self.assertEqual(service._add_dedicated_instance_slot.await_count, 2)
            release.set()
            await service._wait_for_pending_scale_up_tasks()
            self.assertEqual(service.instance_pool.count(), 2)
            self.assertTrue(all(e['activation_kind']=='initial'
                                for e in service._ieee_initial_deployment['ready_events']))
            await service._shutdown_instance_pool()
        asyncio.run(run())

    def test_cancel_initial_wait_joins_factory_before_closing_movement_owner(self):
        _, files, service, queue, engine = self.make('no_handoff')
        async def run():
            started, release = asyncio.Event(), asyncio.Event()
            async def initialize(**_):
                started.set()
                await release.wait()
                return engine, None
            service.engine_factory = AsyncMock(side_effect=initialize)
            deployment = asyncio.create_task(service._start_ieee_initial_deployment())
            await started.wait()
            deployment.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(deployment.done())
            engine.shutdown.assert_not_awaited()
            release.set()
            with self.assertRaises(asyncio.CancelledError): await deployment
            engine.shutdown.assert_awaited_once()
            self.assertFalse(files.owner.host_budget_snapshot()['activation_reservations'])
            self.assertEqual(service.instance_pool.count(), 0)
            self.assertTrue(all(t.done() for t in service._ieee_initial_tasks))
            self.assertFalse(service._pending_scale_up_device_ids)
            self.assertEqual(service._failed_runtime_device_ids, {0})
            self.assertEqual(service._ieee_initial_deployment['state'], 'cancelled')
        asyncio.run(run())

    def test_failed_factory_keeps_unknown_host_reservation_and_prevents_retry(self):
        _, files, service, _, _ = self.make('no_handoff')
        async def run():
            service.engine_factory = AsyncMock(side_effect=RuntimeError('lost startup reply'))
            with self.assertRaisesRegex(RuntimeError, 'lost startup reply'):
                await service._start_ieee_initial_deployment()
            self.assertEqual(service._ieee_initial_deployment['state'], 'failed')
            self.assertEqual(service._failed_runtime_device_ids, {0})
            self.assertEqual(len(files.owner.host_budget_snapshot()['activation_reservations']), 1)
            with self.assertRaisesRegex(ValueError, 'empty owned pending pool'):
                await service._start_ieee_initial_deployment()
            service.engine_factory.assert_awaited_once()
        asyncio.run(run())

    def test_initial_sibling_failure_does_not_hide_behind_ready_sibling(self):
        _, _, service, _, engine = self.make('no_handoff')
        async def run():
            service.instance_pool.min_instances = 2
            async def activate(*_, reserved_device_id, activation_kind):
                if reserved_device_id == 1:
                    raise RuntimeError('second runtime failed')
                sid = service.instance_pool.add_instance(engine, None, owns_engine=True,
                    device_id=reserved_device_id)
                return dict(instance_id=sid, activation_kind=activation_kind)
            service._add_dedicated_instance_slot = AsyncMock(side_effect=activate)
            with self.assertRaisesRegex(RuntimeError, 'second runtime failed'):
                await service._start_ieee_initial_deployment()
            self.assertEqual(service.instance_pool.count(), 0)
            engine.shutdown.assert_awaited_once()
            self.assertTrue(all(t.done() for t in service._ieee_initial_tasks))
            self.assertEqual(service._ieee_initial_deployment['state'], 'failed')
        asyncio.run(run())

    def test_late_initial_failure_updates_deployment_and_prevents_min_pool_retry(self):
        _, _, service, _, engine = self.make('no_handoff')
        async def run():
            service.instance_pool.min_instances = 2
            release = asyncio.Event()
            async def activate(*_, reserved_device_id, activation_kind):
                if reserved_device_id == 1:
                    await release.wait()
                    raise RuntimeError('late initial failure')
                sid = service.instance_pool.add_instance(engine, None, owns_engine=True, device_id=0)
                return dict(instance_id=sid, activation_kind=activation_kind)
            service._add_dedicated_instance_slot = AsyncMock(side_effect=activate)
            await service._start_ieee_initial_deployment()
            self.assertEqual(service._ieee_initial_deployment['state'], 'serving')
            release.set()
            with self.assertRaisesRegex(RuntimeError, 'activation failed'):
                await service._wait_for_pending_scale_up_tasks()
            self.assertEqual(service._ieee_initial_deployment['state'], 'failed')
            self.assertEqual(service._ieee_initial_deployment['failures'],
                             [dict(device_id=1,error_type='RuntimeError')])
            with self.assertRaisesRegex(RuntimeError, 'initial activation failed'):
                await service._ensure_min_instances(True)
            self.assertEqual(service._add_dedicated_instance_slot.await_count, 2)
            await service._shutdown_instance_pool()
        asyncio.run(run())


class SharedDedicatedFactory(unittest.TestCase):
    def test_parent_configuration_and_actual_device_path_are_preserved(self):
        model = dict(backend='vllm', visible_device_ids=[0, 1, 2, 3], device_id=0,
                     tensor_parallel_size=1, enforce_eager='auto')
        engine = SimpleNamespace(device_id=2, shutdown=AsyncMock())
        coord = object()
        async def run():
            with patch.object(runner.SubprocessInferenceEngineProxy, 'spawn',
                              AsyncMock(return_value=engine)) as spawn, \
                 patch.object(runner, 'ResourceCoordinator', return_value=coord) as construct:
                result = await runner._spawn_dedicated_scenario_engine(
                    model_cfg=model, host_visible_ids=[0,1,2,3], cost_cfg={},
                    coord_cfg={'x':1}, hardware_cfg={'gpu_device_ids':[0,1,2,3]},
                    coordination_enabled=True, use_subprocess=True, device_id=2)
                self.assertEqual(result, (engine, coord))
                args = spawn.await_args.kwargs
                self.assertEqual(args['runtime_gpu_ids'], [2])
                self.assertEqual(args['model_cfg']['enforce_eager'], 'auto')
                self.assertNotIn('requested_enforce_eager', args['model_cfg'])
                self.assertEqual(construct.call_args.kwargs['config']['gpu_device_ids'], [2])
                self.assertEqual(model['device_id'], 0)
                engine.shutdown.assert_not_awaited()
        asyncio.run(run())

    def test_controller_failure_retires_the_successfully_created_runtime(self):
        engine = SimpleNamespace(device_id=0, shutdown=AsyncMock())
        async def run():
            with patch.object(runner.SubprocessInferenceEngineProxy, 'spawn',
                              AsyncMock(return_value=engine)), \
                 patch.object(runner, 'ResourceCoordinator', side_effect=ValueError('controller failed')):
                with self.assertRaisesRegex(ValueError, 'controller failed'):
                    await runner._spawn_dedicated_scenario_engine(
                        model_cfg=dict(device_id=0,tensor_parallel_size=1),
                        host_visible_ids=[0], cost_cfg={}, coord_cfg={}, hardware_cfg={},
                        coordination_enabled=True, use_subprocess=True)
                engine.shutdown.assert_awaited_once()
        asyncio.run(run())


class FullPreparationEntryIsolation(unittest.TestCase):
    def test_ieee_release_callback_cannot_start_legacy_gpu_forwarding(self):
        service=runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        service._routing_policy='ieee_confirmed'
        # Nothing beyond policy is needed: no cache hints, capacity guesses,
        # legacy objective or new task may be consulted by this callback.
        self.assertFalse(service._schedule_runtime_gpu_forward(object()))

    def test_actual_stack_cache_access_cannot_start_legacy_promotion_under_ieee(self):
        from faaslora.experiment.experiment_stack import ExperimentStack
        async def run():
            for policy in ('ieee_confirmed','adapter_affinity'):
                with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                    stack=ExperimentStack(adapter_info={'a':{'size_mb':1}}, hardware_cfg={},
                        coord_cfg={'routing_policy':policy},preload_cfg={'dynamic_forwarding_enabled':True},
                        remote_dir=Path(tmp)/'remote',nvme_dir=Path(tmp)/'nvme',host_dir=Path(tmp)/'host')
                    try:
                        stack._ensure_registered()
                        stack._nvme_paths['a']=str(Path(tmp)/'nvme'/'a')
                        stack.sync_local_tier_paths=Mock()
                        stack._can_schedule_explicit_host_promotion=Mock(return_value=True)
                        stack._promote_nvme_hit_to_host=AsyncMock()
                        stack.record_access('a',load_time_ms=1,hit=True)
                        if policy=='ieee_confirmed':
                            self.assertFalse(stack._pending_host_promotions)
                            stack._can_schedule_explicit_host_promotion.assert_not_called()
                            stack._promote_nvme_hit_to_host.assert_not_awaited()
                        else:
                            await asyncio.gather(*stack._pending_host_promotions.values())
                            stack._promote_nvme_hit_to_host.assert_awaited_once_with('a')
                    finally:
                        await stack.stop()
        asyncio.run(run())


class FullPrefixQualification(unittest.TestCase):
    def fixture(self):
        from dataclasses import make_dataclass
        request = dict(request_id='r', adapter_id='a', success=True,
            generation_contract='fixed_length_greedy_v1', timing_contract='ieee_tc_native_v1',
            output_contract_match=True, output_tokens=2, completion_tokens=2,
            readiness_tier_before_dispatch='remote', native_token_timing={'fixture':True},
            gpu_reference_evidence={'fixture':True})
        scenario = make_dataclass('FixtureScenario', [('requests',list)])([request])
        pool = SimpleNamespace(count=Mock(return_value=0))
        service = SimpleNamespace(_routing_policy='ieee_confirmed', _initial_runtime_pending=True,
            instance_pool=pool, _service_profiles=object(), _preparation_profiles=object(),
            model_cfg={'ieee_physical_allocation':True}, _external_replay=None,
            traces=[SimpleNamespace(request_id='r',adapter_id='a',expected_output_tokens=2)],
            _start_ieee_initial_deployment=AsyncMock(), _ieee_initial_deployment={'fixture':True},
            _select_dedicated_device_id=Mock(return_value=1), _pending_scale_up_device_ids=set(),
            _coordination_enabled=True, _add_dedicated_instance_slot=AsyncMock(
                return_value={'activation_kind':'controlled'}),
            run=AsyncMock(return_value=(scenario,{})), _shutdown_instance_pool=AsyncMock(),
            _remote_transfer_evidence=[], _stack=SimpleNamespace(
                preloading_manager=SimpleNamespace(ieee_movements=SimpleNamespace(snapshot=lambda:{})),
                residency_manager=SimpleNamespace(local_source_references=SimpleNamespace(
                    host_budget_snapshot=lambda:{}))))
        return service, request

    def execute(self, service, result):
        from scripts.ieee_tc_preflight import qualify_ieee_full_prefix
        return asyncio.run(qualify_ieee_full_prefix(service,result))

    def test_same_run_path_and_explicit_controlled_category_without_production_claim(self):
        service, _ = self.fixture()
        result = {}
        self.execute(service,result)
        self.assertTrue(result['pass'])
        self.assertFalse(result['production_launch_authorized'])
        self.assertFalse(result['formal_performance_result'])
        self.assertFalse(result['numerical_adapter_correctness_qualified'])
        service._add_dedicated_instance_slot.assert_awaited_once_with(
            True,reserved_device_id=1,activation_kind='controlled')
        service.run.assert_awaited_once_with()
        service._shutdown_instance_pool.assert_awaited_once_with()
        self.assertFalse(service._pending_scale_up_device_ids)
        # The qualification function does not replace/remove the main guard.
        with self.assertRaisesRegex(RuntimeError,'not qualified'):
            runner.ScenarioRunner._require_ieee_full_qualification(service)

    def test_reject_wrong_generation_or_missing_native_evidence_after_retaining_rows(self):
        for field, value in [('success',False),('completion_tokens',1),('output_tokens',1),
                             ('native_token_timing',{}),('gpu_reference_evidence',{}),
                             ('readiness_tier_before_dispatch',''),('adapter_id','wrong')]:
            service, row = self.fixture()
            row[field] = value
            result = {}
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError,'native generation/source'):
                self.execute(service,result)
            self.assertFalse(result['pass'])
            self.assertEqual(result['requests'][0][field],value)
            service._shutdown_instance_pool.assert_awaited_once()

    def test_initial_failure_and_cancel_still_join_shutdown(self):
        for failure in [RuntimeError('startup failed'),asyncio.CancelledError()]:
            service, _ = self.fixture()
            service._start_ieee_initial_deployment.side_effect = failure
            with self.subTest(failure=type(failure)), self.assertRaises(type(failure)):
                self.execute(service,{})
            service.run.assert_not_awaited()
            service._shutdown_instance_pool.assert_awaited_once()

    def test_prefix_failure_retains_partial_rows_separately_without_success(self):
        service,row=self.fixture()
        service.run.side_effect=RuntimeError('controller failed')
        service._interrupted_replay_evidence=[dict(complete=False,planned_request_count=1,
            submitted_count=1,requests=[row],unsubmitted_request_ids=[],collection_errors=[])]
        result={}
        with self.assertRaisesRegex(RuntimeError,'controller failed'):
            self.execute(service,result)
        self.assertFalse(result['pass'])
        self.assertEqual(result['requests'],[])
        self.assertEqual(result['interrupted_replays'][0]['requests'],[row])
        self.assertFalse(result['formal_performance_result'])

    def test_controlled_failure_releases_device_reservation_and_retires_pool(self):
        service, _ = self.fixture()
        service._add_dedicated_instance_slot.side_effect = RuntimeError('controlled failed')
        with self.assertRaisesRegex(RuntimeError,'controlled failed'):
            self.execute(service,{})
        self.assertFalse(service._pending_scale_up_device_ids)
        service.run.assert_not_awaited()
        service._shutdown_instance_pool.assert_awaited_once()

    def test_invalid_prefix_or_existing_deployment_cannot_start(self):
        for field, value in [('_initial_runtime_pending',False),('_service_profiles',None),
                             ('_preparation_profiles',None),('_external_replay',object()),
                             ('traces',[])]:
            service, _ = self.fixture()
            setattr(service,field,value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'empty measured'):
                self.execute(service,{})
            service._start_ieee_initial_deployment.assert_not_awaited()


class ManagedEngineLaunch(unittest.TestCase):
    def test_managed_cleanup_only_revalidates_owned_service(self):
        engine = runner.InferenceEngine({}, {})
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}), \
             patch('scripts.ieee_tc_preflight.verify_current_service') as verify, \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            engine._maybe_kill_stale_gpu_processes()
            verify.assert_called_once_with()
            global_kill.assert_not_called()


class ExternalArrivalControl(unittest.TestCase):
    def setUp(self):
        self.runner = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        now = time.perf_counter()
        self.runner._external_replay = SimpleNamespace(
            observed_times=[now-2, now-.2], context={'replay_t0_s':now-5})
        self.runner._scheduled_arrivals = list(range(4000))
        self.runner._arrival_window_s = 1.
        self.runner._scale_eval_interval_s = 1.

    def test_backlog_and_rate_use_received_requests_only(self):
        r = self.runner
        self.assertEqual(r._arrived_request_count(0.), 2)
        self.assertEqual(r._arrival_rps(0.), 1.)
        self.assertEqual(r._arrived_request_count_at_elapsed_s(100000.), 2)

    def test_ready_prediction_cannot_read_future_adapter_identity(self):
        r = self.runner
        r._waiting_visible_trace_queue = Mock(return_value=['received-adapter'])
        r.traces = ['future-adapter']*4000
        candidates, count = r._scale_up_ready_candidate_queue(
            replay_t0=0., arrived_request_count=2, ready_delay_ms=100000.)
        self.assertEqual(candidates, ['received-adapter'])
        self.assertEqual(count, 2)

    def test_idle_floor_cannot_profile_the_future_trace(self):
        r = self.runner
        r._scheduled_arrivals = [0., 9999., 19998.]
        r._external_replay.observed_times = []
        self.assertEqual(r._derive_trace_scale_down_floor_s(), 1.)

    def test_external_mode_cannot_use_internal_timer(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot fall back'):
            asyncio.run(self.runner._await_trace_arrival(SimpleNamespace(arrival_time=0), 0))

    def test_observation_not_late_dispatch_updates_demand(self):
        r = self.runner
        trace = SimpleNamespace(request_id='a', adapter_id='adapter-a')
        r._external_trace_by_id = {'a':trace}
        r._observe_live_arrived_lora = Mock()
        r._observe_live_waiting_trace = Mock()
        r._stack = SimpleNamespace(record_arrival=Mock())
        r._observe_external_ingress({'request_id':'a', 'server_received_s':12.})
        r._stack.record_arrival.assert_called_once_with('adapter-a', observed_at=12.)
        r._observe_live_waiting_trace.assert_called_once_with(trace)

class ManagedEngineFailure(unittest.TestCase):
    def test_global_cleanup_is_forbidden_inside_managed_launch(self):
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}):
            with self.assertRaisesRegex(RuntimeError, 'owned service scope'):
                runner._kill_stale_gpu_processes()

    def test_native_mode_cannot_use_historical_unbounded_cleanup(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'guarded qualification launcher'):
                engine._maybe_kill_stale_gpu_processes()
            global_kill.assert_not_called()

    def test_native_initialize_tries_exactly_one_configuration(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1',
                                         'name':'existing-local-model'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        engine._resolve_vllm_runtime_settings = Mock(return_value={
            'env_updates':{}, 'enable_chunked_prefill':True,
            'enable_prefix_caching':True, 'tokenizer_mode':'auto'})
        engine._maybe_kill_stale_gpu_processes = Mock()
        engine._try_create_engine = AsyncMock(return_value=None)
        with patch('scripts.ieee_tc_preflight.verify_current_service', return_value={}), \
             patch.object(runner, 'CUDA_AVAILABLE', True), \
             patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, '_check_shm_for_vllm'):
            with self.assertRaisesRegex(RuntimeError, 'engine creation failed'):
                asyncio.run(engine.initialize())
        self.assertEqual(engine._try_create_engine.await_count, 1)
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_chunked_prefill'])
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_prefix_caching'])

    def test_construction_failure_does_not_clean_unrelated_workers_or_hide_error(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        with patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, 'AsyncEngineArgs', side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch.object(runner, 'AsyncLLMEngine', SimpleNamespace(
                 from_engine_args=Mock(side_effect=RuntimeError('native operator failed')))), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'without config retry.*native operator failed'):
                asyncio.run(engine._try_create_engine('model', tp=1, gpu_util=.8, max_len=1024,
                    eager=True, enable_lora=True, max_loras=2, max_lora_rank=16))
            global_kill.assert_not_called()


class ExternalDispatcherIntegration(unittest.IsolatedAsyncioTestCase):
    def setup_runner(self, *, fail=False):
        r = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        r._generation_contract = 'legacy'
        r.traces = [SimpleNamespace(request_id=str(i)) for i in range(3)]
        observed = []

        async def receive():
            for i in range(3):
                if fail and i == 1:
                    raise ValueError('changed external trace')
                now = time.perf_counter()
                observed.append(now)
                yield i, {'server_received_s':now}
                await asyncio.sleep(.002)

        r._external_replay = SimpleNamespace(plan=SimpleNamespace(entries=r.traces),
                                             observed_times=observed, receive=receive)
        r._live_scale_eval_period_s = Mock(return_value=1.)
        r._active_request_count = Mock(return_value=0)
        r._busy_instance_ratio = Mock(return_value=0.)
        r._waiting_visible_trace_queue = Mock(return_value=[])
        r._maybe_run_live_scale_control_evaluation = AsyncMock()
        r._emit_live_snapshot = Mock()
        return r

    async def test_service_completion_does_not_gate_future_ingress(self):
        r = self.setup_runner()
        starts, ends = [], []
        async def run_one(i, trace, *, arrival_released_at):
            starts.append(arrival_released_at)
            await asyncio.sleep(.05)
            ends.append(time.perf_counter())
            return runner.RequestResult(
                request_id=trace.request_id, adapter_id=None, is_burst=False,
                burst_phase='normal', cache_hit=False, cache_tier='backbone',
                lora_io_ms=0., vllm_ttft_ms=1., ttft_ms=1., contention_ms=0.,
                defer_ms=0., tpot_ms=1., e2e_ms=2., input_tokens=1,
                output_tokens=2, cost_usd=0., success=True)
        raw, _ = await r._run_continuous_observed(traces=r.traces, trace_start_index=0,
            replay_t0=time.perf_counter(), run_one_fn=run_one, completed_before_window=0,
            total_requests=3, result=SimpleNamespace(scale_up_events=[], scale_down_events=0),
            coord_enabled=False)
        self.assertEqual([x.request_id for x in raw], ['0','1','2'])
        self.assertLess(max(starts), min(ends))
        self.assertEqual(len(r._external_replay.observed_times), 3)

    async def test_publisher_error_propagates_and_pending_tasks_are_joined(self):
        r = self.setup_runner(fail=True)
        cancelled = []
        async def run_one(i, trace, *, arrival_released_at):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(i)
        with self.assertRaisesRegex(ValueError, 'changed external trace'):
            await r._run_continuous_observed(traces=r.traces, trace_start_index=0,
                replay_t0=time.perf_counter(), run_one_fn=run_one, completed_before_window=0,
                total_requests=3, result=SimpleNamespace(scale_up_events=[], scale_down_events=0),
                coord_enabled=False)
        self.assertEqual(cancelled, [0])

    async def test_main_starts_receiving_before_initialization_and_logs_every_receipt(self):
        with tempfile.TemporaryDirectory(prefix='ptci-') as tmp:
            root = Path(tmp)
            source = root/'source.json'
            source.write_text(json.dumps({'requests':[
                {'request_id':str(i), 'arrival_time_s':i*.01, 'adapter_id':'adapter-a'} for i in range(3)]}))
            plan = FrozenReplayPlan.load(source)
            now = time.perf_counter()
            origin = {'clock_id':local_monotonic_clock_id(), 'deployment_notice_s':now,
                      'replay_t0_s':now+.005}
            context = {**origin, 'plan':plan.identity(), 'address':str(root/'socket'),
                       'nonce':'test', 'frame_limit':16384, 'tiny_witness':False}
            receipt = root/'exec_receipt.json'
            receipt.write_text(json.dumps({'external_replay':context}))
            events = []
            async def start():
                return origin
            publisher = asyncio.create_task(publish_frozen_replay(plan, context['address'],
                                              'test', start, events.append))
            while not events:
                await asyncio.sleep(.001)
            async def initialize_then_serve(*args, external_replay, **kwargs):
                self.assertFalse(external_replay.background_task.done())
                await asyncio.sleep(.05)  # Backend still starting; frontend must receive.
                self.assertEqual(len(external_replay.records), 3)
                async for _ in external_replay.receive():
                    pass
                return 'served'
            with patch.dict(os.environ, {'FAASLORA_TC_EXTERNAL_REPLAY':'1',
                                         'FAASLORA_TC_LAUNCH_RECEIPT':str(receipt)}), \
                 patch('scripts.ieee_tc_preflight.verify_current_service', return_value={'pid':0}), \
                 patch.object(runner, '_main_async_impl', side_effect=initialize_then_serve):
                self.assertEqual(await runner.main_async('unused-config'), 'served')
            await publisher
            log = [json.loads(x) for x in (root/'service_ingress.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['event']=='request_received' for x in log), 3)
            self.assertEqual(sum(x['event']=='request_dequeued' for x in log), 3)
            self.assertTrue(next(x for x in log if x['event']=='service_ingress_terminal')['complete'])
            resource = json.loads((root/'physical_deployment/summary.json').read_text())
            self.assertEqual(resource['n_plan'], 3)
            self.assertEqual(resource['n_terminal'], 0)  # Mock service did not report terminals.
            self.assertFalse(resource['measurement_complete'])
            self.assertIsNone(resource['gpu_seconds_per_correct_request'])


class DeploymentTerminalIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_actual_run_reports_success_exception_and_interruption_once(self):
        for outcome in (SimpleNamespace(success=True), RuntimeError('controlled failure'),
                        asyncio.CancelledError()):
            with self.subTest(outcome=type(outcome).__name__):
                r = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
                r.name, r.baseline_type, r._coordination_enabled = 'fixture', 'faaslora_full', False
                r.engine = SimpleNamespace(backend='vllm')
                r.wl_cfg, r._ttft_slo_ms = {}, 1000.
                trace = SimpleNamespace(request_id='q', adapter_id='a')
                r.traces = [trace]
                r._external_replay = SimpleNamespace(context={'replay_t0_s':time.monotonic()-1}, records={'q':{}})
                r._physical_deployment = SimpleNamespace(terminal=Mock())
                for name in ('_assert_clean_gpu_environment', '_begin_instance_lifecycle_tracking',
                             '_release_live_started_lora', '_release_live_waiting_trace', '_release_live_arrived_lora'):
                    setattr(r, name, Mock())
                for name in ('_attach_ieee_file_pressure', '_ensure_min_instances',
                             '_acquire_dispatch_admission', '_release_dispatch_admission'):
                    setattr(r, name, AsyncMock())
                r._scheduled_offset = Mock(return_value=0.)
                r._prepare_request_execution_plan_cache = Mock(return_value={'q':object()})
                r._exec_request = (AsyncMock(side_effect=outcome) if isinstance(outcome, BaseException)
                                   else AsyncMock(return_value=outcome))

                async def dispatch(**kwargs):
                    await kwargs['run_one_fn'](0, trace, arrival_released_at=time.monotonic())
                    raise LookupError('stop after actual request path; no mock performance aggregation')
                r._run_continuous_observed = dispatch
                expected = type(outcome) if isinstance(outcome, BaseException) else LookupError
                with self.assertRaises(expected):
                    await r.run()
                r._physical_deployment.terminal.assert_called_once()
                call = r._physical_deployment.terminal.call_args
                self.assertEqual(call.args, ('q',))
                self.assertEqual(call.kwargs['interrupted'], isinstance(outcome, asyncio.CancelledError))
                self.assertIs(call.kwargs['result'], None if isinstance(outcome, BaseException) else outcome)

    async def test_failed_return_does_not_skip_other_runtime_cleanup(self):
        r = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        slots = {str(i):SimpleNamespace(instance_id=str(i)) for i in range(2)}
        r.instance_pool = SimpleNamespace(get_slots=lambda:list(slots.values()), remove_instance=slots.pop)
        r._cleanup_removed_slot = AsyncMock(side_effect=[RuntimeError('return unconfirmed'), None])
        with self.assertRaisesRegex(RuntimeError, 'unresolved runtime ownership'):
            await r._shutdown_instance_pool()
        self.assertEqual(r._cleanup_removed_slot.await_count, 2)
        self.assertEqual(slots, {})


class NativePhysicalShutdown(unittest.IsolatedAsyncioTestCase):
    def proxy(self):
        events = []
        proxy = runner.SubprocessInferenceEngineProxy.__new__(runner.SubprocessInferenceEngineProxy)
        proxy._normal_shutdown_completed = False
        proxy._engine_dead = False
        proxy._workdir = Path('/unused-test-workdir')
        proxy._keep_worker_logs_requested = lambda: True
        proxy._preserve_worker_workdir = Mock()
        proxy._rpc = AsyncMock(side_effect=lambda _: events.append('shutdown_ack'))
        proxy._process = SimpleNamespace(poll=lambda: None,
            wait=lambda _: events.append('parent_exit'))
        proxy._close_all_rpc_channels = AsyncMock(side_effect=lambda: events.append('channels_closed'))
        proxy._terminate_process_tree = AsyncMock(side_effect=lambda **_: events.append('owned_cleanup'))
        proxy._physical_allocation = SimpleNamespace(
            wait_workers=AsyncMock(side_effect=lambda **_: events.append('native_worker_exit')),
            release=Mock(side_effect=lambda: events.append('physical_return')))
        return proxy, events

    async def test_shutdown_ack_is_not_the_release_boundary(self):
        proxy, events = self.proxy()
        await proxy.shutdown()
        self.assertEqual(events, ['shutdown_ack', 'channels_closed', 'parent_exit',
                                  'owned_cleanup', 'native_worker_exit', 'physical_return'])

    async def test_failed_native_exit_does_not_return_allocation(self):
        proxy, events = self.proxy()
        proxy._physical_allocation.wait_workers.side_effect = TimeoutError('still alive')
        with self.assertRaises(TimeoutError):
            await proxy.shutdown()
        proxy._physical_allocation.release.assert_not_called()
        self.assertTrue(proxy._engine_dead)
        proxy._preserve_worker_workdir.assert_called_once_with('physical_release_unconfirmed')


if __name__ == '__main__':
    unittest.main()
