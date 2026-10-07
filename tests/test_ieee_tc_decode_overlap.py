"""CPU interval tests: descriptive overlap must not invent causal stall time."""
import copy
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from scripts.ieee_tc.check_d299_full_context import (
    complete_failure_output_hashes, failed_run_partition, project_incomplete)

from scripts.ieee_tc.analyze_d297_full import (
    audit_preparation_attribution, decode_overlap_summary, native_load_overlap_summary, prefill_overlap_summary,
    lifecycle_gaps, paired_timeline_groups, inspect_retained_lifecycle,
    routing_load_consistency, inspect_retained_routing_load,
    routing_length_diagnosis, inspect_retained_routing_lengths,
)


class RetainedRoutingLoad(unittest.TestCase):
    def requests(self):
        rows = []
        for i, (at, admitted, dispatched, last, acquired, q, pending) in enumerate(
                [(9., 10., 14., 20., 13., 0, 0),
                 (11., 12., 15., 21., 14., 1, 1),
                 (15.5, 16., 17., 22., 16.5, 2, 0)]):
            rid = f'r{i}'
            candidate = dict(epoch=i+1, request_id=rid, replica_id='a', adapter_id='x',
                runtime_ready=True, available_slots=8-q, admitted_requests=q,
                active_adapters=['x'] if q else [], max_active_loras=8, pending_loads=pending,
                gpu_utilization_pct=0., last_dispatch_at=0.,
                service_class=dict(tier='remote', prompt_bin=0, output_limit_bin=0,
                    rank_bin=0, footprint_bin=0, representation='remote', admitted_bin=int(q>0)),
                service=dict(d_ms=1., t_ms=2., o_ms=3.))
            rows.append(dict(request_id=rid, adapter_id='x', instance_id='a', success=True,
                native_token_timing=dict(native_clock_id='clock', native_terminal_observed=True,
                    controller_admitted_monotonic_s=admitted, native_dispatch_monotonic_s=dispatched,
                    native_last_token_monotonic_s=last, gpu_reference_acquired_monotonic_s=acquired),
                gpu_reference_evidence=dict(routing_snapshot=dict(kind='ieee_predecision_received_view_v1',
                    selected_replica_id='a', candidates=[candidate],
                    sources={'a': dict(file_captured_monotonic_s=at)}))))
        return rows

    def check(self, rows):
        return routing_load_consistency(rows, clock_id='clock', admitted_edges=[1, 8], delta_ms=1.)

    def test_preparing_and_native_work_are_both_witnessed(self):
        result = self.check(self.requests())
        self.assertEqual(result['requests'], 3)
        self.assertTrue(all(v == 0 for v in result['violations'].values()))
        self.assertEqual([r['witnessed_q_lower_bound'] for r in result['candidates']], [0, 1, 2])
        self.assertEqual(result['candidates'][1]['witnessed_pending_lower_bound'], 1)
        self.assertEqual(result['candidates'][1]['witnessed_predispatch_lower_bound'], 1)
        self.assertEqual(result['candidates'][2]['witnessed_native_lower_bound'], 2)
        self.assertFalse(result['exact_controller_ownership_reconstruction'])

    def test_undercount_pending_and_missing_adapter_are_not_hidden(self):
        rows = self.requests()
        row = rows[1]['gpu_reference_evidence']['routing_snapshot']['candidates'][0]
        row.update(admitted_requests=0, active_adapters=[], pending_loads=0)
        v = self.check(rows)['violations']
        for key in ('q_below_witnessed_lower_bound', 'pending_below_witnessed_lower_bound',
                    'missing_witnessed_active_adapters', 'admitted_bin_mismatch'):
            self.assertEqual(v[key], 1)

    def test_post_token_ownership_is_unknown_not_an_overcount_error(self):
        rows = self.requests()
        rows[2]['gpu_reference_evidence']['routing_snapshot']['sources']['a']['file_captured_monotonic_s'] = 21.
        rows[2]['native_token_timing'].update(controller_admitted_monotonic_s=22.,
            gpu_reference_acquired_monotonic_s=23., native_dispatch_monotonic_s=24.,
            native_last_token_monotonic_s=25.)
        result = self.check(rows)
        self.assertEqual(result['candidates'][2]['witnessed_q_lower_bound'], 0)
        self.assertEqual(result['candidates'][2]['reported_q'], 2)
        self.assertTrue(all(v == 0 for v in result['violations'].values()))

    def test_formula_disagreement_is_detected_independently_of_occupancy(self):
        rows = self.requests()
        snap = rows[2]['gpu_reference_evidence']['routing_snapshot']
        other = copy.deepcopy(snap['candidates'][0])
        other.update(replica_id='b', admitted_requests=0, active_adapters=[], available_slots=8)
        other['service_class']['admitted_bin'] = 0
        snap['candidates'].append(other)
        snap['sources']['b'] = dict(file_captured_monotonic_s=15.5)
        result = self.check(rows)
        self.assertEqual(result['violations']['selection_key_mismatch'], 1)
        self.assertEqual(result['selected_above_min_feasible_q'], 1)

    def test_missing_identity_mixed_clock_and_invalid_observation_rejected(self):
        for case in ('duplicate', 'clock', 'observation', 'source'):
            rows = self.requests()
            if case == 'duplicate': rows.append(rows[0])
            elif case == 'clock': rows[0]['native_token_timing']['native_clock_id'] = 'other'
            elif case == 'observation':
                rows[0]['gpu_reference_evidence']['routing_snapshot']['sources']['a']['file_captured_monotonic_s'] = 11.
            else: rows[0]['gpu_reference_evidence']['routing_snapshot']['sources'] = {}
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.check(rows)

    def test_existing_output_rejected_before_loading_sources(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            inspect_retained_routing_load(root)

    def test_cost_diagnosis_retains_prediction_error_and_no_counterfactual(self):
        candidates = self.check(self.requests())['candidates']
        requests = [dict(request_id=f'r{i}', output_group='2-32', tokens=4, first=10., last=11.) for i in range(3)]
        result = routing_length_diagnosis(candidates, requests)
        self.assertEqual(result['groups'][0]['mean_decode_prediction_error_ms'], -997.)
        self.assertIsNone(result['causal_performance_gain'])
        self.assertTrue(all(r['alternative_performance_known'] is False for r in result['requests']))
        with self.assertRaises(ValueError):
            routing_length_diagnosis(candidates, requests[:2])
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            inspect_retained_routing_lengths(root)


class RetainedLifecycle(unittest.TestCase):
    def physical(self):
        return dict(clock_id='clock', measurement_complete=True, open_lease_ids=[],
            device_intervals={'gpu1': [[10., 20.], [25., 35.]], 'gpu2': [[10., 35.]]}, gpu_seconds=45.)

    def test_physical_gaps_are_measured_not_counterfactual(self):
        result = lifecycle_gaps(self.physical(), dict(clock_id='clock', arrival_start_s=10.))
        self.assertEqual(result['observed_gpu_s'], 45.)
        self.assertEqual(result['released_gap_gpu_s'], 5.)
        self.assertEqual(result['gaps'][0]['released_offset_s'], 10.)
        self.assertEqual(result['gaps'][0]['reallocated_offset_s'], 15.)
        self.assertIsNone(result['causal_savings'])

    def test_incomplete_mixed_clock_overlapping_and_bad_integral_rejected(self):
        for key, value in (('measurement_complete', False), ('clock_id', 'other'),
                ('open_lease_ids', ['open']), ('gpu_seconds', 46.),
                ('device_intervals', {'gpu': [[10., 20.], [19., 25.]]}),
                ('device_intervals', {'gpu': [[10., float('nan')]]})):
            p = self.physical()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                lifecycle_gaps(p, dict(clock_id='clock', arrival_start_s=10.))

    def pair(self):
        row = dict(request_id='a', adapter_id='adapter', prompt_tokens=10, tokens=4,
            canonical_prompt_sha256='prompt', output_group='2-32', arrival_offset_s=10.,
            completion_token_ids_sha256='output', ttft_pass=True, timing_tpot_miss=False,
            timing_joint_pass=True, ttft_ms=2., admission_ms=1., service_ms=1., tpot_ms=1., e2e_s=1.)
        old = dict(requests=[row, dict(row, request_id='b', arrival_offset_s=15.)],
            lifecycle=dict(gaps=[]), activation_offsets=[dict(start_offset_s=-10., end_offset_s=2.)])
        new = copy.deepcopy(old)
        new['lifecycle']['gaps'] = [dict(released_offset_s=10., reallocated_offset_s=15.)]
        new['activation_offsets'].append(dict(start_offset_s=15., end_offset_s=20.))
        new['requests'][0]['completion_token_ids_sha256'] = 'different'
        return dict(d315=old, d318=new)

    def test_common_cohorts_half_open_output_differences_and_empty_na(self):
        result = paired_timeline_groups(self.pair())
        a, b = result['pairs']
        self.assertTrue(a['in_either_release_gap'])
        self.assertFalse(b['in_either_release_gap'])
        self.assertFalse(a['in_either_startup'])
        self.assertTrue(b['in_either_startup'])
        self.assertEqual(result['output_hash_differences'], 1)
        self.assertEqual(result['common_last_ready_offset_s'], 20.)
        empty = [g for g in result['groups'] if g['group']=='after_both_last_ready' and g['label']]
        self.assertTrue(all(g['count']==0 and g['p95_ttft_ms'] is None for g in empty))
        for label in (True, False):
            both = [g for g in result['groups'] if g['group']=='in_either_release_gap' and g['label']==label]
            self.assertEqual([g['count'] for g in both], [1, 1])

    def test_mismatched_population_input_and_arrivals_rejected(self):
        for case in ('duplicate', 'missing', 'adapter', 'arrival'):
            pair = self.pair()
            rows = pair['d318']['requests']
            if case == 'duplicate': rows.append(rows[0])
            elif case == 'missing': rows.pop()
            elif case == 'adapter': rows[0]['adapter_id'] = 'other'
            else: rows[0]['arrival_offset_s'] += .001
            with self.subTest(case=case), self.assertRaises(ValueError):
                paired_timeline_groups(pair)

    def test_existing_output_rejected_before_any_hash_or_analysis(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            inspect_retained_lifecycle(root)


class RegisteredD317Smoke(unittest.TestCase):
    def test_d318_full_audit_has_new_source_and_refuses_existing_output(self):
        from scripts.ieee_tc import analyze_d297_full as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(module, ROOT=root, RAW=None, RUN=None, OUT=None, SOURCE=None), mock.patch.object(
                    module, 'sha256', side_effect=RuntimeError('hash gate reached')):
                with self.assertRaisesRegex(RuntimeError, 'hash gate reached'):
                    module.main('d318')
                self.assertEqual(module.OUT, root/'paper_results/ieee_tc/p2_backend/d318_full_w0/validated1')
                self.assertIn('_r4000_c2_tc_ieee_full_d318_', module.SOURCE.name)
                module.OUT.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    module.main('d318')

    def test_d318_context_is_new_and_reference_cannot_be_changed(self):
        from scripts.ieee_tc import check_d299_full_context as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(module, ROOT=root, RAW=None, RUN=None, OUT=None), mock.patch.object(
                    module.p, 'check_plan', side_effect=RuntimeError('plan gate reached')):
                with self.assertRaisesRegex(RuntimeError, 'plan gate reached'):
                    module.main('d318')
                self.assertEqual(module.OUT, root/'paper_results/ieee_tc/p2_backend/d318_full_w0/context1')
                module.OUT.mkdir(parents=True)
                with self.assertRaises(AssertionError):
                    module.main('d318')
                with self.assertRaises(ValueError):
                    module.main('d318', 'd303')

    def test_d318_full_configuration_changes_only_cold_paths(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        source = (root/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml').read_text()
        expected = yaml.safe_load(source.replace('d315_', 'd318_').replace('tc-d315-', 'tc-d318-'))
        full = yaml.safe_load((root/'configs/ieee_tc/20261007_d318_7b_full_w0.yaml').read_text())
        self.assertEqual(full, expected)
        self.assertEqual(full['workload']['total_requests'], 4000)
        self.assertEqual(full['scenarios'][0]['name'], 'tc_ieee_full')

    def test_d318_explicit_identity_has_independent_no_overwrite_gate(self):
        from scripts.ieee_tc import prepare_d298_capacity as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'paper_results/ieee_tc/p2_backend/d315_restored_full').mkdir(parents=True)
            with mock.patch.object(module.p, 'ROOT', root), self.assertRaises(FileNotFoundError) as error:
                module.verify_restored_full('d318')
            self.assertIn('20261007_d318_7b_full_w0.yaml', str(error.exception))
            (root/'paper_results/ieee_tc/p2_backend/d318_copy_full').mkdir(parents=True)
            with mock.patch.object(module.p, 'ROOT', root), self.assertRaisesRegex(AssertionError, 'never overwrite'):
                module.verify_restored_full('d318')

    def test_smoke_audit_uses_new_output_and_existing_prefix_reference(self):
        from scripts.ieee_tc import analyze_d298_smoke as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(module, ROOT=root, RAW=None, RUN=None, SOURCE=None, REFERENCE=None, OUT=None), mock.patch.object(
                    module.p, 'check_plan', side_effect=RuntimeError('plan gate reached')):
                with self.assertRaisesRegex(RuntimeError, 'plan gate reached'):
                    module.main('d317')
                self.assertEqual(module.OUT, root/'paper_results/ieee_tc/p2_backend/d317_copy_validation/smoke2')
                self.assertIn('d302_20261007', str(module.REFERENCE))
                self.assertIn('_r4000_c2_', module.SOURCE.name)
                module.OUT.mkdir(parents=True)
                with self.assertRaisesRegex(AssertionError, 'never overwrite'):
                    module.main('d317')

    def test_retention_counter_uses_observed_max_not_sum_or_final_events(self):
        from scripts.ieee_tc.analyze_d298_smoke import copy_retention_trigger
        def row(value):
            return dict(gpu_reference_evidence=dict(routing_snapshot=dict(sources={
                'replica': dict(source_collection=dict(identity_frontier_retained_gpu_scopes=value))})))
        result = copy_retention_trigger([row(2), row(7), row(7)])
        self.assertEqual(result['observed_cumulative_retained_scopes_max'], 7)
        self.assertEqual(result['source_snapshot_count'], 3)
        self.assertIsNone(result['exact_rpc_savings'])
        for value in (-1, True, 0.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                copy_retention_trigger([row(value)])
        with self.assertRaises(ValueError): copy_retention_trigger([])
        missing = row(2)
        missing['gpu_reference_evidence']['routing_snapshot']['sources']['replica']['source_collection'].clear()
        with self.assertRaises(KeyError): copy_retention_trigger([missing])

    def test_config_changes_only_names_and_cold_paths(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        old = yaml.safe_load((root/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml').read_text())
        candidate = yaml.safe_load((root/'configs/ieee_tc/20261007_d317_7b_copy_validation_100.yaml').read_text())
        old['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d315_', 'd317_').replace('7b_full_w0', '7b_copy_validation_100')
        old['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d315_', 'd317_').replace('7b_nvme_full_w0', '7b_nvme_copy_validation_100')
        old['storage']['host_cache_dir'] = '/dev/shm/tc-d317-7b-copy-validation-100'
        old['scenarios'][0]['name'] = 'tc_ieee_copy_validation_100'
        self.assertEqual(candidate, old)

    def test_new_identity_does_not_require_old_preparation_absent(self):
        from scripts.ieee_tc import prepare_d298_capacity as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'paper_results/ieee_tc/p2_backend/d315_restored_full').mkdir(parents=True)
            with mock.patch.object(module.p, 'ROOT', root), self.assertRaises(FileNotFoundError) as error:
                module.verify_restored_full('d317')
            self.assertIn('20261007_d317_7b_copy_validation_100.yaml', str(error.exception))

    def test_existing_new_output_and_unknown_identity_rejected(self):
        from scripts.ieee_tc import prepare_d298_capacity as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'paper_results/ieee_tc/p2_backend/d317_copy_validation').mkdir(parents=True)
            with mock.patch.object(module.p, 'ROOT', root):
                with self.assertRaisesRegex(AssertionError, 'never overwrite'):
                    module.verify_restored_full('d317')
                with self.assertRaises(ValueError):
                    module.verify_restored_full('unregistered')


class RegisteredD315Contrasts(unittest.TestCase):
    def test_context_references_have_separate_new_outputs(self):
        from scripts.ieee_tc import check_d299_full_context as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reference, suffix in ((None, 'context1'), ('d303', 'context1'), ('d314', 'context_d314')):
                with self.subTest(reference=reference), mock.patch.multiple(
                        module, ROOT=root, RAW=None, RUN=None, OUT=None), mock.patch.object(
                        module.p, 'check_plan', side_effect=RuntimeError('plan gate reached')):
                    with self.assertRaisesRegex(RuntimeError, 'plan gate reached'):
                        module.main('d315', reference)
                    self.assertEqual(module.OUT, root/f'paper_results/ieee_tc/p2_backend/d315_full_w0/{suffix}')
                    self.assertEqual(module.RUN, root/'results/ieee_tc/p2_backend_qualification/d315_20261007/7b_full_w0')
                    self.assertFalse(module.OUT.exists())

    def test_reference_override_rejects_unregistered_contrasts(self):
        from scripts.ieee_tc import check_d299_full_context as module
        for campaign, reference in (('d303', 'd314'), ('d314', 'd303'), ('d315', 'd300')):
            with self.subTest(campaign=campaign, reference=reference), self.assertRaises(ValueError):
                module.main(campaign, reference)

    def test_existing_d315_output_is_never_overwritten(self):
        from scripts.ieee_tc import check_d299_full_context as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reference, suffix in (('d303', 'context1'), ('d314', 'context_d314')):
                output = root/f'paper_results/ieee_tc/p2_backend/d315_full_w0/{suffix}'
                output.mkdir(parents=True)
                with mock.patch.multiple(module, ROOT=root, RAW=None, RUN=None, OUT=None), self.assertRaises(AssertionError):
                    module.main('d315', reference)


class PreparationAttributionAudit(unittest.TestCase):
    def fixture(self, changed=False):
        from tests.test_ieee_tc_preparation_cost import PreparationIntervals
        from faaslora.preloading.preloading_planner import observed_preparation_interval
        fixture = PreparationIntervals()
        args = fixture.changed_copy_inputs() if changed else fixture.inputs('remote')
        args['admission'].update(preparation_class={'tier': 'remote'}, preparation_profile_id='fixture')
        interval = observed_preparation_interval(**args)
        interval.update(preparation_class={'tier': 'remote'}, preparation_profile_id='fixture',
                        cost_model_updated=interval['profile_eligible'])
        row = dict(request_id='r', adapter_id='a', success=True, gpu_reference_evidence=dict(
            source_admission=args['admission'], receipt=args['native'], remote_preparation=args['remote'],
            local_source_reference=args.get('file_reference'), preparation_interval=interval))
        return row, args['native']['clock_id']

    def test_complete_and_changed_copy_are_recomputed_without_mutation(self):
        for changed in (False, True):
            row, clock = self.fixture(changed)
            before = copy.deepcopy(row)
            result = audit_preparation_attribution([row], clock_id=clock)
            self.assertEqual(row, before)
            self.assertEqual(result['eligible_samples'], int(not changed))
            self.assertEqual(result['ineligible_samples'], int(changed))
            self.assertEqual(len(result['changed_copy_cases']), int(changed))

    def test_fabricated_sample_or_update_is_rejected(self):
        for changed in (False, True):
            for key, value in (('d_ms', 0.), ('reason', 'invented'), ('profile_eligible', None),
                               ('cost_model_updated', None), ('preparation_class', {})):
                row, clock = self.fixture(changed)
                row['gpu_reference_evidence']['preparation_interval'][key] = value
                with self.subTest(changed=changed, key=key), self.assertRaises(ValueError):
                    audit_preparation_attribution([row], clock_id=clock)

    def test_corrupt_copy_and_clock_or_missing_evidence_fail(self):
        for case in ('content', 'clock', 'missing', 'duplicate'):
            row, clock = self.fixture(True)
            if case == 'content':
                row['gpu_reference_evidence']['local_source_reference']['content_verified'] = False
            elif case == 'clock': clock = 'wrong'
            elif case == 'missing': del row['gpu_reference_evidence']['preparation_interval']
            rows = [row, row] if case == 'duplicate' else [row]
            with self.subTest(case=case), self.assertRaises((ValueError, KeyError)):
                audit_preparation_attribution(rows, clock_id=clock)

    def test_gpu_ready_has_no_load_sample_and_empty_is_rejected(self):
        row, clock = self.fixture()
        row['gpu_reference_evidence']['source_admission']['source']['tier'] = 'gpu'
        with self.assertRaises(ValueError): audit_preparation_attribution([row], clock_id=clock)
        del row['gpu_reference_evidence']['preparation_interval']
        result = audit_preparation_attribution([row], clock_id=clock)
        self.assertEqual(result['source_tiers'], {'gpu': 1})
        self.assertEqual(result['eligible_samples'], 0)
        with self.assertRaises(ValueError): audit_preparation_attribution([], clock_id=clock)


class FailureAwarePopulation(unittest.TestCase):
    def test_manifest_does_not_include_its_own_partial_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            for name in ('context.json', 'failed_requests.json'):
                (out/name).write_text('{}')
            expected = {name: hashlib.sha256(b'{}').hexdigest()
                        for name in ('context.json', 'failed_requests.json')}
            for state in ('', '{"partially_written":', '{"complete":true}'):
                (out/'manifest.json').write_text(state)
                self.assertEqual(complete_failure_output_hashes(out), expected)

    def test_manifest_requires_both_failure_evidence_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            (out/'context.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):
                complete_failure_output_hashes(out)

    def test_incomplete_projection_retains_nested_failure_and_excludes_request_tree(self):
        data = dict(run_completed=False,planned_request_count=4000,
            error=dict(type='ConfirmedSourceConflict',message='retained',traceback=['a','b']),
            cleanup_errors=[],measurement_errors=[dict(stage='retained')],
            remote_transfers=[dict(state='published',nested=dict(value=1))],
            completed_scenario_windows=dict(full=dict(requests=[dict(success=True)])))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'outcome.json'
            path.write_text(json.dumps(data))
            self.assertEqual(project_incomplete(path),
                {k:v for k,v in data.items() if k != 'completed_scenario_windows'})
            data['run_completed'] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                project_incomplete(path)

    def fixture(self):
        return dict(metadata=dict(total_requests=2),detailed_results=dict(full=dict(
            completed=1,failed=1,requests=[dict(request_id='a',success=True),
                                        dict(request_id='b',success=False,error='retained')])) )

    def test_partition_does_not_drop_or_impute_failure(self):
        source = self.fixture()
        original = copy.deepcopy(source)
        rows, good, failed = failed_run_partition(source)
        self.assertEqual(source,original)
        self.assertEqual((len(rows),len(good),len(failed)),(2,1,1))
        self.assertIs(failed[0],rows[1])
        self.assertNotIn('readiness_tier_before_dispatch',failed[0])

    def test_invalid_population_is_not_repaired(self):
        for mutation in ('missing','duplicate','unknown','count','no_failure'):
            data = self.fixture()
            s = data['detailed_results']['full']
            if mutation == 'missing': s['requests'].pop()
            elif mutation == 'duplicate': s['requests'][1]['request_id']='a'
            elif mutation == 'unknown': s['requests'][1]['success']=None
            elif mutation == 'count': s['completed']=2
            else:
                s['requests'][1]['success']=True
                s.update(completed=2,failed=0)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                failed_run_partition(data)


class DecodeOverlap(unittest.TestCase):
    def setUp(self):
        self.activations=[dict(activation_id='a',target_replica='a',clock_id='clock',
            category='initial',trigger_at=-2,engine_ready_at=-1),
            dict(activation_id='b',target_replica='b',clock_id='clock',
            category='natural_scaleout',trigger_at=2,engine_ready_at=4)]
        self.thresholds=[dict(lower_exclusive=None,upper_inclusive=None,tpot_s=.6)]
        self.rows=[self.row('1','a',0,1,1,5,5,'gpu'),
                   self.row('2','a',0,2,2,3,3,'host'),
                   self.row('3','b',4,5,5,6,3,'gpu')]

    def row(self,rid,replica,admitted,dispatch,first,last,tokens,tier):
        return dict(request_id=rid,instance_id=replica,success=True,completion_tokens=tokens,
            readiness_tier_before_dispatch=tier,tpot_ms=(last-first)*1000/(tokens-1),
            native_token_timing=dict(controller_admitted_monotonic_s=admitted,
                native_dispatch_monotonic_s=dispatch,native_first_token_monotonic_s=first,
                native_last_token_monotonic_s=last,native_clock_id='clock',actual_prompt_tokens=10))

    def run_summary(self,rows=None,activations=None):
        return decode_overlap_summary(self.rows if rows is None else rows,
            self.activations if activations is None else activations,
            clock_id='clock',arrival_start=0,thresholds=self.thresholds)

    def test_overlap_excludes_self_and_keeps_native_time(self):
        result=self.run_summary()
        row=result['requests'][0]
        self.assertEqual(row['other_startup_overlap_s'],2)
        self.assertEqual(row['same_replica_preparation_envelope_s'],1)
        self.assertEqual(row['mean_other_native_requests'],.25)
        self.assertEqual(row['tpot_ms'],1000)
        self.assertTrue(row['timing_tpot_miss'])
        self.assertEqual(result['startup_union_s'],2)
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_overlapping_startups_count_union_not_sum(self):
        other=dict(activation_id='c',target_replica='c',clock_id='clock',
            category='natural_scaleout',trigger_at=3,engine_ready_at=4.5)
        result=self.run_summary(activations=self.activations+[other])
        self.assertEqual(result['requests'][0]['other_startup_overlap_s'],2.5)

    def test_endpoint_touch_is_not_exposure(self):
        r=self.row('4','a',0,1,1,2,3,'gpu')
        self.assertFalse(self.run_summary(rows=[r])['requests'][0]['startup_overlap'])

    def test_bad_identity_clock_or_timing_rejected(self):
        for field,value in [('native_clock_id','other'),('native_first_token_monotonic_s',6),
                            ('native_dispatch_monotonic_s',-3)]:
            rows=copy.deepcopy(self.rows)
            rows[0]['native_token_timing'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.run_summary(rows=rows)
        with self.assertRaises(ValueError):
            self.run_summary(rows=self.rows+[self.rows[0]])

    def test_empty_groups_remain_unknown_not_zero_latency(self):
        result=self.run_summary(rows=self.rows[:1])
        empty=[g for g in result['groups'] if g['group']=='output_group' and g['label']=='129-256'][0]
        self.assertEqual(empty['count'],0)
        self.assertIsNone(empty['mean_tpot_ms'])


class NativeLoadOverlap(unittest.TestCase):
    def setUp(self):
        fixture = DecodeOverlap()
        fixture.setUp()
        self.rows = fixture.rows
        self.records = fixture.run_summary()['requests']
        for i, r in enumerate(self.rows):
            n = r['native_token_timing']
            loaded = r['readiness_tier_before_dispatch'] != 'gpu'
            acquired = n['native_dispatch_monotonic_s'] - .1
            owner, lease = f"owner-{r['instance_id']}", f'lease-{i}'
            n.update(gpu_reference_owner_id=owner, gpu_reference_lease_id=lease)
            r['gpu_reference_evidence'] = dict(receipt=dict(
                acquired=True, owner_id=owner, lease_id=lease, worker_pid=123 if i<2 else 124,
                worker_rank=0, clock_id='clock', acquisition_operation='demand_load_and_acquire',
                completion_fence_scope='current_worker_cuda_stream',
                acquired_monotonic_s=acquired, completion_fence_ms=100.,
                native_load_invoked=loaded, gpu_resident_before_load=not loaded,
                source_tier_before_acquisition='host' if loaded else 'gpu',
                native_load_started_monotonic_s=1.1 if loaded else None,
                native_load_completed_monotonic_s=acquired if loaded else None,
                load_and_acquire_ms=900. if loaded else 100.))

    def analyze(self):
        return native_load_overlap_summary(self.rows, self.records, clock_id='clock')

    def test_narrow_load_excludes_wait_self_and_other_owner(self):
        result = self.analyze()
        a, b, c = result['requests']
        self.assertAlmostEqual(a['load_overlap_s'], .8)
        self.assertAlmostEqual(a['load_body_overlap_s'], .7)
        self.assertAlmostEqual(a['fence_overlap_s'], .1)
        self.assertAlmostEqual(a['load_decode_fraction'], .2)
        self.assertFalse(b['load_overlap'])
        self.assertFalse(c['load_overlap'])
        self.assertEqual(result['native_load_count'], 1)
        self.assertEqual(result['native_owner_count'], 2)
        self.assertIsNone(a['load_start'])
        self.assertIsNone(a['native_load_ms'])
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_endpoint_touch_excluded(self):
        e = self.rows[1]['gpu_reference_evidence']['receipt']
        e.update(acquired_monotonic_s=1., native_load_started_monotonic_s=.5,
                 native_load_completed_monotonic_s=1.)
        self.assertFalse(self.analyze()['requests'][0]['load_overlap'])

    def test_gpu_protection_before_controller_admission_is_legal(self):
        self.rows[0]['native_token_timing']['controller_admitted_monotonic_s'] = .95
        self.records[0]['admitted'] = .95
        result = self.analyze()
        self.assertEqual(result['requests'][0]['acquired'], .9)
        self.assertFalse(result['requests'][0]['native_load_invoked'])

    def test_no_load_or_no_group_latency_is_not_zero_measurement(self):
        e = self.rows[1]['gpu_reference_evidence']['receipt']
        e.update(native_load_invoked=False, gpu_resident_before_load=True,
            source_tier_before_acquisition='gpu', native_load_started_monotonic_s=None,
            native_load_completed_monotonic_s=None)
        result = self.analyze()
        empty = next(g for g in result['groups'] if g['group']=='load_overlap' and g['label'] is True)
        self.assertEqual(empty['count'], 0)
        self.assertIsNone(empty['mean_tpot_ms'])
        self.assertTrue(result['requests'][0]['fence_overlap'])

    def test_malformed_receipts_rejected(self):
        original = copy.deepcopy(self.rows)
        for key, value in [('clock_id','other'), ('owner_id','unbound'), ('worker_pid',0),
                           ('native_load_completed_monotonic_s',1.8),
                           ('native_load_started_monotonic_s',1.85),
                           ('completion_fence_ms',float('nan')), ('completion_fence_ms',-1),
                           ('native_load_invoked',1), ('gpu_resident_before_load',True),
                           ('acquired_monotonic_s',3), ('load_and_acquire_ms',500.)]:
            self.rows = copy.deepcopy(original)
            self.rows[1]['gpu_reference_evidence']['receipt'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.analyze()
        self.rows = copy.deepcopy(original)
        self.rows[0]['gpu_reference_evidence']['receipt']['native_load_started_monotonic_s'] = 0.
        with self.assertRaises(ValueError):
            self.analyze()

    def test_retained_mapping_and_worker_incarnation_strict(self):
        self.rows[1]['gpu_reference_evidence']['receipt']['worker_pid'] = 999
        with self.assertRaises(ValueError):
            self.analyze()
        self.rows[1]['gpu_reference_evidence']['receipt']['worker_pid'] = 123
        self.records[0]['first'] += .1
        with self.assertRaises(ValueError):
            self.analyze()

    def test_duplicate_rows_or_leases_rejected(self):
        with self.assertRaises(ValueError):
            native_load_overlap_summary(self.rows+[self.rows[0]], self.records, clock_id='clock')
        self.rows[1]['gpu_reference_evidence']['receipt']['lease_id'] = 'lease-0'
        self.rows[1]['native_token_timing']['gpu_reference_lease_id'] = 'lease-0'
        with self.assertRaises(ValueError):
            self.analyze()


class PrefillOverlap(unittest.TestCase):
    def setUp(self):
        fixture=NativeLoadOverlap()
        fixture.setUp()
        self.rows=fixture.rows
        self.records=fixture.analyze()['requests']
        for i,(q,r) in enumerate(zip(self.rows,self.records)):
            n=q['native_token_timing']
            n['native_queued_monotonic_s']=n['native_dispatch_monotonic_s']
            n['native_scheduled_monotonic_s']=n['native_dispatch_monotonic_s']
            if i>0:
                n['native_first_token_monotonic_s']+=.5
                r['first']+=.5
                r['tpot_ms']=(r['last']-r['first'])*1000/(r['tokens']-1)

    def analyze(self):
        return prefill_overlap_summary(self.rows,self.records,clock_id='clock')

    def test_prefill_not_load_and_not_other_owner(self):
        result=self.analyze()
        a,b,c=result['requests']
        self.assertEqual(a['prefill_overlap_s'],.5)
        self.assertEqual(a['prefill_decode_fraction'],.125)
        self.assertEqual(a['peer_prefill_count'],1)
        self.assertFalse(b['prefill_overlap'])
        self.assertFalse(c['prefill_overlap'])
        self.assertEqual(b['prefill_envelope_ms'],500.)
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_union_not_sum(self):
        q,r=copy.deepcopy(self.rows[1]),copy.deepcopy(self.records[1])
        q['request_id']=r['request_id']='4'
        q['native_token_timing'].update(native_dispatch_monotonic_s=2.2,
            native_queued_monotonic_s=2.2,native_scheduled_monotonic_s=2.2,
            native_first_token_monotonic_s=2.7,native_last_token_monotonic_s=3.2)
        r.update(dispatched=2.2,first=2.7,last=3.2)
        self.rows.append(q)
        self.records.append(r)
        result=self.analyze()['requests'][0]
        self.assertAlmostEqual(result['prefill_overlap_s'],.7)
        self.assertEqual(result['peer_prefill_count'],2)

    def test_touching_endpoint_has_no_overlap(self):
        self.rows[0]['native_token_timing']['native_last_token_monotonic_s']=2.
        self.records[0].update(last=2.,tpot_ms=250.)
        self.assertFalse(self.analyze()['requests'][0]['prefill_overlap'])

    def test_wrong_clock_order_and_identity_rejected(self):
        original=copy.deepcopy(self.rows)
        for key,value in [('native_scheduled_monotonic_s',3.),
                          ('native_queued_monotonic_s',float('nan')),
                          ('native_clock_id','other'),('gpu_reference_owner_id','other')]:
            self.rows=copy.deepcopy(original)
            self.rows[1]['native_token_timing'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.analyze()
        self.rows=copy.deepcopy(original)
        self.records[0]['timing_tpot_miss']='False'
        with self.assertRaises(ValueError):
            self.analyze()

    def test_empty_group_latency_unknown(self):
        self.rows=self.rows[:1]
        self.records=self.records[:1]
        result=self.analyze()
        empty=next(g for g in result['groups'] if g['group']=='prefill_overlap' and g['label'] is True)
        self.assertEqual(empty['count'],0)
        self.assertIsNone(empty['mean_tpot_ms'])


if __name__=='__main__':
    unittest.main()
