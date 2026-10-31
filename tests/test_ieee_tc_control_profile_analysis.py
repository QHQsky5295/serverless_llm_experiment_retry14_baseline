"""Offline counter audit fixtures, not performance or numerical qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ieee_tc.export_d298_profiles import (
    audit_control_interleave_snapshots, audit_control_source_run, audit_file_demand_source_run,
    audit_metadata_source_run, audit_measurement_seal, p)


class MeasurementSealIsolation(unittest.TestCase):
    """File identity fixtures only; not measured timing or model evidence."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.prepared, self.measured = [self.root/name for name in
                                                ('raw', 'prepared', 'measured')]
        for path in (self.raw, self.prepared, self.measured):
            path.mkdir()
        self.source = self.raw/'source1.json'
        self.source.write_text('{"fixture": true}\n')
        (self.prepared/'source_spec.json').write_text('{"unchanged_input": true}\n')
        (self.raw/'source_commit.txt').write_text('a'*40+'\n')
        self.checkpoint = dict(passed=True, measurement_complete=True,
            source_run=dict(path=str(self.source), sha256=p.digest(self.source)),
            source_execution_commit='a'*40,
            source_refs=[dict(path=str(self.prepared/'source_spec.json'),
                             sha256=p.digest(self.prepared/'source_spec.json'))])
        self.seal()

    def seal(self, directory=None):
        directory = self.measured if directory is None else directory
        path = directory/'measurement_checkpoint.json'
        path.write_text(json.dumps(self.checkpoint))
        (directory/'measurement_outputs.sha256').write_text(
            p.digest(path)+'  '+path.name+'\n')

    def test_new_measurement_does_not_require_writing_reused_input_directory(self):
        before = {x.name:x.read_bytes() for x in self.prepared.iterdir()}
        paths = audit_measurement_seal(self.raw, self.measured, self.source)
        self.assertEqual(paths[0].parent, self.measured)
        self.assertEqual(before, {x.name:x.read_bytes() for x in self.prepared.iterdir()})
        with self.assertRaises(FileNotFoundError):
            audit_measurement_seal(self.raw, self.prepared, self.source)

    def test_legacy_colocated_seal_remains_valid(self):
        self.seal(self.prepared)
        audit_measurement_seal(self.raw, self.prepared, self.source)

    def test_foreign_run_rejected_even_if_bytes_match(self):
        old = self.root/'old_source.json'
        old.write_bytes(self.source.read_bytes())
        self.checkpoint['source_run']['path'] = str(old)
        self.seal()
        with self.assertRaises(AssertionError):
            audit_measurement_seal(self.raw, self.measured, self.source)

    def test_failed_incomplete_or_other_commit_rejected(self):
        original = copy.deepcopy(self.checkpoint)
        for patch in (dict(passed=False), dict(measurement_complete=False),
                      dict(source_execution_commit='b'*40)):
            with self.subTest(patch=patch):
                self.checkpoint = dict(original, **patch)
                self.seal()
                with self.assertRaises(AssertionError):
                    audit_measurement_seal(self.raw, self.measured, self.source)

    def test_tampered_source_reference_and_seal_rejected(self):
        for path in (self.source, self.prepared/'source_spec.json',
                     self.measured/'measurement_checkpoint.json'):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaises(AssertionError):
                    audit_measurement_seal(self.raw, self.measured, self.source)
                path.write_bytes(original)

    def test_seal_cannot_resolve_members_outside_measurement_directory(self):
        (self.measured/'measurement_outputs.sha256').write_text(
            p.digest(self.source)+'  ../raw/source1.json\n')
        with self.assertRaises(AssertionError):
            audit_measurement_seal(self.raw, self.measured, self.source)


class MetadataProfileInstallationAudit(unittest.TestCase):
    def fixture(self):
        row = dict(policy='cpu_live_mapping_v1', installed=True, poisoned=False,
            banks=2, rows=1024, slots=14, staging_bytes=99040, updates=0,
            reuse_waits=0, copied_bytes=0, gpu_buffers_replaced=False,
            native_mapping_invalidation_preserved=True)
        initial = dict(row, accounted_before_bytes=1000, accounted_after_bytes=100040,
            required_minimum_with_staging_bytes=130040)
        final = dict(row, updates=40, copied_bytes=4000)
        identity = dict(owner_id='worker-owner', worker_pid=123, clock_id='native-clock')
        return dict(model_config=dict(ieee_lora_metadata_policy='cpu_live_mapping_v1',
                max_num_batched_tokens=1024, max_loras=14, ieee_native_host_tensor_budget_bytes=200000),
            metadata_host_installation=dict(identity, configured=True, tensor_budget_bytes=200000,
                metadata_publication=initial,
                workspace_partition=dict(registered_upper_bytes=10000, demand_load_peak_bytes=10000),
                allocation=dict(admitted=True, before=dict(accounted_tensor_bytes=100040))),
            metadata_global_host_budget=dict(total_bytes=1000000),
            workers_after=dict(workers=[dict(pid=123, clock_id='native-clock', metadata_publication=dict(final))]),
            sources_after=dict(identity, metadata_publication=dict(final)),
            sources_after_eviction=dict(identity, metadata_publication=dict(final)))

    def test_actual_receipt_counters_and_accounted_staging(self):
        raw = self.fixture()
        before = copy.deepcopy(raw)
        result = audit_metadata_source_run(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result['actual_pinned_increment'], 99040)
        self.assertTrue(result['triggered'])
        self.assertFalse(result['numerical_correctness_qualified'])
        self.assertIsNone(result['causal_speedup'])

    def test_missing_wrong_identity_accounting_or_inactive_publisher_rejected(self):
        mutations = [
            lambda x: x['model_config'].update(ieee_lora_metadata_policy='other'),
            lambda x: x['metadata_host_installation'].update(configured=False),
            lambda x: x['metadata_host_installation'].update(worker_pid=124),
            lambda x: x['metadata_host_installation']['metadata_publication'].update(accounted_after_bytes=1000),
            lambda x: x['metadata_host_installation']['metadata_publication'].update(required_minimum_with_staging_bytes=0),
            lambda x: x['metadata_host_installation']['allocation'].update(admitted=False),
            lambda x: x['metadata_host_installation']['workspace_partition'].update(demand_load_peak_bytes=1000000),
            lambda x: x['sources_after'].update(owner_id='foreign'),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(updates=39),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(installed=False),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(poisoned=True),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(staging_bytes=0),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(updates=True),
            lambda x: x['sources_after_eviction']['metadata_publication'].update(gpu_buffers_replaced=True),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = self.fixture()
                mutation(raw)
                with self.assertRaises(ValueError):
                    audit_metadata_source_run(raw)
        raw = self.fixture()
        for row in (raw['workers_after']['workers'][0], raw['sources_after'], raw['sources_after_eviction']):
            row['metadata_publication'].update(updates=0, copied_bytes=0)
        with self.assertRaises(ValueError):
            audit_metadata_source_run(raw)


def snapshot(stamp, turns, controls, yields, queued):
    return dict(kind='ieee_native_scheduler_observation_v1', scheduler_pid=123,
        clock_id='native-clock', captured_at=stamp,
        control_interleave=dict(kind='ieee_native_control_interleave_v1',
            policy='atomic_control_yield_v1', input_turns=turns,
            handled_controls=controls, yields_to_native_loop=yields,
            yields_with_queued_input=queued))


class ControlProfileCounterAudit(unittest.TestCase):
    def observations(self):
        # Idle turns can dispatch many controls. Controls need not <= turns.
        return [('start', snapshot(1., 10, 20, 2, 1)),
                ('finish', snapshot(2., 15, 30, 5, 2))]

    def audit(self, observations):
        return audit_control_interleave_snapshots(observations,
            expected_pid=123, expected_clock='native-clock')

    def test_measured_differences_without_causal_claim(self):
        original=self.observations(); retained=copy.deepcopy(original)
        result=self.audit(original)
        self.assertEqual(original, retained)
        self.assertEqual(result['delta'], dict(input_turns=5, handled_controls=10,
            yields_to_native_loop=3, yields_with_queued_input=1))
        self.assertTrue(result['triggered'])
        self.assertTrue(result['queued_input_triggered'])
        for key in ('gpu_idle_seconds', 'causal_speedup', 'model_step_count'):
            self.assertIsNone(result[key])

    def test_no_trigger_is_retained_not_filled(self):
        values=self.observations()
        values[-1][1]['control_interleave'].update(yields_to_native_loop=2,
                                                  yields_with_queued_input=1)
        result=self.audit(values)
        self.assertFalse(result['triggered'])
        self.assertFalse(result['queued_input_triggered'])

    def test_missing_wrong_owner_and_policy_rejected(self):
        mutations=[lambda x:x.pop('control_interleave'),
            lambda x:x.update(scheduler_pid=124),
            lambda x:x.update(clock_id='another-clock'),
            lambda x:x.update(kind='proxy'),
            lambda x:x['control_interleave'].update(kind='proxy'),
            lambda x:x['control_interleave'].update(policy='last-known'),
            lambda x:x['control_interleave'].pop('handled_controls')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                values=self.observations();mutation(values[-1][1])
                with self.assertRaises(ValueError):self.audit(values)

    def test_noninteger_negative_nonfinite_and_counter_subsets_rejected(self):
        for field in ('input_turns', 'handled_controls', 'yields_to_native_loop',
                      'yields_with_queued_input'):
            for value in (-1, True, 1.5, None):
                with self.subTest(field=field,value=value):
                    values=self.observations()
                    values[-1][1]['control_interleave'][field]=value
                    with self.assertRaises(ValueError):self.audit(values)
        for patch in (dict(yields_with_queued_input=6),dict(handled_controls=4),
                      dict(input_turns=4)):
            values=self.observations();values[-1][1]['control_interleave'].update(patch)
            with self.assertRaises(ValueError):self.audit(values)
        for stamp in (float('nan'),float('inf'),-1,True):
            values=self.observations();values[-1][1]['captured_at']=stamp
            with self.assertRaises(ValueError):self.audit(values)

    def test_reset_reversed_time_and_incomplete_sequence_rejected(self):
        values=self.observations();values[-1][1]['control_interleave']['handled_controls']=19
        with self.assertRaises(ValueError):self.audit(values)
        for stamp in (1., .5):
            values=self.observations();values[-1][1]['captured_at']=stamp
            with self.assertRaises(ValueError):self.audit(values)
        with self.assertRaises(ValueError):self.audit(self.observations()[:1])

    def source(self):
        return dict(model_config=dict(ieee_control_interleave='atomic_control_yield_v1'),
            workers_before=dict(workers=[dict(pid=123,clock_id='native-clock')]),
            scheduler_before=snapshot(1.,10,20,2,1),
            source_profile_spec=dict(waves=[dict(role='kernel_warmup_retained'),
                                            dict(role='representative_measurement')]),
            profile_waves=[dict(wave=0,complete=True,
                capacity_before=snapshot(2.,11,21,2,1),
                capacity_after=snapshot(3.,12,22,2,1)),
                dict(wave=1,complete=True,
                capacity_before=snapshot(4.,13,23,2,1),
                capacity_after=snapshot(5.,15,26,4,2))],
            scheduler_after=snapshot(6.,16,27,4,2))

    def test_all_waves_and_zero_trigger_warmup_retained(self):
        result=audit_control_source_run(self.source())
        self.assertEqual(len(result['observations']),6)
        self.assertEqual(result['zero_yield_waves'],[0])
        self.assertEqual(result['waves'][1]['delta']['yields_to_native_loop'],2)

    def test_missing_reordered_incomplete_or_foreign_wave_rejected(self):
        mutations=[lambda x:x['profile_waves'].pop(),
            lambda x:x['profile_waves'].reverse(),
            lambda x:x['profile_waves'][0].update(complete=False),
            lambda x:x['profile_waves'][0]['capacity_before'].update(scheduler_pid=124),
            lambda x:x['model_config'].clear(),
            lambda x:x['workers_before']['workers'].append(dict(pid=124,clock_id='native-clock'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                source=self.source();mutation(source)
                with self.assertRaises(ValueError):audit_control_source_run(source)

    def endpoint_source(self):
        source = self.source()
        for wave in source['profile_waves']:
            del wave['capacity_before'], wave['capacity_after']
        return source

    def test_explicit_endpoints_keep_missing_wave_attribution_unknown(self):
        source = self.endpoint_source()
        original = copy.deepcopy(source)
        result = audit_control_source_run(source, observation_scope='source_run_endpoints')
        self.assertEqual(source, original)
        self.assertEqual(len(result['observations']), 2)
        self.assertEqual(result['complete_wave_count'], 2)
        self.assertEqual(result['delta']['yields_to_native_loop'], 2)
        self.assertTrue(result['triggered'])
        self.assertFalse(result['per_wave_counters_available'])
        self.assertIsNone(result['waves'])
        self.assertIsNone(result['zero_yield_waves'])
        self.assertIsNone(result['causal_speedup'])

    def test_missing_wave_data_never_implicitly_selects_endpoint_audit(self):
        with self.assertRaises(ValueError):
            audit_control_source_run(self.endpoint_source())
        with self.assertRaises(ValueError):
            audit_control_source_run(self.source(), observation_scope='source_run_endpoints')
        source = self.endpoint_source()
        source['source_profile_spec']['capacity_probe_contract'] = {}
        with self.assertRaises(ValueError):
            audit_control_source_run(source, observation_scope='source_run_endpoints')
        with self.assertRaises(ValueError):
            audit_control_source_run(source, observation_scope='unrecognized')

    def test_endpoint_audit_still_rejects_incomplete_and_wrong_owner(self):
        for mutate in (
                lambda x: x['profile_waves'].pop(),
                lambda x: x['profile_waves'].reverse(),
                lambda x: x['profile_waves'][0].update(complete=False),
                lambda x: x['scheduler_after'].update(scheduler_pid=124),
                lambda x: x['scheduler_after'].update(clock_id='foreign'),
                lambda x: x['scheduler_after']['control_interleave'].update(handled_controls=-1),
                lambda x: x['scheduler_after'].update(captured_at=0.5)):
            with self.subTest(mutation=mutate):
                source = self.endpoint_source()
                mutate(source)
                with self.assertRaises(ValueError):
                    audit_control_source_run(source, observation_scope='source_run_endpoints')


class FileDemandProfileAudit(unittest.TestCase):
    def source(self):
        policy='protected_file_owner_turn_v1'
        rows=[]
        sources=('remote','nvme','file_host','native_host','gpu')
        for i,source in enumerate(sources):
            aid=f'a{i}'
            intent=dict(lease_id=f'native-{i}',expected_owner_id='native-owner',
                expected_epoch=10,lora_path=f'/owned/{aid}',expected_content_sha256='a'*64)
            native=dict(lease_id=intent['lease_id'],owner_id='native-owner',lora_name=aid,
                acquired=True,epoch=13,expected_content_sha256='a'*64)
            evidence=dict(state='released',receipt=native,intent=intent,release_receipt=dict(released=True))
            if source in ('remote','nvme','file_host'):
                witness=dict(kind=policy,owner_id='file-owner',lease_id=f'file-{i}',adapter_id=aid,
                    path=intent['lora_path'],tier='host' if source=='file_host' else 'nvme',
                    device=1,inode=i+1,content_sha256='a'*64)
                native['file_source_witness']=copy.deepcopy(witness)
                intent['file_source_witness']=copy.deepcopy(witness)
                evidence['local_source_reference']=dict(witness,content_verified=True)
                native['file_demand_transaction']=dict(kind=policy,observed_epoch=10,
                    validated_epoch=12,owner_epoch_advance=2,live_native_constraints_checked=True,
                    controller_capacity_snapshot_reused=False)
            rows.append(dict(request_id=f'source-profile/w{i}/l0',adapter_id=aid,
                requested_source=source,source_evidence=evidence,**{'pass':True}))
        return dict(model_config=dict(ieee_file_demand_transaction=policy),requests=rows,
            source_profile_spec=dict(waves=[dict(source=x,requests=[{}]) for x in sources]))

    def test_exact_file_and_native_paths_are_separate(self):
        raw=self.source();before=copy.deepcopy(raw)
        result=audit_file_demand_source_run(raw)
        self.assertEqual(raw,before)
        self.assertEqual(result['request_count'],5)
        self.assertEqual(result['file_transactions'],3)
        self.assertEqual(result['advanced_owner_epoch_transactions'],3)
        self.assertEqual(result['max_owner_epoch_advance'],2)
        self.assertIsNone(result['n_correct'])
        self.assertIsNone(result['causal_speedup'])

    def test_missing_invalid_or_foreign_bindings_rejected(self):
        mutations=[lambda r:r['source_evidence']['receipt'].pop('file_source_witness'),
            lambda r:r['source_evidence']['receipt']['file_source_witness'].update(lease_id='foreign'),
            lambda r:r['source_evidence']['local_source_reference'].update(inode=999),
            lambda r:r['source_evidence']['local_source_reference'].update(content_verified=False),
            lambda r:r['source_evidence']['receipt']['file_demand_transaction'].update(validated_epoch=9),
            lambda r:r['source_evidence']['receipt']['file_demand_transaction'].update(validated_epoch=14),
            lambda r:r['source_evidence']['receipt']['file_demand_transaction'].update(owner_epoch_advance=True),
            lambda r:r['source_evidence']['receipt']['file_demand_transaction'].update(live_native_constraints_checked=False),
            lambda r:r['source_evidence']['receipt']['file_demand_transaction'].update(controller_capacity_snapshot_reused=True),
            lambda r:r['source_evidence']['release_receipt'].update(released=False),
            lambda r:r.update(requested_source='gpu'),lambda r:r.update(**{'pass':False})]
        for mutate in mutations:
            raw=self.source();mutate(raw['requests'][0])
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                audit_file_demand_source_run(raw)
        for mutate in (lambda x:x['requests'].pop(),lambda x:x['requests'].append(x['requests'][0]),
                       lambda x:x['model_config'].clear(),
                       lambda x:x['requests'][-1]['source_evidence']['intent'].update(file_source_witness={})):
            raw=self.source();mutate(raw)
            with self.assertRaises(ValueError):audit_file_demand_source_run(raw)


class RetainedControlAnalysisInputs(unittest.TestCase):
    def test_explicit_attempt_paths_are_not_selected_by_recency(self):
        from scripts.ieee_tc.analyze_d455_stages import retained_control_paths
        source = Path('/runs/key/result.json')
        default = retained_control_paths(source)
        self.assertEqual(default[0], Path('/runs/key/launch.json'))
        explicit = retained_control_paths(source, Path('/runs/key/launch_retry5.json'))
        self.assertEqual(explicit[1], Path(
            '/runs/key/launch_retry5.launch/physical_deployment/summary.json'))
        self.assertEqual(explicit[2], explicit[1].parent/'main_outcome.json')

    def test_explicit_attempt_count_matches_sources(self):
        from scripts.ieee_tc.analyze_d455_stages import analyze_retained_controls
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'match source count'):
                analyze_retained_controls(Path(root)/'new', [Path('missing')], launches=[])

    def test_frozen_group_boundaries_are_model_specific(self):
        from scripts.ieee_tc.analyze_d455_stages import (
            retained_control_reference, retained_threshold_group)
        for model, edge in [('7b', 616), ('3b', 506)]:
            with self.subTest(model=model):
                threshold, content = retained_control_reference(model)
                groups = json.loads(threshold.read_text())['groups']
                self.assertEqual(retained_threshold_group(groups, edge), groups[0])
                self.assertEqual(retained_threshold_group(groups, edge+1), groups[1])
                self.assertIn(model, content.name)
        with self.assertRaisesRegex(ValueError, 'model must be'):
            retained_control_reference('13b')

    def test_missing_or_ambiguous_group_is_not_defaulted(self):
        from scripts.ieee_tc.analyze_d455_stages import retained_threshold_group
        group = dict(lower_exclusive=None, upper_inclusive=506)
        for groups, n in [([], 10), ([group, group], 10), ([group], 507),
                          ([group], float('nan')), ([group], 0)]:
            with self.subTest(groups=groups, n=n), self.assertRaises(ValueError):
                retained_threshold_group(groups, n)

    def test_existing_output_is_not_overwritten(self):
        from scripts.ieee_tc.analyze_d455_stages import analyze_retained_controls
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            marker = output/'existing.json'
            marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'new output'):
                analyze_retained_controls(output, [Path('missing-source')])
            self.assertEqual(marker.read_text(), 'preserve')

    def test_empty_or_duplicate_sources_are_rejected_before_reading(self):
        from scripts.ieee_tc.analyze_d455_stages import analyze_retained_controls
        with tempfile.TemporaryDirectory() as root:
            for sources in ([], [Path('same'), Path('same')]):
                with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, 'unique explicit'):
                    analyze_retained_controls(Path(root)/'new', sources)
            self.assertFalse((Path(root)/'new').exists())

    def test_changed_protocol_cannot_be_silently_used(self):
        from unittest.mock import patch
        from scripts.ieee_tc.analyze_d455_stages import analyze_retained_controls
        with tempfile.TemporaryDirectory() as root, patch(
                'scripts.ieee_tc.analyze_d455_stages.sha256', return_value='different'):
            with self.assertRaisesRegex(ValueError, 'frozen metric protocol'):
                analyze_retained_controls(Path(root)/'new', [Path('missing-source')])
            self.assertFalse((Path(root)/'new').exists())


class ControlBusyBounds(unittest.TestCase):
    @staticmethod
    def control(now, sampled=0., elapsed=10., action='no_action'):
        return dict(observed_at=now, action=action, reason='test', score=.2,
            outcome='drained_replica_retired' if action == 'scale_down' else 'no_action',
            target_replica='a', predictive_target_instances=1,
            predictive_observation=dict(history_elapsed_s=elapsed, busy_instance_seconds=sampled))

    @staticmethod
    def request(rid, a, b, replica='a', owner='owner-a'):
        return dict(request_id=rid, admitted=a, last=b, replica=replica, owner=owner)

    def test_owner_union_and_withdrawal_not_double_counted(self):
        from scripts.ieee_tc.analyze_d455_stages import control_busy_time_bounds
        c = self.control(5., action='scale_down')
        c['predictive_observation'] = None
        rows = [self.request('1', 0., 8.), self.request('2', 3., 6.),
                self.request('3', 2., 4., 'b', 'owner-b')]
        result = control_busy_time_bounds([c, self.control(10., 4.)], rows)
        self.assertEqual(result['ratio_quantiles']['0.5'], 4/7)
        self.assertEqual(result['owner_union_intervals'], 2)
        self.assertEqual(result['windows_below_90pct_of_lower_bound'], 1)

    def test_window_clipping_and_zero_busy_unknown_ratio(self):
        from scripts.ieee_tc.analyze_d455_stages import control_busy_time_bounds
        result = control_busy_time_bounds([self.control(10., 2., 4.)],
                                         [self.request('1', 0., 8.)])
        self.assertEqual(result['ratio_quantiles']['0.5'], 1.)
        result = control_busy_time_bounds([self.control(20., 0., 4.)],
                                         [self.request('1', 0., 8.)])
        self.assertIsNone(result['ratio_quantiles']['0.5'])

    def test_repeated_replica_labels_merge_by_native_owner(self):
        from scripts.ieee_tc.analyze_d455_stages import control_busy_time_bounds
        result = control_busy_time_bounds([self.control(10., 6.)],
            [self.request('1', 0., 5.), self.request('2', 1., 6., 'alias')])
        self.assertEqual(result['owner_union_intervals'], 1)
        self.assertEqual(result['ratio_quantiles']['0.5'], 1.)

    def test_invalid_intervals_and_duplicate_requests_fail(self):
        from scripts.ieee_tc.analyze_d455_stages import control_busy_time_bounds
        for rows in ([self.request('1', 3., 2.)], [self.request('1', 0., float('nan'))],
                     [self.request('1', 0., 1.)]*2):
            with self.assertRaises(ValueError):
                control_busy_time_bounds([self.control(10.)], rows)

    def test_incomplete_or_nonmonotonic_control_is_not_exact_release(self):
        from scripts.ieee_tc.analyze_d455_stages import control_busy_time_bounds
        bad = self.control(10., action='scale_down');bad['outcome'] = 'draining_admitted_requests'
        for controls in ([bad], [self.control(10.), self.control(10.)]):
            with self.assertRaises(ValueError):
                control_busy_time_bounds(controls, [self.request('1', 0., 1.)])

    def test_existing_output_refused_without_reading_sources(self):
        from scripts.ieee_tc.analyze_d455_stages import audit_control_busy_windows
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, 'new output'):
            audit_control_busy_windows(Path(root), [Path('absent')])


class SealedCsvEncoding(unittest.TestCase):
    def test_zero_cohort_keeps_schema_but_missing_schema_does_not_create_a_file(self):
        import csv
        from scripts.ieee_tc.audit_d703_3b_full import write_exclusive_csv
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'empty.csv'
            with self.assertRaises(ValueError):
                write_exclusive_csv(path, [])
            self.assertFalse(path.exists())
            write_exclusive_csv(path, [], fieldnames=('request_id', 'latency_ms'))
            self.assertEqual(path.read_bytes(), b'request_id,latency_ms\n')
            with path.open() as h:
                self.assertEqual(list(csv.DictReader(h)), [])

    def test_lf_bytes_round_trip_without_post_hash_normalization(self):
        import csv
        import hashlib
        from scripts.ieee_tc.audit_d703_3b_full import write_exclusive_csv, ref
        rows = [dict(request_id='请求,1', value='quoted "value"\nnext', passed=True),
                dict(request_id='req_2', value='1.2345', passed=False)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'requests.csv'
            write_exclusive_csv(path, rows)
            raw = path.read_bytes()
            self.assertNotIn(b'\r', raw)
            self.assertEqual(ref(path)['sha256'], hashlib.sha256(raw).hexdigest())
            with path.open(encoding='utf-8', newline='') as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual(actual, [{k: str(v) for k, v in row.items()} for row in rows])

    def test_existing_csv_is_not_replaced(self):
        from scripts.ieee_tc.audit_d703_3b_full import write_exclusive_csv
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'requests.csv'
            path.write_bytes(b'original\r\n')
            with self.assertRaises(FileExistsError):
                write_exclusive_csv(path, [dict(value=1)])
            self.assertEqual(path.read_bytes(), b'original\r\n')


class ArrivedFilePreparationAnalysis(unittest.TestCase):
    def fixture(self):
        transfer = dict(transfer_id='t1', artifact_id='adapter-a', loading_clock_id='clock',
            state='published', loading_started_monotonic_s=10.2, published_monotonic_s=10.4,
            client_completed_monotonic_s=10.5, transferred_bytes=1024)
        early = dict(policy='observed_remote_demand_v1', request_id='r1', adapter_id='adapter-a',
            clock_id='clock', state='prepared', target_replica=None, physical_gpu_reservation=False,
            observed_monotonic_s=10.1, started_monotonic_s=10.15, settled_monotonic_s=10.6,
            preparation=dict(state='published', remote_transfer=copy.deepcopy(transfer),
                             _movement=dict(owns_io=True)))
        request = dict(request_id='r1', adapter_id='adapter-a', readiness_tier_before_dispatch='nvme',
            gpu_reference_evidence=dict(arrival_file_preparation=early),
            native_token_timing=dict(native_clock_id='clock', controller_admitted_monotonic_s=10.8),
            external_arrival_timing=dict(planned_arrival_s=10.))
        return request, {'t1':transfer}

    def test_nvme_dispatch_does_not_erase_earlier_remote_wait_or_bytes(self):
        from scripts.ieee_tc.audit_d703_3b_full import arrived_file_preparation_row
        request, transfers = self.fixture()
        before = copy.deepcopy(request)
        row = arrived_file_preparation_row(request, transfers)
        self.assertEqual(row['dispatch_tier'], 'nvme')
        self.assertAlmostEqual(row['file_wait_ms'], 450.)
        self.assertAlmostEqual(row['observation_to_admission_ms'], 700.)
        self.assertEqual(row['owned_transferred_bytes'],1024)
        self.assertEqual(request,before)

    def test_shared_transfer_may_precede_subscriber_but_has_no_second_byte_charge(self):
        from scripts.ieee_tc.audit_d703_3b_full import arrived_file_preparation_row
        request, transfers = self.fixture()
        early = request['gpu_reference_evidence']['arrival_file_preparation']
        early.update(observed_monotonic_s=10.3, started_monotonic_s=10.35)
        early['preparation']['_movement']['owns_io'] = False
        row = arrived_file_preparation_row(request, transfers)
        self.assertEqual(row['transfer_id'],'t1')
        self.assertEqual(row['owned_transferred_bytes'],0)
        self.assertAlmostEqual(row['file_wait_ms'],250.)

    def test_absent_record_is_unknown_and_hint_is_not_a_gpu_readiness_claim(self):
        from scripts.ieee_tc.audit_d703_3b_full import arrived_file_preparation_row
        request, transfers = self.fixture()
        early = request['gpu_reference_evidence']['arrival_file_preparation']
        for state in ('native_presence_hint','file_already_published'):
            early.update(state=state)
            row = arrived_file_preparation_row(request, transfers)
            self.assertEqual(row['dispatch_tier'], 'nvme')
            self.assertEqual(row['file_wait_ms'],0.)
            self.assertIsNone(row['transfer_id'])
        request['gpu_reference_evidence'].clear()
        self.assertIsNone(arrived_file_preparation_row(request, transfers))

    def test_zero_selected_remote_cohort_keeps_global_bytes_and_early_evidence(self):
        import csv
        from scripts.ieee_tc.audit_d703_3b_full import (diagnose_remote_preparation,
            digest, write_exclusive_csv)
        for recorded,remote in ((True,False),(False,False),(False,True)):
            with self.subTest(recorded=recorded,remote=remote),tempfile.TemporaryDirectory() as root:
                raw,stage,output = (Path(root)/name for name in ('raw','stage','output'))
                raw.mkdir(); stage.mkdir()
                request,transfers = self.fixture()
                transfer=transfers['t1']
                transfer.update(http_transfer_id='http1',payload_bytes_verified=2048,
                    headers_received_monotonic_s=10.25, body_received_monotonic_s=10.3,
                    extraction_started_monotonic_s=10.32,extraction_completed_monotonic_s=10.35)
                if recorded:
                    request['gpu_reference_evidence']['arrival_file_preparation']['preparation'][
                        'remote_transfer']=copy.deepcopy(transfer)
                else:
                    request['gpu_reference_evidence'].clear()
                if remote:
                    request['readiness_tier_before_dispatch']='remote'
                    request['native_token_timing']['controller_admitted_monotonic_s']=10.1
                    request['gpu_reference_evidence'].update(
                        preparation_interval=dict(remote_transfer_id='t1',clock_id='clock'),
                        remote_preparation=copy.deepcopy(transfer),
                        receipt=dict(native_load_started_monotonic_s=10.6,
                            native_load_completed_monotonic_s=10.7,acquired_monotonic_s=10.7))
                result=raw/'experiment_results_full_fixture.json'
                result.write_text(json.dumps(dict(detailed_results=dict(tc_ieee_full=dict(requests=[request])))))
                outcome=raw/'launch.launch/physical_deployment/main_outcome.json'
                outcome.parent.mkdir(parents=True)
                outcome.write_text(json.dumps(dict(remote_transfers=[transfer])))
                table=stage/'requests.csv'
                write_exclusive_csv(table,[dict(request_id='r1',tier='remote' if remote else 'nvme',
                    ordinal=0,slo_group='pass',owner='owner',ttft_pass=True,tpot_pass=True,joint_pass=True,
                    ttft_ms=1000.,admitted=10.1,admission_to_acquired_ms=600.)])
                (stage/'diagnosis.json').write_text(json.dumps(dict(passed=True,source_run=str(raw),
                    request_table=dict(path=str(table),sha256=digest(table)),
                    source_refs=[dict(path=str(result),sha256=digest(result))])))
                diagnose_remote_preparation(raw,output,stage)
                report=json.loads((output/'diagnosis.json').read_text())
                self.assertEqual(report['selected_remote'],int(remote))
                self.assertEqual(report['joined_remote_requests'],int(remote))
                self.assertEqual(report['transfers']['n'],1)
                self.assertEqual(report['arrival_preparation']['recorded_requests'],int(recorded))
                self.assertEqual(report['arrival_preparation']['unrecorded_requests'],int(not recorded))
                self.assertEqual(report['arrival_preparation']['owned_transferred_bytes'],1024 if recorded else None)
                with (output/'remote_requests.csv').open() as h:
                    reader=csv.DictReader(h)
                    self.assertIn('request_id',reader.fieldnames)
                    self.assertEqual(len(list(reader)),int(remote))
                with (output/'transfers.csv').open() as h:
                    self.assertEqual(sum(int(row['bytes']) for row in csv.DictReader(h)),1024)

    def test_reuse_is_not_a_new_download(self):
        from scripts.ieee_tc.audit_d703_3b_full import arrived_file_preparation_row
        request, transfers = self.fixture()
        early = request['gpu_reference_evidence']['arrival_file_preparation']
        early['preparation'] = dict(state='reused',_movement=dict(owns_io=False))
        self.assertEqual(arrived_file_preparation_row(request,transfers)['owned_transferred_bytes'],0)
        early['preparation']['_movement']['owns_io'] = True
        with self.assertRaises(ValueError):
            arrived_file_preparation_row(request, transfers)

    def test_wrong_owner_clock_identity_and_omitted_wait_are_rejected(self):
        from scripts.ieee_tc.audit_d703_3b_full import arrived_file_preparation_row
        patches = [dict(request_id='other'),dict(adapter_id='other'),dict(clock_id='other'),
            dict(target_replica='already-selected'),dict(physical_gpu_reservation=True),
            dict(observed_monotonic_s=9.),dict(settled_monotonic_s=11.),
            dict(started_monotonic_s=10.3),dict(state='unverified')]
        for patch in patches:
            request, transfers = self.fixture()
            request['gpu_reference_evidence']['arrival_file_preparation'].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                arrived_file_preparation_row(request, transfers)
        request, transfers = self.fixture()
        transfers['t1']['transferred_bytes'] += 1
        with self.assertRaises(ValueError):
            arrived_file_preparation_row(request, transfers)


if __name__=='__main__':
    unittest.main()
