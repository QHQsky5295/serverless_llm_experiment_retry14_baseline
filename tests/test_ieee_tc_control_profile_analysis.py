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


if __name__=='__main__':
    unittest.main()
