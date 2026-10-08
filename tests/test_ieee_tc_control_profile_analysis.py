"""Offline counter audit fixtures, not performance or numerical qualification."""
import copy
import unittest

from scripts.ieee_tc.export_d298_profiles import (
    audit_control_interleave_snapshots, audit_control_source_run)


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


if __name__=='__main__':
    unittest.main()
