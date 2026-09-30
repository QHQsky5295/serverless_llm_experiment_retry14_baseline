"""Small exact timeline fixtures, no GPU/data regeneration."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import analyze_native_timeline


class ControlTimelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root/name for name in ('projection.json','deployment.json','terminals.jsonl','watchdog.jsonl')]
        self.rows = []
        self.terminals = []
        for i in range(2):
            base = 100 + i
            self.rows.append(dict(request_id=str(i), success=True, instance_id='gpu0',
                scheduled_arrival_offset_s=i, arrival_released_offset_s=i,
                dispatch_window_wait_ms=1000, overall_e2e_ms=5000,
                native_token_timing=dict(native_clock_id='clock', controller_admitted_monotonic_s=base+2,
                    native_dispatch_monotonic_s=base+3, native_last_token_monotonic_s=base+4,
                    controller_completed_monotonic_s=base+5)))
            self.terminals.append(dict(request_id=str(i),success=True,native_contract_matched=True,
                                       instance_id='gpu0',clock_id='clock',at=base+6))
        self.deployment = dict(plan=dict(count=2),arrival_start_s=100,clock_id='clock')
        self.samples = [dict(event='resource_sample',monotonic=at,gpu=dict(
            service_held_gpu_uuids=['gpu0'],devices=[dict(gpu_uuid='gpu0',gpu_utilization_percent=25)]))
            for at in (100,101,102,103,104,105,106,107)]

    def run_audit(self, **kwargs):
        self.paths[0].write_text(json.dumps(dict(requests=self.rows)))
        self.paths[1].write_text(json.dumps(self.deployment))
        self.paths[2].write_text(''.join(json.dumps(r)+'\n' for r in self.terminals))
        self.paths[3].write_text(''.join(json.dumps(r)+'\n' for r in self.samples))
        return analyze_native_timeline(*self.paths,self.root/'output',**kwargs)

    def fail_request(self, i=0):
        self.rows[i].update(success=False,native_token_timing={},dispatch_window_wait_ms=0,
            failure_observation=dict(clock_id='clock',exception_type='TimeoutError',
                kind='controller_task_exception_v1',observed_monotonic_s=106.1+i))
        self.terminals[i].update(success=False,native_contract_matched=False)

    def control_outcome(self):
        events = [dict(observed_at=at,queue_depth=2,active_requests=1,ready_capacity=2,
            ready_instances=1,pending_instances=0,action='no_action',outcome='no_action')
            for at in (100,106,107)]
        path = self.root/'control.json'
        path.write_text(json.dumps(dict(mechanism_events=dict(_ieee_control_events=events,
            _ieee_runtime_quarantine_events=[dict(clock_id='clock',started_monotonic_s=106.5)]))))
        return path

    def test_exact_half_open_intervals_and_gap(self):
        r = self.run_audit()
        self.assertEqual(r['count'],2)
        self.assertEqual(r['observation_s'],7)
        p = r['phase_summaries']
        for name in p:
            self.assertEqual(p[name]['max_concurrent_requests'],2 if name == 'gate_to_terminal' else 1)
        self.assertEqual(p['controller_completion_to_terminal']['mean_duration_s'],1)
        self.assertEqual(p['native_dispatch_to_last_token']['total_request_seconds'],2)
        self.assertEqual(r['sampled_gate_occupancy_counts'],{0:2,1:2,2:4})

    def test_reject_wrong_clock(self):
        self.terminals[0]['clock_id'] = 'other'
        with self.assertRaisesRegex(ValueError,'timing domains'): self.run_audit()

    def test_reject_wrong_replica(self):
        self.terminals[0]['instance_id'] = 'other'
        with self.assertRaisesRegex(ValueError,'replica'): self.run_audit()

    def test_reject_incomplete_population(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_reject_duplicate_request(self):
        self.rows[1]['request_id'] = '0'
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_reject_failed_terminal(self):
        self.terminals[0]['success'] = False
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_reject_missing_timestamp(self):
        del self.rows[0]['native_token_timing']['native_dispatch_monotonic_s']
        with self.assertRaises(KeyError): self.run_audit()

    def test_reject_reordered_timestamp(self):
        self.rows[0]['native_token_timing']['native_dispatch_monotonic_s'] = 99
        with self.assertRaisesRegex(ValueError,'ordered'): self.run_audit()

    def test_reject_missing_gpu_activity(self):
        self.samples[0]['gpu']['devices'] = []
        with self.assertRaisesRegex(ValueError,'activity'): self.run_audit()

    def test_no_overwrite(self):
        self.run_audit()
        with self.assertRaises(FileExistsError): self.run_audit()

    def test_failure_opt_in_retains_all_ids_without_imputation(self):
        self.fail_request()
        r = self.run_audit(allow_failed=True)
        self.assertEqual(r['population'],dict(planned=2,terminal=2,native_success=1,failed=1))
        self.assertEqual(r['phase_population'],'native_success_only')
        self.assertEqual(r['phase_summaries']['gate_to_terminal']['n'],1)
        import csv
        with (self.root/'output/request_occupancy.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual({x['request_id'] for x in rows},{'0','1'})
        self.assertEqual(rows[0]['arrival_to_gate_s'],'')
        self.assertEqual(rows[0]['success'],'False')

    def test_failed_requires_explicit_opt_in(self):
        self.fail_request()
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_mixed_success_flags_rejected_even_when_allow_failed(self):
        self.rows[0]['success']=False
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit(allow_failed=True)

    def test_failure_clock_rejected(self):
        self.fail_request()
        self.rows[0]['failure_observation']['clock_id']='other'
        with self.assertRaisesRegex(ValueError,'failure timing'): self.run_audit(allow_failed=True)

    def test_collected_failure_after_terminal_is_not_service_time(self):
        self.fail_request()
        self.rows[0]['failure_observation']['observed_monotonic_s']=108
        r=self.run_audit(allow_failed=True)
        self.assertEqual(r['failures'][0]['observation_minus_terminal_s'],2)

    def test_task_failure_observed_before_terminal_rejected(self):
        self.fail_request()
        self.rows[0]['failure_observation']['observed_monotonic_s']=105
        with self.assertRaisesRegex(ValueError,'producer boundary'): self.run_audit(allow_failed=True)

    def test_returned_error_observed_before_terminal_is_valid(self):
        self.fail_request()
        self.rows[0]['failure_observation'].update(kind='native_request_execution_error_v1',observed_monotonic_s=105)
        r=self.run_audit(allow_failed=True)
        self.assertEqual(r['failures'][0]['observation_minus_terminal_s'],-1)

    def test_all_failed_has_no_fabricated_success_latency(self):
        self.fail_request(0)
        self.fail_request(1)
        r=self.run_audit(allow_failed=True)
        self.assertEqual(r['population']['failed'],2)
        self.assertIsNone(r['phase_summaries']['gate_to_terminal']['mean_duration_s'])
        self.assertEqual(r['native_by_replica'],{})

    def test_independent_control_history_phases(self):
        self.fail_request()
        r=self.run_audit(allow_failed=True,control_outcome_path=self.control_outcome())
        c=r['control_observations']
        self.assertEqual(c['first_failed_terminal_offset_s'],6)
        self.assertAlmostEqual(c['first_failure_observation_offset_s'],6.1)
        self.assertEqual(c['first_quarantine_offset_s'],6.5)
        self.assertEqual(len(c['by_phase']),3)
        for v in c['by_phase'].values():
            self.assertEqual(v['samples'],1)
            self.assertEqual(v['positive_queue_below_capacity_samples'],1)

    def test_control_counts_rejected(self):
        path=self.control_outcome()
        x=json.loads(path.read_text())
        x['mechanism_events']['_ieee_control_events'][0]['active_requests']=3
        path.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'exceed ready'): self.run_audit(control_outcome_path=path)

    def test_control_wrong_quarantine_clock_rejected(self):
        path=self.control_outcome()
        x=json.loads(path.read_text())
        x['mechanism_events']['_ieee_runtime_quarantine_events'][0]['clock_id']='other'
        path.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'quarantine clocks'): self.run_audit(control_outcome_path=path)


if __name__ == '__main__':
    unittest.main()
