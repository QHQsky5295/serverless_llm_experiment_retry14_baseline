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

    def run_audit(self):
        self.paths[0].write_text(json.dumps(dict(requests=self.rows)))
        self.paths[1].write_text(json.dumps(self.deployment))
        self.paths[2].write_text(''.join(json.dumps(r)+'\n' for r in self.terminals))
        self.paths[3].write_text(''.join(json.dumps(r)+'\n' for r in self.samples))
        return analyze_native_timeline(*self.paths,self.root/'output')

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


if __name__ == '__main__':
    unittest.main()
