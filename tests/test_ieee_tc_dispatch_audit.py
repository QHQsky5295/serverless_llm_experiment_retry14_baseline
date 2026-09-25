import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_serverlessllm_replay import audit_dispatch_timing


class TimingAudit(unittest.TestCase):
    def rows(self):
        return [{'request_id':str(i), 'success':True, 'arrival_time_s':float(i),
            'dispatch_admission_wait_ms':1000., 'replay_dispatch_wait_ms':10.,
            'server_queue_wait_ms':990., 'service_ttft_ms':20.,
            'server_metrics':{'backend_started_at':100.+i, 'finished_at':100.5+i,
                              'queue_wait_ms':980.}} for i in range(3)]

    def test_timing_and_r2_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.json'; p.write_text(json.dumps({'results':self.rows()}))
            before=p.read_bytes(); result=audit_dispatch_timing(p)
            self.assertEqual(result['backend_gap_seconds']['p95_type1'],1)
            self.assertEqual(result['ready_at_enqueue_field_count'],0)
            self.assertFalse(result['counterfactual_performance_measured'])
            self.assertEqual(result['reuse_class_for_tc_main'],'R2')
            self.assertEqual(before,p.read_bytes())

    def test_missing_and_failed_rows_are_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.json'
            for kind in ('failed','missing'):
                rows=self.rows()
                if kind=='failed':rows[0]['success']=False
                else:del rows[0]['server_metrics']['backend_started_at']
                p.write_text(json.dumps({'results':rows}))
                with self.assertRaises(ValueError):audit_dispatch_timing(p)
