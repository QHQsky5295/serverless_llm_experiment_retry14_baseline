import json
import hashlib
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_serverlessllm_replay import (
    audit_dispatch_timing, audit_tc_http_journal, audit_tc_historical_reuse)


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


class HistoricalNativeReuse(unittest.TestCase):
    def fixture(self, root):
        digest = lambda x: hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()
        trace = root/'trace.json'
        trace.write_text('{"historical_fixture":true}')
        contract = dict(event='request_contract', request_id='r', adapter_id='a',
                        target_tokens=2, native_prompt_token_ids_sha256=digest([1, 2]),
                        canonical_prompt_sha256='prompt', source_item_sha256='source')
        arrival = dict(event='request_created', request_id='r', planned_arrival_s=1.,
                       task_created_s=1.01, source_item_sha256='source')
        control = dict(tc_clock_id='c', instance_id='i', tc_http_received_s=1.2,
                       tc_router_enqueued_s=1.3, tc_instance_assigned_s=2., tc_backend_entry_s=2.1,
                       ready_instances_at_enqueue=1)
        observed = dict(control_observation=control, completion_token_ids=[3, 4],
                        completion_token_ids_sha256=digest([3, 4]), native_prompt_token_ids=[1, 2],
                        native_prompt_token_ids_sha256=digest([1, 2]), native_output_tokens=2,
                        native_lora_name='a', native_lora_int_id=1, native_clock_id='c',
                        timing_contract='ieee_tc_native_v1', native_terminal_observed=True,
                        native_dispatch_monotonic_s=2.2, native_queued_monotonic_s=2.3,
                        native_scheduled_monotonic_s=2.4, native_first_token_monotonic_s=2.5,
                        native_last_token_monotonic_s=3., worker_completed_monotonic_s=3.1,
                        native_tpot_ms=500.)
        body = dict(id='r', usage=dict(completion_tokens=2, prompt_tokens=2), metrics=dict(ieee_tc=observed))
        timing = dict(protocol_valid=True, lora_numerical_correctness_qualified=False,
                      ttft_ms=1500., e2e_ms=2200., submit_lag_ms=100.,
                      dispatch_wait_after_submit_ms=900., service_ttft_ms=500.,
                      decode_ms=500., completion_notification_ms=200., router_queue_ms=700.,
                      tpot_ms=500., instance_assigned_s=2., ready_instances_at_enqueue=1)
        return [dict(event='replay_ready', clock_id='c', plan=dict(
                    source_path=str(trace), source_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())),
                contract, arrival, dict(arrival, event='http_headers_sent', client_submit_s=1.1),
                dict(event='http_raw_response', request_id='r', status=200, body=body, client_completed_s=3.2),
                dict(event='http_response', request_id='r', response=timing),
                dict(event='http_replay_complete', N_plan=1, N_arrived=1,
                     N_terminal=1, N_response=1, N_failed=0)]

    def seal(self, root, events):
        replay = root/'replay.jsonl'
        replay.write_text(''.join(json.dumps(e)+'\n' for e in events))
        summary = root/'summary.json'
        result = audit_tc_http_journal(replay)
        result.update(model_profile='llama2_7b', variant='repaired')
        summary.write_text(json.dumps(result))
        evidence = root/'evidence.json'
        evidence.write_text(json.dumps(dict(raw_sources=[dict(
            path=str(replay), sha256=hashlib.sha256(replay.read_bytes()).hexdigest())])))
        return replay, summary, evidence

    def test_reuses_native_clocks_without_promoting_development_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.seal(root, self.fixture(root))
            before = [p.read_bytes() for p in paths]
            result = audit_tc_historical_reuse(*paths)
            self.assertEqual(result['native_revalidated_requests'], 1)
            self.assertLess(result['max_native_metric_recomputation_error_ms'], 1e-9)
            self.assertTrue(result['source_summary_reproduced'])
            self.assertEqual(result['reuse_class_for_tc_main'], 'R2')
            self.assertFalse(result['formal_performance_qualified'])
            self.assertIsNone(result['physical_gpu_seconds'])
            self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_rejects_consistent_but_wrong_derived_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            # Previous component-sum-only checker accepts this wrong TTFT.
            events[-2]['response']['ttft_ms'] += 100
            paths = self.seal(root, events)
            with self.assertRaisesRegex(ValueError, 'native metric recomputation'):
                audit_tc_historical_reuse(*paths)

    def test_rejects_changed_native_ids_clock_and_missing_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            for change in ('token', 'clock', 'submit'):
                events = copy.deepcopy(original)
                if change == 'token':
                    events[4]['body']['metrics']['ieee_tc']['completion_token_ids'] = [3, 9]
                elif change == 'clock':
                    events[4]['body']['metrics']['ieee_tc']['native_clock_id'] = 'another-host'
                else:
                    events[3]['planned_arrival_s'] = .5
                with self.subTest(change=change):
                    paths = self.seal(root, events)
                    with self.assertRaises(ValueError):
                        audit_tc_historical_reuse(*paths)

    def test_rejects_source_and_old_summary_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            paths = self.seal(root, events)
            paths[0].write_text(paths[0].read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                audit_tc_historical_reuse(*paths)
            paths = self.seal(root, events)
            old = json.loads(paths[1].read_text())
            old['counts']['N_response'] = 2
            paths[1].write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, 'summary differs'):
                audit_tc_historical_reuse(*paths)
            paths = self.seal(root, events)
            (root/'trace.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source trace changed'):
                audit_tc_historical_reuse(*paths)

    def test_preserves_failed_offered_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            events[-1:-1] = [dict(events[1], request_id='failed'),
                             dict(events[2], request_id='failed'),
                             dict(event='http_request_failed', request_id='failed', error='unavailable')]
            events[-1].update(N_plan=2, N_arrived=2, N_terminal=2, N_failed=1)
            result = audit_tc_historical_reuse(*self.seal(root, events))
            self.assertEqual(result['counts']['N_failed'], 1)
            self.assertEqual(result['native_revalidated_requests'], 1)
            self.assertEqual(result['failures'][0]['request_id'], 'failed')
