"""Exact host-time partition fixtures; not new performance experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import (summarize_core_host_spans,
    core_interval_overlap, analyze_core_host_timing)


def events():
    def row(identity,parent,boundary,a,b,details):
        return dict(event='native_core_host_span_v1',id=identity,parent=parent,boundary=boundary,
            begin_ns=a,end_ns=b,thread_cpu_begin_ns=a//2,thread_cpu_end_ns=b//2,
            outcome='returned',details=details,preceding_observer_ns=0,preceding_write_ns=0)
    return [row(2,1,'client',2,6,dict(request_type='UTILITY',utility='collective_rpc')),
        row(1,None,'input_queue',0,10,dict(unfinished_before=1,running_before=1,waiting_before=0)),
        row(4,3,'schedule',12,15,dict(scheduled_tokens=11,scheduled_requests=2,multi_token_requests=1)),
        row(3,None,'step',11,31,dict(unfinished_before=1,running_before=1,waiting_before=0,model_executed=True)),
        dict(event='observation_close',monotonic_ns=32,completed_spans=4,truncated=False,
             unfinished_span_ids=[],observer_ns=1,write_ns=1,bytes_before_close=0)]


class NativeCoreAnalysis(unittest.TestCase):
    def test_nested_partition_is_exact_and_half_open(self):
        value=summarize_core_host_spans(events())
        self.assertTrue(value['complete'])
        self.assertEqual(value['spans'],4)
        self.assertEqual(core_interval_overlap(value['index']['client'],4,21),2)
        self.assertEqual(core_interval_overlap(value['index']['input_nonclient'],4,21),4)
        self.assertEqual(core_interval_overlap(value['index']['step'],4,21),10)
        self.assertEqual(core_interval_overlap(value['index']['client'],6,10),0)
        self.assertEqual(core_interval_overlap(value['index']['step'],0,100),20)
        self.assertEqual(core_interval_overlap(value['index']['step'],31,31),0)
        self.assertEqual(value['orphan_span_ids'],[])

    def test_corrupted_nesting_duplicates_counts_and_order_reject(self):
        changes=[lambda e:e[0].update(id=1),lambda e:e[0].update(parent=3),
            lambda e:e[0].update(begin_ns=7),lambda e:e[0].update(end_ns=11),
            lambda e:e[2].update(parent=1),lambda e:e[-1].update(completed_spans=3),
            lambda e:e[-1].update(truncated=True),lambda e:e[3].update(begin_ns=9)]
        for change in changes:
            with self.subTest(change=change):
                rows=events();change(rows)
                with self.assertRaises(ValueError):summarize_core_host_spans(rows)

    def test_truncation_or_missing_close_not_filled_as_complete(self):
        rows=events()[:3]
        incomplete=summarize_core_host_spans(rows)
        self.assertFalse(incomplete['complete'])
        self.assertEqual(incomplete['orphan_span_ids'],[4])
        self.assertTrue(incomplete['overhead_is_lower_bound'])
        rows.append(dict(event='observation_limit',monotonic_ns=31,completed_spans=3,first_unwritten_span=3))
        footer=events()[-1]
        footer.update(completed_spans=3,truncated=True)
        rows.append(footer)
        value=summarize_core_host_spans(rows)
        self.assertFalse(value['complete'])
        self.assertTrue(value['closed'])
        self.assertTrue(value['truncated'])

    def test_empty_closed_process_is_retained(self):
        footer=events()[-1]
        footer.update(completed_spans=0)
        value=summarize_core_host_spans([footer])
        self.assertTrue(value['complete'])
        self.assertIsNone(value['first_ns'])
        self.assertEqual(value['metrics'],[])

    def write_fixture(self, root):
        directory=root/'diagnostic_core';directory.mkdir()
        receipt=root/'exec_receipt.json';deployment=root/'deployment.json'
        owner='/sys/fs/cgroup/test-unit'
        receipt.write_text(json.dumps(dict(allow_exec=True,service_identity=dict(path=owner),
            external_replay=dict(replay_scope='diagnostic_prefix_v1',diagnostic_prefix_count=1))))
        deployment.write_text(json.dumps(dict(plan=dict(count=1),clock_id='clock')))
        header=dict(event='native_core_host_timing_v1',pid=123,start_ticks=10,cgroup=owner,
            formal_performance_result=False,gpu_kernel_timing=False,observer_overhead_included=True,
            clock_id='clock',diagnostic_prefix_count=1,replay_scope='diagnostic_prefix_v1')
        source=directory/'123-10.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in [header]+events()))
        windows=[dict(request_id='r1',worker_pid=123,replica='replica1',tokens=2,tpot_ms=.000017,
                      timing_tpot_miss=False,first=4e-9,last=21e-9)]
        return directory,receipt,deployment,windows,root/'output',source

    def test_file_binding_output_partition_and_no_overwrite(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            args=self.write_fixture(Path(tmp))
            result=analyze_core_host_timing(*args[:5])
            self.assertEqual(result['complete_request_observations'],1)
            self.assertFalse(result['gpu_kernel_timing'])
            with (args[4]/'decode_windows.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[0]['other_or_unobserved_ms']),.000001)
            self.assertAlmostEqual(sum(float(rows[0][key]) for key in
                ('client_ms','input_nonclient_ms','step_ms','other_or_unobserved_ms')),
                float(rows[0]['decode_ms']))
            with self.assertRaises(FileExistsError):analyze_core_host_timing(*args[:5])

    def test_clock_pid_and_prefix_mismatch_reject(self):
        for key in ('clock','pid','count'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                args=self.write_fixture(Path(tmp))
                if key=='clock':
                    value=json.loads(args[2].read_text());value['clock_id']='other'
                    args[2].write_text(json.dumps(value))
                elif key=='pid': args[3][0]['worker_pid']=456
                else: args[3].append(copy.deepcopy(args[3][0]))
                with self.assertRaises(ValueError):analyze_core_host_timing(*args[:5])
                self.assertFalse(args[4].exists())


if __name__=='__main__':
    unittest.main()
