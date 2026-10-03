"""Bounded offline audit fixtures; no model or performance runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import (
    analyze_rpc_breakdown, analyze_control_boundaries, CONTROL_BOUNDARIES,
)


class RpcBreakdownTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root / name for name in
                      ('projection.json','deployment.json','terminals.jsonl','sealed.json')]
        self.rows = [dict(request_id=str(i),instance_id='replica',success=True,
            output_contract_match=True,generation_contract='fixed_length_greedy_v1',
            completion_token_source='vllm_token_ids',routing_decision_us=i,
            native_token_timing=dict(native_terminal_observed=True,
                timing_contract='ieee_tc_native_v1',native_clock_id='clock',
                parent_rpc_transport='native_async_socket_v1',
                parent_rpc_response_pickup_delay_ms=i,
                parent_rpc_thread_resume_delay_ms=0)) for i in range(20)]
        self.terminals = [dict(request_id=str(i),instance_id='replica',success=True,
            native_contract_matched=True,clock_id='clock') for i in range(20)]

    def inputs(self):
        self.paths[0].write_text(json.dumps(dict(requests=self.rows)))
        self.paths[1].write_text(json.dumps(dict(plan=dict(count=20),clock_id='clock')))
        self.paths[2].write_text(''.join(json.dumps(r)+'\n' for r in self.terminals))
        self.paths[3].write_text(json.dumps(dict(source_refs=[
            dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in self.paths[:3]],native_contract_completion_pass=True,
            population=dict(native_contract_matched=20),metric_protocol_sha256='frozen-fixture')))
        self.sha = hashlib.sha256(self.paths[3].read_bytes()).hexdigest()

    def run_audit(self):
        self.inputs()
        return self.execute()

    def execute(self):
        return analyze_rpc_breakdown(*self.paths,self.sha,self.root/'out')

    def test_type_one_and_structural_zero_caveat(self):
        r = self.run_audit()
        m = {v['field']:v for v in r['metrics']}
        self.assertEqual(m['parent_rpc_response_pickup_delay_ms']['p95'],18)
        self.assertEqual(m['parent_rpc_response_pickup_delay_ms']['mean'],9.5)
        self.assertEqual(m['parent_rpc_thread_resume_delay_ms']['zero'],20)
        self.assertTrue(any('structural zero' in c for c in r['caveats']))
        self.assertFalse(r['complete_field_coverage'])
        self.assertFalse(r['new_experiment'])

    def test_missing_null_invalid_not_zero(self):
        name = 'parent_rpc_response_pickup_delay_ms'
        del self.rows[0]['native_token_timing'][name]
        for i, value in enumerate((None,True,float('nan'),float('inf'),-1,'5'),1):
            self.rows[i]['native_token_timing'][name] = value
        r = self.run_audit()
        m = next(v for v in r['metrics'] if v['field']==name)
        self.assertEqual((m['missing'],m['null'],m['invalid'],m['valid'],m['zero']),(1,1,5,13,0))
        self.assertEqual(set(r['invalid_or_absent_request_ids'][name]),{str(i) for i in range(7)})
        self.assertEqual(m['mean'],13)

    def test_missing_all_is_na(self):
        r = self.run_audit()
        m = next(v for v in r['metrics'] if v['field']=='worker_rpc_queue_ms')
        self.assertEqual(m['missing'],20)
        self.assertIsNone(m['mean'])
        self.assertIsNone(m['p95'])

    def test_no_obsolete_alias(self):
        self.rows[0]['parent_response_pickup_delay_ms']=100000
        r = self.run_audit()
        m = next(v for v in r['metrics'] if v['field']=='parent_rpc_response_pickup_delay_ms')
        self.assertEqual(m['mean'],9.5)

    def test_duplicate_rejected(self):
        self.rows[1]['request_id']='0'
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_missing_population_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError,'population'): self.run_audit()

    def test_failed_population_rejected(self):
        self.terminals[0]['native_contract_matched']=False
        with self.assertRaisesRegex(ValueError,'contract'): self.run_audit()

    def test_mixed_clock_rejected(self):
        self.rows[0]['native_token_timing']['native_clock_id']='different'
        with self.assertRaisesRegex(ValueError,'contract'): self.run_audit()

    def test_wrong_transport_rejected(self):
        self.rows[0]['native_token_timing']['parent_rpc_transport']='blocking_socket_v1'
        with self.assertRaisesRegex(ValueError,'contract'): self.run_audit()

    def test_disagreeing_duplicate_field_rejected(self):
        self.rows[0]['parent_rpc_response_pickup_delay_ms']=1
        with self.assertRaisesRegex(ValueError,'disagree'): self.run_audit()

    def test_mutated_input_rejected(self):
        self.inputs()
        self.paths[0].write_text(self.paths[0].read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'pinned'): self.execute()

    def test_wrong_seal_rejected(self):
        self.inputs()
        self.sha='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'): self.execute()

    def test_no_overwrite(self):
        self.run_audit()
        with self.assertRaises(FileExistsError): self.execute()

    def test_oversize_rejected_before_read(self):
        self.inputs()
        with self.paths[0].open('wb') as f:
            f.truncate(256*1024**2)
        with self.assertRaisesRegex(ValueError,'bounded'): self.execute()


class ControlBoundaryAuditTest(unittest.TestCase):
    """Scalar synthetic chain fixtures; not measured service latency."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root/'diagnostic_stacks'
        self.directory.mkdir()
        self.receipt = self.root/'exec_receipt.json'
        self.deployment = self.root/'deployment.json'
        self.owner = '/sys/fs/cgroup/fixture.scope'
        self.receipt.write_text(json.dumps(dict(allow_exec=True,
            service_identity=dict(path=self.owner), external_replay=dict(
                replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000))))
        self.deployment.write_text(json.dumps(dict(contract='physical_gpu_deployment_v1',
            clock_id='fixture-clock', arrival_start_s=200., plan=dict(count=1000))))
        # Lexicographic file order is deliberately not causal role order.
        self.role_pid = dict(parent=310,frontend=330,native=320)
        self.metas, self.rows = {}, []
        for role,pid in self.role_pid.items():
            self.metas[pid] = dict(kind='diagnostic_python_frames_v1', pid=pid,
                start_ticks=50+pid,cgroup=self.owner+'/'+role,
                capture_start_monotonic_s=100.,formal_performance_result=False,
                control_boundaries='request_source_control_boundary_v1',control_overhead_included=True)
        for i in range(20):
            for j,boundary in enumerate(CONTROL_BOUNDARIES):
                role = ('parent' if boundary.startswith('parent') else
                        'native' if boundary.startswith('native') else 'frontend')
                pid = self.role_pid[role]
                self.rows.append(dict(event='request_source_control_boundary_v1',
                    attempt_id=f'{i+1:032x}',boundary=boundary,pid=pid,thread_ident=pid+1000,
                    clock_id='fixture-clock',monotonic_s=190+i*2+j*(i+1)*.001,
                    thread_cpu_s=10+i+j*.0001,
                    byte_count=123 if boundary=='parent_send' else 456 if boundary=='parent_received' else None,
                    outcome='success' if boundary=='parent_terminal' else None))

    def inputs(self):
        for pid,meta in self.metas.items():
            stem=self.directory/f"{pid}-{meta['start_ticks']}"
            stem.with_suffix('.json').write_text(json.dumps(meta))
            stem.with_suffix('.control.jsonl').write_text(''.join(json.dumps(r)+'\n'
                for r in sorted(self.rows,key=lambda r:r['monotonic_s']) if r['pid']==pid))

    def execute(self):
        return analyze_control_boundaries(self.directory,self.receipt,self.deployment,self.root/'out')

    def audit(self):
        self.inputs()
        return self.execute()

    def first(self,boundary):
        return next(r for r in self.rows if r['attempt_id']==f'{1:032x}' and r['boundary']==boundary)

    def test_complete_type_one_grouping_additivity_and_provenance(self):
        r=self.audit()
        self.assertTrue(r['complete_success_coverage'])
        self.assertEqual((r['events'],r['attempts'],r['statuses']), (220,20,{'success_complete':20}))
        m={(v['phase'],v['field']):v for v in r['metrics']}
        self.assertAlmostEqual(m['all','parent_total_ms']['p95'],190)
        self.assertAlmostEqual(m['all','parent_total_ms']['mean'],105)
        self.assertEqual(m['prebusiness','parent_total_ms']['count'],5)
        self.assertEqual(m['business','parent_total_ms']['count'],15)
        spans=[v['mean'] for v in r['metrics'] if v['phase']=='all' and '__to__' in v['field']]
        self.assertAlmostEqual(sum(spans),105)
        self.assertAlmostEqual(m['all','native_thread_cpu_ms']['mean'],.1)
        self.assertTrue(all(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256'] for ref in r['source_refs']))
        self.assertFalse(r['new_experiment'])
        self.assertFalse(r['formal_performance_result'])

    def test_missing_success_is_incomplete_not_zero(self):
        self.rows.remove(self.first('native_ready'))
        r=self.audit()
        self.assertEqual(r['statuses'],dict(success_incomplete=1,success_complete=19))
        m=next(v for v in r['metrics'] if v['phase']=='all' and v['field']=='parent_total_ms')
        self.assertEqual(m['count'],19)
        self.assertAlmostEqual(m['mean'],110)
        self.assertFalse(r['complete_success_coverage'])

    def test_cancellation_allows_native_completion_after_parent_terminal(self):
        terminal=self.first('parent_terminal')
        terminal['outcome']='cancelled'
        terminal['monotonic_s']=190.0055
        self.rows.remove(self.first('parent_received'))
        r=self.audit()
        self.assertEqual(r['statuses'].get('cancelled'),1)
        self.assertEqual(r['statuses'].get('success_complete'),19)

    def test_errors_orphans_and_incomplete_preserved(self):
        self.first('parent_terminal')['outcome']='error'
        for aid,remove in ((2,('parent_begin','parent_terminal')),(3,('parent_terminal',))):
            self.rows=[r for r in self.rows if not(r['attempt_id']==f'{aid:032x}' and r['boundary'] in remove)]
        r=self.audit()
        self.assertEqual(r['statuses'],dict(error=1,orphan=1,incomplete=1,success_complete=17))

    def test_all_incomplete_metrics_na(self):
        self.rows=[r for r in self.rows if r['boundary']=='parent_begin']
        r=self.audit()
        self.assertTrue(all(v['count']==0 and v['mean'] is None and v['p95'] is None for v in r['metrics']))

    def test_partial_final_line_retained_without_fake_event(self):
        self.inputs()
        with next(self.directory.glob('*.control.jsonl')).open('ab') as f:
            f.write(b'{"event":')
        r=self.execute()
        self.assertEqual(len(r['partial_lines']),1)
        self.assertEqual(r['events'],220)
        self.assertFalse(r['complete_success_coverage'])

    def test_duplicate_rejects_before_output(self):
        self.rows.append(dict(self.rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'): self.audit()
        self.assertFalse((self.root/'out').exists())

    def test_mixed_clock_rejects(self):
        self.first('native_ready')['clock_id']='other'
        with self.assertRaisesRegex(ValueError,'identity/clock'): self.audit()

    def test_changed_role_thread_rejects(self):
        self.first('native_ready')['thread_ident']+=1
        with self.assertRaisesRegex(ValueError,'process/thread'): self.audit()

    def test_wrong_causal_order_rejects(self):
        self.first('native_begin')['monotonic_s']=190.0035
        with self.assertRaisesRegex(ValueError,'cross-process'): self.audit()

    def test_invalid_scalar_rejects(self):
        for name,value in [('monotonic_s',True),('thread_cpu_s',float('nan')),
                           ('byte_count',None),('outcome','success'),('attempt_id','x'*32)]:
            with self.subTest(name=name):
                row=self.first('parent_send')
                old=row[name]
                row[name]=value
                with self.assertRaises(ValueError): self.audit()
                row[name]=old
                self.assertFalse((self.root/'out').exists())

    def test_complete_malformed_line_is_not_treated_as_partial(self):
        self.inputs()
        with next(self.directory.glob('*.control.jsonl')).open('ab') as f:
            f.write(b'{}\n')
        with self.assertRaisesRegex(ValueError,'schema'): self.execute()

    def test_nonqualified_owner_rejects(self):
        self.metas[310]['cgroup']='/sys/fs/cgroup/fixture.scope.other'
        with self.assertRaisesRegex(ValueError,'metadata'): self.audit()

    def test_receipt_count_mismatch_rejects(self):
        self.deployment.write_text(json.dumps(dict(contract='physical_gpu_deployment_v1',
            clock_id='fixture-clock',arrival_start_s=200.,plan=dict(count=999))))
        with self.assertRaisesRegex(ValueError,'receipt/deployment'): self.audit()

    def test_missing_metadata_and_oversized_inputs_reject(self):
        self.inputs()
        meta=next(self.directory.glob('*.json'))
        saved=meta.read_bytes()
        meta.unlink()
        with self.assertRaisesRegex(ValueError,'population'): self.execute()
        meta.write_bytes(saved)
        event=next(self.directory.glob('*.control.jsonl'))
        with event.open('wb') as f: f.truncate(128*1024**2+1)
        with self.assertRaisesRegex(ValueError,'bounded total'): self.execute()

    def test_long_line_rejects(self):
        self.inputs()
        next(self.directory.glob('*.control.jsonl')).write_bytes(b'x'*8193)
        with self.assertRaisesRegex(ValueError,'bounded event line'): self.execute()

    def test_no_overwrite(self):
        self.audit()
        with self.assertRaises(FileExistsError): self.execute()


if __name__ == '__main__':
    unittest.main()
