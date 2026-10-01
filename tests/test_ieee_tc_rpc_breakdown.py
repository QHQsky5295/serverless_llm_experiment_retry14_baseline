"""Bounded offline audit fixtures; no model or performance runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import analyze_rpc_breakdown


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


if __name__ == '__main__':
    unittest.main()
