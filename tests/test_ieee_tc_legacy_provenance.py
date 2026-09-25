import json
from pathlib import Path
import tempfile
import unittest

from scripts.eurosys27_v2_provenance import audit_legacy_full_pair


class LegacyFullAudit(unittest.TestCase):
    def test_same_inputs_do_not_imply_same_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); trace=root/'trace'; subset=root/'subset'
            trace.write_text('trace'); subset.write_text('subset')
            paths=[]
            for i in range(2):
                p=root/f'{i}.json'
                p.write_text(json.dumps({'metadata':{'experiment_time':str(i),
                    'shared_trace_path':str(trace), 'shared_adapter_subset_path':str(subset)},
                    'scenario_summaries':{'faaslora_full':{'avg_ttft_ms':i+1}},
                    'detailed_results':{'faaslora_full':{'requests':[{'id':1}]}}}))
                paths.append(p)
            before=[p.read_bytes() for p in paths]
            r=audit_legacy_full_pair(*paths, root/'audit')
            self.assertTrue(r['same_trace'] and r['same_subset'])
            self.assertFalse(r['same_result'])
            self.assertEqual(r['scalar_summary_rows'][0]['main_table'],1)
            self.assertEqual(before,[p.read_bytes() for p in paths])
            with self.assertRaises(ValueError):audit_legacy_full_pair(*paths,root/'audit')
