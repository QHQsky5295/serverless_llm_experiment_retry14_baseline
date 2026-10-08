"""Missing classes are measured with existing inputs, never filled by defaults."""
import copy
import unittest

from scripts.ieee_tc.prepare_d298_capacity import control_short_gap_waves, short_source_gap_waves


class ControlProfileGapSelection(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for rank in (8,16):
            for length in (2,5):
                for i in range(3):
                    self.rows.append(dict(source_index=len(self.rows),
                        source_request_id=f'r{rank}-{length}-{i}', adapter_id=f'a{rank}-{length}-{i}',
                        class_features=dict(tier='gpu',representation='native_gpu_executable_v1',
                            footprint_bytes=100,adapter_rank=rank,prompt_tokens=10,
                            declared_output_tokens=length,admitted_after_accept=1)))
        self.bins = dict(prompt_tokens=[10],declared_output_tokens=[2,5],adapter_rank=[8,16],
                         footprint_bytes=[],admitted_requests=[1,2,4])
        self.missing = [dict(tier=tier,representation=rep,prompt_bin=0,output_limit_bin=0,
            rank_bin=rank,footprint_bin=0,admitted_bin=2)
            for tier,rep in [('host','native_cpu_dense_ab_v1:torch.float16:pinned:packed'),
                             ('host','verified_regular_file_tree_v1'),
                             ('nvme','verified_regular_file_tree_v1')]
            for rank in (0,1)]

    def test_exact_missing_sources_original_inputs_rotated_targets(self):
        before = copy.deepcopy((self.rows,self.bins,self.missing))
        waves = control_short_gap_waves(self.rows,self.bins,self.missing)
        self.assertEqual((self.rows,self.bins,self.missing),before)
        self.assertEqual(len(waves),10)
        self.assertEqual(waves[0]['role'],'kernel_warmup_retained')
        self.assertEqual(waves[0]['source'],'gpu')
        by_id = {r['source_request_id']:r for r in self.rows}
        by_source = {}
        for wave in waves[1:]:
            self.assertTrue(wave['actual_admission_coverage_required'])
            self.assertEqual(wave['background_request_count'],2)
            refs = wave['requests']
            self.assertEqual(len(refs),4)
            self.assertEqual(len({r['adapter_id'] for r in refs}),4)
            rows = [by_id[r['source_request_id']] for r in refs]
            self.assertEqual([r['class_features']['declared_output_tokens'] for r in rows],[5,5,2,2])
            self.assertEqual({r['class_features']['adapter_rank'] for r in rows[2:]},{8,16})
            for ref,row in zip(refs,rows):
                self.assertEqual(ref['adapter_id'],row['adapter_id'])
            by_source.setdefault(wave['source'],[]).append(wave)
        self.assertEqual(set(by_source),{'native_host','file_host','nvme'})
        for source,group in by_source.items():
            self.assertEqual([w['round'] for w in group],[0,1,2])
            self.assertEqual(group[0]['target_source_request_ids'],
                             list(reversed(group[1]['target_source_request_ids'])))
            self.assertEqual(group[0]['requests'],group[2]['requests'])
        self.assertNotEqual([w['source'] for w in waves[1:4]],[w['source'] for w in waves[4:7]])
        actual = [k for source,group in by_source.items() for k in group[0]['intended_missing_classes']]
        self.assertCountEqual(actual,self.missing)

    def test_no_supported_source_or_representation_substitution(self):
        for missing in ([],self.missing+self.missing[:1],
                        [dict(self.missing[0],tier='remote')],
                        [dict(self.missing[0],representation='unverified')],
                        [dict(self.missing[0],admitted_bin=0)],
                        [dict(self.missing[0],output_limit_bin=1)]):
            with self.subTest(missing=missing),self.assertRaises(ValueError):
                control_short_gap_waves(self.rows,self.bins,missing)
        with self.assertRaises(ValueError):
            short_source_gap_waves(self.rows,self.bins,self.missing[:2],source='nvme')
        with self.assertRaises(ValueError):
            control_short_gap_waves(self.rows,self.bins,self.missing,rounds=0)

    def test_missing_background_or_capacity_fails_without_new_inputs(self):
        rows = [r for r in self.rows if r['class_features']['declared_output_tokens']==2]
        with self.assertRaisesRegex(ValueError,'long background'):
            control_short_gap_waves(rows,self.bins,self.missing)
        bins = dict(self.bins,admitted_requests=[1,2,3])
        with self.assertRaisesRegex(ValueError,'capacity'):
            control_short_gap_waves(self.rows,bins,self.missing)
        rows = [r for r in self.rows if r['class_features']['adapter_rank']==8]
        with self.assertRaisesRegex(ValueError,'short target'):
            control_short_gap_waves(rows,self.bins,self.missing)


if __name__ == '__main__':
    unittest.main()
