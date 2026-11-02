import unittest

from scripts.ieee_tc.audit_d703_3b_full import native_startup_log


LOG = '''(EngineCore pid=123) Using V2 Model Runner
Capturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:11]
Capturing CUDA graphs (FULL): 100%|x| 2/2 [00:01]
Graph capturing finished in 20 secs, took 0.71 GiB
Capturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:10]\rCapturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:10]
Capturing CUDA graphs (FULL): 100%|x| 32/32 [00:04]
Graph capturing finished in 17 secs, took 0.31 GiB
init engine (profile, create kv cache, warmup model) took 82.56 s (compilation: 31.00 s)
'''


class NativeStartupTests(unittest.TestCase):
    def test_observed_counts_and_containing_spans(self):
        result = native_startup_log(LOG)
        self.assertEqual(result['worker_pid'], 123)
        self.assertEqual(result['capture_seconds_rounded'], [20., 17.])
        self.assertEqual(result['init_seconds'], 82.56)
        self.assertEqual(result['completed_progress_totals'], {'PIECEWISE': [88], 'FULL': [2, 32]})

    def test_incomplete_or_ambiguous_records_rejected(self):
        for text in (LOG.replace('Using V2 Model Runner', 'Using V1 Model Runner'),
                     LOG + '(EngineCore pid=124)',
                     LOG.replace('Graph capturing finished in 20 secs, took 0.71 GiB', ''),
                     LOG.replace('32/32', '31/32'),
                     LOG.replace('(FULL)', '(OTHER)'),
                     LOG.replace('took 82.56 s', 'took nan s')):
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError):
                native_startup_log(text)


if __name__ == '__main__':
    unittest.main()
