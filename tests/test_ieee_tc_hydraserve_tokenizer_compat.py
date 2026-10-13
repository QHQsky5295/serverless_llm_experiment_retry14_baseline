import json
from pathlib import Path
import tempfile
import unittest
from HydraServe_project.runtime.tokenizer_compat import legacy_bpe_json, materialize_metadata_view


class TokenizerCompatTests(unittest.TestCase):
    def fixture(self, merges):
        return json.dumps(dict(model=dict(type='BPE', merges=merges,
            vocab={'a':0, 'b':1, 'ab':2}, ignore_merges=False),
            normalizer={'type':'Sequence','normalizers':[]}, added_tokens=[])).encode()

    def test_exact_semantic_inverse(self):
        source = self.fixture([['a','b'],['ab','b']])
        result = json.loads(legacy_bpe_json(source))
        self.assertEqual(result['model']['merges'], ['a b','ab b'])
        result['model']['merges'] = [m.split(' ') for m in result['model']['merges']]
        self.assertEqual(result, json.loads(source))

    def test_legacy_bytes_unchanged(self):
        raw = self.fixture(['a b'])
        self.assertEqual(legacy_bpe_json(raw), raw)

    def test_ambiguous_or_mixed_fail_closed(self):
        for merges in ([['a b','c']], [['a','']], [['a','b'],'a b'], [['a']], [3]):
            with self.subTest(merges=merges), self.assertRaises(ValueError):
                legacy_bpe_json(self.fixture(merges))
        with self.assertRaises(ValueError):
            legacy_bpe_json(b'{"model":{"type":"Unigram","merges":[]}}')

    def test_view_never_copies_or_changes_weights(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root/'source'/'test_lora'; source.mkdir(parents=True)
            (source/'tokenizer.json').write_bytes(self.fixture([['a','b']]))
            (source/'adapter_model.safetensors').write_bytes(b'unique weights')
            dest = root/'view'/'test_lora'
            try:
                receipt = materialize_metadata_view(source,dest)
                self.assertEqual(receipt['copied_weight_bytes'],0)
                self.assertTrue((dest/'adapter_model.safetensors').is_symlink())
                self.assertEqual((dest/'adapter_model.safetensors').resolve(),source/'adapter_model.safetensors')
                self.assertEqual((source/'tokenizer.json').read_bytes(),self.fixture([['a','b']]))
                with self.assertRaises(FileExistsError):
                    materialize_metadata_view(source,dest)
            finally:
                if dest.exists(): dest.chmod(0o755)


if __name__ == '__main__': unittest.main()
