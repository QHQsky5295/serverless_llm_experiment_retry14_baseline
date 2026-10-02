"""Native JSON decoding is a representation change, not a state-policy change."""
import json
import unittest
from unittest.mock import patch
import msgspec
from scripts.run_all_experiments import _decode_native_rpc_frame, _encode_rpc_frame


class NativeRPCDecode(unittest.TestCase):
    def assert_exact(self,a,b):
        self.assertIs(type(a),type(b))
        if isinstance(a,dict):
            self.assertEqual(list(a),list(b))
            for key in a: self.assert_exact(a[key],b[key])
        elif isinstance(a,list):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b): self.assert_exact(x,y)
        elif isinstance(a,float): self.assertEqual(a.hex(),b.hex())
        else: self.assertEqual(a,b)

    def test_primitives_identity_and_unicode(self):
        value=dict(owner_id='copy-α',epoch=2**80,leased=True,
            allocation_ids=[0,2**53+1],rank=16,counts={'4':2},sources=[None],
            prompt='中文\n𝄞\t"\\',confirmed=459919.250000003,
            floats=[0.,-0.,5e-324,1.7976931348623157e308])
        raw=_encode_rpc_frame(value)
        self.assert_exact(json.loads(raw.decode()),_decode_native_rpc_frame(raw))

    def test_unchanged_wire_and_no_intermediate_stdlib_decode(self):
        raw=_encode_rpc_frame({'ok':True,'result':{'owner_id':'owner','epoch':3}})
        self.assertTrue(raw.endswith(b'\n'))
        with patch('json.loads',side_effect=AssertionError('stdlib decoder used')):
            self.assertEqual(_decode_native_rpc_frame(raw)['result']['epoch'],3)

    def test_reject_invalid_json_and_nonfinite_without_fallback(self):
        for raw in (b'{bad}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-Infinity}',
                    b'{"x":1e999}',b'{"x":1}trailing'):
            with self.subTest(raw=raw), self.assertRaises(msgspec.DecodeError):
                _decode_native_rpc_frame(raw)

    def test_invalid_utf8_preserves_unicode_error(self):
        raw=b'{"x":"\xff"}'
        with self.assertRaises(UnicodeDecodeError): json.loads(raw.decode('utf-8'))
        with self.assertRaises(UnicodeDecodeError): _decode_native_rpc_frame(raw)

    def test_valid_escaped_nonfinite_word_is_not_changed(self):
        self.assertEqual(_decode_native_rpc_frame(b'{"x":"NaN","y":null}'),
                         {'x':'NaN','y':None})

    def test_duplicate_member_matches_wire_json_semantics(self):
        raw=b'{"x":1,"x":2}\n'
        self.assert_exact(json.loads(raw),_decode_native_rpc_frame(raw))

    def test_decodes_fresh_detached_tree(self):
        raw=b'{"sources":[{"epoch":1}]}\n'
        a=_decode_native_rpc_frame(raw)
        a['sources'][0]['epoch']=2
        self.assertEqual(_decode_native_rpc_frame(raw)['sources'][0]['epoch'],1)

    def test_integer_boolean_and_float_remain_distinct(self):
        r=_decode_native_rpc_frame(b'[true,1,1.0,null]')
        self.assertEqual([type(v) for v in r],[bool,int,float,type(None)])


if __name__=='__main__': unittest.main()
