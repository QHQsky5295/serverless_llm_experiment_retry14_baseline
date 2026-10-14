"""CPU fixtures; no model, kernel build or shared-cache mutation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import ieee_tc_native_toolchain as tc


class ToolchainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        (root/'bin').mkdir()
        for name in ('gcc', 'c++', 'nvcc', 'ninja', 'python'):
            path = root/'bin'/name
            path.write_text('fixture executable: '+name)
            path.chmod(0o700)
        self.python = str(root/'bin/python')
        env = dict(PATH=str(root/'bin'), CC=str(root/'bin/gcc'), CXX=str(root/'bin/c++'),
            CUDA_HOME=str(root), FLASHINFER_NVCC=str(root/'bin/nvcc'), MAX_JOBS='2',
            FLASHINFER_CUDA_ARCH_LIST='8.6', FLASHINFER_NVCC_THREADS='1')
        self.cfg = dict(schema='ieee_tc_native_toolchain_v1', python=self.python,
            environment=env, absent_environment=['FLASHINFER_EXTRA_CUDAFLAGS'],
            workspace_root=str(root/'cache'), executables={
                name:dict(path=str(root/'bin'/filename),sha256=tc.digest(root/'bin'/filename))
                for name,filename in [('cc','gcc'),('cxx','c++'),('nvcc','nvcc'),('ninja','ninja')]})
        self.path = root/'contract.json'
        self.freeze()

    def freeze(self):
        self.path.write_text(json.dumps(self.cfg))
        self.sha = tc.digest(self.path)

    def bind(self, env=None, **kw):
        return tc.bind_native_toolchain(env or {}, python=self.python,
            contract=str(self.path), expected_sha=self.sha, **kw)

    def test_opt_in_no_change_to_legacy(self):
        original = {'PATH':'legacy','KEEP':'1'}
        env, receipt = tc.bind_native_toolchain(original, python=sys.executable)
        self.assertEqual(env, original)
        self.assertIsNot(env, original)
        self.assertIsNone(receipt)

    def test_parent_and_child_identical_no_cache_creation(self):
        original = dict(PATH='foreign', CC='foreign', KEEP='safe', CUDA_VISIBLE_DEVICES='GPU-x')
        before = copy.deepcopy(original)
        env, receipt = self.bind(original)
        child, child_receipt = self.bind(env)
        self.assertEqual(env, child)
        self.assertEqual(receipt, child_receipt)
        self.assertEqual(original, before)
        self.assertEqual(env['KEEP'], 'safe')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-x')
        self.assertFalse((self.root/'cache').exists())
        self.assertFalse(receipt['build_performed'])

    def test_marked_child_cannot_silently_repair_environment(self):
        env, _ = self.bind()
        for key in (*self.cfg['environment'], 'FLASHINFER_WORKSPACE_BASE', tc.BOUND_ENV):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bind(dict(env, **{key:'wrong'}))

    def test_contract_and_binary_tamper_rejected(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'contract SHA'):
            self.bind()
        self.freeze()
        (self.root/'bin/nvcc').write_text('different compiler')
        with self.assertRaisesRegex(ValueError, 'executable SHA'):
            self.bind()

    def test_unfrozen_flags_and_shared_namespace_rejected(self):
        for env in ({'FLASHINFER_EXTRA_CUDAFLAGS':'-O0'}, {'FLASHINFER_WORKSPACE_BASE':'/shared'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.bind(env)

    def test_namespace_changes_with_contract(self):
        first, _ = self.bind()
        self.cfg['environment']['CC'] = str(self.root/'bin/c++')
        self.cfg['executables']['cc'] = self.cfg['executables']['cxx'].copy()
        self.freeze()
        second, _ = self.bind()
        self.assertNotEqual(first['FLASHINFER_WORKSPACE_BASE'], second['FLASHINFER_WORKSPACE_BASE'])

    def test_missing_sha_wrong_python_and_orphan_marker_rejected(self):
        for kwargs in (
            dict(environment={},python=self.python,contract=str(self.path)),
            dict(environment={},python='/wrong',contract=str(self.path),expected_sha=self.sha),
            dict(environment={tc.BOUND_ENV:self.sha},python=self.python)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tc.bind_native_toolchain(**kwargs)

    def test_ninja_resolution_and_nvcc_home_checked(self):
        for key, value in [('PATH','/nonexistent'), ('CUDA_HOME','/nonexistent')]:
            original = copy.deepcopy(self.cfg)
            self.cfg['environment'][key] = value
            self.freeze()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bind()
            self.cfg = original


if __name__ == '__main__':
    unittest.main()
