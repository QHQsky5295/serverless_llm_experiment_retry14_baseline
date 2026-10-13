"""No GPU: reject unsafe Python ABI and unverifiable reuse of old CUDA wheels."""
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.ieee_tc_prepare_hydraserve_source_environment import cuda_distribution_manifest


class CudaReuseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name)
        self.info = self.site / 'nvidia_example_cu12-1.0.dist-info'
        self.info.mkdir()
        (self.info / 'WHEEL').write_text('Tag: py3-none-manylinux1_x86_64\n')
        (self.info / 'METADATA').write_text('Name: nvidia-example-cu12\nVersion: 1.0\n')
        self.library = self.site / 'nvidia/example/lib/library.so'
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b'native CUDA library fixture')
        self.expected = {'nvidia-example-cu12': '1.0'}
        self.record('nvidia/example/lib/library.so', self.library.read_bytes())

    def record(self, relative, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        (self.info / 'RECORD').write_text(f'{relative},sha256={digest},{len(data)}\n')

    def test_python_independent_exact_bytes(self):
        result = cuda_distribution_manifest(self.site, self.expected)
        self.assertEqual(result['nvidia-example-cu12']['version'], '1.0')
        self.assertEqual(len(result['nvidia-example-cu12']['members']), 1)

    def test_corruption_rejected(self):
        self.library.write_bytes(b'X' * self.library.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            cuda_distribution_manifest(self.site, self.expected)

    def test_wrong_abi_rejected(self):
        (self.info / 'WHEEL').write_text('Tag: cp311-cp311-manylinux1_x86_64\n')
        with self.assertRaisesRegex(ValueError, 'Python-independent'):
            cuda_distribution_manifest(self.site, self.expected)

    def test_link_outside_source_rejected(self):
        self.library.unlink()
        self.library.symlink_to('/etc/hostname')
        with self.assertRaisesRegex(ValueError, 'leaves source site'):
            cuda_distribution_manifest(self.site, self.expected)


if __name__ == '__main__':
    unittest.main()
