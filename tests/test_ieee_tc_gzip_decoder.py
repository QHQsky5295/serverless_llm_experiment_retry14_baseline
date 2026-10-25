"""Reuse the established extraction and real-owner contracts with ISA-L."""
import importlib.util
import os
import unittest
from unittest.mock import patch

from faaslora.storage import http_artifact_store as http
from tests import test_http_artifact_store as artifact_tests
from tests import test_ieee_tc_request_lifecycle as lifecycle_tests


HAS_ISAL = importlib.util.find_spec('isal') is not None


class DecoderSelection(unittest.TestCase):
    def test_standard_does_not_require_optional_dependency(self):
        with patch.dict('sys.modules', {'isal': None}):
            client = http.HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        self.assertIsNone(client._gzip_open)
        self.assertEqual(client._gzip_decoder_info['name'], 'stdlib')

    def test_explicit_isal_does_not_fall_back_when_missing(self):
        with patch.dict('sys.modules', {'isal': None}), self.assertRaises(ImportError):
            http.HttpArtifactStoreClient(endpoint='http://127.0.0.1:1', gzip_decoder='isal')

    def test_unknown_decoder_is_not_a_default(self):
        for value in ('', 'auto', 'unknown', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'gzip decoder'):
                http.HttpArtifactStoreClient(endpoint='http://127.0.0.1:1', gzip_decoder=value)

    @unittest.skipUnless(HAS_ISAL, 'optional ISA-L dependency not installed')
    def test_environment_selection_is_frozen_at_client_creation(self):
        from isal import __version__, igzip
        with patch.dict(os.environ, FAASLORA_REMOTE_ARTIFACT_ENABLED='1',
                        FAASLORA_REMOTE_ARTIFACT_ENDPOINT='http://127.0.0.1:1',
                        FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER='isal'):
            client = http.endpoint_from_env()
            os.environ['FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER'] = 'stdlib'
            self.assertIs(client._gzip_open, igzip.open)
            self.assertEqual(client._gzip_decoder_info, dict(name='isal', version=__version__))


@unittest.skipUnless(HAS_ISAL, 'optional ISA-L dependency not installed')
class IsalAtomicPublication(artifact_tests.AtomicArtifactPublication):
    # Existing fixture-client tests now exercise the real alternative reader.
    # Helpers constructing independent clients retain their explicit defaults.
    gzip_decoder = 'isal'

    def test_actual_decoder_receipt_and_close_on_extraction_failure(self):
        from isal import __version__
        self.client.configure_content_manifest(artifact_tests.content_manifest())
        self.client._opener.open.return_value = artifact_tests.SizedResponse(
            artifact_tests.archive_bytes())
        reader = self.client._gzip_open
        opened = []
        def observed_open(*args, **kwargs):
            decoded = reader(*args, **kwargs)
            opened.append(decoded)
            return decoded
        evidence = {}
        with patch.object(self.client, '_gzip_open', observed_open), \
                patch.object(http, '_extract_verified', side_effect=RuntimeError('injected cancel')), \
                self.assertRaisesRegex(RuntimeError, 'injected cancel'):
            self.client.download_artifact('a', str(self.target),
                require_content_manifest=True, evidence=evidence)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        self.assertEqual(evidence['gzip_decoder'], dict(name='isal', version=__version__))
        self.assertEqual(evidence['state'], 'not_published')
        self.assertEqual((self.target/'old').read_bytes(), b'previous-valid-copy')
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['a'])


@unittest.skipUnless(HAS_ISAL, 'optional ISA-L dependency not installed')
class IsalConfirmedPublication(lifecycle_tests.ConfirmedFilePublication):
    gzip_decoder = 'isal'

    def test_actual_runner_records_decoder_and_keeps_owner_verification(self):
        from isal import __version__
        self.assertTrue(self.fetch()[0])
        evidence = self.runner._remote_transfer_evidence[-1]
        self.assertEqual(evidence['gzip_decoder'], dict(name='isal', version=__version__))
        self.assertEqual(evidence['content_verification'], 'owner_publication')
        self.assertFalse(evidence['stream_content_verified'])
        self.assertTrue(evidence['content_verified'])
        self.assertTrue(evidence['confirmed_file_publication']['content_verified'])
