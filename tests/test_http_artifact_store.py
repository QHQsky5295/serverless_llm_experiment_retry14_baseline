from __future__ import annotations

import threading
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from faaslora.storage.http_artifact_store import HttpArtifactStoreClient, RemoteArtifactError
from remote_artifact_node.server import ArtifactHandler, ArtifactServer


def archive_bytes():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        item = tarfile.TarInfo('adapter_model.safetensors')
        data = b'tiny-test-fixture'
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    return output.getvalue()


class AtomicArtifactPublication(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / 'a'
        self.target.mkdir()
        (self.target / 'old').write_bytes(b'previous-valid-copy')
        self.client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        self.client._opener = Mock()

    def test_bad_archive_preserves_previous_destination(self):
        self.client._opener.open.return_value = io.BytesIO(b'invalid-archive')
        with self.assertRaises(tarfile.ReadError):
            self.client.download_artifact('a', str(self.target))
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_unfinished_extraction_does_not_expose_partial_destination(self):
        self.client._opener.open.return_value = io.BytesIO(archive_bytes())
        def interrupted(archive, staging):
            (staging / 'partial').write_bytes(b'incomplete')
            self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
            raise RuntimeError('extraction failed')
        with patch('faaslora.storage.http_artifact_store._safe_extract', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'extraction failed'):
                self.client.download_artifact('a', str(self.target))
        self.assertFalse((self.target / 'partial').exists())

    def test_completed_extraction_is_published_once_and_workspace_removed(self):
        from faaslora.storage.http_artifact_store import publish_directory
        self.client._opener.open.return_value = io.BytesIO(archive_bytes())
        calls = []
        def publish(staging, target):
            self.assertEqual((target / 'old').read_bytes(), b'previous-valid-copy')
            self.assertEqual((staging / 'adapter_model.safetensors').read_bytes(), b'tiny-test-fixture')
            calls.append((staging, target))
            publish_directory(staging, target)
        ok, elapsed, size = self.client.download_artifact('a', str(self.target), publish=publish)
        self.assertTrue(ok)
        self.assertGreater(elapsed, 0)
        self.assertEqual(size, len(b'tiny-test-fixture'))
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.target / 'old').exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_publication_failure_restores_previous_copy(self):
        self.client._opener.open.return_value = io.BytesIO(archive_bytes())
        original = Path.rename
        def rename(path, destination):
            if path.name == 'payload':
                raise OSError('injected publication failure')
            return original(path, destination)
        with patch.object(Path, 'rename', rename):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                self.client.download_artifact('a', str(self.target))
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_failed_restore_keeps_recovery_copy_instead_of_erasing_it(self):
        self.client._opener.open.return_value = io.BytesIO(archive_bytes())
        original = Path.rename
        def rename(path, destination):
            if path.name in ('payload', 'previous'):
                raise OSError('injected filesystem failure')
            return original(path, destination)
        with patch.object(Path, 'rename', rename):
            with self.assertRaisesRegex(RemoteArtifactError, 'previous copy retained'):
                self.client.download_artifact('a', str(self.target))
        workspaces = list(self.root.glob('.a.staging-*'))
        self.assertEqual(len(workspaces), 1)
        self.assertEqual((workspaces[0] / 'previous' / 'old').read_bytes(), b'previous-valid-copy')

    def test_cancellation_after_transfer_does_not_publish_or_delete_previous(self):
        event = threading.Event()
        class CancellingResponse(io.BytesIO):
            def read(inner, *args):
                value = super().read(*args)
                event.set()
                return value
        self.client._opener.open.return_value = CancellingResponse(archive_bytes())
        publisher = Mock()
        with self.assertRaisesRegex(RemoteArtifactError, 'cancelled'):
            self.client.download_artifact('a', str(self.target), publish=publisher, cancel_event=event)
        publisher.assert_not_called()
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_invalid_id_is_rejected_before_local_or_network_work(self):
        with self.assertRaises(ValueError):
            self.client.download_artifact('../escape', str(self.target))
        self.client._opener.open.assert_not_called()
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_existing_loopback_server_roundtrip(self):
        test_http_artifact_store_roundtrip(self.root)


def test_http_artifact_store_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    adapter = root / "demo_lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"demo"}\n')
    (adapter / "adapter_model.safetensors").write_text("demo\n")

    server = ArtifactServer(("127.0.0.1", 0), ArtifactHandler, root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        client = HttpArtifactStoreClient(endpoint=endpoint, timeout_s=10)
        assert client.health()["ok"] is True
        assert client.list_artifacts() == ["demo_lora"]
        target = tmp_path / "fetch" / "demo_lora"
        ok, _elapsed_ms, size_bytes = client.download_artifact("demo_lora", str(target))
        assert ok
        assert size_bytes > 0
        assert (target / "adapter_config.json").exists()
        assert (target / "adapter_model.safetensors").exists()
    finally:
        server.shutdown()
        server.server_close()
