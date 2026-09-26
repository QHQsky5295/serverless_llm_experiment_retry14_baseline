from __future__ import annotations

import threading
import io
import json
import hashlib
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from faaslora.storage.http_artifact_store import HttpArtifactStoreClient, RemoteArtifactError
from remote_artifact_node.server import ArtifactHandler, ArtifactServer


def content_manifest(artifact_id='a', files=None):
    files = files if files is not None else {'adapter_model.safetensors': b'tiny-test-fixture'}
    return {'format': 'artifact_content_v1', 'artifacts': [dict(
        id=artifact_id, files=[dict(path=name, size_bytes=len(data),
                                    sha256=hashlib.sha256(data).hexdigest())
                              for name, data in files.items()])]}


class SizedResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


def archive_bytes(files=None):
    files = files if files is not None else [('adapter_model.safetensors', b'tiny-test-fixture')]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, data in files:
            item = tarfile.TarInfo(name)
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

    def test_routing_identity_uses_exact_frozen_config_and_logical_payload(self):
        config = b'{"r":8}'
        files = {'adapter_config.json': config, 'weights': b'tiny-fixture'}
        self.client.configure_content_manifest(content_manifest(files=files))
        identity = self.client.routing_identity('a', config)
        canonical = json.dumps([dict(path=name, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(files.items())],
            sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(identity['content_sha256'], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(identity['remote_payload_bytes'], sum(map(len, files.values())))
        self.assertEqual(identity['rank'], 8)
        self.assertEqual(identity['remote_representation'], 'tar_gzip_verified_file_tree_v1')
        self.client._opener.open.assert_not_called()
        for aid, raw in [('b', config), ('a', b'{"r":16}'), ('a', b'{ "r":8}')]:
            with self.subTest(aid=aid, raw=raw), self.assertRaises(ValueError):
                self.client.routing_identity(aid, raw)

    def test_routing_rank_does_not_silently_approximate_unqualified_metadata(self):
        for config in (b'{}', b'{"r":true}', b'{"r":0}', b'{"r":8,"rank_pattern":{"q":16}}'):
            client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
            client.configure_content_manifest(content_manifest(files={'adapter_config.json': config}))
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, 'rank'):
                client.routing_identity('a', config)

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

    def test_native_transfer_requires_frozen_manifest_before_network_or_writes(self):
        self.client._opener.open.return_value = SizedResponse(archive_bytes())
        with self.assertRaisesRegex(RemoteArtifactError, 'frozen content manifest'):
            self.client.download_artifact('a', str(self.target), require_content_manifest=True)
        self.client._opener.open.assert_not_called()
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')

    def test_verified_transfer_rejects_same_length_wrong_content_without_publication(self):
        self.client.configure_content_manifest(content_manifest(
            files={'adapter_model.safetensors': b'wrong-test-bytes!'}))
        self.client._opener.open.return_value = SizedResponse(archive_bytes())
        publisher = Mock()
        with self.assertRaisesRegex(RemoteArtifactError, 'content SHA'):
            self.client.download_artifact('a', str(self.target), publish=publisher,
                                          require_content_manifest=True)
        publisher.assert_not_called()
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')

    def test_frozen_manifest_is_immutable_order_independent_and_not_a_size_hint(self):
        payload = content_manifest(files={'b': b'B', 'a': b'A'})
        digest = self.client.configure_content_manifest(payload)
        payload['artifacts'][0]['files'].reverse()
        self.assertEqual(self.client.configure_content_manifest(payload), digest)
        payload['artifacts'][0]['files'][0]['size_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            self.client.configure_content_manifest(payload)
        self.assertEqual(self.client._content_manifest['a']['a'][0], 1)
        with self.assertRaises(TypeError):
            self.client._content_manifest['a']['a'] = (2, '0' * 64)

    def test_invalid_manifest_paths_and_file_identities_are_rejected(self):
        for name in ('../x', '/x', './x', 'a//x', 'a\\x', 'a/../x', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.client.configure_content_manifest(content_manifest(files={name: b'data'}))
        payload = content_manifest()
        payload['artifacts'][0]['files'].append(dict(payload['artifacts'][0]['files'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.client.configure_content_manifest(payload)
        with self.assertRaisesRegex(ValueError, 'both file and directory'):
            self.client.configure_content_manifest(content_manifest(files={'a': b'A', 'a/b': b'B'}))

    def test_verified_transfer_requires_exact_http_body_length(self):
        self.client.configure_content_manifest(content_manifest())
        for header in (None, '', '-1', '0', '1', str(len(archive_bytes()) + 1)):
            with self.subTest(header=header):
                response = SizedResponse(archive_bytes())
                response.headers = {} if header is None else {'Content-Length': header}
                self.client._opener.open.return_value = response
                evidence = {}
                with self.assertRaisesRegex(RemoteArtifactError, 'Content-Length'):
                    self.client.download_artifact('a', str(self.target), require_content_manifest=True,
                                                  evidence=evidence)
                self.assertEqual(evidence['state'], 'not_published')
                self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_unexpected_duplicate_missing_or_wrong_size_member_cannot_publish(self):
        self.client.configure_content_manifest(content_manifest())
        member = ('adapter_model.safetensors', b'tiny-test-fixture')
        for entries in ([member, ('extra', b'x')], [member, member], [],
                        [('adapter_model.safetensors', b'short')], [('../escape', b'data')]):
            with self.subTest(entries=entries):
                self.client._opener.open.return_value = SizedResponse(archive_bytes(entries))
                with self.assertRaises(RemoteArtifactError):
                    self.client.download_artifact('a', str(self.target), require_content_manifest=True)
                self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a'])

    def test_link_cannot_redefine_frozen_regular_file(self):
        self.client.configure_content_manifest(content_manifest())
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            member = tarfile.TarInfo('adapter_model.safetensors')
            member.type = tarfile.SYMTYPE
            member.linkname = '../old'
            archive.addfile(member)
        self.client._opener.open.return_value = SizedResponse(output.getvalue())
        with self.assertRaisesRegex(RemoteArtifactError, 'regular nonsparse'):
            self.client.download_artifact('a', str(self.target), require_content_manifest=True)
        self.assertEqual((self.target / 'old').read_bytes(), b'previous-valid-copy')

    def test_verified_nested_files_publish_with_auditable_byte_counts(self):
        payload = {'nested/a': b'A', 'nested/b': b'BB'}
        digest = self.client.configure_content_manifest(content_manifest(files=payload))
        body = archive_bytes(list(payload.items()))
        self.client._opener.open.return_value = SizedResponse(body)
        evidence = {}
        ok, elapsed, size = self.client.download_artifact('a', str(self.target),
            require_content_manifest=True, evidence=evidence)
        self.assertTrue(ok)
        self.assertGreater(elapsed, 0)
        self.assertEqual(size, 3)
        self.assertEqual(evidence['state'], 'published')
        self.assertEqual(evidence['transferred_bytes'], len(body))
        self.assertEqual(evidence['archive_bytes_declared'], len(body))
        self.assertEqual(evidence['payload_bytes_expected'], 3)
        self.assertEqual(evidence['payload_bytes_verified'], 3)
        self.assertTrue(evidence['content_verified'])
        self.assertEqual(evidence['content_manifest_sha256'], digest)
        self.assertEqual((self.target / 'nested' / 'b').read_bytes(), b'BB')


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
        client.configure_content_manifest(content_manifest('demo_lora', {
            'adapter_config.json': b'{"base_model_name_or_path":"demo"}\n',
            'adapter_model.safetensors': b'demo\n'}))
        assert client.health()["ok"] is True
        assert client.list_artifacts() == ["demo_lora"]
        target = tmp_path / "fetch" / "demo_lora"
        evidence = {}
        ok, _elapsed_ms, size_bytes = client.download_artifact("demo_lora", str(target),
            require_content_manifest=True, evidence=evidence)
        assert ok
        assert size_bytes > 0
        assert (target / "adapter_config.json").exists()
        assert (target / "adapter_model.safetensors").exists()
        assert evidence['content_verified'] and evidence['state'] == 'published'
    finally:
        server.shutdown()
        server.server_close()
