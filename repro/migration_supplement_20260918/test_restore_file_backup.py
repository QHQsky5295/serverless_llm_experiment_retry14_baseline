import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import restore_file_backup as restore


class RestoreChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backup = self.root / 'backup'
        (self.backup / 'objects').mkdir(parents=True)
        self.data = b'preserved experiment input\n'
        digest = hashlib.sha256(self.data).hexdigest()
        packed = gzip.compress(self.data, mtime=0)
        self.row = {'path': 'configs/input.txt', 'sha256': digest, 'bytes': len(self.data),
                    'object': 'objects/' + digest + '.gz', 'object_sha256': hashlib.sha256(packed).hexdigest(),
                    'mode': 0o750}
        (self.backup / self.row['object']).write_bytes(packed)
        self.write_manifest([self.row])

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows):
        (self.backup / 'files.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_roundtrip_idempotence_permissions_and_dedup(self):
        second = dict(self.row, path='inputs/copy.txt')
        self.write_manifest([self.row, second])
        rows = restore.read_manifest(self.backup)
        self.assertEqual(restore.verify(self.backup, rows), 1)
        dest = self.root / 'restored'
        self.assertEqual(restore.restore(self.backup, rows, dest)['restored'], 2)
        self.assertEqual((dest / self.row['path']).read_bytes(), self.data)
        self.assertEqual((dest / self.row['path']).stat().st_mode & 0o777, 0o750)
        self.assertEqual(restore.restore(self.backup, rows, dest)['already_identical'], 2)

    def test_corruption_rejected(self):
        (self.backup / self.row['object']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            restore.verify(self.backup, [self.row])

    def test_packed_roundtrip_and_truncation(self):
        packed = (self.backup / self.row['object']).read_bytes()
        (self.backup / 'packs').mkdir()
        part = self.backup / 'packs/part-00001.bin'
        part.write_bytes(b'padding' + packed)
        row = dict(self.row, pack='packs/part-00001.bin', offset=7, compressed_bytes=len(packed))
        self.write_manifest([row])
        rows = restore.read_manifest(self.backup)
        self.assertEqual(restore.verify(self.backup, rows), 1)
        self.assertEqual(restore.restore(self.backup, rows, self.root / 'unpacked')['restored'], 1)
        part.write_bytes(b'padding' + packed[:-1])
        with self.assertRaises(ValueError):
            restore.verify(self.backup, rows)

    def test_missing_object_rejected(self):
        (self.backup / self.row['object']).unlink()
        with self.assertRaises(FileNotFoundError):
            restore.verify(self.backup, [self.row])

    def test_duplicate_digest_cannot_hide_bad_pack_reference(self):
        (self.backup / 'packs').mkdir()
        (self.backup / 'packs/part-00001.bin').write_bytes(b'invalid')
        second = dict(self.row, path='other.txt', pack='packs/part-00001.bin', offset=0, compressed_bytes=7)
        with self.assertRaises(ValueError):
            restore.verify(self.backup, [self.row, second])

    def test_path_escape_rejected(self):
        for path in ['../outside', '/tmp/outside', 'a/../../outside', './name']:
            self.write_manifest([dict(self.row, path=path)])
            with self.assertRaises(ValueError):
                restore.read_manifest(self.backup)

    def test_symlink_destination_rejected(self):
        dest = self.root / 'dest'
        dest.mkdir()
        (dest / 'configs').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            restore.restore(self.backup, [self.row], dest)

    def test_conflicts_preflight_before_writes(self):
        dest = self.root / 'dest'
        (dest / 'configs').mkdir(parents=True)
        target = dest / self.row['path']
        target.write_bytes(b'user changes')
        with self.assertRaises(ValueError):
            restore.restore(self.backup, [dict(self.row, path='new.txt'), self.row], dest)
        self.assertFalse((dest / 'new.txt').exists())
        self.assertEqual(target.read_bytes(), b'user changes')

    def test_replace_only_recorded_parent_blob(self):
        dest = self.root / 'dest'
        (dest / 'configs').mkdir(parents=True)
        target = dest / self.row['path']
        target.write_bytes(b'old commit content')
        row = dict(self.row, previous_git_blob=restore.git_blob(target))
        self.assertEqual(restore.restore(self.backup, [row], dest)['restored'], 1)
        self.assertEqual(target.read_bytes(), self.data)


if __name__ == '__main__':
    unittest.main()
