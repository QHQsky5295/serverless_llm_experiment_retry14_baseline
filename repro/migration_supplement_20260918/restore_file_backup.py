#!/usr/bin/env python3
"""Verify a captured file backup, or restore it without overwriting unrelated files."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or str(path) != value:
        raise ValueError('Unsafe relative path: ' + repr(value))
    return path


def read_manifest(base):
    rows = []
    seen = set()
    with (base / 'files.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            safe_relative(row['path'])
            digest = row['sha256']
            if not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Invalid content digest')
            if row['object'] != 'objects/' + digest + '.gz':
                raise ValueError('Object path does not match digest')
            if 'pack' in row:
                if not re.fullmatch(r'packs/part-[0-9]{5}\.bin', row['pack']) or row['offset'] < 0 or row['compressed_bytes'] < 0:
                    raise ValueError('Invalid packed object location')
            if row['path'] in seen or row['bytes'] < 0 or not 0 <= row['mode'] <= 0o777:
                raise ValueError('Duplicate path or invalid size/mode')
            seen.add(row['path'])
            rows.append(row)
    return rows


def object_bytes(base, row):
    if 'pack' in row:
        path = base / row['pack']
        if path.is_symlink() or (base / 'packs').is_symlink():
            raise ValueError('Pack storage must not contain symlinks')
        with path.open('rb') as stream:
            stream.seek(row['offset'])
            data = stream.read(row['compressed_bytes'])
        if len(data) != row['compressed_bytes']:
            raise ValueError('Truncated object pack')
        return data
    path = base / row['object']
    if path.is_symlink() or (base / 'objects').is_symlink():
        raise ValueError('Object storage must not contain symlinks')
    return path.read_bytes()


def verify(base, rows):
    checked = {}
    for row in rows:
        identity = (row['sha256'], row['object_sha256'], row['bytes'],
                    row.get('pack', row['object']), row.get('offset', 0), row.get('compressed_bytes'))
        if identity in checked:
            continue
        packed = object_bytes(base, row)
        if hashlib.sha256(packed).hexdigest() != row['object_sha256']:
            raise ValueError('Compressed object checksum mismatch: ' + row['object'])
        h = hashlib.sha256()
        total = 0
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                total += len(block)
                if total > row['bytes']:
                    raise ValueError('Expanded object exceeds declared size')
                h.update(block)
        if total != row['bytes'] or h.hexdigest() != row['sha256']:
            raise ValueError('Original content checksum mismatch: ' + row['path'])
        checked[identity] = True
    return len(checked)


def checked_target(root, relative):
    path = root
    for component in safe_relative(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError('Destination path contains a symlink: ' + str(path))
    return path


def git_blob(path):
    h = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(base, rows, root):
    if root.is_symlink():
        raise ValueError('Destination root must not be a symlink')
    root = root.absolute()
    for ancestor in root.parents:
        if ancestor.is_symlink():
            raise ValueError('Destination ancestor must not be a symlink')
    # Preflight all conflicts before writing any files.
    pending = []
    unchanged = 0
    for row in rows:
        path = checked_target(root, row['path'])
        if path.exists():
            if not path.is_file():
                raise ValueError('Destination is not a regular file: ' + str(path))
            if sha_file(path) == row['sha256']:
                unchanged += 1
                continue
            if not row.get('previous_git_blob') or git_blob(path) != row['previous_git_blob']:
                raise ValueError('Refusing to overwrite unrelated file: ' + str(path))
        pending.append((row, path))
    for row, path in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        checked_target(root, row['path'])
        fd, temporary = tempfile.mkstemp(prefix='.restore-', dir=path.parent)
        temp_path = Path(temporary)
        try:
            h = hashlib.sha256()
            with os.fdopen(fd, 'wb') as output, gzip.GzipFile(fileobj=io.BytesIO(object_bytes(base, row))) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(block)
                    output.write(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError('Object changed after verification')
            os.chmod(temp_path, row['mode'])
            if 'mtime_ns' in row:
                os.utime(temp_path, ns=(row['mtime_ns'], row['mtime_ns']))
            if path.exists():
                if not row.get('previous_git_blob') or git_blob(path) != row['previous_git_blob']:
                    raise ValueError('Destination changed after preflight')
                os.replace(temp_path, path)
            else:
                # An exclusive hardlink avoids clobbering a newly appeared file.
                os.link(temp_path, path)
                temp_path.unlink()
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return {'restored': len(pending), 'already_identical': unchanged,
            'symlinks': 'See symlinks.json; review and remap external targets separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True, type=Path)
    parser.add_argument('--restore-to', type=Path, help='Explicitly write verified files to this directory; default only verifies')
    args = parser.parse_args()
    base = args.backup.resolve()
    rows = read_manifest(base)
    result = {'files': len(rows), 'verified_objects': verify(base, rows)}
    if args.restore_to is not None:
        result.update(restore(base, rows, args.restore_to))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
