#!/usr/bin/env python3
"""Check or apply captured, version-specific installed-package source overlays."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys

p=argparse.ArgumentParser()
p.add_argument('--snapshot',type=Path,required=True,help='One environments/<name> snapshot directory')
p.add_argument('--prefix',type=Path,required=True,help='Restored Conda/venv prefix')
p.add_argument('--apply',action='store_true',help='Apply after every target passes validation; default is read-only')
a=p.parse_args()
rows=json.loads((a.snapshot/'source-overlays.json').read_text())
sites={s.resolve() for s in a.prefix.glob('lib/python*/site-packages')}
pending=[]
for row in rows:
    original=a.snapshot/'source_overlays'/row['path']
    content=original.read_bytes()
    if hashlib.sha256(content).hexdigest()!=row['snapshot_sha256']:
        raise SystemExit('Snapshot checksum mismatch: '+row['path'])
    matches=[s for s in sites if (s/row['path']).is_file() and (s/row['distribution']).is_dir()]
    if len(matches)!=1:raise SystemExit('Missing or ambiguous matching package distribution: '+row['path'])
    target=matches[0]/row['path'];data=target.read_bytes();digest=hashlib.sha256(data).digest()
    installed=hashlib.sha256(data).hexdigest()
    if installed==row['snapshot_sha256']:
        print('ALREADY_MATCHES',row['path']);continue
    b64=base64.urlsafe_b64encode(digest).decode().rstrip('=')
    if 'sha256='+b64!=row['installed_original_hash']:
        raise SystemExit('Refusing to overwrite unexpected source content: '+str(target))
    pending.append((original,target))
    print('VALID_BASE',row['path'])
if a.apply:
    for original,target in pending:shutil.copyfile(original,target)
print('APPLIED' if a.apply else 'CHECK_ONLY',len(pending))
