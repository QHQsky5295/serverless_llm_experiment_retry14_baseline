"""Explicit D761 evidence backup; never stage user dirt or force the V2 branch."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

MAIN=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW=Path(__file__).resolve().parent
OUT=MAIN/'paper_results/ieee_tc/baseline_audit/d761_3b_memory_cause'
sys.path.insert(0,str(MAIN))
from scripts import ieee_tc_preflight as p

def read(path):
    return json.loads(path.read_text())

def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')

def call(*args):
    return subprocess.check_output(args,cwd=MAIN,text=True,stderr=subprocess.STDOUT).strip()

own=p.scope_identity(p.cg_path().name)
save(RAW/'publication_identity.json',own)
assert not call('git','diff','--cached','--name-only')
assert call('git','rev-parse','HEAD')=='7bd3b1848d7481cc87190d8f639c00493748a63e'
assert read(RAW/'seal_receipt.json')['passed']
old=read(RAW/'seal_identity.json')
path=Path(old['path'])
assert p.scope_still_owned(old)
assert all(not file.read_text().split() for file in path.rglob('cgroup.procs'))
resources=p.cgroup_snapshot(path)
subprocess.run(['systemctl','--user','stop',old['unit']],check=True)
assert not path.exists()
save(RAW/'seal_closure.json',dict(identity=old,resources=resources,path_gone=True,
    actual_recursive_procs_empty=True))
for manifest_name in ['manifest.json','seal_manifest.json']:
    for name,sha in read(OUT/manifest_name)['files'].items():
        assert p.digest(OUT/name)==sha,name
for name in ['seal_receipt.json','seal_identity.json','seal_closure.json','publish.py']:
    assert not (OUT/name).exists()
    shutil.copy2(RAW/name,OUT/name)
files=sorted(path for path in OUT.iterdir() if path.is_file())
files += [MAIN/'docs/ieee_tc/EXECUTION_STATUS.md',
          MAIN/'docs/ieee_tc/D761_3B_SERVERLESSLLM_MEMORY_CAUSE_20261023.md']
secret=re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|(?im:^[ \t]*(?:password|passwd|secret|api_key|PRIME_REMOTE_TOKEN)[ \t]*=[ \t]*[\x22\x27]?[A-Za-z0-9+/]{8,})')
for path in files:
    assert path.stat().st_size<2*1024*1024
    assert not secret.search(path.read_text()),path
    if path.suffix=='.py' or path.name.endswith('.py.snapshot'):
        ast.parse(path.read_text(),filename=str(path))
paths=[str(path.relative_to(MAIN)) for path in files]
assert 'configs/generated/lora_manifest_1000.json' not in paths
assert not call('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
call('git','add','--',*paths)
assert set(call('git','diff','--cached','--name-only').splitlines())==set(paths)
call('git','-c','core.whitespace=blank-at-eol,space-before-tab,-blank-at-eof','diff','--cached','--check')
call('git','commit','-m','tc: isolate D760 tokenizer memory retention without a performance replay')
head=call('git','rev-parse','HEAD')
ref='refs/heads/backup/d761-3b-tokenizer-cause-20261023'
call('git','-c','http.version=HTTP/1.1','push','faaslora_origin','HEAD:'+ref)
assert call('git','ls-remote','faaslora_origin',ref).split()[0]==head
save(RAW/'publication_receipt.json',dict(passed=True,head=head,remote_ref=ref,
    force_push=False,v2_synchronized=False,paths=paths,
    resources=p.cgroup_snapshot(Path(own['path']))))
print(json.dumps(dict(passed=True,head=head,remote_ref=ref,paths=len(paths))))
