"""Close owned offline scopes, verify existing tokenizer identity, seal evidence."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MAIN=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW=Path(__file__).resolve().parent
SRC=RAW.parent/'d760_20261023'
OUT=MAIN/'paper_results/ieee_tc/baseline_audit/d761_3b_memory_cause'
sys.path.insert(0,str(MAIN))
from scripts import ieee_tc_preflight as p

def read(path):
    return json.loads(path.read_text())

def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')

own=p.scope_identity(p.cg_path().name)
save(RAW/'seal_identity.json',own)
assert os.sched_getaffinity(0)=={2,3,26,27}
assert read(RAW/'analysis2_receipt.json')['passed']
old=read(RAW/'analysis2_identity.json')
path=Path(old['path'])
assert p.scope_still_owned(old)
assert all(not file.read_text().split() for file in path.rglob('cgroup.procs'))
resources=p.cgroup_snapshot(path)
subprocess.run(['systemctl','--user','stop',old['unit']],check=True)
assert not path.exists()
save(RAW/'analysis2_closure.json',dict(identity=old,resources=resources,path_gone=True,
    actual_recursive_procs_empty=True))

index=MAIN/'paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json'
assert p.digest(index)=='bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275'
data=read(index)
assert len(data['artifacts'])==500
names=['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
       'config.json','generation_config.json']
base=MAIN/'models/LLM-Research--Llama-3.2-3B-Instruct'
facts=[]
for name in names:
    files=[next(f for f in a['files'] if f['path']==name) for a in data['artifacts']]
    values={f['sha256'] for f in files}
    actual=p.digest(base/name)
    assert values=={actual}
    facts.append(dict(path=name,count=len(files),sha256=actual,
                      base_file=str(base/name),size_bytes=(base/name).stat().st_size))
save(RAW/'tokenizer_equivalence.json',dict(passed=True,artifacts=500,
    index=str(index),index_sha256=p.digest(index),files=facts,
    scope='Existing verified remote content index vs current canonical backbone files; no new pool scan.',
    weights_changed=False,serving_adaptation_installed=False))
q=read(SRC/'qualification.json')
for name,sha in (q['protected']|q['preserved']).items():
    assert p.digest(Path(name))==sha,name
manifest=read(OUT/'manifest.json')
for name,sha in manifest['files'].items():
    assert p.digest(OUT/name)==sha,name
for name in ['tokenizer_diagnostic.py','analyze.py']:
    ast.parse((RAW/name).read_text(),filename=name)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
files=['analysis2_receipt.json','analysis2_identity.json','analysis2_closure.json',
       'tokenizer_equivalence.json','seal.py']
for name in files:
    assert not (OUT/name).exists()
    shutil.copy2(RAW/name,OUT/name)
previous=[]
for name in ['result_publication_receipt.json','result_publication_closure.json']:
    destination=OUT/('d760_'+name)
    assert not destination.exists()
    shutil.copy2(SRC/name,destination)
    previous.append(destination)
save(OUT/'seal_manifest.json',dict(passed=True,plan_sha256=p.check_plan(),
    protected_unchanged=len(q['protected']),preserved_user_files_unchanged=q['preserved'],
    files={name:p.digest(OUT/name) for name in files},
    previous_checkpoint={path.name:p.digest(path) for path in previous},
    main_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MAIN,text=True).strip(),
    no_performance_run=True, no_serving_change=True))
save(RAW/'seal_receipt.json',dict(passed=True,identity=own,
    resources=p.cgroup_snapshot(Path(own['path'])),protected_unchanged=len(q['protected'])))
print(json.dumps(dict(passed=True,tokenizer_equivalent=500,protected_unchanged=len(q['protected']))))
