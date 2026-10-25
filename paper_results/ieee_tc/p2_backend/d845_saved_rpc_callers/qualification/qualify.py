"""Adapt D737's bounded unittest driver; preserve its checks and result format."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

RAW = Path(__file__).resolve().parent
ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
template = RAW.parent / 'd737_20261023/test_unittest.py'
source = template.read_text()
tests = [
    'tests.test_ieee_tc_host_execution_projection',
    'tests.test_ieee_tc_identity_recheck',
    'tests.test_ieee_tc_transfer_pressure.MixedOwnedPreparation',
    'tests.test_ieee_tc_transfer_pressure.AutomaticGPUReplacement',
    'tests.test_ieee_tc_transfer_pressure.IntegratedNativeHostReplacement',
]
start = source.index('    tests = [')
end = source.index("    with (RAW/'tests.log')", start)
source = source[:start] + '    tests = ' + repr(tests) + '\n' + source[end:]
source = source.replace('d734_3b_controller_prefix1000', 'd843_3b_controller_full_w2')
needle = "    save(RAW/'test_result.json', dict(returncode=result.returncode,"
assert source.count(needle) == 1
source = source.replace(needle, "    for path, sha in prior['protected'].items(): assert p.digest(Path(path)) == sha, path\n" + needle)
parent = subprocess.check_output(['git', 'show', 'a9a0e7c8d9c40ec21285acc22abec49c6a865d0c:scripts/run_all_experiments.py'], cwd=ROOT, text=True)
current = (ROOT / 'scripts/run_all_experiments.py').read_text()
before, after = ast.parse(parent), ast.parse(current)
old_nodes, new_nodes = list(ast.walk(before)), list(ast.walk(after))
assert len(old_nodes) == len(new_nodes)
changed = []
for old, new in zip(old_nodes, new_nodes):
    if isinstance(old, ast.Constant) and isinstance(new, ast.Constant) and old.value != new.value:
        changed.append(dict(old=old.value, new=new.value, old_line=old.lineno, new_line=new.lineno))
        new.value = old.value
assert len(changed) == 1
assert (changed[0]['old'], changed[0]['new']) == ('source_snapshot', 'source_identity_snapshot')
assert ast.dump(before) == ast.dump(after)
with (RAW / 'ast_scope.json').open('x') as handle:
    json.dump(dict(parent_commit='a9a0e7c8d9c40ec21285acc22abec49c6a865d0c', changes=changed,
        candidate_sha256=hashlib.sha256(current.encode()).hexdigest(),
        template=dict(path=str(template), sha256=hashlib.sha256(template.read_bytes()).hexdigest())), handle, indent=2)
with (RAW / 'qualification_program.py.snapshot').open('x') as handle:
    handle.write(source)
exec(compile(source, str(RAW / 'qualification_program.py.snapshot'), 'exec'),
     {'__name__': '__main__', '__file__': str(__file__)})
