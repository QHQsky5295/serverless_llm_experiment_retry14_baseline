"""Stop only the identity-bound D731 broad-test scope, preserve its evidence."""
import json
import sys
from pathlib import Path
ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
identity = json.loads((RAW/'test_identity.json').read_text())
assert p.scope_still_owned(identity)
before = p.cgroup_snapshot(Path(identity['path']))
result = p.stop_scope_identity(identity, grace_seconds=2)
with (RAW/'broad_test_stop.json').open('x') as f:
    json.dump(dict(reason='Unnecessary broad launcher tests spawned unchanged native imports; narrow to the changed scheduling/ownership path.',
        identity=identity, before=before, result=result, candidate_failure=False,
        tests_incomplete=True), f, indent=2)
assert result['released']
print(json.dumps(dict(released=result['released'], before=before)))
