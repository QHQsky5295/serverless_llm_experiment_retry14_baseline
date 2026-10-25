"""Run the same 67 CPU checks after fixing only the test's caller attribution."""
import hashlib
import json
import sys
from pathlib import Path

raw = Path(__file__).resolve().parent
root = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
prior = json.loads((raw / 'ast_scope.json').read_text())
assert hashlib.sha256((root / 'scripts/run_all_experiments.py').read_bytes()).hexdigest() == prior['candidate_sha256']
source = (raw / 'qualification_program.py.snapshot').read_text()
assert source.count("/ 'tests1'") == 1
attempt = sys.argv[1] if len(sys.argv) > 1 else 'tests2'
assert attempt in ('tests2', 'tests3', 'tests4')
source = source.replace("/ 'tests1'", "/ " + repr(attempt))
with (raw / ('qualification_' + attempt + '.py.snapshot')).open('x') as stream:
    stream.write(source)
exec(compile(source, str(raw / ('qualification_' + attempt + '.py.snapshot')), 'exec'),
     {'__name__': '__main__', '__file__': str(__file__)})
