"""Preserve the failed test version before correcting its caller attribution."""
from pathlib import Path
import sys
root = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
raw = Path(__file__).resolve().parent
attempt = sys.argv[1] if len(sys.argv) > 1 else 'tests1'
assert attempt in ('tests1', 'tests2', 'tests3')
with (raw / attempt / 'test_source.py.snapshot').open('xb') as stream:
    stream.write((root / 'tests/test_ieee_tc_host_execution_projection.py').read_bytes())
