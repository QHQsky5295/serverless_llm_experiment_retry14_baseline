"""Reuse D1062's bounded test receipt for NoHOST consumer/price qualification.

Not a performance experiment and not authorization to enable NoHOST serving.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p

NAMES = [
    'tests.test_ieee_tc_nohost',
    'tests.test_ieee_tc_service_routing',
    'tests.test_ieee_tc_request_footprints',
    'tests.test_ieee_tc_worker_observation.NativeReplacementCostFusion',
    'tests.test_ieee_tc_transfer_pressure.OwnedPreparationPlanning',
]
SOURCES = [
    'faaslora/registry/schema.py',
    'faaslora/experiment/instance_pool.py',
    'faaslora/preloading/preloading_planner.py',
    'faaslora/preloading/planning_cpu.py',
    'tests/test_ieee_tc_nohost.py',
]


def main():
    tag = sys.argv[1]
    if not tag.replace('_', '').isalnum():
        raise ValueError('test attempt tag must be a local identifier')
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (group / 'memory.high').read_text().strip() == str(3 * p.GIB)
    assert (group / 'memory.max').read_text().strip() == str(4 * p.GIB)
    assert (group / 'memory.swap.max').read_text().strip() == '0'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
    before = {name: p.digest(ROOT / name) for name in SOURCES}
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromNames(NAMES)
    # The actual runner's module import exports REPO_ROOT for native workers.
    # This test only needs pure planning subprocesses: multiprocessing restores
    # sys.path for those, while the stdlib resource tracker needs no project or
    # vLLM startup hook. Do not change the production runner/export.
    os.environ.pop('PYTHONPATH', None)
    with (RAW / (tag + '.txt')).open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {name: p.digest(ROOT / name) for name in SOURCES}
    report = dict(passed=result.wasSuccessful() and before == after,
        tests_run=result.testsRun, names=NAMES, failures=len(result.failures), errors=len(result.errors),
        elapsed_s=time.monotonic() - started, sources=before, sources_after=after,
        source_unchanged=before == after, identity=identity, resources=p.cgroup_snapshot(group),
        performance_test=False, nohost_execution_qualified=False)
    with (RAW / (tag + '.json')).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
