"""Bounded D1082 checks using existing HOST fixtures and unittest infrastructure."""
import argparse
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('tests', 'component'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    cg = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (cg/'memory.high').read_text().strip() == str(3*p.GIB)
    assert (cg/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (cg/'memory.swap.max').read_text().strip() == '0'
    result = dict(mode=args.mode, passed=False, scope=p.scope_identity(cg.name))
    try:
        if args.mode == 'tests':
            names = ['tests.test_ieee_tc_worker_observation.' + name for name in (
                'NativeHostBudgetProjection', 'NativeHostFootprint',
                'NativeHostRoutingProjection', 'NativeHostStagedUnionProjection')]
            names += ['tests.test_ieee_tc_hit_host_transaction.' + name for name in (
                'HitHostTransaction', 'AllocatorReobservation', 'NativeWorkerRecheck')]
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.loadTestsFromNames(names)
            report = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            result.update(passed=report.wasSuccessful(), tests=report.testsRun,
                          failures=len(report.failures), errors=len(report.errors),
                          skipped=len(report.skipped), log=stream.getvalue())
            print(stream.getvalue(), flush=True)
        else:
            from faaslora.memory import gpu_monitor as m
            from tests.test_ieee_tc_worker_observation import NativeHostFootprint
            # The parent function is used as a timing/reference implementation,
            # never imported into the live serving module or used as fallback.
            parent = subprocess.check_output(['git', 'show',
                '64fab5706f81aba73e2c5415dc6c02734efe6134:faaslora/memory/gpu_monitor.py'],
                cwd=ROOT, text=True)
            tree = ast.parse(parent)
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_ieee_lora_host_inventory')
            namespace = dict(m.__dict__)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen-D1081-inventory>', 'exec'),
                 namespace)
            reference = namespace['_ieee_lora_host_inventory']
            descriptor, identity = m.native_host_descriptor()
            assert descriptor is not None and identity['kind'] == 'fresh_native_tensor_descriptor_v1'
            assert identity['sha256'] == 'b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044'
            rows = []
            for count in (2, 32):
                source, original = NativeHostFootprint().native_models()
                models = {}
                for index in range(count):
                    template = original[7 + index % 2]
                    # Repeat existing tiny CPU module shapes, not model weights
                    # or a new artifact/workload. No tensor state is cached.
                    modules = {f'{layer}.{name}': SimpleNamespace(
                        lora_a=value.lora_a.clone(), lora_b=value.lora_b.clone())
                        for layer in range(32) for name, value in template.loras.items()}
                    models[index+1] = SimpleNamespace(id=index+1, rank=4, loras=modules)
                native = SimpleNamespace(list_adapters=lambda: dict(models))
                measured = []
                with patch.object(m, '_ieee_pinned_host_allocator_stats',
                                  return_value=(2**30, 2**30, 16, 16)):
                    for block in range(3):
                        for mode in (('parent', 'projection', 'default') if block % 2 == 0
                                     else ('default', 'projection', 'parent')):
                            wall, cpu = [], []
                            for _ in range(5):
                                start_cpu, start = time.process_time(), time.perf_counter()
                                function = reference if mode == 'parent' else m._ieee_lora_host_inventory
                                options = {'include_adapter_footprints': False} if mode == 'projection' else {}
                                inventory = function(native, include_tensor_views=False, **options)
                                summary = m._ieee_pinned_host_observation(inventory)
                                wall.append(1000*(time.perf_counter()-start))
                                cpu.append(1000*(time.process_time()-start_cpu))
                                expected = m._ieee_pinned_host_observation(reference(
                                    native, include_tensor_views=False))
                                assert summary == expected
                            measured.append(dict(block=block, mode=mode, cpu_ms=cpu, wall_ms=wall))
                rows.append(dict(adapter_objects=count, repeated_module_layers=32,
                                 measurements=measured))
            result.update(passed=True, rows=rows, descriptor_identity=identity,
                limitation='CPU-only tiny existing fixtures; allocator counters mocked equally. '
                    'Every native per-view descriptor still executes. No native pinned-memory '
                    'or complete serving performance conclusion.')
        result['source_refs'] = [dict(path=str(path), sha256=p.digest(path)) for path in (
            Path(__file__), ROOT/'faaslora/memory/gpu_monitor.py',
            ROOT/'tests/test_ieee_tc_worker_observation.py')]
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    result['resources'] = p.cgroup_snapshot(cg)
    result['memory_peak_bytes'] = int((cg/'memory.peak').read_text())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('mode', 'passed', 'memory_peak_bytes')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
