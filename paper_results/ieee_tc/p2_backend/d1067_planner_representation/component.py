"""Reuse D193 retained graphs/alternating method for one local JSON decoder.

No production patch, new data, GPU work, or benchmark-sourced speedup claim.
"""
import copy
import hashlib
import inspect
import json
import math
import os
import pickle
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import msgspec
import torch
from scripts import ieee_tc_preflight as p
from faaslora.preloading import preloading_planner as planner
from faaslora.preloading.planning_cpu import execute_planning_message
from tests.test_ieee_tc_planning_cpu import PlanningCPU


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def typed_equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, float):
        assert struct.pack('!d', a) == struct.pack('!d', b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            typed_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            typed_equal(x, y)
    else:
        assert a == b


assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not torch.cuda.is_initialized()
OUT = HERE/'component'
OUT.mkdir(exist_ok=False)
scope = p.scope_identity(p.cg_path().name)
save(OUT/'identity.json', scope)
source = ROOT/'results/ieee_tc/p2_backend_qualification/d169_20261002/7b_probe1.json'
assert source.stat().st_size < 64*1024**2
assert p.digest(source) == 'ead0c5c81acc52c502874e9d45cfba111665f875220e23467c2e716a0d770602'
graphs = [row['sources_before'] for row in json.loads(source.read_bytes())['warm_batches']]
assert len(graphs) == 4
original = planner.freeze_preparation_source_view
namespace = dict(original.__globals__, json=SimpleNamespace(dumps=json.dumps, loads=msgspec.json.decode))
exec(compile(inspect.getsource(original), '<d1067-identical-function-local-decoder>', 'exec'), namespace)
candidate = namespace[original.__name__]
functions = dict(stdlib=original, msgspec=candidate)
boundary = []
valid = dict(integers=[0, -1, 2**64, 2**100, -2**100],
             floats=[0., -0., math.nextafter(0., 1.), sys.float_info.max, 0.1],
             unicode=['制品', '\U0001d11e', '\x00', '\\', '"'], flag=True, empty=None)
typed_equal(original(valid), candidate(valid))
boundary.append(dict(case='finite_native_values', equal=True))
cyclic = {}; cyclic['cycle'] = cyclic
for name, value in [('integer_key', {1: 'x'}), ('tuple', {'x': (1, 2)}),
                    ('nan', {'x': math.nan}), ('inf', {'x': math.inf}), ('cycle', cyclic)]:
    errors = {}
    for label, func in functions.items():
        try:
            func(value)
        except (ValueError, TypeError) as error:
            errors[label] = type(error).__name__
        else:
            raise AssertionError((label, name, 'accepted invalid input'))
    boundary.append(dict(case=name, errors=errors))
# A documented parser-domain difference must not be hidden as a fallback.
surrogates = {}
for label, func in functions.items():
    try:
        func({'path': '\ud800'})
    except (ValueError, TypeError) as error:
        surrogates[label] = type(error).__name__
    else:
        surrogates[label] = 'accepted'
boundary.append(dict(case='unpaired_surrogate', outcomes=surrogates,
    caveat='Not a universal replacement for all Python JSON strings; native RPC already requires valid Unicode.'))
rows, cases = [], []
for index, native in enumerate(graphs):
    # Same current registered-graph projection used by D193; includes all aliases.
    view = {key: value for key, value in native.items()
            if key not in ('native_staging_footprints', 'native_host_allocator')}
    view['native_footprints'] = {key: value for key, value in native['native_footprints'].items()
                               if key != 'host_tensor_views'}
    reference = original(view)
    typed_equal(reference, candidate(view))
    cases.append(dict(snapshot=index, registered=len(native['registered_cpu_adapter_ids']),
        canonical_bytes=len(json.dumps(view, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()),
        canonical_sha256=reference[1], all_values_types_order_float_bits_equal=True))
    for repeat in range(3):
        for label in (('stdlib', 'msgspec') if repeat % 2 == 0 else ('msgspec', 'stdlib')):
            wall, cpu = [], []
            for iteration in range(5):
                start, start_cpu = time.perf_counter(), time.thread_time()
                got = functions[label](view)
                wall.append((time.perf_counter()-start)*1000)
                cpu.append((time.thread_time()-start_cpu)*1000)
                typed_equal(reference, got)
            rows.append(dict(snapshot=index, decoder=label, repeat=repeat,
                wall_ms=wall, cpu_ms=cpu, mean_wall_ms=sum(wall)/len(wall)))
    detached, _ = candidate(view)
    detached['sources'][0]['adapter_id'] = 'mutated-detached-copy'
    assert native['sources'][0]['adapter_id'] != 'mutated-detached-copy'
fixtures = []
for mode in ('handoff', 'residency'):
    fixture = PlanningCPU()
    try:
        args, runner, slot = fixture.inputs(mode)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        for protected_ids in ([], [args['native_snapshot']['slot_adapter_ids'][0]]):
            args['native_snapshot']['replacement_protected_adapter_ids'] = protected_ids
            bundles = []
            for label, func in functions.items():
                with patch.object(planner, 'freeze_preparation_source_view', func), patch(
                        'faaslora.preloading.planning_cpu.uuid.uuid4') as uuid4:
                    uuid4.return_value.hex = 'd1067-fixture-only'
                    payload, receipt = execute_planning_message(pickle.dumps(
                        ('owned_execution_epoch', limit, args), protocol=5))
                bundles.append(pickle.loads(payload))
                assert receipt['frozen_execution_validated']
            typed_equal(bundles[0], bundles[1])
            fixtures.append(dict(mode=mode, protected=protected_ids,
                complete_bundle_and_all_hashes_equal=True))
    finally:
        fixture.doCleanups()
assert not torch.cuda.is_initialized()
save(OUT/'result.json', dict(kind='d1067_local_detachment_decoder_component_v1',
    scope=scope, msgspec_version=msgspec.__version__, python=sys.version,
    source=dict(path=str(source), sha256=p.digest(source)),
    script_sha256=p.digest(Path(__file__)), production_changed=False,
    source_function_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
    source_refs=[dict(path=str(q), sha256=p.digest(q)) for q in (
        ROOT/'faaslora/preloading/preloading_planner.py',
        ROOT/'faaslora/preloading/planning_cpu.py',
        ROOT/'tests/test_ieee_tc_planning_cpu.py')],
    cases=cases, measurements=rows, boundaries=boundary, fixtures=fixtures,
    caveat='Retained 7B native graphs, not current 3B full preparation epochs or serving speedups. No CI. No model, artifact, trace or remote changes.'))
cg=p.cg_path()
save(OUT/'resources.json', dict(scope=scope, peak_bytes=int((cg/'memory.peak').read_text()),
    events=(cg/'memory.events').read_text(), swap_bytes=int((cg/'memory.swap.current').read_text())))
print(json.dumps(dict(cases=cases, rows=len(rows), boundaries=boundary, fixtures=fixtures)))
