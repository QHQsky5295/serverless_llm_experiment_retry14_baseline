"""Bounded option-representation study, reusing the planner's existing fixtures.

In-memory additional zero-demand IDs are CPU test values, not adapters/traces or
serving measurements. No GPU, remote access, or production worker switch here.
"""
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from tests.test_ieee_tc_planning_cpu import PositiveDemandOptions


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    out = Path(__file__).parent / 'component_result.json'
    assert not out.exists()
    case = PositiveDemandOptions()
    rows = []
    try:
        for mode in ('handoff', 'residency'):
            planner, args = case.inputs(mode, {'a': 3, 'd': 1})
            original = tuple(args['options'])
            # Same valid class/footprint, explicitly zero demand. No filesystem
            # artifact, file source, physical victim or generation is added.
            for count in (4, 100, 500):
                options = list(original)
                for i in range(count-4):
                    options.extend(replace(option, artifact_id=f'fixture-unused-{i}')
                                   for option in original if option.artifact_id == 'a')
                args['options'] = tuple(options)
                expected = planner.generate_ieee_epoch(**args)
                measurements = []
                for block in range(3):
                    for sparse in ((False, True) if block % 2 == 0 else (True, False)):
                        cpu, wall = [], []
                        for _ in range(5):
                            begin_cpu, begin = time.process_time(), time.perf_counter()
                            plan = planner.generate_ieee_epoch(**args, positive_demand_only=sparse)
                            plan['source_view'] = args['source_view']
                            selected = planner.validate_ieee_execution_plan(plan)
                            wall.append(1000*(time.perf_counter()-begin))
                            cpu.append(1000*(time.process_time()-begin_cpu))
                            assert selected == expected['selected']
                            assert plan['diagnostics'] == expected['diagnostics']
                        measurements.append(dict(block=block, positive_demand_only=sparse,
                            wall_ms=wall, cpu_ms=cpu, output_options=len(plan['options'])))
                rows.append(dict(mode=mode, logical_fixture_ids=count,
                    input_options=len(options), measurements=measurements))
    finally:
        case.doCleanups()
    cg = p.cg_path()
    result = dict(kind='d1081_option_representation_component_v1', rows=rows,
        source_refs=[dict(path=str(path), sha256=p.digest(path)) for path in (
            Path(__file__), ROOT/'faaslora/preloading/preloading_planner.py',
            ROOT/'tests/test_ieee_tc_planning_cpu.py')],
        scope=p.scope_identity(cg.name), memory_peak_bytes=int((cg/'memory.peak').read_text()),
        memory_events=(cg/'memory.events').read_text(),
        memory_swap_bytes=int((cg/'memory.swap.current').read_text()),
        limitation='Synthetic option multiplicity with existing four-adapter source fixtures. '
            'Not a 500-adapter native graph or end-to-end performance measurement.')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in rows:
        means = {}
        for sparse in (False, True):
            values = [v for m in row['measurements'] if m['positive_demand_only']==sparse for v in m['cpu_ms']]
            means[str(sparse)] = sum(values)/len(values)
        print(json.dumps(dict(mode=row['mode'], fixture_ids=row['logical_fixture_ids'], mean_cpu_ms=means)))


if __name__ == '__main__':
    main()
