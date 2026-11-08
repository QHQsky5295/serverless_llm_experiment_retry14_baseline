"""Cross-run A2 handoff plan capture/replay remains auditable and strict."""
import json
import tempfile
import unittest
from pathlib import Path

from faaslora.preloading.preloading_planner import (
    PreparationCandidate, freeze_preparation_source_view)
from faaslora.registry.schema import StorageTier
from scripts.run_all_experiments import ScenarioRunner


def _plan(view):
    candidate = PreparationCandidate('adapter', StorageTier.REMOTE, StorageTier.GPU,
                                     4096, .5, 10., 0.)
    return dict(kind='ieee_preparation_epoch_v1', mode='handoff',
                source_snapshot_id=freeze_preparation_source_view(view)[1],
                profile_id='profile', cost_sequence=1, demand_observed_at=10.0,
                window_seconds=5.0, total_arrivals=2,
                arrival_counts={'adapter': 1},
                remaining_bytes={'gpu': 4096}, options=[], cost_estimates=[],
                physical_resources_reserved=False,
                selected={'gpu': (candidate,)}, source_view=view)


def _runner(*, capture=None, replay=None):
    runner = object.__new__(ScenarioRunner)
    runner._ieee_handoff_plan_capture_path = capture
    runner._ieee_handoff_plan_replay_path = replay
    runner._ieee_handoff_plan_control_records = []
    runner._ieee_handoff_plan_replay_index = 0
    runner._ieee_handoff_plan_replay_bundle = None
    return runner


class HandoffPlanControl(unittest.TestCase):
    def test_capture_finalize_and_replay_rebind(self):
        view = {'native': None, 'sources': {}, 'files': {}, 'activation_id': 'a'}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'handoff.json')
            capture = _runner(capture=path)
            capture._record_ieee_handoff_plan_capture(
                plan=_plan(view), activation_kind='initial', activation_index=0)
            capture._finalize_ieee_handoff_plan_capture()
            payload = json.loads(Path(path).read_text())
            self.assertTrue(payload['complete'])

            replay = _runner(replay=path)
            current = _plan({'native': None, 'sources': {}, 'files': {}, 'activation_id': 'b'})
            current['demand_observed_at'] = 22.0
            current['arrival_counts'] = {'adapter': 2}
            current['total_arrivals'] = 4
            rebound = replay._apply_ieee_handoff_plan_replay(
                plan=current,
                activation_kind='initial', activation_index=0)
            self.assertEqual(rebound['selected']['gpu'][0].artifact_id, 'adapter')
            self.assertTrue(replay._ieee_handoff_plan_control_summary()['complete'])

    def test_replay_rejects_logical_change(self):
        view = {'native': None, 'sources': {}, 'files': {}, 'activation_id': 'a'}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'handoff.json')
            capture = _runner(capture=path)
            capture._record_ieee_handoff_plan_capture(
                plan=_plan(view), activation_kind='initial', activation_index=0)
            replay = _runner(replay=path)
            changed = _plan({'native': None, 'sources': {'adapter': {'tier': 'host'}},
                             'files': {}, 'activation_id': 'b'})
            with self.assertRaises(ValueError):
                replay._apply_ieee_handoff_plan_replay(
                    plan=changed, activation_kind='initial', activation_index=0)


if __name__ == '__main__':
    unittest.main()
