"""Cross-run A2 handoff plan capture/replay remains auditable and strict."""
import json
import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock
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
    def test_fixed_activation_replays_pending_capture_submissions(self):
        """A2 replay keeps capture submission frontiers while activations load."""
        runner = object.__new__(ScenarioRunner)

        class Pool:
            max_instances = 4
            min_instances = 1

        runner.instance_pool = Pool()
        runner._ieee_fixed_activation_schedule = {
            'events': [{
                'action': 'scale_up',
                'request_index': 20,
                'target_instances': 3,
                'reason': 'capture_pressure',
                'schedule_index': 1,
            }],
        }
        runner._ieee_fixed_activation_schedule_index = 0
        runner._ieee_fixed_activation_schedule_records = []
        calls = []

        def submit(**kwargs):
            calls.append(kwargs)
            return 1

        runner._schedule_background_scale_up_instance_pool = submit
        runner._refresh_ieee_deferred_host_capacity = AsyncMock()
        runner._schedule_ieee_residency_epochs = AsyncMock()

        changed = asyncio.run(runner._evaluate_ieee_fixed_activation_schedule(
            result=object(), coord_enabled=True, replay_t0=0.0,
            backlog=2, active_requests=1, completed_count=18,
            arrived=20, ready=1, pending=1, draining_active=0))

        self.assertTrue(changed)
        self.assertEqual(runner._ieee_fixed_activation_schedule_index, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['target_instances'], 3)
        self.assertEqual(calls[0]['startup_parallelism_limit'], 2)

    def test_fixed_activation_keeps_retirement_barrier(self):
        """Scale-down is not issued while a prior activation is pending."""
        runner = object.__new__(ScenarioRunner)

        class Pool:
            max_instances = 4
            min_instances = 1

        runner.instance_pool = Pool()
        runner._ieee_fixed_activation_schedule = {
            'events': [{
                'action': 'scale_down',
                'request_index': 124,
                'reason': 'capture_release',
                'schedule_index': 3,
            }],
        }
        runner._ieee_fixed_activation_schedule_index = 0
        runner._ieee_fixed_activation_schedule_records = []
        runner._refresh_ieee_deferred_host_capacity = AsyncMock()
        runner._schedule_ieee_residency_epochs = AsyncMock()

        changed = asyncio.run(runner._evaluate_ieee_fixed_activation_schedule(
            result=object(), coord_enabled=True, replay_t0=0.0,
            backlog=2, active_requests=1, completed_count=120,
            arrived=124, ready=4, pending=1, draining_active=0))

        self.assertFalse(changed)
        self.assertEqual(runner._ieee_fixed_activation_schedule_index, 0)
        runner._refresh_ieee_deferred_host_capacity.assert_not_awaited()

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

    def test_replay_rejects_profile_change(self):
        view = {'native': None, 'sources': {}, 'files': {}, 'activation_id': 'a'}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'handoff.json')
            capture = _runner(capture=path)
            capture._record_ieee_handoff_plan_capture(
                plan=_plan(view), activation_kind='initial', activation_index=0)
            replay = _runner(replay=path)
            changed = _plan({'native': None, 'sources': {'adapter': {'tier': 'host'}},
                             'files': {}, 'activation_id': 'b'})
            changed['profile_id'] = 'different-profile'
            with self.assertRaises(ValueError):
                replay._apply_ieee_handoff_plan_replay(
                    plan=changed, activation_kind='initial', activation_index=0)

    def test_replay_allows_live_owner_rebind_but_records_difference(self):
        view = {'native': None, 'sources': {}, 'files': {}, 'activation_id': 'a'}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'handoff.json')
            capture = _runner(capture=path)
            capture._record_ieee_handoff_plan_capture(
                plan=_plan(view), activation_kind='initial', activation_index=0)
            replay = _runner(replay=path)
            changed = _plan({'native': None, 'sources': {'adapter': {'tier': 'host'}},
                             'files': {}, 'activation_id': 'b'})
            rebound = replay._apply_ieee_handoff_plan_replay(
                plan=changed, activation_kind='initial', activation_index=0)
            self.assertEqual(rebound['selected']['gpu'][0].artifact_id, 'adapter')
            record = replay._ieee_handoff_plan_control_records[0]
            self.assertNotEqual(record['frozen_source_view_comparison_sha256'],
                                record['live_source_view_comparison_sha256'])


if __name__ == '__main__':
    unittest.main()
