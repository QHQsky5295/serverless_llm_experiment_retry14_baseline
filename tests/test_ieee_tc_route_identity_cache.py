"""CPU-only qualification for the opt-in confirmed-route footprint cache."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import InstanceSlot, NativeSourceSnapshot
from scripts.run_all_experiments import ScenarioRunner
from tests.test_ieee_tc_request_footprints import full_payload, scoped_payload


class RouteIdentityCacheQualification(unittest.TestCase):
    def _runner(self):
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._ieee_source_observation_waves = {}
        runner._ieee_source_observation_stats = {}
        runner._ieee_route_full_cache = {}
        runner._ieee_route_cache_stats = {}
        slot = InstanceSlot('inst-a', engine=SimpleNamespace(), coordinator=None)
        runner.instance_pool = SimpleNamespace(get_slots=lambda: [slot])
        clock = local_monotonic_clock_id()

        identity_payload = full_payload()
        identity_payload['clock_id'] = clock
        identity_payload['device_uuid'] = 'GPU-00000000-0000-0000-0000-000000000001'
        identity = NativeSourceSnapshot.from_native(
            identity_payload, expected_clock_id=clock,
            received_monotonic_s=time.monotonic())
        identity_wire = identity.identity_view().routing_wire(
            device_uuid=identity_payload['device_uuid'])

        scoped = scoped_payload((4,))
        scoped['clock_id'] = clock
        scoped['device_uuid'] = identity_payload['device_uuid']
        full = NativeSourceSnapshot.from_request_native(
            scoped, requested_adapter_ids=(4,), expected_clock_id=clock,
            received_monotonic_s=time.monotonic())
        full_wire = full.request_wire(device_uuid=identity_payload['device_uuid'])
        slot.engine.ieee_source_identities = AsyncMock(return_value=identity_wire)
        slot.engine.ieee_request_sources = AsyncMock(return_value=full_wire)
        return runner, slot, identity_wire, full_wire

    def test_matching_identity_reuses_measured_footprint(self):
        runner, slot, _, _ = self._runner()

        async def check():
            first, first_meta = await runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4])
            second, second_meta = await runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4])
            self.assertEqual(slot.engine.ieee_source_identities.await_count, 2)
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
            self.assertIsNotNone(first[0][0].sources[0].host_storage_bytes)
            self.assertIsNotNone(second[0][0].sources[0].gpu_slot_capacity_bytes)
            self.assertEqual(first_meta['full_fallback_replica_count'], 1)
            self.assertEqual(second_meta['mode'], 'identity_recheck_cache_v1')
            self.assertGreaterEqual(runner._ieee_route_cache_stats['hits'], 1)

        asyncio.run(check())

    def test_epoch_change_forces_authoritative_full_read(self):
        runner, slot, identity_wire, full_wire = self._runner()
        asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))

        changed_identity = dict(identity_wire, epoch=2, captured_monotonic_s=time.monotonic())
        changed_full = dict(full_wire, epoch=2, captured_monotonic_s=time.monotonic())
        slot.engine.ieee_source_identities = AsyncMock(return_value=changed_identity)
        slot.engine.ieee_request_sources = AsyncMock(return_value=changed_full)
        values, metadata = asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))
        self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
        self.assertEqual(metadata['full_fallback_replica_count'], 1)
        self.assertEqual(values[0][0].epoch, 2)


if __name__ == '__main__':
    unittest.main()
