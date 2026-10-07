"""CPU-only qualification for the opt-in confirmed-route footprint cache."""

import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import InstanceSlot, NativeSourceSnapshot
from scripts.run_all_experiments import ScenarioRunner
from scripts.analyze_control_path_overhead import (
    summarize_route_evidence, analyze_route_evidence, summarize_admission_waits,
    summarize_route_copy_revisits)
from tests.test_ieee_tc_request_footprints import full_payload, scoped_payload


class RouteIdentityCacheQualification(unittest.TestCase):
    def _planning_payload(self):
        value = full_payload()
        value['clock_id'] = local_monotonic_clock_id()
        value['device_uuid'] = 'GPU-00000000-0000-0000-0000-000000000001'
        return value

    def test_planning_measurement_seeds_gpu_and_host_without_second_rpc(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        payload = self._planning_payload()
        original = copy.deepcopy(payload)
        self.assertTrue(runner._ieee_publish_planning_sources(
            slot, payload, expected_engine=slot.engine))
        full = slot.native_source_state
        self.assertEqual(payload, original)
        async def check():
            for aid in (4, 5):
                values, meta = await runner._ieee_collect_confirmed_snapshot_cache(
                    (slot,), requested_adapter_ids=[aid])
                self.assertEqual(values[0][0], full.request_view([aid]))
                self.assertEqual(meta['full_authoritative_read_replica_count'], 0)
                self.assertEqual(meta['selected_revalidation'], 'owner_epoch_or_selected_identity_v1')
                self.assertIsNone(values[0][0].host_tensor_storage_bytes)
            slot.engine.ieee_request_sources.assert_not_awaited()
            slot.engine.ieee_source_identities.assert_not_awaited()
        asyncio.run(check())
        self.assertEqual(runner._ieee_route_cache_stats['planning_snapshot_hits'], 2)
        self.assertEqual(runner._ieee_route_cache_stats['planning_snapshot_seeded_scopes'], 2)
        self.assertEqual(full.host_tensor_storage_bytes, 1536)
        self.assertEqual(full.captured_monotonic_s, payload['captured_monotonic_s'])

    def test_planning_scopes_preserve_existing_gpu_copy_and_host_invalidation_rules(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        runner._ieee_publish_planning_sources(slot, self._planning_payload(), expected_engine=slot.engine)
        gpu = runner._ieee_route_full_cache[('inst-a', (4,))]
        fresh = replace(slot.native_source_state.identity_view(), epoch=2,
            captured_monotonic_s=time.monotonic())
        runner._ieee_publish_identity_frontier(slot, fresh)
        self.assertIs(runner._ieee_route_full_cache[('inst-a', (4,))], gpu)
        self.assertNotIn(('inst-a', (5,)), runner._ieee_route_full_cache)
        # Use the actual fixture's integer scope for direct invalidation:
        with unittest.mock.patch('scripts.run_all_experiments.InferenceEngine._lora_int_id', return_value=4):
            runner._ieee_invalidate_route_scope(slot, 'fixture-adapter')
        self.assertNotIn(('inst-a', (4,)), runner._ieee_route_full_cache)

    def test_late_planning_result_cannot_overwrite_newer_identity(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        old = self._planning_payload()
        runner._ieee_publish_planning_sources(slot, old, expected_engine=slot.engine)
        fresh = replace(slot.native_source_state.identity_view(), epoch=2,
            captured_monotonic_s=time.monotonic())
        runner._ieee_publish_identity_frontier(slot, fresh)
        published = runner._ieee_route_identity_frontier['inst-a']
        before = dict(runner._ieee_route_full_cache)
        self.assertFalse(runner._ieee_publish_planning_sources(slot, old, expected_engine=slot.engine))
        self.assertEqual(runner._ieee_route_full_cache, before)
        self.assertIs(runner._ieee_route_identity_frontier['inst-a'], published)
        self.assertEqual(published, fresh)

    def test_same_epoch_late_planning_result_keeps_newer_frontier(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        old = self._planning_payload()
        runner._ieee_publish_planning_sources(slot, old, expected_engine=slot.engine)
        fresh = replace(slot.native_source_state.identity_view(), captured_monotonic_s=time.monotonic())
        runner._ieee_publish_identity_frontier(slot, fresh)
        published = runner._ieee_route_identity_frontier['inst-a']
        self.assertFalse(runner._ieee_publish_planning_sources(slot, old, expected_engine=slot.engine))
        self.assertIs(runner._ieee_route_identity_frontier['inst-a'], published)
        self.assertEqual(published, fresh)

    def test_planning_publication_requires_enabled_current_owner(self):
        for case in ('disabled', 'removed', 'engine_replaced'):
            with self.subTest(case=case):
                runner, slot, _, _ = self._runner()
                runner._ieee_route_confirmed_snapshot_cache = case != 'disabled'
                expected = slot.engine
                if case == 'removed':
                    runner.instance_pool.get_slots = lambda: []
                if case == 'engine_replaced':
                    slot.engine = SimpleNamespace()
                self.assertFalse(runner._ieee_publish_planning_sources(
                    slot, self._planning_payload(), expected_engine=expected))
                self.assertFalse(runner._ieee_route_full_cache)
                self.assertIsNone(slot.native_source_state)

    def test_planning_invalid_measurement_cannot_partially_publish(self):
        mutations = (
            lambda p: p.pop('device_uuid'),
            lambda p: p['native_footprints'].__setitem__('host_tensor_storage_bytes', 0),
            lambda p: p['native_footprints']['host_allocations'][0].__setitem__('adapter_ids', []),
        )
        for mutate in mutations:
            runner, slot, _, _ = self._runner()
            runner._ieee_route_confirmed_snapshot_cache = True
            p = self._planning_payload()
            mutate(p)
            with self.assertRaises(ValueError):
                runner._ieee_publish_planning_sources(slot, p, expected_engine=slot.engine)
            self.assertFalse(runner._ieee_route_full_cache)
            self.assertIsNone(slot.native_source_state)

    def test_planning_physical_owner_change_is_not_silently_accepted(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        slot.native_device_uuid = 'GPU-other'
        with self.assertRaisesRegex(ValueError, 'physical GPU changed'):
            runner._ieee_publish_planning_sources(
                slot, self._planning_payload(), expected_engine=slot.engine)
        self.assertFalse(runner._ieee_route_full_cache)

    def test_planning_without_measurement_does_not_enter_service_cache(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        payload = self._planning_payload()
        payload.pop('native_footprints')
        with self.assertRaisesRegex(ValueError, 'full measured'):
            runner._ieee_publish_planning_sources(slot, payload, expected_engine=slot.engine)
        self.assertFalse(runner._ieee_route_full_cache)

    def test_planning_same_epoch_footprint_conflict_is_not_hidden(self):
        runner, slot, _, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        p = self._planning_payload()
        runner._ieee_publish_planning_sources(slot, p, expected_engine=slot.engine)
        before = dict(runner._ieee_route_full_cache)
        p['native_footprints']['host_allocations'][0]['pinned'] = True
        with self.assertRaisesRegex(ValueError, 'source (state|footprint)'):
            runner._ieee_publish_planning_sources(slot, p, expected_engine=slot.engine)
        self.assertEqual(before, runner._ieee_route_full_cache)

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
            # Cold scopes use one authoritative read; identity recheck begins
            # only after a scoped footprint has been cached.
            self.assertEqual(slot.engine.ieee_source_identities.await_count, 1)
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
            self.assertIsNotNone(first[0][0].sources[0].host_storage_bytes)
            self.assertIsNotNone(second[0][0].sources[0].gpu_slot_capacity_bytes)
            self.assertEqual(first_meta['full_authoritative_read_replica_count'], 1)
            self.assertEqual(first_meta['cold_full_read_replica_count'], 1)
            self.assertEqual(first_meta['full_fallback_replica_count'], 0)
            self.assertEqual(first_meta['collection_id'], 1)
            self.assertEqual(second_meta['mode'], 'identity_recheck_cache_v1')
            self.assertEqual(second_meta['collection_id'], 2)
            self.assertGreaterEqual(runner._ieee_route_cache_stats['hits'], 1)

        asyncio.run(check())

    def test_concurrent_scopes_keep_distinct_ids_and_joined_waiters_share_one(self):
        runner, slot, _, full_wire = self._runner()

        async def check():
            entered = {4: asyncio.Event(), 5: asyncio.Event()}
            release = asyncio.Event()

            async def read(*, requested_adapter_ids):
                scope = tuple(requested_adapter_ids)
                entered[scope[0]].set()
                await release.wait()
                payload = scoped_payload(scope)
                payload['clock_id'] = local_monotonic_clock_id()
                state = NativeSourceSnapshot.from_request_native(
                    payload, requested_adapter_ids=scope,
                    expected_clock_id=local_monotonic_clock_id(),
                    received_monotonic_s=time.monotonic())
                return state.request_wire(device_uuid=full_wire['device_uuid'])

            slot.engine.ieee_request_sources = AsyncMock(side_effect=read)
            first = asyncio.create_task(runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4]))
            await entered[4].wait()
            second = asyncio.create_task(runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[5]))
            await entered[5].wait()
            joined = asyncio.create_task(runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4]))
            await asyncio.sleep(0)
            release.set()
            a, b, c = await asyncio.gather(first, second, joined)
            self.assertEqual((a[1]['collection_id'], b[1]['collection_id'],
                              c[1]['collection_id']), (1, 2, 1))
            self.assertTrue(c[1]['joined'])
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 2)
            self.assertEqual(runner._ieee_source_observation_stats['collections'], 2)
            self.assertEqual(runner._ieee_source_observation_stats['joined'], 1)
            self.assertEqual(runner._ieee_source_observation_waves, {})

        asyncio.run(asyncio.wait_for(check(), timeout=2))

    def test_epoch_change_forces_authoritative_full_read(self):
        runner, slot, identity_wire, full_wire = self._runner()
        asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))

        changed_identity = dict(identity_wire, epoch=2, captured_monotonic_s=time.monotonic())
        changed_identity['sources'] = [dict(row) for row in identity_wire['sources']]
        changed_identity['sources'][0]['source_id'] = 'target-incarnation-changed'
        changed_full = dict(full_wire, epoch=2, captured_monotonic_s=time.monotonic())
        slot.engine.ieee_source_identities = AsyncMock(return_value=changed_identity)
        slot.engine.ieee_request_sources = AsyncMock(return_value=changed_full)
        values, metadata = asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))
        self.assertEqual(slot.engine.ieee_source_identities.await_count, 1)
        # This mock was replaced after the first cold read; the replacement
        # records the single epoch-change fallback.
        self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
        self.assertEqual(metadata['full_fallback_replica_count'], 1)
        self.assertEqual(values[0][0].epoch, 2)

    def test_scope_local_gpu_identity_reuses_footprint_across_unrelated_epoch(self):
        runner, slot, identity_wire, _ = self._runner()
        asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))
        initial_full_reads = slot.engine.ieee_request_sources.await_count

        # The target GPU copy is unchanged, while an unrelated HOST source is
        # replaced and the owner epoch advances.  A global epoch comparison
        # would reject this, although the fresh identity still proves the
        # target copy and the cached request footprint is complete.
        changed = dict(identity_wire, epoch=2, captured_monotonic_s=time.monotonic())
        changed['sources'] = [dict(row) for row in identity_wire['sources']]
        changed['sources'][1]['source_id'] = 'unrelated-incarnation'
        slot.engine.ieee_source_identities = AsyncMock(return_value=changed)
        values, metadata = asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))

        self.assertEqual(slot.engine.ieee_request_sources.await_count, initial_full_reads)
        self.assertEqual(values[0][0].epoch, 2)
        self.assertEqual(values[0][0].sources[0].gpu_slot_capacity_bytes, 1024)
        self.assertEqual(runner._ieee_route_cache_stats['scope_local_hits'], 1)
        self.assertEqual(metadata['full_fallback_replica_count'], 0)

    def test_scope_local_gpu_identity_rejects_target_copy_change(self):
        runner, slot, identity_wire, full_wire = self._runner()
        asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))

        changed = dict(identity_wire, epoch=2, captured_monotonic_s=time.monotonic())
        changed['sources'] = [dict(row) for row in identity_wire['sources']]
        changed['sources'][0]['source_id'] = 'target-incarnation-changed'
        slot.engine.ieee_source_identities = AsyncMock(return_value=changed)
        slot.engine.ieee_request_sources = AsyncMock(return_value=dict(
            full_wire, epoch=2, captured_monotonic_s=time.monotonic()))
        values, metadata = asyncio.run(runner._ieee_collect_route_identity_cache(
            (slot,), requested_adapter_ids=[4]))

        self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
        self.assertEqual(metadata['full_fallback_replica_count'], 1)
        self.assertEqual(values[0][0].epoch, 2)
        self.assertEqual(runner._ieee_route_cache_stats.get('scope_local_hits', 0), 0)

    def test_membership_change_during_fallback_does_not_publish_cache(self):
        runner, slot, identity_wire, full_wire = self._runner()

        async def check():
            await runner._ieee_collect_route_identity_cache((slot,), requested_adapter_ids=[4])
            cached = runner._ieee_route_full_cache[('inst-a', (4,))]
            changed_identity = dict(identity_wire, epoch=2, captured_monotonic_s=time.monotonic())
            changed_identity['sources'] = [dict(row) for row in identity_wire['sources']]
            changed_identity['sources'][0]['source_id'] = 'target-incarnation-changed'
            slot.engine.ieee_source_identities = AsyncMock(return_value=changed_identity)

            async def changed_membership(**kwargs):
                runner.instance_pool.get_slots = lambda: []
                return dict(full_wire, epoch=2, captured_monotonic_s=time.monotonic())

            slot.engine.ieee_request_sources = AsyncMock(side_effect=changed_membership)
            values, metadata = await runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4])
            self.assertIsNone(values)
            self.assertTrue(metadata['membership_rejected'])
            self.assertIs(runner._ieee_route_full_cache[('inst-a', (4,))], cached)

        asyncio.run(check())

    def test_identity_failure_uses_authoritative_read_without_stale_reuse(self):
        runner, slot, _, full_wire = self._runner()

        async def check():
            await runner._ieee_collect_route_identity_cache((slot,), requested_adapter_ids=[4])
            slot.engine.ieee_source_identities = AsyncMock(side_effect=RuntimeError('unavailable'))
            slot.engine.ieee_request_sources = AsyncMock(return_value=dict(
                full_wire, epoch=2, captured_monotonic_s=time.monotonic()))
            values, metadata = await runner._ieee_collect_route_identity_cache(
                (slot,), requested_adapter_ids=[4])
            self.assertEqual(values[0][0].epoch, 2)
            self.assertTrue(metadata['identity_failure'])
            self.assertEqual(metadata['full_fallback_replica_count'], 1)
            self.assertEqual(runner._ieee_route_cache_stats.get('hits', 0), 0)

        asyncio.run(check())

    def test_confirmed_snapshot_cache_reuses_only_authoritative_entry(self):
        runner, slot, _, full_wire = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True

        async def check():
            first, first_meta = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            # A second route view uses the previously authoritative scoped
            # observation and does not issue an identity RPC or a duplicate
            # full request.  The selected-source transaction is tested by the
            # integration campaign, not inferred from this collector test.
            second, second_meta = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
            self.assertEqual(slot.engine.ieee_source_identities.await_count, 0)
            self.assertEqual(first_meta['full_authoritative_read_replica_count'], 1)
            self.assertEqual(second_meta['confirmed_snapshot_cache_hit_replica_count'], 1)
            self.assertEqual(second_meta['full_authoritative_read_replica_count'], 0)
            self.assertEqual(first[0][0].identity_view(), second[0][0].identity_view())
            self.assertEqual(second_meta['selected_revalidation'],
                             'owner_epoch_or_selected_identity_v1')

        asyncio.run(check())

    def test_confirmed_snapshot_cache_rejects_missing_device_identity(self):
        runner, slot, _, full_wire = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        bad = dict(full_wire)
        bad.pop('device_uuid', None)
        slot.engine.ieee_request_sources = AsyncMock(return_value=bad)
        with self.assertRaises(ValueError):
            asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4]))

    def test_frontier_publication_preserves_just_received_scoped_read(self):
        runner, slot, _, _ = self._runner()

        async def check():
            values, _ = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            snapshot, device_uuid = values[0]
            # Production publishes the identity AFTER the collector installs
            # its authoritative scoped entry. This ordering was not covered
            # by the former collector-only repeated-read test.
            slot.commit_native_sources(snapshot)
            slot.native_device_uuid = device_uuid
            self.assertTrue(runner._ieee_publish_identity_frontier(slot, snapshot))
            self.assertIn(('inst-a', (4,)), runner._ieee_route_full_cache)
            second, meta = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
            self.assertEqual(meta['full_authoritative_read_replica_count'], 0)
            self.assertEqual(meta['confirmed_snapshot_cache_hit_replica_count'], 1)
            self.assertEqual(second[0][0], snapshot)
            self.assertEqual(meta['selected_revalidation'],
                             'owner_epoch_or_selected_identity_v1')

        asyncio.run(check())

    def test_frontier_invalidates_old_identity_but_keeps_matching_new_scope(self):
        runner, slot, _, _ = self._runner()
        values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
            (slot,), requested_adapter_ids=[4]))
        old = values[0][0]
        fresh = replace(old, epoch=old.epoch + 1,
                        captured_monotonic_s=time.monotonic())
        old_entry = runner._ieee_route_full_cache[('inst-a', (4,))]
        fresh_entry = dict(old_entry, snapshot=fresh)
        runner._ieee_route_full_cache.update({
            ('inst-a', (5,)): old_entry,
            ('inst-a', (4,)): fresh_entry,
            ('inst-b', (4,)): old_entry,
            ('inst-a', (99,)): {'snapshot': None},
        })
        self.assertTrue(runner._ieee_publish_identity_frontier(slot, fresh))
        self.assertEqual(set(runner._ieee_route_full_cache),
                         {('inst-a', (4,)), ('inst-b', (4,))})
        self.assertIs(runner._ieee_route_full_cache[('inst-a', (4,))], fresh_entry)
        self.assertFalse(runner._ieee_publish_identity_frontier(slot, old))
        self.assertIs(runner._ieee_route_full_cache[('inst-a', (4,))], fresh_entry)

    def test_frontier_retains_original_gpu_scope_across_unrelated_epoch(self):
        runner, slot, _, _ = self._runner()
        async def check():
            values, _ = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            old = values[0][0]
            entry = runner._ieee_route_full_cache[('inst-a', (4,))]
            fresh = replace(old.identity_view(), epoch=old.epoch+1,
                captured_monotonic_s=time.monotonic(), sources=tuple(
                    (replace(s, source_id='unrelated-new-copy') if s.adapter_int_id == 5
                     else replace(s, gpu_confirmed_monotonic_s=time.monotonic()))
                    for s in old.identity_view().sources))
            self.assertTrue(runner._ieee_publish_identity_frontier(slot, fresh))
            self.assertIs(runner._ieee_route_full_cache[('inst-a', (4,))], entry)
            again, metadata = await runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4])
            # Keep the actual measured token; never claim the old measurement
            # happened at the new epoch/time or grants physical capacity.
            self.assertIs(again[0][0], old)
            self.assertEqual(again[0][0].epoch, old.epoch)
            self.assertEqual(runner._ieee_route_identity_frontier['inst-a'], fresh)
            self.assertEqual(slot.engine.ieee_request_sources.await_count, 1)
            self.assertEqual(metadata['full_authoritative_read_replica_count'], 0)
            self.assertEqual(metadata['identity_frontier_retained_gpu_scopes'], 1)
        asyncio.run(check())

    def test_frontier_rejects_changed_target_slot_rank_or_path(self):
        for mutation in ({'source_id':'new-target'}, {'gpu_slot':None},
                         {'gpu_slot':1}, {'rank':32}, {'lora_path':'/other'}):
            runner, slot, _, _ = self._runner()
            values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4]))
            old = values[0][0]
            fresh = replace(old.identity_view(), epoch=old.epoch+1, sources=tuple(
                replace(s, **mutation) if s.adapter_int_id == 4 else s
                for s in old.identity_view().sources))
            with self.subTest(mutation=mutation):
                self.assertTrue(runner._ieee_publish_identity_frontier(slot, fresh))
                self.assertNotIn(('inst-a', (4,)), runner._ieee_route_full_cache)

    def test_frontier_rejects_target_unknown_unconfirmed_or_missing_registration(self):
        for update in ({'unknown_native_adapter_ids':(4,)},
                       {'unconfirmed_gpu_adapter_ids':(4,)}, {'registered_cpu_adapter_ids':(5,)}):
            runner, slot, _, _ = self._runner()
            values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[4]))
            old = values[0][0]
            fresh = replace(old.identity_view(), epoch=old.epoch+1, **update)
            with self.subTest(update=update):
                self.assertTrue(runner._ieee_publish_identity_frontier(slot, fresh))
                self.assertNotIn(('inst-a', (4,)), runner._ieee_route_full_cache)

    def test_frontier_cannot_confirm_a_measurement_from_a_future_epoch(self):
        runner, slot, _, _ = self._runner()
        values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
            (slot,), requested_adapter_ids=[4]))
        old = values[0][0]
        runner._ieee_route_full_cache[('inst-a', (4,))]['snapshot'] = replace(old, epoch=old.epoch+1)
        self.assertTrue(runner._ieee_publish_identity_frontier(slot, old.identity_view()))
        self.assertNotIn(('inst-a', (4,)), runner._ieee_route_full_cache)

    def test_frontier_does_not_retain_host_footprints_across_epochs(self):
        runner, slot, _, _ = self._runner()
        payload = scoped_payload((5,))
        payload['clock_id'] = local_monotonic_clock_id()
        old = NativeSourceSnapshot.from_request_native(payload, requested_adapter_ids=(5,),
            expected_clock_id=local_monotonic_clock_id(), received_monotonic_s=time.monotonic())
        runner._ieee_route_full_cache[('inst-a', (5,))] = {'snapshot':old, 'device_uuid':'GPU-same'}
        fresh = replace(old.identity_view(), epoch=old.epoch+1)
        self.assertTrue(runner._ieee_publish_identity_frontier(slot, fresh))
        self.assertNotIn(('inst-a', (5,)), runner._ieee_route_full_cache)

    def test_frontier_never_retains_same_epoch_different_owner_or_source(self):
        runner, slot, _, _ = self._runner()
        values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
            (slot,), requested_adapter_ids=[4]))
        state = values[0][0]
        entry = runner._ieee_route_full_cache[('inst-a', (4,))]
        for changed in (
                replace(state, owner_id='different-owner'),
                replace(state, clock_id='different-clock'),
                replace(state, sources=(replace(state.sources[0],
                    source_id='different-copy'),) + state.sources[1:])):
            with self.subTest(changed=changed):
                runner._ieee_route_identity_frontier = {}
                runner._ieee_route_full_cache = {
                    ('inst-a', (4,)): dict(entry, snapshot=changed)}
                self.assertTrue(runner._ieee_publish_identity_frontier(slot, state))
                self.assertNotIn(('inst-a', (4,)), runner._ieee_route_full_cache)

    def test_same_epoch_conflicting_frontier_fails_without_mutating_cache(self):
        runner, slot, _, _ = self._runner()
        values, _ = asyncio.run(runner._ieee_collect_confirmed_snapshot_cache(
            (slot,), requested_adapter_ids=[4]))
        state = values[0][0]
        runner._ieee_route_identity_frontier = {slot.instance_id: state.identity_view()}
        saved = dict(runner._ieee_route_full_cache)
        changed = replace(state, sources=(replace(state.sources[0],
            source_id='different-copy'),) + state.sources[1:])
        with self.assertRaisesRegex(ValueError, 'same native epoch'):
            runner._ieee_publish_identity_frontier(slot, changed)
        self.assertEqual(runner._ieee_route_full_cache, saved)

    def test_event_frontier_reuses_confirmed_absence_without_rpc(self):
        runner, slot, identity_wire, _ = self._runner()
        runner._ieee_route_confirmed_snapshot_cache = True
        identity = NativeSourceSnapshot.from_routing_wire(
            identity_wire,
            expected_clock_id=identity_wire['clock_id'],
            received_monotonic_s=time.monotonic(),
        )
        runner._ieee_route_identity_frontier = {slot.instance_id: identity}
        slot.native_device_uuid = identity_wire['device_uuid']
        values, metadata = asyncio.run(
            runner._ieee_collect_confirmed_snapshot_cache(
                (slot,), requested_adapter_ids=[99]))
        self.assertEqual(slot.engine.ieee_request_sources.await_count, 0)
        self.assertEqual(metadata['frontier_absence_replica_indexes'], [0])
        self.assertEqual(values[0][0].footprint_adapter_ids, (99,))
        self.assertEqual(values[0][1], identity_wire['device_uuid'])


class RouteEvidenceAudit(unittest.TestCase):
    def _requests(self):
        result = []
        for index in range(20):
            row = dict(collection_id=1, identity_replica_count=1,
                       full_authoritative_read_replica_count=index % 2,
                       cold_full_read_replica_count=0,
                       full_fallback_replica_count=index % 2,
                       cache_hits=7, identity_only=3)
            result.append(dict(request_id=str(index), control_path_total_us=1000*(index+1),
                gpu_reference_evidence={'routing_snapshot': {'sources': {
                    'a': {'source_collection': row}, 'b': {'source_collection': dict(row)}}}}))
        return result

    def test_admission_partition_keeps_nested_routing_separate(self):
        rows = self._requests()
        for row in rows:
            row.update(arrival_release_lateness_ms=2., dispatch_window_wait_ms=3.,
                runtime_slot_wait_ms=25., dispatch_admission_wait_ms=30.)
        result = summarize_admission_waits(rows)
        all_metrics = {m['field']:m for m in result['metrics'] if m['group'] == 'all'}
        self.assertEqual(all_metrics['dispatch_admission_wait_ms']['mean'], 30.)
        self.assertEqual(all_metrics['routing_ms']['p95_type1'], 19.)
        self.assertEqual(result['max_partition_error_ms'], 0.)
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_comparison_qualified'])

    def test_admission_partition_rejects_missing_invalid_and_inconsistent_fields(self):
        for mutation in (
                lambda r: r.pop('runtime_slot_wait_ms'),
                lambda r: r.update(runtime_slot_wait_ms=float('nan')),
                lambda r: r.update(dispatch_admission_wait_ms=32.),
                lambda r: r.update(dispatch_window_wait_ms=-1.),
                lambda r: r.update(arrival_release_lateness_ms=True)):
            rows = self._requests()
            for row in rows:
                row.update(arrival_release_lateness_ms=2., dispatch_window_wait_ms=3.,
                    runtime_slot_wait_ms=25., dispatch_admission_wait_ms=30.)
            mutation(rows[0])
            with self.subTest(mutation=mutation), self.assertRaises((KeyError, ValueError)):
                summarize_admission_waits(rows)

    def test_replicated_counters_and_colliding_labels_are_not_event_totals(self):
        result = summarize_route_evidence(self._requests())
        self.assertEqual(result['source_snapshot_count'], 40)
        self.assertEqual(result['observed_collection_labels'], 1)
        self.assertEqual(result['labels_with_conflicting_wave_counts'], [1])
        self.assertEqual(result['observed_cumulative_counter_maxima']['cache_hits'], 7)
        self.assertIsNone(result['exact_rpc_count'])
        self.assertEqual(result['conditional_request_groups']['with_reported_fallback']['n_requests'], 10)

    def test_type1_percentile_not_legacy_interpolation(self):
        result = summarize_route_evidence(self._requests())
        self.assertEqual(result['control_path']['p95_type1_ms'], 19)
        self.assertEqual(result['control_path']['mean_ms'], 10.5)

    def test_missing_duplicate_or_invalid_evidence_fails_closed(self):
        mutations = (
            lambda rows: rows[0].pop('control_path_total_us'),
            lambda rows: rows[0].update(control_path_total_us=float('nan')),
            lambda rows: rows[0].update(request_id=rows[1]['request_id']),
            lambda rows: rows[0]['gpu_reference_evidence']['routing_snapshot'].update(sources={}),
            lambda rows: rows[0]['gpu_reference_evidence']['routing_snapshot']['sources']['a']['source_collection'].update(cache_hits=-1),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self._requests())
                mutation(rows)
                with self.assertRaises((KeyError, ValueError)):
                    summarize_route_evidence(rows)

    def test_empty_group_is_unknown_not_zero(self):
        rows = self._requests()
        for row in rows:
            for source in row['gpu_reference_evidence']['routing_snapshot']['sources'].values():
                source['source_collection']['full_fallback_replica_count'] = 0
        empty = summarize_route_evidence(rows)['conditional_request_groups']['with_reported_fallback']
        self.assertEqual(empty['n_requests'], 0)
        self.assertIsNone(empty['mean_ms'])
        self.assertIsNone(empty['p95_type1_ms'])

    def test_audit_requires_fresh_output_and_retains_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'source.json', Path(temp)/'output'
            source.write_text(json.dumps({'detailed_results': {'example': {
                'requests': self._requests(), 'p95_control_path_total_us': 19050}}}))
            result = analyze_route_evidence(source, output, 'example')
            self.assertEqual(len(result['source']['sha256']), 64)
            self.assertFalse(result['formal_comparison_qualified'])
            self.assertEqual(result['legacy_interpolated_p95_control_ms'], 19.05)
            with self.assertRaises(FileExistsError):
                analyze_route_evidence(source, output, 'example')


class RetainedCopyRevisits(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(3):
            collection = dict(mode='confirmed_snapshot_cache_v1', collection_id=i+1,
                identity_replica_count=0, full_authoritative_read_replica_count=1,
                cold_full_read_replica_count=1, full_fallback_replica_count=0,
                cache_hits=0, identity_only=0, reused_replica_indexes=[])
            source = dict(native=True, tier='gpu', owner_id='owner', epoch=i+1,
                source_id='gpu-copy', path='/same', expected_content_sha256='a'*64,
                footprint_bytes=1024, representation='gpu-slot')
            view = dict(source=source, source_collection=collection,
                native_owner_id='owner', native_epoch=i+1, native_captured_monotonic_s=10.+i)
            rows.append(dict(request_id=f'r{i}', adapter_id='adapter', control_path_total_us=100.,
                gpu_reference_evidence=dict(routing_snapshot=dict(sources={'replica':view}))))
        return rows

    def test_same_copy_epoch_revisit_is_only_an_opportunity(self):
        result = summarize_route_copy_revisits(self.rows())
        self.assertEqual(result['counts']['fresh_gpu_views_same_copy_after_epoch_advance'], 2)
        self.assertEqual(result['affected_request_count'], 2)
        self.assertIsNone(result['exact_rpc_count'])
        self.assertIsNone(result['avoided_latency_ms'])
        self.assertFalse(result['formal_comparison_qualified'])

    def test_changed_target_host_absence_and_capacity_are_not_matches(self):
        for update in ({'source_id':'changed'}, {'tier':'host'}, {'native':False},
                       {'footprint_bytes':2048}, {'path':'/changed'},
                       {'expected_content_sha256':'b'*64}, {'representation':'different'}):
            rows = self.rows()
            rows[1]['gpu_reference_evidence']['routing_snapshot']['sources']['replica']['source'].update(update)
            with self.subTest(update=update):
                result = summarize_route_copy_revisits(rows)
                self.assertEqual(result['affected_request_count'], 0)

    def test_reused_views_are_not_reported_as_fresh_reads(self):
        rows = self.rows()
        for row in rows:
            row['gpu_reference_evidence']['routing_snapshot']['sources']['replica']['source_collection'].update(
                reused_replica_indexes=[0], full_authoritative_read_replica_count=0,
                cold_full_read_replica_count=0)
        result = summarize_route_copy_revisits(rows)
        self.assertEqual(result['affected_request_count'], 0)
        self.assertEqual(result['counts']['reused_target_views'], 3)

    def test_capture_order_not_input_order(self):
        self.assertEqual(summarize_route_copy_revisits(self.rows()),
                         summarize_route_copy_revisits(list(reversed(self.rows()))))

    def test_owner_or_adapter_change_does_not_cross_bind(self):
        rows = self.rows()
        rows[1]['adapter_id'] = 'another'
        self.assertEqual(summarize_route_copy_revisits(rows)['affected_request_count'], 1)
        rows = self.rows()
        view = rows[1]['gpu_reference_evidence']['routing_snapshot']['sources']['replica']
        view['native_owner_id'] = view['source']['owner_id'] = 'another-owner'
        self.assertEqual(summarize_route_copy_revisits(rows)['affected_request_count'], 0)

    def test_missing_and_malformed_membership_fail_closed(self):
        for update in ({'mode':'identity_recheck_cache_v1'}, {'reused_replica_indexes':[2]},
                       {'reused_replica_indexes':[0,0]}, {'full_authoritative_read_replica_count':0}):
            rows = self.rows()
            rows[0]['gpu_reference_evidence']['routing_snapshot']['sources']['replica']['source_collection'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                summarize_route_copy_revisits(rows)


if __name__ == '__main__':
    unittest.main()
