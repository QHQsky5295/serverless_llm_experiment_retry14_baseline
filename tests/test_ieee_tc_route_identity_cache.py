"""CPU-only qualification for the opt-in confirmed-route footprint cache."""

import asyncio
import copy
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
from scripts.analyze_control_path_overhead import summarize_route_evidence, analyze_route_evidence
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


if __name__ == '__main__':
    unittest.main()
