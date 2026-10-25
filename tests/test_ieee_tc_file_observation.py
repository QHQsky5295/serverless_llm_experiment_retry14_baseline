"""One fresh file-owner observation, using existing tiny publication fixtures."""
import asyncio
import copy
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch


def legacy_preparation_snapshot(owner, *, manifests, limits):
    """Independent pre-D153 composition; no new production fallback path."""
    from faaslora.storage.http_artifact_store import _quote_artifact_id
    with owner.lock:
        owner._settle_file_allocations()
        units = {tier: os.statvfs(root).f_frsize for tier, root in owner.roots.items()}
        if any(type(unit) is not int or unit <= 0 for unit in units.values()):
            raise RuntimeError('preparation requires actual destination allocation units')
        if any(path.name not in manifests for path in owner._confirmed_sources):
            raise ValueError('confirmed file owner contains adapters outside the frozen universe')
        artifacts = {}
        for aid, files in sorted(manifests.items()):
            _quote_artifact_id(aid)
            content = owner._expected_content(files)
            observed = owner.source_snapshot(aid)
            if any(row['content_sha256'] != content for row in observed['sources']):
                raise ValueError('preparation file source differs from frozen manifest')
            artifacts[aid] = dict(content_sha256=content,
                logical_payload_bytes=sum(size for size, _ in files.values()),
                sources=observed['sources'],
                targets={tier: dict(tier=tier, representation='verified_regular_file_tree_v1',
                    content_sha256=content, footprint_bytes=sum(
                        ((size+unit-1)//unit)*unit for size, _ in files.values()),
                    path=str(owner.roots[tier]/aid)) for tier, unit in units.items()})
        inventory = owner.inventory()
        budget = owner._file_budget_from_inventory(limits, inventory)
        host = owner._host_budget_from_inventory(inventory)
        sources = {Path(row['path']): row for artifact in artifacts.values()
                   for row in artifact['sources']}
        replacement = owner._file_replacement_capacity_from_inventory(inventory, sources)
        return dict(kind='ieee_file_planning_sources_v1', owner_id=owner.owner_id,
            epoch=owner.source_epoch, clock_id=budget['clock_id'], captured_at=time.monotonic(),
            artifacts=artifacts, budgets=budget, allocation_units_bytes=units,
            managed_host=host, replacement_capacity=replacement,
            physical_resources_reserved=False)


def without_capture_times(value):
    result = copy.deepcopy(value)
    result.pop('captured_at')
    result['budgets'].pop('captured_at')
    return result


class FreshFileObservation(unittest.TestCase):
    def make(self):
        from tests.test_ieee_tc_transfer_pressure import OwnedPreparationPlanning
        from faaslora.registry.schema import StorageTier
        case = OwnedPreparationPlanning()
        self.addCleanup(case.doCleanups)
        fixture, runner, queue, slot, native = case.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        owner = fixture.owner
        manifests = runner._remote_artifact_client.preparation_manifests(runner._ieee_artifact_identities)
        limits = {tier: int(fixture.manager.tier_capacities[StorageTier(tier)].total_bytes)
                  for tier in owner.roots}
        return fixture, owner, manifests, limits

    def test_exact_published_and_cold_outputs_with_500_logical_candidates(self):
        fixture, owner, manifests, limits = self.make()
        manifests.update({f'absent-{i}': dict(manifests['a']) for i in range(496)})
        before = legacy_preparation_snapshot(owner, manifests=manifests, limits=limits)
        after = owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertEqual(without_capture_times(after), without_capture_times(before))
        after['artifacts']['a']['sources'][0]['tier'] = 'changed-export'
        self.assertEqual(without_capture_times(owner.preparation_snapshot(
            manifests=manifests, limits=limits)), without_capture_times(before))

    def test_settled_snapshot_scans_once_and_does_not_probe_absent_adapters(self):
        fixture, owner, manifests, limits = self.make()
        owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertTrue(all(b['settled'] for b in owner._file_allocations.values()))
        manifests.update({f'absent-{i}': dict(manifests['a']) for i in range(496)})
        stats = []
        original = os.stat
        def observed(path, *args, **kwargs):
            if kwargs.get('follow_symlinks') is False:
                stats.append(Path(path))
            return original(path, *args, **kwargs)
        with (patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as inventory,
              patch.object(owner, '_source_observation', side_effect=AssertionError('duplicate scan')),
              patch.object(os, 'stat', observed)):
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertEqual(inventory.call_count, 1)
        self.assertTrue(stats)
        self.assertFalse(any(p.name.startswith('absent-') for p in stats))
        from collections import Counter
        self.assertEqual(set(Counter(stats).values()), {2})

    def test_signature_mutation_withdraws_without_reusing_previous_snapshot(self):
        fixture, owner, manifests, limits = self.make()
        owner.preparation_snapshot(manifests=manifests, limits=limits)
        path = fixture.nvme / 'a'
        file = next(p for p in path.rglob('*') if p.is_file())
        stat = file.stat()
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        epoch = owner.source_epoch
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertNotIn(path, owner._confirmed_sources)
        self.assertGreater(owner.source_epoch, epoch)
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)

    def test_missing_tree_withdraws_and_unknown_copy_is_not_remote(self):
        fixture, owner, manifests, limits = self.make()
        shutil.rmtree(fixture.nvme / 'a')  # Only the existing tiny test fixture.
        epoch = owner.source_epoch
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertGreater(owner.source_epoch, epoch)
        self.assertNotIn(fixture.nvme/'a', owner._confirmed_sources)
        (fixture.nvme / 'a').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)

    def test_unknown_file_and_symlink_roots_are_rejected(self):
        fixture, owner, manifests, limits = self.make()
        manifests['unknown'] = dict(manifests['a'])
        path = fixture.nvme/'unknown'
        path.write_bytes(b'test')
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        path.unlink()
        path.symlink_to(fixture.nvme/'a', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'rejects links'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)

    def test_nested_alias_footprints_equal_individual_source_inventory(self):
        fixture, owner, manifests, limits = self.make()
        source = fixture.nvme/'a'
        file = next(p for p in source.rglob('*') if p.is_file())
        os.link(file, source/'alias')
        os.link(file, fixture.nvme/'external-alias')
        with owner.lock:
            stats = {}
            inventory = owner.inventory(_observed_stats=stats)
            indexed = owner._planning_source_observations(inventory, stats)
            self.assertEqual(indexed[source], owner._source_observation(source))
        # Index equivalence is not confirmation: changed published links must fail.
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            owner.preparation_snapshot(manifests=manifests, limits=limits)

    def test_unsettled_extent_scan_and_private_writer_remain_charged(self):
        fixture, owner, manifests, limits = self.make()
        for bound in owner._file_allocations.values():
            bound['settled'] = False
        with patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as inventory:
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        # Extent settlement is now part of that same fresh double-checked view,
        # not a separate global scan followed by another capacity observation.
        self.assertEqual(inventory.call_count, 1)
        manifests['pending'] = dict(manifests['a'])
        target = fixture.nvme/'pending'
        with owner.materializing(target, budgeted=True) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_copy(transfer, staging, manifests['pending'], limit_bytes=limits['nvme'])
                before = legacy_preparation_snapshot(owner, manifests=manifests, limits=limits)
                after = owner.preparation_snapshot(manifests=manifests, limits=limits)
                self.assertEqual(without_capture_times(after), without_capture_times(before))
                self.assertEqual(after['artifacts']['pending']['sources'], [])
                self.assertEqual(after['budgets']['tiers']['nvme']['active_transfers'], 1)

    def unsettled(self):
        fixture, owner, manifests, limits = self.make()
        file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file() and p.stat().st_size)
        info = file.stat()
        bound = owner._file_allocations[info.st_dev, info.st_ino]
        self.assertTrue(bound['writer_closed'])
        # Restore the conservative ext4 ceiling of this existing tiny fixture.
        bound.update(data_bytes=512*info.st_blocks, ceiling_bytes=6*512*info.st_blocks, settled=False)
        return fixture, owner, manifests, limits, file, bound

    def test_uninitialized_extent_keeps_envelope_in_single_fresh_inventory(self):
        fixture, owner, manifests, limits, file, bound = self.unsettled()
        before = dict(bound)
        with (patch('faaslora.memory.residency_manager._file_extents_initialized', return_value=False),
              patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as inventory):
            snapshot = owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertEqual(inventory.call_count, 1)
        self.assertEqual(bound, before)
        self.assertGreater(snapshot['budgets']['tiers']['nvme']['pending_increment_bytes'], 0)

    def test_successful_settlement_reobserves_path_then_checks_every_path_again(self):
        from collections import Counter
        fixture, owner, manifests, limits, file, bound = self.unsettled()
        counts = Counter()
        original = os.stat
        def observe(path, *args, **kwargs):
            if kwargs.get('follow_symlinks') is False:
                counts[str(path)] += 1
            return original(path, *args, **kwargs)
        with (patch('faaslora.memory.residency_manager._file_extents_initialized', return_value=True),
              patch.object(os, 'stat', observe)):
            owner.preparation_snapshot(manifests=manifests, limits=limits)
        self.assertTrue(bound['settled'])
        self.assertEqual(counts.pop(str(file)), 3)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(bound['data_bytes'], bound['ceiling_bytes'])

    def test_mutation_during_extent_probe_cannot_escape_as_successful_observation(self):
        fixture, owner, manifests, limits, file, bound = self.unsettled()
        stats = {}
        def mutate(*args):
            info = file.stat()
            os.utime(file, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return True
        with (owner.lock,
              patch('faaslora.memory.residency_manager._file_extents_initialized', side_effect=mutate)):
            with self.assertRaisesRegex(RuntimeError, 'changed during extent settlement'):
                owner.inventory(_observed_stats=stats, _settle_allocations=True)
        self.assertEqual(stats, {})

    def test_final_stat_still_rejects_change_after_successful_settlement(self):
        fixture, owner, manifests, limits, file, bound = self.unsettled()
        original, count = os.stat, 0
        def mutate(path, *args, **kwargs):
            nonlocal count
            if kwargs.get('follow_symlinks') is False and str(path) == str(file):
                count += 1
                if count == 3:
                    info = original(path)
                    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns+1_000_000_000))
            return original(path, *args, **kwargs)
        stats = {}
        with (owner.lock, patch.object(os, 'stat', mutate),
              patch('faaslora.memory.residency_manager._file_extents_initialized', return_value=True)):
            with self.assertRaisesRegex(RuntimeError, 'changed while collecting footprint'):
                owner.inventory(_observed_stats=stats, _settle_allocations=True)
        self.assertEqual(stats, {})

    def test_second_stat_change_is_still_rejected_before_observations_escape(self):
        fixture, owner, manifests, limits = self.make()
        owner.preparation_snapshot(manifests=manifests, limits=limits)
        file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file())
        count = 0
        original = os.stat
        def changing(path, *args, **kwargs):
            nonlocal count
            if kwargs.get('follow_symlinks') is False and Path(path) == file:
                count += 1
                if count == 2:
                    stat = original(path)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
            return original(path, *args, **kwargs)
        stats = {}
        with owner.lock, patch.object(os, 'stat', changing):
            with self.assertRaisesRegex(RuntimeError, 'changed while collecting footprint'):
                owner.inventory(_observed_stats=stats)
        self.assertEqual(stats, {})

    def test_private_observations_use_inventory_strings_and_are_fresh_per_call(self):
        fixture, owner, manifests, limits = self.make()
        with owner.lock:
            first = {}
            inventory = owner.inventory(_observed_stats=first)
            names = {name for row in inventory['allocations'] for name in row['paths']}
            self.assertEqual(set(first), names)
            self.assertTrue(all(type(name) is str for name in first))
            file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file())
            previous = first[str(file)]
            os.utime(file, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
            second = {}
            owner.inventory(_observed_stats=second)
            self.assertEqual(first[str(file)].st_mtime_ns, previous.st_mtime_ns)
            self.assertEqual(second[str(file)].st_mtime_ns, file.stat().st_mtime_ns)
            self.assertNotEqual(first[str(file)].st_mtime_ns, second[str(file)].st_mtime_ns)

    def test_inventory_preserves_normalized_root_and_sibling_order(self):
        from contextlib import chdir
        from faaslora.memory.residency_manager import _local_file_inventory
        fixture, owner, manifests, limits = self.make()
        # Reuse tiny fixture; no model weights or trace are generated.
        for name in ('index-a', 'index-ab', 'index-中文'):
            (fixture.nvme/name).mkdir()
        with chdir(fixture.nvme):
            for root in ('.', './', '././', str(fixture.nvme) + '//./'):
                normalized = Path(root)
                expected, pending = [], [normalized]
                while pending:
                    path = pending.pop()
                    expected.append(str(path))
                    if path.is_dir():
                        pending.extend(sorted(path.iterdir(), reverse=True))
                captured = {}
                actual = _local_file_inventory({'nvme': root}, _observed_stats=captured)
                self.assertEqual(list(captured), expected)
                self.assertEqual([p for row in actual['allocations'] for p in row['paths']], expected)

    def test_nested_cross_tree_and_cross_tier_alias_projection_is_exact(self):
        fixture, owner, manifests, limits = self.make()
        file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file())
        roots = tuple(owner.roots.values())
        # Tiny existing bytes only; no LoRA pool or source manifest is generated.
        for root in roots:
            for child in ('index-a', 'index-ab', 'index-中文'):
                nested = root/child/'nested'/'more'
                nested.mkdir(parents=True)
                os.link(file, nested/'alias')
                os.link(file, nested.parent/'second')
        with owner.lock:
            stats = {}
            inventory = owner.inventory(_observed_stats=stats)
            before = copy.deepcopy(inventory)
            actual = owner._planning_source_observations(inventory, stats)
            expected = {child: owner._source_observation(child)
                        for root in roots for child in root.iterdir()}
            self.assertEqual(actual, expected)
            self.assertEqual(inventory, before)
            # Projection uses the scan's existing identities, not new ancestor
            # Path chains or relative_to calls for each file in each tree.
            with (patch.object(Path, 'parents', new=property(
                    lambda _: self.fail('ancestor Path reconstruction'))),
                  patch.object(Path, 'relative_to', side_effect=AssertionError('relative reconstruction'))):
                self.assertEqual(owner._planning_source_observations(inventory, stats), expected)

    def test_projection_rejects_inconsistent_inventory_tier_labels(self):
        fixture, owner, manifests, limits = self.make()
        with owner.lock:
            stats = {}
            inventory = owner.inventory(_observed_stats=stats)
            bad = copy.deepcopy(inventory)
            bad['allocations'][0]['path_tiers'] = []
            with self.assertRaisesRegex(ValueError, 'path/tier count'):
                owner._planning_source_observations(bad, stats)
            bad = copy.deepcopy(inventory)
            allocation = next(row for row in bad['allocations'] if row['kind'] == 'file')
            tier = allocation['path_tiers'][0]
            allocation['path_tiers'][0] = next(t for t in owner.roots if t != tier)
            with self.assertRaisesRegex(ValueError, 'differs from its tier root'):
                owner._planning_source_observations(bad, stats)


class AsyncFileOwnerTurn(unittest.IsolatedAsyncioTestCase):
    """D851 transaction boundary; no GPU or inference performance assertions."""

    def make(self):
        from tests.test_ieee_tc_request_lifecycle import ConfirmedFilePublication
        case = ConfirmedFilePublication()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.assertTrue(case.fetch()[0])
        return case, case.owner

    async def test_cancelled_async_interest_enter_still_exits_real_owner(self):
        import threading
        from contextlib import contextmanager
        case, owner = self.make()
        original = owner.file_movement_interest
        content = owner.source_snapshot('a')['sources'][0]['content_sha256']
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        exits = []
        @contextmanager
        def held(**kwargs):
            with original(**kwargs):
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise RuntimeError('interest enter barrier expired')
                try:
                    yield
                finally:
                    exits.append(threading.get_ident())
        owner.file_movement_interest = held
        async def use():
            async with owner.async_file_movement_interest(intent_id='cancel-enter',
                    target=case.nvme/'a', content_sha256=content, required=True):
                self.fail('cancelled enter reached the consumer')
        task = asyncio.create_task(use())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            self.assertFalse(task.done())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(exits), 1)
        self.assertNotEqual(exits[0], threading.get_ident())
        self.assertFalse(owner._file_movement_interests)

    async def test_async_interest_body_failure_releases_interest_without_holding_owner_lock(self):
        case, owner = self.make()
        content = owner.source_snapshot('a')['sources'][0]['content_sha256']
        with self.assertRaisesRegex(RuntimeError, 'actual consumer failure'):
            async with owner.async_file_movement_interest(intent_id='body-failure',
                    target=case.nvme/'a', content_sha256=content, required=True):
                self.assertIn('body-failure', owner._file_movement_interests)
                view = await owner.call_async(owner.source_snapshot, 'a')
                self.assertTrue(view['sources'])
                raise RuntimeError('actual consumer failure')
        self.assertFalse(owner._file_movement_interests)

    async def controlled(self, owner, operation, *, on_result=None):
        import threading
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()
        worker_threads = []
        def held():
            worker_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise RuntimeError('D851 transaction barrier expired')
            return operation()
        task = asyncio.create_task(owner.call_async(held, on_result=on_result))
        await asyncio.wait_for(started.wait(), 2)
        self.addCleanup(release.set)
        return task, release, worker_threads

    async def test_d851_file_read_runs_off_loop_without_reusing_an_old_snapshot(self):
        import threading
        case, owner = self.make()
        initial = owner.source_snapshot('a')
        seen = []
        task, release, workers = await self.controlled(owner,
            lambda: owner.source_snapshot('a'),
            on_result=lambda value: seen.append((threading.get_ident(), value)))
        try:
            heartbeat = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(heartbeat.set_result, True)
            self.assertTrue(await asyncio.wait_for(heartbeat, 1))
            self.assertFalse(task.done())
        finally:
            release.set()
        result = await task
        self.assertEqual(result['sources'], initial['sources'])
        self.assertGreater(result['captured_monotonic_s'], initial['captured_monotonic_s'])
        self.assertNotEqual(workers, [threading.get_ident()])
        self.assertEqual(seen, [(threading.get_ident(), result)])
        path = case.nvme/'a'/'nested/weights'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            await owner.call_async(owner.source_snapshot, 'a')

    async def test_d851_cancelled_acquisition_is_adopted_then_released(self):
        import threading
        case, owner = self.make()
        adopted = []
        def acquire():
            view = owner.source_snapshot('a')
            source = view['sources'][0]
            return owner.acquire_confirmed(path=source['path'], adapter_id='a',
                lease_id='d851-cancelled', expected_owner_id=view['owner_id'],
                expected_epoch=view['epoch'], expected_content_sha256=source['content_sha256'])
        task, release, _ = await self.controlled(owner, acquire,
            on_result=lambda value: adopted.append((threading.get_ident(), value)))
        try:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(adopted, [])
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(adopted), 1)
        self.assertEqual(adopted[0][0], threading.get_ident())
        reference = adopted[0][1]
        self.assertIn(reference['lease_id'], owner.leases)
        # The existing request/preparation cleanup owner receives the lease
        # even when cancellation coincides with a successful physical acquire.
        await owner.call_async(owner.release, lease_id=reference['lease_id'],
            expected_owner_id=reference['owner_id'])
        self.assertFalse(owner.leases)

    async def test_d851_late_worker_error_is_not_hidden_by_cancellation(self):
        _, owner = self.make()
        adopted = []
        def fail():
            raise RuntimeError('actual owner failure')
        task, release, _ = await self.controlled(owner, fail, on_result=adopted.append)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaisesRegex(RuntimeError, 'actual owner failure'):
            await task
        self.assertFalse(adopted)

    async def test_d851_transaction_lock_covers_snapshot_and_pin(self):
        import threading
        case, owner = self.make()
        loop = asyncio.get_running_loop()
        snapshot_done, competitor_checked = asyncio.Event(), threading.Event()
        lock_observations = []
        def attempt_other_owner_mutation():
            acquired = owner.lock.acquire(blocking=False)
            lock_observations.append(acquired)
            if acquired:
                owner.lock.release()
            competitor_checked.set()
        def transaction():
            view = owner.source_snapshot('a')
            loop.call_soon_threadsafe(snapshot_done.set)
            if not competitor_checked.wait(3):
                raise RuntimeError('D851 competing mutation did not execute')
            source = view['sources'][0]
            return owner.acquire_confirmed(path=source['path'], adapter_id='a',
                lease_id='d851-atomic', expected_owner_id=view['owner_id'],
                expected_epoch=view['epoch'], expected_content_sha256=source['content_sha256'])
        task = asyncio.create_task(owner.call_async(transaction))
        await asyncio.wait_for(snapshot_done.wait(), 2)
        competitor = threading.Thread(target=attempt_other_owner_mutation)
        competitor.start()
        reference = await task
        competitor.join(2)
        self.assertFalse(competitor.is_alive())
        self.assertEqual(lock_observations, [False])
        await owner.call_async(owner.release, lease_id=reference['lease_id'],
            expected_owner_id=reference['owner_id'])

    async def test_d851_release_result_is_adopted_before_cancel_propagates(self):
        case, owner = self.make()
        reference = owner.acquire(path=str(case.nvme/'a'), adapter_id='a', lease_id='d851-release')
        task, release, _ = await self.controlled(owner,
            lambda: owner.release(lease_id=reference['lease_id'], expected_owner_id=owner.owner_id),
            on_result=lambda _: reference.update(state='released'))
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(reference['state'], 'released')
        self.assertFalse(owner.leases)

    async def test_d851_refuses_async_transaction_or_async_adoption(self):
        _, owner = self.make()
        async def invalid(*args):
            return None
        for operation, adopt in ((invalid, None), (lambda: None, invalid),
                                 (lambda: invalid(), None), (lambda: None, lambda _: invalid())):
            with self.subTest(operation=operation, adopt=adopt), self.assertRaises(TypeError):
                await owner.call_async(operation, on_result=adopt)


if __name__ == '__main__':
    unittest.main()
