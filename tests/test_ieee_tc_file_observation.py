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
        original = Path.lstat
        def observed(path, *args, **kwargs):
            stats.append(path)
            return original(path, *args, **kwargs)
        with (patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as inventory,
              patch.object(owner, '_source_observation', side_effect=AssertionError('duplicate scan')),
              patch.object(Path, 'lstat', observed)):
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
        self.assertEqual(inventory.call_count, 2)
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

    def test_second_stat_change_is_still_rejected_before_observations_escape(self):
        fixture, owner, manifests, limits = self.make()
        owner.preparation_snapshot(manifests=manifests, limits=limits)
        file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file())
        count = 0
        original = Path.lstat
        def changing(path, *args, **kwargs):
            nonlocal count
            if path == file:
                count += 1
                if count == 2:
                    stat = original(path)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
            return original(path, *args, **kwargs)
        stats = {}
        with owner.lock, patch.object(Path, 'lstat', changing):
            with self.assertRaisesRegex(RuntimeError, 'changed while collecting footprint'):
                owner.inventory(_observed_stats=stats)
        self.assertEqual(stats, {})


if __name__ == '__main__':
    unittest.main()
