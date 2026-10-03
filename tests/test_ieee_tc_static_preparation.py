"""Static metadata reuse must not become stale source or capacity reuse."""
import asyncio
import copy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from faaslora.storage import http_artifact_store as store
from tests import test_http_artifact_store as http_fixtures
from tests import test_ieee_tc_file_observation as observation
from tests import test_ieee_tc_transfer_pressure as planning


class FrozenDescriptions(unittest.TestCase):
    def test_input_pairs_and_all_export_levels_are_immutable(self):
        raw = {'a': {'weights': [17, 'a' * 64]}}
        frozen = store.FrozenPreparationDescriptions(raw)
        binding = frozen.binding
        raw['a']['weights'][0] = 100
        raw['a']['other'] = (4, 'b' * 64)
        self.assertEqual(frozen['a'], {'weights': (17, 'a' * 64)})
        self.assertEqual(frozen.binding, binding)
        with self.assertRaises(TypeError):
            frozen['a']['weights'] = (99, 'c' * 64)
        with self.assertRaises(TypeError):
            frozen.descriptions['a']['logical_payload_bytes'] = 99
        with self.assertRaises(TypeError):
            frozen.descriptions['a'] = {}
        with self.assertRaises(FrozenInstanceError):
            frozen.binding = ()

    def test_selection_binds_exact_ids_not_input_order_or_name_only(self):
        raw = {a: {'weights': (17, digest * 64)} for a, digest in [('a','a'), ('b','b')]}
        frozen = store.FrozenPreparationDescriptions(raw)
        self.assertIs(frozen.select(['b', 'a']), frozen)
        selected = frozen.select(['b'])
        self.assertEqual(tuple(selected), ('b',))
        self.assertEqual(selected.binding, (('b', frozen.descriptions['b']['content_sha256']),))
        for ids in (['a', 'a'], ['unknown']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                frozen.select(ids)
        self.assertEqual(len(frozen.select([])), 0)
        raw['b']['weights'] = (17, 'c' * 64)
        self.assertNotEqual(store.FrozenPreparationDescriptions(raw).binding, frozen.binding)

    def test_mutable_construction_rejects_invalid_id_path_size_digest(self):
        for raw in ({' a': {'w': (1, 'a'*64)}}, {'a/b': {'w': (1, 'a'*64)}},
                    {'a': {'../x': (1, 'a'*64)}}, {'a': {'w': (True, 'a'*64)}},
                    {'a': {'w': (-1, 'a'*64)}}, {'a': {'w': (1, 'A'*64)}}, {'a': {}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                store.FrozenPreparationDescriptions(raw)

    def test_client_idempotent_configuration_and_legacy_mutable_export(self):
        client = store.HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        with self.assertRaises(ValueError):
            client.preparation_descriptions(['a'])
        payload = http_fixtures.content_manifest(files={'w': b'A', 'z': b'BB'})
        digest = client.configure_content_manifest(payload)
        frozen = client.preparation_descriptions(['a'])
        payload['artifacts'][0]['files'].reverse()
        self.assertEqual(client.configure_content_manifest(payload), digest)
        self.assertIs(client.preparation_descriptions(['a']), frozen)
        legacy = client.preparation_manifests(['a'])
        legacy['a']['w'] = (100, '0' * 64)
        self.assertEqual(frozen['a']['w'][0], 1)
        payload['artifacts'][0]['files'][0]['size_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            client.configure_content_manifest(payload)
        self.assertIs(client.preparation_descriptions(['a']), frozen)


class StaticPreparationOwner(unittest.TestCase):
    def make(self):
        fixture_case = observation.FreshFileObservation()
        self.addCleanup(fixture_case.doCleanups)
        fixture, owner, raw, limits = fixture_case.make()
        return fixture, owner, raw, store.FrozenPreparationDescriptions(raw), limits

    def test_legacy_equivalence_fresh_inventory_and_no_hot_digest(self):
        fixture, owner, raw, frozen, limits = self.make()
        expected = observation.legacy_preparation_snapshot(owner, manifests=raw, limits=limits)
        with patch.object(store, 'preparation_content_sha256', side_effect=AssertionError('hot digest')):
            with patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as scan:
                actual = owner.preparation_snapshot(manifests=frozen, limits=limits)
                self.assertEqual(scan.call_count, 1)
            with patch.object(owner, '_file_inventory', wraps=owner._file_inventory) as scan:
                again = owner.preparation_snapshot(manifests=frozen, limits=limits)
                self.assertEqual(scan.call_count, 1)
        self.assertEqual(observation.without_capture_times(expected), observation.without_capture_times(actual))
        self.assertEqual(observation.without_capture_times(actual), observation.without_capture_times(again))
        self.assertGreater(again['captured_at'], actual['captured_at'])

    def test_export_and_templates_cannot_poison_next_snapshot(self):
        fixture, owner, raw, frozen, limits = self.make()
        first = owner.preparation_snapshot(manifests=frozen, limits=limits)
        expected = copy.deepcopy(first)
        first['artifacts']['a']['targets']['nvme']['footprint_bytes'] = 0
        first['artifacts']['a']['sources'][0]['tier'] = 'bogus'
        first['budgets']['tiers']['nvme']['remaining_bytes'] = 0
        with self.assertRaises(TypeError):
            owner._preparation_targets['a']['nvme']['footprint_bytes'] = 0
        next_view = owner.preparation_snapshot(manifests=frozen, limits=limits)
        self.assertEqual(observation.without_capture_times(expected), observation.without_capture_times(next_view))

    def test_mutable_inputs_are_revalidated_even_at_same_address(self):
        fixture, owner, raw, frozen, limits = self.make()
        owner.preparation_snapshot(manifests=raw, limits=limits)
        with patch.object(store, 'preparation_content_sha256', wraps=store.preparation_content_sha256) as digest:
            owner.preparation_snapshot(manifests=raw, limits=limits)
            self.assertEqual(digest.call_count, len(raw))
        name = next(iter(raw['a']))
        size, digest = raw['a'][name]
        raw['a'][name] = (size, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'differs from frozen manifest'):
            owner.preparation_snapshot(manifests=raw, limits=limits)
        raw['a'][name] = (-1, digest)
        with self.assertRaises(ValueError):
            owner.preparation_snapshot(manifests=raw, limits=limits)

    def test_subsets_do_not_hide_confirmed_copies(self):
        fixture, owner, raw, frozen, limits = self.make()
        owner.preparation_snapshot(manifests=frozen, limits=limits)
        with self.assertRaisesRegex(ValueError, 'outside the frozen universe'):
            owner.preparation_snapshot(manifests=frozen.select(['a']), limits=limits)

    def test_source_mutation_withdrawal_remains_live(self):
        fixture, owner, raw, frozen, limits = self.make()
        owner.preparation_snapshot(manifests=frozen, limits=limits)
        source = fixture.nvme/'a'
        file = next(p for p in source.rglob('*') if p.is_file())
        info = file.stat()
        epoch = owner.source_epoch
        os.utime(file, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            owner.preparation_snapshot(manifests=frozen, limits=limits)
        self.assertNotIn(source, owner._confirmed_sources)
        self.assertGreater(owner.source_epoch, epoch)
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            owner.preparation_snapshot(manifests=frozen, limits=limits)

    def test_second_stat_race_is_not_hidden_by_compiled_metadata(self):
        fixture, owner, raw, frozen, limits = self.make()
        owner.preparation_snapshot(manifests=frozen, limits=limits)
        file = next(p for p in (fixture.nvme/'a').rglob('*') if p.is_file())
        original = Path.lstat
        count = 0
        def changing(path, *args, **kwargs):
            nonlocal count
            if path == file:
                count += 1
                if count == 2:
                    info = original(path)
                    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'lstat', changing), self.assertRaisesRegex(RuntimeError, 'changed while collecting'):
            owner.preparation_snapshot(manifests=frozen, limits=limits)

    def test_pending_writer_capacity_still_changes_after_template_compilation(self):
        fixture, owner, raw, frozen, limits = self.make()
        raw['pending'] = dict(raw['a'])
        frozen = store.FrozenPreparationDescriptions(raw)
        before = owner.preparation_snapshot(manifests=frozen, limits=limits)
        with owner.materializing(fixture.nvme/'pending', budgeted=True) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_copy(transfer, staging, raw['pending'], limit_bytes=limits['nvme'])
                actual = owner.preparation_snapshot(manifests=frozen, limits=limits)
                expected = observation.legacy_preparation_snapshot(owner, manifests=raw, limits=limits)
                self.assertEqual(observation.without_capture_times(actual), observation.without_capture_times(expected))
                self.assertEqual(actual['budgets']['tiers']['nvme']['active_transfers'], 1)
                self.assertGreater(actual['budgets']['tiers']['nvme']['used_bytes'],
                                   before['budgets']['tiers']['nvme']['used_bytes'])
                self.assertEqual(actual['artifacts']['pending']['sources'], [])

    def test_geometry_and_subset_changes_recompile_without_stale_targets(self):
        from faaslora.memory.residency_manager import LocalSourceReferences
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in ('host', 'nvme', 'replacement'):
            (root/name).mkdir()
        owner = LocalSourceReferences({tier: root/tier for tier in ('host', 'nvme')})
        owner.configure_host_budget(1 << 20)
        limits = dict(host=1 << 20, nvme=1 << 20)
        frozen = store.FrozenPreparationDescriptions({
            'a': {'one': (4097, 'a'*64), 'two': (1, 'b'*64)},
            'b': {'one': (1, 'c'*64)}})
        unit = os.statvfs(root).f_frsize
        first = owner.preparation_snapshot(manifests=frozen, limits=limits)
        targets = owner._preparation_targets
        owner.preparation_snapshot(manifests=store.FrozenPreparationDescriptions(frozen), limits=limits)
        self.assertIs(owner._preparation_targets, targets)  # Content binding, not input address.
        with patch('os.statvfs', return_value=SimpleNamespace(f_frsize=unit*2)):
            changed = owner.preparation_snapshot(manifests=frozen, limits=limits)
        self.assertEqual(changed['artifacts']['a']['targets']['nvme']['footprint_bytes'],
                         ((4097+2*unit-1)//(2*unit)+1)*(2*unit))
        self.assertIsNot(owner._preparation_targets, targets)
        restored = owner.preparation_snapshot(manifests=frozen, limits=limits)
        self.assertEqual(restored['artifacts'], first['artifacts'])
        owner.roots['nvme'] = root/'replacement'
        moved = owner.preparation_snapshot(manifests=frozen, limits=limits)
        self.assertEqual(moved['artifacts']['a']['targets']['nvme']['path'], str(root/'replacement'/'a'))
        subset = owner.preparation_snapshot(manifests=frozen.select(['a']), limits=limits)
        self.assertEqual(set(subset['artifacts']), {'a'})
        self.assertEqual(set(owner._preparation_targets), {'a'})
        for invalid in (0, -1, True):
            with patch('os.statvfs', return_value=SimpleNamespace(f_frsize=invalid)):
                with self.assertRaisesRegex(RuntimeError, 'allocation units'):
                    owner.preparation_snapshot(manifests=frozen, limits=limits)

    def test_runner_uses_immutable_input_without_changing_planner_results(self):
        case = planning.OwnedPreparationPlanning()
        self.addCleanup(case.doCleanups)
        fixture, runner, queue, slot, _ = case.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        with patch.object(runner._remote_artifact_client, 'preparation_manifests',
                          side_effect=AssertionError('mutable hot export')):
            result = case.plan(runner, slot)
        sources = result['source_view']['sources']
        self.assertEqual({a: row['selected_source']['tier'] for a, row in sources.items()},
                         {'a':'nvme', 'b':'gpu', 'c':'host', 'd':'host'})


if __name__ == '__main__':
    unittest.main()
