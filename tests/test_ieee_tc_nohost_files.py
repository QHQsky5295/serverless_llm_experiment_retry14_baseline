"""NoHOST file-owner policy with actual local and spawned file ownership.

This qualifies file policy, not native staging authorization or GPU serving.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from faaslora.memory.residency_manager import LocalSourceReferences, ResidencyManager, StorageTier
from faaslora.memory.file_state_process import FileStateProcess, ProcessLocalSourceReferences
from faaslora.storage.http_artifact_store import (
    FrozenPreparationDescriptions, _verified_file_signature,
    preparation_content_sha256, publish_directory,
)


class NoHostFilePolicy(unittest.TestCase):
    owner_type = LocalSourceReferences

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nohost-file-policy-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = {tier: self.root / tier for tier in ('host', 'nvme')}
        for path in self.roots.values():
            path.mkdir()
        self.owner = self.owner_type(self.roots, independent_host_reuse=False)
        if isinstance(self.owner, ProcessLocalSourceReferences):
            self.addCleanup(self.owner.close)
        self.payload = b'unchanged adapter fixture'
        self.files = {'weights': (len(self.payload), hashlib.sha256(self.payload).hexdigest())}
        self.content = preparation_content_sha256(self.files)

    def test_policy_frozen_and_snapshots_explicit_budgets_unchanged(self):
        self.assertFalse(self.owner.independent_host_reuse)
        with self.assertRaises(AttributeError):
            self.owner.independent_host_reuse = True
        self.owner.configure_host_budget(1024**2)
        budget = self.owner.host_budget_snapshot()
        self.assertEqual(budget['limit_bytes'], 1024**2)
        for view in (self.owner.source_snapshot('a'),
                     self.owner.file_preparation_snapshot(),
                     self.owner.replacement_source_snapshot()):
            self.assertIs(view['independent_host_reuse'], False)
        planning = self.owner.preparation_snapshot(
            manifests=FrozenPreparationDescriptions({'a': self.files}),
            limits={'host': 1024**2, 'nvme': 1024**2})
        self.assertIs(planning['independent_host_reuse'], False)
        self.assertEqual(set(planning['artifacts']['a']['targets']), {'nvme'})
        self.assertEqual(planning['budgets']['tiers']['host']['limit_bytes'], 1024**2)
        self.assertEqual(planning['budgets']['tiers']['nvme']['limit_bytes'], 1024**2)
        self.assertEqual(set(planning['allocation_units_bytes']), {'host', 'nvme'})
        if isinstance(self.owner, ProcessLocalSourceReferences):
            self.assertIs(self.owner.identity['independent_host_reuse'], False)
            self.owner.identity['independent_host_reuse'] = True
            self.assertIs(self.owner.independent_host_reuse, False)

    def test_source_guard_excludes_host_but_preserves_external_and_nvme_input(self):
        for path in (self.roots['host'], self.roots['host'] / 'a',
                     self.roots['host'] / 'a' / 'weights'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'NoHOST'):
                self.owner.require_service_source(path)
        for path in (None, self.roots['nvme'] / 'a', self.root / 'external'):
            self.owner.require_service_source(path)

    def test_host_mutations_rejected_before_context_or_publication(self):
        target = self.roots['host'] / 'a'
        with self.assertRaisesRegex(ValueError, 'NoHOST'):
            with self.owner.materializing(target, budgeted=True):
                self.fail('HOST transfer entered')
        with self.assertRaisesRegex(ValueError, 'NoHOST'):
            with self.owner.file_movement_interest(intent_id='forbidden', target=target,
                    content_sha256=self.content, required=True):
                self.fail('HOST movement entered')
        with self.assertRaisesRegex(ValueError, 'NoHOST'):
            self.owner.register_file_preparation_plan(plan_id='bad', targets=[
                dict(adapter_id='a', tier='host', content_sha256=self.content)])
        for verified in (None, {}):
            publisher = Mock()
            with self.subTest(verified=verified), self.assertRaisesRegex(ValueError, 'NoHOST'):
                if isinstance(self.owner, ProcessLocalSourceReferences):
                    self.owner.publish_transfer('nonexistent', self.root / 'staging', target,
                                                verified_files=verified)
                else:
                    self.owner.publish_transfer('nonexistent', self.root / 'staging', target,
                                                publisher, verified_files=verified)
            publisher.assert_not_called()
        state = self.owner.file_preparation_snapshot()
        self.assertEqual(state['plans'], [])
        self.assertEqual(state['movement_interests'], [])
        self.assertFalse(target.exists())

    def test_existing_unverified_host_never_becomes_readable(self):
        target = self.roots['host'] / 'a'
        target.mkdir()
        (target / 'weights').write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, 'NoHOST'):
            self.owner.acquire(path=str(target), adapter_id='a', lease_id='read')
        with self.assertRaisesRegex(RuntimeError, 'without verified'):
            self.owner.source_snapshot('a')
        self.assertEqual((target / 'weights').read_bytes(), self.payload)

    def test_nvme_preparation_acquisition_and_release_remain_usable(self):
        target = self.roots['nvme'] / 'a'
        self.owner.configure_host_budget(1024**2)
        self.owner.register_file_preparation_plan(plan_id='plan', targets=[
            dict(adapter_id='a', tier='nvme', content_sha256=self.content)])
        with self.owner.file_movement_interest(intent_id='request', target=target,
                content_sha256=self.content, required=True):
            with self.owner.materializing(target, budgeted=True) as transfer:
                with self.owner.transfer_workspace(transfer) as staging:
                    self.owner.prepare_copy(transfer, staging, self.files, limit_bytes=1024**2)
                    weights = Path(staging) / 'weights'
                    weights.write_bytes(self.payload)
                    verified = {'weights': dict(size_bytes=len(self.payload),
                        sha256=self.files['weights'][1], signature=_verified_file_signature(weights.stat()))}
                    if isinstance(self.owner, ProcessLocalSourceReferences):
                        self.owner.publish_transfer(transfer, staging, target, verified_files=verified)
                    else:
                        self.owner.publish_transfer(transfer, staging, target, publish_directory,
                                                    verified_files=verified)
            self.owner.finish_file_preparation_target(plan_id='plan', tier='nvme', adapter_id='a')
            view = self.owner.source_snapshot('a')
            if isinstance(self.owner, ProcessLocalSourceReferences):
                held = self.owner.acquire_best(adapter_id='a', lease_id='read',
                                               content_sha256=self.content)
            else:
                held = self.owner.acquire_confirmed(path=str(target), adapter_id='a', lease_id='read',
                    expected_owner_id=view['owner_id'], expected_epoch=view['epoch'],
                    expected_content_sha256=self.content)
            self.assertEqual(held['tier'], 'nvme')
            self.assertEqual(held['confirmed_source']['content_sha256'], self.content)
            self.assertEqual(self.owner.demand_witness(held, content_sha256=self.content)['lease_id'], 'read')
            self.owner.release(lease_id='read', expected_owner_id=self.owner.owner_id)
        self.owner.close_file_preparation_plan(plan_id='plan')
        self.assertEqual((target / 'weights').read_bytes(), self.payload)
        self.assertEqual(list(self.roots['host'].iterdir()), [])


class NoHostFileProcessPolicy(NoHostFilePolicy):
    owner_type = ProcessLocalSourceReferences


class NoHostFileInitialization(unittest.TestCase):
    def test_strict_mode_and_existing_host_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = {tier: Path(folder) / tier for tier in ('host', 'nvme')}
            for path in roots.values():
                path.mkdir()
            for cls in (LocalSourceReferences, FileStateProcess, ProcessLocalSourceReferences):
                for bad in (0, 1, None, 'false'):
                    with self.subTest(owner=cls, mode=bad), self.assertRaisesRegex(ValueError, 'bool'):
                        cls(roots, independent_host_reuse=bad)
            payload = roots['host'] / 'old'
            payload.write_bytes(b'must not delete')
            with self.assertRaisesRegex(ValueError, 'initially empty'):
                LocalSourceReferences(roots, independent_host_reuse=False)
            self.assertEqual(payload.read_bytes(), b'must not delete')
            full = LocalSourceReferences(roots)
            self.assertNotIn('independent_host_reuse', full.source_snapshot('absent'))
            self.assertNotIn('independent_host_reuse', full.replacement_source_snapshot())

    def test_legacy_residency_materializers_cannot_bypass_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            roots = {tier: root / tier for tier in ('host', 'nvme')}
            for path in roots.values():
                path.mkdir()
            manager = object.__new__(ResidencyManager)
            manager.host_cache_dir, manager.nvme_cache_dir = roots['host'], roots['nvme']
            manager.local_source_references = LocalSourceReferences(roots, independent_host_reuse=False)
            source = root / 'source'
            source.write_bytes(b'not a managed copy')
            for method in (manager._materialize_into_tier_dir, manager._materialize_into_tier_dir_locked):
                with self.subTest(method=method.__name__), self.assertRaisesRegex(ValueError, 'NoHOST'):
                    method('a', str(source), StorageTier.HOST)
            # Both a directory and single-file legacy source must be rejected
            # even though the destination is NVMe and no confirmed record exists.
            legacy_dir = roots['host'] / 'unexpected'
            legacy_dir.mkdir()
            legacy_file = legacy_dir / 'weights'
            legacy_file.write_bytes(b'not reusable HOST')
            for method in (manager._materialize_into_tier_dir, manager._materialize_into_tier_dir_locked):
                for forbidden in (legacy_dir, legacy_file):
                    with self.subTest(method=method.__name__, source=forbidden), self.assertRaisesRegex(ValueError, 'NoHOST'):
                        method('a', str(forbidden), StorageTier.NVME)
            self.assertEqual(list(roots['nvme'].iterdir()), [])
            with self.assertRaisesRegex(ValueError, 'NoHOST'):
                manager.publish_local_source(source, roots['host'] / 'a')
            self.assertEqual(source.read_bytes(), b'not a managed copy')
            self.assertEqual(legacy_file.read_bytes(), b'not reusable HOST')
