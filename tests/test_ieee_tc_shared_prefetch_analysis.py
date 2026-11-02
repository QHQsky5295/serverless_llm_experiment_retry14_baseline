import copy
import unittest

from scripts.ieee_tc.audit_d703_3b_full import (
    route_confirmed_preparation_row,
    route_prefetch_transfer_for_native,
)


class SharedPrefetchAnalysis(unittest.TestCase):
    def fixture(self):
        publication = dict(owner_id='file-owner', adapter_id='a', path='/cache/a',
            content_sha256='content', tier='nvme', content_verified=True)
        transfer = dict(transfer_id='transfer', artifact_id='a', state='published',
            loading_clock_id='clock', loading_started_monotonic_s=11.,
            published_monotonic_s=12., client_completed_monotonic_s=12.5,
            target_path='/cache/a', transferred_bytes=100,
            confirmed_file_publication=publication)
        early = dict(kind='route_confirmed_arrival_file_preparation_v1',
            policy='route_confirmed_arrival_prefetch_v1', request_id='r', adapter_id='a',
            physical_gpu_reservation=False, admission_granted=False, state='completed',
            clock_id='clock', started_monotonic_s=10.5, settled_monotonic_s=13.,
            selected_replica_id='replica-1',
            selected_source=dict(owner_id='file-owner', content_sha256='content'),
            preparation=dict(state='published', target_path='/cache/a',
                remote_transfer=copy.deepcopy(transfer), _movement=dict(owns_io=True)))
        request = dict(request_id='r', adapter_id='a', readiness_tier_before_dispatch='remote',
            external_arrival_timing=dict(planned_arrival_s=10.),
            native_token_timing=dict(native_clock_id='clock', controller_admitted_monotonic_s=10.8),
            gpu_reference_evidence=dict(route_confirmed_arrival_file_preparation=early,
                receipt=dict(file_source_witness=copy.deepcopy(publication), lora_path='/cache/a',
                    clock_id='clock', native_load_invoked=True, native_load_started_monotonic_s=13.1)))
        return request, {'transfer': transfer}

    def test_completed_after_admission_joins_without_mutation(self):
        request, transfers = self.fixture()
        before = copy.deepcopy(request)
        row = route_confirmed_preparation_row(request, transfers)
        self.assertEqual(row['owned_transferred_bytes'], 100)
        self.assertFalse(row['completed_before_admission'])
        self.assertEqual(row['task_elapsed_ms'], 2500.)
        self.assertEqual(row['publication_to_settled_ms'], 1000.)
        self.assertEqual(route_prefetch_transfer_for_native(request, row, transfers), transfers['transfer'])
        self.assertEqual(request, before)

    def test_subscriber_can_start_later_but_cannot_charge_bytes(self):
        request, transfers = self.fixture()
        early = request['gpu_reference_evidence']['route_confirmed_arrival_file_preparation']
        early['started_monotonic_s'] = 11.5
        early['preparation']['_movement']['owns_io'] = False
        row = route_confirmed_preparation_row(request, transfers)
        self.assertEqual(row['owned_transferred_bytes'], 0)
        self.assertEqual(row['start_to_fetch_ms'], 0.)
        early['preparation']['_movement']['owns_io'] = True
        with self.assertRaises(ValueError):
            route_confirmed_preparation_row(request, transfers)

    def test_noncompleted_and_missing_are_not_zero_preparation(self):
        request, transfers = self.fixture()
        early = request['gpu_reference_evidence']['route_confirmed_arrival_file_preparation']
        for state in ('skipped', 'cancelled', 'failed'):
            early['state'] = state
            row = route_confirmed_preparation_row(request, transfers)
            self.assertIsNone(row['task_elapsed_ms'])
            self.assertIsNone(row['owned_transferred_bytes'])
        del request['gpu_reference_evidence']['route_confirmed_arrival_file_preparation']
        self.assertIsNone(route_confirmed_preparation_row(request, transfers))

    def test_exact_receipt_and_clock_required(self):
        for key, value in [('request_id', 'different'), ('clock_id', 'remote-clock'),
                           ('started_monotonic_s', 9.), ('settled_monotonic_s', 12.),
                           ('physical_gpu_reservation', True), ('admission_granted', True)]:
            request, transfers = self.fixture()
            request['gpu_reference_evidence']['route_confirmed_arrival_file_preparation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                route_confirmed_preparation_row(request, transfers)
        request, transfers = self.fixture()
        transfers['transfer']['transferred_bytes'] = 101
        with self.assertRaises(ValueError):
            route_confirmed_preparation_row(request, transfers)

    def test_native_file_identity_must_match_and_reuse_is_not_zero_load(self):
        for key in ('owner_id', 'adapter_id', 'path', 'content_sha256', 'tier'):
            request, transfers = self.fixture()
            row = route_confirmed_preparation_row(request, transfers)
            request['gpu_reference_evidence']['receipt']['file_source_witness'][key] = 'other'
            with self.subTest(key=key), self.assertRaises(ValueError):
                route_prefetch_transfer_for_native(request, row, transfers)
        request, transfers = self.fixture()
        row = route_confirmed_preparation_row(request, transfers)
        request['gpu_reference_evidence']['receipt']['native_load_invoked'] = False
        self.assertIsNone(route_prefetch_transfer_for_native(request, row, transfers))
        self.assertEqual(row['transfer_id'], 'transfer')


if __name__ == '__main__':
    unittest.main()
