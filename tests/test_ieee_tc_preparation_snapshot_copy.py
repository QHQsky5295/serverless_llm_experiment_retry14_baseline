"""Value-preserving owner snapshots, not a relaxation of live feasibility."""
import copy
import hashlib
import json
import math
import unittest

from faaslora.preloading.preloading_planner import (
    PreparationCandidate, copy_ieee_preparation_plan, deserialize_ieee_handoff_plan,
    freeze_preparation_source_view, ieee_plan_comparison_sha256,
    rebind_ieee_handoff_plan, serialize_ieee_handoff_plan)
from faaslora.registry.schema import StorageTier


class PreparationSnapshotCopy(unittest.TestCase):
    def view(self):
        source = dict(tier='host', owner_id='owner', content='a'*64,
                      sizes=[0, 1, 2**60], verified=True, ratio=.125, path='制品')
        return dict(native=None, sources={'adapter': dict(selected_source=source,
            confirmed_copies=[source])}, epoch=4, reservations={}, flag=False)

    def test_exact_canonical_hash_and_all_mutable_children_are_detached(self):
        view = self.view()
        detached, digest = freeze_preparation_source_view(view)
        expected = json.dumps(copy.deepcopy(view),sort_keys=True,
                              separators=(',', ':'),allow_nan=False).encode()
        self.assertEqual(digest,hashlib.sha256(expected).hexdigest())
        self.assertEqual(detached,view)
        detached['sources']['adapter']['selected_source']['sizes'].append(3)
        detached['reservations']['new'] = 4
        self.assertEqual(view,self.view())
        view['sources']['adapter']['confirmed_copies'][0]['path'] = 'changed producer'
        self.assertEqual(detached['sources']['adapter']['confirmed_copies'][0]['path'],'制品')

    def test_no_json_coercion_nonfinite_or_cycles(self):
        for bad in ({1:'integer key'}, {'sequence':(1,2)}, {'x':math.nan}, {'x':math.inf}):
            with self.subTest(bad=bad), self.assertRaises((ValueError,TypeError)):
                freeze_preparation_source_view(bad)
        cyclic = {}
        cyclic['self'] = cyclic
        with self.assertRaises(ValueError):
            freeze_preparation_source_view(cyclic)

    def test_plan_preserves_python_selected_candidates_and_detaches_diagnostics(self):
        candidate = PreparationCandidate('adapter', StorageTier.REMOTE, StorageTier.GPU,
                                         4096, .5, 10., 0.)
        plan = dict(source_view=self.view(), selected={'gpu':(candidate,)},
                    diagnostics={'changes':[dict(epoch=4)]})
        copied = copy_ieee_preparation_plan(plan)
        self.assertEqual(copied,copy.deepcopy(plan))
        self.assertIsInstance(copied['selected']['gpu'],tuple)
        self.assertIsInstance(copied['selected']['gpu'][0],PreparationCandidate)
        copied['source_view']['sources']['adapter']['selected_source']['sizes'].append(99)
        copied['diagnostics']['changes'][0]['epoch'] = 5
        self.assertEqual(plan['source_view'],self.view())
        self.assertEqual(plan['diagnostics']['changes'][0]['epoch'],4)

    def test_historical_explicit_plan_without_joint_view_keeps_its_copy_contract(self):
        plan = {'options':[{'a':[1]}]}
        copied = copy_ieee_preparation_plan(plan)
        self.assertEqual(copied,plan)
        copied['options'][0]['a'].append(2)
        self.assertEqual(plan,{'options':[{'a':[1]}]})

    def test_cross_run_identity_ignores_owner_binding_but_keeps_selection(self):
        candidate = PreparationCandidate('adapter', StorageTier.REMOTE, StorageTier.GPU,
                                         4096, .5, 10., 0.)
        source = dict(tier='remote', owner_id='immutable-artifact-origin',
                      content_sha256='a'*64, representation='remote', path=None)
        view = dict(kind='ieee_owned_preparation_view_v1', native=None,
                    files=dict(owner_id='owner-a', epoch=4, captured_at=10.0,
                               artifacts={'adapter': {'sources': [source]}},
                               managed_host={'activation_reservations': {'act-a': 1}},
                               budgets={'tiers': {'host': {'remaining_bytes': 8},
                                                  'nvme': {'remaining_bytes': 8}}}),
                    sources={'adapter': {'selected_source': source,
                                         'confirmed_copies': [source]}},
                    remaining_bytes={'gpu': 4096, 'host': 8, 'nvme': 8},
                    activation_id='act-a', inherited_layout={'slot_count': 1})
        plan = dict(kind='ieee_preparation_epoch_v1', mode='handoff',
                    source_snapshot_id=freeze_preparation_source_view(view)[1],
                    profile_id='profile', cost_sequence=1, demand_observed_at=10.0,
                    window_seconds=5.0, total_arrivals=2,
                    arrival_counts={'adapter': 1},
                    remaining_bytes={'gpu': 4096, 'host': 8, 'nvme': 8},
                    options=[], cost_estimates=[], physical_resources_reserved=False,
                    selected={'gpu': (candidate,), 'host': (), 'nvme': ()},
                    source_view=view)
        bundle = serialize_ieee_handoff_plan(plan)
        restored = deserialize_ieee_handoff_plan(bundle)
        self.assertEqual(ieee_plan_comparison_sha256(plan), bundle['comparison_plan_sha256'])
        rebound_view = copy.deepcopy(view)
        rebound_view['files']['owner_id'] = 'owner-b'
        rebound_view['files']['epoch'] = 9
        rebound_view['files']['captured_at'] = 22.0
        rebound_view['activation_id'] = 'act-b'
        rebound_view['files']['managed_host']['activation_reservations'] = {'act-b': 1}
        rebound = rebind_ieee_handoff_plan(restored, current_source_view=rebound_view)
        self.assertEqual(rebound['selected']['gpu'][0], candidate)
        self.assertNotEqual(rebound['source_snapshot_id'], plan['source_snapshot_id'])
        self.assertEqual(ieee_plan_comparison_sha256(rebound), bundle['comparison_plan_sha256'])

    def test_cross_run_identity_rejects_logical_source_change(self):
        candidate = PreparationCandidate('adapter', StorageTier.REMOTE, StorageTier.GPU,
                                         4096, .5, 10., 0.)
        view = dict(native=None, sources={'adapter': {'selected_source': {
            'tier': 'remote', 'content_sha256': 'a'*64, 'path': 'one'}}},
                    activation_id='a')
        plan = dict(kind='ieee_preparation_epoch_v1', mode='handoff',
                    source_snapshot_id=freeze_preparation_source_view(view)[1],
                    profile_id='p', cost_sequence=1, demand_observed_at=1.0,
                    window_seconds=5.0, total_arrivals=1, arrival_counts={'adapter': 1},
                    remaining_bytes={'gpu': 4096}, options=[], cost_estimates=[],
                    physical_resources_reserved=False,
                    selected={'gpu': (candidate,)}, source_view=view)
        restored = deserialize_ieee_handoff_plan(serialize_ieee_handoff_plan(plan))
        changed = copy.deepcopy(view)
        changed['sources']['adapter']['selected_source']['content_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            rebind_ieee_handoff_plan(restored, current_source_view=changed)


if __name__ == '__main__':
    unittest.main()
