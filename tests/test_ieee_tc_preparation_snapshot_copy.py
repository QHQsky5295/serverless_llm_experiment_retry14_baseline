"""Value-preserving owner snapshots, not a relaxation of live feasibility."""
import copy
import hashlib
import json
import math
import unittest

from faaslora.preloading.preloading_planner import (
    PreparationCandidate, copy_ieee_preparation_plan, freeze_preparation_source_view)
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


if __name__ == '__main__':
    unittest.main()
