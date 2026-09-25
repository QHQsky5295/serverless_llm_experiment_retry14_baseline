"""Exhaustive and edge-case checks of IEEE's conditional planning primitives."""
import itertools
import random
import unittest

from faaslora.preloading.preloading_planner import (
    KnapsackItem, PreparationCandidate, PreloadingPlanner,
)
from faaslora.registry.schema import StorageTier as T

MIB = 1024**2


class IEEEPlanningTests(unittest.TestCase):
    def setUp(self):
        self.p = PreloadingPlanner.__new__(PreloadingPlanner)
        self.p.max_dp_buffer_bytes = 16 * MIB
        self.budgets = {T.GPU: 4*MIB, T.HOST: 4*MIB, T.NVME: 4*MIB}

    def candidate(self, name, target=T.GPU, size=MIB, hotness=.25, source=20, remaining=0):
        return PreparationCandidate(name, T.REMOTE, target, size, hotness, source, remaining)

    def test_benefit_not_density_times_rounded_size(self):
        c = self.candidate('a', size=MIB+1, hotness=.5, source=20)
        self.assertEqual(c.benefit_ms, 10)
        chosen, meta = self.p.select_ieee_insertions([c], self.budgets)
        self.assertEqual(chosen[T.GPU], [c])
        self.assertEqual(meta['gpu']['total_value'], 10)

    def test_sub_mib_and_zero_budget_do_not_round_capacity_up(self):
        for capacity in [0, MIB//2, MIB-1]:
            selected, _ = self.p.select_benefit_items([KnapsackItem('a', 3*MIB//4, 1)], capacity)
            self.assertEqual(selected, [])

    def test_exhaustive_small_problems(self):
        # These are mathematical unit-test inputs, not generated serving traces.
        rng = random.Random(704)
        for trial in range(80):
            items = [KnapsackItem(str(i), rng.randint(1, 12)*MIB//4,
                                  rng.randint(0, 30)/4) for i in range(8)]
            cap_units = rng.randint(1, 9)
            selected, meta = self.p.select_benefit_items(items, cap_units*MIB + MIB//2)
            feasible = (subset for k in range(9) for subset in itertools.combinations(items, k)
                        if sum((x.weight+MIB-1)//MIB for x in subset) <= cap_units)
            optimum = max(sum(x.value for x in subset) for subset in feasible)
            self.assertEqual(sum(x.value for x in selected), optimum, trial)
            self.assertLessEqual(sum(x.weight for x in selected), cap_units*MIB)
            self.assertEqual(meta['algorithm'], 'conservative_mib_dp')

    def test_single_item_cannot_be_used_twice(self):
        selected, meta = self.p.select_benefit_items([KnapsackItem('a', MIB, 3)], 3*MIB)
        self.assertEqual(len(selected), 1)
        self.assertEqual(meta['total_value'], 3)

    def test_large_table_scan_respects_original_bytes(self):
        self.p.max_dp_buffer_bytes = 1
        items = [KnapsackItem('a', 3*MIB//4, 5), KnapsackItem('b', 3*MIB//4, 4),
                 KnapsackItem('c', MIB, 1)]
        chosen, meta = self.p.select_benefit_items(items, 3*MIB//2)
        self.assertEqual([x.artifact_id for x in chosen], ['a', 'b'])
        self.assertEqual(meta['algorithm'], 'raw_byte_density_scan')
        self.assertEqual(meta['selected_bytes'], 3*MIB//2)
        self.assertEqual(meta['dp_buffer_bytes'], 0)

    def test_stable_ties_do_not_depend_on_input_order(self):
        items = [KnapsackItem('b', MIB, 1), KnapsackItem('a', MIB, 1)]
        for values in [items, list(reversed(items))]:
            chosen, _ = self.p.select_benefit_items(values, MIB)
            self.assertEqual([x.artifact_id for x in chosen], ['a'])

    def test_faster_tier_precedes_and_excludes_slower_target(self):
        candidates = [self.candidate('a'), self.candidate('a', T.HOST, remaining=5),
                      self.candidate('b', T.HOST, remaining=5)]
        chosen, _ = self.p.select_ieee_insertions(candidates, self.budgets)
        self.assertEqual([c.artifact_id for c in chosen[T.GPU]], ['a'])
        self.assertEqual([c.artifact_id for c in chosen[T.HOST]], ['b'])

    def test_handoff_uses_density_and_only_one_target(self):
        # HOST saves less but takes much less space: its density wins for a.
        candidates = [self.candidate('a', size=4*MIB),
                      self.candidate('a', T.HOST, size=MIB, remaining=5),
                      self.candidate('b', T.NVME, remaining=10)]
        chosen, remaining = self.p.select_ieee_handoff(candidates, self.budgets)
        self.assertEqual(chosen[T.GPU], [])
        self.assertEqual([c.artifact_id for c in chosen[T.HOST]], ['a'])
        self.assertEqual([c.artifact_id for c in chosen[T.NVME]], ['b'])
        self.assertEqual(remaining[T.HOST], 3*MIB)

    def test_zero_demand_or_nonpositive_gain_is_not_prepared(self):
        candidates = [self.candidate('zero', hotness=0),
                      self.candidate('slower', T.HOST, source=2, remaining=5)]
        chosen, _ = self.p.select_ieee_insertions(candidates, self.budgets)
        self.assertFalse(any(chosen.values()))

    def test_missing_budgets_and_conflicting_epochs_rejected(self):
        with self.assertRaises(ValueError):
            self.p.select_ieee_insertions([], {T.GPU: MIB})
        with self.assertRaises(ValueError):
            self.p.select_ieee_handoff([self.candidate('a'),
                self.candidate('a', T.HOST, hotness=.5, remaining=5)], self.budgets)
        with self.assertRaises(ValueError):
            self.p.select_ieee_insertions([self.candidate('a')]*2, self.budgets)

    def test_invalid_cost_or_non_executable_gpu_rejected(self):
        for kwargs in [{'source': float('nan')}, {'remaining': 1}, {'hotness': 2}, {'size': 0}]:
            with self.assertRaises(ValueError):
                self.candidate('a', **kwargs)

    def test_duplicate_invalid_items_rejected(self):
        for items in [[KnapsackItem('a', MIB, 1)]*2,
                      [KnapsackItem('a', 0, 1)], [KnapsackItem('a', MIB, float('inf'))]]:
            with self.assertRaises(ValueError):
                self.p.select_benefit_items(items, MIB)


if __name__ == '__main__':
    unittest.main()
