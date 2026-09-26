"""Exhaustive and edge-case checks of IEEE's conditional planning primitives."""
import itertools
import random
import unittest

from faaslora.preloading.preloading_planner import (
    KnapsackItem, PreparationCandidate, PreloadingPlanner, PreparationClass,
    PreparationCostModel, PreparationOption,
)
from faaslora.experiment.hotness_tracker import HotnessTracker
from faaslora.experiment.experiment_stack import ExperimentStack
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

    def test_remaining_replacement_uses_each_victim_once_and_precedes_lower_tiers(self):
        candidates = [self.candidate('a', source=40), self.candidate('b', source=20),
                      self.candidate('c', source=16), self.candidate('a', T.HOST, source=40, remaining=5)]
        victims = [dict(adapter_id='old', adapter_int_id=8, eligible=True,
                        usable_bytes=MIB, loss_ms=2.)]
        selected, meta = self.p.select_ieee_insertions(candidates,
            {T.GPU:MIB, T.HOST:MIB, T.NVME:0}, gpu_replacement=victims)
        self.assertEqual([c.artifact_id for c in selected[T.GPU]], ['a','b'])
        self.assertFalse(selected[T.HOST])
        self.assertEqual(meta['gpu']['selected_bytes'], MIB)  # Original insertion set.
        self.assertEqual(meta['gpu']['final_selected_bytes'], 2*MIB)
        self.assertEqual(meta['gpu']['rejected_remaining'][0]['adapter_id'], 'c')
        self.assertEqual(meta['gpu']['virtual_remaining_bytes'], 0)
        self.assertTrue(victims[0]['eligible'])  # Caller snapshot not consumed/mutated.

    def test_replacement_strict_loss_test_does_not_consume_rejected_victim(self):
        candidates = [self.candidate('a', source=8), self.candidate('b', source=4)]
        victims = [dict(adapter_id='old', adapter_int_id=8, eligible=True,
                        usable_bytes=MIB, loss_ms=2.)]
        selected, meta = self.p.select_ieee_insertions(candidates,
            {T.GPU:0,T.HOST:0,T.NVME:0}, gpu_replacement=victims)
        self.assertFalse(selected[T.GPU])
        self.assertEqual([r['victim_adapter_ids'] for r in meta['gpu']['rejected_remaining']], [[8],[8]])
        self.assertTrue(all(r['reason']=='benefit_not_greater_than_loss'
                            for r in meta['gpu']['rejected_remaining']))

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


class IEEEPlanningEntryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.tracker = HotnessTracker(None, 10., clock=lambda: self.now)
        self.planner = PreloadingPlanner.__new__(PreloadingPlanner)
        self.planner.max_dp_buffer_bytes = 16*MIB
        self.stack = ExperimentStack.__new__(ExperimentStack)
        self.stack.hotness_tracker = self.tracker
        self.stack.preloading_planner = self.planner
        self.remote = PreparationClass('remote', 'tar', 'fixture-layout', 0)
        self.host = PreparationClass('host', 'native_tensors', 'fixture-layout', 0)
        self.gpu = PreparationClass('gpu', 'native_slots', 'fixture-layout', 0)
        self.costs = PreparationCostModel({self.remote: 20., self.host: 5.},
            beta=.5, profile_id='fixture-no-model-qualification')
        self.options = [PreparationOption('a', self.remote, self.gpu, 4*MIB),
                        PreparationOption('a', self.remote, self.host, MIB)]
        self.budgets = {T.GPU: 4*MIB, T.HOST: MIB, T.NVME: 0}

    def plan(self, mode='residency', **kwargs):
        return self.stack.plan_ieee_preparation(**(dict(mode=mode, options=self.options,
            budgets=self.budgets, costs=self.costs, source_snapshot_id='fixture-owner-epoch-1') | kwargs))

    def test_actual_stack_uses_arrivals_not_registry_hotness(self):
        empty = self.plan()
        self.assertFalse(any(empty['selected'].values()))
        self.tracker.record_arrival('a')
        self.tracker.record_arrival('b')
        result = self.plan()
        candidate = result['selected']['gpu'][0]
        self.assertEqual(candidate.demand_fraction, .5)
        self.assertEqual(candidate.benefit_ms, 10.)
        self.assertFalse(result['physical_resources_reserved'])
        self.assertEqual(result['selected']['host'], ())
        self.now = 111.
        self.assertFalse(any(self.plan()['selected'].values()))
        # Previously issued plan remains tied to its old observed window.
        self.assertEqual(candidate.demand_fraction, .5)

    def test_handoff_and_residency_apply_different_paper_rules(self):
        self.tracker.record_arrival('a')
        handoff, residency = self.plan('handoff'), self.plan('residency')
        self.assertEqual(handoff['selected']['host'][0].artifact_id, 'a')
        self.assertEqual(residency['selected']['gpu'][0].artifact_id, 'a')
        self.assertEqual(handoff['diagnostics']['host'], 0)
        self.assertNotEqual(handoff['plan_sha256'], residency['plan_sha256'])

    def test_missing_positive_demand_profile_rejects_not_zero_or_nearest_class(self):
        missing = PreparationClass('remote', 'other-representation', 'fixture-layout', 0)
        options = [PreparationOption('a', missing, self.gpu, MIB)]
        self.assertFalse(any(self.plan(options=options)['selected'].values()))
        self.tracker.record_arrival('a')
        with self.assertRaises(KeyError):
            self.plan(options=options)

    def test_stable_hash_and_input_copy(self):
        self.tracker.record_arrival('a')
        first = self.plan()
        self.assertEqual(first['plan_sha256'], self.plan(options=reversed(self.options))['plan_sha256'])
        self.assertNotEqual(first['plan_sha256'], self.plan(source_snapshot_id='next-owner-epoch')['plan_sha256'])
        self.budgets[T.GPU] = 0
        self.assertEqual(first['remaining_bytes']['gpu'], 4*MIB)
        self.assertNotEqual(first['plan_sha256'], self.plan()['plan_sha256'])

    def test_one_atomic_cost_snapshot_and_epoch_consistency(self):
        from unittest.mock import Mock
        self.tracker.record_arrival('a')
        self.costs.snapshot = Mock(wraps=self.costs.snapshot)
        self.plan()
        self.costs.snapshot.assert_called_once_with()
        for options in ([self.options[0]]*2, [self.options[0],
                PreparationOption('a', self.host, self.gpu, MIB)]):
            with self.assertRaises(ValueError):
                self.plan(options=options)
        with self.assertRaises(ValueError):
            self.plan(budgets={T.GPU: MIB})
        with self.assertRaises(ValueError):
            self.plan(mode='legacy')


if __name__ == '__main__':
    unittest.main()
