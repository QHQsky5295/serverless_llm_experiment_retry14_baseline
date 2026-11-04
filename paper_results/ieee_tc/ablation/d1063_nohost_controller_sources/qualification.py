"""Targeted NoHOST controller checks; reuse bounded D1063 receipt runner."""
import importlib.util
from pathlib import Path

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_bounded', RAW.parent / 'qualification.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.RAW = RAW
q.NAMES = [
    'tests.test_ieee_tc_nohost_controller.NoHostController',
    'tests.test_ieee_tc_nohost_controller.NoHostGpuFallback',
    'tests.test_ieee_tc_nohost_controller.NoHostReleaseQueue',
    'tests.test_ieee_tc_nohost_controller.NoHostCostIntervals',
    'tests.test_ieee_tc_nohost_controller.NoHostInstallation',
    'tests.test_ieee_tc_transfer_pressure.ManagedHostOwnership',
    'tests.test_ieee_tc_transfer_pressure.OwnedMovements',
    'tests.test_ieee_tc_transfer_pressure.AutomaticGPUReplacement',
    'tests.test_ieee_tc_transfer_pressure.AutomaticFileReplacement',
    'tests.test_ieee_tc_transfer_pressure.ProactiveCostFeedback',
    'tests.test_ieee_tc_preparation_cost.ProactivePreparationIntervals',
    'tests.test_ieee_tc_nohost_sources.CpuPreparationReleaseWitness',
]
q.SOURCES = [
    'faaslora/memory/residency_manager.py', 'faaslora/memory/gpu_monitor.py',
    'faaslora/experiment/experiment_stack.py',
    'faaslora/preloading/preloading_manager.py', 'faaslora/preloading/preloading_planner.py',
    'scripts/run_all_experiments.py', 'tests/test_ieee_tc_nohost_controller.py',
    'tests/test_ieee_tc_transfer_pressure.py',
]
if __name__ == '__main__':
    raise SystemExit(q.main())
