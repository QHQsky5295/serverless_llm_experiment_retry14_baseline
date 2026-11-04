"""Bounded native NoHOST qualification, reusing the established receipt runner."""
import importlib.util
from pathlib import Path

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_qualification', RAW.parent / 'qualification.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
qualification.RAW = RAW
qualification.NAMES = [
    'tests.test_ieee_tc_nohost.NoHostNativeLifecycle',
    'tests.test_ieee_tc_nohost_staging.NoHostStaging',
    'tests.test_ieee_tc_nohost_native_authorization.NoHostNativeAuthorization',
    'tests.test_ieee_tc_nohost_native_authorization.NoHostPrivateStorageObservation',
    'tests.test_ieee_tc_nohost_sources.NoHostSourceAuthorization',
    'tests.test_ieee_tc_nohost_sources.ProtectedFileCapacityWait',
    'tests.test_ieee_tc_nohost_sources.CpuPreparationReleaseWitness',
    'tests.test_ieee_tc_gpu_references.NativeDemandTransactions',
    'tests.test_ieee_tc_gpu_references.ProtectedFileDemandTransactions',
    'tests.test_ieee_tc_gpu_references.NativeProactiveTransactions',
    'tests.test_ieee_tc_gpu_references.PrivateNativeHostConstruction',
    'tests.test_ieee_tc_gpu_references.NativeFileHostPreparation',
    'tests.test_ieee_tc_worker_observation.NativeHostFootprint',
    'tests.test_ieee_tc_worker_observation.NativeHostRoutingProjection',
]
qualification.SOURCES = [
    'faaslora/memory/residency_manager.py', 'faaslora/memory/gpu_monitor.py', 'scripts/run_all_experiments.py',
    'tests/test_ieee_tc_nohost.py', 'tests/test_ieee_tc_nohost_staging.py',
    'tests/test_ieee_tc_nohost_sources.py', 'tests/test_ieee_tc_nohost_native_authorization.py',
    'tests/test_ieee_tc_gpu_references.py',
    'tests/test_ieee_tc_worker_observation.py',
]

if __name__ == '__main__':
    raise SystemExit(qualification.main())
