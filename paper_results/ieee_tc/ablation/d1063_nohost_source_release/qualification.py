"""Bounded D1063 source/release integration, reusing the existing test runner."""
import importlib.util
from pathlib import Path

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_qualification', RAW.parent / 'qualification.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
qualification.RAW = RAW
qualification.NAMES = [
    'tests.test_ieee_tc_nohost.NoHostNativeLifecycle',
    'tests.test_ieee_tc_nohost_staging',
    'tests.test_ieee_tc_nohost_sources.NoHostSourceAuthorization',
    'tests.test_ieee_tc_nohost_sources.ProtectedFileCapacityWait',
    'tests.test_ieee_tc_nohost_sources.CpuPreparationReleaseWitness',
    'tests.test_ieee_tc_request_lifecycle.NativeCapacityWait',
    'tests.test_ieee_tc_request_lifecycle.SelectedSourceAdmissionIntegration.test_file_and_remote_sources_load_from_confirmed_reference_not_legacy_hint',
    'tests.test_ieee_tc_request_lifecycle.ControllerNativeReferenceLifecycle.test_cached_native_weights_do_not_require_the_original_file',
    'tests.test_ieee_tc_route_identity_cache.RouteIdentityCacheQualification.test_selected_file_recheck_retires_disproved_absence_before_retry',
    'tests.test_ieee_tc_transfer_pressure.OwnedNativeHostMovement',
    'tests.test_ieee_tc_transfer_pressure.AutomaticGPUReplacement.test_native_eviction_before_gpu_action_closes_epoch_and_fresh_same_key_serves',
    'tests.test_ieee_tc_transfer_pressure.AutomaticGPUReplacement.test_native_eviction_before_source_staging_closes_epoch_and_fresh_plan_serves',
    'tests.test_ieee_tc_transfer_pressure.AutomaticGPUReplacement.test_incomplete_or_foreign_absence_witness_is_not_supersession',
]
qualification.SOURCES = [
    'faaslora/memory/residency_manager.py', 'scripts/run_all_experiments.py',
    'tests/test_ieee_tc_nohost.py', 'tests/test_ieee_tc_nohost_staging.py',
    'tests/test_ieee_tc_nohost_sources.py', 'tests/test_ieee_tc_request_lifecycle.py',
    'tests/test_ieee_tc_transfer_pressure.py', 'tests/test_ieee_tc_route_identity_cache.py',
]

if __name__ == '__main__':
    raise SystemExit(qualification.main())
