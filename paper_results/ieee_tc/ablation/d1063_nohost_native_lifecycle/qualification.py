"""Reuse the D1063 bounded receipt; native lifecycle, not GPU qualification."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'qualification.py'
spec = importlib.util.spec_from_file_location('d1063_bounded_qualification', path)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
qualification.RAW = Path(__file__).resolve().parent
qualification.NAMES = [
    'tests.test_ieee_tc_nohost',
    'tests.test_ieee_tc_gpu_references.NativeReferences',
    'tests.test_ieee_tc_gpu_references.NativeDemandTransactions',
    'tests.test_ieee_tc_gpu_references.NativeHostDemandBudget',
    'tests.test_ieee_tc_gpu_references.NativeFileHostPreparation',
    'tests.test_ieee_tc_gpu_references.PrivateNativeHostConstruction',
    'tests.test_ieee_tc_gpu_references.NativeObjectiveReplacement',
    'tests.test_ieee_tc_gpu_references.OwnedNativePreparationPlans',
    'tests.test_ieee_tc_transfer_pressure.NonTargetBindingExpiry.test_binding_receipt_validation_keeps_actual_fallback_leases_on_bad_reply',
]
qualification.SOURCES += [
    'faaslora/memory/residency_manager.py',
    'tests/test_ieee_tc_gpu_references.py',
    'scripts/run_all_experiments.py',
    'tests/test_ieee_tc_transfer_pressure.py',
]

if __name__ == '__main__':
    raise SystemExit(qualification.main())
