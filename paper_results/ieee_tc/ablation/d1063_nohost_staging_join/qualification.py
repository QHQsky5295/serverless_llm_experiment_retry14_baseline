"""Reuse bounded D1063 qualification for native staging ownership only."""
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
    'tests.test_ieee_tc_gpu_references.NativeDemandTransactions',
    'tests.test_ieee_tc_gpu_references.ProtectedFileDemandTransactions',
    'tests.test_ieee_tc_gpu_references.PrivateNativeHostConstruction',
    'tests.test_ieee_tc_gpu_references.NativeFileHostPreparation',
]
qualification.SOURCES = [
    'faaslora/memory/residency_manager.py',
    'tests/test_ieee_tc_nohost.py',
    'tests/test_ieee_tc_nohost_staging.py',
    'tests/test_ieee_tc_gpu_references.py',
]

if __name__ == '__main__':
    raise SystemExit(qualification.main())
