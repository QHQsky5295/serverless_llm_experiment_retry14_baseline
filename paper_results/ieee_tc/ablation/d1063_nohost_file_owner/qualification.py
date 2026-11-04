"""Reuse D1063 bounded qualification for file-owner NoHOST policy only."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'qualification.py'
spec = importlib.util.spec_from_file_location('d1063_bounded_qualification', path)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
qualification.RAW = Path(__file__).resolve().parent
qualification.NAMES = [
    'tests.test_ieee_tc_nohost_files',
    'tests.test_ieee_tc_nohost',
    'tests.test_ieee_tc_file_state_process',
]
qualification.SOURCES += [
    'faaslora/memory/residency_manager.py',
    'faaslora/memory/file_state_process.py',
    'tests/test_ieee_tc_nohost_files.py',
    'tests/test_ieee_tc_file_state_process.py',
]

if __name__ == '__main__':
    raise SystemExit(qualification.main())
