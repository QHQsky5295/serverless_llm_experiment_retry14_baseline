"""Only the changed profile identity and real frozen NoHOST entry, CPU only."""
import importlib.util
from pathlib import Path

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_bounded',
    RAW.parent/'d1063_20261104/qualification.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.RAW = RAW
q.NAMES = [
    'tests.test_ieee_tc_service_routing.FrozenMeasuredInitialization',
    'tests.test_ieee_tc_preparation_cost.FrozenPreparationInitialization',
    'tests.test_ieee_tc_nohost_controller.NoHostInstallation',
]
q.SOURCES = [
    'faaslora/experiment/instance_pool.py',
    'tests/test_ieee_tc_service_routing.py',
    'tests/test_ieee_tc_nohost_controller.py',
]
if __name__ == '__main__':
    raise SystemExit(q.main())
