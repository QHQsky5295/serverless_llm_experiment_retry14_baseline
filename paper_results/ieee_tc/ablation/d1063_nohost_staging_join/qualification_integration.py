"""Reuse the same bounded runner for the existing Full staging integration."""
import qualification

qualification.qualification.NAMES = [
    'tests.test_ieee_tc_transfer_pressure.IntegratedNativeHostReplacement.test_arriving_demand_reuses_staging_and_proactive_work_rechecks',
]
qualification.qualification.SOURCES += ['tests/test_ieee_tc_transfer_pressure.py']

if __name__ == '__main__':
    raise SystemExit(qualification.qualification.main())
