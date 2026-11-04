"""Targeted follow-up for legacy source guards, reusing file policy receipt."""
import qualification

q = qualification.qualification
q.NAMES = [
    'tests.test_ieee_tc_nohost_files',
    'tests.test_ieee_tc_request_lifecycle.LocalSourceOwnership.test_replacement_is_deferred_but_same_copy_and_other_tier_remain_usable',
    'tests.test_ieee_tc_request_lifecycle.ConfirmedFilePublication.test_verified_host_copy_preserves_content_and_lower_tier_after_eviction',
    'tests.test_ieee_tc_request_lifecycle.ConfirmedFilePublication.test_corrupt_tier_copy_does_not_publish_fast_tier_or_invalidate_source',
]
q.SOURCES += ['tests/test_ieee_tc_request_lifecycle.py']

if __name__ == '__main__':
    raise SystemExit(q.main())
