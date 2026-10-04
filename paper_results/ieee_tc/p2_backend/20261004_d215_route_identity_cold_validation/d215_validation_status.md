# D215 7B route-identity cold-scope validation

## Decision

**Revoked after bounded validation; no 4,000-request replay or baseline run is qualified.**

The 100-request run completed 100/100 terminal records, 100 successful requests, and 100 fixed-output contract matches. Observed physical lifecycle usage was **779.8913742139703 GPU-s**; all 4 allocation journals passed the exact owner-event order and open leases were 0.

## Route evidence

The authoritative coordination record reports 107 collections, 7 stale rejections and 1 joined collections. The request snapshots contain 87 unique collection IDs. In those records, 37 collections include a cold full read (88 replica reads), while 60 include a full fallback (142 replica reads).

The emitted `cache_hits` and `identity_only` values are cumulative run counters, not per-request events; they are retained in the JSON but are not used to claim a reuse rate. The repeated authoritative fallback on cached scopes means the intended low-cost identity-recheck path was not dominant enough to justify a formal replay. This is a mechanism qualification failure, not evidence of numerical correctness or a cross-system performance result.

## Delivery and caveats

The outcome contains 29 transfer records, joined one-to-one with the copied remote journal (29 lines; 0 retained out-of-run lines). All joined records use prepublished gzip delivery, with no request-time packing or temporary archive.

The descriptive latency and GPU observations remain available in `d215_validation_curated.json`; they must not enter G1/G2, SLO, ranking, or paper comparison tables.
