# D228: opt-in native token-ID capture for numerical correctness

Status: CPU-qualified implementation; no GPU or remote run.
Date: 2026-10-05 (Asia/Shanghai)

## Motivation and boundary

D225 showed that D210 and D219 have identical request/input/output-length
contracts but 136/4,000 completion-token digests differ.  A digest is enough
for audit joins but cannot identify the first divergent token or compare a
runtime output with an independent adapter reference.  D228 adds the smallest
possible observation needed for the bounded correctness gate: exact native
completion IDs are retained only when an explicit model flag is true.

The default remains unchanged.  Without `ieee_token_id_capture: true`, the
runner writes only `completion_token_ids_sha256`, exactly as before.  The flag
does not change sampling, batching, routing, admission, resource ownership,
timing, or any IEEE equation.  It is not permitted in a formal performance
run unless the run is explicitly labelled as a correctness diagnostic.

## Evidence contract

When enabled, the existing `native_token_timing` evidence object contains:

- `completion_token_ids`: the native integer `token_ids` sequence in order;
- `completion_token_ids_capture_contract`:
  `fixed_length_greedy_v1_native_ids_opt_in_v1`.

The sequence is validated as a list of non-negative Python integers before it
crosses the controller boundary.  The fixed-output path still requires the
native output count to equal its request target, and the existing SHA remains
present for compact joins.  Text re-tokenization, expected-token counts, or
SSE chunk counts cannot populate this field.

## CPU qualification

`tests/test_ieee_tc_token_capture.py` covers the three required cases:

1. default-off behavior leaves the digest-only schema untouched;
2. explicit opt-in preserves the exact integer sequence and contract tag;
3. strings and negative IDs fail closed.

The 43 existing native-timing and EuroSys runner tests also pass.  Python
compilation and `git diff --check` pass.  This is state/serialization evidence,
not a claim that a GPU run or numerical adapter correctness has passed.

## Use in the next bounded gate

A fresh 22-request correctness configuration may set
`model.ieee_token_id_capture=true` and use new output/cache roots.  It must
reuse the D166/D167 request identities and frozen content index, retain D207
execution mapping and D208 coordinate checks, and run only after the unchanged
physical-memory gate passes.  The captured sequences are then compared with an
independent original-checkpoint PEFT reference, including nonzero, zero, base,
and same-adapter repeat controls.  Exact equality is the primary criterion;
any non-determinism is reported with first-divergence/logit evidence rather
than hidden by a tolerance.

No D228 GPU result, `N_correct` upgrade, SLO value, or performance ranking exists
yet.
