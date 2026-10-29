# D953 7B historical `isal`/file-state diagnostic outcome

## Scope

This run was a single 4,000-request validation diagnostic for the current
7B PrimeLoRA candidate. It reused the frozen W2 trace, adapter map, endpoint,
vLLM 0.30.0 runtime, and D931 resource configuration, while additionally
enabling the historical `FAASLORA_FILE_STATE_PROCESS=1` and
`FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal` settings. It was not a formal
comparison point and was intended only to determine whether that historical
environment explained the older D917 result.

- Plan SHA-256: `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`
- Result directory:
  `results/ieee_tc/p2_backend_qualification/d953_20261030/7b_isal_file_state_full_w2/`
- Requested requests: 4,000
- Native terminal records: 2,204/4,000
- Native contract-complete records: 2,204/4,000
- Interrupted replays: 7
- Formal-comparison qualified: no
- Physical resource measurement complete: no

## Why the run is rejected

At request 2,204, a background residency preparation observed a later native
source epoch and correctly raised `PreparationPlanSuperseded` at
`native_staging_source`. During the required ownership-settlement path,
`_execute_ieee_file_preparation_plan` attempted to join an intent that had
been generated before the shared movement queue accepted it. The intent was
therefore not present in `OwnedMovementQueue._intents`, and cleanup raised a
`KeyError` instead of closing the preparation epoch cleanly.

This is an execution-integrity failure, not evidence of a latency, GPU-memory,
or remote-transfer result. The partial run is retained as diagnostic evidence
and is excluded from every comparison, aggregate, and figure.

## Corrective change

The ownership invariant is now explicit: a file-movement intent is added to a
plan's cleanup set only after `OwnedMovementQueue.submit()` accepts it. The
queue entry callback is invoked at that boundary, so a pre-submit validation or
context failure cannot create a later cleanup join for a non-existent physical
operation. The regression test is
`OwnedFileMovement.test_file_intent_registration_callback_matches_queue_ownership`.

Code backup commit: `f1033be6` (`Fix owned file intent registration before cleanup`).
The commit was pushed to both `origin` and `faaslora_origin` as
`backup/d953-intent-registration-fix-20261030`.

Validation after the change:

- `tests/test_ieee_tc_service_routing.py`: 72 passed.
- New intent-registration regression test: passed under the vLLM 0.30.0
  environment.
- `tests/test_ieee_tc_transfer_pressure.py`: 194 passed; 12 pre-existing
  failures remain in unrelated native-host fault fixtures (the suite reports
  194 passed, 12 failed). They are not attributed to this change.

## Decision

D953 is **incomplete and rejected**. It does not replace D931, does not
support a claim about the historical `isal`/file-state environment, and does
not authorize starting baseline campaigns. A fresh full validation may be
considered only after the corrective commit passes the bounded smoke and the
same frozen configuration is rerun from a cold deployment.
