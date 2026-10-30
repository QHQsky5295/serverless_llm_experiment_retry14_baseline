# D958 7B historical file-state diagnostic outcome (2026-10-30)

## Scope and provenance

D958 reran the D957 diagnostic after the async owner-context correction. It
used the unchanged D931 7B W2 configuration, trace, 500-adapter map, real
remote HTTP endpoint, `FAASLORA_FILE_STATE_PROCESS=1`, the historical `isal`
decoder, and the D838 dependency overlay. The output and NVMe cache were new
campaign paths. The governing plan hash was
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`.

This run was launched with `FAASLORA_FORMAL_RUN=0`; therefore it is a
diagnostic and not a replacement for the frozen PrimeLoRA mainline candidate.

## Replay and lifecycle evidence

- Requests completed: 4,000/4,000.
- Request failures: 0.
- Physical journal: 4,000 planned, 4,000 terminal, 4,000 native-contract-complete,
  0 interrupted.
- `launch.json`: `pass=true`, service and replay return codes both 0.
- Shutdown cleanup errors: none.
- Post-shutdown file owner: zero plans, zero movement interests, zero live
  contexts/leases/materializations/workspaces.

The last item is the decisive D958 result: the D957 orphan-context failure did
not recur after the correction.

## Diagnostic performance values

The run produced a complete result file, retained for audit but excluded from
the mainline comparison:

| Quantity | D958 value |
|---|---:|
| Average end-to-end TTFT | 1,361.2 ms |
| P95 end-to-end TTFT | 3,257.8 ms |
| Average TPOT | 31.4 ms |
| E2E SLO attainment (`TTFT <= 5 s`) | 96.625% |
| Average E2E | 4,854.7 ms |
| Physical lifecycle GPU-seconds | 13,310.3524 |
| Scale-up / scale-down events | 13 / 13 |

The larger physical residency than the current D931 candidate is an
environment diagnostic observation, not evidence that the historical decoder
or file-state process should be selected for PrimeLoRA. It must not be used to
retune the mainline or to claim a baseline comparison.

## Retention and cleanup

The raw D958 launch, replay, physical journal, and result JSON remain
immutable. The duplicate NVMe cache may be reclaimed only after this outcome
is committed; the cache is not part of the evidence needed to establish the
zero-live-dependency invariant.
