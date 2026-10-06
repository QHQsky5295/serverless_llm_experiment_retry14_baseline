# D275: stale selected GPU copy is a re-routeable conflict

Date: 2026-10-06
Campaign: `p2_backend_qualification`
Scope: CPU-only native-reference contract; no GPU replay, remote service, or
performance measurement.

## Question and falsifiable hypothesis

D266 exposed a native integer/source rejection, but its retained failure record
does not include the old source, requested source, live-copy state, or
preparation-plan state.  The next narrow hypothesis was therefore independent
of the historical attribution:

> If a router selects a confirmed GPU copy and that copy is legally retired and
> replaced by an equivalent, same-content copy before the serialized native
> worker acquires it, the owner should return the existing
> `required_source_changed` re-resolution result.  It must not treat the stale
> path as an attempted live relabel, and it must not mutate the replacement.

This is the behavior required by the existing controller path, which abandons
the selected reference and performs whole-request re-routing for
`required_source_changed`.  It does **not** accept a new path, retry the native
load, or bypass source identity checks.

## Controlled witness

The CPU fixture performs the following real owner operations:

1. load adapter 4 from `/existing/adapter-4` with immutable content digest
   `a…a` (64 hexadecimal characters), then release the request lease;
2. capture the native source snapshot (including the confirmed GPU source ID);
3. deactivate and evict the old copy;
4. load the same logical adapter and digest from
   `/managed-host/adapter-4`, keeping that replacement lease live;
5. submit the delayed old snapshot with `required_source_tier=gpu`, the old
   source ID, old path, and the same digest.

The assertion is `acquired=false, reason=required_source_changed`, with the
replacement snapshot byte-for-byte unchanged.  Two negative controls submit
the same delayed witness with a different logical adapter name or a different
content digest; both must remain fail-closed with the historical source-conflict
exception.

## Result

| Check | Outcome |
|---|---:|
| Delayed equivalent copy returns typed re-route | PASS |
| Replacement remains live and unchanged | PASS |
| Wrong logical identity remains rejected | PASS |
| Wrong content identity remains rejected | PASS |
| Focused native-reference suite | 135 passed |
| Request lifecycle + pending GPU pipeline + transfer-pressure suites | 365 passed |

The initial pre-fix witness was intentionally run first and raised the source
conflict at `_validate_source_binding`; it is the red observation motivating
the change.  No production behavior was changed before that observation.

## Minimal implementation change

`IEEEBackendGPUReferences._load_and_acquire` now:

- preserves future-epoch rejection and lease-id checks;
- when an observed source ID is stale, returns a typed re-route only if the
  current live copy has the same logical adapter name and verified immutable
  content digest;
- invokes the unchanged strict `_validate_source_binding` for cross-adapter,
  content-mismatch, staged-source, and unqualified callers;
- leaves all native caches, pins, leases, and epochs unchanged on the typed
  conflict.

The existing controller handles `required_source_changed`; no retry or
fallback was added.  The test uses the vLLM 0.30.0 environment but allocates no
GPU and makes no claim about CUDA correctness or request performance.

## Interpretation boundary and next gate

This closes a demonstrated stale-selection contract bug class.  It does **not**
prove that D266 was caused by this branch: D266 lacks the required witness, and
the D275 fixture is not a replay of D266.  It also does not qualify readiness,
physical GPU release, SLO, G1/G2, or any baseline comparison.

Before another 7B GPU readiness run, the change must pass the complete
source/reference, request-lifecycle, pending-pipeline, and transfer-pressure
tests (recorded above), then the existing numerical/common-warm/physical-release
gates must be rechecked.  Any future replay remains a new, explicitly
identified run and must retain failure evidence.

## Provenance

- Test: `tests/test_ieee_tc_gpu_references.py::NativeSelectedCopyWitness::test_stale_gpu_selection_after_same_content_replacement_is_reroutable`
- Implementation: `faaslora/memory/residency_manager.py`,
  `IEEEBackendGPUReferences._load_and_acquire`
- Runtime: `/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python`,
  vLLM 0.30.0
- Resource scope: `MemoryHigh=3G`, `MemoryMax=4G`, `MemorySwapMax=0`,
  `AllowedCPUs=2,3,26,27`
- Existing primary sources consulted: vLLM 0.30.0 `worker_manager.py` cache
  ownership path and the repository's frozen IEEE source/lease contract.
