# D352 — one-transaction HOST storage evidence on a confirmed GPU hit

## Question and mainline

7B optimization, before 3B or external baselines. D349 reduced physical GPU
occupancy to 11314.883 GPU-s but its provisional timing-only joint attainment
was 3668/4000; D336 was 3837/4000 at 13814.334 GPU-s. Neither native token counts
nor these timing-only values establish numerical correctness or final G1/G2.
The frozen V1 ruler and outstanding common reference qualification are unchanged.

D351's completed first1000 diagnostic found real engine-core CPU control work:
work-entry pending-reference calls averaged 131.201 ms, and demand-load calls
106.739 ms. For the 108 short-output TPOT misses, client-control spans occupied
41.88% of mean decode time. These are conditional, overlapping host observations,
not CUDA kernel time, a causal effect size, or a removable-latency guarantee.
Do not rerun D351 or subtract its measured overhead from ordinary runs.

Hypothesis: a confirmed GPU-hit demand transaction walks the same complete HOST
storage graph twice, despite executing no HOST load between those checks.
Removing only the second walk can reduce core interference without relaxing
budget enforcement. Ordinary Full4000 replay, not CPU tests, decides its value.

## History and source basis

- D329 (`D329_HOST_STORAGE_DEPENDENCY_AUDIT.md`) demonstrated that model ID and
  owner epoch do not determine tensor storage, aliasing or pin state. All six
  counterexample tests remain. This is **not** an epoch/TTL/cross-request cache.
- D330 planning publication, D180/181 scoped observations, D189 file descriptions
  and D193/194/202 physical projections already removed other redundant work;
  this change does not repeat them or remove their freshness requirements.
- [vLLM v0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  places native adapter mutation in the single-threaded core loop. The local
  UniProc bridge executes this worker transaction synchronously on that owner.
- [vLLM v0.30 LRU cache](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/cache.py)
  implements cache pinning by adding a key to a set; it is not tensor pin_memory.
- [PyTorch v2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  queries pointer pin status. Repeated physical observations have CPU cost;
  background allocator processing is a reason to retain fresh counters.
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  motivates measuring decode interference under latency constraints. Its
  prefill/decode scheduling result does not prove our control-path hypothesis;
  D351 supplies the local evidence. No scheduling policy is imported here.

## Bounded change and proof obligation

The native worker retains the existing dense vLLM0.30/Torch2.13/loader checks.
The first budget check still measures every current registered/staged HOST
tensor, validates representations and aliases, and reads allocator occupancy.
Only if the transaction already holds a confirmed GPU copy may its postcheck
reuse those just-measured storage totals. There is no retained graph cache.

Between checks the selected path runs owner/source validation, native cache
pins, the unchanged CUDA completion fence, and reference publication. It invokes
no loader, staging registration, cache insertion/eviction, or tensor mutation,
and never yields the owner thread to another core operation. Live model and
staging owners retain the reachable tensors. The completion fence only waits
for current-stream work; it does not execute another Python preparation task.

Background pinned-allocator completion may change process occupancy. Therefore
the postcheck always calls host_memory_stats again and applies exactly the same
accounting and workspace predicate. A counter increase can still reject the
transaction; failure poisons the owner as before, with no invented rollback.
Scope is process pinned allocations plus measured pageable LoRA storage, not RSS.

Cold loads, HOST activations, unconfirmed GPU copies, staging/register paths and
owners without the qualified callback keep both full walks. Every new request
starts with a fresh full inventory, including when the adapter/epoch is unchanged.
Source snapshots, route scoring, Eq.2 preparation classes, Eq.8 admission, all
nine paper formulas, copy fences and physical/reference invariants are unchanged.

Receipts record `host_allocation.after_storage_observation` as either
`same_transaction_confirmed_gpu_hit_v1` or `fresh_inventory`. Before/after counters
and workspace outcomes remain independently visible. A trigger count without
native GPU-hit identity is an error, not an optimization success.

## Validation and run contract

CPU tests cover exact accounting with staged/shared storage, fresh allocator
growth/unavailability, budget rejection, failure poisoning, transport replay,
cold/HOST/unconfirmed paths, per-operation freshness, and the actual worker
callback's loader qualification and workspace check. The broader native-reference,
physical-inventory, lifecycle and transfer suites are also run under 3/4 GiB
auxiliary limits. Failed test attempts remain in raw evidence.

The first test attempt exposed an observation refactor ordering regression for
unavailable allocator statistics; the measurement/pure-accounting split restores
the prior unknown-state behavior. It also exposed a pre-existing shutdown test
fixture missing `slot.engine`, required by the unchanged shutdown implementation;
only that fixture is completed, no cleanup policy is changed. The next attempt
found the new native-worker test importing a v0.30-only module in the v0.10 CPU
test environment; an explicit test import fixture replaces that dependency.

Minimal GPU validation: original W0 first100, native vLLM0.30, unchanged D349 C14
configuration/profiles, no heavy observer. Require all native contracts, closed
physical leases, at least one eligible transaction-local postcheck and zero
misclassified triggers. This is not full-workload performance evidence or
numerical qualification. Maximum two minimal validations with new evidence.
Then ordinary Full4000 compares to preserved D349 and D336; no threshold changes,
no configuration stacking, no 3B/baseline advance and no performance figures yet.

Only output/cache paths differ in the D352 YAML. Existing artifacts, source trace,
published remote gzip objects, SLO definitions, allocation boundaries, resources,
and remote services are reused. New results never overwrite historical runs.

Plan SHA: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
V1 SHA: `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.

## Completed minimal validation

Execution commit `3d232cf9790be0702ebbbdef4abbc2be8620cf48`, verified on V2.
618 CPU tests, 25 tests in the native environment, then 23 analysis tests passed.
The first100 run completed on 2026-10-08, with all100 native contracts and
binding checks passing, four physical leases closed and actual GPU release
confirmed before the remote services were stopped.

| Diagnostic observation | D352 first100 |
|---|---:|
| Completed/native-contract matched | 100/100 |
| Mean / P95 TTFT | 876.166 / 2592.862 ms |
| Mean / P95 TPOT | 28.035 / 34.319 ms |
| Timing-only joint pass count, provisional D170 ruler | 98/100 |
| Observed physical GPU occupancy (prefix only) | 798.516624 GPU-s |
| Same-transaction postchecks / full postchecks | 62 / 38 |
| Incorrectly classified postchecks | 0 |
| Remote transfers / wire bytes / request-time packaging | 29 / 50650865 / 0 |
| Service peak / high,max,OOM,swap | 18934951936 B / all zero |

The selected dispatch tier was GPU for61 requests, whereas acquisition was
already-confirmed GPU for62. Eligibility is deliberately the actual serialized
acquisition state, not a previously selected tier; an intervening preparation
can make a source faster. All62 satisfy the actual receipt invariant and invoke
no loader. This difference is not a mislabeled dispatch snapshot.

Comparison to D345 is descriptive only because its scaling controls differ.
Four output digests differ from D345 despite matching prompt/control identities;
the retained binding audit cannot identify a numerical cause. N_correct remains
unknown, and this prefix is not formal qualification or a performance selection.
No CI or figures are produced. The ordinary D353 Full4000 run now tests the same
candidate/config against preserved D349 and D336, without the D351 observer.

Source: 8129393 B, SHA
`e7f8ff5881988698c6b5dca0cccbd3f433d0d86fb289ed9646c82fb3409a0c84`.
Curated audit: `paper_results/ieee_tc/p2_backend/d352_hit_host_transaction/smoke1/`.
