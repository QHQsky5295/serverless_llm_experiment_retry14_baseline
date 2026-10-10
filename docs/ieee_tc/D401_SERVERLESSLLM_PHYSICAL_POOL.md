# D401 — ServerlessLLM physical GPU pool and ready-time provenance

2026-10-11. CPU validation only. This extends the D399/D400 native qualification
adapter; it is not a new model result, a baseline ranking, or numerical LoRA
qualification. PrimeLoRA policies and the nine IEEE equations are unchanged.

## Evidence and measurement decision

The D398 re-audit of D72/D73 and earlier native-store qualifications shows why
logical replica counts are insufficient: the native checkpoint store itself
creates multi-GPU contexts. Retiring a model actor does not release the store's
physical ownership. Conversely, looking up only surviving Ray actors loses
source and containment evidence for legitimately retired serving instances.

Use the existing main-repository PhysicalGPUAllocation, NativeGPUCensus and
PhysicalGPUDeployment, rather than a second resource estimator. The explicit
`physical_lifecycle=native_store_pool_v1` HTTP configuration requires
`ingress_mode=service_pre_ready_v1`; legacy configurations are unchanged.

The whole selected native store/engine pool is exclusively allocated immediately
before native stack creation, after preceding CPU-only configuration work. This
is allocator-held capacity, not measured CUDA busy time. Startup is included;
logical actor scale-down does not shorten the pool lease. For a given GPU its
store and engine processes are counted once by the shared physical union reducer.
Do not substitute discounted billing, logical instance counts, utilization, or
one-second sampled occupancy for these allocation/return boundaries.

This is a disclosed static native-store pool deployment, not a claim that every
possible ServerlessLLM deployment must permanently reserve the same devices.
The native store topology and this allocator contract must both be verified in
the affected GPU qualification before using the resulting resource number.

## Lifecycle and provenance

1. Create a fresh deployment ledger bound to the exact external frozen replay,
   deployment notice, fixed business t0 and host monotonic clock identity.
2. Allocate all selected physical UUIDs before spawning the native launch child;
   bind its actual PID, birth identity and service cgroup.
3. After store startup and again before teardown, bind actual service-owned
   NVML compute contexts, including the store. Reject an unallocated device.
   The existing allocator verifies coverage and retains kernel pidfds.
4. Before a backend publishes RUNNING, save a durable exclusive ready receipt
   from the actual actor: PID/start ticks, clock, cgroup/affinity, native backend
   and store source hashes, checkpoint, native loading mode and LoRA enablement.
   Each served request carries the receipt ID/hash. This is positive ready-time
   provenance, not an exit receipt or proof of a numerical adapter effect.
5. Audit receipts even when serving fails. Retired actors need not still exist.
   A receipt cannot bind two instance IDs; changed, missing, symlinked, foreign
   clock or post-entry receipts fail. Legacy no-receipt mode keeps its original
   survivor lookup and explicitly fails on missing retired instances.
6. Stop only the run-owned native stack. Wait for the bound kernel process-exit
   notifications, then require a fresh native census before returning the pool.
   Failure to prove release leaves an open lease, not a guessed release time.
   A startup with no confirmed workers does not fabricate an empty exit event;
   the existing allocator retains the unqualified spawned lease.
7. Feed actual external success/failure/cancellation terminals into the existing
   deployment reducer. Successful terminals must have the validated native HTTP
   response and exact source-request binding. Failures stay failures; cancellation
   and a partial journal stay incomplete. No missing requests become timeouts.
8. Write the immutable raw qualification result, then the canonical physical
   sidecar bound to its SHA. Full GPU-s requires all planned terminal outcomes
   and closed leases. Native token-contract completion is kept separate from
   numerical correctness (`n_correct=null` at this layer).

The store lease covers the entire owned pool continuously, so accounting does
not depend on catching every short-lived actor with periodic sampling. Ready
receipts cover **served** actor source/containment; they are not presented as
a complete actor event history. The external contained launcher/watchdog remains
the final owner of forced cleanup if ordinary qualification teardown fails.

Primary-source checks: NVIDIA documents its query as returning processes with
compute contexts; that observation is not an allocator's release event.
[NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
Ray documents actor termination separately from serving results; the latest
documentation was checked for that distinction, not used as proof of installed
version-specific behavior.
[Ray actor termination](https://docs.ray.io/en/latest/ray-core/actors/terminating-actors.html).
Actual versioned local source and D398 native history govern this adaptation.

## Validation and status

- Baseline CPU suite: 89 tests passed in 7.093 s; 12 new tests plus expanded
  native-response checks. The earlier 88-test pass is development validation,
  not an independent performance repeat.
- Existing shared physical accounting suite: 32 tests passed in 1.305 s.
- Both suites: MemoryHigh=3 GiB, MemoryMax=4 GiB, swap=0, CPUQuota=400%,
  CPUs 2/3/26/27, existing ServerlessLLM Python environment.
- Tests use real temporary receipt files/process identities and synthetic
  native modules/device events. Existing transport tests use tiny loopback
  HTTP fixtures. No test creates an experimental adapter pool or workload.
- Synthetic two-GPU/ten-second accounting returns 20 GPU-s, without duplicate
  store/engine billing. Complete failure and cancellation retain distinct
  terminal/incomplete states. The native source hook order and immutable
  raw-result binding are checked.

No GPU model was launched in D401. `remote_qualified=false` and no new baseline
performance qualification remain deliberate. No production remote service was
mutated, no approved D78/D80 cache was republished, no figure was made, and no
user-dirty replay/relay file was changed or staged.

Next: recheck disk-growth and host-resource admission, source/environment and
remote published-service identities, then qualify the affected repaired 7B
model path once. Reuse the existing checkpoint, full static pool index, original
trace prefix and contained launcher. Do not repeat D72/D73 polling diagnostics.
Only after qualification select/freeze the public configuration and execute the
matched full 7B replay. Valid Resident reference, remaining 7B baselines, W1/W2,
3B, ablations/mechanisms/sensitivities are still open.
