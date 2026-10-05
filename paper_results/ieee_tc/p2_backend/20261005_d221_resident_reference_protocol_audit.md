# D221 — Resident-vLLM reference protocol audit

Status: read-only launcher audit; no model, remote service, or GPU run. The
existing D213 repeats remain diagnostic-only. This record defines the smallest
protocol changes needed before a new Resident reference is eligible.

## Existing runner findings

The ordinary vLLM runner in the baseline repository already provides useful
semantic ingredients: the frozen W0 trace/subset, fixed-length native vLLM
token IDs, static or dynamic LoRA modes, startup logs, post-run process cleanup,
and per-request serving summaries. It is not yet a V1 Resident reference for
five independent reasons:

1. **Notice origin.** Replicas are started sequentially and replay begins
   after all are ready; the runner does not publish one common deployment
   notice at (t=-60) s and define business arrival (t=0) from that clock.
2. **Resource domains.** The vLLM server processes, replay client and monitor
   are not proven to be in the separately bounded service/auxiliary domains
   required by the protocol.
3. **Physical lifecycle.** The runner records a fleet/unit envelope and final
   GPU emptiness, but does not emit a per-GPU acquire/hold/release integral
   with owner identity and overlap-safe intervals.
4. **Delivery contract.** Existing D213 runs use a static local adapter pool;
   they do not bind the same prepublished remote first-touch delivery contract
   to each replay request.
5. **Timeout/protection.** The ordinary runner's historical timeout envelope
   differs from the common 1,800-second qualification protection.

These are contract mismatches, not evidence that vLLM is slow or incorrect.
The old successful D213 values must remain in the diagnostic archive and must
not be promoted by offline renaming.

## Eligibility conditions for the replacement

The new launcher must reuse, without regeneration:

- the existing 7B W0 trace and 500-adapter subset;
- the fixed-length greedy v1 prompt/target/token-ID contract;
- the existing vLLM environment and its verified native token path;
- the one-time prepublished read-only remote artifact cache.

Before launch it must create one owned service resource domain for all vLLM
workers and one independently bounded auxiliary domain for replay/watchdog;
record effective cgroup limits, CPU affinity and parent limits; and refuse to
run if a worker escapes its owned identity. The service notice and replay
publisher must share a clock ID and carry `deployment_notice_s`, `replay_t0_s`
with exactly the frozen lead time, full request-map identity and trace/subset
SHA values.

For each physical GPU, the lifecycle ledger must contain an acquisition event
before model startup, owner and CUDA UUID, contiguous hold intervals covering
startup/serving/drain, a release event after worker exit, and a final independent
NVML/cgroup census. Overlapping intervals on one GPU are unioned, not summed.
The reference run is ineligible if a release timestamp is backdated, inferred
from utilization, or missing after forced cleanup.

The common run must also retain request-level remote transfer identity,
artifact/content SHA, bytes, delivery mode, no-pack state, and local preparation
spans. A one-time preload/initialization transfer is recorded separately and is
not silently merged into request-scoped first-touch transfers.

## Qualification order

1. CPU-only contract and trace/subset/hash audit.
2. 100-request smoke with all 500-adapter identities reachable.
3. One 1,000-request compatibility run under the common notice and lifecycle
   ledger.
4. A complete 4,000-request Resident run; only after it is complete and
   correct repeat it two more times for the per-model/per-workload Resident
   reference required by M2.

Every attempt, including launcher/resource failures, receives a new immutable
campaign directory and remains in the audit table. No result is accepted by
changing the timeout, dropping the remote path, or folding the replay client
into the service budget after the fact.

## Boundary

This audit does not claim that ordinary vLLM lacks a valid serverful baseline;
it only states that the current D213 runner cannot supply the frozen physical
Resident reference. The next code change belongs in the baseline harness and
must be committed there separately. PrimeLoRA optimization and the D219
diagnostic are not rewritten while this reference is being qualified.
