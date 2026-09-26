# Physical GPU time: owner events plus independent native observations

This is a P1/P2 measurement qualification, not a new performance result. It
implements plan §4.5 and the overlap test in §13.2. The original no-model witness
and the later dedicated-runtime qualification are distinguished below.

## Why historical billing is not the new G1 metric

The existing runner's `_begin_instance_lifecycle_tracking` creates initial
records after preload, reconstructs a negative creation offset, and assigns
ready offset zero. `_summarize_infra_cost_from_lifecycles` implements historical
instance/lifecycle billing. Neither establishes physical UUID allocation and
release. Those historical outputs remain unchanged and separately identified.

The falsifiable measurement hypothesis is that summing replica/process or
startup/preparation activities can overcount a shared device, while equating
idle/ready/shutdown with release can undercount retained CUDA resources. G1
requires the union of actual physical ownership intervals, not either proxy.

`PhysicalGPULedger` in the existing metrics collector now reduces ordered,
same-clock, evidence-linked allocation-owner events. Per physical GPU UUID it
merges overlapping leases; TP devices count separately. It partitions the union
into the four mutually exclusive plan windows. CPU preparation without a GPU
lease adds no GPU time. Missing release remains right-censored, full offered
counts remain intact, and zero correct completion has no finite per-correct
resource score. A successful inference population without positive allocation
evidence is rejected.

The reducer does **not** create ownership, verify its own evidence identifiers,
or qualify the controller. It must receive events from the real allocation owner
before it can provide M1/M2 evidence. D27 below adds a real dedicated-runtime
owner journal; complete Full/other-baseline owner coverage and formal aggregation
remain open. Old lifecycle events are deliberately not silently relabelled.

## Independent observer

The existing external watchdog now supports native NVML GPU UUID, compute and
graphics process census, PID birth identity, cgroup, affinity, raw v2 memory
fields and utilization. Unknown identities and API errors do not become empty
GPU lists. A zero-utilization compute process still blocks a clean launch.
Known owned workers that leave the service domain remain owned for auditing;
PID reuse does not transfer ownership to another process. External compute
occupancy causes a scoped safety stop, initially unattributed rather than an
unsupported baseline-OOM or external-interference claim.

NVML is used because NVIDIA specifies stable device UUIDs and native process
queries; its command-line output is not a substitute for a versioned API.
[NVML device queries](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html),
[nvidia-smi compatibility guidance](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
vLLM sleep can free much of a model's memory without stopping its server: memory
reduction is not automatically a returned device lease.
[vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/).

The watchdog runs in qualified system Python, which lacks the binding. It reuses
the existing official `pynvml.py` as one explicit SHA-locked module, without
importing torch/vLLM or editing either environment:

```text
FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
```

These variables are required for non-tiny guarded launches. No CLI fallback or
automatic binding substitution is permitted. Initial native observation precedes
the exec permission. The watchdog also samples after service-domain cleanup;
an unconfirmed native-context release prevents successful qualification. It
never resets a GPU or kills the pre-existing display process.

Query start/end delimit an observation, **not** a bound on all transitions
between samples. Short-lived contexts can be missed. Therefore polling is a
cross-check of native workers and allocation events, not an exact GPU-time
integrator. Real worker birth/exit, scale-out containment and owner-event
completeness must still be qualified. Empty cgroup + clear native census alone
do not prove that the scheduler returned a reservation.

## Executed witness and delivery table

Existing 7B seed42 trace prefix, 32 requests, 8x diagnostic rate, simulated
2-second async initialization and 1.5-second consumer stall. This is the same
small no-model protocol used for ingress qualification, not a new workload.
Formal preparation remains 60 seconds; diagnostic settings are not promoted.

| Check | Result | Interpretation |
|---|---:|---|
| Physical union algebra | 12 tests pass | 10-second allocation with 6-second overlap remains 10 GPU-s |
| Native census semantics | 9 tests pass | Unknown, foreign, reused and escaped PID/API cases covered with fakes |
| Actual transported requests | 32/32 | Three received before simulated readiness |
| Actual physical UUID coverage | 4/4 | Existing display process preserved, no compute processes |
| Periodic census samples | 11 | Separate initial and terminal observations also preserved |
| Periodic query duration, min/median/max | 2.868 / 3.906 / 5.218 ms | Idle no-model observations only, not under-load overhead proof |
| Initial query duration | 43.799 ms | Not hidden in the periodic latency summary |
| Service peak | 120,496,128 bytes | Small witness, not a model-memory result |
| Service high/max/OOM events | 0/0/0 | Witness bounds respected |
| Service domain removed / native contexts clear | Yes / yes | No model-GPU allocation/release qualification inferred |

Receipt: `paper_results/ieee_tc/safety/gpu_census_ingress_attempt1.json`, SHA256
`3fc3e1e7b50817e84fa3f1bf173b86faaa987ebd62c07feb8683d12a7458c4cf`.
Raw `.launch/` records are retained, all receipt hashes checked; curated summary
is `gpu_census_ingress_summary.json`. The later foreign-compute abort branch is
unit-tested, not exercised by this idle witness. No interference was injected.

Development findings are retained: the first direct binding probe failed before
NVML initialization because module registration was missing. The loader now uses
normal `sys.modules` registration before executing the verified source bytes.
Separately, v1 memory reported 489,226,240 bytes while v2 reported raw used
15,466,496 and reserved 473,759,744 bytes. The monitor records explicit v2 fields
without mistaking driver-reserved memory for the experiment's allocation.

## D27: bind ownership to dedicated runtime creation and teardown

Historical review: the existing dedicated proxy already owns a fresh worker
process per runtime, but neither its `shutdown` RPC nor several best-effort
cleanup calls establish that the physical card has returned. The falsifiable
hypothesis is that explicit allocation before spawn plus native teardown
confirmation can supply the missing lifecycle events without altering any of
IEEE's nine policy equations. This is measurement/ownership work, not a latency
optimization or a new autoscaler.

`PhysicalGPUAllocation` extends the existing metrics module and dedicated proxy:

- A service-local cooperative exclusive lock prevents assigning the same card
  to two dedicated runtimes. NVML indices are resolved to physical UUIDs; the
  fresh interpreter receives those UUIDs before any CUDA/backend import.
- Acquisition is an actual reservation before process creation, so it includes
  startup while the runtime holds that card. Preparation which occurs without
  a reservation adds no GPU time. This is not utilization-based accounting.
- The native worker reports its actual CUDA UUID. The parent cross-checks it
  against the allocated UUIDs and independent NVML process/cgroup identities.
- Return occurs only after the dedicated parent has terminated and a fresh
  census finds no compute, unknown, or service-owned context on those cards.
  Existing external display contexts are preserved. Shutdown acknowledgements,
  empty adapter caches and GPU idle percentages cannot close a lease.
- Append-only per-runtime journals survive owner failure. A dropped OS lock
  with an unclosed journal blocks reuse within the same launch. Failed release
  is not backdated to request completion. Forced whole-service cleanup still
  leaves the measurement censored until separately reconciled; no synthetic
  release event is manufactured.

The locks coordinate this guarded local service, not unrelated users or an
external cluster scheduler. Independent watchdog checks remain mandatory.
Shared-runtime logical replicas must share a single actual runtime lease; they
must not acquire one lease per logical slot. Direct in-process engines and
other baseline allocation owners are **not qualified by this path**.

The opt-in `ieee_physical_allocation` path is exercised through the existing
`backend-model-check --qualification-mode native_lifecycle`, using the original
four-request prefix. It does not change traces, adapters, the output contract or
Full routing. Eight additional owner tests cover exclusivity/reallocation,
living/lingering/unknown contexts, crash journals, partial TP contention,
actual-worker UUID checks, and missing birth identity. The initial targeted
owner/worker/launcher suite has 48 passing tests; model outcome is recorded
after cleanup, not inferred from those tests.

### First actual 7B attempt: failed release qualification, preserved

| Check | Observed result |
|---|---|
| Existing prefix / native token targets | 4/4 matched, 551 output tokens |
| Allocated UUID vs actual CUDA worker | Match, one physical RTX 3090 |
| Allocation / native-worker journal | Present before execution |
| Return at proxy shutdown | Rejected: compute/owned/unknown context remained |
| Subsequent external cleanup | Service removed, native contexts clear |
| Physical resource result | Censored; no complete GPU-s/request reported |
| Safety samples / service peak | 82 / 5,650,046,976 bytes |
| Service high/max/OOM | 0/0/0 |

Raw result `llama2_7b_physical_lifecycle_attempt1.json`, SHA256
`979c9074cc1965f626ed3eaa3a6837e6932ea00505c7b52232396bfc5373835f`;
launch receipt SHA256
`340d8b84b1b0e12014b229b688bab574ab337d6361dd7d7cb5ca362c30f70b56`.
Both are under `results/ieee_tc/p2_backend_qualification/model_20260926/`.

The native worker was PID419261; the proxy parent was PID418429. The old proxy
sent TERM immediately after the shutdown acknowledgement while native teardown
was still executing. The post-run observer eventually saw clear contexts, but
that does not retrospectively supply an owner return timestamp. This attempt
is a measurement qualification failure, not a formal performance point.

Correction: allow the cooperative shutdown to finish before forced cleanup,
and pin the native workers' kernel pidfds at their verified birth identities.
Wait for exit readiness before the final census/return decision, within the
existing normal60s cleanup allowance (15s reserved for the existing TERM/KILL
path). No fixed sleep, ignored context, or backdated return is introduced.
Kernel pidfds support exit notification and avoid PID-reuse ambiguity:
[Linux pidfd_open specification](https://man7.org/linux/man-pages/man2/pidfd_open.2.html).
The corrected attempt is a new retained run, not an overwrite of this failure.

Environment qualification found both existing Conda Python and the candidate
venv lack `os.pidfd_open`; host glibc2.35 also lacks that wrapper. The kernel is
Linux6.8 x86-64 and supports pidfds. A small explicit ctypes UAPI binding uses
syscall434, verified against the installed `unistd_64.h` and the official
[Linux6.8 x86-64 syscall table](https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl).
This is a fixed ABI identifier, not a latency parameter or a polling fallback.
Unsupported architectures and kernel errors fail. A real child-process exit
handle test accompanies mocked lifecycle cases. Its stdlib-only child disables
site initialization, avoiding unrelated backend imports in the exit witness.

### Second actual attempt: complete owner measurement, forced teardown exposed

Four requests again matched all551 native output tokens. The physical lease
closed after102.163931280s, including48.769133136s after the last native token.
The parent returncode was -15: this is measured bounded forced teardown, **not**
a successful graceful-shutdown claim or a resource-efficiency result.

This new evidence identified a separate ordering dependency: the worker uses
`async with server`, and Python3.12 `Server.wait_closed()` waits for accepted
connections to close. The proxy awaited parent exit before closing its idle
RPC pool. The candidate's installed asyncio source and the
[official Server contract](https://docs.python.org/3.12/library/asyncio-eventloop.html#asyncio.Server.wait_closed)
confirm the dependency. Close those owned idle connections immediately after
the stop acknowledgement, then await actual worker exits. No shorter timeout,
unconditional delay or changed model policy is used. A third, separately retained
run tests this newly identified cause; it is not a repetition without evidence.

### Third actual attempt and delivered status table

| Attempt | Target matches | Physical GPU-s | Last token to return | Parent exit | Outcome |
|---|---:|---:|---:|---:|---|
| 1 | 4/4 | N/A, censored | N/A | Not confirmed in owner journal | Release qualification failed |
| 2 | 4/4 | 102.164 | 48.769s | -15 | Measured, forced cleanup |
| 3 | 4/4 | 60.738 | 7.642s | 0 | Native exit and physical return confirmed |

Every attempt used the same four existing requests and551 native output tokens;
request IDs, adapter IDs, prompt hashes and output-token hashes match across all
three. Attempt3 has an explicit native-worker exit event before final census and
return. Its74 resource samples show peak5,684,371,456 bytes, high/max/OOM all0.
All three service domains were removed, native contexts cleared, display sessions
preserved, and their empty auxiliary scopes stopped after identity checks.

This is a single small diagnostic per code version, not a paired statistical
performance study. The measured durations include startup and final cache
cleanup; do not normalize them into main-workload GPU-s/request or extrapolate
to Full scale-out. Do not repeat this prefix now that its question is answered.
Next integrate the remaining Full ownership/admission and real profile gates.

Curated JSON/CSV: `paper_results/ieee_tc/p2_backend/20260926_7b_physical_lifecycle.*`.
They retain all three attempts, raw/launch/watchdog/journal hashes, per-attempt
limitations and actual attempt3 runtime-source hashes. Final regression:
666 functional tests and56 safety/census/replay tests pass, no failures or skips.
The eleven additional owner tests include the real kernel exit witness; two
proxy tests enforce connection close → parent exit → native exit → physical
return and prove that failed native exit cannot return the allocation.

The rationale uses the official NVML device/process API and the candidate
[vLLM 0.30 AsyncLLM implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py).
The [sleep-mode documentation](https://docs.vllm.ai/en/latest/features/sleep_mode/)
likewise distinguishes memory reduction from stopping the server; sleep alone
does not satisfy this experiment's physical-return contract.

## D55: whole-deployment collection in the actual external-replay runner

Plan review keeps G1 as physical possession, not discounted ready time. D54
solves CPU entry capacity only when incoming staging fits; inspection of the
actual dense loader and pinned allocator does not justify reclaimable-byte
credit when physical bytes are exhausted. That work stays open. D55 advances
the independent Full lifecycle gate rather than adding another staging test
campaign. No IEEE policy equation, budget, workload or artifact is changed.

Historical D27 already proved one dedicated runtime's acquire/exit/return path.
It did not connect every Full activation and request population to one reducer.
Old instance accounting does retain removal events, but logical removal is not
physical return. The new `PhysicalGPUDeployment` reuses the qualified owner's
append-only journals and `PhysicalGPULedger`; it does not reconstruct leases
from ready/idle or instance-pool membership.

- The actual external-replay entry creates a fresh deployment record before
  model initialization, bound to its exact plan hash, clock and notice/arrival
  boundaries. Every dedicated initial/scale-out/reinitialized runtime is forced
  through the existing physical allocator. Scenario overrides cannot disable
  this accounting or select an unowned in-process backend.
- All owners retain the **existing launch-wide lock directory**, not separate
  locks per replica or collector. Fresh deployment evidence cannot overwrite
  an old run or create a competing allocation namespace. Per-run journal paths
  are deployment evidence, not model-config/profile identity. The physical
  accounting enablement itself remains in model identity; measured profiles
  must use that same enabled configuration, with no profile-hash exception.
- Request terminals are observed on the actual `run_one` return/exception path.
  Global cancellation is an interruption, not an invented request timeout or a
  terminal used to declare full completion. Offered IDs/denominator never shrink.
- Final reduction occurs after the actual work/ingress cleanup, including the
  failure path. It reads every owner journal, including removed replicas and
  failed startup attempts. Clock/identity/order errors or truncated journals
  fail; missing return remains censored. No ready or successful-RPC timestamp
  closes an allocation. Failed cleanup of one runtime does not skip the others.
- The sidecar retains journal/request SHA, physical UUID unions and the four
  windows. Scenario metadata links to it without relabelling legacy billing.
- `n_native_contract_complete` checks request/adapter identity, native token
  source/count, fixed contract and digests. It is **not numerical adapter proof**:
  `n_correct=null`, per-correct resource=null and comparison eligibility=false
  until the independent correctness/qualification analysis supplies evidence.
  There is no flag that converts a token match into semantic qualification.

The approach still relies on qualified native owner events and independent
[NVML device/process observations](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html).
The candidate's [vLLM0.30 sleep documentation](https://docs.vllm.ai/en/v0.30.0/features/sleep_mode/)
supports keeping memory reduction distinct from runtime termination. Neither
source provides a Prime performance result.

### D55 correctness table (constructed events, no GPU/model run)

| Question | Evidence |
|---|---|
| Same GPU released then reacquired | The unallocated gap is excluded; old lease remains in the journal set |
| Another physical GPU overlaps | Three leases on two devices total12 GPU-s in the fixture |
| Four exclusive windows | 1+4+4+3=12 GPU-s; no startup/preparation double count |
| Failed startup with open allocation | Only observed GPU-s, no completed resource score |
| Global replay interruption | Full offered denominator retained, interruption not a fake terminal |
| All requests fail but physical release completes | Resource measurement can complete; correctness/ranking cannot |
| Malformed/foreign-clock source journal | Rejected, not silently omitted |
| First runtime cleanup fails | Other owned runtimes still receive cleanup; failure remains visible |
| Existing replay wrapper and actual request path | Final sidecar exists on normal wrapper exit; success/error/cancellation recorded once |

Initial70 checks had one fixture assertion mismatch from JSON tuple→list
serialization; normalize the assertion, not the measurement. Next70 pass.
Initial19-module919 and expanded20-module937 checks pass. Final frozen-source
20-module937 functional checks,70 checks under installedvLLM0.30.0/torch2.13.0+
cu130 and56 safety/census/replay checks pass without failure/error/skip; CUDA
remains uninitialized. All nine owned scopes are empty with no high/max/OOM
events and stopped. These counts overlap and are not independent experiment
repetitions. Full receipt and source hashes are in EXECUTION_STATUS.

This is connected measurement logic with CPU tests, **not Full GPU qualification**.
No D27/source-prefix model rerun, remote174 service, new weights/traces, formal
baseline/main/ablation/sensitivity result, or manuscript/old-result mutation.
Actual multi-activation ownership/terminal coverage, A4 dispatch linkage and
correctness-qualified resource analysis still require the integrated model run.

## Remaining mainline work

Connect and qualify real allocation-owner events, native scheduler/KV/iteration
state, atomic admission and actual worker lifetime across the Full deployment.
The candidate backend is already installed; reuse it and its compile caches.
Do not repeat completed sequential/source-prefix checks as new performance evidence.
The common HTTP baseline transport, remote disk gate, Serverless model repair
pair, formal main comparisons, ablations and sensitivity remain open.
