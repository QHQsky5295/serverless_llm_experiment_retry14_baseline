# Physical GPU time: owner events plus independent native observations

This is a P1/P2 measurement qualification, not a new performance result. It
implements plan §4.5 and the overlap test in §13.2. No model was launched.

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
before it can provide M1/M2 evidence. It is not yet wired to a qualified model
allocator; the old lifecycle events are deliberately not silently relabelled.

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

## Remaining mainline work

Connect and qualify real allocation-owner events, native scheduler/KV/iteration
state, atomic admission and actual worker lifetime. Repeat native clock/stream,
LoRA execution and release checks on the candidate backend after its existing
installation completes. No concurrent model run while installation is active.
The common HTTP baseline transport, remote disk gate, Serverless model repair
pair, formal main comparisons, ablations and sensitivity remain open.
