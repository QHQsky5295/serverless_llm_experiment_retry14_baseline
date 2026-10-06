# D285 — Resident-vLLM 7B resource-snapshot pilot

**Date:** 2026-10-06  
**Identity:** seed 42, the existing W0 trace and 500-adapter subset, 100
requests from the frozen 4,000-request trace. This is a protocol pilot, not a
formal Resident reference repeat.

## Integrity result

- 100/100 requests completed successfully.
- 100/100 fixed-length requests matched their target native token count and
  prompt/target request map; all completion counts came from vLLM `token_ids`.
- The replay used the receipt's absolute monotonic business origin. The first
  request had approximately 18,984 ms of dispatch/admission wait, consistent
  with the measured 73.44 s maximum startup and the 60 s deployment lead plus
  client prewarm. Startup was not subtracted from user TTFT.
- The four replicas were launched concurrently and became ready in
  72.66–73.44 s; the fleet startup value is the maximum, 73.44 s.
- The remote service reused the immutable prepublished gzip delivery cache and
  was stopped after the run. No request-time archive creation was performed.

## Runtime resource evidence

The receipt contains two active snapshots, `replicas_launched` and
`replay_started`. At both snapshots each vLLM unit reported:

```text
MemoryHigh=77309411328       (72 GiB)
MemoryMax=85899345920        (80 GiB)
MemorySwapMax=2147483648     (2 GiB)
TasksMax=4096
```

The replay unit reported `MemoryHigh=MemoryMax=4294967296` (4 GiB),
`MemorySwapMax=0`, and `TasksMax=128`, in a separate auxiliary slice. Cgroup
paths, PIDs, active states, and the common monotonic clock ID are retained in
the lifecycle receipt. Final NVML inventory was empty apart from the normal
driver baseline allocation (15 MiB per GPU); no experiment unit remained.

## Descriptive pilot values

The 100-request diagnostic has mean/P95/P99 user TTFT 752.28/828.54/8019.83
ms, mean TPOT 25.15 ms, mean E2E 5102.4 ms, and 97% attainment of the
development 5-second TTFT display threshold. Its lifecycle sidecar observes
2254.870 GPU-s, of which 293.766 GPU-s are startup, 1961.104 GPU-s are
ready-state, 431.890 GPU-s are active service, and 1529.213 GPU-s are idle-ready.
These numbers are descriptive pilot evidence; the frozen warm SLO and the
three-repeat Resident budget reference are still open.

## Boundary and next action

D285 closes the Resident pilot protocol gate: common clock, concurrent
startup, native output contract, active cgroup limits, remote delivery, and
cleanup are all evidenced. It does not freeze `U_ref` or a G2 budget because a
100-request pilot is not a three-repeat full workload. The next step is the
pre-registered 7B Resident reference series under the same frozen contract,
after which the common warm threshold and budget can be evaluated. No 3B or
external baseline run is authorized by this pilot alone.
