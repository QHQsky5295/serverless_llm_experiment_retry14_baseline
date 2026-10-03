# D184 — ordered pending registration / GPU protection: CPU qualification

Prime7B development; parent 99ea14a6e035ca1116cde3e04eedfac35d54e744.
Last complete runtime a4f4c0630534ed2c4b6f20f4f84e1f262884904f (D182).
No new GPU replay or model qualification claimed. CPU validation completed.

## Evidence and falsifiable hypothesis

D183 reuses all 4,000 D182 request records. The 1,300 selected-GPU requests
spent mean 1,254.938 ms from the global gate to source admission: recorded
routing 587.842 ms and residual 667.096 ms. The residual mixes registration,
source acquisition, retries, resumption and capacity waiting; it is NOT the
measured duration of one RPC. GPU acquisition precedes admission. The later
1.267 ms admission-to-generation span does not establish free acquisition.

Existing control order crosses controller → dedicated frontend twice:
native pending acknowledgement, then selected-GPU reference acknowledgement.
Hypothesis: executing this ordered sequence within the dedicated frontend
removes one controller roundtrip/resumption and reduces pre-engine waiting,
without changing either native owner operation or the router's information.
Only a subsequent ordinary complete replay can establish a net improvement.

Primary sources inspected 2026-10-03:

- [vLLM performance update](https://vllm.ai/blog/2024-09-05-perf-update): CPU
  processing and process separation can matter; its speedup is not our estimate.
- [vLLM v0.30.0 core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py):
  asynchronous frontend/core utility transport. We retain that native transport.
- [Python asyncio guidance](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code):
  blocking work delays event-loop tasks; no measured CPU-causality claim follows.

## Dependency boundary

| Path | Decision | Preserved requirement |
|---|---|---|
| Native GPU + proactive profile | One frontend-owned ordered sequence | Pending ack before exact GPU compare/acquire |
| Native HOST | Unchanged | Controller pending-load count updated between registration and protection |
| File HOST/NVMe/remote | Unchanged | Live identity recheck and local source ownership |
| Source-only/no admission profile | Unchanged | No invented pending policy |

The sequence is NOT atomic across scheduler and GPU owners. Other tasks may
interleave at awaits; native owner/epoch/copy checks still decide acquisition.
Known no-acquisition conflict closes the pending intent and retries the entire
router. It cannot silently load a changed GPU miss. Lost/malformed replies
retain possible pending and GPU ownership and withdraw/quarantine the runtime;
closing pending alone cannot settle an unknown GPU pin.

The frontend rejects non-GPU/malformed identities before registration, validates
the pending ack before mutation, and returns both native receipts. The controller
still validates the original GPU receipt. It records both intents before the
cancellable RPC. HOST-budget attachment/check remains in acquisition before this
sequence; normal Full runtimes already attached it at initialization. This does
not claim identical wall-clock interleavings to the old two-RPC path.

Nine equations, capacity/profile, SLO/timeout, generation and remote protocol are
unchanged. No native scheduler/owner algorithm, completed-state cache, fallback,
new weights/trace, sleep or retry limit is introduced.

## Validation results

| Phase | Executed tests | Errors | Status |
|---|---:|---:|---|
| tests1 | 103 | 0 | pass |
| regression1 | 1005 | 0 | pass |

Targeted execution took 2.834 s in unittest (13.69 s process wall including
imports); regression took 101.959 s (114.44 s process wall). Targeted discovery
ran 36 existing imported tests twice; this was corrected to a module import
before regression. Both exact source archives are retained. No serving change
was made between these phases. Counts overlap and are not statistical repeats.
There are 11 new test methods, including multi-case identity/ack negatives.

| Check | Observed result |
|---|---|
| Pending-before-GPU ordering | Waiting pending ack prevents GPU operation |
| Invalid source or pending receipt | Rejected before the relevant mutation |
| Actual router / native source-owner request | Same GPU tier, no native load; pending closed and reference released |
| HOST path | Separate register and pending-load update retained |
| GPU source changes during registration | First pending closed; fresh router selects HOST; no hidden GPU miss load |
| Lost/corrupt combined reply | Controller remains owned, replica draining; no generation or false release |
| Actual worker/proxy, cancelled during either owner stage | Exact pending/owner/lease retained; pending close cannot clear GPU uncertainty |
| Broad lifecycle, routing, pressure, planner and smoke coverage | 1005 pass |

Two controller/frontend exchanges become one on the guarded GPU path; native
registration and reference acquisition remain two ordered owner operations.
This is a structural call-count change, NOT a measured latency speedup.

Both runs used MemoryHigh/Max 3/4 GiB, swap.max=0, CPUs 2,3,26,27, no CUDA
inference. Peak RSS was 1,169,372/1,205,948 KiB. Final memory events and swap
were zero. Actual scope identities were 059ed3d45c3b4480a404d2e77a804258 and
cd187ddca6c14e0fb4ab79a476039d0e; both inactive with empty identity/control path
after completion. Protected-artifact checks cover 147 entries with zero change.

Interpretation: the bounded dependency and failure-path tests support the
implementation contract. They do not establish numerical model correctness,
SLO feasibility, GPU-s saving or G1/G2 superiority. No test-time performance
comparison/CI is meaningful for these synthetic fixtures.

Next: checksum/scope/secrets verification and tested backup, then one ordinary
7B Full W0 replay at the same D182 cap4 / D157 profiles, 4000 source42 requests,
60-second notice, fixed-output contract and 1800-second qualification timeout.
Compare complete pre-engine/TTFT/TPOT/resource outcomes, not only GPU-hit rows;
preserve regressions and invalid results. No second candidate/configuration
change. 3B and external baselines remain paused. Before a new heavy run, recover
space only from audited disposable caches and repeat resource preflight; the
new-heavy 150 GiB and running 100 GiB floors remain unchanged.
