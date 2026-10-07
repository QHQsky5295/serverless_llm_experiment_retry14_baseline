# D312: atomic file observation and pin for native HOST preparation

## Main-line question

D311 terminated before its original 4,000-request W0 replay completed. It has
1,738 arrived, 1,732 submitted and 1,640 terminal/native-contract-complete requests;
16 interrupted requests are separately recorded. N_correct remains unknown.
Observed GPU time is 6,656.206557878992 GPU-s, not a complete workload cost.
No performance ranking, SLO acceptance, confidence interval or figure follows.

The first serving traceback is an entry owner/epoch conflict in
`_queue_ieee_native_host_preparation.execute` while acquiring its file source.
The launcher's generic protocol/launcher classification is retained, but its
connection loss is not evidence that the underlying failure was merely external.
All raw failure data remain in `d311_20261007`. Local GPU contexts were released;
the three captured remote services were stopped monitor-first and subsequently
rechecked inactive with MainPID 0. No new remote cache or artifact pool is needed.

## History and falsifiable hypothesis

- D307 (`a578fab`) fixed the same observation-to-pin transaction in request-driven
  `_ensure_local_async`; it did not modify native HOST preparation's caller.
- D310 (`4ea7daa`) removed a redundant post-validation self-epoch rejection inside
  an already locked acquisition. Strict owner/epoch checks on entry remain.
- D311 executed `d04aa2290ca99e28be0878a4a2f872afa44643db`. Its native HOST caller
  still observed the file owner and acquired the lease as separate transactions.
- All four production `acquire_confirmed` sites were inspected. Request-driven
  acquisition and staged victim-file holds already use the owner transaction.
  Route admission deliberately validates its earlier selected snapshot; refreshing
  that snapshot silently would change routing semantics and is not this repair.

Hypothesis: a background file publisher can advance the shared owner epoch
between native HOST preparation's current observation and lease acquisition.
The absence of an asyncio yield does not exclude an actual OS thread.
The precise competing thread in D311 was not recorded. The deterministic CPU
interleaving below demonstrates the code defect, not a replay of that missing event.

## First-principles correction and primary sources

Hold the existing file-owner RLock across current snapshot, exact path/content
selection and `acquire_confirmed`. Release it before native RPC, I/O movement or
any await. The read lease then protects the file for the existing asynchronous
materialization and cancellation protocol. No catch/retry, fallback, relaxed
identity/epoch validation, added capacity, parameter change or new control law.
All nine IEEE equations and the candidate's measured profiles remain unchanged.

The frozen [vLLM v0.30.0 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
explicitly relies on its single-threaded core loop for the relevant manager
operation. Prime's file publications also run on background threads, so that
assumption does not transfer. This is a concurrency-model comparison, not a
claim that vLLM implements Prime's confirmed file protocol or caused the bug.
[Python's RLock contract](https://docs.python.org/3.12/library/threading.html#rlock-objects)
supports nested owner methods while excluding other threads until the outer
transaction exits. Both sources were checked on 2026-10-07.

## Validation and limits

Two tests use the real owner lock and a competing thread at the acquisition
boundary: first native HOST load, and reuse of an already materialized HOST
tensor. Without the production change both reproduce the exact entry-epoch
exception (2 errors, 0.054 s). With the change, the publisher cannot enter the
transaction. Each operation still makes one acquisition attempt, releases file
and HOST leases, leaves GPU slots unchanged and releases transfer accounting.
Every native RPC additionally asserts that the file lock is not held.

The targeted native HOST/request lifecycle suite passed 199 tests in 3.677 s.
Before/after CPU peaks were 502,837,248 / 510,472,192 bytes, with zero high/max/OOM/
OOM-kill/swap events. Exact owned empty scopes were checked then stopped.
The broader suite passed 972 tests in 91.805 s, including basic smoke, GPU
references, transfer pressure, file observations, launch and failure accounting.
Its complete log is retained under `d312_20261007/cpu_regression.log`, SHA
41963d36a395a765602eab187d910ca6df3572cfc87b1c08060181b2f03d4c83.
Peak memory was 2,627,743,744 bytes, with zero memory events/swap; the exact empty
scope was stopped. These are correctness checks, not GPU performance experiments
or numerical inference qualification. CPU receipts are in
`paper_results/ieee_tc/p2_backend/d312_host_transaction/cpu_checkpoint.json`.

## Next action and execution boundary

Retain the D311 failure-only audit, run the broader CPU suite, and push the tested
checkpoint. Then use a fresh identity for minimal native validation followed by
the original full W0, with D309/D310/D311 settings and D307 profiles unchanged.
No repeated calibration, budget sweep, 3B/baseline work or figures during this
repair. Keep the sequence 7B -> 3B -> external systems -> comparisons -> ablations.
Warm/Resident/numerical qualification remains open; do not claim G1/G2 completion.

The one-time read-only delivery-cache permission was already implemented by D78:
1,581,215,261 archive bytes across the existing two 500-adapter pools. Reuse these
published objects, not a second extracted pool; no per-request packing and no
post-hoc subtraction from actual transfer latency.

## Native100 replay preparation (2026-10-07)

Continue the already registered repair validation, not a new tuning point.
`20261007_d312_7b_prefill_validation_100.yaml` is identical to D310 except
exclusive result/NVMe/HOST paths. Config SHA
e40990b308ae9912b09db55d3ad26c4d71be1a037033966e25b9437e8399f86f.
The existing prelaunch verifier compares the complete configuration and native
child with the measured D307 context:40service/24preparation classes bind.
Prelaunch receipt SHA366a5996b1c466b9db51582c745e62103f97e2f471ebacb023a2d4017a177658.
No profiles, weights or input traces were regenerated. The prelaunch CPU scope
was checked empty and stopped, peak496,459,776B, memory events/swap0.

Use existing runner `run_d293_7b_prefetch_validation_100.sh d312` and data-only
`analyze_d298_smoke --campaign d312`; no new experimental framework. Compare
descriptively to D310, preserve improvements and regressions, and do not infer
whole-W0 cost or formal SLO from the100-request prefix. No serving change since
the tested repair checkpoint1d92ec4. Native full4000 follows only after cleanup
and audit. The complete source plan and frozenV1 were reread for this step.

Before launch, read-only remote checks found eno1 negotiated1000Mbit/s/full,
not100Mbit/s. Artifact server SHAa31fa921... and monitor SHAda2c7d04... are
unchanged. Two initial read-only lookup attempts used nonexistent interface/
script paths; neither changed services or data. Actual route/unit definitions
supplied the paths used for the successful checks. No hardware extrapolation
or remote service adjustment is inferred from link speed.

## Native100 completed and audited (2026-10-07 12:34+08)

Execution9b1c1171d559a0c4b65b421215def4f4025b60ea;
resultSHA15fd5e6cb3147da440b0254244ca62ae882e8db475c13151e47ef7f6cc3ba7c2.
100/100successful returns, fixed-length native timing and ten saved binding
checks pass.29transfers/50,650,865wirebytes pair with remote, no request packing.
265resource samples: peak19,406,376,960B, minimum host available93,270,421,504B,
zero high/max/OOM/OOM-kill/swap. Native GPUcontexts released and service path
removed. Exactremoteidentities checked thenmonitor-first stopped12:33:49;
allthreeunitsinactive/success. Auxiliary and audit scopes empty-checked/stopped.
Physical measurement is complete with no open leases; N_correct stays unknown.

| Same100-request prefix | D310 | D312 | Descriptive reduction |
|---|---:|---:|---:|
| Mean TTFT / ms | 1175.321556 | 1018.513423 | 13.34% |
| P95 TTFT / ms | 2994.614306 | 2401.525333 | 19.81% |
| Mean TPOT / ms | 27.899638 | 27.591759 | 1.10% |
| P95 TPOT / ms | 34.857805 | 33.803534 | 3.02% |
| Physical GPU-s | 805.409961 | 799.027391 | 0.79% |

All shown candidate observations improved, but this is one run of an identical
prefix, not a causal speedup or statistical superiority result. D170 provisional
timing passes98/100, versus96/100previously; it is not the final common SLO or
numerical correctness qualification. Do not extrapolate GPU-s or replace the
retained failedD309/D311 evidence. Reuse D307 profiles; next is original full4000
W0 with unchanged settings, in freshD313paths after tested backup.

Curated data: `paper_results/ieee_tc/p2_backend/d312_host_validation/smoke1/`;
auditSHA098c400b984db85e246df112c61139faae2122a97673a85c628f5cf86f58c3f6.
Raw and curated completion receipts retain exact source and remote checksums.
No figure, source recalibration, new workload or baseline run was performed.
