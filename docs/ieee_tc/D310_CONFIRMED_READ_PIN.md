# D310: preserve content identity across bounded file-allocation refresh

## Main-line question and disposition

D309 is a complete 7B W0 replay with 4,000 terminal records, 3,999 successful
returns and one failed request. It is not an all-correct run, regardless of
successful process cleanup. Keep N_plan=4,000; no G1/G2 acceptance, comparison
ranking, confidence interval or figure is produced from this run.

Failed request: req_03242 / medical_lora_0110, target186, actual0. The saved error
is `ConfirmedSourceConflict: confirmed file source footprint epoch changed`.
The saved dispatch tier is empty; do not replace it with an inferred hit or miss.
Source SHA: 811aadf1cc61d14f58b7ebf59a54f1406d05c8e161003f8fd6da947da662257b.

## History, hypothesis and first-principles repair

- D90 (857fc7e493fd6d2fa0660e1a81805670c8d7a9bc) introduced bounded physical
  extent accounting. Already charged allocation envelopes remain required.
- D93 separated planning snapshots from their own footprint refresh ordering.
- D307 (a578fab) made snapshot-plus-pin atomic against other owner publications.
  It did not prevent the same locked transaction from refreshing its own view of
  physical blocks.
- Before D310, acquire_confirmed checked owner/epoch on entry, validated the
  source, then checked epoch again. Successful validation can update footprint
  and increment epoch without changing verified content or exceeding allocation.
  The second check rejected exactly this otherwise valid read transaction.

Hypothesis: a legal allocation-observation change inside an already reserved
envelope is being treated as content invalidation. A deterministic fixture
reproduced the same exception with unchanged path, inode, timestamps and content.
This demonstrates the code-path defect, **not** the precise kernel event that
occurred in D309; that event was not recorded.

The repair retains the entry owner/epoch check and all source content, inode and
allocation-envelope validation. It removes only the redundant post-validation
epoch comparison. The owner lock excludes competing publication mutations;
validation's own footprint refresh still advances epoch, invalidating older
planning snapshots. The read lease cannot enlarge capacity or modify the copy.
There is no catch/retry, fallback, sleep, fsync workaround, increased budget,
new tuning parameter, or change to the nine IEEE equations.

## Primary-source checks and their limits

Linux [stat(2)](https://man7.org/linux/man-pages/man2/stat.2.html) distinguishes
allocated blocks from logical size and warns that fields can reflect different
observation moments. The ext4 [extent mapping documentation](https://docs.kernel.org/filesystems/ext4/ifork.html)
and [FIEMAP interface](https://docs.kernel.org/filesystems/fiemap.html) support
separating content identity from allocation/mapping observations. They do not
prove which filesystem event caused the original failure.

The frozen vLLM [v0.30.0 LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
validates a newly loaded adapter before the LRU eviction step. Its native adapter
pin and capacity operations provide a useful separation of validation and
residency. It does not implement Prime's file-owner or confirmed-tier protocol;
this is a design comparison, not attribution of Prime's bug to vLLM.

## Actual validation, including unsuccessful setup

| Check | Observed result | Interpretation |
|---|---|---|
| Initial two-test fixture | 2 errors: immutable manifest already frozen | Fixture setup failed; not root-cause evidence. New fixture client used afterward; production immutability unchanged. |
| Corrected fixture before serving repair | 2 tests, 1 error, 0.060s | Bounded footprint transition reproduces the exact exception; over-envelope rejection still works. |
| Targeted checks after repair | 32 passed, 2.143s | Both footprint directions readable; stale entry epoch, modified content and exceeded envelope rejected. |
| Broader CPU regression | 950 passed, 98.677s | Request lifecycle, transfer pressure, file observations, artifact store, GPU references, basic smoke and launch checks. |
| Failure-audit population tests | 19 passed, 0.014s including existing overlap tests | Complete offered population preserved; no success-validator relaxation. |

All CPU runs used the existing LLM_vllm0102 environment, CPUs2,3,26,27,
MemoryHigh3GiB/MemoryMax4GiB/swap0, offline model access. The broad regression's
peak was2,647,199,744B; high/max/OOM/OOM-kill/swap observations were zero.
Its exact owned, empty scope was stopped after checking InvocationID and reading
cgroup.procs. Retained tool output is an extract where truncated, not a full log.
The small artificial files are test fixtures, not new LoRA weights or trace data.

These checks do not constitute 7B performance qualification. The source-profile
and control settings are not retuned in this repair. First preserve the D309
failure-aware audit and tested backup, then perform a minimal native smoke and
a fresh complete replay under the same candidate settings. Do not repeat the
source calibration, scan more budgets or start3B/baselines without new evidence.

## Immutable evidence locations

- Raw run and cleanup receipt: results/ieee_tc/p2_backend_qualification/d309_20261007/.
- CPU test receipts: paper_results/ieee_tc/p2_backend/d310_source_read_pin/.
- Failure-only audit: paper_results/ieee_tc/p2_backend/d309_full_w0/failure_audit1/.
- Audit uses existing request vectors, native timing, binding, transfer and seal
  helpers. The ordinary audit_run all-success gate remains unchanged.

The user's one-time read-only delivery-cache approval is already implemented by
D78: two500-adapter pools,1,581,215,261 archive bytes, no second extracted pool.
Reuse those immutable objects. Offline publication is before deployment, not a
post-hoc subtraction from request latency; actual fetch/competition remain timed.

## D309 offline audit completed

All4,000 offered input identities match the retained D303 common input map.
All3,999 successful returns pass the ten saved binding checks and native timing
identities, with maximum TTFT/TPOT/stage error0ms. The failure stays outside those
conditional success checks and inside the offered population.132transfers and
131,480,060wire bytes match the remote journal; no request-time packaging.
4,040resource samples show peak20,676,182,016B,minimum host available93,657,604,096B,
zero swap/high/max/OOM/OOM-kill. Lifecycle observation is15,744.284432824003GPU-s;
this is not a per-correct feasible-ranking result. The protected147-entry seal
retains only the previously authorized baseline replay script change; its raw
pass flag is correctly false rather than rewritten to true.

D310's next100-request native check reuses allD308settings and D307profiles.
The actual child config matches the measured profile config;40service and
24preparation classes bind. Only new output/cache paths differ. This prelaunch
check does not execute GPU inference or qualify performance.

## Native100 check completed (not full-workload acceptance)

Source execution commit4ea7daacab6014d5aa3a1d874f74da6c3c9ea34f.
Source result SHA5410400edf56e5fd1c814f4e8cb58b9fcac81f5f71dbbce9d282086061af44ea.
100/100successful, native generation/timing/binding pass;29matched transfers,
50,650,865wirebytes and no request-time packaging. No memory protection/swap
events. GPU contexts and service domain released; remote monitor stopped first
at11:26:33+08 and allthree ownedremoteunits were inactive/successful afterward.

| Single100-request prefix | D308 | D310 |
|---|---:|---:|
| Mean TTFT / ms | 1021.630202 | 1175.321556 |
| P95 TTFT / ms | 3061.757701 | 2994.614306 |
| Mean TPOT / ms | 28.791707 | 27.899638 |
| P95 TPOT / ms | 40.205643 | 34.857805 |
| Physical GPU-s | 684.796752 | 805.409961 |

Tail metrics improved while mean TTFT and physical resource use worsened.
This mixedsingle-run observation does not support overall superiority or a
causal performance benefit from the correctness repair. ProvisionalD170timing
passes96/100; numerical correctness and commonfinalSLO qualification remainopen.
Do not extrapolate this prefix's GPU-s to4000requests or replace the failedD309.
Next is the preregistered complete4000same-candidate replay in freshD311paths,
without retuning, extra source calibration, figures or externalbaseline work.
