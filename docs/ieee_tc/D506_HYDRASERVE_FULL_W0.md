# D506 — HydraServe full W0: preregistered single observation

## Actual outcome: invalid/incomplete due to adaptation lifetime cap

The run below was launched once and stopped after a verified launcher/platform
error. It is **not** a qualified performance result or a native HydraServe
system-resource failure. Keep the preregistration and all failure evidence.

The isolated adapter's `ContainerRecipe.timeout_s=1800` was passed to
`podman run --timeout=1800` for storage, CPU provider and every GPU container.
This option limits the **whole container lifetime**, not startup or a request.
The installed CLI help and the matching [Podman 3.4.4 documentation](https://docs.podman.io/en/v3.4.4/markdown/podman-run.1.html#timeout-seconds)
confirm that meaning. All six exact-owned containers exited after approximately
1,800 seconds; captured State records show `OOMKilled=false`. This was a bug in
our isolation adaptation, not evidence that HydraServe cannot serve this load.
The prior short qualifications did not run long enough to expose the cap.

Final publisher record: N_plan=4000, N_arrived=N_terminal=2229,
N_response=1202, N_failed=1027 (includes cancellations). The 1,771 unarrived
requests are not relabelled timeouts. No full SLO/GPU-s ranking or finite
per-correct cost is derived from this truncated execution. Full native numerical
correctness remains unclaimed; the strict full-run audit was not executed.

The exact launcher was interrupted using a verified pidfd so its existing
cleanup path could stop its own service, publisher and watcher. tmux exit=2;
launch classification=`protocol_or_launcher_error`; actual GPU contexts clear,
service resource group removed, launch auxiliary recursively empty and closed.
Peak service memory 50,427,858,944 bytes; high/max/OOM/OOMkill/swap events zero.
Remote 3B/7B/monitor stopped by their recorded InvocationIDs **after** local
release; ports 18080/18081 absent and collected logs SHA-verified.

Two diagnostic-helper failures before the interrupt are also retained: host
Python 3.10 could not parse Podman's nanosecond ISO timestamps; the existing
3.12 environment parsed and saved the evidence but lacks `os.pidfd_open`.
The separate host pidfd helper then delivered the checked signal. Neither
failure changed the running policy or triggered a rerun.

Next correction is narrowly scoped: remove the inappropriate per-container
lifetime timer and leave service lifetime with the native controller and the
existing bounded campaign/watchdog cleanup. Do not replace 1,800 with an
arbitrarily larger constant. Retain request protection, startup checks, native
scheduler/cache/batch settings and common evaluation thresholds. Validate the
specific lifecycle contract, preserve this attempt, then run a new full-W0
attempt. No performance parameter search or baseline optimum claim is allowed.

Stopped container metadata remains in the isolated Podman store; capture its
logs/state and remove only these exact-owned, exited objects before the next
launch. Unique weights, source traces, raw failures and the unrelated k3s
cluster remain untouched. Full remaining project scope is unchanged.

## Original preregistration (retained)

2026-10-13. This record is written before launch; no result is asserted here.
The next step follows D505's 1,000-request compatibility audit, not another
elementary qualification. Original D503 runtime integration is unchanged.

## Question and fixed comparison contract

Obtain the missing HydraServe row for the available 7B W0 baseline comparison.
Reuse all 4,000 existing W0 requests, 500 accessible logical adapters, canonical
prompt and native fixed-length greedy output contract. Do not generate weights,
new traces or a replacement adapter pool. This is one preregistered development
observation, not an independent holdout or a search for HydraServe's optimum.

- W0 view SHA: `a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d`.
- Source trace SHA: `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`.
- Preregistration SHA: `033911e0ae567ae8c3dd02dc0bf0be65739465ae0acd10113fa3899c06654b24`.
- Runtime registration remains official `ModelSet[0]` mapped to existing 7B:
  `(13, 9000, 200, 1000000, 1)`, native batch 8, USE_CACHE=0, SLOW_EXPR=0.
  The final 1 is the registered logical model count, not an autoscaler cap.
- **9000/200 ms are native control inputs, not our evaluation thresholds.**
  Evaluation uses unchanged D394 common model/length thresholds, manifest SHA
  `2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3`.
- For native prompt length <=616 / >616, common TTFT thresholds are
  2995.348713684507 / 4633.345350589479 ms, and TPOT thresholds are
  59.74409694902162 / 85.24698166015581 ms. No post-hoc tolerance.
- V1 plus single-run V2 remains authoritative: all offered work and native
  binding/accounting first; joint SLO at least 3800/4000 before G1 resource
  ranking. Infeasible and incomplete results are retained, not optimized away.
  No repeated performance runs, CI or winner filtering.

## Faithful path and scope

Original modified vLLM 0.4.2, Scheduler, resource placement, startup overlap,
pipeline/consolidation code and 30-second idle retirement remain unchanged.
Use the isolated rootless Podman platform on **one physical host / four GPUs**;
do not access the unrelated k3s cluster or invent multiple nodes. Native
selection in qualifications produced pp1 instances; record actual events in
this run, not forced pipeline or consolidation events.

Reuse D78/D80 immutable published remote objects, no request-time packing,
no prefetch of the whole pool, cold per-run local download cache. Actual
request-triggered reads, transfer contention, extraction and materialization
remain in latency. Administration and hashing occur before deployment notice;
remote changes are forbidden during inference. Fresh authenticated health
must actually finish successfully before launch; the launcher checks receipts.

Common open-loop replay starts 60 seconds after deployment notice regardless
of ready time. Keep the common 1,800-second development request guard. Keep
original scheduling, cache and batch configuration; no baseline parameter search.

## Measurement and safety

Use native token IDs and native request/adapter/index/path/content evidence;
never use HTTP success or expected token counts alone as correctness. Raw
`n_correct=null`, numerical qualification flags and `formal_run=false` remain
unchanged. A new full-run analysis may derive V1 binding/accounting metrics
only after the full evidence audit; it must disclose independent numerical,
trained-adapter-diversity and unexercised pipeline limitations.

Physical GPU time is the actual container-reservation interval union, from
before container creation through actual release, not utilization. Record
all four disjoint lifecycle windows and actual native GPU contexts. The
qualified pp1 claim-to-create path has no intervening await; if multi-rank or
additional cache reservations occur, inspect their allocation boundaries before
claiming complete physical accounting. No invented earlier bookkeeping times.

Service memory high/max 72/80 GiB, swap max 2 GiB; independent auxiliary max
4 GiB, swap 0. Keep common CPU separation, external watchdog, 102 GiB empty-start
memory gate, 150/100 GiB inference-disk rules and 20 GiB predicted new growth.
No simultaneous build, compression, large scan, second inference or remote
maintenance. Stop only exact-owned experiment processes; retain failure evidence.

Preparation CPU scope was recursively empty and closed; no GPU has yet run for
D506 at the time of this preregistration. Existing runtime source hashes match
D505; no algorithm changes. Inputs/configuration/source snapshots are under:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d506_20261013`.

## Ordered next actions and untouched scope

Fresh gates -> one complete W0 replay -> actual local/remote release -> full
request/resource audit -> save and back up observed result -> compare all
available baseline rows -> optimize PrimeLoRA with a falsifiable hypothesis.
Do not start a PrimeLoRA candidate before this comparison.

Remaining 7B W1/W2/G2, 3B optimization/baselines, dLoRA B6, A1–A5, S1–S13,
cost/CE/Pareto analysis, design/reviewer documents and final academic figures
remain required. Figures are deferred until PrimeLoRA is near target. No
13B, SGLang, paper edits, historical overwrites or new datasets.
