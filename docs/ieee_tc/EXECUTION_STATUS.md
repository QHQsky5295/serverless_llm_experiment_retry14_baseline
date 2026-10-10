# IEEE TC execution status — current, D404 terminal

Updated 2026-10-11. This is a current checkpoint, not a replacement plan.
The former 10,982-line status is preserved byte-for-byte in
`EXECUTION_HISTORY_THROUGH_D402.md`, SHA
`cbfd51321c2bb2fdc0e2c767054cc5e68a16176360235f6a855ec214cd5b9037`.
It includes every earlier D-series entry and D402 start/closure. Search it and
the related reports before changing code. Old NEXT sections are historical.
No approved experiment or failure evidence was removed. This split prevents
whole-status reads from truncating current instructions after compaction.

## Authority and unchanged evaluation

- Read `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` fully at task start,
  after compaction and before experiments. It and the full-name snapshot
  `PLAN_APPROVED_20261010_FULL_NAMES.md` have SHA
  `f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5`.
- Before comparisons/configuration selection read `METRIC_PROTOCOL_FROZEN_V1.md`
  (SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`)
  and `METRIC_PROTOCOL_SINGLE_RUN_V2.md`
  (SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`).
- G1: all4,000 correct + common joint TTFT/TPOT SLO>=95%, then minimum real
  lifecycle GPU-s/request. G2: same correctness/SLO and U<=0.75U_ref, then W1/W2
  P95TTFT and observed joint-SLO difference. CE supplementary. Relative TPOT
  regression alone is not failure; never change the evaluation ruler to win.
- One performance run per frozen key, no repeated means/CIs or best-repeat
  selection. Preserve development/reuse identity, all failures and old results.
- Order: sealed7B PrimeLoRA ->7B baselines, ServerlessLLM first -> comparison and
  necessary optimization ->3B -> ablations/mechanisms/sensitivities.
- Full names PrimeLoRA/ServerlessLLM; historical IDs unchanged. No figures until
  near target, no13B/new SGLang/paper edits/new workload or weight pools.

## Reusable PrimeLoRA evidence

D397 seals D371 as reused development W0, not unseen formal test:4,000 native/
source/adapter binding contracts,3,908 joint passes(97.70%),14,482.523924GPU-s,
3.620631GPU-s/request; mean/Type-1 P95TTFT1150.798/2837.896ms.
See `D397_7B_SINGLE_RUN_SEAL.md` and its curated seal. Actual132 published HTTP
deliveries,zero packing,one phase-attribution join caveat. Mostly-zero/repeated
LoRA content limitation remains. D379(94.425%) and D386(94.85%) are retained
below-threshold alternatives. D394 freezes7B warm thresholds; valid Resident
U_ref is OPEN. D395/D396 have measurement/ownership gaps and cannot set it.

## D402 — terminal, affected path passed; scheduler defect prevents final freeze

Raw exclusive root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d402_20261011`.
Execution commits: main348c5f3e750fd78a2f877c25bd4c06d85b4806bc;
baseline d0fea52483f2906a5e68a1ec96aed42135498c67. Existing trace,pool,
checkpoint,native vLLM0.10.2,D401 source view. No live GPU job; do not restart.

- 100/100 native contracts,0failed,17,369 output tokens. E2E/TPOT/native maximum
  reconstruction error0ms. Independent numerical LoRA qualification not added.
- All29 real published-gzip transfers join request/receipt/content/server:
  50,650,865 wire bytes,724,665,100 verified materialized bytes,71cache-hit
  requests,0request-time packs. This is not500-ID coverage or full W0.
- Physical four-card native-store pool closes at1544.433269GPU-s,no open leases.
  Raw n_correct=null and full/numerical qualification flags remain unchanged.
- Mean TTFT138.039s/router queue135.379s. Four logical instances include one
  stuck starting allocation;3engines served. No comparison of this prefix with
  PrimeLoRA's4,000-request observation or selection as a baseline optimum.
- Executed scheduler SHA
  b7779e41f451a3328eb73337533d6d9b117aafe6ee6b78f9b28e9b5f5e2b0567
  enumerates pending items then pops stale list indices. Its unaltered control-
  loop AST with3pending requests reproduces IndexError,2resolved futures,wrong
  leftover item and no metadata update. Live logs match that boundary but lack
  an exception traceback; retain this distinction, not an all-delay attribution.

Report: `D402_SERVERLESSLLM_7B_PATH_RESULT.md`.
Audit: `paper_results/ieee_tc/baseline_audit/d402_7b_path/audit.json`, SHA
`b1102a8b839ac198d512381ac6b0709fb7ab1bedebe6d38143aa4e1b05275ba8`.
Companion evidence_manifest binds77 retained files. Recipe reuses existing
analyzer/native validators. CPU audit passed; baseline native/router/launcher
regression suite70tests passed in7.699s,4-GiB/zero-swap scope. No performance repeat.

## Exact next action and remaining scope

D404 TERMINAL 2026-10-11:100/100 native contracts,0failed,17,369tokens;
four actual served engines/ready receipts; exact repaired source and contained
actor readback,live loop/no exception/empty allocation queue. No repeat needed.
Raw `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d404_20261011`.
Execution main abd53201f0ed5959d554541a4b86beae2cfef108,baseline3cc25a8.
Source scheduler SHA7f42b0cce82f970392b3d484767a13563fb2d29f6c6e870aa8f29b86fb972b91.
TTFT mean82.940567s/P9587.723942s;queue82.332853s;serviceTTFT433.976789ms.
All29real HTTP/50,650,865B/724,665,100verified payload bytes;71cache hits,0packs.
Native/timing recompute error0;physical GPU-s1362.0847175046802,closed leases.
Report D404_SERVERLESSLLM_QUEUE_QUALIFICATION.md;curated d404_7b_queue/audit.json
SHAde80d7f8adbe92e4aaec8a6a4950d905ffd3da263a5c7acdff31c4c514290f3e;
companion manifest125files.100request contracts identical to D402;startup timing
differs,so do not attribute all latency change to queue repair or compare to D371.
GPU/service/aux released;remote3ownedunits stopped;loader overlay fully restored.
Baseline evidence checkpoint6f432c8a079fde9c8db656d8063f6ceef2ae20dd pushed.
Protected before/after unchanged. No live job. Full baseline/config optimum OPEN.

1. D403 opt-in `request_identity_v1` repair implemented in existing baseline
   source adapter; vendor/historical views untouched. Original ordering,placement,
   scaling,loading,migration and cadence preserved. Official sibling FCFS source
   supports identity removal. 81 CPU tests passed in7.055s,4GiB/zero-swap scope.
   See main `D403_SERVERLESSLLM_QUEUE_REPAIR.md` and baseline detailed report.
   Baseline checkpoint3cc25a87f3895a408a16efe2ef9c75d0e8780ff3 pushed/verified.
2. D404 affected7B scale-out path passed above. Do not repeat100 requests or the
   D72/D73 polling pair. Next audit existing500-ID native-engine coverage: D80's
   500-ID HTTP/content gate alone is not engine coverage. D72/D73's1000 prefixes
   are R2/local/failed,not new-contract compatibility;reuse diagnostics/assets,
   supplement only missing affected gates. Document scope before GPU work.
3. Reuse/audit prior500-ID and1,000-request gates where contracts permit;
   reasonable public configuration validation/freeze ->one matched full W0.
   Historical target2/max-seqs4/min1/max4/keep10 is diagnostic,not selected optimum.
4. Preserve D371. No PrimeLoRA retuning without comparison evidence or3B jump.

Still open:complete7B comparisons,Resident reference,W1/W2/G2,3B optimization/
comparisons,A1-A5,S1-S13,CE/cost supplements,review/design documents,later figures.
Main roles:vLLM,S-LoRA,ServerlessLLM,HydraServe,Loquetier,ElasticLocality;
dLoRA only approved3B scope. HydraServe/Loquetier not automatically qualified.
The full plan and preserved history contain all approved matrix details;
this status is not a shortened experiment plan.

## Safety,remote,data protection

- One heavy job. Inference disk start max(150GiB,100GiB+1.5growth),stop100GiB;
  service memory high72/max80GiB,swap2GiB,CPUs4-23,28-47;aux4GiB,swap0,
  CPUs2,3,26,27. Actual worker containment/watchdog/cleanup precede GPU runs.
- D404 service peak49,713,250,304B,swap0,high/max/OOM0;host available minimum
  61,525,995,520B,disk minimum162,448,949,248B. Recheck before heavy work.
- All D402/D404 workers/service scopes released;four cards15MiB/0%. Loader overlay
  restored to exact preimages,added loader removed;empty aux scope stopped.
  Remote3ownedunits inactive/MainPID0,ports absent.
- Strict SSH alias primelora-artifact-174,lab14@10.199.227.174:8122;
  data192.168.4.174 ports18080/18081. Credentials outside Git.
  Actual current link1000Mb/s/full;historical100Mb/s was a different state.
- Reuse D78/D80 immutable gzip cache;no republication/request packing. Real reads,
  transfers,competition,local preparation remain in latency;no cumulative-span
  subtraction or live remote administration. Artifact-only reserve+growth disk
  gate does not relax inference gates.
- Small resolver journals/receipts,Ray logs and scheduler copies preserved under
  raw D402/D404. Private caches `/tmp/tc-d402-7br` and `/tmp/tc-d404-7br` remain;
  only verified re-creatable objects may be reclaimed after reference checks,
  never raw records,unique weights or frozen traces.
- Protected147-entry checks before/after identical,SHA
  96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d.
  Sole historical mismatch:user replay edit,unchanged in this campaign.
- Never stage main `configs/generated/lora_manifest_1000.json` or baseline
  `scripts/replay_openai_trace.py` (04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22)
  and `scripts/run_serverlessllm_relayserve_continuation.sh`
  (337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499).
  Preserve all unrelated tracked/untracked work.
- Tested backups:main faaslora_origin/retry14_continuous_queue_v2;baseline
  origin/main separately. No force,secrets,raw-large-data. Update this status
  before yielding;full goal remains active until genuinely complete.
