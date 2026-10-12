# IEEE TC execution status — D462 live pin-query candidate rejected

Updated 2026-10-12. This is a current checkpoint, NOT a replacement plan.
Complete preceding status is byte-identically archived in
EXECUTION_HISTORY_THROUGH_D461.md (cmp passed), SHA
6af1969d46e767fe9c5287398d140ff917a6f765ad66ef220e8ac197ea08de7b.
Earlier D448/D433/D402 histories remain. Historical NEXT/RUNNING entries are
not current instructions. No evidence or approved task was removed.

## Mandatory authority and evaluation

- Read full /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md at task start,
  after compaction and before experiments. Source/approved FULL_NAMES snapshot
  SHA f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
- Read BASELINE_EXECUTION_RULES_20261011.md. Only PrimeLoRA performance optimization.
  Baselines permit necessary compatibility/correctness/measurement/safety work,
  no performance configuration search. Preserve actual outcomes and failures.
- Before comparison/configuration: read METRIC_PROTOCOL_FROZEN_V1.md and
  METRIC_PROTOCOL_SINGLE_RUN_V2.md in full. SHAs respectively
  5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 and
  193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
- G1: all4000 native/source/adapter contracts and common joint TTFT/TPOT >=95%,
  then minimum physical lifecycle GPU-s/request. G2: same + U<=.75 U_ref,
  then W1/W2 P95TTFT and observed joint-SLO difference. CE supplementary.
  Single TPOT regression is not failure. Raw unknown numerical flags stay unknown;
  full input/source/adapter binding is not independent kernel numerical proof.
- One performance run per frozen execution key; no repeated means/CI/significance,
  winner filtering or threshold changes. Preserve development/reuse identity.
- Order: sealed7B PrimeLoRA -> 7B baselines -> comparison/PrimeLoRA optimization
  ->3B -> ablations/mechanisms/sensitivities. W0 alone does not close W1/W2/G2.
- Full names PrimeLoRA/ServerlessLLM. No figures before near target, no13B,
  new SGLang, paper edits, new weights/traces or pool regeneration.
- Every optimization: logs/history + online primary sources + one falsifiable
  hypothesis + tests/minimal validation/new full replay. IEEE nine formulas fixed.

## Current action — D462 native qualification completed, candidate rejected

Read full PLAN/status/rules/V1/V2. Used run-experiment and optimize-for-gpu.
Extended only existing ieee_tc_preflight.py with diagnostic backend-pin-check,
plus tests. NO production faaslora code/configuration or baseline changes.
D461 observed179 exact pin-query frames, not CPU share; D329 caches unsafe,
D352 same-transaction reuse and D355 atomic interleave already implemented.

One pre-exec launch rejected absent explicit NVML binding, CUDA never ran.
Original helper/receipt preserved. v2 uses existing runner's hash-locked NVML
and allocator environment; unique actual CUDA qualification passed.53 CPU tests,
0 errors/failures/skips (5.626s),9 actual pinned/pageable/offset/alias/register/
unregister comparisons,4 complete inventory equality modes. Diagnostic-only AST
clone substitutes one live query and guards the initialized context at entry.
1536 small tensor views, no backbone/LoRA pool/trace/remote operation.
Actual binding13.4.3, CUDA runtime13040, driver13000, Torch2.13.0+cu130;
one mapped libcudart path. Preserve actual versions, not just wheel label.

20 preregistered alternating local pairs: direct faster only2/20.
Native/direct medians9.317179/10.220996ms; median paired ratio1.082391.
Means12.895695/10.290082ms are dominated by one native77.382471ms call;
cause unknown, record retained. Apparent mean20.2% improvement is NOT sufficient
evidence of a useful optimization. REJECT Python-binding candidate; no production
integration, prefix/full replay, repeated microtest or blind extension build.
See D462_LIVE_PIN_QUERY_QUALIFICATION.md and p2_backend/d462_live_pin/.
CPU safety/error branch mocks are distinct from actual CUDA failure evidence.

Native service/watchdog exit0, actual context release and service path removal
verified.4 samples max observed365572096B, high/max/OOM/swap0; CPU test peak539017216B.
147 protected entries unchanged. Raw:
 /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d462_20261012/
CPU1 and launch2/3 exact InvocationID/inode/actual recursive cgroup.procs empty
closed. Curation4 completed, final exact closure/source/secret/diff checks and
backup still NEXT. No GPU run remains. Latest disk161125584896B (~150.06GiB),
fresh gates before another heavy job. Whole goal ACTIVE; actual qualification
and rejection are progress, not a blocker or completed system optimization.

## Next mainline

Seal/backup D462, then return to retained end-to-end control wait evidence:
distinguish necessary new observations from duplicate calls within one control
turn. Check D180/181 scoped observation, D193/194/202 projection, D330 publication,
D352 transaction reuse and D355 interleave before proposing anything.
Do not repeat D456 profiling, D458, sealed baselines, C8/capacity/threshold sweeps,
pin-query microtests, stale HOST caches or already implemented planner offload.
No new candidate until a concrete redundant boundary is established.
D458 remains the latest Full observation, W0 feasible but not lowest GPU occupancy.
HydraServe independent native qualification remains open, not a reason to alter
the user's unrelated Kubernetes cluster or abandon the PrimeLoRA optimization mainline.

## Sealed W0 comparison and remaining resource gap

Six-system D454 matched input:4000 requests,458224 native tokens,132 logical IDs.
Contract SHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.
All sealed rows retained, no repeats:

| System/run | Joint SLO | GPU-s/bound request | P95 TTFT ms |
|---|---:|---:|---:|
| PrimeLoRA D371 | .977 | 3.620630981 | 2837.895685 |
| ServerlessLLM D414 | .98125 | 4.022440686 | 1052.246361 |
| vLLM D423 | .99475 | 4.028844291 | 1200.506860 |
| S-LoRA D435 | .94425 | 3.789269468 | 17699.067366 |
| Loquetier D445 | .0435 | 4.249595809 | 555110.032229 |
| ElasticLocality D453 | .95775 | 2.473178968 | 2420.607829 |
| PrimeLoRA D458 latest | .9645 | 2.994084616 | 2847.386152 |

S-LoRA/Loquetier diagnostic_nonfeasible; no baseline performance tuning.
D458 resource17.3049% below old PrimeLoRA,21.0622% above ElasticLocality.
All4000 native/source/adapter/lease contracts + independent D435 input pass;
raw n_correct=null retained. Same-backend output-digest changes220 vs old,
204 vs ElasticLocality retained without unsupported rounding explanation.
D458 source SHA8d65bf7a9113909451d72a822098e1e464e27d031a7b0ea71357bf30fd52205c.
Matched comparison SHAd8188014ac448205e33198f81e9bb12a793dea664a1de90e5c504535ea500a42.
D458 meanTTFT1145.751632/meanE2E4987.147144/P95TPOT58.675797ms.
Natural22 triggers=19ready+3notready; initial1. No global optimum/G2 claim.
Firstfour prior systems and D458132fetch/131480060wireB; S-LoRA/Loquetier500/
420740410B. No isolated remote-transfer advantage on this W0.
500 logical adapters:498zero/2nonzero,4weightSHA/6tree classes; diversity limitation.
Loquetier native static4x125 shards,batch2/context1024; not native scale-out.

D394 common SLO: <=616 input TTFT2995.348713684507/TPOT59.74409694902162ms;
>616 TTFT4633.345350589479/TPOT85.24698166015581ms.
protocols/20261010_7b_slo_d394_frozen.json SHA
2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
W0 U_ref16115.377162149176,Q_ref1200.5068596918136ms,B12086.532871611882.
W1/W2 references/G2 OPEN. No short-run extrapolation or refreeze.

D459/D460 retained diagnosis:98.33% of D458-minus-Elastic GPU gap is during
common arrival window. New mean decode3636.602 vs3102.616ms; mean other native
occupancy2.653914 vs2.223436, not kernel batch size.31/42 new TPOT misses have
no same-owner proactive-load overlap;33 have peer prefill. No single-load causal
explanation or latency subtraction. Pure planning already offloaded.
D457 file-view optimization reduced controller pickup but did not remove all delay.
Detailed physical-binding/stage/raw evidence in D459/D460 docs and curated folders.

## HydraServe and common workload plumbing

User confirms existing k3s is NOT for this project; appropriate adaptation authorized.
Read HYDRASERVE_ISOLATION_DECISION_20261012.md before next change.
D447 rootlessPodman3.4.4 metadata passed (empty image/container store,9 CDI names).
No actual GPU container/K8s/native LoRA pipeline/consolidation qualification yet.
Prefer isolated project-owned reversible deployment, no existing cluster mutation,
unknown credentials or permission widening; no repeated ownership question.
Native clean HEAD8ae605de354ccfa2e9095514cdb4a9e9c56aa56b preserved.
Read BASE HYDRASERVE_REUSE_REVIEW_20261011.md,
HYDRASERVE_LORA_IDENTITY_REVIEW_20261012.md,
D447_HYDRASERVE_PLATFORM_AND_LAYER_WITNESS.md before adaptation.
CPU stage-local/global-name witness is not GPU/numerical proof.
Never substitute bare vLLM and label it HydraServe.

W1 input plumbing still open: FrozenReplayPlan.load already implements half-interval/
30s phase transform and rejects unfrozen W2 mappings. BASE load_http_replay_plan
currently calls it without profile (W0); explicit common transformed-input interface
must be qualified before W1, not just relabelled. No new trace or baseline tuning.

## Safety, ownership, storage and backups

- MAIN=/home/qhq/serverless_llm_experiment_retry14_baseline,
  branch retry14_continuous_queue_v2, remote faaslora_origin.
  Prior pushed HEAD f303fdb5a5bd0785e641cf8bfcc42e5f107ab3cf.
  BASE=/home/qhq/serverless_llm_baselines,main,origin,
  unchanged HEAD71c52cf6f7331dc2a171ec83db5f1011eaf6d5a6.
- One heavy job. Disk start>=max(150GiB,100GiB+1.5 growth), runtimefloor100GiB.
  Service high72/max80GiB,swap2GiB,CPUs4-23,28-47; actual workers contained
  before init. Auxhigh3/max4GiB,swap0,CPUs2,3,26,27. Separate replay/watchdog.
- Close only exact InvocationID+inode+actual recursive cgroup.procs empty.
  Never use virtual-file stat size. Never stop unrelated processes.
- SSH primelora-artifact-174=lab14@10.199.227.174:8122, strict known host.
  Data192.168.4.174 ports18080/18081, inference192.168.4.178.
  Token outside Git at /home/qhq/.config/primelora-tc-d75/artifact.token; NEVER print.
  Latest actual link1000Mb/s/full; historical100Mbps different state, cause unknown.
- Published readonly gzip D78/D80 reused (1581215261B), no republish/extracted
  pool/request packing. Real reads/transfer/contention/materialization counted.
  No live remote maintenance/hashscan/cleanup/tests/build; do not subtract spans.
- Existing 7B pool /home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix;
  original seed42 input and existing environments reused.
- Protected147 seal: original results/ieee_tc/slora_qualification/d435_20261011/protected_before.json.
  SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
  NEVER stage MAIN configs/generated/lora_manifest_1000.json, SHA
  3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
  BASE scripts/replay_openai_trace.py SHA
  04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
  scripts/run_serverlessllm_relayserve_continuation.sh SHA
  337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
  Preserve nested S-LoRA10 historical patches/default diff SHA
  7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60.
  Preserve all unrelated tracked/untracked user files.
- D459 compressed10 closed old main_outcomes losslessly, reclaimed0.866GiB;
  receipts map old paths. Do not assume every historical main_outcome is plaintext.
- Explicit task-only staging; source/diff/test/secrets checks before push.
  No force, raw large evidence, secrets, history overwrite or unique-data deletion.
  Update status before yielding; never claim the whole goal complete.

## Remaining full approved scope — not canceled

7B PrimeLoRA optimization and W1/W2/G2/common references; remaining HydraServe
native qualification and comparisons;3B optimization/all baseline comparisons;
dLoRA3B B6; all A1-A5 including confirmed-state, elasticity and first-service;
all S1-S13 including layer/mismatch/contention motivation, arrival/GPU/network/
workload/KV/control/backend/EOS/design sensitivity; offline CE/cost/Pareto;
B1/B2/C1/C2/C3, formula/novelty/two-review evidence documents; later academic
figures and verified V2 delivery. Reuse only same contracts. No 13B/new SGLang.
Full original plan plus archived histories retain every setting.
