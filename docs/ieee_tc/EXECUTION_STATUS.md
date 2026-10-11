# IEEE TC execution status — D435 S-LoRA full W0 preregistered

Updated 2026-10-11. This is a current checkpoint, NOT a replacement experiment plan.
Full earlier status is preserved byte-for-byte in EXECUTION_HISTORY_THROUGH_D433.md;
archive SHA60aeae9c98335a1b9c8d131441cb052b0351e09ac48f6b07d57ecf7ab6560d98.
earlier history through D402 remains EXECUTION_HISTORY_THROUGH_D402.md.
Search those histories and D-series reports before changing code. Their old NEXT/
RUNNING entries are historical. No approved experiment or failed evidence was removed.

## Mandatory authority and evaluation

- Read full /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md at task start,
  after compaction and before experiments; source and approved FULL_NAMES snapshot
  SHA f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
- Read BASELINE_EXECUTION_RULES_20261011.md. No non-PrimeLoRA performance
  optimization/configuration search; only necessary compatibility, correctness,
  measurement and safety adaptations. Reuse qualified prior points and report
  their actual outcomes, including nonfeasibility; do not call unsearched points optimal.
- Read METRIC_PROTOCOL_FROZEN_V1.md and METRIC_PROTOCOL_SINGLE_RUN_V2.md
  before configuration/comparison. Their SHAs remain respectively
  5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 and
  193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
- G1: all4000 native/source/adapter contracts and common joint TTFT/TPOT >=95%,
  then minimum physical lifecycle GPU-s/request. G2: same + U<=.75U_ref,
  then W1/W2 P95TTFT and observed joint-SLO difference. CE supplementary.
  Relative TPOT regression alone is not failure. Raw unknown numeric flags stay unknown;
  derived V2 binding metrics require complete evidence, not HTTP200 alone.
- One performance run per frozen execution key, no repeated means/CI/significance
  or best-repeat selection. Preserve development/reuse identity and every failure.
- Order: sealed7B PrimeLoRA -> remaining7B baselines -> comparison/necessary
  PrimeLoRA optimization ->3B -> ablations/mechanisms/sensitivities.
- Full names PrimeLoRA/ServerlessLLM. No figures before near target; no13B,
  new SGLang, paper edits, new weights/trace or pool regeneration.

## Current action — D435 full W0 pending launch

D435_SLORA_FULL_W0.md preregisters one original4000/132ID/458224token W0 using
the exact D432/D433/D434 native config. No baseline runtime/config/analyzer changes.
Raw /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011;
private /tmp/tc-d435-7bs; tmux name tc-d435-7b reserved but not launched yet.
37CPU checks passed1.696s; fresh preflight passed,147oldprotected entries unchanged.
D434578-source-file hashes still match after object cleanup; three user-file SHAs,
native config/runtime/source-view and historical tenfilepatch unchanged.
NEXT task-only backup -> input/cachefreeze -> remote health -> one gated full4000.
No live GPU/remote job at this preregistration checkpoint. Full goal remains ACTIVE.

## D434 closed — do NOT repeat

D434 original1000 compatibility completed by2026-10-11 22:44:35Asia/Shanghai.
1000offered/submitted/terminal/native/source/adapter/timingcontracts,0failures,
122510native tokens,60IDs;fouractualTP1workers served/500-ID registries.
E2E/TPOT/native recomputation error0ms. U3542.956034773262GPU-s,
3.5429560347732623/binding-validrequest, all physical leases closed.
CommonD394 TTFT750/TPOT999/joint749of1000=74.9%,NONFEASIBLE prefix,
not fullW0/ranking/reference/optimum. Mean/P95TTFT37636.062/236234.733ms,
TPOT33.501/49.954ms,E2Emean41782.526ms. Raw unknown numerical flags unchanged.
Source0-99/100-499/500-999 joint1/100,249/400,499/500; last500 meanTTFT241.488ms.
Do not drop startup from primary denominator or infer fullW0 from a short prefix.
Fullstaticpool500HTTP/420740410wireB/12985984450verifiedmaterializedB,0packs.
Preparation span95.131748s; native ready notice+221.662/320.676/421.199/518.432s.
Deployment transfers counted once,requestwire0/preparation references1000.
Servicepeak62617686016B,hostavailablemin60923981824B,diskmin150234251264B,
high/max/OOM/swap0. ActualGPUreleased/servicepathremoved/exit0;remote3ownedunits
stopped by InvocationID/MainPID0/portsabsent; resolver and remote evidence preserved.
Strict existing slora_resident_rr_v1/prefix1000 analyzer passed, no runtime/analysis changes.
ReportD434_SLORA_1000_COMPATIBILITY.md;curated baseline_audit/d434_7b_slora_compatibility/:
auditSHA05a595c69063efc92484a208325b3476fd4c49c857a6a450b281845af39660a0;
578-file/63678229B evidence_manifest SHA
bf444b0c1ce339e6a63342b6484e8cc2dd1d15fbe2d317163d9a9378c65ff7ab.
All578 source-file SHA/size entries independently verified. Closure backed up:
MAIN869fdff76c57a5b8de6efe718d453d891cec2809 and
BASE33fc840bc282839628a8275fe016671b2541c955, remoteSHA verified.
Audit/seal aux0005/0006 exactidentity/empty-tree closed, high/max/OOM/swap0.
Aux0007 and cleanup0008 now closed after exactidentity/empty-subtree checks.
Only verified private objects reclaimed:5000files/12985984450payloadB,
observed freed13004435456B,free163189366784B. Originals/raw/remote cache retained.
Cleanup-only aux peak3221749760B/high4003/max0/OOM0/swap0; this was after
performance and did not alter the D434 performance resource record.
Receipt paper_results/ieee_tc/storage/20261011_d434_download_cache.json.
147protected entries unchanged,postSHA23b9be29909ef728f3450ea7e32f592419d81e319012d383ea290ecd618cfd0d.
Three protected user-file SHAs and nested10filepatch unchanged.
NEXT freshgates -> preregister/backup and execute ONCE D435 full4000W0 SAME native config.
No100/500/1000 repeats,baselineparametersearch,oldsealedW0repeat or3Bjump.
Cleanup restored about151.98GiB; launch still requires fresh complete safety gates.
Full goal ACTIVE/all remaining approved scope OPEN. Previous turn launched D434;
this turn monitored same live handle to terminal and completed strict audit, progress.

Raw: /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d434_20261011
Private: /tmp/tc-d434-7bs. Reuse run/start/stop/health/freeze/preserve/seal recipes.
Original1000viewSHA0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981.
Nativeconfig slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d.
D430 source-view manifest SHA
51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004.
DP4/TP1/RR/sequentialstartup/packedHOST/native swap nonBMM;
input984/context1024/tokenbudget14000/batch1024/running1000.
Full500 HOST preparation afternotice, business notice+60, actualGPU allocations,
qualificationprotection1800s; no freeGPU/adapterpreload. ExistingTritoncache2055files/62246523B.
Prelaunch37CPUtests pass1.858s inexistingouterPython3.12; earlier systemPython3.10
missingnumpy attempt retained, no install/codeworkaround. No performance retry.
Execution MAIN47be5422cbf65c8c0905f78668a83c188b7b9ccf,
BASE5059cb0b86c5a5378a772adaa25aa80bb7a1b240 both pushed/remoteSHAverified.
Runningdoc backupMAIN677a53e049bea6118d8f4e9d559179ada2098e3c notexecutioncommit.
Serviceprimelora-tc-svc-b0ce62d358e94489b1d5958c0e83ede0.scope,
InvocationID18c0f4b4575a45be824a81b181b46476 now released/removed;notrestart.
Notice1235966.54994207,arrival_start1236026.54994207,end1237105.9642220077.
Remote priorlink1000Mb/s/full, unchangedD78/D80 once-only gzip cache;no republish.
Closedremote3B/7B/monitorInvocationIDs158b40360fdf46bb818a036493b0246c/
e0525a84e0af4598a1e715bfac1e7255/1028fcd55b374827a52f7d705422add0.
Remote7Bclock8e030525ed52454e89698c6a7e7b6287;copiedtransferSHA
092504a38e8e2b7f9f57be60442cd2c1068db89d36a3777c43236007f407aa6c,
monitorSHAb6c9853df74bb7721af442f618e967f46ff66a0d1c459f1e9ef4ec44ffaa798e.
Aux0001-0008 closed. No remaining D434 process scope.
No figures,paper/backend/policychange or newweights/trace.

## D433 closed — do NOT repeat

500-ID functional coverage:500 offered/submitted/terminal/native/source/adapter,
68925 native tokens,0failures,exact500IDs,four actual workers served/500-ID registries.
E2E/TPOT/native recompute error0ms. D394 TTFT268/TPOT494/joint266 of500=53.2%;
NONFEASIBLE functional view, not fullW0, ranking, reference or optimum.
U1743.8732772790827 GPU-s,3.487746554558165/request.
Fullpool500HTTP/420740410wireB/12985984450materializedB,0packs.
Deployment bytes counted once; per-request references are not extra transfers.
Peakservice62410342400B,high/max/OOM/swap0; all physical leases closed.
D433 report: D433_SLORA_500_ADAPTER_COVERAGE.md.
Curated baseline_audit/d433_7b_slora_coverage/audit.json SHA
7cc9f8e220e0b5cf761cd5c416e6d3126d22f42cb77f46494af10341feedf3f7.
576-file evidence_manifest SHA
cb03ba5ddf53241e427712c285197bdcb3f89def15c2480f45e6872a4b20003c.
All576 source hashes verified.37CPU tests passed1.636s before launch.
Execution MAIN627aebff3fdaf67822946e52eacb7c9291f2eb30,
BASE15077ab4aeeac9532b1cb8d9f18567d76883163c.
Closure and cleanup backups subsequently pushed/verified; latest MAIN
29e98482d0375181fb020f53747d29ecdb136336,
BASE219ad363a83e44487d5df4831861a0edad7e3d2f.
All GPU/remote/aux scopes closed by actual identity and empty process trees.
Only /tmp/tc-d433-7bs/remote_artifacts/objects reclaimed after references, retained
source bytes/inodes/hash and process checks:5000files,12985984450payloadB,
freed13004427264B. Originals/raw/remote cache retained. Three unreadable sameUID
process entries disclosed. Do not repeat cleanup or redownload to repeat a passed audit.
Receipt paper_results/ieee_tc/storage/20261011_d433_download_cache.json.
D432 original100 passed; bounded D385 numerical witness95exact/5differences,
3known margins/2unknown retained; no full numeric proof or exact-token gate.

## Sealed W0 comparison and reference

D397 seals PrimeLoRA D371 reused-development W0, not independent holdout:
4000/132IDs/458224 native tokens,97.70% joint,3.6206309810630626 GPU-s/request.
D414 ServerlessLLM98.125%,4.022440685881069;D423 vLLM99.475%,4.028844290537294.
PrimeLoRA GPU-s about10% lower, but P95TTFT2837.896ms vs1052.246/1200.507ms
and jointSLO not leading. All meet95%; not all-metric/cross-scenario/G2 optimality.
Allthree132HTTP/131480060wireB: no W0 remote-byte-reduction evidence.
See D424_MATCHED_7B_W0.md and paper_results/ieee_tc/main_comparison/d424_7b_w0/.
W0 U_ref16115.377162149176,Q_ref1200.5068596918136ms,B12086.532871611882GPU-s.
W1/W2 references and G2 OPEN. D395/D396 cannot set reference due measurement gaps.
D394 SLO manifest paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json,
SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
<=616input:TTFT2995.348713684507ms/TPOT59.74409694902162ms;
>616:TTFT4633.345350589479ms/TPOT85.24698166015581ms. Do not regenerate.
Mostly-zero/repeated weights:500logical IDs,498zero/2nonzero,4weightSHA/6tree classes.
Native bindings, independent numerical evidence and content diversity are distinct.

## Safety, ownership, storage and backups

- MAIN=/home/qhq/serverless_llm_experiment_retry14_baseline,branch retry14_continuous_queue_v2,
  remote faaslora_origin. BASE=/home/qhq/serverless_llm_baselines,main,origin.
- One heavy job. Disk start>=max(150GiB,100GiB+1.5growth), runtimefloor100GiB.
  Service memory.high72/max80GiB,swap2GiB,CPUs4-23,28-47.
  Auxhigh3/max4GiB,swap0,CPUs2,3,26,27. Actual workers contained before init.
  Separate replay/watchdog; stop only exact owned units after identity checks.
- Strict SSH alias primelora-artifact-174=lab14@10.199.227.174:8122.
  Data192.168.4.174ports18080/18081,inference192.168.4.178;credentials outside Git.
  Token file /home/qhq/.config/primelora-tc-d75/artifact.token, never print.
  Current pre-D433 link1000Mb/s/full; historical100Mbps separate state, cause unknown.
- D78/D80 published gzip reused, no request packing. Real reads/transfers/competition/
  local materialization counted. No live remote administration/hashscan/cleanup,
  tests/second heavy job. Never subtract cumulative spans to invent corrected latency.
- Original7B pool /home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix;
  original seed42 trace and existing environments reused.
- Protected147-entry seal D433raw/protected_before.json. Historical results unchanged.
  NEVER stage MAIN configs/generated/lora_manifest_1000.json SHA
  3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df;
  BASE scripts/replay_openai_trace.py SHA
  04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
  scripts/run_serverlessllm_relayserve_continuation.sh SHA
  337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
  Preserve nestedS-LoRA ten historical patches/default diff SHA
  7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60.
  Preserve all other unrelated tracked/untracked files.
- Explicit task-only staging, diff/tests/source/secrets checks; no force/rawlarge/secrets.
  Update current status before yielding. Do not claim full goal complete.

## Remaining full approved scope — not canceled

Remaining7B baseline qualifications/comparisons, W1/W2/G2 and references, PrimeLoRA
optimization where justified,3B optimization/comparisons,dLoRA3B,ElasticLocality,
Loquetier/HydraServe native qualification, all A1-A5 and S1-S13, CE/cost sensitivity,
two-review action/design documents and later academic figures. Reuse only same contracts.
B1/B2/C1/C2/C3 and formula/novelty evidence remain required. Hardware limits disclosed.
Full original plan + preserved history define every setting; this tracker is not a shortened plan.
