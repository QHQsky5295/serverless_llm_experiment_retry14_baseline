# IEEE TC execution status — D521 bounded metadata publication

Updated2026-10-14. Prior status preserved BYTE-FOR-BYTE in
EXECUTION_HISTORY_THROUGH_D519.md, SHA
e0ed5415b779effec429b48ea716abde2e3e9b1a379d5abbc11073cdca2f3a99.
Older RUNNING/NEXT paragraphs there are historical, not current process state.

## Current authoritative handoff

D521 default-off `cpu_live_mapping_v1` integrated via existing runner/worker
HOST-budget attachment; no config/run enabled yet. Read
D521_METADATA_PUBLICATION_CONTRACT.md. Fixed pinned staging K*(48N+24L+32),
K=actualnativeconcurrency2; D476 N1024/L14 gives99040B. Exactcopyeventbefore
reuse, sameexecutionstream/thread, nativeGPUaddresses/CPUflags preserved,
currentmapping+slotlayout invalidation/activation/LRU/admission untouched.
ExistingHOSTallowance must fit staging plus unchangedloadworkspace; actual
postallocation check and futureallocatorwide checks charge allpinnedbytes.
Noinstalleddependency edit, newweights/traces/pools or graphreplacement.

D521 CPU qualification COMPLETE:441tests PASS/0fail/error/skip in cpu2/.
Firstattempt441 had3errors fromtwo oldfakeworkerfixtures lacking nativecfg;
new13tests passed. Fixtures nowinclude actualdefaultempty additional_config;
productionchecks notweakened. Failedattempt retained. ActualtwoCPUscopeclosure
recursiveempty/pathgone; cpu2peak2288648192B,high/max/OOM/OOMkill/swap0.
RAW results/ieee_tc/p2_backend_qualification/d521_20261014; curated
paper_results/ieee_tc/p2_backend/d521_metadata_publication. NoCUDAinitialized,
no realDMA/graph/model qualification/performanceclaim. BackupNEXT; actual
backup_receipt/closure will establishcompletion. No livejob.

NEXT ONLY minimal native device metadata/graph/nonzero-LoRA qualification,
including realworkerinstallation/HOSTaccounting and updatecount evidence.
Then ONEordinaryoriginal4000W0 at frozenD476 plussolemetadatastrategy ifqualified.
Do NOTrepeat D520/D521CPUtests, D519diagnosis, HydraServe orbaselinefullruns.
Allfullmatrix remainsACTIVE. No GPU qualification config preparedyet.

D520 checkpoint0f5671ce432bd62e9cb55ca8c5a774c607e51070 pushed/remote-equal;
311CPUtests/96nativeCPUoracle/fiveofficialsources/protected147 archived.

D519 outcome4e272a8be91db44ab1c6e10d2a91e7e7d1609d4a pushed/remote-equal.
D520 read source/history and added ONLY an unintegrated pure CPU metadata helper
faaslora/utils/lora_metadata.py + tests. No serving caller, no installed dependency
edit, no allocator/policy/SLO change, no GPU/remote/k3s operation.
Read D520_NATIVE_METADATA_DEPENDENCY.md. Official native mapping currently makes
GPU reductions/unique and CPU branches from information already present on CPU.
Avoiding that round trip is a new falsifiable hypothesis, not a proved speedup.
High host-free driver time may overlap waiting; do NOT restore unlimited cache,
sweep thresholds, subtract it from step time or repeat D56–60.

D520 CPU qualification COMPLETE:311tests PASS/0fail/error/skip,96 actual installed
Torch2.13 CPU kernel-metadata oracle cases, five officialv0.30 files byte-identical.
CPU scope actualrecursiveempty closed/pathgone; peak3112230912B,events0/swap0.
Raw results/ieee_tc/p2_backend_qualification/d520_20261014; curated
paper_results/ieee_tc/p2_backend/d520_metadata_lowering/cpu_qualification.json.
Explicit tested checkpoint backup NEXT; actual backup_receipt/closure establish
completion. Do not rerun completed qualification. No live inference or CPU job.

D520's then-pending integration is now implemented byD521above. GPUqualification
and ordinary4000 stillpending; neitherCPU result constitutes scientificclosure.

## Authority and execution rules

Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this status before
work/after compaction/before experiment. PLAN SHA
d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83.
Also read STORAGE_PRIORITY_20261013.md,BASELINE_COMPLETION_ORDER_20261012.md,
BASELINE_EXECUTION_RULES_20261011.md,BASELINE_SCOPE_20261014.md,
SLO_BUDGET_RATIONALE_20261014.md. D483 storage maintenance DONE; do not repeat
broadscans. HydraServe necessary adaptation/full4000 D510 DONE; do not restart it.
Shared k3s belongs to another project: no service, credential, permission or
resource changes. Qualified HydraServe isolated rootless Podman path retained.

Published comparison: PrimeLoRA, vLLM(SOSP2023), S-LoRA(MLSys2024),
ServerlessLLM(OSDI2024), HydraServe(NSDI2026), Loquetier(NeurIPS2025).
dLoRA(OSDI2024) only3BW1/W2. ElasticLocality is ONLY a composite ablation,
not baseline/internal-strong-control; historical stronger results retained.
Do not relabel/remove failures or infer improvement from changing the roster.
Only necessary baseline compatibility/correctness/measurement/safety adaptation;
no performance sweeps. Abnormal results still require evidence-based defect review.

One performance run/frozenkey; no CI/repeat means/winner filtering. Full names
PrimeLoRA/ServerlessLLM. Figures deferred until near target. Chinese updates as
paper evidence: completed/current scientific question/remaining experiments.
No new weights/traces/pools/13B/SGLang/paper source edits/history overwrite.
Use existing runners/environments. CPU metadata fixtures are not new workloads.

## Retained D519 evidence (not a new performance point)

Production main runner restored byte-identicalD476 db44e8f068eb30c69466f32ad3d147ffbc45f9e2.
Only D518 opt-in component observer differs in existing serving modules.
ONE1000 diagnostic finished nativecontracts1000/0failure/122510tokens;
independent numerical n_correct remainsnull. All11allocationsreleased; remote
60fetch/74953014wireB/packing0. Servicepeak20621922304B/events0/swap0.
Full component qualification FAILED due64MiB cap in primaryprocess,10/11journals
complete,279/1000 whole-process request coverage. Partial analysis explicitly
retains47038 realclosedsteps;1orphan not imputed. Do NOT repeat analysis/GPU.
45513single-token steps:mean23.669581ms/threadCPU7.468172,allocation.125878,
backgroundfree10.206529,input.919288,LoRAactivation2.100393,scheduler.230154.
987multitoken:step83.052146/CPU82.809049,execute80.685168/LoRA22.072073.
Nested notadditive; no puredecode/prefill/GPUidle inference.
Sourceaf2c2680e375c3c5d14720a810d30d9e6892852b7951fcdb6e1522083005e7b1.
Resultmanifest21c9299fee58c93e993dc7c4e03b244e6a290de243c1b501c0bf23be389c5f0e.
Result doc D519_STEP_DIAGNOSTIC_RESULT.md and result1 evidence remain immutable.

D516 rejected source-query candidate:4000nativecontracts, joint94.775%<95,
U11011.303222888382,2.752825806/request, P95TTFT3321.357967ms. Not accepted;
not proof of causal harm from one run. RestoredD476, no repeats.
D517/D518/D519 diagnoses/qualifications/backups completed; do not redo them.

## Frozen evaluation and sealed 7B W0

Before comparison/configselection read FULL METRIC_PROTOCOL_FROZEN_V1.md and
METRIC_PROTOCOL_SINGLE_RUN_V2.md, SHAs
5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22,
193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
G1 all4000correct/nativecontracts + jointTTFT/TPOT>=95%, then physicalGPU-s.
G2same + U<=.75Uref, W1/W2P95TTFT/observedSLOdifference; CE supplementary.
Numerical binding/count evidence is not independent numerical proof; unknown
n_correct staysunknown.95%/75% project points, not universalindustrynorms.
Keep approved90/95/99%/.5/.75/1/multiplier sensitivities including negatives.

D513 six-system published table SHA
044c04d6bb5b894761628146a65de248a53436cdd4ba23346837eccb3473b17b:
(system, joint, GPU-s/boundrequest, P95TTFTms)
PrimeLoRA D476 .97425,2.694612200556614,2315.363336;
ServerlessLLM D414 .98125,4.022440685881069,1052.246361;
vLLM D423 .99475,4.028844290537294,1200.506860;
S-LoRA D435 .94425,3.7892694676132525,17699.067366 NONFEASIBLE;
Loquetier D445 .0435,4.249595808607468,555110.032229 NONFEASIBLE;
HydraServe D510 .68725,2.4155381525526174,8851.575365 NONFEASIBLE.
HydraServe all34workerspp1: does not prove pipeline/consolidation performance.
ElasticLocalityD453 .95775/2.473178968/2420.607829 remains ablation evidence.
PrimeLoRA U10778.448802226456 is33.011%/33.117% belowServerlessLLM/vLLM,
but lowerattainment/highertail; no all-metric/cross-scenario optimum claim.

D394 SLO SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3:
input<=616/>616 TTFT2995.348713684507/4633.345350589479ms,
TPOT59.74409694902162/85.24698166015581ms.
W0Uref16115.377162149176,B12086.532871611882,Qref1200.5068596918136.
W1/W2refsOPEN; W0 is NOT W1/W2/G2closure.
Matched4000contractSHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.

## Safety, protection and publishing

One heavy job. No build/tests/hashscan/compression/backup/remote changes during
inference. Diskstart>=max150GiB,100GiB+1.5growth; runtimestop100GiB.
Servicehigh72/max80GiB,swap2,CPU4-23,28-47; auxhigh3/max4GiB,swap0,CPU2,3,26,27.
EmptyhoststartMemAvailable>=102GiB; otherwise remaininggrowth formula.
External16GiBhoststop. Actualworker restrictions before load; no blindOOMretry,
globalGPUreset/reboot/driver changes/unknowncleanup. Closure checks exact
Invocation/inode,actual recursive cgroup.procs and capturedpathgone.
Latest read-only disk~162GiB,hostavailable~104GiB; gate freshly before next job.

MAIN retry14_continuous_queue_v2/faaslora_origin. BASEHEAD
09baa9379bdc3518f18f408d2cad3763c2754554, unchanged. Protected147 seal
originalresults/ieee_tc/slora_qualification/d435_20261011/protected_before.json
SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
NeverstageMAINconfigs/generated/lora_manifest_1000.json SHA
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
BASEscripts/replay_openai_trace.py SHA
04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
scripts/run_serverlessllm_relayserve_continuation.sh SHA
337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
Explicit paths only; checksum/tests/diff/secrets before commit/push; no rawlarge
files, credentials or forcepush. All old data/failures/userdirt preserved.

RemoteSSHprimelora-artifact-174 lab14@10.199.227.174:8122 strictknownhost;
data192.168.4.174ports18080/18081 inactiveafterD519. Token outsideGit
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
D78/D80readonlygzip1581215261B reuse, no republish/extractedpool/requestpacking.
Latestactual1000Mb/s/full; old100Mbpsdistinctstate,causeunknown. Real necessary
read/transfer/contention/materialization count; no subtracting old accumulated
spans. Remote start/health must explicitly complete before launch.

## Full remaining scope — ACTIVE

7B PrimeLoRA evidence-driven optimization;7BW1/W2/G2/commonrefs;
3Boptimization/allbaselines;dLoRA3BB6;A1–A5confirmedstate/elasticity/handoff/
hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/load/GPU/network/
workload/KV/control/backend/EOS/designsensitivities;offlinecost/CE/Pareto;
B1/B2/C1/C2/C3/formula/novelty/two-reviewdocs;lateracademicfigures/V2delivery.
No stage closure from this CPU-only result, no goal completion or scope reduction.
