# IEEE TC execution status — D411 full4000 W0 RUNNING; do not restart

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

D411 RUNNING since2026-10-11 03:12:24 Asia/Shanghai, tmux tc-d411-7b.
Do NOT restart or launch another heavy job. Main execution
896af899396983350c4544a9d0277283bc28339d, pushed and remoteSHAverified;
baseline execution2a5dfd1154ba93c82d9b7076fde0a8521543933b.
Baseline running-checkpoint backupf8af38bcb7a401114e3782e54bd5b1396a40fc76
pushed toorigin/main andremoteSHAverified;not the executioncommit.
Service primelora-tc-svc-cdae00875efc4791b2fad4ef286ef88f.scope,
InvocationID426554ae3dcf4dd2999ceeecf4a2e012;aux
primelora-tc-aux-d4110000000000000000000000000001.scope,
InvocationIDe1a1176feced4bd5ac90df1c38152e09.
Deploymentnotice1166817.112284243,business t0=1166877.112284243.
Full source beforelaunch validated4000/4000,132actualIDs,458224targettokens,
arrivalwindow3963.9089031219482s (~66min). No final performance result yet.
Remote link1000Mb/s/full,publishedcache reused;7Bclock
remote-process-monotonic:158e79ec85e345c19ea6fb6250184f23.
Remote3B/7B/monitor InvocationIDs respectively
3dadc35fa0e8430d8ef0a231e943546c,f5a284ee1ad643d6b4417e3987500387,
43c6e563e84f43689b0b27a570ea1255. Keep unchanged during business.
Loaderoverlay INSTALLED with exact7-memberreceipt inD411raw;restore AFTER
actualGPUrelease from that receipt. Existingwatchdogactive,4000requestcontracts
registered;no pressure/OOM/swap in initialsample. Continuousguard authoritative.
Preflight passed;protected147-entryhashsame96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d,
sole unchanged historicaluserreplaymismatch. No oldresultchange.
NEXT: monitor existingtmux/localjournals only,no live remoteadministration or
hash/cleanup. Afterterminal actualGPUrelease ->restoreoverlay ->stopownedremote
via D411stop_remote.sh ->preserveworker/resolver/remoteevidence ->auditfull4000
with existinganalyzer/commonD394 ->save/backup ->7Bnextbaseline. Do not repeat
this key for a better result,retunePrimeLoRA,or jump3B beforecomparison.

D411 preparation checkpoint follows. One full original
4000W0 development observation with unchanged D410 public graph configuration.
See D411_SERVERLESSLLM_FULL_W0.md. Existing analyzer now explicitly validates
full4000 count/source_count/W0/rate1 and exact offered view;19CPUtests passed
0.095s in4GiB/zero-swap scope. No online baseline code/algorithm change.
Raw root `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d411_20261011`;
existing runner/config/source view prepared, no new trace/pool/checkpoint.
Baseline execution2a5dfd1154ba93c82d9b7076fde0a8521543933b.
Native compilation diskcache now3858files/104734877B,aggregateSHA
5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791;
recorded beforelaunch, distinct from cold LoRA/HOST/KV/GPU state.
D410 downloadedobjects ONLY reclaimed afterbyte/source/inode/reference checks,
600files/1474429899payloadB,1476268032B observedrelease;
diskafter161905242112B. Originalpool/rawlogs/remoteimmutablecache retained.
Three inaccessible sameUIDproc entries disclosed; receipts inD410raw.
Preparation andlaunch steps completed above. No100/1000 repeat,
no PrimeLoRA tuning or3B jump. The following D410 next-action is superseded.

D410 TERMINAL, started 2026-10-11 02:40 Asia/Shanghai, released before03:00.
Original1,000 W0 requests,
no adapter remapping, fixed_length_greedy_v1, public enforce_eager=false as D409;
no new policy or backend-limit changes. This is development validation, not
full4,000 qualification or a baseline optimum. Do NOT restart the closed run.
Raw root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d410_20261011`.
Tmux `tc-d410-7b`; execution main d6fbaaf245f236e6f05babb2f1b7f7919e8e02f7,
baseline336576bc8a3e7aca34febad52372911289a9ca87.
Service primelora-tc-svc-1913f29ca4e94a49accc7c3057578dc1.scope,
InvocationID e29a72bb3c284637a1b6f62d655fd6da; aux
primelora-tc-aux-d4100000000000000000000000000001.scope.
Deployment notice1164859.37841325, business t0=1164919.37841325;
unchanged60s notice/1800s qualification protection/commonD394SLO.
Preflight passed, actual contained launch/watchdog receipt retained. GPU released,
service domain removed,emptyaux stopped;overlayRESTORED,remote3ownedunitsSTOPPED
by exactInvocationID and portsabsent. No live job.
Remote7Bclock remote-process-monotonic:672039983ce04b839837c414cf6adaf3;
3B/7B/monitor InvocationIDs respectively b00f633737b34674904ea8415758db2c,
4f9d03b0696440d88229ab2742a73596,5c3bd0ca8fb0484598261d680cd81186.
Authenticated health confirms unchanged published-gzip protocol; no new cache
publication, packing or local pre-materialization. Current remote link1Gbps/full.
Cold per-run LoRA cache; native compiler diskcache retained separately:
75,240,459B/290files, aggregateSHA
f8c62659046c116bea0ce79501c52c8824979ca8a676bb5c04cb74671cf7b9a8.
Four actual native compiler-cache hits verified,3.538..3.685s loads and2s/.09GiB
captures each. Actualready notice+138.113..176.535s;all startup/capture counted.
D408 eager1000 is the development comparison; D406500-ID/D409 bounded
reference evidence reused, no repeat just to obtain exact-token agreement.
1000/1000native/source/adapter/content/timing contracts,0fail,122510tokens,
60IDs,4actualservedengines;native/E2E/TPOTrecomputeerror0ms.
TTFTmean2672.734439/P9513276.391335ms;TPOTmean27.846560/P9536.115468ms;
E2Emean5996.914935ms. CommonSLO TTFT915/TPOT1000/joint915=91.5%NONFEASIBLE.
PhysicalU4611.030857GPU-s,4.611031/binding-validrequest,completeleases.
60HTTP/74953014wireB/1474429899verifiedpayloadB,940cachehits,0packs.
All1000D408/D410input/adapter/prompt/target/content contracts matched.
Descriptive joint improvement24.1pp/meanTPOT reduction56.19%,butUincreased0.58%:
nativefourcardstorehold remains,not claimedGPU savings orstrictcausalestimate.
Source0-99/100-499/500-999joint46/100,400/400,469/500;all85missesareTTFT.
Raw unknown numerical flags unchanged. Full4000 ranking remainsOPEN.
Report D410_SERVERLESSLLM_GRAPH_VALIDATION.md;
audit paper_results/ieee_tc/baseline_audit/d410_7b_graph_validation/audit.json,
SHAfa1be0a21dd3b9f842eb41a75b618cf59e30509ad069de29c69e69ee51e3d44a.
Companionruntime_evidence and161-filemanifest,manifestSHA
93811678b42a7fc5658985dba25b5acd180d75b10a5baf6a3de4ee6094f5509a.
15analysisCPUtests passed0.041s. Protectedbefore/after147entries unchanged,
same96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d.
Baseline closurebackup2a5dfd1154ba93c82d9b7076fde0a8521543933b pushed to
origin/main and remoteSHAverified;not the D410 executioncommit.
Next: completeevidencebackup -> reclaim only verified D410downloadobjects and
recheckdisk -> extendexistingpopulationanalyzer tofull4000 withCPUtests ->
one complete4000W0 observation viaexistingqualifiedrunner/publicgraphpath.
Currentdisk160430882816B is below150GiB startgate;do notlaunchuntilrechecked.
Do not repeat100/1000 gates or use95%failureasreason todelete this observation.
No new local parameter sweep, no PrimeLoRA retuning or3B jump. All entries below
are retained historical checkpoints; their NEXT instructions are superseded.

D409 TERMINAL 2026-10-11 02:27 Asia/Shanghai: existing 100-request affected-path
qualification, original W0 prefix, explicit --no-enforce-eager; no other public
backend/scaler fields changed. Raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d409_20261011`.
Tmux `tc-d409-7b`; main execution5a3dac21ffe52e0651c988584feecda003f1a64d,
baseline d6dcf14449e595f461bb933007b62c87423377d2.
Service scope primelora-tc-svc-8d8865fa28cd4c50bbcc940747b9fff2.scope,
InvocationID df291d0a5a9c42b2a14a0866a110f3b9; aux
primelora-tc-aux-d4090000000000000000000000000001.scope.
No live job; do not restart D409. Retained
launch7b_repaired.launch/watchdog.jsonl,replay.jsonl,service.log and native logs.
Native vLLM0.10.2 already
sets inductor compile_threads=1; no new thread parameter/build introduced.
Remote reused D78/D80 read-only published gzip cache,link1000Mb/s/full,
7Bclock remote-process-monotonic:ad5c8352384b424e9db29edaaa2a1f88.
Three owned remote units STOPPED by matchingInvocationID/portsabsent; GPU
released/service removed/emptyaux stopped; overlay RESTORED from exact backups.
No cache republication/request packing. Worker/resolver/remote evidence retained.
100/100 native/source/adapter/timing contracts,0failed,17369tokens,4servedengines.
TTFTmean63240.114772/P95111377.363558ms;TPOTmean31.699636/P9534.570687ms;
jointSLO0/100,TTFT0/100,TPOT100/100. Shortgate NONFEASIBLE,not a fullW0 optimum.
U1033.029004GPU-s,29HTTP/50650865wireB/724665100verifiedpayloadB,71cachehits,
0packs. Four actual PIECEWISE capture completions;no silent eager retry.
Independent D385 reference reused offline:97/100exact,3differences kept with
existing first-divergence/top2 witnesses;raw unknown numerical flags unchanged.
Audit_v2 SHA9c35cbb2661567d718f9671e605cf69b38099fdbcefbebbf9c688f2200db1218,
paper_results/ieee_tc/baseline_audit/d409_7b_graph/audit_v2.json.
See D409_SERVERLESSLLM_GRAPH_RESULT.md.15analysisCPUtests pass0.052s.
Evidence_manifest126retainedfiles allSHAverified;manifestSHA
52c597fe13b0b0bb2782b1a533eba8d12a10a1b2ad5f2a26518107fac2a30fcb.
Baseline closure backup336576bc8a3e7aca34febad52372911289a9ca87 pushed to
origin/main and remoteSHA verified;this is not the D409 execution commit.
Next: original1000 graph/public-config validation on a new key; record retained
native compile diskcache separately from cold per-run LoRA cache,all actual
startup cost remains counted. Do not expand unrelated knobs or repeat100 gates.
Nativecache path ~/.cache/vllm/torch_compile_cache/1c637ca91b/rank_0_0/backbone.
Keep PrimeLoRA D371; then full4000 W0. Disk at gate minimum161426763776B;
recheck>=150GiB+growth before another heavy job. No cache deleted this gate.
The CPU-only D409 entry below is a preserved earlier checkpoint.

D407 COMPLETE 2026-10-11: no GPU repeat. Existing D404100request outputs joined
to independent D385 HF/PEFT reference with source/input/adapter/content hashes:
100/100 identities,94exact sequences,all4weight classes. Six differences kept;
only3share D385's independently recorded divergence position/token and top2
margin0/.015625. Other3margins unknown,not assumed rounding. Eleven CPU tests
pass0.050s. Audit paper_results/ieee_tc/baseline_audit/d407_7b_reference/audit.json
SHA7843794689b7aeaee4ac8ed067de5871ef858729130c7f185a0c9566e6c0b5d4.
See D407_SERVERLESSLLM_REFERENCE_REUSE.md. Frozen V1/D393 do NOT require exact
cross-backend tokens; do not reintroduce D391's superseded full-exact oracle.
Bounded numerical evidence is reusable; raw unknown fields stay unchanged.
D408 COMPLETE (2026-10-11 02:07 Asia/Shanghai): original1000 W0 requests,
trace_prefix_v1,no coverage remapping,60actualIDs,target4/min1/max4/keep10,
same D406 native backend/eager mode. Raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d408_20261011`.
Execution main242b3c455e13ac10920f0af8a04695af05f52a7f,
baseline25848cfb57ef15351f800daf7062cffb2a0d3807. No new weights/traces/cache.
1000/1000 native/source/adapter/content/timing contracts,0fail,122510tokens,
4actual served engines,live scheduler/no exception/empty allocations.
TTFTmean4683.000724/P9536802.779816ms;TPOTmean63.561872/P9568.849439ms;
E2Emean12420.357748ms. Common D394 TTFT828/TPOT800/joint674 of1000:
67.4% joint,NONFEASIBLE,not a4000-request comparison/config optimum.
Physical U4584.427947GPU-s,4.584428/binding-valid request,complete releases.
60HTTP/74953014wireB/1474429899verified payloadB,940cache-hit requests,0packs.
Raw n_correct=null/numerical flags unchanged; reuse bounded D407 separately.
Audit v2 SHA3a159e40315aa5291e70972b621dd77a1222f46c0b308785022c514cfcb28ca4,
paper_results/ieee_tc/baseline_audit/d408_7b_compatibility/audit_v2.json;
earlier audit.json retained. Companion evidence_manifest189files.
Report D408_SERVERLESSLLM_1000_RESULT.md. Offline analyzer15CPUtests pass0.050s.
GPU contexts released/service removed/empty aux stopped/overlay RESTORED.
Remote3ownedunits stopped by InvocationID,MainPID0/portsabsent; transferred
remote logsSHAverified,resolver+worker evidence retained. Protected147entries
before/afterSHA96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d
unchanged,sole historical user replay mismatch. Health prelaunch corrected from
system Python lackingnumpy to existing nativeenv before any business request.
No live job; do NOT restart D408 or earlier completed gates.

D409 CPU-QUALIFIED, NOT GPU-RUN: D408 startup and steady decode both matter.
Ready receipts notice+167.977..168.029s; first100meanTTFT35.417s. Source100-499
and500-999 jointSLO78.5%/70.6%,TPOT~63ms. Helper had fixed eager=true, whereas
official pinned backend exposes false/hybrid. Existing helper now has explicit
--no-enforce-eager,default unchanged,only one public backend config field differs.
No RR/scaler/loader-policy change or silent eager retry. Baseline89CPUtests pass
7.234s in4GiB/zero-swap scope; see baseline docs/ieee_tc/D409_SERVERLESSLLM_PUBLIC_EXECUTION_MODE.md.
Baseline checkpoint d6dcf14449e595f461bb933007b62c87423377d2 pushed to origin/main;
remote branch SHA verified. D408 evidence manifest189files SHA
dbaa0ab5343027b29b0bcc964c1601a8d876f42c80fad63d1dd71ba220ed06b3,all189verified.
Next: full plan/status + safety reread, one new-key affected-path100 graph/native
timing/adapter/memory gate, then original1000 compatibility if valid. Reuse D406
content/full-ID evidence, verify graph slot execution rather than assuming it.
Keep target4/min1/max4/keep10,backendlimits,FP16,common60s/1800s unchanged.
Hypothesis is decode reduction versus capture startup/memory trade-off,not proven.
Do not expand unrelated parameters; retain failure and return to full4000W0
after this bounded public-config check. Do not repeat numerical probes merely
for token exact-match selection. Preserve PrimeLoRA D371 throughout.

D405 COMPLETE 2026-10-11: audited historical replay files: clean7B, earlier7B,
3B each4000reported successes but132actual logical adapter IDs. Original7B
first500/1000 touch46/60; D404 native100 touches29. D80 whole-pool HTTP evidence
does NOT establish whole-pool engine execution. No historical500engine gate
found in the audited sources. Inventory JSON SHA
62b649e75ca1f6bfd5fd38367716d17e3d78231d1991b512aa10a0c9fa6ed0eb.
Existing baseline helper now has opt-in adapter_coverage_v1, strictly500,
qualification-only/nonformal. Existing first500 prompt/arrival/target rows,
sorted static pool IDs, explicit original/derived row provenance; no new trace
or weights. Publisher/physical ledger/deadline share the exact view.87CPU tests
passed7.478s in4GiB/zero-swap auxiliary scope. Baseline checkpoint
7a37e1f9b3220b26f9f0166e36e2821e84aa1d4b pushed and remote hash verified.
See D405_SERVERLESSLLM_COVERAGE_INVENTORY.md and baseline detailed report.

D406 TERMINAL 2026-10-11; former tmux `tc-d406-7b` ended. Exclusive raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d406_20261011`.
Execution main6eed716bed1ed994ab1d35e7a00c6ecf0469e4d7,baseline7a37e1f9b3220b26f9f0166e36e2821e84aa1d4b.
500/500responses/native contracts,exact500logical IDs,0failure,68,925tokens;
fouractualservedengines,live scheduler/no exception/empty queue. Native/E2E/TPOT
maximum recompute error0ms.500realHTTP transfers420740410B,verifiedpayload
12985984450B,0packs/0cache-hit requests. Physical GPU-s2717.3239195765927,
complete leases. Raw n_correct=null/independent numerical=false preserved.
Audit SHA24da017f3612e2e15c8533b03d30ffa3ad612c8b52e6dfa3a7bd2c445cb419d7;
main paper_results/ieee_tc/baseline_audit/d406_7b_coverage/audit.json.
Evidence manifest602files allSHAverified,manifest SHA
676f15919945163bd88196c582fa6ee1d19dfaeb85d1854ae6973c15c977d7dc.
Baseline evidence backup25848cfb57ef15351f800daf7062cffb2a0d3807 pushed/verified.
Report D406_SERVERLESSLLM_500_ADAPTER_RESULT.md. ActualGPU released,overlay
RESTORED,remote3ownedunits stopped/portsabsent,empty ownedauxscopes stopped.
Protected147entriesbefore/afteridentical;sole old mismatch user's replay edit.
Remote link1000Mb/s/full. Fullbyteaudit passed before cache-only reclamation:
objects /tmp/tc-d406-7br/remote_artifacts/objects removed;originalpool/remote
cache/receipts/journals/rawlogs retained. Observed reclaimed13003907072B;
diskafter162353836032B. Raw cache_cleanup_before/after.json record checks.
Reauditing removed cache bytes requires rematerialization; do not redownload
wholepool merely to repeat an already verified gate. Independent numerical
qualification/actual content diversity remain distinct from native binding.
View SHA286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d.
Public target4 matches existing engine max-seqs4/max-loras4; min1/max4/keep10
unchanged. D404 observed overlap2/engine supports capacity validation, not GPU
saturation. Functional mapping is not a performance comparison or optimum.
Next audit reusable numerical qualification,then original1000 compatibility/
public configuration validation and frozen full4000 W0. Existing analyzer's
explicit500view validator:6CPUtests pass0.036s; D404 original-population
reanalyzed numeric fields unchanged, no performance repeat. No live job.
Do not restart completed D406,D404 or old D72/D73 pair.

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
2. D404 scale-out and D406 whole-pool native execution gates passed above.
   Do not repeat them or D72/D73 polling pair. D72/D73's1000 prefixes remain
   R2/local/failed,not new-contract compatibility;reuse diagnostics/assets.
3. Audit/reuse independent numerical correctness evidence before promoting
   full-system qualification; original1,000 compatibility and reasonable public
   configuration validation/freeze ->one matched full W0.
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
- D406 service peak64,123,109,376B,swap0,high/max/OOM0;host available minimum
  59,220,758,528B,disk minimum149,497,323,520B. Recheck before heavy work.
- All D402/D404/D406/D408/D409 workers/service scopes released. Loader overlay
  was restored to exact preimages after each run;empty aux scopes stopped.
  D410 has also released GPU/service/aux,restored overlay and stopped ownedremote
  units by exactInvocationID. D411 is the only live heavyjob;see identity above.
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
- D408 after audit reclaimed only600 downloaded object files at
  /tmp/tc-d408-7br/remote_artifacts/objects. Original source hashes/different
  inodes,closed ownedruntime,readable same-UID fd/cwd/root+fuser and tracked
  future config references checked;3inaccessible proc entries disclosed.
  Retained allraw/receipt/journal and remoteimmutablecache. Observed release
  1476124672B;diskafter162219474944B. Rawcache_cleanup_before/after.json.
  Re-auditing removed bytes requires rematerialization;do not redownload just
  to repeat a passed check. Recheck>=150GiB plus predictedgrowth before D409.
- D408servicepeak52653588480B,high/max/OOM/swap0;hostavailablemin59081170944B.
- Before D410, D409's downloaded objects only were reclaimed after byte/source
  hash, distinct-inode, closed-runtime and readable process/reference checks.
  Removed /tmp/tc-d409-7br/remote_artifacts/objects; retained original pool,
  remote immutable cache, rawlogs, receipts/journals and source symlink targets.
  Three inaccessible same-UID proc entries disclosed in raw cleanup receipts.
  Reclaimed725557248B; diskafter162078203904B. See D409 raw
  cache_cleanup_before.json and cache_cleanup_after.json. Historical evidence
  manifest predates this cleanup and is not rewritten. Do not redownload for
  a repeated already-passed byte audit. D410 launch rechecked growth/resource
  gates; continuous watchdog was active throughout that run.
- D410 terminal safety:servicepeak52803334144B,high/max/OOM/swap0;
  hostavailableminimum58742841344B,diskminimum160506687488B.
  Watchdog closed normally afteractualrelease. D410objects are still retained
  after completed byteaudit; reclaim only this rebuildable cache with ownership,
  hash and reference checks before nextheavylaunch. Keeprawlogs and allreceipts.
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
