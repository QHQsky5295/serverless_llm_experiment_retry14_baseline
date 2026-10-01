# IEEE TC execution status

## CURRENT — D152 finished, analyzed and sealed; backup pending

2026-10-02 02:36 +08. Goal ACTIVE/incomplete; this turn PROGRESS. No live
exec/session/GPU/remote/analysis job. Runtime remains backed
26fa0a086bf91895e667a1996662971a526c860b. No new optimizer or nextGPU prepared.
External baselines PAUSED. Stay on 7B until actual per-model G1/G2 acceptance.

D152 ordinary Full4000/source42/W0 ran01:11–02:20, sameD150configuration/D89profiles,
no profiler/prefix. OnlyD151 fresh readonly native routing observation candidate.
Native-contract4000/0fail; numericaladapter/commonSLO unqualified, n_correct=null.
All four GPU allocations released, U=15994.237882167974GPU-s. Mean/P95TTFT
97.961747894/210.630781154s; TPOTmean/P95 36.850619034/59.529330164ms.
VsD150 meanTTFT-66.980%,P95-61.929%,GPU-10.937%; RPCpickup+13.302% retained.
Still grossly too slow, NOT model acceptance, formal superiority or G1/G2 proof.
All15 non-output-hash contract fields match4000;109outputhashdifferences OPEN.
Dispatchwait96.493725659s=98.501%TTFT; service1.468022235/native.320694262s.
Phases: arrival→gate95.252677690, gate→source1.241047969,
source→dispatch1.147327973, native4.345848121,
last→controller.575936233, controller→terminal.293962738s.
Gate→terminal7.604123034s is UPPER ENVELOPE, not precise permit release.
Mean native/gate-envelope concurrency4.361448/7.631419; max8.
Controls1745allinwindow;1624withqueue/778belowreadycapacity.
Planning1993completed. Remote132pairs/131480060B/allpublished/packing0.
Resource4062samples:peak20379181056B,minhost93121208320B,swap/events0/no warnings.
Firstepochs initial1/natural3;0quarantine. Timingerrors0ms;pre-dispatch4000/conflicts0.

Cleanup COMPLETE:
- Localservice/aux automatically removed, PID/GPU/cgroupabsence verified02:22:41.
  No manual retained-ID cleanup on removedscopes. Exactremote3B/7B/monitor
  stopped02:21:23; journal+monitor copied AFTERstop/sourceSHA matched.
- metadata64105 CLOSED0/4.05s/RSS435640KiB,
  actual6cbb7e2178634059b4a0298ef5376ce9 absenceverified.
- projection8221 CLOSED0/78.68s/RSS99840KiB,
  actual8bbc6bc7aa2e41edbd158a9af114026b absenceverified02:29:53.
  Source271163137B streamedONCE. Never reproject/reparse oldgiantoriginals.
- curation8324 CLOSED0/7.63s/RSS454460KiB,
  actualf3eba0c4d6024653b403485facc5bab1 absenceverified02:30:32; failuretable0.
- occupancy43787 CLOSED0/3.08s/RSS447068KiB,
  actual5df8b477357a4b13ab2010694c96e76b absenceverified02:31:26;
  events0 atanalysisend in transcript. Otherfinalcounters unavailable after
  automaticremoval, not reconstructed.
- verification10106 CLOSED0, actual03695fbd493a4834b5b42f3509fe95ae,
  absenceverified02:34:21;255frozenrefs/47curatedrefs/147protected/oldPlan/V1PASS.
  SHA826be525804888ff7d9c1568f761a7c2f1625afb2eb2e36945f21ae77e42daa5.
- bundle returned0, actual27aef7b84c0743048639763dec2977db nowinactive/emptyID,
  absence verified02:37:50 bycleanup_bundle_full1.sh/log.78members68175B,
  SHA559712dfb6855b0e3815ce02c6ca1a85cd4ed3b8caee833cf057f6ef0daca3d6.
  No largecompression or repeatedpublication. Metadatacompact126001682B
  fitsunchanged128MiBguard. AllCPUanalysis3/4GiBswap0CPU2,3,26,27.

Final tables/interpretation in SEALED docs/ieee_tc/D152_FULL_W0_FULL1.md.
CuratedSHA6a1cd918beb5df1331b7a41725af7531335e25321fe1fce65e7ada7bdc21b5e7.
ProjectionSHA6d3bcc3e57b9babedd001aa93437e1e357176f9af010bf6b6291064d29be75d0.
Outputs paper_results/ieee_tc/p2_backend/20261002_d152*.
Raw results/ieee_tc/p2_backend_qualification/d152_20261002.
Do NOT repeat completed candidate tests, projection, curation, seal or cachecreation.
Only10 publicdownloadarchives+10headers removedprelaunch by exactauditedallowlist;
allocatedfreed2151706624B; installedenvs/uniqueassets/rawresults untouched.
Latestfree162404921344B/MemAvailable110045548KiB; recheckbeforefutureheavy.

Afterclosure, authoritativePlan amended with latestper-model/NewG1G2Prime-only
directives; original1525lines retained byte-for-byte after insertedsection,
oldPlan snapshot stillfe6c05b..., V1unchanged. NewPlan/fullmirror1537lines
SHA0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7.
No oldrunidentity rewritten. Planchange is sequencing/acceptanceclarification,
not metricdefinition change. FullPlan/status/V1/skills read this continuation.

NEXT explicit newdocs/curated/archive/Planbackup,
neverstage usermanifest. Then source/history/primaryreference diagnosis of
remaining non-generation waiting; no blindcap/deadline increase or rerun.
Need newmetric oldPrime gap audit and legitimate commonreference calibration;
don't resume externalbaseline campaign. Bothmodelperformance/numericadapter/
warm/Resident/M1M2/A1-A5/S1-S13 remain OPEN.

## Completed history — do not repeat

Full preceding ledger preserved VERBATIM in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D152_ANALYSIS_20261002.md,
SHAd18c53f9e63b75a240dc09aa67badf8f501efc827b11dc56a8f4d48296d4aaa2.
OldLIVE/NEXT entries are historical, superseded by CURRENT. Other archives
referenced there remain intact. D150/D151 results and qualification are complete.

D151 source/test qualification backed26fa0a0;737PASS. Only fresh native routing
observation omits unused staging/allocator output; ownerrefresh/graph/device
identity/clock stilllive, fullphysical/planning source_snapshot unchanged.
D150 sealed/backed805bf611:4000native/0fail,U17958.297896,
mean/P95TTFT296.677792/553.254599s; TPOT41.273491/69.890503ms.
Reuse D15034MBprojectionSHA768e25abbda38e3acea540806945d6fb5b68b3161b233a81adab3706c7509012
and curatedSHAee01b56e9f44d8eced77528e99b488e3efb1515d7edba0d0fd9e5a3c729f72d2.
NEVERreanalyzeD137/D145/D14815GBoriginals or repeatedonce-onlyremote cache.
D145fourRPCtimeoutcauses still unresolved; D146preservedfailurecontextwithout
changingtimeouts/retries; laterzero failures don'tretroactivelyclosecause.

Legacy3Baudit12000rows independentlyverified/backed2608d027; oldlocal/remote
meanTTFT.8813136/1.0872257s vsD118122.1465s, currentD13812.588627s/P9545.713688s,
TPOT38.58/98.35ms. This improvement is NOT per-modelacceptance.
Oldnative/prompt/physicalallocation commoncontracts incomplete; R2 notcausal.
NewmetricoldPrimeaudit cannot replace missingfields with legacyCE ordiscount.
Legacygate doc LEGACY_CURRENT_PERFORMANCE_GATE_20261001.md sealed; do notedit.

## USER CLARIFICATION — 2026-10-02 02:13 +08, NEW metrics for old/new Prime

Current work is ONLY PrimeLoRA audit/optimization; external baseline experiments
and tuning stay PAUSED until Prime reaches the per-model goal. The comparison
target is previous PrimeLoRA evaluated with THIS campaign's frozen NEW core
metrics, not merely old TTFT/CE or superiority over the previous slow candidate.
G1=all correct+common joint SLO then lifecycle physical GPU-s/request;
G2=common resource budget+SLO then tail latency and SLO attainment/noninferiority.
The user still explicitly rejects tens-of-seconds mean TTFT as a completed
optimization, but this is not permission to replace G1/G2 with TTFT alone.

Reuse old measurements only where new-metric fields/contracts can genuinely be
recovered. Do not invent native correctness, GPU allocation/release, common
warm thresholds, or budgets from legacy CE/discount billing. If those are
missing, identify the exact gap and perform only necessary matched Prime-old/
Prime-new evidence collection, not an external baseline campaign. Common
measurement calibration must remain distinct from external system comparison.
This supersedes any interpretation of the earlier update as an old-TTFT-only
acceptance test. Applied immediately and synchronized in the 2026-10-02 Plan amendment after
D152 provenance closure. Frozen V1 itself is not changed.

## USER UPDATE — 2026-10-02 02:02 +08, per-model performance acceptance

Latest goal explicitly rejects mean TTFT in the tens-of-seconds range as an
acceptable finished model. Each base model must reach the actual performance
goal before moving to the next; mere improvement over a very slow development
candidate, passing token counts, or finishing a replay is NOT model acceptance.
Compare with canonical legacy results and explain the remaining gap using
input/generation/resource contracts and measured stage evidence; differences in
contracts must not become an excuse for closing an obviously poor result.

D152 7B is already in flight: finish this unchanged run, then continue focused
7B optimization/validation until accepted before advancing to 3B or baselines.
The unresolved 3B latency/TPOT and legacy gap remain mandatory, not removed.
Retain nine equations, source-based falsifiable optimization, frozen V1 G1/G2,
fair comparisons and all failures. Do not invent a new 10-second SLO from the
phrase "十几秒", relax common requirements, or select favorable runs.
This directive is effective immediately. Synchronized into the authoritative Plan AFTER D152 closure/provenance checks;
original Plan identity is preserved. Frozen V1 was not edited.

## Authority and frozen protocols

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before task/experiment and after compaction. No subagents authorized.
- Plan1537lines SHA0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7.
  Full mirror: docs/ieee_tc/PLAN_APPROVED_20261002_PRIME_FIRST.md.
  Oldfe6c05b... retained in PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md;
  do not rewrite D152/earlier run provenance to new Plan identity.
  Execution authorization supersedes historical plan-mode/noexecution wording.
- Before comparison/configuration selection/comparative figure read FULL
  docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md,312lines,
  SHA5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22.
  G1=allcorrect+jointSLO95% thenminimumphysicalGPU-s; G2=commonbudget+SLO,
  thenP95TTFT. CE supplementary. Warm/reference numerical thresholds NOTfrozen;
  development5000ms/liveCE are NOTformal SLO qualification.
- Preserve nineIEEE equations/core. Each optimizer: historical evidence +
  current primary literature/code + falsifiable hypothesis + minimal causal
  validation + ordinaryFull. One candidate; no future knowledge/test-point
  tuning/winner filtering, newweight/trace generation or13B.
- Eachrun cleanup→validation→table/figure→interpretation→next. Academic plotting:
  plansection11,IEEE singlecolumn3.45in,TimesNewRoman,nooverlap,below bold(a).
  Failure/status tables appropriate; no n1CI/rankingfabrication.
- Chinese updates≤60s, paper/evidence terms, completed/current/outstanding.
  Backup testedcheckpoints to faaslora_origin/retry14_continuous_queue_v2.
  Noforcepush/credentials/rawlargeGit. Baseline changes inownrepo.

## Current mainline and reusable evidence

- BaselinesPAUSEDHEAD9e2cf28903ed11bc8ee891dd4cc9636b94307573. D74Serverless3B
  repaired complete/analyzed/plotted/backed; preparedoriginal NEVERrun.
  ResumeafterPrimeFull:Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe.
  DisplayServerless, not-new. AllformalM1M2,ablations,sensitivities outstanding.
- Numericadapter distinction pending:3B500/500zero,7B498/500zero,2/4distinct
  weightSHA. Nativecount≠numeric proof; no authoritynewweights,don'trepeatedlyask.
- D122 diagnostic-prefix contract optionalFAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT:
  explicitindexview/sourceSHA/count/defaultFullguard; formalprefix forbidden.
  521finaltestsPASS,backedd4825f6. Not a servingoptimizer; doNOTrepeat tests.
- D121 actualsavedstateCPUcopy/JSON/SHA component only,backed716e6a5; D120
  D119queue/conditionaloccupancy,backeda7d8673. Neitherproveswholequeuecause.
- D1197B4000:2890native-success,1086Timeout+24returnedownershiperrors;
  22866.332606GPU-s,61leases/57quarantinesreleased. Firsttimeout precedes
  ownershipfailure; earlierbacklog cannot be attributedsolelytolaterquarantine.
  Original9.48GB NEVERwholeload/reprojectagain. Boundedprojection29MB reusable.
  FinaldocD119_FULL_W0_FULL1.md,backed4ee00af.
- D1183B4000:all4000nativecontractsuccess,0failure,15866.211830GPU-s,
 4leasesreleased/0quarantines; largewaiting,notnumericadapter/SLOqualified.
  Original9.18GB neverwholeload. Backedb4c194b.
- D11748f808 avoidsIEEE legacyterminalscale-down; settlespreparationownersbefore
  retire. D1160ef466c non-targetmembership fix. D1142697758 actualasyncTCP avoids
  shared32-executor starvation. D1115442e62 parses samein-flight snapshotonce.
  Alreadyinruntime; no repeatedobsoleteprobes or safetyguardremoval.
- D115 all4000native butshutdownfailure; ownedHOST/NVMe roots remain RETAINED:
  /dev/shm/tc-d115-3b-full10/tc_ieee_full and d115_20260929/3b_nvme_full10/tc_ieee_full.
  DoNOTdelete or claimworkspacecleanup. Historicalquarantine/error evidence kept.
- D88profiles3B368/368,7B92/92;D89initializers
  paper_results/ieee_tc/p2_backend/d89_{3b,7b}_initialization/manifest.json.
  Use requested_model_config PARENT,notresolvedchild; no reprofile.
- cap2 historyApr13/Apr22maxloras8 KV differsfromcurrent7Bmaxloras4/.30;
  no cap4qualification. DoNOTblindcapacitytune orraise1800squalificationdeadline.
- Prior warmerreference interface audit WARM_REFERENCE_REUSE_AUDIT_20260930.md.
  D88/D89 profiles are not commonwarmSLO samples; don'tinflatewarmreference with
  Primeprotected-source/controlwall. Reference protocol remains pending.

## Environment, remote and safety

- Repo /home/qhq/serverless_llm_experiment_retry14_baseline,
  branchretry14_continuous_queue_v2,remotefaaslora_origin. Baselineownrepo/main.
  results symlink=/home/qhq/serverless_llm_experiment/results.
- Nativeenv /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM.30/torch2.13/CUDA13/Python3.12.12/SM86.
  CPUenv /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OSguards/usr/bin/python3 (modelenvlacks pidfd_send_signal).
  py-spy/home/qhq/anaconda3/envs/splitwise_official_20260615/bin/py-spy.
  Noinstall/driver/globalptrace change.
- 3Bcap8/slots8/cpu32/gpu.72;7Bcap2/slots4/cpu24/gpu.70;
  HOST16GiB/native2/NVMe16,W5/movement3,scale2s/beta.5,min1max4.
  3Bservicebin28.114717726756954ms,7B27.4336ms. Developmentonly.
- Remotealiasprimelora-artifact-174 lab14@192.168.4.174:8122,strictBatchMode,
  key~/.ssh/primelora_artifact_174_ed25519_20260925(0600);
  token~/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
  FingerprintSHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
  Artifactonly, noGPUinference. HTTP3B18080/7B18081 direct,noSSHdatatunnel.
- Immutablecache/home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b},
  both500/500,total1581215261B. Once-onlycreationfulfilledD78/D80; NEVERrebuild.
  Localeno1np0/remoteeno1 1000/full verifiedbeforeD123; historical100Mbps
  notcurrent. NoagentNICchanges. Actualtransfers/competition retained,packing0.
- Service72/80GiB/swap2,CPUs4–23,28–47;aux/CPUanalysis3/4GiB/swap0,
  CPUs2,3,26,27. ONEheavyjob; limitactualworkersBEFORElaunch.
  No global kill/ray-stop/reset/reboot. Only exact-owned identity cleanup.
  No remoteconfig/restart/hash/cleanup during inference.
  Inference disk150/100GiB unchanged;artifactnodeindependentincrementalrule.
- Contentindices inputs/20260927_3b_remote_content_index.json
  SHAbd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  inputs/20260927_7b_materialized_content_index.json
  SHAe85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  Do not use misnamed7b_remote_content_index diagnostic4998filelist.

## Protection and archives

- Seal147protected:paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify via scripts.ieee_tc_preflight.verify_seal; oldfinal_v2/figs/paperunchanged.
- NEVERstage configs/generated/lora_manifest_1000.json,unrelatedAAAIarchive,
  oldfigures,regenerate_motivation_figs.py,rejectedpreviews. Explicitpaths only.
- Prior1439-line ledger archived VERBATIM in
  docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D123_20260930.md,
  SHAdf8bc100fe6eaa4bb00ccbc857b7a469ecaca5e56e4ce28f6f76b883cd692c70.
  Historical LIVE/NEXT handles are superseded, not commands to execute.
- Earlier THROUGH_D110,D91_D100,D81_D91,D78_D80,D67_D77,THROUGH_D26,D27_D52,
  D53_D66 archives remain intact. Currentstate above has priority.
