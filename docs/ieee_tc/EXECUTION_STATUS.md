# IEEE TC execution status

## CURRENT — D123 diagnostic complete; backup next, no live job

2026-09-30 17:39. Goal ACTIVE/incomplete. Previous goal turn VERIFIEDWAIT;
current turn PROGRESS: D123 finished, exact cleanup, CPU/request analysis,
failure/timing audit, final tables, 71 tests, source/protected checks and small
evidence bundle done. NO GPU/replay/remote/analysis job remains. Baselines PAUSED.
No serving optimizer/configuration change accepted this run. Do NOT repeat
D123 replay/profile/projection/CPU/metadata/curation/copy/tests/bundle.

- Original first1000 W0 diagnostic view, source4000, all500 adapters accessible.
  Same D119 configuration/D89 profiles, only owned paths and diagnostic count.
  Runtime HEAD d4825f64ae51096dd619cd9b26dc09b166c8dc7e, already backed D122.
  1000 planned/submitted/started/terminal;916 native-contract success,84Timeout.
  Formal/numeric-adapter/Full/SLO qualification NOTpassed; n_correct=null.
- Conditional916 mean/P95TTFT700.900821/1692.545729s;
  controllerE2E707.796039/1701.858600s (NOTclientreceipt).
  Dispatchmean694.305124s=window684.191716+slot8.330000+release1.783407.
  service6.595697s=prenative6.330967+native.264730; decode3.657414s;
  worker→controller3.212812s. TPOTmean31.420830ms/P9550.771127ms.
  Native prompt/token/hash/tier checks916PASS; GPU333/HOST317/NVMe211/Remote55,
  conflicts0. Success subset only, NOTall-dispatch/A4/numericadapterproof.
- TIMING CHECK FAILS:3 requests(req00063/00330/00539) each have same TTFT,
  controllerE2E,dispatch absolute-boundary errors2.792140/3.205084/3.948905ms.
  Original1ms unchanged. service/dispatch decomposition andTPOTerrors0.
  Firstcurator stopped atfirstviolation, script/log/time retained. Attempt2
  ONLY recordsallviolations with pass_tolerance=false, no data/threshold change.
  Source: run_one globaladmission G, then _exec_request_in_reservation slot
  start S, selectedadmission T. rawdispatch=(G-A)+(T-S), missingS-G. Source
  reading supports timing-boundary gap, not hundreds-second queue explanation.
  No timing-production fix or causal optimizer implemented. Do not rerun just
  to get an error below1ms; reconcile authoritative absolute boundaries.
- All84timeouts lack generation-submission/source-admission records; default
  zero stages/emptytier/native dictionary doNOTprove nodispatch/zerowait.
  Firsttimeoutreq00369offset2249.536751s. No returned ownership-unresolved error.
  2quarantines released separately; no aggregate timeoutcause proven.
- Physical11695.308335912065GPU-s,5leasesallreleased,2quarantinesreleased;
  4ready+1cancelledactivation,firstfour1initial+3natural. HOST/NVMeownedroots
  removed.3247samples,peak20695658496B,minhost92699725824B,
  high/max/OOM/swap/warnings0. Main/service/replay/watchdogexit0,launchpasstrue.
  Execution/cleanup success is NOT correct workload/performance qualification.
- Remote60UUID/content/bytepairs74953014Bbothends,allpublished,packing0.
  Services/monitor exactstopped17:22:16,localemptyauxsame time.
  Correct journaltransfers-73cd7890ada447aaac2204aec8281de0.jsonl copiedONCE,
  remote_7b_prefix1_transfers.jsonl SHA
  a51c18941e6c2479fead2d76dd5bead30c5d7167f4913ba490df7e8047f50502.
  Monitorremote_monitor_prefix1_final.log SHA
  d7b111494a2aad73a341f87a60abdb68db68fcd5ff559f63e7fcaaf4c9704dd1.
  BothlocalSHA==remoteSHA; doNOTrecopy/rebuild once-only published cache.
- CPUprofile279902GILsamples,67errors/67stackwarnings; parent-only100Hz
  py-spy0.4.2,overheadunmeasured. Exclusive nearestproject(non-save):
  owned_preparation_inputs59998(21.44%),filepreparation54296(19.40%),
  sendRPC37606(13.44%),footprints31492(11.25%). Denominator ALL samples,
  not walltime/phase shares. saveancestry24896/import1811/other253195.
  Source615ff/745ff ownedinputs deepcopies+hashesfullview;
  runner17429ff fileexecutor deepcopiesplan. Hypothesis, NOTcausal proof.
- Original2886956618B SHA
  70c3a0fe24dd3edf1108183090ba5dcc8d762ca18ebd014b733b9b1c7848be70.
  Projection8171319B completed354.24s/RSS35328KiB/exit0; don't repeat/read
  original whole. Normalmainoutcome15077467B; profile31053756B.
  CPU12.40s/RSS301600KiB,metadata.50s/51900KiB,curator2 9.96s/89300KiB,
  failurehelper.15s/34320KiB; all finalanalysesexit0.
- Curated paper_results/ieee_tc/p2_backend/20260930_d123_7b_full_w0_prefix1.json
  SHA93614036c20e9b978fe5e98c99163f77235ee1e7a7770110e36bcea63c271b65;
  failurebreakdownSHA9af5c025465fada4704961c834d7ae74bd0e2c2b26e66ba40f5880b59d2e1fa6.
  Doc D123_7B_PREFIX_CONTROLLER_PROFILE.md contains full tables/caveats.
- Evidence71checksPASS2.453s;122frozenrefs,47curatedrefs,147protected verified.
  VerificationSHA280e9b565d5ca439e61b6d0bd6946db64cbce433d13ef26e862126ecdbcb76d3.
  All CPU/metadata/projection/curator1/curator2/evidence scopes exactemptyclosed;
  finald5457d6c24df4449ab7fbf47882b1725 closed17:37:59,events0.
  Bundle50029B/75memberseachSHAverified,
  SHAf0a2f617d7c33a00fe9a6b519fd9eec1965c3f4c28bad72f4e2c79b79ca84bc5.
  Initial bundlechecksum callwrongcwd failed; correctedcheckPASS, NOrepackage.
  Git diffPASS; stagedsecrets check/commit/push next. Neverstage usermanifest.

NEXT: finish scoped Git backup. Then reconcile small timing boundary using
existing records/source, and ONE bounded causal test of actual preparation
copy/validation path before selecting an optimizer. Use current primary-source
literature/code and preserve all owner/content/epoch/budget checks. No blind
unchanged Full, raised deadline, blanket threads/capacity or guard removal.
Return ordinary Full validation after candidate; then warm/Resident→baselines
→M1/M2→A1–A5/S1–S13. Whole goal incomplete. No subagents.

## Authority and frozen protocols

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before task/experiment and after compaction. No subagents authorized.
- Plan1525lines SHAfe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
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
- Latest17:38 allprimelorascopesinactive/GPUempty; lastdisk252719472640B.
  Recheckrealresourcesbeforeheavy,not a futureguarantee.
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
