# IEEE TC execution status

## CURRENT — D138 complete, analyzed and verified; backup pending

2026-10-01 13:03. Goal ACTIVE/incomplete. Current turn PROGRESS: same evidence
session4277 completed0; verification, small bundle and exact-owned cleanup done.
No live experiment or analysis handle. No serving/config change, new optimizer,
GPU replay or baseline. Once-only published cache fulfilled; NEVER rebuild.
3B AND 7B performance remain OPEN. Baselines remain PAUSED.

- SAME projection2846 completed0 at12:48:28; exact emptydomain closed12:48:45,
  memory events0. 27:47.70/RSS97536KiB. Output33831248B SHA
  8221aaf5f60fad975eb4ae92bd9b5d2a0a7cc5d9b0e07bc3e00d6389b30322eb.
  NEVER reproject original9.4GB. Monitoring cells6327/6334/6337 all finished.
- Curator8268 completed0, actual d5b76f96f6ea4c4ebc0c0216b9832d1b,
  31.41s/RSS363440KiB; failure audit0.57s/RSS97304KiB. Emptyclosed12:49:56.
  Occupancy25366 completed0, actual f9258f03291b41dba280dcdb16b0873a,
  2.87s/RSS339676KiB; emptyclosed12:50:29. Both analysis domains events0.
  Occupancy helper retained old D137 service precheck; actual cleanup targeted
  exact D138 empty domain correctly. Executed helper retained unchanged, actual
  D138 service/occupancy inactive supplement verified. Not a wrong-process stop.
- All4000 native-contract success,0fail; exact prompt/input/target/count/timing
  matches withinrun, maximum timeline/TPOT recompute error0ms. Numeric adapter
  qualification and commonwarmSLO PENDING; n_correct null/eligiblefalse.
- D118/D138 all4000 IDs,adapter,arrival,prompt/source hashes,input/target/actual
  counts and contracts match. OutputtokenID hashes:3748same/252different(6.3%).
  All differentIDs retained. Cause UNPROVEN. vLLM0.30 official reproducibility/
  batch-invariance docs identify a possible cause, NOT evidence of this cause.
  No historical env reconstructed from today's default; no new env flag set.
- 3B mean/P95 TTFT12.588626567/45.713687960s; controllerE2E17.517594720s;
  dispatch8.540798232s(67.85%meanTTFT); serviceTTFT4.047828335/native.581056899s.
  Mean/P95 TPOT38.582978680/98.346633191ms. GPU15861.330546541s.
  D118 descriptive change: TTFTmean -89.69%,P95 -82.98%; TPOTmean +28.80%,
  P95 +51.81%; GPU -0.0308%(no material resource gain). Both n1/multiplechanges/
  outputdifferences; no isolated causal estimate, CI or formal winner.
- Stage means:arrival→gate4.458793090s,gate→source4.082005143,
  source→native3.466771436,native→last4.157320471,last→controller1.352704580,
  controller→outer.975115882. Gate→outer14.033917513s is upperenvelope,not
  exactpermitrelease or GPU billing. Native meanconcurrency4.190201/peak23,
  perreplicapeak8. 1462controlsamples/921queuepositive&activebelowcapacity;
  samples do not prove continuous idle or causal source of waiting.
- Final tables/interpretation saved docs/ieee_tc/D138_FULL_W0_FULL1.md. Its
  preverification wording about pending seal is superseded here; document SHA
  pinned in verifier, do not casually edit. CuratedSHA
  b20177aa13cc677fea2c151d752d1ede10a78d01de2d258fc02303c0624d87bb;
  failureSHA a8409a86632bcfcd14d2d01247e9fe6aa7e5cfd3ae230a6c02c7c2e8f208572d;
  occupancySHA d49ae6eeb3b2c81eb00ce85100b8af99fd28f44c75bce4ec0fca7d3d7b1e5649.
- Verifier4277 completed0,2.66s/RSS28964KiB, actual
  bd7ba7f8714b446e809e4ae76e164de0. 184frozen/43curated rehashed/147protected
  PASS,9.4GB source stat+alreadysealedhash reused,95tests reused from sealed
  D137 (test/analyzer code unchanged), NOT rerun. New crossrun function fixtures
  captured separately. EvidenceSHA
  795a10b12fb56adaa8aadbbeb1aea93e6e86e53522c9d88f8db0c8e96bc43521.
  Exactemptyclosed13:01:18, memoryevents0.
- Small bundle complete67members/53168B, SHA
  fc602a8dd0bbe08534ddded90e96080254b206b496d82d31e7e3ae9816f3ad01;
  checksum/memberchecksPASS. Actualbundle78aa7951bf5544e2a814d543ba7d3db4,
  completed0 and exactemptycleanup done,events0. Initial cleanup syntax command
  used wrong cwd and stopped before execution; corrected cwd ran helper ONCE.
  No rawlarge data or model copied. Lastdisk182146555904B, recheck beforeheavy.

NEXT: explicitfiles diff/secrets/checksum→commit/push→verify remoteHEAD.
Precommit13explicitfiles/80payloads secrets and bundle-member hashes PASS;
user manifest excluded. csv.writer outputs CRLF; ordinary diff-check flagged
those line terminators only. Recheck with cr-at-eol (all other whitespace rules
retained) PASS. No sealed CSV bytes/values/hashes changed to silence the check.
Then one evidence-led question for large 3B/7B performance gap, TPOT regression
and output differences; no optimizer chosen yet. Do NOT run baseline, blindly
increase capacity/deadline, remove guards, repeat completed tests/projections or
declare 3B finished. Warm/Resident, baselinequalification, M1M2/A1–A5/S1–S13 pending.

## D138 historical wait through12:40 — superseded, not execution instructions

2026-10-01 12:40. Goal ACTIVE/incomplete. Previous turn VERIFIED WAIT;
current turn VERIFIED WAIT on SAME2846 plus bounded-curator comparison
extension and isolated fixture checks. No new serving optimizer or replay.
No subagents. Baselines PAUSED; 3B/7B performance OPEN. No serving/config change,
new optimizer or new replay. Once-only cache already fulfilled; NEVER rebuild.

- SAME D138 normal exit verified12:19:15: service inactive/tmux finished,
  launch.pass true, service/replay/watchdog returncodes0; actual native GPU
  release and service path removal true. Launch SHA
  48533d56fc48673ecd2e769e04950416c12f20fabc5fc76a40e35845c9f80bd0.
- All4000 planned/arrived/submitted/terminal and native-contract matches,0fail.
  Numerical adapter/common warm SLO still PENDING, n_correct null/eligiblefalse.
  DO NOT close 3B performance from completion or claim formal qualification.
- Physical15861.330546540965GPU-s; four leases/allreleased/noopen/noquarantine.
  Windows36.03676840296248prearrival/15708.746717182745arrival/
  18.813937864266336drain/97.73312309099128cleanup. No formal cost ranking.
- Exact remote3b/7b/monitor stopped12:19:28 after localterminal/GPUrelease;
  local aux f2dee3270e6246c1970e1510d60207c4 confirmedempty,events0,closed12:19:29.
  Do NOT repeat cleanup. No remote configuration/management during inference.
- Remotejournal transfers-34483b35f5c0466e8444ecd2c7968bb9.jsonl matches
  healthclock remote-process-monotonic:679f672f06904165afbb229e1bb32e70.
  Copied ONCE; local/remote SHA equal:
  journal a3d288495fe38932fe39c8e266440ae4da96bc4a6edc8f3d0c716dbfb1744b8a;
  monitor 773b57fa4ceff1e9c7ca5636bab9d8c57ce1653a6e32e61060dfaa6268cc8606.
  132UUIDpairs,306360162Bbothends,5139892912Blogical,allpublished/verified,packing0.
- Metadata session31657 completed0,3.26s/RSS313656KiB. Actual
  317073ed84204f1881b2f9d124f71680,3/4GiBswap0CPU2,3,26,27,events0;
  exactemptyclosed12:20:40. Whitespace-only137586951→89976825B,fixturesPASS,
  completevalues/numeric lexemes/line boundaries preserved;128MiBguard unchanged.
  PreliminarySHA eff18aba7af785d81711e81cfbebd7eddf07f94b744735d58abcd999160783c9.
  4557samples,peak37299908608B,minhost79907352576B,high/max/OOM/swap/warnings0.
- Original normal JSON9401622521B, inode109731443,
  mtime_ns1790828305989859516. NEVER whole-load or repeat projection.
- ONE unchanged D96 streaming projection launched12:20:40:
  exec session2846 LIVE; primelora-d138-full1-project-20261001.scope,
  actual InvocationID8cbc472b763f427a922a06c2022b3de0,3/4GiBswap0CPU2,3,26,27.
  Actual timePID882202/jqPID882352,Tasks2/MemoryCurrent9678848 at12:23.
  Output full_full1_request_projection.json is PARTIAL until exit0/time receipt.
  Resume SAME session; no second pass. cleanup_projection_full1.sh prepared
  with actual identity,syntaxPASS,NOTexecuted; requires emptydomain afterexit.
  At12:26:52 SAME2846/InvocationID confirmed LIVE: jq882352 CPU6m11s,
  sourcefd4position2137767936/9401622521B,MemoryCurrent23080960B,events0.
  This is verified progress of the same analysis, not a new run/result.
  Latest12:32:10 SAMEactualidentity/PID LIVE, CPU11m29s,
  sourcefd4position3948826624/9401622521B,MemoryCurrent40501248B;
  latestcheckedmemoryevents0. Read-only polling cell6327 completed allsix
  observations; it is NOT the actual analysis handle. Resume2846 only.
  No new analysis output/optimizer/config/remoteoperation/replay this turn.
  Latest12:39:26 SAMEactualidentity/PID LIVE,CPU18m45s,
  sourcefd4position6392455168/9401622521B,MemoryCurrent68669440B;
  latestcheckedmemoryevents0. Pollingcell6334 completed; notanalysis2846.
- Existing summarize_full_full1.py,collect_full1_failure_breakdown.py,
  curate_full_full1.sh,run_occupancy_audit1.sh prepared/reviewed,NOTexecuted.
  Currentcurator now additionallyreuses D137 ID-sorted historicalaudit pattern
  tocompare D118/D138 all4000offered: adapter/arrival/prompt/input/target/native
  count/outputhash/sourceitemSHA. ReadsONLYexisting34MBD118projection, verifies
  pinnedsourceandsealedref; no original9GBparse. Missingfields/duplicates fail;
  outputdifferences preserved,notwinnerfiltered. Actualcomparison NOTexecuted.
  AST-extractedfunction fixturesPASS (reorder,outputchange,targetunchanged,
  duplicate/missingIDs,missingprompt). Toolcapturenotedtruthfully in
  contract_audit_fixture_tool_capture.txt; notredirectedexperimentlog.
  New source_refs includepreviouscurated/projection andfixturecapture. Ordinary
  serving/source/config unchanged. D138doc hasprovisionalaudit-methodnote.
  Normal schema confirmed. All postrunanalysis actualdomain identities except
  metadata/projection remain unknown; bind observed ones later,not inheritedIDs.
- docs/ieee_tc/D138_FULL_W0_FULL1.md added with provisional status/GPU/remote/
  resource tables and source SHA. Final latency/timing/occupancy/comparison
  NOT yet claimed. Finish tables before evidenceverification/scopedbackup.
- Full1525linePlan/1778priorledger/312MetricV1 reread aftercompaction. Skills
  monitor-experiment/analyze-results/academic-plotting/github-sync read fully.
  Feishu absent/W&B not enabled. Plan/V1 unchanged. User files untouched.
  Superseded1778lineledger archived verbatim, SHA verified equal to original;
  activeledger condensed254lines beforethisnote. No historical evidence lost.

NEXT: SAME2846→actualterminalreceipt→exactemptycleanup→existingcuration/failure→
exactcuratoremptycleanup→occupancyaudit→fulltables/interpretation→verification/
bundle/scopedbackup. No newoptimizer/GPUrun/baseline beforeclosure.
Then evidence-led 3B/7B performance gap work; no blind capacity/deadline increase,
guard removal, repeated completed probes, old large projections or winnerfilter.
Warm/Resident,baselinequalification,M1M2/A1–A5/S1–S13 all remain outstanding.

## D138 execution identity (not a new launch instruction)

Raw results/ieee_tc/p2_backend_qualification/d138_20261001.
Started ONCE11:02:08 tmux tc-d138-3b-full1, NORMAL FINISHED12:19; neverrestart.
RuntimeHEAD2608d027d82df5a79719153e114ff06a7a70679b, production identical
233ccc29 (docs/evidence-only commit). Multi-change requalification since
D118runtime48f808ab; not isolated causal ablation or formal comparison.
Same D1183B config except3freshownedpaths; Full4000/source42/W0/formal0,
sameD89profiles; cap8/seq8/slots8/cpu32/batch4096/gpu.72/context1024/FP16TP1,
no prefix/chunked prefill/profiler,deadline1800,source weights/traceunchanged.
Trace sourceSHA4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d;
viewSHA0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37;
subsetSHA6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b.
Preflight184refs/147protected PASS,receipt1368a6f5c216d96fd21c32abe36e28fe8ba06f8596f8a25ce0c1aea7b7b1a0d2.
BothNIC1000/full,nochanges. Allprelaunch/healthscopes closed beforelaunch.
Actualserviceunitprimelora-tc-svc-1f05557629544bcb813b18787c26af99.scope,
invocation66ecce9242f84e359eefc18dd39a9a1a,72/80GiBswap2CPU4–23,28–47.
Actualauxprimelora-tc-aux-4590b9aa7523492da77c0f2ed966ea6f.scope,
invocationf2dee3270e6246c1970e1510d60207c4,3/4GiBswap0CPU2,3,26,27.
Allfour leases endedrelease/worker_returncode0:
GPU0/5fd74446d375418491065591a0ca22e9 at335016.171668082;
GPU1/3e46c5bca3dd49de91649f73837194a8 at335025.572814232;
GPU2/01d8bdd6b1054062bb7004256bb258fa at335035.353050915;
GPU3/bbf99349c8254c3c894fd343714e2e44 at335044.559847262.
Exactremote3bPID1582707/c396658a91dd498c90ff3211f4aa76e4;
7bPID1582709/84734126467248f7ba4bada8ce80341e;
monitorPID1582712/a75037f098e64b448272836cfea7338f,
unitprimelora-artifact-monitor-d138full1.service,allstopped.
Remote monitor /home/lab14/primelora_remote/tc/d138_20261001/remote_monitor_3b_full_full1.log.
Diskfree rose duringrun,NOagentcleanup/compression; causeunestablished,
do not infer contamination/performance causality fromfree-spacechangealone.

## Completed checkpoints — DO NOT REPEAT

D137 audited and backed2608d027d82df5a79719153e114ff06a7a70679b,
exactremoteHEADverified10:56:24. All14explicitfiles/bundle/secretschecksPASS,
userfilesneverstaged. Productionunchanged233ccc29.
4000native/0fail,4leasesreleased/noquarantine,GPU20319.723388s;
mean/P95TTFT603.644819/1141.327139s,TPOT39.014920ms.
Dispatch601.277516s=99.61%meanTTFT,service2.367304/native.317747s.
Timing/native/prompt/targetPASS0ms; numericadapter/commonSLOpending.
D135conditionalmean/P95 +12.87/+12.26%,GPU+1.63%;n1/differentpopulations,
NOTacceptedFullperformancegain or causalCI. FIFOcorrectnessfixretained.
D137curatedSHA9d6e028dd9caa49283510f4db585e79170067addc70345f0397f95f27b5bb04d;
occupancySHA dd641909a930d4cc90e08e543c86e157630257ffd73d6140084e5b34c5b76311.
95testsPASS; evidenceattempt1 onlywrong expectedpopulationlabel;
attempt2reusedtests/correctlabelPASS. EvidenceSHA
18b496f98daf9aa48aab8d98d68b75507df33798bd8b29b470e75d3a98da4c17.
72member56033BbundleSHA15b529dd8653e42b72911392e9e9a2fe46fec7d1be30b1f4ff7934a4cc2128ee.
DocD137_FULL_W0_FULL1.md and LEGACY_CURRENT_PERFORMANCE_GATE_20261001.md
SHApinnedinverification; doNOTeditcasually.
OriginalD13715.2GB and completed34MBprojection mustNOTbere-read/reprojected.

Historical3Baudit backed in samecheckpoint; independent verification12000rows
PASSED, no needrepeat. Curated20261001_legacy_3b_contract_audit.json
SHA39d086ee40d7d8c182f1c6c6f6bb95d8dec162897e8435544b8b3387895db272.
Oldlocal/oldremote/D118 input2981921/2981921/2594938;
output447515/447447/458224;IDs/adapters/arrivalmapsidenticalafter sorting.
OldpromptHash0,new4000; native provenance missingold. D118nativeinputequals
content+1specialtoken;maxcontent759. R2notcausal,countsnot100xmorework but
queue-amplificationpossible. OldTTFTalreadyincludesplannedarrivalqueue.
Oldlocal/oldremote/D118 meanTTFT.8813136/1.0872257/122.146517856s,
P952.21323/3.7441838/268.605684342s; D118dispatch114.1778663/native.475647s.
Oldmax2vsD118max4;olddiscountcostnotnewphysicalU. DoNOTblameallgap onremote.
D118all4000native,15866.211830GPU-s;oldruntime48f808notcurrentcandidate.
D136FIFO RED→GREEN→1068testsPASS,backed233ccc29; don'trepeattests/probe.
D1353999native/1Timeoutreq03061,19992.947476GPU-s,backed220f430.
D134selectedplanbindingbackedc164f87;D133failedHOST retained/backedf5e10db.
D132compactvalidatedfreshobservationsbacked30b0715;D1313749native/251fail.
D130sealedplanningbackedd0384e4;D1293415native/585fail.
Earlier archive recordsallfullaudits/failedattempts/sourceevidence intact.
No newoptimizerselected until D138 finalclosure.

## Ledger archive

Full preceding1778-line ledger preserved VERBATIM in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D138_LAUNCH_20261001.md,
SHA871e92cda031f82da6e30917d2d895177dde30a01d24b7b2e03ad5ae55c1c29e.
Its historical LIVE/NEXT entries are superseded by CURRENT; never restart them.
Earlier throughD129 archive remains unchanged:
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D129_20261001.md,
SHAf2a51eb3106b217bc285f12451694c1b33318fde5e62354ec02fc0a81b91ebe2.
D128runtime75deef6,D125result61c9bcf,D126/D127e1f7f6e alreadybacked.

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
- At12:20 allserving/remote/auxdomains closed; ONLY D138 projection below LIVE.
  Lastdisk182296768512B afterclosure; recheckbeforeheavy,not a futureguarantee.
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
