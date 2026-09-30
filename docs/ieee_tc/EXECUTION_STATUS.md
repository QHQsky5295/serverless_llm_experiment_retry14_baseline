# IEEE TC execution status

## CURRENT — D126/D127 complete; return to one control-path architecture question

2026-09-30 21:15. Goal ACTIVE/incomplete. Currentturn PROGRESS, no subagents.
No GPU/remote/analysis/test jobs live. BaselinesPAUSED. ServingHEAD61c9bcf unchanged
(runtimee7d4c91); no codec/capacity/deadline/thread/guard changes. Once-onlycache
fulfilledD78/D80; NEVERrebuild. D125/D126/D127 completed work MUSTNOTberepeated.

- D126 final tables/sources verified:6inputSHA,all4000IDs,additivephase checks,
  metricV1/planSHA,147protectedPASS. Basic288testsPASS22.535s;wholeverify32.07s/
  RSS1166252KiB;actual3/4GiBswap0CPU2,3,26,27,events0.
  Exact6750e92e139342688d71f98f9178a66eemptyclosed beforeD127.
- D127 reusedD123profile31MB toattributeRPC:37014samplesraw_decodeunder5266,
  554atline5266itself,within37606sendRPC/all279902GIL. HistoricalNOTD125wall.
  Same479167B savedD887Bresponse(2adapters/512alloc),msgspec0.21.1:
  3alternatingrounds×10each,stdlibmean5.515074ms→msgspec2.583254ms;
  canonicaldecodedvaluesSHAidentical,largeint/double/UnicodePASS.
  NonfiniteJSON/isolatedsurrogatebehavior differs;no silentfallback proposed.
- Firstprobe rejected3B253310434Bcontainer by24MBinputguardBEFOREload;
  exit1/3.77s/RSS345216KiBretained. Probe2 measuredONLY7B20.34MBsource;
  3Bexplicitnotmeasured,nohugeprojection/raiseRAM.4.32s/RSS345216KiBexit0.
  Exactprobe1 5c546dd1a42a4346ade5b736cf06cb48 andprobe2
  6d0073d8b6a545df9b500fa4a8c632e4emptyclosed,events0.
- Decision:componenteffectsupported,NOTselectedasstandaloneFulloptimizer;
  doesn'testablishmulti-secondwaitcause. No onlinechange/Fullretry. Archive
  thismicro;donotrepeatit. DocsD126_CONTROL_OCCUPANCY.md and
  D127_RPC_DECODE_DIAGNOSTIC.md contain tables/boundaries/primarysources.
- CuratedD1264files+verification;D127summary/samples. Finalcurationchecked
  allsourceSHA andD126references;29member12033Bsmallbundle
  20260930_d126_d127_analysis_sources.tar.gz SHA
  89e609ce74fdcf454356dbdb252b98b0dcec6abdcb1b2ef79338a955c1b90dd4.
  Membersverified;checksumPASS(correctcwdafteronewrongcwdread,notrepackage).
  Finalizeexit0 exactc95ba81f5e7942ff8b33be4c5f8d6234emptyclosed;noheavyjob.
  Finalize.28s/RSS48000KiB,events0.13stagedfiles/41payloadsecretchecksPASS.
  CSVretainsanalyzer'sCRLF(likeD120);defaultdiffcheckreportedonlyCRaswhitespace.
  Recheckwithper-commandcr-at-eol(no repoconfigchange),preservesverifiedSHAs.
  Scopedbackupnext;NEVERstageusergeneratedmanifest.

NEXT: inspect one architecture question using CURRENTcode+existingCPUevidence:
which frozen-snapshot preparation computations can be separated from request
event-loop advancement without changing demand/cost epoch,selectedset,live
physicalrechecks,cancellation/ownershiprelease. No blanketthreads,staleTTL,
capacity/deadlineincrease,moretinydecoderprobesoridenticalFull. ChooseONE
falsifiablecandidate, boundedverification thenordinaryFull ifsupported.
PrimeFull→warm/Resident→Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe
→M1/M2→A1–A5/S1–S13 remainpending. No formalSLO/numericadapter qualification.

## Previous checkpoint — D126 offline occupancy complete; no new optimizer accepted

2026-09-30. Goal ACTIVE/incomplete. Current turn PROGRESS, no subagents.
No GPU/remote/analysis job live. Baselines PAUSED. D125 backup COMPLETE at
61c9bcfff368916f200744f4300d234efe51be1c; do NOT repeat backup/replay/projection.
D78/D80 once-only published delivery cache fulfilled; NEVER rebuild.

- D126 reuses D125 31.7MB projection + normal outcome + deployment/terminal/
  watchdog via UNMODIFIED analyze_control_path_overhead.py --native-timeline
  --allow-failed. Analysis3472 exit0,2.60s/RSS259048KiB; actual3/4GiBswap0,
  CPU2,3,26,27; events0. Exact604c6eaf67584cbb83270b5c16c35003emptyclosed20:54:38.
- All4000IDs retained;3456native-success/544failed. Successconditional phase
  means: arrival→gate877.394043s,gate→source2.761062,source→native2.473673,
  native→last3.876349,last→controller1.629483,controller→terminal.697782.
  Gate→terminal11.438349s isupperenvelope(actualreleaseearlier),NOTGPUbilling.
  Observation5763.144352s;5687resourcesamplesutilsamplemean35.853719%,notcausal.
- Firstfailureterminal3366.880645s,quarantine4490.755947s. Beforefailure
  1160controlsamplesmeanqueue575.069828;earlybacklogcannotbeexplainedsolelyby
  laterquarantine. activebinding≠gate≠nativeoccupancy;failureworkmaypersist.
- Curated20260930_d126_control_occupancy/{summary.json,request_occupancy.csv,
  sampled_occupancy.csv,control_observations.csv}; summarySHA
  e366042d59a8f55be335792641aac636e568bbed9a4f2300467c441cf320f21f.
  DocD126_CONTROL_OCCUPANCY.md hasfullconditional/phase/failuretables+caveats.
- Read fullplan/status/metricV1 + analyze-results/run-experiment/academic-plotting.
  Sourcechecks: parent_rpcjsonloads syncinloop; workerjsonencoding; native
  HOSTallocation/alias validation; preparation copies/selector. PrimaryPython
  asyncio + officialvLLMperf/0.30serialization read. No codec/thread/process/
  stale-cache/capacity/deadline/guard change implemented or accepted.

NEXT: verify D126 sources/protected state, scoped checkpoint. Singlecausal
controlCPUcandidate only after existingD123exactcallstack attribution; then
bounded same-input validation and ordinaryFull ifsupported. No identicalFull
retry or more oldlargeprojection. Warm/Resident/baselines/M1M2/A1A5/S1S13 remain
pending; fullgoalACTIVE. D125 timingfixretain,copycomponentcandidate alone
doesNOTestablishqualifiedFull orcausalwhole-system improvement.

## Previous checkpoint — D125 Full4000 diagnostic complete; qualification FAILED

2026-09-30 20:49. Goal ACTIVE/incomplete. Current turn PROGRESS. No subagents.
Baselines PAUSED. No GPU/remote/analysis/test job remains. All exact-owned domains
closed. D78/D80 once-only delivery cache authorization already fulfilled; NEVER rebuild.
Do NOT repeat D125 replay/projection/metadata/curation/failure/test/bundle work.

- Planned/submitted/started/terminal4000;3456 native-contract successes,
  524TimeoutError,20RuntimeError (17parent+3subprocessownershipunresolved).
  All-correct gateFAILS. Numericadapter/commonwarmSLO remainpending; noM1/M2rank.
- Conditional3456 mean/P95/P99TTFT882.915085/1785.200278/1796.354608s;
  controllerE2E888.134610/1790.791201/1798.612494s, NOTclientreceipt.
  Dispatch880.155105s=window875.873124+slot2.761062+release1.520920;
  service2.759980=prenative2.473673+native.286307;
  decode3.590042,worker→controller1.584736s;TPOT35.300291/P9559.949057ms.
- All3456timingTTFT/E2E/dispatch/decomposition/TPOTmaxerrors0ms,
  unchanged1mstolerancePASS. D124boundaryfixsupported; copycandidate NOTproven
  sufficientFullimprovement. No change toformula,deadline,capacity,threads,guards.
  GPU1160/HOST1272/NVMe924/Remote100confirmed,conflicts0(successsubsetonly).
- Firsttimeoutreq01539offset3366.969737s;firstownershipreturnreq03011offset
  4491.386125s. Earlyqueue/timeoutprecederecovery. 524timeoutgenerationunrecorded
  ≠nodispatch/zerowait;20returnednot_submittedonlydescribestheirgeneration.
  Initialfourruntimes3001successes;24replacementruntimes455successes.
- Physical23059.33509171981GPU-s,36leases/35quarantinesallreleased,noopenleases.
  Allcleanupcomplete;6482samples,peak40832913408B,minhost79962181632B,
  high/max/OOM/swap/warnings0. Execution/launcherpass≠qualifiedFull.
- Remote129UUID/artifactbytepairs129137809Bbothends,packing0;
  128published/1notpublished. Fullowner/span/job/requestcrosscheckPASS for
  req03300'scomplete780745BarchivefollowedbysubscriberTimeoutErrorcancellation.
  Contentnotpublished/notverified, notnetworkbyteloss; notalltimeoutcauses.
- Projection45356andobservationcell4745FINISHEDexit0:35:04.25,RSS92928KiB,
  output31744572B,SHAa1940952e8f1db0b691de80cee1082fb5f21e10f0141a1e50aae3e184ee17200.
  Exactf043db73c34d4b39a1b071efd4acdc34emptyclosed20:44:17.
  Original11510552337B SHA93c17465221aed19dc993bf04cbfbefa720b692e6cf4465e88aef76b33f648a2.
  NEVERwholeload/reprojectagain; smallprojectionreusable.
- Curation19965exit0:36.47s/RSS266576KiB;failure.64s/122780KiB;
  actual3/4GiBswap0CPU2,3,26,27,events0. Exactbc0748fe9e514add88e2175726b16266
  emptyclosed20:45:12. No curatorretry or data/criterion weakening.
- Finalcurated20260930_d125_7b_full_w0_full1.json SHA
  c416733c929a7ccd58045f71e882a59f463d377b6261bade62def383da7d3498;
  failurebreakdownSHAf5e33da0e720ea7dd14c7e4dbf6cf938b5779b1770c417a6abe2479bf91bb8a4.
  DocD125_FULL_W0_FULL1.md containsfinalstate/phase/failuretables and decision.
- Evidence30586exit0:71testsPASS3.083s;130frozenrefs,39curatedrehashed,
  147protectedPASS. Hugeoriginalhashreusedwithstat, notreread. VerificationSHA
  966650d0960c87baf8d3575a9d8fac64497ad3c6c2aa1522aed7cc9ddde5729d.
  Exactb845127004eb4dce9a2696feb3f8eb01emptyclosed20:46:41,memoryevents0.
- Small55member49150BanalysisbundleSHA
  a97ccf2402e4e5962fddaa3cfe5c08bff042449d480269f20859b3a8d90ef8df,
  individualmembersverified. Fullskills/plan/status/metricV1read; Feishuabsent.
  Scopedbackupnext(afterdiff/checksum/secrets), servingruntimee7alreadybacked.

NEXT: finish scopedD125backup, then a SINGLE evidence-led PrimeFullbottleneck:
useexistingD125smallrequestprojection+D123CPUevidence/sourcehistory to distinguish
global-entrybacklog fromtimeoccupyinglimitedserviceopportunities. Consultprimary
implementation/literature before a falsifiable causal test/optimization. DoNOT
repeat identicalFull, oldlargeprojections, blanketincreasecapacity/thread/deadline
or removephysical/ownershipguards. D124timingretain;copyremaincomponent-supported
candidate,notFullcausalwin. AfterPrimeFull,warm/Resident→Serverless→vLLM→S-LoRA→
dLoRA3B→Loquetier→HydraServe→M1/M2→A1–A5/S1–S13 remainpending. EntiregoalACTIVE.

Postbackup2026-09-30 20:52: eightD125files committed/pushed
61c9bcfff368916f200744f4300d234efe51be1c; exactremoteHEADverified.
Bundlechecksum,gitdiff,63staged/archivepayloadsecretschecksPASS;
usermanifest/unrelatedfilesneverstaged. No heavyjobremains,lastdisk240804945920B.
This postpushledgernote isnotruntime/config change. DoNOTrepeatD125closure/backup.

Boundedsourcefollow-up(noedit/noexperiment) reviewedD120/D121docs andcurrent
run_all_experiments.py:11644–11743,14425–14635,15156ff,15831ff,15956ff.
Externalreplayglobalgate=runtime_groups×runtimecap; run_one holds it through
_exec_request and _finish_runtime_request_reservation, thenreleases beforeouter
terminal. Consequentlysuccessgate→outer isoccupancyupperenvelope, NOTnative-only
capacity. No proofyettoincreasegate/cap; cancellation/ownerreleasecannotbedropped.
ExistingD120run_occupancy_audit2.sh + scripts/analyze_control_path_overhead.py
acceptallow-failed and smallprojection, preserveallIDs; D125sameanalysisNOTrunyet.
D123oldCPUsummaryreusedwithoutreprofiling: _footprints11.2511% allGILsamples,
sendRPC13.4354%,fileinventory4.4048%,sourceobservation2.5341%; samplefractions
notcurrentD125walltime. D124targetedcopiesalreadychanged, no extrapolatedshares.
CurrentNativeSourceSnapshot._footprints independentlyvalidatesallstorageunion/
sharing/capacitymetadata; currentfile_source_snapshot restats managedsources.
DoNOTcache stalephysicalstate ordeletechecks; nextcandidate stillrequiresprimary
sourcecomparison and causalvalidation. No optimizerselected/implemented yet.

## D125 projection-stage history — superseded by completed analysis above

2026-09-30 20:27. Goal ACTIVE/incomplete. Current turn PROGRESS. No subagents.
Previous turn PROGRESS (run completion, cleanup, metadata and projection launch).
Baselines PAUSED. Request/replay/launcher finished; exact-owned cleanup complete.
No GPU/remote service is running. ONE CPU-heavy job: new D125 streaming projection.

- All4000 terminals:3456success/native_contract_matched,524TimeoutError,
  20returnedfailures with null terminal error_type. All-correct gateFAILED;
  numericaladapter/commonSLO remainpending. DoNOT call successful qualifiedFull.
- Finalmain/replay/watchdogexit0,launchpass=true, nativeGPUreleaseconfirmed,
  servicepathremoved.36physicalleases released/noopen,23059.33509171981GPU-s;
  windows pre47.79901311098365/arrival15808.675131062744/
  drain7151.112570960191/cleanup51.7483765858924.35quarantinesreleased.
- 6482resource samples,peak40832913408B,minhost79962181632B;
  high/max/OOM/swap/warningsall0. IncludesresultserializationafterGPUrelease.
  Rawresult11510552337B written20:05:54;serviceinactive/launchready20:07:33.
- Localexactemptyauxclosed20:07:44; remote3B/7B/monitor exactIDs/PIDs stopped
  20:07:45. Allstoplogs retained. No cache/weight/trace regeneration.
- Copied remotejournaltransfers-8566d2902a8b4a53b3d69c990de693ce.jsonl ONCE,
  localremote_7b_full1_transfers.jsonl SHA
  8a7f41746218a8545fcb0b7c1083823c1405e33fcefff6a47176a3c277a1b359;
  remote_monitor_full1_final.log SHA
  3f8637b1e2e2819c5f8a12567c74649b5118fec55da71d032850b6a27861bf48.
  BothmatchremoteSHA.129UUID/artifactpairs,129137809Bbothends,packing0;
  128published/1notpublished(code_lora_0099,242537bf3c584484b62451aa70d6fa4b).
  Unpublishedreceived780745B,archiveverifiedtrue/contentfalse,no payloadverified;
  doNOT invent cancellationcause or validcachehit. Fullauditpending.
- MetadatareuseD119passed1.52s/RSS220104KiB in3/4GiBswap0 CPU2,3,26,27;
  exactmetadata scope5a0a1c03fe544d4f81d147c2b1c28c4b emptyclosed.
  full_full1_preliminary.json SHA
  874a38245d0bbae84218e2cf4a54ea4ada4a108fe4d65d518bc007643600d9f6.
- LIVEprojection: unified exec session45356, unit
  primelora-d125-full1-project-20260930.scope,
  invocationf043db73c34d4b39a1b071efd4acdc34,actual3/4GiBswap0CPUs2,3,26,27.
  ExistingD96project_full_attempt3.jq, ONCEnewD125raw→
  d125_20260930/full_full1_request_projection.json. DoNOT wholeload or restart.
  time/scope receipts full_full1_projection_{time,scope}.txt.20:09events0.
- Preliminarystatus table docs/ieee_tc/D125_FULL_W0_FULL1.md written;
  latency/failure/timingvalidation and D124decision NOTcompleted.
  Previousobservationcells4684/4707/4711finished, NOTrunningexperimenthandles.
  No optimizer, capacity/deadline/thread/guard change; no newGPUrun/test/backup.

NEXT: wait exactprojection45356 toexit, inspect exit/time/actualscope and
cleanupthatemptydomain. Adapt existingD119curator/failureaudit to OBSERVEDD125
population/size/journal and retain alltimingviolations; 1unpublishedtransfer is
diagnostic, notpermission to inventcontentvalidation. Fullstatstable/interpretation,
source/protected checks, testedscopedbackup. No repeatedoldlargeJSONprojection.
Thendecide candidate/nextPrimeFullstep fromevidence; warm/Resident→baselines→
M1/M2→A1–A5/S1–S13 remainpending. EntiregoalACTIVE.

20:12:22 checkpoint: projection45356 STILL LIVE, jqPID2783789, elapsed3:25,
CPU3:24, rchar1143332847B, RSS12056KiB; exactscopeabove2tasks,current10424320B,
high/max/OOMall0. Finalprojectionfile isnotready; doNOTread/restart it.
Prepared (NOTexecuted) D125summarize_full_full1.py,collect_full1_failure_breakdown.py,
curate_full_full1.sh,cleanup_projection_full1.sh using existingD119scripts.
Only observedD125counts/sizes/identities plus D123record-all-timing-violations
handling; toleranceunchanged1ms. Remoteexpected128published+1notpublished,
conditionalvalidation remainsstrict, no falsecontentpass. Exactprojectcleanup
expects invocationf043db73c34d4b39a1b071efd4acdc34. ShellsyntaxchecksPASS;
curation/failureaudits NOT run, doNOTclaimfinalstats. Beforecurationverifyfinal
projectiontimeExitstatus0, cleanupitsactualemptyresource domain, then runcurator
inprimelora-d125-full1-curate-20260930.scope actual3/4GiBswap0CPU2,3,26,27.
GitdiffcheckPASS; usermanifest/unrelateduntrackedpreserved. No newbackup yet
becausefinalanalysis/testedcheckpoint isnotready. Servingruntimee7alreadybacked.
Useronce-onlycacheauthorization reconfirmed; alreadyfulfilledD78/D80,
NEVERrebuildpublishedcache. FinalD125doccurrentlyexplicitPRELIMINARYstatustable.

20:26:59 checkpoint: SAMEprojection45356 LIVE, exactinvocationunchanged,
jqPID2783789elapsed18:02/CPU18:01,rchar5983236079B of11510552337B source,
RSS44480KiB,scope2tasks/current43745280B,high/max/OOMall0,disk240832892928B.
Observationcell4730finished15polls; thisisNOTprojectioncompletion.
No largefilewholeload/reprojection/newGPUjob/test/remoteoperation/backup.
ReadfullAGENTS/plan/status/metricV1/monitor+analyze skills thisturn.
One bounded direct-record analysis changes unpublished-transfer explanation:

- RemoteUUID242537bf3c584484b62451aa70d6fa4b outcome sent,read/write780745B;
  localarchivecompleteandverified;body275557.939614691,extractstart275557.939679471,
  clientend275557.999528606, loadingstart275557.786223007, extractdurationnull.
- Samefileowner011c6036f1174af08a798a700bf6aa05/adaptercode_lora_0099 and
  containingI/Ospan links pressure313e24fb3517480988a44297d768e02d:
  operation_outcome cancelled,caller_cancelledtrue,errorCancelledError,
  io_started275557.784389969/io_joined275557.999871948/finished275558.009436967.
- coordination_after_shutdown.ieee_movements jobe59c71ef08344b6398739ac5007af0df
  statecancelled,unique demand subscription2d528d0eb29e407bb7971be2e784001d,
  plan_idrequest:req_03300,withdrawn275557.948074521.
  replayplanned273757.9452582016,contract1800splanned_arrivaldeadline;
  loadingbegan1799.840965safterarrival,withdrawal2.816msafterdeadline.
  requestterminalTimeoutError at275558.114980423,clockssame.
  This supports timeout-associated demand cancellation, NOTbyte-loss;
  doesNOTexplain allpriorwait or all524timeouts.
- Addedmatching check+fulltransfer/pressure/job/request evidence retention to
  PREPAREDcurator; syntaxASTPASS, notexecutedbeforeprojectionfinishes.
  D125docupdated direct-recordfinding; finalcrosscheck/latency/failurespending.
  GitdiffcheckPASS. No servingruntime/config/thresholdchanges. No speculative
  optimizer accepted. NEXTremains exactprojection→emptycleanup→curation→table/
  tests/source/protectedverification→backup→evidence-basedPrimeFullnextstep.

## D125 launch and historical live checkpoints (superseded by completion above)

2026-09-30 19:45. Goal ACTIVE/incomplete. Current turn VERIFIED WAIT. No subagents.
Previous turn VERIFIED WAIT. D125 launch turn was PROGRESS; exact live service
invocation repeatedly polled at45s intervals, never restarted.
D124 previous turn PROGRESS; its tests/micro/backup are complete, do NOT repeat.
Baselines PAUSED. Only one heavy run: ordinary 7B W0 Full4000, no prefix/profiler.

- Runtime HEAD e7d4c91aa4347f26a1f08672b54a5ad4b3228d78, already backed.
  Same D119 model/workload/config except three fresh owned paths; D89 profiles.
  Only D124 timing interval and preparation-copy candidate differ from D123 runtime.
  cap2/slots4/cpu24/.70, 1800s planned-arrival protection unchanged.
  Source4000/selected4000, viewSHA
  a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
  fixed_length_greedy_v1, real published delivery, no new cache/weights/trace.
- Raw results/ieee_tc/p2_backend_qualification/d125_20260930;
  run 7b_full_w0_full1; launcher log full1_console.log;
  launch evidence 7b_full_w0_full1/launch.launch/.
  tmux tc-d125-full1 started18:17:15. DO NOT launch duplicate or modify runtime.
- Service primelora-tc-svc-09bcef61bfa14b20bc099a54c88ea642.scope,
  invocation b94b9b492d3b448bb266e5289b790c89;
  aux primelora-tc-aux-cb5a52372bb94bd3b012b0b3a2e68b5f.scope,
  invocation61fa7a7966244155b68ebc78d3191d7c.
  Actual service72/80GiBswap2 CPU4-23,28-47;
  aux3/4GiBswap0 CPU2,3,26,27; watchdog/publisher separately verified.
  deployment_notice270684.173485817, replay_t0=270744.173485817,
  last planned arrival274708.0823889389 (common monotonic clock).
- Remote D80 3B/7B services active before inference; invocation respectively
  4007a21f66ae4769b286665a4c17edd2 /05550087e4564ee4a53a40575cd633ec.
  monitor primelora-artifact-monitor-d125full1.service,
  invocationde88cbacbc284ffaa072e5ff02d73ef4;
  /home/lab14/primelora_remote/tc/d125_20260930/remote_monitor_7b_full_full1.log.
  Both healthPASS prepublished_gzip_v1/timingv2. NEVER rebuild D78 published cache.
  No remote management/config/hash/cleanup during inference.
- Prelaunch130source refs and147protectedPASS; receiptSHA
  ff03e949dbadeb952226b78553b602169bcb17a7a681ccb55930dc6b3d345538.
  Local112.5GBavailable/disk252.65GB; remote102.15GBavailable/146.06GBdisk;
  localeno1np0/remoteeno1 both1000/full. Qualified check, not future guarantee.
  Empty prelaunch scope3623d319701a4ee08d90c537f176632a and healthscope
  d0e91fb489bb48c08837a9e14d1ae8e4 exactly stopped, memory.events0.
  First37watchdogsamples: nohigh/max/OOM/swap/warning/escaped/foreigncompute;
  oneownedGPU loading. No performance conclusion from startup.

NEXT: monitor this exact run to terminal; no extra probes/edits/heavy work.
Then exact-owned cleanup, bounded existing analysis/table, timing/native/source
and full-population checks, GPU lifecycle, failed requests, D124 acceptance decision.
Do not whole-load/reproject old D119/D118/D123 huge originals. Preserve all failures.
If justified follow with 3BFull, then warm/Resident→baselines→M1/M2→A1–A5/S1–S13.
Numeric adapter distinction and formal SLO remain pending; this is development.

Live checkpoint18:21:02: 75 terminals/75 success/75native_contract_matched,
0returned failures so far; latestsubmitted req00086. Not final completion or
numericadapterqualification. Watchdogsample223, 4ownedGPUs, hostavailable
97720356864B, servicecurrent14837551104B/peak17804238848B, disk251965861888B;
high/max/OOM/swap/warnings/foreigncompute/escaped all0. watchdog.stderr empty.
Business replay remains LIVE; preserve exact scopes above and DO NOT rerun.
No new source optimizer/testing/backup needed while this run is active.

Live checkpoint18:29:26: same exact service invocation, 443terminals/443success/
443native_contract_matched, no returnedfailures. Latest publisherreq00532,
so arrival crossed first500 rotation; successful-prefix completion is NOTfull
qualification. Lastliveconsole531arrived/440done/91backlog; not a synchronized
final population tally. Conditional displayed TTFT is rising; doNOTclaim queue
resolved or compare this partial with D119Full/D123profiled prefix.
Watchdogsample721: 4ownedGPUs, available94718832640B, current18242101248B,
peak18259025920B,disk251500208128B; high/max/OOM/swap/warning/foreigncompute/
escapedall0. No restart, addedprobe, runtimeedit, remoteoperation or hashscan.
Continue exact LIVErun; no newheavyjob. End→cleanup→validate→table→interpret.

Live checkpoint18:38:54: 850terminals/850success/850native_contract_matched,
0returnedfailures, latestpublisherreq01161. Same service invocation active527tasks.
Lastliveconsole1155arrived/845done/310backlog, conditionalTTFTcontinuesrising;
no finalperformance/Full/numericadapter/SLO qualification claim.
Watchdogsample1281: 4ownedGPUs, available92453351424B,
current20787978240B/peak20863315968B,disk251192037376B;
high/max/OOM/swap/warning/foreigncompute/escapedall0. No newheavywork, optimizer,
test, sourceedit, remoteoperation, scan or backup. Observational toolcell4647
finished its10polls; it is NOT the experiment. tmux/service continueLIVE.
Next turn keepmonitoring exact D125 until realterminal, then existing cleanup/
boundedanalysis/table/decision. Never restart merely because monitoringcellended.

Live checkpoint18:51:35: 1372terminals/1372success/1372native_contract_matched,
0returnedfailures. Latestpublisherreq02108 (2109submitted); lastliveconsole
2108arrived/1366done/742backlog, asynchronous snapshots, notfinalmetrics.
Same exact invocation remainsactive527tasks; detailedCPUprofiler stillabsent.
Watchdogsample2032: 4ownedGPUs, available90639912960B,
current24113881088B/peak24159137792B,disk250751459328B;
high/max/OOM/swap/warning/foreigncompute/escapedall0.
Currentturn15confirmedpolls45sapart, toolcell4650finishedobservationonly;
experiment remainsLIVE in tmux/exact scopes. No other task/replay/optimizer,
remotechanges/hashscan, tests or source changes. Previous turn VERIFIEDWAIT.
Continue same D125; doNOT infer completion/timeout from endedobservationcell.

Live checkpoint19:04:13: 1879terminals/1879success/1879native_contract_matched,
0returnedfailures. Latestpublisherreq03068; lastliveconsole3064arrived/1875done/
1189backlog, asynchronous snapshots. Overallcompleted1879/4000, NOT100%Full
completion. Statusreportclarified that nofailures means observedterminals only; all4000
remain denominator. ConditionalTTFTcontinueshigh; no final qualification/claim.
Exactsame serviceinvocation active527tasks; watchdogsample2779, 4ownedGPUs,
available87391760384B,current27185434624B/peak27207659520B,disk250496278528B;
high/max/OOM/swap/warning/foreigncompute/escapedall0.
Currentturn15polls45sapart; toolcell4653finishedobservationonly, notexperiment.
No runtime/config change, probe, tests, remoteoperation, scan, cleanup or backup.
Continue exactLIVE D125 torealterminal. Do not launchanotherheavyjob or duplicate.

Live checkpoint19:19:32: 2492terminals/2481success/2481native_contract_matched,
11TimeoutError. Latestpublisherreq03802 (3803submitted); lastliveconsole
3777arrived/2485done/1292backlog, asynchronous snapshots, NOTfinalmetrics.
Full4000 all-correct gate cannot pass this attempt because returnedfailures exist;
retain remainder/cleanup evidence, no deadline change or success-only selection.
DoNOT infer numericadapter/SLO qualification or causal optimizer improvement.
Same exact serviceinvocation active527tasks; watchdogsample3687, 4ownedGPUs,
available83858169856B,current31077879808B/peak31096963072B,disk250152910848B;
high/max/OOM/swap/warning/foreigncompute/escapedall0. Timeoutcause notproven
by resource safety or livequeue alone. Fullrun remainsLIVE.
Observationcell4657 finished15polls (continued across contextcompaction), not
experimentcompletion. ReadfullAGENTS/plan/status/monitor skill; goalstillACTIVE.
No runtime/config change, probe, tests, remoteoperation, scan, cleanup or backup.
Only ledger checkpoint edited. Continue same exact D125 torealterminal, then
cleanup/validation/table/interpretation; no addedheavyjob or duplicate launch.

Live checkpoint19:32:11: 2992terminals/2975success/2975native_contract_matched,
17TimeoutError. Publisherreplay_complete observed19:24:38; all4000arrived,
1008notterminal atlatest requestaudit. Lastliveconsole4000arrived/2986done/
1014backlog isasynchronous, notfinalmetrics. OrdinaryFull remainsinDRAIN,
notcompleted. All-correct gatefailed; no finalSLO/numericadapter/performanceclaim.
Same exact serviceinvocation active527tasks; watchdogsample4435, 4ownedGPUs,
available81721077760B,current33458544640B/peak33552990208B,disk249889742848B;
high/max/OOM/swap/warning/foreigncompute/escapedall0. GPUholding continues.
Currentturn15confirmed45spolls, observationcell4670finished, NOTexperiment.
Readfullplan/status/AGENTS/monitor skill. No runtime/config change, tests, probes,
remoteoperation, scan, cleanup or backup. Only ledgercheckpoint edited.
Continue exactLIVE D125 throughactualterminal/release, then boundedvalidation/
table/interpretation. BaselinesstillPAUSED; no duplicate/newheavyjob.

Live checkpoint19:45:02: 3629terminals/3211success/3211native_contract_matched,
401TimeoutError plus17failureterminals withnullerror_type. All4000arrived;
371notterminal. Lastconsole3619done/381backlog isasynchronous, notfinalmetrics.
At19:33:10 firsttwo nulltypefailures appeared: req03011 activation9cbcc8...
andreq03014 activation0562b...; service.log19:33:17 explicitly showed
native RPC ownership unresolved; new generation =2, readyinst/runtimes0 and
ownedreplica removals. Laterall17nulltypefailures needfullerror reconciliation;
doNOT automatically labelall17samecause. Service autonomouslyretired/recreated
replicas, observedGPUholding1–4; no manualrestart or remoteoperation.
Firsttimeout/backlog precededthisrecovery; donotattributeearlydelayonlytolater
ownership/quarantine. Failedattempt remains retained, NOTcompletequalifiedFull.
Same exact serviceinvocation active515tasks; watchdogsample5196, 4ownedGPUs,
available81430134784B,current33489313792B/peak34640678912B,disk249521926144B;
high/max/OOM/swap/warning/foreigncompute/escapedall0. Resource safety alone
doesnotproveperformancecause. NativecountNOTnumericadapterproof.
Currentturn15confirmed45spolls, observationcell4678closed, NOTexperiment.
Readfullplan/status/AGENTS/monitor skill; oneboundedread oflivefailureterminals/
logtail fornewfailuretype, no offlineanalysis/probe/tests/configchange/backup.
Continue exactD125 toactualterminal/release, then cleanup→boundedvalidation/
table→interpretation→next. BaselinesPAUSED; no newheavyjob or duplicatelaunch.

## Previous checkpoint — D124 candidate verified and backed

2026-09-30 18:08. Goal ACTIVE/incomplete. Current turn PROGRESS. No subagents.
Baselines PAUSED. No GPU/real-remote job started this turn; D78/D80 cache unchanged.
Do NOT repeat D123 analysis/replay, D124 timing/probe/tests/curation/bundle.

- D124 fixed actual missing timing interval: pass global-admission timestamp G
  into reservation path, end wait/start service at one T. Original raw formula
  (G-A)+(T-S) omitted S-G. D123916success signed gaps allnonnegative,
  mean.043464436ms,max3.948904981ms; three >1ms unchanged failures.
  Old rows/files untouched. Absolute boundary recovery is controller completion,
  not client receipt. Timing feeds online TTFT control, new runtime identity.
- One optimizer candidate: freeze JSON owner view via SAME canonical bytes→
  json.loads+SHA; detach all mutable values, reject type coercion/nonfinite/cycles.
  Whole plan retains ordinary deepcopy for Python/dataclass/tuple components.
  All owner/epoch/content/budget/live/source/selection guards unchanged.
  No capacity/thread/timeout/feedback formula changes; no filelock guard removal.
- New actual-line D123 GIL attribution confirms both deepcopies prominent;
  no complete source_view in normalmainoutcome. No huge raw re-projection.
- Paired component CPU3trials, alternating order, reference source5e3407b:
  4adapter owned-input fixture1.050577→.604413ms; copy+selector1.528904→1.253710ms;
  real retainedD88 7B native component38.796561→13.169825ms (66.05% mean decrease).
  Same values/SHA/selected, copy isolation and tamper rejectionPASS.
  Fixture isnot500workload; D88nativeonly2adapters/512alloc. NOFullspeedupclaim/CI.
- Tests209 timing/lifecycle/retirement BEFOREpreparationcandidate;
  189 preparation/copy/transfer plus425smoke/launch/replay WITHcandidate, allPASS.
  Native microenvironment unchanged;3/4GiBswap0CPU2,3,26,27, nohigh/max/OOM.
- Firstmicrofailed wrongD121summarypath; source/log/timeexit1retained, nofinaldata;
  secondonlypathfixed,11.42sRSS1066052KiBexit0. Two curator errors (trailingstdout
  logregex; generatedregexsyntax) retained; correctedcurator only, no test/probe rerun.
- Curated paper_results/ieee_tc/p2_backend/20260930_d124_preparation_detachment/:
  summarySHA6c7e1278dc23f023789989a54bb08575a103c1a8cad727b7708b2a534e23b0e5;
  table/sampleCSV;46smallmembers48312BbundleSHA
  8a7d1557422abd9d1c396f13726664f0a7923ac57be4e9624f61fbf5727177f1.
  Protected147PASS. Doc D124_TIMING_AND_PREPARATION_DETACHMENT.md has table,
  first-principles literature/code references, equality scope and failure history.
- All exact-owned timing/preparation/probe2/verify/curate/curate2 scopes emptyclosed;
  finalcurate3 identity6d9b9a8803764279aa320aaa107e10cf emptyclosed18:08:14,
  memory.events0. No live work.
  Bundle checksum and57staged/archivepayload secrets checksPASS;13sourcerefsSHA
  verified. NewCSV CRLF→LF formatting only after diffcheck; values unchanged.
  Scoped backup next; NEVERstage usermanifest.

Postbackup2026-09-30 18:10: D124twelvefiles committed/pushed
e7d4c91aa4347f26a1f08672b54a5ad4b3228d78, exactremoteHEAD verified.
Only usermanifest tracked dirty before this note; old/unrelated files preserved.
Allprimelorascopesinactive/GPUempty, disk252668096512B. DoNOTrepeatbackup.
This ledger note is not a serving/config change. Wholegoal remainsACTIVE.

NEXT: ordinary 7B Full4000, no detailedCPUprofiler/prefix, same
D119configuration/D89profiles/realpublished delivery. Only D124timing+copydiff.
Verify completion/timing/backlog/GPUlifecycle/cleanup, then3BFull ifappropriate.
Do not add more component probes. Candidate NOTaccepted as Full improvement.
Then warm/Resident→baselines→M1/M2→A1–A5/S1–S13, allstillpending.

## Previous checkpoint — D123 diagnostic complete and backed

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

Postbackup2026-09-30: nine scopedfiles committed/pushed
5e3407b16b62e5c8a85e81027b4c24f55e367a65; exactremoteHEAD verified.
Diff/bundle/memberSHA/83staged-and-archive-entrysecrets checksPASS;
userdirtymanifest neverstaged. DoNOTrepeat backup or completedD123 checks.
This postpush ledger note is not a runtime change. Old1439line historySHA
verified identical afterarchive. No analysis/GPU/remote job remains.

Historical D123 NEXT (completed by D124 above): reconcile small timing boundary using
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
