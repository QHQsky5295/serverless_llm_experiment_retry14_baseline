# IEEE TC execution status

## CURRENT — D103 profiling interrupted; cleaned; source-coverage counterexample next

2026-09-28 17:20. Supersedes ALL historical live/NEXT instructions below.
Goal ACTIVE/incomplete,baselines PAUSED. NoGPU/remotejob. RuntimeFull8 source
2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d unchanged; evidenceHEAD e6c7831.

- D103profiling didNOTcomplete. Planned4000,arrived59,submitted58;27savedrequest
  records=11success/native-contract+1failed_untyped(req_00018)+15CancelledError.
  Physicalsummary12noninterruptedterminal/15interrupted. DoNOTfabricateoutcomes
  fortheotheroffered/submitted/unarrivedrequests; original4000denominator retained.
- Native primaryexception: ieee_prepare_host ->
  residency_manager.py:2505 proactive_host_prepare_and_acquire:
  'replacement epoch lacks the current owned source/fallback set'. Guardchecks
  coverage OR name/pathidentity OR missingGPUconfirmation. Whichsubcondition
  failedisNOTyetproven. DoNOTassumeprofiler-causedorjustremoveguard.
- GPUplan38aa40e1f6fb49a58b1ac539c82837a4 registeredepoch1/allslotsNone;
  laterexplicitdeferrals epoch96/100; adapter416753 native_outcome_unresolved,
  nofailedRPCreceipt. Mainfailurepropagatedthroughresidencytaskreaping.
- LaterpublisherBrokenPipe/ConnectionReset; launchclassification
  protocol_or_launcher_error:'external replay failed; no silent internal timer
  fallback',serviceexit-15,replayexit1. KeepouterclassificationANDearlierservice
  cause. Profilerterminatedbeforewrite: NOspeedscope,emptytimefile,NOCPUresult.
- Observed346.7111347940081GPU-s only; fullgpu_secondsnull.4leasesallreleased,
  cleanupcomplete,nativeGPUrelease/servicepathremovedtrue.190resourcesamples,
  peak19720454144B,minhostavailable93658480640B,high/max/OOM/swap/warnings0.
  All12remoteUUIDpairs published/contentverified,27884609Bbothends,packing0.
- Mainexit/GPUemptyverified; exactremote3B/7B/monitorstopped17:15:31 andemptyaux
  cleaned. Remotejournalidentifiedbyclock:transfers-663b80d656c544038e1fc3b447a0d16f.jsonl,
  copiedonceasrawD103/remote_3b_profile1_transfers.jsonl;monitorcopiedonce. No repeats.
- Curator collect_profile1_failure.py adaptedD101; exit0,147protected/34sourcechecks
  PASS. Curated paper_results/ieee_tc/p2_backend/20260928_d103_controller_profile1_failure.json
  SHA79827bcd881d581d6bc1f9b605b9aa14a892c41af12e2a418e27876ae717aaa3.
  Table/doc D103_CONTROLLER_PROFILE_INTERRUPTED.md. Curator scope
  primelora-d103-failure-curation-20260928.scope invocation371854d44a984f5d9ec06dd639fa7beb,
  exactidentity+empty-verified/stopped; rawD103/curation_cleanup.log. No live task.
- NEXT scopedbackupofD103failureevidence,then ONE CPUcounterexample usingactual
  owner/plannerfixtures: valid source-set change afterfrozenplanregistration.
  Read D99/D100history+currentprimarysources; isolateexactguardclausebeforefix.
  Separate legitimateplanobsolescencefromidentity/confirmationdamage. No arbitrary
  catch/retry,newGPUreplay,profilerlifecyclepatchorparser-sharingpatch.
- Full8lastcomplete remains3970/4000+30TimeoutError. NeitherD103norD102advances
  formalqualification/ranking.7B,warm/Resident,baselines,M1M2,A/S stillpending.

## D103 — superseded live notes (do not restart this run)

2026-09-28 17:12. Supersedes ALL historical live/NEXT instructions below.
Goal ACTIVE/incomplete; baselines PAUSED. EvidenceHEAD e6c783110046842a9aaa0cd52b96e22de8c040de
pushed and exactremoteHEADverified. Runtime remains Full8 production2f1bc4b;
NO production/config strategy change during this run.

- ONE live tmux tc-d103-3b-profile1; canonical3BFull4000W0 plus parent-only
  py-spy0.4.2 sampling100Hz/GIL/threads/speedscope. Diagnostic only, NOTformal
  performance or a successful Full qualification. No shortened/generatedtrace.
- Raw results/ieee_tc/p2_backend_qualification/d103_20260928; launcher
  run_3b_full_w0_profile1.sh/config3b_main_config_profile1.yaml. New threeowned
  paths only versusFull8. profile_controller_python.sh resetsFAASLORA_PYTHON to
  actualnativePython beforelaunch; workers are NOT recursively profiled.
- Service primelora-tc-svc-7eab36c069a6440a823e1534d27addde.scope,
  invocationf15daa9feed7437a99511a7f078277b6;72/80GiB,swap2GiB verified.
  Auxiliary primelora-tc-aux-3c2957f5f8614fa5a55d102b96356681.scope,
  invocation2d064f2a857e4b39aba61a962047173e;3/4GiB,swap0 verified.
  ActualprofilerPID2393281,controllerPID2393282. Gateallow_exec andwatchdogready
  present;17:11:50startup ongoing,no warning/abort. DoNOTrestart/relaunch.
- Remote3B PID1920627 invocation5e0e2d1cc6cc4ca585bc69556256cf0e;
  7B PID1920629 invocation2a63e91d71084fc796300d301ddb87fd;
  monitord103profile1 PID1920632 invocation51e939641e194381858cc67b93bed33f.
  Remote3Bclock remote-process-monotonic:4983b6cc05da4803813edb91933457ce.
  Remote journal filename MUST be found byclockafterrun,not guessedUUID.
  Monitor /home/lab14/primelora_remote/tc/d103_20260928/remote_monitor_3b_full_profile1.log.
- PrelaunchbothNIC1000/full,GPUidle,hostavailable113846681600B,disk306419744768B,
  swap0. Remoteavailable104460210176B,disk147216875520B. Original147protected
  and34source refsPASS; prelaunchverificationSHA
  bda16d0e5aa789936c4a6e4e18d9eadb8cf8a71485a6a0da2d8d801bd9375f54.
  InitialsystemPythonhealthclientimportnumpyfailedbeforeHTTP; emptyfilekept,
  existingCPUenvclienthealth subsequentlyverifiedboth. No environmentinstall.
- NEXT monitor same run through normalterminal+GPUrelease+serialization+profiler
  exit. Profileonlywrittenatend; absentfilewhileliveisNOTfailure. Nootherheavyjob,
  sourcechanges,remotehash/config/cleanup. Thenidentity-scopedcleanup→validate
  completedprofile/exit→CPUstack table+outcomestatus→interpretation→next.
- Speedscope retains sampleordinal,notwalltimestamp (officialv0.4.2sourcechecked).
  Use stack ancestry toseparatecontrol work from finalserialization; DON'Tmap
  index/rate toarrivalclock orclaimnestedCPUpercentagesareadditive. Sampling
  overheadunquantified; noneofthisrunentersformalM1/M2ranking.
- D102completedandbackedup; no repeatparserprobes/cache/profileinitializers.
  Prime3BFull3970/4000prior,7B,warm/Resident,baselines,M1M2,A/S remainpending.

## D102 — completed CPU diagnosis (superseded next-action instructions)

2026-09-28 17:07. Supersedes ALL historical live/NEXT instructions below.
Goal ACTIVE/incomplete; baselines PAUSED. Production remains Full8 source
2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d. No GPU/remote job or new Full replay.

- D102 parser hypothesis measured on retained, SHA-qualified D88 source payload:
  one owner,8 registered adapters,1792 HOST allocations. One parse2.460/4.465/
  2.536ms;32 repeats83.054/95.047/77.491ms. Outputs identical. NOT Full8's
  routing-wave distribution; no timeout attribution or performance improvement.
- Initial Full8-readiness input correctly rejected for missing native_footprints;
  failed probe retained. No invented footprint and no repeated large-run scan.
- Decision: do NOT implement parse-sharing from this evidence. Need actual
  complete controller CPU profile; retain formulas/config/deadline/trace.
- Existing py-spy0.4.1 sampled294 stacks but had ECHILD shutdown error; notused.
  Existing0.4.2 sampled248,actual native Python3.12.12 child completed,exit0
  (3.07s,10752KiB peakRSS). No install/ptrace/sysctl changes. Witness is CPU-only,
  not a serving/performance result or guarantee of profiler overhead.
- All four CPU-probe scopes identity+empty-verified/stopped17:05. Curator exit0,
  147protected+24unchangedsourcechecks PASS; prior47 evidence-smoke reused with
  unchanged source. Curator94b3faa648054e96ac653af1ca4c296d also exactidentity+
  empty-verified/stopped; rawD102/curation_cleanup.log. No live analysis task.
- Table/doc D102_NATIVE_SNAPSHOT_CPU_DIAGNOSIS.md; curated
  paper_results/ieee_tc/p2_backend/20260928_d102_snapshot_cpu_diagnosis.json
  SHA f4cb0e1c53e36d66e1d68c499fb3e3d9946499f14267d07ea4d5f41e87d7fa36.
  Raw results/ieee_tc/p2_backend_qualification/d102_20260928; don't rerun probes.
- NEXT scoped evidence backup, then ONE profiling diagnostic of unchanged
  Full4000W0 controller using qualified existing py-spy0.4.2. Canonical external
  replay/resource gate retained; wrapper only, no shortened/generated trace,
  no source change. Profile business interval separately from finalserialization.
  NOT formalperformance or a new optimization-qualification claim. No blindretry.
- Full8 remains3970/4000native-contract-matched,30TimeoutError. Prime3BFull,
  7BFull,warm/Resident,baselines,M1M2,A/S all incomplete. Once-only published
  delivery cache already fulfilled; NEVER rebuild/duplicate pools.

## D101 Full8 — completed checkpoint (superseded next-action instructions)

2026-09-28 16:42. Supersedes ALL historical live/NEXT instructions below.
Goal ACTIVE/incomplete; baselines PAUSED. Runtime2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d
unchanged. Inference,remote,projection,curation andevidencecheckallfinished.
Allownedanalysisunits exactidentity+emptyverified/stopped; no job to restart.
- Evidencecommit8aa3607defd14cdc491708079cb3053d98546f62 pushed16:43:22 to
  faaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
  Fourfilesonly(doc,ledger,two curatedJSON); no runtimechange/userdirty staged.
  DoNOTrepeatbackup,projection,curation,smoke. CurrentHEAD8aa3607; runtimeidentity
  in Full8 remains2f1bc4b. Thispost-push ledgerupdateisnotanewruntimeconfiguration.

- Full4000:3970success/native-contract-matched,30TimeoutError,0otherreturned.
  Completeexecutionbutfailedqualification; no formalSLO/numerical/rankingclaim.
- Physical19874.486280712983GPU-s,4leasesallreleased;1initial+3natural_scaleout,
  zeroquarantine/replacement. All132remoteUUIDpairspublished/contentverified,
  bothends306360162B,requestpacking0.5593resourcesamples,high/max/OOM/swap0.
- Fullprojectionexit0:1210.15s,RSS137856KiB;raw9889140123Bimmutable,projected33MB.
  All4000identitiesmatched.3970conditionalmeans:TTFT734.065505s,E2E740.126818s,
  dispatch721.191880s (=window703.630776+slot15.681931+release-late1.879173),
  service12.873625s,nativeTTFT0.484660s,TPOT31.080834ms. Nofailedlatencyfabricated.
- All30timeoutgeneration/source-admissionevidenceunrecorded;notproofofnodispatch.
  Firstcontrollerobservationreq_01685offset3478.693559s,offeredage1800.231357s.
  No finalstage/rootcauseinferencefromnullinstance. Completefailurelistretained.
- Supersession255:251registration/3nativeGPU/1filefallback;sameeventspropagate
  through3planlists,don'ttriplecount. Statsrequests10952/collections1049/joined9903,
  RPC4115,stale6727,membership4. RejectioncountsNOTrequestsorproof ofCPUcause.
- Doc/status tableD101_FULL_W0_ATTEMPT8.md;curatedfullSHA
  86b795c8df33bb1d6984e0c25c455f0cf2645228b821382ef6a54f4ef9d2fdb9;
  failureSHA8a604531df3cae5ef416e5b0b5ed182ca2c8c946be947fa062c21dbad2eb9593.
  Filenamespaper_results/ieee_tc/p2_backend/20260928_d101_3b_full_w0_*.json.
- Source24/147protectedchecksPASS;47evidence-smokePASS1.586s. ReceiptsrawD101/
  full_attempt8_evidence_verification.json,full_attempt8_evidence_smoke.log,
  metadata_scopes_cleanup_full8.log,analysis_scopes_cleanup_full8.log.
  No need to repeat extraction,curation,smoke,cache/download/profile/prefixes.
- NEXT ONEcausalCPUprofile usingrecordednativeview,
  checkingrepeatedNativeSourceSnapshot.from_native/footprintconversionperwaiter.
  Thisisacandidateonly,runtimecost/timeoutcausalitynotmeasured;requirehistory,
  primarysources,measuredprofilebeforeimplementation. DoNOTweakenowner/freshness/
  physicalguards,extenddeadlineorblindFull9. 7B/warm/Resident/baselines/M/A/S pending.

## D101 Full8 — superseded projection/curation preparation notes

2026-09-28 16:17. Supersedes ALL historical live/NEXT instructions below.
Full goal ACTIVE/incomplete; baselines PAUSED. Runtime remains pushed
2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d; no new production changes.

- Full4000 W0 attempt8: all4000terminal,3970success/native-contract-matched,
  30TimeoutError. No RuntimeError terminals. Success99.25%, NOTfull-success,
  SLO/numerical/ranking qualification. Do not lengthen deadlines or discard failures.
- NativeGPUcomputeempty16:01:47; mainPID656236finishedby16:12:03, inference
  scopeinactive/tmuxgone. launch.jsonpass/nativeGPUrelease/servicepathremoved
  true, SHAfc74807a715761402e2956989f04b7d4ca6e73702d156e644eb2463550f8aec7.
  FullJSONcomplete9889140123B; NEVERwholeload it. Serialization took several
  minutes AFTER GPUrelease; usephysicaljournals,notlegacyconsoleInfraGPU/CE.
- Exactownedremote3B/7B/monitorunitsandemptylocalauxstopped16:13; receipts
  rawD101/remote_stop_full_attempt8.log andlocal_aux_cleanup_attempt8.log.
  No inference/remote job remains. Published cache and weights unchanged.
- Correctremotejournalidentifiedbyclock18c0b8ad9e2a40a596e744480615f5b5:
  /home/lab14/primelora_remote/tc/d80_20260927/3b/
  transfers-42ec47b0a908481fb4660685c4d9cf52.jsonl. CopiedonceasrawD101/
  remote_3b_full_attempt8_transfers.jsonl (132lines/108956B). Remote monitor
  copiedonceasremote_monitor_full_attempt8_final.log (93799561B). No morecopies.
- BoundedpreliminarycollectorCOMPLETED: rawD101/full_attempt8_preliminary.json,
  adaptedcollect_full8_preliminary.py. All4000IDsunique,cleanup/remotechecksPASS.
  Physical19874.486280712983GPU-s;4leasesallreleased;1initial+3naturalactivations;
  3970successesonoriginalfourruntimes,0replacement/0quarantineevents.
  Windows:pre_arrival48.35857119099819,arrival15755.358407963766,
  drain3976.913227008248,cleanup93.8560745499708GPU-s.
  5593resource samples,peak37106450432B,minhostavailable81964531712B;
  high/max/OOM/swap/warnings0. All132remoteUUIDpairspublished/contentverified;
  client/serverbytes306360162equal;verifiedlogical5139892912B;requestpacking0.
  Source-observation stats:requests10952,collections1049,joined9903,
  rpc_invocations4115,membership_rejections4,stale_rejections6727. These counts
  doNOTaloneprovewhyrequests waited; no causalclaim orformalCE/ranking.
- Completedprelimscopeprimelora-d101-full8-preliminary-20260928.scope,
  invocatione9fb0aa78e9a481dbeff7d013a750082,Tasks0; stillactive/emptyat16:17,
  cleanonlyafterexactidentity+emptycheck. 3/4GiB/swap0,CPUs2,3,26,27 verified.
- NOW LIVE: tmux tc-d101-full8-project; unit
  primelora-d101-full8-project-20260928.scope,
  invocation995760a38b7140bea5183f49195c6aca. At16:17:22Tasks2,
  timePID1793358/jqPID1793365,CPUaffinity2,3,26,27,3/4GiB/swap0readbackverified.
  ReusesunchangedD96streamfiltervia rawD101/project_full_attempt8.sh.
  Scope receiptfull_attempt8_projection_scope.txt;logfull_attempt8_projection.log;
  outputfull_attempt8_request_projection.json;time/exitreceipt
  full_attempt8_projection_time.txt. Partial/emptyoutputNOTfinishedorfailed;
  pollactualunit/tmux/CPUprocess. DoNOTlaunchduplicate orotherheavyjob.
  Reverified16:19:40sameunitactive,CPUprocessadvancing. ActualinputFD4points
  tofullJSON;FD3istimeoutput,NOTinputprogress. Don'tinferstallfromFD3pos0.
  Reverified16:24:44sameinvocation/Tasks2/tmuxlive;jqCPU00:07:50,RSS50068KiB,
  inputFD4position3849326592of9889140123B. Notcomplete; no duplicateparser.
- 16:34:54sameprojectionlive,inputFD4position8838045696,RSS107544KiB;
  no duplicate/heavy companion. Hostavailable106GiB,swap0,diskfree286GiB.
  Exactemptypreliminaryscope stopped16:33:51; receipt
  rawD101/metadata_scopes_cleanup_full8.log. Preparedadaptedfullcurator and
  all30timeoutfailureaudit(summarize_full_attempt8.py,
  collect_full8_failure_breakdown.py),ASTPASS; guardedcurate_full_attempt8.sh
  bash-nPASS. NOTexecutedbeforeprojectionexit0. No productionchange.
- Read-onlycounteraudit(noimplementation/test/newhypothesisaccepted):
  runner:_ieee_request_snapshot incrementsstale_rejections whenANYslot rejects
  anincomingNativeSourceSnapshot; returnsNone andrequestloopobservesagain.
  InstanceSlot.accepts_native_sources rejectsolder epoch,orsameepocholdercapture;
  owner/clockmismatch andsameepochdifferentcontentraiseinstead. No TTL/physical
  memorythresholdinthiscounter;6727isNOT6727failed/distinctuserrequests.
  Alsoobserved:sharedin-flightRPCrawviewsarestillconverted/footprint-validated
  byNativeSourceSnapshot.from_native foreachwaiterbeforeresponsefreshnesscheck.
  PotentialrepeatedpureCPUworkonly; runtimecost/timeoutcausalityUNMEASURED.
  D96shared-observation/D98selected-copydocsread. DoNOTweakenstateguards or
  implementparse-sharingbeforefullcuration/table/causalprofile/primary-sourcecheck.
- NEXT finishstreamprojection→adaptD100fullcurator/failurebreakdownusingactual
  counts→protected/sourceSHAchecks→failure/status table/doc→scopedbackup.
  Onlythenchoosenextcausalbottlenecktest; no blind Full9/7B/baseline replay.
  WarmSLO/Resident,M1M2,A1–A5,S1–S13 remainpending. No objective completion.

## D101 Full8 — superseded live monitoring and prepared-work notes

2026-09-28 14:37. Supersedes all earlier NEXT/LIVE instructions. Goal ACTIVE,
baselines PAUSED. Source2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d pushed and
remoteHEAD verified. Runtime source MUST NOT change during this run.

- Live tmux tc-d101-3b-full8; canonical3BFull4000W0attempt8. Actualsourceonly
  capacityindexdiff; configuration identical toD100 exceptthreeownedpaths.
  Main launcher results/ieee_tc/p2_backend_qualification/d101_20260928/
  run_3b_full_w0_attempt8.sh;config3b_main_config_attempt8.yaml.
- Service primelora-tc-svc-963f4bfd092f405bbbda9054be5e0cc1.scope,
  invocationa0c80b8d841744aba5fbb8b69cf62848;actual72/80GiB/swap2GiB.
  Auxprimelora-tc-aux-b948ec8b043d42ef98fe201ce8307d7e.scope,
  invocationf0356581f6aa4830bc8de65e7a0e621a;3/4GiB/swap0.
  exec_receipt allowexec/resourcegate present,watchdog active,noevent/abort.
- Remote3Bservice PID1711024 invocationdb85615b059d4ebea7406edbf85766cc;
  7B PID1711026 invocation9633ee9f3f14422fad6d271d07411954;
  monitord101full8 PID1711029 invocationc60c678495734efabedf75b2f83cf032.
  ALLactiveafterauthorizedstart; don'trestart/reconfigure/hash/cleanup duringrun.
  New3Bclock remote-process-monotonic:18c0b8ad9e2a40a596e744480615f5b5.
  Remote monitor /home/lab14/primelora_remote/tc/d101_20260928/
  remote_monitor_3b_full_attempt8.log. JournalfilenameUUID!=clockUUID; identify
  afterrun byclock,notguessedfilename. Existingpublishedcacheunchanged.
- PrelaunchbothNIC1000/full;GPUcomputeempty;hostavailable114498306048B,
  disk316649136128B,swap0;remoteavailable104441331712B,disk147311034368B.
  147protectedentries/24prelaunchsources verified. Final source13SHAsPASS.
  Initialofflineverifierfile_digest unsupported onOSPython retainedlog;
  correctedincrementalSHA passed. Prelaunchstdout INFOprefixpreservedin.log;
  canonicalprelaunch_verification.json parsedwithoutalteringfieldvalues.
- Evidence in rawD101/3b_full_w0_attempt8/launch.launch/{service.log,replay.jsonl,
  watchdog.jsonl,service_ingress.jsonl,physical_deployment/request_terminals.jsonl}.
  Latefullresultin3b_outputs_attempt8; NEVERwholeloadmultiGBJSON.
- NEXT monitor same run with boundedreads, no newheavyjob. Finish→exactowned
  remote/localcleanup→curation→status table→interpretation. Evaluatecompletion,
  stagedwaiting,recovery/supersession,actualresources; no109xend-to-endclaim.
  DoNOTredoCPU1233/cache/profile/prefixes,alterdeadline oradvancebaselines/7B.
- Firstservingcheck~14:39:27submitted/27terminal/27success/nativecontract;
  noerrors,1runtime. Service13681917952B,hostavailable100415647744B,
  high/max/OOM/swap0,noguardwarning/abort. Tinyearlyprefixnotqualification.
  LegacyterminalCE,5000msSLO andloaded0 countersarenotIEEEformalmetrics.
- Verifiedwait16:04:35: SAME serviceinvocationa0c80b8d841744aba5fbb8b69cf62848,
  activeTasks152. Submitted4000,terminal4000,success3970,nativecontract3970,
  TimeoutError30;0pending. All4000terminalobserved16:00:55; finalcensusfour
  runtimes/3naturalscaleups, noobservedreplacement. Success99.25%, NOTfullsuccess.
  nvidia-smicomputeempty16:01:47and16:02:50; noGPUjobremains,butmainPID656236
  stillR/CPUactive. At16:04:35onlythisPIDinservice,CPUtime01:21:57,25GBRSS.
  Service26116255744B,hostavailable88857026560B,diskfree316479647744B,
  high/max/OOM/swap0,noabort/warning;watchdogsample5188.
  This turn is a verified wait, not a newoptimization; runtime/remote unchanged.
  NEXT wait SAME mainprocess tofinishserialization, no restart/duplicate. Read planfull completed
  this continuingmonitor task; SHAunchanged. Baselines/7B/warm/A/Sremainpending.
- main_outcome.jsonexists31933995B, butlaunch.jsonnotyetpresentat16:02:50.
  FullJSONfirstobservedwriting~590MiBat16:04:35in3b_outputs_attempt8. Fileexists
  doesNOTmeancomplete; donotread/parse/hashwhilewriting. Actualscope/tmuxstill
  live. Waitserviceexit/launchexit,thenownedcleanup/remotejournalcopy/bounded
  preliminarycollectorandstreamprojection. Noheavyanalysisyet. Ignorelegacy
  consolesummaryCE/InfraGPU/MaxRep; finalphysicaljournalsareauthoritative.
- Replayjournalnowhasreplay_complete:N_plan=N_arrived=N_submitted=4000.
  Lastreq_03999plannedarrival88690.05531338794,client_submit88690.057797548,
  socket_drain88690.058309985. Arrivalstreamandrequestdrainended; serviceisnow
  serializingresults. DoNOTcleanup/stoppublishedservicesuntilactualmainexit.
  Postarrivalphysicaloccupancyremainsinfinalmeasurement,nottruncatedatlastarrival.
- Firstfailures: req_01685 TimeoutError terminalat88204.748153334,
  plannedarrival86404.60861233814,observedage1800.139540995864s;
  req_01808 TimeoutError terminalat88268.925795175,
  plannedarrival86462.22203282459,observedage1806.7037623504148s.
  Bothnativecontractfalse/instance_idnull; nullaloneNOTproofofdispatchstage.
  Completefailurestagesawaitfinalrequestevidence. ObservationagenotfailedTTFT.
  Keepfullrunandfailuredenominator; no performance-basedearlystop orlongerdeadline.
- Boundedlivephasejoin atmonotonic87077.949435485 preserved inrawD101/
  full_attempt8_live_phase_snapshot_87077.json;JSONsyntaxPASS. First500all500
  success;secondgroup499/500,third487/500;latergroupsalsohadsuccesses. Oldest
  pendingreq_00992age1274.0996293755s;pendingageNOTfinalrequestlatency. Source
  journalslive/unhashed,nonatomicsnapshot,NOTproof ofstage/starvationcausality.
  FollowtheseIDs afterfullresult; don'tchangetimeouts/schedulingmidrun.
  Followup15:27:32: req_00992/01055/01075/01107 nowterminalsuccess/nativecontract,
  atmonotonic87080.279304747/87202.008355711/87317.574112952/87354.963659276.
  req_01179alsosuccess/nativecontract at87482.279566966. Allfiveobservedoldest
  pendingarenowsuccess. EarlierlongwaitdoesNOTimplypermanentstarvation;
  thesecompletedwithouttimeout/cancel. StillnotFull/SLOqualification.
- Secondboundedphasejoin preserved rawD101/full_attempt8_live_phase_snapshot_87959.json,
  at87959.136814633: first1000allcomplete;group3=499/500,group4=489/500,
  group5=476/500. Oldestpendingreq_01478age1726.2634623046s. At15:34:41,
  req_01478/01508/01533alreadyterminalsuccess/nativecontract at
  87964.152081586/88020.696212701/88069.377773154;req_01685/01808notyetterminal.
  Thusobservedlongwaitingpersisted,butthese3werenottimeouts. Samecaveats:
  nofinalphase/performancequalificationorcausaldiagnosisfromlivesnapshots.
  PlanandmetricSHArechecked15:28:32unchanged;runtime/scripts/testsnoGitdiff.
- PreparedONLY/bothbash-nPASS: rawD101/stop_services_full_attempt8.sh and
  cleanup_local_aux_attempt8.sh, adaptedD100identity-scopedcleanup. RemotePID/
  invocationvaluesmatchstoredD101activationreceipt; requiresfreshcheckafterrun.
  Localauxaddsactualserviceinactive/GPUcomputeemptyguard,plusinvocation/empty
  cgroupchecks. NONEexecuted. Remote3servicesmustNOTstopwhilelocalinferenceactive.
- Offlineprojectionpreparedonly: rawD101/project_full_attempt8.sh, bash-nPASS.
  ReusesunchangedD96jqfilter,guardsexactD101serviceinactive/allGPUcomputeempty,
  absenttargetandreceipt,noclobber. NOTrunwhileinferenceactive. Aftercleanup,
  launchinbounded4GiBanalysisresourcegroup; NEVERwholeloadfullJSON. Finalcurator
  mustuseactualFull8counts,notD100hardcodedcounts. NoGPU/source/configchange.
- PreliminarycollectorpreparedONLY: rawD101/collect_full8_preliminary.py,
  ASTsyntaxPASS,NOTexecuted. AdaptsexistingD100collector (paths/clock/type),
  addsserviceinactive/GPUcomputeemptyguard;remoteUUIDcardinalityfromactual
  journalsratherthanold132;emptyquarantinelistsremainempty,notinventedevents.
  Requiresall4000terminals,cleanup/release/remotecontentchecks; failuresretained.
  Runonlyaftercleanup/correctremotejournalcopyinbounded4GiBanalysisresourcegroup.

## D101 CPU checkpoint — completed (superseded launch instructions)

2026-09-28 14:34. This section supersedes ALL previous LIVE/NEXT instructions.
Full goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/CPU job remains.
D100 complete failure evidence pushed ae8a5caffaadadad6b45a9750b26cf1ccb9b3a3c;
remote HEAD verified. Don't repeat D100 analysis/backup.

- D101 single hypothesis tested: synchronous file-capacity helper repeats an
  all-inode scan per source. Saved D100 census133sources/1199allocations shows
  original5.918/5.811/5.857s,141512statcalls; index0.059/0.051/0.050s,2calls.
  Exact outputSHA matches. Mocked recorded-device lookup/no actual file IO;
  reconstructed source/protection state, NOT measured live epoch/end-to-end.
- Only productionchange: ephemeral per-inventory ancestor/device index in
  LocalSourceReferences._file_replacement_capacity_from_inventory. Same usable
  bytes,protected/held/moving exclusion; no persistent cache or relaxedguards.
  Nine equations/profile/config/trace/deadline/remote unchanged.
- 1186CPU checksPASS55.068s +47OSguardsPASS0.216s=1233distinct; all5CPUscopes
  identity+emptychecked/stopped14:33:19. No repeated regressions needed.
- Doc D101_FILE_CAPACITY_INDEX.md and curated20260928_d101_file_capacity_index.json
  SHA b1f3c41fa3df6edf9945d5a199be9e97c6beba1896ff5b8f9df67271ddf251ef.
  Source/protected147 validated; next scopedbackup5files, then same-contract
  independent3BFull4000W0attempt8. No further speculativeoptimization.
- RawD101 results/ieee_tc/p2_backend_qualification/d101_20260928; original/index
  CPUprobe/scripts/testlogs retained, no artifact regeneration. Full8 NOTstarted.
- After backup: adapt existing D100 launch/config/remoteactivation/prelaunch
  with new owned d101/attempt8 paths and finalcommit only; compareparsedconfig
  excluding3ownedpaths. Recheckidle/resource/NIC/remoteprotectedserviceidentity,
  start existingpublishedservices/monitor once, launch canonicaltmuxFull8.
- AfterFull8:cleanup→analysis→table→interpretation; doNOTjump to7B/baselines
  untilqualification. W0historicalcomparisonlimitedtooneoptimizationsource,
  noformalpairedCI/CE ranking/claimof109xend-to-end.

## Completed D100 Full7 — failure evidence (superseded work instructions)

2026-09-28 14:24. Supersedes ALL historical LIVE/NEXT instructions.
Full goal ACTIVE/incomplete; baselines PAUSED. No inference, artifact service,
remote monitor, playback or analysis process remains. All nine completed analysis
scopes identity+empty-verified and stopped; receipts rawD100. Runtime source
6815beea8453ba56894e206a0ac4ad914ce7b3cf unchanged/already pushed.

- Full4000:2445 success/native-contract-matched,1450TimeoutError,105RuntimeError.
  All105=parent native RPC ownership unresolved; new generation not_submitted.
  Success61.125%; NO Full/SLO/numerical/ranking qualification.
- Physical22449.901937858973 GPU-s,85leases allreleased; cleanup complete.
  6117samples,peak30640865280B,minhostavailable88298586112B;OOM/high/max/swap0.
- Remote132UUIDpairs,129published/contentverified,3not_published;packing0.
  Client305066543B/server socketwritten306357477B; partialreceived retained.
- Supersession303:302native_registration,1native_gpu_source. This confirms one
  D100 sourceabsence branch, NOT its causal share of fullcompletion.
- Firstfour1initial+3natural_scaleout=2124success;later61runtimes321success;
  quarantine83allreleased. Recovery exists but remains frequent/unstable.
- Conditional2445means:TTFT828.854s,E2E835.362s,dispatch812.435s
  (=window790.440+slot19.840+release-lateness2.154),service16.420s,
  nativeTTFT0.389s,LoRAIO14.597s,parentRPCoverhead4.061s,TPOT25.736ms.
  Nestedspans notadditive; nofailedrequest latency fabricated.
- Firsttimeout offset2888.776s;firstreturnedownershipfailure3739.963s.
  Cannot assume laterreturnederrors caused the earliest waiting.
- Streamingprojection exit0,959.99s,RSS81408KiB;all4000identitysets checked.
  Curated20260928_d100_3b_full_w0_attempt7.json SHA
  ff70e419f638d94d9581c72b2ddc372c279a02d8cac97c307934095aaf3ea4a6;
  companion20260928_d100_3b_full_w0_failure_breakdown.json. Table/doc
  D100_FULL_W0_ATTEMPT7.md complete. Rawoutputs notoverwritten.
- 147protectedentries unchanged;47evidence-smokePASS2.122s. Don'trepeat.
  NEXT sourceRefs/secrets/diff/scopedbackup of evidence/docs/archive.
- THEN single causalCPUprofile hypothesis: synchronous file-owner planning/
  capacity construction blocks controller progress. Actual helper currently
  scans all1199capturedallocations once per confirmedsource and repeatsstat/
  ancestor parsing. Profile original on saved metadata (no livecache/noGPU),
  don't claim this proves all failure causality. Only optimize after evidence.
  Read primary Python asyncio/vLLM profiling docs. No blindtimeout/retry change,
  weakerownershipguards,newGPUretry,7Badvance orbaselinebefore qualification.

## Authority and frozen contracts

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before task/experiment and after compaction. No subagents authorized.
- Plan1525lines SHAfe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
  Executed authorization overrides historical plan-mode/noexecution sentences.
- Before configuration selection/comparison read FULL METRIC_PROTOCOL_FROZEN_V1.md,
  SHA5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22.
  G1=allcorrect+commonjointSLO95%,thenminimum lifecycleGPU-s/request;
  G2=commonbudget+SLO,thenP95TTFT. CE supplementary. No frozenwarm/reference
  values yet; development5000ms is NOTformalSLO.
- Preserve nine IEEE equations/core. Every optimization uses history,current
  primary-source paper/code,evidence-based hypothesis,causaltests,fullreplay.
  No hidden fallback,newweights,trace regeneration,13B,oldresults overwrite.
- User's once-only published-cache approval alreadyfulfilled D78/D80. DoNOT
  rebuild it or duplicate extracted pools. Preparation beforedeploymentnotice,
  no requestpacking. Actualtransfer/necessaryread/localprepare/competition stay
  measured; legitimate tiercache/fewerfetch advantage preserved.
- Baselines paused at9e2cf28903ed11bc8ee891dd4cc9636b94307573. D74Serverless3B
  repaired analyzed/plotted/backedup;prepared original NOTrun. Resume onlyafter
  PrimeFull:Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe. DisplayServerless.
- 7BFull,warmSLO/Resident,M1M2,A1–A5,S1–S13 incomplete;442conditional slots are
  notcompleted/uniquejobs. Numerical adapterdiscrimination pending:3B500/500,
  7B498/500zero,2/4distinctweightSHA. Nativecount!=numericalproof. Noauthority
  newweights; do not repeatedlyask. Otherimplementation canprogress.

## Completed reusable work and recent evidence

- D78/D80cache both500/500,1581215261B;D81fourway12/12both. No repeatdownloads.
- D88nativeprofiles3B368/368,7B92/92. D89initializers in
  paper_results/ieee_tc/p2_backend/d89_{3b,7b}_initialization/manifest.json.
  Use originalD88 requested_model_config PARENT,notresolvedchild;no reprofile.
- D90prefixes100/100both,cleanup/tier/nativeverified;driverglobal8/2 versus
  canonicalaggregate32/8. Notfullconcurrencyqualification. DoNOTrepeatprefixes.
- D91mainassembly4000/500both;D92planned-arrival1800sdeadline/outcome/cleanup.
- D93snapshotorder;D94physicalownerretirement;D95unsubmittedgeneration boundary;
  D96shared in-flight sourceobservation;D97ownedquarantine recovery;
  D98selected-copy identity;D99nativefallback supersession;
  D100confirmed sourceabsence terminates oldqueuejob without poisoningfreshplan.
  Complete docs/raw/evidence for each remain. These are NOToverallqualification.
- D100 final1230distinct CPUchecks passed,sourcecheckpoint6815beea pushed/remotely
  verified. Full7 complete failure evidence archived/pushedae8a5ca; no repeat.
- Earlier full failures retained: D96Full3 1417success/2581timeout/2returned,
  D97Full4 1609/2243/148;D98Full5interrupted623success+504cancelled;
  D99Full6interrupted843success+302cancelled. DoNOTcompareconditionaltruncated
  means asfullperformance orattributecrossversiondelta toonefix.

## Assets, safe execution and exact current paths

- Repo /home/qhq/serverless_llm_experiment_retry14_baseline,
  branchretry14_continuous_queue_v2,remotefaaslora_origin. Baselineownrepo/main.
- RawD100 results/ieee_tc/p2_backend_qualification/d100_20260928/;
  main outcome3b_full_w0_attempt7/launch.launch/physical_deployment/main_outcome.json;
  launch.json,parentphysicalsummary,request_terminals,replay,watchdog retained.
  Largerfullresult in3b_outputs_attempt7,NEVERwholeload. Results symlinkto
  /home/qhq/serverless_llm_experiment/results; resolvepaths carefully.
- Nativeenv /home/qhq/.venvs/primelora_vllm0300_tc_20260925,vLLM0.30/torch2.13/
  CUDA13SM86. CPUtests /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OSguards /usr/bin/python3 (pidfd). No reinstall/driverchange.
- Currentmodelconfigs3Bcap8/slots8/cpu32/gpu.72;7Bcap2/slots4/cpu24/gpu.70;
  HOST16GiB/native2/NVMe16,W5/movement3,scale2s/beta.5,min1max4.
  3Bservicebin28.114717726756954ms,7B27.4336ms—not2. Developmentonly.
- Remotealias primelora-artifact-174 lab14@192.168.4.174:8122,strictBatchMode;
  key~/.ssh/primelora_artifact_174_ed25519_20260925,0600,outsideGit.
  Token~/.config/primelora-tc-d75/artifact.token private NEVERprint.
  FingerprintSHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
- HTTP3B18080/7B18081 directbypassambientproxy. BothNIC1000/full verified
  beforeFull7;older100Mhistorical,noagentnetworkchanges. No physical100G claim.
- Cache /home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b};
  sourcepools /home/lab14/primelora_remote_artifacts/{llama32_3b_a500_v1_modelscope,
  llama2_7b_a500_v2_publicmix}. No13B. D100servicesstopped; D101stateatCURRENT.
- D100remote3Bclock403d2ad90d034daea81f3fa1c569b322 (remote-process-monotonic).
  Original3Binvocation5ea730db/7B274dce41/monitora0cbef80 andlocalaux5c9135ed
  checkedbeforestop;rawremote_stop_full_attempt7.log/local_aux_cleanup_attempt7.log.
  Don'treuseoldPIDs. JournalUUID differsfromclockUUID;correctcopy alreadydone.
- Correctindicespaper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json
  SHAbd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  20260927_7b_materialized_content_index.json
  SHAe85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  DoNOTusemisnamed7b_remote_content_index diagnostic4998filelist.
- Safetyservice72/80GiB/swap2,aux3/4GiB/swap0;actualworkerscontainedBEFOREspawn.
  ServiceCPUs4–23,28–47,aux2,3,26,27. Oneheavyjob. No global kill/ray-stop/reset,
  hostreboot or unrelatedcleanup. Diskinference150/100GiB unchanged;artifact
  nodeindependentincrementalspace rule. Latestavailable~107GiB/disk~295GiB/swap0;
  recheckbeforeanylaunch. No remotehash/restart/cleanup duringinference.

## Protection, reporting, backup and history

- Seal147protectedentries paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify via scripts.ieee_tc_preflight.verify_seal. Oldfinal_v2/figs/paperunchanged.
- NEVERstage configs/generated/lora_manifest_1000.json or unrelatedAAAIarchive,
  oldfigures,regenerate_motivation_figs.py,rejectedpreviews. Explicitpaths only.
- Eachrun cleanup→validation→status table/figure→interpretation→next.
  academic-plotting forrealfigures; failuretable isappropriate, noadvantageplot
  forfailedrun. Chinese paper-evidence reports≤60s duringwork.
- Backup testedcheckpoints/evidence tofaaslora_origin/retry14_continuous_queue_v2
  afterdiff/sourceSHA/protection/secrets/smoke. No forcepush,credentials,rawlargeGit.
- Verbatimpriorledger archived EXECUTION_HISTORY_D91_D100.md,
  SHA536c559d2d7724f1f00b021c9830082aefc328e38cf4fe5d6dec163a326e2054.
  Thisarchive containsSUPERSEDED livecommands,not fresh instructions. Earlier
  EXECUTION_HISTORY_D81_D91,D78_D80,D67_D77,THROUGH_D26,D27_D52,D53_D66 preserved.
