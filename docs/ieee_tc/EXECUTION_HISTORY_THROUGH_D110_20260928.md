# IEEE TC execution status

## CURRENT — D110 completed diagnostic; bounded request projection LIVE

2026-09-28 21:13. Supersedes ALL historical LIVE/NEXT notes below.
Previous goal turn VERIFIED WAIT; this turn PROGRESS: inference/profiler ended,
owned services cleaned, preliminary+CPU curation complete, stream projection started.
Goal ACTIVE/incomplete; baselines PAUSED. Runtime244ceb6571c3f1d04a9f24594115f3433115d619
unchanged/already pushed. No production/config/remote mutation during inference.

- Canonical3BFull4000W0 profiling4 COMPLETED but failedqualification:
  4000planned/arrived/submitted/terminal,2904success/nativecontract,
  999TimeoutError+97untypedfailure. No formalSLO/numerical/rankingclaim.
  Main/profiler exited21:08:58; final launch.json21:09:00 pass=true,
  nativeGPUrelease/servicepathremovedtrue; actualGPUcomputeempty21:09:16.
  FullJSON complete5474371755B; NEVERwholeload. Detailedprofiling notordinary
  performance; no attribution of crossversiondelta toD109.
- Physical22773.316472914972GPU-s,62leasesallreleased;59quarantinesallreleased.
  Initialfour1initial+3natural served1997success;51later successfulruntimesserved907.
  6557resourcesamples,peak31748009984B,minhost87575101440B,
  high/max/OOM/swap/warnings0. ThisdoesNOTaloneestablishfailurecausality.
- Exactremote3B/7B/monitorandemptylocalauxstopped21:09:45,receipts
  remote_stop_profile4.log/local_aux_cleanup_profile4.log. NoGPU/remotejobleft.
  Matchingclockjournaltransfers-915c0c48238446c3a9f31e42d6cf0032.jsonl
  pathrecordedremote_journal_profile4_path.txt,copiedONCEas
  remote_3b_profile4_transfers.jsonl (133lines),monitorONCE108813154B as
  remote_monitor_profile4_final.log. DoNOTrepeatcopies/cache/profile/assets.
  133UUIDpairs,132published/1not_published;bothends308673720B,requestpacking0.
  Unpublished ecommerce_lora_0093 received2313558B,archiveverifiedtrue but
  contentnotverified/published; don't turn byteequality into successfulmaterialization.
- Bounded preliminary+CPU curator completedexit0,actual3/4GiB/swap0/CPU2,3,26,27.
  Outputsfull_profile4_preliminary.json/full_profile4_cpu_samples.json.
  Scope7b1fc4ad720643c1a9b56f2342b2e592 exactemptychecked/stopped21:11:28;
  memorypeak567046144B,high/max/OOM0;metadata_scopes_cleanup_profile4.log.
  Fullprofile58314278B/559878GILsamples/99reportederrors,1.01slagwarning.
  Exclusive nearestproject(non-save_results):footprints130957(23.39%),
  sendRPC105947(18.92%),ownedinputs47149(8.42%),fileplan40659(7.26%).
  save_resultsancestry70046/import2187/other487645 mutuallyexclusive samples.
  Fractions≠walltime/latency; profiler overheadunmeasured. GNUtime wrapsprofiler
  ANDwaiteddescendants,notisolatedcontrollerCPU/RSS. No causalfixyet.
- NOW LIVE: tmux tc-d110-profile4-project; unit
  primelora-d110-profile4-project-20260928.scope,
  invocation1d9c4c260d5c46f0a9b35873687c3f0d,Tasks2. Actual3/4GiB/swap0,
  CPU2,3,26,27 verifiedbeforejq;timePID922595/jqPID922605.
  21:12:46jqCPU1:16,RSS14180KiB,inputFD4position601780224/5474371755B.
  UnchangedD96streamfilter;project_full_profile4.sh nowaddsactualenvelopechecks,
  newSHA2c64f4f8b5b6b06e931a7e29accbf1eaa9f084df037d725985e4d4aee48c2a88.
  Scope/log/time/outputfull_profile4_projection_scope.txt,
  full_profile4_projection.log/full_profile4_projection_time.txt,
  full_profile4_request_projection.json. PartialfileNOTcompletion/failure.
  PollSAMEhandles; no duplicateparser/secondheavyjob. No newGPUexperiment.
- Provisional failure/status+CPUtable D110_CONTROLLER_PROFILE_FULL.md created;
  finalcuration/SHA/backup stillPENDING. Preparedsummarize_full_profile4.py and
  collect_profile4_failure_breakdown.py reuseD101; inspectagainstactualprojection
  beforeexecute. Fullcuratorchecks56prelaunchsources+147protected; notrunyet.
  NEXT projectionexit0→boundedfullcuration→failurestage/doc update→source/checksum/
  secrets/evidencechecks→scopedbackup. Onlythenonecausalbottleneckprobe using
  history+primarysources. Don'trestartFull/extenddeadline/weakenidentityguards.
- PlanandmetricFULLread previouscontinuingturn,bothSHAunchanged21:07.
  Userdirtymanifestpreserved; onlyagentledger/newdoc changed intrackedpaths.
  Ordinary3BFull,7BFull,warm/Resident,baselines,M1M2,A1–A5,S1–S13 incomplete.

## D110 — superseded live inference notes (do not restart)

2026-09-28 19:19. Supersedes ALL historical live/NEXT notes below.
Previous goal turn VERIFIED WAIT: same D110 handles live and advancing;
D109 candidate,1256checks,evidence andscopedbackup complete.
Goal ACTIVE/incomplete; baselines PAUSED. ONE live tmux
tc-d110-3b-profile4. Runtime244ceb6571c3f1d04a9f24594115f3433115d619 pushed;
no production/config changes during this run.

- Canonical3BFull4000W0 + existing parent-only py-spy0.4.2/100Hz/GIL/threads/
  speedscope. Diagnostic NOTformalperformance. ReusedD107launcher/config;
  onlytestedD109source andnewoutput/NVMe/HOSTpaths differ. No regeneratedtrace,
  profiles,deliverycache or shortprefix; no timeout change.
- Rawresults/ieee_tc/p2_backend_qualification/d110_20260928. Launcher
  run_3b_full_w0_profile4.sh,3b_main_config_profile4.yaml,wrapper.log.
  ActualprofilerPID3804577/controller3804578 started19:18:14, bothlive/advancing
  at19:18:49. WorkerPythonrestoredtonativeenv; no recursiveprofiling.
- Serviceprimelora-tc-svc-2128ab9284fb4b24b24fa2c3d943ae2e.scope
  invocationaf03934185c5402c8a8fd12c46a9b034,actual72/80GiB/swap2GiB.
  Auxprimelora-tc-aux-e68b997a669245248f0629d42b5ead2c.scope
  invocationa2c63f6a67f045f59f2bb3537c32f568,actual3/4GiB/swap0.
  allow_exec=true/watchdog_ready; production_launch_authorized=false is
  qualificationonly,notmissinguserauthority. External4000/sourcecount4000;
  deploymentnotice101542.292337055,replayt0=101602.292337055,viewSHA
  0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37.
- Remote3B PID2023698 invocation2106531fb8a74b30bf813857a75ef95d;
  7B PID2023700 invocation091c8efd0c9c49049e7a999c61279ce9;
  monitord110profile4 PID2023703 invocationd8c3e6a0e2b147ecab5e24c01bc4a44f.
  Bothauthenticatedhealth=true/prepublished_gzip_v1. 3Bclock
  remote-process-monotonic:c6ad47d3841446a2b5da3c3dd8303cb6.
  JournalfilenameMUSTbeidentifiedbyclockAFTERrun,notguessedUUID. Monitor
  /home/lab14/primelora_remote/tc/d110_20260928/remote_monitor_3b_full_profile4.log.
- Prelaunch147protected+56sourcesPASS; receiptSHA
  2bcbeb065e7bf84e815f52c6fa9881f6f37fb0c62aff3ad08d9fbc4209110415.
  BothNIC1000/full viaactualsysfs; localethtoolnotinPATH,noinstallneeded.
  Host106GiBavailable,disk306345537536B,swap0,GPUcomputeemptybeforelaunch;
  remoteavailable104415956992B,disk147103141888B. Plan/metricreadfullunchanged.
  Prelaunchscope6b0bc5bc0548459fa1c3bf6e2c241cda exactemptychecked/stopped,
  prelaunch_cleanup.log; no otherCPUjob. Feishufileabsent,W&Bdisabled.
- PreparedONLY stop_services_profile4.sh andcleanup_local_aux_profile4.sh,
  actualIDsabove,bash-nPASS. Neitherexecuted. Reverifyidentityandmain/profiler/
  GPUexitbeforecleanup. NOremoteconfig/restart/hash/cleanupduringinference.
- NEXT monitorSAMEhandles through terminal,GPUrelease,serialization,profilerexit.
  Missing/partialprofilewhileliveNOTfailure. Then scopedcleanup->boundedcuration
  ->status/CPUstacktable->interpretation. Samplesnotwalltime; no additive nested
  fractions or unmeasuredprofileroverheadclaims. No secondheavyjob/newoptimization.
- Verifiedlive19:21:44: profiler3804577/controller3804578 sameparents/starttime,
  CPUadvancing; requestjournal76submitted/63terminal/63success/nativecontract,
  nofailedterminal yet. Fouractualinstances serving; onlyearlyprefix,notFull
  qualification orcausalperformanceproof. Watchdogsample207:service16626700288B,
  hostavailable97883426816B,high/max/OOM/swap0,no warnings/abort. Runtime Gitdiff
  empty acrossfaaslora/scripts/tests. DoNOTrestart justbecause samplingfile
  absentwhileprocesslive. Legacyconsole5000msSLO/CE/cachecountersnotfinalmetrics.
- Verifiedwait21:06:10: SAMEservice/profiler/controller live; Tasks156,
  profilerCPU26:42/controllerCPU1:31:02. All4000terminal remains
  2904success/nativecontract+999TimeoutError+97untypedfailure. FullJSON
  stillWRITING4095696461B; no finalizedlaunch/profile yet. GPUcomputeempty
  previously20:59:54; reverifyafterexit. Watchdog6391:service19163672576B,
  hostavailable99453394944B,high/max/OOM/swap0,noabort/warning.
  Disk302098771968B free; runtime/script/testdiffempty,HEAD244ceb657unchanged.
  ThisturnVERIFIED WAIT,notblocked; no duplicate inference/remote change/cleanup.
  Plan+ledgerreadFULLaftercompaction,metricV1alsoFULLread;bothSHAunchanged.
- PreparedONLY D110 postrun adaptations, syntaxchecked, NONEexecuted:
  collect_full_profile4_preliminary.py (D101terminal/resource/remotejoin),
  collect_profile4_cpu.py (D107sampleaccounting;save_results ancestry separated),
  curate_profile4_preliminary.sh (actual3/4GiB/swap0/CPUguard),
  summarize_full_profile4.py andcollect_profile4_failure_breakdown.py
  (D101fullcurators,actual2904/999/97;dynamicremotecounts/quarantines),
  existingproject_full_profile4.sh (D96unchangedstreamfilter).
  CPUfractionsusesampledenominator,NOTwalltime;samplingwarningsretained.
  Afteractualmain/profilerexitandGPUempty: exactremote/localcleanup,identify
  remotejournalbyclock andsave raw/remote_journal_profile4_path.txt,copyONCEas
  remote_3b_profile4_transfers.jsonl+remote_monitor_profile4_final.log.
  Cleanupreceiptsremote_stop_profile4.log/local_aux_cleanup_profile4.log.
  Thenboundedpreliminary+profilecuration under
  primelora-d110-profile4-preliminary-20260928.scope; wrapperrecordsactualscope.
  Finish/emptyidentitycleanup(save metadata_scopes_cleanup_profile4.log),then
  oneboundedprojection scopeprimelora-d110-profile4-project-20260928.scope,
  savefull_profile4_projection_scope.txt/log/time. No secondheavyjob orwholeload.
  Finalcuratorsmustbeinspectedagainstactualcompletedmetadata BEFOREexecution;
  preparedassertionsnotproof. Fullprojectionstillrequiredforrequestcauseclaims.
- Historicalverifiedwait20:59:40: ALL4000terminal;2904success/nativecontract,
  999TimeoutError+97untypedfailure=1096failed. SAMEserviceinvocation
  af03934185c5402c8a8fd12c46a9b034,Tasks156;profiler3804577/controller3804578
  stillliveCPUadvancing. GPUcomputeempty20:59:54;mainserializingnewfullJSON
  (236247429Bthen),launch.json/profileNOTyetfinal. DoNOTparse/hashpartialJSON,
  stopremoteearly,startsecondrun orcallthisqualified. Watchdog6007:
  service15022866432B,hostavailable99373334528B,high/max/OOM/swap0,noabort.
  Plan1525andledger1008linesreadFULLaftercompaction. Existingcleanup/projection
  scripts reread,NOTexecuted. Nextwaitsameprocesses→exit→identitycleanup→curation.
- Historicalverifiedwait20:44:05: SAMEserviceinvocationaf03934185c5402c8a8fd12c46a9b034,
  Tasks463; profiler/controllerIDs unchanged andCPUadvancing. Requestjournal
  3602terminal/2581success/nativecontract,927TimeoutError+94untypedfailure.
  ReplayCOMPLETE:N_plan=N_arrived=N_submitted=4000,4000uniquesubmitted.
  Watchdogsample5084:service29630275584B,hostavailable89540927488B,
  high/max/OOM/swap0,no warnings/abort. No restart/config/source/remotechange
  orsecondheavyjob. Thisispartialliveevidence,notperformancequalification;
  continueSAMErun,notanotherattempt. Plan+ledgerreadFULLaftercompaction;
  plan/metricSHAunchanged. 20:34:38diskfree301680930816B;
  tmuxsameactive,runtimediffemptyacrossfaaslora/scripts/tests,userdirtypreserved.
  Livejournalreadsarenonatomic;398outstandingrequestsremainindrain.
  Samecontinuingmonitor task,plan/metricSHAreverifiedunchanged20:35. No new
  experiment,cleanup,remotechange or sourcepatch during this verified-wait turn.
  Apy-spystack-readwarningat19:45:26 retained; actualcontroller/profilerstilllive
  andadvancing. Countsamplingerrorsafterexit; notrequestfailureorrestartreason.
- Read-onlyboundedlivephasejoin atmonotonic103348.180079318 (stdoutonly;
  non-atomic/liveunhashedjournals): first500=500success,second500=368success,
  third500=5success,fourth319arrived=4success. Oldestunterminated
  req_00607/00608/00630/00675/00840 hadobservedages1003.2958/1003.0309/
  999.3581/968.4654/821.0640s. At19:49:40noneoffivehadterminalrecordyet.
  AgesareNOTfinalTTFT/E2E norproofofstage/starvation; followtheseIDsafterfull
  outcome. Firstjqexpressioncompileerrorcorrectedbeforeanyanalysisoutput;
  no service/input/sourcechange. DoNOTlengthendeadlineorstartsecondrun.
  Followup19:53:56: req_00608/00607/00630 nowterminalsuccess/nativecontract
  at103574.836515345/103576.789667459/103665.592832808,respectively;
  At19:56:45req_00675alsosuccess/nativecontract(at103826.93054957).
  FouractualcompletionsdisprovepermanentstallforthoseIDs,butdoNOTestablish
  cause,acceptablelatencyorFullqualification. req_00840laterTIMEDOUT,below.
- Firstfailuresnowobserved: req_00840TimeoutErrorat104332.384430151,arrival
  102527.11608659723,observedterminalage1805.268343553762s. Then00856/00857/
  00858/00890TimeoutErrorat104352.618678979/104352.825375279/104359.718959933/
  104403.154322665;allinstance_idnull. NullisNOTproof ofdispatchstage.
  Existing1800squalificationdeadlineunchanged; no failedTTFTfabricated.
  CurrentcandidatehasNOTachievedallcorrectFull; continue SAMEdiagnostic to
  completion/profilefinalization,notperformance-basedearlystoporblindrerun.
- Newfailure/recoveryobserved20:21onward: first6untypedfalse-terminals
  req_01994/02009/01992/02008/02011/02010 bindinitial8ab6f378runtime;
  error_typenull,successfalse. Consoleaggregate reportsnativeRPCownership
  unresolved/newgeneration. Exactperrequestcauseawaitsfullprojection; doNOT
  equateerror_typenullwithsuccessorcallallfailuresTimeoutError. Read-onlymonitor
  jqnowcountsALLsuccess!=true andseparatelylabelsuntyped; no runtimemodification.
  Oldinstancesremoved,newinstancesadded;20:22:11new1db7b3a0added,subsequent
  actualad0a3a9e/cafd5426/2f7319bdserving20:25. Allreplacementcostsretained.
  Watchdogat105381.708896679:newcomputePIDs340923/346902inSAMEservicecgroup;
  foreigncompute/escaped/unresolvedlistsEMPTY; existinggraphicsPID2875untouched.
  No high/max/OOM/swap/abort. EventorderdoesNOTproveinitialtimeoutcause.
  Profilerreported1.01sbehind20:22; retainlag/stackreadwarningsandqualifyCPU
  samplinginterpretationafterfinish,doNOTchangeratesmidrunorassumezerooverhead.
- PreparedONLY rawD110/project_full_profile4.sh by adaptingD101launcher;
  unchangedD96streamfilterSHA18e3450632be803cfa7499db18e8bb4df44c6853b931ed60de561e22a0cb5eb0.
  NewscriptSHAc00e895e34ee5b75f65309aa6920dbfde32a9d6a5aee9adf671e07239ca17805,
  bash-nPASS. NOTrun. Requiresfinal launch.json,sourcefile,serviceinactive,
  emptyGPUcompute,absenttargets/noclobber. Aftercleanup runin3/4GiBanalysis
  envelope,neverwholeloadlargeJSON; retainsfailedrequests/notformalranking.
- LastcompleteFull3970/4000+30Timeout unchanged; D10711success+16cancelprefix.
  D109CPUqualificationisNOTfull-replay/SLO/numericalqualification. Ordinary3B,
  7BFull,warm/Resident,baselines,M1M2,A1-A5/S1-S13 remainpending.

## D109 — completed candidate checkpoint (superseded launch instructions)

2026-09-28 19:14. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/analysis job remains.
Candidate HEAD244ceb6571c3f1d04a9f24594115f3433115d619 committed/pushed;
exact faaslora_origin/retry14_continuous_queue_v2 HEAD verified19:14.
Seven scoped files only; userdirty preserved. Diff/source20/schema/secrets
checks PASS. Do NOT repeat backup, regression or curation. This post-push
ledger note is not a runtime/config change.

- Single D108-proven contract defect fixed: non-target same-content copy change
  expires the old path-bound objective before cost/admission/eviction. Native
  explicit no-operation receipt, strict controller identity/content evidence,
  actual stale-race handling, existing joined closure. No second optimization;
  nine equations/profile/config/trace/deadline/generation/remote unchanged.
- Final1209 related checks PASS57.384s +47 OSguards PASS0.327s =1256distinct.
  Eight new tests included; 16 malformed view/target subcases and15 constructed
  RPC boundary cases. Actual native negative receipt+real stale race separately
  checked; constructed receipt not claimed as measured concurrency. Invalid/lost
  reply retains ACTUAL file leases/deletion protection. Real changed-file bytes
  still fail. Fresh residency completes; handoff never silently replans.
- Earlier logs preserved: five targeted PASS0.424s; target2 one new test had15
  errors due non-staged fixture; receipt3 valid case expected raise instead of
  actual outer superseded return. Only fixtures/assertions corrected; production
  unchanged after initial candidate. Do NOTrepeat old D108 error-expected probe.
- Five CPU scopes exactidentity+empty-checked/stopped19:11; high/max/OOM0.
  Curator exited0; actualemptyadbf717859b848489bb3ebc69dfacd00 stopped19:13:36,
  high/max/OOM0. 147protected entries unchanged; previous50refs:45unchanged,
  fivekeys/four intendedsourcefiles changed (runner has absolute/relative keys).
- Doc/status table D109_NON_TARGET_BINDING_EXPIRY.md; curated
  paper_results/ieee_tc/p2_backend/20260928_d109_non_target_binding_expiry.json
  SHA4b05d85b9fb4fc5da393f4f3bde395f89425e779ab314615f247df70a35c4100.
  Rawresults/ieee_tc/p2_backend_qualification/d109_20260928; firsttargetlog remains
  d108_20260928/candidate_target1.log. No data/weights/cache regeneration.
- NEXT ONE same-contract
  canonical3BFull4000W0 controller-profiling diagnostic usingD107 launcher and
  existingpy-spy0.4.2. Only source/newownedpaths differ. No shortprefix/reprofile/
  deadline change/newoptimization. Cleanup->table->interpretation before next.
- CPU correctness is NOTFull/performance/SLO/numerical qualification. Lastcomplete
  Full3970/4000+30Timeout unchanged; 3B/7BFull,warm/Resident,baselines,M1M2,
  A1-A5/S1-S13 pending. Host106GiBavailable,disk286GiB,swap0; recheckbeforelaunch.

## D109 — superseded in-progress notes (do not repeat tests)

2026-09-28 19:06. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/heavy job running.
HEAD06ec406d23234c65ab105d2da51db4025ce54a82 already pushed; D109 production
candidate UNCOMMITTED in planner, native owner and runner, with transfer tests.
Do NOT repeat D108 counterexample (it intentionally expects the old error).

- Single D108-proven hypothesis: legal non-target path rebinding invalidates a
  frozen objective, not the serving system. Explicit native no-operation receipt
  before pricing/admission/eviction; strict controller identity/content proof;
  existing joined plan closure. No formula/config/trace/deadline/remote changes.
- First five targeted tests PASS0.424s: native negative receipt without mutation,
  damaged logical/GPU state remains fatal, malformed views rejected, actual
  residency reaping and fresh epoch, observation-to-RPC stale deferral then expiry.
  Raw log retained at d108_20260928/candidate_target1.log (not a new D108 probe).
  Empty target scope primelora-d109-target-20260928.scope invocation
  8c9eaf5b53a1459b9c104605f39e148b awaits identity/empty-checked cleanup.
- NEXT add adversarial receipt/current-file-content checks, then full regression
  and separate OS guards; cleanup -> curated status table -> scoped backup.
  Only then same-contract canonical Full4000 diagnostic; no blind GPU retry.
  CPU tests do not establish performance or full-workload qualification.
- Last complete Full remains3970/4000+30Timeout; 3B/7B Full, warm/Resident,
  baselines, M1M2/A1-A5/S1-S13 and numerical adapter qualification pending.
  Once-only published delivery cache already fulfilled; NEVER rebuild it.

## D108 — completed evidence (superseded next action)

2026-09-28 18:53. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/analysis job.
Runtime b7161bdcafe05117dcc9434f12ede5a7003c121d unchanged. D107evidence
30165e382965d320531347d3ba65c477d0362263 pushed/exactremoteverified18:48;
three scopedfiles, no userdirty staged. Do NOT repeat D107backup/curation/copies.
D108 evidence06ec406d23234c65ab105d2da51db4025ce54a82 also committed/pushed,
exactremoteHEADverified18:53:32;three scopedfiles,source/schema/secrets/diffPASS.
Don'trepeatD108backup/probe/curation. Thispostpushledgernoteisnotruntimechange.

- D108 actualowner/file-publication CPUprobe reproduces D107nativeguard on a
  legal NON-targetcopychange: targetcunchanged,non-targetbNVMe→HOSTsameSHA;
  frozen source domain stillcoversCPU,allnamesmatch,GPUconfirmationscomplete.
  Epoch20→29fromactualevict+demandload/acquire/release,notmanuallyeditedmaps.
  Controlbeforechange reachespricing sentinel; changedcaseValueErrorbefore
  pricing/admission,noextra preparationload/eviction/statechange;closedclean.
  This proves adefectpath, NOT the missingD107pair norFull8timeoutcause.
- ReusedAutomaticGPUReplacement/MixedOwnedPreparation+existingnativecachefixtures;
  nocuda/newweights/trace/pool. Newprobe1PASS +21existingtestsauto-discovered
  fromimportedTestCase=22totalPASS2.051s. DoNOTrepeatjusttoreducecount.
  Rawprobe_non_target_source_copy.py/probe.log preserved. No productionpatchyet.
- Probeandcurator bounded3/4GiB,swap0,CPU2,3,26,27 viauser services;exit0,
  automaticallycollected. Endcgrouppeak/eventsNOTcaptured; don't inventzero.
  CurrentprimaryvLLM0.30worker_manager revisited: serializedLRUoperations ≠
  atomicmultiRPCplan. Owner_validate_source_binding permitsnon-targetreload
  afterretirement; frozenwhole-objectivepathcheck currentlytreatsitfatal.
- Curator147protected/50unchangedsourcesPASS. Doc/status table
  D108_NON_TARGET_SOURCE_COUNTEREXAMPLE.md; curated
  paper_results/ieee_tc/p2_backend/20260928_d108_non_target_source_probe.json
  SHA5df6564602cb24393739c4389ff38d2cb98569b7874efbc52c2bb836eb07eaea.
  Rawresults/ieee_tc/p2_backend_qualification/d108_20260928.
- NEXT ONE candidate to unify legitimate non-target bindingexpiry beforecost/eviction.
  Mustcover observation→nativecommit race; explicitplan/hash/lease/epoch-bound
  no-operationreceipt+strictcontroller validation+existingjoinedclosure.
  Preserve identity/content/rank/owner/clock/GPUconfirmation/physicalguards;
  unknownRPCoutcome remainsunresolved; don'treleasefallbackrefs onbadreply.
  DoNOT merelyaddparentcheck/blanketcatch/old-planretry, parser/profilerchange,
  deadline/config/formulachange, orblindFull replay. Tests first, then fullreplay.
- LastcompleteFull8=3970/4000+30Timeout; D107interrupted11success+16cancel.
  NoFull/SLO/numerical/rankingqualification. 3BFull,7B,warm/Resident,baselines,
  M1M2,A1–A5,S1–S13 pending. Publishedonce-onlycache alreadyfulfilled;no rebuild.

## D107 — completed evidence (superseded next action)

2026-09-28 18:46. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete, baselines PAUSED. Runtime b7161bdcafe05117dcc9434f12ede5a7003c121d
unchanged. Main/profiler exited; allownedremote/localaux stopped18:37:41 after
identity checks. Curation finished exit0; exactempty scope49d8827a2a24457ba3ed8ab990bbac26
stopped18:45:44,high/max/OOM0. No GPU/remote/CPU job to restart.

- Canonical3BFull4000W0 profiling3 interrupted: planned4000,arrived67,submitted66;
  27savedterminals=11success/native-contract+16CancelledError. No fabricated
  unarrived/missing latencies. No full-replay/SLO/numerical/ranking qualification.
- Primarynative ValueError atresidency_manager.py2505: replacement epoch source
  identity or GPU confirmation changed; RPCwrap RuntimeError. Guard checks ALL
  overlappingCPUsource identities and GPUconfirmations, not D105parenttargetguard.
  Exact failed predicate/pair notsaved. Failedplan437932963ab644518524d54c838f4f27:
  11deferred/6completed/2native_outcome_unresolved(441461,497501); closeacknowledged.
  LaterreplayConnectionReset→outerprotocol_or_launcher_error; service wrapper0
  is profiler exit, NOTsuccess; replay1. DoNOTcatch/ignoreguard orblindrerun.
- Physicalobserved361.1592227360088GPU-s,fullnull,4leasesallreleased,cleanupcomplete.
  197samples,peak19968495616B,minhost93360402432B,high/max/OOM/swap/warnings0.
  12UUIDpairsallpublished/contentverified,27884609Bbothends,packing0. Correctjournal
  transfers-873b6e43e0a24ca7be0162f0c0368ea1.jsonl identifiedbyclockecec12a0,
  copiedonce+monitoronce. Remote rg absent; fallbackgrep used. No weights/cachechange.
- Finalizedspeedscope1595034B,4562GILsamples/2samplingerrors. Notfullworkload;
  2015nearestprojectstacks inmoduleimport,444canonicalprompt,335sendRPC,138fileinventory.
  Exclusive stacklabels≠wallphase; inclusivecounts overlap; timeRSS/CPUbelongs
  toprofilerwrapper. No Full8timeout attribution or performance claim.
- Curator reusedD105 withactualcounts/profileaccounting; source50+147protectedPASS.
  Doc/tableD107_CONTROLLER_PROFILE_NATIVE_CONFLICT.md; curated
  paper_results/ieee_tc/p2_backend/20260928_d107_controller_profile3_failure.json
  SHA6724b9d09a41880c5b4f97bf6d24d2f39f11edc799d46e27459ed25710b0c4c0.
  Rawresults/ieee_tc/p2_backend_qualification/d107_20260928. Don'trepeatcuration,
  remotecopies,Full8projection,D1061248checks,publishedcache,profiles orprefixes.
- NEXT source/schema/secrets/diffchecks andscopedevidencebackup (doc/ledger/JSON).
  THEN ONE actualownerCPUcounterexample for non-target source-copy changes in
  frozen nativeobjective; distinguish legitimateexpiry from identity/GPUdamage.
  Readprimarysources/historybeforefix. No secondoptimization orblindGPUretry.
- LastcompleteFull8=3970/4000+30Timeout;3BFull,7B,warm/Resident,baselines,M1M2,
  A1–A5,S1–S13,numericalLoRAqualificationremainpending. Host106GiBavailable,
  disk286GiB,swap0at18:43; recheckbeforelaunch.

## D107 — superseded live notes (do not restart)

2026-09-28 18:34. Supersedes ALL historical live/NEXT notes below.
Previous goal turn PROGRESS: D106 candidate+1248checks+curation+backup completed.
Goal ACTIVE/incomplete; baselines PAUSED. ONE live tmux tc-d107-3b-profile3.
Runtime b7161bdcafe05117dcc9434f12ede5a7003c121d already pushed/verified;
no production/config strategy changes during inference.

- Canonical3BFull4000W0 plus existing parent-only py-spy0.4.2/100Hz/GIL/threads/
  speedscope. Diagnostic NOT formal performance ranking. Reused D105 launcher/
  D103 wrapper semantics, existing published artifacts/trace/profile. Only D106
  source and new output/NVMe/HOST paths differ; no shortprefix/newdata/reprofile.
- Raw results/ieee_tc/p2_backend_qualification/d107_20260928; launcher
  run_3b_full_w0_profile3.sh/config3b_main_config_profile3.yaml. Actual profiler
  PID3304221/controller3304222 started18:33:07, bothlive/advancing18:34:12.
  Wrapper restores native Python for workers, no recursive profiling.
- Service primelora-tc-svc-3d4d08076a4947f68fd7101dc7c84bb9.scope,
  invocation56ba387dcd3344c4a48bd833867fb6bd; actual72/80GiB/swap2GiB.
  Aux primelora-tc-aux-dea14a6c7b92485f8312bfc4b7e5e589.scope,
  invocation83d938039ada4fc0b27445072be40289; actual3/4GiB/swap0.
  exec_receipt allow_exec=true/watchdog_ready; production_launch_authorized=false
  denotes qualification-only gate, not missing user execution authority.
  External replay4000/sourcecount4000, deployment_notice98835.834135072,
  replay_t0=98895.834135072. ViewSHA0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37.
- Remote3B PID1987507 invocation6173ac37740f49aab5cde096ff9e20fa;
  7B PID1987509 invocation3ea35330573e4ff990414b817d866368;
  monitord107profile3 PID1987512 invocationabf98a2321df455fa20b5f592f7a2e62.
  Both authenticatedhealth true/prepublished_gzip_v1. Remote3Bclock
  remote-process-monotonic:ecec12a020cf4688acb77b62e88d81e6. Journal filename
  must be identified by clock AFTERrun, not guessedUUID. Monitor
  /home/lab14/primelora_remote/tc/d107_20260928/remote_monitor_3b_full_profile3.log.
- Prelaunch147protected/50sources PASS; receiptSHA
  06ca4321ceeb186bb98b191ddf5ed7c30cde53be3d8dffa94971fa6507a78dce.
  Localavailable113759309824B,disk306394681344B,swap0,GPUcomputeempty;
  remoteavailable104442892288B,disk147126325248B. ActualrouteNICeno1np0→eno1,
  both1000/full. Initial guessedlocalNIC query nonexistent, corrected via route;
  no NICconfigurationchange. Plan/metricSHAunchanged; Feishu absent/W&Bdisabled.
- 18:34:25firstinstance added,serving4000.18:34:28oneactualEngineCorePID3312254
  onGPU,service4304912384B,host109194498048B,high/max/OOM/swap0,no warnings/abort.
  Startup/progress only, NOTfullsuccess/native-numerical/SLOqualification.
- Prepared ONLY stop_services_profile3.sh andcleanup_local_aux_profile3.sh;
  actualIDs above, bash-nPASS. NEITHER executed. Reverify allidentities and
  main/profiler/GPUexit before cleanup. Do not restart remote or scan/hash pools
  during inference. No second heavyjob/sourcepatch/install/compression.
- NEXT monitor SAME handles through terminal, GPU release, serialization and
  profiler exit. Missing/partial profile while live is NOTfailure. Then scoped
  cleanup→boundedcuration→status+CPUstack table→interpretation. Speedscopeordinal
  isnotwalltime; don't add nestedpercentages or claim measured samplingoverhead.
- D106backup/tests complete, doNOTrepeat. LastcompleteFull8=3970/4000+30Timeout;
  D103/D105no usableprofile. Ordinary3BFull,7B,warm/Resident,baselines,M1M2,
  A1–A5,S1–S13,numericalLoRAqualification remainpending. Goalnotcomplete.

## D106 — completed candidate checkpoint (superseded next action)

2026-09-28 18:28. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/CPU job. Candidate HEAD
b7161bdcafe05117dcc9434f12ede5a7003c121d committed/pushed; exact remote HEAD
verified18:28. Six scoped files only; userdirty preserved. Schema/source22/
secrets/diff checks PASS. Do NOT repeat backup, tests or curation. No D106 owned
units or GPU compute remain. Post-push ledger note is not a new runtime config.

- Actual file/native owners reproduced D105's old fatal guard in two subcases:
  SHA-identical HOST/NVMe copies with native HOST-only or GPU-ready state. This
  proves a defect path, NOT the missing D105 exact failing pair or timeout cause.
- One candidate: require complete same-owner native ID/name/rank/source evidence
  and both current file publications with frozen content SHA; expire old path-
  bound objective through existing joined closure. No relabel/load/success or
  retry of old objective. Three boundaries: staging, native-HOST queue, GPU action.
  Missing/corrupt evidence remains error; no formula/profile/config/trace/deadline/
  remote change. Real modified-file publication rejection and25 malformed cases.
- Actual mixed executor/residency reaping, old close and fresh next epoch PASS;
  demand-loaded object remains bound to HOST, only legitimate demand load occurs.
  Handoff does NOTsilently replan. Prior sibling-failure/unacknowledged-close
  checks retained. No parser/profiler lifecycle or second optimization.
- Final1201 related checks PASS56.176s +47 OSguards PASS0.155s =1248distinct.
  Earlier logs retained: target6PASS; integration fixture errors then19/20;
  handoff corrected remaining-capacity fixturePASS; initialregression1200/1201
  due wrong expected exception type for actual modified file. Production fix
  unchanged after initial candidate; final suite includes allnewchecks.
- Eight CPU scopes exactidentity+empty-verified/stopped; high/max/OOM0. Curator
  exit0,147protected and22source refs verified; sourceSHA recheckPASS18:27.
  Curator df9138cac7c64f11b440a9a27ceabdcf also exactempty-verified/stopped18:26:48.
  Host105GiBavailable,disk286GiB,swap0 then; recheck beforeinference.
- Doc/table D106_CONFIRMED_ALTERNATIVE_COPY.md;curated
  paper_results/ieee_tc/p2_backend/20260928_d106_confirmed_alternative_copy.json
  SHA51184ccb0a8fa4aa4955c49d161e43ffc92a426bcdd8553c39f489b100fb3cd3.
  Raw results/ieee_tc/p2_backend_qualification/d106_20260928; doNOTrecreateassets.
- NEXT ONE same-contract canonical3BFull4000W0 controller-profiling diagnostic using D103
  tool/wrapper and D105 launcher; only D106 source and newownedpaths differ.
  Reusepublishedcache/trace/profile, no shortprefix or newoptimization. Diagnostic
  notformalranking; finish→cleanup→curation/table→interpretation. No blindretry.
- LastcompleteFull8=3970/4000+30Timeout; D103/D105profiles unusable. Ordinary3B,
  7BFull,warmSLO/Resident,baselines,M1M2,A1–A5,S1–S13,numericalLoRAqualification
  remainpending. CPUcorrectness isnot full-replay or performancequalification.

## D105 — completed failure evidence (superseded next action)

2026-09-28 18:04. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote/analysis job. Runtime
00f0f8da3061f89cfbb857200da1ac9b20fd1560 unchanged; NO new production fix.
Evidence HEAD2b16bb25c449be081673b1e187a53eb7755b2c8b committed/pushed andexact
remoteHEADverified. Three scoped doc/ledger/curated files only; userdirty preserved.
Schema/source19/secrets/diffchecksPASS; prior1241checksreusedwithunchangedsource.
Do NOT repeat backup/curation/tests. No owned D105 units/GPUcompute remain18:04.

- Canonical3BFull4000W0 controller profile2 interrupted: planned4000,arrived59,
  submitted58;24terminalrecords=9success/native-contract+15CancelledError.
  Physical9noninterrupted/15interrupted. Preserve missing/unarrived outcomes;
  do not invent remaining request latencies or count tiny prefix as qualification.
- Primary ValueError 'native HOST target source identity changed', controller
  _queue_ieee_native_host_preparation execute line17137. Checks native adapter
  name/path against task name/selected path BEFOREfile-reference acquisition.
  Failing snapshot and exact pair not saved; neither exact adapter nor legal
  cross-tier copy mismatch versus identity damage proven. Different fromD103.
  D104 still NOTfull-replay-qualified. No guard bypass/catch-and-retry.
- Plan94a17251eb0f4daba317658ae372257c failed,4deferred/1completed attempts,
  close acknowledged. Six nativeHOST preparationrecords(no file-heldattempts),
  allrequestedNVMe paths; doesNOTprove which failed/why or that allsix failed.
  LaterreplayConnectionReset→outerprotocol_or_launcher_error,service-15/replay1.
  speedscope ANDtime0B: NO usableprofile/CPU attribution, don't claim completion.
- Observed356.32732690899866GPU-s only;fullnull;4leasesallreleased,cleanupcomplete.
  190resourcesamples,peak19528437760B,minhost94120902656B;high/max/OOM/swap0.
  11remoteUUIDpairsallpublished/contentverified,25563136Bbothends,packing0.
- Actualmain/profiler/GPUexitverified; exactownedremote3B/7B/monitor andlocalaux
  stopped17:58:54. Matchingjournalbyclock copiedonce as remote_3b_profile2_transfers.jsonl
  fromtransfers-1f3f211067cd4fb49cf27a5bb7925070.jsonl. Monitorcopiedonce4.98MB.
  Curator147protected/44sourcesPASS; exactemptyc77e206e6acd4da1901b9a09baad0990
  scopestopped18:02:15,high/max/OOM0. Host106GiBfree/disk286GiB/swap0then.
- Doc/table D105_CONTROLLER_PROFILE_SOURCE_CONFLICT.md;curated
  paper_results/ieee_tc/p2_backend/20260928_d105_controller_profile2_failure.json
  SHA3e2fcd6c1768dd86bac29af5d08d6ec918c91a8b9a2fb5439817982a5790f933.
  Raw results/ieee_tc/p2_backend_qualification/d105_20260928. DoNOTrepeatcuration,
  copies,cache/profile/prefixes or unchangedD1041241checks.
- NEXT ONE actualowner CPU
  counterexample for target source-copy conflict; read D98/D104history+primary
  vLLM source semantics. Distinguish valid cross-tier samecontent from damaged
  name/content; don't infer rootcause just from path. No blindGPUreplay/profiler
  lifecycle patch or second optimization before causal evidence.
- Ordinary3BFull,7B,warm/Resident,baselines,M1M2,A1–A5,S1–S13 pending; lastcomplete
  Full8=3970/4000+30Timeout. No SLO/numerical/ranking qualification.

## D105 — superseded live notes (do not restart this run)

2026-09-28 17:55. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. ONE live canonical3BFull4000W0
diagnostic, tmux tc-d105-3b-profile2. Runtime HEAD00f0f8da3061f89cfbb857200da1ac9b20fd1560
already pushed/verified; no source/config strategy changes during inference.

- Raw results/ieee_tc/p2_backend_qualification/d105_20260928. Reused D103
  wrapper/tool (py-spy0.4.2,100Hz,GIL,threads,speedscope), only D104 source and
  new owned output/NVMe/HOST paths differ. Same config confirmed; no reprofile,
  generated trace/cache or short prefix. Detailed profiling NOTformal ranking.
- Service primelora-tc-svc-cf9bacc2da4e4e78b640b701a3775d46.scope,
  invocation32c0f865d66e4c1ca3dde42e0d3d7aa4;72/80GiB,swap2GiB verified.
  Aux primelora-tc-aux-e154037b3d844ff3b55506be41a52748.scope,
  invocation989755a57ee142dbbf582d809d2d72d4;3/4GiB,swap0 verified.
  allow_exec=true,watchdog_ready present. ActualprofilerPID2872037/controller
  PID2872038, bothstarted17:54:20; wrapper resets worker Python so no recursive
  profiling. Startup ongoing17:55, no warning/abort/high/max/OOM/swap events.
- Remote3B PID1955796 invocation464a10760b704399afb7b32ceb068cb8;
  7B PID1955798 invocation959d2282c1e74ef3b489ab54afb9d416;
  monitord105profile2 PID1955801 invocation9f0655bfbfc34a229c17159a8380b06b.
  Remote3Bclock remote-process-monotonic:8b3d35bad8d14f0f99bc7a07e8d4d1ba.
  Monitor /home/lab14/primelora_remote/tc/d105_20260928/remote_monitor_3b_full_profile2.log.
  Correct transfer journal MUST be identified by clock AFTER run, not guessedUUID.
  Immediate first health probe preceded listening: connectionrefused/emptyfile
  retained. SamePIDs/invocations subsequently listening and both authenticated
  health verified; no restart. *_ready.json holds successful receipts.
- Prelaunch147protected and44source refs verified; prelaunch_verification.json
  SHA2f3a611ecdb9018755180e68af64868fd86092a07beedf183a68efb919a2cc45.
  Localhostavailable113900486656B,disk306427527168B,swap0,GPUcomputeempty;
  remoteavailable104480198656B,disk147172294656B. BothNIC1000/full.
- NEXT monitor SAME run to terminal, GPU release, serialization and profiler exit.
  Profile only written at exit; absent while running NOTfailure. No second heavy
  job, source modifications, remote hash/config/restart/cleanup during inference.
  Then exactowned cleanup→boundedcuration/status+CPUstack table→interpretation.
  Speedscope sample index is NOTwalltime; separate business/serialization by stack
  ancestry, don't add nested percentages or claim measured profiler overhead.
- D1041241checks/backups COMPLETE; do not repeat. Full8 remains3970/4000+30Timeout;
  D103 profile absent. No Full/SLO/numerical/ranking qualification. Final ordinary
  nonprofiledqualification,7B,warm/Resident,baselines,M1M2,A1–A5,S1–S13 pending.

## D104 — completed checkpoint (superseded next action)

2026-09-28 17:45. Supersedes ALL historical live/NEXT notes below.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote job. D104 CPU work complete;
candidate HEAD 00f0f8da3061f89cfbb857200da1ac9b20fd1560 committed/pushed, exact
remote HEAD verified. Nine scoped source/tests/evidence/doc files only; userdirty
preserved. Do NOT repeat backup/CPU probes or recreate delivery assets.

- Two actual owner/queue counterexamples reached D103's old line2505. Only
  source-set coverage failed: legitimate demand added an ID after registration,
  with original source identities and GPU confirmations valid. This establishes
  a defect path, NOT the exact missing predicate in D103's failed RPC.
- One candidate fix: explicit expired source-domain outcome before victim cost
  indexing/eviction; strict complete same-owner later-state evidence; old plans
  close through existing supersession/queue lifecycle. Worker cost calculation
  now occurs AFTER the same serialized owner's validity checks. No formula,
  profile, config, deadline, trace, generation or remote change. No blanket catch,
  old-objective retry, parser sharing or profiler lifecycle changes.
- Native identity/confirmation damage, incomplete observations and malformed or
  unknown RPC receipts still fail; invalid replies do not release file fallbacks.
  Confirmed GPU-ready target reuse remains legal despite unrelated source growth.
  Mixed worker+actual residency reaping tested; fresh valid plan executes.
- Final1194 related tests PASS54.583s; separate47 OS checks PASS0.102s =1241
  distinct checks. Final affected277 PASS8.213s included, not extra repeats.
  Prior17/275/1193 intermediate logs retained, not final-version substitutes.
- Seven CPU scopes exactidentity+empty-verified/stopped; high/max/OOM0.
  Curator completed exit0,147 protected and21 source refs verified. Raw
  results/ieee_tc/p2_backend_qualification/d104_20260928. Status table/doc
  D104_PREPARATION_SOURCE_DOMAIN.md; curated
  paper_results/ieee_tc/p2_backend/20260928_d104_preparation_source_domain.json
  SHA01dfa6a7eef53bb01f6998fff5a2f793c0fd218d63e424ed73ec8d82f384d6fd.
- Curator scope6ba5406cfcaf4ea691a155c7ab05c6af exactidentity+empty-verified
  stopped; raw curator_cleanup.log. No D104 live scopes. Host106GiB available,
  disk286GiB free,swap0 at17:45. Recheck before inference, not a future guarantee.
- NEXT ONE same-contract canonical3B
  Full4000W0 controller-profiling diagnostic using unchanged D103 wrapper/tool.
  Only D104 source + new owned paths differ; reuse trace/cache/profile. Still
  diagnostic, not formal performance qualification. Keep standard resource gates,
  external replay, actual GPU release and independent remote monitoring. No short
  prefix, no second optimization, no duplicate run; cleanup→table→interpretation.
- Full8 remains3970/4000 +30TimeoutError. D103profile absent; CPU cause not
  established. No Full/SLO/numerical/ranking qualification. 7B,warm/Resident,
  baselines,M1M2,A1–A5,S1–S13 pending. Final non-profiled qualification still needed.

## D103 — completed failure evidence (superseded next action)

2026-09-28 17:20. Supersedes ALL historical live/NEXT instructions below.
Goal ACTIVE/incomplete,baselines PAUSED. NoGPU/remotejob. RuntimeFull8 source
2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d unchanged. D103evidencecheckpoint
587ce9f476cbeadb21bccaa283b77c5fb13d1382 pushed; exactremoteHEADverified.
Onlydoc/ledger/curatedJSONcommitted; no userdirty or runtimechange. D102checkpoint
e6c783110046842a9aaa0cd52b96e22de8c040de also pushed. DoNOTrepeatbackups.

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
- NEXT ONE CPUcounterexample usingactual
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
