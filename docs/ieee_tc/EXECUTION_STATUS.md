# IEEE TC execution status

## CURRENT — D114 native async transport: tested candidate, backup pending

2026-09-29 00:21. Goal ACTIVE/incomplete; baselines PAUSED. Previous goal turn
D113 was PROGRESS. This turn PROGRESS: actual TCP causal probe,one transport
candidate implementation,227native+288legacytests,retained-payload before/after,
phase-status table/document and source/protection verification. NO GPU/remote
experiment started. Candidate full validation remains REQUIRED; not a claim
that D112's whole waiting time is explained or end-to-end performance improved.

- ParentHEAD2943c7187195b3f9afbf6f06c0bc283d370ff27f,runtimebefore5442e62.
  Candidate ONLY scripts/run_all_experiments.py plus two tests. RuntimeSHA
  b2e4f65e4a20c4c56f27df9fff88075c17570021e3035ed83634badef75cdd18.
  Native I/O now nonblocking socket connect/send/recv/close; legacyunchanged.
  Same30sconnect/300soperation/8MiBprotocol,channelpool,owner guards/progress
  ordering/cancelunknown/no-blindretry. No capacity/timeout/formula/config tuning.
  parent_rpc_transport=native_async_socket_v1; threadresume0 denotes no thread
  stage, not zero whole controlwait. JSON encode/decode not separately optimized.
- Originalactualproxy+TCP,event-held32generationreplies saturate default32
  executorworkers; newcontrolopenfuturequeued/notrunning,workerhasnotseen it.
  Releasing generation permits progress. Onegenerationcontrolproceedsnormally.
  AftercandidatecontrolcompletesBEFORErelease at1and32; noexecutorsubmission
  incontrolprobe. Separate32generationregression forbidsANYexecutor use.
  No timedinjection/newGPUowners/weights/trace. Conditionssynthetic dependency
  fixture, NOT D112saturation proof or walltime acceleration measurement.
- RetainedD88actual8adapter1792HOSTallocation reply1328413B; complete
  nontransportbody SHA4efdcb0d7483c21656f6132caadd535f8ad1c9d5bb35791d622eddf879c461c4
  identical before/after. OriginalsourceSHA7377ed4e706e1eb81db79d89d780d1205d04e76345f5ede8b8a6c9c81397e953.
- FirstCPUprobe failed onlycomparisonincludingoldtransporttiming. Prior
  source/logretainedattempt1_sources.tar.gz; corrected compares completebody
  excludingonlytransporttiming. Beforeprobe source retainedbefore_sources.tar.gz.
  Rawafter production_optimization:false is stale templatelabel; curatedcorrects
  candidatemetadata explicitly, no rawobservations overwritten.
- Native227PASS3.224s,existingbasic288PASS28.840s. Candidateprobe10.83swall
  includingimports,peakRSS1686928KiB. Allcompletedscopeshigh/max/OOM0.
  Fourprobe/testscopes verifiedexactownedemptyandstoppedbycurator.
- 62priorfrozenrefschecked;3expectedchangedrefs(twoaliasesofrunner plus
  request_lifecycle),allothersunchanged.147protectedunchanged. Candidateprobe
  sourceSHAsverified;beforecodecheckedagainstparentGit andarchivedprobescript.
  MetricV1/sourceplanunchanged. No oldfig/resultwrites,cache rebuildorGPUrun.
- Curatedpaper_results/ieee_tc/p2_backend/20260929_d114_native_async_transport
  summarySHAa76afd819eee82d9974770e0d2634fd180bf57f0e38b1a4ffd5f836343901d5a;
  36KiBsmall_sources.tar.gz17members verified,companionSHA256SUMS.
  Rawresults/ieee_tc/p2_backend_qualification/d114_20260929;
  DocD114_NATIVE_ASYNC_TRANSPORT.md fulltable/sourcebasis/limits.
  Curationscope5b37c5bc383d4ff8ab2941e353f4f812 exactemptychecked/stopped,
  events0. No experiment/analysisjob remains; doNOTrepeat completedD114 checks.
- NEXT preflight/backup and ONE ordinary canonical3B4000W0 Full10 candidate
  replay, same D112trace/subset/config/D89initializers/remote publishedcache.
  No py-spy or furtheroptimizer. D113 E2Econtroller-vsouterboundary mapping
  remains pending formalmetricreconciliation; don't silently overwriteoldE2E.
  NoD115config/path/remoteactivation preparedyet. Afterfulljudgecandidate,
  continuePrimeFull/7B,thenwarm/Resident andbaseline mainline. Allformalmatrices,
  ablations/sensitivities/numericadapterqualification remain pending.

## D113 completed reference — offline control occupancy backed up

2026-09-29 00:05. Goal ACTIVE/incomplete; baselines PAUSED. This turn PROGRESS:
extended existing control analyzer,10 targeted tests,complete D112 timeline audit,
phase table/documentation, source/protection verification and owned cleanup.
NO new GPU/remote run; production5442e62 unchanged. D113 uses D112 bounded
projection only, never repeats9.49GB extraction or reconstructs artifact cache.
Ten scopedfiles committed/pushed2943c7187195b3f9afbf6f06c0bc283d370ff27f;
exactremoteHEAD verified2026-09-29 00:06:30. Userdirtymanifestnotstaged.
DoNOTrepeat D113 analysis/tests/verification/backup. This postpushledgernote
is not a production/configuration change; no D113 work remains.

- 4000native-contract requests,commonclock/ID/replica/order complete.
  Observation4471.156393s. Mean/P95 seconds:
  pre-gate247.747071/474.045999;gate→source14.023919/37.794468;
  source→native9.538420/50.207334;native→last3.555022/8.587821;
  last→controller3.088190/13.204306;controller→outerterminal2.550083/7.827683.
- Mean occupancy gate→terminal29.303948, native dispatch→last3.180405;
  peak32/25. Every replica nativepeak8. Gateends at outerterminal afterrelease,
  hence upper envelope, not exact instrumented release. Request-s≠GPU-s.
  4412resourcesamples;4071with pregatebacklog;3625gateoccupancy32;
  samplemeanheldGPUutil22.564143%. Not timeweighted/kernelcausal evidence.
- Reject simplistic totalgate8/GPUcompute-saturation interpretation. No safety
  guard removed, no production change/accepted optimization. Controllerterminal
  vs outerterminal mean2.550083s gap exposed; formal c_r boundary mapping still
  needs ingress-notification audit, not silently relabeling D112 originalE2E.
- First test failed solely due legacy top-level matplotlib import, no analysis.
  Lazy import inside legacyplot; no newenvironment/install. Second10PASS0.054s;
  analysis1.40s peak93456KiB. Both scopes exact-ownedemptyclosed,high/max/OOM0.
  Verification scope exited0 and exactempty identity
  2e201dcc855847fcbd3c381e12c6763d stopped00:05:21,events0;
  no GPU/remote/analysis job remains.
- 62frozenproductionsources,5analysissources and147protected unchanged;
  plan/metricV1 exactSHA unchanged. SummarySHA
  f9463cc0d4d8c837d9eeba5f381990eb8f6a48cf47934951cde799ac60eace30;
  verificationSHA99f899582b38aaab781b3f388d2e5f3454ada06bf91d73f57bd54ab5cf41be42.
  Curateddirpaper_results/ieee_tc/p2_backend/20260928_d113_control_occupancy;
  rawresults/ieee_tc/p2_backend_qualification/d113_20260928.
  small_analysis_sources.tar.gz preserves6smallscript/log/scopereceipts,
  each memberSHAverified. DocD113_CONTROL_OCCUPANCY.md fulltable/caveats.
  CSV uses csv.DictWriter standardCRLF; defaultgit whitespacecheck flagged
  lineendCR,not changed data. Check with explicitcr-at-eol plus normal
  blank-at-eol/blank-at-eof/space-before-tab PASS; no resultrewriting.
  Bundle.sha256 checkPASS; scopedsecretscheckPASS; userdirtymanifestnotstaged.
- NEXT ONE bounded actual-RPC CPU probe using saved realnativepayload:
  test whether serialization/decode/progress-confirmation sharing execution
  resources produces control waiting. Separate actualsocket/loop/queue timing,
  do not injectlongsleep or call simulatedtiming E2Egain. If unsupported,
  archive and inspect sourceconflict/preparationwait. No blindnextGPUreplay.
  PrimaryPythonasyncio andvLLM0.30worker docs reread/webverified; nativeworker
  mutations rely on singlethreadcore, don't parallelize them unsafely.
- Outstanding:7BFull,warm/Resident,numericadapterqualification,baselines,
  M1/M2,ablations/sensitivities. OncepublishedcachefulfilledD78/D80,no rebuild.

## D112 completed reference — ordinary 3B Full9 fully backed up

2026-09-28 23:47. Goal ACTIVE/incomplete; baselines PAUSED. This turn PROGRESS:
projection, native timing/readiness checks, final tables, source/protection/test
verification and exact-owned cleanup completed. NO experiment/analysis remains.
Runtime5442e62fb977a6bd9a90ca4220e1b7d0872e3d2a unchanged/already pushed.
DoNOTrepeat D112 replay, projection, preliminary, curation,70checks or transfers.
Six scoped evidence/doc/source-bundle files committed/pushed as
c03a1b1b4660b8b4bec825918a56e9eb9ec7fb56; exact remoteHEAD verified23:49.
Userdirty manifest notstaged. No remaining D112 work; doNOTrepeat backup.
This post-push ledger note is not a production/configuration change.

- Canonical3B4000W0: planned/arrived/submitted/terminal/success/nativecontract
  ALL4000;0failure. Not numerical adapter qualification or formalSLO/ranking.
  All4000ID/prompt/native-token checks; timing maxerror0.992767ms,TPOT/error
  dispatch/service decomposition0. Native selectedsource4000/4000 confirmed
  pre-generation with clock/tier/order matching;GPU2460/HOST652/NVMe743/Remote145.
- Mean/P95 TTFT271.850703/507.604630s,E2E277.952591/515.315555s.
  Mean dispatch261.770959s =window245.487414+slot14.023888+arrivalrelease2.259656.
  Mean serviceTTFT10.079744s =pre-native9.538420+native.541324.
  Native decode3.013698s,worker-completion.051245,worker→controller3.036945s.
  TPOTmean32.354962ms/P9575.468296ms. Nested spans and P95 never summed.
  D110profiler differs; no single-change causal gain/pairedCI claim.
- Physical17920.27838478402GPU-s,4leasesallreleased,1initial+3natural,
  0quarantine/replacements.5101resource samples,peak36525568000B,
  minhost82589007872B,high/max/OOM/swap/warnings0.
  GPUreleasedbefore JSONserialization; postrelease serialization not GPU time.
- Remote132UUIDpairs,132contentverifiedpublished,306360162Bbothends,packing0.
  Journaltransfers-ea079e4e3eaf47bebdde42dc4fae0fec.jsonl matchedclock
  396cebb9473941329267a80b0833fd12; copiedONCEplusmonitor.
  Remote3B/7B/monitor andemptylocalaux stopped23:22:30; publishedcacheunchanged.
- Full9493608892B immutable; boundedD96jq projection1170.78s,
  peakRSS137856KiB,exit0; projection33837876B. Neverwholeload/repeat.
- Firstcuration passedrequestchecks thenfailed one nonexistent legacy source
  filename; originalscript SHA33ee5055c6228a17d2565d747bb5de5810d89c4f19107e1d5467ab0df8a63e7d
  retained as summarize_full_full9_attempt1.py with wrapperlog. Second only
  corrected source reference, no data/formula/production changes; SUCCESS.
- 62frozenrefs+33curatedsources+147protected unchanged;70 evidencechecks
  PASS2.404s. Sourceobservations12783requests/1035collections/11748joined/
  4057RPC/4054parses,8507stale,3membershiprejections. Not failure request counts.
- All5postprocessscopes exactempty cleaned;metadata earlier/projection/
  curate1/curate2/evidence at23:46:43,events0. No GPU or remote/analysis jobs.
  Logs metadata_cleanup_full9.log/analysis_scopes_cleanup_full9.log.
- Raw results/ieee_tc/p2_backend_qualification/d112_20260928.
  Curated paper_results/ieee_tc/p2_backend/20260928_d112_3b_full_w0_full9.json
  SHA222ae5b62e5a75fb81f700b08276eb9e3586a5768f6653b4e280f55a74d4290d;
  evidence verification20260928_d112_evidence_verification.json
  SHA90759c615db2843d995e6aefb601f479f231da0da715f65a4f1b6507343a6de8.
  DocD112_FULL_W0_FULL9.md has complete status/timing/readiness tables.
  13KiB20260928_d112_analysis_sources.tar.gz preserves12small run/analyzer
  sources (including failedcuratorandD96filter), no rawlarge/weights/credentials.
  Companion.sha256:13entries and eacharchive member verified.
- NEXTinspect dispatch-window
  progression against this complete run before choosing ONE causal bottleneck
  probe. Do not remove identity/physical guards or raise deadlines as workaround.
  No blind nextGPUreplay. Return to Prime IEEE Full mainline, not baseline work.
  7BFull,warm/Resident,formalmatrices/ablations/sensitivities and numericaladapter
  qualification remain pending. Once-only publishedcachefulfilledD78/D80;
  NEVERrebuild. Sourceplan andmetricV1 SHA unchanged/fulltextread retained.

## D111 completed reference — shared parser backed, Full pending at that point

2026-09-28 21:50. Goal ACTIVE/incomplete; baselines PAUSED. No GPU/remote run
started this turn. D111 modifies ONLY scripts/run_all_experiments.py production
collection parsing; tests/test_ieee_tc_request_lifecycle.py adds/extends checks.
Runtime/evidence5442e62fb977a6bd9a90ca4220e1b7d0872e3d2a committed/pushed;
exact faaslora_origin/retry14_continuous_queue_v2 HEAD verified21:50.
Base HEAD6593db7, runtime parent244ceb657. Candidate not Full-qualified yet.
Five scoped files only; diff/source26/protection147/tests275/secrets checks
passed. Userdirty manifest notstaged. DoNOTrepeat D111 probes/tests/backup.
This goal turn PROGRESS. This post-push ledger note is not a runtime change.

- Same D102-qualified actual D88 payload: oneowner/8adapters/1792HOSTallocations,
  actual runner collection with explicit fake RPC/file/NVML boundaries. First
  probe failed ONLY frozenset summary serialization; raw failed source/log kept.
- Before2:32waiters share1RPC but parse32times. Candidate:1RPC/1parse;
  all normalized routing row hashes and livecounts0..31 identical, sourceSHA
  unchanged, next request freshRPC. No long-lived/finished-view cache.
- Three32-waiter CPU wave timings before293.484/87.709/87.742ms,
  after6.185/6.455/6.020ms. Diagnostic only, not GPU/E2E/timeout causality/CI.
- Parser moved inside same in-flight collection; immutable return. Eachwaiter
  still checks membership/currentepoch/wholecollection beforepublication and
  recalculates files/counts/feasibility/cost. Nativeguard/formulas/timeoutunchanged.
- 275targetedtests PASS6.047s; noGPUstarted, high/max/OOM0 in completedprobes/tests.
  Doc D111_SHARED_SNAPSHOT_PARSING.md contains full table/caveats/sourcebasis.
- All4 probe/test scopes plus curation exactempty stopped; nojobs remain.
  Curated20260928_d111_shared_snapshot_parsing.json
  SHAeb57f019112f8b0dcdfab305dc985843d98e47d96401e989d956ae2d6f617873;
  26sourcechecks and147protected unchanged; gitdiffcheckPASS.
- NEXT: preflight and ONE ordinary canonical3BFull4000W0 with same frozen
  inputs/config/initializers, newownedpaths and noadditionaloptimizer. No py-spy;
  therefore don'tattribute a D110-to-next timingdelta wholly to this candidate.
- Once-only published cache fulfilledD78/D80; NEVERrebuild. OldD110 allcomplete,
  doNOTredo replay/projection/curation/copy/tests/backup. Ordinary3BFull,7BFull,
  warm/reference,baselines,formalmatrices andnumericadapterqualification pending.
- Latest21:50 hostMemAvailable114491949056B,swap0,disk300476743680B,GPUempty,
  no primelora scopes/services active. Recheck at next launch. All next-run
  paths/config/remoteactivation/standardmonitor still NOTprepared or started.

## D110 completed reference — superseded CURRENT, not a command to rerun

2026-09-28 21:34. Supersedes ALL historical LIVE/NEXT notes in archived ledgers.
Goal ACTIVE/incomplete; baselines PAUSED. No GPU, remote, replay or analysis job
remains. Runtime244ceb6571c3f1d04a9f24594115f3433115d619 unchanged/already pushed.
This goal turn PROGRESS: full projection/curation, failure classification,
test synchronization repair, final checks and exact-owned cleanup completed.
Do NOT repeat D110 replay, projection, curation, remote copies or 47 checks.
Evidence commit6593db70a38e0337f72e690fcff0f7ebdd09b1ee pushed;
exact faaslora_origin/retry14_continuous_queue_v2 HEAD verified21:34.
Seven scoped files only, checksum15/source56/protection147/secrets/diff checks
passed; userdirty manifest not staged. Do NOT repeat backup. This post-push
ledger update is not a runtime/configuration change.

- Canonical3BFull4000W0 profiling4: all4000planned/arrived/submitted/terminal;
  2904success/nativecontract,999TimeoutError,97RuntimeError. Complete execution
  but failed Full qualification. No formal SLO/numerical/ranking claim.
- Main/profiler exited21:08:58; launch.json21:09:00 pass=true, actual GPU compute
  empty21:09:16, native release/service path removal confirmed. Exact remote3B/
  7B/monitor and empty local auxiliary stopped21:09:45. Weights/cache unchanged.
- FullJSON5474371755B retained immutable. Stream projection695.02s,119424KiB
  peakRSS,exit0; output28941897B. All4000IDs/native success token/prompt hashes
  checked. NEVER wholeload original, redo projection or overwrite raw files.
- Physical22773.316472914972GPU-s,62leases allreleased.59quarantines allreleased.
  6557resource samples,peak31748009984B,minhost87575101440B;
  high/max/OOM/swap/warnings0. First4 runtimes(1initial+3natural)1997success;
  51later successful runtimes907success. Counts do not establish failure cause.
- Remote133UUIDpairs,132published/1not_published; bothends308673720B,
  requestpacking0. Unpublished ecommerce_lora_0093 received2313558B,
  archiveverified but contentnotpublished; byteequality is not materialization.
  Correct journal transfers-915c0c48238446c3a9f31e42d6cf0032.jsonl found by clock
  c6ad47d3841446a2b5da3c3dd8303cb6; copied ONCE plus monitor108813154B.
- Conditional2904 means: TTFT1086.991756s,E2E1092.913897s,
  dispatch1074.098357s (=window1049.606544+slot21.816368+release-late2.675445),
  service12.893400s,nativeTTFT0.319634s,TPOT23.186419ms.
  No failed latency fabricated, no additive nested spans, no formal ranking.
- All999timeouts lack generation/source-admission evidence; NOT proof of no
  dispatch. Firsttimeout req_00840 offset2730.165340s.97RuntimeError=
  96parent+1subprocess wrapped unresolved-native-ownership; all this generation
  not_submitted,NOT proof earlier RPC ended. Firstreturned req_01994
  offset3661.265477s; cannot explain earlier timeouts solely by later errors.
- Parent-only py-spy0.4.2/100Hz/GIL/threads:58314278B,559878valid samples,
  99samplingerrors,1.01s lag warning. No measured profiler overhead or cross-run
  causal performance claim. Exclusive nearest-project(non-save_results):
  footprints130957(23.39%),sendRPC105947(18.92%),ownedinputs47149(8.42%),
  fileplan40659(7.26%). save_results ancestry70046/import2187/other487645.
  Samples≠wall-time phases. GNUtime wraps profiler AND waited descendants,
  not isolated controller CPU/RSS.
- Fullcurator56prelaunchsource checks+147protected unchanged;33source refs.
  Doc/status+CPU+request-stage table D110_CONTROLLER_PROFILE_FULL.md.
  Curated paper_results/ieee_tc/p2_backend/20260928_d110_3b_full_w0_profile4.json
  SHA1697c656f400efd7c1551df1b63f7f52c86d9fcfb290c934f2e0a1caf9c6ef41;
  companion20260928_d110_3b_full_w0_failure_breakdown.json
  SHA73a2b0d046b36d51b3c399afac69d1ad2fd8bf628cf9013f59e091e26ff6a74c.
  Additional20260928_d110_evidence_verification.json includes final checks
  and all3 smoke attempt/log identities.
- Initial smoke46/47: test assumed incomplete replay after start(), but real
 5-packet fixture could already finish. TEST ONLY repair uses real socket with
  event-gated first-prefix reception; no production/timeout/sleep workaround.
  Second46/47: added complete-case assertion used last event, which may be
  request_dequeued after terminal. Fixed by selecting ingress-terminal event.
  Final47/47PASS1.757s; both earlier failures retained. Test source SHA now
  f7217e085e02e331c962312aefe8ef1aaa1b37c7041e23d493149bd26b020074.
  tests/test_ieee_tc_external_replay.py was NOT one of56frozen refs; those
  remain unchanged. Production/data/generation/formulas unchanged.
- Preliminary scope7b1fc4ad720643c1a9b56f2342b2e592 cleaned21:11:28.
  Projection1d9c4c260d5c46f0a9b35873687c3f0d,curationfe1e9d6b9b1f4d42abb640777f3e9fe7,
  smoke071b619d1e3640929d4514ba0c81d562/smoke2 5d98c4833af9469c970f2ed41d73b654/
  smoke3 8eea8ce3fec4419f8feaf7441390e52a exact-empty checked/stopped21:29:21.
  All high/max/OOM0. Raw analysis_scopes_cleanup_profile4.log.
- NEXT ONE causal CPU probe of repeated parsing in the actual coalesced native
  observation path. D102 old8-adapter snapshot probe not sufficient alone;
  don't repeat it unchanged or infer all timeouts explained by23.39%samples.
  Retain same-source owner/epoch/clock/content/freshness and physical guards.
  No long-lived stale-state cache, arbitrary sleeps, deadline changes or
  blanket catch/retry. No newFull replay until minimal causal validation.
- Current goal incomplete: ordinary3BFull,7BFull,warmSLO/Resident,baselines,
  M1M2,A1–A5,S1–S13 and numerical adapter qualification all still pending.
  Published once-only delivery cache already fulfilled D78/D80; NEVER rebuild.

## Authority and frozen contracts

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before task/experiment and after compaction. No subagents authorized.
- Plan1525lines SHAfe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
  User execution authorization overrides historical plan-mode/noexecution text.
- Before comparison/configuration selection/comparative figure read FULL
  docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md,
  SHA5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22.
  G1=allcorrect+jointSLO95%,thenminimum physical lifecycleGPU-s/request;
  G2=commonbudget+SLO,thenP95TTFT. CE supplementary. Warm/reference numerical
  thresholds NOTfrozen; development5000ms is NOTformal SLO.
- Preserve nine IEEE equations/core. Each optimization: history + primary
  literature/code + falsifiable hypothesis + causal minimal validation + full
  replay. One candidate at a time; no data/weight/trace regeneration,13B or
  oldresult overwrite. Test-point tuning and result selection prohibited.
- D110 read-only source review completed: D102 diagnosis, NativeSourceSnapshot
  _footprints/from_native, Python3.12 asyncio official CPU-blocking docs and
  vLLM0.30 LoRA worker single-core-loop/identity semantics. Links in D110doc.
  No production optimization has yet been accepted from this profile.
- Baselines paused at9e2cf28903ed11bc8ee891dd4cc9636b94307573. D74Serverless3B
  repaired analyzed/plotted/backedup; prepared original NOT run. Resume only
  afterPrimeFull:Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe.
  Display name Serverless, not-new.
- Numeric adapter distinction remains pending:3B500/500 and7B498/500zero,
  2/4distinctweightSHA. Nativecount≠numeric proof; no authority newweights,
  don't repeatedly ask. Other implementation work can progress.

## Reusable history and assets

- Latest ordinary complete Full is D101Full8:3970/4000+30Timeout,19874.486281GPU-s,
 4leases,0quarantine; source2f1bc4b,curated/pushed8aa3607. DoNOT re-extract.
  D110 detailed profiling has different instrumentation and source; no paired
  CI, ordinaryperformance delta or claim D109 alone caused regression.
- D102 parser microprobe2.460/4.465/2.536ms at8adapters1792HOSTallocations;
 32repeats83.054/95.047/77.491ms. Rejected optimization pending actual profile;
  do not mislabel as D110 wave distribution. Existing py-spy0.4.2 qualified.
- D104 source-domain expiry;D106 confirmed same-content target-copy expiry;
  D108 real-owner non-target counterexample;D109 explicit no-op non-target
  binding expiry. D1091256distinct CPUchecks alreadyPASS and backed244ceb657.
  Do not repeat obsolete-error-expected D108 probe or unrelated regressions.
- D88 profiles3B368/368,7B92/92;D89initializers
  paper_results/ieee_tc/p2_backend/d89_{3b,7b}_initialization/manifest.json.
  Use original requested_model_config PARENT,notresolvedchild; no reprofile.
- D90prefix100/100both is notcanonical aggregateconcurrency qualification;
  D91assembledfull4000/500both;D92plannedarrival1800sdeadline.
  D93–D101 CPU/source/reference/lifecycle fixes have archived evidence, not
  automatic Full/performance qualification.
- RawD110 results/ieee_tc/p2_backend_qualification/d110_20260928;
  results symlink resolves to /home/qhq/serverless_llm_experiment/results.
  Fullfinal3b_outputs_profile4/...d110_3b_full_w0_profile4.json;
  requests full_profile4_request_projection.json; finalstats inpaper_results.
- Correct contentindices paper_results/ieee_tc/inputs/
  20260927_3b_remote_content_index.json SHAbd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  20260927_7b_materialized_content_index.json SHAe85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  Do not use misnamed7b_remote_content_index diagnostic4998filelist.

## Environment, remote and safety

- Repo /home/qhq/serverless_llm_experiment_retry14_baseline,
  branchretry14_continuous_queue_v2,remotefaaslora_origin. Baselineownrepo/main.
- Nativeenv /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM0.30/torch2.13/CUDA13/Python3.12.12/SM86. CPUenv
  /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;OSguards/usr/bin/python3.
  No install/driver/globalptrace change.
- Current3Bcap8/slots8/cpu32/gpu.72;7Bcap2/slots4/cpu24/gpu.70;
  HOST16GiB/native2/NVMe16,W5/movement3,scale2s/beta.5,min1max4.
  3Bservicebin28.114717726756954ms,7B27.4336ms. Development only.
- Remotealiasprimelora-artifact-174 lab14@192.168.4.174:8122,
  strictBatchMode,key~/.ssh/primelora_artifact_174_ed25519_20260925 (0600).
  Token~/.config/primelora-tc-d75/artifact.token private NEVERprint/stage.
  FingerprintSHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
- HTTP3B18080/7B18081 direct. Localeno1np0/remoteeno1 both1000/full verified
  beforeD110; historical100Mbps is NOTcurrent. No agent NICchanges.
- Remote immutable cache /home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b},
  both500/500,total1581215261B. No rebuild/pool copy. Profiles and raw true
  network transfer retain latency/competition; no requestpacking.
- Actualservice72/80GiB/swap2,CPUs4–23,28–47; aux3/4GiB/swap0CPUs2,3,26,27.
  CPUanalysis3/4GiB/swap0sameauxCPUs. Oneheavyjob. Limits BEFOREworkers.
  No global kill/ray-stop/reset/reboot; only exact-owned identity cleanup.
  Inference disk150/100GiB unchanged;artifact nodeindependent incrementalrule.
  No remoteconfig/restart/hash/cleanup during inference.
- Latest21:29host106GiBavailable,disk300527575040B,swap0,GPUcomputeempty.
  Recheck before next heavy task; this is not a future resource guarantee.

## Protection, reporting, backup and archive

- Seal147protected:paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify via scripts.ieee_tc_preflight.verify_seal; oldfinal_v2/figs/paper unchanged.
- NEVERstage configs/generated/lora_manifest_1000.json or unrelatedAAAIarchive,
  oldfigures,regenerate_motivation_figs.py,rejectedpreviews. Explicitpaths only.
- Eachruncleanup→validation→status table/figure→interpretation→next.
  academic-plotting forfigures; failuretables appropriate. Chinese paper-evidence
  updates≤60s duringongoingwork; no engineering checks sold asperformance results.
- Backup testedcheckpoints/evidence tofaaslora_origin/retry14_continuous_queue_v2,
  afterdiff/sourceSHA/protection/secrets/smoke. No forcepush/credentials/rawlargeGit.
- Previous1105-line ledger archived VERBATIM at
  docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D110_20260928.md,
  SHA f616d36b19a0f2e5e39267a2367a3c7597c1417287bf00f025408365c6f17ed6.
  Historical LIVE/NEXT handles are superseded, not commands to execute.
  Earlier EXECUTION_HISTORY_D91_D100,D81_D91,D78_D80,D67_D77,
  THROUGH_D26,D27_D52,D53_D66 remain intact.
