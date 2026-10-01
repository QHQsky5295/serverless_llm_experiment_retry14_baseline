# IEEE TC execution status

## CURRENT — D154 ordinary 7B Full sealed; checkpoint backup pending

2026-10-02 04:30 +08. Goal ACTIVE; PROGRESS, no live jobs. D154 completed,
validated, tabled and sealed. Source39cc3c1 unchanged; no nextoptimizer selected.
4000native/0fail; numericaladapter/commonSLO remainOPEN, n_correct=null.
U16052.781459837977GPU-s (vsD152 +0.366%); mean/P95/P99TTFT
103.898017548/210.479250493/220.402708432s (mean +6.060%,P95 -0.072%).
TPOTmean/P9536.847975193/56.133295002ms. NOT model acceptance or G1/G2 proof.
Dispatch102.435813815s=98.593%TTFT; service1.462203733/native.326907505s.
Phases arrival→gate101.177903361; gate→source1.257910454;
source→dispatch1.135296228; native4.354499659; last→controller.559013280;
controller→terminal.317432635s. Gate→terminal7.624152256 is UPPER ENVELOPE.
Native/gate-envelope averageconcurrency4.355350/7.625642,max8. Control1763total,
1762inwindow/1656queued/783belowreadycapacity,1postwindowretained. Planning1954
completed,935.311941workerCPU-s. Outputhash123changed vsD152,all15otherfields
4000match; numericalcauseOPEN. Timingerrors0,dispatch4000/conflicts0.
Resources4080samples/peak20384194560B/minhost92605038592B/swap-events0.
132publishedtransfers/131480060B/packing0. Fourphysicalleasesallreleased.
Decision: retain semantically qualified redundant-scan simplification, NOT a
demonstrated performance gain. No repeatedmicrotests/replay; next inspect
remaining request-pipeline non-generation occupation with source/history/primary
references. No blindcap/deadlineincrease; baselinesPAUSED until bothmodelgoals.

Postprocessing complete:
- localabsence04:22:01; exactremote stop thenSHAverifiedcopy. No active remote.
- metadata2361 CLOSED0/4.13s/RSS448864KiB,actualc2406f98860b43dd80b63b15dae1f17b.
- projection38714 CLOSED0/79.36s/RSS99840KiB,actual346633f706484b9d937f5a15a8a71a19.
  271585048Boriginal streamedONCE; neverreproject. ProjectionSHA
  fe4583726bfebf2befc321b3a92bf911836edc57a3fa417eec2021e256757f24.
- curation65999 CLOSED0/7.69s/RSS464960KiB,actual5619225f222c49fdaf4588d1c7f43450.
- occupancy82482 CLOSED0,actual66a661417d5a427db612913f1d395057;
  finalevents0 shown; otherauto-removedscopefinalcountersunavailable,not0.
- verification94172 CLOSED0,actuald46e142259d1408db18dad5c5990c838;
  266frozenrefs/47curatedrefs/147protected/Plan/V1PASS,SHAbba003fb52a8614eca2eeac71a118618d0e191a6072b999cad3b5c528d279c82.
- bundlecompletedexit0,actualab9e31d81def4cd98840c023fd4f61b8;
  allscopeabsencesverified.65members56994B,SHAd34860f355e84431ee74c0f1c814a36c3f0759634dbd1dc680c3672465d37d84.
Sealed docs/ieee_tc/D154_FULL_W0_FULL1.md DON'TEDIT. Curated
paper_results/ieee_tc/p2_backend/20261002_d154_7b_full_w0_full1.json
SHAdf5139616f6d2e24d139b26c9766c93400839321a84ab07cac12c73792d905f2.
128MiBguardunchanged,compact129733056B. No newGPUrun/sourcechange.
NEXT explicit checkpointchecks/backup then focusedread-only bottleneck audit.
Prior current entries below retained as historical, superseded by this closure.

2026-10-02 04:24 +08. D154 launch PASS, all4000 terminal/native-contract successes,
no failures. Local service/aux automatically removed; exact PID/cgroup/GPU
absence receipt04:22:01. Remote exact3B/7B/monitor stopped after localrelease;
copies SHAverified. Journal transfers-d8e65a825b6b4ce8854eaa3a8c6d5a70.jsonl,
clock matches frozen2b08d4753ddd40518b9d22609fc55d65. No liveGPU/remote jobs.
Preliminary U16052.781459837977GPU-s/all4released;132published UUIDpairs,
131480060online bytes/3300789780logical bytes/packing0. Resources4080samples,
peak20384194560B/minhost92605038592B/swap0/high-max-oom0/no warning.
Numericaladapter/formalSLO stillpending; no comparative interpretation yet.
Metadata2361 CLOSED0/4.13s/RSS448864KiB, actualc2406f98860b43dd80b63b15dae1f17b;
absenceverified04:23:45. Metadata128MiBguardunchanged. Single bounded projection
LIVEsession38714, unitprimelora-d154-full1-project-20261002.scope. Never reproject
original after success. Curation/table/seal/backup stillpending. Source unchanged
39cc3c1. FullPlan/status/V1 and analysis/plotting skills reread this continuation.
GoalACTIVE; baselinesPAUSED; no nextcandidate or GPUrunselected.
Disk afterworkspacecleanup161741139968B; recheck150GiBgate beforeheavylaunch.
NEXT finishprojection→exactanalysiscleanup→curation/table→interpretation/backup.
Below04:09 RUNNING record retained as history, superseded by this terminal state.

2026-10-02 04:09 +08. ONE ordinary Full launched03:09:28 in tmux
tc-d154-7b-full1. Source39cc3c1147f1699a4e7cddb54996be2564a895c3 BACKED.
4000/source42/W0,sameD152config/D89profiles,no profiler/prefix. D153 file
observation is solecandidate; approvedPlanpointer is provenance-only update.
Do NOTrelaunch orchange code/config/remote duringinference. GoalACTIVE.
BaselinePAUSED; bothmodelacceptance/newmetricoldPrime gap stillOPEN.

- Service primelora-tc-svc-3bfabb65d8a64a77a5d47380c926f69f.scope,
  actualfb3a7d977e9c43b0bae3b8e82cc790bd,72/80GiBswap2readback.
- Aux primelora-tc-aux-318ecb4a5f8e4fd18f56487c98ee456e.scope,
  actuala3ac14f901b44b7583cda07827d43f74,3/4GiBswap0readback.
  Replay614753/watchdog614760verifiedexactauxCPU2,3,26,27. Nativecores618019,
  622796,622884,623230 observed in service group; firstcore's CPU4–23,28–47
  verified earlier. Sample3541memory20289523712B/peak20300824576B,
  host94739906560B,swap/events0,no warning/abort/foreignGPU. Disk159156215808B.
  Disk now below150GiB new-heavy-run gate, above100GiB running stop floor;
  do not clean/compress during inference. Recheck after native workspace cleanup
  and audit only reconstructible owned data if space remains insufficient.
  Live04:09:17:3591/4000reported-success,3675arrived,0fail; incomplete,no comparison.
  Full native-contract cross-source validation is pending, not inferred from banner.
- Remote3B2767751/fa041c728de74c0c9cb00d62e724c751;
  7B2767753/940a068243f846a2968f56528cb8e16f;
  monitor2767756/9accccc401234562856b1efa23b39afe,
  unitprimelora-artifact-monitor-d154full1.service.
  7Bclockremote-process-monotonic:2b08d4753ddd40518b9d22609fc55d65.
  Monitor/home/lab14/primelora_remote/tc/d154_20261002/remote_monitor_7b_full_full1.log.
  BothNIC1000/full; immutableD78/D80deliverycache unchanged.
- Prelaunch266refs/147protected/Plan/V1PASS;
  SHAa63468029ab5b1a1e2b2aa1a9a459a6e7fe70aa71d0d36ecf56d21584603b740.
  prelaunch28293closed0 actual676d6ed9b4f740efaf29d5004e7d7f1b;
  healthreturned0 actualbb858db8c23540b8a48da088350b01c8;
  bothabsenceverified, finaleventsunavailableafterremoval.
- Rawresults/ieee_tc/p2_backend_qualification/d154_20261002.
  stop_services_full1.sh/stop_remote_after_full1.sh/copy_remote_full1.sh
  preparedwithactualidentities/clock; shellsyntaxPASS, NOTexecuted.
  Six D152 analysis helpers reused for D154 paths/service/remote clock and
  D152 sealed prior-comparison SHAs; syntax PASS, NOTexecuted. Localauxcleanup
  and automatic-removal verification helpers likewise prepared, NOTexecuted.
  Analysis scope InvocationIDs/cleanup helpers await actual post-run launches.
  DoNOTpartialparse/hashresults. FullPlan/status/V1 reread aftercompaction.
  Monitoring cells7583/7597/7604/7608/7611/7614/7617 closed normally; not experiment completion.
  tmux run remains LIVE. No outstanding tool session/monitor cell to resume.
  03:56:54 exact service/aux InvocationIDs revalidated active; 04:09:17 service
  still active exactID/tasks533. This continuation is a VERIFIED WAIT, not a
  newly completed experiment. Plan/V1 hashes unchanged from full prior reading.

NEXT monitorSAMErun→terminal/nativephysicalrelease→exactremote stop/copy and
localauxclosure→boundedmetadata/ONEprojection/curation/table→interpretation.
CompareusingsealedD152projection/curated,not itsoriginal. No secondoptimizer,
cap/deadlinechange,duplicateprofile/cachepublication orbulkI/O duringrun.

## D153 checkpoint and D154 preparation — completed history

03:08 +08 D153 backup COMPLETE39cc3c1147f1699a4e7cddb54996be2564a895c3,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
Push84503/verify46575closed0;10explicitfiles/54payloadchecksPASS.
Onlyuserlora_manifestdirty; runtimefrozen. D154 reusedD152sixlaunch/configscripts,
freshpaths only, verification adaptedtoD153identity/newapprovedPlan. Notyet
launched; currentlypreparingboundedprelaunch/remotehealth. NoGPUorremotejobyet.
FullPlan/status/V1 read; beforeD154Plan/status reread includingtruncatedgap.
Priorentriesbelowhistorical; doNOTrepeatD153tests/seal/publication.

03:05 +08 D153 seal COMPLETE, backup pending. Curated
paper_results/ieee_tc/p2_backend/20261002_d153_file_observation.json
SHA40358c5acae7d448cdeb4e0cefdd57c1e06f080452031b099e9286efe2302cbe.
Bundle45members239944B SHAe4b67a786e9d214f271494cdeb9db88802915ebd562b71f9117fc5b80b606a2e;
checksums/syntax/secrets/147protectedPASS. SealedD153doc don'teditafterthis.
PreflightSNAPSHOT nowexplicit20261002approvedmirror; oldsnapshotretained and
mismatchstillrejects. verify1 rejectedoldpointer(exit1,
actualfb9aaf52d80c483fab7397c40ccc5ded); retainedscript/log.
preflight1wrongmodelinterpreterlackedpidfds:68tests2fail2error,exit1,
actual53614131170f4bc6a17d9cf7fc03a334,session42774closed;
preflight2qualified/usr/bin/python3:68PASS1.184s,exit0,
actualef0eea922da64130a899a61a0c619ac2,session60759closed.
verify2/session84867CLOSED0 actual1d72f46557df48d7bc3f83fc64b12cbf;
allabsencesverified; no livejobs. No servingchangeafter813tests.

2026-10-02 02:59 +08. Goal ACTIVE. Only Prime7B; externalbaselinesPAUSED.
D152 remains sealed/backed; no repeat original parsing/closure. D153 single
candidate in faaslora/memory/residency_manager.py derives confirmed file
signatures/footprints and budgets from one fresh double-checked owner inventory.
No cross-callcache or removal of execution/physicalchecks. Extentsettlement
skips only when no closed unsettledwriter; ordinaryinventory stillfull.
Targeted18PASS1.421s, regression813PASS131.814s(command144.02s/RSS1200136KiB).
targeted1/session54921 CLOSED0 actual0ffa72e945804136bc2d5ceaec964a54;
regression1/session52090 CLOSED0 actuala4d7e7c65cb448db8158c20c9c3b3721;
micro1/session31678 CLOSED0 actual6955eb57ff4c4a8f968826252ea652b7.
Allscopeabsences verified; finalcgroupcounters unavailableafterautomaticremoval.
Raw results/ieee_tc/p2_backend_qualification/d153_20261002.
Component tinyfixture7trees:500names old9inventories/3143stats/56.185ms vs
new1/48/36.492ms,3alternatingpairs;exactoutputexceptcapturetimes.
4names3.854→3.563ms. Not servinggain,CI,orG1G2acceptance.
Historicalcalleraudit570D143samples:59LocalSource.source_snapshot inclusive,
57frompreparation;39inventory(30_file_inventory/9_source_observation),
29NativeSource._footprints;18file/16GPUexecutionobjectives. NotCPUpercentages
orcurrentfrequency;overlapnotadditive. ThreefilesAST/byteidentitybeforecandidate.
Caller audit scope nowabsent; historicalInvocationIDNOTcaptured. No rerun.
Doc D153_FILE_OBSERVATION_DIAGNOSIS.md has qualification/componenttable.
NEXT seal/backup then one ordinaryFull4000/W0 sameD152parameters/profiles,
no profiler/prefix/capchange. D154 launch not yet prepared orauthorizedasformal.
No liveGPU/remote/testjob. Sourcecurrentlydirtycandidate; HEAD53efe130.

## PRIOR — D152 finished, analyzed, sealed and BACKED

02:40 +08 backup COMPLETE53efe130e6871d387af128c3101aed4cef635b30,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
14explicitfiles/91payload syntax/secrets/bundle/diffPASS; usermanifestexcluded.
Push88371/verify98154 CLOSED0. Plan amendment and verbatim history included.
No liveGPU/remote/analysis/exec jobs. Do NOTrepeat closure/tests/publication.
Read-only nextbottleneck source/history/primary-reference audit in progress;
no new candidate/configuration or GPU run selected. Serving unchanged26fa0a0.

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

NEXT source/history/primaryreference diagnosis of
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
