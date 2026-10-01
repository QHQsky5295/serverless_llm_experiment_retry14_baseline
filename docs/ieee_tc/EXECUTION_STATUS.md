# IEEE TC execution status

## CURRENT — D147 qualified; backup then capacity reclamation/ordinary Full

2026-10-01 19:39 +08. D147 candidate2 QUALIFIED, not yet Full-performance
accepted. No liveexec/session/GPU/remote/CPUjob; all scopes empty/events0/stopped.
Do NOT repeat completed tests, query microtests or D143 stack accounting.

- Green2 sixPASS/.008s, actual9666671689aa4c0ca21647412c8bae8a cleaned.
- Observe2 actualf4405fdaaad44aa29ade3129f7d8bd60 completed0/16.41s,
  RSS1139108KiB; exactempty/events0 cleaned. Nativeenv, CUDAuninitialized.
  All32 paired idle queries agree within CLI rounding; first candidate's
  reserved-memory mismatch resolved with explicit NVML memoryv2. On physical
  1/2/3 median query59.7824/66.9568/61.0174ms -> .0923/.1295/.1069ms.
  Initialdevice0 branch unchanged, its variation NOTcausal. These are query
  diagnostics, not end-to-end gains or independent replay repeats.
  ResultSHA15e915f8c1ea29ce4fd43cf3dab7b3924b2857c7cc2bbd14f39b139bfea7dc51.
- Finalregression543PASS22.880s/command33.41s/RSS1193040KiB;
  actual23022241cc55420a81f11a17395116e5 exactempty/events0 cleaned19:38:06.
- Verification204unique sources (216 historical path entries normalized)/147protected/Plan/V1/sourcevariants/syntax/secrets
  and archive re-read PASS. Actual972dba01e1b545dea35301c7a1c1b10d complete0,
  10.22s/RSS1140732KiB; empty/events0 stopped19:39. No global resource changes.
  CuratedSHA4660f4579a003af2ab7e7038c0c439178e55e3e955838d9956696cc9c916a4a6.
  Bundle49members546769B SHA6a0400e02005a4e91f64575550d52fa77b3954163ab8c262587b759b5226d063.
  D147_DIRECT_DEVICE_QUERY.md includes two diagnostic tables and caveats,
  SHApinned; do not editafterseal. Verificationownlog/cleanup localnotinbundle.

NEXT explicitsevenfile backup (no usermanifest), safe reclaimable disk audit,
then ordinary7BFull with unchanged config/profiles/no profiler. Full performance
and D145failures NOT resolved by microtest. No newoptimizer while testingthis.
Disk~151.7GiB; no deletion/compression yet. Baselines PAUSED. Bothmodels,
3BTPOT/outputhash,numericadapter,warm/Resident,M1M2/A1-A5/S1-S13 remainOPEN.
GoalACTIVE; latest once-onlyremote cache permission alreadyfulfilledD78/D80.

## D147 qualification history — superseded by CURRENT

2026-10-01 19:34 +08. Goal ACTIVE; Prime-first, baselines PAUSED. No GPU
inference or remote operation. D146 already BACKED7f42cbd; do not repeat it.
Read full Plan/status/V1 and vLLM/optimization, run-experiment, github-sync,
academic-plotting skills. Primary NVIDIA NVML/SMI and vLLM CPU/GIL docs checked.

D143 archived controller samples establish a concrete synchronous subprocess
path: 570 request-window main samples; snapshot inclusive43, check_output42.
Of the42: request execution16, reservation cleanup16, live display10.
Counts are NOT wall/CPU percentages; no new GPU profiling or giant JSON read.
Initial TP1 stack monitor only covers device0; scaled-out device queries use
CLI. Do NOT expand monitor.devices, because this also changes accounting.
Candidate only substitutes fresh NVML queries for this IEEE hint CLI path;
UUID-bound authoritative utilization, equations, budgets, polling cadence,
timeouts and request ownership unchanged. Legacy branch unchanged.

- First6 targetedtests RED6failures, then GREEN6/.008s. First affectedregression
  543PASS22.685s,33.30scommand/RSS1192120KiB. All exact domains empty/events0.
- First idle-query microtest nativeenvironment passed execution17.99s but
  candidate1 NOT accepted: v1 memory.used includes driver reservation while
  CLI excludes it. Old means0.455566GiB vs0.014648GiB are semantic difference,
  not mere display rounding. Raw firstresult SHA
  b4620467bfc5a8df442a28c56145b7627697df2d5c0cff5233ddd85fbe7ce1d5.
  Firstsource/test snapshots preserved as candidate1_runner.py/candidate1_tests.py.
- Bounded API probe actual6e85416f8d92496b97a2ee80beb03739 confirmed device3:
  v1 used489160704; v2 used15400960/reserved473759744/total25769803776.
  Probe stdout in tool transcript; exact empty/events0 cleanup saved.
- Candidate2 requests nvmlMemory_v2 explicitly; tests assert version. Actual
  returned bytes remain unrounded (CLI MiB rounding separately qualified).
  Current green2 execsession76681; next check completion/identity/cleanup,
  then ONE regression2 and observe2. No repeats of D143 stack aggregation;
  observe2 reuses sealed first counts, checks all32 idle pairs within CLI
  rounding and records timings. Ordinary Full remains PENDING, not a gain.

Raw d147_20261001; no D147 curated/seal/commit yet. Disk~151.7GiB; proven
rebuildable cleanup still needed before largeFull; no deletion/compression.
Next finish candidate2 tests/evidence/table/seal/backup, then reclaim safe
capacity and ordinary Full. No newcap/deadline/config tuning. Bothmodel
performance,3BTPOT/outputhash,numericadapter,warm/Resident,formalM1M2/A1-A5/
S1-S13 OPEN. Once-only immutable remote cache already fulfilled, never rebuild.

## D146 qualified and BACKED — completed history

19:16 +08 backup COMPLETE7f42cbdd143ab917927237abcf136bc2dc18e608,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
Sevenexplicitfiles/30payload secrets/checksum/diffPASS; usermanifestexcluded.
No liveexec/session/GPU/analysis/remotejob. Do NOT repeat D146 tests/sealing.

2026-10-01 19:16 +08. D146 bounded failure-observation change COMPLETE:
same RuntimeError/message, connect30s/send-recv300s, no retry/ownership/release/
formula/config changes. Operation, call-local phase, originalexceptionclass,
attempt, parentclock, dispatchhandoff and actualguardexpiry now survive both
existing requestfailure collectors. Scalarallowlist; no prompt/kwargs/tokens.
CPython3.12.12 primary timeouts.py readonline; unavailable docs notused.
No GPUrun selected/launched and no performance gain claimed. D145's four
failures stillunknown; don't infer poolstarvation/retirementbug/D144causality.

- Red6tests/8subcaseerrors reproduce missingattribute, notunderlyingD145timeout.
  Actual923c40e44ed64631a4d33db0b95431df empty/events0 cleaned.
- Same6tests GREEN0.031s, actualdef1eda2d5484c59b3139e0f1b59932b empty/events0.
- ONEaffectedregression508PASS22.810s, command31.22s/RSS1081892KiB;
  actual4773f20c81744bb5a42ba50c928a9ee0 empty/events0 cleaned19:13:16.
  Allactual3/4GiBswap0/taskset2,3,26,27, CPU Python3.12.12. Do NOTrepeat.
- Verification216frozenrefs, declared2changedsource/testfiles checkedagainst
  parentd0cfbf31;147protected/Plan/V1/syntax/secrets/bundle PASS.
  Actualbbc8a69e4f264ab8908986e47182eb7d complete0/1.25s/RSS85648KiB,
  exactempty/events0 stopped; no livehandles/jobs.
- CuratedSHA3cc80f33368b81926bbfa0f0affdb47b52a5a5ef53baa772a12d52dca16b17b2.
  Bundle23members270791B SHA5b77e9332eec624e01b5299393cde12cfbbc5fc5a693e6561a1bde5547856288.
  D146_NATIVE_RPC_FAILURE_CONTEXT.md has status/semantics/boundaries table;
  SHApinned, do noteditafterseal. Rawd146_20261001 contains actualreceipts.
  Scopeverification's ownlog/time/cleanup remainlocal, notinthealreadycreatedbundle.

NEXT returnto evidence-backed non-inferencewaitingcandidate;
noGPUreplay justforlogging. Backupabove supersedes thepreviouspendingstep.
Disk~151.7GiB needs provenrebuildablecleanup before nextlargeFull; no deletions
orcompressionyet. BaselinesPAUSED; bothmodelperformance,3BTPOT/outputhash,
numericadapter,warm/Resident,M1M2/A1-A5/S1-S13 remainOPEN. GoalACTIVE.

## D145 backed closure — completed history

18:56 +08 D145 backup COMPLETE d0cfbf31dc70b9e50c21cab8e38e8644ef47c101,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEAD verified.
13explicitfiles/77payload secrets+syntax+bundlePASS; usermanifestexcluded.
CSVnativeCRLF retained to preserve sealedSHA; diffcheck withcr-at-eolPASS,
no trailing-space exclusions beyond recognizingCSVlineendings. No servingedit.
Allownedanalysis/inference/remotejobs inactive; NOlivehandles. GoalACTIVE.
NEXT useexistinghistory/source to narrow RPCfailure and non-inferencewaiting;
disk151.7GiB free needs safe provenrebuildable cleanup before furtherlargework.
Do not repeat D145 closure/projection/tests or immutable deliverycachecreation.

19:00 continuation read-only source/history work: D139/D143/D136 fullyread;
vLLMskill + optimization/troubleshooting references fullyread. Generic skill
QPS/quantization/cap/timeout suggestions NOTapplied; frozen V1 controls.
Primary vLLM CPU/GIL performance note and tagged0.30 core/UniProc source
rechecked online; Python3.12asyncio-dev page returned503, no claim frommissingpage.
Currentsource confirms _rpc wraps first_exc withoutcmd/operation/phase,
while failed_request outercollector stores onlytype/text. No oldlog canbe
assumed to contain those fields. Native control calls usefreshconnections;
generation usespool; doNOTinfer poolstarvation fromerrortype. _exec_request
finally/earlyselectedsource paths bothcanpropagateouterexceptions.
Fourrequest ingress/terminal rows inspected(nooriginalgiantJSON): 03842-44
arrivedanddequeued nearplannedtime;03996serverreceived~22.06s late. Notproof
of300sRPCphase orcause. Existingnative retirement/pending/frontend code also
read; NOdemonstratedretirementfencebug orperformancecandidate selected.
Next boundedtask: preserve exactRPCoperation/phase failurecontext viaexisting
error path, qualify onlyaffectedCPUtests withoutchangingfailure/retry/timeout/
ownership semantics; first inspectexistingtests/collector beforeediting.
This is measurementcompleteness, NOT a claimedservingoptimization andNOT
authorizationforanotherGPUdiagnostic alone. Then returnto oneevidence-backed
non-inferencewaitingcandidate and ordinaryFull. No newfiles/scriptsprepared,
no servingchange, no newrun. Do not repeat closedD143sampling/D144microtests.
Source main_outcome/projection/curated data remain frozen; no deletion/compression
performed. All formal andpreviousopenissues stayOPEN. ThisturnPROGRESS: sealed
andbackedD145; nextdirection constrainedbyactualmissingfailureevidence.

## D145 closure — completed history

18:54 +08 closure PASS:216frozenrefs/44curatedrefs rechecked,147protected
unchanged. Existing giantoriginal streamSHA reused withstat, no reparse/hash.
Failurepopulation3996/4,controls2053/2052/1outside,alltiming/remote/cleanup
invariants verified. D145_FULL_W0_FULL1.md contains final diagnostic tables,
conditional comparison and failure boundary; SHApinned, do not editafterseal.
VerificationSHAf509fc500e0abf58a63d558be2bc3e0bea974abbbc14b26d2a31c0005d0032da.
Verification actual910a165269e34f78804a307175748ce0 completed0/2.52s/RSS27696KiB,
empty/events0 cleanup18:53:31. Existing unchanged tests reused, notrerun.
Bundle64members58086B SHAafb300b2f0a7f2286a3275ad8f7e77e4a15447296d37ac82dd36c4755d81ce93;
actual8a564a77fa4c466bb134c1c27d5ea13d completed0, exactempty/events0 cleaned.
No livehandles/analysis/GPU/remotejobs. NEXT explicitD145docs+curated+bundle
secrets/checksum/diff -> commit/push; usermanifest/unrelatedfilesexcluded.
Then new source-based bottleneck diagnosis, not another speculativefullrun.
No newservingchange or optimizer selected. GoalACTIVE/incomplete.

## D145 finished-analysis state — superseded by CURRENT above

2026-10-01 18:48 +08. Goal ACTIVE/incomplete. D145 ordinary7BFull4000,
projection, curation/failure breakdown and failure-aware occupancy all FINISHED.
NO live exec cell/session/GPU/inference/analysis job. Historical LIVE handles
below are closed; NEVER wait on or restart them. Baselines remain PAUSED.

- 3996 native-contract successes/4 failures; full-completion FAIL, numeric
  adapter/commonSLO still pending. Serving fc486083 unchanged/backed.
- All four error firstlines: RuntimeError: subprocess_native_rpc_failed_no_retry:
  TimeoutError:. Outer RuntimeError classifier timeout0 does NOT mean no nested
  timeout. Actual RPC operation/phase not recorded; cause UNKNOWN. No evidence
  these are outer1800s deadline errors. Generation state unrecorded; null/default
  fields do not prove zero work/no dispatch. Three quarantines all released.
- Before first error, long queue already existed; arrival window ended18.756min
  earlier. Do not attribute all queueing to later quarantines.
- Curated SHA8a75bdc7f4600094f31ce92201acb6a13d9b8c3760d60665af2eb1414be7733f.
  Successful mean/P95TTFT678.390986/1316.054617s; TPOT40.859549/68.869382ms.
  mean dispatch675.934327s, serviceTTFT2.456658s, nativeTTFT.341165s.
  All3996 reliable timing errors0ms; confirmedpre-generation3996/conflicts0.
  Physical21346.039381GPU-s,6leasesallreleased. Remote132/131480060bytes,
  requestpacking0, immutablecacheunchanged. Fullnotacceptedperformancegain.
- Occupancy all4000IDs/3996native/4failed; phasepopulationnative_success_only.
  Gate→terminalmean10.308862s is UPPER ENVELOPE, NOT measuredpermitrelease.
  Native4.712198s; meanconcurrency3.534531(native)/7.732484(upperenvelope).
  Controls2053total/2052inwindow/1afterwindow preserved. No causalGPUproof.
- Projection completed0 at18:37:18,43:44.88,RSS99840KiB,original14869059366B
  readONCE,projection34384866B. NEVER reproject/re-hash giantoriginal.
  Actuald7ce840a89db418ea1749078629b4d56 cleaned18:37:39.
- Curation actual327593c6e1b24e39b38dfa1a72bc465d completed0,48.07s,
  RSS367428KiB; exactempty cleaned18:39:10. Failurecollector completed0.
- Occupancy actual2a7b1258d4af462da0d7d8e15ce8fe3d completed0 at18:39:14;
  exactTasks0/cgroupempty/events0 cleanup18:47:50, inactive confirmed; receipt
  cleanup_occupancy_full1.log. Previous metadata/projection/curation cleanup
  stdout exists in tool transcript, not raw*_cleanup.log; do not fabricate logs.
- Last18:47disk162918825984B(~151.73GiB),hostavailable113723953152B.
  Recheck and reclaim only provenrebuildable/duplicate resources beforebigRun.
- Plan/currentledger/V1 and analyze-results/github-sync/monitor skills readFULL
  this turn; no serving/analyzer changes or repeated regression tests.

NEXT: final D145 diagnostic table/interpretation -> verify sources/protected147/
secrets and small evidence bundle -> explicit scopedcommit/push. Reuse D137
closure but adapt failures3996/4 and actual cleanup evidence; no repeats of
95tests/legacy3Baudit. RPC-only analyzer NOTapplicable; no guardweakening.
Only after D145 closure select new evidence-backed optimization; actual RPC
timeout phase currentlyunknown. No cap/deadline increase. Numericadapter,
3BTPOT/outputhash, bothmodelperformance,warm/Resident,M1M2/A1-A5/S1-S13 OPEN.

## D145 analysis history — superseded by CURRENT above

2026-10-01 17:56 +08. Goal ACTIVE/incomplete; this turn PROGRESS: terminal
cleanup, stopped remote services, copied matching remote evidence, completed
bounded metadata audit, started one streaming request projection. No new GPU
run, serving edit, optimization selection or baseline work. The user-approved
once-only delivery cache remains D78/D80; NEVER rebuild.

- D145 terminal launch PASS; service/replay/watchdog returncodes all0, native
  GPU context release and service-path removal confirmed. All4000 terminal,
  3996 native-contract successes/4 RuntimeError failures. Full completion FAIL;
  numerical adapter and common SLO qualification still pending.
- Physical measurement complete:21346.03938088799 GPU-s,6 allocations all
  released,0 open leases. Three native-ownership quarantines all released.
  First quarantine354366.783569071,req03842/03844; last354601.48183647,
  req03996. Complete request errors still pending projection; this does NOT
  establish root cause or explain all waiting.
- Metadata6226samples:peak53154447360B,minhost73221296128B,swap0,
  high/max/oom/oom_kill0,no warning/abort. Remote132 UUID pairs,allpublished,
  client/server bytes both131480060,logical3300789780B,requestpacking0.
- 17:52 terminal service inactive/GPUempty; exact remote3B/7B/monitor stopped
  with frozen identities,allinactive/MainPID0. Local auxiliary exact
  e212cd03e1ba4a0fa4e973e3a08e7cbd empty/events0 and stopped17:52:26.
- Remote journal discovered by frozen clock:
  /home/lab14/primelora_remote/tc/d80_20260927/7b/transfers-095485df19224b5c929684a45bb87d2b.jsonl.
  Copied AFTER stop; source/copy SHA34898fde7e8c03f837d639a8614e1ca728ed7c33d44cc1da486395dbc015cbc8;
  monitorSHAd8cab00b941c083a8dc2111d3e9ea234b94babab1cd3e176f6baeeb7d69a5859.
- Metadata actuala11377c470304d3983ff2aa320a6dde0,3/4GiBswap0/taskset2,3,26,27;
  PASS3.19s/RSS315712KiB,empty/events0,stopped17:53:33. Both whitespace
  fixtures passed. Source133795919B/compact89727383B,allvalues retained,
 128MiB guard unchanged. PreliminarySHA
  6a6e831ad0f0f4d327a878324596bd97343fd9f66dfe66e513ba5eb3dbbbe3c4.
  Terminal launchSHA681158ea3dc1e005609785db564997f981f64351b1bd23616163bb8480db92c3.
- **ONE LIVE CPU projection** started17:53:33,exec_command session99752,
  unitprimelora-d145-full1-project-20261001.scope,
  actualInvocationd7ce840a89db418ea1749078629b4d56. Actual3/4GiBswap0/
  taskset2,3,26,27; unchangedD96 jq streaming projection. Source14869059366B
  read ONCE; partial output must NOT be read/hashed/parsed. Previous D137
  same-size projection took46min, so do not infer a stall from zero output
  before final reduction. Do NOT relaunch, reparse or run heavy parallel work.
  Latest18:23:17 SAME actualInvocation verifiedactive2tasks/68489216B,events0;
  source offset10160427008B of14869059366B, steadily increasing. No partial
  output read/hash/parse. Monitorcell6889 completed/closed; monitorcell6892
  LIVE, polling SAME exec_command session99752 every45s. Session99752 is the
  projection, not GPU replay. Resume cell6892 with functions.wait; do not
  launch a duplicate monitor/projection. Loop40polls stops on commandexit.
- Lastdisk162971496448B at18:20,hostavailable110909648KiB.
  One projection only; recheck before another heavy task. No deletions.
- Plan/V1 readFULL aftercompaction; unchanged protocol used. monitor-experiment,
  analyze-results,academic-plotting,github-sync SKILL.md readFULL. No plotted
  figure selected; failed qualification uses provisional status table.

17:58 static postprocessing audit found D139 RPC-only analyzer requires
native_contract_completion_pass and4000native successes (analyze_control_path_overhead.py
lines520-566), so prepared run_rpc_audit1.sh is NOT applicable to D145. Do NOT
execute it or weaken that guard. Existing curator reports conditional RPC
fields, and existing --native-timeline --allow-failed retains all4000 IDs and
failure clocks; use these for D145. No analyzer/source edits or repeated tests.
Previous goal turn PROGRESS (cleanup/metadata); this turn has new applicability
evidence plus verified SAME projection wait. Fullplan/ledger/skills reread;
completeV1 already read preceding turn, reread before actualcomparison.

NEXT: poll SAME projection -> verify Exit0/all4000 and source identity ->
capture exact projection empty-domain cleanup -> run prepared bounded curation
and failure breakdown -> failure-aware occupancy and conditional RPC fields -> table/interpretation/seal/
backup. cleanup_metadata_full1.sh now exists with actual identity, alreadyrun.
Other prepared helpers remain unexecuted. Do not repeat metadata/remote copy/
old projections/D144 tests. Full exception text is the next new evidence;
no speculative patch or cap/deadline change. Baselines PAUSED;3BTPOT/outputhash,
bothmodelperformance,numericadapter,warm/Resident,M1M2/A1-A5/S1-S13 OPEN.
18:00 prepared cleanup_projection_full1.sh by reusing exact empty-domain
protocol, bound to actuald7ce840a89db418ea1749078629b4d56; syntaxPASS, NOTrun.
It requiresTasks0/cgroupempty and cannot stop the live projection. New
D145_FULL_W0_FULL1.md is an unsealed provisional status table, not a formal
comparison. Current serving code remains backedfc486083; no new tests or
source changes. No backup claim for provisional D145 documents yet.
18:11 new lightweight timeline evidence (no secondary heavy analysis):
deploymentarrival_end353241.4238260309; firstfailureterminal354366.783730505,
18.7559984minlater. Service logalreadydone3731/fail0/backlog269 beforeerrors.
Therefore laterquarantines cannot explain all earlierbacklog; keep failure
correctness andthroughput/controlwaiting asdistinctquestions. Added to
provisionalD145doc; exactlatencydecompositionstillpendingprojection. No causal
CPU/GPU-specific claim. PreviousgoalturnPROGRESS(analyzerapplicability)+
verifiedwait; currentturnnewtemporal evidence+verifiedSAMEprocesswait.

## D145 launch/save history — superseded by CURRENT above

2026-10-01 16:06:08. Launched ONCE in tmux tc-d145-7b-full1. Runtime backed
fc48608387e75d4400034ee412d3456c835ba41c. Goal ACTIVE/incomplete; launch turn
PROGRESS (ordinary Full launched, not another diagnostic). Baselines PAUSED.
Latest user cache permission already fulfilled D78/D80; NEVER rebuild.

- Reused D137 ordinary launcher/config/remote helpers. Exactly three fresh
  owned paths, otherwise identical 7B configuration/D89 profiles. Full4000,
  source42/W0/formal0; no prefix or detailed stack/CPU profiler. New D144
  candidate only since D143; comparison to D137 also contains D141 correctness
  delta, so not an isolated causal estimate or formal SLO comparison.
- Preflight PASS216refs/147protected/Plan/V1; SHA
  26039dcdaf3e98e23031f8c5e275d98a14a5724ce74521e7777aa547a08a2dd4.
  Actual8b13d994524f49189e389e26f1ccaf9e emptyclosed/events0; health actual
  767a6845b1be424c988cee124af77ec8 emptyclosed/events0. Predictedgrowth32GiB;
  latest prelaunch local free178151333888B,hostavailable112424366080B.
- Actual service primelora-tc-svc-46f7a42b36914d08af1aef936f577752.scope,
  Invocation a7c0d7d9c1df471e8111176d0dc8a5c7,72/80GiB/swap2/CPU4-23,28-47.
  Aux primelora-tc-aux-7b16694bb9d447769eb74d80de743d98.scope,
  Invocation e212cd03e1ba4a0fa4e973e3a08e7cbd,3/4GiB/swap0/CPU2,3,26,27.
  Limits read back. Replay publisher1853942;watchdog1853984.
- Remote3b PID1918283/6f3e7a1bf1c04a1a9086e84fa46d42b1;
  7b PID1918285/360a8200c5264d8fb0735eafbb38b45b;
  monitor1918289/fbcc9cc86e9d4d5d8469839c49e75fd3,
  unit primelora-artifact-monitor-d145full1.service. BothNIC1000/full.
  7B healthclock remote-process-monotonic:edf4d3248d484db5a297c3d8a45d0e89.
  Remote log /home/lab14/primelora_remote/tc/d145_20261001/remote_monitor_7b_full_full1.log.
  Exact stop helpers prepared, NOT executed. No remote management/hash/copy/
  cleanup during inference. Cache unchanged; no new weights/traces/pools.
- Raw results/ieee_tc/p2_backend_qualification/d145_20261001.
  Last startup sample55:service6559006720B,host105597747200B,events0/no warning.
  No terminal launch.json or final performance result yet. Old apparent running
  run-* scopes all TasksCurrent0; not active competing analyses, untouched.

Latest17:42:42 SAME run LIVE, exact Invocation verified; requests all terminal:
arrived4000/done4000/ok3996/fail4, backlog0. Request phase ended; result-save/
final cleanup still running. Full-completion requirement NOT passed.
Service125tasks; actual domain
unchanged. Four GPU cores1856562/1861679/1861977/1862164 previously verified
in exact domain/CPU4-23,28-47. Sample5715: service40164638720B,
host75120238592B,swap0/events0/no warning. Last disk check17:42:59
175291957248B, watchdog ongoing; exact aux Invocation active/5tasks last
confirmed16:47:23.
No terminal launch.json, serving/config change,
remote management or second experiment. Previous and current turn VERIFIED
Current turn gained new failure evidence: request_terminals req_03842/03843/03844
RuntimeError at354366.783730505/354366.784857652/354366.785862901; native
contract false. Bounded service tail says subprocess_native_rpc_failed_no_re...
(truncated display; full error cause not yet available); live runtimes fell4->2.
Fourth RuntimeError req_03996 at354601.481890174. All four native_contract=false;
no timeout observed in these terminal rows. Full error text not yet inspected.
No OOM/high/max/swap or safety abort observed. Continue SAME saving process,
no retry or safety change. Normal source JSON now writing:2781019661B at
17:42:59 (fresh mtime), NOT complete, no reading/hashing/projection yet.
GPUcompute list empty17:40:42, but final launch receipt absent; wait for actual
terminal release/cleanup confirmation. Old terminal banner (cost/local-sim/5000ms) is legacy display, not
frozen V1 qualification; use postcleanup physical/native/remote evidence.
Monitoring cells6781/6796/6799/6814/6826/6831/6835 completed and closed.
No live exec cell/session. Actual experiment is tmux/service/aux above; NEVER
relaunch. Continue lightweight polling of that exact invocation and writing
file. Inspect normal versus interrupted final schema before analysis.
Frozen V1 reread fully this turn; SHA unchanged. analyze-results and
academic-plotting SKILL.md read fully, latter's plotting references not yet
selected/read (no figure generated). No result analysis or new GPU run.

During this wait, reused D137 raw postcleanup helpers into d145 (PREPARED ONLY):
collect_full1_preliminary.py,inspect_full1_metadata.sh,project_full_full1.sh,
summarize_full_full1.py,curate_full_full1.sh,collect_full1_failure_breakdown.py,
run_occupancy_audit1.sh. Existing D138 compare_request_contracts reused verbatim;
compares only the already sealed34MB D137 projection, never its15GB original.
On current failed requests the cross-token contract comparison is explicitly
unavailable, not silently matching missing fields; failure tables still retained.
Curator corrects the two previously nonexistent parent_response_* names to the
actual parent_rpc_* producer fields (D139 evidence). No serving change.
Bound D145 remoteclock/service unit/runtimecommit in raw helpers. Shellsyntax
PASS; Python syntax/functional execution still pending after inference.
cleanup_local_aux_full1.sh bound to exact actual auxiliary identity, NOT run.
run_rpc_audit1.sh reuses unchanged D139 analyzer, single D1457B only, explicit
future sealed SHA argument; no repeat of previous audits/tests. Remote journal
filename still unknown: discover/copy ONLY after terminal and exact remote stop.
No new outputdata/projection/curatedresult produced; do not run helpers early.
16:42 static reread of prepared metadata/projection/curation/failure helpers:
normal outcome and exact terminal/resource prerequisites retained; no execution,
source change or acceptance of a performance result. Source-reference map does
not pin the mutable status ledger. Reuse existing helpers only after cleanup.
16:49 prepared copy_remote_full1.sh by reusing D143's copy protocol. Exact
journal still unknown and must be discovered by frozen health clock AFTER
shutdown, then passed as a validated explicit path. Helper requires terminal
physical release, inactive local service/no GPU, stopped remote services, fresh
outputs, matching clock and source/copy SHA. Syntax PASS only; NEVER executed
during inference. Curator source_refs includes this helper; no serving edit.

NEXT: monitor SAME run -> terminal/physical cleanup -> exact remote stop/copy
-> bounded analysis/table/interpretation -> scoped backup. Do not repeat D144
tests or old giant projections; no second optimizer/cap/deadline changes.
3BTPOT/outputhash,bothmodelperformance,numericadapter,warm/Resident,baseline,
M1M2/A1-A5/S1-S13 remain OPEN. Do not infer SLO qualification from token counts.

## D144 candidate qualified and BACKED — completed history

2026-10-01 15:49. D143 diagnostic BACKED f70595dc98fb9b2ae013171142bbb87085faa991;
remote exactHEADverified. Sixexplicitfiles/61payloadsecrets/checksum/diffPASS,
user manifest excluded. D143 closed, don'trerun. Allremotesinactive/cacheunchanged.

D144 bounded candidate in gpu_monitor.py: queryis_pinned onceperview (still
comparealiases), andwhenstagedempty copythecurrentregisteredplaininventory
instead ofwalkingthesametensorsagain. No cross-callcache, formula/capacity/
timeout/backend/workload/remote changes. Nonemptystaging still fullyobserved.
IEEEowner,epoch,leases,budgets,content,errors,fences remainunchanged.
PrimaryvLLM0.30 model_manager.list_adapters/UniProcExecutor plus PyTorch2.13
Memory.cpp verifiedonline; localowner.staged_models inspected. Thirdparty
genericperformancetargets notused. DocsPytorch2.13webpagesunavailable; native
taggedsourceprovided actualisPinnedPtr implementation instead.

- Two minimaltests RED:9pinqueriesfor6views;empty-stagingquery2inventories.
  Actualc7f1cbd7858a4579b8ce1a3672e4fe47 emptyclosed15:47:14/events0.
- SAMEtests GREEN aftercandidate:2PASS0.005s, actual294e5f85469a440b9ee0329f658cbc34,
  emptyclosed/events0. Extendedtestalso checksindependentreturnedpayloads,
  nextobservationfreshness andnonemptystagingstilltwo inventories. No GPUused.
- ONEregression completed628PASS77.150s,command86.41s/RSS1122388KiB;
  actualbf26b36d67434255a3701fa70181320a,emptyclosed15:51:03/events0.
  Do NOT rerun. All actual3/4GiBswap0/taskset2,3,26,27; rawd144_20261001.
- Seven existingtinyCPUfixture cases compare exactbackedf70595helper output/
  errors to candidate: all equal. shared/partial/packed/stagedalias9->6pinqueries,
  empty0->0;invalidextraTensor4->2,pinningconflict7->4 withsameerrors. RealCPU
  Torch2.8.0+cu128/CUDA_VISIBLEempty, NOTnative2.13CUDAperformancequalification.
  Actualfa521b88303a4ab7ad1c92ec040238df emptyclosed15:52:48/events0.
- Curator1 failed BEFOREoutputs: preflight refs mixabsolute/relativepaths,
  changedfile allowlist comparedonlyrelativepaths. Retainedfailedsource/log;
  cd435e4ab8424ffbaf96e05a3665d17c emptyclosed15:55:15/events0.
  Curator2 normalizesidentity;206prelaunchrefs checked, declared3changedfiles
  againstexactparentSHA andallotherrefsagainstcurrentbytes;147protected/Plan/V1/
  source/syntax/secretsPASS.
  Actual4ece9a4811d24efe81063f6e21b60b3d complete0/.73s/RSS31100KiB,
  exactemptyclosed/events0. No tests or GPU rerun tofixanalysispathhandling.
- CuratedSHA61a5ecc421d798cb383be44dc984afa99f5caa5ab8fa39d4cee910d9071b5ee8;
  bundle37members105940B SHA19100fc752c6bdfb3072684eb3e0de847a183ab3423308fb2fe3c92f5e18b3cc.
  DocD144_SAME_OBSERVATION_HOST_INVENTORY.md SHApinned; do not editafterseal.
  Only3source/testfiles changed; not usermanifest. No liveexec/tmux/GPU/scope.
  Sourcecandidate qualified; backup completed below.

BACKUPDONE fc48608387e75d4400034ee412d3456c835ba41c pushed to
faaslora_origin/retry14_continuous_queue_v2; exactremoteHEAD verified15:57:03.
Eightexplicitfiles/45payloadsecrets/checksum/diffPASS; usermanifestexcluded.
No liveexec/tmux/GPU/analysisscope; bothremoteartifactservicesandmonitorstopped.
Lastdisk178149642240B; recheckbeforeheavy. Thisbackupnote is theonlynewledger
deltaaftercheckpoint. Do NOT repeat RED/GREEN/regression/equivalence/curation.

NEXT: reuse existing ordinaryFull launcher/preflight/remotehelpers for ordinary7BFull
4000/currentprofiles/config,noprofiler. No claimedTTFT/SLO/GPU-simprovementyet.
Bothmodelperformance,3BTPOT/outputhash,numericadapter,warm/Resident,baseline,
M1M2/A1-A5/S1-S13 remainOPEN. BaselinesPAUSED. No nextGPUprepared/launched.

## Recent history archive

D138 closure, D139–D143 diagnostics and D144 qualification remain fully
preserved in [the verbatim ledger archive](EXECUTION_HISTORY_THROUGH_D145_LAUNCH_20261001.md).
All936lines verified byte-identical before removal from active ledger;
SHA03b4a33bbb4eca65400049eb411b24a871704a8ef1aee34ceb4cf10ce81a0496.
Historical LIVE/NEXT instructions there are superseded by CURRENT above.
Do not repeat completed tests, diagnostics, projections or remote publication.

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
- Historical12:20 onlyD138projectionwaslive; CURRENT overrides: allfinished.
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
