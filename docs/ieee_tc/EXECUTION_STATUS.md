# IEEE TC execution status

## Latest — D95 unsent generation boundary CPU-verified; checkpoint then source-conflict diagnosis

2026-09-28. Supersedes older NEXT/LIVE. No live GPU/remote run; baselines PAUSED.
Actual runner/proxy tests reproduce D93 req00333's pre-send rejection boundary:
old code attempts retirement with no native generation. Two red tests failed;
34 targeted and 848 related regression tests PASS (0.224s / 48.032s).
Receipt distinguishes positively not_submitted from may_execute/unobserved;
only the former avoids nonexistent-generation retirement. Lost replies retain
ownership; other unresolved RPCs are never cleared. No retry/formula/profile/
trace/deadline/remote change. D95 doc has immediate correctness status table;
curated 20260928_d95_generation_submission.json has seven source/log SHA refs.
Raw d95_20260928. D94 already backed up; do NOT repeat its work or tests.

NEXT: verify protected files/secrets, scoped D95 commit/push, then ONE CPU causal
source-conflict workstream. Per-request full source collection can partially
commit; global native epoch changes on acquire/release even without copy/tier
change. Quantify actual repeated work before choosing a correction. Existing
source incarnations may distinguish unrelated reference mutation from replaced
copy; never simply remove epoch/tier/capacity/pin checks. No optimization yet.
No unchanged GPU replay. Long queue remains unexplained; D95 is not Full success.
After causal validation return canonical3B4000 then7B; no repeated D78/D80 cache,
D88/D89 profiles or D90 prefixes. Warm/Resident/M1/M2/A1-A5/S1-S13 pending.

## Latest — D94 physical-owner retirement fixed; CPU verified, no GPU replay

BACKED UP: df09f25986168b296941883978caeb15a9e5036f pushed to
faaslora_origin/retry14_continuous_queue_v2; fresh remote SHA matched.
Six scoped files, 841 tests, five source/log SHA references, 147 protected items
and credential exclusion passed. User manifest excluded. No extra backup loop:
NEXT is the unsubmitted-versus-uncertain generation boundary below.

2026-09-28 04:41. Supersedes all older NEXT/LIVE instructions below.
No live inference/remote monitor. Baselines remain PAUSED at 9e2cf289.
D93 failure evidence already pushed df47980; no need to repeat that backup.

D94 actual CPU counterexamples reproduced: 4 draining owners allowed GPU 0;
scale-down removed membership before asynchronous shutdown; failed shutdown
lost its owner. Pre-fix 3 tests: 2 failures/1 error. Fix selects from all retained
members and keeps draining membership through teardown across retirement paths.
Only successful cleanup removes membership. Physical/native checks unchanged;
no blind retry, no IEEE formula/profile/trace/deadline/remote change.

19 targeted tests PASS 0.471s; final 841 regression PASS 49.844s in bounded
offline CPU scope. Raw d94_20260928/regression1.log; doc D94_PHYSICAL_OWNER_RETIREMENT
contains immediate status table; curated 20260928_d94_physical_owner_retirement.json.
147 protected entries and plan/metric SHA unchanged. No GPU performance result.

NEXT: scoped checkpoint backup, then continue the same failure diagnosis:
1. req00333 was withheld BEFORE generate send/binding, yet retirement was attempted;
   distinguish unsubmitted from uncertain submitted work without weakening guards.
2. Bound/test concurrent source-observation and selected-source conflict amplification.
Do NOT rerun unchanged GPU workload just because physical selection is fixed.
Long waiting is still unexplained; all-draining liveness is not restored by this fix.
Do not repeat D78/D80 cache, D88/D89 profiles, or D90 prefixes. Complete canonical
3B Full4000 then7B only after causal validation. Warm/Resident/M1/M2/A/S pending.

## Latest — D93 full4000 attempt2 FAILED/CLEANED; backup then causal CPU diagnosis

BACKEDUP df4798095caf2458d7e4c46254cc647561391b01 PUSHED/freshremote matched.
Three scoped evidencefiles;19refs/147protected/secretsPASS;288basic smoke tests
PASS24.142s in bounded4GiB/offline scope, no runtime edits. NoGPU/tmux/remote
remains. Rawfull2_evidence_push_receipt.json. NEXT is causalCPUwork, notanother
evidence/backup/wait loop or GPU replay. OlderLIVE is superseded.

2026-09-28 04:28. Supersedes LIVEbelow. No live model/tmux/service/remote monitor;
baseline9e2cf289 remainsPAUSED. Execution17bb348 unchanged. Outcome originalerror
IEEE activation failed from physical GPU still has compute/owned/unknown contexts.
Fourready initial/natural runtimes laterdraining afterdeadline/ownership trouble;
fournewactivationattempts unresolved. Outerexternalreplay failure issecondary.
Population:4000planned,2513publisher submitted,2482ingressreceived,2417runner
tasks;320success/native matched,29TimeoutError,1RuntimeError(req00333),2067
whole-runCancelledError. 1583notrunnerstarted includesbuffered/inflight/future;
do NOTcallall1583 unarrived or only2513 theoffereddenominator.

All4physicalleasesreleased, nativecensusclear,servicepathgone,nohardkill;
ownerHOST/NVMe rootsremoved; outcome cleanup/measurementerrors empty.
2309resourcesamplespeak21293617152B,minhost95746981888B,high/max/OOM/swap0.
U_obs8901.365137GPU-s incomplete NOTfullcost. 45remoteUUIDpairsexact,
104449531wireB/1756692668verifiedlogicalB,requestpacking0. Remotejournalfile
transfers-24d46dbbdee2450ca4d4109438b8379c.jsonl has clock3b7dfc18...;
clockUUIDandfilenameUUID differ. Finalremote monitor38MiBcopied; matchingmonitor
a19e7d1b... stopped AFTERinference; bothartifactservices+monitor inactive/MainPID0/
success. Emptylocalaux matchingfaadb5a... stopped; noactualGPUjob remains.

Curated20260928_d93_3b_full_w0_attempt2.json SHA9750d4b0053f83f3cdd16527ab5f96ebcba86a3638f2586863db9215551fc3ea,
19sourceSHArefs/147protectedPASS. D93doc immediatefailuretable complete.
Rawmain_outcome584MiB staysunchanged; D92curatoradaptation usesbounded4GiB jq
projection (full_attempt2_analysis_projection.json), notfullPythonJSONtree.
Conditional320success:meanuserTTFT540.250s,dispatch517.526s,service22.724s,
native.430s; notfullperformance/ranking/CI. 201successwithselected-source retries,
1242total/max47; NOTfullsnapshotdiscardcount. Onefile+oneGPUhandofffailed
ConfirmedSourceConflict, exactpredicate notcaptured; noFullqualificationclaim.

NEXT scopedfailure-evidencebackup, thenONEcausalCPU workstream: deviceallocation
uses get_slots(runningonly) although drainingowners stillphysicallyholddevices;
testactualselector/cancellation/retirement boundaries. req00333waswithheld by
unresolvedRPC BEFOREnewgeneration, butcleanup triedunknownnativebinding.
Do NOTrelaxphysical/referenceguards or addblindretry. Afterownershipcorrectness,
testnativeobservation/sourceconflict amplification; mainlinefullreplay mustthen
revalidateall4000before7B/baselines. No newGPUjob/profile/cache/fullpoolwork now.
Warm/Resident/M1/M2/A/S remainpending, zero-weight numerical limitation unchanged.

## LIVE — D93 canonical 3B4000 Full W0 attempt2, 2026-09-28 03:39:55

04:17:02 progress:2384submitted/330terminal/315success/15TimeoutError,
315native-contractmatched;watchdogsample2197,service19.51GiB,
hostavailable89.54GiB,disk309.39GiB,high/max/OOM/OOMkill0.
Actualscopeactive at every poll; freshest sample0.74s old.
Scope remained active, nooutcome/launchterminal yet. Backlog grows and native
generation remains much shorter than user waiting; NOTcompletequalification.
Auxiliaryinvocationfaadb5a13abc4616bb7026362a7a55d5.
Continue monitoring THIS run; no duplicate launch or source changes.
This turn was verified monitoring/read-only diagnosis, no runtime edits or new
experiments. Tool-only polling cells1711/1715/1721 were ended; they are NOT the
experiment. Tmux/service/watchdog/external publisher remain live. Do NOT try to
resume those closed tool cells, restart the experiment, or treat slow progress
as a termination condition. Planned4000arrivalspan3963.9s plus fixed1800s
planned-arrival deadlines; terminal files must be checked, not guessed from ETA.

New qualification evidence: first4timeouts observed04:15:34. req00150/00152/
00157/00160 terminal at1803.015/1802.980/1800.209/1800.324s from plannedarrival.
Thus this configuration CANNOT pass4000/4000 completion even if the remaining
run completes. Keep the same replay until its terminal/authorized safety stop,
preserve alloffered4000 and failures. Do NOT advance7B with a success claim or
change timeout/trace midrun. The4terminal rows have instance_id=null because
_run_offered_request passes result=None on exception; this is NOT proof they
timed out before replica selection/native dispatch. Need retained per-request
failure observations after run end. These are1800sdeadline failures, NOT5sSLO
classification. NoOOM/high/max or source-snapshot ValueError observed so far.

Read-only candidate diagnosis (NOT proven, no runtime edits): each routing
attempt gathers source_snapshot from all native workers; each response rebuilds
HOST inventories twice and the GPU tensor inventory. Partial per-slot commit can
then reject a gathered view that another request has overtaken, immediately
restarting the entire gather. Global admission bounds these callers (3B32), so
do NOT claim every offered waiter polls all workers. Native controls already use
fresh connections, NOT the generation-channel pool; shared-pool deadlock is not
established. Inspect actual end-of-run stages and test count/progress separately
after cleanup before accepting an optimization. Official v0.30 core.py input
queue is drained before stepping, worker_manager.py keeps native mutation in
the single-threaded core; these motivate bounded/coalesced observation research,
not proof of this run's exact cause. Sources checked 2026-09-28:
https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py
https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py

Source17bb348602f00c66520c66d9b6074cc70a1e514e PUSHED/frozen; tmux
tc-d93-3b-full2. Rawd93_20260928/3b_full_w0_attempt2/launch.json and
launch.launch/,3b_full_w0_attempt2_console.log; D92 attempt1 unchanged.
Canonical existingrunner, actual4000 externalarrivalmap,60snotice,1800splanned-
arrivaldeadline, sameD88/D89 configuration; only codefix andnewownedroots.
Auxa917b3643e2c482290182c00eec26bc2; serviceabd1a98f3730468ab6e0ec5ab4ecdd42
scope/invocationb92baf0317bc4f288304426507c98a06. Watchdog1578138 actualworkers
inside72/80GiB/swap2 domain. Beforelaunch13refs/147protectedPASS,bothhealthPASS.
Remote3Binv2e00b66fe4e24c25905d6320fe90ae46 PID823499;
7Binv9abca9bd988f47b2b349abd5ad7b1b71 PID823501;
monitorinv a19e7d1b00ec43ed89939cdb95daae51 PID823504,
unitprimelora-artifact-monitor-d93full2.service,
remote tc/d93_20260928/remote_monitor_3b_full_attempt2.log.
NOsource/remoteconfiguration/restart/hash/cleanup duringinference. MonitorSAME
attempt toterminal thencleanup/validation/status table BEFORE7B/nexttask.
AllolderLIVE/NEXT notes superseded. NoFull/SLO/numericalqualification yet;
baselinesremainPAUSED. Do NOTlaunch duplicate, reprofile orrebuildcache.

## Latest — D93 snapshot counterexample fixed and CPU-verified; full replay next

BACKEDUP17bb348602f00c66520c66d9b6074cc70a1e514e PUSHED/freshremote matched.
Six scopedfiles/13SHArefs/147protected/secrets PASS; usermanifest untouched.
Prelaunch localdisk334048542720B,MemAvailable113298712KiB,swap0,GPUidle;
bothNIC1000/full. Remote services inactive before new attempt activation.

2026-09-28 03:38. No live inference/remote/tmux; baselines9e2cf289 PAUSED.
CPU counterexample reproduced D92's same complete-file-view ValueError when
the previous capacity observation ages without content/signature changes.
Original D92 exact predicate remains unknown because failing fields were absent.
Fix: finish source refresh BEFORE epoch freeze; derive HOST/file/replacement
capacity from one inventory, reuse each confirmed source once. Same owner lock,
content/ref/reservation/capacity checks and execution revalidation retained.
Guard unchanged, now records failing owner/epoch/time/universe fields.
No equation/config/profile/trace/remote semantics change, no retry/sleep/fsync.

Targeted28 tests PASS1.672s; final838 regression PASS48.077s/offline.
First regression stopped because old dummy-model test attempted online HEAD;
log retained, no model download. Rawd93_20260928 andD93doc immediate status table.
3B attempt2 config reuses D92 with ONLY new output/cache roots. Do NOT repeat
CPU assembly, prefixes, whole-pool/cache work or D88/D89 profiling.
NEXT verify protected/curated, scoped backup, one3B4000 W0 canonicalFull attempt;
cleanup/validation/table BEFORE7B. All warm/Resident/M1/M2/A/S remain pending,
zero-weight numerical discrimination caveat unchanged. No Full success yet.

## Latest — D92 full4000 attempt1 FAILED/CLEANED; CPU snapshot diagnosis next

Evidence BACKEDUP b280c23e84ad020ee8727a13215dbad8065be847; fresh remote SHA
matched again after compaction. Raw full1_evidence_push_receipt.json records it.
No live experiment. D93 CPU hypothesis: source footprint refresh may advance
the owner epoch after the budget was captured, even under the existing lock.
Not yet proven to be the exact D92 failing predicate (inputs were not retained).

2026-09-28 03:26 supersedes LIVEbelow. Originalmain08cbf45:4000planned/
27submitted/23success/4cancelled/3973unsubmitted; incomplete NOTperformance.
OriginalValueError:automaticplanningrequiresonecompleteconfirmedfile-owner view,
residencyepochinputguard. ExactpredicateinputNOTcaptured; doNOTassumecause.
12residencycomplete/5superseded/1failed;file14complete/5superseded/2cancelled.
1activationready/3cancelled;all4physicalleasesreleased,actualGPU/servicegone.
13HTTPUUIDpairs exact,30210124wireB/542464828verifiedlogicalB,requestpacking0.
164resourcesamplespeak19352363008B/minhost95952441344B;high/max/OOM/swap0.
Mainoutcome preservespartial/native/UUID/mechanismevidence;no snapshot/cleanup
errors,ownedHOST/NVMe removed. Outerpublisher subsequentlyfailed,launchlabel
protocol_or_launcher_error/service-15/watchdog0 doesNOTreplaceoriginalValueError.
Observed329.281494GPU-s ispartialU_obs,NOTfullperformance/correctrequestcost.

AFTERinference,matchingremote monitor0574f3ff stopped;bothservices+monitor
inactive/MainPID0/success. Finalmonitor/currentjournal copied;13pairsverified.
Aux72637aed131d4e358bab9f87d1da642a stopped ONLYafteremptyprocs/populated0.
NOliveGPU/model/tmux/remote. D92docimmediatefailuretable andcurated
20260928_d92_3b_full_w0_attempt1.json (16SHArefs) complete;147protectedunchanged.
NEXT evidencebackup thenCPU isolateactualpreparation_snapshot vs complete-view
guard;oneexistingownerlock alreadycoverscollection,so don'tassertsimplemissing
lock. NoGPUrerun/7B/baseline orcompletedprofile/cachework before causaltest.
Allformalmatrices/warmSLO/Resident/numericalcorrectness remainpending.

## LIVE — D92 canonical 3B4000 Full W0 attempt1, 2026-09-28 03:18:53

Source08cbf45c1920019373ab4dfd8ba1e7840a44c519 unchanged/pushed.
tmux tc-d92-3b-full1; rawd92_20260928/3b_full_w0_attempt1/launch.json,
launch.launch/ and3b_full_w0_attempt1_console.log. Existingcanonicalrunner+
actualexternal4000arrivalmap,60snotice,1800splanned-arrivaldeadline,D88parent/
D89profiles,original500pool. One runtimecap8/aggregateupto32, naturalcontrol.
This isdevelopmentqualification, NOTformalM1/M2/SLO/numericalcorrectnessproof.
Aux22de466b3b1a403ba7d38f3510128417,service7ec035e7dfd74913abb40c8323990bea
scope/invocation500bf55e04f445abbecc785af4bba01e;guardedactualworkers+watchdog.
Beforelaunch bothdirecthealthPASS,20SHArefs/147protectedunchanged,GPUidle,
312GiBdisk/108GiBMemAvailable/swap0,bothNIC1000/full.
Remote3B9329cc89ccaa428cadbfa69331b536a6 PID804093;
7B3b6407dd455642d08f790c0dbb9596d5 PID804095;
monitor0574f3ffd87f483796d33caf6994e8e7 PID804098,
unitprimelora-artifact-monitor-d92full1.service,
logremote tc/d92_20260928/remote_monitor_3b_full_attempt1.log.
NOsource/remoteconfiguration/restart/hash/cleanup duringinference. MonitorSAME
attempttoterminal;cleanup/validate/table BEFORE7B oranyotherexperiment.
Baselines9e2cf289PAUSED. D78/D80/D81/D88/D89/D90prefixescomplete,dontrepeat.
AllpriorNEXT/nostart/livestate notesbelow arehistorical, not restart instructions.

## Latest — D92 deadline/outcome/owned cleanup verified; Full replay next

BACKEDUP08cbf45c1920019373ab4dfd8ba1e7840a44c519 PUSHED/freshremoteSHA
matched. Sixscopedfiles,835tests,20sourceSHArefs and147protected/secretschecks
PASS. UserdirtymanifestNOTstaged. Noactualreplayyet. Rawcheckpoint_push_receipt.

2026-09-28 03:16. No live model/tmux/remote; baseline9e2cf289 PAUSED.
Implemented explicit finite request deadline from PLANNEDarrival; development
1800s. Alreadyexpired work is not submitted. Physical terminal recorded OUTSIDE
timeout conversion; externalCancel remains interruption, cleanupfailure remains
failure, no fabricated nativecompletion/release. Legacyunset behavior unchanged.
Canonical main finally retains partial requests/remoteUUID/mechanism journals and
original+secondaryerrors. Only fresh run-owned HOST/NVMe roots can be cleaned;
actual file owner/ref/movement/identity checks retained. No IEEEformula/config/
profile/input/remote-protocol change. Historicalledger archived verbatim, notlost.

835related/basic/externaltests PASS50.078s; actualCPUmain both4000/500,1800s,
exactD88child+D89profiles, deliberatepreactivationfailure outcome andownedcleanup
PASS14.180s. NOGPU/HTTPperformance inthesechecks. Rawd92_20260928 andcurated
20260928_d92_main_deadline_outcome.json (20verifiedSHArefs),D92docimmediatetable.
147protectedunchanged,planSHAfe6c05b0 andmetricV1SHA5f0732ef unchanged.
Redtests andfirstoutcometest wrongkeywordfixture retained; finalregressionpasses.

NEXT scopedbackup thenonecanonical3B4000 W0developmentreplay viaEXISTINGscope/
actualexternalpublisher/physicalledger usingnewD92config (nowdeadlinewired).
Cleanup/validation/table BEFORE7B. DoNOTrepeatCPUassemblies,D90prefixes,
D78/D80/D81publication/coverage,D88/D89profiles orresumeServerless.
Full/M1/M2/A/S notqualified/started; warmSLO/Resident unmeasured andexistingzero-
weight numericaldiscrimination limitation remains. Preparedconfig is NOTa run.
Safety at03:11:312GiBdisk/108GiBMemAvailable/swap0/GPUempty, recheckbeforelaunch.

## Latest — D91 main-entry integration CPU-verified; no live inference

BACKEDUP1148078783831ab96e2d817c596161fb50410670 PUSHED/freshremoteSHA matched.
Sixscopedfiles,22SHArefs,147protectedentries andsecretschecksPASS;userdirtymanifest
NOTstaged.825tests andbothactualmainassemblies complete;no D91 unit/model/tmux/
remote job remains. Rawcheckpoint_push_receipt.json. NEXT deadline/failure/cleanup
integration,notanotherCPUassembly/profile/backup loop. BaselinesremainPAUSED.

2026-09-28 02:56. Supersedes older NEXT/LIVE notes. Completed actual canonical
`_main_async_impl` assembly for BOTH models:4000 existing requests/500 adapters,
exact D88 child configuration after original parent assembly,D89 measured profiles.
One/four-runtime capacity3B8/32,7B2/8. Stops deliberately before activation;
ingress/notice are explicit CPU fixtures,NOT real transport/resource/performance
qualification. Main raw input reuse now skips raw dataset construction; IEEE
remote setup only reads frozen index+SHA-checked small configs, no payload scan,
repair/generation or local fallback. Actual external ingress determines open-loop
mode independently of provenance labels; old non-external behavior retained.

Unconditional Full rejection now replaced with explicit executable prerequisites:
pending native owned TP1, exact measured profiles+activation layout, actual shared
demand/movement/admission binding,published remote/no artificial delay, started
external full-map replay,common60s notice and physical ledger. Does NOT certify
numerical correctness, SLO or superiority; receipt formal_comparison_qualified=false.
Existing per-worker/ownership/source/reservation/release checks unchanged.

Final825 related/basic/external testsPASS47.471s;99 launch-onlyPASS11.591s earlier.
Initial807 regression had1failure ONLYold exact error text; test still requires
rejection and no legacy startup,updated to new missing-owner reason. Red input
counterexamples2failure/3error retained. Final-source real main assemblies both
PASS in13.485s. Rawd91_20260928 incl assembly_final/,curated
20260928_d91_integrated_main_entry.json,D91_INTEGRATED_MAIN_ENTRY.md status table.
No new GPU/remote/weights/trace/profile/baseline run. No tmux/model/remote job.
Disk312GiB,host108GiBavailable,swap0. Protected/source verification and scoped
backup follow; NEVERstage userdirtymanifest or unrelateduntrackedfiles.

NEXT complete canonical main finite deadline FROM plannedarrival (development
1800s) and distinguish request timeout from whole-run interruption; preserve
interrupted replay/native/UUID/mechanism evidence on main exception; owned HOST/
NVMe cleanup receipt. THEN3B full4000 development replay viaexisting scope/native
environment/externalpublisher/physicalledger,cleanup+validation+table before7B.
Config files prepared byCPU audit lack the yet-to-wire request deadline and are
NOT launch-authorized scripts. Do NOT blindly launch them or repeat D90 prefixes,
D78/D80/D81/D88/D89. Baselines9e2cf289PAUSED. Full/M1/M2/A/S notqualified/started;
warmSLO/Resident stillunmeasured;originalzero-weight numerical limitation remains.

## Authority and recovery

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before work, after compaction, and before experiments. AGENTS.md applies.
  No subagents authorized. Default mode; execution authorized.
- Approved snapshot PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md. Plan SHA:
  fe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
  Historical snapshots and unsuccessful attempts remain preserved.
- Active goal incomplete. G1=correct complete workload/common SLO then minimum
  physical lifecycle GPU-s; G2=common budget/SLO then tail TTFT. CE supplementary.
  M1/M2/A/S formal matrices have NOT started; no optimality claim.
- D74 Serverless3B repaired finished/analyzed/plotted/backed up. Baselines PAUSED.
  Do NOT run the prepared3B original. Complete Prime IEEE Full first.
- User APPROVED once-only immutable compressed delivery cache, not new weights,
  trace or a second extracted pool. D78 publication and D80 delivery functionality
  are COMPLETE; DO NOT repeat either full pool.
- Formal artifacts are already published before deployment notice. No request
  packing/compression/formatting/full-pool hashing. Necessary reads, transfer,
  request-caused competition and local loading stay observed. Preserve legitimate
  NVMe/HOST/GPU reuse/fewer fetches. Never subtract old cumulative packing from E2E.
- Only artifact-only node exits150GiB disk floor; use actual incremental space,
  log and safety reserve with1.5margin. Inference150/100GiB and ALL OOM rules remain.
- No unrelated remote restart/config/hash/cleanup during actual inference.
  Shared service conditions freeze after qualification. No silent local fallback.


## Completed checkpoints and remaining mainline

- D78/D80: once-only published gzip cache complete, both500/500 HTTPcontents
  verified,1,581,215,261B. Request packing0. D81four-way12/12both complete.
  Do NOT rebuild publication/re-download whole pools or repeat those checks.
- D88: admission-enabled owned-native source observations complete:
  3B368/368,7B92/92, remoteUUIDs exact, allresources released, plots/backup done.
- D89: measured initializer exports complete in
  paper_results/ieee_tc/p2_backend/d89_{3b,7b}_initialization/manifest.json.
  Use original D88 requested_model_config PARENT, not resolved childconfig.
  Frozen content classes covered, actualbinding verified; do NOT reprofile.
- D90: functional100-request prefixes completed/cleaned:3B100/100,7B100/100;
  initial1/controlled1/natural2 and four releasedleases each, nativecounts and
  dispatchtiers complete, no requestpacking/resourcefault. Descriptive only:
  limiteddriver globallycaps8/2 despitefourruntimes. CanonicalD91main instead
  aggregate32/8. Do NOT repeatprefixes. Failures retainedinarchive andD90doc.
- D91: canonical actualCPUassembly both4000/500 and825tests PASS,pushed1148078.
  No actual canonicalFullGPUreplayyet. PreparedD91configs lackdeadline;
  cannotlaunchunchanged. Rawresults/ieee_tc/p2_backend_qualification/d91_20260928.
- Next: deadlinefromplannedarrival, mainfailure-evidence andownedHOST/NVMe
  cleanup; tests,backup;3Bfull4000 then cleanup/validate/table before7B.
  Existingrun_all_experiments_user_scope.sh/externalpublisher/physicalledger
  andnativeCUDA13 environmentonly. Oneheavyjob,nofallback/newframework.
- AfterPrimeFull,resumebaselines:Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe.
  Serverless3Boriginal preparedbutPAUSED. Do notlaunchitnow.
- WarmSLO/Residentreferences,M1/M2,A1–A5,S1–S13/formalcomparisons pending;
  442conditionalcoreslots NOTcompleted/uniquejobs. No numerical/SLO/rankingclaim.
- Existingzero-weight limitation:3B500/500,7B498/500zero;2/4distinctweightSHAs.
  Wrong/right controls nondiscriminative. Noauthoritynewweights;don'treask
  repeatedly. Otherimplementationcanprogress independently.
- Currentdevelopmentcontrols:3Bcap8/slots8/cpu32/gpu.72,7Bcap2/slots4/cpu24/gpu.70;
  HOST16GiB/native2/NVMe16,W5/movement3,scaleinterval2/beta.5,min1/max4;
  service_bin_ms staysactualD88measured modelvalue, NOT2ms.
  NOTfinalSLO/optimalworkingpoints.
- NativePATH includesvenv/bin,CUDA_HOME=/usr/local/cuda-13.0,FLASHINFER_NVCC
  itsbin/nvcc,MAX_JOBS2;allocatoruncached_background_v1,PYTORCH_ALLOC_CONF
  pinned_max_cached_size_mb:0,pinned_use_background_threads:True. OfflineHF/
  TRANSFORMERS,PYTHONNOUSERSITE. AuxiliaryCPUtaskset2,3,26,27.
- Latest147protected/planunchanged;runtime1148078pushed;noactiveGPU/tmux/remote,
  disk312GiB/MemAvailable108GiB/swap0. Recheckactualstatebeforelaunch.

## Assets and current service state

- Main /home/qhq/serverless_llm_experiment_retry14_baseline,
  retry14_continuous_queue_v2. Baseline /home/qhq/serverless_llm_baselines/main,
  HEAD9e2cf28903ed11bc8ee891dd4cc9636b94307573, PAUSED.
- Native /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM0.30/torch2.13/CUDA13; no reinstall/driver upgrade.
  CPU tests /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OS guards /usr/bin/python3 (required pidfd API).
- Real remote alias primelora-artifact-174, lab14@192.168.4.174:8122,
  strict BatchMode/host check. Fingerprint
  SHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
  Private key ~/.ssh/primelora_artifact_174_ed25519_20260925,0600,outsideGit.
  Local/remote token ~/.config/primelora-tc-d75/artifact.token,0600.
  NEVER print values/put credentials in journals or Git.
- All remote services/monitors stopped after D90; D91 was CPU-only.
  No local model or live tmux at latest checkpoint.
  Do not reuse old PIDs/invocations. Unit fragments/cache remain deployed.
- Cache root /home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b}.
  Source pools /home/lab14/primelora_remote_artifacts/
  llama32_3b_a500_v1_modelscope and llama2_7b_a500_v2_publicmix.
  HTTP18080/18081, no13B.
- Correct indices ONLY:
  paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json,
  SHA bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  20260927_7b_materialized_content_index.json,
  SHA e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  Canonical HTTP SHA3Bac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e;
  7B684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050.
  Misnamed7b_remote_content_index4998files is local diagnostic, NOT valid full7B.
- External reboot~15:06, not agent initiated: localboot
  aef67ed2-79d1-46bc-82e4-37401a5b45bf; remote
  b53c1f16-1dd2-4741-9271-6b75e86582ec. Current direct link1000Mb/s/full BOTH
  endpoints and remote peer advertises1000. D77 earlier100M historical only;
  physical reason not confirmed. No agent network change, no inspected qdisc cap.
- Main results symlinks to /home/qhq/serverless_llm_experiment/results.
  D80 snapshots/failed activation retained. No old results overwritten.

## Protection, reporting and backup

- Common inference72/80GiB/swap2GiB, auxiliary4GiB; one heavy run at a time.
  Actual workers contained BEFORE start, external watchdog, native release.
  No global kill/ray-stop/reset/remote reboot or unrelated cleanup.
- Protected147entries seal
  paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify with scripts.ieee_tc_preflight.verify_seal.
- Preserve/exclude user configs/generated/lora_manifest_1000.json, AAAI archive/
  dir, oldfig7/fig2/fig3,scripts/regenerate_motivation_figs.py; rejected D72/D73
  previews untracked. Baseline unrelated dirty files untouched.
- Main push faaslora_origin/retry14_continuous_queue_v2; baseline own origin/main.
  Before push diff/secrets/protected-name/checksum/test/smoke; fresh remote SHA.
  Raw large logs/credentials never Git. Explicit paths only, no force push.
- Each run cleanup→validation→table/figure→interpretation→next. Chinese reports
  at least each minute during work from paper-evidence view. Functional checks
  use status tables, not manufactured performance plots. Use academic-plotting
  for actual figures per plan IEEE single-column/TNR/no overlap rules.
- Relevant pre-optimization evidence: P1_FORMULA_IMPLEMENTATION,
  P2_BACKEND_QUALIFICATION, PHYSICAL_GPU_MEASUREMENT, RESOURCE_QUALIFICATION,
  EXTERNAL_REPLAY_QUALIFICATION, ARTIFACT_CONTENT_AUDIT,
  NATIVE_ADAPTER_NUMERIC_CONTROL,P0_FULL_PROVENANCE,REMOTE_ACCESS.
- Verbatim archives contain SUPERSEDED live/next-action text, not fresh directives:
  EXECUTION_HISTORY_D78_D80.md SHA090798cbdf7ab651797e8fa7df2245f7c0dc3c3fe911397b37a67183feb3f99e;
  D67_D77 SHA608d83d2440b27cd071d5e85a3d41d92d8b74c100952bf530ec58c5e7c8dab21;
  THROUGH_D26 SHA4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb;
  D27_D52 SHAbd56dec4a43d7e03f35e8a9b1b26ac598378df4ca75429e0d1344d237e0d1dfe;
  D53_D66 SHA47b9a76adad6b7dacb1977133389c56cebbb921531d0c1921a5f2ea5618ee35c.

## Verbatim recent history archive

Full pre-compaction ledger retained unchanged in EXECUTION_HISTORY_D81_D91.md.
Older LIVE/NEXT headings are historical, not restart instructions.
