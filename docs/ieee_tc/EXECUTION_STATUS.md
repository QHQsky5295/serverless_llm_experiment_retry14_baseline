# IEEE TC status — D508 necessary adaptations CPU-qualified; next native GPU qualification

LATEST CHECKPOINT2026-10-13: BASE5a8e65fffcf91beb646ca7843789303dc142f511
PUSHED/remote-equal. D508 is CPU-only progress, NOT a new performance result.
No GPU experiment/remote service running. Existing unrelated k3s untouched.
Read BASEdocs/ieee_tc/D508_HYDRASERVE_ADAPTATION_OWNERSHIP_CAPACITY.md.

Necessary changes now implemented:
- http_contract.consume_native_stream preserves original per-chunk AnyIO
  off-loop iteration under one shielded owner. Cancellation signals stop after
  pending next yields, JOINS actual owner then propagates; repeated cancellation
  cannot close a concurrently running generator. Native counts remain owned
  until real close. This is NOT instantaneous backend preemption; existing
  transport/request/init/cleanup guards unchanged. Actual read errors propagate.
- request_contract/controller_runtime render opt-in cancellation_ownership=True:
  remove only cancelled unassigned request; if assigned before chat starts,
  close exact unstarted coroutine/release once; after chat starts its existing
  exception owner handles release. Demand history/FIFO/native policies unchanged.
- native_delivery.render_overlays opt-in lora_admission_capacity=True:
  max_loras = source-pinned native Model default batch(8) * current PP_SIZE
  on ALL3EngineArgs sites. max_cpu_loras retains native None->max_loras.
  No catalog-sized500cache, rank/max_num_seqs/LRU/policy changes or parameter sweep.
  Destination reads nativeupdatedPP_SIZE; existing reusedworker slots may retain
  larger initial capacity. New run MUST assert actual Model.batch_size==derived8.

100/100 fullCPUtests and36/36 actualnativePython3.10 targetedtests PASS.
Tests use actual worker-thread blocked next/repeatedcancel/error/counterrelease,
plus original rendered queued/assigned-before-start/started cancellation paths;
AST checks isolate lifecycle changes and max_loras-only EngineArgs addition.
FirstCPUtest stopped on ambiguous source splice (5errors); failedsource/log kept;
precise allocate_instance anchor fixed then33testsPASS. No GPUtest failure here.
Pinned official worker/executor sources re-fetched byte-identical: LoRA buffers
allocated duringload_model BEFORE actual free-memory/nativeKV calculation.
Still need actual mixed-adapter GPU/memory qualification; codeorder is not proof.

D508RAW originalresults/ieee_tc/hydraserve_qualification/d508_20261013.
Qualifiedsource/overlaySHA snapshots saved;65-file roundtripchecked archive
BASEpaper_results/ieee_tc/hydraserve/d508/adaptation_evidence.tar.gz175118B.
summary.json includes priorD507diagnosis andqualification. Protected147/userdirty/
vendorclean/fresh100tests/diff/secrets/explicitstage/push PASS. CPU1–5actually
emptyclosed; CPU4peak89985024B, allhigh/max/OOM/OOMkill/swap0. BackupCPU6closure
andMAINstatusbackup follow; no liveGPUjob. D507 archives/failures unchanged.

NEXT MAIN-LINE: finish checkpoint closure/MAINbackup, then NEW native mixed-LoRA
concurrency/memory qualification using existing qualified commonHTTP/fullnative
scheduler and original inputs (e.g. affected100prefix, NOT basic import/setup
ladder). Enable BOTH opt-ins above; verify actual batch and native distinctadapter
overlap, token/content/clock binding and full physicalrelease. Exact-owned five
staleD507container records remain in privatePodmanstore; logs/State already saved
and backed up BEFORE any exact-ID cleanup. No broadprune/foreignk3s access.
Then ONE new affected full4000HydraServe ->allavailablebaselinecomparison using
latestPrimeLoRAD476 ->PrimeLoRAoptimization. No timeout/SLO relaxation, oldresult
overwrite, repeatwinnerfilter or immediate PrimeLoRA candidate. Full remaining
7B/3B/W1/W2/G2/A1–A5/S1–S13/cost/docs/finalfigures below remains ACTIVE.

## D507 preceding terminal checkpoint (sealed, not current work)

LATEST CHECKPOINT: D507 failure-aware outcome SEALED and BASE pushed/remote-equal
988f7662b6282c925dee08a7c72ee2ce20a532b2. Fresh90CPUtests/protected147/userdirt/
vendorclean/secrets/diff passed.613-file full local archive31,630,351B retained
outside staged Git;602-file compact curated archive1,523,017B SHA
11a8bcc1b9208821957ec87cac4683f708ea091968a04866e752411f0ecfdd2c backed up.
Five remaining native containers' State/stdout/stderr preserved, no containers
deleted; original idle path had already removed w1 with recordedrelease.
CPU7/8/10/11/12/13/14/15/16 actually empty/closed; no GPU inference live.

D508 NEW READ-ONLY diagnosis PASS, NOT a GPU run: pinned official arg_utils/
config/core scheduler/controller re-fetched and byte-identical to nativevendor.
All three executed EngineArgs sites enabled LoRA but did not override max_loras
or max_cpu_loras. Actual defaults max_loras=1,max_cpu_loras=None->1; native
controller admission batch8. Original backend defers distinct adapters when
max_loras is occupied. In3632responses BEFORE first timeout, every worker had
peak distinct-adapter decode overlap1, cross-adapter overlapping pairs0; same
adapter peakrequest overlap5/5/6/5. This directly supports a capacity-mismatch
hypothesis; no claimed intervention speedup yet. Raw d508_20261013/
adaptation_diagnosis.json records sourceSHA/online URLs and exact observations.

NEXT: finish MAIN backup, then necessary HydraServe adaptations: single-owner
iterator cancellation plus LoRA batch capacity derived from original per-instance
admission (with actual memory qualification), NOT arbitrary capacity sweep or
larger timeouts. Preserve native policies/pipeline/consolidation/old failures.
Investigate complete-stage effect and validate before a NEW affected full4000
execution key. No more elementary import/tokenizer tests. After faithful actual
HydraServe result, full available baseline comparison, then PrimeLoRA optimization.
Mainline/fullmatrix remains ACTIVE; do not accept abnormal baseline performance
without adaptation review under latest user clarification.

Updated2026-10-13. Current checkpoint, NOT replacement plan. Entire preceding
status byte-identically archived EXECUTION_HISTORY_THROUGH_D491.md; cmp passed,
SHA017a48a3675c0aa9e2f1baff09552539a0ebe64123816a2ad883f25a7f5063d1.
Older D477/D461/D448/D433/D402 histories remain. Their NEXT/RUNNING notes are
historical, not current instructions. No approved task/outcome/failure removed.

## D507 current outcome (2026-10-13; terminal, nonfeasible; audit pending)

The SAME full4000 run ended: N_plan=N_arrived=N_terminal=4000,
N_response=3876, N_failed=124. These are response counts, not yet independently
audited V1 contract counts; raw n_correct remains unknown. Do NOT restart this
baseline or extend request timeouts because of the failed outcome. Original
success-only audit.py remains unexecuted. A new failure-aware audit is next.

Post-terminal service finalization exceeded the existing 60s guard; launch
classification protocol_or_launcher_error, service/watchdog -15, replay 1,
hard_kill=true/released=false; tmux exit91 at original close_audit assertion.
This POST-terminal closure failure does not erase the preceding 124 actual
request timeouts. Original launch/close logs and preregistration remain unchanged.
There is no original terminal GPU census: full physical closure cannot be assumed.

NEW recover_closure.py verified original service Invocation/inode and recursively
empty descendants, plus empty native GPU census; stopped ONLY the two empty owned
service/launchaux scopes. Actual recovery time recorded, not backdated release.
Peak54397161472B, all high/max/OOM/OOMkill/swap0 over5773 samples. Service
Invocationd855457eebbb44fe93ca7d792ca987ad,inode24207809 gone; aux9 Invocation
0b984cf4de2844b7b93388ceac2387d4,inode24207609 gone. RecoveryCPU7/remoteCPU8
also actually empty/closed. Exact-owned remote3B/7B/monitor stopped AFTER local
release, both ports absent; transfers/monitor SHA matched remote originals.
No k3s or unrelated service accessed/changed. No GPU experiment is live.

NEXT failure-aware native/request/source/timing/remote audit, observed/right-censored
physical accounting, seal/back up outcome; then full available baseline comparison
using latest PrimeLoRA D476, followed by PrimeLoRA optimization. Full approved
matrix remains ACTIVE; failure does not authorize baseline performance search.

LATEST user goal supplement: unreasonably poor baselines require evidence-based
adaptation review, not automatic acceptance under the no-performance-search rule.
Read updated BASELINE_EXECUTION_RULES_20261011.md. D507 failure-aware audit now
PASS: 3876 native contract responses,124 timeouts,6/4000 joint SLO. 105 server
generator-close overlap errors occur AFTER client timeouts (19 other cancellations).
3632 responses preceded first timeout. Poor pre-timeout queuing is observable;
later responses/retirement may be affected by OUR cancellation adapter. Do not
declare faithful full-result closure or attribute all slowness to original policy.
Necessary cancellation ownership review/repair remains before a clean full baseline
result and PrimeLoRA optimization; no timeout/SLO relaxation or winner filtering.
Original physical ledger has4acquires/1release: exact U is unknown; audited lower
22932.369175774GPU-s, conservative recovery upper24496.389781413GPU-s, NOT a
precise ranking value.132remote transfers131480060wireB, all bound including5
first-touch transfers belonging to failed requests. First offline audit correctly
stopped on these missing failure-owned transfers; v1 source/error preserved,
failure-aware v2 included their retained artifact receipts and passed.

Native log capture first stopped on stale Podman Running state after conmon kill;
actual originalPID/birth gone, recursivegroups gone. Keep stale State as evidence,
do not rewrite it. Read-only capture v2 uses actual ownership/absence, not labels.
No containers removed. Final curation/backup and cancellation-path review next.

### D507 preceding launch/wait history (no longer running)

LATEST OUTCOME WARNING: late http_request_failed events now observed in the SAME
run. Bounded tail of launch_gate.launch/replay.jsonl confirms TimeoutError at
~1800s from planned arrival, e.g. req_03614 arrival1404068.3725305658,
completion1405868.379617821; req_03620 similarly. Initial submission lag was
milliseconds; native allocation logs show queue waits around1709–1724s.
These are actual request-deadline failures, NOT the removed container-lifetime
timer. Watchdog5233 stillhigh/max/OOM/OOMkill/swap0,warningfalse/abortnone.
Full denominator4000 staysfixed. Later terminal_count mixes responses AND
failures; do not label it allsuccessfulresponses. Exactpopulation/source/timing/
resource attribution requires postclosure audit; do NOT infer rootcause or
change runtime/timeout/configuration whilelive. Existing audit.py requires4000
native-contract responses and is now NOT the appropriate success audit; leave
it unexecuted and preserve it. After actualtermination/ownedclosure, derive a
NEW failure-aware observation with all planned/arrived/response/failed counts,
native/source binding for actualresponses and physical union, not a feasible
G1 point. Preserve failures and proceed to full availablebaseline comparison;
no baseline performance tuning or rerun merely because it misses the goal.

ONE tmux primelora-d507 panePID3014164 isalive; run_and_close.sh launched once.
Service primelora-tc-svc-e55e30421ffe4d4d868962062271f68d.scope,
Invocationd855457eebbb44fe93ca7d792ca987ad,inode24207809; launchaux9.
Actualexternalreplay matchespreregistered4000W0view. Deploymentnotice
1400584.845739947; replayt0=1400644.845739947, samequalifiedmonotonicclock.
Nativeprivate storage initialized, CPUprovider startupundercommon80GiB underway.
Latestinitialwatchdogsample8:events0/swap0,warningfalse/abortnone; no fullclaim.

Beforelaunch BASE3eec32869c657e8020581da4354ee9ad8b91a95d and
MAINec36afd489a3edcaa3357e398ec8ad6102a15f77 pushed/remote-equal;
90freshCPUtests/protected147/userdirty/archiveSHA/secrets/diff passed.
CPU1–6 actualemptyclosed. Authenticatedremotehealth AND source/protected/resource
gates explicitlycompleted/readPASS BEFORElaunch, freshactual1000Mb/s/full.
Remote3BInvocatione37895d5a09a4b6d9d4790afb586f1b1;
7B362722f2e67b48b794ecd2cf8b8271df;monitord9aeae341231412d8259a9953ad9e70f.
D78/D80readonlypublishedobjectscoldper-run,reusedwithoutrepublishing.

DO NOT rerunprepare/launch/audit oroverwrite. Monitor actualtmux/process and
watchdog/service logs. No remoteadmin/build/hashscan/compression/secondGPUtask
whileactive. run_and_close.sh closeslocally thenexact-ownedremote/collectslogs.
Onlyafteractualclosure use strictaudit.py ifall4000complete; otherwiseclassify/
retainfailure. Auditnotyetexecuted. Thenfullavailablebaselinecomparison ->
PrimeLoRAoptimization. Fullremainingmatrix atfileend remainsACTIVE.

Latestactualwatchdogsample136:11HTTPresponsesobserved; nativeGPUprocesses
3018560/3021262/3022430 onphysicalGPUs0/1/2, originalfourthallocation stillonline
controllerdecision. tmux3014164alive;service42.3983GiB,hostavailable62.5168GiB,
disk170.5484GiB,high/max/OOM/OOMkill/swap0,warningfalse/abortnone.
No prefixperformanceconclusion, no runtime/remotechanges. This runningnote is
localpendingpostrunbackup; precedingtestedcheckpointalreadyremotelybackedup.

Verified waiting thisturn: same actualtmux3014164 stayedalive; observedresponses
74->129->180->242->327->395. Latestwatchdogsample756:service45.4206GiB,
hostavailable60.1092GiB,disk169.6068GiB,high/max/OOM/OOMkill/swap0,
warningfalse/abortnone. Originalqueuewait retained; no service/runtime/remote
change, hashscan, repeat, performanceanalysis or extraheavyjob. Awaitthissame
full4000run andactualclosure; strictaudit.py stillNOTexecuted.

Read-only postrun-entry inspection (NOT a newcomparison): reuse MAINscripts/
ieee_tc/check_d299_full_context.py matched_controller_rows/matched_native_entry/
matched_request_contract/matched_relative_observation and existingD476
RAWp2_backend_qualification/d476_20261012/compare_curate.py source-bound join.
MAINpaper_results/ieee_tc/main_comparison/d454_7b_w0/comparison.json contains
OLDERPrimeLoRA row; use latestsealedD476 matched_comparison.json.candidate,
keepallotherbaseline rows/failures and historicalPrimeLoRA references explicit.
D476 provenance key is refs (not source_refs); matchedcontractSHA
384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399,
458224nativeoutputs/132actualIDs. AfterHydraServeclosure: completeitsstrictaudit,
source/content/prompt/target/timing/resource cross-join thenallavailablecomparison.
Do not assume sharednominalW0label alone provesmatchedcontract. No comparison
or PrimeLoRA optimization has been executed early; wholeapprovedscopeACTIVE.

Further verified wait: the SAME tmux panePID3014164 remained alive while observed
HTTP responses advanced431->653->730->798->841. Latestwatchdogsample1417:
service46.3260GiB,hostavailable59.5219GiB,disk169.1500GiB; high/max/OOM/
OOMkill/swap0,warningfalse/abortnone. The fourth nativeGPU workerPID3024474 is
recorded in the service resource domain; no foreign compute process observed.
Native instance queue waits remain in the measured timeline. No restart, runtime
or remote changes, build, scan, compression, extra heavy job or result filtering.
Full PLAN/status and frozen V1/single-run V2 were reread; prepared audit.py was
inspected read-only, NOT executed. Continue this original full4000 run, then check
actual local/remote closure before audit/curation/backup and allavailablebaseline
comparison. A live prefix is not a completed result or numerical correctness proof.

Next goal turn classified VERIFIED WAIT: same actual panePID3014164/process alive
through887->926->1021->1093->1138->1177->1219->1296 observedHTTPresponses.
Watchdogsample2109:service47.1220GiB,hostavailable59.0971GiB,disk168.7353GiB,
high/max/OOM/OOMkill/swap0,warningfalse/abortnone. At sample1942 monotonic
1402552.737769622 all four original nativeGPU PIDs3018560/3021262/3022430/
3024474 were still present with their original birth identities. Later ps at
sample2028 showed elapsed33:15/32:25/32:10/31:41 respectively: actual workers
survived beyond the erroneous D506 1800s container lifetime boundary. This is
lifecycle-correction evidence, NOT full-run or SLO/numerical qualification.
No changedruntime/remoteconfiguration, restart, additionalexperiment, audit or
compression. Continue SAME full4000; postclosure audit/curation/backups and all
availablebaselinecomparison remain next, followed by PrimeLoRA optimization.

Subsequent verified waiting: same live tmux/process3014164, responses1340->1395
->1457->1503->1563. Latestwatchdog2488:service47.3699GiB,hostavailable58.8991GiB,
disk168.7848GiB,allhigh/max/OOM/OOMkill/swap0,warningfalse/abortnone. No restart,
runtime/remotechanges, secondheavyjob, prematureaudit or ranking. Original full
4000 remains live; fullclosure/audit/curation/comparison still await this run.

Further VERIFIED WAIT: actualtmux/process3014164 stayedalive; response observations
1624->1684->1760->1810->1868->1913->1971->2023. Latestwatchdog3025:
service47.9664GiB,hostavailable58.5087GiB,disk168.5479GiB; high/max/OOM/OOMkill/
swap0,warningfalse/abortnone. Same originalrun beyond half of plannedresponses,
not a second experiment and NOT a completed/audited result. No runtime/remote/
sourcechanges, heavyparallelwork, prematureanalysis or configurationselection.
Next remains SAMErun completion + actualclosure -> strictaudit/curation/backup
-> allavailablebaselinecomparison -> PrimeLoRAoptimization; fullscopeACTIVE.

Latest VERIFIED WAIT: originaltmux3014164 actualalive beyondonehour; responses
2081->2150->2238->2305->2383->2437->2493. Watchdog3528:service48.5495GiB,
hostavailable58.1380GiB,disk168.1780GiB,high/max/OOM/OOMkill/swap0,
warningfalse/abortnone. No newexperiment, runtime/remotechanges, earlyaudit or
performancecomparison. Keepawaiting SAMEfull4000 andactualclosure, then follow
the previouslyprepared strictaudit/curation/backup/fullcomparison sequence.

Continuation VERIFIED WAIT: sameactualtmux/process3014164 alive; observedresponses
2526->2598->2653->2714->2759->2812->2867->2921->2974->3030. Watchdog4210:
service49.2424GiB,hostavailable57.5918GiB,disk167.8147GiB,high/max/OOM/OOMkill/
swap0,warningfalse/abortnone. No restart, source/runtime/remotechanges, extra
heavyjob or prematureanalysis. Original full4000 stillRUNNING; morethanthree
quarters of responses observed, not yet complete/audited. Await SAMErun and
actualrelease, then existing strictaudit/curation/backups/fullbaselinecomparison.

Further VERIFIED WAIT: actualsame tmux/process3014164 remainedalive; responses
3065->3111->3161->3221->3267->3303->3372->3409->3450->3489. Latestwatchdog4894:
service49.9644GiB,hostavailable57.2171GiB,disk167.5438GiB,high/max/OOM/OOMkill/
swap0,warningfalse/abortnone. No config/source/remotechange, restart, parallel
heavyjob, prematureaudit or performanceclaim. Full4000 stillRUNNING; awaitits
actualterminal+closure. Existing requestguards remainfrozen, failuresretained.

### D507 preparation history (completed before launch above)

D506 failure sealed BASEf1591bff7ab9d828b5697e3308aece6ba676ba7d and
MAINbcc2d8ab4091f56de6db5e12c6685384367449d3, bothpushed/remote-equal.
AllsixstoppedD506container native stdout/stderr/State/ownership captured BEFORE
exact-ID removal; no uniqueasset or oldevidence deletion. Store-specificfresh
Podmanpause exactPID/birth/UIDverified andclosed; unrelatedk3s untouched.

NecessaryD507adaptation removesContainerRecipe.timeout_s and Podman --timeout;
NOT replacedwithlargerTTL. Nativecontroller ownsretirement; existingbounded
campaign/watchdog ownsabnormalcleanup. Requestguard1800/nativeinit/transport/
CLI60/nativeScheduler/modeltuple9000/200/cache/batch8/idle30/SLO unchanged.
TwoCPUregressionsadded; all11isolated-platformtestsPASS. Full90tests runatbackup.

Full4000D507preparedONCE, sameW0view/trace/pool/generation/realremote/60snotice/
resourceenvelope/physicalledger asinvalidD506; no newelementaryGPUqualification.
Raworiginalresults/ieee_tc/hydraserve_qualification/d507_20261013.
PreregSHAab9c8692f31d824dd8ee89be6fae0d4e7db6241cea6d15e086653ec7f85f637e.
66files roundtripSHA lifecycle_correction_evidence.tar.gz4,553,660B,
SHAef1e52716f787de4e29fc744318c6bf6672bae6eda63196c9006e1425d51db29,
underBASEpaper_results/ieee_tc/hydraserve/d507/; includesnativeD506logs and
newexecution sources. D506originalarchiveunchanged. Runtimeonlychangedsource
isolated_platform.py; oldsnapshotsverified, no nativepolicysearch/performanceclaim.
CPU1/2actualemptyclosed,events0/swap0; prepareCPU3completed,closure next.

NEXT protected/tests/secrets/backups, explicitlyawait/read remote_start then
authenticatedhealth/resource/source/protected gates; closeCPUscopes then
ONE tmux full4000D507. Do NOT rerunprepare/overwrite. No remoteoperations/builds/
hashscans/compression duringinference. Actualclosure thenstrictfullaudit (onlyif
complete), saveoutcome ->allavailablebaselinecomparison ->PrimeLoRAoptimization.
Full7B/3B/W1/W2/G2/A1–A5/S1–S13/cost/docs/finalfigures remainACTIVE.

D507 BASE3eec32869c657e8020581da4354ee9ad8b91a95d PUSHED/remote-equal.
Fresh90CPUtests/protected147/userdirty/vendorclean/secrets/diff/archiveSHA allPASS;
CPU3preparepeak140460032B/events0/swap0 actualclosed. BASEbackup4closure next,
then MAINstatusbackup andfreshremote/starthealth gates; stillNO GPUlaunch.

## Authority and main line

Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md before work,
after compaction and before experiments; SHA
f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
Also BASELINE_COMPLETION_ORDER_20261012.md, BASELINE_EXECUTION_RULES_20261011.md,
STORAGE_PRIORITY_20261013.md. D483 maintenance COMPLETE47,340,646,400 allocatedB
reclaimed; no repeated broad storage sweep by default. D486 metric recheck DONE.

LATEST ORDER: faithful HydraServe actualresult -> all availablebaseline comparison
-> PrimeLoRA optimization. No new PrimeLoRA candidate now. Only PrimeLoRA
performance optimization; baseline compatibility/correctness/measurement/safety
only, no performance tuning/configsearch. Existing k3s belongs to another project;
do not access/change cluster/credentials/permissions/resources. Isolated rootless
Podman adaptation authorized; no barevLLM substitution, fakeReady or fake nodes.

## D506 actual outcome (2026-10-13; STOPPED, incomplete/invalid)

tmux primelora-d506 dead exit2; no GPU job. Nativecontainers each terminated at
~1800seconds because OUR isolated_platform.py passed --timeout=1800 to Podman,
which limitswholecontainerlifetime, NOT startup/request. Installedhelp andofficial
Podman3.4.4 docs verified. Sixexact-owned State records savedin
RAW/container_lifetime_failure.json;allOOMKilledfalse. NotnativeHydraServe
performance/system-resourcefailure; classificationprotocol_or_launcher_error.

Publisherterminal: N_plan4000,N_arrived=N_terminal2229,N_response1202,N_failed1027
(includescancelled). Remaining1771notarrived, notfakedtimeouts. No fullSLO/GPU-s
ranking, no numericalcorrectnesspromotion, no full_auditexecution. Rawfaileddata
andallpreregistration/sourceSHA retained. D506 mustNOTberestarted/overwritten.
ExactlauncherpidfdSIGINT triggeredexistingownedcleanup; servicegone/nativeGPU
contextsreleased;launchaux9Invocation585b136f0cdd4295a167ef05aa7ba707 emptyclosed.
Peak50427858944B;high/max/OOM/OOMkill/swap0. Remote3B/7B/monitor exact-owned
stoppedAFTERlocalrelease;portsabsent/logSHAmatched. CPU1–6closed. Failedhelper
ISOparser/pidfd attempts preserved; hostpidfdhelper deliveredtheactualsignal.

NEXT seal/backup failure evidence BEFOREsourceedits; capturestoppedcontainer
logs/state thenonlyexact-owned cleanup. Removeinappropriateper-containerlifetime
timer, relyingonnativecontroller+existingboundedcampaign/watchdogcleanup. DoNOT
arbitrarilyincreaseTTL, change1800requestguard/SLO/nativepolicy, ortunebaseline.
Validateonlythislifecyclecontract, thennewfull4000attempt; notmoreelementary
qualification. Thenfullavailablebaselinecomparison ->PrimeLoRAoptimization.
ReadBASEdocs/ieee_tc/D506_HYDRASERVE_FULL_W0.md. Failurecuration/push stillPENDING.
All7B/3B/W1/W2/G2/A1–A5/S1–S13/cost/docs/finalfigures remainOPEN;goalACTIVE.

Failurecuration nowCOMPLETE:439explicitfiles,5,143,756B roundtripSHAarchive
BASEpaper_results/ieee_tc/hydraserve/d506/failed_lifetime_evidence.tar.gz plus
summary.json. Protected147/sourcechecks passed; populationaudit preserves
2229arrived/1202responses/1027failed-or-cancelled/1771notarrived. No fullperfrow.
CPU7curation actualemptyclosed; sourceunchanged so far. Failurebackup follows;
postrunstrictaudit.py remains unexecuted and is not applicable to thispartialrun.
DoNOT reruncuration/overwrite. Stoppedcontainer cleanup and lifetimefix stillPENDING.

Failure BASEf1591bff7ab9d828b5697e3308aece6ba676ba7d nowPUSHED/remote-equal,
fresh88CPUtests/protected147/userdirty/source/vendorclean/secrets/diff allPASS.
ArchiveSHA8b4230015f526977dc02d288ea50c4ce5763c1293951fa7d58a18ce8c4dad99f,
summarySHA07e45a196e4a0e81d480a0a307163a74be265ad6118edbb090f9389077413107.
CPU7curationpeak84885504B/events0/swap0 actualclosed; CPU8BASEbackupclosed.
MAINstatusbackup next withseparateCPU10, no runningheavyjob. SourcehasNOTyetbeen
fixed; nextnecessarylifetimecorrection isnotbaselineperformancetuning. Preserve
container logs/State BEFOREexact-owned removal; do notoverwriteD506 orrunits
strictsuccessaudit. Freshdisk183285915648B(~170.7GiB),MemAvailable112647868416B;
newlaunch requiresfreshgates. Entiregoal/fullmatrix stillACTIVE.

### D506 actual launch history (no longer running)

tmux primelora-d506 panePID2871594 actualalive; run_and_close.sh launched ONCE.
Service primelora-tc-svc-1f9b667fcb1a436cbb8671c628d6c18f.scope,
Invocationf0b714394f5c4c2ab4e975f96174040c,inode24179351. Launchaux9.
exec_receipt external4000 W0 view matchespreregistration; deploymentnotice
1397084.676304775,replayt0=1397144.676304775 samequalifiedmonotonicclock.
Watchdog actualsampling/events0/swap0 atstart, originalprovider/storage startup
undercommonservice80GiB. No completedperformance claim. DoNOTrepeatlaunch or
prepare. Monitor actualtmux/process/logs, notold state labels; letclosure finish.

Beforelaunch: remote_start command completed/readsuccess, thenauthenticated
health for3B/7B AND resource/source/protected147 gates completed/readPASS;
CPU5exactemptyclosed Invocation760f9db358654f3bbbee40ae43cbe220,inode24178872.
FreshMemAvailable112281948160B,disk~185.8e9B;20GiBgrowth/common150/100GiB
gatepassed. Freshremoteactual1000Mb/s/full. Remoteowned3BInvocation
ff6b5cb44a1246129935e78de7bc7951;7Be83b90cd88c046c38832752343683727;
monitorf6a51e684b114b609901f2a09e0cbf6d. D78/D80readonlyobjectsreused.
NO remotechanges, builds, hashscans, compression or anotherGPUtask whileactive.
run_and_close.sh closesactuallocalresources thenexact-ownedremote andcollects
receipts. Onlyafterterminal/closure inspectaudits; preservefailures/incomplete.

Preregistration backup complete beforelaunch: BASE329a021df779581381380e2e8afea22194778f2a,
MAINea372931ac3eb9962084ea055ea0094dbc67054f bothpushed/remote-equal.
CPU1–5allactualemptyclosed (firstbackupCPU2failure retained), fresh88testsPASS.
Current runningstatus is local until nextclosed checkpoint; no backup duringrun.
Latest actualwatchdogsample111: nativeprovider/storage initialized, first6HTTP
responses observed, nativeGPUprocess2876097 onphysicalGPU0; tmuxstillalive.
Servicecurrent43243175936B,peak43381321728B,hostavailable69054889984B,
diskfree185507975168B, high/max/OOM/swap0, warningfalse/abortnone.
This isprogress only, NOT full-request contract audit or performance result.
Lateractualsample395:136HTTPresponsesobserved;service44.8455GiB,hostavailable
60.5747GiB,disk171.6309GiB,events/swap0,warningfalse/abortnone;sametmuxalive.
No performance conclusion fromprefix. Prepared RAWaudit.py reusingD503audit,
with4000exactpopulation, duplicate/index/path checks, independentphysicalunion/
nativeworker lifetime binding andcommonD394 SLO. SyntaxcheckedONLY; NOTexecuted.
ExecuteunderboundedCPU ONLYafterlocalclosure andremote_stop receipts pass. It
preservesrawn_correct=null andwritesnewV1binding/accountingobservation; anyfailure
isretained inaudit_failure.json, never apartialqualifiedranking or changeddenominator.
Full-source/request/resource audits/curation/backup remainPENDING, not assumedpass.
Latestverifiedwaitsample532:236HTTPresponsesobserved,tmux2871594stillalive;
service45.0571GiB,hostavailable60.4832GiB,disk171.5410GiB,events/swap0,
warningfalse/abortnone. Sameoriginalrun; no restart/newexperiment/remotechanges.
Subsequent verifiedwaiting: same tmux/process stayedalive while responses advanced
272->347->412->458->517. Latestwatchdogsample961:service45.5924GiB,
hostavailable60.5728GiB,disk171.3153GiB,high/max/OOM/swap0,warningfalse/abortnone.
Originalqueuewait increased; no latency subtraction, runtimeadjustment or early
performance conclusion. Remainonthissame4000run; postreleaseaudit stillNOTexecuted.
Further verifiedwaiting:539->617->725->811->874->924->1018responses;
same originaltmux2871594 alive. Latestwatchdogsample1568:service46.6368GiB,
hostavailable59.9837GiB,disk170.9241GiB,events/swap0,warningfalse/abortnone.
This is the ongoing full4000 run, NOT a new1000qualification or completedrun.
No runtime/remote/config changes, extraheavyjob, repeats, analysisexecution or
earlyresource-return claim. Awaitremainingrequests and actualclosure, thenaudit.
Afterresult: fullavailablebaseline comparison ->PrimeLoRAoptimization.
3B/W1/W2/G2/A1–A5/S1–S13/cost/docs/figures remainOPEN, fullscopeACTIVE.

### D506 preregistration history (prepared before the actual launch above)

Raw originalresults/ieee_tc/hydraserve_qualification/d506_20261013 prepared ONCE.
Unchanged D505 native runtime/controller/backend/modeltuple/cache/idle policy;
only request population1000->4000 and campaign identity changed. Full existing
W0 view SHA a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d,
same source trace/pool/fixed-output/open-loop/60s notice/realremote contract.
Preregistration SHA033911e0ae567ae8c3dd02dc0bf0be65739465ae0acd10113fa3899c06654b24.
Read BASEdocs/ieee_tc/D506_HYDRASERVE_FULL_W0.md. Native9000/200ms control
inputs remain separate from commonD394 EVALUATION thresholds; no baseline tuning.
Keep formal_run=false/qualification identity/raw n_correct=null. Derive full
V1 binding/accounting observation only after actual complete evidence audit;
do not require new cross-backend exact token equality or invent numeric proof.

Preparation CPU1 actual empty/closed: Invocationedc250ddd76a40f6b7075ec531585b96,
inode24176808, peak135823360B/events0/swap0. No D506 GPU or remote service
has started at this checkpoint. Next back up preregistration, await/read remote
start and authenticated health/resource/source/protected gates, close CPU scopes,
then launch one tmux full4000 with unchanged20GiB growth and common safety gates.
Do not repeatprepare/overwrite files. Run closure automatically, then collect
remote evidence only afterlocalrelease. Pipeline/migration/consolidation scope
remains limited to actual events; one physicalhost, unrelatedk3s untouched.
After fullresult: allavailablebaseline comparison ->PrimeLoRA optimization.
Entire remaining matrix at end of thisfile remains ACTIVE.

D506 preregistration BASE329a021df779581381380e2e8afea22194778f2a pushed,
independently remote-equal; fresh88CPUtests/protected147/source/userdirty/vendor/
secrets/diff gatesPASS. Firstbackup CPU2 usedhostPythonmissing aiohttp/numpy;
3importerrors preserved, nothingstaged/GPUstarted. QualifiedexistingCPUenvironment
passedall88; no install/runtimechange/testskip. CPU2/3 actualemptyclosed.
MAINstatusbackup follows beforefreshremotehealth and guardedlaunch.

## D505 actual qualification COMPLETE (2026-10-13; not full W0 performance)

BASEcd3baf4251e840d983177d4c6e8f7e8f103f78a4 pushed/remote-equal. No liveGPUjob.
ReadBASEdocs/ieee_tc/D505_HYDRASERVE_1000_COMPATIBILITY.md. Original1000-prefix
completed1000/1000/1000/1000 offered/arrived/terminal/native-contract responses,
0failed;60requested logicaladapters,60request-driven realremote transfers,
74,953,014wireB,allreceipt/content/transferIDs matched, no requestpack/temp.
Input/adapter/nativeindex/nativecount/clock/process/timing auditPASS. Twoexisting
nonzero outputs matchD502, NOT independentnumericalproof; raw n_correctnull.
RecallD393/V1: no new cross-backend exact-token-equality substitute for binding/
accounting qualification; stillno HTTP200/name/count-only correctnesspromotion.

Mean/P95TTFT302606.357839/523997.128948ms; TPOT62.883717/79.802997ms;
E2E310140.194658/527501.026815ms. Mean submitlag4.030806ms,instancewait
260795.201367ms,assignment-to-first41807.125667ms,decode7491.086070ms;
artifactresolve225.439184ms isNESTED,notadditive. PrefixNOTfullW0ranking.
Physicalactualcontainer-reservationunion6554.059487593GPU-s,6.554059488/offered;
windows pre0/arrival4106.776171554/drain2270.325859637/cleanup176.957456402.
Allnativecontextsqualified/leasesclosed. D503pp1boundarycaveatunchanged.
Same originalScheduler/modeltuple/batch8/USE_CACHE0/SLOW_EXPR0; fourpp1instances
onONEphysicalhost,original30sidleretirement, no policytuning orforcedpipeline.

tmuxprimelora-d505 actualdead exit0. ServiceInvocation9af869eb405a4d339f67d6cc4497562e,
inode24109586 actuallygone; launchaux9Invocationcd65124d387f4e138a2a61461b1f0852,
inode24109386 recursivelyemptyclosed. Peak49,888,940,032B; high/max/OOM/swap0.
AllGPU/provider/storage/workers/pause released; exact-ownedremote3B/7B/monitor
stopped AFTERlocalrelease,portsabsent,logSHAverified. Prelaunchhealthpassed.
No remoteadmin/scans/compressionduringinference. CPU1–4/6emptyclosed; curation6
peak59,330,560B/events0/swap0. CPU5unused;backup7/8closureseparate. Free
185,855,791,104B(~173GiB); fresh150/100GiB+growth gates requirednextlaunch.

518explicitfiles roundtripSHA;protected147/userdirty/vendorclean/secrets/diff/
fresh88CPUtests/pushPASS. BASEpaper_results/ieee_tc/hydraserve/d505/
compatibility_evidence.tar.gz7,780,211B SHA
d7e1b5092d8caa558888490520c87f5264ab8f45e72d4e3a259b7832085839fd;
summarySHAef538402ebe7cca3ab26767fe40476b4f07068f79386f19dc927e60adadce08f.
Raworiginalresults/ieee_tc/hydraserve_qualification/d505_20261013 preserved.
Neverrerunprepare/launch/audit/curation oroverwrite. D505audit/checkpoint alreadyDONE.

NEXT onefull4000HydraServe usingunchangedqualifiednativeconfig/no performancesearch,
explicitlyrecordnativeruntime9000/200ms versuscommonD394 EVALUATIONthresholds;
do not use9000/200asevaluationorcallthisdefaultpointoptimal. Keepdevelopment/
qualificationidentity,sourceW0view/fixednativegeneration/realremote/physicalunion.
Thenfullavailablebaselinecomparison ->PrimeLoRAoptimization. Don'trepeatelementary
qualification. Numerical/fullpipeline/KVmigration/consolidation remainlimited/
OPEN,notfabricatedorforced. Full7B/3B/W1/W2/G2/A1–A5/S1–S13/docs scopeACTIVE.

### D505 preparation and execution history (preserved, no longer running)

Raworiginalresults/ieee_tc/hydraserve_qualification/d505_20261013. Mechanically
reuseD503/D504nativepath, unchanged modeltuple/backend/scheduler/idle/cachepolicy.
Original trace_prefix_v1 first1000 requests, NOT500-IDcoverageview. No newweights/
pool/trace. Existing500IDmetadata/CPU88tests reused; source/currentSHA matched.
FirstCPUprepare failed BEFOREgeneratedinputs/services/GPU becauseharness wrongly
expectedrequest_viewabsent; D503explicitlyhadtrace_prefix_v1. Exactvalueassertion
corrected inprepare2.py, originalfailure/source preserved; secondpreparePASS.
CPU1/2actualemptyclosed. RemoteD78/D80readonlyservicesstartPASS,1000Mb/s/full.
StandaloneauthenticatedhealthANDprelaunchresource/protected147/sourcechecks
explicitlycompleted/readPASS beforelaunch. CPU3actualemptyclosed. Launchalso
requiresbothprelaunchreceipts; no repeatingD504asynchronousprerequisiteerror.

tmuxprimelora-d505 panePID2546714 ran run_and_close.sh once, originalcommon
externalHTTP/openloop/60snotice/coldper-runcache, growth20GiB. Outcome/closure/
remoteownedstop/independentaudit nowcompletedabove. No build/compression/scans/
remotechangesduringinference. DoNOTrerunprepare/launch or overwriteoldresults.
NumericalLoRA/fullpipeline/KVmigration/consolidation stillOPEN; no fabricated
n_correct. Then4000 actualHydraServe ->allavailablebaselinecomparison ->
PrimeLoRAoptimization. Fullmatrixbelowunchanged. D504BASEd8069f5ffba814c9c4252cc5c238d43c0988669e
andMAIN286ec631fef869766dc8fdd20241d248f6a60ae5 pushed/remote-equal;
allD504CPU1–8actualemptyclosed. No backup/build whileD505GPUactive.
Historical watchdog sample227:48 HTTPresponses observed, actual4GPUs allocated;
service44.14GiB,hostavailable61.17GiB,diskfree173.47GiB,high/max/OOM/swap0.
This wasprogressonly, notthefinalresult. Postreleaseaudit.py (D503audit adapted
onlycount/configpath) subsequentlyranAFTERclosure/remote-stop success andpassed.

## D504 actual qualification COMPLETE (2026-10-13; not W0 performance)

BASEd8069f5ffba814c9c4252cc5c238d43c0988669e pushed/remote-equal. No GPU job.
Actual500/500 native-contract responses,500 distinct logicalIDs,0 failed,
500 request-driven realremote transfers/420,740,410 wireB, allmatchedto local
receipts; no request packing/temp. Native input/adapter/length/time contract PASS.
n_correct remainsnull; no numerical equivalence/fullpipeline/KVmigration/
consolidation qualification. Onephysicalhost,fouroriginalpp1 instances, original
30s idle path releasedall. This adapter_coverage_v1 isNOT W0prefix/performance.

Observed physical container-reservation union4604.206954383GPU-s,
9.208413909/offered; pre0/arrival2239.345318356/drain2227.887869176/
cleanup136.973766851. Mean/P95TTFT289289.305915/505736.984892ms,
TPOT60.719879/72.787144ms; no comparison tofullW0. D503 boundarycaveatunchanged.
ServiceInvocation827508296d1a42a28fa73b8c3147628e,inode24089907 actuallygone;
launchaux9Invocation404bb6498f9f4170b1b99b59dd0f44f2,inode24089865 emptyclosed.
Peak61,854,396,416B; high/max/OOM/OOMkill/swap0; actualGPU/provider/storage/
pausegone. Remoteownedservices stopped AFTERlocalrelease; portsabsent.

Original tmuxexit92 ispostruncollectionerror, notgenerationfailure. Missing
prelaunchhealthreceipt made originalcollectorstopbeforeSSH. Firstofflineaudit
was also premature, stoppedbeforeoutputs. Originalerrorspreserved. Newpostrelease
healthreceipt (NOT retroactiveprelaunch), originalInvocation-owned stop and
SHA-verified collection succeeded; subsequentofflineauditPASS. Futureprerequisites
must be explicitly awaited/read beforelaunch oranalysis; never assumeasyncpass.

1396explicitfiles roundtripSHA; archive5,206,190B underBASEpaper_results/ieee_tc/
hydraserve/d504/coverage_evidence.tar.gz, SHA recordedinimmutable summary.json.
Protected147/userdirty/vendorclean/secrets/diff/88freshCPUtests/pushPASS.
CPU1–6actualemptyclosed; curation6peak52,097,024B/events0/swap0. Backup7/8
closures separatelyrecorded. Free187,416,596,480B (~174.5GiB), freshgatesneeded.
ReadBASEdocs/ieee_tc/D504_HYDRASERVE_500_ID_COVERAGE.md. Raworiginalresults/
ieee_tc/hydraserve_qualification/d504_20261013. Neverrerun/overwriteD504.

NEXT1000compatibility ->4000 actualHydraServe ->allavailablebaselinecomparison
->PrimeLoRAoptimization; no more elementarywitnesses or baselinepolicytuning.
Numericalqualification explicitlyOPEN; cannotinventcorrectnessorforcepipeline.
Fullremainingmatrixbelowunchanged; goalACTIVE. NofiguresuntilnearPrimeLoRAtarget.

### D504 preparation and sequencing history (preserved, no longer running)

Prepared raworiginalresults/ieee_tc/hydraserve_qualification/d504_20261013 by
mechanically reusing D503 native/common HTTP/remote/physical path. No source or
baseline-policy changes. Explicit adapter_coverage_v1 maps sorted500 existing
IDs onto first500 existing prompt/arrival/target entries; not an ordinary W0
prefix, no replacement trace or weights generated. All500 small configs matched
frozen contentSHA:350rank8/150rank16, fourq/k/v/o targets; single tokenizerSHA
already exercisedD503. This metadata audit isNOTGPU/numericalcoverage. Native
ranklimit16 unchanged. Existing poolmaterializedbytes12,985,984,450;20GiB growth
gate includes views/temporary/log allowance. No fullpoolprefetch.

CPUprepare1actualemptyclosed. RemoteD78/D80 services started exact-ownedD504;
no republish/request-time packing. Prelaunch2 FAILED at authenticated health
connection-refused immediately after service start (protected147/userdirty/source
checks had passed). The agent launched before incorporating that failure; this
ordering error is retained in prelaunch_failure_notes.md, not described as passed.
CPU2actualemptyclosed. Independent guarded-launch resource gates and publisher's
authenticated content-manifest handshake subsequently passed BEFORE deployment
notice. D504 remains functional diagnostic only; no remote restart/config change
or in-run health retry. Do not silently promote to performance or repeat the key.
D50388CPUregressions/sourcefreeze retained. D504 ran in tmux primelora-d504 with
common externalpublisher/60snotice/coldper-runcache; run_and_close.sh automatically
checks actual release then attempts remote collection (failure/recovery above).
No remotechanges/builds/scansduringrun.
DoNOT rerunprepare/overwriteinputs. Ifpassed:1000compatibility ->4000HydraServe
->fullavailablebaselinecomparison ->PrimeLoRAoptimization. Numerical/fullpipeline/
KV/consolidation OPEN. Fullremainingmatrixbelow unchanged. D503MAINbackup
b0d3ea628973399ceaa9f66a7990b77c9cddae50 pushed/remote-equal;allCPU1–8closed.

## D503 completed (2026-10-13; NOT a full performance point)

BASE77b5f3a0b5bcb2588aa515ae0bdfc9db2a317d59 pushed/remote-equal. MAIN preceding
f4b7e66ef85065cc306b48f0b4e48673ea4c2bb0; this status backup follows. No live
heavy job. Read BASEdocs/ieee_tc/D503_HYDRASERVE_COMMON_HTTP.md. Goal remains ACTIVE.

ACTUAL original full Scheduler + common external HTTP/open-loop publisher:
100/100 offered/arrived/terminal/native-contract-complete, 0 failed; 29 unique
adapters, 29 request-driven real-remote transfers, 50,650,865 wire bytes. All
transfer IDs/content/receipt SHA matched; no remote request packing/temp archives.
Native counts/canonical input/adapter/timing checks passed. Original idle loop
released all 4 naturally created pp1 instances on ONE physical host. Two nonzero
requests' outputs equal D502; n_correct stays null, numerical equivalence OPEN.

Physical actual-container reservation union 1033.412008754 GPU-s, 10.334120088 per
offered request; windows pre0/arrival523.945161059/drain352.671394928/cleanup
156.795452766. Actual native GPU contexts qualified, all leases closed. Boundary
starts BEFORE container creation, not first CUDA use; earlier native bookkeeping
is not reconstructed. pp1 claim-to-create has no intervening await; multi-rank
sequential create/cache reservations require separate boundary audit if exercised.
Prefix mean/P95 TTFT42512.344/78862.363ms, TPOT55.529/63.429ms; not compared with
full 4000 W0 points and not attributed to a specific mechanism without evidence.

88 CPU tests and independent postrun audit PASS. New opt-in queue/assignment/
backend-ready/native-submit observation ASTs leave original policy unchanged.
Original synchronous generator consumed off-loop; no fabricated V1 scheduler
timestamps or bare-vLLM replacement. Physical union supports native co-location.
Original test ModelSet tuple remains QUALIFICATION identity, not final common-SLO
performance config. No baseline tuning/search. Fresh official pinned main/engine/
scheduler/ModelInfo matched; no vendor edits or existing-k3s access.

D503 service Invocation218c1cd76a97484f945dd51301c11ff5,inode24076121 actualgone;
aux9 Invocation789e74a51b27450c824fb72a3b5b27d0,inode24075921 actualemptyclosed.
Peak47,493,828,608B, high/max/OOM/OOMkill/swap0; all workers/provider/storage/pause
and GPU contexts released. CPU1–5closed; curation6peak26,066,944B/events0/swap0
closed; BASEbackup7exactclosed; MAINbackup8 closure separately retained. Remote
owned3B/7B/monitor stopped AFTERlocalrelease; ports18080/18081absent. Actual link
1000Mb/s/full beforelaunch; historical100Mb cause unknown. No changes during run.
Free200,991,637,504B (~187GiB); fresh launch gates required, no broadcleanupneeded.

445 explicit files roundtripSHA; protected147/userdirty/vendorclean/secrets/diff/
88-tests/explicit-stage/push PASS. BASEpaper_results/ieee_tc/hydraserve/d503/
http_qualification_evidence.tar.gz SHA
7cce2da2186a7725381628f6019d89036dc1272132d4baed902cb5c71816857a;
summarySHAa55a4d0250f5b089a216ceda5e7a052091db0a1906f878d7a14b0779f7d6a7e3.
Raworiginalresults/ieee_tc/hydraserve_qualification/d503_20261013. Do NOT rerun
D503 prepare/launch/audit/checkpoint or overwrite sources/results. No new pools.

NEXT500-ID functional ->1000 compatibility ->4000 actual HydraServe -> all
available baseline comparison ->PrimeLoRA optimization. Reuse qualified common
HTTP/native scheduler/remote/physical path, not more two-request/import tests.
Numerical LoRA/full pipeline/KV migration/consolidation remain OPEN; 500 static
registered IDs do not establish 500-ID execution. Full remaining scope below is
unchanged; no new performance ranking or baseline optimum is claimed.

## D502 preceding checkpoint (two-request delivery qualification, NOT performance)

BASEc4dff5e0487f058cd36f4512c2c610bf0be4e78a pushed/remote-equal; MAIN preceding
02f737f72ad5c9431ab7b128ba5e2f6c224fc051. No live heavy job. Read BASE
docs/ieee_tc/D502_HYDRASERVE_REAL_REMOTE.md. Fullscope/mainline/goalACTIVE unchanged.

ACTUAL fullnativeScheduler+originalcoldstart/SHMfastload/init -> two request-driven
remoteLoRAs -> native152/89tokens -> original30sidlerelease PASSED. Two native
outputsequences equalD501; binding/token identity isNOTnumericalequivalence.
finance14,718,801B + medical14,736,516B wire, bothcoldmiss/contentverified andremote
pack_performed=false/temporary_created=false, transferIDs matchedserverrecords.
Actualnetwork1000Mb/s/fullbeforelaunch, historical100Mbcause stillunknown.
ExistingD78/D80readonlypublishedcache reused, no republish/newweights/pool/trace.
Only2objectsfetched;500staticdescriptionsdoNOTmean500-IDfunctionalqualification.

New NativeArtifactViews reuses PublishedArtifactResolver and lossless tokenizer
view; no extraweightcopy, no prefetch/policytuning. BoundRequest opt-inprepare runs
AFTERnativeassignment/init BEFOREexistingbackendsubmit. Common300stransportguard,
original5ssubmittimeout unchanged, totalrequestguard/actual latencyretainallcost.
Perobject metadata lock/receipt/statguard; cancellationjoinswriter; errorsrelease
originalinstancecounts. Officialscheduler/engine refetched/bytesmatched. Policy
ASTunchanged. CPUfirst3renamepermissionsfailuresrecorded(cpu_failure_notes.md),
noGPUstartedthen; fixedstagingpermissions/sealedfinalpublication,80testsPASS.

D502peak42970333184B; high/max/OOM/OOMkill/swap0. Allactualworkers/provider/storage/
pause/GPUgone. svcInvocation9d1230e11d7d482e9b882d6e8dc0a2ae,inode24053924;
aux9Invocation728b0375b7c34d2d87d98d1e14c8693c,inode24053882 actualemptyclosed.
CPU1–5closed; curation6peak20238336B/events0/swap0closed. Backups7/8exactclosures
separate. Remote3B/7B/monitor exactInvocation stopped AFTERlocalGPUrelease;
18080/18081absent/logSHAverified. No remotechangesduringrun. Free201828442112B
(~188GiB); freshgates required. No broadcleanupneeded/repeated. Existingk3suntouched.

228explicitfiles roundtripSHA, protected147/userdirty/vendorclean/secrets/diff/
80tests/pushPASS. BASEpaper_results/ieee_tc/hydraserve/d502/remote_scheduler_evidence.tar.gz
SHA0862b0d94dc54116dd097b33a8254547fe418930f83cbc336ca56263c205b3ac;
summarySHAe3e1e0750d11701b6a73934bf67eff65e092f51e8e225762f554917644adfb3e.
Raworiginalresults/ieee_tc/hydraserve_qualification/d502_20261013. DoNOTrerunD502
prepare/launch/checkpoint or repeat2requesttest. Rawremoteobjects/view metadataare
smallownedworkingcache, no automaticdelete withoutreferences/receipts check.

NEXT advance commonHTTP/open-loop frontend and physicalGPUledger with existing
prepare_ieee_tc_serverless_stack.py replay_frozen_http/FrozenReplayPlan interfaces;
retainactualoriginalScheduler, notbarevLLM orclient-sideartifactpreparation.
100 ->500-IDfunctional ->1000 ->4000 actualHydraServeresult ->allavailablebaseline
comparison ->PrimeLoRAoptimization. NumericalLoRA andfullpipeline/KVmigration/
concurrentconsolidation qualificationremainOPEN; don'tinvent n_correct. Integrate
necessaryqualification withactualnextworkload, notmoreelementaryimports/tests.
OriginalnativeModelSettestregistration inD501/D502 isqualificationidentity, NOT
frozencommonSLO/performanceconfig. ReadV1/V2beforecomparison/configselection.

## D500–D501 preceding checkpoint (NOT a performance result)

BASE2a7f8d7ee780ca12c75768882ea6547507adf322 pushed/remote-equal; MAIN preceding
af50deb60858e0c1e28a9ade7257bf51070e3ca4. No live heavy job. Read BASE
docs/ieee_tc/D501_HYDRASERVE_NATIVE_SCHEDULER.md. Goal/mainline/fullscope unchanged.

D501 actual original Scheduler.process_request -> queue/ours/prefetch -> original
ResourceManager coldstart -> native provider/privateSHM/fastload/app/init ->
Request/Instance/ChatEngine completed existing finance152 and medical89 tokens.
Original30s idle loop deleted actualworker and restored rgpus[[24]*4]. Onephysical
host4GPUs, nativepp1; no inventednodes or forcedconsolidation. OriginalModelSet[0]
testregistration retained, not finalcommonSLO/performanceconfig selection.

New fullcontroller source/SHA loader moves only Kubernetes SDK import into its
five originalusingfunctions; no fakeSDK/cluster, nativeHTTP/TCP/algorithmunchanged.
Exact flat-module aliases/conflict gates. Existing k3s untouched. D500 failed
after first actual152tokens becauseharness comparedcontainerPID1 tohostPID;
secondnot submitted. Itsfinalcleanup Python lackedpidfd andresult.jsondoesnotexist;
outerqualifiedcleanupreleasedall. Failedraw preserved. D501 matchesentryPID and
hostbirth/boot,retainsliveidentity/timeNScheck; onlyhostpidfdcleanuphelper uses
/usr/bin/python3. NativePython3.10.20 stillruns actualcontrollerdependencies.

D500peak42516488192B; D501peak42705088512B; high/max/OOM/OOMkill/swap0.
D501svcInvocation7d7de1bee24445e19a7158621db9bed3,inode24043224;
aux9Invocationec70d7d5d49045ca9287feb9236f30c3,inode24043182 actualemptyclosed.
AllGPU/provider/storage/worker/pausegone; CPU1/2alreadyclosed. Curation3peak
41578496B/events0/swap0actualclosed. Backupclosures separatelyrecorded.
75CPUregressionsPASS,protected147/userdirty/vendorclean/secrets/diff/pushPASS.
376explicitfiles roundtripSHA; BASEpaper_results/ieee_tc/hydraserve/d501/
scheduler_evidence.tar.gz SHA84852e27255f9ddf7255b7f683b1e9e2cf846e847b1feee0d32eb420e1f6f88e;
summarySHA71bad7c8603d2e5739c6585c32094de2e48bbbf24b6779cdc6a61d55068cf768.
Raworiginalresults/ieee_tc/hydraserve_qualification/d500_20261013,d501_20261013.
No repeated D501 launch/prepare. Free~188GiB; freshgates required. No remotechanges.

NEXT request-driven realremote LoRA integration using existing PublishedArtifactResolver
and qualified tokenizer metadata view; resolveafteractualinstanceassignment, no
futurefetch/fullpoolpreload/weightcopies/newtrace. KeepnativeScheduler/commonreplayer.
Then100 ->500-ID ->1000 ->4000 actualHydraServeresult ->fullavailablebaselinecomparison
->PrimeLoRAoptimization. Nativebinding isNOTnumericalLoRAequivalence; numerical,
remote,fullpipeline/KVmigration/concurrentconsolidation remainOPEN. No newranking.

## D498–D499 preceding checkpoint (actual GPU generation; NO performance result)

BASE a52dca9399e4c4a2e0aeb546e1b516d2f9cbf8cf pushed/independently remote-equal.
MAIN predecessor2921e92068678856d6e3ffcafc8918a54ebc13bb; statusbackup follows.
Read BASEdocs/ieee_tc/D499_HYDRASERVE_GENERATION_AND_RELEASE.md. No liveheavyjob.

D498 ACTUALLY ran prepared GPU witness: original Request/Instance/ChatEngine ->
native app with qualified lossless tokenizer views. req_00000 finance152/152 and
req_00021 medical89/89; nativebinding/counts/clock/processidentity/counters PASS.
Original provider/twoexistingHFshards/nativepacking/two-threadwire/private13GiB
SHM/C++fastload and /init200 used. NOT fullScheduler,realremote,numericalLoRA,
pipeline/KVmigration/consolidation or performance qualification. No newinput/pool.
D498previousCPUonly summary/archive staysimmutable, not retrospectively changed.

D499 necessary SHM connectionownership repair implemented as opt-in shm_control.
Original startup asyncio.run closesitsloop; original StreamWriter laterclose on
uvicornloop fails. NativeShmLease ownsnonblockingsocket independentofthatloop,
loop.sock_* performs pendingI/O; socketclose sendsnativeEOF. Samewire/header,
loaderarguments/thread, originalstartupoverlap/init/consolidationcalls andcache/
schedulerpolicy. Cancel/loaderfailure closesandreraises; duplicatebeginrejects.
Onlytwo connectionfunctions/globalowner changed; defaultolderoverlaysunchanged.
Pinnedofficialapp refetchedSHA matches; CPython3.10 docs reviewed. No newfallback.

D499 ACTUAL GPU witness PASS: native readonlyqueryflag1(knownbutcleared) and
free_buffer(0,13476855675) recorded BEFORE generation/workerdelete while exact
workerPID/birthstillalive. Bothsameexistingrequests subsequentlycompleted152/89;
outputIDs equalD498forboth. ThisprovesSHMpoolrange reusable, NOT thatpreallocated
tmpfspagesreleasedtohost or wholememoryreduced. 7newCPUtests(reaLoopbackTCP),
full70testsPASS. Initialpreparequoteanchorfailurepreserved; finish_prepare only
completedabsentfiles. D499reusesD498adapterviews; nocopy/repack/regeneratedtrace.

D498peak42909892608B; D499peak43116400640B; bothhigh/max/OOM/OOMkill/swap0.
ActualGPUclear/provider/storage/worker/pausegone andservicegone. D498svcInvocation
5fbfe99c8c744bb7a79ffb85c88af7f5,inode24026359; aux9Invocation5338caa3a458439b829abbd8ed1b8a44,
inode24026317. D499svcInvocationb958e31b01ca4aa39487a834b39ba927,inode24032704;
aux9Invocation04fbb66a4b704cf991ef4c1594c791dc,inode24032662. Allactualemptyclosed.
D499CPU1/2/6closed; curation3peak34811904B/events0/swap0closed; backup4/5receipts
separateinraw. Free201933180928B(~188GiB); freshlaunchgates stillrequired.

407explicitfiles archivedwithroundtripSHA; protected147/userdirty/vendorclean/
diff/tests/secrets/explicitstage/push gatesPASS. BASEpaper_results/ieee_tc/hydraserve/d499:
generation_release_evidence.tar.gz SHA943da4c15e09f6535b084dadb710240d4eb04cb53d08e82094d1a92c71fc92fe;
summarySHA72c392496aba3ba67e828a0e996ca5f01aeaa1829df4e983d9a30217227e8fa2.
Raworiginalresults/ieee_tc/hydraserve_qualification/d498_20261013,d499_20261013.
D498livecode subsequentlyevolved; verifyits executed_sources snapshots against
itsmanifest, NOT currentnative_delivery as ifthatranduringD498. No remotechanges.

NEXT actual fullScheduler/realremote integration, NOT repeatD498/D499/prepare or
elementaryimports. Original Scheduler.process_request/check_model_loop and
create_cold_start_instance mustusequalifiedLifecyclePlatform,originalours/prefetch/
placement/counters; currentlywitnessusesalreadyinitializedInstance. Nativeutils.py
imports Kubernetes/Aliyun atmodulelevel; isolate platform-only dependency without
fakecluster or replacingoriginalHTTP/TCPrequestfunctions. Nativefiles under
vendor.../scripts/kubernetes/vllm/src (NOT top-levelsrc). Keeponephysicalhost4GPUs.
Connectexistingrequest-drivenrealremote LoRAresolver andtokenizerview,original
fullScheduler/commonreplayer;100->500-ID->1000->4000 actualHydraServeresult ->
allavailablebaselinecomparison -> PrimeLoRAoptimization. NumericalLoRA andfull
pipeline/KVmigration/concurrentconsolidation remainOPEN. Fullremainingmatrixbelow.

## D497–D498 preceding checkpoint (failure preserved; NO performance result)

BASE cfa5ba2715eaf85b95fb454c2f528b11eae97ac9 pushed/independently remote-equal.
MAIN predecessor fb6fc54792b26288cdaffaa6e1ea9b9b55826ea5; status backup follows.
Read BASEdocs/ieee_tc/D498_HYDRASERVE_TOKENIZER_COMPAT.md. No live heavy job.

D497 ACTUALLY launched native provider/private13GiBSHM/C++GPUfastload/originalapp
and Request/Instance/ChatEngine with first existing finance request req_00000,
target152, canonicalpromptIDs,greedy/ignoreEOS/correctLoRAname/index/path. Init200,
health200, completionHTTPheaders200 but engine failed BEFORE first native output.
Servertrace: get_lora_tokenizer -> TokenizerFast.from_file -> ModelWrapper JSON
deserialization. ZERO completed requests; medical req_00021 target89 not submitted.
Client incompletechunk stream is consequence, not networktimeout/performance.
No token or numericalcorrectness claim. D497 original overlays/logs/failure retained.

Cause: existing adapter BPE JSON stores merge pairs (tokenizers>=0.20); pinned
HydraServe tokenizers0.19.1 expects strings. Primaryofficial0.19.1/0.20.0source
fetched/archived. New runtime/tokenizer_compat.py ONLY lossless pairs->legacy
serialization, exact JSON inverse; reject spaces/ambiguous/mixed shapes. No
base-tokenizer fallback, backend upgrade, slowtokenizer switch or policy tuning.
Immutable adapter view symlinks original weights/configs, writes tokenizer.json
only. TWO converted metadata files,1156966B each; weight copies0. Originalpool
untouched. Originalsource dirs + backbone + view must be readonlymounted together.

D498 CPU qualification PASS: existing0.22.2 original vs native0.19.1 converted,
each2adapters x all4000existingtrace texts encoding+decoding hashes identical;
fullvocab/all32000single-token decodes/special map+IDs/BOS/EOS identical. No
GPUgeneration tested onnewview yet; not a universal library/numerical proof.
63CPUregressionsPASS(4new+59prior). Protected147/userdirty/vendorclean/diff/secrets
andexplicitBASEpush gates passed. Existingk3s untouched. No remote service changes.

D497peak42372599808B; high/max/OOM/OOMkill/swap0, actual GPUclear/servicegone,
worker/provider/storage/pause released. svcInvocation0d318768a0774e5a9f9a9329629b3b6c,
inode24018345; aux3Invocation728720af2b57465eab273d8ced571dda,inode24018303,
exactemptyclosed. D497prepare1/2closed. D498CPU1–4 actualemptyclosed, peak<=
217980928B/events0/swap0. Curation5 actualemptyclosedpeak33546240B/events0/swap0.
Backup6/7 closures in raw receipts. Free201944530944B(~188GiB), fresh gates needed.

231 explicit source/failure/CPU qualification receipts archived with roundtripSHA:
BASEpaper_results/ieee_tc/hydraserve/d498/tokenizer_compat_evidence.tar.gz
SHA3ef88b51c13da22f3d430aeeeacea6297173b57f429ab4bb28aa733bff9edaaa;
summarySHA4e86bc2bca6b49e12372b5991d0001ddc5cd699119a6defd56c8f32622e137e5.
Raw originalresults/ieee_tc/hydraserve_qualification/d497_20261013,d498_20261013.

NEXT: D498 actual GPU request witness is ALREADY PREPARED, NOT LAUNCHED.
Do NOT rerun prepare.py/tokenizer_witness or overwrite outputs. Read fullPLAN and
thisstatus, fresh idle/resources gate, then tmux bash rawd498/launch.sh. Dedicated
launchaux is d498...0009; close_audit.py matches9 (CPU1–4 alreadyclosed). Source
manifest frozen, same D497 native protocol/inputs; only qualified tokenizer view.
request_witness.py unchanged; mounts include original symlinktarget directories.
After terminal, close_audit.py records actualrelease; preserve any newfailure.
Need native binding/token/clock checks, not just init200. D498curatedcheckpoint
explicitly says GPUrun=False; anylater GPUevidence gets NEWappend-onlycuration,
not overwritten summary/archive. Scripts/source are preserved there for recovery.

Then fix separately native startupStreamWriter ownerloop lifetime/release before
repeated/multireplica loading; integrate existing realremote request-drivenresolver
and originalfullScheduler;100->500-ID->1000->4000 HydraServeactualresult -> all
availablebaselinecomparison -> PrimeLoRAoptimization. Full pipeline/KVmigration/
concurrentconsolidation correctness stillOPEN. No barevLLM replacement/fakeReady.

## D496 preceding checkpoint (CPU contract tests, NOT GPU generation/performance)

BASE9459dd459799ab7eecbad769b102bb009708c038 pushed/independentlyremoteequal.
MAIN predecessor0f23fd1aa72a2664b89471ee92a8c743be37fbbc; statusbackup follows.
Read BASEdocs/ieee_tc/D496_HYDRASERVE_REQUEST_CONTRACT.md. No liveheavyjob.
NEW runtime/request_contract.py: opt-in FrozenRequest travels through original
Request.prompt/Instance/ChatEngine. CanonicalinputIDs,adapterID/path/index and
target passed to native completion, greedy/ignoreEOS/nostop. Originalsyncstream
and5secondsubmissiontimeout retained. EntireScheduler/Model/Request AST unchanged;
originalresp_generator successfulprofiling AST unchanged. No baselineparamsearch.
Error/cancel/streamclose nowreleaseactualnativecounts once, notswallowed/unboundreply.

Native RequestOutput construction snapshots perf_counter; CompletionOutput copies
tokenIDs because originalpoints atmutatingsequence list. Native completion SSE
addsdeltaIDs/nativebinding/processPID/startticks/bootID/snapshottime. No text
retokenization/chunkcount/expected-count replacement; NOT kernelpertokenclock.
Native sequencebinding isNOT numericalcorrectness. Crossprocessclock/workerreceipt
stillmustbequalifiedwithcommonprotocol; actualGPUgenerationNOTyettested.
StaticLoRAregistry SHA/native1-basedorder onTWO nativecompletioninitializations;
metadataregistration doesn'treadweights,fetch,claimready orvalidatefiles.
native_delivery.render_overlays(...,request_contract=True) composesD480/D495;
defaultD495pathunchanged andoldoverlays/resultsnotmodified. Cleanvendoruntouched.

59CPUtests PASS(14new+45prior), includingactualrenderedRequest->Instance->ChatEngine
withtestresponse doubles, emptytext/multitokenchunks/errors/cancel/coldclose/source
drift/registry/AST/outputalias checks. NOT fullschedulerHTTP/GPUinferencequalification.
Officialengine.py fetchedonline/SHAc0722e5512fd7ef88a19146f85850d9715fe8002baf49de4fe33aed377174337.
protected147/userdirty/nativeclean/diff/secrets/explicitstage/backup allPASS.
BASEpaper_results/ieee_tc/hydraserve/d496/request_contract_evidence.tar.gz SHA
aefe5ce374ec64cf0f4ef5944f627588b7f8fba73f2369e81c6cf84ec9bce94d;
summarySHAbeb0a6d8e3edce79d3a8689bf2e81f02833ea95c58cbeec8e120f58835366800.
Raw originalresults/ieee_tc/hydraserve_qualification/d496_20261013.
CPUaux1/2exactInvocation/recursiveemptyclosed; curation3 peak33443840B/events0,
swap0 actualemptyclosed; backup4exactownedemptyclosed. No GPUjobstartedthischeckpoint.
Free201967214592B(~188.1GiB), available110184624KiB; freshgatesneeded.

NEXT advance ACTUAL generation (not another import/container or onlyCPUtest):
reuseD495fullnativeprovider/SHM/fastload/app recipe + opt-inrequestoverlay and
staticreadonlymetadataregistry foralreadyexistingadapters. Qualifyoriginalupper
Request/Instance path withnativeoutput/LoRAbinding andfixedcommoninputs, then
integrateexistingrealremote request-drivenresolver/replayer andfullactualScheduler.
No barevLLMreplacement. PreserveD495/SHA/source snapshots andfailedattempts.
Before repeated/multireplicaloading, fixnative startupStreamWriter ownership:
asyncio.run closesitsloop; laterend_load_model_shm hitsEventloopclosed andstorage
free observedonlyatworkerdelete. Needprotocol-preservingownerlifetime+releaseproof.
Registryexpects HYDRASERVE_LORA_REGISTRY + _SHA256; JSONlist{adapter_id,
adapter_index,adapter_path}, contiguous1-based nativeenumeration. DoesNOT fetch.
Then100->500-ID->1000->4000 HydraServeactualresult -> allavailablebaselinecomparison
-> PrimeLoRAoptimization. Pipeline/KVmigration/concurrentconsolidation correctness
OPEN; no performanceclaim from D495initialization or D496CPUcontract.

## D494–D495 preceding checkpoint (NOT a performance result)

BASE c334727892b7b7fa2c63c4fd10cf3f124f9695a1 pushed/independentlyremoteequal.
MAIN preceding9446e814ebdde563878cc9b156028756663f6441; currentstatusbackup follows.
Read BASEdocs/ieee_tc/D495_HYDRASERVE_NATIVE_DELIVERY.md. No liveheavyjob.
Actualnative CPUprovider/two-existing-HFshards/packing/two-threadTCP/private13GiB
SHM/C++GPUfastload/originalapp+LoRAmanager/KVinit allconnected. D495 /init200,
/health200,/v1/models200. OriginalResourceManager actualGPUselection/release used;
GPU-visibleUUID verified beforepermit. Onephysicalhost, notfour inventednodes.
Readonly metadata/symlinks only; no tensorcopy/combinedcheckpoint/newdataset.

D494 originalfailure retained: nativebatchdefault65536 violates ownLoRAkernel
ceiling65528, so/init500. D495 explicitlyuses source-AST-derived65528 onthree
LoRA-enabledEngineArgs sites; notparametersearch/performanceoptimization.
Officialconfig re-fetchedonline, SHA5a17e5ca3741314175daa825024172be0decb56dd411d6e40ff146cfe15d8332.
ProviderwithoutLoRAdefaultunchanged; originalpacking/wire/storagepolicies retained.

Bothrunscommonhigh72/max80GiB/swap2,CPU4-23/28-47; wholeprovider/storage/worker
incommonparent. D494peak42040569856B; D495peak42318667776B; allhigh/max/OOM/
OOMkill/swap0. Workers/provider/storage/pausegone; actualGPUclear,svcgroupgone,
exactInvocation/inode emptyauxclosed. D495svcInvocation303886b8151e43debae95deca14549cf,
inode23993694; auxInvocationeacea1d57eaa4b42a66c8c7dfb4d335a,inode23993652.
45CPUregressionsPASS; protected147/userdirty/nativeclean/secrets/diff/PUSHallPASS.
186explicitfailure/success/source records roundtripSHAchecked:
BASEpaper_results/ieee_tc/hydraserve/d495/native_delivery_evidence.tar.gz SHA
7cc57b5e05fc09e9fb34f76abe787955c549d1f67ae9578ba3e725fc4e62fbf4;
summarySHA526a43234a21ad053d196eeea79d597e3c3f1439fe1754b42bd628cc4bbc3a98.
Raw originalresults/ieee_tc/hydraserve_qualification/d494_20261013,d495_20261013.
Latestdiskfree201970425856B(~188.1GiB); available110223124KiB; freshgates required.

NEXT: nativeRequest/ChatEngine input+LoRA+fixed-output+native-token/timingcontract,
not another elementary import/container/SHM test. Officialsrc/engine.py chat uses
temperature1/max_tokens1024, Request in src/scheduler.py lacksadapter/target.
Native vllm/entrypoints/openai/serving_completion.py alreadyhasLoRARequest and
token_ids; keep actualHydraServe scheduler, don't replacewithbarevLLM. Registration
llama2_7b hasnochattemplate; choose necessarycanonical-input adaptation explicitly.
Also observed nativeapp startup asyncio.run closesoldloop; end_load_model_shm
laterreports Event loop is closed. Storagefree_buffer observedonlyaroundworker
deletion; notproof of promptsteady-stateSHMrelease. Trace control-socketownership
before multireplica/consolidation; protocol-preserving correctnessrepair ifneeded.
Nativeactualgeneration/per-requestLoRA/pipeline/KVmigration/concurrentconsolidation
remainOPEN. Then100->500-ID->1000->4000 HydraServe -> availablebaselinecomparison
-> PrimeLoRAoptimization. No newperformancepoint/ranking/claimfrominitialization.

## D492–D493 completed (NOT inference/performance)

BASE929b18ad1f84b8d3433de03126430794c8ae707e pushed/independentlyremoteequal.
MAIN predecessora108123ff7a5bf6b9976683952e02c70e79358d3 remoteverified;
status/history backup follows, exactreceipt D493raw. No live heavyjob.
Raw originalexperiment results/ieee_tc/hydraserve_qualification/d492_20261013
and d493_20261013. Read BASEdocs/ieee_tc/D493_HYDRASERVE_CONCRETE_LIFECYCLE.md.

NEW HydraServe_project/runtime/isolated_platform.py actually implements D480
LifecyclePlatform, reusing D477/D481/D491 containment/IPC/address recipe.
stdlib-only container_entry.py awaits actualPID/birth/affinity/limits/privateIPC
checks BEFORE nativecommand; no torch import before original app. One physical
host inventory; explicit native-image mapping and immutable localimage, readonly
mounts and selectedUUID. Original ResourceManager still chooses GPU, creates,
awaits /init and releases. Addresses are NOT Ready; deletion requires actual
exit/recursiveempty childgroup. Hostnetwork is NOT kernel network isolation.

D493 ran original local_server with D481 address-only overlay, actual1GiB
privateSHM allocation/touch and originalhealth/free PASS. Original ResourceManager
created TWO realcontainers,gpu_id0/1; both read privateobject and nativehealth.
Synthetic NO-MODEL /init endpoints delayedacks; controller actuallywaited then
restored resourcebookkeeping only afteractualremoval. NOT modelReady/inference/
fastload/GPUarithmetic. CDI rewrites NVIDIA_VISIBLE_DEVICES to void while
injecting selectedGPUdevicefiles; don't infer noGPU from thatenv. NoCUDA/model.

Allworkers/storage/pausegone; scope1Invocation4fc23cef1c1f485f8c865213ec3e8cec,
inode23976812 actualrecursiveemptyclosed. Parenthigh3/max4GiB/swap0,
CPU2,3,26,27; childmax2GiB/swap0. Peak1450344448B,allhigh/max/OOM/OOMkill/swap0.
Free201994956800B(~188.12GiB); freshgatesneeded. NOT formalserviceenvelope.
D492 failed BEFOREcontainers because baseline scripts shadowed mainpreflight.
D493 imports preflight FIRST then addsBASEpath. Originalcode/error retained;
no claimedoriginalexitcode. ScopeInvocationf21ceb89e2bb4681a350dcdc49878e50
actuallyemptyclosed. NotGPUfailure/OOM/baselineperformancefailure.

Afteractualrun: entryJSON atomicpublication and boundedCLI cancellation awaiting
completion beforecleanup added. Originalexecutedsources retained *_executed.py;
don't claim laterrevisions were exercised inD493containers. Updatedcode38CPUtests
PASS includingactualboundedsubprocesscancel, priorpolicy/source andD479/D491tests.
Testscope2closedpeak27721728B/events0.95explicit source/success/failure/resource
records archived/roundtripSHAverified; protected147/userdirty/nativeclean/secrets/
diff/explicitstagePASS. BASEpaper_results/ieee_tc/hydraserve/d493:
archiveSHA0f025ce5fbfa4fe3c67d58e34d2c3020af8388ed2e72fa58b83db7117ae05ff5,
summarySHAa8ea86c80a3407d7007f014f67186cea027cadadbe3a114fea15423aa57bfb0d.
Curation3actualemptyclosedpeak12320768B/events0; backup4closurefollows.
Officialresource_manager.py refetchedonline SHA equals pinnedinput. Webtool429
notusedasevidence; boundedrawGitHubfetch succeeded. No nativecheckout changes.

## D494 precursor detail (D495 current next action above)

D494 (2026-10-13) NEW actual CPU provider loaded existing TWO 7B HF shards,
native packing/partition/two-thread wire -> private13GiB SHM -> C++ GPU fastload
completed. GPU UUID gate actually passed. Original app /init returned500 before
engine construction: native batch default65536 exceeds its own LoRA kernel cap
65528. This is a concrete compatibility failure, NOT a performance result.
Raw d494_20261013 retains executed source overlays, logs,failedresult and closure.
Common72/80GiB,swap2,CPUs4-23/28-47; observedpeak42040569856B,events/swap0;
GPUreleased/servicegone/exact-owned emptyauxclosed. Existingk3s untouched.
NEW readonly metadata links/shard iterator/import overlay/provider lifecycle
passed15CPUtests. No wholecheckpointcopy/download/partition rewrite. Nativeapp
LoRAenable/extra_vocab0 necessary for existingweights; native LoRA/batch/cache
defaults otherwise retained. NEXT revision only clips batchtokenlimit to exact
source-pinned native LoRA ceiling65528 (not performance tuning); D495initnowpassed.
Upper requestLoRA/nativeoutputcontract stillOPEN. No PrimeLoRA candidate now.

Use concreteplatform with D491lora_worker_binding overlay. Preserve modified
vLLM0.4.2/nativepipeline/startupoverlap/consolidation. Need actualprovider->private
SHM->C++fastload->app/init (D495passed) AND upperleveladapter/fixed-output/native-tokencontract.
Then100->500-ID->1000->4000 HydraServeactualresult -> fullavailablebaseline
comparison -> PrimeLoRAoptimization. Do NOT repeat install/build/elementaryCUDA/
container or sharedbinding tests instead of advancing this integration.

Newsourcefacts (readoriginals beforeimplementation):
- remote_server defaultsmodel_set3 including13B; restrict explicitexisting7B/3B
  registration. NativeModel CPU-packed byte strings+retainedCPUmodel duplicate
  memory; accountincommon80GiB, preserve nativepartition/serialization/two threads.
- provider's offline safetensors_weights_iterator reads ONLY MODEL_PATH; existing
  7B hasTWOHFshards. Necessaryreadonlymulti-sharditerator adaptation; no combined
  fullcopy. NEVERdownloader.partition onuniqueweights (pp1rewrites/deletes).
  Onlysmallconfig/tokenizerlinks/partitionmanifest/duplicateJSON metadata views.
- LOCAL_SERVER=1 FAST_COLDSTART=1 ENABLE_PARA=0 ISoriginalSHMfastpath.
  App starts torch.ops.start_load_shm BEFOREvLLMimports. Wrapper mustnot addheavy
  torchimport ahead. ThreeEngineArgs siteslackLoRAenable; upper Request/ChatEngine
  lacksadapter/generation/nativeclockcontract; low-levelsequence/broadcast hasLoRA.
- local_server touchesSHM_SIZEGiB andclearsprivatepsm_*; neverhostshm. This1GiB
  test isNOTfull7Bcapacity. Actualprovider+SHM+workers need80GiB audit+watchdog.
- native slow_exprdefault0/USE_CACHEdefault0 preserved. Scheduler acts perPHYSICAL
  NODE, so onehost4GPUs naturallyselectspp1; no inventednodes or forcedconsolidation.
  Controlled pipeline/KVmigration/concurrentconsolidation correctness stillOPEN.

## Qualified components and reusable environment (don't redo)

BASE/vendor_new_baselines/HydraServe_main_20260520 cleanHEAD
8ae605de354ccfa2e9095514cdb4a9e9c56aa56b. D480/D481/D491 sourcepinned overlays.
BASE/installs/hydraserve_native_d484_20261013: Python3.10,Torch2.3.0+cu121,
xformers0.0.26.post1,Transformers4.43.1,flash_attn2.5.8/Torch2.3/cp310/ABIfalse,
Triton2.3.0,Ray2.9.3,nativevllm0.4.2 originalPython+ownD485kernels.
ReadonlyPythonbase anaconda3/envs/hydraserve_py_20260520; readonlyCUDA12.4.131
andnvidia namespace from slinfer_official_20260612. Neverpip-upgrade sharednamespace.
D486private_home/.config/vllm/nccl/cu12/libnccl.so.2.18.1 SHA
005503b62b9fe466b6b2f60ba2ad1daa4571196ffcb82a3d77a3c775bfcd5ac8;
setVLLM_NCCL_SO_PATH. D485build_fastload/lib/libst_pybinding.so SHA
f11b31a82f654d7c8950d85e05461d31b5cc45469720deb1e784d47022571e44,
mount/vllm-workspace/libst_pybinding.so. ImportPASS, actualSHMload/appinitD495PASS;
end-to-endgenerative correctnessstillOPEN.

D487Punica/SiLU/FlashAttention GPU arithmeticPASS; D488nonzeroLoRA128positionsPASS.
D491 actualnativeDefaultModelLoader/Llama/ModelRunner2loadboundaries/oneLRUmanager
pp4rank3->pp2rank1 ONONEGPU,32packedmodules global16–31/twoalreadyactiveLoRA;
maxrelativeL2.0002201656752731651,fixed.005tolerance;8decoderobjects/head/slots
shared. CompleteoriginalCPUinit/model.to/postprocess, onlyHFiteratorcontrolled.
NOTattention/KV/pipeline/KVmigration/concurrentconsolidation/fastload/e2etokens.
ReadD491doc; D489circularimport/D490incompleteTESTfactoryfailures retained.

Ubuntu22.04imagee8df6c74f650ad385efe2e20103eca9b0ad50690fb3c63d49fa00762412bbf55,
D477container_proxy_attempt/container_storage vfs; runtime
/run/user/1001/primelora-hydra-d477-proxy/runroot; Podman3.4.4/NVIDIAruntime1.19.
Use/usr/bin/python3 hostpidfdcleanup. Pauseinitiallyownscope: verify/adoptBEFORE
workers, notclaimborninparent. CPUcontrollerundelegated; taskset+actualaffinity.
ReuserecipeD493/executed_probe.py plusBASE D480/D481/D493docs.
Originalbackbone models/meta-llama--Llama-2-7b-hf readonly. financefullSHA
971901c3140f83f57cbb19fcc86d33b4d010f6d8fc15fbec621f0347bc31a258;
medical d2b1dcfe42193917d0b2e0f728e735f24ad119dd2c77cb9b20c983c7f688fa5c.
EqualheaderSHA isNOTequalfullweights.500logical=498zero/2nonzero,4weightSHA/
6treeclasses limitationretained. No newmodels/traces/pools/13B.

## Frozen evaluation and sealed7BW0 (do not repeat)

Beforecomparison/configselection read METRIC_PROTOCOL_FROZEN_V1.md and
METRIC_PROTOCOL_SINGLE_RUN_V2.md FULL; respectiveSHAs
5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22,
193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
G1 all4000contracts+commonjointTTFT/TPOT>=95%,thenphysicalGPU-s. G2same+
U<=.75Uref,W1/W2tail/observedSLOdifference; CEsupplementary. Unknownnumerical
n_correctstaysunknown. Oneperformance/frozenkey; noCI/repeatmeans/winnerfilter.
TPOTrelative regressionalone notfailure; retain development/reuseidentity.
Sealed(joint,GPU-s/boundrequest,P95TTFTms):
PrimeLoRA D476 .97425/2.694612200556614/2315.363336;
ServerlessLLM D414 .98125/4.022440686/1052.246361;
vLLM D423 .99475/4.028844291/1200.506860;
S-LoRA D435 .94425/3.789269468/17699.067366 nonfeasible;
Loquetier D445 .0435/4.249595809/555110.032229 nonfeasible;
ElasticLocality D453 .95775/2.473178968/2420.607829.
HydraServe missing; PrimeLoRA stilllosesElasticLocalityresource. No newcomparison.
D3947BSLO input<=616/>616:TTFT2995.348713684507/4633.345350589479ms,
TPOT59.74409694902162/85.24698166015581ms.W0Uref16115.377162149176,
B12086.532871611882,Qref1200.5068596918136.W1/W2refsOPEN; qualifyactualW1
transform, not relabeldefaultW0. W0 isNOTW1/W2/G2closure. Historiesretaindetails.

## Safety, protection and publishing

Oneheavyjob; no experiment withbuild/install/hashscan/compression/remotechanges.
Diskstart>=max150GiB,100GiB+1.5growth; runtime100GiB. Servicehigh72/max80GiB,
swap2,CPU4-23,28-47; auxhigh3/max4GiB,swap0,CPU2,3,26,27. Emptyservicestart
MemAvailable>=102GiB; otherwise remaininggrowthformula. Actualworker restriction
BEFOREmodel/external16GiBhost/100GiBdiskguard. No blindOOMretry/globalGPUreset/
reboot/driver/unknowncleanup. Close exactInvocation/inode+actualrecursiveempty,
never virtualfilesize or systemdactivelabel. No rawlargefiles/credentials/forcepush.

Protected147seal originalresults/ieee_tc/slora_qualification/d435_20261011/protected_before.json
SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
Neveredit/stageMAINconfigs/generated/lora_manifest_1000.json SHA
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
BASEscripts/replay_openai_trace.py SHA
04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
scripts/run_serverlessllm_relayserve_continuation.sh SHA
337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
Explicitstage/source/checksum/tests/secrets/diff; MAINfaaslora_origin/
retry14_continuous_queue_v2 andBASEorigin/main; verifiedremotebackupcheckpoints.

RemoteSSHprimelora-artifact-174 lab14@10.199.227.174:8122 strictknownhost;
data192.168.4.174ports18080/18081inactiveD476.TokenoutsideGit
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint. D78/D80readonlygzip
1581215261B reuse,no republish/extractedpool/requestpacking. Latest1000Mb/s/full;
old100Mbpsdistinctstate,causeunknown. Realnecessaryread/transfer/contention/
materializationcount; no oldspan subtraction. No remotechangesduringexperiments.

## Remaining fullscope — goalACTIVE

HydraServeactualresult/fullavailable7Bcomparison/PrimeLoRAoptimization;
7BW1/W2/G2/commonrefs;3Boptimization/baselines;dLoRA3BB6;A1–A5confirmedstate/
elasticity/handoff/hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/
load/GPU/network/workload/KV/control/backend/EOS/designsensitivity;offlineCE/cost/
Pareto;B1/B2/C1/C2/C3/formula/novelty/two-reviewdocs;lateracademicfigures/V2delivery.
NineIEEEequations unchanged. No13B/newSGLang/paperedits/newweights/trace/pools/
oldresultoverwrite. Nofiguresuntilneargoal. Chinese paper-evidence updates, full
names PrimeLoRA/ServerlessLLM. Setup/lifecycle checks areNOTperformanceexperiments.
