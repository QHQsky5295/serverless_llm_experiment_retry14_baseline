# IEEE TC execution status

## USER PRIORITY UPDATE — 2026-10-06, no figures before optimization targets

The latest explicit user goal supersedes the old per-run plotting requirement.
Until Prime is close to its agreed correctness/SLO/GPU-s goals, retain raw data,
minimal comparison records and interpretations only; no new performance figures
or figure styling/QA work. Finish the current missing numerical measurement, then
return to7B optimization, then3B, then baselines and goal-driven fair comparisons.
Preserve paper equations/core semantics and the frozen evaluation ruler. Existing
figures/history remain untouched. Source plan SHA remains0c808509...; this is a
user execution-priority override, not a silent rewrite of the approved source plan.

## CURRENT — D367 complete W0 audited; formal qualification remains open

2026-10-09: A bounded CPU-only D368 diagnostic isolated a concrete startup
barrier before any code change.  D367's first runtime was ready 4.336 s after
the frozen replay origin, while the first dispatch was around 24.36 s.  Using
the exact D367 7B trace/config, synchronous parent-side request-plan
preparation took 19.101 s for 4,000 requests (first request 137.63 ms,
requests 2--100 mean 4.322 ms); an asyncio heartbeat run took 19.847 s and
observed a maximum heartbeat gap of 19.847 s.  This is direct event-loop
starvation evidence, not an autoscaling or remote-transfer attribution.  The
diagnostic used an owned 4-GiB auxiliary scope, performed no CUDA/GPU/remote
work, and the scope is inactive.  The full evidence and the falsifiable
next-step candidate (per-arrival preparation in a bounded executor for native
external replay, with the existing legacy cache retained) are in
`docs/ieee_tc/D368_STARTUP_PLAN_PREPARATION_DIAGNOSTIC.md`.  No serving code or
performance result had changed at the time of that initial diagnostic; the
candidate implementation and its unit qualification are recorded below.

The minimal candidate then passed a CPU-only 4,000-request validation with the
same D367 trace and fixed-output contract: 4,000/4,000 plans completed in
19.654 s, with 194 event-loop heartbeat ticks and a maximum heartbeat gap of
134.43 ms (versus the synchronous 19.847-s gap). The two earlier diagnostic
failures were harness omissions of the workload contract and are explicitly
excluded from evidence. The candidate executor scope was stopped. Existing
unit/lifecycle qualification now passes (3 new async tests plus 220 selected
launch/lifecycle tests). The next permitted action is one complete unchanged
7B W0 GPU replay to measure dispatch/TTFT/GPU-s effects; no 3B, baseline, or
figure work is authorized before that replay is audited.

2026-10-09: D367 completed one full 7B W0 development replay with
4000/4000 terminal/native-contract returns and zero terminal failures. The
immutable source is
`9c6bfb1929692949636e0a2b64aacf9d7707e2e449a31709114f38d11b636941`; physical
lifecycle GPU time is 13701.34494326613 GPU-s. The offline projection reports
mean/P95 TTFT 1309.9077/3175.8600 ms and mean/P95 TPOT 31.7334/54.5125 ms;
3835/4000 requests satisfy the provisional timing upper bound. This is not a
formal G1/G2 result because `N_correct` is intentionally unknown and the
common warm/Resident references are not frozen.

D367 context audit against D358 found exact equality for all checked request
identity fields and shared trace/adapter-subset hashes. It records 132 remote
transfers (131480060 wire bytes), no request-time packing, and 4029 resource
samples with peak service memory 20733222912 B, zero swap/max/OOM events, and
minimum host available memory 92912091136 B. There are 227 output-token hash
changes across independent development runs; this is retained as run variation,
not a causal performance claim. D367 is now offline-audited; no GPU or remote
service is active. No figures are generated.

The final ruler is unchanged: G1 requires all correct plus joint SLO >=95%
before comparing physical GPU-s/request; G2 requires all correct plus the
common budget and joint SLO before comparing W1/W2 P95 TTFT and SLO
non-inferiority. Raw TPOT regression alone is not a veto, but correctness and
common-reference evidence cannot be inferred from this replay. Next is closure
backup, then a first-principles 7B optimization/qualification decision; 3B,
baselines, M1/M2, ablations, sensitivity, and figures remain outstanding.

Closure backup completed and pushed as commit `2cc2d8ca37eeac25a0fa736aa84e940aa32db680`;
local HEAD equals `faaslora_origin/retry14_continuous_queue_v2`. The closure
archive has 42 members and 1082473 bytes. Two seal-attempt path/staging failures
were recorded in the raw D367 directory; their rebuildable partial outputs were
removed only after recording hashes. Neither attempt launched GPU or remote work.
The user-owned generated LoRA manifest remains unstaged and unchanged by this
closure.

The request-path boundary and final metric interpretation are recorded in
`docs/ieee_tc/METRIC_BOUNDARY_DECISION_20261009.md` (SHA-256
`845b7869fb4916f66fe777e20d269688889d8c21b7a5438013db3cc024e0b68d`). It is
an explanatory companion to the frozen V1 protocol, not a redefinition of the
G1/G2 ruler. The current read-only link audit is 1000 Mb/s/full on both ends;
the historical D77 100 Mb/s observation is retained as a separate link-state
diagnostic and is not pooled with D367.

## D366 native100 COMPLETE, audited, sealed and backed

2026-10-09:100/100/0fail,source0788a2b972887baafffc87cec7a501f8e078298eb294b7497bedcb386fd41905,
6097205B,execution0da5917. ActualGPUrelease/servicepathremovalpass,tmuxended.
stop_remote.sh executedONCE,3ownedremoteunitsinactive. collect_closure.py closed
launcher199a483c and collected bilateralSHA-matched journals. run_audit.py closed
resume starter4a0bf4be andcollectione535a956;69CPUchecks+strictauditpassed,[0,0].
Offline ef43e6d7 closed by seal_checkpoint.py without hardkill.332589B closure
archive/checksums/protected147/secrets passed,11publication paths staged. Sealer
afc30fe6 is empty; finish_backup.py closes it and records pushed-head equality.
Consult raw closure_backup_receipt.json before repeating anycleanup. Do not repeat finished
start/stop/collection/audit. Initialfailedstarterac000931 andpreparation5b09/2b72
alreadyclosed. Curatedsmoke1 includes33filetransactions(29Remote/4NVMe),58GPU/9HOST,
8advancedownerepoch,max5,currentconstraintsrechecked.29fetches50650865wireB,pack0.
262resourcesamples,peak11054030848B,minhost101917089792B,high/max/OOM/swap0.
Mean/P95TTFT694.858512/1538.731795ms,mean/P95TPOT29.218495/40.759924ms,
physicalGPU335.652851436s. VersusD363prefix:TTFT−23.63%/−36.99%,GPU−58.21%,
TPOT+11.38%/+29.43%;retainregression,single-run/noCI/nocausalclaim.
D170timing-only99/100;N_correct/commonwarm/Resident/formalG1/G2stillopen.
NEXTbackup,thenD367unchangedFull4000againstD358/D353/D336,nofurtherretuning.
SeeD366_COMPLETE_PREFIX_DECISION.md;7B→3B→pausedbaselinesunchanged,noplots.

### D366 live/preparation history — terminal

D366 execution0da5917af192a8e9055ac9c0651462c427e00355 pushed/equality checked.
Firststart(ac000931) closed preparation scope2b7249b3 and started remote services,
then systemPython health import failed (missingnumpy), BEFORE tmux/GPUrequest.
Retained start_attempt1_failure.json. resume_start.py closed only failedstarter,
verified SAME remoteInvocationIDs, used existing modelPython for health, and
started one tmux tc-d366-file100. No service restart/configchange or new package.
Remote actual1000Mbps/full,clock remote-process-monotonic:e27c6f05cf5d4a11b7f9681b3855ebc0.
Live pane555968, launcher199a483c identitycaptured; resume starter4a0bf4be owns
tmuxserver and must not be closed while live. Service639b9ca4 resourceguard active.
Watchdogsample32:host107573637120B,peak5012262912B,high/max/OOM/swap0,noabort,
disk161906069504B. Startupinprogress, no performance/source result yet.
Never rerun start.py/resume_start.py or capture_launcher.py. NEXT monitor same
handle→actualrelease→stop_remote.sh ONCE→exact-owned cleanup/evidence→audit→backup.
No tests, compression, serving changes or figures during measurement.
Continuation sample123 on samepane555968:16done/24arrived/0fail,
host107320127488B,peak7061876736B,high/max/OOM/swap0,noabort. Same live replay,
not a blocker. Offline analyze_d298_smoke now has drafted d366 registration,
descriptive referenceD363 and unchanged strict file/native/timing/resource gates;
UNTESTED untilGPUrelease. rg found no serving-path import; sourceexecution remains
0da5917. Rawcollect_closure.py and success-only run_audit.py reused fromD363 and
prepared,NOTexecuted. Audit uses resume_launch_gate.json for starter4a0bf4be,
not already-closed firststarterac000931. If anyrequestfails, do not invoke the
success-only analyzer; retain offeredpopulation and use existingfailureaudit.

2026-10-09: D365 backup_receipt.json confirms local/remote f7ccb13803cdcc3056c141aae6fe1cef371226bb,
verifier closed without hardkill. All D365 cleanup/seal/backup actions finished;
do not repeat. D366 reuses D363 configuration with only3output/cache paths changed,
same D362 200/24profiles, same existing W0 prefix100 and publishedremote archives.
Existing runner has an explicit d366 identity; no additional serving changes or
parameter tuning.69native timing/analysis tests,50OS guards and200/24strict profile
assembly passed. First publication check rejected one extra YAML EOF blank line;
all first tests/binding succeeded and first archive retained. Corrected onlythat
LF (binary SHA proof), reused tests, reran exact binding; final publicationpassed.
Firstscope5b09a622closedonce. Finalscope2b7249b3 is empty and start.py owns its
cleanup. Launch uses preparation_final.json/.sha256 and preparation_scope_final;
do not rerun either preparation helper or substitute first-attempt manifests.
Finalarchive15567B, disk161906073600B,host112072265728B. Source backup beforelaunch.
No D366 GPU result exists yet. See D366_FILE_IDENTITY_VALIDATION.md.
Mainline7B→3B→pausedbaselines unchanged; no new figures or asset generation.

## D365 deterministic file-demand identity repair; CPU qualification complete

2026-10-09: D364 cleanup/backup receipt passed, local/remote both e8f8a513.
Do not repeat D364 helpers. Reused D310's tiny real publication/lease fixture;
unchanged serving code failed all four bounded st_blocks refresh cases, while
the over-envelope negative case passed. This establishes a demand_witness code
defect, not the exact filesystem event behind D364's req_03511 (operands absent).
Repair compares immutable publication fields/schema, not the two mutable physical
allocation fields. Fresh signatures/envelopes, real live lease, logical size,
membership/content/representation checks and physical accounting remain. Failed
publication checks now retain both observations. No retry, tuning or formula change.
77 targeted checks and662 broader checks in EACH existing Python environment
passed. Peak647557120B,high/max/OOM/swap0. No GPU/remote work, no figures. Raw d365_20261009 retains
before/after sources/logs/receipts. All four exact-owned test scopes closed without
hardkill by seal_candidate.py;1726962B evidence archive/41members verified. Do not
repeat test cleanup or sealing. verify_checkpoint.py closes the sealer, then
close_verifier.py records final remote equality; consult raw backup_receipt.json
before repeating any closure. See D365_FILE_DEMAND_IDENTITY.md.
NEXT: source backup, then same-config native100 and
Full4000, reusing D362 profiles after strict binding verification. Do not regenerate
assets/profiles without new evidence or jump to3B/baselines. Mainline7B remainsopen;
D364 staysfailed; numericalcorrectness/commonwarm/Resident/formalG1/G2 remainopen.

## D364 Full W0 COMPLETE with one failure; 7B NOT qualified

2026-10-09: complete4000 terminal population,3999successful/native-contract
returns,1failure req_03511 education_lora_0127. Sourceb90a4f6874d0f5f8a9c0a1ad8ddf749430e11c46b32c35c9eaa71c3edb6762aa,
298535077B; executioned98ac3aaf388983f8baa8ec17cc13461909a54e.
Failure occurs before native demand loading at demand_witness line886:
`ValueError: file demand source content is not confirmed`. Full traceback is
retained in failure_observation, not truncated. Compared operands are not logged;
mutable footprint comparison is a code-level lead, NOT established historical
root cause. Do not bypass content/signature/lease/capacity guards.

ActualGPUrelease and servicepathremoval confirmed; tmux ended, fourGPUs idle.
stop_remote.sh executed ONCE:3ownedunitsinactive. collect_closure.py stopped
launcher c7256c6a and storage bf65f1d5, verified bilateral remotejournalSHAs.
First run_audit attempt closed starter79252da1 and collection1bb64e85, then
failed importing numpy in systemPython; noGPUrerun or source modification.
Originalhelper+SHA and audit_attempt1_failure.json retained. Resume verified
already-closed receipts (did NOT repeat closures), stopped failedaudit c74b1c16,
and used existing analysisPython for parsing.142CPUtests and D309/D313 reused
failure-aware audit passed; offline_execution_receipt returncodes[0,0].
Offline scope dc7b8599 closed by seal_checkpoint.py without hardkill.
Failure-evidence archive3403463B passed source/checksum/protected-file/secrets
checks;12publication paths explicitly staged. Verifier scope2e56a516 is empty;
finish_backup.py owns its final cleanup and remote-equality receipt. Read
raw/closure_backup_receipt.json before repeating any cleanup/backup step.

Failure audit:10native binding checks3999each, successful timing errors0;
N_correct/N_good remainunknown,all_correct_gate=false,ranking_eligible=false.
PhysicalGPU13431.62944322708s is failed-run observed resource, not qualifiedG1.
132remote fetches/131480060wireB,packing0;4029resource samples,
servicepeak21129539584B,minhost93283561472B,high/max/OOM/swap0.
Threeclosedhistorical main_outcome logs(D349/D353/D358) compressedlosslessly
using unchangedD339 procedure; roundtripSHAsverified,1357942784Breclaimed.
Uncompressed historical main_outcome paths now have .json.gz retained siblings;
curated provenance/contentSHA unchanged. CurrentD364 source/failure untouched.

NEXT: confirm closure_backup_receipt.json, then deterministic validation of
immutable content versus mutable physical-footprint hypothesis,
with historicalsource and fresh primary-source checks before serving changes.
No extra GPU run, figures, baseline or3B work. Mainline7B→3B→pausedbaselines.
Do not invoke the all-success D364 analyzer/contexts or repeat completedhelpers.
The raw run_audit.py now contains explicit failure-only/resume checks and has
completed; do NOT rerun it. SeeD364_COMPLETE_FAILURE_DECISION.md.

### D364 live history — terminal; retained evidence

2026-10-09 NEW evidence at sample3284: first request failure in the final phase.
Terminaljournal req_03511 has success=false, native_contract_matched=false,
error_type=ValueError. Live3524terminal/3523success/1failure. Service log reports
`file demand source content is not confirmed`, raised by demand_witness in
faaslora/memory/residency_manager.py:886. Do not remove/bypass this guard or
restart/relabel the run. No resource abort/high/max/OOM/swap event; same live
pane285600. Continue the unchanged full replay and retain all terminal evidence.
The existing success-only run_audit.py MUST NOT be executed for this outcome.
After actual release, stop_remote.sh and collect_closure.py remain applicable
if their original launch/ownership gates pass; then reuse the existing explicit
failure-aware audit path (D309/D313), without weakening all-success analyzers.
The source code compares confirmed_source including mutable footprint fields;
this is a diagnostic lead, NOT yet attribution of req_03511. Inspect final
request/evidence and prior source-allocation history before any serving change.
No formal G1/G2 eligibility is possible for this failed offered population.

2026-10-09: D364 source ed98ac3aaf388983f8baa8ec17cc13461909a54e pushed and
remote equality verified. Existing guarded runner started all4000 W0 in
tmux tc-d364-full, pane285600; do not duplicate this run. Preparation scope
4808f958 closed once by start.py. Starter scope79252da1 owns tmuxserver until
the session ends; do not close while live. Captured launcher c7256c6a identity
in rawd364/launcher_identity.json. Service scopea5ba6bb9 is owned and monitored.
Remote actual1000Mbps/full, prepublished_gzip_v1 healthy; no regenerated assets.
At watchdog sample75 serving4000 began. Latest verified sample164: same pane,
host97368481792B, peak19330891776B, high/max/OOM/swap0, disk161078657024B,
four heldGPUs, noabort. Live37complete/44arrived/0fail is partial, not a source
audit or performance result. Below150GiB while running is not below the100GiB
stop floor; never lower the next-start gate. launch.json remains pending.
Continuation verified same live pane285600 at watchdog494:361complete/366arrived,
0fail; host96141508608B, peak19678195712B, high/max/OOM/swap0,
disk160411750400B,4heldGPUs,noabort. Observed natural6scaleups/3scaledowns;
no completed source or release receipt yet. Previous goal turn made progress
(prepared, tested, pushed and launched); this turn is a verified wait on the
same live handle, not a blocker. Source plan read in full again; no config change.
During replay only lightweight monitoring and preparation of offline-only analysis
entry points; no serving changes/tests/analysis execution/compression.
Continuations verified same live pane285600 at samples530/612/674/734, then
782/851/912/979/1037/1107, then1148/1243/1304/1371 and1402/1474/1533/1593.
Further verified samples1645/1778/1871 and1916/1997/2055/2156/2219, then
2414/2490/2546, then2577/2647/2702/2763/2829/2898/2957 on same live
pane285600. Latest3211complete/3214arrived/0fail in seventhphase.
Host95169900544B, peak21129539584B,
high/max/OOM/swap0,disk158596390912B,4heldGPUs,noabort;12scaleups/9scaledowns.
Earlier asynchronous resource/log samples differed (4heldGPUs/2displayedpool);
do not infer an instantaneous difference from unmatched sample times. FinalGPU-s
uses actual allocation/release union, never displayed logical replica count.
launch.json stillpending; no sourceaudit, comparison or post-release helper
execution. Previous/current turns are verified waits, no blocker and no reason
to restart. Mainline7B remains open; no additional serving/config change.
Post-release collect_closure.py reused fromD363 and run_audit.py reused fromD358
are prepared in rawd364, NOT executed. The latter closes starter79252da1 and the
collection scope only after release. Existing analyze_d297_full/check_d299_full_context
now have drafted d364 registration, refsD358/D353/D336 and reused strict
file_demand_transaction_trigger fromanalyze_d298_smoke. Closure/reference fixtures
extended to20261009 without relaxing oldtests. These four offline-only files are
uncommitted and UNTESTED; run the bounded CPUtests only after GPUrelease, then
execute the audits. git diff --check passed; serving/model/launcher paths have
no changes, and rg found no imports of these offline analyzers in those paths.
No performance data were analyzed during GPU measurement. Executioncommit remains
ed98ac3; eventual analysiscommit must be recorded separately.
Latest continuation reread the full source plan after compaction and inspected
the offline-only diff/run_audit helper without execution. No new serving change,
tests, compression or remote operation; current source/closure still pending.
The one-time published-cache approval is already implemented and recorded;
reuse existing immutable archives, do not regenerate a second delivery pool.
Following continuation revalidated the same live handle throughout, read the
complete frozen V1 again, and inspected stop_remote.sh/collect_closure.py without
executing either. Still a verified wait, not a blocker; no restart or new run.
NEXT: monitor same run -> actual GPU release -> stop_remote.sh ONCE -> exact-owned
closure and remote journals -> failure-aware audit retaining all4000 -> backup.
Do not invoke the success-only full comparison or success-only run_audit.py.
D363 cleanup already finished; never repeat it. D364 start and raw references
are sealed in preparation_sources.tar.gz. Numerical correctness/formalG1/G2open.

### D364 preparation — passed

2026-10-09: D363 is closed/backed up at a95aac413995675106693f67cd303f7d11f1366c.
Do not repeat its cleanup. D364 reuses the candidate and D362 profiles unchanged;
only three output/cache paths differ from D363 YAML. Existing runner removes
the first100 prefix for all4000 W0. Bounded prelaunch checks, source backup and
fresh safety gate precede launch; no D364 GPU result exists yet.
Prior prefix TTFT/GPU regressions retained. Full target phase2 remains untested.
Mainline 7B -> 3B -> paused baselines; no new candidate, figures or thresholds.
Preparation passed62 native analysis tests,50 system-Python OS guards and exact
200service/24preparation class assembly. D363 closure checksums/source reverified,
protected history unchanged except the established baseline-script exception.
Startup gate: disk161510105088B,host112101826560B. Preparation archive13694B.
Scope4808f958 is empty; start.py owns its single exact-identity cleanup before a
fresh gate. Source checkpoint must be pushed before launch; no GPU run yet.
See D364_FULL_FILE_DEMAND_VALIDATION.md; next start.py then capture_launcher.py.

## D363 first100 COMPLETE and audited; retained prefix evidence

2026-10-09: D363 completed100/100/0fail usingpusheda5a2355beb8747d0efe2c4b550e998a39c165a7c.
SourceSHA4228717a06b169ce308d8241e967513dbda6424e7bb2fa6d1b1b796f42a442ff,
8388963B. ActualGPUrelease/servicepathremovalpass, tmuxended. stop_remote.sh
ranONCE,3ownedunitsinactive; bilateraljournalSHAmatch. Launcher dc1dd5cb closed
by collect_closure.py. Starter9450d59f andcollectionae7a3304 closed byrun_audit.py;
do not repeat. Offlineaudit43325a34 closed byseal_checkpoint.py, nohardkill.
Closureevidence301484B sealed,12publicationpathschecked. Seal scopee5d58541
finalclosure/remote-equality recorded inraw/closure_backup_receipt.json by
finish_backup.py; consult thatreceipt before attempting furthercleanup.
62postreleaseCPUchecks andexistingaudits passed, newfileauditcovers100requests.
35filetransactions(29Remote/6NVMe),4advancedownerepoch,max3;61nativeGPU/4nativeHOST
guards retained. Currentconstraintsrechecked, no oldcapacityreuse. Controltrigger
100receipts/4owners. Remote29fetches/50650865wireB/packing0,actual1000Mbpsfull.
Servicepeak19176263680B,hostmin93320040448B,high/max/OOM/swap0.

Prefixmean/P95TTFT909.898093/2441.886618ms; mean/P95TPOT26.233966/31.492178ms;
physicalGPU803.25265209s. VersusD357prefix: TTFT+13.79%/+36.39%,GPU-s+22.69%,
TPOT−1.44%/−6.32%. Retainregressions; no performancewin inferred. D170timing-only
98/100passes,2TTFTmiss,0TPOTmiss. NumericalN_correct/formalwarm/Resident/G1/G2open.
NoCIfromoneshortrun. Sourceheld-file/lifetime checks passed; mechanismexposure
isnotavoidedRPCcount orcausalspeedup. Diskrecoveredabove150GiBaftershutdown,
no manualdeletion. Nofigures orservingchanges.
NEXT afterseal/backup: D364 complete4000W0, unchangedcandidate/configexceptpaths,
compareD358/D353/D336; targetphase2notcontainedinprefix. Noadditionaltuning before
fullreplay. SeeD363_COMPLETE_PREFIX_DECISION.md. Mainline7B→3B→pausedbaselines.

### D363 preparation history — completed

D363 preparation passed430model/native tests and50OSguard tests, exact child
model identity,200/24profile loaders and unchangedD358controller verification.
First480test attempt had4OSfixture errors because the wholepreflightmodule ran
inmodelPython lackingpidfd_send_signal. Retainedfailure; actualwatchdogalready
uses /usr/bin/python3. Reranall430nonOS tests natively and50OS tests inactual
watchdogPython, noproductioncodeorcheckweakening. Failedscope9a8465f closed.
Preparation scope1e0ca791 is empty; start.py closes it once beforefreshlaunchgate.
Peak693661696B,high/max/OOM/swap0; archive19920B. NoGPUvalidationyet.

2026-10-09: D362 closed/backed up, noGPUorremotejoblive. Reuse its exact fresh
profiles forD363 first100, sameW0/sourceinputs, no newassets. D363 config differs
fromD358 only paths, protected_file_owner_turn_v1, identity-bound initializerIDs
and D362 profilebindings. Existing runner extended with explicitd363 identity.
No newservingchange, autoscaler/tokens/capacity/SLO/timeout remainunchanged.
Bounded native CPUtests and exact child-model/profile assembly verification
are the next launchgate; do not claim GPUvalidation until it actuallyfinishes.
Rawd363_20261009 has reused launch/health/stop/capture helpers; preparation script
seals source/testchecks beforepush. See D363_FILE_DEMAND_VALIDATION.md.
Afterqualification: first100→actualrelease→ownedcleanup→receipt/timing/fileaudit
→unchanged4000W0 againstD358/D353/D336. Nofigures.7B→3B→pausedbaselines unchanged.

## D362 calibration COMPLETE and exported; retained reference

2026-10-09: the same D362 source run completed1669/1669 native calls,
including34warmups and1635representative measurements. All200service classes
and24preparation classes covered; no complementary run needed. SourceSHA
7f53d261ed66ae4c5386237da3b4c425197dda3c18c4764e7dbfbcc3d15a457e;
execution8adb30d1390f5fa139061219b03f709cee55ea05. GPU2204.462957785s
is calibration occupancy, NOT Full4000 cost. Native contracts/identity receipts
pass; numerical N_correct and formalwarm/Resident/G1/G2 qualification remainopen.
Remote1669pairedtransfers/3406922201wireB/requestpacking0; reusedpublishedcache.
969filetransactions;191sawunrelatedownerepochadvance,max7,livechecksretained,
no oldcapacitysnapshotreuse. NativeGPU/HOST path guards retained. This verifies
mechanism exposure, not a causal performancegain. Controlinterleave triggered.
Servicepeak5453787136B,hostmin107043381248B,high/max/OOM/swap0.

Actual GPUrelease/servicepathremovalconfirmed; tmuxended. stop_remote.sh ran
ONCE,3exactownedremoteunitsinactive. run_analysis.py closed starter233a4ec
and launcher0dc1ff through collector, downloadedSHAverifiedjournals, passed50CPU
checks and all source/file/control/coverageaudits. Offline52d3e1c closed by
seal_measurement.py; seal54eca708 closed by export_profiles.py. Fresh strict
service/preparation loader exportpassed; nooldtiming relabelled, nofigures.
Measurement evidence2076337B archived, rawretained. Export scope8011d759 and
checkpoint290343df closed exact-owned, nohardkill; noGPUcompute remains.
Completed data/profiles backed up at8bcc327da2e9177db098f9ec06125fc2652b4b55;
remote branch equality verified.51archivemembers checked, usermanifestunstaged.
closure_backup_receipt.json records finalcleanup; do not repeat oldclosures.

NEXT: D363 first100 usingD362 measured profiles andD358 unchangedcontrollers,
then complete4000W0 againstD358/D353/D336. Onlycandidate remainsD361 protected
file-demandtransaction; no second servingchange/tuning before Fullvalidation.
Keepmainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities.
One-time cacheapproval alreadyrecorded inplan; do NOT regenerate weights,
traces, publishedarchives or aseconduncompressedpool.

### D362 live history — completed, retained reference

2026-10-09 live: source checkpoint8adb30d1390f5fa139061219b03f709cee55ea05
pushed and remote equality verified. Same existing collector launched in tmux
tc-d362-source (pane PID120971); do not launch a duplicate. Source1 and launch
outputs are under raw d362_20261009. Remote negotiation1000Mbps/full, published
gzip services healthy; no assets regenerated. Watchdog sample104: actual GPU0
serving, servicepeak5453787136B,hostavailable108586762240B,high/max/OOM/swap0,
disk161821388800B,noabort. First measured GPU-source calls have completed;
this is partial calibration, no Full or performance conclusion.
Live launcher identity is in launcher_identity.json (unit0dc1ff...); service
aaefd1098e0f4e60ab2c1992d92ee0d2. Starter scope233a4ec... still owns tmuxserver
PID120970. Attempted empty-scope check correctly refused closure, sent no signal;
starter_close_deferred.json records it. Do not close starter until session ends.
Preparation scope4b2ce7... closed by checkpoint;checkpoint750cb9... closed by
start.py. Do not repeat those closures. Next monitor SAME measurement→native
release→stop_remote.sh once→run_analysis.py (owns empty starter closure,
collect_closure.py, CPU tests and offline file-witness/control/class audit).
Do NOT separately run collect_closure.py or close_starter.py before that wrapper.
No tests/hash/compression during GPU. D356 analysis/seal helpers have been reused
under D362 raw paths with1669/1635/34counts and explicit file-demand audit;
they have NOT been executed yet. Source checkpoint remains8adb30d.
Latest verified same live handle at watchdog sample231: file_host wave11 calls
completing;host108204048384B,peak5453787136B,high/max/OOM/swap0,
disk161402839040B,noabort. Startup150GiB and running100GiB gates unchanged.
Continuation verified wait: same live tmux pane120971 at watchdog sample477,
365/1669 native completion log events (not a completed/calibration-qualified
result).Host108320727040B,peak5453787136B,high/max/OOM/swap0,
disk161775316992B,noabort. Full source/launch JSON closure is still pending.
This turn did not change serving code, restart services or run any tests/analysis.
Only post-release helpers were prepared from D356, and the live handle was
revalidated. Previous goal turn was progress (536tests, pushed8adb30d,launch);
this is a verified wait, not a blocker. Keep the full goal active.
Second continuation verified the same live pane120971 through watchdog sample823:
644/1669 completion log events, including NVMe wave50. Host108652412928B,
servicepeak5453787136B,high/max/OOM/swap0,disk161394970624B,noabort.
No restart, second experiment or live serving change. Full source/launch result
and release remain pending. This is another verified wait, not a blocked turn.
Third continuation: five scheduled50s checks repeatedly confirmed live pane
120971, ending at watchdog1180 with922/1669completion events. Host108460974080B,
peak5453787136B,high/max/OOM/swap0,disk161790672896B,noabort. Source/launch
completion files still pending; no duplicate launch, service mutation, analysis,
compression or other GPU job. The prior/current turns are verified waits on the
same ongoing measurement, not equivalent blockers. Keep goal active.

2026-10-09: D361 candidate remains the only serving change. Extended existing
D356/D344 preparer/runner/exporter for its exact runtime identity. Only completed
fixed-output lengths transfer via an explicit contract; old timing never moves.
Reused exact135waves/1669calls (1635representative/34warmups), existing artifacts,
trace and published delivery cache.536CPU tests passed in native environment.
At preparation checkpoint no GPU measurement had run; no performance/numerical
qualification claimed. Prepared sequence: checkpoint/push and fresh gates→D362 source calibration→strict file-witness,
control counter and class coverage audit→first100→Full4000 against D358/D353/D336.
Do not repeat D361 cleanup (allscopes closed, backup verified5f3053a).
See D362_FILE_DEMAND_PROFILE_REFRESH.md. Nofigures;mainline unchanged.

## D361 opt-in protected file-demand candidate CPU-qualified; retained reference

2026-10-09: implemented one D360-grounded candidate, defaultoff:
model.ieee_file_demand_transaction=protected_file_owner_turn_v1. Local file owner
attests its actual held/content-verified lease; native worker validates directory
incarnation and all LIVE source/pin/capacity/HOST constraints in one serialized
load/acquire turn. No unrelated globalepoch equality requirement for this explicit
immutable-file witness; strict legacy rule remains without it. Exact nativeGPU/HOST
guards and proactive paths unchanged. New receipts bind observed/liveepochs and
file witness to native lease; controller validates before releasingfile. Unknown
or cancelled outcomes keep both owners. No concurrency/timeout/scaling/SLO changes.
Tests: first71had3failures in newfixture which wrongly assumed adapter1 present;
test corrected to real unrelatedload/release, productionchecks not weakened.
417passed, then final443passed in actual nativevLLM environment and443CPU.
Native1/current difference is one explanatorycomment only: exactsources retained,
SHA andAST equality verified. Max observed testpeak632295424B; allhigh/max/OOM/
swap0. NoGPU inference,remote operations,newassets orfigures. NumericalN_correct
and formalG1/G2 remainunknown, no performancebenefit or7Bclosureclaimed.
All4testscopes closed exact-owned. Evidence1287483B/31members in
paper_results/ieee_tc/p2_backend/d361_file_demand_transaction/. Two non-destructive
archive-path failures preserved (raw results are symlinked outsideworktree);
their empty scopes closed once. Finalseal scopee24ae05014cc4101ba8ce9012d2b99e7
will be closed by checkpoint verifier; consult raw d361_20261009 receipts first.
NEXT qualification prerequisite, NOT another candidate: new field changes strict
service/preparation profile identity. Reuse D356/D344 source-measurement cases,
collector/exporter to refresh affected timing with candidate enabled; do not
delete the field from identity or relabel old timing asnew. Extend existing
length-only transfer contract for this explicit policy difference, retaining
all remaining exact checks. Then sourcecoverage/transactionaudits→first100→Full4000.
Compare D358/D353/D336, physicalGPU-s and complete timingoutcomes; D170 remains
provisional, formalwarm/Resident/correctnessopen. See D361_PROTECTED_FILE_DEMAND_TRANSACTION.md.
Mainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities; nofigures.

## D360 preparation receipt diagnosis COMPLETE; retained reference

2026-10-09: reused all4000 D353/D358 source-admission/native-acquisition receipts;
no GPU/remote run or figure. One-time delivery cache approval reconfirmed: cache
already published and D358 reports132transfers/zero request-time packing; do not
regenerate assets.61CPU tests pass. Stale rechecks593→1609, affectedrequests267→263,
max12→105; final native load mean41.888→42.492ms(1416→1403loadedrequests).
All-request admission→acquired154.966→272.560ms; acquired→engine108.973→132.087ms.
Second500phase rechecks108→1023; admission→acquired157.351→925.850ms while
final load45.670→44.183ms. All phases/tiers retained; conditional cohorts not
matched causal effects. req00615 has105rechecks and42.814s admission→acquired,
but its final transaction reuses GPU with no load: do not invent a file-load span.
Code/history and officialvLLM0.30 worker support a falsifiable globalepoch-chasing
hypothesis: nonnative file-demand commands reject unrelated worker epoch changes,
then repeat controllerRPC; cached-copy acquisitions already use exactcopy identity.
No claim that every model step advances epoch or all pre-load time is pureRPC.
NumericalN_correct/formalqualification remainunknown; noCI from one developmentrun.
Analysis scope00d6dff1b3e242f6a72ba538b197062f closed exact-owned, nohardkill;
peak962719744B,high/max/OOM/swap0. Data and39520B/9member evidence archive in
paper_results/ieee_tc/p2_backend/d360_preparation_wait/. Checkpoint verifier closes
seal7d55fcf071a74afca4b48d7684993b65; consult raw d360_20261009 receipts before
any cleanup. Detailed decision D360_RETAINED_PREPARATION_RECHECKS.md.
NEXT one opt-in worker-local atomic file-demand validation/load/acquire candidate,
with heldcontentverifiedfile witness, live owner/capacity/pin/HOSTbudget checks,
replay/cancel safety; no stale-capacity reuse or changes to cached/proactive paths.
CPU contract tests →qualified smoke →fullW0 versus retained D358/D353/D336;
noautoscaler/concurrency/timeout/threshold retuning. Serving code unchanged in D360.
Mainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities; nofigures.

## D359 retained TTFT diagnosis COMPLETE; retained reference

2026-10-09: reused D353/D358 complete4000 request/native clocks, activations and
control samples; no GPU run or remote operation. User approval of one-time
published delivery cache reconfirmed; it already exists, do not republish assets.
Existing analyzer extended with explicit D353/D358 contrast and tested strict
eight-stage join.56CPU tests pass; both runs all4000 stage identities error0.
MeanTTFTdelta+305.810531ms decomposes into release lateness−7.796119,
pre-acquire+.952123,acquire→resume+102.918227,runtime wait+50.272683,
admission→engine+140.707844,engine entry→queue+38.294134,
native queue−5.464494,prefill envelope−14.073867. Routing is nested, not additive.
Second500phase contributes77.0354555%of globalmeanTTFTincrease; all8phases and
length/tier groups retained. Phase2NVMe admission→engine567.120→2971.520ms
(111→137requests),GPUhit58.310→124.053ms(313→294); post-routing conditional,
not matched causal effects. Startup-overlap/capacity samples are contextual:
D35823waiting samples/5definite-capacity-covered,not proof allothershadfreeslots.
Do not label this a prefill regression, solely network cost, or pure RPC time.
One development run each; noCI,N_correct/formalqualificationremainunknown.
Analysis scope62aa8ccf5dc7453bbcd57ec5b8ec103d closed exact-owned by seal helper;
peak987635712B,high/max/OOM/swap0. Newdata/evidence archive66187B in
paper_results/ieee_tc/p2_backend/d359_ttft_partition/. Sealing completed;
checkpoint verifier owns closing sealaad84ade991a4b02adbb62e17bdc0349 and checking
explicit stage/checksums/protected files before push. Exact closure/remote equality
receipts are retained under raw d359_20261009; consult them before any cleanup.
NEXT reuse retained source-admission/HOST/native-load/acquire receipts to test
one hypothesis: control queue/roundtrips on nonGPU preparation prolong occupancy
of admitted-but-not-executable work during scaleout. Distinguish actual load
from waiting before changing implementation; do not repeat whole perf run or
stack capacity/autoscaler/timeout tuning. Detailed decision D359_RETAINED_TTFT_DIAGNOSIS.md.
Mainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities unchanged; nofigures.

## D358 Full4000 COMPLETE and audited; retained reference

2026-10-09: same D358 replay finished4000/4000/0failed. Do NOT rerun.
launch.json pass=true; service domain removed; native GPU contexts released;
all service/replay/watchdog returncodes0. Remote stop_remote.sh executed ONCE;
three exact-owned units inactive/ports absent; do not repeat. Bilateral journal
downloads match remote SHA. Launcher e41ce634... closed. Collection scope
c3773d21025b44cdba383307e8a82db8 closed by the offline helper, no hard kill.
Source0d6c188c302d83f57d3f581181d639c2dd37ed486b5817ce2027127ba055fe51,
300387150B; execution commit205cff857d39a70fec28e77bc40baf0afc1442ae.
GPU-s13567.637457336183; native contracts4000, numerical N_correct unknown.

Existing analyzers extended for D358 after release only;120CPU checks and all
five commands passed (full audit +D353/D336/D349 contexts). Offline scope
b0d78f826ee848b099f41a3d07d4df22 closed exact-owned, no hard kill.
Complete audit: D170 timing-only3821/4000=.95525,
TTFTmiss161/TPOTmiss20 (overlap2). Mean/P95/P99 TTFT1439.619434/3432.846578/
15336.032382ms; mean/P95TPOT31.105686/51.073428ms. Compared withD353,
decode threshold failures decline but TTFT and GPU-s worsen. Do not close7B
or claim formal qualification; D336 remains earlier service-quality context.
Full trigger4000receipts/17owners, queued and unqueued yields retained. Remote
132transfers/131480060wireB/packing0. Resource4028samples,peak23338115072B,
hostmin91898331136B,high/max/OOM/swap0. Stage identities and protected files pass.
OutputSHA differences160vsD353/215vsD336/155vsD349 retained, not N_correct.
Second500phase meanTTFT3599.418ms vsD3531714.758ms,TTFTmiss90vs53;
TPOTmiss3vs87. Short2–32output TPOTmiss18vs145,TTFTmiss76vs41.
This is a real decode/first-token trade-off; do not report as a full improvement.
U rises241.066567GPU-s vsD353 (+1.809%), drops246.697039vsD336 (-1.786%).
Complete phase/length/lifecycle comparisons and all three references sealed in
d358_full_w0/; archive71566B/27files roundtrip verified. Seal scope
311e96d24c0b481ca6acbd6cbf5eddd2 closed by backup verifier. Checkpoint checks
passed36staged paths/29checksums/27archive members, with no secrets or user
manifest staged. Verifier a83c991215ed45c994cbc447d6351897 is empty; exact
closure and remote equality receipt go to raw/closure_backup_receipt.json.

Disk recovered automatically during normal service shutdown to161978200064B,
above150GiB startup floor; no manual compression/deletion was performed. Fresh
bounded-analysis resource gate passed. No new figures,3B,baselines or retuning.
NEXT after checkpoint/push, reuse D358/D353 complete wait/stage/activation
evidence to isolate second-phase TTFT regression →one literature/history-grounded
hypothesis. No identical rerun seeking luck, threshold changes or early3B.
See D358_COMPLETE_W0_DECISION.md. Plan/V1 unchanged; formal warm/Resident/
numerical qualifications,3B,baselines,main comparisons/ablations remain OPEN.

### D358 live measurement history — completed

Live check2026-10-08 (watchdog sample1028): same tmux pane PID3985152,
commandpython3,pane_dead0;875arrived/869done/869ok/0fail,971sbusiness elapsed.
This and the preceding goal turn are verified waits on that actual live handle;
the implementation turn closed D357 and launched D358. Host92.58GiBavailable,
servicepeak19.03GiB,high/max/OOM/swap0;disk150.31GiBfree,noabort. No analysis,
tests, extra GPU work or remote changes during this live measurement. Early
startup queue remains included; current partial latency/legacy5sSLO is not a
configuration-selection or formal result. Full4000, actual release and offline
frozen-metric audit are still required. Continue monitoring the SAME run.
The original rotation500 boundary has been crossed without clearing legal caches.
If disk falls below150GiB, do not launch a new heavy task; the existing run's
100GiB safety-stop threshold is unchanged. Any needed reclaim is post-release,
limited to verified reconstructible caches, not unique measurements or assets.
Read-only history crosscheck during monitoring: D295 completed all3registered
numeric blocks (162calls); its single-request repeatability hypothesis was not
confirmed. Do not reopen PEFT/numeric repetitions without new evidence. D170
already has1536measured warm calls, but cross-model common-batch/numerical
qualification remains open. D285 is only a100request Resident pilot, not U_ref.
These existing records do not change the current W0 candidate or authorize
early3B/baseline advance. After D358, inspect second500phase and2–32token misses
alongside full physicalGPU-s and prior D353/D336 evidence, not partial live CE.

D358 launched2026-10-08 23:00:53CST after preparation/remote branch verification
at205cff857d39a70fec28e77bc40baf0afc1442ae. tmux tc-d358-full is LIVE. Do NOT
relaunch, run tests/analysis/compression, or change runtime/config/remote services
while this performance replay runs. Actual remote link again1000/full; reused
published cache, no republishing. Gate disk162857697280B (just above150GiB floor).
Launcher auxe41ce63423a445fb93aff4c1bc1184c7 identity saved in
results/ieee_tc/p2_backend_qualification/d358_20261008/launcher_identity.json.
Service scope0366bf23e9614fb7a942a3c4da25a98c; sample29host100.72GiB,
servicepeak4.12GiB,high/max/OOM/swap0, no warning/abort, still initial preparation.
Starter aced703515214c329bed949f872637e0 closed exact-owned by capture_launcher.py.
D357 audit scopee7942feb61a74a49b32048f6623ff00d and D358 preparation
scope25f8ab64f8994e3a94d210aea6830e3c closed; all D357 work is CLOSED.

NEXT monitor same4000 →actual GPU/service release →D358 stop_remote.sh ONCE
(not yet run) →reuse D353 collect_closure.py with D358 identity/path →bounded
full native/context/control audit. Extend existing analyze_d297_full.py and
check_d299_full_context.py registration AFTER release; add existing
control_interleave_trigger to Full audit, preserve per-owner/zero rows. Compare
D353 and D336 (D349 secondary), including short-output TPOT and physicalGPU-s.
Do not infer N_correct/formalSLO from timing-only pass. No figures/baselines/3B.
Remote monitor primelora-artifact-monitor-d358-full1.service, log directory
/home/lab14/primelora_remote/tc/d358_20261008_full1;7Bclock
remote-process-monotonic:cfba427c5c1547eeaa480422474ac180.

### D357 closure and D358 preparation evidence

D357 executed atd861130722b7ad1151ee8549d29fdb3a3462560a.100native contracts/
bindings passed, sourcee7a7d50c727c49b26f3d569ec1d9778d377eebd8ddb73288f83293f0d4eb6674
(7314012B). GPU-s654.7159873729106; mean/P95TTFT799.6601438790094/
1790.3715358115733ms; mean/P95TPOT26.61758728932433/33.61822651688951ms.
D170timing-only joint98/100,TTFTmiss2,TPOTmiss0. N_correct/formalSLO unknown.
Fourowners/100finalreceipts audited; per-owner yield deltas753/336/213/7,
queued206/100/40/0. Actual trigger gate passed; no causal savings inferred.
29transfers50650865wireB,packing0. Servicepeak18468663296B,
hostmin93995520000B,high/max/OOM/swap0. All localGPU/service and remote units
released; stop_remote.sh executed ONCE, do not repeat.41post-run checks passed.
Starter98eb829ced734e2aa66a6b8aae19ffdd, launcher2aca020dca3942d5a01badb0d4f1839e
and collector d358c144b67f4c9a8d41d4b4f0b2ed79 scopes closed exact-owned.
Analysis e7942feb61a74a49b32048f6623ff00d empty, to close in D358 preparation.

D358 uses the exact D357 candidate/profiles/controller, only new paths and no
100-prefix. RAW d358_20261008/7b_full_w0, config20261008_d358_7b_full_w0.yaml;
existing runner d358, tmux tc-d358-full. NOT LAUNCHED. Prepare/seal/push D357
closure and D358 preparation, then fresh resource gate →ordinary4000W0. Remote
monitor primelora-artifact-monitor-d358-full1.service. Full comparison toD353,
D349/D336 context is mandatory; no short-run extrapolation, figures or3B/baselines.
See D357_COMPLETE_D358_FULL_DECISION.md. Formal warm/Resident/numerical identity,
main comparisons/ablations/sensitivities remain OPEN. Sourceplan/V1 unchanged.

### D357 preparation history — completed

D357 Full receipt analyzer is now implemented and169CPU checks passed. Each
final registration binds request/intent, retired scheduler owner, actual TP1
worker PID and clock; snapshots are ordered per owner, single/zero-trigger
owners retained. No model-step/GPU-idle/causal savings inferred. Existing runner
and all sealed preparation refs unchanged. Additive analysis preparation archive
6579B verified, original invalid checksum retained alongside valid checksum.
Analysis test scopefe93ee8dd90a4db583322a18f37273ee closed exact-owned. Empty
analysis seal scope27d8550cb174428ea8e167758f22855a remains to close separately
via analysis_seal_scope.json; original preparation scope remains for the starter.
NEXT checkpoint/push analyzer, then start_validated.py → Full100. No GPU run yet.

D356 original+gap measurements CLOSED; all local GPU/remote services released.
Do not relaunch either source run or rerun either remote stop helper.
Gap100/100native calls passed, all six intended HOST/NVMe classes measured;
SHA554c2f6d46d6d6e78e6a60b67c8209e58f2d491d57a19d3d7f64018af3a59794.
GapU211.8763492850121GPU-s, startup40723.33782503847ms,100transfers/
357507340wireB,packing0; servicepeak5702094848B,hostmin106783211520B,
high/max/OOM/swap0. Gap control delta3154turns/1685controls/945yields/473queued.
48offline checks passed; gap measurement archive241474B verified.

Strict merged export PASSED:200/200service classes,24/24preparation classes;
1669native calls=34warmup+1635measurement,1302preparation samples.
Service SHA23e9d4e565fef1530e82e9e831642cef79c0ada396a202de63f47970f77163af;
preparation SHA61ca9ff297ab1bcdb9d6b3ca3ac76f2457614142885c475ea573d8f02172e983.
Curated d356_control_profiles/initialization/manifest.json binds two separate
source runs/clocks, no old timing fill/defaults. First offline export invocation
failed only at Python module lookup before data loading; original log/receipt
retained, -m invocation passed. Export scopeb7640a7156274ed59203ca7b9e19ff34
closed exact-owned/no PIDs/no hard kill. All D356 scopes closed.

D357 code now adds a copied control counter/PID snapshot to the existing native
register receipt only when enabled. No new RPC/polling/policy; default off and
canonical entry unchanged.128CPU+128native-env tests PASS. New Full100 config
uses D356 profiles and matches its actual child model through existing Full
assembly/strict loaders; D353 controller/soft8 unchanged. Prelaunch passed:
disk162907930624B,host112198033408B. D357 NOT launched; no performance result.
Preparation scopes cpu1(1d59f00418704d989625d68435d297f9),native1
(6f4f436eb594463cb329300b35cc5d79),prelaunch(a4783f4eb29b406bb1ef7ffaf2b8d183)
closed by exact identities during successful checkpoint; archive14045B verified.
The generated preparation.sha256 contains literal backslash-n separators, so
it is retained as invalid; preparation_validated.sha256 uses unchanged measured
file digests and passes sha256sum. Use new start_validated.py, not start.py.
Actual preparation.json/source archive are unchanged. Checkpoint aux
ce28ba1bd8664634808e3053f664ddaa complete/empty, to close via preparation_scope.json
at launch (do not pre-close it then rerun a helper that expects it live).

NEXT finish/test Full counter audit, then use start_validated.py and existing
runner d357 (tmux tc-d357-control100),100→4000W0. Preparation seal and its strict
loader checks passed; tested code/curated data are included in this checkpoint.
Fresh plan/V1/resource
checks before launch. Need actual per-owner Full trigger evidence, not D356
source counters. User cache approval already implemented: reuse existing cache,
no republishing. Mainline7B→3B→pausedbaselines. No figures; warm/Resident/numeric
qualification, all main comparisons/ablations/sensitivities remain OPEN.

### D356 six-class supplement live history (completed)

2026-10-08 22:25 CST: preparation backed/local=remote at
d8ef6a9d6ed85a1a2b3d79d553b249adc26470ed; existing runner d356-gaps launched
in tmux tc-d356-gaps. Fresh gates passed, remote actual link1000/full, both
published services healthy. Starter auxa5f420464d2647f38a57f4ca15ad58a0 may
own tmux; do not stop until session exit and exact-owned PIDs empty.
Service scopef43caadf36a44773a467ad33ab3d4065; sample16host111611330560B,
servicepeak1138294784B,high/max/OOM/swap0; still initializing at that sample.
No full-source rerun. No runtime/config changes or offline heavy work while live.

NEXT monitor same10waves/100calls → actual GPU/service release → gaps1
stop_remote.sh once → collect_closure.py and starter cleanup by exact identity.
Remote 7B clock remote-process-monotonic:d85182ed29414b6f9925d8d89e072f11;
monitor unit primelora-artifact-monitor-d356-gaps.service. Then run only bounded
offline audit and merge. Original measurement at parent remains immutable.
The preparation checkpoint had one offline hash-literal transcription failure;
original failed helper/receipt retained; corrected checkpoint_attempt2 passed,
closed that empty scope, and did not alter user files or measurements.

### Six-class supplement preparation history

Original D356 measurement backed and remote-confirmed at
835e1863f79de5b6fe365850a5bb0b91973e873d. Gap preparation/150CPU tests passed.
Existing collector/long-background selector reused for six HOST/NVMe classes:
10waves/100native calls planned (10warmup,90measurement), same C14/control policy.
Spec SHAae7e630b4e21bfa61b875e90c0ef6252912b04b34bdbca2d67858c10ae0d403e;
10existing adapters only, original500 static pool unchanged. No GPU run yet.
RAW d356_20261008/gaps1; CURATED d356_control_profiles/gaps1.
Preparation aux9fe11bf44ecd4231a51c5253f0659dcb complete/empty; close by its
preparation_scope.json during checkpoint. Helpers adapted from same source run;
remote monitor uses separate d356-gaps unit/log, published cache unchanged.
Next checkpoint/push, then gated existing runner d356-gaps in tc-d356-gaps tmux.
After measurement verify actual class coverage before compatible merge/export.
No full-source rerun, old timing fill, default fill, figures or baseline work.

### D356 original measurement closure

D356 completed125waves/1569native calls (24warmup+1545measurement),
source SHA1be8c0b963176599e4f1a3096fe250046383697675efff8be83d29ebd80b0b12.
All native calls passed their source contract. Actual service coverage194/200,
preparation24classes complete; six missing short-output/high-admission classes:
rank8/16 at HOST native pinned, HOST verified file tree, NVMe verified file tree.
No old latency/default fill; no profile export or Full launch yet.

Across252 scheduler snapshots, atomic_control_yield_v1 observed30693input turns,
26350handled controls,15422native-loop yields (10515with queued input). No
zero-yield wave. These are control counters, NOT GPU steps/idle or causal speedup.
Calibration U=2059.8417760439916GPU-s, startup89667.33829397708ms;
1569bilaterally paired remote transfers/3049414861wireB/request packing0.
Service peak5343715328B; host minimum107159056384B; high/max/OOM/swap0.
N_correct remains unknown; Full/common-SLO/formal qualification remains OPEN.

Actual local GPU/service and remote services released; stop_remote executed once.
Starter and launcher exact-owned scopes closed. Offline collection/48CPU tests/
analysis/seal all passed, analysis scope closed. Measurement archive1869663B
verified;147protected entries unchanged except known baseline script, user
manifest unchanged. Six missing classes are retained, not hidden by the seal.
Curated measurement_checkpoint.json and measurement_outputs.sha256 bind evidence.
The empty seal aux scope95e1fe9cbb4803fdc37a21f4b46e6a72 was subsequently
closed by recorded identity with no owned PIDs/no hard kill. All scopes closed.

NEXT: reuse existing long-background/short-target gap selector for only these
six HOST/NVMe classes; actual admission counts determine coverage. No full
source rerun/capacity change/output change. Strictly merge only compatible,
complete measured evidence. Then Full100→4000W0 if checks pass. Before Full,
retain interleave counters in existing native admission receipts (source
counters alone cannot prove Full trigger); pending test file is NOT yet run
and the runtime receipt change is NOT implemented. No extra RPC/polling.
Mainline7B→3B→pausedbaselines; no figures; commonwarm/Resident/numeric correctness
and all later matrices still OPEN. Published delivery cache reused, not rebuilt.

### D356 live-run history (completed; do not relaunch)

Continuation at 2026-10-08 21:52:36 CST: SAME tmux/nativePID3784253 verified;
native samples progressed through zero-indexed wave62 (125 planned),watchdog1001.
Hostavailable108351361024B,servicepeak5343715328B,high/max/OOM/swap0;
noabort/foreigncompute. No runtime/config/remote changes,tests,hashscan,
compression,newGPUrun or figures during measurement. Existing published cache
is reused under the user's reconfirmed one-time-cache approval; do not rebuild.

Offline analysis PREPARED ONLY (UNTESTED; run after actual release):
- scripts/ieee_tc/export_d298_profiles.py: strict d356 registration and native
  cumulative-counter audit. All scheduler/wave snapshots, owner+clock, monotonic
  counters and subset checks; zero-trigger waves retained. Counts do not prove
  model steps,GPU idle or causal acceleration. Strict timing loaders unchanged.
- tests/test_ieee_tc_control_profile_analysis.py: seven fixture tests for valid,
  zero-trigger,missing,foreign,reset and malformed observations; not run yet.
- d356raw/analyze.py adapted from D344,1569native/24warm/1545measurement calls,
  actual coverage+bilateraltransfer+capacity+counter checks, requestCSV and
  measurement summary. Deliberately NO immediate export (seal must come first).
- d356raw/run_analysis.py: after stop_remote.sh, bounded4GiB/auxCPU runner;
  starter closure -> collect_closure.py -> analysis tests -> analyze.py. It uses
  exclusive receipts/logs and must not be rerun wholesale after partial failure.
- d356raw/seal_measurement.py: after successful offline receipt, closes that
  exact-owned scope, retains closed evidence+SHA,checks protection and original
  preparation checksums against execution215c165,then creates measurement seal.
  Export remains separate: existing exporter main('d356') only after valid seal.
Only git diff --check has passed for these edits; no claim of passing tests.

IMPORTANT: starter scope a77ccf64dbbe4cda92ac12bde0d5c4c6 is NOT empty while
tmux lives: direct cgroup.procs read showed PID3783232. The older empty note
below was inaccurate. Never stop it before session exit and no owned PIDs;
prepared run_analysis.py checks both. Actual service/replay scopes unchanged.

2026-10-08 21:38 CST. D356 execution and verified remote V2 checkpoint
215c165fb0f78cafe73625ec596f9c331d88cb33. tmux tc-d356-source is live;
existing run_d298_capacity_source.sh d356, native single-GPU source collector.
Actual VLLM::EngineCore PID3784253 belongs to service scope
primelora-tc-svc-315fb95212c84930a90c0f9d77268ca4.scope, CPUs4–23,28–47;
GPU0 acquired. Native source samples now completing, not merely launch success.
Watchdog sample122: servicepeak5343715328B,hostavailable108612218880B,
high/max/OOM/swap0,noabort. Remote startup1000/full,unchanged published service.

RAW results/ieee_tc/p2_backend_qualification/d356_20261008;
CURATED paper_results/ieee_tc/p2_backend/d356_control_profiles.
Source outputs will be source1.json/source1_launch.json; do not read whole
growing outputs or rerun collector. launcher_identity.json captured actual aux
52ed60dd78294bb7b2025137bd0a6e0b / InvocationID69c2648aa8134b12932fe97cfb535ee5.
Starter scope a77ccf64dbbe4cda92ac12bde0d5c4c6 is completed/empty but not yet
closed; exact identity in launch_gate.json. All previous preparation scopes closed.
No live tool handle; tmux/watchdog own ongoing run. Preparation checksum binds
this ledger's PRELAUNCH version at215c165, not subsequent live status updates.

NEXT monitor existing run → verify actual local GPU/service release → prepared
stop_remote.sh ONCE → collect_closure.py (exact launcher identity, bilateral logs).
If no transfer on failure, stop helper final grep may return1; verify stopped
units and absent matching clock independently, do not rerun the whole script.
Close completed starter scope by recorded identity after confirming no owned PIDs.
Then adapt/reuse D344 analyze.py and existing exporter for d356 (not yet adapted),
CPU-test offline changes, verify actual200service/24preparation class coverage,
native counts/binding, interleave counters in scheduler snapshots, profiles and
resource/transfer contracts before export. Old latency profiles remain rejected.
If coverage missing, report actual missing classes and supplement only those.
After valid export: new exclusive Full100 then4000W0; original D355attempt retained.
No G1/G2/full/numeric qualification from calibration. Mainline7B→3B→baselines;
all commonwarm/Resident/N_correct and remaining matrices OPEN. No figures.

### D355 failure and D356 preparation history

D355 execution a9986c3e591d01ea295445aee78a5b609c45db63 was backed then launched.
Full100 stopped before model creation/arrivals: strict service model identity
rejects the added control policy.0 terminals/owner events,0 observed GPU-s;
not a valid performance point or evidence against the optimization. All local
GPU/service paths released; exact-owned remote units inactive. Stop helper rc1
was its final grep finding no transfer file, not failed service stop; independent
clock-ID scan confirms no transfers. Failure closure archive204532B verified.
See d355_control_interleave/failed_smoke.json. Do NOT rerun this attempt/stop.

Decision: preserve strict timing-profile identity. D356 reuses original D344
measurement cases (125waves/1569calls incl24warmups), refreshes affected timings
under SAME C14 plus atomic_control_yield_v1. Explicit length-only transfer may
reuse native completed fixed-output lengths; no old latency/SLO transfer.
127 CPU tests and preparation passed before the above launch. New profile
must validate before another exclusive Full100/full4000. D353 controller soft8
unchanged; no extra candidate/capacity sweep.7B→3B→pausedbaselines remains order.

### D355 qualification history (before the failed entry)

D354 backup local/remote bbdb21f428a77d306461b74cd27a162531fe77d8 confirmed.
D355 adds opt-in model.ieee_control_interleave=atomic_control_yield_v1:
after a complete Prime control call, return to native loop when it has work.
FIFO/native handler, atomic pending/GPU/HOST transactions and nine equations
remain; no timer tuning, extra threads, recursive steps or stale-state cache.
Version/AST/topology/owner guards; default off. D353 config/profile fixed apart
from opt-in and isolated output/cache paths. Snapshot counters record trigger.

110 tests pass in CPU env and110 in installed native env; actual native handler/
step API also passes with synthetic executor/work and NO CUDA initialization.
Initial109-test failure retained with source archive: Python AST empty-field
normalization and invalid-config fail-fast path repaired, not a GPU retry.
All4 test scopes closed;preparation scope identity in d355raw/preparation_scope.
Preparation archive290292B validated;147-entry protection checked,only recorded
historical baseline replay change;user manifest unchanged/notstaged.

Next: backup tested checkpoint,close empty owned preparation scope, fresh gates,
start unchanged published remote service, first100 via existing runner d355.
Require100/100 native contract/binding, trigger counters and actual GPU cleanup;
then ordinary full W0 if safe. No SLO/GPU-s qualification from a100-prefix.
Do not run two heavy tasks or advance3B/baselines. Formal warm/Resident/N_correct
and all remaining matrices OPEN;no figures. Currentbest-state is not declared.
Doc D355_ATOMIC_CONTROL_INTERLEAVE.md;raw d355_20261008;curated
paper_results/ieee_tc/p2_backend/d355_control_interleave/preparation.json.

### D354 retained input-turn evidence (completed)

D353 backup confirmed local=remoteV2 d6714fdf3b539f676c4a212c520eec21c9e4d6a1;
all prior scopes closed. User reconfirms immutable published delivery cache;
reuse existing cache, no request packing or new artifact pool.

D354 reuses all12 complete D351 native streams/1000 decode windows with SHA
and exact prior interval checks. 50613 input turns;8533 nonempty work-entry turns,
2228 multi-client and2085 multi-utility. Work-entry client339.385908s;
after-first-utility client117.460968s (subset, not extra latency). Short TPOT
misses104/108 overlap that suffix,mean399.549100ms versus46.638575ms for204
passing short outputs. Descriptive intrusive pre-D352 evidence, not D353 causal
attribution, GPU idle or projected speedup.18tests pass;analysis peak259985408B,
high/max/OOM/swap0. No serving/config/metric change, GPU run or figures.

Decision D354_CONTROL_TURN_EVIDENCE.md: supports ONE next candidate, bounded
interleaving of complete Prime control transactions with ordinary native model
steps. Preserve FIFO, idle/abort/shutdown, atomic pending/admission/reference,
native topology and nine equations. Candidate NOT IMPLEMENTED/QUALIFIED.
First correctness/minimal validation, then ordinary Full W0 against D353;
include control delay, TTFT/TPOT, joint timing and physical U trade-offs.
Do not implement cross-request HOST cache or blindly sweep capacity/softtarget.

Curated paper_results/ieee_tc/p2_backend/d354_input_turns/analysis1;
raw results/ieee_tc/p2_backend_qualification/d354_20261008.
Analysis complete. Initialseal rejected original manifest's empty-file selfSHA;
six data outputs/all sources intact. Preserve original/failure;use separate
analysis1/validated_manifest.json excluding self, no reanalysis/GPU rerun.
Outer closure had the same empty-file selfSHA issue;validated_closure.json
preserves original closure and checks every non-self entry. No data rewrite.
Both prior failures retained. All owned scopes closed;backup completion is in
raw/closure_backup_receipt.json. Use validated indexes, not original self entries.
Mainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities,allOPEN.
D353 provisional3752/4000 stillnotqualified;formal warm/Resident/N_correct OPEN.

### D353 completed checkpoint (historical)

Checkpoint verification passed:31exactstagedpaths,25filehashes,46archivemembers,
syntax/diff/secrets checks and147-entry historical protection. User manifest
unchanged/notstaged. Backup completion/remoteSHA must be read from
d353raw/closure_backup_receipt.json before repeating any commit/push work.
Latest completed verifier scope identity is in closure_checkpoint_verification;
the backup closer must verify it is owned/empty before stopping it.

2026-10-08 20:33 CST: all offline stages complete. Native4000/4000,
all10binding checks4000 each,stage identity error0;N_correct/formalG1G2 OPEN.
U13326.570889990195GPU-s;mean/P95/P99TTFT1133.808903/2967.689615/
6632.252923ms;mean/P95TPOT36.672525/75.301481ms. Unchanged D170timing-only
3752/4000=93.8%,48short of3800. D349comparison:latencyimproves butU+17.779%;
D336comparison:U−3.531%butjoint−2.125pp. NOTqualified;do notadvance3B/baselines.
2584eligible transaction-local HOSTpostchecks/1416full,misclassified0.
132remote transfers,131480060wireB,packing0;allownedremote/local servicesclosed.
4030resourcesamples,servicepeak22306799616B,high/max/OOM/swap0.
OutputSHAchanges167vsD349,247vsD336retained,notarithmeticqualification.

Initialaudit100tests+Full+contextD349passed;D336function-levelregistration
guard failed beforewriting output. Originalsource retainedin
d353raw/check_d299_full_context.audit1.py,provenanceinanalysis_source_retention.
Fixed sharedregistered-reference whitelist;103tests+remainingD336contextpassed.
NoGPurerun,statisticalchange,outputoverwriteorhiddenfailure. Firstreceipt[0,0,0,1]
andresume[0,0]retained. Allcollection/analysis scopesclosed. Latestseal scope
identityinclosure_seal_scope.json,emptyandpendingcheckpointverifierclosure.
46members/78481Bclosurearchive verified;phase/length/lifecycle tablescomplete.
Curatedpaper_results/ieee_tc/p2_backend/d353_full_w0/;decisiondocument
D353_COMPLETE_W0_DECISION.md. SourceSHA0fe856cf44923ec66fb91aed5ed9d088891bedb7c2b603d416e78b04f0751c9d,
298685749B. Sixhistorical closedlogs compressedlosslessly,3399909376Breclaimed;
storage receipts retained,150GiBstartgateunchanged. Nofigures/newGPUrun.

Next: confirmclosure_backup_receipt(remoteV2),thenreuseD353
remainingphase/short-decodeandD351nativeCPU evidence forONEnewhypothesis.
Do not repeat D353/analysis,blindcapacity/softtarget sweeps or oldnumericalblocks.
Mainline7B→3B→pausedbaselines→M1/M2→ablations/sensitivities,allOPEN.

### D353 completed-run closure history

2026-10-08 20:18 CST closure verified:4000/4000 terminal/native-contract
complete,failed0,measurement_complete=true,open_lease_ids=[];N_correct remains
null. Physical U=13326.570889990195 GPU-s, not yet a qualified G1 result.
launch.json pass/native_gpu_context_release_confirmed/service_path_removed=true;
service/replay/watchdog returncodes0, nvidia-smi compute list empty, tmux exited.
All3 owned remote services were then stopped with the prepared stop_remote.sh;
remote_stop.log has3 inactive/PID0 confirmations. Do not rerun launch or stop.
The remote transfer/monitor logs have NOT yet been collected; Full latency,
SLO, output/binding and actual HOST-postcheck trigger audit remain pending.

Continue the prepared order: bounded lossless closed-log compression to restore
150GiB startup headroom (current free160566366208B), collect_closure.py,
then run_audit.py (pending tests and existing full analyzers). Do not rerun
whole helpers after partial completion; preserve exclusive-create receipts.
HEAD remains06ca566; analyzer edits remain untested. Mainline7B→3B→baselines,
no figures, unchanged V1/D170 development ruler, formal qualification OPEN.

### D353 monitoring/preparation history (completed run; not live)

Latest verified wait, 2026-10-08 19:50:08 CST: same tc-d353-full session
confirmed live repeatedly; sample2449, 2686/2686 successful and native-contract
matched terminal records, failed0 (not numerical qualification). Host available
92.173GiB, high/max/OOM0, no warning/abort; disk148.019GiB, above100GiB running
floor. Service invocation was independently rechecked unchanged at19:34:03.
No serving/remote/configuration changes, tests, full analysis, compression,
figures or new experiment in this continuation. Plan/V1 SHA remain
0c808509.../5f0732ef...; the just-read full contracts are unchanged. Existing
offline analyzer edits below remain UNTESTED, pending actual run closure and
restoration of150GiB startup headroom. Do not restart the live full replay.

Post-run storage recovery is PREPARED, NOT EXECUTED:
d353raw/storage/compress_historical.py reuses the unchanged D339 guarded
procedure via AST replacement of TARGETS only. Six closed raw main_outcome
logs (D314,D315,D318,D332,D335,D346), total3854041079B before compression,
have source SHA references in their retained curated manifests. Only lightweight
stat/manifest checks were done while running; full SHA/open-file/protected-result
checks and two decompression roundtrips must run in bounded3/4GiB aux AFTER GPU
release. D336/D349 reference outcomes and all raw result JSONs are untouched.
Use this existing prepared helper rather than re-searching/rebuilding cleanup.
The helper must refuse when GPU processes remain, and records exact reclamation;
recheck actual disk gate after completion, never assume an estimated saving.

2026-10-08 19:22:09 CST continuation: same live tmux tc-d353-full, sample793,
592 terminal records. Host available103445147648B, servicepeak18989367296B,
high/max/OOM/swap0, no warning/abort/foreign compute. Disk160229117952B;
the post-run disk-headroom recovery below remains required. No serving,
configuration, remote-service or measurement changes; no new GPU run.

Prepared ONLY offline D353 registration in existing analyze_d297_full.py and
check_d299_full_context.py. It requires the post-release collection receipt,
matching source path/bytes/SHA and local/remote closure, retains all existing
4000-request/binding/timing audits, adds the existing hit_host_transaction_trigger,
and registers D349 comparison plus separately labeled D336 historical context.
Extended tests/test_ieee_tc_hit_host_analysis.py with four closure/source-selection
cases; existing five trigger cases remain. These edits pass git diff --check
ONLY: tests/imports/full analysis are NOT YET RUN and must wait until measurement
ends and resource gates pass. Do not commit them as tested yet. Running source
HEAD remains06ca566; only these offline files and this status are our dirty edits;
the generated manifest remains the user's untouched edit.

After actual closure and collection, first run the pending analysis tests, then
analyze_d297_full --campaign d353, then check_d299_full_context --campaign d353
and its --reference d336 variant. This is now PREPARED in d353raw/run_audit.py,
adapted from D349's bounded wrapper and completion_receipt schema. It requires
actual collection identity/source SHA, physical release, remote closure,4000
native contracts and the unchanged resource gate. It closes only the exact-owned
empty collection scope and records all four command results. NOT EXECUTED or
import-tested yet; no analyzer has consumed live data. Use the prepared helper,
not D349's hardcoded collection identity. A partial audit failure must retain its
receipt/output and resume only the remaining step with a new log; do not rerun
the entire wrapper and overwrite existing exclusive-create artifacts.

Monitoring continuation, 2026-10-08 19:16:11 CST: SAME tmux tc-d353-full
is live, 307 request terminal records at sample440. This is progress, not a
completed performance result. Host available103428857856B, service
peak18989367296B, memory high/max/OOM/swap0, no warning/abort or foreign
compute. Disk free160378716160B is below the150GiB new-heavy-job gate but
above the100GiB running floor. Continue this run unchanged; after its actual
closure, recover headroom using already authorized lossless closed-log
compression before collection/analysis requiring the startup gate. Do not
compress, scan full archives or change remote services during measurement.
Plan and frozenV1 were reread in full. This continuation made no serving,
configuration, analyzer, remote or GPU-run changes, and ran no heavy tests.
No completed experiment should be repeated; D353 remains the sole live run.

2026-10-08 19:08:44CST launched tmux tc-d353-full. Do NOT relaunch.
ExecutionHEAD=independentlyverifiedremoteV2=
06ca566eb96eb1b619b7c4fd71d0db5da3d2c62b. D352 analysis/closure and D353
preparation backed in17exactpaths. Firstpush had transientTLSfailure; ordinary
retry succeeded withoutweakeningTLS,changingcommit orrerunningexperiment.
Receipts in d353raw/initial_push_failure.json and backup_receipt.json.
Raw: results/ieee_tc/p2_backend_qualification/d353_20261008/7b_full_w0.
Config: configs/ieee_tc/20261008_d353_7b_full_w0.yaml.
SameD349 C14/profile/softtarget configuration exceptthreeisolated output/cache
paths; onlyD352 HOST postcheck candidate, noobserver, noSLO/formula change.
All4000originalW0requests, ordinaryopenloop60snotice. No prefix extrapolation.

Serviceunit primelora-tc-svc-ee76b1c3d7f44d138b7894294eacc958.scope,
invocationcebede387f2b47ffa41099a52baf2f4f,inode20733583.
Launcherunit primelora-tc-aux-42c3457a9f38445d85dc0da636ecd640.scope,
invocation9c95e48699134426bf65aa9718bae729,inode20733541.
Exactlauncheridentity captured in d353raw/launcher_identity.json. D353 setup01–03
and allD352scopesclosed. tmux launchedoutside thetemporarysetupscope this time;
actual replay/watchdog stayincommon4GiBaux, service72/80GiBswap2GiB.
No liveexec/toolsession; tmux+independentwatchdog ownrun. At sample23, host
available109589159936B,servicepeak3050856448B,high/max/OOM/swap/warning0.
Freshstartdisk161469927424B;150GiBstartand100GiBrunningfloorunchanged.
All3ownedremote services active, NIC1000/full, 7Bclock
remote-process-monotonic:0766b356266b4c48b3d370a611e2ded8.
Reuseimmutablepublishedcache; no requestpacking, no remoteinterference allowed.

Next: monitor currentrun only; after launch.json confirmsactualGPUrelease and
servicepathremoved, execute d353raw/stop_remote.sh, then collect_closure.py in
boundedauxscope. Test the prepared existingfullanalyzer D353 extension, require4000nativecontracts,
fullbinding/leaseclosure,62-styleeligiblepostcheckcounts (notfixed62threshold).
ComparefullD349 andD336 withunchangedV1/D170provisionalruler; keepnumerical
N_correctnull untilqualified. No extraoptimization or secondminimalrun now.
Ifdiskbelow150GiBafterFull, firstreuseauthorizedlosslesscompression onclosed
nonprotectedrawoutcome afterSHA/open-file checks; never lowergate/deleteinputs.
No figures,3B orbaseline advance; G1/G2/numerical/commonreferencesremainOPEN.

### D352 completed result and D353 preparation history

D352 first100 finished, native-contract/binding100/100, timing-only98/100 under
unchanged provisionalD170 ruler; mean/P95TTFT876.166/2592.862ms, mean/P95TPOT
28.035/34.319ms. U798.5166242971318GPU-s is prefixONLY, notFullperformance.
62eligible same-transaction HOST postchecks,38fresh-inventory postchecks,
zeroincorrecttrigger. SelecteddispatchGPU61 versusacquisitionconfirmedGPU62:
eligibility is actualserializedacquisition, notpreviousroutingtier. 4leasesclosed,
GPUrelease/servicepathremovalverified, all3remote servicesinactive. 29transfers,
50650865wireB, no requestpacking. Servicepeak18934951936B;264samples;high/max/
OOM/swap0. 23analysis tests passed; analysispeak103325696B. No figures.
SourceSHA e7f8ff5881988698c6b5dca0cccbd3f433d0d86fb289ed9646c82fb3409a0c84,
8129393B. Audit smoke1complete; fouroutputdigestdifferences fromD345 retained,
numericalcauseunknown. D345controls differ, onlydescriptivecontext. N_correctnull.
Allsetup/launcher/collectionscopesclosedexcept currentcompletedanalysis08,
identityin offline_execution_receipt.json. Do notrerunD352oranalysis.

D353 config and existingrunner case prepared: D349 identical exceptisolated
output/NVMe/HOST paths, ordinaryFull4000,noheavyobserver. Requiresanalysis backup,
source/config/preflightgate and remote startup beforelaunch; NOT launchedyet.
Next exact-pathbackup plusfullrun; compareD349 andD336 usingfrozenV1, noSLOchange,
noextraoptimization. Mainline7B→3B→baselines; formalG1/G2/commonreferenceOPEN.

### D352 launch history (completed)

2026-10-08 18:57CST minimal100 LIVE in tmux tc-d352-hit100. Do not relaunch.
Execution HEAD=remoteV2=3d232cf9790be0702ebbbdef4abbc2be8620cf48;
11 exact files backed after618 CPU tests and25 native-environment tests.
Native aux peak619245568B, no high/max/OOM/swap. Source/profile/protection
binding and fresh preflight passed; all147 historical entries unchanged except
the previously recorded baseline script. User manifest untouched.
Raw: results/ieee_tc/p2_backend_qualification/d352_20261008.
Service: primelora-tc-svc-1fc479be6e60412dafde070603d266bb.scope,
invocation ec23b4d024c74b07bc0239add204d5a0, inode20728154.
Launcher: primelora-tc-aux-5086875d39764353ad5d249cc12f8c14.scope,
exact identity in launcher_identity.json. No exec/tool cell remains live.
Setup01–05closed; setup06 contains the tmux serverPID3410239, so the cleanup
guard correctly refused to stop it. Leave it until tmux exits; never kill live
launcher/service. This 1.1MiB control-process scope is recorded separately;
actual replay/watchdog remain in the common4GiB auxiliary scope. No service
resource settings changed. capture_launcher.py failed only its empty-setup
assertion AFTER successfully saving launcher_identity.json; do not rerun it.
All3 owned remote services active, 7B clock remote-process-monotonic:
ee4328c3e0144e68b119b15ee8e79de3. RemoteNIC1000/full, immutable published
cache reused. At sample62 servicepeak7073648640B, no resource events/warnings.
After run: requirelaunch.json nativeGPUrelease/servicepathremoved, then use
stop_remote.sh, collect/validate100/100 and postcheck triggers, close exact-owned
launcher/setup scopes. No performance inference from this prefix. OrdinaryFull
4000 remains next; N_correct/formalG1/G2 remainOPEN. No figures/3B/baselines.

### D352 preparation history

D351 is fully analyzed/backed at 6e6f5be0262c7ae2ebe4dbc5fad442dfb08ebbd4;
do not repeat its run, analysis or archival. D352 implements only one candidate:
on a confirmed GPU-hit acquisition, reuse the current transaction's freshly
measured HOST storage totals for the postcheck, but re-read allocator counters
and re-evaluate workspace. No cross-operation cache, source/admission bypass,
formula/config/SLO change or fence removal. Cold/HOST/unconfirmed paths retain
both walks. Every new request retains its own full physical inventory.

618 CPU tests passed in attempt3 after preserving two earlier failure logs:
unavailable-counter ordering regression corrected; old shutdown fixture supplied
missing engine field (production shutdown unchanged); new worker test's explicit
v0.30 import fixture corrected for CPU env. Tests include actual callback/workspace
and D329 freshness counterexamples. Max observed aux peak528838656B, no high/max/
OOM/swap events. Native-environment test and preparation binding follow. No GPU
run or remote start yet. Code is NOT backed yet; do not launch before checkpoint.

Next: native CPU check, source/config/protection/preflight binding, exact-path
commit/push, then unchanged D349 first100 diagnostic using D352 isolated paths.
Require native contract/closure and eligible postcheck trigger; then ordinary
Full4000 comparison, no performance figures. Mainline remains7B→3B→baselines;
N_correct and formalG1/G2/common references remain OPEN.

### D351 diagnostic COMPLETE/ANALYZED/SEALED

18exactpaths/25archive members validated; standardCSV CRLF triggered default
gitwhitespacecheck,then command-local cr-at-eol checkpassed withALLresultbytes
andSHAunchanged. NoGitglobal/configchange. Failedcheckretainedinraw.
Backup completion/hash is recorded in d351raw/analysis_backup_receipt.json;
read that receipt before repeating any checkpoint work. All analysis isDONE.

2026-10-08 18:20CST finished the SAME first1000 run; never relaunch D351.
All1000 native-contract requests completed, all1000 decode windows bound to
complete core observations;12closed core processes,171284hostspans,notruncation.
12physicalleases closed; actualGPUrelease and servicepathremoved proven.
Source70478940B,SHA df53608d95fc85ffeca113a4ff2d8ac8ad85bfd1df64e9b9c5d00fe15ef1eb3c.
U3297.097114621196 GPU-s is instrumented prefix,NOTformal/full4000 performance.
Mean/P95TTFT1922.098/10327.093ms,mean/P95TPOT46.742/116.659ms;
N_correct remainsnull. Do notcompare this observerrun asunmodifiedFullspeedup.
All3ownedremote services stopped,portsabsent,60transfers74953014wireB,packing0,
downloadSHAmatch. Launcher andcollection08/analysis09/storage10 nowclosed;
operation11 andfirstseal12 areclosed;seal13 finished,identityin
analysis_checkpoint.json,closedbythebackuphelperafterownershipcheck.
No liveGPUjob,toolcellor terminalsession.106analysistestspass;peak509108224B,
servicepeak21160337408B,1191samples,high/max/OOM/swap/warningsall0.

Short2–32 TPOT-miss108requests:mean decode1359.664ms/client569.394ms(41.88%),
step760.867ms. Nonmiss204:515.375/119.584/386.721ms. Notcausalpairedgroups.
Work-entrycontrol:source_snapshot767calls/mean119.553ms/91.697host-s;
pending_gpu_reference558/131.201ms/73.210s;
demand_load504/106.739ms/53.796s;routing553/70.370ms/38.915s.
ThreadCPU/host98.29–99.82%fortheseoperations:actualCPUcontrolcost,notsimply
GPUwait/remote. HostspanNOTGPUkernel;asyncGPUcanoverlap. Observer48.551s
includeswrite39.647s;nooverheadsubtraction. Dataanalysis1+operation_detail1.

NewhypothesisafterhistoryD329/D330andcurrentofficialcore/worker/PyTorchreview:
withinONEserializedconfirmedGPU-hittransaction, test whether alreadymeasured
HOSTphysicalgraphcanbereusedacrossbefore/aftercheckswhilenativeHOSTobjects
cannotmutateandallocatorcountersremainfresh. NOTepoch/TTL/model-IDcache;
noequation/budget/safetyrelaxation,noRPCreorder/fenceremoval. Proveallcallback/
asyncpreparationmutationboundarieswithexistingfixturesbeforeimplementation.
ThiscandidateisNOTimplemented; donotrepeatD351orcapacity/softtarget512scans.
Thenboundedminimalvalidation(max2withnewevidence)andordinaryfull4000replay.

Diskfellbelow150GiBaftercollection;reusedD339losslessprocedureonclosedD341
main_outcomeonly,two roundtripSHAsverified,noactivefile,freed490188800B.
ExactrestorepathandSHAinrawstorage/summary.json. Noinput/historyloss;fresh
150GiBstartgate stillmandatory. Analysis/documentation/archivalonlythisturn,
noservingpolicychange,figures,3B orbaselines. G1/G2/numerical/commonreferences
remainOPEN;goalACTIVE. Verifybackupreceipt,thenonecandidateabove.

### D351 launch history (completed; below is the original live snapshot)

2026-10-08 18:00CST launched in tmux tc-d351-core-prefix1. Do NOT relaunch.
Execution HEAD=independently verified remoteV2=
15e66c9b05b4c5ee8998f219206886c37a00f810.13exactpaths,11archive members
backed after202CPUtests/secrets/syntax/diff/protection/source-hash checks.
Raw results/ieee_tc/p2_backend_qualification/d351_20261008/7b_full_w0_prefix1.
Service unit primelora-tc-svc-29b7437f29f2433bae86ec9e605aa673.scope,
invocation85ad122bedd74ae985f50150a68f579c,inode20700585.
Launcher unit primelora-tc-aux-091aadfbe85645f9a543f3dd328948b5.scope,
invocation14bab6604c8d495eafbc2f2e8c192f09,inode20700543;
exact identity in launcher_identity.json. No live tool/exec cell; tmux/watchdog
own the running job. Setup scopes01–07 closed, service/launcher untouched.
All4 initial native core processes emit diagnostic_core/*.jsonl. At watchdog
sample130:host available91.45GiB,servicepeak17.93GiB,high/max/OOM/swap0,
no warning/abort. First requests complete with native token contract; full
1000 completion/correctness/timing conclusion NOT yet established.
systemd inotify-watch warning recurred (known fromD307); actual cgroup limits
and independent polling watchdog are readable/active, no global changes made.
Remote three owned services active; NIC1000/full. Existing immutable gzip
cache reused, no request packing/rebuild. Start/health/stop helpers in d351raw;
remote output /home/lab14/primelora_remote/tc/d351_20261008_prefix1.
Health helper initially ran under systemPython missingnumpy; reran unchanged
health script using existing qualified CPUenv, succeeded without restart.
Remote invocations and clockID retained in remote_start.log/remote_health.json.
Do not adjust or inspect remote during measured run. Completion: wait for
launch.json cleanup proof, stop owned remote with stop_remote.sh, retain logs,
verify1000/1000 native contract and observer completeness/overhead, analyze
input/client/step time allocation before choosing one optimization. Do not
reuse heavy numerical observer or call these host spans GPU kernel time.
This turn PROGRESS: implemented/tested/backed observer and actually launched
the bounded experiment. No figures, no3B/baselines; G1/G2 remainOPEN.

### D351 native-core host timing preparation

7B mainline, D350 hypothesis only. New opt-in host spans cover input queue,
client/allowlisted utility, engine step and schedule token allocations; no CUDA
readback, prompt/RPC payload logging or serving policy change. Default does not
install hooks. Existing D328 wrapper/D293 runner reused, D349 C14 configuration
unchanged except three isolated output/cache paths; original W0 first1000 only.
Formal comparisons forbidden for this observer. N_correct/G1/G2 remain OPEN.
Initial49 and final202CPUtests pass; resources high/max/OOM/swap0. Actual native
source hashes, unchanged profile/config binding,147-entry history protection,
frozen Plan/V1 and user manifest checked. Disk161346510848B; fresh heavy gate
still required, do not relax150GiB. No GPU launched, no remote service started.
Preparation AST check initially rejected inherited shutdown; corrected against
actual EngineCore inheritance, plus a log-marker escape check. Failure evidence
and original preparation script retained; no inference/results affected.
Next exact-path checkpoint/push, fresh gate, start frozen remote service and
one bounded first1000 diagnostic; validate/cleanup/analyze before any candidate.
No new capacity/softtarget scans, no3B/baselines/figures. d351_core_timing/prelaunch.json.

### D350 retained diagnosis COMPLETE/SEALED/BACKED

LocalHEAD=independently checked remoteV2=
45bade67ae2240cbac7da82d894af20917adaae8.23exactpaths,18curatedhashes,
10archive members verified; source/output SHA,66CPUtests,secrets,syntax,diff,
147-entry history protection checked. Known baseline source difference only;
user lora_manifest remains unchanged/unstaged. Analysis01/control02/seal03
all closed, no liveGPU/toolcell/execsession. Remote services were not started
or changed. Backup receipt retained at d350_20261008/backup_receipt.json.
Postbackup disk161348571136B. This turn PROGRESS: new retained-data diagnosis,
all-phase/control join, bounded next hypothesis, tested exact-path checkpoint.
Do not repeat D350 analysis/closure/backup. Next native-core timing evidence.

D350 adds zero GPU runs. Reused complete D336/D346/D349, frozen Plan/V1,
D337/D347 interval projections, same input/arrival identities4000each.
66 CPUtests pass. Analysis1 peak687878144B; high/max/OOM/swap0; scope01
closed. Control-context scope02 and checkpoint03 finished/closed with ownership
and resource receipts. No running GPU job or live tool session.
Existing analyzer minimally extends registered D336/D349 contrast; no serving
code, equations, warm threshold, workload, artifact or remote change.

D349 native request-associated loads1975→1408 vsD336 (−28.71%), butTPOT
misses80→239. Peer-prefill-overlap requests2288/76miss→2873/234miss;
short2–32 overlap449/74miss→560/191miss; withoutprefill665/3→554/5.
Native other-request overlap1.394933→3.071052, NOTkernelbatch. Conditional
groups overlap/differ; no causal subtraction or statistical CI.
Phase2 meanarrival→admission646.961→2125.243ms; maxsampledqueue29.
At1047.085s scale-down then1051.152s scale-up; nextready1096.373s.
Do not tune cooldown to these observed timestamps. All8phases retained.

Control export original manifest accidentally hashed itself while empty.
Original preserved; validated_manifest.json verifies unchanged fourtables and
removes only self-reference. First repair helper lacked Python3.10file_digest;
failed before writing, corrected streamedSHA. No performance data regenerated.
Doc: D350_DECOUPLED_CONTROL_DIAGNOSIS.md. Data:d350_decoupled_timing.

NEXT one bounded native-core scheduling/utility timeline hypothesis: burst
utility work may delay generation steps, confounded with peer-prefill envelope.
Read historical D328 and current official vLLM0.30core/UniProc/scheduler;
reuse existing diagnostic attachment/runner and original1000requestprefix.
First check available same-boundary evidence; add only absent bounded timing
observation if needed. No synchronous GPU readback, no numerical-observer timing
reuse, no alteration to utility order/scheduler/fences before evidence. Do not
repeat D350 or softtarget/capacity/fixed512 scans; no3B/baselines/figures yet.
Disk161351729152B, only~0.27GiB above150GiB start floor; fresh gate and
verified rebuildable-cache cleanup required before nextheavy if growth needs it.
G1/G2,numerical correctness,finalcommonreferences remain OPEN. GoalACTIVE.

### D349 complete4000 W0 AUDITED/SEALED/BACKED (preceding checkpoint)

Checkpoint localHEAD=independently verified remoteV2=
242eb5dfe1ba1bb170cfdacdaa4ed14dbc4f5ea6.26exactpaths,21curatedhashes,
23archive members passed staged-content/secrets/syntax/diff/protection checks.
All D349 localservice/launcher/collection05/audit06/seal07/verifier08 scopes
closed; all three ownedremote services inactive. No liveGPUjob, toolcell or
execsession remains. closure_backup_receipt.json preserves independent check.
Postbackupdisk161370935296B (only slightly above150GiB); recheckbefore any
nextheavy task and clean only verified rebuildable resources if required.
This turn made PROGRESS: full replay, all audits, comparisons, decision and
remote backup complete. Do not redo D349 closure or backup.

All4000 requests completed with native contract counts4000 and zero request
failures. Launch pass=true; actual nativeGPUrelease confirmed; service path
removed; all17 physical leases closed. U=11314.882957193302GPU-s;
N_correct remains null, no formal G1/G2 acceptance. Source SHA
c289b97bd146ae8ec355f59ea5abd373b271e9848814ea3b60ddff58cfdd97e7,
287707308B. Do NOT repeat inference. All three owned remote services stopped,
ports absent; remote transfer/monitor downloads SHA-match. Launcher and
collection05 scopes closed. Runtime serving code was NOT changed.

Existing analyzers extended with strict D349 identity; both analyzers return0,
90 CPUtests pass. Audit06 scope closed. Seal07 finished successfully (23archive
members/60161B) and was closed by exact-owned checkpoint cleanup. No live
exec session or GPU job. All-phase/length and D336/D346 comparisons saved.
Timing-only3668/4000=91.70%,132short of3800. D3463655, D3363837.
MeanTTFT1349.357788ms/P953248.721206ms/P9912351.693921ms;
meanTPOT38.486231ms/P9586.733863ms. U−11.0161% vsD346/−18.0932% vsD336;
P95TTFT−8.0574% vsD346, butP99+51.1286%. REJECT as qualifiedG1/G2 gain;
do not hide genuine resource reduction or latency regressions.
Same1114short requests:TPOTmiss196 vsD346185/D33677. Phase2/3 jointmisses
147/49 vsD346128/29 andD33647/16. No more manual softtarget/capacity sweeps.
132realremote transfers/131480060wireB/packing0; outputSHAchanges177 vsD346,
all retained. Native consistency is not numerical qualification. Servicepeak
20519321600B,4026samples,high/max/OOM/swap0; auditpeak995151872B/events0.
Read complete Plan and frozenV1 this turn. Next retained phase2/3
admission/decode/load evidence and one new hypothesis
after historical and primary-source review. Do not repeat D349, finished
capacity/profiles/numericalblocks, or start3B/baselines. Nofigures. GoalACTIVE.
Decision:docs/ieee_tc/D349_COMPLETE_W0_DECISION.md.

### D349 execution history (completed; monitoring snapshots below are historical)

Continuation VERIFIED_WAIT: same D349 execution remains active. Latest check:
watchdog sample2892 / monotonic956569.717038418; Live arrived3169,
done3166, ok3166, fail0, elapsed47:43, three runtimes. Hostavailable99167064064B,
servicepeak20519321600B, high/max/OOM/swap0, disk159360118784B, noabort.
Service invocation43b39a9b2c8543708ab107deab58c2c7 is active/running;
tmux panePID2986339 alive. No terminal launch.json yet. Full Plan reread in
three chunks after context recovery; SHA unchanged. Frozen V1 fully reread.
Read-only observation cells4285/4293 were stopped after their polling windows;
this stops only observation, NOT the tmux experiment, service or watchdog.
There is no outstanding tool cell/session to resume. Resume light monitoring
of this SAME execution; do not relaunch. No serving-code change, remote
adjustment, concurrent heavy work, figures or early performance acceptance.
Do not apply the150GiB start floor as a running abort; the100GiB stop threshold
is unchanged. Recheck/clear only verified rebuildable caches before any NEXT
heavy job. Await full completion and actual release, then analyze as below.
Goal remains ACTIVE; unchanged running state is expected, not a blocker.

D349 launched2026-10-08 15:59Asia/Shanghai in tmux primelora_d349.
Execution/independentremoteV2 hash54d69e859438b38e8851f9a9cce1f40e9bca3131;
18exactpaths and15archivemembers verified. AllD348/D349preparation/checkpoint/
gate/healthauxscopes closed. Current runMUSTNOT be relaunched.
Localraw results/ieee_tc/p2_backend_qualification/d349_20261008/7b_full_w0;
launcheraux3de5233572024d0c9b725b65456cd3fa,invocation
674736526d07472094886a4e9634657c,inode20642991;service
c7bb35d8edd04c69a8acd9391d91c769. Identity receipt is launcher_identity.json.
Ownedremote3services active,monitor d349-full1;remoteNIC1000/full,
publishedgziphealth verified,7bclock remote-process-monotonic:95c20cb25b6e40de81a840478b82dedc.
Existing start/stop/capture procedures reused; no cache regeneration/packing.

Freshgate disk162235592704B,host112964812800B,GPUempty. FullPlan threechunks
reread immediately beforelaunch. Atwatchdogsample120,host95320625152B,
servicepeak17154289664B,high/max/OOM/swap0,noabort;first14completednative
requests have0failure. This is startup/prefix only, notperformanceacceptance.
No livetoolterminal handles;tmux/service/watchdogcontinue. Onlyoneheavyjob.
NEXT monitorcomplete→actualGPUrelease→exactownedremote stop/transfercollection
→existingfull/context analyzers extendedwithstrictD349identity→compareD346
andD336 onunchangedtiming/lifecycle ruler→backup/decision. Do notdoCPUheavy
hash/analysis oranotherexperiment duringthisrun. Do notstopbasedonpartial
timing,hidefailures,changethresholds,orstart3B/baselines. GoalACTIVE;PROGRESS.

D348 reused333C14 GPU-ready samples plus66long-request8/14stress measurements.
Actual admission1–14coverage does NOT identify a steady-load latency curve:
count1 requests later see9.32new admissions on average; short-output1–6 has
15/75TPOTmisses while short9/10 has0/6under different background conditions.
No fitted/interpolated optimal target; proposed value remainsnull.65CPUtests
passed, analysisreturn0, peak508760064B,high/max/OOM/swap0. Ownedanalysis scope
closed byD349preparation. No GPU orremote experiment repeated;no figures.
See D348_C14_PROFILE_IDENTIFIABILITY.md and d348_soft_target_evidence/analysis1.

D349 preservesC14runtime,200/24profiles and IEEEcontrol law; only the absolute
soft-load workingpoint fromD336 is restored: S8,C14 givesqueue8,upper7/14=.5,
lower7/28=.25. This is an explicit causal configuration contrast, NOT an
inferred optimum or a new hardcap8. SameadmittedA/readyN givesactive ratio
A/(7N)asD336. Preparation verifies exactmodel/profileequality and only3control
fields plusexclusivepaths differfromD346.252CPUtests pass(54.201s),peak
2988187648B,high/max/OOM/swap0. Disk162238025728B,host112451342336B atpreflight.
No new100prefix/nativequalification needed forunchanged paths; nextonecomplete
original4000W0 aftertestedbackup,freshPlan/status/gateandremotevalidation.
D349preparationscope01 andcheckpoint02/gate03/health04closed;backup/freshgate
completed andrunstarted asabove. Noexternalbaseline or3B, nofigures.

## Previous — D346 and D347 COMPLETE/AUDITED/BACKED

D347 backup independently verified localHEAD=remoteV2=
0c4af0db7b3776ca13d60a6da7ffb9cbbf8edde1.17exactpaths,12sealedoutputs,
5failed-attemptsource/log versions retained;allD34701–04scopes nowclosed.
Rawbackup_receipt.json/checkpoint_cleanup.json recordexactclosure. Noopen
terminalhandle,GPUexperimentorremote service fromthese campaigns remains.
This continuation madePROGRESS;fullgoal remainsACTIVE. Do notredo D346,
D347analysis,backup,oldcapacityprobes,numericalblocks orfinishedcalibration.

D346 checkpoint localHEAD/independent remoteV2 both
33def6fbbdc0c5f2e0c9173f7a0f910514ff960b;22paths,17hashes,25archive members
verified. AllD346service/launcher/audit/seal/verifier resources closed. First
seal attempt used systemPython withoutNumPy; no inference repeated, preserved
attempt then existingCPUanalysis environment completed seal. No code rollback.

D347analysis2 COMPLETE,96CPUtests pass,peak696680448B/high/max/OOM/swap0.
ReferenceD336projection reused. Actual17readyepochs+1cancelled retained;the
cancellednaturalstart has no engine_ready_at and serves no request. Oldanalysis1
stopped on this unsupported assumption;its exactcode/log/archive preserved.
Onlymeasured closedstartup intervals enter overlap;cancelledrecord retained
separately,all4000requests/allphysicalleases stillcounted,noinventedreadytime.
NoGPU/remoteexperiment running. Analysis03andcheckpoint04closed/verified.

Firsttwofixed500-requestphasesjointmisses61→197;allphases163→345.
Meanothernativeoverlap1.394933→3.211822;scaleupdecisions10→17,retire7→16.
Same518arrivalcohortin eitherclosedstartup:TPOTmiss31→116;outside49→125.
Prefill/load overlap alsoincreases;notisolatedcausality. SeeD347diagnosisdoc.
NEXT hypothesis: useexistingC14D343/D344measured service/length evidence to
derive latency-constrained softtarget distinctfromphysicalC14, preservingIEEE
controller. Firstcheckprofilecoverage;nointerpolated/defaultsamples,nonew
capacitysweep,nonewSLO/batch8ruler,notyetimplementedoraccepted.

D346 execution f4f6d4784d22b5f1afb1b7c6f7ed26bd24d14851 completed4000/4000
native-contract requests, no failure. ActualGPUrelease/service-removal and
ownedremote3unit shutdown/portsabsent/bilateraljournal hashes verified.
No GPU experiment is running. Do not repeatD346/D345/D344/D343.

FullD346: U12715.646901GPU-s vsD33613814.334497 (−7.95%);
meanTTFT1307.344449ms/P953533.423788ms/P998172.970001ms;
meanTPOT38.533414ms/P9585.352486ms. Same provisionalD170timingruler:
3655/4000(91.375%) vs3837/4000(95.925%), −4.55pp;145shortof3800.
TPOTmisses241vs80. REJECT this capacity-derived working point as G1
improvement; retain genuineC14native feasibility but no third capacity sweep.
One developmentrun each, noCI. Ncorrect/commonfinalreferences/G1G2 OPEN.
Same4000input identities;244outputSHAchanges retained,not waived as numerical
correctness. 132realremote fetches/131480060wireB/packing0. TiersGPU2591,
HOST515,NVMe760,remote134. Servicepeak22479060992B,minhost92149645312B,
4069samples,high/max/OOM/swap0. 83analysis tests and both analyzers passed.

Existing full/context analyzers extended only with strictD346identity; no
serving/runtime edit. RawsourceSHA52e7d3fddde1f4bd8ae05831dd4ce0d69e291f9c2b9aba3a7ef5b659a987151e.
Curated paper_results/ieee_tc/p2_backend/d346_full_w0/{validated1,context1}.
Decision docs/ieee_tc/D346_COMPLETE_W0_DECISION.md.
Main audit e18ae... invocation035a7c4012174dafa2157fbcd63f24fd completed/closed.
Postrun disk~151GiB: recheck exactavailable/growth before nextheavy task.

NEXT: the measured-service softtarget question above; D347seal/backup DONE.
History and officialvLLM0.30/Ray2.59/Knative sources checked; theysupport the
capacity/target distinction,notcausalperformanceproof. No blind rerun,
thresholdrelaxation orfigures.
7B target remainsOPEN, then3B, thenpausedbaselines/M1M2/ablations/sensitivities.
D294/D295 registered numericalblocks alreadycomplete; do not rerun them.

### Completed smoke and prelaunch checkpoint

D345100/100native counts/bindings verified. MeanTTFT848.309953ms/P952032.469623ms,
meanTPOT28.546773ms/P9536.128674ms,physical759.836137GPU-s(prefixonly).
Timing-only98/100 underretainedwarmcandidate; numericalNcorrectunknown,
notformalSLO/G1/G2. TiersGPU62/HOST4/NVMe5/remote29,29bilateralfetches,
50650865wireB,packing0; peak18237345792B,minhost94142713856B,events/swap0.
RealGPUrelease/localservicegone/remote3unitsinactive verified;tmuxended.
SourceSHA17d50430c65fc3b2cdb70cc596044f0bbab7dd5d24674c6223777182b310de19.
Existingboundedanalyzer and timing tests return0/0, nofigures.

D346 is SAME model/profiles/controls; onlyexclusivepaths and no100prefix.
Prelaunch passes200/24classes, disk163080966144B/host112219115520B,GPUempty.
Seal/backup D345andD346setup, freshgate andFull4000launch are nowcomplete.
No additionalshorttests/capacitypoints. Compare fullD346toD336, retainalltradeoffs.
Do not reinterpret98/100prefix asfullqualification. No3B/baselines yet.

### D345 launch receipt (completed)

D345 started in tmux primelora_d345 at2026-10-08 13:49Asia/Shanghai.
Executioncommit2ab680ef1b70c2604a0149cf4043704b6ec801e6 independently matches
remoteV2. Freshdisk163125723136B/host112778514432B,GPUempty;remotehealthy,NIC1000/full.
Onlyoneheavyjob. Localraw d345_20261008/7b_capacity_validation_100;
launcheridentity captured in d345_20261008/launcher_identity.json:
aux3f017eb3b9814262ada58b65850cd45c,invocation43cebd2266f949afa34085193ec4e10a,
inode20580442. Service372862e170e14dc4a0fb56bee54e37a3. Do not relaunch.
Remote monitor d345-smoke1; exact-ownedstop/collect prepared. Wait realterminal,
release,thenbounded existing smokeanalyzer (addD345identity), not performanceclaims.
Preparation/exportscopes22/D34501–04closed; no outstandingtoolhandles.

Strict same-C14collection validator and existingexporter now retain1545measured
samples with200service/24preparationclasses (1212loadsamples). Two rawsources
remain separate SHA/clock namespaces;24warmups excluded, no default/C8fill.
376/376CPUregressionpassed. Productionprofile loaders and Full childassembly
match the predeclaredD344contract. ExportmanifestSHA
6f22b0938235cd1fe854a26d794b74415b7afee15f213634559590ee2390a012.
ServiceSHA794d00bac794da9c076b34da38cc310c90d70330b9414ef02de09faab9c7cd32;
preparationSHAbfe0a6f87a65a1835bf0d374f8045f28e0c09c41d5416960b1c4a218ad2893be.
No inference rerun, figures, weights/trace regeneration or remote cache rebuild.

D345 config and existingFullrunner branch ready: diagnosticprefix100, Nplan4000,
sameC14profilebindings/D336controlrule, capacityderivedtargets14, noD341prebuild.
Prelaunch passes200/24classes, disk163130814464B/host112257400832B,GPUempty.
Next exact-ownedCPUclosure/seal+backup, freshgate, publishedremote start and
Full100 in tmux; thenvalidate/release andFull4000. Do not repeatcalibration.
Numericalcorrectness, commonreferences, G1/G2 and7Btarget remainOPEN; no3B/baselines.

### Prior D344 measurement backup (complete)

BACKUP/CLOSURE VERIFIED: localHEAD and independently read remoteV2 both
4bf53e13c5d6d79d0ca6c785683345b1671b7a80.13exact paths,10measurement hashes,
secrets/diff and retained analysis correction checked. Seal20exactlyclosed,
peak47194112B/events0/swap0; no D344aux/service units remain, GPUcomputeempty.
Disk163159347200B. All tool handles closed. Measurement/closure/backup are DONE;
do not repeat. Next is data-only matched multi-source export, thenFull100/4000.

Gap40/40native calls complete;10warmup+30measurements. Two missingGPUshort
classes each have3observations; actualshort admissions9,10,9,10,9,10,with8original
long backgrounds still live in every case. Unionwithsource1 covers200/200service
classes; preparation24classes retained fromsource1. No forcedwait/barrier or
repeated1529matrix. Both rawmeasurements retained; noFullrun or profileexport yet.
Gap physical111.405661GPU-s,startup40.735613s,40controlledsetupfetches,
143002936wireB,packing0;servicepeak5701263360B,minhost106936963072B,events/swap0.
GPUreleased,servicepathremoved,remote3unitsstopped,portsabsent;bilateral40
transferSHA/bytesverified. All14–20auxiliaryscopesclosed andcheckpointbackedup.
No liveexperiment or terminalhandle torestart.
GaprawSHA764509162dd0889b469e6150ae42aed8ca018f1899271d7b1789406e04508e4d;
execution7431ad6bde014182289c5e48daff68b4008cd428. Auditarchive157485B.

Offlineanalyzer1 incorrectly compared initializedmodeldict against plannedchild
including4addedstartupobservations. Preservedanalyzer/failure andinitialoutputs;
analyzer2 usesexisting model_identity plus all originalkeys unchanged,exact4
loaderobservationsvalidated. No servingchange,datamutation orGPUrerun. Two
coverage/summaryversions identical. Targetcoverage/nativeCSV/union sealed.

NEXT: extendexisting measured
profileexport tovalidate sameC14config/runtime,input/resourcecontracts and
nativegeometry across BOTH sourceSHA namespaces. Keep all1515+30measured
samples (24warmups excluded) with their own clocks/ownership; require200classes,
24preparationclasses. No neighbor/default/C8timingfill. Existing single-run
export must still reject incomplete input. After export, existing100Fullsmoke
and4000W0 usingD336controlrules/C14capacity-derivedtargets; NOTD341prebuild.
Do not repeatD343probes,D344source/gaps. Commonwarmbatch8unchanged;numeric,
finalcommonreferences,G1/G2 stillOPEN. Then3B,pausedbaselines/M1M2/ablations.

### D344 gaps1 launch (complete; do not relaunch)

Prelaunch7431ad6bde014182289c5e48daff68b4008cd428 independently matchesremoteV2.
Freshgate disk163448926208B/host112862265344B,emptyGPU;remotehealthy,NIC1000/full.
tmux primelora_d344_gaps,raw d344_20261008/gaps1. Launcheridentity stored:
primelora-tc-aux-8416b3834a2c41f280fe90c9bc9df4a0.scope,
invocation78a4700d14ca4c558580be717bd271a2, inode20567651.
Service primelora-tc-svc-b8556d36117e4d40b7a5bedf5501b0b0.scope.
Wait actualterminal/release; do not repeat/relaunch. Exact-owned stop/collection
scripts prepared; newremote monitorunit d344-gaps, remote path d344_20261008/gaps1.
Then bounded analyze.py checks40calls,30measurements,each target>=3observations;
not automaticFull export. No concurrent heavy job/hash/compression.

### D344 gaps1 preparation (complete)

Completed D344source1 evidence backed up183bd8c56a0075be9f8a6521ba1fcec619474642,
independent remoteV2 verified. Seal09exactlyclosed,peak367312896B/events0/swap0.
Gap selector extends existingprepare_d298_capacity/runner, not serving code:
8existing256-token background requests precede2existingshort targets (rank8/16).
8comes from missingadmissionbin lower boundary, not a tuned runtime constant.
Alternate only target order across3measurement rounds; retained10-call warmup
plus30measured=40native calls. Same C14config,actual KV guard,Fullcontract and
V1warm ruler. No wait/barrier/forcedadmission, newtrace/weights or extra capacity.
16targetedCPUtests/inputassembly and374/374expandedCPUregression pass.
Next: checkpoint, freshgate and same sourcecollector d344-gaps; retain actual
coverage,release,remoteaudit. Multi-source profile validation beforeFull required.
Do not repeat source1. G1/G2,correctness/commonreferences and7Btarget remainOPEN.

### Completed D344 source1, two genuine gaps

Source1 completed1529/1529native calls (14warmup+1515measured); actualGPUrelease,
service removal and exact-owned remote closure verified. No live tmux/GPU run.
198/200service classes,24preparation classes and all static content/source pairs
covered. Missing only GPU-ready short-output rank8/rank16 with actual admission
9–14. Six short-only GPU waves reached at most6actual admissions;8–12requests
finished before the final admission. This explains the sampling gap, not a
runtime capacity failure. Do not fill neighboring/default timings or rerun1529.
1847.578253physicalGPU-s,1529controlled setupfetches,2906411925wireB,packing0;
servicepeak5706301440B,hostminimum106838941696B,all memoryevents/swap0.
Native token/timing/binding checks pass; numerical N_correct remainsunknown.
Profile export intentionally withheld. Commonwarmbatch8/reference/G1/G2 OPEN.

Measurement rawSHA09d2f2b688c7cbce1591bb8b5ab4af8f28d3df1f26377b109b231ec2440c924e;
source execution833ddd6dcc84e31700ad8bf13c5b449c56fe0f72. Curated sixoutputs plus
measurement_outputs.sha256 under d344_capacity_profiles; audit archive1711340B.
Raw135772770B retained, not duplicated into Git. Auxiliary04–09closed. Protected147
unchanged except recorded historical baseline replay script; usermanifest intact.

NEXT: backup completed evidence, then same C14collector/config with original
long-output GPU-ready background ahead of the two existing short targets. Derive
background count from missing admission-bin boundary8; actual native events
must prove9–14. Merge only compatible measured profile evidence; no forced wait,
new capacity point, synthetic request or new artifact. Then100Fullsmoke/4000W0.
No figures,3B orbaseline advance. No new cache: approved once-only published
archives already exist (1581215261B); user approval confirmed again this turn.

### Earlier D344 launch (completed; do not relaunch)

PRELAUNCH BACKUP VERIFIED:833ddd6dcc84e31700ad8bf13c5b449c56fe0f72 matches
independent remoteV2. Preparation/test/checkpoint scopes closed; freshgate
disk163657302016B,host112820686848B,GPUempty. Frozen remote healthy, NIC1000/full.
Launched existing runner in tmux primelora_d344_source, raw d344_20261008,
source1_launcher.log/source1_launch.launch.1529planned nativecalls; no Fullrun.
Wait original measurement, do not relaunch. Then actualrelease, ownedremote
stop/transferaudit/coverageexport; Full100/4000 next. G1/G2 remainOPEN.

Live handoff: source launcher identity recorded in raw/launcher_identity.json:
primelora-tc-aux-3799f099215848feb788a8ffbcaf23ec.scope,
invocation3dc2f2e322c84cb8aa447018840e8885. Service is the guarded D344source
run; no open terminal handle to restart. Last inspected measurement progress
870/1529native sample events throughwave68 (first505measured-call round plus
14warmups complete; secondround long-output classes underway). Last watchdog
sample1021: servicepeak5.315GiB,hostavailable100.9GiB,noabort. These are running
samples, not finalpassed. Original tmux and exact service scope confirmed live;
no new launch. This continuation is VERIFIED WAIT, not a blocker or completed
experiment; Plan/V1 hashes unchanged. Finish the same measurement before work
that would compete for resources or reinterpret its final outcome.
Raw start/stop/health/collect_closure scripts reuse D343 exact-owned procedure;
collect_closure checks the recorded launcher identity. analyze.py uses existing
source-domain/export validators and saves coverage1 before conditional export;
if incomplete, preserve missing classes, do not relabel timing or rerun whole
matrix blindly. Existing tmux run must finish before stop_remote.sh/closure/
bounded analysis. Read this live state first next turn; D343probe/backup and
D344CPUpreparation/checkpoint are complete and must not be repeated.

Continuation audit: previous turn=PROGRESS (implemented/tested/backedup/launched
D344). This turn verified the same live tmux/worker measurement and reread full
Plan/V1; no experiment restarted. Raw analyze.py now validates all native timing
and bilateral transfer bytes even when coverage is incomplete, and retains a
measurement_summary1 before conditional profile export. Do not run it until
source terminal/release and exact remoteclosure. No new hypothesis/configuration
or heavy task while source calibration is live. Do not confuse its controlled
three rounds with independent Full runs or numerical adapter qualification.

2026-10-08: full Plan/V1 reread; D343 completed evidence not repeated. Existing
D298collector extended with D344source mode. Same C14native config, D321output
edges24/91/217/256, admitted1/2/4/8/14; first1000originalrequests only.121waves,
14warmup+1515measured=1529calls,41existingadapterIDs.15targeted CPU tests pass;
SHA-bound config/length/source assembly passes. Expanded CPU regression373/373
passes (including source-domain/export, lifecycle and launcher checks). No new
weights/trace or GPU run.
All timing classes will be freshly measured; commonwarmbatch8/SLO unchanged.
Only D336capacity-derived controller targets follow C14rule; no other knob.
NEXT: expanded CPU check/checkpoint, freshgate, existing tmux source calibration;
then actualrelease/remoteclosure/coverage export, Full100smoke and4000W0.
No figures,3B orbaseline advance. Numerical/commonreference/G1G2 remainOPEN.

## HISTORY — D343 C14 native probe COMPLETE; affected profiles are D344

BACKUP/CLOSURE VERIFIED: localHEAD and independent remoteV2 both
2741f59d378f6a4b591eb1fca42c464e56e5d017. All D343 C15/C14 inference,
collection,analysis,checkpoint scopes closed; final verifierpeak27127808B,
events/swap0. GPU compute empty; remote services already stopped.9finalpaths
and8curated output hashes verified; standard CSV CRLF validated explicitly,
measurement bytes unchanged. Prior9a4968d snapshot identities checked from Git.
Usermanifest unchanged/unstaged. Disk163663269888B. D343 run/analysis/seal/
backup are DONE; do not repeat. All tool handles closed. Current mainline is
C14 affected timing-profile refresh, then100Fullsmoke and4000W0, not another
capacity probe, plot,3B orbaseline task. G1/G2 remainOPEN.

RESULT:80/80native contract calls,14warmup+66measured. All7waves956freeKV
before/after>=896needed; native interval peak14 and measured common decode
intersection7.18–7.43s. Three within-process pairs (not independentruns):
wave token/s201.043838→237.038846(+17.90%), SAME8requests TPOT32.787416→
46.992471ms(+43.32%), nativeTTFT288.272724→323.235279ms(+12.13%). This is
sameC14runtime8/14stress-wave tradeoff, not oldC8/newC14Full advantage.
OutputSHA changes2/2/1pairs; N_correct and finalwarm/Resident/G1G2 remainOPEN.
227.890459physical GPU-s,80setupfetches258013231wireB,packing0; startup87.647s
includes firstnewconfig initialization. Servicepeak5449039872B;events/swap0.
GPU released, service removed, exactlauncherauxclosed,3remoteunits inactive,
remote SHA/80transferbytes verified. Analyzer finished in scope11, awaits exact
closure/seal/backup only. C15 alreadyclosed/rejected; no3rdminimalcapacitypoint.
NEXT after backup: reuse D298 collector to refresh only changed C14 timing
classes, retaining V1warmbatch8 and D336 paper control rules; then existing
100Fullsmoke/4000W0. Probe GPU-only classes are not a completeFullprofile.
No repetition of completed D343 probes, no figures, no3B orbaseline advance.

### Earlier C14 launch (completed; do not relaunch)

C14 prelaunch backup verified9a4968dcee02febb590630ba8b8a9e189c4de83d.
Checkpoint scope10closed. Fresh disk163729219584B/host112718135296B,
empty GPU, frozen published remote healthy. tmux primelora_d343_cap14,
raw d343_20261008/cap14/source1_launch.launch.80planned calls, no Full
launch. Wait original run, retain failure/success, actualGPUrelease then
owned remote stop and analysis. Do not rerun C15 or generate new inputs/figures.

C15 initialized with942actual free KV blocks, below960required.0/84inference
calls; initial schema guard incorrectly expected[] for integer0 unretired
iterations. Corrected guard tested against retained native snapshot still
rejects942<960. No OOM.101.518437physical GPU-s,15setup fetches/39670961wireB,
packing0; service peak5450616832B,high/max/OOM/swap0. NativeGPU released,
service scope removed, launcher scope exactly closed, remote3units stopped,
journal SHA/bilateral15transfer bytes verified. First result is DONE.
One explicit followupC14=floor(942/64),80calls from same indexing rule, new
cap14config/spec/raw paths.40follow-up CPU tests pass. No auto fallback or
new workload. Next checkpoint/backup then freshgate and C14, not repeatC15.
Current analysis scope09 awaits exact closure in followup checkpoint.

### Earlier D343 launch record (completed; do not relaunch)

Backup independently verified:2800caec4dab56819459a863b03d20033e03dfde on V2.
All preparation/test/checkpoint scopes closed. Fresh gate:disk163769196544B,
MemAvailable112784265216B, emptyGPU. Existing published remote services started
and authenticated; no new artifact cache.84-call native probe launched in
tmux primelora_d343_probe through existing gated-launch. Raw d343_20261008;
source1_launcher.log/source1_launch.launch. No repeated run or Full launch yet.
Wait for original result/release; then stop only the owned remote invocations,
collect native capacity/TPOT evidence and decide whether changed profiles/full
are justified. Numerical correctness/common references/G1G2 stay OPEN.

Full Plan/V1 and completed D342 read. Native current vLLM docs/source checked.
One candidate: actual sequence/runtime/LoRA capacity8→15, derived from D169's
1022free blocks and64blocks/max-context request. Fresh larger-runtime guard
requires960free blocks and drained native state; no shrinking or guessed KV.
Reuse D298 native_source_matrix, D168 existing long-context distinct-adapter
indexes from first1000,84calls (15warmup+69measured), alternating8/15 three rounds.
Base remains D336/D298, not D341's unaccepted prebuild switch. Common warmbatch8
and SLO stay unchanged. Explicit completed-length-only transfer keeps old timing
profiles separate.27initial CPU tests pass; no serving algorithm change and no
GPU run yet. New config/spec are developmental, not a selected Full working point.
Expanded CPU regression371/371 passes. First driver lacked spawn main guard;
retained failure and corrected driver, no model run or serving-code workaround.
Next: regression/checkpoint, fresh resource/remote gate, bounded native probe;
if safe/useful refresh changed profiles then Full100/4000.7B goals OPEN, then3B,
paused baselines/M1/M2/ablations/sensitivity. No plots or new input data.

## HISTORY — D342 retained dispatch-window diagnosis COMPLETE and backed up

BACKUP/CLOSURE VERIFIED: localHEAD and independent remoteV2 both
70a962bb622647c06ccdc02e892e6b3c359f79e8.19exact paths committed/pushed,
all13curated hashes verified, earlier2source versions preserved. Both analysis
scopes and checkpoint scope exactly empty/stopped; final scope peak34529280B,
events0/swap0. GPU compute census empty; all tool handles closed. Usermanifest
unchanged/unstaged. D342 analysis/context/seal/backup are DONE; do not repeat.
Current mainline remains7B capacity/TPOT feasibility, then3B and paused baselines.

2026-10-08: full Plan/V1 re-read; approved published remote cache already exists,
reuse without regeneration. D341 closure/audits/backup remain DONE. Next data-only
test splits release→gate-resume at acquire entry, using both complete4000 sources.
Current alternative explanations: pre-acquire event-loop delay versus await-gate
capacity/resumption delay. Exact permit grant/release absent; sampled controls
must not be treated as exact live gate state. No serving/config change accepted.
Existing analyzer extended and boundary/identity/overlap tests added. New output
d342_dispatch_wait/analysis1 only; 4GiB/no-swap auxiliary scope, no figures.
Disk163794300928B, hostavailable112809058304B, GPU census empty before analysis.
G1/G2 and 7B target remain OPEN; then3B, baselines, main comparisons, ablations.

RESULT: both4000 splits pass, identity error0. Pre-acquire mean28.539295→26.052328ms;
acquire-to-resume43.789969→212.001188ms. D341 phase1 accounts98.96% of the
second interval sum. At756.640562s,8definitely-owned permits cover ready capacity8,
37acquire waiters and3pending activations.24control samples observe waiters,
18have the conservative ownership lower bound cover capacity. This supports
local capacity scarcity while replacement capacity starts, not an exact causal
attribution of all delay. D336 same period had2ready replicas; trajectories differ.
No serving/config change, GPU run, remote operation, new input or figure.
72split/regression tests plus34follow-up capacity-bound timeline tests pass;
analysis peaks1001353216B/61218816B,all memory events/swap0. Context tool77208
finished, analysis tool60340finished. Both exact analysis/context aux closed.
Checkpoint verification passed19exact paths,output/source SHA,archived earlier
source versions,secrets/AST/diff and147protected entries with solely the known
historical baseline-script change. Usermanifest unchanged/unstaged. Disk
163786104832B. Save/push this checkpoint, close its exact empty auxiliary scope,
and independently read remoteHEAD; no new inference in D342.
NEXT after backup: one bounded 7B capacity question—does verified C8 constrain
useful batching relative to actual KV/LoRA/iteration capacity and TPOT? Reuse
history/native resource evidence and official same-version implementation before
any candidate; do not blindly double concurrency, hold extra GPUs for free, or
change the ruler. Do not rerun completed D341/D342 or restart old startup probes.

## HISTORY — D341 complete4000 audited; no full-run improvement

BACKUP/CLOSURE VERIFIED: local HEAD and independently read remote V2 both
5eed59ebabe2ef9d7b34b86836185ed5e3c72913.24 exact paths,19curated hashes,
26archive members, startup references, protected/secrets/syntax/diff checks PASS.
All D341 collection/audit/startup/seal/verifier scopes exactly empty and stopped;
final verifier peak34103296B/events0/swap0. GPU compute census empty. All tool
handles closed. Disk163799715840B. Usermanifest unchanged/unstaged. D341 closure,
analysis and backup are DONE; do not repeat them next turn.

NEXT source location already identified: run_all_experiments.py13051–13171
dispatch gate; external replay is open_loop, so YAML workload concurrency2 does
NOT impose a global2 clamp. Gate limit is current runtime groups times measured
runtime capacity8. The permit spans the entire request and grant occurs on
release/capacity notification. Its recorded wait also includes event-loop
yield/resumption before acquire, not just permit saturation. Inspect retained
request/activation/control timelines to separate these before changing anything.
No new serving change or optimization hypothesis accepted on source reading alone.

2026-10-08: goal ACTIVE/current turn PROGRESS. Same D341 run completed4000/4000
native-contract/terminal records,0failure. Actual GPU release, service removal,
exact-owned auxiliary closure,3remote services inactive/ports absent and remote
log SHA checks passed. Observer3717 and audit80273/startup51805 are CLOSED.
No replay restart, serving change, new pool/trace, remote interference or figure.

83 offline tests and existing full/context analyzers PASS. All10 binding checks
and time identities pass;219output-token SHA changes vsD336, N_correct unknown.
U14350.604333GPU-s vs13814.334497 (+3.88%); mean/P95TTFT1407.676/3319.578ms
(+16.26/+13.99%); P99TTFT13265.145ms vs4766.143ms. Mean/P95TPOT31.287/56.075ms
(-2.97/-5.09%); meanE2E5.002403s(+3.08%). Provisional timing3813 vs3837, NOT
formal SLO. All10 native spawn/unquantized/no-background witnesses verified;
actual activation mean43.784759s vs47.371069s in11D336 epochs, not a paired
causal estimate. Initial preparation51.077484s vs51.449530s. See D341 decision.
Remote132/131480060B/packing0 unchanged.4037resource samples,peak20673847296B,
minhost94667337728B,high/max/OOM/swap0. SourceSHA9f87d2e10ac42d08717d40a7fbafe725c16916eb4e7d34aa70cee177fd39b664.

Retained wait partition identifies dispatch-window mean72.329->238.054ms as
main component of total queue609.246->802.372ms increase, NOT root-cause proof.
Candidate not promoted to qualified working point. NEXT after current evidence
checkpoint: inspect retained long dispatch-window waits and associated live
capacity/activation history; then one evidence-based7B hypothesis. No blind
D341 repeat, import/fork/prebuild retest, higher SLO threshold or premature3B.
G1/G2/N_correct/commonwarm/Resident/old-new comparison remain OPEN;3B and all
baselines/M1M2/ablations/sensitivities still pending. Backup closure underway;
latest verified source checkpoint remains eea5cd2c4b4267b53d7e392d0882ac2110e33ecb.

## HISTORY — D341 unchanged 7B full4000 running (now complete)

2026-10-08: D340 smoke audit/cleanup/backup DONE. Local and independently read
remote V2 eea5cd2c4b4267b53d7e392d0882ac2110e33ecb,17 exact paths.18 analysis tests,
bundle/source hashes, syntax/diff/secrets checks passed; user manifest untouched.
First checkpoint stopped before commit on one YAML EOF blank line; attempt1
evidence/binding kept, syntax-only correction separately frozen in full_binding2.
All D340 analysis/backup scopes closed; final backup peak29069312B/events0/swap0.
Do NOT repeat any completed D340/CPU/import/compression work.

D341 fresh gate passed (disk164785528832B, hostavailable112804376576B), using
launch_full_gate.py and full_binding2.json. D80 remote services reused unchanged,
both authenticated health passed, actualNIC1000Mbps/full. Launched tmux
tc-d341-full1 via existing runner d341; raw d341_20261008/7b_full_w0. Replay
declares4000/4000 source, no diagnostic prefix, original traceSHA efb903254fcd...
and W0view a5331be2e220... . First watchdog samples show no events/abort.
Continuation verified wait: same tmux/service invocation remained live through
30 spaced45s polls across two continuations; actual frontend/native core PIDs
confirmed at each final check. Plan and V1 hashes remain unchanged.
Latest continuation 2026-10-08 10:38+08 is VERIFIED WAIT, not a blocker:
actual frontend/native core PIDs remain live under the same service invocation.
Terminal journal2791success/0failure observed. Last watchdog sample2509:
hostavailable88.727GiB, service18.185GiB, swap0, no memory events/abort.
Disk162179891200B; Plan/V1 hashes unchanged. Full run remains active; no
relaunch, serving/config change, remote intervention or heavy offline work.
Observer exec cell3717 remains running (45s spaced low-overhead checks);
resume that handle before starting any new observer. It does not own the
inference tmux and its monitoring duration is not a replay timeout.
User reapproved once-only delivery cache; D78/D80 already supplies it. Reuse,
do not recreate. Current D341 tests on-demand unused quantization preparation,
NOT removal of remote request-time packing (already accomplished earlier).
Owned aux96e54c3f52d34e1e9487846dfe92c4de invocationaa1a762df48b48748f27175020766281;
prepared collect_closure.py requires this exact identity after completion.
Do not relaunch, change serving sources, scan caches or run concurrent heavy jobs.
NEXT: monitor through complete4000, then actual GPU release/exact owned cleanup,
stop only matching remote invocations, collect/verify remote records and use
existing full analyzer extended for D341, compare to D336 full (not D334 prefix).
No figures. G1/G2/N_correct/commonwarm/Resident/old-new qualification still OPEN;
3B then baselines/ablations/sensitivities remain pending. No goal closure.

## D340 100-prefix completion

2026-10-08: D340 completed100/100 native-length/timing/control-binding checks;
2 actual native spawn/unquantized/no-background witnesses. Mean/P95TTFT
784.9585397/1718.4081011ms, mean/P95TPOT29.8732831/40.4399100ms,
U330.7621931GPU-s.29 transfers50650865B, no dynamic packing, remote1000Mbps/full.
Service peak9471033344B, high/max/OOM/swap0. ActualGPU release, service scope
removal, exact aux closure and3remote units inactive/ports absent confirmed.
Offline18tests + strict audit passed; numericalN_correct and finalcommonSLO
remain UNKNOWN. D334 prefix context has different controls, NOT causal savings.
No figures. D341 fullW0 config/launcher prepared: identical D340 candidate,
all4000 original requests, exclusive output/cache paths. Next seal/checkpoint
smoke evidence and full-run config, fresh gate, then actual D341 GPU replay.
Do not repeat D340 or D339/D338; G1/G2/3B/baselines remain OPEN.

## D340 RUN RECORD

2026-10-08: fresh launch gate passed; frozen D80 remote services started with
verified identities, both authenticated health checks passed. Actual NIC1000Mbps/
full again; no cache regeneration. Existing guarded runner launched in tmux
tc-d340-smoke1, raw d340_20261008/7b_prebuild_validation_100. Native spawn and
on_demand_v1 retained. Initial watchdog samples: no memory events/abort/foreign
compute. Do not relaunch or change serving sources while active. Next: monitor,
exact-owned local/remote cleanup, strict data-only100-prefix audit, then unchanged
candidate complete4000 if qualification passes. No figures or baseline tasks.
Health check's first stdlib-Python package import lacked NumPy before any HTTP
request; management script now loads the unchanged stdlib-only HTTP module by
file, authenticated health passed. No serving/config change or failed GPU run.

## D340 PRELAUNCH BACKUP

BACKUP/CLOSURE VERIFIED: local and independently read remote V2 both
df56fb08a9e727557b6b7a78b9e262b5c09eb44c,25 exact paths; archive/source SHA,
diff/syntax/secrets checks pass. User manifest unchanged/unstaged. All D339/D340
CPU/compression/prelaunch/backup scopes closed exactly; final backup peak106885120B,
events0/swap0 and GPU compute census empty. CPU validation/profile binding and
backup are DONE. Do NOT repeat them, D338 imports, D339 fork, or compression.
Next actual task is fresh gate/remote start and existing guarded D340100-prefix
runner (native spawn/on-demand Humming); no GPU replay has been launched yet.

2026-10-08: D339 native-fork candidate passed170 unit tests but real-import safety
precondition FAILED: CUDAfalse,2 Humming background reaper threads,4 OS tasks.
No GPU fork run; patch archived then fully reverted. D339 CPU scopes empty/stopped,
peak2369814528B/events0/swap0. Do not retry fork or relax the thread guard.
Three closed historical failed-run main_outcome logs losslessly compressed via
the prior procedure; double roundtrip SHA/source provenance passed, failures kept.
Freed4213305344B; available164836679680B at completion, above150GiB. Compression
peak3221626880B/high3599/max/OOM/swap0, exact scope empty/stopped. No GPU replay.

New single candidate D340: upstream HUMMING_DISABLE_PARALLEL_BUILD=1 for actual
unquantized FP16, native spawn retained. D339 observed unnecessary import-time
background preparation, but its latency contribution remains unmeasured. Source
history/official/local source audited.228 CPU tests pass; actual native import
has only MainThread, CUDAfalse, spawn retained, no Humming reapers. Existing
160 service/24 preparation profile classes bind; preflight disk164820172800B,
host available112265232384B. No GPU launch yet.
New config reuses D336 controls and original trace/pool,100-prefix smoke only;
full4000 required before performance selection. No sitecustomize deletion, fork,
new artifact cache, trace regeneration, policy/metric change or figures.
NEXT backup tested candidate, fresh guarded remote/
local gate, then D340 100-prefix. Do not repeat D337/D338 or compression.
G1/G2/N_correct/commonwarm/Resident/old-new qualification remain OPEN;3B and all
later baselines/matrices pending. D80 remote delivery cache remains the approved
shared cache, reused without regeneration. See D339/D340 design notes.

## HISTORY — D338 CPU import diagnostic done; no unsupported startup shortcut

BACKUP/CLOSURE VERIFIED: local HEAD and independent remote V2 both
3aab0f6a40468f0b9e26671b94a5db2113d1e18e.9 exact paths, archive/source SHA,
diff/syntax/secrets/protected checks passed; user manifest unchanged/unstaged.
Diagnostic and backup scopes empty/stopped; final backup peak28143616B,events0,
swap0, GPU compute census empty. Current disk160627277824B remains below150GiB.
D338 diagnosis/closure/backup are DONE; do not spend next turn repeating them.

2026-10-08: one bounded CPU-only import diagnostic completed, NO new GPU replay,
serving/config change, figures, artifact cache or trace regeneration. Two paths:
repository startup entry10.163524s + runner.661289s, totalwall13.004194s;
neutral entry.023166s + runner12.791289s, totalwall14.404769s. One observation
each, fixed order, importtime overhead/CPU affinity2,3,26,27; not live activation.
Bypassing sitecustomize shifts required imports; it does NOT establish10s savings.
Do not remove startup hooks or subtract timing on this evidence. Both CUDA
initialized=false, compute census empty; peak1686216704B,events0/swap0.
Initial launcher read unavailable cpuset file and stopped before native imports;
failure retained, actual affinity verified. See D338_STARTUP_IMPORT_DIAGNOSTIC.md.
No production candidate accepted; do not repeat this microtest or D337 analysis.
NEXT: restore disk headroom with reference-checked rebuildable cache recovery
(currently149.6GiB<150GiB, no heavy launch); then actual worker CPU initialization,
engine/core and ready boundaries for ONE startup candidate/minimal validation.
Do not treat uv cache size as reclaimable: inspected native wheels hardlinked.
Known histories D293 prefetch/D307-D315512budget/D329 unsafeHOSTcache remain closed.
G1/G2/commonwarm/Resident/N_correct/old-new7B qualification remain OPEN;3B then
baselines/ablation matrix pending. User reapproved published artifact cache;
existing D80 cache reused, not regenerated. Closure/backup verified above.

## HISTORY — D337 retained full-run trade-off localized; next startup-path hypothesis

BACKUP/CLOSURE VERIFIED: local HEAD and independent remote V2 both
cfe420d13678c67c4d2f123879634292187a8c32.18 exact paths, source/output SHA,
syntax/diff/secrets checks passed; user manifest unchanged/unstaged. Analysis,
startup extraction and backup scopes are all exactly owned-empty and inactive;
backup peak63369216B/events0/swap0, GPU compute census empty. Disk160641273856B
still below150GiB. D337 analysis and backup DONE; do not repeat them next turn.
Latest user goal reconfirmed: use agreed frozen G1/G2, no CE-only or changed
ruler, remain7B then3B, no figures/baselines before the model targets.

2026-10-08: D337 analysis COMPLETE, no new GPU replay/service/config change or
figures.73 CPU tests pass; D335/D336 each4000 retained intervals analyzed with
the same D319/D327 method. In the common288 startup-arrival requests, provisional
joint timing misses22->51; other3712 misses111->112. This decomposes the observed
net30 count increase, NOT a causal startup-attributable fraction. In release-gap
common2033 requests P95TPOT57.089->66.388ms; average peer-native occupancy
.983->1.395 overall, load count1977->1975 and load seconds71.086->71.201.
After both last-ready, common946 P95TTFT2940.804->3225.216 while P95TPOT improves
44.406->40.694. Startup is worth targeting but cannot explain all outcomes.

All11 actual worker PID/activation logs retained: activation44.314–51.429s,
mean47.371; model loadmean9.379s, init/profile/KV/warmupmean11.501s; all11 AOT
cache hits, compilemean.279s. Eachhasone runtime LoRA JIT notice without duration.
Nested rounded log spans not summed or cross-subtracted from monotonic times.
See D337_CAPACITY_STARTUP_TRADEOFF.md and d337_capacity_lifecycle/analysis1.
No more subdivision/re-audit/rerun of D336. NEXT one7B question: fresh frontend/
core initialization, handshake and ready-publication repeated work, not deleted
TTFTguard, higher targets, max2cap, eager fallback or repeated512budget test.
Inspect actual code/history/official implementation; if needed one bounded
no-GPU startup/import measurement before selecting a minimal production change.
Unknown N_correct/commonwarm/Resident/old-new G1/G2 remain OPEN; 3B/baselines
and latermatrix paused. Disk below150GiB still prohibits a new heavy launch.

## HISTORY — D336 full4000 complete/audited/backed up; resource–latency trade-off

08:48+08 BACKUP VERIFIED: local HEAD and independent remote V2 both
321d45b49ef1bcae1ef43bd69e1a77d92fa081ad.23 exact staged paths/17curated hashes/
24archive members,secrets/syntax/diff checks pass;user manifest unchanged and
unstaged. Final verifier empty/stopped,peak48525312B,events0/swap0. All current
local/remote experiment and analysis scopes closed; GPU compute census empty.
D336 closure/analysis/backup are DONE, not another task to repeat. Available
disk160669462528B below150GiB; no new heavy launch before evidenced recovery.
NEXT mainline: use retained full request/decode/control evidence and history to
distinguish the TPOT/queue trade-off before ONE next7B hypothesis. Do not declare
formal SLO, restart D336 unchanged, repeat numerical/profile blocks, or jump3B.

08:46+08:79 CPU tests and both reused full/context analyzers pass. Exact-owned
offline scope empty/stopped,peak1064529920B,events0/swap0.24-member49850B closure
archive sealed with curated hashes; no inference/remote service active. D336
analysis is DONE; do not repeat this run or audit-only cycle. See
D336_COMPLETE_W0_DECISION.md. Backup was pending at08:46 and is verified above;
return to7B with the retained latency/decode evidence, not3B/baselines.

All4000 native counts/10binding checks/time identities pass,207outputSHA changes
vsD335 retained,N_correct unknown. Mean/P95TTFT1210.845/2912.194ms,
mean/P95TPOT32.245/59.083ms,U13814.334497GPU-s. RelativeD335 U-12.28%,
P95TTFT+1.90%,P95TPOT+12.01%,meanE2E+3.98%; timing-only3837vs3867,
NOT formal SLO. RelativeD332 physicalU also lower, but no full dominance claim.
Seven release gaps total2033.436636GPU-s vs97.092069: observed U reduction
1933.718657equals1936.344567more gap time minus2.625910envelope increase.
Remote132/131480060B/packing0;4033resource samples/peak21622865920B,
minhost93376729088B,events/swap0. Candidate is a resource operating point with
quality cost, not G1/G2 qualification. No higher targets/deleted guard/max2 clamp.
Do not repeat D334/D335/D336/profile/numerical blocks without new evidence.

## HISTORY — D336 completed and offline comparison launched

2026-10-08 08:42+08: all4000 terminal/native-contract records,0interrupted,
launch pass and actual GPU release verified; physical U13814.334496588912GPU-s,
N_correct unknown. Result SHA b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05.
Exact-owned replay scope stopped empty; both remote services/monitor inactive,
PID0/portsabsent. Transfer/monitor downloads SHA match remote, no request packing.
Only extended existing analyzers for the new D336 identity; no serving changes,
no new inference or figures. Full source plan/V1 and relevant skills reread.
NEXT: finish bounded full/context checks versus D335, retain D332 context,
interpret complete latency/resource/control trade-offs, preserve unknown numerical
qualification, seal/backup then choose one next7B step. Do NOT relaunch D336.
Fresh disk160747806720B below150GiB: no new heavy performance launch until
evidenced recovery; no relaxed gate or deletion of unique measurements.7B final
qualification/3B/baselines/all later matrix remain OPEN.

## HISTORY — D336 full4000 W0 running observations

2026-10-08 07:31+08: actual replay/service launched in tmux tc-d336-full1, pane
PID1511285, existing guarded runner case d336, no prefix/profiler. Executed HEAD
0175a045ad7fe1a4f177330e5dda405e87b63955 independently matches remote V2.11 exact
staged paths/4 curated SHA/54 archive members passed secrets/syntax/diff checks;
user manifest unchanged/unstaged. Preparation is DONE; do not relaunch D336.

Fresh gate disk161554620416B/memory112754069504B. Five preparatory CPU scopes
closed exactly; final verifier empty/stopped, peak49602560B/events0/swap0.
Legacy empty systemd scope labels were observed, but a direct cgroup PID census
found no active local Prime experiment before launch; no unrelated scopes killed.
Remote frozen services/monitor started under current identities,link1000/full,
3B/7B authenticated prepublished_gzip_v1/artifact_timing_v2.7B clock
a94b0b3d55744c01aa56228f4ee6cbb7. Existing D78/D80 approved cache reused without
regeneration. User's authorization is already persisted in source plan§2026-09-27.

NEXT: monitor this same ordinary4000W0 to completion, release exact-owned local
and remote resources, reuse full/context analyzers for D336 vs D335 with D332
context, preserve failures/negative effects/output changes/unknown N_correct,
then choose the next7B mainline step. No code/config/profile/remote changes or
heavy analysis/compression during measurement. No figures.7B final G1/G2,
numerical/commonwarm/Resident/old-new qualification,3B and baselines remain OPEN.

07:33:31+08 monitor: same pane PID1511285 and replay active,25 terminal records,
127 watchdog samples; service peak18578640896B,host available94969180160B,
high/max/OOM/swap0,warning/abortnone. Early first5 service observations included
mean TTFT11.016s from startup/arrival backlog; preserve this regression warning,
but do not extrapolate five requests to full4000 or change the candidate midrun.
No tool process remains pending; long experiment continues in tmux with its
independent watchdog. No further cache audit/setup/backup repeat is needed.

07:43:47+08 VERIFIED WAIT continuation: actual same tmux pane PID1511285/ps
live,735 fresh watchdog samples,541/4000 reported successful terminals,0fail.
The original trace crossed its first500-request rotation boundary. Actual held
GPU samples4->2->4->3->2 demonstrate release/re-expansion, NOT a complete-run
resource-saving or SLO claim. Service peak19331055616B,current host available
103178145792B,high/max/OOM/swap0,foreign compute0,warning/abortnone. Full source
plan and monitor skill reread; no code/config/profile/remote mutation, no cache
scan/compression/GPU restart/baseline/figure work during measurement. The bounded
foreground observation cell ended only; tmux/replay/watchdog/remote remain live.
Next still finish THIS4000run, not another setup/audit-only/relaunch cycle. The
source trace spans3963.9s; ongoing elapsed time alone is not a blocker.

07:55:49+08 VERIFIED WAIT continuation: same actual tmux pane PID1511285/ps
live, elapsed24:29, watchdogsample1448; latest service report1339/4000successful,
0fail,1342 terminalrecords. Physical heldGPUs4, servicepeak20601348096B,
hostavailable95761985536B,high/max/OOM/swap0,foreigncompute0,warning/abortnone.
Complete source plan/AGENTS/monitor skill reread after context compaction.
No serving/config/profile/remote changes, no analysis/cleanup/compression or
other performance runs. Observation cell3446 ended only; actual tmux, service,
watchdog and frozen remote remain running. This is expected verified waiting,
not an impasse or a new experiment. Next still finish THIS4000run, exact-owned
closure, then full D336-vs-D335 analysis with D332 context; no relaunch/repeated
setup/cache generation. User's approved delivery cache already exists and is
reused. All final qualification/3B/baselines/later matrix remain OPEN.

08:14:02+08 VERIFIED WAIT continuation:24 bounded observations of the same
actual tmux pane PID1511285/ps completed; service report2780/4000successful,
0fail,2789 terminalrecords,watchdogsample2526,elapsed42:42. HeldGPUs fell4->3
near the1500request boundary and stayed3 across subsequent samples through
the2500boundary. This is physical release evidence, NOT final GPU-s/SLO gain.
Servicepeak20601348096B,hostavailable98614521856B,high/max/OOM/swap0,
foreigncompute0,warning/abortnone. Source plan/AGENTS/monitor skill fully reread.
No serving/config/profile/remote edits, no heavy analysis or interference work.
Observation cell3452 completed normally; actual experiment remains LIVE and
must NOT be restarted. This turn/previous turn are verified waits, not blocked.
Next: finish same full4000, exact-owned release, full/context analysis then
evidence-based7B decision. All final qualification/3B/baselines remain OPEN.

## HISTORY — D336 capacity-derived soft-target candidate prepared, not launched

2026-10-08: full source plan/V1 and D88/D298/D300/D335 history rechecked. Latest
user approval of once-published delivery cache reconfirmed; D78/D80 already
provides this cache, so no regeneration or second extracted artifact pool.
No new GPU/remote run yet. D335 closure/backup remain DONE, not rerun.

ONE new configuration factor: apply original D88 capacity rule to actual C8:
queue_upper8, active_upper.875, active_lower.4375. Min1/max4, actual cap8,
backend, profile160/24, delay guard5000/2339.924886ms,45s window/38s cooldown,
2s evaluation, D333 serving code, generation and V1 ruler unchanged. Soft target
need not equal hard capacity; candidate is a test, not a proven correction or
optimum. No artificial max2 cap. See D336_CAPACITY_CONTROL_WORKING_POINT.md.

Factored original capacity derivation into a reusable helper with deterministic
tests; new config/runner identity D336. CPU validation and backup precede full
4000 W0. No repeated short smoke/profiling/numerical blocks. Disk below150GiB
requires evidenced recovery first; only nine public download archives proposed,
installed/extracted packages/environments/models/results untouched. First audit
found absent local SHA metadata and removed NOTHING; second audit verifies
official object bytes by SHA rather than relying on MD5. Failure evidence kept.
187 CPU tests PASS54.171s, peak2504757248B/events0/swap0; prelaunch exact config
and child/profile checks PASS, peak515174400B/events0/swap0. Nine reviewed public
archives removed1590808576B; installed packages and all evidence untouched.
Preflight disk161556115456B/memory112497967104B, no threshold relaxed.
Next: seal/close bounded CPU work, checkpoint, fresh gates, then
one ordinary full4000 W0. Numerical/commonwarm/Resident/old-new G1/G2 and all
3B/baseline/later matrix remain OPEN. No figures before target qualification.

## HISTORY — D335 full4000 W0 COMPLETE/audited; full-run gain NOT established

BACKUP VERIFIED: local HEAD and independent remote V2 both
cc6046607cee16d7409e2a542ca8468dfd93cffe.22 exact staged paths,17 curated hashes,
24 archive members,secrets/syntax/diff checks pass;user manifest unchanged and
unstaged. Final verifier peak36073472B/events0/swap0, exact-owned-empty/stopped.
All local/remote experiment and analysis scopes closed; GPU census empty. D335
closure, analysis and backup are DONE: do not repeat this as another audit-only
turn. Available disk159981162496B is below150GiB; no new heavy launch yet.
Next task is resource operating-point/queue-pressure evidence, not D335 rerun,
D334 smoke, D328 sampling, D295 numerical blocks or another backup-only cycle.

2026-10-08 07:16+08:79 tests and full/context analyzers pass, offline exact-owned
scope empty/stopped; peak1094164480B/events0/swap0.24-member50508B closure archive
and curated hashes sealed. No performance/GPU/remote service is active. D335
analysis is DONE, not a task to repeat. See D335_COMPLETE_W0_DECISION.md.

All4000native counts/10binding checks/time identities pass,199outputSHA changes
vsD332 retained,N_correct unknown. Mean/P95TTFT1202.490/2857.905ms,U15748.053154
GPU-s:2.29%/5.25%/2.38%worse thanD332. MeanE2E4.667260s slightlybetter; P95TPOT
52.747ms worse. Timing-only3867/4000 versus3901,NOTformalSLO.2009GPU requests
prove newcoreturntrigger,notcausal saving. Remote132/131480060B/packing0;
4037samples/peak20462186496B/minhost92627652608B/high,max,OOM,swap0.

Observed U delta365.564940GPU-s equals5.105955envelope increase plus360.458985
less released-gap time. Scale-up/down9/6->4/1, runtimewait344.607->347.121ms.
Do not accept D333 as a verified performance gain, rerun unchanged for a winner,
or infer that deleting TTFTguard/reducing replicas automatically meetsSLO.
NEXT checkpoint then return to resource operating-point/queue-pressure evidence
and history; choose ONE falsifiable7B step before changing configuration. No
newcandidate selected in this audit. Final qualification andlaterstages remainopen.

## HISTORY — D335 offline comparison launched

2026-10-08 07:11+08: D335 finished4000 successful native-contract terminals,
no interrupted requests, launch pass and physical GPU release confirmed. Service
scope removed; exact-owned replay scope empty/stopped. Both remote services and
monitor stopped after measurement, PID0/portsabsent; downloaded transfer/monitor
logs match remote SHA. No GPU/performance run remains. Do not relaunch D335.

Raw result SHA a3b6ee044db0ba46915e94ad1de8e67f638b3b0f188fed7eaac1e69078ffaa97.
Physical summary U15748.053153870977GPU-s, N_correct unknown. Preliminary resource
regression versus D332 U15382.488214008044 is retained; no success claim from the
short D334 prefix. Reusing existing full/context analyzers in a3/4GiB,swap0 offline
scope; no code/config/profile/remote changes to serving, no figures or new inference.
User's once-published delivery-cache approval reconfirmed: D78/D80 cache already
exists and was reused, not regenerated. Request-path packing remains forbidden.
Free disk159992184832B is below150GiB: no new heavy/GPU run until fresh gate and
authorized evidenced storage recovery; thresholds remain unchanged.
NEXT: complete audit, interpret full wait/control/lifecycle and trigger evidence,
save negative result and checkpoint, then choose the next single7B research step.
Numerical/commonwarm/Resident/old-new G1/G2 and3B/baselines remain OPEN.

## HISTORY — D335 full4000 W0 launched; D334 is complete, no repeated smoke

2026-10-08 05:58+08: continuing the unchanged D333 serving candidate, ordinary
4000-request W0 launched with registered runner case d335 in tmux tc-d335-full1.
Raw: results/ieee_tc/p2_backend_qualification/d335_20261008/7b_full_w0.
Actual HEAD e0b1060d906ee401bba5468c4330d6cb0b114c43 independently matches V2.
Source SHA/profile/model checks passed, only3 exclusive result/cold-cache paths
differ from D332. No prefix limit or profiler; existing trace/artifacts reused.
Prelaunch scope peak514818048B/events0/swap0, exactly owned-empty and stopped.
Fresh gate disk161100333056B/memory112859860992B, no threshold relaxed.
Remote frozen published services started; link1000/full, both authenticated
health prepublished_gzip_v1/artifact_timing_v2.7B clock45a7f67912d04ded9c47c7c3863abb1c.
No packaging/cache regeneration or baseline/3B work. No figures. Do not edit
serving/configuration or run heavy analysis during this performance run.
NEXT: monitor actual launch/watchdog, complete replay, release exact owned local
and remote resources, validate native counts/timelines/U/remote/trigger against
D332, save all regressions/unknown correctness, then decide next7B mainline step.
Full/numerical/commonwarm/Resident/old-new G1/G2 are not yet qualified.

06:00:42+08 monitor: replay/service are actually running, not merely a submitted
launch. Latest service report35arrived/33success/0failure,3natural scale-ups,
4live runtimes.148 watchdog samples; current service13.764GiB/host available
91.279GiB,high/max/OOM/swap0,foreign compute0,abort/warningnone. Partial warmup
latencies (mean2455/P9512709ms at33 requests) are NOT complete-run estimates;
retain startup regressions and wait for the full4000 before judging D333.
tmux tc-d335-full1 remains active; no tool session pending. No new code changes
or push during performance measurement. Existing e0b1060 backup contains the
exact serving candidate/config/runner. Finish this run; do not relaunch it.

06:44:26+08 verified-wait continuation: SAME tmux session/pane PID1086842 live,
watchdog2737 samples,3030/4000successful terminals/0failure (3031arrived). The
run has crossed the first six500-request rotation boundaries and recorded a
natural4->3->4 replica sequence. These are partial observations, not G1/G2.
Peak service18.709GiB/latest host88.684GiB,high/max/OOM/swap0,no warning/abort.
Read existing D324/D327 lifecycle evidence and queue-display/control code only;
queue_visible is cumulative display, not itself IEEE control queue_depth. Do not
infer a queue bug from that label. No edits, new analysis, policy/configuration
changes, numerical repeats, baseline or figures. Plan/V1 SHA unchanged.
The temporary foreground polling cell was stopped after the observation window;
the experiment, independent watchdog, replay and frozen remote services continue.
NEXT still finish D335, then exact resource closure and offline comparison. Do
not confuse ending an observation window with experiment termination or restart.

## HISTORY — D334100 COMPLETE/audited; next D335 full4000, no more short tests

2026-10-08 05:53:23+08 BACKUP VERIFIED: local HEAD and independent remote V2 both
e0b1060d906ee401bba5468c4330d6cb0b114c43.12 exact staged paths,7 curated hashes,
17 archive members,syntax/diff/secrets pass;user manifest unstaged and unchanged.
No active experiment/GPU/remote service or pending tool process. Cleanup and
D334 analysis are DONE. Next D335 full4000; do not turn next cycle into another
closure/backup audit. Source-serving candidate7458783 remains unchanged by the
analysis-only e0b1060 commit. Full goal remains active; no result claim closed.

2026-10-08 ~05:52+08: D333/D334 is real mainline progress, not another backup-only
cycle.74587830983bd28fd4791ed8ffeec9192262c56d was pushed/independently verified
before the actual D334 run (05:44:46 start).100/100 success/native token counts,
all10 binding checks100,TTFT/TPOT recomputation error0;45GPU requests carry exact
single_native_core_turn_v1,18HOST/8NVMe/29Remote keep separate paths.4 outputSHA
changes vsD331 attempt2 retained,N_correct unknown; not numerical/G1/G2 qualified.
Mean/P95TTFT804.848/2012.726ms,U618.357456GPU-s,meanTPOT28.152ms (worse than
prior prefix27.775). Prefix means/quantiles are not Full4000 claims;noCI/figures.

29fetch/50650865B,matched remote IDs/bytes/SHA,packing0;network1000/full before
start.263resource samples,servicepeak19400253440B/minhost93944946688B,events/swap0.
Physical release confirmed;exact current remote services/monitor inactive/PID0/
portsabsent. Local aux and offline scopes empty/stopped,offlinepeak91394048B.
Existing smoke analyzer extended for d334 and strict trigger;15tests pass.
17-member13910B closure archive and curated hashes sealed. No activeGPU/remote
experiment remains. Last disk161112862720B barely above150GiB;fresh gate required.

NEXT: checkpoint this closure, then D335 unchanged candidate ordinary4000W0
against D332. Use registered runner case d335/config20261008_d335_7b_full_w0.yaml.
Do NOT repeat D334, D328 profiles, D295 numerical blocks or D332 analysis. Do NOT
start3B/baselines. Whole goal remains active;old-new/commonwarm/Resident/numerical
and actual7B/3B targets remain open. See D333_NATIVE_PENDING_GPU_TURN.md.

## HISTORY — D333 native pending/GPU core turn CPU-verified

2026-10-08 05:44+08: one new7B candidate implemented. D184 had fused only the
outer controller/frontend exchange; D333 fuses the remaining two native utility
calls for selected GPU sources. Original tokenizer/descriptor, register-before-
protection, exact-source checks, unknown ownership and cancellation remain.
HOST pending-load update participates in the routing key and is NOT reordered.
No nine-equation/profile/config/SLO/remote change. See D333_NATIVE_PENDING_GPU_TURN.md.

113 focused and476 expanded CPU tests PASS (overlap, not589); normal error/unknown
receipt/source conflict and cancellation are covered.3/4GiB,swap0,events0; exact
test/prelaunch scopes stopped. Initial non-UUID launcher name refused before tests;
record retained, no guard relaxed. D334 config/profile check passes: only3 output/
cold-cache paths vsD332, actual model and160/24 classes unchanged. Candidate is
NOT GPU/numerical/performance-qualified. Curated evidence sealed. Backup then
D334100 original W0 prefix using real prepublished remote, then D335 ordinary4000
if valid. No D328/D295/D332 repeats, no figures, no3B/baselines yet. Fresh launch
resource gate mandatory; old-new/commonwarm/Resident/numerical/G1/G2 remain OPEN.

## HISTORY — D332 full4000 COMPLETE and audited; no rerun/figures

2026-10-08 05:28+08 BACKUP VERIFIED: local HEAD and independent remote V2 read
both7b30792518a6cf9707fc5ae22e8db5585114f7d4.23 explicit staged paths,17 curated
hashes,38 archive members,syntax/diff/secrets checks pass;user manifest unstaged
and unchanged. Final verifier peak35004416B/events0/swap0, exact-owned-empty and
stopped. No active GPU/performance/remote experiment or tool process. Before this
status note only the user manifest was tracked-dirty. Raw backup receipt retained.
NEXT IS THE NEXT7B RESEARCH QUESTION, NOT another D332 audit/backup-only turn.
Use existing entrance/runtime wait evidence, historical changes and current primary
sources; avoid repeating finished profiles or numerical blocks. Storage only barely
cleared150GiB: fresh launch gate remains mandatory. Goal active; all later stages
and formal qualification still open.

2026-10-08 ~05:20+08: actual run ended4000/4000 successful native terminals,
0failed, all ten binding checks4000, stage error0. GPU physically released and
service scope removed; exact remote units inactive/PID0/portsabsent after release,
downloaded logs match remote hashes. Local replay auxiliary scope empty/stopped.
Full/context analyzers both exit0;75 existing tests pass in bounded offline scope
55b68817... invocation04794b97b33b478391cda418a1c592e4; peak1153286144B/events0.
Offline scope independently verified empty/owned and stopped;21-member49102B
closure archive round-trip SHA verified. Curated hashes sealed; checkpoint push next.

Same complete inputs as D326;186 output token SHA changes retained, N_correct
unknown. Mean/P95 TTFT1263.168/3142.674 ->1175.624/2715.336ms; physicalU
15690.824152->15382.488214GPU-s;P95TPOT53.023->50.413ms. Timing-only provisional
3842->3901/4000, NOT formal SLO. Startup53.177s is worse; older D323 E2E/P99
still better. Planning publication2931/seeded68507/hits8408 actual triggers,
not avoided RPC or causal latency counts. Remote132/131480060B/packing0.
Servicepeak21662269440B/minhost93462282240B/high,max,OOM,swap0. No figures/CI.
See D332_COMPLETE_W0_DECISION.md. Source/plan/V1/old data and user manifest protected.

Storage closure: removed only two verified public download archives,530714624B;
environments/extracted packages/compiler caches/models/results untouched. Official
libcublas API SHA was absent; first audit refused, second streamed actual official
bytes and confirmed SHA before removal. Initial archive-summary path typo preserved;
validated2 receipt complete, all four storage scopes empty/stopped. Observed free
161164591104B is only just above150GiB, not a future launch gate guarantee.

NEXT: next single7B question
from retained entrance/runtime wait boundaries and current official sources.
Do not repeat this audit, D331 short runs, D328 profiling or D295 numerical blocks.
Keep D330 candidate; performance improvement observed, formal G1/G2/old-new/common
warm/Resident/numerical qualification OPEN. No3B or baselines yet; goal active.

## HISTORY — D332 ordinary full4000 monitoring (now complete)

2026-10-08 04:37:55+08 VERIFIED WAIT: same live tmux and service invocation,
terminal2109/4000 successes,0failures. Sample2033 peak19945795584B, high/max/OOM0,
no abort. Disk159448276992B: still above current-run stop floor, below next-launch
floor; no cleanup during measurement. Read-only observer3166 ended, actual run
independently confirmed active afterward. No new experiment, code, config, remote
operation or figure. This is the continuation of the same verified monitoring task;
full plan already read in this active context and unchanged SHA rechecked.
Next remains complete4000 -> physical release -> owned remote stop/log closure ->
full/context analysis vsD326 -> evidence-backed next7B action. Do not claim G1/G2
qualification from these partial success counts. Goal active, not blocked.

2026-10-08 04:23:39+08 VERIFIED WAIT: independently rechecked same active
tmux/service invocation after ending read-only observer cell3162 (not the run).
Terminal journal1081/4000 successes,0failures. Sample1188: hostavailable95915139072B,
servicepeak19300335616B,swap0,high/max/OOM0,no abort. Disk159956058112B is below
150GiB NEW-heavy-launch floor but above100GiB current-run stop floor. Do not lower
either threshold or clean during measurement; after release/audit, inspect only
owned reconstructible caches and record any safe reclamation before another run.
No new performance experiment completed, no code/configuration/remote changes.
Full run still has planned arrivals through about05:10:40+08 plus drain/release.
Next remains monitor this exact run -> closure -> existing D326 comparison path.

2026-10-08 04:15:43+08 VERIFIED WAIT: same live tmux/service invocation checked
repeatedly, terminal journal528/4000 successes,0failures. At sample719 service
peak18870632448B/hostavailable96286015488B, high/max/OOM0, no safety abort.
Read-only observer cell3156 was ended after polling; experiment tmux and owned
service remain ACTIVE (independently rechecked), not stopped or restarted.
No new serving changes, figures, profiling, remote operations or heavy work during
measurement. Remaining planned arrivals extend to about05:10:40+08, followed by
drain/release; this is not a completion deadline. Next step is the same live run,
not a replacement run. The preceding turn made progress (D331 sealed, D332 launched);
this turn is a verified wait for actual4000 evidence, not a new completed experiment.

2026-10-08 04:03:33+08 ACTUAL launch in tmux `tc-d332-full` through the existing
runner `d332`. Backed source1134455c146b7cda8a0787b35744e07d6452b49f independently
verified on faaslora_origin/V2. Twelve explicit paths;24 tests;seven curated hashes;
credential/diff/config checks pass. User manifest untouched/unstaged. Test scope
9a3a2d22... empty/stopped, peak496025600B, events/swap0.
Fresh gate disk161652736000B/MemAvailable112965509120B. Protected147-entry seal
has only previously documented baseline replay difference, not an all-unchanged claim.
Replay receipt confirms4000/4000 source42/W0, no prefix or detailed observer;
same D326 policy/profile/60s notice, D330 candidate plus final counter export.
Service cb8e3f9908ea46bc82fc0faf35839da0 invocation2f5f5f2592ef40559b2138d929b344c8;
aux01ecb60b13994a49954b6c4c0934e927 invocation13b503a1e64842cab0fa4bad3e729368.
Guard allow_exec=true; production_launch_authorized=false is expected development
identity, NOT formal qualification. Real allocation/ingress observed; no result yet.
Remote published delivery unchanged; startup1000/full. Services3b/7b invocations
c0482664740240c3a541b56fd221e5f6 / 9c737a91fec44fa7bbf7390772b6cdad;
monitor d332-full1 invocation37266b0a3731471fbab348cd78b02f9b.
7B clock remote-process-monotonic:1106bcf2cf934df29e987e73654810fa.
Remote log directory d332_20261008_full1. Raw D332 start/health/gates and exact
owned stop script saved (stop script PREPARED, not executed during measurement).
At sample83 host available107706286080B/servicepeak7120441344B/events0, no abort.
Early service log subsequently confirms12/4000 completed,12ok/0fail (partial only).
Initial startup waiting is retained, not subtracted or treated as a complete result.
NEXT monitor full completion -> actual GPU release -> owned remote stop/download SHA
-> existing full/context analyzers extend only D332 identity/hash -> compare D326.
Do not treat startup/partial completion as an outcome, rerun, tune, or draw figures.
No more short tests; no external baselines. G1/G2/numerical/common warm/Resident
remain open;7B then3B then baselines. Goal stays active.

## HISTORY — D331 attempt2 qualified; D332 full4000 prepared

2026-10-08 ~03:58+08 PROGRESS: actual attempt2 complete/released and audited.
100/100 native quantity/ten binding checks; planning publication163/seeded2287/
hits162 observed. Selected-conflict invalidations3 retained, no failed requests.
These counts are NOT RPC or latency savings. Mean/P95 TTFT900.317/2290.801ms,
P95 TPOT39.398ms, prefix U711.198790GPU-s; numerical N_correct remains unknown.
Remote29/50650865B/packing0, startup1000/full; exact units stopped, portsabsent,
downloadSHA matched. Servicepeak18945822720B/minhost93540143104B, events/swap0.
No figure, CI or G1/G2 claim. First run and all output changes retained.
D332 full config + runner case prepared: same D326 settings, only3exclusive
paths; same D330 serving candidate + post-shutdown counter export. NEXT actual
ordinary4000 after safety gates/checkpoint, not more short-run optimization.
Audit scope4da2a1a8... empty-owned/stopped; closure and seven curated hashes sealed.
24 existing launch/export/timing tests passed before D332 checkpoint; no serving
code or parameter changes in this checkpoint (runner identity + exclusive paths).
7B->3B->baselines mainline intact; full comparison target D326, not D322 prefix.

## HISTORY — D331 attempt2 launched (now complete)

2026-10-08 03:52:43+08 actual launch in tmux tc-d331-smoke2, existing runner
d331-attempt2, source01d390de3cf06d12cca41619c44d234200f7775d independently backed.
Fresh gate disk161703772160B/MemAvailable112892461056B, protected seal only known
baseline replay difference, config only three exclusive paths. Source42/W0 first100,
same D326 policies/profiles/60s notice; no detailed observer or new input.
Service d9ac19117bd84beab443c5b291144900 invocationc5a81fad75214327bc6d28b92c089032;
aux14565b2901bf4bbbaa91f66c9a9e7028. Guarded launch allow_exec=true.
Remote unchanged published delivery, startup1000/full; 7B invocation
6a85d4bbc0be407aa3d41f19486f7a79, monitor ebaee053d7e74b69ae940aec2f4abc0a,
remote output d331_20261008_smoke2. New receipts under raw d331/attempt2/.
Do not relaunch on observation timeout. NEXT complete -> GPU release -> owned
remote stop/log SHA -> existing analyzer --campaign d331-attempt2 -> if native
and positive planning counters pass, prepare/launch ordinary4000 D332 vsD326.
No more short repeats without new error/evidence. No figures or external baselines.

## HISTORY — D331 first run complete; counter export repaired

Backup verified2026-10-08T03:50+08: local HEAD and independently read remote
V2 branch both01d390de3cf06d12cca41619c44d234200f7775d. 16 explicit paths,
18 tests,7 bundle hashes,30 archive member checks,8 executed-source bindings,
29 evidence references,syntax/diff/credential checks pass. Actual D331 serving
source remains af0b771; analysis-time source and post-shutdown export repair
are separately identified, not retroactively claimed as executed.
Final verifier first rejected its own literal private-key scan marker (no secret
exposure/output); marker construction corrected, pre-publication rerun passed.
Both final scopes3774c231.../81d4c9b5... peak127021056/127029248B,
events/swap0, empty/stopped/inactive. User manifest unchanged/unstaged; no live
GPU/tmux/remote experiment. Disk161704726528B at last read, not a future gate.
NEXT ACTUAL d331-attempt2 after fresh gates; do not repeat audit/backup-only work.
Attempt2 has NOT launched. Raw remote scripts must use NEW attempt2 paths/units,
not overwrite original D331 receipts. Main source plan and frozen V1 unchanged.

2026-10-08 ~03:47+08 PROGRESS: actual D331 completed and released GPU.
100/100 native contracts and ten control bindings pass; mean/P95 TTFT
967.853/2726.455ms, P95 TPOT31.667ms, prefix U797.567960 GPU-s. Numerical
N_correct unknown. Remote29 transfers/50650865B, packing0; startup1000/full.
Servicepeak19428937728B, min host93287440384B, high/max/OOM/swap0.
Exact owned remote services inactive/PID0/portsabsent; downloaded logs SHA match.
No live tmux/GPU; auxe2274f94... empty/stopped, cleanup receipts retained.

Mechanism gate NOT passed: final outcome omitted existing _ieee_route_cache_stats.
First audit failed KeyError after valid native/binding checks; raw audit1.log and
partial smoke1 retained. Same-run smoke2 explicitly says unobserved_export_missing,
qualification_pass=false. Ordinary cache hits are NOT substituted for planning hits.
Only repair: export existing counters in final post-shutdown outcome field list.
No online policy/parameter/snapshot/RPC change. 18 tests pass incl real finalizer
success/error/drain export; missing counters remain unknown. Offline scopes
51c6b8f2.../b31c7d2f... peaks94003200/497688576B, events/swap0, empty/stopped.
See D331_PLANNING_PUBLICATION_SMOKE.md. D322 same-input context has worse current
TTFT/U, but intervening lifecycle repair means it is not a causal D330 contrast.
Seven output SHA differences retained; no improvement/formal G1/G2/figure claim.

NEXT backup tested export repair, then ACTUAL existing runner d331-attempt2 with
fresh plan/resource/remote gates (exclusive config prepared). One necessary100
rerun for lost observation, same serving candidate; not another audit-only turn.
Actual positive planning publication/seeded/hit required. If qualified, ordinary
full4000 unchanged D330 candidate vsD326 next. 7B->3B->baselines mainline intact;
numerical/common warm/Resident/new-metric acceptance OPEN; goal active.

## HISTORY — D331 launched (now complete)

2026-10-08 03:37:58+08: launched existing guarded runner d331 in tmux
tc-d331-smoke, backed serving commit af0b7717c8c79db5fbd8b0e61532ea0a1976e5d0.
Fresh source/hash/remote-branch/resource gate passed: disk161763442688B,
MemAvailable113043435520B; protected seal only known baseline replay difference.
Service scope2fe841a7c5b24c13b7e383085c01da01, invocationbdce9aee9c9f4c0fa350c2d9ffed9d85;
auxe2274f94accd40089b45fbda626f4eba; actual launcher allow_exec=true.
Remote unchanged published delivery, startup1000/full; 7B service invocation
32e963a31ece4f309ea11e1299bdfc52, monitor762b324c234f40dc9e5401b5eed12706.
At sample22 servicepeak1955422208B, events/swap0, no safety warning.
No performance/qualification claim yet. Do not relaunch on a poll timeout.
NEXT finish this run -> actual GPU release -> exact-owned remote stop/logs ->
native/binding/timing/source-publication audit -> same candidate ordinary4000
only if qualified. No figures, 3B or external baseline; goal remains active.

## HISTORY — D330 tested candidate; D331 prepared

Backup verified2026-10-08T03:33+08: local HEAD and independently read remote
V2 branch both af0b7717c8c79db5fbd8b0e61532ea0a1976e5d0. 16 explicit paths,
485 tests,6 bundle hashes,26 source references,syntax/diff/credential checks
pass. D329's earlier instance_pool source verified at bbbcf429, not falsely
claimed unchanged after D330. Archive members verified against raw sources.
Final verifier peak106463232B/events0/swap0, exact-owned-empty/inactive.
Before this post-push note only user manifest remained tracked-dirty; never
staged. GPU census empty; no active experiment/tool job. NEXT actual D331
qualification after fresh gates, NOT another backup-only or dependency audit.

2026-10-08 ~03:30+08 PROGRESS. Implemented ONE candidate: initialized planner's
already measured native graph also seeds the existing confirmed-route scopes.
Original identity/time retained, no extra RPC, no TTL/physical-inventory cache,
same live planner/budget and selected-copy atomic checks. HOST invalidation and
D316 GPU-copy rule unchanged. New projection removes global HOST union from
request scopes. Publication/hit counters added, not latency-savings estimates.
See D330_PLANNING_SOURCE_PUBLICATION.md. No new GPU run/performance claim/figure.

485 related tests pass60.858s incl actual planner/collector composition. Initial
81-test attempt has3 fixture/assertion failures retained (two object identity
assertions, one pinning mutation was no-op); only tests corrected. Actual scopes
c56f4057.../f6119e46... peak1887268864/1919430656B,events0/swap0,empty/stopped/inactive.
D331 config plus existing guarded runner `d331` prepared: originalW0 first100
indices, D326 controls/model unchanged, new exclusive paths, no detailed observer.
Prelaunch production loader160/24profiles + exact D298 child-model match pass;
disk161765138432B,MemAvailable112507600896B, protectedseal onlyknownbaseline
replaydifference. Preflight scopeb5b65d76...peak514015232B/events0/swap0,empty/stopped.

NEXT: backup tested candidate, then actual D331 via existing runner after fresh
plan/status/resource/remote checks; not another analysis-only/backup-only turn.
Need100/100 native contracts/bindings/timing, real GPU release, actual planning
publication/hits, remote packing0. Then SAME candidate ordinary4000 W0 vsD326,
all failures/retirements/output changes retained. Short run cannot prove G1/G2.
No remote service started/changed this turn. D78/D80 once-onlycache already done.
7B->3B->baselines->comparison->ablations/sensitivities remains; numerical/common
warm/Resident/old-new new-metric acceptance OPEN. Goal active; no figures.

## HISTORY — D329 dependency tests complete; measured planning publication candidate

2026-10-08. 32 CPU tests pass (6 new tiny real-tensor witnesses plus existing
HOST footprint/projection checks). Owner epoch/model ID alone fails to detect
inner tensor/storage/alias/representation changes. Pinning flip is MOCKED, not
real cudaHostRegister; no claim these changes occur in ordinary workload.
Reject epoch-only physical inventory cache, not all controlled-lifetime designs.
No production change/GPU run/remote operation/performance gain in D329.
CPU scope c912c63246284a72ac2101d0de587caa, invocation69f01c77bdbe43e78d6fc92497c01385,
inode20218224, peak496173056B/events0/swap0, exact-owned-empty/stopped/inactive.
Data in d329_host_dependencies; plan/V1/user manifest unchanged, protected seal
only known baseline replay SHA04f3eb52... difference. No figures; goal active.

NEXT D330: investigate reusing initialized planner's already freshly measured
native source graph for existing confirmed route scopes. D289/D301/D316 are
already implemented, not new candidates. Preserve actual capture identity/time,
selected-source atomic validation, full fresh planning/budget checks; no extra
RPC, TTL or forged footprint. First bounded tests, then actual replay only if
causal publication and resource gates pass. Do not repeat D328/D329 audit.
7B remains unqualified; then 3B, baselines, comparisons, ablations/sensitivities.
Common warm/Resident/numerical/old-new new-metric qualification remains OPEN.
User's once-only delivery-cache approval already fulfilled D78/D80; no rebuild.

## HISTORY — D328 complete; current native HOST observation is the next study

Checkpoint backed up 2026-10-08T03:02+08: local HEAD and independently read
remote V2 branch both bbbcf429107a4d8743305326824b93521d62a8c0. 14 explicit
paths, 25 related tests pass; 71 source and 7 analysis references verified,
12 bundle hashes checked. CSV uses the analyzer's standard CRLF; whitespace
check passed with cr-at-eol recognition, without rewriting measurements.
Credential-pattern checks include the 9,252-byte analysis-source archive.
Final verification scope inactive, peak496934912B/events0/swap0. GPU census empty.
52 historical active scope records inspected read-only: all empty, none populated;
not stopped or misreported as running experiments. User manifest unchanged/unstaged.
No active experiment or tool job. D329_HOST_STORAGE_DEPENDENCY_AUDIT.md records
why pinning cannot simply be omitted: it feeds the HOST service/preparation class.
NEXT continue that one dependency/candidate task, not another D328 audit or
backup-only turn. This post-push tracker note is later than the sealed checkpoint.

2026-10-08: the one 1,000-request diagnostic and offline audits completed.
Native quantity/ten binding contracts 1000/1000, failures0; numerical N_correct
unknown, formal G1/G2 still OPEN. Mean/P95 TTFT1039.602/2344.315ms,
mean queue/service463.476/576.126ms, P95 TPOT65.589ms; observed U4176.348493
GPU-s, six allocations released. Detailed observer included; not an ordinary
Full performance comparison. Keep late unused activation and all observations.
Remote60 matched transfers/74953014bytes, packing0; 1000/full negotiation at start.

1403 complete control attempts: mean total119.264ms; pre-native62.946ms,
native body11.905ms, frontend-ready to parent21.526ms. These are RPC spans,
not request TTFT or disjoint aggregate CPU time. Business core stacks repeatedly
show host inventory tensor.is_pinned(), not token-event add. Sampling is GIL-biased,
not CPU-time percentages. Current evidence locates work; it does not establish
the entire cause or a performance gain. See D328_CURRENT_PATH_DIAGNOSTIC.md.

Service peak19472388096B, min host93709967360B, high/max/OOM/swap/warnings0.
GPU released, tmux ended, exact owned remote units stopped and journals SHA matched.
Remote stop's trailing rg check failed127 (rg absent); separate readback succeeded.
Both offline scopes and run aux verified empty and stopped; final caller scope
92741856d81c4abc907cc3f5b239aefd peak15417344B/events0/swap0. No serving changes.
Curated analysis1 under d328_current_path; raw closure receipts retain all failures.

History D180/D189/D193/D194/D202 reviewed: scoped footprints, static descriptions,
planning projection and view-table removal already exist. Do not redo them or
remove an already-removed duplicate pinning query. NEXT one owner-lifecycle
HOST-observation dependency study using current code/official source, then one
qualified candidate and ordinary full replay. No TTL confirmed-state cache,
budget weakening, repeated D328/D327 diagnostic, figures or external baselines.
Backup this audited milestone with explicit paths; user manifest never stage.
7B ->3B ->baselines ->comparison ->ablations/sensitivities. Numerical/common warm/
Resident/old-new new-metric acceptance remains OPEN; goal active.

## HISTORY — D328 launched, now complete

2026-10-08 02:12:40+08 launchedtmux `tc-d328-prefix1`, backedsourcecommit
24bf5c9976cc7cbaf9d056ccebeac99d39f26edf, independentlyverifiedonremoteV2branch.
Guardedexecallowtrue; original4000source42/W0first1000indexview,traceSHAefb90325...
isfixed. Actualservice35d0769692944320938140937dfe517b invocation786febd56336408d81d0ccfc0d234f44;
aux459360e636af4f7285d72f200b5d58d4. Observerfileswritten; firstrequestscompleted.
AtresourceSample79:hostavailable106961186816B,servicepeak7123456000B,
high/max/OOM/swap0. ThisisinitialhealthNOTfullresult,performanceorformalSLO.
Remotehealthpublished/clock8bc15e2cece54d578ae59e7c4d2cd8b0, startup1000/full;
3b/7b/monitoractualIDsandpaths inrawrunning_checkpoint1.json andremote_start.log.
NOremotechanges/cleanup/hashscanorotherheavyjobduringmeasurement. NEXT waitfor
thisrun,localGPUreleasefirst,thenexactownedremotestop/receipts,existingstackand
control-boundaryanalysis. DoNOTrelaunch/repeatD327 orbackup-only. Goalactive.

## HISTORY — D328 prepared and safety checked

2026-10-08 ~02:12+08 PROGRESS. Recovered1,444,200,448B byremovingONE
publicPyTorchdownloadarchiveafterofficialSHA/size/HEAD/reference/open/linkchecks;
installedandextractedpackages/resultsuntouched. Diskbackabove150GiB; no gatechange.
Twoauditedcleanupscopesinactive,events/swap0. D328reuseofD326configandD200observer
isready: only3exclusivepaths,originalW0first1000indices,160/24profilesunchanged.
13tests10.570spass,CPUpeak515080192B/events0/swap0,scopeinactive.
Prelaunchactualjointloader/model/protected/preflightpass; usermanifestunchanged;
147sealonlyknownbaselineSHA04f3eb52...difference. Config/runtime/metric unchanged.
Next: backuptestedcheckpoint, thenstartfrozenremotesandtmuxD328; do notstopat
anotherpreparation-onlyturn. NoGPUrunyet, nofigures. FollowD328docforclosure.
Numerical/commonwarm/Resident/old-new/G1G2remainOPEN;7B->3B->baselineorderunchanged.

## HISTORY — D327 closes lifecycle-only explanation; 7B steady-path measurement next

Backupverified2026-10-08T02:01+08: HEAD and independentremote both
e424881aeb9770f44b9c37a8cbf19fdc65ad78a5;17explicitpaths, credentialchecks clean.
Beforethispostpushnote onlyuser manifesttrackeddirty. No activejob/GPUservice;
bothCPUscopesinactive. NEXT is diskrecovery/currentboundeddiagnostic below,
not another D327 analysis/test/backup-only turn. Goal remains active.

2026-10-08 ~02:00+08 PROGRESS. Reused sealed D323/D326 full4000 results with
the existing D319 event-defined common-cohort method; no inference, remote,
serving/config/profile/threshold changes or figures. Explicit registered pair,
source SHA gates and fresh output only.63CPU tests pass1.507s; fullanalysisexit0,
scope d3cf52ba149b4331a3c30121af82042b peak1150197760B/events0/swap0,
3/4GiB CPU2,3,26,27; exactownedempty scope stopped/readbackinactive.
Finalcheckpoint22source/9analysisoutput/14bundleSHAchecks pass,syntax/diffpass;
verificationscope124a60e149604aad8a3d7464dcc85ccf peak30908416B/events0/swap0,
empty/stopped. GPUcomputeempty; disk160669499392B (<newheavy150GiB).

After BOTH last engine-ready (common1366.513851s), same2703requests:
P95TTFT2951.051->3350.996ms(+13.55%), mean1230.998->1334.201ms;
meanadmission+62.783ms/service+40.420ms. Provisionaljointmiss80->104;
P95TPOT50.603->48.502ms. Startup-union333requests P95TTFT3308.344->2430.856;
outside3667requests2773.251->3176.239. All172outputSHAchangesretained.
NotnumericalN_correct/formalSLO/CI orcausalretirementestimate. Earliercache/
routinghistory canpersist; no claimthatretirementhasnoeffect. Thisrejects
theoverstrongexplanationthattheentireregressionisonlycurrentstartupscarcity.

No lost-release-wake demonstrated: after539.732 no_free_device, latercontrol
samples haveactive<=8/cap16,queue<=2,TTFTbelowupper until570.433active9;
no_action follows IEEEscore. No retry/timerpatch. D325 lifecycle remains.
Read D319/D324/D326, D139/D140/D143/D187 history andcurrentcode; pureplanner
alreadyoffloaded/staticdescriptionsalreadyreused. Ray2.59,HPA,Pythonasyncio,
vLLMofficialCPUanalysischecked; do notimporttheirrulesorgainpercentages.

Data: d327_drain_lifecycle/analysis1 (completegroups,8000intervalrows,provenance),
execution_receipt.json; doc D327_DRAIN_LIFECYCLE_COHORTS.md. Once-onlycache
approval already fulfilled byD78/D80: no rebuild/newunpackedpool. Original
147sealrawfalse onlyknownbaselineSHA04f3eb52..., usermanifestunchanged.

NEXT testedcheckpointbackup, thenONEcurrentD326-configuration bounded
python_frames_v1 diagnostic usingexisting1000index/observers tolocatepresent
controller/frontend/core waiting; no repeatD140100Hz/Cwatchdog/oldcomponent
optimizations. Firstrecoverdisk>=150GiB usingauditedrebuildablecache/lossless
archive, neverlowergates/deleteuniqueevidence. NoGPUdiagnosticprepared/launched
yet. Endthislifecyclelogpartitiontask; don'tspendanotherturn repeatingit.
7B->3B->baselines->comparisons->ablations/sensitivities; numerical/commonwarm/
Resident/old-new/newmetricgoalsOPEN. No figures. Goalactive.

## HISTORY — D326 complete and audited; resource/latency trade-off, 7B still open

Backup verified2026-10-08T01:47+08: localHEAD andindependentremote branch both
42e746249ca50c2c0dcb5696f81c501b41c48be7.21explicitpaths backedup; diff,14output
checksums,32sourceidentities andcredentialchecks pass; usermanifestneverstaged.
Beforethispostpushnote onlyuser manifestwastrackeddirty. FourCPUscopesinactive;
verificationpeak32788480B/events0/swap0. GPUcomputeempty. No activeexperiment
or tooljob; no serving/config changes inthisauditturn. NEXT below nowmeans
the7Bcapacity/activation bottleneck task, NOT anotherbackup-only/audit turn.

2026-10-08 ~01:46+08 PROGRESS. D326 original4000 complete/native contracts4000,
failure0, tenbindingchecks4000 each, stageerror0. Allinputs/trace/subset matchD323;
172outputSHA changes retained, numericalN_correct unknown. Full data audits exit0.
MeanTTFT1190.032->1263.168ms(+6.15%), P952836.346->3142.674(+10.80%),
P99+14.47%; P95TPOT53.987->53.023ms(-1.79%); U15727.543212->15690.824152
GPU-s(-0.23%). ProvisionalD170 timing3874->3842/4000(-0.80pp), NOTformaljointSLO.
ComparedtoD318U stillhigher. No winner/CI/G1/G2qualification or figures.

Threeactualownedscale-ins, firsttwo withdrawwith1admittedrequest, allreleased,
noquarantine/unresolved/sourceerrors. Gaps115.647->169.279GPU-s; envelope+16.912,
netU-36.719; accountingidentity notcounterfactualsavings. Aftersecondwithdrawal
newactivationrequested~4.187slater; one no_free_device thenlateractivation.
Do notcallthisprovenTTFTcause oraddtimers. Existingrecords nowcaptured in
d326_full_w0/lifecycle_trigger.json. Remote132/131480060wirebytes/packing0,
sameasD323.4045resourcesamples,peak20616302592B,minhost93612437504B,events/swap0.
LocalGPUreleased andownedremoteunitsstopped, exactreceiptscopied/hashesmatched.

61CPUtests1.428spass; fullauditsactualscope9b0403532db3465c94dfb33cdb057f90,
peak1204662272B/events0. NarrowprojectionfirstwrongOSPython lackednumpy before
dataread; failurepreserved, existingCPUenvcorrected(noinstall), actualscope
e54e1a0693364718aea0bc32b73eb967 peak28590080B/events0. All3scopes checkedempty
andstopped. No inference rerun, serving/config/profile/threshold changes orfigures.
See D326_COMPLETE_W0_DECISION.md; rawoffline_execution_receipt.json recordsall.

NEXT backupthisauditedcheckpoint, thenONE7Bbottleneckcheck/candidate around
service-capacity/activation waiting afteractualwithdrawal, usingexistingrecords,
history andofficialsources. Do NOTredoD326audits/full/prefix/profilecalibration.
D325correctlifecycle retained; itsperformancehypothesisonlymixedsupport. Disk
belownewheavy150GiB: auditedrebuildablecache/losslessarchive beforeanynextGPUrun,
neverlowergates/deleteuniqueevidence. Numerical/commonwarm/Resident/old-new/G1G2
remainopen. Mainline7B->3B->baselines->comparison->ablations/sensitivities retained.
Goalactive; thisiscompletedrun/auditprogress, notblocked oroverallcomplete.

## HISTORY — D326 complete, offline audit started (now finished)

2026-10-08 ~01:37+08 PROGRESS. Original D326 completed without restart:4000
successful terminals/native-count contracts,0failures; numericalN_correct remains
unknown. Service/replay/watchdog exits0/0/0; launchpass, physicalreleaseconfirmed,
servicepathremoved. ActualU15690.824152139015GPU-s, no open leases. No latency/SLO
or superiority conclusion until existing audits complete. GPU compute census empty.
Source result9e1057f50d1cc8dafacfd4f232216cd5979ac6c7e9f2a479dcd26f654295150a.
Remote monitor and both owned artifact units identitychecked/stopped/PID0/success;
portsabsent. Stop shell141 is from trailingls/head underpipefail after stops;
separate readback0 proves inactive state. Rawremote_stop.json preserves both.
Remote journal andmonitor copied andSHA matched; once-onlyD78/D80cache unchanged.
Raw/curated completion_receipt.json created; existing two analyzers extended only
with D326identity/date/healthschema. Tests/fullofflineaudit next in3/4GiBswap0scope.
No serving/configuration/threshold changes, inference rerun, figures orbaselines.
Diskfree160713785344B below NEWheavy150GiB; boundedCPUaudit allowed. Do notlaunch
anotherheavyjob or relax gates. Mainline7B->3B->baselines remains; goalactive.

## HISTORY — D326 original4000 W0 observed live (now complete)

VERIFIED WAIT 2026-10-08T01:12:06+08:00: same tmux tc-d326-full and supervisor
PID3776599 remain live. Incremental terminal journal has3450 successful terminals,
3450 native-contract matches and0 failures out of4000 planned; this is not audited
N_correct or a final performance/SLO result. Watchdog sample3165 has warning=false,
abort_reasons=[], all memory high/max/OOM events0 and service swap0. Host available
95318446080B, service current20169625600B, disk free158878326784B. Disk is below
the new-heavy150GiB floor but above the current-run100GiB stop floor; do not launch
another heavy task. Latest GPU census has no foreign compute or escaped worker.
Two physical release events observed: lease ce3b4107e821445cad49770cfda717f4 at
897710.524354009 and a5959023cf1b439db1b45db2402f57f0 at897769.713167422;
both worker_returncode0, released GPU absent from that event's service-held list,
unresolved census empty. Later new acquire events are retained: this is evidence
of actual retirement and reactivation, NOT net GPU-s savings or SLO qualification.
Do not change policy on partial-run observations. This turn is verified waiting,
not another performance experiment or a new optimization.
User reconfirmed one-time read-only compressed delivery-cache permission; D78/D80
already implemented it. No cache rebuild, remote modification, code/config change,
cleanup, compression, figures or new experiment in this monitoring turn. The
4000-request run must finish and physically release before its offline comparison.
Continue observing the same process; no restart on a poll timeout. Goal active.

2026-10-08T00:19:08+08:00 PROGRESS. Actual tmux tc-d326-full/panePID3776599
python3 supervisor live; runner reached Phase1 preparation. Source808b36f49c8d2a1362fd3ce80017bde5e264e80f,
runner734a51f... andconfig3c5ad20b... rechecked immediately beforestart. Original
4000requests/500adapters/W0 (3963.9sarrivalspan), no diagnosticprefix or newdata.
Productionjoint profile entry160service/24preparation passed; wholeconfig differs
fromD323 only3exclusiveoutput/cachepaths, realchildmodel exactlyD298contract.
Preflight passed:free161932050432B,MemAvailable112429875200B; protected147
rawfalseonlyknownbaselineSHA,user/plan/V1 unchanged. CPUprelaunchscope
0e0a4e6e7790478581f7a9a5e0875dd6 peak499892224B/events0,empty/stopped.

Actualservice primelora-tc-svc-5301b69d226d421788cd039b19c6b961.scope,
InvocationID6c7c1d46975d4436b046f57166a9cd64,high/max72/80GiB,swap2GiB.
Aux primelora-tc-aux-480967520f8b4cbab4eee9b16c8e548a.scope,
InvocationID3ff5f9e68bb64696965ceedd62acf537,high/max3/4GiB,swap0.
Raw results/ieee_tc/p2_backend_qualification/d326_20261008/7b_full_w0.
Sibling start_remote.sh/prelaunch_binding.json/remote_health.json/source_commit.txt
preserve preparation andidentity;existinglauncher d326branch used,not rebuilt.

Remote actual1000/full,unchangedpublishedcache;3b/7b services wereinactive before
start,now ownInvocationIDs d887fcd717384a4bbbb54873a8787ec2 /
6c55fbed79e8418ebf51535c269a6980. Monitor primelora-artifact-monitor-d326-full.service
87dc34f553794a9c86d6e2a2e35ab9dc. Healthclock
remote-process-monotonic:c289f13e41fd4cb38cebfaab50a47319;monitorpath
/home/lab14/primelora_remote/tc/d326_20261008_full/remote_monitor.log.
3Bartifactservice idle only; no3Binference. Do notalterremote duringmeasurement.

NEXT observe sameprocess throughcomplete/release; preserve anyfailure, no restart
onpolltimeout. Then identitycheckedremote stop/copymatchingtransferjournal,
data-only analyze D326 againstD323/D318 with frozenV1 andactualphysicaldrain.
No serving/config/code changes, heavyanalysis, cleanup/compression, nextGPUrun,
calibration,figures orbaselines duringmeasurement. Numerical/formalSLO/G1/G2
remainopen; launch is not a performance result. Goalactive.

## HISTORY — D325 owned scale-in CPU-qualified; next D326 complete W0

Backup verified2026-10-08T00:14:13+08:00: localHEAD andremote branch both
808b36f49c8d2a1362fd3ce80017bde5e264e80f. Eight explicitpaths backedup; staged
diff/credential-pattern checks pass. Before thispostpushnote usermanifest was
theonly trackeddirtyfile. All7D325scopes inactive; noGPUcomputeprocess. Diskfree
161935097856B (~150.81GiB), rechecknew-heavyfloor before D326. Noremote change.
This turn made tested implementation progress; not a wait/blocker and not a
performance conclusion. Nextturn must proceed to D326 actual prelaunch/full,
not another checkpoint-only/log-slicing/smoke turn. Goal remains active.

FINAL CPU CHECK2026-10-08 ~00:11+08:426tests pass55.804s; then6configuration
tests pass1.293s. Final runnerSHA734a51f1789861e1f2e023890057b911ae5b341da0d07bd92ceb73c4b1abc627.
D326config3c5ad20b... only3exclusivepaths differ fromD323; original4000 preserved,
existinglauncher explicitd326 clears diagnosticprefix. NO GPU/remote run yet.
Six exactInvocationID testscopes verifiedempty andstopped; events/swap0; final
testpeak1967718400B. Protected147 rawfalse onlyknownbaselineSHA04f3eb52...,
user manifest/plan/V1 unchanged. Receipt d325_owned_drain/validation_checkpoint.json.
Verification scope79eda4861b5d45bc92e53e3191b9a9d1 alsoempty/stopped,
peak44032000B/events0;11source/logreferences rehashed andsyntax/diffpass.
Tested explicit-path backup pending. NEXT freshsafety/profile/remote checks and
ONE D326 fullW0; do not redo100prefix/calibration/D324analysis or figures.

Earlier checkpoint2026-10-08 ~00:08+08 PROGRESS, no GPU/remote launch or figures. Source plan fully
read and frozen V1 read, hashes unchanged. D325 implements D324's ONE candidate:
withdraw -> drain actual admitted users and published source reads -> owned
cleanup -> confirmed physical return. No control/profile/threshold/formula change.
Normal scale-in and exceptional quarantine share one retirement owner. Draining
live work is not queued work; physical membership remains held until actual ack.
Min/routing withdrawal is synchronous before HOST-query await. Fatal requests
retain failure/device exclusion, cannot self-join retirement. Failed returns are
not silently retried at common shutdown; cancellation joins already owned work.

Reproduction failed old implementation (busy nonprimary stayed running). First
34tests had two old fixture-structure errors, fixed in tests rather than runtime
defaults.39tests pass31.044s;365expanded pass55.766s. Final expanded suite now
was running with added source-query ownership checks in bounded3/4GiB/swap0 scope
primelora-tc-d325-finaltests.scope. Logs d325_20261008/{tests2,regression,finaltests}.log.
At that checkpoint final pass was not yet known; see final check above. No active
GPU context. Earlier scope/resource receipt and cleanup are now verified above.

NEXT: final regression -> preservation/resources -> explicit tested backup ->
ONE D326 original4000W0 with D323-identical policy/model/profiles, new exclusive
paths. No repeated100-prefix/calibration; old100prefix does not cover311s event.
See D325_OWNED_SCALEIN_DRAIN.md. No performance gain/new SLO qualification yet;
7B->3B->baselines->comparisons->ablations remains. Goal active.

## HISTORY — D324 retained-data diagnosis complete; next is owned scale-in drain

Backup verified2026-10-07T23:47:07+08:00: localHEAD andremote branch both
8cd699a2bd4ead7bf26c83f83325f67fc5bcd5e6.12explicitfiles backedup; staged diff/
credential-pattern checks pass; usermanifest neverstaged. Beforethispostpushnote
itwas theonly tracked dirtyfile. All3D324scopes inactive; noGPU/remote taskstarted.

2026-10-07 ~23:45+08 PROGRESS, not a new performance run. Completed the ONE
bounded D318/D323 service/lifecycle check using original4000 each. No inference,
remote change, figures, serving/config change or threshold tuning. User cache
approval already implemented D78/D80:1581215261B/500permodel, do not recreate.

Selected O MAE2453.322128->671.100584ms(-72.65%); signed mean-62.701939->
-110.232014ms retained. Short/long bias reduced, selected-above-minQ2807->1855;
all five load/class/routing contradiction checks0. Sameinputvectors4000 and
all209outputSHA differences retained. No numerical/formalSLO/G1/G2/CI claim.

Physical decomposition is exact, not causal: envelope delta-8.060216GPU-s;
observed release gaps324.251432->115.647227GPU-s, reduction208.604205;
net U+200.543989GPU-s. Both runs2actualretirements, not fewerretirements.
See D324_SERVICE_LIFECYCLE_DIAGNOSIS.md and d324_service_lifecycle/analysis1.

NEXT SINGLE candidate: deterministic reproduction then owned withdraw->drain->
physicalrelease, matching IEEE source line215 (SHA35da2d89...). D48faf5c85
implemented idle-only retirement; current nonidle SCALE_DOWN returns
awaiting_drained_replica but has already resetlow_since. D323observedthis at
311.818820s(queue0/active6/cap32/pending0), nextactualdecision674.883314s.
This is NOT363seconds of proven savings. Do not alter 3-signal law, thresholds,
38s cooldown, profiles, routing formula or SLO; no kill-active/timer-reset-only
patch. Must reconcile new-admission withdrawal, owned active completions,
ready/admitted counts, physicaldevice reservations, planners and shutdown.
Ray2.59.0official graceful-shutdown code/K8sHPA checked for mechanism boundary.
No serving fix yet. Do not re-runD323/calibration or keep slicing these logs.

53CPUtests6.689spass. Testscope e8a66b39d72e442289fd8cef2061c533 peak500260864B;
datascope b2c8e0ceca8140a885efff59e6e95531 peak1489567744B; high/max/OOM/swap0,
actualempty/populated0 beforestop. No activeGPUrun. Final23source/6outputSHAchecks
pass; protected147 rawfalse onlyknownbaseline replaySHA; usermanifest unchanged.
Verification scope6d83d2e67d0b4d73b14190c3b17faed7 peak13500416B/events0/swap0,
empty. Diskfree161963716608B. Receipt d324_service_lifecycle/validation_checkpoint.json.
Explicit-path Git backup follows; no serving/config/profile change to stage.
Mainline remains7B->3B->baselines->comparisons->ablations/sensitivity; goalactive.

## HISTORY — D323 complete and audited; 7B resource goal remains open

Backup verified 2026-10-07T23:29:03+08:00: local HEAD and remote branch both
535949ff5a80a50301d7c266206e8ab51e711c04.18 explicitly selected files committed;
user manifest was not staged. Before this post-push note it was the only tracked
dirty file. CSV CRLF was normalized to LF only; stripped-CR byte comparison
passed and original CSV/manifest were retained in raw with hashes. Numeric
values unchanged; formatter provenance is explicit in context1/manifest.json.
All output SHA checks, syntax, diff and credential-pattern checks passed.
No GPU compute process remains. This goal turn is PROGRESS: complete replay,
actual release, data audits, safe archival and verified remote backup.

PROGRESS 2026-10-07 ~23:20+08: D323 original4000 W0 finished, no restart.
All4000 arrived/submitted/unique successful terminals/native-contract matched;
ten binding checks4000 each, stage identity error0. Service/replay/watchdog0,
native GPU release confirmed, servicegroupremoved, GPU compute census empty.
N_correct remains unknown; no numerical/formal SLO/G1/G2 qualification claim.
Source a790cd59968a0d8d33bc017938467c1f818c5e5c617fb9628ab25a4843b5e991.
Full audits in p2_backend/d323_full_w0/{validated1,context1}; completion receipt
also retained there and in raw d323_20261007. See D323_COMPLETE_W0_DECISION.md.

Versus D318: meanTTFT1292.091->1190.032ms(-7.90%), P953031.176->2836.346ms
(-6.43%), P95TPOT64.226->53.987ms(-15.94%); physicalU15526.999223->15727.543212
GPU-s(+1.29%, worse). D170 timing-only3818->3874/4000, NOT formaljointSLO.
All input identities match;209 output tokenSHA changes retained, arithmeticopen.
TotalU delta+200.543989 comes mainly from arrival-window+202.026954, notcleanup.
132HTTP transfers/131480060bytes matched,pack0.4039resourcesamples,
peak20414410752B,minhost93245304832B,high/max/OOM/swap0.

Remote units identitychecked and stopped AFTER local release; monitor and3b/7b
inactive/PID0/success. Journal clock6fd32ba11d7f42cb9ddef89eea7d01c8, raw copied
to remote_transfers.jsonl and remote_monitor.log; remote_stop.json preserveschecks.
No serving/config changes this turn. Existing offline analyzers extended with
the frozen D323 result identity and flat health schema;5configurationtests pass,
both full audits exit0. CPUanalysis scope d80c17278daa4bb081951acd8a20078d had
peak1087778816B/events0/swap0, empty beforestop. Old full auxiliary also empty/stopped.

Storage: D299/D300/D303 historical main_outcome logs losslesslygzip archived;
source and two decompressedSHA checks match prior audits before uncompressed
duplicate removal. Source/candidate requestresults and oldsubmission unchanged.
Freed1885448860B; free161993199616B atcheck. Compression scope650453d936dc47c49fed60d7564a4b73
peak260530176B/events0/swap0, empty/stopped. Receipt storage/20261007_d323_lossless_compression.json.
Protected147 rawfalse ONLYknownbaseline replaySHA; usermanifest untouched.

NEXT ONE bounded data-only D318/D323 physical-lifecycle and service-estimate
check using existing records; tested audit/scripts/curated backup is complete.
Do not rerunD323/smokes/profile calibration or tune another policy before that
evidence. Still7B first, then3B, thenbaselines; nofigures. Goal active.

## HISTORY — D323 complete4000 W0 running observation (now finished)

Latest VERIFIED WAIT 2026-10-07T22:37:55+08:00: same tmux/PID 3163492 live;
2407 request-terminal records, not audited N_correct. Watchdog sample 2228 has
warning false / abort []; high/max/OOM 0. Service current 19297914880 B,
host available 95549202432 B, disk free 158779609088 B (~147.87 GiB).
The current run remains above the 100 GiB stop floor; no next heavy task until
the 150 GiB start gate is satisfied. Source plan read completely this turn,
including the middle section initially truncated by tool output. Same process
was repeatedly checked live, advancing 1938 -> 2055 -> 2168 -> 2315 -> 2407;
previous turn and this turn are VERIFIED WAIT, not blockers or full-run results.
No serving code/config/remote changes, heavy analysis, cleanup, compression,
new run or figures. NEXT remains the same run through actual release, then
data-only validation and comparison under frozen V1.

2026-10-07T22:30:21+08:00 checkpoint: 1814 terminals, sample 1781 clear;
disk free 159153844224 B. User cache approval rechecked: D78/D80 already published
1581215261 B (~1.47 GiB), both 500-adapter HTTP/content coverages complete;
D323 saved health is prepublished_gzip_v1. Do not recreate the cache, unpack
another pool, or alter the remote service during measurement.

Latest VERIFIED WAIT2026-10-07T22:20:50+08:00: same tmux/PID3163492 live,
1119request terminals; watchdog current resource sample clear,high/max/OOM0.
Servicecurrent18218110976B,hostavailable95930593280B,diskfree161270140928B
(~150.19GiB; near new-heavy-task start floor, not current-run100GiB stop floor).
No heavy analysis, code/config/remote changes, cleanup, compression or newrun.
Latest100service-loglines have noERROR/Traceback/failed; not a full result audit.
This continuation and previous are verified waits on the actual running job,
not blockers. Source plan fully reread (filled output truncation); no scopechange.
Next remains the SAME run through terminal/release, then data-only validation.

Verified-wait checkpoint2026-10-07T22:12:03+08:00: exact tmux panePID3163492
still live(python3),518request terminals, latest independent watchdog resource
sample clear. Servicecurrent17274376192B,hostavailable96297955328B,high/max/OOM0;
diskfree161900462080B. No code/config/remote service change, heavyanalysis, new
run or figure this continuation. Complete source plan reread. Previous turn was
PROGRESS(smoke complete/full launch); this turn VERIFIED WAIT on actual live
run, not a blocker. Do not treat the518terminals as audited N_correct or a full
result. Continue same process; full analysis only after terminal/release evidence.

2026-10-07~22:00, mainline PROGRESS. Result/prelaunch backup local/remote exact
f13002351fb160e4aa6e703e821f4e2e8c3edfeb. D323 actual gate andjoint profile entry
passed; original4000/source4000/W0, no diagnosticprefix. tmux tc-d323-full,
panePID3163492 (python3 supervisor), actual service scope
primelora-tc-svc-c066567257c74357a7b600f992cab6b4.scope,
InvocationID2c5fd59029894f9cabf43f7f111d5231. Auxiliary scope
primelora-tc-aux-65b20ed83355492285e9176a4768622a.scope. Watchdog live/clear,
service reached Phase1 preparation. Raw base:
results/ieee_tc/p2_backend_qualification/d323_20261007; run7b_full_w0.
Beforestart free163067568128B,MemAvailable110324568kB; no GPU compute process.
Full-prelaunch verifier pass,peak501596160B,events/swap0,actualempty beforestop.

Remote 1000/full, unchanged frozen published cache. Own remote unit identities:
3b4e834daaf0cd4d169bf353a83a21929e;7be1424c403a3443e3be6f28cddbc82606;
monitor primelora-artifact-monitor-d323-full.service/42ee2675070a40e8a55ce0bbfa3f1fb3.
Health clock remote-process-monotonic:6fd32ba11d7f42cb9ddef89eea7d01c8; direct
client health JSON has flat fields (not old wrapped health schema). Remote monitor
path /home/lab14/primelora_remote/tc/d323_20261007_full/remote_monitor.log.
Start/health/sourcecommit saved in rawbase. No serving/code/remote changes or
other heavy work during measurement. Goal active; no figures/3B/baselines.

NEXT observe existing run to completion, actual native GPU release and cleanup.
Then stop only these remote InvocationIDs, copy monitor and unique transfer
journal matching health clock. Extend existing analyze_d297_full for D323 actual
source SHA, reuse data-only binding/timing/physical/preparation checks; compare
to retained D318 (and D315 context). Do not extrapolate prefix or claim formal
N_correct/commonSLO/Resident qualification. No new smoke/calibration/code tuning
before complete-run evidence. Usermanifest/plan/V1 unchanged. Only this postlaunch
status note is an own uncommitted edit after testedbackup; do not stage userdirty.

### D322 attempt2 complete and D323 preparation

100/100 native complete; all binding checks pass. MeanTTFT848.836ms vs955.135,
P95TTFT1933.006vs2118.484ms;P95TPOT32.477vs37.949ms. GPU-s717.875820vs710.223238
(+1.08%,worse). One prefix each, noCI/full/numerical/formalSLO claims. D170 timing
99/100 provisional.29remote transfers/50650865bytes/pack0;266watchdog samples,
peak19324653568B,minhost93391679488B;events/swap/abort0. ActualGPUreleased and
servicegroupremoved, remote identities checked/stopped. Source05f4071 backedup.
Offlineanalysis passed,peak88944640B/actualempty. Data d322_length_candidate/smoke2.
NEXT tested result backup and D323 original4000W0, no more short runs/config
changes. Preparation gate running; do not claim full launch until actual tmux/
service handles recorded. Usermanifest/oldresults intact. Nofigures/3B/baselines.

### D322 attempt2 launch record (finished)

Tested backup normalpush local/remote05f4071e611ae5aa76803d9d983512778dd1d42e.
Started tmux tc-d322-smoke2 PID3119060; existing gated-launch/watchdog active,
actual joint entry passed and Phase1 preparation reached. Raw run is
results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100_attempt2.
Remote start/health/sourcecommit in sibling attempt2; clockfcffcd8d65fa4e93b15fb9468191b23b.
Remote units3b/7b InvocationIDs0fcb5c1ce8744311b08c9ea49d481acc/ddac16ddba0f4926a7febe3f1a45b7a0;
monitor primelora-artifact-monitor-d322-smoke2.service/8baa051bd9864e77b321d4b19c387117.
Link1000/full again. No concurrent heavy work or figures. Observe actual finish,
release, stop only owned remote identities and copy matching journal/monitor,
then analyze_d298_smoke --campaign d322-attempt2. Do not relaunch blindly.

### First attempt and correction

Firstlaunch48b6ea9:0/100arrived, no GPU lease; contexts released/servicegroup
removed, remote InvocationIDs checked/stopped. Not a performance measurement.
Actual runner required whole preparation/service contexts equal. D321 legitimately
has separate historical/gap input provenance; no hash rewriting is allowed.
Correction retains independent full payload/context/model validation and shared
backend/resource/model equality, permits distinct calibration input provenance.
No formula/EWMA/control/preparation data change.5targeted tests pass1.874s;
full138tests pass53.897s; peak2242187264B/events/swap0. Actual joint-entry
prelaunch pass160service/24prep; source references/protected147 checked. New
attempt2 paths prepared, original evidence kept. RunnerSHA571c08a2090f6967...
NEXT joint-entry preflight, testedbackup, one attempt2 smoke, then D323 complete
4000W0 if valid. No newcalibration,3B/baselines orfigures. See D322 document.

### D322 preparation history

Only D321 service initialization is changed from D318; model/control/preparation/
generation/inputs/serving formulas unchanged. New exclusive configs and existing
launcher/analyzer extensions prepared. Full-config test caught an accidental
admission-profile-ID text substitution in the new full YAML before GPU execution;
corrected, failed CPU log retained. Revalidation and backup precede smoke.
Revalidation:134tests passed50.029s; production loaders verified160service/24prep
classes, exact D318 configuration comparison and protected147 (rawfalseonlyknown
baseline change). CPUtestpeak2400415744B,verification501542912B;events/swap0.
No GPU/remote launch yet. Do not restart calibration, tune another policy, draw
figures or advance to3B/baselines. See D322_LENGTH_CANDIDATE.md. Goal remains active.

## HISTORY — D321 complete:160classes measured; NEXT one7B candidate smoke/full

PROGRESS2026-10-07.86native calls completed (81measured+5warmup),164.025575605
GPU-s, servicepeak5705359360B,hostmin107083677696B,175watchdogsamples,events/
swap/abort0. Backendreleased/servicegroupremoved; remote services identity-
checked and stopped.86server/clienttransferIDs match179757983wirebytes,pack0.
Initialization1 combines4000D315+240D298+81gap=4321samples,160/160classes;
19gaps actuallyfilled,edgesunchanged[24,91,217,256],minclassn1(disclosed).
serviceSHA21f02c42ad6f39a5ea3a4245b09caa81a4e505502667dcc8b93667926e9d23db;
bindingJSONfa27b09796bad42b9a34e0b7ceaf872ad33e26cd44e7b7be106aa8567f47806f.
This is measured initialization, NOT performancegain/numerical/formalSLO pass.
Sourcecommit4728b16 backedupbeforeGPU. Finaltargeted75tests pass0.590s;
earlierfull213pass50.423s.10newtests. Plan/V1/serving/usermanifestunchanged.

NEXT after tested result backup: prepare D318-equivalent100-request7B Full
with onlynewservicebinding, one smoke then original4000W0; assess data before
accepting. No more lengthcohort/profilematrix,512budget,numericalsmallrepeats,
3B/baselines orfigures. Preparationprofile/control/backend unchanged. Formal
warm/Resident,N_correct,oldPrime newmetrics stillunknown; objective staysactive.
Allownedjobs stopped, no liveGPU/remote/toolsession. No unique data deleted;
free163138781184B (~151.94GiB), recheck floor before next heavy run.
See D321_DECLARED_OUTPUT_CALIBRATION.md. History exporter exactfb9a9f33...
archived beforemergeextension;22merge sources/3outputs verified. Protected147
rawpassfalseonlyknownbaseline04f3eb52...; before/afterreceipts identical.

Backup closure2026-10-07: normalpush verified local/remote HEAD both
f799172660ca54314561deae6b6b95b0349862f3.11explicit result/code/docpaths;
no usermanifest/rawlogs/figures/credentials staged.12checkpointreferences
rehashed successfully. No GPU PID,MemAvailable110277796kB,free163136770048B.
Onlythis postpush note is a new ownedit. NEXT one7B100-requestcandidate smoke
then4000W0 above; do not restart calibration or another backup-only turn.

### D321 preparation and prelaunch record

One candidate rule fixed before performance evaluation: Type-1 quartiles of
D315 historical declared limits, merge ties, append cap256. Keep nine equations,
EWMA/routing/control/V1 unchanged. Extend existing D298 exporter, retain actual
D298 measurements and complete D315 native intervals; no pooled/default fills.
213CPUtests passed50.423s;8new. History export complete4000+240samples,
141/160classes,edges[24,91,217,256].19missing:17fileHOST,2remote.
gaps1 spec prepared22waves/86calls (81measured+5warmup),13existing adapters,
first1000original requests. Actual gaps must be verified, no assumed filling.
See D321_DECLARED_OUTPUT_CALIBRATION.md. No freshGPU/remote run or figure yet.
NEXT tested backup then only86gapcalibration calls; then one minimal validation
and full4000W0. No repeated D320 grouping or old source matrix.7B first,3B and
baselines remain pending. Cache authorization already fulfilledD78/D80.

D321 prelaunch backup: normalpush verified local/remote exact
4728b16024de548c3630262eae0372f978f16509.13explicitpaths, no usermanifest,
rawlogs orfigures;213tests pass. All history/gap source/outputSHAs checked;
protected147rawpassfalse onlyknown baseline replay04f3eb52..., no newchange.
No GPU PID, all4cards15MiB/0%, MemAvailable110384196kB,free163170385920B.
Only ownedD321CPUscopes were stopped afteractualempty; no populated oldTCscope.
Remote published-service start uses unchangedD318procedure/campaignpaths only.
NEXT86-call source calibration via existingrun_d298_capacity_source.sh d321-gaps
in tmux; no new cache, trace,weights,preparationprofile orFullpolicy change.

## HISTORY — D320 complete: Q omission rejected; output-length calibration is next

2026-10-07. This turn PROGRESS, data-only: both retained4000-request W0 logs,
31392candidate views. Q/pending/active-adapter lower-bound contradictions0;
admitted-bin and exact routing-key disagreements0. Preparing/native work are
included; no serving/formula change warranted by the omission hypothesis.
Exact reserve/release times absent: zero contradiction is not exact equality.
See D320_ROUTING_LOAD_AND_LENGTH.md, d320_routing_load/analysis1 and length1.

New actionable evidence: all declared outputs2..256 share one output bin.
D318 short2..32(n1114): predictedO3.090s/observed0.521s; long129..256(n1444):
3.581/6.582s. Overall signederror-0.063s hides absoluteerror2.453s. D315 same
direction. Selectedabove-min-Q2536->2807;1833->1907 lack predictedD advantage
versus the defined lightest candidate. Not proof of wrong placement or causal
latency gain: unselected outcomes unknown. No performance/numerical qualification.

NEXT single7B candidate direction: declared-output class/initialization quality,
not new Q bias or controller thresholds. Reuse historical completed native
samples/admission features and existing exporters; audit class coverage first,
freeze legitimate static/history-based bins before validation, fill only real
necessary measurement gaps (no nearest/pooled/zero profiles), then one candidate
minimal validation and full4000W0. Keep all9formulas/EWMA/routing/input/V1 rules.
Do not repeat D320 cohort grouping,512budget,D294/D295small numerical repeats,
old full profile matrix,3B/baselines or figures. Formalwarm/Resident/oldPrime new
metrics and N_correct remainunknown; objective active. This is not an optimal
Full freeze. User cache approval already implementedD78/D80; do not rebuild it.

334CPUtests pass3.776s (7new); earlier274pass3.605s. Analysispeak1212862464B,
lengthjoin82448384B,tests556945408B; events/swap0; actualempty/populated0 verified
before stopping ownedCPUscopes. No GPU/remote job. Exactanalysis1 analyzer SHA
275ce44d... retained in provenance because the independent length mode was added
afterward. No old output overwritten. NEXT verify/protect/push tested checkpoint,
then the declared-output candidate above; not another open-ended audit cycle.

Verification20:40+08:26source references and6outputs checked, analysis1 source
resolved to exact275ce44d...archive; no oldmanifest rewrite. Protected147rawpass
false retains only known baseline replay difference SHA04f3eb52...; usermanifest
3b69bd0a...unchanged. Plan/V1/serving unchanged. Free163197087744B; no deletion.
Hashverify scope3b1053ff...peak27148288B, events/swap0, actualempty/populated0.
validation_checkpoint.json records identities; backup pending normalpush.

Backup closure2026-10-07 20:41:50+08: normalpush succeeded; exact local and
faaslora_origin/retry14_continuous_queue_v2 HEAD both
b67139bb8067daf7cfebf418dd13bdade35a4c43.14explicit paths; whitespace and
credential-pattern checks passed, usermanifest/rawlogs/figures notstaged.
No GPU PID or live D320 job; all owned CPU scopes stopped afteractualempty.
Onlythis postpush note is a new ownedit. NEXT declared-output calibration
candidate above; no repeatedbackup/audit/grouping. Full objective staysactive.

## HISTORY — D319 complete: reactivation alone does not explain tail tradeoff

2026-10-07 20:16+08. This turn PROGRESS: reused D315/D318 complete4000each,
no new inference/remote/figures/serving edits. D319_LIFECYCLE_TAIL_DIAGNOSIS.md;
new d319_lifecycle/analysis1 (9outputs,22source references rehashed) and
validation_checkpoint.json.364CPUtests pass18.470s. Analysis1exit0 and actual
scopeempty/populated0 before identity-checked stop; peak1041989632B,events/swap0.
Testspeak1095843840B,events/swap0; all owned CPU jobs stopped. No GPU PID.
Plan0c808509.../V1 5f0732ef... and serving262b8f58.../planner1e0c900b... unchanged.

Actual release gaps324.251432GPU-s versus observedU difference338.618264GPU-s
(95.757% descriptive ratio, NOT causal savings). Both-run final-ready boundary
1113.182358s: same2963arrivals P95TTFT3050.342->3149.713ms, P95TPOT56.803->
62.352ms; provisionaljointmiss124->135. Therefore no support for attributing
all regression to two reactivations. Native load2168->2135,total74.269->74.021s;
load-overlap1400->1614requests,prefill-overlap2217->2436,meanothernative1.466->
1.695. Ownprefillmean198.472->198.720ms. Association only, not kernel time or
latency to subtract. All240outputSHAchanges retained; N_correct/formalSLO and
warm/Resident/oldPrime newmetrics unknown; noCI oroptimalFull freeze.

Decision: close this data-only investigation; no repeated startup/load grouping,
no 512budget/profile/numerical repeats, cooldown sweep,3B/baselines or figures.
NEXT targeted7B evidence/code check: does the router's measured Q_i/service view
cover all assigned/preparing/native-not-completed work across lifecycle changes?
Compare existing routing_snapshot/native evidence and historical/official code;
one falsifiable input-consistency hypothesis, not a replacement routing formula.
If no omission, reject that proposed defect; no speculative load-bias coefficient.
Control retirement await is NOT proof that independent dispatcher/request tasks
stop; do not patch lifecycle from that observation alone. Full objective active.

Protected147rawpassfalse retains only known baseline replay SHA04f3eb52...;
userdirtymanifest3b69bd0a...unchanged/notstaged. Free163236904960B at20:16;
no unique data deleted, newcurated~6.7MiB. NEXT normal tested backup then above
7B input-consistency question. This is diagnostic progress, not performance gain.

Backup closure2026-10-07 20:18:51+08: normalpush succeeded; exact local and
faaslora_origin/retry14_continuous_queue_v2 HEAD both
bfc7dfa774506128764cc8ead093fa3b322f8dea.15explicit paths, no usermanifest,
rawdata orfigures staged; whitespace/secrets checks passed. All ownedD319CPU
scopes empty andstopped, no live tool session/inference/remote task. Onlythis
post-push note is a new own edit. NEXT above targeted7B routing-input consistency
audit, not another backup cycle, repeatedinterval analysis orGPUreplay. Goalactive.

## HISTORY — D319 data-only lifecycle/tail diagnosis in progress

2026-10-07. Full source plan and frozen V1 reread; same hashes. Retained
D315/D318 complete4000 each; no inference/remote/figures or serving edits.
Existing analyzer extended to reuse D304-D306 validated interval functions,
with actual physical release gaps and same-request arrival cohorts from the
union of observed event windows.43CPUtests passed6.549s including5new tests.
New output d319_lifecycle/analysis1 only; preserve output hash changes,
unknown numerical/formal qualification and noCI. See D319_LIFECYCLE_TAIL_DIAGNOSIS.
NEXT bounded offline analysis -> interpret -> save/test/backup -> next single
7B falsifiable bottleneck. Do not rerun D318, numerical/profile diagnostics,
start3B/baselines or drawfigures. Goal active.

## HISTORY — D318 full4000 audited: resource-tail tradeoff, not overall superiority

2026-10-07. This turn PROGRESS: same registered full run completed once;
cleanup and complete native/binding/remote/context audits passed.359CPUtests
passed23.322s. No live inference, remote or audit task. Full result decision:
D318_COMPLETE_W0_DECISION.md; curated d318_full_w0/validated1,context1 and
validation_checkpoint.json. All9raw evidence,10prelaunch sources and both
analysis manifests/outputs rehashed. Serving/config/profile unchanged since
D317. Existing analyzers alone extended with explicit D318 paths and no-overwrite
tests; no figures, assets or GPU repeats. Source plan/V1 unchanged.

Versus registered D315: meanTTFT1304.383->1292.091ms(-0.942%), P95TTFT
2934.366->3031.176ms(+3.299%), P99TTFT4610.766->4459.486ms(-3.281%).
MeanTPOT32.558->33.461ms(+2.771%), P95TPOT62.152->64.226ms(+3.337%);
meanE2E4.896654->4.955792s(+1.208%). PhysicalU15865.617487->15526.999223
GPU-s(-2.134%).4000native completions;10bindings each4000; TTFT/TPOT/stage
errors0. All240outputhashdifferences retained. D170timing3818vs3820 of4000
is provisional only, N_correct/formalSLO unknown, noCI/G1G2qualification.

Mean routing177.688->141.098ms nested in admission681.229->652.184ms;
retained-scope observedmax24216 over15473views, not exactRPC savings.
D3185scale-ups/2retirements versusD3153/0; unchanged control law, different
closed-loop trajectory. Both132HTTPfetches/131480060wirebytes, no packing.
4039resourcesamples,peak20559921152B,minhost93445820416B,events/swap0.
Remote all3ID-verified then monitor-first stop,inactive/PID0/success. Local
aux empty verified then stopped. CPUaudit90118a61...peak1154301952B,events0,
actualempty/populated0 then stopped. Protection147rawpassfalse retains only
known baseline replay difference; usermanifest3b69bd0a...unchanged/notstaged.

Decision: retain tested D316 implementation and measured tradeoff point; reject
the claim of overall superiority and do NOT freeze an optimal Full. This local
candidate's short/full evidence chain is closed; no repeated prefix/profile or
threshold scan. NEXT tested explicit-path backup + remoteHEADverification,
then return7B goal-level analysis: reuse D315/D318 full control/native intervals
to inspect retirement/activation and preparation/decode overlap underlying the
tail tradeoff, against D304-D315 history. Read primary papers/code before any
next serving edit; no hardcoded control/SLO changes from mere correlation.
D294/D295's3registered numerical repeats are already complete, do not reopen
that diagnostic indefinitely. Formalnumerical/warm/Resident/oldPrime newmetric
evidence and both model goals remainopen; no3B/baselines/figures yet. Goalactive.

Publication check:18explicit paths only, no usermanifest/rawlogs/figures staged.
Default whitespace checker flagged standard CSV CRLF; kept exact audited bytes
and used command-local cr-at-eol plus all usual whitespace checks (passed), no
global Git setting or source-data rewrite. Secrets-pattern scan passed. Final
hash verifier5010ffc9...peak7913472B,events/swap0, actualempty/populated0 then
stopped. All source/output hashes valid; backup pending normalpush.

Backup closure2026-10-07 19:57:50+08: normalpush succeeded; exact local and
faaslora_origin/retry14_continuous_queue_v2 HEAD verified
8c001480e091daaa6633eb65c700d7bab8f5e575.18explicit files, usermanifest never
staged, unrelated dirty/untracked files preserved. No compute GPU PID remains.
Only this post-push status note is new own modification. NEXT is the data-only
7B goal-level investigation described above, NOT backup repetition, replay,
D295/PEFT repeats,3B/baselines or figures. No task/session/remote unit remains
live from D318. Full objective stays active and unqualified.

## HISTORY — D318 full4000 finished and resources released; offline full audit next

2026-10-07 19:45+08. Previous turn VERIFIED WAIT; this turn PROGRESS: watched
SAME native run through4000/4000terminal,4000success/native-contract-matched,
zero errors. launch.json pass, service/watchdog/replay returncodes0, actual
native GPU contexts clear and service path removed. No inference remains live.
No source/config edits or heavy work were performed during measurement.

19:43:05+08 verified all three captured remote InvocationIDs; stopped monitor
FIRST. Its existing EXIT trap stopped both owned artifact units. All3inactive,
PID0,Resultsuccess. Exact remote transfer and monitor files copied, remote/local
SHA agree:6f531f875b5dbf9919e99ed2292969c3a4a159dfa0632b4784083d5fe21c5fb8
and46381817a4f61e1927063d09f0ec685bfee5362e0080885375180b9566821d10.
Aux220968afb6cd41638ce89b292725c2ff verified actual cgroup.procs EMPTY and
populated0 before stop; peak332750848B, memoryevents/swap0, finalinactive.
One-time delivery cache unchanged; no request-path packing or new asset pool.

Raw result SHA4c781c7b10890293130b425e7cc94dc80c96f3b3dd8037f89db53e1fb08ffb1b;
launch SHAaf05d97d8ce53c4181c8dea5a36a0b019d5c3408d9333aa6c240254e7be6fcbf.
Raw root d318_20261007/7b_full_w0. Complete physical summary measurement=true,
open leases empty; use actual schema gpu_seconds (not a missing prefixed
field). Full audit and comparison NOT yet performed; N_correct/formalSLO unknown.

NEXT save completion receipt; minimally register D318 in existing full/context
analyzers, D315 reference, NEW d318_full_w0/validated1 and context1 only. Reuse
existing copy_retention_trigger for branch evidence, not exact RPC savings.
Bounded CPU tests/audit, then full outcome interpretation and tested backup.
Do not relaunch D318, repeat prefixes/profiles, start3B/baselines or drawfigures.
Goal active; numerical/warm/Resident/oldPrime newmetrics remain open.

## HISTORY — D318 full4000 W0 LIVE; continue SAME run, never relaunch on wait timeout

2026-10-07 18:33:36+08 launched ONCE in tmux `primelora_d318_full`, existing
runner argument `d318`. Tested registration/configuration checkpoint
`efa58b19dfc70df8daa8f9b94b4d78cd9ae531a2` backed up; exact remote HEAD verified
before measurement. Complete source plan and V1 reread, no changes. No figure.

Local service `primelora-tc-svc-1fe0f468e8444f5fbd0fe77e3331dfd8.scope`,
Invocation `9897e0d9862f4357a1b778c6fc14d929`, cgroup inode19893095;
aux `primelora-tc-aux-67d604d46048488d8c0dfbaa5abb71cf.scope`,
Invocation `220968afb6cd41638ce89b292725c2ff`; watchdog2267776/start87646566,
actual identities and affinity verified. Existing gate allow_exec=true,
production_launch_authorized=false correctly denotes development qualification,
not formal M1/M2. Initial read-only probe incorrectly asserted production=true;
subsequent probe corrected this interpretation, no source change/run restart.
External replay plan4000, W0, original trace SHAefb903254f..., 60s notice.

Remote identities captured in d318_20261007/launch_identity.json:
3B ba65a25947b4492ab450026ba7f3a068/PID2138151;
7B 4552e568ad5c40339f980136ef818f60/PID2138153;
monitor556d5d9bfab342aaae248894555dfff4/PID2138156.
Direct authenticated health prepublished_gzip_v1, link1000/full. Actual 7B
journal transfers-399d729f6ba54e6ba16af813a13c6f1f.jsonl under D80/7b;
remote monitor d318_20261007/remote_monitor.log. No repacking or cache rebuild.
Initial watchdog sample50 fresh, events/swap0/no abort. This is not performance
evidence. No serving/config/test/heavy analysis/compression/Gitpush during run.

NEXT monitor SAME full run. After launch.json confirms terminal/local physical
GPU release, verify captured remote IDs, stop monitor FIRST (existing EXIT trap
stops both captured artifact services), then verify inactive/PID0/success and
collect exact journal/monitor. Stop local aux only when same identity and actual
empty/populated0. Reuse existing full analyzers to NEW D318 output versus D315,
preserving all failures, TPOT/output differences and missing formal metrics.
No new prefix/profile/budget scan, 3B, baseline or figures. Goal active; neither
model qualified; numerical/warm/Resident/oldPrime newmetrics remain open.

Live verification2026-10-07 18:37:07+08: SAME service Invocation/inode and
watchdog PID/start confirmed; sample206 fresh0.868s.72terminal/72success/
72native-contract-matched, no error type. Hostavailable100405628928B,
servicepeak19260325888B, all memoryevents/swap0, no warning/abort. No launch.json
yet: measurement still running, not missing-final-result failure. This turn is
PROGRESS (registered/backed/launched/verified actual full run); continue waiting
on the recorded handle, do not relaunch or mark the objective complete/blocked.

Continuation2026-10-07 18:45:19+08 is a VERIFIED WAIT, not a blocker: actual
service Invocation/inode and watchdog PID/start rechecked throughout, latest
sample692 fresh0.536s;516terminal/516success/516native-contract-matched.
No errors, memoryevents/swap0, no warning/abort; hostavailable96244633600B,
servicepeak19260325888B. All three actual remote InvocationIDs/PIDs remained
unchanged and active at18:43. Source plan/V1 SHA unchanged. No source/config
changes, new experiments, heavy analysis or figures during measurement.
launch.json still absent because same full run is LIVE. Resume SAME D318;
terminal/local physical release -> identity-owned remote cleanup -> existing
full analysis versus D315, not another launch or short-prefix interpretation.

## HISTORY — D318 unchanged full4000 registered and tested; fresh launch next

2026-10-07 18:28+08. Previousgoalturn PROGRESS: D317 complete100andbacked.
Thisturn PROGRESS: fullplan andfrozenV1 reread, unchangedSHAs; explicitD318
existingrunner/config registered. ExactlyD315config exceptfreshoutput/cache
paths, servingSHA262b8f58... unchangedafterD317. No new performance mechanism,
profile, assets, coefficients orprotocol. D318clears diagnosticprefix,4000W0.
Configuration2b2a2d408d399c8cb3798ea8628e1c816f230a9b6145c4d430e5c3182d8c1959.
Actualchild/40service/24preparationprofilebinding matches; D317raw8/final5hashes
verifiedbeforefull. Protected147sealonlyknownbaselinereplaydifference/passfalse;
userdirtymanifestunchanged/notstaged. No currentlocalGPU/remoteexperiment.

357CPUtests pass23.280s. Prelaunchscoped78f6a6bcfc8439a873888b9c8bc9605,
3/4GiB/0swap,CPUs2,3,26,27,peak1158090752B; events/swap0. ActualprocsEMPTY
andpopulated0 verified beforestop. Bindings/checkpoint saved d318_copy_full;
prelaunchlogSHA969f73670cb61eb66cfe768eeb9a11dda42df0d14a01cf782039fadd6234528d.
Registration D318_COPY_REUSE_FULL_W0.md fixes D315asfullreference; failures,
TPOTregression,outputdifferences andunknownN_correct/formalSLO remainexplicit.

NEXT explicit-pathcommit/push+verifyremoteHEAD;freshresourcechecks;startsame
D80artifactservices/newD318monitor;directauthhealth;tmux fullrun d318ONCE.
No profile/shorttestrepeat,3B/baselineorfigures. Duringmeasurementdo notchange
serving/config orrunheavyanalysis/tests. AfterlocalreleaseverifyremoteIDsand
stopmonitorFIRST; itsknownEXITtrapstopsartifactunits,thencheckinactive/PID0.
Only thenreuseexistingfullanalyzers tonewD318outputs versusD315. Goalactive;
neithermodelachieved,newmetricsoldPrime/numerical/warm/Residentremainopen.

## HISTORY — D317 native100 complete/audited; unchanged full4000 next

2026-10-07 18:22+08. PROGRESS: single D317 run complete100/100 successful native
contract requests,0failed; ten bindingchecks each100; dispatch tiers100/100;
native TTFT/TPOT andpartitionerrors0. NewGPU-copy retention branch observedmax
402 across280retainedviews (NOT exactRPCsavings/finaleventtotal). Nativeleases
closed,servicepathremoved,service/watchdog/replay returncodes0. No live GPU,
CPU or remoteexperiment. Sourceplan/V1 unchanged; goalactive,7Bnotqualified.

Mean/P95TTFT955.134897/2118.484378ms,mean/P95TPOT28.916611/37.948561ms,
U710.223238332GPU-s,meanarrival-to-admission322.925526ms,nestedrouting78.414707ms.
100-prefix only; N_correctnull; D170timing98/100 is NOTformalSLO. D302 same
1024/control/profile prefix is descriptive context only (intervening source
correctnessrepairs): mean/P95TTFT-4.535%/-7.041%,U-11.159%,butmean/P95TPOT
+5.984%/+10.448%. All4outputhashdifferencesretained; noCI orcausalwin.

29remoteUUIDs/50650865wirebytes matchexactlocal/remotesources; no packing or
republication.263resourcesamples,peak18820423680B,minhost93895622656B,
events/swap0. Localrelease18:14:00;18:14:37ourID-verifiedmonitorstop triggered
the EXISTING remote EXIT trap tostopbothownedartifactservices. Subsequent
redundant IDcheckfailed becausealreadyinactive; notin-run outage. Journals and
monitorSHA agree locally/remotely. Allremoteunitsinactive/MainPID0/success.
Aux747cbc14...actualempty/populated0verified thenstopped,peak161136640B.

Final66CPUtests pass0.132s; analysis4boundedscopes allactualempty/populated0
thenstopped,events/swap0. Two offlineanalyzer errors retained: explicitfilename
wasr100butfrozenconfig-namedfileisr4000(metadata/offeredstill100); then attempted
anunexportedfinalcounter. Corrected to exactsource andrequiredperrequestfield;
no fallbackdefault/noGPUrerun. smoke1partial4files retained; finalsmoke2/audit
and5curatedfiles validated.8rawinputs/8prelaunchsourcehashesverified; protected
147sealonlyknownbaselinereplaydifference/passfalse;userdirtymanifestunchanged.
Validationcheckpoint and D317_NATIVE_COPY_REUSE_VALIDATION.md documentall.

NEXT tested explicit-path backup,remoteHEADverification; thenregisterD318 using
existingfullrunner, sameD315config/currentD316servingcode exceptfreshoutput/cache
IDs, full4000W0. No more100repeat,profiles,budgetscan,D295/PEFT or localcosmetics.
Audit fullwaiting/TTFT/TPOT/U/failures/outputdifferences versusD315, thenaccept
orrevokecandidate. Do notstart3B/baselines/figures ordeclareG1/G2done. Numerical,
warm/ResidentandoldPrime newmetrics remainopen. Supersedesalllivehandlesbelow.

Backup closure2026-10-07 18:23:44+08: normalpush complete; localHEAD and exact
faaslora_origin/retry14_continuous_queue_v2 verifiedef7b6488e5e40510692dd41f97e
97378ede7765b. Explicit14-file commit passed hashes/whitespace/secrets checks;
usergeneratedmanifestneverstaged,otherassetsunchanged. NoGPUcomputePIDs or
populatedexperiment scopes. Only this post-push statusnote isnew ownmodification.
Next continuation must register unchanged fullD318, not redoD317analysis/tests
or assumeD317isstillrunning. Goalactive; remainingfullcampaignnotcomplete.

## HISTORY — D317 native100 live; continue same run, never relaunch on wait timeout

2026-10-07 18:09:32+08 launched ONCE in tmuxprimelora_d317_7b, existing
run_d293_7b_prefetch_validation_100.sh d317. Tested configuration/entry checkpoint
65732328273a9566b81dce8659fbe461990e24a7 pushed and exactremoteHEAD verified.
8source hashes unchanged. Prelaunch113061494784Bavailable/disk164417269760B,
no computePIDs/populatedexperiment scopes. Manualpreflight probe initially
looked for wrong `pass` instead of `preflight_pass`; stopped before remote/GPU
startup, then checked correct existing field. No protection/configuration edit.

Localserviceprimelora-tc-svc-d1daf6aca4a340709b5b44fdc29abad5.scope,
Invocation56e512df33954c76b4e0ac319e565093,inode19877820; auxc29b0545...;
watchdog2163279/start87502082 actualidentity verified. Fullcapturedlocal/remote
identities in d317_20261007/launch_identity.json. Resource sample20fresh,
events/swap0/noabort; startup only, not full performance evidence.
Remote3Ba78ef87f...,7B054008b9...,monitor8107408f...;1000/full; authenticated
directhealth prepublished_gzip_v1. Exact7Bjournal from actual/proc/fd captured;
no cache rebuild, token exposure, packing, networkchange or extra load.

NEXT monitor SAME100run; afterlaunch.json confirmsrelease, stopcapturedremote
monitorFIRST thenartifactservices; collectexactjournal/monitor; audit100native
bindings/tier/releases and retainedscope trigger. Then unchangedfull4000 if
safe. No serving/config/test/fullanalysis/Gitpush during measurement. Nofigures
or3B/baselines. Numerical/warm/Resident/oldPrime metrics remainopen,goalactive.

## HISTORY — D317 registered 100-request native validation, launch pending

2026-10-07 18:08+08. D316 tested change backed up to b9975ef28c37195ec4d0c806
bfabce36f7265f40; exact remote branch HEAD verified. No performance win claimed.
D317 adds ONLY an explicit existing-runner identity, cold output/cache paths
and scenario name. Configuration8eccbe6648036c947ed4a8d67b2837a19169aa74b796
07d486734110a8761350 equals D315 otherwise. It uses the existing100-prefix,
not a regenerated trace. D29840service/24preparation class bindings and actual
native child match; protected147/user manifest checks pass with known rawseal
exception unchanged. No new model profiles, safety thresholds or evaluation.

64 registration/route/analysis tests pass0.138s; prelaunch binding passes.
BoundedCPU Invocationefdda784094b45a19ba10b0d1d246d8f peak558866432B,
3/4GiB/0swap,CPUs2,3,26,27,events0. ActualprocsEMPTY/populated0 checked then
stopped. Prelaunchlog SHA c0e84ffdbfc1011d81b043b6535317574706af5c5d1b086
c6e2c11080439f56a. See d317_copy_validation/prelaunch_binding.json for8sources.
ServingSHA remains262b8f58... already validated by419regressions/289smoke tests.

NEXT back up registered entry, verifyremoteHEAD, then fresh host/disk/GPUgate;
start unchanged owned artifact services+new monitor (no publication rebuild),
authenticated direct health, launch d317 ONCE in tmux. Register100success,
nativecount/binding/tier/physicalrelease and nonzero retained-GPU-scope branch;
short performance cannot qualify G1/G2. Then one unchanged full4000 versusD315
if safety/trigger pass. Preserve any failures and stop local optimization after
this candidate's full interpretation. No figures,3B,baselines or protocol edits.

## HISTORY — D316 single event-confirmed GPU-copy candidate CPU-validated; D317 next

2026-10-07. PROGRESS: D315 full-log reuse analysis found2357 retained fresh-GPU
views with unchanged target-copy identity after global epoch advancement,
affecting1261of4000requests. These are observed opportunities, NOT exact RPC
counts or predicted latency savings. Native owner epoch also changes on pins
and unrelated copies. Historical D200/D201/D216/D289/D301 and official vLLM0.30
worker/core were checked; D216's extra identity RPC is NOT revived.

One serving change retains an ORIGINAL measured GPU-copy record when an
already-received newer native event confirms unchanged owner/clock/target
sourceID/slot/path/rank. No new TTL, RPC, profile or coefficient. HOST-only
records still invalidate; selected-source atomic recheck/pin remains mandatory.
IEEE equations,1024backend configuration, D298profiles, generation and frozen
metrics unchanged. Cumulative retained_gpu_scopes only proves branch activity.
Old-code targeted test reproduced deletion;419 regressions and289 smoke tests
pass. Prior27 analysis/route tests also pass. No GPU performance run this turn.

CPU scopes3/4GiB/0swap,CPUs2,3,26,27: every actual cgroup.procs empty and
populated0 verified before stop; events/swap all0. Verifier5e4f37d32a7d454ca647
cd963b8f2aaa peak27131904B stopped afteridentity check. Analysis6input/2output
hashes pass. Protected147seal retains ONLY known baseline replay difference,
rawpassfalse preserved; dirty usermanifest3b69bd0a... unchanged/notstaged.
ServingSHA262b8f58ca12b129d0292f3e6da94ed3f5732171f959a2d0280c89e8cda46ef6.
Evidence and validation_checkpoint in d316_route_copy_revisits; interpretation
in D316_EVENT_CONFIRMED_GPU_COPY_REUSE.md. No new figures/assets or remote work.
Approved one-time delivery cache already exists D78/D80; do not rebuild it.

NEXT: explicit-path Git checkpoint and verified push, then register D317 once
using existing100-request runner and D315 unchanged1024 configuration. Safety,
native binding/count/tier and retained-copy trigger must pass before unchanged
candidate full4000. Do not infer full performance from100prefix. No repeatD315
audit/D295/PEFT/profile/budgetscan or jump to3B/baselines. Numerical/warm/Resident
and oldPrime newmetrics remain open; neither model nor G1/G2 qualified.
Fullgoal remainsactive. This entry supersedes D315's next-investigation task.

## HISTORY — D315 full4000 audited; request-path waiting is next, not another budget scan

2026-10-07 17:40+08. PROGRESS: complete D315 analysis and both registered
comparisons (D303 and D314) finished; 341 CPU tests passed19.032s. Full source
plan and frozen V1 reread; both SHAs unchanged. No serving/configuration edits,
new GPU runs, figures or new assets. No local/remote/CPU experiment remainslive.
The user-approved one-time compressed delivery cache was already published in
D78/D80 and reused; do not rebuild it in response to the repeated approval.

All4000 native count/request/adapter/owner/lease/slot/clock checks pass, stage
sum maximum error0; actual leases closed. N_correct remainsnull. Output hashes
differ216 vsD303 and244 vsD314, retained for audit, not numerical qualification.
Mean/P95TTFT1304.383/2934.366ms, mean/P95TPOT32.558/62.152ms,
meanE2E4.896654s, physicalU15865.617487GPU-s. D170 provisional timing3820/4000
is NOT formal SLO. ComparedwithD314: P95TTFT-12.923%,U+0.123%; comparedwithD303:
P95TTFT+1.193%,P99TTFT+7.817%,P95TPOT-8.741%,U+2.517%. Singledevelopment
runs, noCI or superiority claim. Preserve all regressions; keep1024 only as
the current development startingpoint, not as achievement of G1/G2.

Mean arrival-to-admission681.229ms is52.226%ofTTFT. Its partition is187.154ms
arrival-to-serverrelease,28.901msrelease-to-globalentry,465.175msentry-to-slot.
Routing177.688ms is nested, not additive. Conditional authoritative-read
groups are not causal evidence for removing checks. 99.362%ofU occursin the
arrivalwindow;3scaleups/0scaledowns.917controlsamples havequeue/active lowbut
TTFT blocking; longestall-low35.197s does NOTjustify fitting38scooldown to35.
Next question is actual request-path dependency/serialization and its resource
consequences, preserving confirmedstate/atomicpin/nineformulas and frozenSLO.

Remote132transfers/131480060bytes matched; no on-request packing.4042resource
samples: servicepeak20568260608B,minhost93507256320B,high/max/OOM/swap0.
Preparations2163eligible/39ineligible; changed-copybranchtrigger0, so noclaim
that zerofailures provedD313cause. Protected147seal retainsonlyknownbaseline
replaydifference/passfalse. Userdirtymanifest3b69bd0a...unchanged/notstaged.
Servingrunner/planner source SHAs unchanged fromD314correctnessfixes.

Curated d315_full_w0/validated1,context1,context_d314 manifests verifiedall
10/7,12/2,12/2input/outputhashes. TestInvocation702cb38f..., analysis39b30b04...,
verifier48916a67...all3/4GiB/0swapandCPUs2,3,26,27; actualprocsEMPTY/populated0
checked before stopping. All events/swap0. Validationcheckpoint binds scripts,
tests/logs/resources. Full interpretation: D315_COMPLETE_W0_DECISION.md.

NEXT: explicit-path tested Gitbackup, verifyremoteHEAD; then returnto7Bwaiting
bottleneck investigation with historical evidence and official source before
choosing one falsifiable change. Do not scan budgets, repeatD295/PEFT, jump to
3B/baselines or generatefigures. OldPrime newmetrics and numerical/warm/Resident
qualification remainopen. Keepfullgoalactive. Thisnote supersedes the raw
completion receipt's historical audit-pending flag without rewriting it.

Backup closure 2026-10-07 17:41:22+08: normal push completed. Local HEAD and
exact faaslora_origin/retry14_continuous_queue_v2 remote HEAD both verified as
e6cb59e099021fa528be2865718c259bab3e8b4e. Explicit20-file staging passed the
CR-at-EOL-aware whitespace check and credential-pattern scan;341 tests and
manifest identity checks were completed before commit. Usermanifest unchanged
3b69bd0a...andneverstaged; unrelated untracked/dirty assets preserved. No raw
large logs, credentials or figures committed. This post-push note is the only
new own modification. No experiment remainslive. The next continuation must
investigate7Brequest-path waiting, not repeat completedD315/audit/backup tests.

## HISTORY — D315 full4000 finished and released; complete audit next

2026-10-07 17:19:45+08. This continuation moved from VERIFIED WAIT to PROGRESS:
the existing D315 run completed4000/4000 successful native-contract requests,
0failed, then released allGPUcontexts. No new run was started. Observer2052
completed; observer2064 completed45polls and observed launch.pass=true. Both
observers are closed; do NOT resume or restart them. No localGPU/remote/CPU job
is live. Sourceplan fully reread thisturn, SHA0c808509... unchanged.

service/watchdog/replay returncodes all0; service cgroup path absent and native
GPUrelease confirmed. Physicalmeasurementcomplete=true, openleasesempty,
U15865.617487298907 GPU-s. N_correct remainsnull; this is not numerical/SLO/
G1/G2 qualification or a comparison decision. Main result330113142B,
SHA6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f;
launchSHAd0ed2535e2a8ee8ac268eaefae586934b336e96cbc0285573927a3f96ea5bc55.
main_outcome706070523B: use existing streaming projection, not fulltree loading.

Allthree capturedremoteInvocationIDs matched immediately before stopping
MONITORFIRST at17:18:18; allthree theninactive/MainPID0/Resultsuccess/exit0.
Exact7Bjournal andmonitor copied into D315rawroot without overwriting files.
LocalauxInvocationf744b608... actualprocsEMPTY/populated0verified beforestop,
peak333459456B, allmemoryevents/swap0; nowinactive. NoGPU computeprocesses remain.
Postcleanupdisk164462583808B/hostMemAvailable110393476KiB are readings, not a
futuregate. Rawcompletion_receipt.json records identities, source hashes and
the still-pending audit boundary. No figures, new assets or measured-code edits.

NEXT: reread frozenV1, extend the EXISTING analyze_d297_full.py and
check_d299_full_context.py with explicit D315 source/new output identities,
test and run under the existing bounded CPU protocol AFTER a fresh resource
check. Retain all-success/failure separation and preparation-attribution checks;
do not weaken gates. Compare the complete run with D303 and D314, retaining
all output differences and provisional-ruler caveats. Save interpretation,
protected/user-file checks, then tested Gitbackup. No more token-budget scan,
D295/PEFT repeat, 3B, baselines or figures before the required 7B work.
Codecheckpoint remainsbbd6bf39...; only own EXECUTION_STATUS ismodified beyond
it, plus the pre-existing usermanifest (neverstaged). No Gitpush duringmeasurement.
The oldPrime new-metric comparison and numerical/warm/Resident qualifications
remain open; neither7B nor3B is declared optimized. Keep the fullgoal active.

## HISTORY — D315 restored 1024 full W0 running and verified waits

Started ONCE2026-10-07 16:08:46+08 in tmuxprimelora_d315_full, existingrunner
explicitd315. Tested checkpointbbd6bf39f65d753c344b9af07ee4224d19022556 pushed;
remoteHEADmatched16:06:39. All10prelaunch input/outputSHA checks passed.
Service serving source remainsD314'sc4631e81 content. No configuration, service,
tests, fullanalysis, compression or remote management changes during measurement.

Localserviceprimelora-tc-svc-598ea76bc9b64f3db0bdc475c4363e1c.scope,
Invocationdada997e891c41d28bea0c7df75fd3fc,inode19802320;
auxprimelora-tc-aux-252682b3914047f4b693bade5bb2df80.scope,
Invocationf744b6082cbd411e9bba77d40e616a82. WatchdogPID1631479/start86777493,
actualcgroup verified; sample36fresh,host108465807360B/service4069486592B,
memoryevents/swap0,noabort. Thisis startup evidence,notcompleteperformance.
Rawresults/ieee_tc/p2_backend_qualification/d315_20261007/7b_full_w0.
Do NOTrestart on observation timeout. Continue SAMElivehandles.

Remote3BInvocationac49bbe70e45453ebf5cf44422897237/PID1957266;
7B56daa9f145774211832ad5126a869d16/PID1957273;
monitorfd4701f5153643498eb4f31914e05063/PID1957283,
unitprimelora-artifact-monitor-d315.service. Exact7Bjournal verifiedvia/proc/fd:
/home/lab14/primelora_remote/tc/d80_20260927/7b/transfers-5ad4f4b9abc74fd89a0c0750a4ac9ff9.jsonl.
Monitorlog /home/lab14/primelora_remote/tc/d315_20261007/remote_monitor.log.
Authenticated DIRECT healthpass,clockremote-process-monotonic:ff1156ae3f864d9194396dd5707cac92;
bothNICs1000/full. Initialmanualurllib default-proxyhealthprobe returned502;
retainedasprelaunch probeerror, noGPUrunthen. Correctdirectprobe uses same
ProxyHandler({}) asunchangedrealartifactclient; noendpoint/server/networkchange.
Similarlyinitial manualchecksumcommandusedPython3.11-onlyfile_digest onOS3.10;
correctstandardSHA256verifiedall10files, nopayloadrewrites. Thesearechecks,
notperformancefailures. Feishuconfigurationabsent; noW&Bconfigured.

NEXT: monitor this4000replay; afterterminal/localrelease verifycapturedremoteIDs,
stopMONITORFIRST thenartifactservices,collectexactjournal/monitor,streamlarge
outcomes andrunexistinganalyzers tonewoutputs. Do notpublishpartialprefixaswin.
Nofigures or3B/baselines. Numerical/warm/Resident/oldPrime newmetrics remainopen.

16:12:20+08 verified live wait: same localserviceInvocationdada997e...and
watchdogPID1631479/start86777493 checked directly. Sample210fresh,
74terminal/74success/74native-contract-matched/0failed; host96610721792B,
service16481558528B,disk165034491904B,events/swap0,noabort; no launch.json.
Monitoring cell2039 isLIVE (20polls at45s; lastreturnedpoll2). Resume thatcell
with wait, not anotherlaunch; if it expires inspect sameactualrun handles.
Capturedlocal/remoteidentities areinrawlaunch_identity.json. No serving/config
changes afterlaunch. Thisis partialrunstatus,notG1/G2/numericalqualification.

2026-10-07 16:34:21+08 VERIFIED WAIT continuation: previous monitorcell2039
completed its20polls; that did not terminate or restart the experiment. New
monitorcell2052 is LIVE (20polls at45s, latestreturnedpoll7). Resume it withwait;
on observer completion continue checking the SAME actual experiment handles.
Local serviceInvocationdada997e.../inode19802320, auxInvocationf744b608...,
watchdogPID1631479/start86777493 and actualauxcgroup were reverified. Remote
three originalInvocationIDs/PIDs remainactive; eno1 still1000/full. No remote
restart, configuration change or cache reconstruction was performed.

Latest sample1514/monotonic869309.00416726:1439terminal/1439success/
1439native-contract-matched/0failed; hostavailable96122785792B,
service18316713984B,diskfree163726761984B,allmemoryevents/swap0,noabort.
No terminal launch.json yet. This is partial run evidence, not numerical,
joint-SLO, G1/G2 or superiority qualification. Sourceplan fully reread after
compaction, SHA0c808509... unchanged; frozenV1 fully reread, SHA5f0732ef...
unchanged. D315 registration reread; no new comparison or selection made.
monitor-experiment skill used; Feishuconfig absent, noW&Bconfigured. No new
serving/config edits, tests, full-data analysis, figures, compression, heavy
jobs or Git push during measurement. Tested checkpoint remainsbbd6bf39...
User-approved read-only delivery cache already exists fromD78/D80 and is reused.
After terminal/localrelease follow the captured-ID/MONITOR-FIRST cleanup and
existing analyzer path above. Keep7B->3B->baselines; unresolved qualification
and oldPrime new-metric comparison remain open. Goal remains active.

2026-10-07 16:44:24+08 continued VERIFIED WAIT: cell2052 completed20polls;
actual D315 remainslive, neverrestarted. Newmonitorcell2064 is LIVE
(45polls at45s, latestpoll1); resume thatobserver. Same localserviceInvocation
dada997e... and watchdogPID/start identity verified on everypoll. At16:42:54
allthree originalremoteInvocationIDs/PIDs were directlyreverified active;
NIC1000/full unchanged. Latest sample2108/monotonic869911.127163333,
2243terminal/2243success/2243native-contract-matched/0failed,
host95524122624B/service19196420096B/disk163265880064B,events/swap0,noabort.
Sourceplan fullyreread thiscontinuation/SHAunchanged. Only read existing
D296/D303/D314 conclusions and existing analysis interfaces while waiting;
no new performance analysis, changes to measured code/config, newhypothesis
experiment, heavy task or remote operation. Full replay and post-release
audit remain pending; no claim of G1/G2 or numerical qualification.

## HISTORY — D315 restored configuration prelaunch preparation

2026-10-07 16:05+08. Previous continuation was PROGRESS: completed D314
evidence was pushed and exact remote HEAD verified, not a new experiment.
This continuation prepares the already registered return to measured D303
settings while keeping all current ownership/preparation correctness fixes.
No serving code or IEEE equation changed; D314's 512 candidate remains rejected
as an overall gain. Full source plan and frozen V1 reread, SHA unchanged.

New D315 config differs from D303 only in output/NVMe/HOST paths. Actual child
exactly equals D298 measured model contract; existing40service/24preparation
classes load with exact SHA/context. Both source and profile data are reused,
not recollected. ConfigSHA3a87e96b9ed1ae525320aec69a743c02714738fef3368af074c0dd980be24e31.
431CPU tests passed72.830s; actual configuration binding passed. Test/prelaunch
scopeInvocation80ec7e70dc8f429294daf04c107cec0f, peak2602958848B, memoryevents/swap0.
Existing147protection seal retains only known baseline replay change/passfalse;
userdirtymanifest unchanged3b69bd0a... andneverstaged. No figures or new assets.

NEXT: tested backup, then existingrunner explicitd315 in tmux fororiginal4000W0
ONCE. Independent localcold paths, same60snotice/control/1800squalification,
existingpublished remotegzipcache. Capture newlocal/remoteInvocationIDs; at
terminal verifyGPUrelease thenstop remoteMONITOR FIRST andretain exactjournal.
No100prefix/profile/budgetscan. Formalnumerical/warm/Resident andoldPrime
newmetrics comparison remainopen. Keep7B->3B->baselines; no prematureclosure.
SeeD315_RESTORE_MEASURED_WORKING_POINT.md andd315_restored_full/prelaunch_binding.json.

## HISTORY — D314 full4000 audited; 512 candidate is not an overall improvement

2026-10-07 15:49+08. This continuation made PROGRESS after verifiedwaiting:
D314 finished4000/4000successful native-contract requests,0failed,normalrelease.
No liveGPU/remote/CPUjobs remain. Monitorcells1969/1976 completed; doNOTresume
them orrestartD314. Allthree launchreturncodes0,physicalmeasurementcomplete,
openleasesempty,U15846.161075063865GPU-s. FullplanandfrozenV1 reread/SHAunchanged.

FullsourceSHA5cbc1b213ab54d282ee2b41a3ff22c0fde09d8615d900005c2676e2bd9ff4c9e;
launch890095eed63884d27c948ae241d716445467da21814d01e0771d24e789e83b61.
RemoteallcapturedInvocationIDs verified,MONITORFIRST stopped15:40:42; allinactive/
PID0/success. Exactjournalandmonitor copied; rawcompletion_receipt.json retained.
Localaux9d487d05...empty/populated0verified,peak333615104B,events/swap0,stopped.

Existinganalyzers extended explicitD314identity andtested preparationattribution
recomputation;338tests passed31.714s. CPUtestInvocation3eb33f53...peak1175941120B,
actualauditInvocation8c851ab3...peak1083670528B,bothhigh/max/OOM/swap0,empty/stopped.
Canonicaloutputs d314_full_w0/validated1 andcontext1. All4000input/binding/time
checks pass;251outputhashdifferences vsD303 retained,notnumericalqualification.
132HTTPtransfers/131480060bytesmatched,nostreamedpacking;4042safetysamples,
peak20745805824B,minhost93557907456B,events/swap0. Protectedsealrawpassfalse keeps
onlyknownbaseline replaychange. Usermanifest3b69bd0a...unchanged/notstaged.

Mean/P95TTFT1466.345/3369.848ms,mean/P95TPOT32.635/62.166ms. ComparedwithD303:
P95TPOT-8.72%,butP95TTFT+16.21%,GPU-s+2.39%; provisionaltiming3787/4000vs3810.
NoformalSLO/G1/G2claim,CIorfigures. Preparation2059eligible/45ineligible;
newchanged-file-copybranchtrigger0,sozerofailures donotproveD313historicalcause.
SeeD314_COMPLETE_W0_DECISION.md. Servingcodeunchangedc4631e81; newaudit/scripts/
tests/doc/curated changesawaittestedbackup. No newcandidate launched.

NEXT: back up this completed evidence, then return toalreadymeasured1024working
point withcorrectnessfixes andexactmatchingexistingprofiles. No morebudgetscan,
D307reprofile,100prefixforwinning,PEFT/D295repeat,3B/baselinesorfigures. Required
7Bperformance/commonreference/oldPrime comparison remainsopen. Do not shorten
38scooldown fromobserved37.366sall-lowspan; that isnotcausal tuning justification.

Prebackup identity verification15:51+08 passedall10inputs/7outputs ofvalidated1
and12inputs/2outputs ofcontext1. VerifierInvocation66377e72...peak10514432B,
events/swap0,empty/populated0verified/stopped. validation_checkpoint.json binds
analysissource/tests/logs andallthreeCPUscopes. Nofulltestlogfilewascaptured;
338pass/31.714s isfromrecordedcommandoutput,notinventedlog. Explicitstagingonly.
Defaultgit whitespacecheck flagged onlyexistingcsv.DictWriter CRLF innew
comparison.csv. Preserve itsverifiedbytes/SHA; useCR-at-EOL-aware diffcheck
while retainingblank-at-eol/blank-at-eof/space-before-tabchecks. Nooutputrewrite.

2026-10-07 15:58+08 backup closure: push session25411 completed successfully.
Local HEAD and faaslora_origin/retry14_continuous_queue_v2 both resolve to
e35ef9f749724eef7155a3781378681aac44b4b9. This checkpoint contains D314's
complete audited evidence, analysis tests and explicit rejection of the 512
candidate; it does not change serving code. Full source plan and frozen V1
reread; both SHA values unchanged. User manifest SHA remains3b69bd0a...,
unstaged; existing unrelated dirty/untracked files are preserved. Only this
post-push status note is new own work after that checkpoint.

The asynchronous approval for one-time read-only compressed delivery is already
implemented by D78/D80: two 500-adapter manifests,1581215261 archive bytes in
total, no second extracted pool. Continue reusing it; this confirmation does
not require a duplicate build. No remote operation or new GPU run
was performed during backup closure. Fresh read-only local check found no GPU
compute processes,165582110720 free disk bytes and110446036kB MemAvailable;
these readings are not a substitute for the next run's full admission gate.
Next remains the 7B 1024 working-point revalidation described above, with
correctness fixes retained. Formal numerical/warm/Resident qualification,
3B closure, baseline comparisons, ablations and sensitivities remain open.

## HISTORY — D314 full4000 W0 running and verified waits

2026-10-07 14:32+08. This continuation made progress: deterministic defect
reproduction, principled identity/attribution fix,873passingtests,protected/
profile checks, commit/push and new original4000replay. Tested checkpoint
c4631e81b1ca4510fdb0fe59391dc6e6c3a07c2d pushed andremoteHEADverified14:29:44.
Started ONCE14:30:58 in tmuxprimelora_d314_full viaexistingrunner explicitd314.
Rawresults/ieee_tc/p2_backend_qualification/d314_20261007/7b_full_w0.
No serving/configuration changes during measurement; no figures or tests.

Localserviceprimelora-tc-svc-5e8f3b6c7a0448dda9a0edb48f87603d.scope,
Invocationcc4bbfa3de4048d9a1e4ef2738593ea0,inode19741350;
auxprimelora-tc-aux-956df753ceb943acb4fa802ab204c595.scope,
Invocation9d487d05aa784fc2bb2bd82c3f6f583f. WatchdogPID1194219/start86190714.
At14:31:52sample53fresh,host106681475072B/service6090342400B,
high/max/OOM/swap0,noabort; initialpreparation, notcomplete. Observationtimeout
isNOTterminal. Continue samejob; do not restart. Latestlaunchmetadata is
exec_receipt.json (top-level service_identity; watchdog_ready.watchdog_process).

Remote3BInvocation055ca0dfb4ee428fb427cf9ab0a4caf2/PID1814168;
7B2f2f0090d7a14ba4961044bd250146cb/PID1814170;
monitorf9fb4b9ccacd41519c5e7bcefae71409/PID1814173,
unitprimelora-artifact-monitor-d314.service. Authenticated healthpassed,
clockremote-process-monotonic:32617c08aa344ba2b81d0c97da118668.
ActualNIC1000/full; existingpublishedcache reused, notrebuilt. Exactjournal
verifiedvia/proc/1814170/fd/3:
/home/lab14/primelora_remote/tc/d80_20260927/7b/transfers-1e437b0cb6ce4d6cba98d9ff8399f33e.jsonl.
All captured identities in rawlaunch_identity.json. After localterminal/cleanup,
verifyallthree remoteIDs, stopMONITOR FIRST, thenbothartifactservices; copy exact
journal asremote_transfers.jsonl andD314remote_monitor.log. Do NOTcopywholepools.

Fullplan/V1alreadyread; hashesunchanged. Userdirtymanifestnotstaged. Noother
newheavyjob; no Feishuconfiguration. Keep7B->3B->baselines; numerical/warm/
Residentqualificationstillopen. D313remains3999success/1failed andcannotrank.
Thisrunmayalsofail; preservealloutcomes and exactnewpredicates, notassumevictory.
Fresh14:32:59: sample118,16requestterminals;host97227976704B,
service15571558400B,high/max/OOM/swap0,noabort. Businessreplaystarted;
sameInvocationstillactive,launch.jsonabsent. live_handle_receipt.json saved.
No terminal interpretation/retuning based on this early partial prefix.

2026-10-07 14:49:35+08 verified-wait continuation: resumed monitoring cell1950
and completed its16lightweight polls, not a second experiment. Same service
Invocationcc4bbfa3... remainsactive; watchdogPID1194219/start86190714 andexact
auxcgroup reverified14:46:51. Latest sample1101fresh,914terminal/914success/
914native-contract-matched/0failed. Hostavailable95802503168B,
service18113466368B,diskfree165259665408B; high/max/OOM/swap0,noabort.
No terminal launch.json. Monitoring cell isclosed; actualrun isNOTcomplete.
Continue same D314 handles. Partial observations do not establish numerical
correctness, SLO qualification or performance superiority. No source/config/
remote changes, tests, full-data analysis, cleanup, figures or newGPUjobs.
Fullsourceplan reread, SHA0c808509... unchanged; currentcode stillc4631e81.
Read-only existing analyzer inspection confirms post-release D314 needs explicit
new output identity; keep all-success and failure paths separate, streamlarge
outcomes, never weaken gates or reuse D313 failed run asall-success reference.
Rawcheckpoint: d314_20261007/monitor_144935.json. Userapproved publishedcache
alreadyexists andisreused; do not rebuild or duplicate theartifactpool.
Keep7B->3B->baselines, numerical/warm/Residentqualification remainsopen.

2026-10-07 15:04:14+08 continuing VERIFIED WAIT: same serviceInvocation and
watchdogPID/startticks reverified each poll. Newmonitorcell1969 isactive (40polls,
45s spacing; latestpoll16); actualexperiment neverrestarted. Latest sample1968,
2029terminal/2029success/2029native-contract-matched/0failed; host95560052736B,
service18971758592B,disk164604743680B; high/max/OOM/swap0,noabort,launch.jsonabsent.
Remote lightweightreadonlycheck ~14:57 confirmed allthree originalInvocationIDs
unchanged andeno1 still1000/full; no management/config/download actions.
Fullplan reread SHAunchanged; initialtooloutputtruncation wascovered bysmaller
rereads. Feishuabsent. D295history reviewed: allthree registered numerical
repeatblocks COMPLETE (162generations); do NOT reopen thisboundeddiagnostic.
FormalFullcorrectness isstillunproved, but that isnot authorization forindefinite
repeatedPEFT/numerical tests. Return afterD314audit to7B SLO/GPU-s mainline.
Readonly serving-code review identified retained evidence.preparation_interval
reason/profile_eligible/cost_model_updated; post-release audit should count
changed-copy cases andverify attribution, not infer repairtrigger fromzero failures.
No fullanalysis/tests/figures/newheavyjobs; code remainsc4631e81. Continue same
monitorcell1969 untilit ends, thenrevalidate exactlivehandle, neverrestart ontimeout.

15:22:18+08: cell1969 completed all40polls normally; experiment isSTILLlive.
SameInvocationcc4bbfa3... andwatchdog1194219/start86190714; sample3037,
3299terminal/3299success/3299native-contract-matched/0failed. Host95310749696B,
service20261634048B,disk163726778368B; high/max/OOM/swap0,noabort;
terminal launch.jsonabsent. Newlightweightmonitorcell1976 follows SAMEjob forup
to24more45s observations; resume thatcell, notlaunch asecondexperiment.
ReadonlyD307registration confirms fullcomparison should use retainedD303 and
reportallTTFT/TPOT/GPU-s effects with combinedbudget/profile/correctness-fix
caveats; nofurtherbudgetscan orwarm/PEFTrepeat withoutnewcausal evidence.

## HISTORY — D314 copy-identity repair tested; backup then full4000

2026-10-07 14:29+08. Progress: deterministic existing request-lifecycle fixture
proved that legal Remote->NVMe publication followed by confirmed same-content
HOST loading raises D313's original path-equality error. Historical req_00216
root cause remains unproved (missing fields); do not relabel that history.
Fix separates safe immutable-copy identity from complete-path d attribution.
Both confirmed file publications, frozen content, same file owner and exact
native consumption must match. File release still occurs immediately after
native acquisition; new bounded receipt records that protection boundary.
Changed-copy sample is ineligible/None, not zero; service D and user timing
remain complete. No retry, forced NVMe, disabled HOST, or formula/config change.
Remote validation exceptions now preserve failed predicate booleans in the
existing bounded error/traceback collector. D312 owner transaction remains.

873 CPU tests passed40.538s; logSHAf3751bb45d65fc30ed2c0a19780102c85c064b85008f1280b62f6808a76cd9d7.
Negative owner/content/path/clock/stage/consumption cases, wrong evidence cleanup
and cancellation pass. Initial867passed+one nonexistent test module invocation
is retained as failed cpu_tests.log, not hidden. Allfive CPUtest scopes had
zero high/max/OOM/swap, empty procs/populated0 verified and stopped; finalscope
099ad1caa3184223b3e7c9eea567621a peak1170784256B. Prelaunchscope
3cc0a4a96851490c9ccb4cbde8a028c2 peak496173056B/events/swap0, empty/stopped.

Fresh D314 full config SHA bd772f035b9c8c1e026dc5ec1880fd1d79d3730f1fdab383d37405bfd66d0fa3;
only result/coldpath identities differ from D313. Same assembled child and all
40service/24preparation classes validated against D307. Existing runner adds
explicit d314 key. Raw/cpu logs under results/ieee_tc/p2_backend_qualification/
d314_20261007; curated CPU/prelaunch checkpoint under d314_copy_identity.
Protected147seal retains only known authorizedbaseline replaychange/rawpassfalse.
UsermanifestSHA unchanged3b69bd0a..., neverstaged. Sourceplan0c808509... andV1
5f0732ef... fullyread. Source runnerSHAf78615de77253215942a9bee553ae3d4bec9fc1164d08ecf187cd46a56c8206e.

NoGPU/remote/CPUjobrunning at this checkpoint. Next: tested commit/push, verify
remoteHEAD, start existing published-cache remote services with new captured
identities, then D314 original4000 W0 ONCE. No repeat100prefix, profiles or tuning.
Keep7B->3B->baselines. Nofigures; numerical/warm/Resident andG1/G2 remainopen.
Design/evidence: D314_PREPARATION_COPY_IDENTITY.md. After measurement stop captured
MONITOR FIRST, retain journal, stream audit; all failures retained.

## HISTORY — D313 full4000 audited; deterministic preparation-path test next

2026-10-07 14:07+08. This continuation is progress after verified waiting:
all4000requests reached terminal states,3999native-contract-complete/1failure.
launch.json pass=true means normal launch/cleanup, NOT all-correct qualification.
Allthree returncodes0, nativeGPUcontextrelease andservicepathremoved confirmed;
physicalmeasurementcomplete=true,openleasesempty,U_observed15647.60522277304GPU-s.
N_correct/N_good remainunknown andranking/formalqualificationfalse. Source result
SHA091c7ffe720b413610a2a366ed4e18315ec9896c68f6f1080d279e913d229f7c;
launchSHA83dd8df0237bb8f5caa082fd62bb44da9f7cc3bd495994cdd7afb2418a2b5607.

LocalauxInvocationd533b4cb... verifiedemptyprocs/populated0,peak335048704B,
events/swap0,stopped13:55:33. NoGPUcomputeprocesses remained. Allthree remote
InvocationIDs matched capturedidentities, MONITOR FIRST stopped13:55:34;
allthreeinactive/MainPID0/Resultsuccess. Exactjournal andmonitor copiedonce:
remoteSHA50618d61f529eaf8d5fe193f88448b84020fcede19402c46c8961f2a6395ad8d;
monitorSHA8b6528a93b4e1db409d02447778738a9a5a634dad356f6465b5c40281bd3fd6a.
Rawcompletion_receipt.json containscleanup/failurefacts. Do NOT restart D313.

Streamed304MiBrequestresult underboundedCPU scope: exactfailure req_00216,
support_lora_0040,target7; nontruncatedtraceback confirmspreloading_planner.py:317.
Failed gpu_reference_evidence isEMPTY, so six-way failingpredicate isnotretained.
Nativepath/remote_preparation/selectedsourcecannotberecoveredfromthisrow; the
HOST/NVMehypothesis remainsunconfirmed. Inspection scope138c461d... stopped after
emptycheck,peak1527808B,events/swap0. Smallreadonlysourceanalysis confirms why
pre-generationexceptionlosescontext; thislimitationisrecordedinD313triagedoc.

Existing failure-aware analyzer extended with explicitD313sourceSHA/samplecount;
no all-success gate weakened. Auditpassed inputidentity/nativebinding/timing/
resource/remotechecks while correctlyretaining thefailedall-correctgate.132/132
HTTPtransfersmatched,131480060wirebytes,nopacking.4036samples,peak20682014720B,
minhost93777801216B,high/max/OOM/swap0. Protected147seal retains onlytheknown
authorizedbaseline replaychange, rawpass=false retained. Nofigures/ranking.

Initialoffline manifestincorrectlyincludeditsownempty-filehash. Preserve
failure_audit1 anditsproducersource incommitb74cc96eb015fa70b92c0c993e3b63791c930221;
correctedfailure_audit2 excludesmanifestself-hashing andall16input/2outputSHA
checks pass. Context/failurepayloadsbyte-identical, noGPUrerun ordatareplacement.
ContextSHAd04e93b1c86c67e776ad1b9251e380cca03fb0f4c621ea1368f2a82e49b018d8;
manifestSHA8439a6259bc510ce68055fa271a5b41aff36bf26d7c98f888caad7415b01711f.
332tests passed23.564s, logSHA8091bbc6c10c5e83b414541cb65ca34c9852fe75a0ef5f21945b94a333b20344.
AllCPUaudit/testscopesemptychecked/stopped,latestInvocation561dc5f...peak1169936384B,
events/swap0. NoGPU/remote/CPUjobrunning. completion_checkpoint.json savesproofs.

Backup5851098a54c1625e2459247578dc8bd4f5031df1 pushed tofaaslora_origin/
retry14_continuous_queue_v2; remoteHEADmatchedexactly14:08:16. Usermanifest
unchanged/notstaged; onlythispost-pushstatusnoteisnewuncommittedownwork.
Next: deterministic existing-fixture test
of legal Remote->NVMe followed by selected HOST file loading. This isnot yet a
provenhistoricalrootcause; preserve strictcontent/owner/clock/pathnegatives and
normalleasecleanup. No newGPUreplay until tested/backed-up correction. Serving
source/config/profiles unchanged, no retuning,3B/baselinesorfigures. PlanandV1
fullyread; sourceplanSHAunchanged0c808509... . Numerical/warm/Resident qualification
stillopen; keep7B->3B->baselines anddo notdeclareG1/G2achieved.

## HISTORY — D313 full4000 W0 RUNNING with request failure; retain full evidence

2026-10-07 12:54+08 update: this turn found NEW correctness evidence, not a
blocker or terminal job. Same service Invocation80a935ab... and watchdogPID727967/
start85562420 live. At12:53:49 sample447fresh, hostavailable96813424640B,
service16905019392B, high/max/OOM/swap0, noabort. Live counters308terminal,
307success/1failure. Exact failed requestreq_00216, target7, at856011.559573558,
ValueError, native_contract_matched=false, no instance/prompt/token hash.
Source requestSHA7ff6c2640bc25b95b5777e5123aa3bf30b735985bbdb1528e0d0f804ac726c15.
Service failure reason prefix maps topreloading_planner.py:317:
`remote preparation interval lacks its verified transfer identity`.
This is not yet a proven root cause, remote bandwidth failure or recurrence of
D311 HOST source pin failure. Exact source/path/span needs terminal audit.
Do NOT suppress the check, invent transfer IDs, erase failed request or tune
while live. Continue same complete replay (no safety reason to interrupt);
retain all4000offered and failed outcomes. This run cannot pass all-success
qualification. After normal completion/release, stream full outcome and correlate
failed preparation with actual request/background fetch and remote journal.
No source changes/tests/figures/3B/baselines while measuring. Previous turn was
progress (prepared/tested/backed-up/launched), current run is verified live.
Latest13:26:40: sameInvocation live, sample2391, live2603terminal2602success/
1failure;host95657398272B/service19522818048B,disk165352411136B. Memory events/
swap0,noabort. Current continuation=verified wait: resumed the existing monitoring
cell through its last four observations, then ten more40s-spaced actual polls;
also checked the live service/log at13:20:20. The monitoring cells ended, but the
experiment did NOT end. No terminal launch.json, restart, additional failure,
serving/configuration change, test or figure. Earlier read-only causal-path/
negative-test/official-source triage remains a hypothesis; failed field unknown.
Five500rotation boundaries passed. Plan still0c808509..., fully reread this turn.
No comparison/new experiment; continue same run and audit after resource release.
Fresh monitoring receipt: rawroot/monitor_132640.json. Source code remains the
tested/pushed0cf3c1d checkpoint; documentation/status changes are uncommitted.

2026-10-07 12:46+08. Launched ONCE12:46:15 in tmuxprimelora_d313_full via
existing runner explicit d313. Source0cf3c1dd10a4894516feff203f38d346ee5389aa
was tested/pushed and remote HEAD verified before launch. Full plan/V1 read.
Rawresults/ieee_tc/p2_backend_qualification/d313_20261007/7b_full_w0.
Same D311 controls/model/generation/D307 profiles and D312 serving repair;
original4000 W0, fresh coldpaths, no repeated100smoke/calibration/retune.

Localserviceprimelora-tc-svc-5f22e0fd9d164d4da6e436853ec74651.scope,
Invocation80a935ab83d448ea855fe407fd7b8cf7,inode19676953;auxiliary
primelora-tc-aux-51f52d57d77045ad8e9c3dbb174cd6c6.scope,
Invocationd533b4cbfccc48509c1045ea0cd15ad8. At12:47:48 sample90fresh,
hostavailable107632066560B/servicecurrent5390348288B, all memory events/swap0;
business arrivals started, original4000not yet complete. No safety abort.
WatchdogPID727967/start85562420. Actual exec receipt is service_identity at
top level. Observation timeout is NOT experiment terminal. Monitor same handles.

Remoteidentity captured in rawlaunch_identity.json:
3B66c5f09539d14991b3034c41e2f7e12c/PID1667350;
7Bff8c400381a848879e77a7ad8abc32ae/PID1667352;
monitorda6d8927efd3426fb98d7043f9ad5fb4/PID1667355,
unitprimelora-artifact-monitor-d313.service. Authenticated health passed;
clockremote-process-monotonic:1429d5adb4824bfc8531934865660c99.
Actualeno1NIC1000/full. ExactjournalD80/7b/transfers-e2f7e6a634244c26b8cb241bbb075335.jsonl.
Reuse existing published gzip cache, no rebuild. During measurement no service
change, full-result analysis, tests, bulk hashing or remote reconfiguration. Small
read-only failure-source/history inspection does not alter the measured service.
Read-only triage is saved inD313_REMOTE_PREPARATION_READ_ONLY_TRIAGE.md: possible
Remote->NVMe followed by fastest HOST file choice creates differing valid paths,
but actual failed-field/root cause remains UNKNOWN until terminal evidence.
No repair implemented, no validation weakened. OfficialvLLM/Python source
checked; plan fully reread before this triage. At13:06:50 sample1218fresh,
1119terminal1118success1failure, same live job,noresourceevents. After localterminal/
cleanup, verify all3remoteinvocations, stopMONITORFIRST then copy exactjournal
asremote_transfers.jsonl andD313remote_monitor.log to rawroot. Stream full audit
after release; keep all failures/unknown numerical/warm/Resident qualification.
Keep7B->3B->baselines; no figures or target qualification claimed yet.

## HISTORY — D313 original full4000 prepared; tested backup then launch once

2026-10-07 12:45+08. Plan and frozen V1 fully reread; D312 audited checkpoint
f9a8ef7245428ac1442ac496a0dd943557854e58 is already pushed/remote-verified.
Fresh D313 config differs from D311 only in output/cold cache identities;
model/control/generation/arrival/D307 profiles unchanged, same D312 repair.
Config SHA44e9506d196140f063582fa42fefb7aefbe6c57560a5477edcb0af04f365065a.
Existing prelaunch verifier passed40service/24preparationclasses, exact D307
child binding and D312 raw evidence hashes; receiptSHAfe0d372577941dfbff25d17fee8731754e270aa3b4fac572fd9d5d4b3871a323.
332 basic/capacity/artifact tests passed22.756s. Prelaunch/test scopes checked
empty/stopped, peaks496627712/1169313792B, memory events/swap0. First prelaunch
management command rejected unsupported scope CPUAffinity before any process
was started; log retained, then existing taskset form used. Not a GPU attempt.
Protected147seal retains only known authorized baseline replay change; raw
pass=false preserved. No serving/parameter change or new profiles/figures.

No GPU/remote experiment running yet. Existing remote wrapper copied with new
D313 identity only; one-time read-only gzip cache reused, never rebuilt. Next:
commit/push this tested checkpoint, start remote with captured identities/health,
then existing runner d313 ONCE in tmuxprimelora_d313_full. Capture service,
auxiliary and watchdog handles and monitor the same run, no restart on timeout.
After terminal cleanup stop captured remote MONITOR FIRST, save exact journal,
audit full outcome (stream large JSON). No more100smoke/calibration/retune.
Numerical/warm/Resident qualification remains open;7B->3B->baselines, no figures.

## HISTORY — D312 native100 completed/audited; backup then unchanged full4000 next

2026-10-07 12:35+08. This turn=progress: native100samecandidatecompleted/audited.
100/100successful; resultSHA15fd5e6cb3147da440b0254244ca62ae882e8db475c13151e47ef7f6cc3ba7c2.
Launchpass/all3returncodes0,nativeGPUcontextrelease/servicepathremovedconfirmed.
Remoteidentitieschecked thenMONITOR FIRST stopped12:33:49;all3inactive/MainPID0/
success. Exactjournal/monitorcopiedonce,SHA36eb05e9.../cedc9023....
AuxInvocation7c38192a2f21450182a9cbebdd1c9be0 emptychecked/stopped,
peak161038336B/events0/swap0. AnalysisInvocation662d695916b049cfa3d2479a5dfc354f
emptychecked/stopped,peak121987072B/events0/swap0. NoGPU/remote/CPUjobrunning.

Existing data-only audit passes fixed-output native timing/10binding checks;
29transfers/50650865wirebytesmatchremote,nopacking.265resourcesamples,
peak19406376960B,minhost93270421504B,zero high/max/OOM/swap. Protected147seal:
only known authorizedbaseline replaychange, rawpassfalse preserved.
MeanTTFT1018.513423ms,P952401.525333ms,meanTPOT27.591759ms,P9533.803534ms,
physicalU799.027391GPU-s. VersusD310: meanTTFT-13.34%,P95TTFT-19.81%,
meanTPOT-1.10%,P95TPOT-3.02%,U-0.79%. Oneprefix observation, no causal/statistical
win orwhole-W0claim. D170provisionaltiming98/100only;N_correct/formalSLOunknown.
Curatedpaper_results/ieee_tc/p2_backend/d312_host_validation/smoke1/;
auditSHA098c400b984db85e246df112c61139faae2122a97673a85c628f5cf86f58c3f6.
Raw/curatedcompletion_receipt.jsonsaveallidentities/cleanup/sourcehashes.

Next: backup audited results, prepare freshD313original4000W0 with ALLD311model,
control,generation,arrivalandD307profilesunchanged; onlynewoutput/coldpaths.
Use existingrunner/prelaunch, noextra100repeat, profiling, budgetscan orretune.
NoD313config/service/launchexistsyet. Fullplan/V1readthisturn; rereadnextturn.
Keep7B->3B->baselines;nofigures. Numerical/warm/Residentqualificationremainsopen.

## HISTORY — D312 native100 RUNNING; monitor the same handles, no retune

2026-10-07 12:29+08. LaunchedONCE12:28:23 in tmuxprimelora_d312_smoke,
existingrunner explicitd312, backed-upsource9b1c1171d559a0c4b65b421215def4f4025b60ea
(remoteHEADverified). Rawresults/ieee_tc/p2_backend_qualification/d312_20261007/7b_prefill_validation_100.
SameD310settings/D307profiles; no additional servingchange. Plan/V1readfully.

Localserviceprimelora-tc-svc-8ec460d17f094114b2654dcdb5733822.scope,
Invocatione75f86030ea4499d8fb53a7da928f367,inode19665756;auxiliary
primelora-tc-aux-6ad537d54fce4e2eab844ac6cb053516.scope.
WatchdogPID650198/start85455208,service sample21fresh,noabort/events/swap.
Hostavailable111091310592B,service1903022080B;initializing,notcomplete.
Do not relaunch when observation waits expire. No serving/config change whilelive.

Remoteidentitiescapturedinrawlaunch_identity.json:
3B263cad0014bd4467a225cf536c16e7f4/PID1647451;
7B21063adba0ac4c1688ee3fa150cc9f8b/PID1647453;
monitor3dc7f42e524b45038711e6562c67068b/PID1647456,
unitprimelora-artifact-monitor-d312.service. Actualeno1NIC1000/full.
Authenticatedhealthpassedprelaunch;7Bclockremote-process-monotonic:db3ca84d79724e23b25386eb272ce8a1.
ExactremotejournalD80/7b/transfers-3edd13a5dee847a39564f5e2d5272cc2.jsonl.
Afterlocalterminal/cleanup,verifyallremoteidentities thenstopMONITOR FIRST,
copyexactjournalassmoke_remote_transfers.jsonlandmonitorassmoke_remote_monitor.log.
Runexistinganalyze_d298_smoke --campaign d312 inboundedCPUdomainafterrelease.
Nofigures,recalibration,3B/baselines;formalnumerical/warm/Residentqualificationopen.

## HISTORY — D312 native100 prepared, same settings; backup then launch once

2026-10-07 12:27+08. Previous goal turn=progress:972testrepair/failureaudit/push
completed, remoteHEAD3118892b67392b53ccb77b8659af5f8efeaed386 verified. This turn
full plan and frozenV1 reread. Existingrunner/prelaunch/analyzer only extended
with explicitd312identity. Config equalsD310exceptresult/NVMe/HOSTpaths;
SHAe40990b308ae9912b09db55d3ad26c4d71be1a037033966e25b9437e8399f86f.
Prelaunch binding passes40service/24preparationclasses usingD307measurements,
receiptSHA366a5996b1c466b9db51582c745e62103f97e2f471ebacb023a2d4017a177658.
Prelaunchscope820be435...empty/stopped,peak496459776B/events0/swap0.
309basic/audit smoketests passed21.987s;scopeb413af3e...empty/stopped,
peak1169199104B/events0/swap0. No serving/config/profile/formula change.

Remoteinactive, actualeno1NIC1000/full; server/monitor SHAunchanged. No cache
rebuilt. Newremote-start wrapper copied existingD311commandwithnewuniqueID,
but not executed yet. NoGPU/remote experiment running. Nexttestedbackupthen
remote start/health/capturedidentities, launch once in tmuxprimelora_d312_smoke.
After measurement stop capturedMONITORFIRST, copy exactjournal, release all
resources and runexistingdata-onlyd312audit. Full4000samecandidatefollows; no
recalibration, parametersearch, 3B, baselinesorfigures. Finalqualificationopen.

## HISTORY — D312 transaction repair validated/backed up; fresh native100 next

2026-10-07 12:20+08. No GPU/remote experiment or CPU job is running. Source plan
reread in full after compaction; SHA unchanged0c808509... . This turn made new
correctness/evidence progress, not a blocked or completed-goal condition.

D311 incomplete audit is saved at
paper_results/ieee_tc/p2_backend/d311_full_w0/failure_audit1/.
Context SHA6258bd0366dd9e0383b8ab381f46654cb0bcf82cffb0defea4a6c00c8324cd85;
manifest SHAb591e669348c90af2be09daa2c55fa31c6d7eda5f94a291f6830bc82eea3dceb.
452,534,210-byte main_outcome streamed, SHA62c764d3ae13600e42a572f3f8671d3a8110a1ded2970530f493346cc6cefd92.
N_plan4000/arrived1738/submitted1732/terminal1640/interrupted16 retained;
N_correct/N_good unknown, no ranking, latency comparison or figures.77/77remote
transfers matched,88,341,140wire bytes,no request packing.1746resource samples:
peak19,653,382,144B,minhost93,354,754,048B,zero high/max/OOM/swap/abort.
Protected147-entry seal retains only known authorizedbaseline replay change;
rawpassfalse is preserved, not claimed147/147pass. Allremoteunits rechecked
inactive/MainPID0/success this turn. Audit scope8b001129... empty/stopped,
peak74,440,704B,events0/swap0.

All four production acquire_confirmed callers inspected. Native HOST preparation
alone missed the current-observation+pin owner transaction. Before repair two
CPU interleavings reproduced exact entry owner/epoch exception; after holding
the existing RLock only across snapshot/select/acquire,199targeted and972broader
tests passed (3.677s/91.805s). Native RPC is explicitly tested outside the lock;
single attempt, source/lease preservation and GPUslot invariants verified.
Route admission's stale selected-snapshot rejection remains strict. No retry,
fallback, relaxedcheck, formula/config/profile/budgetchange. D310 remains intact.
Regression full logSHA41963d36a395a765602eab187d910ca6df3572cfc87b1c08060181b2f03d4c83;
scope5e276f5c...peak2,627,743,744B/events0/swap0,emptychecked/stopped.
CPUreceipt paper_results/ieee_tc/p2_backend/d312_host_transaction/cpu_checkpoint.json.
Design/history/source comparison: D312_NATIVE_HOST_SOURCE_TRANSACTION.md.

Tested checkpoint1d92ec4e78fd5aa33355f06a028e3f0efe7f35a8 pushed to
faaslora_origin/retry14_continuous_queue_v2; remote branch HEAD matched exactly.
Diff/staged-path/credential-pattern checks and all19source SHA checks passed;
the user's generated manifest was not staged and its SHA remained unchanged.

Next: fresh D312 native100 identity using same D310 settings
and D307 profiles, followed only after cleanup/audit by fresh full4000 W0. No
new configuration/remote service has been prepared or launched yet. Reuse existing
runner/prelaunch/smoke analyzer; only add explicit new identities and coldpaths.
Do not repeat calibration or tune around failure; no3B/baseline work or figures.
Numerical/warm/Resident qualification remains open. Keep7B->3B->baselines.
One-time delivery cache already exists fromD78; reuse it, do not rebuild.

## HISTORY — D311 full W0 aborted; native HOST file snapshot/pin race, no rerun yet

2026-10-07 12:05+08. The original job is TERMINAL, not a polling timeout.
launch.json SHA0573b52eb763b157f0742cbdf5d1fccbd1541e88d77f0f8ee58fb268087e66af:
pass=false, service=-15/replay=1/watchdog=0. Service traceback originates in
_queue_ieee_native_host_preparation.execute, run_all_experiments.py:19112,
acquire_confirmed entry check: ConfirmedSourceConflict, owner/epoch changed.
The generic launch classification protocol_or_launcher_error is retained, but
its replay connection loss follows the serving exception; do not blame remote
bandwidth, OOM or claim a launcher-only failure from that generic field.

Replay reports N_plan=4000, arrived=1738, submitted=1732. Physical summary
reports 1640 terminal/native-contract-complete, 16 interrupted, N_correct=null,
measurement_complete=false and observed U=6656.206557878992 GPU-s. This is
truncated observation, not full performance or a feasible ranking point.
Main outcome is 452,534,210 bytes: use streaming projection, NOT json.load.
No ordinary final result JSON was produced. Preserve all failure evidence.

Local GPU context release and service-path removal confirmed; live GPU compute
census empty. Exact auxiliary InvocationID checked, cgroup.procs read empty;
scope stopped, peak260755456B, high/max/OOM/swap0. Remote three identities
checked, monitor stopped FIRST at12:04:54; all three units inactive/MainPID0/
Resultsuccess/ExecMainStatus0. Exact remote journal and monitor log copied once
to raw root; SHA3d1fe5be... andf642f83f... respectively. No experiment running.

Next: finish bounded failure/cleanup audit and preserve tested backup; inspect
all production acquire_confirmed call sites. D307 already locks snapshot+pin
in _ensure_local_async, and staged victim-file holds also lock. The native HOST
preparation path reads snapshot then pins without the shared owner transaction.
Reproduce competing publication with existing CPU fixtures before repairing;
retain strict admission's intentional stale-snapshot rejection. No catch/retry,
relaxed epoch/content check, formula change, retune or repeated source profile.
Short/full replay only after tested correction and fresh execution identity.
Keep 7B->3B->baselines, no figures. This turn has new failure evidence/progress,
not a blocked condition; do not restart D311 or overwrite its raw directory.

## HISTORY — D311 full4000 W0 monitored until native preparation exception

Verified wait 2026-10-07 11:54:25+08: same tmux remains live; service and
auxiliary InvocationIDs rechecked unchanged at 11:49:45. Sample 1181 is
fresh: 1,061 arrived / 1,054 successful / 0 failed, four natural runtimes.
Host available 89.386 GiB, service 16.955 GiB, disk 155.300 GiB; memory
events 0, swap 0, no abort/warning or foreign GPU compute. Final launch.json
absent. Progress from 779 successful at 11:49 to 1,054 is observed, not inferred
from process liveness. This is the original active 4,000-request run, not
stalled/restartable. Continue the SAME job next turn; no code/config changes,
new experiments or performance conclusions during measurement. Full plan
reread this turn; frozen V1 was read in the preceding monitoring turn, and no
new system comparison/configuration selection was performed. Turn classification
is verified wait, not blocked. Remote 3B/7B/monitor identities last rechecked
unchanged at 11:46; monitor log fresh. No remote management actions performed.
Read-only D294/D295 review confirms the bounded numerical-repeat diagnostic is
complete; do not reopen it during this full run or infer Full qualification.
AuxInvocationbf066f492f8748ac99c7dcafba87b927 captured andverified.

2026-10-07 11:35+08. LaunchONCE11:34:27 in tmuxprimelora_d311_full,
explicitd311 existingrunner. Sourcecheckpointd04aa2290ca99e28be0878a4a2f872afa44643db
pushedandremoteHEADverified. Rawresults/ieee_tc/p2_backend_qualification/d311_20261007/7b_full_w0.
Original4000arrivalmap,sourceSHAefb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c,
viewSHAa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d,
notice851316.427208622,business851376.427208622(sameclock,60sgap).
No config/control/generation/servingchange beyond priorD310sourcefix.

Localserviceprimelora-tc-svc-b0462f9fa6c24b62bc057e0aa62f5905.scope,
Invocationdeaba153790e4cab90d7140255cd97a6. Auxiliary
primelora-tc-aux-1318ecc33e0c408bbe077b8e4c6a9912.scope.
WatchdogPID410730,startticks85131615 verified against/proc;sample12fresh,
hostavailable104.40GiB,noabort/foreignGPU/memoryevents/swap.
Initializationinprogress; final launch.json/result notpresentyet.

Remoteidentitiescapturedinrawlaunch_identity.json:
3Bf2fdfa0ff2644680b2ab03062988c9af/PID1576638;
7Bbc3a2940e95d4177a96a91e7c7572ca0/PID1576640;
monitor8bbd44ed2ed14ae6b30d453288bec9aa/PID1576648,
unitprimelora-artifact-monitor-d311.service. NIC1000/full.
7Bclockremote-process-monotonic:d909ae3a97bb45d997db20d5bb145317;
exactjournalD80/7b/transfers-02418f60c5b944dcb0c1696978d947a2.jsonl.
Immediateprelaunchhealthwasconnectionrefused; SAMEcapturedprocessbecamelistening
and authenticatedhealthpassed BEFOREinference. Norestart/configchange; preserve
theprelaunchattempt. No requestpacking; D78cacheunchanged.

Next: monitor SAMEhandles. Do not runheavyanalysis/tests/compression,change
servingcode/config,orremoteservicesduringmeasurement. Afterterminal/localrelease,
verifyallremoteidentitiesthenstopMONITOR FIRST,copyexactjournal/monitorlog,
auditallrequests/resources/input/token/binding/timing/protectedseal. Existing
completeanalyzersdoNOTyetacceptd311; addexactterminalsourceSHAaftercompletion,
preservingallsuccessgate. Ifanyfailure,usefailure-awarepath,notfakepass.
Nofigures/3B/baselines, no7Bqualificationorcomparativewinclaim.

## HISTORY — D311 full4000 prepared; same candidate, no serving change

2026-10-07 11:32+08. Previousgoalturn=progress: D310complete/audit/backupdone,
83989b5eaff007d66a18c7cbf28790ae50cdbdcb remoteHEADverified. FullplanandV1
reread (truncatedplanopeningrereadseparately). D311followsregisterednextstep.
Freshconfig copiesD309except uniqueoutput/coldcachepaths; actualchildmatches
D307measuredconfig,40service/24preparationclasses bind; D310model/controlsame.
Prelaunchpassed configSHAe865efc2f3d964f40f1335d54a91ebdfd2bd2406cbc52e56666ecf3e254e434e.
CPUprelaunchscope73d766c86bb0457ab3932d55ab6e482b emptystopped,
peak496689152B/events0/swap0. NoGPU/remote service startedyet.
289basic smoketests passed24.013s; scope13178c17efb843bd9c35343872618c49
peak1170231296B,events0/swap0,emptystopped. Protectedseal147entrieschecked:
onlyknownauthorizedbaselinereplay change, rawpassfalse retained. Sealscope
3b73840e0fc0414180efbcec055cd661 emptystopped. Backup/launchnext.
RemoteD80server/monitorSHA unchanged,
priorD310servicesinactive. FullpreparationnotesD311_FULL_READ_PIN.md.
Do not repeatD310, recalibrateprofiles orchangebudget; nofigures/3B/baselines.

## HISTORY — D310 native100 completed/audited; same settings full4000 next

2026-10-07 11:28+08. D310100/100successful,no failures. ExactsourceSHA
5410400edf56e5fd1c814f4e8cb58b9fcac81f5f71dbbce9d282086061af44ea.
Nativefixedlength,timing,10bindingchecks pass;29transfers/50650865wirebytes
matchremote,no packing.266resourcesamples,peak18923728896B,minhost93902819328B,
high/max/OOM/OOM-kill/swap0. Launch/service/watchdog/replayallpass0;
nativeGPUreleaseconfirmed,servicepathremoved,tmuxabsent,computeGPUcensusempty.
Allremoteidentitieschecked thenMONITOR FIRST stopped11:26:33; all3inactive,
MainPID0,Resultsuccess,status0. Auxiliaryemptychecked/stopped,peak160755712B.
Analysisca29fb5a5a964c9ebadc7a2d27e8c85f empty/stopped,peak122183680B,events0.
Rawandcuratedcompletion_receipt.json retainidentitiesandSHA. No jobs running.

FrozenV1data-onlysummary: meanTTFT1175.321556ms,P95TTFT2994.614306ms,
meanTPOT27.899638ms,P95TPOT34.857805ms,physicalU805.409961GPU-s.
VersusD308 same100inputs: meanTTFT+15.04%,U+17.61%;P95TTFT-2.19%,P95TPOT-13.30%.
This is a mixedsingle-prefix observation, NOToverallimprovement, acceptance or
causal speedup. D170provisionaltiming96/100only; N_correct/formalSLOstillunknown.
Nofigures. FailedD309notdiscarded; newsourcefixnotclaimedas performancegain.
Curatedpaper_results/ieee_tc/p2_backend/d310_read_pin_validation/smoke1/.

Next: backuptheseauditedresults, then freshD311original4000W0 with ALLD309/D310
model,profile,controls,generation andarrivalsettings unchanged exceptnewcoldcache/
outputpaths. ReuseD307profiles andexistingrunner, verifychildcontextasbefore.
Do not repeatcalibration, scanbudgets, rerunD310or switchto3B/baselines.
Formalwarm/Resident/numericalqualification remainsopen; noG1/G2claim.
Sourcecheckpoint4ea7daacab6014d5aa3a1d874f74da6c3c9ea34f is alreadypushed.

## HISTORY — D310 native100 smoke running after tested backup

Latestlivecheck11:24:12+08: sameownedserviceInvocationcbf01e8e466544cb9d1a1463691e958a,
watchdog352543/start85054807. Sample150fresh,33arrived/27successful/0failed,
4naturalruntimes, noabort/foreigncompute/memoryevents/swap. Hostavailable91.25GiB,
service14.12GiB. This is live100prefix, not finalperformanceor a completedrun.

2026-10-07 11:22+08. Checkpoint4ea7daacab6014d5aa3a1d874f74da6c3c9ea34f
pushed tofaaslora_origin/retry14_continuous_queue_v2; remoteHEADverified.
StartedONCEtmuxprimelora_d310_smoke, existingrunner explicitd310. No newsettings.
Rawresults/ieee_tc/p2_backend_qualification/d310_20261007/7b_prefill_validation_100.
Localserviceprimelora-tc-svc-3e3bf2defcf9472a9f4cbe05ca589ed4.scope;
auxprimelora-tc-aux-4b3595c0098d4dacad2edb2d992143b2.scope,
auxInvocation6a65cc88866347f3924fa8a0a2066734. Livewatchdogsample10, noabort,
foreigncompute ormemoryevents; initializationinprogress, finalresultnotyetpresent.
Do not relaunch or change serving code/config during measurement.

Remoteexactidentities:3Bd82da67794e24457aae316d7f6e19a3d/PID1561841;
7Bf8d8da3ae3614737a8a0a0449ad32fbc/PID1561843;
monitore75fe1878ab448c0b4cebf2f0afaa6da/PID1561846,
unitprimelora-artifact-monitor-d310.service. ActualNIC1000/full. Authenticated
7Bclockremote-process-monotonic:1cea38e08e8641f6b971a59703448e45.
ExactremotejournalD80/7b/transfers-4b63e08b20494a8e84835f9c18be73c6.jsonl.
Afterlocalcompletion verifyidentities andstopMONITOR FIRST, then capturejournal
as smoke_remote_transfers.jsonl andmonitor as smoke_remote_monitor.log.
PrelaunchOSpythonhealth importfailed(numpyabsent),noHTTPrequestmade; existing
CPUenvironmenthealthpassed. Noenvironmentinstall/service/configchange made.

## HISTORY — D310 repair/prelaunch preparation complete, before native smoke

2026-10-07 11:19+08. Fullplan/V1 reread. User's once-only delivery-cache
approval is already implemented by D78:1,581,215,261 archive bytes across two
500-adapter pools. Reuse, no new pool or republishing. Goal stays7B→3B→baselines.

D309 failure-only audit completed using existing validators, not a relaxation
of the all-success gate. All4000 offered input identities match retainedD303;
3999 successful returns pass10binding checks, native fixed-length count and
TTFT/TPOT/stage identities (maxerror0ms). The one failed request is preserved
verbatim.132remote transfers/131480060wirebytes pair exactly, no request packing.
4040resource samples: peak20676182016B,minhostavailable93657604096B,swap0,
high/max/OOM/OOM-kill0. Complete physical lifecycle15744.284432824003GPU-s;
N_correct/N_good remainunknown, no feasible ranking or performance acceptance.
Seal147entries has only previously authorizedbaseline replay script change;
old results unchanged, do NOT claim rawseal147/147pass.
Curated: paper_results/ieee_tc/p2_backend/d309_full_w0/failure_audit1/.

D310 deterministically reproduced exactfootprint-epoch exception with unchanged
content under a legal allocation observation inside already charged envelope.
Actual D309kernel event remainsunobserved. Removed only post-validation self-epoch
rejection; entryowner/epoch,content,inode,allocation andlease protection retained.
No retry/catch/fallback/budget increase or formula change. HistoricalD90/D93/D307
and officialLinuxstat/ext4/FIEMAP/vLLM0.30code reviewed. DetailsD310_CONFIRMED_READ_PIN.md.
Initial fixture's immutablemanifest setup failure preserved separately.
Targeted32tests passed; broader950tests passed98.677s; added failure-accounting
path plus existingoverlap19tests passed. CPU regressionpeak2647199744B,
auditpeak1017995264B; allmemoryevents/swap0. Exactempty scopesstopped.

NextD310100prefix config freshlyprepared; child config equals measuredD307,
40service/24preparationclasses stillbind; everyD308model/control/workload setting
unchanged except unique output/cache paths. ConfigSHAe9a453df6327d46f10777d2d3d3251ff5386dde01c76b32285193ea2a7099c7a.
Existingrunner explicitd310case. Prelaunchbindingpassed; no model/remote service
launched yet. Save testedcheckpoint first, then native100smoke and, only after
cleanup/audit, original4000freshreplay. No extra calibration/budgetscan,figures,
3B or baseline work. No7Bqualification claim.

## HISTORY — D309 full4000 finished with ONE source-footprint conflict; correctness first

2026-10-07 11:00+08. D309 replay finished naturally:4000 terminal records,
3999 successful returns,1 failure; no restart or configuration change during
measurement. Exact failed request req_03242/medical_lora_0110, requested186,
actual0, error `ConfirmedSourceConflict: confirmed file source footprint epoch changed`.
This is NOT an all-correct run and cannot enter G1/G2 feasible ranking. Do not
drop that request, fabricate its timing/tier, or relax the successful4000 validator.
Numerical qualification remains unknown independently of this visible failure.

Final launch.json exists/pass=true, service/watchdog/replay returncodes0;
native_gpu_context_release_confirmed=true, service_path_removed=true. Native
GPU census empty, local service inactive, tmux absent after actual completion.
4040 resource samples, terminal census and service-domain-gone events recorded.
All live checks showed no memory protection or foreignGPU event; full archived
resource/remote/input/token/binding/timing/protected-seal audit still pending.

Remote identities all checked BEFORE stopping captured MONITOR FIRST at10:56:43.
Its trap released both ownedartifactservices. All3units inactive/MainPID0/
Resultsuccess/ExecMainStatus0. Exact journal copied to
results/ieee_tc/p2_backend_qualification/d309_20261007/remote_transfers.jsonl,
monitor to remote_monitor.log. Immutable completion_receipt.json now records
these observations and the one failure. No service remains active.

SourceSHA811aadf1cc61d14f58b7ebf59a54f1406d05c8e161003f8fd6da947da662257b;
launchSHA9dce5dd3b42759a8990759ee29e6b23857706d7289dfa6e023153d4cab5e5716;
remotejournalSHA646cac692b6ba1a32377575036bece9f0e789ad22e301c042cb4cfc0e70cbf0a;
remotemonitorSHA953b02441c558f94080b03b8211b9eaba1d359a7f625df1539822709e1b92ad2.
Final source310MiB, main_outcome669MiB: use bounded CPU scope/streaming for
further inspection, not repeated unbounded whole-file loading.
Read-only failure inspection scope15dbf434ff8548dc978dba3345464d01 stopped empty;
3/4GiB/swap0,allowedCPUs2-3/26-27,peak810909696B,allmemoryevents/swap0.
Replay auxiliary scope48c579b34c2b4393b08967d9e1c7373e also observed empty
and stopped; peak333053952B,allmemoryevents0. Both local domains nowinactive.

Next: preserve a failure-aware audit, inspect req_03242 and related source
history, and reproduce exact footprint-epoch conflict deterministically before
changing serving code. _validated_source publishes physical-footprint updates
by incrementing source_epoch; acquire_confirmed checks that epoch again after
validation (residency_manager.py:801). This is a code-path lead, not yet proof
of the filesystem/contending-event cause. Review official mechanisms before
repair; preserve content/owner/inode checks and physical budget constraints,
not a catch/retry/fallback workaround. No new token-budget scan or immediate
GPU rerun. Existing analyze_d297_full/check_d299_full_context have NOT been
extended to d309; audit_run correctly requires all requests successful, so
do NOT bypass it for this failure. No figures,3B or externalbaseline work.
Serving code/config remains checkpoint1b6aab450094a5dc63c9104e32aa5b713f10eb13
(already backed up). Goal remains7B→3B→baselines, no7Bqualification claim.

## HISTORY — D309 full4000 W0 running; same owned job monitored to completion

Verified wait 2026-10-07 10:40+08: same tmux and service InvocationID live;
watchdog4122999/start84491727 verified against/proc, sample3115 age<1s.
Progress3056→3210→3276→3388 terminal; arrived3397/4000,success3387,failed1.
Natural scale-in then scale-out observed after first500 boundary; no intervention.
Latest fullresource checkservice18.78GiB,hostavailable88.81GiB,disk155.12GiB;
allmemoryevents0,noabort/warning. Remote3B/7B/monitor same capturedInvocationIDs live;
no foreignGPUcompute, final launch.json absent. This is an actively progressing
full replay, NOT a blocked/no-progress/restart condition. No concurrentheavywork,
code/configchange,backupcompression or figures. Monitor SAME handles next.

At business49:30 first failure became visible in service.log:
`ConfirmedSourceConflict: confirmed file source f` (truncated summary).
Read-only source inspection locates footprint-epoch check in
faaslora/memory/residency_manager.py:801; full failure record is still pending.
Do not infer the precise race/OS cause from this prefix. No safety event or
foreign GPU compute observed. Continue original replay; no restart/replacement,
retry patch or tolerance relaxation. This run cannot qualify as all-correct.
After actual completion/release/remote cleanup, inspect full failed request and
preserve a failure-aware audit before any performance comparison. Do not weaken
the all4000 validators to make it pass or drop the failed offered request.

2026-10-07 09:48+08. Fullplan/status/V1 reread beforelaunch (large output was
truncated; missingmiddle rereadseparately). Testedcheckpoint
1b6aab450094a5dc63c9104e32aa5b713f10eb13 pushedandremoteHEADverified.
LaunchedONCE09:47:48 in tmuxprimelora_d309_full, explicitd309case. Raw
results/ieee_tc/p2_backend_qualification/d309_20261007/7b_full_w0.
SameD308candidate/profile/control,4000originalW0, no additionaltuning.
Localserviceprimelora-tc-svc-b60d677d0e3d49bc9134ec14cde1dcf2.scope
invocation95140ab7bf8e45188e6304f6f89a6945;
auxprimelora-tc-aux-dc5c7e5471cb4d03a6ee9139216b8a2c.scope;
watchdog4122999/start84491727. Original72/80GiB/swap2 andaux3/4GiB/swap0
attachedbeforemodelinit; allow_exec=true, no foreigncompute. Sample29,
hostavailable101.18GiB/service4.06GiB, no warning/abort. Disk169618407424B atlaunch.

Remote3Bd40b78efa7fe45799ab2cb6100a157cf/PID1422317;
7Bfc7d0cbb731a4069aef2117e84f6f50a/PID1422319;
monitorprimelora-artifact-monitor-d309.service
4fbee00a03ea414cb52f5a4a0e2426cb/PID1422322.
AuthenticatedhealthBEFORElaunch: remote-process-monotonic:01a8b562d7724c7db091fccdea8f5ddb.
Exactjournaltransfers-18e0ba2321024fce97f97790b53a62c0.jsonl inD80/7b;
monitorlogremote/tc/d309_20261007/remote_monitor.log. NIC1000/full again.
DoNOTstop/restartremote,alterconfig,hash/compress/install,orparallelheavywork
duringrun. MonitorSAMEjob. AftercompleteandactualGPUrelease, stopcaptured
remoteMONITOR FIRST (trap releasesownedartifactunits), verify3unitsinactive/PID0,
copyexactjournalto d309_20261007/remote_transfers.jsonl andmonitorlog, savecleanup.
SealactualsourceSHA/completionreceipt beforeexistinganalyzerextension(d309 notyet
addedtoanalyze_d297_full/check_d299_full_context; extendexplicitidentityafterresult,
no arbitraryhashbypass). CompareD303 all4000 withsameD170temporaryruler, retain
tradeoffs, noCI atn1/noformalqualification. D308shortresultNOTcandidateacceptance.
No figures, no3B/baselines/numericaldetour. Fullgoalactive; nofinal7Bclaim.

## HISTORY — D308100 complete/audited; D3094000 prepared, next actual full replay

2026-10-07 09:47+08. D308 completed100/100, native token/source/time/binding
checks pass; sourceSHA d3a660845b47b5d32db6664dccad6a2fc48818f61ea7abbfab095ec7867c04b7.
Launchpass/returncodes0, nativeGPUrelease andservicepathremoved confirmed.
StoppedremoteMONITOR FIRST withrecordedInvocationID;3unitsinactive/PID0/exit0.
29UUIDpairs/50650865wirebytes allmatched; no requestpacking.264samples,
peak18296344576B,minhost94835871744B,high/max/OOM/swap0. Historicalseal147
unchangedexceptsameknownbaselinescript. Nofiguresornewpool/trace.

D308TTFTmean1021.630202/P953061.757701ms,TPOTmean28.791707/P9540.205643ms,
physical684.796752GPU-s. VersusD302100:U-14.34%,TTFTP95+34.35%,TPOTP95+17.02%.
Retainregression; cannotchoosecandidate from100prefix. D170timing-only97/100,
formalSLO/Ncorrectunknown; neither7Bqualificationnorstatisticalsuperiority.
Curatedd308_prefill/smoke1 containsaudit,binding,comparison andsourceidentities.

D309configpreparedfromD308, only4000extent/scenario/freshpaths; actualchild and
allprofiles/controls equal. Newbindingreceipt verifiesD303onlydeclaredcandidate
differences, D308nativeauditrawSHA,and exactD307measuredchild. ConfigSHA
e193de4e4f1bfc2793f22f60c2eb6702ac7c9643ac949d7f53bb10facbafa8a5.
Existingrunnerexplicitd309case added; prephelper --verify-prefill-fullpassed.
No servingcodechange. CPUauditandprelaunchscopesempty/stopped;readbacksretained.
NEXT backup exactfiles, rereadfullplan/status/V1 beforelaunch, D309remote wrapper
withfreshmonitoridentity/sameD80services/D78cache;healthBEFORElaunch. Runexisting
run_d293_7b_prefetch_validation_100.sh d309 onceinownedtmux.4000W0 mustcomplete
beforeaccept/reject. No additionalcandidate,warm,source,numeric orbaseline detour.
Afterlocalrelease: stopcapturedremoteMONITOR FIRST,copysourcejournalandmonitor,
sealcompletionSHA, extendexistinganalyze_d297_full/check_d299_full_context explicit
d309identity andcompareD303. No benchmarkthresholdchange. Goalactive7B→3B→baselines.

## HISTORY — D308 Full100 RUNNING; monitor same owned experiment

2026-10-07 09:40+08. Fullplan/status/V1 read before launch; tested checkpoint
6a29eb0277c92ed84a8e53107315414d97dcf0b6 pushed/remoteHEADverified.
D308 launchedONCE at09:39:34 in tmuxprimelora_d308_smoke using explicitd308
existingrunner. Originalmemory/disk/gpu gates passed; no foreigncompute.
Localserviceprimelora-tc-svc-ed858a4613544374a64b3a299dd5af5e.scope,
invocation4d69d69534ed4a4dbffe0682100ba8c2;
auxprimelora-tc-aux-e332c63284be4b84b15797236655f701.scope,
watchdog4086031/start84442331.72/80GiB/swap2GiB limitsandseparateauxverified.
Remote3Bc463d999180b4dc1b241b5737e3621cf/PID1410843;
7Ba726ed1ebf7148bb8e9407ecfd9bbc3f/PID1410844;
monitorprimelora-artifact-monitor-d308.service
0cefa1cc28164f3590b259aa8ea91a4a/PID1410848.
Authenticatedhealth capturedBEFORElaunch, clock
remote-process-monotonic:f0fa3a3c358641fb91d133304b8b95b3;
journaltransfers-d7142310de8e4b848c5b2825a20e833e.jsonl. NIC1000/full.
ReuseD80service/D78publishedcache; no rebuild,hashscan,compression or concurrent
heavywork duringmeasurement. Monitorcurrentjob; localrelease then stopowned
remoteMONITOR FIRST, verifybothartifactservicesinactive, copysmoke_remote_*
logs/journal. Run existinganalyzer --campaign d308 onlyaftercomplete/pass.
Ifvalid proceed4000W0. Nofigures,3B/baselines; goalactive.

## HISTORY — D307 source complete; D308 bound Full100 prepared, next actual replay

2026-10-07 09:39+08. Fullplan/status/V1/run/monitor/githubskills reread.
D307 attempt2 completed248/248 (8warmup+240measurement;48per source), launchpass,
nativeGPUreleased, sourceworkspacesremoved. SourceSHA
2d1611f5c0b11a7c77d31b26e68adb9bdb4c1d323576679a6451919b0c543750;
actualsourcecommitf0813771f35ffe30463ae19d1abec38f89658f14. No repeatedsourceconflict.
248UUIDpairs/1059355374wirebytes allmatched/no requestpacking, NIC1000/full.
429.007688GPU-s,437samples,peak5700685824B,minhost107253698560B,events/swap0.
3/8 outputgroupsvary acrossprocessingconditions; numericalqualificationNOTproved.
StoppedcapturedremoteMONITOR FIRST, bothartifactservicesinactive/PID0/exit0;
exactjournal/monitor/stoplogs retained. No GPU/service live now.

Existing exporter d307-attempt2 succeeded; new initialization/service.json SHA
f9e6efb10e69d5d1e321a7a500e9502c2240ab108c56395fe54fe7b2b82bd73a,
preparation.json e0706f8aaced852c0308de6a81e70d3b6d169b9ae95ecac4e8ae38b23ae9f951.
40service/24preparationclasses,500staticdomain covered,not500numericalproof.
D308100 config prepared fromD302; only512budget,profileidentity/newtimings,
freshoutput/cache/scenarionames. D303controls unchanged. ActualFullchild equals
D307measuredchild; exactSHA/model/contextloaders pass.19profiletests pass.
CPUscopesempty/stopped; readbacks retainedd308_prefill/cpu_checks.json.
No newservingcode, formulas, control changes or figures. Usermanifest untouched.

NEXT backup exactD307profiles/D308preparedfiles, originalgates, startnewowned
remote d308 service monitor, authenticatedhealth BEFORElaunch. Use existing
run_d293_7b_prefetch_validation_100.sh d308 ONCE inownedtmux. Analyzer extended
with explicitd308 only; samechecks and D302descriptivecontext. Aftercomplete,
localrelease thenremoteMONITOR FIRST, joinjournal, audit; ifvalid prepare4000W0.
No repeatedwarm/source/500coverage,3B orbaseline work. Finalwarm/Resident/
numerical/G1G2 stillopen.7B optimization mainline and goalactive.

## HISTORY — D307 attempt2 source measurement RUNNING, same config/cases

2026-10-07 09:25+08. Fullplan/status/V1/run/monitor/githubskills reread; previous
turn progress was verified failure plus atomic-source repair, not a wait/replan.
SAME248cases/config/spec startedONCE in tmuxprimelora_d307_attempt2 at09:24:43.
Actualsourcecommitf0813771f35ffe30463ae19d1abec38f89658f14; source1 under
d307_20261007/attempt2 (not parent failedsource1). Health capturedBEFORElaunch,
clockremote-process-monotonic:75c3ae1040484b56b1ba2ff8ec46a554, NIC1000/full.
Journaltransfers-fd35a777142b42649c6f234002efd9fe.jsonl, not clockUUID.
Remote3B b6d43ee1e0944ef88c264b218a209d44/PID1390776;
7B82719170624f4fa1aa7e886edf0eadf4/PID1390778;
monitorprimelora-artifact-monitor-d307-attempt2.service
b170316587144a3a83fc70c54cc31f3c/PID1390781.
Localservice1c5dc94a679446ffbe6eab698d3dc111.scope
invocationd5b3d64c28374675bf5e8d199fb5ae94;
auxc053020e16834680b6bdd378df02939b.scope;
watchdog4020338/start84353156. Originalgatepassed; no relaxedresource limits.
No concurrentheavy task/codechange. Monitor SAMEjob; afternativeGPUrelease stop
capturedremoteMONITOR FIRST, verifyartifactunitsinactive/PID0, copyexactjournal/
monitor/stop logs. Export_d298_profiles.py d307-attempt2 onlyif complete/pass.
Then newbound100Fullsmoke and4000W0; nofigures,3B orbaselines. Goalactive.

## HISTORY — D307 attempt1 failed; atomic source pin fixed and CPU-tested, retry next

2026-10-07 09:21+08. Source run actually executed and stopped on correctness
conflict, NOT OOM/network/budget evidence. Planned248,started16,completed15,
232notstarted. source-profile/w1/l6 translate_lora remote verified publication
then ConfirmedSourceConflict at _ensure_local_async snapshot→acquire gap.
SourceSHA44cfe310a60e48fde07ef2c7b26cec56077bc4a68f09d874fdb3a069b81aeead;
raw retainedd307_20261007/source1*.json. No initialization profiles exported.
Observed118.506740GPU-s;16remoteUUIDs/68345508B alljoined/no packing. NIC1000/full.
130resourcesamples,peak5451767808B,minhost107744985088B,allmemoryevents/swap0.
Actual GPU context release confirmed and source workspaces removed. Owned
remote MONITOR stopped FIRST; existingtrap released3B/7B artifactservices,
allinactive/PID0/exit0. Journals/logs copied. failureaudit.json is named
paper_results/ieee_tc/p2_backend/d307_prefill/attempt1_failure.json.

Read history3c86c85/8cff52a/2b54c69 and officialPythonRLock/vLLMworker manager.
Two deterministic CPU interleavings reproduced SAMEerror beforefix (2errors).
Fix: one existing owner RLock transaction for re-observe/select/pin after
materialization. Network/nativeRPC/await remainOUTSIDE; no retry/fallback,
no relaxed epoch/content checks, no formula/source-class change.57related
CPUtests pass. Tested checkpointa578fab9f92d8577a4ce9658b0e87e62a8f79470
pushed andremoteHEADverified. Exact8files staged, SHA/diff/syntax/credential
checks passed; usermanifest untouched. Serving source now
6dcbabbfc0ce580f7982d6927fa4da51910a781974388fdd37abccdef14d12bc.
ThreeCPUscopesempty/stoppedbyInvocationID;peak<556MB,allmemoryevents/swap0.
Feishu configabsent, no notification. Disk169436831744B>150GiB, GPUcensusclear.
Comparison withD303 includesnecessaryatomicityrepair plusbudget/profile;
doNOT claimpurebudgetcausality. Exact competing thread in rawfailure unknown.

NEXT explicit run_d298_capacity_source.sh
d307-attempt2 uses SAMEconfig/spec with newraw d307_20261007/attempt2, never
overwriteattempt1. Rawattempt2 already containsONLY start_remote.sh (syntaxpass),
derived fromD298 withONLYnewrun/monitoridentities. Remote outputplanned
d307_attempt2_20261007, monitorprimelora-artifact-monitor-d307-attempt2.service.
No remote or local attempt2 started yet. Use preparedwrapper,authenticated
health BEFORElaunch (existingcondaPython, OS-Python lacksnumpy); reread
fullplan/status/V1 before launch and pass original safety gates. Export arg
d307-attempt2 aftersuccess+release only. Then100Fullsmoke/newprofiles,4000W0.
No extra tokenbudget sweep/warm/coverage/cache rebuild/numeric qualification/
figures. No new GPUrun currently;7B→3B→baselines mainline and activegoal retained.

## HISTORY — D307 source measurement RUNNING; monitor same owned job

2026-10-07 09:13+08. Fullplan/status/V1 and run/monitor/github skills reread.
Reused D298 remote wrapper with only fresh D307 identities; unchanged D80 cache,
service and monitor. Remote NIC1000/full observed. Source launched ONCE in tmux
primelora_d307_source at09:12:59, source commit5b6280a37b39cd606f0ef9a57a6977062cb21e99.
Local svc62d579c7735e443bb897fb1779476c22.scope invocation0165438c1bc141638fad2d2bd2f8679f;
aux9970f8aa443d4ae3b90905036a4ea3c9.scope, watchdog3969818/start84282767.
Existing gated-launch attached watchdog before model init;72/80GiB,swap2GiB,
separate3/4GiBaux/swap0 readback passed. Initial foreign compute empty.
No new serving code/config changes or parallel heavy work while measuring.

Remote3B invocation5a0a3fd2eb794d198f3c995437dee4ae/PID1378250,
7B17c1d050da0b4c05814e1c67c6426ecf/PID1378252,
monitor b2488db854e74ed1a1068e756e4eda27/PID1378255.
Authenticated direct health clockremote-process-monotonic:18dfc3312d2f453190f544a0f3ebc009;
journalfilename transfers-81e442f033a64c6dbd7b1bb1342e95b1.jsonl (not clockUUID).
Initial OS-Python health helper lacked numpy; no HTTP request was issued. Correct
existing conda helper succeeded at09:13:10 during engine initialization after
gated launch. This timing is recorded, not claimed as a prelaunch health receipt.
Raw d307_20261007/{start_remote.sh,remote_start.log,remote_health.json,source_commit.txt,
source1_launcher.log,source1_launch.launch/} retain actual identities.

NEXT monitor SAME248-call source run, no duplicate launch. After local release,
stop captured remote MONITOR FIRST, verify its trap releases both owned artifact
services; copy exact journal/monitor/stop logs and export_d298_profiles.py d307
in bounded CPU scope. Then100Fullsmoke/new measured profiles, then4000W0. No
repeat warm,500HTTPcoverage,numerical qualification,cache publication or figures.
7B goal not yet achieved;3B/baselines/formal comparisons still pending.

## HISTORY — D307 single token-budget candidate prepared; source measurement is next

2026-10-07 09:06+08. Fullplan/V1 reread, once-only remote cache authorization
alreadyimplementedD78/D80, no rebuild. D306checkpoint63fd6a56d9d4d8c2aebd7753818df1c530484002
pushed andremoteHEADverified. D307 keepsD303 controls/cap8/slots8/rank16/context1024,
changes onlynativeiterationbudget1024→512, with newly measured initialization
timings REQUIRED. Hypothesis/primarysources/history in D307_NATIVE_PREFILL_BUDGET.md.
No GPU/remote/service behavior change executed, no figure. Goal remainsactive.

Existing initializer had exact performance config binding even for completed
lengths. Explicitnative_fixed_output_length_transfer_v1 now accepts ONLYdeclared
iterationbudgetchange under identical fixed-output contract; verifies oldnative
completion/input/sourceSHA, records source/targetidentity, changesprofileIDs.
Naturalstop/extrachange/implicitreuse reject; exact service/preparation timing
loaders unchanged and still reject oldbudget. No futuretargetprior, no latency
relabel. D170512uniquecompletedlengths reused, NOT remeasured.58CPUtests pass.

ReusedD298 prep/collector/export scripts; addedexplicitd307 path, same248cases
(8warmup+240measurement),31waves/eightreps,500staticpool/rank≤16. Actual child
bindingpass,9nestedreferenceSHAsverified. Source configSHA
56704b6189eef8c878f66125d3867adafc680cc2386b3e1c616fce139c0436ec;
specSHA0a1044fb7233a8784fd0a336e2ee7de01c55d7801043990c1708fc75242df258.
Preparedcurated p2_backend/d307_prefill/{prepare,development_contract,source_spec,
launch_config_binding,prelaunch_verification}.json. Rawd307_20261007 existsEMPTY.
DoNOTrepeatprepare oroverwrite, doNOTpromote preparationtoGPUmeasurement.

D115historical9.19GBmain_outcome losslesslycompressed to260.19MB; twiceverified
SHA66f2019b... thenonlyuncompressedduplicate removed. Failure retained, request/
curated/publicationunchanged. storage/20261007_d307_d115_lossless_compression.json.
Free169499455488B>150GiB. Compressionhigh19656/3833events(noOOM/swap), separate
fromperformance. systemdmemoryinotifywarn retained; directlimitsreadable. Existing
watchdogattachment/livememory/diskgate mustpassagainbeforeGPU, no protocolchange.
FiveCPUscopesempty/exit0,peak<0.60GB,noevents/swap; nativeGPUcensusclear.
Protected147soleknownbaselineSHAunchanged, usermanifest/servingrunnerunchanged.

Tested preparationcheckpoint95e6d35134aa264619213a96f77bf0debe2e8ea6 pushed and
remoteHEADverified at09:07+08; exact16files staged, diff/syntax/reference/credential
checks passed. Only preexisting usermanifest remains tracked-dirty. All fiveCPU
andtwo storage scopes empty/stopped byInvocationID; no inference live.
NEXT readfullplan/status/V1 andrun-experimentskill,
reuseD298 start_remote.sh withONLY freshD307remote/run/monitoridentities andsame
D80files/cache. Confirm actualNIC/health, captureInvocationIDs andjournal identity.
Launch existing run_d298_capacity_source.sh d307 ONCE aftergates inownedtmux.
Then monitorSAME248job, localrelease, stopcapturedremoteMONITOR FIRST (ownedtrap
stopsartifactunits), joinjournalUUID/clock andcopylogs, export_d298_profiles.py d307.
JournalfilenameUUID != healthclockUUID. DoNOT repeat500HTTPcoverage/poolpublication.
Thenprepareexisting100Fullsmoke withD303controls/newboundprofiles andsamecandidate,
then4000W0. Finalsharedwarm/Resident/numerical/G1G2stillopen; no3B/baselinesuntil7B.

## HISTORY — D306 prefill overlap complete; one smaller native token-budget candidate next

2026-10-07. Full plan/V1 reread. User once-only remote cache authorization already
implemented D78/D80; reuse without rebuilding. D306 reuses D300/D303 all4000 and
D305 sealed intervals, no GPU/remote/service/config/figure work. 25 CPU tests pass.
Output p2_backend/d306_prefill/analysis1, summarySHA
636e9fa6863a1d84a6da3324614a3b8226e05a9fd9e445799b0ee57088492964.
18 manifest checks pass, D305 outputs unchanged; protected147 retain only known
baseline script difference, userdirtymanifest/servingrunner unchanged.

D303 129/139 TPOT timing misses overlap same-owner other-request prefill envelope;
27/36 no-startup/no-load misses do so. Own-prefill mean202.050→200.492ms did NOT
slow, other native occupancy1.636→1.801. Envelope NOT kernel/co-batch witness;
duration-dependent exposure/concurrency confounding, no causal subtraction/CI.
All requests retained, D170 only provisional, N_correct null/G1/G2 still open.
See D306_PREFILL_OVERLAP.md; do not extend endless conditional offline analyses.

NEXT D307 single preregistered model-specific native tokenbudget1024→512 candidate,
everything else D303 including soft controls/cap8/slots8/rank16/context1024 fixed.
512 is a controlled halving, NOT a fitted automatic optimum. Reuse D298 source
collector/representatives/waves/completed lengths; newly measure affected timing
profiles with exact config binding, then100smoke and4000full. Do not relabel old
latency or repeat unrelated warm/numerical qualification. Review historical
D298/D300 and officialvLLM0.30/Sarathi before code; references in D306 doc.
No D307 config or GPU run yet. Disk160586784768B<150GiB gate, audit reusable
caches/lossless log compression first; no threshold reduction/unique evidence loss.
CPU three scopes empty, peaks26.1/754.7/12.9MB, no memory events/swap.
Mainline7B→3B→baselines→comparisons→ablations/sensitivities remains; goalactive.

## HISTORY — D305 narrow load overlap complete; distinguish prefill/concurrency next

2026-10-07. Fullplan/V1 reread, user's once-only remote cache authorization
confirmed already implemented D78/D80; do NOT regenerate cache/pool or repeat
remote qualification. No GPU/remote/serving/config/figures in this turn.
D300/D303 source4000 + sealedD304 reused with strict owner/native receipt joins.
Output p2_backend/d305_native_overlap/analysis3; summarySHA
e29817f0a5f449a4ab4f6a5e2114d030e2840994c651b11a4307027aa7ca0bbb.

D303 97/139TPOTtimingmisses overlap same-owner request-native-load,42doNOT;
36have neither load nor otherstartup overlap. Loadcount2097→2121, mean34.232→
36.138ms. Mean per-request load-overlap/decode ratio0.671→0.798%, notcausal
latencyorGPUutilization.129/139overlapacquirefence butfenceduration short;
doNOTremovefencesfromcorrelation. Conditionalconcurrencydiffers,all4000retained,
N_correctnull/D170onlyprovisional/noCI. No evidence yet to chooseasync-loadas
primaryfix. ReadD184/D202/D209, officialvLLM0.30UniProc/worker/optimization.

Initialread-onlyquery usedwrongJSONkey; analysis1/2 wrongly boundedacquirebelow
bycontrolleradmission. D184 GPUprotectionprecedesadmission; correctedanalyzer,
notservingcontract. Regressiontestadded,20CPUtests pass. Failedattempts retained.
Seven3/4GiBswap0CPUscopes nohigh/max/OOM/swap; successfulanalysispeak751775744B.
SeeD305_NATIVE_LOAD_OVERLAP.md for fulltable/interpretation/limits.

NEXT one retained native_scheduled→first-token peer-prefill envelope check,
especially36no-startup/no-loadtailrequests, preserving owner/clock/concurrency.
Envelope is NOTkernelprefilltime. No GPUdiagnostic or D304/D305 rescan needed.
Use this to select one falsifiable candidate, not endlessconditionalgroups or
blind tokenbudget/NUMA/concurrency tuning. History/officialsources beforechange.
Finalsharedwarm/Resident/G1/G2 remainopen;7B→3B→baselinesunchanged. Diskstill
below150GiBnew-heavy gate; auditedrebuildablecachecleanupbeforeanynextGPU.
19manifestchecks pass,D304outputs unchanged.147sealretainssoleknownbaseline
scriptdifferenceSHAunchanged,userdirtymanifest/servingrunnerunchanged. All8CPU
scopesempty/stoppedbyInvocationID,noevents/swap; verificationpeak13611008B.
GPUcensusclear,disk160605229056B. Exact12-filestage/diff/credentialchecks passed.
Tested checkpoint e3344141699c6bc242db72feb91647514ae98f38 pushed andremoteHEAD
verified. No inference running; next narrowprefill/concurrencycheck, not a
repeatloadanalysis or fullreplay. Goalactive.

## HISTORY — D304 decode-tail localization complete; narrow native-load overlap next

2026-10-07. Fullplan/V1 reread. Previous D303 audit was progress; this turn reuses
D300/D303 source4000 and activation clocks, no GPU run/config/warm/figure work.
Existing analyze_d297_full.py gains --decode-tail-output; new strict interval
helper/tests, no serving change. Output p2_backend/d304_decode/analysis1.
Summary SHA4f7338731ae2eee6f88888a1e1c4d2c28cf90bef45c9badfb11a03c49b148e3f.
13CPUtests pass. D300/D303 TPOT misses118/139 match sealed audit.

D303:36/139misses overlap other-replica startup,103doNOT; rejectstartup-only
explanation.131/139are2-32outputtokens.112/139overlap same-replica broad preparation
envelope; peer-nativeoccupancy mean1.636→1.801. All conditional, duration-dependent
exposure, NOTcausal stalls or independent replicates. Entire4000rows retained.
Acquirecompletionfence4000fields:mean1.266588ms/P950.164327/max121.364890.
DoesNOTcoverreleasebarriersorallutilityCPU; donotremove safetyfencesfromthis.
ReadD176/D184/D202/historycd5b68d andofficialvLLM0.30UniProc/worker/optimization.
UniProc worker method synchronously executes before Future, plausible locus,
notproof. FullrecordD304_DECODE_TAIL_DIAGNOSIS.md containsallcounts/limits.

NEXT use EXISTING receipt native_load_invoked/start/end/acquired/fence fields to
intersect NARROW same-owner load+acquire windows with peer decode, contrast broad
envelope/concurrency. All4000havefields, GPUhitloadtimestampsnullbycontract,
notzero. No newGPUdiagnostic needed yet; keepanalysis1immutable. Do NOTblindly
changeNUMA,concurrency,tokenbudget,scalingthresholds,or repeatwarm/numerical/full.
One evidence-led candidate only after this distinguishes actualnativework.
N_correct/finalsharedwarm/Resident/G1/G2 remainopen;7B→3B→baselineunchanged.
Disk160617480192B<150GiBnew-heavy gate, no lowering; auditrebuildablecachebefore
nextGPU. AllsixownedCPUscopes bounded3/4GiBswap0, noevents, empty/stoppedbyID;
inspect/lifecycle/tests/analysis/receipt/verify peaks719650816/719773696/26763264/
742457344/719581184/12849152B.17manifestinput/outputSHAchecks pass; protected
147seal retainssoleknownbaselinescriptdifference, usermanifest/servingsourceSHA
unchanged. Ordinary diffcheck andexact10-filestage/credentialscan passed.
Checkpoint a56c0b192481bfb94bd52cf64b51b64d1ff566df pushed andremoteHEADverified.
No inference running; next is the narrow retained native-load overlap, not a
new full replay or another broad startup-correlation analysis. Goalactive.

## HISTORY — D303 fully audited; confirmed-state fix retained, 7B goal still open

2026-10-07 08:15+08. Full plan/V1 reread; existing analyzers extended only for
D303 frozen identity/new outputs. SAME completed4000 reused, no GPU rerun,
warm/profile/numerical repeat, figures,3B or externalbaseline work.
validated1/audit.json SHA29508b8165e940eb98f55f848442c77f277c2ea413d535229a1f0be8df710ca2;
context1/context.json SHA01004237b17c14e3566d75abf7ba78f2706cf771fe23ef86e7ad040731226ff8.
4000native-contract complete, zero failure; all inputfields equalD300 and10binding
checks4000/4000.191outputhashchanges retained, NOTnumerical qualification.
132remoteUUIDjoins/131480060wirebytes/no requestpacking; same remote work asD300.
4039resourcesamples; peak22123409408B/minhost93849612288B,swap/high/max/OOM0.
Eightphysicalleasesclosed; remoteexit73postmeasurementguardexplicitlyretained.
Protected146/147 andsoleknownbaselineSHA unchanged; usermanifest/source unchanged.

Mean/P95TTFT1288.659541/2899.768391ms,mean/P95TPOT34.463616/68.104561ms,
E2E4.964285s,physical15476.010963GPU-s. VersusD300:meanTTFT7.3434%lower,
P95TTFT5.3744%lower,U0.5985%lower BUTP95TPOT10.2034%higher. D170timing-only
3810/4000=95.25%,+0.10pp,notformalSLO. N_correctnull/sharedfinalreferencesopen.
Strictwaitpartition162.868184+37.218513+460.790672=660.877369ms;
meanrouting214.665988→174.322781ms(-18.7935%),notadditiveagain.
Observedcachehitmax1537→3118; finalviewauthoritativeread3178→3040,
notexactRPCtotals/causalconditionalgroups.7scaleups/4downs vs5/2closed-loop.
TTFTtimingmiss81→55 butTPOT118→139; keepallregressions. Onefullrunperconfig,noCI.

56timing/route/waittests pass. ThreeboundedCPUscopes3/4GiB,swap0,CPU2,3,26,27
peaks740319232/1063010304/501297152B,noevents,empty/stoppedbyInvocationID.
NativecgroupreadinitiallyusedwrongUIDpath; correctedfromsystemdControlGroup,
nochangeinlimitsoranalysis. Source/config/oldresultsnotmutated.

All30manifestinput/outputSHAchecks pass; bounded checksum scopepeak2379776B,
noevents/empty/stopped. Exact14-file stage andcredential-marker checks passed;
CRLF-aware whitespace check preserves validatedCSVbytes. Analysis checkpoint
c59b38e5606673ee40055960b9fc57bfd0946238 pushed andremoteHEADverified.
NEXT singleevidence-led7Bdiagnosisofdecode-tail/regressionandremainingadmissionwait.
Reviewhistory/officialsourcesbeforeanyservingchange; noautomaticthresholdsweep,
soft=hardtarget,orwarm/source/numericalrepeat. RetainD301correctnessfixbutdoNOT
declare7Bclosed. Beforeanynextheavy, disk160551415808B is below150GiBgate;
auditendedrebuildablecaches/referencesandsafecleanup,neveruniqueevidence.
NoGPUorremoteservicerunning;goalactive. FullreportD303_FULL_CONFIRMED_SCOPE.md.

## HISTORY — D303 full4000 completed/released; full audit next, do not rerun

2026-10-07 08:02+08. Same D303 completed4000/4000native-contract requests,
zero reported failures. launcher/replay/watchdog0, native GPU census clear,
servicepath removed, eight allocation journals released and no open leases.
PhysicalU15476.010963284993GPU-s; mutually exclusive windows sum verified.
N_correct remainsnull; no formal G1/G2/SLO or CI claim. Serving/config unchanged.
SourceSHA9bd86f5647a2723fa0aed67a98b92a72b477f725ef91b98a0182f348eaa7598e;
launchd2005aee486b8d0d8ac3fb8f0af9db3ba3a58a8940cd713fea01c58000d5ba9f;
physicalsummary594b964caf8dfabe6ba1c32fbb81c8c4ae6fd5c35577002a5044c01e0625a35c.
Read-only monitor session10742 exited after terminal receipt; no inference live.

After confirmed localrelease, both remote artifact units stopped by captured
InvocationID;inactive/PID0/success. Remote monitor exited73 at08:02:58 after
artifact stop (failed/PID0 retained), NOT allremote returncodes0. Its frozen
script uses73 for InvocationID mismatch, same-host journal confirms shutdown
order.3798samples/no safety_abort. SeeD303_FULL_CONFIRMED_SCOPE.md and
completion_receipt.json. Future teardown: stop captured monitor FIRST after
localrelease; its existing owned_stop trap stops matching artifacts; then verify.
No source/remote config change or rerun warranted by this post-measurement guard.

Remote transfers-a9a0e0da1b75417caa0d37ced0cb7d4f copied asremote_transfers.jsonl
SHAeed52207fbe0e3e711b8189c82043ea361c029a2e7191cd2dbfb5c2d031f925e;
remote_monitor.log SHA22c3d05984ba926ec3f1d25d36abf59eea4678de634db737a8f150b68e364664;
monitor script SHAda2c7d041e7fae858a404a7b104eade5162741faeaf9788d3a484080cf599b28.
Allthree local/remoteSHAs agree. Eight source/receipt hashes rechecked; protected
seal remains146/147, same soleknownbaseline-script difference; usermanifest and
servingsource hashes unchanged. One CPU metadata check used unsupported system
Python hashlib.file_digest; reused existing p.digest and verification passed,
no data mutation/performance rerun. Bounded scopes peaks14766080/11505664B,
nohigh/max/OOM, both empty. Last local disk160627437568B is below150GiB
new-heavy-run gate but above100GiB stop; do not launch nextheavy until safe
rebuildable-cache cleanup/reference audit, never delete unique evidence.

NEXT fullplan/V1 reread and existing analyze_d297_full.py/check_d299_full_context.py
extend onlyD303 identity, fresh curatedvalidated1/context1, same D300 reference.
Need native/binding/remote UUID/resource/seal checks and strict wait partition;
all full analysis is pending. No figures, warm/numerical repeat,3B/baseline work.
Goalactive; first complete audit and retain both gains/regressions before next
first-principles optimization. Do not rerun4000or100 merely because audit pending.

## HISTORY — D303 full W0 running, continue SAME run

2026-10-07 06:53:45+08 launched via existing runner argd303. Preregistered
checkpoint26f0b3fa133c53e9ad2bedb9857d6844e71f2a39 pushed/remoteverified first.
tmuxprimelora-d303-7b, panePID3358498; watchdog3358585.
Serviceprimelora-tc-svc-03b6fd78b5b14ff784b2ab064f16bb1f.scope invocation
4bcae8e9570a40ad835a653fdce32e13;
auxprimelora-tc-aux-431b908ca24247869ebb9d2f733e00cd.scope.
Rawd303_20261007/7b_full_w0, no prefix limit; new local cold caches, same D300
controls and D301 single fix. Initial native GPU census clean; watchdog ready.
07:26:41+08: 2016/4000terminal, all success=true, no resource abort. Same live
service invocation, launcher3358498 and watchdog3358585 verified throughout.
Observed servicepeak19671531520B, hostavailable95787307008B, high/max/OOM0.
Read-only poll session10742 checks those handles every45s and exits on launch.json;
resume that monitor (or verify same service directly), do not relaunch inference.
Full arrival window is3963.908903s; this is expected trace wait, not a blocker.
This continuation reread the full plan and inspected existing offline analyzers;
no serving/config changes, reference repeat, data processing or new experiments.
Live-handle checkpointb15372dc6ea9c38925f43a537a97bfcc34ddd4cd pushed and
remoteverified. No final outcome yet; do not launch another run or repeat D302.

RemoteunchangedD80 units and existing published cache, NIC1000/full.
3Bartifact invocation1dbe1c0f69a64958922fa3454d2403a2/PID1208555;
7B3840148199eb49e38f5cf952d4b59563/PID1208557;
monitor-d303bc2c06b4bab8433dbaf60b045b60cf53/PID1208560.
Healthclockremote-process-monotonic:327b103d3c164082a92a8e9e56f34855;
remote monitor/home/lab14/primelora_remote/tc/d303_20261007/remote_monitor.log.
No cache regeneration/packing or additional downloads. Prelaunch aux scope
f2053170009047aea03774b7aaad3799 peak13701120B/noevents, empty and stopped.

NEXT monitor this full4000; after verified physical release stop ONLY these
captured remote invocations, collect journal by healthclock and monitor with SHA
agreement, extend existing full analyzer to new output and compare D300/D303.
No figures, reference repeats, 3B or externalbaseline. Goal active and G1/G2 open.

## HISTORY — D303 full W0 registered, launch next (do not repeat D302)

2026-10-07 06:53+08. Full source plan and frozenV1 reread. D303 keeps all D300
configuration fields except new output/NVMe/HOST paths; structural comparison
passes. Single implementation change remains D301 confirmed-scope publication,
already371tests and D302100 validated. No additional tuning, reference or source
measurement. Config SHA18a5d633fac9a5cf13d4be260d323bb7796ad8698c4d7fd2e0493de141fcb3e5.
Serving source SHA d6ec59a9877023eafc9686251bdebce48a470e3af6c55efcc76fe2df8e3d2078.
Protected146/147 unchanged, sole known baseline-script difference SHA unchanged.
Fresh check four GPUs idle, host113370583040B available, disk161766203392B;
runner will recheck all safety gates at actual launch. No figures or baseline work.
NEXT backup preregistration, reuse D80 remote units/cache and launch existing
runner argd303 in tmuxprimelora-d303-7b. Capture live handles here immediately.
Full run remains exploratory; N_correct/shared final warm/Resident and G1/G2 open.

## HISTORY — D302100 completed/audited; next full W0 for the same D301 fix

2026-10-07 06:43+08. SAME D302 finished100/100native-token-contract requests,
0failure. launcher/replay/watchdog0, physicalGPUreleaseconfirmed, servicepath
removed. Threecapturedremoteunitsstoppedonlyafterlocalrelease;inactive/PID0/success.
Rawsource0a90b00fec4580979f3d200dade7910f345f5c5356f8f7fe6b05bb6560a6ff0e.
Curatedp2_backend/d302_frontier/smoke1/audit.json SHA
1a7f9a853905ae36db594b775efc483e2eee17747eb630af046340c2c42a84bb;
alloutputs/rawrefs verified.29transferUUIDs/50650865wirebytes,nopacking;
remote/localjournal andmonitorSHAsagree.264resourcesamples,peak19125301248B,
minhost94000787456B,high/max/OOM/swap0.147sealretainssoleknownbaselinescript
mismatch;nohistoricalresultsorusergeneratedmanifestchanged.

Mean/P95TTFT1000.507728/2278.951706ms,mean/P95TPOT27.283961/34.358794ms,
physical799.428274GPU-s,timing-only99/100. NotformalN_correct/SLO/G1/G2orCI.
D298descriptivecontext:meanTTFT12.2572%lowerbutGPU-s2.4156%higher;
meanrouting103.884988->112.065777msworse,authoritativefinalview70->66,
observedcachehitmax13->40. D298hasdifferentlowercontrolreference,notcausal
effectofpublicationfix.3outputhashchangesretained,binding100/100bothruns.
SeeD302_FRONTIER_VALIDATION.md. Nofigures,warm/profile/numericalrepeat,3B
orbaselinework. D301testedcodecheckpoint1b336d4alreadypushedbeforelaunch.

Data/analysischeckpoint61a0f6fd1ed6b0d02a0c1c2354aff23044fbc406pushed and
remoteverified;exact10filestage/checksum/secretschecks pass. Onlytrackeddirty
fileisusergeneratedmanifest,notstaged. Bothownedanalysis scopesempty/stopped.
NEXT onecomplete4000W0withD300config
andtheD301identity-publicationfixonly. Do notrepeat100orchangecontrolthresholds.
Recheckfullplan/V1 andnewoutput/resourcegates beforelaunch. NoGPU/remote
inferencecurrentlyrunning;goalactive,7Btargetandfullcampaignremainopen.

## HISTORY — D302 same single-candidate100 validation running, do not relaunch

2026-10-07 06:40+08. D301 checkpoint1b336d4e18ecc7e1ee935fa286cbea4711167d4c
pushed and remoteverified beforelaunch. Fullplan reread immediately beforeD302.
Config84c48c9e780cf2e31b01f3aa931f78354d1d9a4d4133eb07fe0357dfd9cf91bc;
only100-prefix/scenario/output/cache identities differ fromD300 configuration.
Launched06:38:25 via existinglauncher argd302, tmuxprimelora-d302-7b, PID3292247.
Serviceprimelora-tc-svc-8c605ee41180465abcaad53cc03b8730.scope invocation
355d6a41430f432ca82e970411c564dc, watchdog3292261,
auxprimelora-tc-aux-7ef89ae61f84485ab2d2c5f4e549f532.scope.
Rawd302_20261007/7b_frontier_validation_100. 27/100terminal at06:40:42,
no failure/memoryevents/abort; peak19125301248B,hostavailable98264129536B.
Startfree161811652608B, same150/100GiB and72/80GiB protections.

RemoteNIC1000/full; unchangedD80units andD78/D80publishedcache,no rebuild.
Captured3Bartifact invocationbd7921094f7f468a8f6c685f65bb92c6/PID1191484;
7Bcaaea886964b45fc9e456645b6d74763/PID1191487;
monitor-d3024317b7ea000a4905a26d05585c27a728/PID1191491.
Healthclockremote-process-monotonic:b78902bb54d74713a532a8c927d55f1e.
Monitorlog/home/lab14/primelora_remote/tc/d302_20261007/remote_monitor.log.
Beforelaunch, onehealth attempt usedOSpython lackingnumpy; no request replay
or inference started. UsedexistingCPUenv successfully; no dependency install.

NEXT finish SAMErun, verifyphysicalrelease, stoponlycapturedremoteinvocations,
copyjournalbyclock andmonitor, then existinganalyze_d298_smoke.py --campaign d302.
D298context hasdifferentlowercontrolreference; notsinglefactorcausalcontrast.
Newwaitsummary andsourceprotection check beforefull4000 againstD300. No figures,
3B/externalbaseline,warm/source/numericalrepeats. Goalactive,G1/G2stillopen.

## HISTORY — D301 repeated-read cause reproduced/fixed; D302 validation pending

2026-10-07. Still 7B optimization, no new performance run this turn yet.
Full source plan and frozenV1 reread after compaction. D301 reuses complete
D299/D300, new data-only output p2_backend/d301_wait/validated1; input/output
hashes retained. Mean D300 arrival/admission partition196.073016 +33.577356
+526.683647=756.334019ms; nested routing214.665988ms is not added again.
Actual external admission uses runtime capacity, not workload concurrency2.
3,178 final-view requests with authoritative read have mean routing267.622156ms;
822 without have9.927908ms. Conditional groups, not causal gains or RPC totals.

Found D289 publication bug: collector writes a fresh authoritative scope, then
identity publication unconditionally deletes it. Two production-order regression
tests FAIL on oldcode. Narrow fix retains only SAME owner/clock/epoch/source
identity entries, invalidates changed/malformed entries, leaves selected-copy
revalidation and IEEE equations untouched. Reviewed D200/D214/D215/D289 history,
IEEE received-state/recheck semantics, official vLLM0.30 LRU and Ray queue docs.
371 final CPU tests pass; no new parameters or generation/reference changes.
See D301_CONFIRMED_SCOPE_PUBLICATION.md. All five owned offline/test scopes
empty/stopped after identity checks; peaks<1GiB, nohigh/max/OOM.

NEXT backup this tested checkpoint, then existing launcher argd302 for100-request
7B diagnostic using D300 model/profiles/controls and new output/cache paths.
No warm/source/numerical repeat,3B/external baseline or figures. If qualified,
advance same single candidate to full4000 W0; do not call100 a G1/G2 outcome.
Formal numerical correctness/shared references and optimization remainopen.

## HISTORY — D300 full4000 audited; resource/latency tradeoff, 7B target still open

2026-10-07 06:14+08. SAME D300 completed4000/4000native-contract requests,
zero reported failures. Six physical leases allclosed, launch/replay/watchdog0,
native GPU census clear and service scope removed. Source SHA
0964d7a58feeebadf607e72f6f6ea7a09b949d6df44da684bb86d0c27e8be974.
Captured remote monitor stopped only after local release; all three services
inactive/PID0/success. Remote journal transfers-4fed620e4e2a48518c284170fb890217
and monitor log copied; local/remote SHAs agree. No request-time packing/cache
rebuild, rerun, warm/numerical repeat,3B/baseline or figure work.

Native/context audits complete innewp2_backend/d300_full_w0/validated1 andcontext1;
all4000inputfields matchD299;193outputhashchanges retained, numericalqualification
notinferred.132remoteUUIDjoins/131480060wirebytes/no packing.4038resourcesamples,
peak20565889024B/minhost94115979264B,high/max/OOM/swap0. Protectedseal146/147
unchanged, soleknown baseline-script SHA unchanged.35timing tests pass.

Mean/P95TTFT1.390791/3.064464s,mean/P95TPOT33.284821/61.798980ms,
E2E5.055340s,physical15569.192088GPU-s. VersusD299:GPU1.743968%lower,
P95TTFT6.231563%worse,P95TPOT1.094118%better;provisionaltiming3806/4000=95.15%,
down0.55pp. NotformaljointSLO/N_correct/G1G2orCI. Retainasexploratoryresource
tradeoff,notfinalqualifiedwinner;donotdiscardmerelybecauseTTFTrises. Fiveups/two
downs,all6leasesclosed. Meanarrival-to-admission0.756334sremainslargeststage.
Native/contextanalysisscopesempty/stopped;peaks734461952/1062809600B,noevents.
Nofigures,newGPUrun,warmsource/numericalrepeat,3Borexternalbaseline.

Data checkpoint e27b7d0b7ee3af485f281133b955636a3f0906d6 pushed toV2; allcurated
and source-reference checksums match, exact13-file stage/secrets checks passed.
Userdirtygeneratedmanifest wasnotstaged.35tests pass; owned testscope empty/stopped.
NEXT oneevidence-led7Bquestionabout
avoidablearrival-to-admissiondelayversusnecessaryreadiness/capacityconstraints.
ReviewC2softtargets/cap8historyandprimarysourcesbeforeanynewcandidate;notanother
lowerthresholdsweeporautomaticsoft=hardtarget. D300fullreportincludesalltradeoffs.
G1/G2,sharedfinalreferencesandfullcampaignremainopen. NorelaunchofcompletedD300.

## HISTORY — D300 original4000 W0 running; one request-path scale-in reference change

Live check2026-10-07 05:51:07+08: SAME service invocation6108fa3d... remains
active, launcher2875210 and watchdog2875311 live (2736/2735s elapsed).2967/4000
request terminal rows,0reportedfailure,0native-contract mismatch; notN_correct.
Two natural scale-ins and two subsequent reactivations observed; currently4
replicas, scale_up5/scale_down2. First scale-in at05:08:39 removed GPU2; lease
1ee6babca943409da3198498c0e57151 has release828167.456849019, workerreturn0,
GPU-56e0f9e0-c5f2-58b3-08b0-5f46beb9a036 absent from remainingheld census.
GPU2 reacquired828389.666044347, an observed222.209195328s unheld interval.
Second release: GPU1 lease3900e859356244d6b78b823bbe5cc653 at828728.669339698,
workerreturn0; reacquired828785.151136132, gap56.481796434s. These are actual
idle intervals, NOT complete-run netGPU-s improvement or preservedSLO evidence.
RemainingGPUs kept serving; global all-contexts-clear is correctlyfalse during
this partial release. No counterfactualGPU-s/performance claim.
At sample2698: hostavailable89.0292854309082GiB, high/max/OOM0/noabort.
Earlier sample1402 servicepeak19226914816B; latestdiskcheck161191829504B,
above100GiBstop; no storagework during measurement. This turn reread the full
source plan after compaction and verified continuing request completion on the
same live handles; no rerun/config/remote change, comparative analysis or figure.
Previously observed physical-reclamation intervals remain partial-run evidence.
Read-only SSH at05:46 confirmed all three captured remote invocations remain
active with the original PIDs; no service restart or configuration operation.
Frozen metricV1 reread in full at05:50; source plan SHA still0c808509... .

2026-10-07 05:06+08. Checkpoint26ec812ba5b15e8c4f5b18d1e99a58445b7c39b2
pushed and remoteverified BEFORElaunch. Config25debb773cc004a134e026b148c23d9af2565a8a7041db98e74f438efa65a378.
Started05:05:32 in tmuxprimelora-d300-7b, existinglauncher argd300. No source,
warm, numerical,3B/baseline or figure work. Original4000/4000source/W0/rate1,
traceefb903... viewa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
Raw results/ieee_tc/p2_backend_qualification/d300_20261007/7b_full_w0.

Service primelora-tc-svc-fd56e3b4c5a94d35bc741edc052037cd.scope invocation
6108fa3d32a84f9fb423d6c3a1587a35; watchdog2875311, aux
primelora-tc-aux-211d58503eb34befa558e228c1136e91.scope. Sample33: service
peak5046140928B, hostavailable107976048640B, high/max/OOM0, no warning/abort.
Same72/80GiBservice,3/4GiBaux,150/100GiBdisk. Startfree162910732288B.
Deploymentnotice827980.548269617, arrivalstart828040.548269617, sameclock.

Remote monitorprimelora-artifact-monitor-d300.service invocation
b035a165def248f898dc28cd38992426;7B31b65455bc1c48e5a84329f3575636f8;
3B67e4789b281046f5b2360a9776732232 (artifactonly). ExistingD80unitfiles match;
NIC1000/full; no cache rebuild. Authenticateddirecthealth succeeded before
deployment at827978.965600904, clockremote-process-monotonic:5d1c9a55e63b45f4ae1c112785523137.
Remote monitorlog /home/lab14/primelora_remote/tc/d300_20261007/remote_monitor.log.
Only stop these captured invocations after actual local release; capture current
transferjournal using the processclock, not a guessed journalfilenameUUID.

NEXT monitor THISrun, do not relaunch. Finish->physicalrelease->ownedremote stop
->native/input/remote/control/resource audit->compareD299/D300 under unchangedV1.
One observed development result cannot establish CI or finalG1/G2. Both may
regress; preserve all outcomes. D299requestsource remainsraw; selected historical
main_outcome files nowgzip with double-SHA receipt, not deleted evidence.
Allfour newlyowned empty compression/check scopes stopped beforelaunch.
Goal active, 7B optimization unresolved;3B and allbaselines staypaused.

## HISTORY — D300 single scale-in reference candidate validated; backup before full W0

2026-10-07 05:05+08. D299 remains complete; no inference currently running.
Full plan and frozenV1 reread, prior checkpoint197a64e matches remote. Reviewed
IEEE §3.1, D88 derivation, D297/D299 histories and Ray/Knative official docs/code.
ONE candidate D300 changes only ttft_lower_ms from281.278192 to2339.924886ms,
the Type-1 P95 of all100 already-audited D298 Full requests. This is an empirical
request-path development reference, not a revised evaluation SLO or proven
optimum. Queue/active limits,38scooldown,45swindow and every mechanism stayD299.
See D300_REQUEST_PATH_CONTROL_REFERENCE.md for rationale and limitations.

Config25debb773cc004a134e026b148c23d9af2565a8a7041db98e74f438efa65a378;
native reference audit recomputed; exactone nonidentitydifference verified.
56controller/lifecycle/timing tests pass. Same historical D299 observations have
420candidate-low samples and one43.252538s span; no counterfactualGPU-s claim.
No repeat warm/source/numerical run, figures,3B or external baseline.

Storage: five completed historical main_outcome logs losslessly gzip-compressed,
source/archive SHA round trips verified twice before rawduplicate unlink.
Request sources and curated evidence unchanged; receipt in paper_results/ieee_tc/storage/
20261007_completed_outcomes_lossless_compression.json. Free162918813696B;
unchanged150/100GiB thresholds. Both compression jobs no high/max/OOM.
146/147protected entries unchanged; soleknown baseline-script SHA unchanged.

NEXT save/push tested checkpoint, stop only empty owned check scopes, then reuse
existing remote startup/monitor and protected launcher argd300 for original4000W0.
Do not rerunD299, do not sweep this lower threshold or change evaluation limits.
On finish release/validate/data-only compare actualphysicalU, TTFT,TPOT,timing,
control/release events and request identities, including all regressions.
G1/G2, finalsharedwarm/Resident and N_correct remainopen; goal active.

## HISTORY — D299 full4000 W0 completed and audited; 7B resource goal still open

2026-10-07 04:46+08. Same D299 finished4000/4000native-contract requests;
zero reported failures,4physical leases closed, service/replay/watchdog0,
native GPU census clear, service scope removed. Exact-owned remote monitor and
both artifact services stopped/inactive/PID0/success. No run was restarted.
Source SHA df066b365fa13ecc25b225a793c362706157224965b8a7b360981d8cf45887ab.
Curated native/stage/control audit in p2_backend/d299_full_w0/validated1;
input/remote/resource/protection comparison in context1. All4000 inputs match
D297 and control bindings pass;213output hashes differ, N_correct remainsnull.
132remote UUID joins,131480060wire bytes,no request-time packing.4046watchdog
samples,peak20523708416B,minhost94323630080B,high/max/OOM/swap0. Protectedseal
146/147unchanged, sole known baseline script edit preserved.35timing tests pass.

Mean/P95TTFT1.306468/2.884702s, mean/P95TPOT32.918675/62.482613ms;
physical15845.533076GPU-s. VersusD297:P95TTFT14.14%lower, physicalonly0.09554%
lower, P95TPOT0.67137%worse. ProvisionalD170timing upper3828/4000=95.70%,
notformaljointSLOorG1/G2. Sharedfinalwarm/Resident/numericalqualificationopen.
No figures or3B/baseline/warm-repeat. D295bounded numerical diagnostic remains
closed; do not recreate it merely because formalcorrectness is not yetclosed.

Resource observation:3scale-ups/0downs;1335lowqueue+lowactive samples blocked
byTTFTlower. The observed lowqueue/lowactive/no-pending maximum now61.728443s,
six spans>=38scooldown, versusD29716.882191s. This supports re-examining
paper-consistent scale-in eligibility; it doesNOTpredict counterfactualGPU-s.
SeeD299_CAPACITY_FULL_W0.md for complete tradeoffs, provenance and small
post-run inspection errors (no performance changes/retries). Both bounded
analysis jobs completed; their empty owned scopes were stopped.

NEXT verify saved checkpoint/upstream for checked data/scripts/docs; safely recover disk to unchanged
150GiBstartmargin (04:46free160457490432B, below150GiB); then inspect IEEE/control
history plus primary sources for ONE observable resource-control hypothesis.
Retain D299 as latency-improved development evidence, not finalacceptedconfig.
No freshGPUlaunch,3B/baselines,thresholdrelaxation,warm-repeat or figures yet.

## HISTORY — D299 original4000W0 RUNNING; D298 calibration/smoke complete

2026-10-07 03:28+08. Checkpoint1fa3f708ee434dccd58509bea56ba097f2a669bd
pushed/remotelyverified BEFORElaunch. Config803693f4cccbdf4a5d4719b6d77f3b611cb380fe5e4342583f7d19e7be2ff10d
diffchecked: same D298model/control/profiles; only100→4000/scenario/output/cache.
Started03:27:50 in tmux primelora-d299-7b. Raw d299_20261007/7b_full_w0.
Replayreceipt4000/4000source/W0/rate1/no prefix, viewSHA
a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
Service primelora-tc-svc-a9a8e9715d484ac5939ffd02427acf48.scope invocation
323ed5b94d1644dcbcc5a9d63f4e46f8;watchdog2438146 in aux
primelora-tc-aux-d98edc92a9544a3abf13ef64ba32d1ce.scope. Sample29:
hostavailable108712308736B,peak4366774272B,high/max/OOM0/no warning.
Startdisk161679175680B exceeds unchanged150GiB gate;100GiB stopunchanged.

Remote monitor primelora-artifact-monitor-d299.service invocation
c9540d87c0494ee480efa4c92eb36598;7B13c92b11397a4906a9a5b216b1f3b867;
3B1a41e205be4641bbba2f950314d944da (artifactonly). NIC1000/full,D80units/cache
unchanged. Firstdirecthealth queried before listen and refused; noGPUlaunch
occurred. Sameinvocations thenlistening; authenticated health2 succeededat
822117.95993913 BEFOREdeployment, remoteclock5998fee5b2904ea78256f4aaaff2d3a2.
No restart, serving fallback or altered measured delay. Failed emptyhealthfile
retained; successful receipt raw/remote_health2.json. Remote monitor log
/home/lab14/primelora_remote/tc/d299_20261007/remote_monitor.log.

NEXT monitor SAME4000W0job throughterminal/actualrelease, stop only captured
remote invocations, locate actualjournal byclock/transferIDs (filenameUUIDdiffers),
collect/AUDITviaexistingD297Full+native/provisionalhelpers WITHOUTfigures. Keep
D298prefixresource regression visible; comparecompleteD299toD297/D219onlyafter
all4000. No newoptimizer/profiling/hashscan/compression/cleanupwhilelive. No3B,
baseline or warm-repeat. N_correct/commonwarm/Resident/G1G2 remainOPEN.

03:34+08 verified wait on SAME D299: service active/invocation unchanged,
tmuxconfirmedlive;196arrived/189normalcompleted/0reportedfail. Watchdog377:
hostavailable89.938GiB,peak17.365GiB,no safety warning, disk149.644GiB.
Run passed150GiB startgate; below150now prohibits nextheavylaunch, not current
validmeasurement (100GiB stopgate unchanged). Remote monitor exactinvocation
confirmedactive03:31, no service/config/cache operations. Full source plan reread;
no new candidate, code/tests/analysis/figures/GPU launch during this wait.
Keep monitoring same job through all4000 and actual release; do not restart
on an observation timeout. D298 data and launchcheckpoint remainbacked a77ffcb.

03:40+08 verified wait on SAME D299 (no new launch): 535 arrived,
534 normally completed, 0 reported failures; service invocation remains
323ed5b94d1644dcbcc5a9d63f4e46f8 and active. Watchdog sample729:
host available89.946GiB, service peak17.365GiB, high/max/OOM0, no warning;
disk149.485GiB, above100GiB running stop threshold. Remote monitor invocation
c9540d87c0494ee480efa4c92eb36598 confirmed active03:39 by read-only SSH.
Latest user reply explicitly permits the one-time read-only delivery cache;
this reconfirms the already implemented D78/D80 published-cache boundary.
Reuse it unchanged; do not regenerate archives or duplicate weights/pools.
Source plan fully reread and SHA0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7
unchanged. No code/config changes, extra performance job, figures, analysis,
compression, cleanup or remote service changes during this observation.
Continue through actual completion/release, then data-only full-run audit;
7B goals still open, followed by3B, baselines and the retained full matrix.

03:44+08 continuation classified as verified wait: SAME service invocation
confirmed active/running; tmux pane also live03:43. Latest856arrived/850normal
completed/0reportedfail. Watchdog1003: available89.546GiB, peak17.365GiB,
high/max/OOM/swap0, no warning; disk149.198GiB. No new job or intervention.
Frozen V1 reread in full and existing D297 data-only audit inspected (read-only)
for the eventual post-release D299 audit. No result analysis or performance
comparison executed on incomplete data. Continue same4000W0; do not restart.

03:50+08 verified wait, SAME active/running local invocation;1176arrived/
1174normalcompleted/0reportedfail. Sample1311: hostavailable89.649GiB,
peak17.365GiB, high/max/OOM0/no warning, disk148.901GiB. Exact remote monitor
invocation confirmedactive03:49. No run/config/cache changes or extra work;
original trace continues, and all4000 plus physical release remain required.

04:05+08 SAME D299 still running after continuous low-overhead observation:
2426arrived/2420normalcompleted/0reportedfail. Local service and remote monitor
invocations revalidated unchanged/active. Sample2241: hostavailable89.177GiB,
servicepeak17.982GiB, high/max/OOM0/no warning, disk148.315GiB. No intervention,
configuration changes, additional GPU/CPU analysis, packing or cleanup.
This remains a verified wait; continue original4000W0 through physical release.

## HISTORY — D298 Full100 complete; D299 original4000W0 prepared

2026-10-07 03:26+08. D298smoke100/100native-contract completed,0reportedfail,
fourGPUcontextsreleased,launch/watchdog0/0,ownedremoteunitsinactive/PID0/success.
Data-only audit reuses D290 comparison/binding and D171 timing helper, saved
paper_results/ieee_tc/p2_backend/d298_capacity/smoke1/. Inputmatch/controlbinding
100/100forD296/D298;9outputhashdifferencesretained,NOTnumericalqualification.
D298vsD296 meanTTFT1.140273vs2.129072s,P95TTFT2.339925vs10.771356s;
meanTPOT29.797313vs34.997670ms,P95TPOT39.253765vs44.405718ms.
PhysicalGPU-s780.572493vs727.912111(+7.23%): NOTresourceimprovement.
ProvisionalD170timingupper97/100, noformalSLO/N_correct/G1G2acceptance.
29remoteUUIDjoins50650865wirebytes/packing0;266watchdogsamples,
peak19154620416B,hostminimum93601243136B,high/max/OOM/swap0.
Remotejournal774ad88...SHA7102ce906f913e8fbeeb869267ea2ed2f13ba312683027f4ad389218cecbed82;
monitorSHA5278820e3fe83e453446945d19dd3f23cdefc1d7ce9a5276d2f4c94cb2fde8a7,
bothcopiedandremote/localSHAequal. Protection146/147sameknownbaselineedit.

D299 is the predeclared full4000W0 of SAME D298 configuration, not another
optimizer. Config/launcher branch prepared, fulltracecountrestored/no prefix;
allmodel/control/profile/publicenvelopeunchanged. See D299_CAPACITY_FULL_W0.md.
NEXT verifyconfigdiff/checkpoint/push, then one guarded4000W0. ComparetoD297Full,
notD296prefix; no figures/warm-repeat/3B/baselines. Finalsharedreferences,
numericalcorrectness/resourcegoalsremainOPEN. Noexperimentcurrentlylive.

## HISTORY — D298 Full100 smoke RUNNING; source calibration complete

2026-10-07 03:19+08. Checkpoint9cd6a4aa323d7484b5b08f29e97018f74f3de0aa
pushed/remotely verified, then SAME prepared candidate launched once in tmux
primelora-d298-smoke. Raw d298_20261007/7b_capacity_validation_100.
Service primelora-tc-svc-9c2f957cd5174d60a6d81d411517ba66.scope invocation
549590abfe3842aeb4ca60ede5e00855; watchdog2397137 in aux
primelora-tc-aux-d1e6a90a22d74e75a6c10fa7199ced67.scope. Sample21: hostavailable
111105257472B, servicepeak1939734528B, high/max/OOM0/no warning. Stillpreparing.
Inference startdisk161739771904B meets unchanged150GiB threshold.
Remote monitor primelora-artifact-monitor-d298-smoke.service invocation
85dc6cf708d94a0fb2abd7807333d3d8;7Bf3fb04d7a85e455e85f7015edce90bde;
3Bed01e4e1cba5400fa07f7b28764412c6 (artifactonly). RemoteNIC1000/full.
Authenticated direct health at821595.054949011 before launch uses existing
HttpArtifactStoreClient, clock00210148961142b18fb410298f6c7c16. First manual
urllib health used environment proxy and returned502 BEFORE inference; empty
smoke_remote_health.json retained; direct receipt is smoke_remote_health2.json.
Serving client already uses direct access; no serving change or rerun occurred.
Remote monitor log/home/lab14/primelora_remote/tc/d298_smoke_20261007/remote_monitor.log.
NEXT monitor THIS smoke to actual resource release, stop exact captured remote
instances, collect by actual journal filename and clock/UUID matches, audit
native/binding/timing/resource/remote data WITHOUT figures, then original4000W0
if qualification permits. No warm repeat, fresh optimizer,3B or baseline whilelive.

## D298 source calibration complete; Full smoke preparation history

2026-10-07 03:17+08. SAME source job completed248/248 native-contract calls,
240representative+8warmup, with48measured calls per source. Existing validators
and profile loaders cover40service/24preparation classes, no extrapolated fills.
Curated d298_capacity/initialization/manifest.json binds new timings to exact
capacity8/slots8/rank16/.92 model; D170 completed-length evidence reused.
Source SHA f1d9bd576fef9da92ae4e566b75df1292c7ab51f320eb53f7fa4483f4e25d474.
All248remote transfers match UUID/content/archive/bytes,1059355374wire bytes,
no request-time packing. Controlled source reset downloads are NOT Full misses.
One physical GPU held476.461610s, actual context released, scope removed;
484resource samples,peak5456633856B,hostminimum107680460800B,high/max/OOM/swap0.
Two of8repeated output groups vary; retained, not numerical correctness proof.

Exact-owned remote monitor/7B/3B invocations stopped, allinactive/PID0/success.
Journal filename has its own UUID (de4093...), not the health clock UUID;
initial mistaken filename lookup failed after successful service stop. Actual
file verified by clock/transfer joins and remote/local identical SHA. No rerun.
Data-only export reused existing D157/D89 validators; no serving code changes.
Protection146/147unchanged, sole previously known baseline script SHA preserved.
No figures or warm-repeat.11profile-domain/export tests pass; actual Full child
assembly exactly matches the measured contract, old workload/controller equal.
Smoke config SHA6dc8bf632d07a0b87191a659f024b639a39a59648380ff55613b3a8ca5da4ebd.
NEXT checkpoint/push, then one guarded100-request Full smoke with fresh owned
caches and the same published artifacts. Only after complete validation run
4000W0. G1/G2, numerical correctness and shared final references remainOPEN;
D297 is still latest completed Full. No3B/baselines until7B target work closes.

## HISTORY — D298 source measurement RUNNING; D297 remains completed Full

2026-10-07 03:05+08. Checkpoint783fa42e6b3ba1cd0e034d8e0cddcd3d72c9b42e
pushed and remote SHA verified BEFORE launch. The one source-calibration job
started03:03:59 in tmux primelora-d298-source. Raw/source1_launch.launch contains
guarded execution and resource evidence. Actual service unit
primelora-tc-svc-30e40ed8dc7e43cfaa7c593a0c94ad41.scope,
invocationb34a7b85dc6549fb96b21d68ec9f461c; watchdog2330044 in auxiliary
primelora-tc-aux-6f07b2a115444c259f1ad833c3e94380.scope. Sample73:
servicepeak5456633856B,hostavailable109734871040B,high/max/OOM0,no warning.
Still initializing the native runtime; no completed source measurement yet.
No concurrent new GPU run, compilation/install campaign, cleanup or figures.

Remote units use the unchanged D80 files/cache/monitor; actual NIC1000/full.
Monitor primelora-artifact-monitor-d298.service invocation
7f8ab050a35c464d87a489395aa2ceaf;7B8decefe0633e40fbb4054a11433e839a;
3B58857eeff82f447f822b839507f9b656 (artifact service only, no3B inference).
Authenticated direct health at820672.26981241 preceded local launch.
Remote7B clockremote-process-monotonic:0c7bd6b595fb40a98648d03e568050b5.
Start/health receipts retained under d298_20261007; immutable cache reused.
NEXT monitor SAME job, release actual owned GPU, stop ONLY captured remote
monitor/invocations, reconcile transfers and audit/export source profiles via
existing helpers. Then100-request Full smoke and4000W0, not warm-repeat/3B/
baselines. No fresh candidate while this run is live. G1/G2 remainOPEN.

03:06+08 SAME job now measuring:24/248native source samples logged, including
the8warmup calls; no final source-profile acceptance. Watchdog131, peak still
5456633856B, available109195345920B, high/max/OOM0/no warning. GPU0owned only;
other3cards idle. Live-status checkpoint9b5b7a2 pushed. Continue this run.

## D298 preparation history

2026-10-07 02:56+08. D170 raw JSON losslessly archived; two roundtrip SHA checks
preceded removal of the uncompressed duplicate. Curated/projection unchanged;
receipt in paper_results/ieee_tc/storage. Free disk161841979392B (~150.73GiB),
unchanged150/100GiB thresholds. No live GPU/remote experiment or serving change.

Recovered already-qualified D169/D170 capacity8/rank16/fraction0.92 evidence.
Do not repeat warm/capacity measurement. D298 preregisters one capacity-aligned
Full candidate, keeping paper/controller/metric contracts unchanged; actual
service/preparation profiles must be refreshed, not copied from D157cap4.
Next: strict offline native completed-length reuse + existing source collector
preparation, then bounded qualification/Full. No winner claim, figures,3B or
external baselines. The final reference/numerical/G1G2 gaps remain open.

03:02+08 preparation completed: D170 projection/index/model/native-count checks
reuse512unique completed requests without extra GPU calls. New means100.8641/
125.5689, no old timings relabelled. Source contract248calls=240representative+
8warmup, eight existing distinct adapters,31waves; config/actual child assembly
and static500maxrank16 verified. Existing source collector unchanged. Offline
initializer accepts explicit warm evidence in addition to existing source data.
20targeted tests pass; combined209/213 pass with4wrong-interpreter OS errors;
actual system Python OS suite50/50passes. Failed attempts retained. Checkpoint
then one guarded real-remote source measurement, not a new warm/Full result yet.

## HISTORY — D297 complete 7B W0 audited; targets still OPEN

2026-10-07 02:36+08. Previous turns were verified waits on one job, not failed
launches. D297 finished4000/4000 native-contract requests, no reported failures;
all4physical leases closed and GPU compute processes cleared. Matched remote
monitor stopped only its captured invocations; both artifact units inactive/PID0.
132/132remote transfers join by UUID/content/archive/byte contract,131480060wire
bytes,0request-time packing. Immutable delivery cache was reused, not regenerated.

Data-only audit reuses existing native timing/binding/provisional-SLO helpers;
new narrow driver `scripts/ieee_tc/analyze_d297_full.py` does not change serving.
Curated `paper_results/ieee_tc/p2_backend/d297_full_w0/validated1/`.
Mean/P95TTFT1.527145/3.359722s; mean/P95TPOT37.883698/62.065918ms;
physical15860.686503GPU-s,3.965172GPU-s/offered (NOT per-correct).
Provisional D170 timing upper bound3786/4000=94.65%, below95%; N_correct=null,
formalSLO/budget/G1G2 remainunqualified. D219 same input/plan diagnostic gives
3759/4000=93.975%,P95TTFT3.619799s,15911.712503GPU-s. AverageTTFT/TPOT regress;
126output hashes differ. Single observations/different serving versions: noCI,
single-change attribution, numerical pass or overall victory. D295 remainsclosed.

Resource hypothesis REFINED: all1861control samples lack all-low; all1859known
P95values exceed281.278ms(min1525.370ms).308lowqueue/lowactive samples are gated
by TTFT. BUT removing that predicate in the unchanged observed path still yields
at most16.882s consecutive lowqueue/lowactive/no-pending, below38s cooldown.
Thus changing TTFTlower alone is NOT supported as a solution; do not retune it
in isolation or claim counterfactual savings. Further7B optimization must use
both latency and lifecycle evidence and new falsifiable, paper-consistent logic.

4037watchdog samples: servicepeak19.031GiB,hostmin87.140GiB,high/max/OOM/swap0.
Analysis used3/4GiB scope,peak750845952bytes,no high/OOM. First analysis rejected
integral-float prompt-count serialization; explicit finite/integral conversion
then passed, original rejection retained.8timing-helper tests pass. Protection
146/147unchanged; only preexisting baseline replay script mismatch, untouched.
No figures, new GPUrun,3B/baseline run, new configuration or cache generation.
Free disk160137027584bytes(~149.14GiB): before another heavy launch recover space
using audited ended-work caches/lossless compression; preserve unique evidence.
NEXT finish checkpoint/provenance, safe disk recovery, then evidence-led7B
candidate selection. Do not reopen completed D170 measurement or boundedD295.

## HISTORY — D297 complete 7B W0 RUNNING; historical snapshots below

2026-10-07 01:20+08 live verification. The pre-registered replay was launched
at01:16 in tmux `primelora-d297-7b`, after checkpointa40df0d was pushed and remote
head verified. Existing immutable delivery cache is reused (latest user approval
reconfirmed); no new packaging, weights, trace or expanded artifact pool.
Direct authenticated remote health completed atmonotonic814229.57178896,
BEFORE deployment notice814247.332460961; first arrival origin814307.332460961.
Execution receipt confirms W0,4000/4000 source requests,rate_scale1.0,no prefix.
At the01:20 snapshot79arrived/74normal completions/0reported failures; this is
partial operational progress, NOT numerical correctness or final performance.
Watchdog sample220: servicepeak18.240692GiB, hostavailable90.266891GiB,
high/max/OOM/swap0, no safety warning; free disk149.509441GiB. Launch passed
150GiB start gate; now below it prohibits another heavy launch, not this valid
run (100GiB stop gate unchanged). Do not compress/clean/hash-scan during replay.

Service `primelora-tc-svc-72524b343c5c45a699736635a24184bd.scope`,
invocation `3d9973309a41446baa8df1e8ac763a91`; watchdog PID1851665 in independent
aux scope `primelora-tc-aux-7f9be371ba2644b5a885484a29b73f98.scope`.
Remote monitor `primelora-artifact-monitor-d297.service`, invocation
`bec592d608814aa7a9a4512c7d364bff`;7B/3B units have matching captured identities
in remote_start.log (3B artifact unit idle; no3B inference). Actual NIC1000/full.
Full launch/health/provenance details in D297 document and unique raw directory.
NEXT monitor this run to terminal and actual GPU release, then stop ONLY its
remote monitor/invocations, collect immutable transfer/monitor evidence, audit
and save data without figures. Do not start warm/Resident/3B/baselines alongside.
N_correct, commonwarm/Resident and G1/G2 remainOPEN. No completed4k result yet.

01:28+08 verified wait on SAME tmux/service invocation:518arrived,516normal
completions,0reported failures. Watchdog sample696, servicepeak18.240692GiB,
hostavailable89.911167GiB, high/max/OOM0, no safety warning. The matching remote
monitor invocation was independently verified active; no remote adjustment.
Read-only preparation for final analysis recovered the COMPLETE D170 warm
measurement (1536+16 calls); do NOT rerun it merely because the final common
reference remains unfrozen. Reuse its sealed candidates and the D171
`provisional_timing_bound` helper for clearly provisional timing diagnostics.
D219's physical total is15911.7125GPU-s per D240, NOT legacy4317.05; its old
sidecar is `physical_deployment/summary.json`. The short-run comparison helper
hardcodes a100-request limitation and expects the newer sidecar: don't apply
it unchanged to a full/legacy comparison or silently weaken its checks.
Final analysis must wait for actual GPU release and run in the bounded auxiliary
domain. No figures/new code/config/tests/heavy analysis while this run is live.

01:33+08 SAME live run:846normal completions/0fail,3scale-ups/0scale-downs so far.
Read-only D88/D156/controller audit finds old cap2-derived soft thresholds are
retained with cap4, and the281.278ms lower threshold came from a shorter native
measurement than the actual overall-TTFT control signal. This is a falsifiable
post-run resource-reclamation question, NOT a proven bug or permission to retune
mid-run. Existing control event fields suffice; evaluate all low predicates,
continuous intervals/pending resets/retirement outcomes after completion.
Current official Knative/HPA sources and explicit non-causal boundaries are in
D297 doc. No next candidate selected, no source/config modification.

01:36+08 quarter-run verified wait:1023arrived/1016normal completions/0fail,
3scale-ups/0scale-downs, same active service invocation. Resource sample1143:
hostavailable89.508228GiB, servicepeak18.240692GiB, high/max/OOM/swap0, no warning,
disk148.657516GiB. Keep this run live; no second launch/cleanup/analysis now.
Two documentation-only changes preserve the above reference reuse and hypothesis;
serving code/config untouched. Final analysis and resource recovery remain after
terminal/actual release, not after a quarter of the requests.

01:50+08 half-run verified wait (no implementation work this turn): SAME tmux
and active service invocation;2019arrived/2010normal completions/0fail. Live
counts still3scale-ups/0scale-downs/4runtimes. Sample1957: available89.630GiB,
peak18.240692GiB, high/max/OOM0/no safety warning; disk148.149006GiB. Full
result/release not yet available. Plan SHA reconfirmed unchanged at turn start.
No new tests/analysis/remote changes/figures/GPU launch; continue this exact job.
The next run remains prohibited until the existing150GiB start gate is met.

02:04+08 verified wait continues on SAME D297 (not a new experiment):3059arrived,
3056normal completions/0fail at02:03:33. Sample2776: available88.982GiB,
high/max/OOM0/no safety warning. Exact remote monitor invocation reconfirmed
active at02:01; no remote configuration/service/cache operations. Source plan
SHA unchanged. Continue through remaining requests and actual physical release;
no final result, correctness/SLO/G1G2 qualification or new candidate yet.

02:13+08 verified wait on SAME run:3553arrived/3550normal completions/0fail,
4runtimes, no duplicate launch. Watchdog sample3342: hostavailable88.988GiB,
high/max/OOM0 and no warning. Remote paired services/monitor retain the exact
registered invocations (rechecked02:10); no packaging/configuration/restarts.
Full source plan and V1 reread after compaction; native audit and provisional
timing helpers inspected read-only for post-release reuse. No serving edits,
heavy analysis, cleanup or figures. Final result and physical release pending;
continue monitoring this job, then collect/audit before selecting another task.

## HISTORY — D297 complete 7B W0 prepared; candidate unchanged

2026-10-07. Previous goal turn was PROGRESS: completed D296, saved its resource/
latency trade-off and pushed300c789. Current checks reconfirm stopped GPU/remote
state and preserve all history. D297 restores all4000 original W0 requests using
the D296 model/control candidate; only count/scenario label/output/cache paths
differ. Existing launcher extended with explicitd297 branch and no prefix.
No new serving optimization, numerical repeat,3B/baseline or figures.

Important mechanism check: legacy effective_capacity_admission metadata=false
is not the current native proactive E(t) gate. D296 has101GPU plans with
capacity_only=false,144attempts (115completed/29deferred),109explicit proactive
receipts. Current native policy derives capacity_only from coordination_enabled.
Do not reapply D226's historical legacy-path inference or activate the optional
demand capacity witness just to alter metadata. See D297 doc/source evidence.

Configecf75d4d... structurally checked; launcher syntax and94system-Python
open-loop/guard tests pass. Same servingSHA cc821e08..., prior tests reusable.
NEXT backup/check protection, start unchanged remote+monitor, direct authenticated
health BEFORE local launch, fresh disk/memory gate, then one fullW0 development
replay. Formal numerical/SLO/Resident/G1G2 remainopen. No new GPU result yet.

## HISTORY — D296 short replay complete; cleanup dependency reduced, G1/G2 still open

2026-10-07. D296 actual remote100/100 completes with native fixed counts and
all saved dispatch/adapter/slot bindings verified. Physical measurement is complete;
service/watchdog/replay exit0, GPU contexts clear and service domain removed.
Terminal cleanup85.545934→36.264719GPU-s; four independent cards return within
8.479–9.568s after terminal instead of9.586–33.336s. Total769.307664→727.912111
GPU-s (-5.38% single-run observation). Mean TTFT2.170015→2.129072s, but P95
10.538035→10.771356s and mean/P95TPOT worsen slightly. No latency win/causal
overall speedup/qualification claim. Earlier D290 total714.256679 is still lower.
100-prefix is not4000 qualification. Five output hashes differ (59/74/77/84/91),
saved bindings all pass; N_correct remainsnull and no extra numerical repeats.

267 resource samples; servicepeak17.710510GiB, hostmin87.808937GiB; all high/max/
OOM/swap0. Remote29transfers/50,650,865wirebytes, prepublished only/no packing;
actual1000Mbps/full. Paired monitor and both artifact units stopped by invocation,
inactive/PID0/success. No3B inference. Initial health used ambient proxy/502;
direct health succeeded during local startup, deviation retained. No service
reconfiguration. Old protected results146/147 before/after; same baseline script.

Candidate implementationccc0fd8 pushed before replay. Data/interpretation:
`paper_results/ieee_tc/p2_backend/d296_lifecycle_validation_100/` and D296 doc.
No figures. NEXT retain this candidate for complete7B development replay and
commonwarm/Resident/oldPrime new-metric evidence; do not repeat100 solely to win,
reopen bounded D295, or move to3B/baselines. Full numerical qualification, shared
SLO/budget and G1/G2 all remainOPEN. All inference/remote services currently stopped.

## HISTORY — D295 all three starts complete; return to7B lifecycle/SLO optimization

2026-10-07. Three registered blocks complete162 native generations, each54/54,
with slot content and independent execution-journal validation and actual cleanup.
Within-block single/concurrent exact counts25/27,25/27,23/27. Across starts,
single exact22/27,24/27,25/27; concurrent24/27,26/27,23/27 (pairs1–2,1–3,2–3).
All first differences involve a tied side and at most.03125 top2 logprob margin.
The single-stability/batch-only explanation is contradicted, NOT established.
No token tolerance added, no mismatch relabelled exact, no extra repeats planned.
Raw archives/receipts and the full audit are retained; final compact data in
`paper_results/ieee_tc/p2_backend/d295_numerical_repeat/three_blocks/`. No figures.
Block3 service/GPU release confirmed;789 samples, peak5.298267GiB, host min
100.030857GiB, high/max/OOM/swap0. Old seal146/147; same pre-existing baseline
script is the sole difference. No historical results overwritten.

NEXT: D296 lifecycle hypothesis (doc adjacent). Historical D293 terminal cleanup
85.546GPU-s, independent card returns9.586/17.403/25.221/33.336s after terminal.
Four CPU counterexamples reproduce serial dependence; candidate now groups
shared-engine aliases and joins independent runtime returns concurrently while
retaining physical-release checks. Focused34tests pass. Broader model-Python test
was stopped on persistent auxiliary high reclaim (no max/OOM); old CPU environment
passes634 tests, with one unrelated pytest import error. The analyzer's correct
pytest entry separately passes22; system-Python guards75pass. Failed/incomplete
logs remain. Candidate/config are ready for backup and fresh resource gate.
Checkpointccc0fd8 pushed and remote head verified. D296 bounded100-request replay
started in tmux `primelora-d296-7b` with unchanged D293 settings and fresh output/
cache paths. No result yet. Source protection146/147, same known baseline file only.
Remote units/monitor started; actual link1000/full; idle3B artifact unit exists only
to reuse the paired monitor, no3B inference. First standalone health probe used
ambient proxy and returned502; direct health then passed during local startup.
This ordering deviation is saved, not represented as prior-to-launch qualification.
The existing real fetcher already disables ambient proxies. No remote configuration
changed during the run. Next: complete/release, no-figure audit and compare measured
terminal GPU-s before acceptance/full qualification.
Numerical/Full N_correct, commonwarm/Resident, oldPrime-newmetrics and G1/G2
remain open. Do not revisit D295 or move to3B/baselines/figures now.

## HISTORY — D295 block2 complete; single-request stability hypothesis contradicted

2026-10-07. Block2/attempt1 completes54/54 diagnostic generations and cleanup.
Within block2 single/concurrent exact25/27 (differences7/59). Across blocks1/2,
single exact22/27 (differences9/13/59/77/84), concurrent exact24/27 (9/13/77).
Every observed first difference has a tied side and the other top2 margin at
most.03125. Full native execution/slot revalidation passes. Thus "single stable,
only concurrent changes" is NOT supported; do not repeat that provisional claim.
This is numerical sensitivity evidence, not proven kernel causality or N_correct.
Complete the registered third start without changing configuration or tolerance,
then end this bounded series and return to7B SLO/lifecycle optimization.

Block2 raw032a8043... losslessly compressed to11,194,069bytes, gzipaea19bde...;
only verified uncompressed duplicate removed. Receipt beside archive. Audit
`paper_results/ieee_tc/p2_backend/d295_numerical_repeat/blocks_1_2/` validates
both journals in23.16s, maxRSS1,733,608KiB.790 watchdog samples; service peak
5.298527GiB, host minavailable100.046379GiB, high/max/OOM/swap0; actual GPU
release/service removal confirmed. No figures. NEXT launcher block3/attempt1
after fresh gate; all serving/config SHA unchanged from block1. No3B/baselines.

## HISTORY — D295 block1/attempt2 complete;25/27 single/concurrent exact,2 near-tie divergences

2026-10-06. Completed54 native fixed-length generations (27 per arm), with actual
single batches1 and concurrent batches1/2/3/4; slot readback and independent
forward-journal revalidation pass.25 paired sequences exact;req7 first differs
at token80 (logprob margins0/.03125),req84 at token38 (.015625/0). This supports
batch/execution-condition numerical sensitivity but does NOT explain all historical
six differences, prove kernel arithmetic, or qualify N_correct/G1G2. One of three
independent native starts done; no extra seeds or tolerance relaxation.

GPU released/service domain removed;840 watchdog samples, service peak5.286GiB,
min available100.415GiB, high/max/OOM/swap0. New346-MiB detailed JSON compressed
losslessly to10.67MiB; raw SHA60949fa4... matches decompression, gzip83dcac90... .
Only the verified uncompressed duplicate was removed, receipt retained. Read
`results/ieee_tc/p2_backend_qualification/d295_20261006/block_1_attempt_2/numerical_repeat.json.gz`
directly; original raw name no longer exists. Do not rerun block1 as missing data.
Curated data/status and source hashes in
`paper_results/ieee_tc/p2_backend/d295_numerical_repeat/block_1/`; no figure per
latest user instruction.31 analyzer/capture CPU tests pass; actual audit16.97s,
maxRSS1.306GiB, high/max/OOM0. Full explanation in D295 doc.

NEXT: fresh resource gate, then existing launcher `...run_d295_7b_numerical_repeat.sh 2 1`;
inspect/record/cleanup/compress before block3. Reuse same27 inputs/config/code;
do not expand numerical diagnostics or repeat HF/PEFT. Then return to7B performance
optimization. D293 cleanup85.546GPU-s and serial independent-runtime release are
one untested resource hypothesis, not a subtraction from resource measurement.
Commonwarm/Resident, oldPrime-newmetrics, per-model G1/G2 still OPEN. No3B/baselines.

## HISTORY — D295 block1 launch refused before GPU; unique retry next

2026-10-06. D294's missing measurement now has an opt-in extension to the existing
backend-model-check: same 27 original request indexes (0–21 plus59/74/77/84/91),
single-request and concurrent4 arms, three separately launched native runtimes.
Actual token IDs/top2 probabilities, slot content and forward metadata retained;
no Full performance point, new input, kernel/precision/batch-invariance change,
or remote service. See `D295_CONTROLLED_NUMERICAL_REPEAT.md` and new config/launcher.
468 model/basic tests pass;75 guard tests pass in the required system Python.
An initial combined model-Python test invocation had4 guard API failures and539
passes; those failures remain documented, not suppressed. Old seal146/147 with
only the same pre-existing baseline replay script difference. Fresh disk/headroom
gate passes narrowly; recheck before launch. No new GPU results yet. Inspect each
block and publish status table before next; no unchecked three-block loop.
7B acceptance, commonwarm/Resident and G1/G2 still OPEN; no3B/baselines.
Block1 first launch was refused because the new launcher used a readable name
instead of the existing required UUID auxiliary scope. Zero model/request work.
Original `block_1/` and source4bba237 preserved; only launcher ownership naming
corrected, explicit fresh attempt suffix added. Next block1/attempt2, same inputs.

## CURRENT — D294 saved binding audit passes; numerical cause remains unresolved

2026-10-06. Reused D290/D293 immutable logs and frozen content index only;
no GPU, remote service, model, trace, or serving change. All100 rows per run
have consistent adapter/owner/lease/native integer ID/slot/content/clock/
release/token-digest witnesses. All six output-difference rows have identical
native prompt token-ID hashes. This is control-witness consistency, NOT
actual kernel arithmetic proof or `n_correct=100`. Actual batch membership,
full sequences and native logits are missing in these runs; do not infer the
first differing token from hashes or declare batching causality established.

Audit/doc: `docs/ieee_tc/D294_SAME_BACKEND_BINDING_AUDIT.md` and
`paper_results/ieee_tc/p2_backend/d294_output_binding_audit/`. Existing
diagnostic analyzer extended;20 CPU tests pass. D293 backed at24519cf and
CSV-format-only correction3af37b4; old source hashes/plots remain unchanged.

NEXT: exact pre-registration/CPU qualification of the missing controlled
same-backend repeat using existing native token-capture path and existing
request indexes. D246 single native run and D247 cross-backend19/22 are NOT
three same-backend controlled repeats; do not redo the cross-backend study.
Then fresh safety gate and only the minimal missing measurement. No blind
Full replay, batch-invariance/precision changes, relaxed correctness, new
weights/traces, or progression to3B/baselines. Numerical/commonwarm/Resident/
oldPrime-newmetrics/G1G2 remain OPEN. Last GPU state idle, remote7B stopped.

## CURRENT — D293 bounded prefetch run complete; latency/resource trade-off, not acceptance

2026-10-06. D293 completed 100/100 native fixed-output requests with full
physical allocation/release evidence. Frozen V1 Type-1 recomputation gives
mean/P95 user TTFT 2.170015/10.538035 s versus D290 13.558675/39.448237 s.
Initial readiness fell 103.504→50.496 s, but physical GPU-s **increased**
714.256679→769.307664 and P95 service TTFT worsened 1760.353→1933.152 ms.
This is a single-prefix diagnostic, not a causal prefetch effect or G1/G2 pass.
File-cache warmness is not controlled after D291/D292 startup attempts.
No common SLO/budget is frozen; both sidecars keep `n_correct=null`.

D291 failed before requests on profile identity; D292 loaded weights but
failed before requests on parent/child derived-loader receipt equality.
Their artifacts remain failures, not denominator entries. Existing commits
175b37a/ec7fdb0 separate deployment loader settings/observations from
post-ready service identity and configuration intent; D293 used 632bd7d.
All inputs/target/native counts match D290, but six output hashes differ:
req_00005/00059/00074/00077/00084/00091. Do not assume numerical correctness
from success/counts or assume an adapter bug from hash inequality.

Offline analyzer repaired double-counted arrival lateness in queue, uses
Type-1 rather than legacy interpolated percentiles, rejects incomplete data,
and preserves old figures. Audit/table/plots:
`docs/ieee_tc/D293_PREFETCH_VALIDATION_AUDIT.md`,
`paper_results/ieee_tc/p2_backend/d293_prefetch_audit/`,
`figs/ieee_tc/p2_backend/d293_prefetch_audit/`.
Twelve focused offline tests pass; TNR regular/bold embedded, 3.45-inch page.
The initial dependency/font failures are analysis setup failures, not replays.

Owned GPUs released and remote 7B artifact service stopped. Remote delivery
cache authorization already fulfilled by D78/D80; no recreation. Both NICs
currently report 1000/full; historical 100-Mbps cause remains unproven.
Service peak18.048GiB/events-swap0; minimum host available87.301GiB.
Fresh disk gate required before another heavy task; never lower 150/100GiB.
Old seal146/147 unchanged: sole mismatch is pre-existing dirty baseline
replay_openai_trace.py, untouched here; protected historical results unchanged.

NEXT: finish six-row same-backend identity/batching audit using D225–D249
evidence and actual source, before minimal missing-data collection or full
7B qualification. Keep numerical/G1/G2 gaps open; 7B before3B beforebaselines.
Do not replay D291/D292, regenerate adapters/traces/cache, announce resource
saving, or use D290's slower point as proof of exceeding old Prime.

## CURRENT — D288 Resident reference paused by the revised Prime-first order

2026-10-06. D288 launched the first full 7B Resident reference attempt under
the repaired CPU-affinity and resource contract, but it was deliberately
stopped before replay completion after the execution order was revised to
finish/analyze the existing ServerlessLLM run and then complete PrimeLoRA
first.  The attempt has no replay JSON and no valid request denominator; it is
classified as `user_priority_stop` and excluded from every Resident reference,
budget, SLO, G1/G2, and baseline-ranking calculation.  Its raw directory and
receipt are preserved, and the lifecycle closed only after owned workers and
the final NVML census were clear.  The baseline-side audit is
`docs/ieee_tc/D288_RESIDENT_REFERENCE_PRIORITY_STOP.md`.

The D287 100-request CPU-affinity pilot remains a valid protocol qualification
only.  The Resident three-repeat series is registered but paused.  The next
main-line work is the frozen metric/remote-boundary decision note, followed by
PrimeLoRA 7B correctness and performance optimization; 3B and all external
baseline comparisons remain paused until the per-model Prime gate is met.
The resulting metric/remote accounting freeze is recorded in
`docs/ieee_tc/METRIC_REMOTE_BOUNDARY_HYDRA_FREEZE_20261006.md`.

## CURRENT — D287 7B Resident CPU-affinity pilot passes; formal reference series authorized

2026-10-06. After D285 exposed that a user-level slice's `AllowedCPUs` field
was not sufficient evidence without delegated cpuset, the baseline runner was
repaired at commit `1115b16`: each Resident vLLM and replay transient unit
now receives systemd `CPUAffinity`, and active receipts record the kernel
`Cpus_allowed_list` for each main PID.  The focused contract audit is in the
baseline repository at `docs/ieee_tc/D286_CPU_AFFINITY_CONTRACT_AUDIT.md`.

D287 is a fresh 100-request 7B seed-42 pilot using the unchanged W0 trace,
500-adapter subset, fixed-length native-token contract, and prepublished
remote delivery.  It completed 100/100 requests, matched all 100 target
native token counts, and used `vllm_token_ids` for every completion.  At both
active snapshots, all four vLLM PIDs reported configured `CPUAffinity=4-23
28-47` and kernel `Cpus_allowed_list=4-23,28-47`; the replay PID reported
`2-3,26-27`.  The 72/80-GiB service, 4-GiB auxiliary, swap, task, common
clock, final-NVML-empty, and remote-delivery checks also passed.  Its
760.11-ms mean TTFT, 875.83-ms P95 TTFT, and 2,254.12 GPU-s are descriptive
pilot values only.  The pilot table and provenance are in
`baseline/docs/ieee_tc/D287_RESIDENT_CPU_AFFINITY_PILOT.md`.

This closes the CPU-affinity qualification blocker but does not freeze a
budget.  The next admissible action is the pre-registered three-repeat,
4,000-request 7B Resident reference series for `U_ref`; 3B and all external
baselines remain paused until the 7B Prime numerical/common-warm/Resident
qualification gate is complete.

## CURRENT — D285 Resident 7B pilot passes the protocol gate; reference series still open

2026-10-06. A fresh 7B seed-42 pilot completed 100/100 requests with the
fixed-length native-token contract (100/100 target matches and no trace-token
fallback). The deployment notice was published before startup and the replay
used the same absolute monotonic origin. Four replicas were launched
concurrently and became ready in 72.66--73.44 seconds; the first request's
approximately 18.98-second dispatch wait is consistent with the 60-second
lead and measured startup, so startup was not removed from user TTFT.

Unlike D282/D283, D285 records active unit snapshots while the units still
exist: every service unit read back 72-GiB `MemoryHigh`, 80-GiB
`MemoryMax`, 2-GiB `MemorySwapMax`, and `TasksMax=4096`; the auxiliary replay
unit read back a separate 4-GiB/zero-swap/128-task envelope. Cgroup paths and
PIDs are retained, final NVML cleanup succeeded, and the remote immutable
delivery service was stopped. The audit is in
`docs/ieee_tc/D285_RESIDENT_PILOT_RESOURCE_SNAPSHOT.md`; the cumulative pilot
table is under
`paper_results/ieee_tc/p2_backend_qualification/20261006_d282_d283_resident_pilot_audit/`.

This closes only the 100-request Resident protocol pilot. Its descriptive
752.28-ms mean TTFT, 25.15-ms TPOT, 97% development-threshold attainment and
2254.870 GPU-s are not a formal budget reference. The next admissible action
is the pre-registered 7B full Resident reference series needed to freeze
`U_ref`/G2; 3B and external baselines remain paused.

## CURRENT — D282/D283 Resident pilots diagnose shared-clock and startup barriers; no reference budget frozen

2026-10-06. D282 completed 100/100 fixed-output 7B requests and passed the
request-level generation/cleanup checks, but its deployment notice was emitted
only after all replicas were ready and the replay used a process-start clock.
It is therefore a generation/cleanup diagnostic, not a Resident performance
point.  D283 moved the notice before startup and passed the absolute business
clock to replay; it then exposed serial replica initialization (approximately
66.53, 68.56, 66.77 and 68.58 seconds) because the launcher waited for each
replica before starting the next.  The pilot reached 20 successful requests
before being deliberately stopped after the startup-path diagnosis, so it has
no complete replay output and is also excluded from all performance
denominators.  Both pilots were cleaned up, the managed remote service was
stopped, and NVML was empty.

The audit and compact status table are in
`docs/ieee_tc/D282_D283_RESIDENT_CLOCK_STARTUP_AUDIT.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d282_d283_resident_pilot_audit/`.
The shared-clock repair is baseline commit `dd1ca72`; the parallel-launch and
72/80-GiB Resident default repair is `effb149`, both pushed to the baseline
repository.  The next admissible action is a new unique 7B pilot after the
parallel-start path passes CPU/dry-run checks.  No warm threshold, Resident
budget, G1/G2 point, 3B run, or external baseline comparison is frozen.

## CURRENT — D281 remote 7B delivery health gate passes; Resident pilot still held

2026-10-06. The existing owned 7B artifact unit was started through the
verified SSH alias and checked over the direct experiment NIC.  Authenticated
health returned `prepublished_gzip_v1` with `artifact_timing_v2`; the immutable
manifest contains 500 objects and its response hash is recorded in
`docs/ieee_tc/D281_REMOTE_7B_HEALTH_GATE.md`.  The unit readback shows the
expected 2-GiB high / 4-GiB max / 128-task auxiliary envelope.  No archive was
regenerated, no inference process was started, and no GPU result was produced.

This closes only the remote delivery prerequisite.  The next Resident action
is still gated by the local 150-GiB disk floor, memory headroom, GPU-idle and
cleanup checks; after a pilot the remote unit must be stopped by identity.  No
Resident budget, warm SLO, G1/G2 or baseline ranking is frozen.

## CURRENT — D280 re-enables the audited Resident-vLLM protocol; no GPU run yet

2026-10-06. The baseline harness commit `d9db181` removes only the historical
quarantine guard around the already audited Resident-vLLM protocol path.  CPU
checks, shell syntax, receipt compilation, frozen W0 trace/subset validation,
DP4/TP1 launch-spec generation and the fixed-output native-token dry-run passed.
The path now explicitly requires `dynamic_remote`, a prepublished endpoint,
separate service/auxiliary resource domains, deployment notice timing and
post-stop NVML release before a real receipt can close.  No model was loaded,
no remote request was sent and no GPU performance sample was produced.

The new provenance boundary is documented in
`docs/ieee_tc/D280_RESIDENT_PROTOCOL_ENABLEMENT.md`.  D280 does not promote
D213's local-static runs and does not freeze a Resident budget.  The next
authorized step is a fresh remote health/resource gate followed by one
protected 7B Resident pilot; only after that pilot passes will the three-run
Resident reference series be considered.  Prime's combined D278+D279
readiness diagnostic remains diagnostic-only (`n_correct=null`), so 3B and all
external baselines remain paused.

## CURRENT — D279 completes; combined readiness diagnostic reaches the pre-registered sample count

2026-10-06. The permitted identical cold replicate D279 completed with
4,000/4,000 terminal successes, eight complete phases, ten scale-ups, complete
dispatch tiers and zero tier-invariant conflicts.  Its single-run first-service
count is ten, so the generic strict analyzer reports the expected
`rerun_required=true`.  D278 contributes ten more.  An explicit allowlisted
configuration audit found only run/output/cache identity differences between
the parsed YAML files, with equal trace, adapter subset, generation contract,
request-map hash, seeds and feature gates; the combined diagnostic count is
therefore 20.  The compact table is in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d279_readiness_analysis/`.

D279's descriptive mean/P95 user TTFT is 2,062.5/6,242.5 ms and service TTFT
is 661.0 ms; its physical sidecar observes 15,895.426 GPU-s but leaves
`n_correct=null`.  These values are diagnostic and not G1/G2 or baseline
ranking results.  Local GPU release, service-path removal, and the owned
remote artifact process cleanup were confirmed.

The D278-versus-D219 audit is retained, but its different parent swap and
short-window PSI observations are only a correlated host-state difference.
`memory.swap.current` is cgroup swap usage and PSI `avg10` is a percentage of
stall time; neither observation proves that the replay incurred swap I/O or
that it caused the 151-s startup difference.  D278 therefore remains a
diagnostic stratum with an unresolved cause, not an “environment pollution”
exclusion and not evidence of an intrinsic PrimeLoRA latency regression.  The
unfavorable D278 values remain preserved and were not discarded merely because
they are slow.  The completed D279 evidence does not by itself identify the
cause, so no causal optimization decision is made from this diagnostic.

The metric boundary remains frozen: G1/G2 require correctness and common joint
SLO; CE is supplementary.  The no-delay storage point is a path diagnostic in
the bandwidth sensitivity, not a measured or rigorous performance upper bound.
The next action is to preserve the combined readiness evidence and begin the
next pre-registered 7B Prime optimization/qualification step only after the
CPU audit and table review.  Baselines and 3B remain paused.

## CURRENT — metric boundary frozen; D278 startup latency retained as an unresolved host-state diagnostic

2026-10-06. The operational metric and remote-boundary checklist is archived in
`docs/ieee_tc/METRIC_DECISION_FREEZE_20261006.md`; the parent V1/Remote Scope
V2/Final V2 protocol hashes remain unchanged.  The note freezes G1 as
physical-card GPU-s per correct request after `N_correct=N_plan` and joint SLO,
G2 as P95 user TTFT under the common 0.75 reference GPU-time budget, and CE as
supplementary.  It also records that the approved prepublished gzip creation is
deployment-time work, while request-triggered remote read/transfer/receive/
materialization remains in user E2E.

A read-only comparison of D219 and D278 is archived in
`docs/ieee_tc/D278_STARTUP_ENVIRONMENT_AUDIT.md` and its curated CSV.  Both
runs completed 4,000 requests with the same 7B trace, subset, generation
contract and prepublished delivery, but D278 began under approximately 12 GiB
parent swap and non-zero memory PSI, versus approximately 0.22 GiB and zero
short-window PSI for D219.  D278's engine/preload interval was 204.5 s versus
53.6 s; mean service TTFT stayed 681.7 ms versus 694.3 ms, while the first 100
user TTFTs averaged 142.6 s.  This is not evidence of a 4--5x native inference
regression, and D278's mean/P95 user latency is not promoted to a Prime
optimization target or a G1/G2 comparison point.

The current host preflight passes the Plan's 102 GiB memory headroom and
150 GiB new-task disk floor with no active experiment GPU processes; the
available disk is only slightly above that immutable floor, so no heavy replay
is authorized until its predicted growth and cleanup receipt are audited.  The
artifact-node independent capacity implementation is covered by five focused
preflight tests (5 passed); inference-host memory/disk/OOM rules are unchanged.
No Prime code change or GPU replay was made in this checkpoint.  The next
paper-level action is a clean 7B startup/resource-stratum validation under the
frozen contract; only after that can a causal readiness optimization or the
formal first-service blocks be considered.  3B and all baselines remain paused.

## CURRENT — D278 first-service audit bounds the planned-match question before the D279 replicate

2026-10-06. A CPU/offline join of the ten D278 natural scale-up epochs to
their first actual dispatches is archived in
`docs/ieee_tc/D278_FIRST_SERVICE_AUDIT.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d278_first_service_audit/`.
The ten rows all carry confirmed dispatch snapshots and complete tier fields;
the three very large waits (159--183 s) are concentrated in the initial
scale-out backlog, while native service TTFT remains 681.7 ms on average.

The zero planned-adapter matches are a legacy-prefix instrumentation boundary,
not a measured native match rate.  The IEEE path deliberately rejects that
prefix and starts owner-bound native file plans; eight activations also have
native GPU target plans, with two of the ten first-service adapters appearing
in those target lists.  The request field
`proactive_admission_evaluated=false` describes the request branch, not whether
the background native plan evaluated admission.  The audit identifies a
falsifiable optimization question—whether an owner-bound readiness publication
changes first-service delay—but no code change or performance claim has been
made.

The metric interpretation used from this point is recorded in
`docs/ieee_tc/METRIC_PROTOCOL_FREEZE_ADDENDUM_20261006.md`; it records G1/G2
as the primary gates, HydraServe-style warm/cold decomposition as a disclosed
reference, and the 100-Mbps versus current 1-Gbps link as separate strata.
Baselines and 3B remain paused; D278 is not a G1/G2 or SLO result.

## CURRENT — D278 completes post-D277 replay; runtime failure removed but first-service gate remains open

2026-10-06. The fresh same-configuration 7B eight-phase diagnostic completed
the full envelope: 4,000/4,000 requests arrived, submitted, terminated and
succeeded; all 4,000 native fixed-output/token-ID contracts and dispatch-time
tier snapshots are present; and the tier consistency validator reports zero
invariant conflicts. Tier counts are GPU 1,165, HOST 1,823, NVMe 897 and
Remote 115. Ten natural scale-ups and seven scale-downs occurred.

The D276 native-capacity-wait failure did not recur after the D277
lease-to-plan witness repair: no terminal error, failure observation or
capacity-conflict diagnostic was emitted, and the physical sidecar closes with
zero open leases. This is a runtime compatibility result, not proof that the
repair caused every earlier failure to disappear; the run is still
`formal_run=false` and `n_correct=null`.

The pre-registered first-service minimum remains unmet: only 10 observations
were produced, below the required 20. D278 is therefore excluded from
readiness, joint-SLO, G1/G2, lifecycle and baseline ranking denominators. Its
descriptive mean/P95 user TTFT is 6,895.5/22,698.6 ms, mean/P95 E2E is
11,204.6/27,220.7 ms, service TTFT is 681.7 ms, and physical owner-union use
is 15,349.592341 GPU-s; none is a published performance claim. Watchdog
evidence shows minimum host `MemAvailable` 94.23 GiB, service peak 20.47 GiB,
and zero high/OOM/oom-kill events. Local GPU cleanup completed; the managed
remote 7B artifact service was stopped and port 18081 was inactive. The
approved prepublished gzip cache was reused without request-time packing.

The compact status and immutable hashes are in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d278_readiness_retry9/`
and the detailed boundary is in `docs/ieee_tc/D278_7B_READINESS_POST_D277.md`.
The next action is a CPU/offline audit of why ten scale-ups yield only ten
first-service samples; only the explicitly permitted identical cold replicate
may follow. Baselines and 3B remain paused.

## NEXT GATE — D278 post-repair 7B readiness diagnostic registered; launch held

2026-10-06. After the CPU red/green reproducer and the 489 focused tests
passed, D278 is registered as one fresh, same-configuration cold diagnostic.
It changes only the D276 run/result/cache identity and tests whether the
lease-to-plan receipt repair prevents the observed capacity-wait failure.  The
existing seed-42 W0 trace, 500-adapter subset, fixed-output contract, vLLM
environment, remote prepublished delivery and all resource/control profiles
are frozen.  The config and launcher are
`configs/ieee_tc/20261006_d278_7b_readiness_8phase_full_retry9.yaml` and
`scripts/ieee_tc/run_d278_7b_readiness_8phase_full_retry9.sh`; their hashes and
acceptance boundary are sealed in `docs/ieee_tc/D278_7B_READINESS_POST_D277.md`.

No GPU replay has started in this registration step.  The final preflight must
repeat memory/disk/process/GPU cleanup, verify the existing remote service
without repacking, and preserve the exploratory/non-ranking identity.

## CURRENT — D277 identifies and repairs the proactive-lease witness schema defect

2026-10-06. D276 showed that a selected-adapter capacity failure could leave an
empty `gpu_reference_evidence` object because the exception escaped before the
inner RequestResult envelope.  The new bounded
`native_capacity_conflict_v1` diagnostic copies only the final blocker rows,
lease IDs, external-pin state, preparation-plan IDs, owner/epoch and request
identity onto the exception.  It does not retry, evict, relax fail-closed
checks, or infer a release.  Full native snapshots and cache contents remain
excluded.

The red/green CPU reproducer used a real proactive preparation lease and a
request-driven acquisition against the same native capacity.  Without the
plan-task binding it fails closed as before; with the exact lease-to-plan
binding it waits for the acknowledged preparation release and then succeeds.
The worker now emits `preparation_leases=[{lease_id,plan_id}]`, and the
controller accepts it only with one live engine-scoped plan task.  Unknown,
duplicate or mismatched identities remain rejected.  This is the minimal
source-level repair suggested by D276's observed failure path; it does not
change routing, eviction, admission thresholds or retry policy.

The request-lifecycle suite passes 185/185, and the GPU-reference plus
transfer-pressure suites pass 304/304 after the schema change (489 tests
across the three suites).  This is not yet a
readiness or performance result.  After the post-fix suites pass, one new
same-configuration cold readiness replay may test whether the D276 failure
reappears and whether the first-service sample gate can be met.
The protected-artifact seal check remains false only for the pre-existing
change to the sealed baseline file
`serverless_llm_baselines/scripts/replay_openai_trace.py`; the file was not
edited or staged here.

## CURRENT — D276 7B readiness replicate closes with one localized-but-unproven capacity-wait failure

2026-10-06. The registered same-configuration cold replicate completed its
4,000-request replay envelope after D275, with 4,000 terminal records. 3,999
requests succeeded and one (`req_00862`, `translate_lora_0030`) failed before
dispatch with `RuntimeError: native capacity conflict has no acknowledged
release path`. The retained traceback reaches `_wait_native_reference_capacity`,
but the failed terminal has an empty `gpu_reference_evidence` object. Thus the
artifact identifies where the controller rejected the wait, not which
candidate lease blocked it; it does not prove that D275 caused or fixed the
failure and does not authorize a blind replay.

Among dispatched records, pre-dispatch tier coverage is 3,999/3,999 and tier
invariant conflicts are zero. The run exercised ten natural scale-ups, seven
scale-downs, and ten first-service observations, so it misses the
pre-registered first-service minimum of twenty. The physical sidecar is
complete (15,878.133222 GPU-s, no open leases, `n_correct=null`) and cleanup
is complete; service/replay/watchdog return code is zero, native GPU-context
release is confirmed, and the owned service path was removed. No service or
auxiliary OOM/oom-kill occurred; service memory peaked at 21,792,223,232 bytes
and host `MemAvailable` remained 102.91--122.41 GiB.

D276 is consequently diagnostic-only and is excluded from readiness, SLO,
lifecycle, G1/G2 and baseline denominators. Its compact status table is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d276_readiness_retry8/d276_status.csv`;
the full claim boundary and artifact hashes are in
`docs/ieee_tc/D276_7B_READINESS_RECHECK.md`. The next falsifiable action is
CPU-only: exercise a proactive GPU-preparation lease and a request-driven GPU
acquisition against the same native capacity, retain the exact candidate and
release receipts, and test whether preparation leases are represented in the
acknowledged-release witness set. No new full replay is authorized until that
contract is either rejected or repaired with focused tests.

## CURRENT — D275 stale selected GPU copy now re-routes without relaxing identity checks

2026-10-06. A new CPU-only, falsifiable contract test isolated a race class
that was not distinguished by D266's incomplete witness.  It selected a
confirmed GPU copy, retired it, installed the same logical adapter and
verified immutable content from a new managed path, and then submitted the old
selection.  Before the change, the request raised the native source-conflict
exception at `_validate_source_binding`; this red observation was captured
before modifying production behavior.

`IEEEBackendGPUReferences._load_and_acquire` now returns the existing typed
`required_source_changed` result when (and only when) an observed source ID is
stale while the current live copy proves the same logical adapter name and
content digest.  The controller's existing whole-request re-resolution path
therefore remains authoritative.  Future-epoch and lease checks remain strict;
wrong logical identity, wrong content, staged-source mismatch, and callers
without a verified digest still reach the unchanged fail-closed validator.  No
native cache, pin, lease, or epoch is mutated by the typed conflict.

The focused native-reference suite passes **135/135**; the request-lifecycle,
pending-GPU-pipeline, and transfer-pressure suites pass **365/365** under the
bounded CPU scope.  Evidence and the compact table are in
`docs/ieee_tc/D275_STALE_GPU_SELECTION_RECHECK.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d275_stale_selection_recheck/d275_cpu_contract.csv`.
This closes a demonstrated stale-selection contract bug class, but does not
reconstruct D266 (which lacks old/requested/live source and plan witnesses),
and is not a GPU readiness, performance, SLO, G1/G2, or baseline result.  The
next gate is rechecking the existing numerical/common-warm/physical-release
requirements before any new 7B GPU readiness attempt; no replay is authorized
solely by D275.

Checkpoint `8cff52a` contains the implementation, test, and compact table; it
was pushed to `faaslora_origin/retry14_continuous_queue_v2`.  The pre-existing
dirty manifest and unrelated untracked review/figure artifacts were not staged.

## CURRENT — D274 3B historical bottleneck is profile-bound; no replay authorized

2026-10-06. A CPU-only audit compared the retained exploratory D138
Llama-3.2-3B projection with the fully resolved current 3B profile. D138's
12,588.6-ms mean TTFT is not a model-compute measurement: mean native vLLM
TTFT is 581.1 ms, while dispatch/admission, ingress queue, slot and
pre-runtime spans account for most of the user-visible delay. However, D138
used concurrency 8, dedicated `ieee_confirmed` coordination, explicit measured
service/preparation profiles and no-delay storage; the current profile resolves
to concurrency 4, automatic `adapter_affinity` coordination, predictive
scale-up and an effective-capacity gate. The historical value therefore
cannot be used to justify a current control-path rewrite or a formal 3B claim.

The audit and resolved-profile digest are sealed in
`docs/ieee_tc/D274_3B_HISTORICAL_BOTTLENECK_AUDIT.md` and
`paper_results/ieee_tc/p2_backend/20261006_d274_3b_profile_audit.json`.
They register falsifiable hypothesis H3B-1: a fresh current-profile prefix,
when authorized, must distinguish residual dispatch/readiness/slot delay from
native inference before any 3B full replay. This CPU-only step does not
authorize a GPU replay, 3B run, baseline, SLO, G1/G2 point or ranking. The
active gate remains the 7B numerical/common-warm/physical-release evidence;
the user-visible 3B 12.6-s figure remains historical and unqualified.

## CURRENT — D273 source-plan boundary audited; no validator change retained

2026-10-06. A CPU-only falsification check examined whether the verified
same-content path-migration shortcut could bypass an active preparation plan.
The direct validator call accepts such a source only when no plan identity is
carried; this is the request-side input that the controller is designed to use
to supersede a stale plan and advance the source epoch. The complete
transfer-pressure suite then passed 169/169 tests, including staging,
native-HOST, handoff, movement, and native-commit supersession boundaries; the
focused GPU-reference suite passed 135/135. The exploratory source edit was
fully reverted and no production validator change remains.

The evidence therefore distinguishes an unplanned demand that must trigger
plan supersession from an operation explicitly owned by a frozen preparation
plan. D273 does not identify D266's missing old/requested source state and does
not authorize a GPU replay, readiness point, G1/G2 point, or baseline
comparison. The audit is retained at
`paper_results/ieee_tc/p2_backend_qualification/20261006_d273_source_plan_supersession_audit/d273_status.md`.

## CURRENT — D272 separates source-path migration from content identity

2026-10-06. A CPU-only contract qualification inspected the unchanged native
source validator after the D271 partial diagnostic. Six focused source/identity
tests passed, covering live-copy relabelling, verified same-content migration
after complete retirement, preparation-plan ownership, host guards, and the
distinct CPU/GPU source views. The retained evidence is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d272_source_identity_contract/d272_status.md`.

This establishes the implementation's intended boundary: a new path is legal
only for the same logical adapter after all old native copies and preparation
plans retire and the immutable content SHA agrees; a live copy, active plan,
different logical name, or different content remains fail-closed. It does not
identify which predicate fired in D266, whose pre-instrumentation record lacks
the old/requested source, content, live-state, and preparation-plan fields.
Therefore D266 remains unresolved, no validator relaxation or retry is
justified, and D272 authorizes no GPU replay, readiness point, G1/G2 point, or
baseline comparison.

## CURRENT — D271 does not reproduce D266 and remains a partial diagnostic

2026-10-06. D271 replayed the registered 7B eight-phase diagnostic with the
bounded source-binding witness enabled. It observed 231 arrivals and 224
successful terminations before a deliberate SIGINT after the historical D266
position `req_00198` had completed. The native integer-ID reuse marker,
structured source-binding conflict, and retained failure observation were all
absent in the observed prefix. This rules out a claim of an immediate
deterministic failure at that position under this replay state, but it does not
reproduce D266, identify its old/requested/live source transition, or prove a
repair.

The run is recorded in
`paper_results/ieee_tc/p2_backend_qualification/20261007_d271_identity_diag_8phase/d271_status.md`.
It is classified as a manual-stop partial diagnostic with
`pass=false` and `production_launch_authorized=false`; the raw evidence is
retained and excluded from readiness, performance, SLO, G1/G2, and all ranking
denominators. The owned service path was removed and cleanup reported
`released=true`, but the terminal census did not prove allocator-level/native
context release. No OOM, cgroup high/max/oom/oom_kill event, remote mutation, or
historical-result overwrite occurred. The intermediate latency values are not
formal measurements.

D266 therefore remains unresolved. The fail-closed validator is unchanged, and
no further heavy identity replay is authorized without a new falsifiable
hypothesis and a witness that links old/requested/live source, content identity,
and preparation-plan state at the native rejection. The active execution gate
is the complete common warm reference plus physical whole-GPU allocation and
release evidence for Prime; baseline comparison and G1/G2 ranking remain
blocked until that gate is met.

## CURRENT — HydraServe/remote metric boundary audit frozen as a non-mutating addendum

2026-10-06. The authoritative metric calculation remains
[`METRIC_PROTOCOL_FINAL_V2.md`](METRIC_PROTOCOL_FINAL_V2.md), with the frozen V1 and
remote-scope V2 parents unchanged. A read-only audit of the HydraServe NSDI'26 paper,
the official repository, and the D77/D78 link records is retained in
[`METRIC_PROTOCOL_HYDRASERVE_REMOTE_AUDIT_20261006.md`](METRIC_PROTOCOL_HYDRASERVE_REMOTE_AUDIT_20261006.md)
(SHA-256 `ea5b82c0ddfdfcb4ac736f8fe76862a976cccb9ad081a4ec26eae4efe7d022ac`). It
confirms that HydraServe's published primary evidence is cold-start TTFT,
TTFT/TPOT SLO attainment, and a resource/cost trade-off—not CE—and that its
memory-time cost model is not a substitute for this project's physical GPU-s
lifecycle metric. The addendum also freezes the classification of the approved
one-time read-only gzip delivery cache as deployment-time work, while all
request-triggered remote read/send/receive/materialization remains in user E2E.
D77's 100-Mbps negotiation and D78/current 1-Gbps/full negotiation remain separate
execution states; no historical 100-Mbps coverage run is promoted to a performance
or G1/G2 point. This documentation step authorizes no new GPU replay.
The read-only 2026-10-06 state (both endpoints 1000/full, remote boot ID retained,
no service or link mutation) is recorded at
[`20261006_d270_link_readonly.json`](../../paper_results/ieee_tc/remote_qualification/20261006_d270_link_readonly.json)
(SHA-256 `c03f299d2a3c58c44cf4e9384058d3c3731dd2eb0850f2211c0ef4523aca8240`).

## CURRENT — request-level source-binding witness is instrumented and unit-tested

2026-10-06. The controller now retains a bounded, request-local diagnostic when
the unchanged native rejection marker `native integer ID reused for a different
adapter source` escapes selected-source preparation. The evidence binds the
offered request ID and adapter to the selected instance, submitted native
intent, selected source identity, owner/epoch and the original bounded worker
error text. Full routing snapshots are deliberately excluded to avoid
duplicating the adapter pool in every failure record. The exception is still
re-raised unchanged; there is no retry, fallback, validator relaxation, route
change, or ownership decision based on this field. The normal reservation
cleanup and unresolved-native-work handling therefore remain authoritative.

The same evidence is emitted in the request failure observation when the
exception is collected outside the inner execution path, closing the evidence
gap seen in D266 without retrospectively altering D266. A focused regression
test verifies that a synthetic D266-shaped error keeps request/source identity
and drops unrelated full-pool metadata; the complete request-lifecycle and
native-reference suites pass: **317 passed, 19 warnings**. This is an
instrumentation qualification, not a readiness, performance, G1/G2, or repair
result. A future bounded replay may use it only to capture the exact rejection
predicate; no replay is authorized solely because this instrumentation exists.

## CURRENT — D269 retry2 completes a phase-preserving identity diagnostic; no repair claim

2026-10-06. D269 retry2 replayed a fresh 500-request 7B prefix as **one
phase**, preserving the phase structure relevant to D266's `req_00198` rather
than repartitioning the prefix into D268's eight mini-phases. The structured
source-binding diagnostics were active, with no retry, fallback, validator
relaxation, or changed routing objective. The raw result and audit are stored
at `paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix_retry2/`.

All 500/500 requests terminated successfully (500 successes, zero failures),
including request 198 with the same source-request SHA as the D266 failure
position. The run observed three natural scale-ups and no scale-downs. No
terminal source-binding exception appears in the service or retained worker
logs. The launch receipt reports zero service/replay/watchdog return codes,
native context release and service-path removal; cgroup `high`, `max`, `oom`,
and `oom_kill` counts stayed zero. The physical-release proof field remains
`proves_physical_lease_release=false`, so the allocator-level release claim is
not promoted.

This is stronger than D268 for replay-state control, but it still does **not**
prove that the D266 identity transition was reproduced or repaired. The
terminal schema has no adapter ID/source path, retry, or transient-conflict
counter. The retained `main_outcome.json` does verify an
`education_lora_0007` NVMe source, content digest, preparation plan, and
completed native host preparation, but it does not link that preparation to
request 198. The analyzer also reports `vllm_available=false` although worker
logs show real vLLM 0.30.0/LoRA execution. Consequently the result is
exploratory, not a readiness, performance, G1/G2, or ranking point. Descriptive mean/P95
TTFT is 1205.5/2802.7 ms and mean/P95 E2E is 6302.1/12317.7 ms for the 500
request prefix; these numbers are not compared with formal runs.

The initial D269 attempt remains separately classified as a
`protocol_or_launcher_error` caused by a copied `source_audit_sha256` typo;
its GPU request count was zero and its receipt is retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix/`.

The validator remains fail-closed and unchanged. The next gate is a
read-only comparison of D266 and D269 source-selection, content-digest,
live-copy, and preparation-plan state. D269 does not authorize a new full
readiness replay or any formal Prime optimization claim.

The read-only comparison is now complete. D269's native outcome contains a
verified `education_lora_0007` NVMe source and content digest, one deferred
`stale_snapshot` preparation, and a later completed preparation from the same
source. D266 has no structured source witness at its rejection, and neither
run maps the native preparation record to request 198. Thus the comparison
confirms that D269 exercised the adapter path and that ordinary stale-snapshot
revalidation can complete, but it cannot identify D266's rejection predicate
or establish a repair. The validator remains unchanged; no new heavy replay
is authorized until a future witness can capture old/requested source,
content, live-copy, and preparation-plan state at the exact rejection. The
side-by-side evidence and the controlled-difference table are retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix_retry2/d266_d269_source_state_comparison.md`.

## CURRENT — D266 identity conflict unresolved; D267 configuration failure; D268 prefix diagnostic is non-causal

2026-10-06. The authorized D266 fresh cold rerun did not produce a formal
result. At request `req_00198` (227 arrivals observed before manual stop), the
native worker rejected adapter `education_lora_0007` with
`ValueError: native integer ID reused for a different adapter source` during
`_validate_source_binding`, called from `demand_load_and_acquire`. The
terminal journal records the request as failed with no instance, prompt hash,
or completion tokens. The worker traceback and all raw journals remain at
`results/ieee_tc/p2_backend_qualification/d266_20261006/7b_readiness_8phase_full_retry7/`.
The run is incomplete and is excluded from readiness, performance, SLO,
G1/G2, and ranking denominators. Watchdog samples show no OOM/high event and
the exact owned scopes were cleaned; the manual interruption means D266 has
no final successful launch receipt suitable for a formal release claim.

The failure is not an integer-ID hash collision: the frozen adapter IDs are
unique. D257 had an analogous source-binding failure for a different adapter,
and D265 succeeded for the same `education_lora_0007` request position. This
supports a path/source-state hypothesis, but the old and requested live
bindings at the D266 failure are not yet jointly captured. The strict live
relabeling validator therefore remains unchanged; no retry, fallback, or
relaxed source policy is justified.

The post-D269 non-GPU source-state contract check selected 31 source/identity
tests from `tests/test_ieee_tc_gpu_references.py`; all 31 passed (103 tests
deselected). It confirms the intended semantics: a live native copy cannot be
relabelled to another path, while an old retired copy can be reloaded from a
new tier when the immutable content identity agrees. This is a code-contract
check, not a reconstruction of D266's missing old/requested source witness.

D267 was a fresh configuration-only attempt. Its service profile rejected a
hand-copied `source_audit_sha256` identity before model startup, with no GPU
inference. The failure and its receipt are retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d267_identity_diag_prefix/d267_status.md`.

D268 then corrected the configuration and ran a fresh 210-request bounded
prefix. It completed 210/210 requests with no terminal identity-conflict
failure, ten scale-ups, seven scale-downs, clean native-context release, and
owned service-path removal. There is no direct native identity-conflict
counter in the terminal schema, so this is an indirect non-recurrence
observation. The strict cleanup field
`proves_physical_lease_release=false` remains a physical-release warning.
This is not a repair result: the prefix retained eight
multi-cycle phases, so it used seven 26-request phases and one 28-request
phase rather than D266's eight 500-request phases. Its later phases reached
88.62 s mean TTFT and 87.26 s P95 dispatch-window wait, which is a queue/drain
effect of this diagnostic partition and not an acceptable final performance
point. D268 metadata remains exploratory (`formal_run=false`), with empty
frozen-settings/source-config hashes and `vllm_available=false`; its full
evidence and caveats are summarized in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d268_identity_diag_prefix/d268_status.md`.

The current decision is therefore **no repair claim and no new GPU replay**.
The next gate is a source-level comparison of D266 and D268 replay/phase
semantics and a controlled, falsifiable identity-state diagnostic. Only after
that audit identifies the changed state needed to reproduce or rule out the
conflict may another heavy readiness run be authorized. The readiness gate,
G1/G2, and the user's latency target remain open.

## CURRENT — D265 7B eight-cycle Full readiness diagnostic completes; one gate remains open

2026-10-06. The fresh 7B `d265_7b_readiness_8phase_full_retry7` diagnostic
completed its pre-registered 8-cycle workload without changing the frozen
trace, 500-adapter subset, model, fixed-output contract, vLLM environment,
profiles, resource envelope, or read-only prepublished remote delivery. The
raw result is retained at
`results/ieee_tc/p2_backend_qualification/d265_20261006/7b_readiness_8phase_full_retry7/`.
Its result SHA256 is
`545da871f734a0a7b6dbf25fa97af66465a6cdc9db33ca67182d535c0ca1256a`, launch
receipt SHA256 is
`faf08552233d7869926e562e32f3ba2dc5b44860cbb7b73d9f56453100e8c834`, config
SHA256 is
`95717b4b1a6a53c0f0f1184f552672c909bcc91f6bbc5db40a8eec3dd84c754a`, and
launcher SHA256 is
`d24565ae4e02d56fb56c0c6a0a18c040c729d5f9bb54aac5b4a73dcfd77d6027`.

The categorical execution contract is complete: 4,000/4,000 requests
arrived and terminated successfully, with zero failures, 4,000/4,000 native
fixed-length/token-ID contracts, 4,000/4,000 canonical prompt hashes and
native timing observations, 100% pre-dispatch tier coverage, and zero tier
invariant conflicts. The tier partition is GPU 1,190, HOST 1,684, NVMe 994,
and Remote 132; each tier's readiness booleans are internally consistent.
The run exercised eight 500-request phases, ten natural scale-up events and
seven natural scale-down events. The terminal journal contains no nonempty
`failure_observation`, and the service, replay, watchdog, physical allocation
journals, GPU-context release and owned service-path removal all closed
successfully. Physical owner-union accounting reports 15,883.368 GPU-s
(3.970842 GPU-s per offered request), with no open lease; this is the canonical
physical field and must not be replaced by the legacy lifecycle estimate.

The readiness diagnostic gate is **not yet passed** solely because the
pre-registered first-service sample minimum is 20 per model/variant. D265 has
10 natural scale-out first-service observations (all ten are dispatch-time
Remote/cold and none has a planned-adapter match). The existing analyzer
therefore reports `passed=false`, `rerun_required=true`, while all other gate
components pass. The protocol explicitly allows one additional identically
configured cold run; D266 is authorized only to supply the missing independent
first-service observations, then the two runs will be aggregated without
pooling requests as independent repetitions.

The result remains a **diagnostic**, not a formal PrimeLoRA comparison or
G1/G2 point: metadata intentionally marks `trace_role=exploratory` and
`formal_run=false`, `n_correct=null`, and the frozen-settings/source commit
fields are empty. The metadata also reports `vllm_available=false` despite
the native vLLM token-timing path; this provenance inconsistency is retained
as an audit caveat rather than silently repaired. The descriptive overall
TTFT/E2E values (2,159.8/6,510.8 ms mean; 7,581.6/14,457.7 ms P95) are not
used as ranking evidence. A readiness summary table and diagnostic figures
are stored in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d265_readiness_retry7/`
and `figs/ieee_tc/p2_backend_qualification/20261006_d265_readiness_retry7/`;
the first-service figure is correctly omitted because n=10 is below the
published minimum. The request-local routing hardening has no recurrence in
this run, but D265 does not prove that it caused the earlier D258 exception.

No remote configuration, packing, cache regeneration, or unrelated workload
was performed during measurement. The remote records report the approved
`prepublished_gzip_v1` delivery with `remote_pack_performed=false`; request-
triggered transfer remains part of the application timeline, while one-time
publication and unrelated remote work remain outside it under the metric
protocol. D265 is therefore closed as a successful execution diagnostic with
one sample-count gate outstanding, not as evidence that PrimeLoRA has already
met its final optimization target.

## CURRENT — D258 retry6 stopped after an unlocalized controller exception; routing hardening is diagnostic

2026-10-06. The fresh 7B eight-phase Full diagnostic
`d258_7b_readiness_8phase_full_retry6` was intentionally interrupted after
approximately 74 arrivals. Scale-out reached four instances and host/GPU
memory remained healthy, but one request terminated before reservation binding
with `AttributeError: 'NoneType' object has no attribute 'replica_id'`. The
terminal record has no instance, dispatch snapshot, prompt hash, or completion
tokens. No native-RPC failure, worker transport error, or OOM was observed.

This run is an incomplete diagnostic and is excluded from every readiness,
performance, SLO, G1/G2, and ranking denominator. The manual interruption means
its final release proof is incomplete; the raw terminal journal, service log,
watchdog trace, and launch receipt remain at the unique retry6 result root.
Post-cleanup GPU/process checks are clear.

The exact controller source line is not established by this artifact. The
request was observed before reservation binding, but the collector retained
only the exception type/message and the enclosing run was manually
interrupted. In particular, the old selection call and the subsequent read of
`last_ieee_decision` had no intervening `await`; the shared diagnostic field is
therefore a plausible robustness hazard, not a demonstrated cause of this
failure. It must not be reported as the root cause.

The repair keeps the immutable decision returned by the selection call in the
request-local execution context and retains `last_ieee_decision` only for
diagnostics and legacy tests. This is defensible request-local semantic
hardening, not a causal explanation, and it adds no retry, fallback, altered
SLO, or changed routing objective. Focused routing, pending-GPU, and
request-lifecycle tests pass; a fresh 100-request bounded smoke also completed
with 100/100 terminal successes and complete dispatch-tier observations. That
smoke is still diagnostic only. Before another 4,000-request readiness run,
the collector now preserves a bounded Python traceback for controller task
exceptions so any recurrence can be localized from the immutable raw result.

## CURRENT — D260 bounded RPC reproducer confirms a repairable combined-reference schema defect

2026-10-06. The fresh 7B eight-request bounded reproducer
`d260_7b_rpc_prefix8` completed with 8 arrivals and 8 terminal records (7
successes, 1 failure). It is a diagnostic only and is excluded from all
performance, SLO, G1/G2, and ranking denominators. The launch receipt passed
service, watchdog, replay, cleanup, GPU-context release, and service-path
removal checks; the failed request and all worker logs remain at the fresh
result root
`results/ieee_tc/p2_backend_qualification/d260_20261006/7b_rpc_prefix8`.

The new worker-side error instrumentation identifies the failure before native
GPU mutation: the controller supplied a combined GPU reference containing the
immutable `expected_content_sha256` field introduced by D258, while
`InferenceEngine.ieee_register_pending_gpu_reference` still required the old
eight-field set. The worker therefore raised
`ValueError: combined registration requires an exact selected GPU reference`,
the parent correctly treated the reply as ownership-uncertain, and the worker
was retired. vLLM initialization completed normally and no OOM or transport
failure was observed. This is a data-flow contract mismatch, not evidence of
an algorithmic readiness or capacity failure.

The minimal repair accepts exactly the legacy reference or that reference plus
one validated lower-case SHA-256 content identity; all other fields and invalid
digests remain rejected before registration. The repair is covered by the
pending-GPU pipeline and native-RPC ownership tests. D260 config and launcher
identities are respectively
`077bb73938e778b5dbad38bcdb33098a0cb53e36276a710a133a726a6b317ade` and
`a9075a8710a3eb5fccbb1a48d4dcb9a6a0c8f386f42739da`.

No new 4,000-request GPU replay is authorized until this repair is committed
and a fresh bounded smoke confirms that the digest-bearing combined path reaches
the native owner and returns an acknowledged receipt without uncertainty.

## CURRENT — D261 post-repair bounded smoke passes the native combined path

2026-10-06. The first D261 launch was a gate-only rejection because its fresh
configuration omitted the required preparation-profile `ewma_beta`; it created
no request result and passed GPU-context and scope cleanup. That evidence is
retained at `d261_20261006/7b_rpc_prefix8` and is excluded from all denominators.

The corrected retry used a new result/cache identity
`d261_20261006/7b_rpc_prefix8_retry2` with the same frozen trace, adapter
subset, model, vLLM environment, and remote delivery contract. All 8/8
requests completed successfully, including four requests routed with a
confirmed GPU source and the digest-bearing combined pending/admission path;
all eight native intents carried `expected_content_sha256`. The result contains
zero `subprocess_native_rpc_failed_no_retry` records and zero ownership-uncertain
guards. Every request's native acquisition receipt echoed the expected content
identity, and the launch receipt confirms service, watchdog, replay, cleanup,
GPU-context release, and service-path removal.

This bounded smoke is diagnostic rather than a performance result. It clears
the schema-repair gate for a fresh D258 4,000-request readiness diagnostic, but
it does not by itself establish readiness, SLO, or G1/G2 performance. The
corrected config and launcher hashes are
`85e94ff08cce8cfa2aaebf999ed122b2464047802d5631caeeb8917063342663` and
`0a5cafe0121a7fa3bf3cf28e1d00701041d6340a68e59e405978d289795e7af8`.

## CURRENT — D258 retry5 is an interrupted diagnostic; native RPC ownership failures remain unresolved

2026-10-06. The fresh 7B 8-phase Full diagnostic
`d258_7b_readiness_8phase_full_retry5` was started with the registered config and
launcher identities below, but was stopped after the failure mode became
decisive. At interruption, 177 requests had arrived, 100 had terminal records
(81 successes and 19 failures), and the run had already recorded 15
`subprocess_native_rpc_failed_no_retry` failures plus four
`native RPC ownership unresolved; new generation withheld` guards. The partial
run therefore fails the 4,000/4,000 readiness gate and is excluded from every
performance, SLO, G1/G2, and ranking denominator. Its raw logs, watchdog trace,
terminal journal, and worker-failure workdirs are retained at the fresh result
root; no old result was overwritten.

The failure is not attributable to host memory pressure or GPU OOM: watchdog
samples show zero cgroup OOM/high events, host memory remained available, and
the vLLM workers reached `engine ready` before their parent-side native RPC
channel became unresolvable. The strict no-retry behavior is intentional and
correct for safety, but the repeated empty/closed native exchange is a new
runtime defect to localize before another readiness run. The interruption also
caused the watchdog's final physical-release proof to be incomplete; current
GPU/process checks after cleanup are clear, but this partial run cannot provide
release evidence for a readiness claim.

The next action is a source-level RPC/worker-lifecycle audit using this raw
evidence (including the first failing request and its worker log), followed by a
minimal focused reproducer and tests. A new GPU replay is not authorized until
the cause is explained and the repair is test-qualified.

## CURRENT — D258 native identity/capacity repair is test-qualified; rerun still gated

2026-10-06. D257 remains excluded from formal readiness because it completed
3,993/4,000 requests and exposed two separate runtime defects: six requests
could not identify an acknowledged release owner for a proactive native
preparation lease, and one request rejected a path change for the same
immutable adapter content. The original D257 evidence is unchanged.

D258 repair code now carries the manifest `content_sha256` through demand,
proactive preparation, file-to-native-HOST preparation, the vLLM bridge, and
the worker admission payload. A path change is accepted only for the same
logical adapter, with a verified equal digest and after the old native copy is
no longer live; legacy commands without a digest retain strict path identity.
Native capacity receipts expose optional preparation `plan_id` witnesses, and
the controller waits only for the matching same-engine preparation task. A
failed or cancelled task is not treated as capacity. No polling, forced
eviction, retry of an unresolved native RPC, or external-pin release was
introduced.

Validation completed in the qualified vLLM 0.30.0 environment:

- 484 native GPU-reference, request-lifecycle, and transfer-pressure tests;
- 177 scheduler-observation and service-event tests;
- 289 basic smoke tests and 19 external-replay contract tests;
- compile and `git diff --check` checks.

The added tests cover digest-preserving path relocation after retirement,
rejection of a mismatched digest, and worker-side digest publication. No GPU
replay was started after this repair. The next gate is a fresh D258 7B
8-phase diagnostic with the existing 4,000-request contract; it must satisfy
4,000/4,000 terminal success, complete dispatch tiers, zero tier conflicts,
at least eight natural scale-ups and at least 20 true first-dispatch samples
before any readiness claim. D257's seven failures remain visible and are not
reclassified.

The fresh diagnostic is registered as
`d258_7b_readiness_8phase_full_retry5` with config SHA-256
`fbb61e70edf83c89d1c272374a54c08e87b1ea04e3378e9eea9cc05cf50988b8` and
launcher SHA-256
`f561b5980f3aff4c8d8fd713b9b0f1da87d553dfe087873026307a8647d54349`.
Its result, NVMe, and `/dev/shm` roots are fresh and its fields match D257
after identity substitution; no old result directory is reused.

## CURRENT — D249 fresh audit closes retry6 provenance and quarantines stale retry4

2026-10-06. A fresh read-only integrity audit of D247 retry6 confirms an
internally consistent bounded diagnostic: 22/22 requests completed, 19/22
native/HF sequences match exactly, and the three remaining first divergences
are HF top-two near ties (margins 0.015625, 0, and 0.015625). The strict result
therefore remains `pass=false`; no tolerance, post-hoc rewrite, or successful
subset enters the correctness denominator. Cleanup, GPU-context release, and
owned-service removal are confirmed.

The overall D247 evidence set is `FAIL` for integrity because retry4 predates
the strict predicate fix and contains a contradictory `pass=true` field.
Retry6 alone is an internally consistent `WARN` diagnostic; it does not become
a semantic pass through the fresh audit.

The audit also found that the older retry4 artifact reports 17/22 exact matches
but `pass=true`, which contradicts the current strict predicate. Retry4 is now
explicitly quarantined as stale contradictory evidence. Its raw files remain
unchanged and are retained only for provenance. D249 permits no semantic,
performance, SLO, G1/G2, or baseline claim from D247; the next admissible step
is the already registered 7B physical/admission/readiness bounded diagnostic.
Details and immutable hashes are in
`docs/ieee_tc/D249_D247_RETRY6_INTEGRITY_AUDIT.md`.

## NEXT GATE — D247 independent 7B PEFT sequence reference prepared; no GPU run yet

2026-10-06（D246 后的独立语义门槛）. D246 已经证明固定输出的原生 token-ID
序列可以完整穿过真实 worker、JSON/RPC、controller 和 terminal sidecar，但其
`n_correct` 仍为 null，不能直接作为正确性或正式资格证据。D247 注册一个只读的
22-request 原始 checkpoint + PEFT 对照：复用 D246 的结果、W0 冻结 trace、同一
canonical renderer、现有 7B checkpoint 和 500-adapter 池；不生成新权重、工件、负载
或服务结果，也不进入性能分母。

D247 在单张可见 GPU 上以离线 HuggingFace/PEFT、FP16、eager attention、同一
`fixed_length_greedy_v1` 目标长度和关闭 EOS 的语义逐请求生成，并在生成前校验
prompt hash、prompt token-ID hash、target map、adapter safetensors keys/dtype/value。
验收要求 22/22 完成且 native token-ID 序列逐元素完全一致；任何不一致都保留首个
分歧位置并停止，不使用文本重分词、容差或成功子集筛选。D236 的首位置参考仅作
辅助证据，不能替代该序列级门槛。

本条目前仅登记协议、脚本和静态检查，尚未启动 GPU。启动前仍需重新执行内存、磁盘、
GPU 空闲、进程清理和结果路径唯一性检查；若 D247 通过，下一步才是按计划恢复受保护的
7B Full/资格路径；3B、外部 baseline、G1/G2 和比较图仍暂停。详细协议见
`docs/ieee_tc/D247_PEFT_SEQUENCE_REFERENCE.md`。

两次启动尝试都在 GPU 加载前被门控拒绝，且没有创建结果或 launch receipt：第一次是
CUDA/PEFT 环境的 Python 3.12.12 缺少 `signal.pidfd_send_signal`，第二次虽换用系统
Python，但没有从预期的共享 auxiliary cgroup 启动。两者都不是请求失败或序列不一致，
空路径也没有计入实验。脚本现在同时固定两项启动契约：由系统 Python 承担外层
watchdog，并通过 4-GiB systemd auxiliary scope 启动它；CUDA 环境仅作为受限子进程
执行 PEFT。资源门、输入、checkpoint、adapter、生成语义和输出路径均不变，修正后需
重新通过同一启动门运行一次。

随后一次启动已通过内存/磁盘预检，但因缺少显式 NVML binding/SHA 在 GPU 加载前被
拒绝；该 launch receipt 作为 `d247_20261006` 的失败门控证据保留，不进入分母。检查
确认 11 个旧 `primelora-tc-aux-*` scope 均 `TasksCurrent=0`，随后只停止这些空 scope
以释放 systemd/inotify 资源；没有触碰任何有任务的 scope。下一次使用新的
`d247_retry3_20261006` 身份，并显式固定已验证的 NVML 路径和 SHA；若通过，才可进入
独立 PEFT 序列比较。

D247 retry3 随后通过了 containment、NVML、watchdog 和清理门，并生成了唯一结果，
但子进程在加载模型前报 `ModuleNotFoundError: scripts`；结果为
`measurement_complete=false`、请求数 0，没有 token 或 PEFT 序列，故不构成语义失败。
根因是 systemd 子进程 cwd 不受保证，而参考函数隐含依赖仓库根目录。修正已改为在
导入既有 renderer/workload plan 前显式加入仓库根目录，retry4 使用新的
`d247_retry4_20261006` 身份；retry3 结果和 launch 证据均保留。

D247 retry4 完成了独立原始 checkpoint + PEFT 生成并安全清理，但 22 条序列只有
17 条逐元素一致，`matched_reference_consistency_pass=false`；5 条差异中有两条是
HF 参考在 EOS token 处提前停止，而 native `fixed_length_greedy_v1` 明确要求
`ignore_eos`，另有少数请求在更早位置出现后端数值分歧。因此 retry4 是有效的
语义诊断、不是资格通过，也不进入任何性能分母。实现中的顶层 `pass` 已修正为只有
逐序列全匹配才为 true，避免把“测量完成”误读为“语义通过”。下一步 retry5 使用
显式空 EOS 集合和与服务一致的 KV cache，在全新身份上复核；3B、baseline、G1/G2
仍暂停。

D247 retry5 按修正后的 empty-EOS + KV-cache 语义完成并安全清理，结果为 19/22
逐元素一致，`measurement_complete=true` 但 `pass=false`。剩余 3 条均在目标长度
内生成完成，差异分别发生于 finance/ecommerce 请求的第 12–28 个位置，不再是 EOS
提前停止；这说明跨 vLLM FLASH_ATTN/FlashInfer 与 HF eager/PEFT 的 FP16 数值路径
可能在近 tie 处选择不同 token，但尚未足以把它客观归类为纯数值差异。下一步只做
一次 retry6 margin diagnostic：记录 HF 首分歧点 top-2 分数、margin 和 native token
是否位于 top-2，不改变生成或验收语义。若为近 tie，则将精确序列门降级为诊断证据并
以共同 fixed-length/adapter/prompt contract 作为资格前提；若不是，则继续停在语义
问题定位，绝不进入正式 Full/G1/G2。

D247 retry6 完成了预登记的 margin diagnostic。严格谓词仍为 19/22、顶层
`pass=false`；3 个差异位置的 native token 均在 HF top-2，FP16 top-1/top-2
margin 为 `0.015625/0/0.015625`。所有 prompt/tokenizer/target/adapter tensor
identity 检查、GPU-context release、service scope cleanup 通过。依据冻结 V1
“跨后端不要求逐 token 相同、同后端变化必须审计”的条款，D248 将严格序列结果保留为
诊断而不把近 tie 误判为 adapter 或请求语义错误；同时不把它改写成 22/22 exact pass。
D248 允许下一步进入已登记的 7B physical/admission/readiness bounded diagnostic，
随后才可进行 4,000-request Full qualification。3B、外部 baseline、G1/G2 和比较图
仍暂停。完整裁决与 immutable hashes 见
`docs/ieee_tc/D248_CROSS_BACKEND_SEQUENCE_RECONCILIATION.md`。

D249 新鲜审计随后确认 retry6 内部证据一致，但发现旧 retry4 产物把 17/22
精确匹配错误标成 `pass=true`。retry4 已隔离，不进入任何正确性分母；retry6
仍保持严格 `pass=false`。因此下一项只执行预登记的 D250 7B Full/Delayed
8-phase readiness 诊断，规格见 `docs/ieee_tc/D250_7B_READINESS_8PHASE_DIAGNOSTIC.md`。

D250 Full 的第一次启动在模型加载前被拒绝：外部 replay 校验遗留了
`multi_cycle_phases == 1` 限制，和已批准的 8-phase readiness 协议冲突。该
attempt 的 `N_arrived=0`、`service_returncode=1`，watchdog、GPU context 和
scope cleanup 均正常，不能归类为服务或性能失败。D251 记录了最小修正：保留
单一冻结 replay、单一 scenario/run、trace SHA 和 guarded receipt，仅移除错误
的 phase-count equality；修正后必须使用新的 launch identity 重试，不覆盖这次
门控失败证据。

D252 retry1 通过了外部 replay 顶层门，但在场景构造器内再次触发了相同的
`multi_cycle_phases == 1` 旧限制，仍在模型初始化前结束，`N_arrived=0`。D253
记录了第二处边界修正：构造器只拒绝 tiny-witness，继续绑定同一 guarded receipt、
trace request order、context 和正 phase count。D252 仍是独立失败证据，后续重试
必须使用新的 D254 identity。

D254 retry2 已进入真实 vLLM Full stack，但在首个 phase 收到 6 条请求后触发
`external replay cannot reset at phase/drain boundaries`，随后安全退出；这是
同一路径中第三个把单窗口 replay 假定为全 4,000 条的旧守门器。D255 记录了
最小修正：单一 ingress consumer 加连续全局 cursor，8 个窗口严格按
`(start_index,count)` 接续，默认一次性 API 保持不变；非连续窗口、第二 consumer
和不完整窗口 fail-closed。D254 的 6-request partial attempt 不进入任何分母，
下一次必须使用新 launch identity。

## CURRENT — D246 passes real-worker token transport; independent semantic gate remains

2026-10-06（D243 retry2，7B 22 请求固定输出诊断）. 在 D245 修复后，新的唯一
evidence root 完成 22/22 请求，失败 0。所有请求均有
`output_contract_match=true`、`timing_contract=ieee_tc_native_v1`、真实非负整数
token-ID 序列及匹配 digest，序列长度逐请求等于目标长度；终端 journal 也为
22/22 `success=true/native_contract_matched=true`。这证明类型修复已穿过真实
worker、JSON/RPC、controller 和 terminal sidecar，不是单元测试假象。

该运行仍不是 Full 正确性或性能结果。物理 sidecar 绑定结果 SHA，记录
`n_plan=22,n_terminal=22,n_native_contract_complete=22,n_correct=null`、无 open
lease、294.1641511830967 physical GPU-s；`n_correct` 必须等待独立原始 checkpoint
PEFT 序列比较。D236 的首位置参考不能替代这一步。watchdog 无 OOM/high/swap
事件并确认 GPU context 释放；远端 7B 服务已停止。详细证据见
`docs/ieee_tc/D246_TOKEN_CAPTURE_RETRY2.md`。3B、外部 baseline、G1/G2 和正式
Full replay 继续暂停。

## CURRENT — D245 localizes the D243 retry1 token-capture failure

2026-10-06（数值诊断失败后的进程间证据修复）. D243 retry1 completed the
protected 22-request replay and released all four allocation owners, but all
22 requests failed with `opt-in completion token capture is invalid`.  The
source-level worker contract attaches a token-ID list; the dedicated parent
proxy and the parent RPC timing annotator independently applied scalar
coercion, turning that list into a numeric fallback before controller
validation.  Focused tests reproduce this path, but the GPU-level causal
interpretation remains provisional until retry2.  The retry is excluded from
correctness, SLO, G1/G2 and ranking denominators.

D245 adds one shared validator-preserving normalizer for both parent boundaries.
It preserves only the opt-in `completion_token_ids` list after checking
non-negative integer IDs; ordinary timing fields keep their legacy scalar
normalization and the default-off digest-only schema is unchanged.  A new
encode/decode → parent-normalization → proxy-return integration test passes.
Pinned tests pass: 58 service-routing, 4 token-capture, and 25 native-timing
tests (87 combined).  Full evidence and the claim boundary are in
`docs/ieee_tc/D245_TOKEN_CAPTURE_RPC_FIX.md`.

The retry1 physical sidecar is cleanup-complete (`n_plan=22`, `n_terminal=22`,
no open leases, 318.4623162969947 observed physical GPU-s) but reports
`n_native_contract_complete=0` and `n_correct=null`; no low-cost or performance
conclusion is admissible.  Its metadata also reports `vllm_available=false`
while the service log records dedicated vLLM/GPU execution attempts; this is a
provenance warning to resolve by recording worker runtime identity in retry2,
not evidence that the retry was eligible.  The next action is a fresh guarded
retry with a new result identity, followed by the existing bounded
numerical-reference scope; 3B and all external baselines remain paused.

## CURRENT — D244 closes a profile-identity contract failure; D243 retry is held

2026-10-06（D243 首次启动失败，已完成因果审计）. D243 在第一条模型请求完成前
被冻结 service profile 校验拒绝：`ieee_token_id_capture` 是 D228 明确的纯观测
开关，但旧的 `model_identity` 将它当作运行语义字段，导致 profile/configuration
上下文不匹配。一次 replay header 已建立后，runner 安全停止；没有 request
terminal、GPU allocation、token sequence、SLO 或性能数据。远端 7B 发布服务已
停止，失败原始证据保留在唯一 D243 launch root，不进入任何分母。

该问题不通过手工改 profile SHA 或放宽校验解决。D244 对第一性原则的修正是让
`FrozenServiceProfiles.model_identity` 及 stdlib preflight identity 共同排除唯一的
诊断字段 `ieee_token_id_capture`，与已存在的 placement-only 排除保持一致；模型、
权重、采样、路由、准入、profile 文件、环境/资源/input context 和论文公式均不
变。普通 backend/model 配置变化仍 fail-closed。D244 checkpoint 的服务路由 58
项和 token-capture 3 项聚焦测试通过；下一次重试必须使用新结果身份，并重新执行全部本地主机/
GPU 门槛。详情见 `D244_OBSERVATION_FLAG_PROFILE_IDENTITY_FIX.md`。

## CURRENT — D243 7B native token-capture diagnostic prepared; GPU launch held for final gates

2026-10-06（D228 后的数值正确性诊断，尚未启动 GPU）. 已登记新的 22-request
Llama-2-7B 配置 `configs/ieee_tc/20261006_d243_7b_token_capture_22req.yaml`，其
SHA-256 为 `b57acd9c6e2e61b20535afac697be6ad81fc39767cd672262221d91cd61cb213`。
它复用冻结的 D166/D167/D207/D208 请求与内容索引、D227/D238 的模型/profile、
`fixed_length_greedy_v1` 合同和 D78/D80 已发布 gzip 工件；仅显式开启
`ieee_token_id_capture`，并使用新的结果/NVMe/HOST 根目录。没有生成权重、LoRA
工件、trace 或第二份解压池。

离线 YAML、路径和 SHA 检查通过。清理遗留离线扫描后，当前一次 preflight 观察到
`issues=[]`、`MemAvailable=121,837,080,576` 字节、推理目录剩余
`172,863,201,280` 字节，均高于未改变的 109,521,666,048 字节内存和
161,061,273,600 字节磁盘起始门槛；GPU 空闲状态仍需在启动前再次读取。远端
D233 的一次性发布 gzip 缓存、内容 SHA 和独立 artifact-disk 峰值规则已具备，
但服务 invocation、结果 sidecar 和独立 PEFT 对照仍不存在。

D243 只允许作为数值诊断：必须得到 22/22 成功的固定输出、逐请求原生非负整数
token IDs、完整 dispatch/admission/physical sidecar，并与原始 checkpoint PEFT
参考做精确序列比较；它不能进入 G1/G2、SLO、性能排名或 baseline 分母。启动前
会重新做本地内存/磁盘/GPU/cleanup 与远端服务状态检查；任一失败均保留证据并停
止，不降低阈值。详细协议见 `D243_TOKEN_CAPTURE_22_DIAGNOSTIC.md/.json`。

## CURRENT — D242 registers one readiness-publication hypothesis; host gate still closed

2026-10-06（离线瓶颈登记，无 GPU 重跑）. D235 的证据把下一处可证伪瓶颈
收敛为一个边界：新 dedicated runtime 在 engine 初始化后即可进入 confirmed
router，而 activation preparation 可能仍在执行；D219 新副本首 60 条中只有
10 条是 confirmed GPU source，最早首批 TTFT 达 11,288.8 ms。D242 将已存在的
`ieee_handoff_policy=delayed` 作为第一受控变量，Full 为配对参考，NoHandoff
只作机制边界；不复用 legacy handoff 字段，也不把非 GPU source 改称 GPU-ready。

当前 inference-host `MemAvailable` 在清理遗留离线扫描进程后约 93 GiB，仍低于
批准计划和 preflight 的 102 GiB（80+4+16+2）要求，且 swap 已有使用。因此没有
启动模型、远端服务或 replay。D242 规定内存门通过后的顺序：数值/输出 gate、
D226 显式 admission bounded diagnostic、Full/Delayed natural-scaleout 诊断，
然后才考虑新的 7B 留出块。3B 和外部 baseline 继续暂停。

## CURRENT — D241 canonical physical-resource sidecar gate implemented

2026-10-06（测量完整性实现，无 GPU 重跑）. D240 要求的物理生命周期验收已
落实：受保护运行在结果 JSON 写入且所有 allocation owner 完成清理后，生成
`physical_resource_summary.json`，其中的 `physical_gpu_seconds_total` 只来自
最终 owner-union summary，并绑定 allocation journal、terminal、summary 和
结果文件的 SHA-256。旧的 `infra_gpu_seconds_total` 与
`instance_lifecycle_log` 明确不能替代该字段。正式 Fig. 9 的
`--formal-matrix` 路径会拒绝缺少 sidecar、未完成释放、分母不完整或哈希不匹配
的记录；旧结果没有被重写。

物理生命周期单元测试 32/32 通过，Python 编译和 diff 检查通过。更大的 launch
测试中有 7 个环境阻塞（当前 Python 无 NVML、测试环境缺少可选 vLLM），不是本次
sidecar 变更引起的协议失败。D241 记录实现与验收；没有启动模型、远程 artifact
服务、baseline 或正式 replay。PrimeLoRA 7B qualification/optimization 仍是
唯一下一步活动，3B 与外部 baseline 继续暂停。

## CURRENT — D239 freezes remote latency scope, 100-Mbps interpretation, and HydraServe-aligned metrics

2026-10-06（只读指标与结果审计）. A new protocol audit,
`docs/ieee_tc/D239_REMOTE_METRIC_HYDRA_PRIME_AUDIT.md`, records the current
evidence and does not start a GPU or remote service. The current artifact route
is 1000 Mb/s/full on both observed endpoints with zero reported NIC errors. The
historical D77 100-Mbps/full state remains a separately labelled negotiation
state: its physical cause was not established, so it is neither silently
relabelled as 1-Gbps nor treated as proof of a permanent 100-Mbps device. The
formal contract keeps request-caused remote read/send/receive/write/extract and
contention in user TTFT/E2E, while one-time prepublished gzip construction,
administration, scans, cleanup, and request-external packing stay outside the
request result and are separately audited.

The metric decision is now explicit: HydraServe's warm-derived TTFT/TPOT SLO
logic is a disclosed reference, while PrimeLoRA's primary comparison remains
G1 (verified physical GPU-s/request under common correctness and joint SLO) and
G2 (scale-out/churn P95 user TTFT under a common GPU-time budget and joint SLO).
CE and idle-ready pricing remain secondary sensitivity quantities. The audit
also records that D210→D219 7B confirmed-snapshot Full reduced mean TTFT and
control-path time but increased P95 TTFT and changed the physical owner-union
GPU lifecycle only from 15,916.8239 to 15,911.7125 GPU-s (−0.032%). The
previous 4,317-GPU-s values were legacy primary-instance billing fields, not
the frozen G1 resource quantity. The next optimization target is therefore
readiness/first-service tail, not another unvalidated routing rewrite. Existing
3B D138 is a diagnostic, not a qualification point: its remaining mean TTFT is
dominated by observable dispatch/admission, runtime-slot, and LoRA-I/O terms.

No metric threshold, denominator, execution key, historical result, or paper
source was changed. PrimeLoRA 7B qualification remains the active gate; 3B and
external baselines stay paused. The corrected physical-lifecycle reconciliation
is documented in D240; deterministic JSON/hash/whitespace checks pass;
the fresh same-family D239 reviewer did not return within the audit window and
is recorded as `BLOCKED_REVIEWER_TIMEOUT`, not PASS.

## CURRENT — D238 retry4–6 expose natural proactive-coverage conditions; no admission result

2026-10-06（受控诊断审计）. Three fresh Llama-2-7B, seed-42, 22-request
prefixes completed with the native fixed-output contract and complete cleanup.
Retry4 ran the old post-drain observation and found seven HOST sources but no
GPU target selected by the live planner. Retry5 moved the diagnostic into
natural in-flight demand (`arrived=4`, `active=1`, `backlog=1`) and found no
native HOST source on the selected owner. Retry6 used a larger in-flight
window (`arrived=12`, `active=4`, `backlog=6`), found one HOST source, and
observed a frozen Remote→GPU target for a different adapter; the diagnostic
correctly stopped at `mixed_gpu_sources` rather than relabeling it as HOST→GPU.

All three report zero proactive admission decisions, zero admit/defer/reject
events, and `physical_kv_reservation=false`. Thus the retry sequence is a
coverage result, not proof of an ordinary request-path defect and not a
qualification of (E_i(t)). The run-level TTFT/E2E and GPU-s values remain
diagnostic only; no G1/G2, joint SLO, lifecycle ranking, or baseline comparison
is admissible. Full evidence, hashes, and the claim boundary are sealed in
`docs/ieee_tc/D238_RETRY456_DIAGNOSTIC_AUDIT.md` and its JSON sidecar. The
integrity audit is `docs/ieee_tc/D238_RETRY456_EXPERIMENT_AUDIT.md`; the fresh
same-family reviewer did not return within the audit window and is recorded as
BLOCKED, not PASS.

The next action is not another arbitrary short prefix. First pre-register a
legitimate later frozen-trace source window (or report that proactive coverage
is unavailable), then use paired Full/CapacityOnly frozen owner/epoch evidence
with typed decision, allocator/workspace, cancellation and release receipts.
The remote artifact services are stopped and no local GPU compute process
remains.

## CURRENT — D238 retry3 completes the bounded diagnostic but still does not exercise proactive admission

2026-10-06（受控诊断）. A fresh 22-request prefix was run with the protected
`diagnostic_prefix_v1` replay scope after fixing the witness UUID binding and
making the prefix explicit. All 22 requests completed with the native
fixed-output terminal contract; the service, replay, watchdog, lease cleanup,
and GPU-context release checks passed. The remote artifact unit was stopped
after the run and no local GPU compute process remained.

This is still a diagnostic-only result. The machine-readable diagnostic reports
`state=not_eligible` and
`reason=requires_one_initialized_ieee_runtime_and_profiles`; proactive GPU
admission decisions and physical KV reservations are both zero. Three natural
scale-up events occurred, but none had a planned adapter. Thus the run confirms
bounded request-path execution and a remaining proactive-coverage gap; it does
not qualify physical admission, ordinary demand-route correctness, G1/G2, SLO,
lifecycle ranking, or any baseline comparison. The observed 3,926.38-ms mean
TTFT and 10,375.84-ms mean E2E are retained only as startup/queue diagnostics.

The immutable result SHA is
`cefb72492f4c533b9c5cb813a63a20a817cc6ec2d8180507276b8a06091ff355`; the
launch receipt SHA is
`021b6d635259af906a4ca851c66231846026503b84d4ad1b0493294a6add0399`.
Full evidence and the claim boundary are sealed in
`docs/ieee_tc/D238_RETRY3_DIAGNOSTIC_AUDIT.md` and its JSON sidecar. The next
diagnostic must create an initialized runtime/profile and use a frozen-state
Full-versus-CapacityOnly counterfactual; sequential evaluation on one mutable
owner is not valid.

## CURRENT — D238 corrects the D227 admission interpretation

2026-10-06（语义复核）. A fresh read-only review of the IEEE PDF, the
formula-to-code record, and the D227 trace found that \(E_i(t)\) is a proactive
GPU-residency budget. The paper explicitly gives request-driven loading
priority over proactive preparation. D227's 22 request-driven source
resolutions (14 remote, 6 GPU, 2 HOST) submitted no proactive residency
candidate; its zero
admission decisions therefore show insufficient proactive coverage, not a
proven defect in ordinary demand loading. The raw D227 contract, cleanup,
pressure and non-ranking boundaries remain unchanged.

The current checkpoint is being kept within that boundary: when the explicit
gate is active it can emit a serialized core-to-worker observational capacity witness for a
request-driven load, using `capacity_only=true`; it leaves
`proactive_admission_evaluated=false` and `physical_kv_reservation=false`.
When the gate is absent, ordinary native demand loading is preserved. This
witness is diagnostic only and is not a formal physical-admission or G1/G2
qualification. The next bounded action, after code tests and a checkpoint, is a
proactive HOST→GPU decision diagnostic; the full replay remains held.

The detailed correction is `docs/ieee_tc/D238_ADMISSION_SEMANTIC_REASSESSMENT.md`
and its JSON sidecar.

## HISTORICAL — D237 D227 retry2 had no proactive-admission coverage

2026-10-06（受控诊断）. The original D227 launcher first stopped before model
construction because the guarded environment omitted the published-remote
delivery mode variable; that attempt is retained as a protocol/launcher
failure. After restoring the exact approved delivery environment, retry2
completed a bounded 22-request prefix with 22/22 native fixed-output terminal
contracts, return code 0, and confirmed GPU-context cleanup. It is still not a
performance or ranking result.

The important raw observation is that the YAML gate was `true`, but the native
request-driven route recorded zero proactive GPU-admission decisions and zero
physical KV reservations. Every inspected request reports
`physical_capacity_qualified=false`; the native pending journal explicitly
declares that it reserves no physical KV blocks. The original D237 wording
called this a request-path integration issue. D238 supersedes that causal
interpretation: the run exercised demand-driven loading but did not submit a
proactive HOST-to-GPU candidate, so it proves insufficient proactive coverage,
not a defect in request-priority demand loading.
The full replay remains held until a physical owner/epoch reservation
transaction, release semantics, and bounded pressure tests are implemented.

The retry's user TTFT is dominated by 134.4-s engine initialization and queue
wait (mean TTFT 101.27 s; mean service TTFT 0.758 s), so no latency, SLO,
CE, or lifecycle ranking is inferred. The physical owner-union deployment
journal reports 348.72 GPU-s; the separate 602.51 GPU-s billing field includes
a 300-s idle tail and is not the lifecycle metric. All leases were released.
The current Plan startup equation is `80+4+16+2=102 GiB`; older status prose
said 109.5 GiB as a stale byte-equivalent description. D227 used the current
Plan SHA and 102-GiB equation; future manifests must carry the reconciled
canonical value explicitly.

Full evidence and source-line analysis are sealed in
`docs/ieee_tc/D237_D227_RETRY2_ADMISSION_AUDIT.md` and its JSON sidecar.

## CURRENT — D236 independent PEFT reference passes; full Prime qualification remains open

2026-10-05（受控数值参考）. A single guarded 7B `backend-peft-reference`
run completed with `pass=true`, `measurement_complete=true`, and
`matched_reference_consistency_pass=true`. It evaluated the five pre-registered
roles (nonzero-A, zero, nonzero-B, base, and nonzero-A-repeat) against the
existing native receipt and the byte-verified frozen W0 trace. Prompt/tokenizer
identity matched; all three adapters passed exact tensor-value/dtype checks.
The independent full-vocabulary reference reports zero-vs-base difference 0,
nonzero-A and nonzero-B differences, and an identical repeated-A result.

This closes only the bounded numerical reference gate. The result explicitly
remains `semantic_full_pool_qualification=false` and
`production_launch_authorized=false`; it is not a 4,000-request correctness
denominator, SLO result, lifecycle-budget measurement, or ranking point. The
launch receipt and result hashes, plus the exact interpretation boundary, are
sealed in `docs/ieee_tc/D236_PEFT_REFERENCE_RESULT_AUDIT.md` and its JSON sidecar.
The service and auxiliary scopes were stopped, the GPU compute context was
released, and no remote service or replay was started.

The current Plan SHA `0c808509…` contains the explicit no-service startup
equation `80+4+16+2=102 GiB`, while older status prose still says 109.5 GiB.
D236 used the current Plan equation; this discrepancy must be reconciled and
recorded in the next run manifest before the D227 physical-admission or any
performance run.

## CURRENT — D235 confirmed scale-out first-service audit separates schema gap from real tail cause

2026-10-05（离线代码—结果核对）. The three D219 scale-up events were
triggered at visible request count 4 and took 44.91–46.57 s to activate. The
legacy `scaleup_first_service_*` counters are zero by construction in the
`ieee_confirmed` path: activation rejects legacy handoff plans and the event
registrar strips their fields, so those counters cannot be interpreted as
first-service readiness evidence. In contrast, all 3,003 requests served by
the new replicas carry `confirmed_dispatch_snapshot=true`; the first 20
requests per new replica contain only 10 GPU-ready cases out of 60 (16.7%),
with 23 HOST, 12 NVMe and 15 Remote cases. The earliest new-replica requests
were confirmed NVMe/Remote/NVMe and included a 11,288.8-ms TTFT.

The result is diagnostic only: it does not qualify G1/G2 and does not justify
reusing the legacy handoff. It does establish that the confirmed snapshot is
being captured at dispatch, while a newly published replica may still be
selected with a non-GPU source and perform demand loading. The next
first-principles optimization must therefore create a native, evidence-bound
ready-time preparation epoch or a measured admission objective that accounts
for the activation source state; it must not relabel a prediction as GPU-ready.
Details and exact hashes are in
`docs/ieee_tc/D235_SCALEUP_FIRST_SERVICE_AUDIT.md`.

## CURRENT — D234 old/new 7B Prime audit identifies a partial gain and an unresolved tail

2026-10-05（离线结果核对）. The existing 4,000-request ordinary Full (D210)
and confirmed-snapshot Full (D219) were compared field by field without
relabeling either as a formal G1/G2 point. D219 improves mean TTFT from
1650.5 ms to 1508.4 ms (−8.6%), mean E2E from 6034.1 ms to 5848.0 ms
(−3.1%), dispatch-admission wait from 938.4 ms to 814.1 ms (−13.2%), runtime
slot wait from 615.8 ms to 485.9 ms (−21.1%), and mean control-path time from
325.9 ms to 84.7 ms (−74.0%). P95 E2E improves 3.0%.

The same comparison is not uniformly better: P95 TTFT rises from 3415.8 ms to
3620.0 ms (+6.0%), legacy TTFT attainment falls by 0.10 percentage points,
GPU lifecycle time is unchanged (4317.01 vs 4317.05 GPU-s), and GPU cache-hit
rate is not higher. Thus the confirmed-snapshot path has a measured control
path benefit but has not yet demonstrated G1/G2 or a lifecycle reduction. The
100-request D217 retry10 prefix is not a regression counterexample: its cold
prefix cache hit rate is only 74% and its dispatch-window P95 is 10.67 s, so it
is a separate readiness/qualification diagnostic.

This evidence narrows the next first-principles optimization target to
scale-out/readiness tail behavior, not another blind routing rewrite. Numerical
correctness and physical admission evidence remain prerequisites; no code or
metric threshold is changed based on these descriptive differences.

Detailed values and calculation are in
`docs/ieee_tc/D234_7B_OLD_NEW_PRIME_AUDIT.md`.

## CURRENT — D233 remote boundary and independent artifact gate rechecked read-only

2026-10-05 20:09+08. A read-only recheck found the current direct experiment
route at `1000 Mb/s/full` on both the remote `eno1` and local `eno1np0`; both
interfaces are up. The remote 3B/7B artifact units are inactive and no
18080/18081 listener is present. No remote configuration, link setting,
service restart, cache mutation, or traffic-generation action was performed.

The historical D77 observation of remote `100 Mb/s/full` remains a separate
diagnostic. Its recorded peer advertisement was limited to 10/100; the exact
switch/port/cable cause was never established. It is not merged with the
current 1-Gbps state and cannot be used to claim that the machine-room link is
physically limited to 100 Mbps. The metric protocol therefore keeps
request-triggered transfer in user TTFT/E2E, while excluding one-time
prepublished packing, administration, scans, cleanup and unrelated remote
work. A 100-Mbps historical point, a current 1-Gbps point and application
no-delay sensitivity points must remain separately labelled.

The approved immutable gzip delivery caches are present on the remote node:
both manifests report `complete=true`, 500 artifacts, mode
`published_gzip_delivery_v1`, and `preparation_outside_inference=true`; total
archive bytes are 1,581,215,261 (about 1.47 GiB). Current remote free space is
143,522,885,632 bytes (133.67 GiB). The previously sealed independent bound
is 97,407,250,432 bytes (90.72 GiB), derived from the declared concurrent-pack
peak, 64-MiB log allowance, 16-GiB safety reserve and 1.5 margin. Thus the
artifact node passes its own measured gate despite being below the old 150-GiB
floor; the inference host's 150/100-GiB disk and memory/OOM gates are unchanged.

Evidence and calculation details are in
`docs/ieee_tc/D233_REMOTE_BOUNDARY_RECHECK.md`. This recheck is a prerequisite
confirmation only; it does not qualify Prime 7B or authorize a new GPU run.

## CURRENT — D232 independent PEFT-reference inputs are CPU-audited and ready

2026-10-05（本轮离线核查）. The existing completed native five-arm control
receipt `results/ieee_tc/p2_backend_qualification/model_20260926/llama2_7b_nativenumeric5_attempt1.json`
was checked as an input to the already implemented `backend-peft-reference`
gate; no model was loaded and no GPU or remote service was started. It is a
completed, exploratory observation with one same-prompt source request and the
roles nonzero-A, zero, nonzero-B, base, and nonzero-A-repeat. Its trace SHA is
identical to the current frozen W0 trace. All three referenced adapter files
and the local Llama-2-7B safetensors index exist and match the recorded bytes.

The old native control's identical generated token sequences are not treated as
correctness evidence. The independent reference must still recompute the
first-position logits with the original checkpoint and PEFT, then compare them
with the captured native first-token probabilities. A CPU-only input audit is
sealed in `docs/ieee_tc/D232_PEFT_REFERENCE_INPUT_AUDIT.md`. Once the unchanged
109.5-GiB memory gate passes, the next GPU action is this bounded reference
gate—not another 4,000-request replay or a baseline.

## CURRENT — D231 D217 retry10 bounded route qualification completed; formal gate still open

2026-10-05（本轮核查）. The retained retry9 receipt remains unambiguous: its
replay receipt and service ingress both declared the full 4,000-request source,
so it is a launcher/protocol failure with no performance denominator. The
retry10 directory also contained a later bounded run that had not yet been
recorded in this status file; it is now audited below rather than rerun.

The new launcher
`scripts/ieee_tc/run_d217_7b_confirmed_snapshot_cache_validation_retry10.sh`
sets `FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=100` before invoking the guarded
external-replay wrapper, passes the retry10 YAML explicitly, and preserves the
qualified NVML binding, allocator policy, W0 trace, fixed-output contract,
confirmed-snapshot route gate, and fresh output/NVMe/HOST roots. A CPU-only
static audit verified 100 requests, seed 42, the route gate, fixed generation
contract, qualification/external-replay flags, prefix value, and fresh paths;
`bash -n`, Python YAML validation, and `git diff --check` passed. No additional
GPU worker was started by this audit.

The bounded retry10 run itself completed 100/100 requests with zero failures,
100/100 fixed-output contract matches, 100/100 native token-timing records and
100/100 confirmed-dispatch snapshots. Its detailed audit is
`docs/ieee_tc/D231_D217_RETRY10_BOUNDARY_AUDIT.md`. It is still diagnostic-only:
the launch receipt records `production_launch_authorized=false`, every request
has `physical_capacity_qualified=false`, no GPU-admission decision was made,
and no independent numerical reference establishes `N_correct`. It therefore
cannot enter the common-warm SLO, Resident budget, G1/G2 or any ranking
denominator. The 109.5-GiB gate remains unchanged for the next authorized
diagnostic.

## CURRENT — D229 final remote metric and artifact-disk protocol frozen; 7B mainline still held

2026-10-05（本轮）. Added `docs/ieee_tc/METRIC_PROTOCOL_FINAL_V2.md`, a single
versioned calculation entry that composes the immutable V1 and Remote Scope V2
protocols. It freezes the HydraServe-aligned warm-reference 5x TTFT/2x TPOT
thresholds, joint correctness/SLO denominator, physical GPU lifecycle G1,
common-budget P95 TTFT G2, E2E decomposition, true-remote inclusion/exclusion,
CE sensitivity, paired-run statistics, and publication/manifest checks. HydraServe
primary sources were rechecked: its NSDI'26 paper reports warm-derived TTFT/TPOT
SLO attainment and GPU memory-time cost, not CE as a universal primary metric;
the official repository identifies its modified vLLM 0.4.2 implementation and
separate remote storage server. No old protocol or result was edited.

The remote artifact node was queried read-only after all services were stopped:
`eno1` reported 1000 Mb/s/full, auto-negotiation on, link detected, and zero
reported RX FCS/RX errors/TX errors/collisions. This is kept separate from the
historical D77 100-Mb/s negotiation state; the cause of that earlier state is not
claimed. Request-triggered remote read/send/receive/write/extract and contention
remain in the user timeline; one-time published-cache creation, administration,
hash scans, cleanup and request-external packing remain outside the request path
and are never subtracted from E2E. No remote service was started or modified.

The artifact-node gate is confirmed as an independent peak rule already present
in `scripts/ieee_tc_preflight.py`: `required = safety_reserve + ceil(1.5 *
(concurrent_pack_peak + log_growth))`, with explicit free-byte/inode and later
quota evidence. It deliberately does not call the inference-host
150/100-GiB `disk_required` rule; inference memory/OOM/disk gates are unchanged.
The five targeted artifact-gate CPU tests passed, as did Python compilation and
`git diff --check`; the unchanged
109.5-GiB physical-startup memory gate still forbids the next D225/D227 GPU
diagnostic. Prime 7B numerical correctness, common warm SLO, Resident budget and
G1/G2 remain open; no 3B or baseline run is authorized.

## CURRENT — D228 opt-in native token capture CPU-qualified; numerical GPU gate still held

2026-10-05 19:45+08. The runner now has a default-off
`ieee_token_id_capture` diagnostic flag.  When explicitly enabled it carries
the already observed native completion `token_ids` through the existing
`native_token_timing` evidence object, validates non-negative integer IDs, and
retains the compact SHA.  Ordinary Full results do not change schema or
behavior.  The new three-case unit test passes; the existing 43 native-timing
and EuroSys runner tests pass; Python compilation and diff checks pass.

This is only preparation for the D225 bounded numerical gate.  No D228 config,
GPU worker, remote artifact service, or token-capture result was started.  A
future fresh 22-request config must opt in explicitly, use distinct roots, and
compare sequences against an independent original-checkpoint PEFT reference.
Until the unchanged 109.5-GiB physical-startup gate passes, `N_correct` remains
unknown and no Prime replay, 3B run, baseline, SLO, or ranking is authorized.

## CURRENT — D227 admission-gate diagnostic configuration prepared; execution held

2026-10-05 19:35+08. A CPU-only audit created the new bounded 7B diagnostic
configuration `configs/ieee_tc/20261006_d227_7b_admission_gate_22req.yaml`.
It differs from immutable D219 only by an explicit
`effective_capacity_admission_enabled=true`, a 22-request bounded prefix,
fresh output/NVMe/HOST roots, and a diagnostic scenario identity.  The same
model, frozen trace/adapter content index, generation contract, vLLM backend,
profiles, resource envelope and remote delivery contract are retained.  YAML
parsing, recursive intended-difference comparison, scenario-map gate merge,
fresh-path absence and `git diff --check` all pass.  The configuration SHA is
`76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73`; D219's
reference SHA remains
`5072d681dc961a44b03ce9f3794b5c13d878f902c6f962cfba13999d5d3a0b63`.

No GPU worker, local result directory, or remote artifact service was started
for D227.  The unchanged physical-startup memory gate remains below threshold
(about 105.6 GiB available versus 109.5 GiB required), so the 22-request run
is not authorized yet.  The design note and status table are
`docs/ieee_tc/D227_ADMISSION_GATE_22_DIAGNOSTIC.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d227_admission_gate/d227_status.md`.
Even after execution, D227 is diagnostic-only and cannot by itself establish
numeric adapter correctness, common SLO, G1/G2, or a full-system ranking.

## CURRENT — post-audit safe hold; Prime 7B acceptance remains the main line

2026-10-05 18:40+08. The D222–D224 Resident audit and quarantine are complete
and backed up. No local GPU compute process, owned experiment scope, or remote
artifact service is active. The current host has about 105 GiB `MemAvailable`
against the unchanged 109.5 GiB physical-startup gate and about 163 GiB free
disk; no memory-cache purge, unrelated-process stop, threshold relaxation, or
GPU retry is authorized. This is a safety hold, not a completed experiment.

The Prime 7B line remains before 3B and external baselines. D219 is a complete
single-seed route/full diagnostic, but numerical adapter correctness, common
warm SLO, Resident budget, and G1/G2 remain open. The next admissible work is
bounded CPU/offline correctness and protocol preparation, followed by one
freshly gated 7B validation only when the unchanged resource gate passes.

## CURRENT — D225 output-difference audit completed; numerical gate remains open

2026-10-05 19:10+08. A read-only audit paired the immutable D210 and D219 7B
request records by all 4,000 request IDs. Adapter IDs, canonical prompt hashes,
input lengths, requested/actual output lengths, native token source and fixed
generation contract match for 4,000/4,000. Completion token-ID SHA differs for
136/4,000 (3.40%). This is an observation about two separate runtime attempts,
not evidence of an incorrect adapter: all 136 rows use a different activation
instance, while cache-tier and ready-state changes also occur among rows whose
hash is unchanged. The logs retain only token-ID digests, not the sequences or an
independent semantic reference, so `N_correct` remains unknown.

The audit and a falsifiable H0/H1 next gate are sealed in
`docs/ieee_tc/D225_OUTPUT_DIFF_AND_NUMERICAL_GATE.md` and the status table in
`paper_results/ieee_tc/p2_backend/20261005_d225_output_diff/`. The next GPU
action, only after the unchanged memory gate passes, is a bounded 22-request
diagnostic that reuses D166/D167/D207/D208 inputs and adds opt-in token/reference
capture; it is not a performance run. No full replay, 3B run or external baseline
is authorized by this audit. No local GPU process or remote artifact service was
started for D225.

## CURRENT — D226 found a disabled admission gate in D219

2026-10-05 19:25+08. A read-only source/config audit found that D219's
`tc_ieee_full` YAML omits `effective_capacity_admission_enabled`; the runner and
`ResourceCoordinator` both default the missing key to `false`. D219's result
metadata therefore records `false` and zero admission decisions/admit/defer/reject
events. This is a static configuration fact, not evidence that the workload had
no capacity pressure. D219 is retained as confirmed-snapshot route/full diagnostic
only, not promoted to a complete Full mechanism result or G1/G2 denominator.

The root-cause chain and the sole next variable (a new config with the gate
explicitly true, first checked by a bounded 22-request diagnostic) are sealed in
`docs/ieee_tc/D226_D219_ADMISSION_GATE_AUDIT.md` and its status table. The new
diagnostic remains gated by D225 numerical correctness preparation and the
unchanged memory/cleanup protocol; no GPU or remote service was started for D226.

## CURRENT — D222–D224 Resident-vLLM prototype RETIRED; no qualified Resident result

2026-10-05 18:25+08. A post-run protocol audit found that the D222 prototype
wrapper did not satisfy the frozen Resident reference contract.  The baseline
runner used the inherited vLLM 0.10.2 environment, not the Prime/vLLM 0.30.0
backend; its effective service cap was 96 GiB without a verified 72/80 GiB
`MemoryHigh`/`MemoryMax` contract; the deployment notice was emitted only after
all replicas became ready; replay used a later independent time origin; and
the receipt recorded an initial `nvidia-smi` inventory as “acquisition” and
closed intervals using only a final memory threshold, without a physical lease,
owner identity, or overlap-union proof.  Dynamic remote fetch/load was also
placed in the replay auxiliary domain.  These are protocol defects, not
performance observations.

The launch also selected `dynamic_remote`; the repository's ordinary
remote-fair vLLM reproduction rule requires round-local `static_remote` staging
unless a separately resource-bounded exception is qualified.  D222–D224 did
not qualify that exception, and their receipts do not capture a vLLM package or
source identity.

The prototype entrypoint and `VLLM_RESIDENT_PROTOCOL_V1=1` path are now
fail-closed in baseline commits `0a8e5f7` and `70bbbfc`.  Historical source and
raw logs remain untouched.  D222 is a
CPU/syntax and dry-run artifact only; D223 is a launcher/protocol failure; D224
completed 100/100 requests with the native fixed-output token contract, but
that contract does not prove numerical adapter correctness or physical
resource accounting.  D224 is therefore retained solely as a generation-
contract diagnostic, not a Resident baseline, lifecycle GPU-s/request sample,
joint-SLO result, G1/G2 input, or ranking point.  No 1,000-request follow-up is
authorized from this wrapper.

The next Resident reference, if needed, must reuse the existing qualified
common preflight, resource-domain, physical lease, common-clock, and cleanup
infrastructure.  The D222–D224 audit is recorded in
`docs/ieee_tc/D222_RESIDENT_PROTOCOL_WRAPPER.md`; the numeric diagnostic table
is `paper_results/ieee_tc/p2_backend_qualification/20261005_d222_d224_resident_audit/d224_diagnostic_table.md`.
No remote service or GPU is running after cleanup.

## CURRENT — D219 7B confirmed-snapshot full replay COMPLETE; formal acceptance remains open

2026-10-05 15:45+08. D219 completed the frozen 7B W0 exploratory full replay
with 4,000/4,000 terminal requests, 4,000/4,000 native fixed-output contract
matches, zero terminal failures, and four physical allocation journals closed
in the required acquire → worker_spawn → native_workers → native_workers_exited
→ release order. The remote artifact service was stopped after local cleanup;
the local GPU census is idle and no owned leases remain. The remote journal
contains 133 artifact-transfer events: 132 replay-linked records all use the
prepublished read-only gzip contract with `pack_performed=false`; one earlier
`finance_lora` preload/initialization event is retained and classified in
`d219_remote_unmatched.csv`, but is excluded from the replay denominator. No
request-time packing was present.

The confirmed-snapshot route evidence is complete at request level: dispatch
snapshots, dispatch tiers, owner/epoch checks, source-tier equality, prompt
hashes and timing-contract fields each cover 4,000/4,000 requests. The source
collection mode is `confirmed_snapshot_cache_v1` for all observed rows, with no
reported full fallback or identity-only rows. There are 252 requests with at
least one stale recheck (530 observed rechecks); these are selected-state
validation observations, not a causal performance estimate. Run metadata sets
`effective_capacity_admission_enabled=false`, and observed admission,
planning, promotion and hierarchy-transition counters are zero; D219 therefore
does not qualify the full physical-capacity/admission mechanism. The producer
also labels `readiness_routing_enabled=false` for the `ieee_confirmed` policy,
although 6,404 readiness-aware decisions are observed; this is recorded as a
metadata predicate/schema mismatch, not as evidence that confirmed routing was
disabled.

Descriptive values, using the frozen V1 Type-1 percentile recomputation, are
mean/P95 TTFT 1,508.362/3,619.799 ms, service TTFT 694.289/1,843.186 ms,
mean/P95 E2E 5,848.015/12,025.636 ms, mean TPOT 37.566 ms, throughput
1.00853 requests/s, and legacy SLO display 98.275%.
Observed physical GPU time is 15,911.713 GPU-s (3.977928 GPU-s per offered
request). Relative to the immediately preceding D210 descriptive run, mean
TTFT is lower by about 8.62% and mean E2E by about 3.08%, while P95 TTFT is
higher by about 5.98%; physical GPU time is effectively unchanged (about
0.00091 higher). Thus D219 is not an across-the-board improvement and does not
close the optimization gate.

The run is deliberately **not** a formal G1/G2 or cross-system result:
`n_correct` is unknown because the native token contract does not establish
numerical adapter correctness; common warm-derived thresholds and Resident
budget references are not yet frozen; and there is only one exploratory seed
with no paired held-out blocks. The 4,000-row timing audit has zero maximum
absolute error for timestamp algebra, stored TTFT/E2E/service fields, and
native TPOT recomputation; these checks validate timing-field consistency, not
numerical adapter correctness. The 98.275% value is retained as a descriptive
legacy field, not as the frozen
joint TTFT/TPOT SLO attainment. The full
curation, status table and diagnostic figure are in
`paper_results/ieee_tc/p2_backend/20261005_d219_full/` and
`figs/ieee_tc/p2_backend/20261005_d219_full/`; the curation correction is in
checkpoint `4e25e39`, pushed
to `faaslora_origin/retry14_continuous_queue_v2`.

**Next:** perform the remaining 7B numerical/output correctness audit and the
common warm/Resident reference qualification using the frozen protocol before
starting 3B or any external baseline. If those gates pass, use D219's frozen
candidate only as a development input for held-out paired blocks; if they do
not, keep Prime-only optimization focused on the measured tail/control-path
bottleneck. Do not claim a PrimeLoRA win or start the baseline matrix from
D219 alone.

Offline inventory D220 confirms that D166/D167, D207, D208 and the D63 native /
independent-PEFT controls already cover the four inspected 7B tensor classes;
repeating those single-prompt or coordinate checks would add no new hypothesis.
They still do not provide a request-bound semantic oracle for all 4,000 D219
requests, so `N_correct` remains unknown. D170 warm measurements remain
provisional and are not a frozen Resident budget. The next evidence must close
one of these two distinct gaps rather than relabel existing diagnostics.

## CURRENT — D217 retry8 launch correction prepared; retry7 evidence reclassified conservatively

2026-10-05 13:12+08. The retry7 log audit does not establish a distinct
post-initialization handoff deadlock. It establishes only that the dedicated
worker did not publish `ready.json` before the 600-s deadline: the checkpoint
load completed, the first CUDA-graph profiling pass and kernel warmup completed,
and the process was then terminated without an engine-ready receipt. The strict
allocation journal therefore remains correctly unqualified; no route,
latency, SLO, or GPU-s/request value is inferred. The earlier wording that
called this a proven handoff problem is superseded by this narrower statement.

The source/config audit found that retry7 used the default shared vLLM cache,
and its cold run populated a new AOT artifact (`b7a8b0...`) immediately before
termination. The known-good D214 launcher explicitly exported the cache roots;
the vLLM source and official compilation documentation describe the cache as a
reusable, configuration-keyed startup artifact. A fresh retry is therefore
justified only as a one-variable startup validation: same D217 route, trace,
generation, resource envelope, and 100-request prefix; fresh output/NVMe/HOST
roots; explicit reuse of the already populated cache; unchanged 600-s timeout.
It is not a performance run and cannot authorize a full replay by itself.

The retry8 configuration was corrected to the authoritative native-host source
SHA and committed/pushed as `42fb789`; no user files were staged. The remote 7B
artifact service was started and authenticated against the immutable published
cache. Local disk, memory, and GPU gates passed immediately before launch.

The first retry8 launch was rejected by the unchanged preflight before any
model process or GPU worker was started: `MemAvailable` was
107,886,551,040 bytes versus the required 109,521,666,048 bytes. This is a
`protocol_or_launcher_error`, not a system failure or performance observation.
Two transient auxiliary scopes created by the abandoned detached launch were
stopped by their exact unit identities; the remote 7B service was stopped and
the local GPU process census is empty. The failed receipt is retained under the
retry8 raw directory and is excluded from every denominator. No timeout,
cache, route, or generation conclusion is drawn from it.

The next cache-startup attempt (retry9) did reach native vLLM readiness and
reused a previously compiled AOT artifact, but its launcher omitted the
guarded `FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=100` setting. The external
publisher therefore began the full 4,000-request trace (`arrived=118/4000`)
instead of the intended bounded prefix. I stopped the exact owned
auxiliary/service scopes before treating it as a campaign; remote 7B was
stopped, GPU processes are absent, and the partial receipt/logs remain under
the retry9 raw directory. This is a launcher/protocol failure with no
performance denominator, not a PrimeLoRA result. The corrected retry will add
only the already-supported bounded-prefix environment variable; no
algorithmic, cache, or resource change is authorized by this correction.


## CURRENT — D217 retry7 reached native vLLM initialization but did not qualify worker readiness; route candidate remains pending

2026-10-05 12:56+08. A fresh D217 bounded qualification run was attempted with
the confirmed-snapshot route gate enabled, the unchanged W0 first-100-request
prefix, byte-verified shared trace/subset paths, the qualified NVML binding,
the frozen allocator policy, and `FAASLORA_WORKER_START_TIMEOUT_S=600`. The
remote 7B service authenticated successfully and delivered `finance_lora`
from the immutable `prepublished_gzip_v1` cache: 14,718,801 archive bytes,
`remote_pack_performed=false`, 470.49 ms end-to-end client transfer stage,
and 20,938,675 verified extracted bytes. This is valid delivery-path evidence,
not a request-performance sample.

The run reached an owned native vLLM EngineCore on GPU0. The cold 12.55-GiB
checkpoint load completed in 258.91 s (model load reported 300.36 s), followed
by torch.compile (82.69 s), CUDA graph capture, KV-cache construction and
kernel warmup. However, the dedicated worker never emitted its required
ready-file before the 600-s startup gate; no request was admitted and
`request_terminals.jsonl` is empty. The runner terminated with
`RuntimeError: native worker lifetime unqualified; physical lease retained`;
the launch receipt classifies this as `qualification_command_failure`, not an
algorithmic or latency result. The receipt SHA256 is
`5c858a9646b3398451a4a3a8e1e0db09a415ee5a714a9405cc9d7a89e47e231a`.

The strict physical journal records acquire → worker_spawn →
release_deferred(`native_worker_lifetime_unqualified`), so no GPU-s/request,
route hit rate, SLO, G1/G2 value, or cross-system denominator is inferred.
The watchdog observed the native context disappear and reported no OOM/high/max
event; the owned local auxiliary scope and remote artifact service were then
stopped by exact unit name. The run is retained under
`results/ieee_tc/p2_backend_qualification/d217_20261005/7b_confirmed_snapshot_cache_validation_retry7/`
and excluded from all rankings. This establishes that increasing the timeout
from 180 s to 600 s was necessary but not sufficient: the remaining issue is
the final worker-readiness/engine handoff after model initialization, not a
proven route or workload effect. Before another heavy run, audit that
readiness protocol and its prior successful D216 counterpart; do not simply
increase the timeout again or advance to 3B/baselines.

## CURRENT — D217 retry2/retry3 launch-contract failures retained; confirmed-snapshot GPU validation still pending

2026-10-05 12:15+08. Two fresh D217 retries were completed under distinct
campaign directories and are retained as protocol evidence only. Retry2 passed
the unchanged physical-memory preflight but stopped before model startup because
the launcher did not provide the required explicit NVML binding and binding SHA;
its receipt SHA256 is
`517ae75931f8917816d367bafc633dddfbca5d586f6d263c42d0b212ae864834`.
Retry3 supplied the qualified NVML binding and passed the same preflight, but the
runner stopped during initial deployment before a native worker/GPU context was
created. The frozen `uncached_background_v1` HOST allocator candidate conflicted
with the runner's inherited `expandable_segments:False` aliases; its receipt
SHA256 is
`cec1e5665f910e467891dba73404cef9532cdf1b6682d2acabb04128d89baa1d`.

Neither retry emitted request terminals, lifecycle allocations, route evidence,
or performance measurements. Both are therefore excluded from candidate
selection, G1/G2, figures, and all cross-system denominators. The service cgroup
reported no OOM/high/max event, all owned scopes were stopped by exact unit name,
the GPU terminal census confirmed no owned native context, and the remote
published-cache service was stopped after local cleanup. The authenticated
remote artifact path and immutable `prepublished_gzip_v1` cache were unchanged.

The allocator test selection passes 16/16. It establishes the next launch
contract without changing the algorithm: export the model's frozen
`FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1` before the
fresh runner process imports `run_all_experiments.py`, so the import-time legacy
aliases are not introduced; the child then derives the verified
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True` setting. A
fresh retry is authorized only with this environment, the already-qualified
NVML binding, unchanged D217 config semantics, new output/NVMe/HOST roots, and
the unchanged memory/disk gates. No 4,000-request replay, 3B run, or external
baseline is authorized until the 100-request route candidate produces native
GPU evidence and passes its pre-registered mechanism and control-path checks.

The immediately following retry4 also stopped before deployment and is retained
separately. Its receipt SHA256 is
`0f7bde9cc045042b873596f56bb87d95da14be2ce22cdfe7013a9916273a1b6c`. The
unchanged preflight passed, but the qualification supervisor did not receive the
NVML binding because the launch command used the non-existent variable
`FAASLORA_TC_NVML_BINDING_SHA256`; the code contract is
`FAASLORA_TC_NVML_SHA256`. No service log, request, model, or GPU evidence from
retry4 is admissible. This naming error is now corrected in the next launch
command; no runner or algorithm code is changed.

Retry5 then passed the NVML and allocator launch contracts and reached the
runner, but the supplied adapter-subset path was not present on this checkout.
The run therefore stopped during input resolution before model/GPU startup; its
receipt SHA256 is
`fe6cfa85ac21510170ee75e87b3d8c63b44b9ec3705cccf1c2a94853a3d4ffc7`.
The replay trace was otherwise the intended frozen 4,000-request trace. A
read-only content audit found the same trace bytes (SHA256
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`) and the
same adapter subset bytes (SHA256
`aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`) in the
retained `11_remote_fair_main_local_sim_v3/v4` shared-artifact directories.
Retry5 is excluded from all performance evidence; the next fresh retry uses
those existing byte-verified paths and new result/cache roots. No artifact is
regenerated or overwritten.

Retry6 reached the native deployment path with all input and launch contracts
valid. It verified one real remote `prepublished_gzip_v1` transfer
(`finance_lora`, 14,718,801 archive bytes, 547.7 ms client stage) and created a
GPU context of about 13.8 GiB on the owned RTX 3090. The vLLM cold checkpoint
load did not publish the worker-ready file within the existing 180-s
`FAASLORA_WORKER_START_TIMEOUT_S` window; the parent then retired the worker and
the strict physical journal correctly refused a release because no native-worker
confirmation had been recorded. The receipt SHA256 is
`b2a0cd61e5005538102dab1c38c126010c0d497441eff4f2257ce16dd3bd541c`.
There were zero request terminals, zero route observations and no performance
sample; retry6 is excluded from G1/G2 and all rankings. The remote service and
local scopes were stopped after the exact cleanup path.

This establishes a launch-time, not algorithmic, constraint: D216 succeeded
with a warm page cache, whereas retry6's cold 12.55-GiB checkpoint load exceeded
180 s. The next fresh retry keeps the same model, vLLM, route candidate,
workload, resource envelope, and remote cache, and exports
`FAASLORA_WORKER_START_TIMEOUT_S=600` so the pre-existing cold-start evidence can
complete. This is an explicit startup contract, not a performance tuning knob;
no request is admitted until the same native-worker confirmation, lifecycle,
generation and dispatch evidence gates pass.

## CURRENT — Remote-scope metric protocol V2 frozen; artifact-node disk gate separated from inference host

2026-10-05 11:55+08. A new protocol document
`docs/ieee_tc/METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` was added without
modifying the frozen V1 or reclassifying any historical run. Its SHA256 is
`485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`.
It freezes the distinction required for the real-remote experiment: request-
caused remote open/read/send/receive/write/extract, shared bandwidth, miss
count/bytes/concurrency and legitimate residency remain in the measured service
path; one-time published-cache construction, remote admin, hash scans, cleanup,
logs and request-external packing do not. Those excluded activities remain
audited and are never subtracted from E2E. All systems use the same immutable
published gzip content while retaining their legitimate cache/residency behavior.

The protocol makes the primary claim independent of CE: correct completion and
common warm-derived joint TTFT/TPOT SLO (HydraServe-aligned 5x/2x reference
倍率) precede G1 lifecycle physical GPU-s/request and G2 common-budget P95 TTFT.
CE, idle-ready factors and prices are supplementary sensitivity/Pareto evidence.
It records that the historical D77 100-Mbps observation was a separate peer-
advertisement state; the current read-only check on 174 found local/remote
1-Gbps full-duplex links and no NIC errors. No 10/100-Gbps or H100 claim is made.

The preflight code now exposes `artifact-disk-check`, a read-only gate that
computes the independent bound
`safety_reserve + ceil(1.5 * (concurrent_pack_peak + log_growth))` and checks
explicit artifact-node free bytes/inodes. It does not call the inference-host
150/100-GiB rule, start services, alter transfer concurrency, pack files or
delete data; inference-host memory/disk gates are unchanged. The pure arithmetic,
filesystem and CLI gate tests pass (5/5 targeted). A remote read-only census found
143,676,846,080 free bytes and 30,895,524 free inodes on `/home/lab14`; this is
reported as capacity evidence only, not a launch authorization because formal
delivery uses the already published cache and still requires explicit bound,
quota and service checks.

No D217 GPU validation was started: the unchanged inference startup gate still
has insufficient physical headroom. Prime-only optimization remains at the
confirmed-snapshot candidate decision; 3B and external baselines remain paused.

## CURRENT — D217 confirmed-snapshot route candidate CPU-qualified; GPU validation held by startup memory gate

2026-10-05 11:25+08. A new Prime-only 7B candidate was implemented behind the
disabled-by-default `ieee_route_confirmed_snapshot_cache` gate.  A scoped route
footprint is populated only by the authoritative native request endpoint.  On
later candidate views, non-selected replicas may reuse that immutable confirmed
observation; the selected source is still checked by its owner/epoch/source
transaction (or by one selected-replica identity read for a non-native source).
Any selected conflict invalidates only that `(replica, adapter-scope)` entry and
forces a fresh authoritative read.  This is not a TTL, polling, future-demand
oracle, or unconditional last-known tier decision.  IEEE equations, admission,
generation, workload, profiles and resource limits are unchanged.

The route-identity suite passes 14/14 under the qualified vLLM Python
environment, including authoritative-only cache creation, no duplicate full
read on reuse, missing device identity fail-closed behavior, epoch/source
conflict handling, membership rejection and Type-1 route-evidence checks.
Python compilation, YAML parsing and `git diff --check` also pass.
The broader request-footprint, service-routing, scheduler-observation and
request-lifecycle regression selection passes 292/292 in the bounded auxiliary
3/4-GiB scope.

One fresh D217 diagnostic configuration and distinct output/NVMe/HOST roots were
created.  The existing remote 7B artifact unit was started through its managed
systemd user service and authenticated health/manifest checks reported
`prepublished_gzip_v1` and the frozen 500-adapter set.  The first launcher
attempt stopped before execution because the new output parent did not exist;
the second stopped in preflight with `insufficient_physical_startup_headroom`
(`MemAvailable` about 106--108 GB versus the frozen 109.5 GB requirement).
The failed auxiliary scopes were stopped by their exact unit names.  No model,
GPU worker, replay request, or new result was produced; the D217 directory only
contains the preflight receipt.  A no-write page-cache release did not clear
the margin, so the experiment remains held rather than weakening the gate or
terminating unrelated services.
Because no local inference worker was admitted, the remote 7B artifact unit was
stopped again after the health check; its managed unit and published cache
remain unchanged for the next gated attempt.

**Next:** when the host passes the unchanged startup gate, rerun the same D217
command from its existing empty campaign directory.  Compare its bounded
control-path mean/P95 with D215/D216, audit selected-conflict invalidations and
all lifecycle/contract fields, then decide whether a 4,000-request replay is
permitted.  Do not start 3B or any external baseline before that decision.

## CURRENT — D216 retry7 bounded validation COMPLETE; mechanism active but candidate revoked for full-replay selection

2026-10-04 16:00+08. The fresh D216 retry7 was run only after stopping and
restarting the owned 3B/7B artifact services. The authenticated health check
passed after the service bind completed, and the exact gated replay then used
the immutable published 7B cache. This removes the retry5/retry6 launch-time
authentication mismatch; neither failed attempt is treated as algorithmic or
performance evidence.

The bounded W0 prefix completed 100/100 planned, arrived, submitted,
terminal, successful and native-contract-matched requests. All 100 actual
output lengths equaled their per-request targets, all prompt hashes and
dispatch-time readiness fields were present, and four physical allocation
journals closed in the required order with zero open leases. Launch, service,
replay and watchdog returned 0; native GPU contexts and the owned service path
were released. Observed lifecycle usage was 852.302629 GPU-s (8.523026 per
offered request). The fixed-output contract does not establish numerical
adapter correctness, so `n_correct` remains unknown and the attempt is not a
formal G1/G2 or cross-system result.

The D216 source-level predicate was active: the cumulative `cache_hits`
counter reached 30 in the final route collection, with zero identity failures.
The emitted counters are cumulative and are retained as mechanism evidence,
not converted to a request-level hit rate. The run contained 337 source
snapshots/106 collection IDs; aggregate fields were 3,509 cache hits, 1,386
identity-only observations, 1,136 full authoritative reads, 732 full
fallbacks and 404 cold full reads. These counts are not independent samples.

The primary control-path check is mixed. D216 mean/P95 control path was
238.515/692.622 ms. Relative to D215's 236.004/947.056 ms, the mean is 1.06%
higher while P95 is 26.84% lower. Relative to D214's 462.790-ms mean, D216 is
lower, but that comparison alone cannot satisfy the pre-registered candidate
gate. Because the primary mean did not decrease against the immediately prior
candidate and only one bounded block exists, **D216 is revoked for full-replay
selection**; no 4,000-request replay, 3B run or external baseline is
authorized. The lower P95 is retained as a diagnostic signal for the next
source/log audit, not as a selected advantage.

The 29 joined remote transfer records are all `sent` under
`prepublished_gzip_v1`, with `pack_performed=false`, no temporary archive and
50,650,865 server-written bytes. The result therefore has no retry7
authentication or request-time packaging failure. The full bounded curation
and status table are in
`paper_results/ieee_tc/p2_backend/20261004_d216_route_identity_scope_validation_retry7/`.

**Next:** preserve retry7 and its status table, audit why scope-local reuse did
not reduce the primary mean (collection-wave multiplicity and fallback
population first), and propose one new falsifiable candidate only after that
audit. D216 must not be blindly retried; 3B and external baselines remain
paused.

## CURRENT — D216 route-evidence audit corrected; candidate remains revoked

Backup checkpoint `32487b1` (pushed to `faaslora_origin/retry14_continuous_queue_v2`)
seals the retained retry7 curation, corrected route-evidence audit, wave-ID
provenance fix, and remote/metric decision document. The protected manifest,
old figures and unrelated user files were excluded.

2026-10-04 22:20+08. A bounded CPU-only audit of the retained D215/D216 request
evidence found that the producer-side collection counter and the emitted
collection label are not interchangeable. D216 reports 106 producer collections
after shutdown but only 83 labels in request evidence; D215 reports 107 and 87.
Seven D215 labels and nine D216 labels contain conflicting per-wave counters.
The old collector read the global collection counter after an asynchronous native
await, so a concurrent scope could advance it before the waiting wave emitted its
metadata. Snapshot sums and cumulative counter sums therefore cannot be presented
as RPC totals. Exact RPC count remains unknown.

The runtime fix is narrow evidence provenance: a wave captures its ID at creation
and all joined waiters return that ID. It does not change routing, cache
acceptance, admission, any IEEE equation, workload, SLO, or resource profile.
The route-identity suite plus request-footprint/service-routing/lifecycle/timing
tests passed 275/275 under the auxiliary 3/4-GiB scope, with no GPU or remote
service active.

The corrected Type-1 control P95 values are D215 945.799 ms and D216 690.333 ms;
the D216 mean remains 238.515 ms versus D215 236.004 ms (−1.064% lower-better
direction, i.e. a 1.064% regression). D216 has 337 observed source snapshots
versus 304, 76/100 requests with four source snapshots versus 65/100, and
scaleup_affected 51/100 versus 39/100. Fallback-conditioned control means are
306.126 ms for D216's 67 reported-fallback requests and 101.243 ms for its 33
without reported fallback; these groups are diagnostic, not causal contrasts.

The corrected audit is in
paper_results/ieee_tc/p2_backend/20261004_d216_route_evidence_audit_v2/ and
explicitly records the exact-RPC unknown status, Type-1 recomputation and
population mismatch. The status-table decision is unchanged: mechanism and
lifecycle gates pass, but D216 is revoked for full replay. No 4,000-request,
3B or baseline run is authorized until a new falsifiable candidate explains
fallback/membership cost without relying on future demand or test-point tuning.

## HISTORICAL — D216 scope-local identity candidate CPU QUALIFIED; bounded GPU validation pending

2026-10-04 14:25+08. The D215 source audit produced one falsifiable next
hypothesis: global owner epochs advance for unrelated adapter churn, while the
requested GPU copy can retain the same owner/source identity. D216 keeps a
fresh identity RPC but permits reuse only for a currently confirmed GPU target
whose adapter identity, path/rank, GPU slot, registration and cached scoped
footprint all match. Host-only targets, target-copy changes, unknown or
unconfirmed entries, identity failure and membership changes remain fail-closed
to the authoritative scoped read. The nine equations, admission/budget checks,
workload, generation contract, lifecycle accounting and resource envelope are
unchanged.

The route-identity, request-footprint, identity-recheck, service-routing,
scheduler-observation and request-lifecycle suites pass **298 tests** under the
bounded auxiliary resource domain. Two new tests specifically prove that an
unrelated source incarnation/epoch change can reuse the target GPU footprint,
while a target source-id change still forces a full read. This is semantic
qualification only; no GPU performance result exists yet.

The next authorized action is one fresh 100-request 7B W0 validation using the
exact D215 trace, adapter subset, published delivery cache, D157 profiles and
resource envelope. It must show a nonzero `scope_local_hits` path, reduced
control-path cost, and no lifecycle/coverage regression before a 4,000-request
replay can be considered. 3B and external baselines remain paused.

## CURRENT — D215 REVOKED after bounded route-identity validation; Prime gate remains open

2026-10-04 14:05+08. The D215 cold-scope candidate completed its authorized
100-request 7B diagnostic, but it did not establish the intended low-cost
identity-recheck path. The run is retained as mechanism evidence and is
excluded from every G1/G2, joint-SLO, resource-saving, ranking, and baseline
denominator.

The strict execution contract passed: 100/100 terminal records, 100/100
successful requests, 100/100 fixed-output/native contract matches, four exact
physical allocation journals, zero open leases, and 779.891374 GPU-s observed.
The run was nevertheless not a formal comparison: the token contract does not
prove numerical adapter correctness, and the legacy 5,000-ms display is a
TTFT-only diagnostic rather than the frozen joint TTFT/TPOT SLO.

The decisive mechanism audit is in
`paper_results/ieee_tc/p2_backend/20261004_d215_route_identity_cold_validation/`.
The coordination record reports 107 collections, 7 stale rejections and 1
joined collection. Request evidence contains 87 unique collection IDs; 37
include cold full reads (88 replica reads) and 60 include authoritative full
fallbacks (142 replica reads). The emitted `cache_hits` and `identity_only`
fields are cumulative counters, so they are not converted into a reuse rate.
Repeated authoritative fallback on scopes that had already been cached means
the proposed path was not dominant enough to justify a 4,000-request replay.

Descriptive diagnostic values were routing/control mean/P95 236.004/947.056
ms, dispatch/admission mean/P95 1,719.253/10,809.149 ms, E2E TTFT mean/P95
2,372.378/11,132.852 ms, service TTFT mean/P95 653.125/1,805.050 ms, E2E
mean/P95 8,622.024/19,239.451 ms, and legacy TTFT-only attainment 90%.
These values are diagnostic only and are not compared as accepted performance
improvements.

**Decision:** revoke D215; do not run its 4,000-request replay, 3B, or any
external baseline. The next step is a source-level audit of why the cached
footprint is rejected after a fresh identity observation, followed by one new
CPU-qualified hypothesis. No stacked optimizer or blind retry is authorized.

## HISTORICAL — D215 cold-scope candidate CPU QUALIFIED; superseded by the terminal validation block above

2026-10-04 13:25+08. D214's stage audit identified a specific control-path
candidate rather than a generic latency tuning opportunity: on a cold
`(replica, requested-adapter-scope)`, the identity-recheck implementation could
perform a compact identity RPC and then an authoritative scoped RPC, although
the compact endpoint does not carry the scoped footprint required for routing.
D215 changes only this cold path. It performs one authoritative scoped read on a
cold scope, publishes that measured footprint, and starts fresh identity
rechecks only after the exact scope is cached. Identity mismatch, malformed
coverage, epoch change, or RPC failure still fails closed to a per-replica
authoritative read. No last-known tier, TTL, admission/profile/formula,
workload, generation, or resource-envelope change is included.

The candidate passed 296 related CPU tests (route identity, request footprints,
identity recheck, service routing, scheduler observation and request lifecycle)
under the bounded auxiliary scope. Python compilation and `git diff --check`
also pass. This is semantic qualification only; it is not evidence of GPU
performance improvement.

The fresh bounded validation configuration is
`configs/ieee_tc/20261004_d215_7b_route_identity_cold_validation.yaml`. It
reuses the exact D214 W0 prefix, trace and adapter-subset identity, D157
profiles, fixed-output contract, published read-only artifact delivery, and
resource envelope, with fresh D215 output/NVMe/HOST/receipt roots. The next
authorized action is one 100-request 7B validation. It must pass the existing
request, native timing, physical allocation/release and transfer audits before
its routing/control, dispatch, service-TTFT, E2E and GPU-s observations are
compared with D210/D214. A full 4,000-request replay is allowed only if the
bounded run reduces the predicted control-path cost without a new correctness,
lifecycle or resource failure. 3B and external baselines remain paused.

## CURRENT — D214 retry4 COMPLETE; lifecycle-qualified diagnostic, formal G1/G2 still open

2026-10-04 12:45+08. The D214 route-identity recheck full replay completed and
was closed before any new task was started. The launch, replay and watchdog
returned 0; native GPU-context release was confirmed, the owned service path
was removed, local `nvidia-smi` was empty, and the remote artifact service was
stopped only after local release. The protected artifact credential was then
rotated on both endpoints without restarting the service; the token value is
not stored in the repository or status log.

The strict lifecycle/request audit has 4,000/4,000 terminal records,
4,000/4,000 successful requests, 4,000/4,000 native fixed-output contract
matches, 4,000 unique request IDs, zero terminal errors, and no missing
canonical-prompt/source/token-ID/clock/instance fields. The target is
request-specific (`min(source_expected,256)`), so 3,055 requests have a target
other than 256; the run must not be described as a uniform 256-token replay.
All four physical allocation journals have exactly
`acquire → worker_spawn → native_workers → native_workers_exited → release`,
with zero open leases. Observed lifecycle usage is **15,907.670119 GPU-s**
(3.976917530 GPU-s per offered request).

The immutable delivery evidence contains 132 retry4 transfer records. The
copied remote journal contains 266 lines: 132 join one-to-one with retry4
outcome IDs, while 134 retained out-of-run records are excluded from retry4
counts and bytes. Every joined record reports `prepublished_gzip_v1`,
`pack_performed=false`, and no temporary archive. The main outcome records
`remote_pack_ms=0` and
`remote_pack_duration_basis=absent_by_published_contract`; request-time
packaging therefore did not enter this replay. The joined server-written bytes
are 131,480,060. The raw transfer and lifecycle sources remain unchanged.

Descriptive end-of-run values are E2E TTFT mean/P95/P99
1,953/4,233/6,262 ms, service TTFT 686/1,875/3,868 ms, TPOT 37.2 ms, and E2E
6,285/12,522/14,907 ms. The log's 97% `slo@5000ms` is a legacy TTFT-only
display, not the frozen joint TTFT/TPOT SLO. Relative to D210's descriptive
1,650.511 ms mean and 3,415.442 ms P95 TTFT, retry4 is not an accepted latency
improvement; its lifecycle value is only a diagnostic observation and is not a
formal saving claim. The identity-recheck mechanism was active (5,160 source
collections, 492 joins, 1,602 stale rejections), but these counters do not
establish causal G1/G2 benefit.

Compact, bounded curation (3/4 GiB auxiliary scope) is stored at
`paper_results/ieee_tc/p2_backend/20261004_d214_route_identity_full_retry4/`;
the JSON/CSV/status files retain source hashes and the raw evidence remains
immutable. `formal_comparison_qualified=false` remains intentional:
`n_correct` is not inferred from the native token contract, and common warm
SLO/Resident reference values are not frozen. No figure or ranking is generated
for this diagnostic; the status table is the appropriate paper artifact.

The curated JSON SHA256 is
`69b7865e299d02e9c185d77f0a175be0258702d5b1dfaaa75972bf0208ae5ccf`; the
metrics CSV SHA256 is
`f9422421b45c69b4e749b0ef4618da8f3ed0fc3980b3c01d497fd5c46100f685`.
The D210/D214 stage comparison table and interpretation are in
`d214_vs_d210_stage_comparison.{csv,md}`. It shows routing/control mean rising
from 325.863 ms to 462.790 ms while service TTFT falls from 712.107 ms to
686.015 ms; the added control path dominates the user-visible regression.

**Decision:** D214 passes lifecycle and serving-contract integrity, but does not
close the 7B Prime optimization gate. Before 3B or any external baseline, the
next action is a single new, evidence-backed 7B bottleneck hypothesis (after
the required old/new comparison and numerical-correctness/SLO audits), followed
by a fresh validation/full replay only if that hypothesis predicts improvement.
No blind retry or stacked optimizer is authorized.

## HISTORICAL — D214 retry4 LIVE; superseded by the terminal block above

2026-10-04 11:36+08 continuation: previous turn PROGRESS/VERIFIED WAIT,
current turn VERIFIED WAIT on the same live tool session `46064`. Independent
PID and systemd checks confirm service invocation unchanged and active; the
replay has not terminated. Bounded terminal audit now reads 1,279 terminal /
1,279 success / 1,279 native-contract matches, zero duplicate request IDs and
no terminal error types. This is a partial planned-4,000 population, not
numerical-adapter qualification, formal SLO or G1/G2 acceptance. Latest watchdog
sample 1401: service 18,072,084,480 bytes, host available 97,054,093,312 bytes,
free disk 177,581,068,288 bytes, service swap/events zero, no warning/abort,
foreign compute or escaped worker. Plan/V1 SHA unchanged. No runtime edits,
new experiment, heavy analysis, remote configuration or cleanup occurred.
Continue THIS run to terminal; do not restart on an observation timeout. The
credential-rotation follow-up was pending at that time and is now complete.

2026-10-04 11:16+08. Goal ACTIVE/current turn VERIFIED WAIT. User's once-only
read-only compressed delivery-cache authorization remains fulfilled by D78/D80;
the immutable published cache is reused, not regenerated. The full Plan, status
and V1 protocol were reread in this continuation. No baseline/3B is running.

Lifecycle fix checkpoint `28b810d` and the fresh retry4 configuration `c40a823`
were pushed to `faaslora_origin/retry14_continuous_queue_v2`; unrelated user
changes and `configs/generated/lora_manifest_1000.json` were not staged.
The same W0 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`
and subset SHA `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`
are reused; only output/NVMe/HOST/receipt roots changed from retry3.

The protected replay is LIVE in tool session `46064`, service scope
`primelora-tc-svc-8fc8702482d74551bf58db5771ff9db1.scope`, invocation
`98a92d8e38b942408d811b5c299d4612`, and auxiliary scope
`primelora-tc-aux-e1e337b26fca4c59ae2bc3acc4d4fef4.scope`. Raw receipt root:
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_full_retry4/`.
At watchdog sample 242, 93 terminal records existed (partial, not acceptance).
Four native GPU PIDs 2792275/2796895/2797323/2797552 were service-owned with
the frozen CPU affinity. Service memory was 16,560,996,352 bytes, host available
97,482,391,552 bytes, service swap and memory events all zero; no warning,
abort, foreign compute or escaped worker. Free disk was 178,698,461,184 bytes.

Do NOT start another replay, change runtime source/config, or alter remote
delivery while this run is live. Next: monitor THIS session to terminal,
verify local physical release, then exact-owned remote stop/copy, strict
bounded analysis, status table/figure as appropriate, and backup. Formal
numeric correctness/common-warm/Resident reference/G1G2 remain open; a raw
development 5-second attainment value is not the frozen joint SLO.

Historical security note (resolved in the terminal block above): an authenticated pre-run curl diagnostic mistakenly echoed
the artifact access token into tool output. Its value is not stored here or in
Git. After the live replay ends and remote evidence is retained, rotate the
local/remote protected token together and verify authenticated access before
the next run. Do not change the credential mid-measurement. Avoid shell tracing,
verbose authenticated curl, or credential-bearing process-argument output.

## CURRENT — D214 7B full replay retry3 completed requests but failed strict physical-owner journal audit

2026-10-04 11:05+08. Goal ACTIVE/current turn PROGRESS. The cached-startup
retry3 completed the entire PrimeLoRA-only W0 replay: 4,000/4,000 requests
arrived, submitted, terminated and succeeded under the frozen
`fixed_length_greedy_v1` contract, with native token IDs and per-request targets
capped at 256 (not a uniform 256-token workload).
The replay process returned 0 and all owned GPU contexts were released. The
raw result and launch evidence remain immutable under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_outputs_full_retry3/`
and `.../7b_route_identity_full_retry3/`.

The run is nevertheless **not formally qualified** and is excluded from G1,
G2, SLO, resource and ranking tables. The service returned 1 during final
physical-deployment reduction because one of the four allocation journals
contained a second `native_workers_exited` event after its `release` event.
The other journals had the required acquire → worker_spawn → native_workers →
native_workers_exited → release order. This is a concurrent retirement
accounting race (two shutdown paths waited on the same native worker), not an
OOM, model, workload, remote-delivery or algorithmic failure. The reducer
correctly failed closed rather than converting an ambiguous lifecycle into a
GPU-s/request value. The run recorded no cgroup max/oom/oom_kill event and
preserved its incomplete qualification receipt.

The fix is limited to lifecycle evidence: `wait_workers()` now serializes
concurrent waiters and records `native_workers_exited` exactly once; it does
not weaken release checks or alter any PrimeLoRA routing/admission/profile
logic. The new regression test and the lifecycle/native-retirement/launch
suite passed (178 tests, 0 failures). A fresh retry4 will reuse only the
already-populated compile cache and the exact trace, adapter subset, backend,
resource envelope and D214 feature gate, while using new output/NVMe/HOST and
receipt roots. No baseline or figure is authorized until retry4 passes the
strict request, native-generation, dispatch/readiness, journal-order and
cleanup gates.

## CURRENT — D214 7B full replay retry3 launch evidence retained; retry4 pending

The retry3 launch receipt is retained with `formal_comparison_qualified=false`
and `pass=false` solely because of the physical-owner journal-order reducer
failure described above; this does not invalidate the 4,000 completed request
records as raw diagnostics. The four journal IDs and the unmodified result
files are kept for audit. No result directory is overwritten, and no
performance interpretation has been generated.

## CURRENT — D214 7B full replay retry1 rejected before model initialization; launcher-only retry authorized

2026-10-04 09:24+08. Goal ACTIVE/current turn PROGRESS. The first 4,000-request
D214 full replay was retained as an incomplete launch attempt and is excluded
from every performance, SLO, resource, and ranking table. Its receipt is
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_full_retry1/launch.json`:
the service and replay returned 1, while native GPU-context release was
confirmed and the owned cleanup path was already gone. No model worker reached
inference and no request entered the algorithmic result denominator.

The fail-closed launcher rejected the run during the initial background
scale-up because the parent environment did not carry the frozen D157 native
HOST allocator contract (`uncached_background_v1` with
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True`). This is a
protocol/launcher error, not evidence against D214, the published remote
delivery, or PrimeLoRA latency. The remote 7B artifact unit was stopped after
the attempt and read back inactive; local `nvidia-smi` showed only the normal
driver baseline (15 MiB per GPU), with no owned worker processes.

The retry preserves the exact model, vLLM backend, D157 service/preparation
profiles, W0 trace, 500-adapter subset, fixed-output contract, resource
envelope, and D214 feature gate. It changes only the fresh output/NVMe/HOST
roots and supplies the already-qualified allocator/PATH environment before
the protected user-scope launch. The failed retry1 directories remain
immutable; no old result is overwritten. The full replay remains a Prime-only
candidate check; baseline comparison and formal G1/G2 claims remain blocked
until a complete replay passes the strict correctness, release, and evidence
gates.

The immediately following retry2 launch also exited before the guarded
service scope was created: the preflight attempted to stat the configured
receipt parent `.../d214_20261004/7b_route_identity_full_retry2/`, which had
not yet been created. No model, cache, GPU context, request, or remote
artifact transfer was started. This second result is therefore another
`protocol_or_launcher_error`, retained in the execution log but excluded from
all denominators. The retry remains valid with the same fresh cache/workspace
roots; only the empty receipt-parent directory is created before the next
invocation.

That guarded retry then reached model initialization but failed before any
worker-ready event. The worker log records weight loading, torch.compile,
FlashAttention, CUDA-graph capture and KV-cache setup; no native request
terminal was produced. The parent reached its existing 180-second startup
deadline and teardown retained the physical lease because no native worker
confirmation/exit evidence had been recorded. The launch receipt reports
`service_returncode=-15`, `replay_returncode=1`, native GPU-context release
confirmed, and zero cgroup `max`, `oom`, or `oom_kill` events. The physical
deployment summary therefore remains incomplete with
`gpu_seconds_observed=182.01148`, `n_terminal=0`, `n_correct=null`, and an
open lease journal retained as failure evidence; it is not a resource or
performance sample.

The only falsifiable working explanation is initialization variance around
the first build of the fresh vLLM/Inductor cache: D214 bounded retry7 recorded
an engine activation of about 155.4 seconds under the same frozen profile,
whereas this fresh-cache attempt did not publish ready before 180 seconds.
This is not yet established as the cause. The next attempt reuses the
now-populated read-only compile-cache contents, changes only fresh output,
NVMe, HOST and receipt roots, and keeps the timeout and all algorithmic
settings unchanged. If cached initialization still misses the deadline, the
startup qualification path will be audited before another replay.

## CURRENT — D214 7B route identity-recheck bounded validation PASS; full replay decision pending

2026-10-04 09:10+08. Goal ACTIVE/current turn PROGRESS. The seventh D214
bounded validation is the first attempt that reached the native 7B worker under
the frozen D157 service profile. It used the existing W0 trace prefix (100 of
4,000), the existing 500-adapter subset, `fixed_length_greedy_v1`, the
published remote delivery contract, and only the D214
`resource_coordination.ieee_route_identity_recheck_cache` gate. The venv
`PATH` was made explicit so the already-qualified FlashInfer `ninja` tool was
visible; no algorithmic or workload parameter was changed.

All 100 planned requests arrived, submitted, terminated and passed the native
output contract. The strict counts are 100/100 target-length matches,
100/100 native `vllm_token_ids`, prompt hashes, `ieee_tc_native_v1` timing,
native terminal observations, confirmed dispatch snapshots, and complete
pre-dispatch readiness fields. The route evidence contains 291 per-replica
source observations, all in `identity_recheck_cache_v1` mode. The final
monotone mechanism counters recovered from concurrent evidence are 54 cached
full-state reuses and 198 identity-only observations, with 0 identity
failures and 0 membership rejections. The per-observation
`full_fallback_replica_count` field was observed at most 4 concurrently, but it
is an instantaneous snapshot and must not be interpreted as a total number of
authoritative fallbacks. These are mechanism counters, not request-level
denominators; the auditable extraction and caveat are in
`paper_results/ieee_tc/p2_backend_qualification/d214_20261004_route_identity_validation/d214_7b_retry7_curated.json`.

The launch receipt passed, native GPU-context release was confirmed, the owned
service path was removed, local `nvidia-smi` was empty after completion, and
the D214-owned remote artifact units were explicitly stopped and read back as
inactive. Raw result SHA256 is
`1163f60f4092a2d2ea907630d80057da184cadb15ceb65d0a8f6b772a6269a9e`, and the
launch receipt SHA256 is
`b1bb5ad48ffd896f7d9484b6c7997d4e342e0dd2517f4e9e8e7a268826edfd49`.

The diagnostic had mean overall TTFT 65,663.985 ms and zero 5-second SLO
attainment because the 100-request prefix overlapped model startup; these
values are explicitly non-formal and are not a performance claim. Service-path
TTFT was 593.653 ms mean / 1,471.231 ms P95, while the measured control path
was 138,920.583 us mean / 348,461.808 us P95. A complete 4,000-request replay
is now scientifically justified as a candidate check of the control-path
hypothesis, but it is not yet started: it must use fresh output/cache roots,
the same frozen inputs and environment, and the full replay acceptance gates.
No baseline, 3B, comparison, or ranking is authorized before that decision is
audited. The bounded run remains diagnostic and does not establish G1/G2.

## CURRENT — D214 route identity-recheck candidate CPU QUALIFIED; no GPU replay yet

2026-10-04 08:35+08. Goal ACTIVE/current turn PROGRESS. After the D213
protocol audit, the next PrimeLoRA-only candidate is a feature-gated route
observation optimization. D210 request projection identified the remaining
control-path cost: each confirmed route performed a full scoped native
footprint read even when the owner/source epoch was unchanged. D214 keeps a
fresh identity-only owner read for every replica and reuses a prior measured
scoped footprint only on exact owner/epoch/clock/source/slot identity match;
identity failure, unknown/unconfirmed native state, epoch change or incomplete
footprint falls back to the original full endpoint. No IEEE equation, profile,
admission, lease or SLO definition changed, and the gate defaults OFF.

The implementation is in `scripts/run_all_experiments.py`, documented in
`docs/ieee_tc/D214_ROUTE_IDENTITY_RECHECK_CACHE.md`, and tested by the new
CPU-only `tests/test_ieee_tc_route_identity_cache.py`. The affected request
footprint/lifecycle suites plus the two D214 tests ran in the declared
auxiliary CPU domain: 199 tests passed, 0 failed, 0 swap/OOM evidence. This
qualifies only state-machine correctness; it does not claim a latency gain.

The bounded candidate configuration is now
`configs/ieee_tc/20261004_d214_7b_route_identity_validation.yaml`; it reuses the
D210 7B W0 trace, D157 profiles and published remote delivery, changes only
the feature gate plus new output/cache paths, and leaves the user manifest
untouched. Next is one 100-request W0 validation replay with strict cleanup/
dispatch evidence, followed by a decision from the measured control-path and
TTFT change about whether one complete replay is scientifically justified. No
new baseline, 3B, comparison, or long vLLM run is authorized before this Prime
candidate gate is resolved.

The first launch attempt reached the runner but was rejected before model
initialization because the controller-only gate had been placed in `model`,
changing the frozen service-profile identity. It produced no GPU inference and
is retained as a failed launch diagnostic under the first D214 validation
directory. The gate is now stored under `resource_coordination`, so the model
profile identity remains exactly D157; the retry must use a new output
directory and must not overwrite that failed attempt.

The second bounded attempt is also retained without scientific interpretation.
It passed the external-replay preflight and loaded the shared W0 trace prefix
(100 of 4,000 requests), but the runner stopped before model/GPU initialization
because the configuration still referenced the first attempt's already-created
NVMe and `/dev/shm` workspace roots. The runner's fail-closed contract correctly
raised `FileExistsError: IEEE replay requires fresh per-run workspace roots`;
`N_arrived=N_submitted=0` and no GPU inference occurred. Its launch receipt and
service evidence are under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry2/`.
This is a launch hygiene failure, not evidence against the identity-recheck
candidate. A third bounded validation must use a new output directory and new
workspace roots, and must leave both earlier attempts immutable.

The third bounded attempt also stopped before model/GPU initialization. The
resource preflight passed and the external replay published the same 100-request
diagnostic prefix, but the guarded launcher rejected the attempt because the
required SHA-locked NVML binding variables were not supplied. It therefore
reported `explicit NVML binding and SHA required before native launch`, released
the service scope, and left `N_arrived=N_submitted=0`; there is no algorithmic or
performance interpretation. The retained receipt is under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry3/`.
The next attempt adds only the already-qualified binding
`/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py`
with SHA256
`4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea`, plus
fresh cache roots; the feature gate and workload remain unchanged.

The fourth bounded attempt passed the NVML/resource gates and reached the
runner, but stopped during remote-storage setup because the command omitted the
already frozen 500-adapter subset path. The fail-closed error was
`IEEE input setup requires nonempty unique adapter IDs`; no cache root was
created, no model was initialized, and the watchdog confirmed native GPU
contexts were released. Its complete receipt is retained at
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry4/`.
This is a command/provenance error, not a route-candidate result. The next
attempt supplies the existing subset JSON paired with the existing W0 trace;
it does not regenerate or alter the adapter pool.

The fifth bounded attempt supplied both frozen inputs and reached the full-stack
preload path. It then failed before model initialization because the parent
process had imported the runner's legacy allocator default
(`expandable_segments:False`) while the frozen D157 profile requires the
explicit `uncached_background_v1` native HOST allocator. The runner correctly
rejected this inherited conflict (`native HOST allocator candidate conflicts
with inherited allocator tuning`). No request was served; the service scope and
native GPU context were released. The evidence remains at
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry5/`.
The next attempt will launch the same profile with its already-specified
allocator policy preconfigured in the parent environment, not by changing the
profile or candidate algorithm.

The sixth bounded attempt reached the real 7B worker and began GPU model
initialization, but FlashInfer's first sampling-kernel JIT failed with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`. The qualified
venv contains the `ninja` executable, but its `bin` directory was not present in
the inherited worker `PATH`; this is an environment propagation failure. The
external publisher observed 13 arrivals before the service connection closed,
with no terminal requests, so the attempt is incomplete and cannot support a
performance or route-semantic claim. The worker, service scope and GPU context
were released; raw worker logs are retained under the retry6 launch evidence.
The next attempt adds the venv `bin` directory to `PATH` only, preserving the
same code, allocator policy, trace, adapter subset and fresh workspace rule.

## CURRENT — D213 Resident repeats COMPLETE as serving diagnostics; V1 reference REJECTED by protocol audit

2026-10-04 08:20+08. Goal ACTIVE/current turn PROGRESS. Before this
classification the full Plan, this status file and METRIC_PROTOCOL_FROZEN_V1
were reread; `analyze-results`, `run-experiment`, `academic-plotting` and
`github-sync` instructions were reread as applicable. D213 repeat3 completed
4000/4000 successful requests, so all three retained repeats pass the fixed
output/native-token serving gates. Repeat3 descriptive values are mean TTFT
230.662 ms, Type-1 P95 TTFT 275.283 ms, mean E2E 3089.306 ms, Type-1 P95 E2E
6969.348 ms and mean TPOT 25.271 ms. GPU processes and the owned unit are
stopped; post-run `nvidia-smi` is empty. The new repeat3 compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat3_seed42_fixed.json`.

The bounded CPU analysis was run in the auxiliary 3/4 GiB, CPU 2,3,26,27
resource domain. It produced the repeat table and the fail-closed audit at
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_resident_reference_audit_v2/`;
audit SHA `8d6948a0cedfbdf648a572c67e08eec142c43c3182d60e7e50c5abd71171de72`.
The three replays are **not** eligible to define V1 `U_ref` or the G2 budget:
they used a local static adapter pool rather than the common published remote
delivery, launched four replicas sequentially and started replay only after
all were ready rather than using the common `t=-60s` notice, placed playback
and service in the same resource domain, and recorded only a whole-unit
envelope—not per-GPU allocation/release integrals. The baseline summary also
uses a synthetic startup-plus-300-second-tail lifecycle model, and the runner
used a 7200-second timeout rather than V1's 1800-second qualification guard.

This is a protocol/launcher mismatch discovered in our own reference path,
not a vLLM performance failure and not a reason to hide the successful
serving evidence. The prior repeat1/2 compact receipts remain immutable
historical files; the new audit manifest supersedes only their earlier
“eligible reference candidate” interpretation. No fourth long replay will be
started under the same invalid contract. No figure or PrimeLoRA ranking is
generated; a status table is the scientifically appropriate deliverable.

Next: first construct and CPU-audit a protocol-compliant ordinary-vLLM
reference launcher with common notice, published remote delivery, separated
playback/monitor resources, 1800-second protection and per-GPU lifecycle
events. Only after that gate passes may the three-repeat Resident reference
be rerun. PrimeLoRA 7B formal acceptance, 3B, baselines, M1/M2, ablations and
sensitivity experiments remain open; no configuration or SLO was changed.

## CURRENT — D213 7B Resident repeat2 COMPLETE and eligible; reference mean pending

2026-10-04 06:47+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before acceptance. The
ordinary vLLM Resident candidate completed the same existing W0 trace as
repeat1: 4000/4000 records, 4000/4000 successful requests, all HTTP 200, and
strict `fixed_length_greedy_v1` validation passed for native `vllm_token_ids`,
target length, prompt hash and timing decomposition. Descriptive values are
mean TTFT 231.091 ms, P95 TTFT 274.771 ms, mean E2E 3089.880 ms, P95 E2E
6968.416 ms, and mean TPOT 25.268 ms. These values are a second seed-42
repeat under the same trace/subset and are not a formal ranking.

The systemd journal records the owned four-GPU service envelope from
05:35:44 to 06:46:21 (4237 s). The runner stopped all four owned vLLM trees;
the unit became inactive and post-run `nvidia-smi` showed no compute apps.
The conservative lifecycle envelope is therefore 16948 GPU-s for this
repeat. The compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat2_seed42_fixed.json`.

Paper-evidence interpretation: D213 now has two complete ordinary-vLLM
Resident repeats eligible for the three-repeat reference set. The lifecycle
envelopes remain conservative per-repeat observations; they are not yet a
frozen `U_ref`, G2 budget, or acceptance threshold. Repeat3 must use the same
trace, subset, generation contract and resource envelope. No performance
figure or PrimeLoRA ranking is justified before that reference protocol and
the common warm-SLO audit are complete.

## CURRENT — D212 ordinary vLLM fixed-output qualification PASS; Resident replay still pending

2026-10-04 04:16+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before qualification
analysis. After the CPU audit and two retained launch-only failures, the
third short 7B ordinary-vLLM qualification completed with four DP=1/TP=1
servers, static registration of the existing 500-adapter subset, and the
existing W0 trace (seed 42, no new trace or weights). The run completed
100/100 requests successfully under `fixed_length_greedy_v1`; the strict
validator passed target length, native `vllm_token_ids`, prompt hash and
decomposition gates for every request. Replay elapsed time was 189.8485 s.
All four services exited normally and the owned GPU/cgroup scopes were
released. Compact provenance is in
`paper_results/ieee_tc/p2_backend_qualification/d212_vllm_7b_seed42_q100_static_fixed_attempt3.json`.

Paper-evidence interpretation: this closes the ordinary-vLLM measurement
qualification precondition, but it is not a Resident reference, warm-SLO
sample, G1/G2 budget, or performance comparison. The next action is a
read-only semantic audit and then the required complete Resident reference
runs, with common delivery and strict 4000-request acceptance. No figure is
generated for this short qualification; a status record is the appropriate
artifact.

## CURRENT — D213 ordinary vLLM Resident semantic audit PASS; complete reference pending

2026-10-04 04:22+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before the audit. The
ordinary baseline runner, replay client and generated DP4/TP1 launch
specification were checked read-only. Neither source contains
`faaslora_full`, `ieee_confirmed`, confirmed-tier propagation, Prime handoff,
hierarchical residency, forwarding or Prime admission/scale-control logic.
The specification contains only ordinary vLLM LoRA, batching, KV/cache and
round-robin endpoint settings; its four services use the existing static
500-adapter pool. The audit is documented in
`docs/ieee_tc/D213_VLLM_RESIDENT_SEMANTIC_AUDIT.md`.

Paper-evidence interpretation: the ordinary vLLM implementation is now a
valid Resident candidate in semantics, and D212 supplies its fixed-output
measurement qualification. This does not freeze a numeric reference or a G2
budget. The next run must use the existing 7B W0 trace, complete 4000/4000
under the fixed contract, release all service scopes, and then be repeated for
the V1 reference protocol; no ranking or figure is justified before that.

## CURRENT — D213 7B Resident repeat1 COMPLETE and eligible; reference mean pending

2026-10-04 05:36+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before acceptance. The
ordinary vLLM Resident candidate completed the existing W0 trace in full:
4000/4000 records, 4000/4000 successful requests, all HTTP 200, and strict
`fixed_length_greedy_v1` validation passed for native `vllm_token_ids`, target
length, prompt hash and timing decomposition. Descriptive values are mean
TTFT 232.078 ms, P95 TTFT 276.110 ms, mean E2E 3092.043 ms, P95 E2E
6977.803 ms, and mean TPOT 25.300 ms. These are one seed-42 repeat, not a
formal ranking.

The systemd journal records the owned four-GPU service envelope from
04:21:49 to 05:32:17 (4228 s). The runner stopped all four owned vLLM trees;
the unit became inactive and post-run `nvidia-smi` showed no compute apps.
The conservative lifecycle envelope is therefore 16912 GPU-s for this repeat.
The compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat1_seed42_fixed.json`.

Paper-evidence interpretation: D213 is the first complete ordinary-vLLM
Resident repeat eligible for the three-repeat reference set. Its lifecycle
envelope is not yet a frozen `U_ref`, and no G2 budget or ranking is derived
from one repeat. Repeat2 and repeat3 must use the same trace, subset,
generation contract and resource envelope, then the reference mean and
SLO/budget audit can be computed. No performance figure is generated yet;
the single-repeat status table is the appropriate artifact.

## CURRENT — D212 vLLM fixed-output path CPU QUALIFIED; Resident replay still pending

2026-10-04 03:47+08. Goal ACTIVE/current turn PROGRESS. Before any comparison,
the full Plan, this status file and METRIC_PROTOCOL_FROZEN_V1 were reread.
D212 audited the ordinary vLLM runner after the D211 Resident candidate was
rejected for retaining `faaslora_full`/`ieee_confirmed` semantics. The old
runner did not propagate `fixed_length_greedy_v1` to replay and the replay
client had no vLLM-native token-ID source. No GPU, remote service, model,
trace, adapter or paper result was changed in this checkpoint.

Baseline-harness commit
`a98ee47d174e5830d1b08da56ad8b10c4e23a671` was pushed to its `origin/main`.
It adds vLLM `return_token_ids=true`, `vllm_token_ids` validation, explicit
generation-contract propagation, and the empty-text/native-ID observation
fix. The pre-existing dirty Serverless changes in that repository remain
unstaged and untouched. CPU-scoped tests passed: fixed-contract 9/9 and
runner/config 23/23; syntax, shell and diff checks passed. A fixed-contract
dry-run with the existing W0 trace confirmed the 256-output/759-prompt caps
and contract reached the vLLM launch/replay specification; it is not a
performance run.

Paper-evidence interpretation: D212 repairs a measurement precondition for a
future vLLM comparison. It does not qualify ordinary vLLM Resident, establish
numeric adapter correctness, freeze warm SLO or budget references, or compare
PrimeLoRA with vLLM. Next construct and read-only-audit the ordinary vLLM path
without Prime control semantics, then run the short qualification replay before
the required three complete Resident runs. See
`docs/ieee_tc/D212_VLLM_FIXED_OUTPUT_PATH_AUDIT.md`.

## CURRENT — D211 Resident reference candidate rejected before formal use

2026-10-04 03:12+08. Goal ACTIVE/current turn PROGRESS. A fixed-four-runtime
7B Resident reference candidate was stopped after semantic audit, before it
could be used to define (U_{ref}) or a G2 budget. The candidate disabled
handoff, hierarchical residency, forwarding, scale control and effective
capacity admission, but still ran through `faaslora_full` with
`routing_policy=ieee_confirmed`. Its selected-replica path therefore retained
Prime's confirmed-tier/physical IEEE routing machinery and was not the
ordinary vLLM reference required by V1 §7. Continuing would contaminate the
reference rather than answer the budget question. This is a protocol
qualification rejection, not a performance result.

The candidate used vLLM 0.30.0, four TP=1 dedicated runtimes, 7B W0, the
existing 4,000-request/500-adapter seed-42 trace, and the fixed-length greedy
contract. Config SHA `da13a1ec5c9e0954138ec9287b7ff9c341ab718ea7985ed07609f0c57c19bae9`;
launch receipt SHA `f2eb5c1aac936cdf23df59ff8f6008000ff67ae9f7d8b35a43cc7318dc9ee0e4`.
The first attempt was rejected before GPU startup because `max_concurrent_loads`
did not match the measured shared movement owner; that failure is retained.
After restoring the measured limit 3, the second attempt reached the serving
phase, observed 1,642 received/dequeued requests and 1,624 successful live
completions (planned denominator 4,000, no request failures observed), then
was stopped. It ended `service_returncode=-15`, `replay_returncode=-2`,
`classification=safety_abort_unattributed`, and watchdog return 1 because the
forced stop did not prove native context release at the stop instant. The GPU
was subsequently empty, the owned scopes were gone, and no partial value is
eligible for ranking or budget estimation.

All raw replay, ingress, service, watchdog and launch evidence is archived at
`results/ieee_tc/p2_backend_qualification/d211_20261004_resident_ref1/`;
the semantically invalid attempt is under
`7b_resident_ref1/launch.launch.attempt2_semantic_invalid/`. The exact remote
3B/7B transfer journals and monitor log were copied only after local release;
remote services are now inactive, with the recorded remote NIC still
1000/full. No remote repacking, hash scan or reconfiguration occurred.

Paper-evidence interpretation: D211 does not establish a Resident reference,
does not freeze a warm/budget manifest, and does not advance 7B acceptance.
The next step is to construct and read-only-audit an ordinary vLLM path that
does not import `ieee_confirmed` or `faaslora_full` control semantics, then run
a short qualification replay before the required three complete Resident
runs. See `docs/ieee_tc/D211_RESIDENT_REFERENCE_ATTEMPT2.md`. No figure is
generated for this incomplete/invalid candidate; the plan calls for a status
table rather than a misleading performance plot.

## CURRENT — D210 ordinary 7B Full COMPLETE, analyzed; 7B formal acceptance still open

2026-10-04 01:46+08. Goal ACTIVE/current turn PROGRESS. D210 completed and
closed; do NOT repeat this replay or add another optimizer before the next
acceptance evidence. Full Plan/status/V1 reread; analyze-results and
academic-plotting read. No baselines/3B started. The D210 checkpoint is
committed and pushed at `2f366779b8a8388a550b5f03e088187ff1da294b`, with the
user's manifest modification and unrelated dirty paths excluded.

D210 ordinary same-config 7B W0 replay (D209 candidate, current HEAD
be7c2951a64d7f381eb0a4345f5701561dcf3549) completed 4000/4000 terminal,
4000/4000 success, 4000/4000 native contract, zero terminal errors. Launch
pass/service/replay/watchdog return 0; native GPU context released and local
scope/GPU cleanup verified. Remote stopped only after local release; exact
transfer journal copied and SHA-checked. No remote reconfiguration or packing.

D210 vs prior D203 one-run descriptive evidence: mean TTFT 1650.511 vs
1741.609ms (5.23% lower); P95 TTFT 3415.442 vs 3853.586ms (11.37% lower);
mean E2E 6034.115 vs 6173.824ms (2.26% lower); mean TPOT 38.397 vs
38.587ms (0.49% lower), but P95 TPOT 63.920 vs 63.039ms (1.40% higher).
Physical GPU U=15916.823905 GPU-s vs D203 15926.245594 (-0.059%); this is not
yet a lifecycle saving claim. Remote 132 pairs/131480060 client and server
bytes/3300789780 verified logical bytes/packing0. Service peak20272316416B,
minimum host available92068065280B, swap/high/max/OOM/watchdog warnings all0.

Offline projection and occupancy audit ran under 3/4GiB CPU scopes with source
and compact SHA retained. Curated result JSON SHA
597620d2ef2d927c971136a56be337887e5bd201895320ea7bf21d6e43028c66; CSV SHA
7ae260cb0b5d2827e3a0f318487a80a51d10879959727a643f2aef5f84fc0eca.
Figure PDF SHA fe72e897f468fe53c0a01e695a2dfdb361c99482da35ed2e95ac2020a3b8a043;
figure visually inspected, single-column/TNR layout, no overlap. D210 report:
docs/ieee_tc/D210_7B_ORDINARY_FULL.md. The first 128MiB metadata projection
attempt is retained as attempt1; final 165266073B compact projection uses the
previously qualified 192MiB bounded path, original 498984004B outcome remains.

This is observation/contract evidence, not formal G1/G2 acceptance: numerical
adapter correctness, common warm SLO thresholds, Resident GPU budget reference,
and paired multi-block evidence remain open. P95 TPOT trade-off is retained.
NEXT: perform the necessary 7B common warm/Resident reference and correctness
acceptance work before 3B or baselines; do not declare the model accepted from
this single improvement.

### D210 live/replay history (superseded by COMPLETE above)

2026-10-04 00:29+08. This turn PROGRESS: one qualified ordinary replay launched
00:11:47 in tmux tc_d210_full1, now VERIFIED WAIT, goal ACTIVE. Do NOT launch
another replay/optimizer/qualification or stop based on observation timeout.
Service primelora-tc-svc-77e8c69b01e547a78275658afd562ee0.scope,
invocation b3359f20fd7e42199afba473684542ca/inode16524917,72/80GiBswap2.
Aux primelora-tc-aux-e41fba3b3ec440ce8804ea3fa07fbf1d.scope,
invocation42585c2ff6e1458db99213aeeb966b37,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice551157.263463389/businessstart551217.263463389;
plannedarrivalend555181.172366511. Detailed observers OFF,4000W0 unchanged.

Latest physical_deployment/request_terminals.jsonl:876terminal/876success/
876native_contract_matched/error_types empty. Partial ONLY; numerical/full/SLO/
performance acceptance unknown. Early monitor looked for request_terminal in
replay.jsonl (which only logs creation/submission); those zero counts were NOT
actual absence of service terminal records. Correct source now used above.
Watchdog1035/monotonic552205.321853412:service17849155584B,hostavailable
95135272960B,events/swap0,no warning/abort/foreign/escaped. ActualGPUworkers
4049851/4054776/4055069/4055288 all service-owned, originalCPUaffinity preserved.
Freshdisk161290977280B; running100GiB safe. No other heavy/analysis/tool sessions.
Prelaunch/health scopes inactive/identity+paths empty. No remote operations since
activation/health; readonly published cache reused without repacking.
7Bhealthclock remote-process-monotonic:254154de56494625aaba44191e5bd06a.
Exact-owned remote stop/copy wrappers prepared/bashed-n PASS, NOT executed;
must wait local terminal + physical release. No local cleanup wrapper prepared
yet; adapt existing D203 wrapper to exact live identities/PIDs after terminal.

NEXT monitor SAME tmux/service/watchdog -> terminal/local release -> exact remote
stop/copy -> bounded existing D203 metadata/projection/curation/occupancy/timing
analysis -> table/interpretation/scoped backup. Avoid whole-loading large source
JSON or reprojecting historic D20334MB projection. No Git mutation while running.
All numerical/commonreference/legacy G1G2 and subsequent matrix gaps remain OPEN.

### D210 prelaunch history (superseded by LIVE above)

2026-10-04 00:11+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread after compaction; combined-output Plan-tail truncation repaired by
explicit tail read. run-experiment/monitor-experiment read; no subagents.
D209 already SEALED/BACKED, no repeat tests/probe or additional optimizer.
Once-only published cache permission fulfilled D78/D80; reused, not recreated.

Raw results/ieee_tc/p2_backend_qualification/d210_20261004. Reused D203 ordinary
4000 W0 launch/config/prelaunch; only freshowned paths/current source identities.
D157cap4/profile/generation/60snotice/1800s qualification unchanged; detailed
frame/control/execution observers OFF. Current HEADbe7c2951; no runtime edits.
Prelaunch078ec3e435bb6fcca8b9c21c9150d97db0fa521b4752c408e5981f5ec42c986e:
614source refs/147protected/resourcePASS, session90968 CLOSED0. Health CLOSED0.
Remote activated only BEFORE measurement; NIC1000/full and authenticated health
PASS. 3Bpid1813549/id de107771cd5544868a3210f635259bfb;
7Bpid1813551/id97ca094dd9de49e1b5d0aee40ca4dd78;
monitor-d210full1pid1813554/id72aa8b760743474f9c707a572f749370.
NEXT ONE ordinaryFull in tmux -> actualworker/resource checks -> terminal/local
release -> exact-owned remote stop/copy -> bounded reused analysis/table/backup.
No remote config/hash/cleanup/restart during replay; no second experiment.
Numerical/output-difference/commonwarm/Resident/old-newG1G2 remain OPEN. Current
candidate is CPU-qualified only, not performance/model acceptance. 7B then3B then
baselines; all M1M2/A1–A5/S1–S13 retained.

## Previous checkpoint — D209 acquisition identity CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-04 00:01+08:
be7c2951a64d7f381eb0a4345f5701561dcf3549. Push44490 CLOSED0; independent
ls-remote42250 CLOSED0 exactmatch.10explicitfiles/126staged+archiveentries
scope/secretsPASS; checksum/diff/syntax/smoke/qualificationPASS; usermanifest
excluded. No stagedfiles, GPUcompute, experiment scopes/tmux or live tools.
Remote remains stopped/untouched. D209 DONE, doNOTrepeat tests/probe/backup.
NEXT one ordinary same-config7B4000Full after fresh resource/source gate, then
cleanup/table/interpretation. GoalACTIVE; no performance/model/G1G2 acceptance.

2026-10-03 23:59+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1 read after compaction;
optimize-for-gpu, vLLM+optimization reference, run-experiment, analyze-results and
academic-plotting read. No subagents/GPU/remote operation. Published delivery cache
permission fulfilled D78/D80, not regenerated. D208 COMPLETE, not repeated.

One candidate only: `_acquire_runtime_gpu_reference.observe_source` consumes fresh
identity/tier/epoch/path, not physical footprint. Reuse qualified identity-only
endpoint and monotonic identity frontier. Original native demand transaction
keeps before/after full HOST allocation checks, slot/copy guards and references;
initial routing classes/planner/admission/config/formulas/SLO unchanged.
Acquisition evidence explicitly identity-only; no invented 0 capacity.
Official vLLM0.30 worker/model manager and CPU optimization docs checked;
D178/D181/D202 histories distinguished. No immutable physical-descriptor cache
implemented: consumer audit found this simpler redundant query. Runtime change
confined to ONE runner method; native owner/worker/planner/admission unchanged.

Raw results/ieee_tc/p2_backend_qualification/d209_20261003. All CPU3/4GiBswap0/
CPUs2,3,26,27; terminal events/swap0, actual units/paths absent:
- tests1 session79687 CLOSED1/id3cfeefd1977746bd94f4dc1b8a7df4c8:
  222tests/1error, old fixture native_source_state.owner_id now identity-only.
- tests2 session83761 CLOSED1: same222/1error, next old .sources assertion;
  both corrected to explicit identity record, production candidate unchanged.
- tests3 session2940 CLOSED0:222PASS2.862s, including3new budget/clock checks.
- regression1 tmux ENDED/id36b025de7b0c4809b4d402b30a112882:
  1144PASS157.105s, peak2050699264B; tests overlap, not independent repeats.
- probe1 session60898 CLOSED0/id312059a7ac2c4147b00c1b2767e236a4:
  exact-parent five-source pairs/10records PASS; D20334MBprojection reused.
- curate1 session27710 CLOSED0/id85a9f923ea1b4c7e9430c2b2228acf44:
  147protected/PlanV1/unchanged-runtime AST/sourceSHA/syntax/shell PASS.
Failure/source snapshots retained. No GPU/inference/remote operation or live
tool sessions/scopes/tmux. No hardware/profile/config/threshold changes.

GPU/HOST component paths unchanged. HOSTfile/NVMe/Remote fullgraph1->0,
identity1->2, initial targetgraph1->1 and totalfreshreads3->3. Each1demand
transaction, same tier/holds/releases, no remaining references. D203 source
populations:GPU1189/nativeHOST1766/HOSTfile36/NVMe873/Remote136. Lastthree1045/
4000=26.125%; population coverage, NOT actualRPCcounts or latency attribution.

Doc D209_ACQUISITION_IDENTITY.md SEALED35cec3282f9d6c972288d74ee44d1c7d015f198e00b2ba185f00778206476415;
curated8946a278312da1d2e9a20878ed9f92432cd81d14017702d3b14a1c76ed08985d;
60member2721576B evidencebundlea21feb4e99b370fafbc96d058d91554a8685a574a560cc10b433217e175b6d0c.
Qualification table delivered; no GPU timing gain/CI claimed. DoNOTedit sealed
doc/results. github-sync read. NEXT exact10path scoped checked backup, then
ONE ordinary7B4000Full same D203/D157cap4config/profile, freshownedpaths,
diagnostic observers OFF, current source/Plan/V1 identity. Reuse D203 launch/
preflight/remote/cleanup/analysis. Lastdisk162226733056B, fresh gate beforeheavy.
No repeat D207/D208/D209 or additional optimizer before ordinary validation.
7B numeric/output differences/commonreference/old-newG1G2 remain OPEN, then3B,
thenbaselines; allM1M2/A1–A5/S1–S13 retained. No additional optimization stack.

## Previous checkpoint — D208 isolated LoRA coordinate arithmetic QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 23:34+08:
a19365ef15f9d03885d1981b2e28c4fbbc445955. Push69434 CLOSED0; independent
ls-remote44602 CLOSED0 exactmatch.7explicitfiles/31staged+archive entries
scope/secretsPASS; diff/checksum/syntax/GPUqualification PASS; usermanifest
excluded. No stagedfiles, GPUcompute, experiment scopes/tmux or tool sessions;
remote remains stopped/untouched. Curate actuala835f487f84e4ccba3f5280586eb8812
scope/path absent, events/swap0. Lastdisk162263064576B, freshgate before heavy.
Curated bca88d6cdd8555632175f891f8b176428247be49a4c65c63b15782b1a1434451;
doc e1b92d06c65c35322e24d2e51118cc2717e6a5aa85e0fbe8165ed724604057fd.
D208 DONE; doNOTrepeatprobe/curation/backup. NEXT remaining full-request
correctness/output-difference + common-reference/old-newPrime G1G2, as below.
Goal ACTIVE, no model acceptance or mainmatrix completion.

2026-10-03 23:31+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read after compaction; grouped output truncation repaired by separate fullread.
analyze-results/vLLM+troubleshooting/run-experiment/monitor-experiment/
academic-plotting/github-sync read; no subagents. No production code/config/
formula/SLO/native-package change, remote operation or cache generation.
Once-only cache permission fulfilled D78/D80. D207 already backed, not repeated.

ONE minimal arithmetic question following D63/D167/D207: installed native
shrink/expand on coordinate inputs must equal directly read checkpoint columns.
Official v0.30 test_punica_ops.py/utils.py fetched/read; generic broad tolerance
NOT imported. Four existing content classes, all32layers qkv/o, FP16/4slots/
rank64padding,16even A-input coordinates/all64 B-rank coordinates, mixed base
rows and two slot assignments. Native eager + CUDA Graph replay checked.
No new model/adapter/workload, no new observation framework or Full run.

Raw results/ieee_tc/p2_backend_qualification/d208_20261003. arithmetic1 PASS,
512 combinations/673710080 compared elements/0 mismatches, maxabs0. Each128
layer/module/op reference group distinguishes nonzero-adapter swap and base;
combined16782968 differing elements. Elements/padding/repeated coordinates NOT
independent samples, no performance/CI/Full/n_correct upgrade. Arbitrary-input
roundoff/fullmodel arithmetic and D203126hashdifferences remain unproven.

Service b741410a96a44636b72b4a2a705ac28e / invocation240e1426d709418b9cf9430459fd389b,
inode16495613, actualGPU0worker3849963 service-owned CPUs4–23,28–47.
Auxe03a7ae7d7234f4aa3edacbb8a73a234. 22watchdogsamples/peak782970880B/
minhost111552618496B, allguardevents/swap0. Service/watchdog0, physicalcontext
release/pathremovedtrue; exactcleanup23:26:54. No GPU/tmux/scopes/remote live.
GPU script20.305s diagnosticwall, NOT inference timing or lifecycle saving.

Curate1 session6376 CLOSED0, CPU3/4GiBswap0/CPUs2,3,26,27; terminalevents/swap0,
unitidentity/pathabsent. Independently checked full512keycoverage/128controls/
sourceSHA/actualworker/resource/147protected/PlanV1/runtimeunchanged, syntax and
shell smokePASS. No GPU/curation failure. One launcher-preparation JS index
typo before writes and one extra-newline archivalcopy corrected before curation;
not measurement attempts. Exact prelaunchprotocol retained and SHA matched.

Doc D208_LORA_COORDINATE_ARITHMETIC.md/table SEALED, doNOT edit after binding.
24member bundle SHA b479466b36c2b9118575e38de62e5436546f75be323f627105b3a9470cfffe61.
NEXT exact7path checked backup, then RETURN mainline. Do NOT repeat D166/D167/
D207/D208 or broaden kernel proof absent a specific new failure/hypothesis.
Remaining: audit output-difference using evidence capable of distinguishing
batching arithmetic from wrong adapter; commonwarm/Resident and necessary
matched old-newPrime evidence for actual7BG1G2, then3B, thenbaselines. Existing
126outputhashes alone cannot recover firstdivergenttokens/logits. No silent
batch-invariance toggle, tolerance change or speculative Full. AllM1M2/A1–A5/
S1–S13 remain pending. Native-count completion != formal numerical acceptance.

## Previous checkpoint — D207 actual 22-request execution metadata QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 23:11+08:
8efaad5dc7c169881f1ea21d3b5cac96d319c4be. Push25309 CLOSED0; independent
ls-remote97019 CLOSED0 exactmatch.13explicitfiles/120staged+archive entries
scope/secretsPASS; diff/checksumPASS; usermanifestexcluded. No stagedfiles,
GPUcompute/runningexperiment scopes/tmux/tool sessions; remote untouched/stopped.
Latestdisk162280996864B; freshgate before newheavy, floorsunchanged.
D207 qualification and backup DONE; do NOT repeat them. NEXT remaining
correctness/output-difference and common-reference evidence for actual old-new
7B G1G2 acceptance, not a further observation framework or speculative Full.
Read V1 FULL before comparison/config selection. 3B/baselines/latermatrix remain
pending; goal ACTIVE, not model completion. Details and retained failures below.

2026-10-03 23:10+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read
after compaction; no subagents. vLLM/troubleshooting/run/monitor/academic-plotting/
github-sync read; grouped academic read truncation repaired by individual fullread.
Once-only published-cache approval fulfilled D78/D80, no regeneration or remote
operation. No optimizer/configuration/equation/SLO or native site-package changes.
No system comparison/config selection, V1 unchanged. Production Full still has
NO execution observer. Changes only observer class + opt-in preflight qualification.

Actual V2 source matched official v0.30 hash; previous D206 assumed older runner.
Forward and sample are separate native calls. FULL graph replay / PIECEWISE outer
call / NONE model forward explicitly observed. Internal request IDs retained
verbatim, joined offline to input-side lease via EXISTING frontend exact-ID +
core retirement acknowledgement. No suffix removal, adapter/order guessing,
disabled randomization, eager bypass or loosened arithmetic tolerance.

D207 attempt4 COMPLETE/pass:22/22 correct native-count contracts,3917tokens;
3961iterations=3917nonempty+44empty,31468events/24drainedsnapshots. Independent
native scheduled/completed deltas both3917. Every real forward/logits mapping
matches intended adapter, physical slot and GPU metadata/layer/buffer identity.
Actual graph modes FULL/NONE only; PIECEWISE still CPU-only. Four content classes
reuse D166inputs; no newweights/trace/contentreaudit. Does NOT qualify numerical
arithmetic, concurrentFull/fullpool/n_correct,126outputchanges or formalG1G2.
Raw35555149B SHA993c2956ee2c007dd678327932dc5368e4f454f43595587e8d3197088a46467d;
launchd8a47fdcad9ce9f9336f94f4bce50c708c7f626f658e2c0ba4aa153fc1541489.
Service24c20e6f2a2e45ae87251d87393528b9/invocation405173e431a146c3b32d32f300ee4626,
auxf17bf7b4769440279cf3a3f3a270ad95, nativeworker3749672 actualservice-owned/
CPUs4–23,28–47. return0/watchdog0, physicallease released/pathsgone/GPUempty,
exactcleanup23:06:14. No liveGPU/tmux/remote/tool sessions.

All three failures retained: attempt1 pre-request admitted=[] schema;
attempt2 pre-request absent oldrunner _model_forward; attempt3 firstrequest
observer compared external/internalID, nativecoreexception,0completed/partial
journal=startonly. Not numeric/servingperformance failures. All released, guards0.
FinalCPU tests4 416PASS20.225s/id2089ff8f4ee7437b919a0edf45de6f2e; earlier
tests1/2/3/regression retained (overlap, notindependent).3/4GiBswap0events0.
Bounded inspect4 session93325 CLOSED0; actualmetadata printed, scopegone.
Curate1 session40444 CLOSED0/id8da81aa9213547c8801c125596fed75e;3/4GiBswap0,
protected147/actualinput-SHA/independentjournal/ASTscope/unchanged serving
policy and frontend/scheduler/owner/config checksPASS. No curation failure.

Doc D207_NATIVE_EXECUTION_PREFIX.md SEALED6934aa43e44a204ef92b1b3b42d0be7de92d15dde2fe36a2775c8569415c4867;
curatedc55595d89796854604502b911a71856a3a3c626383e53c872b7d3917ded3db2c;
80member2041377B bundleac2f3dd3f3809ca2295eb9b06d053c80cff6f4b8947783e877664a5d093101a7.
Qualification/attempt/request tables delivered, no speedupfigure/ranking. Do NOT
repeat D205/D206/D207 probes or22requestrun absent a new actual hypothesis.
NEXT exact13path checked backup, then remaining correctness/output-difference
evidence and common-reference/newmetric old-newPrime acceptance. Read V1 FULL
before comparing or selecting configuration. Return mainline, not more observer
framework: actual7BG1G2acceptance→3B→baselines. Commonwarm/Resident/numeric and
legacycomparison stillOPEN; all M1M2/A1–A5/S1–S13 retained, not finished.

### D207 preparation and failed-attempt history (superseded above)

23:02+08 update: attempt3 installed observer, but first native request rejected
because external req_1 != vLLM randomized req_1-ab14385f; adapter963725 unchanged.
Completed0/22; no numeric mismatch concluded. Native core died from observer
exception, partialjournal=startonly; failure/tail retained. launch285a6985a4ca515fe2224641b64cfb3ede351f7811ae9e9c8786e3f1925dea82,
service8f62138b8cde4944b1ad73f9013e25c4/invocation4cf9895be6474e8784bea51f92fdcc03;
return2/watchdog0, release/pathgone true, exactcleanup22:56:10. AllGPU/scopesgone.
Correction reuses existing idempotent native retirement receipt (frontend exact
external/internal map + core retired acknowledgement). Observer collects internal
IDs verbatim in separate counters; offline checker joins ONLY independent receipt,
not suffix removal/adapter/order inference. Existing frontend/owner unchanged.
Tests4 416PASS20.225s/id2089ff8f4ee7437b919a0edf45de6f2e/session4760 CLOSED0,
3/4GiBswap0/events0/pathabsent,147protectedPASS. NEXT preparedattempt4 sameinputs/
config aftergate, newpaths. Noobserver data accepted yet; allG1G2gapsremainOPEN.

2026-10-03 22:55+08 continuation. Full Plan/status reread after compaction;
skills vLLM/troubleshooting/run/monitor/academic-plotting/github-sync read.
Attempt1 native admitted=[] schema error before requests, fixed; tests2
34PASS1.681s/id2ae3289130c84c498780f845d4feffbc. Attempt2 observer assumed
older runner `_model_forward`, absent in actual `vllm.v1.worker.gpu.model_runner`.
No requests submitted/no journal. launch877ba8d843e9ef1ef30f2c859fe5e3cf35ddd2f8f1b646909a1d8ef214fa22d1,
service d5d2fbb8eaa341bfa0265d02225bd7a7/invocation8dd62442608b4f6c93bde6645a0371ea,
return2/watchdog0, physicalrelease/pathremoved true. Exact cleanup verified
22:45:58, service+aux72adc254e61b42b98d0e4f75c491e6a0/path absent/GPUempty.

One diagnostic-only correction targets actual nativeV2 explicitly, no default
serving policy/site-package/config/formula/SLO change. FULL graphmanager replay,
PIECEWISE outer call and NONE forward observed; native prepare_inputs batch
bound across separate execute_model/sample_tokens; no double-counted PW child.
Execution contract vllm_v2_split_forward_sample_v1; checker requires split
events and actual request-state/lora identity. Official V2 runner exactSHA
174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce matchesinstalled.
Tests3 414PASS20.539s/session24652 CLOSED0/id3f90f697428c42ce9af64b2cb38fc243,
3/4GiBswap0CPU2,3,26,27, events/swap0, scopeidentity/path absent. Same tests
overlap preceding groups; no numeric/actualCUDA qualification yet. Source
curl/check session4710 CLOSED0. No remote/cache generation or subagents.
NEXT fresh source/resource gates and SAME22request config under new attempt3,
not anotherFull/optimizer. D207doc DRAFT, noD207commit yet. Allcorrectness/G1G2/
7Bacceptance then3B/baseline/formalmatrixgaps OPEN.

### D207 preparation history (superseded above)

2026-10-03 continuation. Goal ACTIVE/current turn PROGRESS. Full Plan/status,
vLLM/troubleshooting, run-experiment, monitor-experiment, academic-plotting and
github-sync read. No subagents. Existing preflight gained only opt-in
native_execution_metadata plus journal checker; production serving unchanged.
Independent input leases/owner/adapter/target, native begin/end, all ordered
forward/logits boundaries and independent scheduled/completed counter deltas
checked. No timing/arithmetic/fullpool/n_correct upgrade. Reuse D16622request
configuration/content evidence; no content re-audit or optimizer/config change.

Raw results/ieee_tc/p2_backend_qualification/d207_20261003.
CPU3/4GiBswap0CPUs2,3,26,27, events/swap0, units/paths absent:
tests1 session46537 CLOSED0/idb49d47a9b7e64acda59565664ed074f4,
33PASS1.677s (9new tests); regression1 session13055 CLOSED0/
id33603cdeb68546a5a3f1964362491bd1,376PASS19.617s (groups overlap).
One premature read of terminal regression files, later captured/verified.
147protectedPASS. Doc D207_NATIVE_EXECUTION_PREFIX.md DRAFT/table.

Disk belowNEWheavy150GiB; cache1 PyPI-only5–200MiB read-only audit found only
187609088B; NOT APPLIED. session25966 CLOSED0/idd2772e4ffed64b70a1fd9cb83e61aa5c.
cache2 one official PyTorch CUDA2.0.1+cu118 cp39 archive SHA matched index,
but urllibHEAD403, session84639 CLOSED1; no deletion/outputaudit, log/source kept.
cache3 same public URL via curlHEAD200/size, exactSHA
eb55f29db5744eda8a96f5594e637daed0d52278273005de759970e67cfa6a5a,
fresh reference/owner/open-file guards. audit25447 CLOSED0/
id5e9887208165418dbcc1b50f4794519a; audited2267332608allocatedB.
Reviewed auditSHAa750b307d1acb75d8ad0faafa96486b257bd4d4bf4d0ac1a2eac30e0993cfe63.
Apply43193 CLOSED0/id80c28ab4422a411883e1125384a6273e: exact one archive+header
removed,2267332608B freed, four tracked repos/installed environments unchanged.
Unit/path absent, events/swap0. Disk162355134464B aboveNEWheavy150GiB.
All cache scopes same3/4GiBswap0; no model/results changes. Original audits kept.

NEXT freshgate then ONE existing22request actualGPU diagnostic in tmux,
prepared run_7b_execution1.sh (new paths only, D166 config unchanged).
No remote service/cache regeneration needed. Whole numerical/126outputdifferences/
commonwarm/Resident/old-newG1G2 remainOPEN. 7Bacceptance->3B->baselines unchanged.
AllM1M2/A1–A5/S1–S13 retained. Do NOT redo D205/D206 or these CPU tests.

## Previous checkpoint — D206 native execution observer CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 22:09+08:
82ac6b5e21457ddfb0a667597a39e874d1561974. Push21943 CLOSED0;
independent ls-remote36584 CLOSED0 exactmatch.11explicitfiles/62staged+archive
entries scope/secretsPASS; bundle/diff(cr-at-eol)PASS; usermanifestexcluded.
Verification6acc1bcf7ab46880c01afdd779e91f2327a112be74222dd9d8b1cf22945d9458.
No stagedfiles, live experiment scopes/GPU compute/tool sessions; remote remains
stopped/untouched. Lastdisk160026431488B belowNEWheavy150GiB. Do NOT repeat
D206 tests/backup or D205 probe. NEXT existing preflight integration -> matching
lightweight check -> fresh resource/reclaim gate -> ONE22request actualGPU
diagnostic as below. All correctness/acceptance/later-matrix gaps remainOPEN.

2026-10-03 22:07+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read;
vLLM+troubleshooting, run-experiment, academic-plotting, github-sync read.
No subagents, GPU replay, optimizer/config/formula/SLO changes, remote operation
or cache creation. Latest once-only permission already fulfilled D78/D80.
No system comparison/config selection; V1 unchanged. No model acceptance.

ONE isolated observer added in gpu_monitor + existing observation RPC optional
argument through InferenceEngine/proxy. Default None installs NOTHING in Full.
Native begin/end binding independent of batch; execute/forward/logits boundaries,
actual GPU metadata readback/stream fence, padding/graph mode, all real scheduled
rows, actual executed model-tree module identity/Punica/A-B and metadata pointers.
Bounded64bindings/<=10000iterations/<=32MiB serialized buffer; explicit fail,
no silent truncation. Drain retains counters/sequence/failure; install rollback,
atomic owner check before restore. Dense TP/PP/DP1/nonspec/no ubatching only.
No site-package patch or default inference policy change. Metadata collector
does NOT certify arithmetic/graph replay/fullpool/n_correct. Ordinary runtime
last measured remains D203/D202. Official fixed-v0.30 runner fetched/read;
Punica URL fetch failed, installed exact D205 source SHA reused, not invented.

Raw results/ieee_tc/p2_backend_qualification/d206_20261003. Sequential scopes
3/4GiBswap0CPU2,3,26,27; terminal events/swap0, actual paths+units absent:
- tests1 session71031 CLOSED0/id69b86c389dc64f0e93ea9a18dd5c8a84;
  494PASS19.793s (initial12 new tests). Source/log retained.
- tests2 session63695 CLOSED0/idf90f839be9274b119075df39c076af67;
  496PASS20.416s after model-tree identity/atomic-unhook/rollback additions;
  final14newtests. CPU fake tensors/graph label, NOT actual GPU execution.
- curate1 session23740 CLOSED0/id52d05dcbf78b4646bcae2d9f44a2e14c;
 147protected/sourceSHA/syntax/unchanged AST outside observer+three opt-in RPCs
 PASS; owner/planner/instance/dedicated-worker/preflight unchanged.
No failed test/curation attempts. One initial read-only scope absence check
guessed uid1000; corrected to actual recorded uid1001 and verified explicitly.
No live GPU/experiment/tool sessions. Remote stopped/untouched. Disk160034369536B
belowNEWheavy150GiB; no GPU/build or deletion/relaxed floor this turn.

Doc D206_NATIVE_EXECUTION_OBSERVER.md SEALED; do NOT edit after binding.
Curated7f44ff456079a7646dca4da50c2dcccd6cd3abec525c8c4d688b1d18f5e501df;
39member1164889B bundle19c4f473800ed0daffaed3bd7463a333dfcfed18661f994b6ee4f33d54d19f36.
CPU qualification/status CSV table, no performance figure or ranking.
NEXT exact11path tested backup, then connect existing preflight short-prefix
qualification to start/read/stop + external lease/adapter/sequence/whole-iteration
checks; lightweight integration check -> fresh disk/resource gate and ONE existing
22request/four-content-class actualGPU diagnostic. Reuse D166/D167 content proof;
no repeated D205CPU probe, D206tests or speculativeFull/new framework. Isolated
local-artifact diagnostic needs no remote service. Preserve failures/missing.
All numeric/126output differences/commonwarm/Resident/legacyG1G2 remainOPEN.
7Bacceptance->3B->baselines; all M1M2/A1–A5/S1–S13 remain outstanding.

## Previous checkpoint — D205 execution-metadata checker CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 21:40+08:
0a6c22ebf666a9f97d36c90639927ea1e8b0ad65. Push46148 CLOSED0;
independent ls-remote33898 CLOSED0 exactmatch.11explicitfiles/128staged+archive
entries scope/secretsPASS; bundle checksum/diff(cr-at-eol)PASS, usermanifest
excluded. No staged files, experiment scopes, GPU compute or live tool sessions.
Remote remains stopped/untouched. Plan/V1 SHA unchanged. Lastdisk160057401344B
belowNEWheavy150GiB; no GPU/build before freshgate and audited reclaim.
NEXT isolated execution observer qualification below; do NOT repeat D205 CPU
checker/probe or backup. All numeric/G1G2/model/formal gaps stay OPEN.

2026-10-03 21:38+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read;
vLLM+troubleshooting, run-experiment, academic-plotting, github-sync read.
No subagents, GPU replay, optimizer, configuration/formula/SLO changes, remote
operation or cache creation. Latest once-only cache approval fulfilled D78/D80.
No system comparison/config selection; V1 unchanged, no new acceptance claim.

Actual vLLM0.30 request->batch->slot->Punica->kernel path audited using installed
files + official fixed-version source. Existing manager already invalidates on
slot-layout change; DO NOT reimplement it. Kernel consumes grouped sorted rows,
counts/starts/active slots and flags, separately for token and sampler. Logical
callID and before/after slot contents alone do not cover this execution boundary.

Added ONE offline preflight validator, no production hook/caller. All previous
preflight AST nodes and serving sources unchanged. External expected backend
request bindings, batch order/counts, physical slot uniqueness, token/sampler
indices, groups/permutation/counts/starts/grid/no-LoRA checked. All-base stale
arrays explicitly unused per native early return. DenseTP1/nonspec scope only;
pure checker NEVER self-certifies GPU collection/arithmetic/fullpool/n_correct.
Current installed LoRAKernelMeta CPU probe8cases/34wrongcontrols rejected;
CPU Torch2.8, NOT nativeTorch2.13/CUDA/compiled-graph qualification. New10tests.

Rawresults/ieee_tc/p2_backend_qualification/d205_20261003. All3/4GiBswap0,
CPU2,3,26,27, terminalevents/swap0, paths/unitsabsent/GPUempty:
- tests1 session89400 CLOSED1/id94c7ee8174e64410875bd1e30872b440;
  384run/2failure3error: four old OSguard tests used model Python withoutpidfd;
  one old dynamic-importfixture lacked sys.modules registration for dataclass.
- tests2 session70196 CLOSED0/idcec5ab5190044d01a676da1c1034137c;
  337PASS21.462s. Only fixture normal-import registration repaired, guards unchanged.
- os1 session63454 CLOSED0/id7908320ad53440b4be48c0eca25f6e42;
  remaining47PASS.096s under qualified /usr/bin/python3. Tests2 final readonly
  seal verification and os1 briefly overlapped; no concurrent GPU/build/inference.
- probe1 session97917 CLOSED0/id43bcb4ac254f484aa06657db7d04aa5a;
  9.73s/RSS1101660KiB,8cases34negativecontrols,CUDAuninitialized.
- curate1 session75120 CLOSED1/ida043f8ac98e24de7b242da775ecc3740;
  JSON parser rejected existing INFO stdout prefix before any curated write.
- curate2 session85357 CLOSED1/id6a7fb690f92642dbaad125460a816e38;
  overescaped prefix regex rejected same retained stdout before curated writes.
- curate3 session7690 CLOSED0/idbbf3fc760b7c4d8588410b8dd081b3c5;
  strict known-prefix parser corrected only in separate third analysis. No tests/
  GPU rerun for curation. All scripts/logs/source snapshots/failures retained.
Some readonly rg/sed searches used missing guessed paths; corrected, no measured
GPU failure. No live tool/GPU/remote jobs. Disk~160068288512B stillbelowNEWheavy
150GiB; no lowered floors or deleted data.147protectedPASS.

Doc D205_NATIVE_EXECUTION_METADATA.md SEALED; do NOT edit after binding.
Curatedfd3fc3e95c1419ae60e1ddc1dbf4b7b8c25256fdb93350be30e07b668c532385;
verification49266635ff76d14b3a7870262406c6d73bffb196efd04d031016b9f2d1b01ba4;
84member1171392B bundlebb90a790b3b14d10d7c4615a76b568e1f7c9134e6fa3398cf65921d06f921379.
Correctness status table/CSV delivered, no performance ranking. Tests1/curate1/
curate2 failures are not omitted or reclassified as GPU failures.

NEXT exact11path tested backup, then isolated native forward-boundary observer
qualification (existing worker extension/preflight, no production overhead),
then ONE existing22request/4contentclass diagnostic only after disk/resource
gate. Bind independent begin_use/backend IDs, actual device metadata, all real
iterations/padding/graph path and layer wrapper/buffer identity. Python-only
operator hook may miss CUDA graph replay; don't call eager-only evidence graph
qualification. Reuse D166/D167 content proof; no repeated loose D63 or speculative
Full. D204126changedoutputs/numeric/commonwarm/Resident/legacyG1G2 stillOPEN.
7Bacceptance->3B->baselines; all M1M2/A1–A5/S1–S13 remain outstanding.

## Previous checkpoint — D204 retained-data diagnosis COMPLETE, SEALED and BACKED

Backup verified2026-10-03 21:10+08:
8d45f0775842136032a5f803cfbd301c0467289d. Push9201 CLOSED0; independent
ls-remote37620 CLOSED0 exactmatch.18explicitfiles/35staged+archiveentries
scope/secretPASS; checksum/diff(cr-at-eol)PASS; usermanifestexcluded.
No stagedfiles, GPUcompute or experiment scopes remain; remote untouched/stopped.
VerificationSHAed367dd51fe40366c77e72acef3c4fce55137af5e4ce0225a50571c987903512.
Lastdisk160086003712B stillbelownewheavy150GiB; freshgatebeforeGPU/build.
NEXT identity-execution observation boundary audit below, not repeatD204,
anotherFull or cachecreation. All numeric/G1G2/model/formalmatrixgaps OPEN.

2026-10-03 21:08+08. Goal ACTIVE/current turn PROGRESS. No serving/code/config/
formula/SLO changes, GPU replay, remote management, new weights or traces.
D203 retained34MBprojection + D195projection + D196table reused; no big-log
reprojection. Once-only cache permission fulfilled D78/D80, not regenerated.
analyze-results/vLLM+reference/run-experiment/academic-plotting/github-sync read.
Full Plan/status/V1 read, hashes unchanged, no subagents.

D203 preengine1447.966992ms. D195->D203 six nonoverlap meanms:
arrival223.603735->194.293671;gate198.569185->123.963843;
routing519.377555->341.223842;residual414.846893->296.681148;
source-to-handoff539.380491->441.574187;handoff-to-native65.913325->50.230300.
One run each, noCI or causal attribution.205TTFT-onlymisses have2050.550355ms
source-to-handoff vs350.114900ms in3711timingpasses; conditionalnotcausal.
Eight500-request index groups timingpass470/469/469/471/463/469/439/461,
so not merely startup. Same161preengine-only thresholdcrossings;
3711/4000timing-only still89short3800, NOformalSLO/modelacceptance.

Outputchanged126same3874:finance11/103,medical2/66,other113/3831.
RetainednativeadapterintIDs matchlogicalmapping4000/4000 inbothversions;
notper-tokenexecution/numericproof. SourcecontentSHAequal672pairs;3328native
sourcepairshavenocomparablefileSHA, explicitUNKNOWN, notfilled. Outputhashes
alone cannot recover firstdivergenttoken/logits. OfficialvLLM0.30 defaults do
notguaranteereproducibility; batchinvariance supportsCC>=8.0 andhasLoRA branch,
but not testedhere andNOTproofwhy126changed. No toggledbackendflag/tolerance.

AllCPU3/4GiBswap0CPUs2,3,26,27; scopes/pathsgone,GPUempty,events/swap0:
analyze1 session11110 CLOSED1/id7dd3cfe6b37546b68983f1a4b129eab3,
10.68s/RSS1107260KiB; incorrectuniversalfileSHAassumption raisedKeyError on
native source; originalscript/log/partialtables retained. analyze2 separate
outputdir, session71305 CLOSED0/idd976272d79854f8dad5d84c78269486a,
11.15s/RSS1110264KiB; only offline schema coverage corrected. finalize1 session
16583 CLOSED0/idc4f8f5b0677a437ba01ff2fa78e1106d;147protectedPASS, independent
CSVsummary/documenttables/sourceSHA/syntaxsmokePASS, productiontreeunchanged.
Read-only searches with guessed missingpaths failed then corrected; no serving
or GPU measurement failure. No live tool/GPU/remote jobs. Disk160080211968B,
belowNEWheavy150GiB; CPUanalysisonly, safetyfloorsunchanged.

Doc D204_REMAINING_MISSES_AND_IDENTITY.md SEALED (doNOTeditafterbinding).
Curated39c0a6d608dc0f2d652a75620ef066fb7ccb35092e5421f85c65cc5eded0e1ab;
17member28611B bundle024a903561b4af739c5146df6c437967ba7b83cf5c4b88ae64a245fe93bb731b.
Rawresults/ieee_tc/p2_backend_qualification/d204_20261003; validcurated
paper_results/ieee_tc/p2_backend/20261003_d204_control_path_attempt2.
Failedfirst20261003_d204_control_path remainslocalprotected, notformalresult.

NEXT exact18path backup, then inspect native token-to-adapter/slot execution
observation boundary using existing22requests/4contentclasses/checkpointslot
evidence; one minimal falsifiable identity diagnostic, not repeatedlooseD63
probabilitytest. Performance question remains selected-source preparation
queue vsactualwork; doNOTspeculatively rerunFull or repeatD176/D184/D202.
7Bnumeric/commonwarm/Resident/old-newG1G2 OPEN, then3B,thenbaselines.
All M1M2/A1–A5/S1–S13 retained; no goal/modelcompletion.

## D204 preparation history — superseded by COMPLETE above

2026-10-03 continuation. Full Plan/status/V1 read; analyze-results, vLLM and
optimization reference, run-experiment and academic-plotting read. No subagents.
Latest once-only remote cache approval already fulfilled by D78/D80; no rebuild,
remote operation or GPU replay. D203 remains COMPLETE/SEALED/BACKED below.
Only bounded D196 analysis wrapper adapted to D203, adding retained timing-outcome
and output-hash/adapter-identity joins against D195; no raw log reprojection.
Current official vLLM0.30 reproducibility/batch-invariance and LoRA kernel source
read: default output reproducibility is not guaranteed; NOT proof of the cause
of126differences. No backend flag, numerical tolerance or SLO changes.
NEXT one CPU-only3/4GiBswap0CPU2,3,26,27 analysis -> cleanup -> exact table and
interpretation -> scoped backup. New GPU/build below150GiB remains disallowed;
latestdisk160098562048B, availablememory112895528960B, GPU/scopesempty.
All7Bnumeric/commonwarm/Resident/old-newG1G2,3B and later matrix gaps OPEN.

## Previous checkpoint — D203 ordinary 7B Full COMPLETE, analyzed, SEALED and BACKED

Backup verified2026-10-03 20:46+08:
81984d41240b09b8d3615ecf4e4b18fa204cbc28. Push46638 CLOSED0;
independent ls-remote51034 CLOSED0 exactmatch.16explicitpaths/103staged+archive
entries passed scope/secrets; bundle checksum and diff(cr-at-eol)PASS.
Usermanifest/unrelated work excluded; no staged files remain. No live GPU,
experiment scopes/tmux/tool sessions; remote remains stopped. No need to repeat
D203 qualification, analysis or backup. Lastdisk160103985152B; fresh gate before
newheavy, currently below150GiB. Goal ACTIVE, no model acceptance.
NEXT retained-data current-stage/remaining-miss and output-identity audit with
history/current official implementation; one falsifiable hypothesis before
new optimization/replay. SevenB must satisfy actual G1/G2 old-new target before
3B/baselines. All numerical/reference/legacy comparison and full matrix gaps
below remain OPEN; D203 merely improved over D195, not final acceptance.

2026-10-03 20:45+08. Goal ACTIVE/current turn PROGRESS. Same D203 Full ended,
all cleanup/analysis complete; no GPU/remote/tmux/analysis/tool sessions live.
FULL Plan/status read for this task; V1 FULL reread before comparisons; hashes
unchanged. monitor-experiment/analyze-results/run-experiment/academic-plotting/
github-sync read; no subagents. No serving/code/threshold/configuration edits.

D195 -> D203 (one development replay each, NOT causal/CI/model acceptance):
meanTTFT2.290592642511532->1.7416089653320668s; P955.207960044150241->
3.8535859030671418s; P998.059697074699216->5.9235012114513665s;
meanTPOT40.26261691433204->38.58740864479467ms;
P95TPOT66.60925991794404->63.038562519337844ms;
U15915.263361512974->15926.245593711035GPU-s (+.069%, no resource saving).
Same provisional D170 timing joint3476->3711/4000=92.775%, +235/+5.875pp;
still89short3800. TTFT-only205/TPOT-only75/both9/failure0. n_correct UNKNOWN,
commonwarm/Resident not frozen, NOformalG1G2 or legacy-Prime superiority.
15nonoutputfields identical;3874outputhashsame/126changed OPEN. Dispatch4000
complete/conflicts0:GPU1189/HOST1802/NVMe873/remote136. AlltimeidentitiesPASS.
Source4337reads/4145collections/16511RPC/192joins/274stale rejects.
Ownedplanner2797complete/708.411902008CPU-s (+sideeffect); init2complete.
Meanpreengine1.447966992s;161requests' preengine span alone exceeds threshold.
Retain D202 candidate; still 7B UNACCEPTED. No blind Full without new hypothesis.

Successful offline scopes3/4GiBswap0CPU2,3,26,27, terminal events/swap0,
identity/path absence verified:
- metadata5:97583 CLOSED0/894e5a9fe73a4ec1841af9c5e762c0d9;
- projection:73044 CLOSED0/327728052db74524a778ee2e4d004bf6;
- curation:7872 CLOSED0/c68e6246a4f7402b96caace77c78d86d;
- occupancy:38953 CLOSED0/12bf3659a50a45f48de1667b533ec027;
- timing:71332 CLOSED0/ab9e76eb3f2f42458b33b81403d859d1;
- evidence:10665 CLOSED0/a8d302de3707458a859d40c2fef39d1e;
- bundle:23272 CLOSED0/0d77932f0b724b009054fcaee867c6a9.
Metadata4 size-guard failure retained as below, not GPU failure; no rerun of
projection or inference. One template-preparation assertion and a doc patch
context mismatch occurred before writes; corrected. One bundle receipt read
used wrong cwd, corrected; checksumPASS. No additional measurement failure.

Doc D203_7B_ORDINARY_FULL.md exacttable and caveats SEALED; do NOT edit after
verification binding. Curated8b92e298f7093295a5b062f45e8502d4c68b9d71e93aa606049116be044763e3;
requestprojection34389917B SHA3e5250e0b34c152f06e7c520bd84469cfa483e2f6414da5d428f848929ffc8cc;
verification10149d2e36f6b3690c09f14cf5a60f43c5c86be28eebe05210610c581177c1be;
87member54106B bundlebe6cdb5f7e2d651e480f5ae0b3b92e6c2a9eff1a6a35fa4a118d25cf6f8fa5e0.
548frozenrefs/147protected PASS; current analyzer matches D199 qualification,
all old helper ASTs and timing tests unchanged. Prior31/71 tests reused and
actual D203 full-data checks passed. No new production tests needed.
NEXT exact16path scoped backup, then retained-data remaining-control/identity
audit against history + official source; one hypothesis/minimal validation.
No repeated D200/D201 or same optimizer/Full without new evidence. Keep
7Bacceptance->3B->baselines and allM1M2/A1–A5/S1–S13 outstanding. Latestdisk
160106127360B BELOW NEWheavy150GiB; fresh resource gate/reclaim audit required
before any GPU/build, not a reason to relax floors or delete unique evidence.

## D203 collection history — superseded by COMPLETE above

2026-10-03 20:29+08 analysis continuation, goal ACTIVE/PROGRESS.
Metadata4 session89280 CLOSED1 after successful stream projection exceeded
128MiB input-size guard (173117160B). No Python collector ran, no OOM evidence;
jq exit0/peak728448KiB. Original wrapper/source/output retained; actualscope
4c10f0f1e1cc4514967b4e459657dcf9 absent; final cgroup counters unavailable.
Metadata5 reuses same projection, no reprojection/GPU rerun. New per-run
192MiB cap justified by previous same-schema RSS and conservative expansion;
3/4GiBswap0 unchanged. Exact two-field control/quarantine projection817902B
keeps shared analyzer128MiB guard unchanged; parent full evidence retained.
97583 CLOSED0, actual894e5a9fe73a4ec1841af9c5e762c0d9, peak610660KiB/events-swap0;
cleanup_metadata5_full1.sh verified scope+path absent/GPUempty.
Preliminary4000success/native contracts; fourleasesreleased;
U15926.245593711035GPU-s,132UUID/allpublished/131480060wireB/
3300789780logicalB/requestpacking0/no mismatches. 4063watchdogsamples,
servicepeak20148785152B/minhost92553940992B/events-swap0/warnings0.
Numerical/SLO/performance qualification remains OPEN.
Request projection LIVE session73044,
primelora-d203-full1-project-20261003.scope actual327728052db74524a778ee2e4d004bf6,
3/4GiBswap0CPU2,3,26,27. Existing D96 streaming jq unchanged.
NEXT wait SAME projection->prepared cleanup_projection_full1.sh->existing
curation/occupancy/timing->table->backup. No optimizer or new GPU run.

2026-10-03 20:22+08. Current turn PROGRESS after verified wait. SAME Full ended,
4000terminal/4000success/4000native-contract, error_types empty; numeric correctness
and formal SLO/G1G2 still UNKNOWN. launch pass/service-replay-watchdog0,
native_gpu_context_release_confirmed/service_path_removed true. tmux absent,
exact service/aux invocation and cgroup paths absent; six recorded PIDs absent,
GPU compute empty. Local cleanup receipt recorded20:19:30. No new GPU replay.
Only after local release, exact-owned remote3B/7B/monitor stopped; stop21985
CLOSED0, allinactive/MainPID0. Journal found by frozen healthclock:
transfers-258535c3794b42d4bdb15bcaadab0c72.jsonl. Copy31392 CLOSED0; localjournal
SHA342e454b70324433f179729283a5a0ee95b602c717346dc26229ee26ab5cfa1a,
monitor35a46797fcf41ff801a4f2241992fb763b0a33402aca065581a3c38d5a0c1052.
No remote management during inference; no cache regeneration.

Plan/status already FULL read for continuous D203 task; hashes unchanged,
V1 FULL reread before analysis. Existing D195 metadata/request projection,
curation/failure/occupancy/timing wrappers adapted only to D203 paths/identities
and previous D195 comparison; no serving/formula/configuration change. Metadata
scope primelora-d203-full1-metadata4-20261003.scope LIVE, session89280;
3/4GiBswap0CPU2,3,26,27, tinyfixturesPASS. Source main_outcome522819369B;
streaming projection before bounded JSON load. All original sources retained.
NEXT wait SAME analysis; verify cleanup; bounded request projection -> full
curation/occupancy/timing -> exact table/interpretation -> scoped backup.
No further optimizer/3B/baseline before D203 closure. Lastdisk160405495808B,
NEWheavy150GiB not met; CPU bounded analysis allowed, safety floors unchanged.
All numerical/commonwarm/Resident/old-newG1G2 and later matrix remain OPEN.

## D203 live-monitor history — superseded by TERMINAL above

2026-10-03 20:07+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. FULL Plan/status/AGENTS/V1 read after compaction; unchanged SHA.
monitor-experiment/analyze-results/run-experiment/academic-plotting read;
grouped skill output truncation repaired by separate missing-span read.
Latest3563terminal/3563success/3563native-contract/error_types empty, still
partial and NOT numeric/full/SLO/performance acceptance. Same live tmux,
service/aux invocation IDs and four native GPU workers service-owned.
Watchdog3356/monotonic536472.716791287:service20003020800B,
hostavailable95990538240B, events/swap0/no warning/abort/foreign/escaped.
Disk158055075840B: running100GiB safe, NEWheavy150GiB not met. No new
experiment/optimizer, large-log analysis, remote management, cleanup or live
tool session. D195 existing bounded metadata/request projection, curator,
timing/occupancy and evidence wrappers read only; 15 templates also cached
in functions store d195_scripts, NOT adapted or executed for D203 yet.
Once-only readonly delivery-cache approval already fulfilled D78/D80; reuse,
no regeneration. NEXT continue SAME Full; confirmed terminal/local release
-> prepared exact remote stop/copy -> adapt existing bounded D195 analysis
-> full table/interpretation/backup. All acceptance and later matrix gaps OPEN.

2026-10-03 19:56+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. Same live tmux/service/aux IDs and four owned native workers.
Plan/V1 SHA unchanged from full task read; monitor-experiment reread. Latest
3040terminal/3040success/3040native-contract/error_types empty. Still partial,
not numerical/full/SLO/performance acceptance. Watchdog2749/monotonic
535857.757065129:service19541774336B, hostavailable94885326848B, events/swap0,
no warning/abort/foreign/escaped. Disk158530756608B; running100GiB safe,
newheavy150GiB not met. No new experiment, code, analysis, remote management,
cleanup or live tool session. Continue SAME Full, then local terminal/release,
prepared exact remote stop/copy, bounded existing analysis/table/backup.
All numerical/reference/old-new/G1G2 and remaining matrix gaps stay OPEN.

2026-10-03 19:49+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. Same tmux/service/aux identities and four owned GPU workers live.
Plan/V1 SHA unchanged from full task read; monitor-experiment reread. Latest
2517terminal/2517success/2517native-contract, error_types empty; NOT full,
numeric, SLO or performance acceptance. Watchdog2317/monotonic535420.08886423:
service19164942336B, hostavailable95112548352B, events/swap0/no warning/abort/
foreign/escaped. Disk158814269440B, running100GiB safe/newheavy150GiB not met.
No new experiments/code/analysis/remote management/cleanup/tool sessions.
NEXT keep this exact replay, then verified localrelease -> prepared remote
stop/copy -> bounded D195 analysis/table/backup. All acceptance gaps remain OPEN.

2026-10-03 19:43+08 same-task monitor continuation; previous/current VERIFIED
WAIT, goal ACTIVE. Plan/V1 hashes remain unchanged from full task read;
monitor-experiment reread. No new task, optimizer, analysis or remote changes.
Latest2024terminal/2024success/2024native-contract, error_types empty; still
partial, not numeric/full/SLO/performance acceptance. tmux tc_d203_full1 and
exact service/aux IDs below active; same four GPU workers service-owned.
Watchdog1958/monotonic535056.356041721:service18536136704B, hostavailable
95256092672B, events/swap0/no warning/abort/foreign/escaped. Disk159137865728B,
running100GiB safe/newheavy150GiB not passed. No live tool sessions, no cleanup
or extra workload. NEXT continue SAME run to terminal and verified release;
then prepared remote stop/copy -> bounded D195 analysis/table/backup. All
7B acceptance/3B/baseline/formal/ablation/sensitivity work remains OPEN.

2026-10-03 19:37+08 ongoing-monitor continuation, previous/current VERIFIED
WAIT; goal ACTIVE. Same live tmux tc_d203_full1 and exact service/aux invocation
IDs below, no new replay or production changes. Plan/V1 SHA unchanged from full
read of this continuous task; monitor-experiment reread. Latest1525terminal/
1525success/1525native-contract/error_types empty. Partial only, not full/numeric/
SLO/performance acceptance. Watchdog1600/monotonic534693.630221785: service
18148175872B, hostavailable95432777728B, events/swap0/no warning/abort/foreign/
escaped. Four same native GPU PIDs still service-owned. Disk159427059712B above
running100GiB but below newheavy150GiB; no extra heavy work or cleanup. No tool
sessions or remote management. NEXT same replay monitoring; ONLY after local
terminal/release execute prepared exact-owned remote stop/copy, then bounded
D195 analysis adaptation/table/backup. No baseline/3B resumed; all gaps OPEN.

2026-10-03 19:30+08 ongoing-monitor continuation: previous/current turns
VERIFIED WAIT on same live handle; goal ACTIVE. Plan/V1 SHA rechecked unchanged
from full read in this continuous monitoring task; monitor-experiment reread.
No new task/replay/optimizer/analysis/remote management/cleanup/backup. Latest
1069terminal/1069success/1069native-contract, error_types empty; partial only,
numeric/full/SLO/performance acceptance NOT established. Same four GPU PIDs
and service/aux invocation IDs below, both active; tmux tc_d203_full1 alive.
Watchdog1186/monotonic534274.213385683:service18105544704B, hostavailable
95398572032B, events/swap0/no warning/abort/foreign/escaped. Disk159549440000B:
above running100GiB but below NEWheavy150GiB; no other heavy work. No live tool
sessions. Continue exact Full; terminal->local release->remote stop/copy->bounded
existing analysis/table/backup. All 7B/3B and later acceptance gaps remain OPEN.

2026-10-03 19:22+08 continuation: previous turn PROGRESS (qualified ordinary
Full launched); this turn VERIFIED WAIT on the same live tmux/service/aux.
Full Plan/status and monitor-experiment read, no truncation. No new replay,
optimizer, analysis, remote operation, cleanup or backup during measurement.
Only D195 exact local cleanup wrapper adapted to current identities, bash-n
PASS, NOT executed. analyze-results skill inspected; no results interpretation.
Live sample696/monotonic533777.731856753:517terminal/517success/517native
contracts, no error_type. Numeric correctness and full completion UNKNOWN;
do NOT label this partial population as performance acceptance. Four native
PIDs2744241/2749026/2749402/2749657 still service-owned. Service/aux invocation
IDs below unchanged; both active, tmux tc_d203_full1 alive. Service17176203264B,
hostavailable95960543232B, events/swap0/no warning/abort/foreign/escaped.
Disk160102776832B, running100GiB floor safe, NEWheavy150GiB not passed.
No live tool sessions. Remote remains untouched during this run. NEXT continue
this exact replay -> confirm local terminal/releases -> exact remote stop/copy
-> bounded existing D195 analysis/table/backup. Goal ACTIVE, all later work OPEN.

19:13+08 monitor: sample168/monotonic533242.800717055,46terminal records;
NOT complete/native/numeric/performance acceptance. Four actual GPU workers
2744241/2749026/2749402/2749657 all service_member=true, CPUs4–23,28–47.
Service15317975040B/hostavailable97029894144B/events-swap0/no warning/abort/
foreigncompute/escapedworkers. tmux alive. Disk160672710656B below NEWheavy
150GiB but above unchanged RUNNING100GiB floor; no other heavy work allowed.
No live tool sessions; only this GPU replay/aux and owned remote services live.
No remote management since health/start; cleanup scripts are prepared but
must NOT run until confirmed local terminal and physical release. Continue
monitoring this exact run, no second replay/optimization/qualification.

Launched2026-10-03 19:10+08 in tmux tc_d203_full1. Raw
results/ieee_tc/p2_backend_qualification/d203_20261003. Existing D195 ordinary
4000 config unchanged except owned paths; diagnostic prefix/frame/control OFF.
Prelaunch0586567f3ea5c862ed2e49e7180804f1de6458fc6c34e46c17e2a122a7b28a63:
548source references/147protected/resource pass; session60778 CLOSED0.
Prelaunch actual13a35835166d4e418e7ea1f7ab6804e4; health actual
3fc0bb56b64d41cea35488d238c44f97 immediateCLOSED0. Both scopes absent.
Service primelora-tc-svc-b45ff477137e448999b428d39a8e14f0.scope,
invocationbcd562dc2b5148a59aed8c38d466e933/inode16336682,72/80GiBswap2.
Aux primelora-tc-aux-7fb06b5a677d4b62b58776aefd5653ac.scope,
invocation85dc6c7dc6fa4137b23d912186204f66,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice533073.53751656/businessstart533133.53751656. Native GPU
worker2744241 observed; verify all scale-out workers from live watchdog next.
Remote ACTIVE, both NIC1000/full and health checked before measurement:
3Bpid1536346/id987612c9cfd2405eadc6e096b071535d;
7Bpid1536348/ide8904853bdeb472595e5a9f54718339c;
monitor-d203full1pid1536351/id0845c03e49ad4254ab473f32ef132b61.
7Bclockremote-process-monotonic:a211cea09c844dcfa7e40d11d0a5ef83.
Exact stop/copy scripts prepared, NOT executed; only after local terminal and
physical release. No remote management/cleanup/hash scans during replay.
NEXT monitor actual workers/resources -> existing cleanup -> bounded D195
analysis adaptation -> exact table/interpretation -> scoped backup. No result
or model qualification yet. All later acceptance/matrix gaps remain OPEN.

2026-10-03 19:09+08 preparation history. Full Plan/status/V1 reread; grouped output truncation in
status tail corrected by explicit bounded reread. run-experiment/monitor-experiment
read; no subagents. D202 checkpoint below already backed; no repeated tests or
optimizer. Reuse D195 ordinary4000 config/launcher/preflight with only fresh
d203_20261003 output paths and current source/evidence identity. Prefix and
detailed frame/control observer OFF. Candidate/evidence HEAD2fc3630d.
Latest census GPUempty/no experiment scopes/tmux; remote was stopped. Disk
161182912512B and hostavailable112587755520B, fresh prelaunch still required.
User once-only cache permission already fulfilled D78/D80, not regenerated.
NEXT bounded prelaunch + owned remote activation/health, then ONE ordinaryFull;
monitor actual workers, retain failures, cleanup->table->backup. No performance
claim; all numeric/commonwarm/Resident/old-new/G1G2 and later work remain OPEN.

## Previous checkpoint — D202 physical HOST representation CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 19:00+08:2fc3630d684e73f5bfbae1b49f1ac937488444bf.
Push69680 CLOSED0; independent ls-remote44698 CLOSED0 exactmatch.10explicitpaths/
114staged+archiveentries scope/secretPASS; checksum/diffPASS. Usermanifest and
unrelated files excluded. No live GPU/tmux/analysis/tool sessions; remote still
stopped. Lastdisk161197293568B, freshgate before newheavy. NEXT ordinary4000Full
below; backup DONE, no need to repeat qualification or add an optimizer.

2026-10-03 18:59+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read; Plan/status reread before qualification (one grouped output truncated,
missing span then reread in bounded chunks). Plan/V1 hashes unchanged.
optimize-for-gpu, vLLM+optimization reference, run-experiment, analyze-results,
academic-plotting and github-sync fully read; no subagents. User once-only
delivery cache authorization already fulfilled D78/D80, not recreated; no
remote operation or GPU replay this turn. Parent5fadd45fb6a8be47c7a6e39ed65d8b1f4f2ecc4d.

ONE candidate: five physical HOST consumer calls use existing graph-only
projection. All current storage/alias/dtype/pinning/staged/allocator checks,
GPU pool_tensor_views, owner, admission, equations, config and return isolation
retained; isolated worker observation/default helper full detail unchanged.
Official vLLM0.30 UniProc/model manager/CPU optimization read; D144/D174/D194
history and direct dynamic consumers audited. No stale cache/extra concurrency/
timeout change. Runtime change only gpu_monitor.ieee_gpu_reference plus helper
documentation; other runtime AST and owner/planner/admission source unchanged.

Rawresults/ieee_tc/p2_backend_qualification/d202_20261003. AllCPU3/4GiBswap0,
CPUs2,3,26,27; terminal events/swap0, scopes+paths absent/GPUempty:
- tests1 79run/2fail1error, session95372 CLOSED1,
  actualc8f34f31684140e988af5708199d6640. New fixture missed configure call and
  valid same-observation length state; tests corrected only, runtime unchanged.
- tests2 79PASS .047s, session27662 CLOSED0,
  actual7cc5a12ca100432fa248b6e58cca017f.
- regression1 1122PASS156.731s, tmux ended,
  actual1a24cf33f08e4995aa845074942a9113; peak2052943872B. Groups overlap.
- probe1 fixture namespace error before measurements, session38069 CLOSED1,
  actuala807c3ac302d4df9b0a0aa7a47d9f6db. Exact parent globals lacked mocked GPU
  pool observer. Original script retained; probe2 explicitly shares fixture.
- probe2 PASS, session74384 CLOSED0, actuald286e5a9132f4d878dfc5302e137a3f1.
- curate1 PASS, session84672 CLOSED0, actuale750606de0dd484b9793150f7097b7e9.
All failure/source snapshots retained. No production safety check weakened.

7exact-parent extension tinyCPU cases preserve all other fields/errors;
same pinningcounts6(nonempty)/14(stagedalias), description queries0. Four retained
D169 physical snapshots exact NativeSourceSnapshot/allocator equal. JSONbefore/
after1388807/550023,2025691/766427,3301339/1199835,3936172/1415212B.
HOSTonly deepcopy ms26.140382/9.928758,75.266975/14.411368,
96.688978/60.121510,111.970420/30.927552; all24blocktimings preserved.
CPU Torch2.8/three alternating blocks ten copies each, not CUDA/RPC/wallrequest,
no n3workload CI/performance acceptance. Retained-state source SHA verified.
Doc D202_PHYSICAL_HOST_REPRESENTATION.md + table SEALED;147protectedPASS.
Curated0590051fcffff3eaaed25eb107789a95a51923d49db6deccbcb909ffbb3296ae;
62member1017908B bundleb42256a2ab62dd9a266dc5e2ffc962b9943a2ce9fa783c7a323a56d024007f31.
Lastdisk161210089472B (~150.14GiB), freshgate before newheavy. No cleanup/install.

NEXT exact scoped tested backup, then ONE ordinary7B4000Full same D195/D157cap4
config/protocol/profile, fresh owned dirs, diagnostic observer OFF. No repeat
D200/D201 or another optimizer. Follow existing ordinary launcher/prelaunch/
cleanup/analysis/table. Numeric correctness/128hashchanges/commonwarm/Resident/
old-newG1G2 still OPEN. 7Bacceptance→3B→baselines; allM1M2/A1–A5/S1–S13retained.

## Previous checkpoint — D201 native-wait association COMPLETE, SEALED and BACKED

Backup verified2026-10-03 18:35+08:5fadd45fb6a8be47c7a6e39ed65d8b1f4f2ecc4d.
Push58926 CLOSED0; independent ls-remote34944 CLOSED0 exactmatch.8explicitpaths/
19staged+archiveentries passed scope/secret checks; checksum/diffPASS. No
serving edits; CPU intervalfixtures and actual retained-data analysis passed.
Usermanifest/unrelated files excluded. Both analysispaths absent/GPUempty;
no live tool sessions/remote service. NEXT one audited physical-HOST
representation candidate below; no need to repeat D200/D201 or backup them.

2026-10-03 18:33+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread; unchanged. analyze-results/run-experiment/academic-plotting/vLLM/
github-sync read; no subagents. User once-only remote delivery cache permission
already fulfilled D78/D80, not regenerated. No GPU replay/remote operation,
serving change or optimizer added. Current runtime still D198 diagnostics,
last ordinary Full D195. Latest evidence HEAD before backup1852c43.

Official vLLM0.30 core/UniProc/client and local call-chain audited. Synchronous
utility executes in core before returning Future; client async does not move
owner work off-thread. Same-thread Prime preparation/inventory can occupy the
query path, not proof all wait is queueing or native inference bottleneck.
Reused exact SHA-sealed D200 bounded frame reader/control qualification;
no new framework or big-log projection. Capture interval must entirely fit
[frontend_native_send,native_begin); process incarnation and main thread match.
159 unique process/sample pairs cover226/4204calls; 1boundary-ambiguous capture,
0missing/truncated in-intervalmain. HOST deepcopyline940 occurs48times
(11/12/10/15); HOST pointercheckadd103 occurs59times. NOTCPU/wall percentages,
not latency attribution/critical-path savings. Full counts and allotherframes
retained; unsampled calls not zero. 8boundaryfixturesPASS.

Association1 session32248 CLOSED0,actual85a69431dba240a883b2ebc9ff29518d,
3/4GiBswap0CPU2,3,26,27,peak109821952B/events0/scope+pathabsent.
Curate1 immediateCLOSED0,actualaf02bac61f2744e8875d216a63695e30,sameenvelope,
events/swap0/scopeidentityempty. No live jobs/tool sessions.147protectedPASS.
One read-only receipt command used wrong cwd; corrected; checksumPASS and
no analysis failure/artifact rewrite. No production tests needed (no edits).
Doc D201_NATIVE_WAIT_ASSOCIATION.md + primary exacttable SEALED.
Curatedc55ee1c3190a26724240d35ab21847187b0d314d6fc7d252bab0d54d6152604d;
verificationf8ed73d05081aa8da95fb2cc62300c35ed023584213b34bdbb885b6897bcd2dc;
11member31930B bundleccb5c2788b2a97aae291a1920d925831576bf7b810f1d3beb104391685c092eb.
Rawresults/ieee_tc/p2_backend_qualification/d201_20261003. Disk161228591104B,
freshgate before heavy task. No cleanup/deletion or environment install.

NEXT exact scoped backup, then ONE physical-HOST-consumer representation
candidate. Auditdynamicconsumers; keep fresh fullallocation/alias/dtype/pinning/
staged/allocator graph, GPU pool_tensor_views, owner/admission/formulas and
D144 mutable-return isolation. Existing host_tensor_views has no explicit
runtimeconsumer outsideproducer; remove only proven unused work, not physical
checks or stale-cache shortcut. Reuseactualsnapshots/tinyfixtures then ordinary
4000Full if qualified. D174 applied onlyrouting; D144 duplicate-pin/traversal,
D181targetscope/D194planning alreadydone, don'trepeat. No speculativeFull,
off-threadowner mutation, capacity/timeoutincrease or D197encoderrevisit.
SevenBnumeric/128hashchanges/commonwarm/Resident/old-new/G1G2remainOPEN;
7Bacceptance→3B→baselines; allM1M2/A1–A5/S1–S13retained.

## Previous checkpoint — D200 detailed 7B prefix COMPLETE, SEALED and BACKED

Backup verified2026-10-03 18:14+08:1852c43aa55b89923e23c591ecf68137e3d0c965.
Pushsession97990 CLOSED0; independent ls-remotesession81035 CLOSED0 exactmatch.
9explicitpaths/90staged+archive entries passed scope/secrets; usermanifest and
unrelated files excluded. Bundle checksum PASS; standard CSV CRLF triggered
plain git whitespace diagnostic, rerun explicit cr-at-eol check PASS with other
whitespace rules retained; sealed CSV bytes NOT rewritten. No production change
or new smoke needed; existing qualified analyzers passed actual D200 data.
All experiment scopes and GPU compute absent, no live tool sessions. Remote
owned services stopped and copied; no runtime to restart. Finaldisk161245675520B.
NEXT official/local vLLM utility scheduling/response audit, one evidence-backed
hypothesis, then minimal qualification; no speculative Full. Goal remainsACTIVE.

2026-10-03 18:11+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread after compaction; hashes unchanged. monitor-experiment/analyze-results/
run-experiment/academic-plotting/github-sync read. No subagents or optimizer.
Latest once-only cache permission already fulfilled D78/D80; reused, not rebuilt.

D200 ended:1000/1000 terminal/native-contract successes, zero failures, numeric
n_correct UNKNOWN. launch pass/service-replay-watchdog0, physical release true;
launchSHAc8f083b721368eb9fa0e5bf4908a81b29a9ef3d64127a68a6e3f691557577d03.
Four physical leases/4452.071693249978GPU-s. Exact local service/aux paths and
ten observed PIDs absent; GPU census empty; tmux ended. Only then exact-owned
three remote services stopped (57398 CLOSED0), allinactive/MainPID0. Transfer
journal transfers-981699c3da6f4a13a75dfb14e95c6f1b.jsonl copied with SHA check
(91954 CLOSED0). 60UUID pairs/allpublished/74953014wireB/1474429899logicalB,
requestpacking0/no byte mismatches. No remote changes during inference.
1191watchdogsamples/servicepeak18999349248B/minhost92882001920B/events-swap0,
no warning/foreigncompute/escaped workers. Remote/local runtime now STOPPED.

Reuse D187 collection/curation, only D200 identities/paths and D199 boundaries
added. No production code changes. Analysis sequential3/4GiBswap0CPU2,3,26,27:
- boundaries1 actual4f58160d4d854163afd2a6d4f5f49f3c/session15587 CLOSED0;
- metadata actual245057c1874643929bb63dd7676a3693/session31782 CLOSED0;
- frames actualb372a59811a14a1cb71db16c25bf77f9/immediate CLOSED0;
- curation actual2750a8509b5443a8aba318f8a342e493/session72138 CLOSED0.
Allterminalmemory.events0; actual identity/path absence checked in seal.
46244boundaryevents/4204attempts ALLcomplete-success, no partial/orphan/cancel/
error; count agrees with source_stats4204RPC. ALLbusiness includingdrain.
Means ms:parenttotal179.796957; frontend-send→native-begin96.328020;
nativewall12.315801/nativeCPU9.520286; native-ready→frontendreceive24.726790;
frontend-ready→parentreceive35.193664. NOT pure queue/transport times, not
request critical-path attribution; fourconcurrentRPCs not additiveasTTFT.
Tenprocessframe populations:controller/planner/fourfrontends/fourGPUcores;
6truncatedstacks/0missingmain/0partial. OccurrencesNOTCPUpercentages.
Controller537businesssamples, nearestproject decode45/fileinventory26/planning17.
No ordinaryFull comparison/n1CI/performance/model/SLO qualification inferred.

Doc D200_CONTROL_BOUNDARY_PREFIX.md and exact-stage Markdown/CSV tables SEALED.
Curated23ffb565ccfb93f74bc4283bca82b56eab417989e4eeb4b4a4626176819e683d;
verification56c032f3f2160c258fc86918ab949ab1647e0bbbee9c2895e4b760c89dee9691.
Bundle81members736572B SHAd4c8a06afaf5a0aab48158deaa9cc54c56db9d77988ee282b5af3844d2aead1d.
147protected/537prelaunchrefs PASS. Full raw request JSON not reparsed; no new
native-timeline/dispatch completeness or E2E identity claim from this prefix.
seal1 session16563 CLOSED1: path cache1 nonexistent (actual cache), before
result artifacts; original helper/log retained, no GPU rerun. Failedscope
c7a68fa362cc4e508f8b633c89185ad4 absent; terminal counters not captured.
seal2 corrected only exact archive path/new attempt, session59160 CLOSED0,
actualbf7bbe7b25954331ad39479e3b7ce662; events0/scopeidentity/path absent.
No live tool sessions. Lastdisk161249439744B; freshgate before any nextheavy.

NEXT backup exact D200 checkpoint, then read actual vLLM0.30 utility scheduling/
response + Prime caller against official docs/code and historical optimizations;
one falsifiable hypothesis before any serving change. D197 summary production
still DEFERRED; no repeat serializer study/stale confirmed-state shortcut or
speculative Full. SevenB numeric/commonwarm/Resident/old-new/G1G2 gaps OPEN.
7B acceptance→3B→baselines; entire M1M2/A1–A5/S1–S13 still retained.

## D200 launch history — superseded by COMPLETE above

17:46+08 last read:34terminal records,968control-event lines across tenprocess
files (one startup-only process has0events); not complete or success acceptance.
watchdogsample152: service14967316480B/hostavailable97511907328B/events-swap0,
no foreigncompute/escaped/guardabort. tmux live. Prior D199curate/D200cache1/
cache2audit-apply/prelaunch/health scopes all not-found/identity+path empty.
No live tool sessions; ONLY GPUdiagnostic+ownaux+three remote services active.

17:45+08 actual worker check: watchdogsample108,monotonic527952.964302317;
GPUcomputePIDs2364655/2368679/2369096/2369310 all service_member=true and
affinity4–23,28–47. Service memory17061773312B/hostavailable95008206848B;
memory.events all0/swap0/guardwarningfalse/no abort. Actual workers verified,
not merely launcher. No analysis/remote management during live measurement.

2026-10-03 17:44+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read; full Plan/status reread before GPU launch (large grouped output truncated,
801–1537/status reread separately to EOF). Plan/V1 unchanged. No new optimizer.
D199 backup c75ef3750b4fc739f9351f713578fb2bbe9f3ca1 verified below.
D200 ONE detailed diagnostic launched in tmux tc_d200_prefix1; raw
results/ieee_tc/p2_backend_qualification/d200_20261003. Do NOT launch another.
Same D195 config/D157 cap4/60s/source42/fixedoutput/1800s; only prefix1000 and
three fresh owned paths differ. D198 boundaries + existing 2s frame observer.
Prelaunch4e04fad348e3e3f4c35c86870ac59a35592622cb9e8a1abdeff825b796d02362,
537refs/147protected PASS; prelaunchsession60394 CLOSED0, health CLOSED0.
Actual service primelora-tc-svc-5c3eb2f2a2c14d249e224f5311a2b94e.scope,
invocation03661341910f44188c81ffede5224537, inode16281283, 72/80GiBswap2.
Aux primelora-tc-aux-dfbd3ac771f348919fc489b5daaba34f.scope,
invocationb7721e1d3b674280a2825cbf4079c77d,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice527844.502615776/businessstart527904.502615776.
Initial live samples: no guard events/foreign compute/escaped workers; GPU
workers still starting at this record. Need verify actual GPU processes next.
Once-only D78/D80 remote cache reused, no repacking. BothNIC1000/full prelaunch.
Remote services ACTIVE: 3Bpid1451251/id2bfcb78c0818438daa6d1a03cc42d170;
7Bpid1451253/id14ae44c2bd5f461db9227a961fe3cc57;
monitor-d200prefix1 pid1451256/ide5020271cb6c4780905090375efacf15.
7B healthclockremote-process-monotonic:d79dce1802504d40b319489cb4591e6f.
Exact remote stop/copy scripts prepared, MUST NOT execute before local terminal
and physical release; no remote operations, hash/cleanup/config change during run.
Cache2 audit/apply completed49806/45551 CLOSED0; identities
a10851c61fa24ff1b0218eab094252ba/c8d4d676fea74ce3ba990a2ce9ba76d4.
52publicwheelarchives/104files875450368allocatedB reclaimed after exact public
SHA/size/HEAD/reference/ownership/openfile checks; fourtrackedprojects unchanged,
installedenvs/models/results/custombuilds untouched. Both3/4GiB/events-swap0.
AuditSHA706b48b37ff17689f204bc15ccc16667cbce6cb2045f41665e09db264164e06d;
newlaunchdisk161602117632B above150GiB; running100GiBfloor unchanged.
Cache1 read-only9archive audit not applied, superseded. No stale deletion retry.
NEXT monitor -> exact localrelease -> exact remotestop/copy -> bounded D199
analysis + existing stack/native terminal analysis -> table/interpretation/backup.
No claim of current completeness, numericalcorrectness, SLO or performance gain.
7B acceptance -> 3B -> baselines; all later matrix retained. D200doc DRAFT.

## D199 completed — offline control-boundary analysis CPU QUALIFIED and SEALED

Backup verified2026-10-03 17:36+08: c75ef3750b4fc739f9351f713578fb2bbe9f3ca1,
pushsession51612 CLOSED0, independent ls-remotesession28448 CLOSED0 exactmatch.
9explicitpaths/42staged+archiveentries passed scope/secret check; checksum/diff
PASS. First checksum invocation used wrong cwd (read-only missing file), rerun
in bundle directory PASS; no evidence or test failure. D199 doc/results sealed.
D200 prep history: cache1 read-only10–30MiB audit ended96663 CLOSED0; only158302208B
eligible under previous package allowlist, not applied. Fresh metadata inspection
identified additional exact public scientific/build-tool wheel names; cache2
read-only audit started with expanded explicit names, same SHA/size/HEAD/ref/
ownership/open-file checks. No GPU/remote changes. Do not apply stale cache1.

2026-10-03 17:34+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread; hashes unchanged. analyze-results/run-experiment/academic-plotting/
github-sync read. Latest once-only remote cache permission acknowledged as
already fulfilled D78/D80; no recreation, no remote operations this turn.
Existing analyzer only: strict bounded ID/clock/process/role/ordering joins,
Type-1 complete-success summaries, explicit missing/error/cancelled/orphan and
partial-last-line counts. No serving/formula/SLO/configuration change. No
performance gain inferred; no GPU launch. D199 doc and curated evidence SEALED.
Tests1 31PASS .234s, actual98860922cc054a6aa5835a1e6cd84d8e/session86164 CLOSED0.
Regression1 71PASS .357s, actualad9f576254a844a4a8664ef676563cd5/session97731 CLOSED0.
17newtests; test groups overlap. Both3/4GiBswap0CPU2,3,26,27/events0; scope/path
absence verified by curation. Source archives retain exact test wrapper history.
Curate1 actual97257bc145c5498fa68e5c9cc77922c4/session40264 CLOSED0,events/swap0.
147protected/PlanV1 unchanged; existing analyzer bodies AST unchanged; only one
offline script changed in runtime tree. CuratedSHA
48a9a62805f40ded7a86979b7e06066eb07b1c8fbde98c7b89b37e204009d7f1.
24member79329B bundleSHA299c0c16fc0ba5b2b7dec429daecede04e266928798e17b70f960de0f14c36c6.
NEXT exact scoped backup, audited disposable-cache recovery/fresh gate, then
ONE current-source1000-prefix detailed diagnostic. Disk160749322240B last
check is below NEWheavy150GiB. Do not relax floor, repeat D198 tests, or run
speculative Full/new optimizer. 7B acceptance -> 3B -> baselines remains order;
all numerical/commonreference/old-new/G1G2 and full matrix gaps retained below.

## Previous checkpoint — D198 control-boundary diagnostic CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 17:15+08:
79dc8f12668c7d255846e97c7a461c95e8d580d4; pushsession52715 CLOSED0 and
independent ls-remotesession95840 CLOSED0 match. Exact13paths/70staged+archive
entries passed scope/credential checks; checksum/AST/shell/diff checks PASS.
Usermanifest excluded. D198 sealed doc/results unchanged. Final GPU census
empty; no experiment scopes/tool handles (unrelated podman untouched).
NEXT strict existing-analyzer grouping qualification below, then one detailed
prefix after fresh resource gate. No new optimizer/Full or repeated D198 tests.
Disk160754200576B below NEWheavy150GiB; no floor relaxation or data deletion.
Goal ACTIVE/current turn PROGRESS; all model/G1G2 and later work remain OPEN.

2026-10-03 17:13+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read after compaction; Plan/V1 hashes unchanged. optimize-for-gpu, vLLM and
its optimization reference, run-experiment, academic-plotting, github-sync read.
No subagents/independent semantic-review claim. User once-only cache permission
already fulfilled by D78/D80; cache reused, not recreated, no remote operation.
Current HEAD81f998c9c36f98cbbfece56807d87c0d05129a74; last ordinary runtimea0c4a9d.

ONE diagnostic-only change in logger/native worker/runner/dedicated worker:
existing qualified nonformal python_frames_v1 prefix gate now emits read-only
request-source RPC boundaries with one attempt ID/common local monotonic clock.
Default emits no fields/files. No scheduling/owner/physical inventory/formula/
configuration/timeout/SLO change, no new optimizer. Detailed code and caveats in
D198_CONTROL_BOUNDARY_DIAGNOSTIC.md (SEALED; do NOT edit). New10 CPU tests; actual
native physical computation stubbed in loopback, separate real-owner fixtures;
not CUDA/model correctness/performance qualification. Missing/failed/cancelled
events remain incomplete, not filled zero; async CPU deltas not request CPU.

Targeted40PASS49.586s, actual63d3ac2a573843668c7295e326df82f1/session66942 CLOSED0,
3/4GiBswap0CPU2,3,26,27; events/swap0/scopeabsence verified. One premature
read of terminal memory files occurred while tests still active; no experiment
failure or relaunch, terminal receipts later present. Exact source snapshots
retained in raw results/ieee_tc/p2_backend_qualification/d198_20261003.
Regression1 1100PASS156.099s, actual29cd7d6a3c03410687302bdd4caa1613,
same envelope/events-swap0; scope/path absent and tmux ended. No failed tests;
groups overlap, not independent repeats. Exact sources unchanged after tests.
Curation1 actual993191de83674736861148be55ec23de/session62949 CLOSED0,
same envelope/events-swap0/scope absence verified. 147protected/PlanV1/sourceSHA/
allowed-function AST and unchanged owner/planner/admission checks PASS.
CuratedSHA81ce5ed6e4c6de5313b4e62586cb643c3850255b5275593b520442a3549f73b8.
29member1127008B bundleSHA79210323e4539b0c9e026046fdfbd2bb69f71aac65d5ee5c22ae0237a6f924de.
Doc/curated SEALED; do NOT edit after evidence binding. Qualification table
delivered; no misleading speedup graph. No live GPU/remote/analysis/tool jobs.
No CUDA/model/numeric/performance acceptance. Exact13path backup verified above.

NEXT qualify strict bounded offline boundary grouping in
existing analyzer (ID/clock/completeness/order/cancel; no filled zero) -> audited
disk recovery/fresh gate -> ONE current-source1000-prefix detailed diagnostic.
No speculative Full/new optimizer or repeated D198 tests before new evidence.
GPU/build blocked by NEWheavy150GiB disk floor, NOT an overall goal blocker:
160772681728B last read; CPU bounded work allowed; running100GiB/memory floors
unchanged. Actual7B numerical/128hashchanges/commonwarm/Resident/oldPrime-newmetrics/
G1G2 OPEN before3B/baselines. Entire M1M2/A1–A5/S1–S13 retained.

## Most recent retained evidence — do not rerun

- D197 request GPU summary component study SEALED/BACKED81f998c9.
  Eleven cases/66 timing rows/110 negatives/7 CPU fixtures; source states identical.
  Bytes -76.925–95.274%, ten component deltas .557–.636ms, last1.146ms.
  NOT native inventory/actual RPC/CUDA/end-to-end. Production candidate DEFERRED;
  no summary implementation, repeated encoding study or speculative Full.
- D196 control-path retained-data diagnosis SEALED/BACKEDbac5f8f.
  D190→D195 preengine2101.355→1961.691ms (85.641% current meanTTFT);
  routing563.304→519.378ms, residual415.139→414.847ms NOT one RPC.
  Four source queries concurrent, not serial. Fresh control connections, not
  generation channel pool. No source-RPC boundary retention before D198.
  GPU-selected1165 conditional population differs from previous1232; don't
  infer causal improvement from those groups. Native source admission occurs
  before source→handoff1.219ms. Local ingress not remote174HTTP.
- D195 ordinary4000Full SEALED/BACKEDa42f250, runtimea0c4a9d. Complete native
  countcontracts4000/failure0; numerical n_correct UNKNOWN.
  D190→D195 meanTTFT2.433103089→2.290592642511532s, P95
  5.351091139→5.207960044150241s; P997.788593317→8.059697074699216s REGRESSION.
  MeanTPOT39.830671→40.26261691433204ms REGRESSION; P9567.727418→66.609259918ms.
  InternalE2E7.069757720→6.895221204s. PhysicalU15919.807325829→
  15915.263361513GPU-s(-.02854%), no established resource saving.
  SAME provisional D170 joint3375→3476/4000=86.9%, still324short3800.
  All15nonoutputfields identical;3872outputhashsame/128changes OPEN.
  Dispatch4000complete/conflicts0:GPU1165/HOST1773/NVMe928/remote134.
  4520source reads/4218collections/16806RPC/302joins/442stale rejects.
  2196plans=2init+2194owned;2193complete+1cancelled-discarded; CPU565.902328697s.
  132remote UUIDpairs/allpublished/131480060wireB/3300789780logicalB/packing0.
  Four physical leases released; local/remote/analysis jobs CLOSED, doNOTrestart.
  Reusable34407428B projection SHA805a6b8c5aa8f89b7394a4eba8b593f5ada95c6f3048f5a9f2d6c1c06e759697.
  CuratedSHA31594e530d42341c2eb876e18daf2d7a3f39a4c9cb728cd14f3b4bda45da4134.
  Metadata121405834B within128MiBguard; never wholeload249/361MBoriginals.
- D194 planner uses existing complete registered routing_source_snapshot,
  not target-scoped routing wire; qualified75targeted/1070regression, BACKEDa0c4a9d.
- D192 guarded encoder REJECTED; current compact messages show no consistent gain.
  D189 immutable static descriptions already implemented; no static-cache redo.
  D184 pending+GPU frontend pipeline already implemented; D181 target scoped
  routing already implemented. Don't wholesale remove legacy hints: scale-down
  still consumes queue hints. No removed physical invariants or TTL stale cache.

## Verbatim ledger archive

Previous2542-line status preserved verbatim in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D197_20261003.md,
SHA3ff837acaf1d7ca1fe513afbf969fc12fb6034a665253004baefc50d4ae43639.
All old LIVE/NEXT text is historical; CURRENT above supersedes it.
No old docs/results/failed evidence rewritten or deleted.

## Completed history and acceptance gaps — do not repeat

Previous ledger preserved VERBATIM in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D178_QUALIFICATION_20261003.md,
SHA3828059ffcb4822821f60acb11c3c169341e20d4031fc3a20b87d9eb6cfb6eb4.
OldLIVE/NEXT text there is historical, superseded by CURRENT above.
D152 archive d18c53f9e63b75a240dc09aa67badf8f501efc827b11dc56a8f4d48296d4aaa2
and all earlier archives remain unchanged. No old evidence deleted.

D176 held-HOST transaction qualified45targeted/947regression/8parentfixtures,
backedb4b10c1; D177providesperformanceobservation, notmodelacceptance.
D175 table/projection/backed95de3c3; D174 representation/newmeasurements done.
D172/D173 decode and D163/D164 membership qualified but no establishednetgain;
do not replay those hypotheses. D159/D160 CPUworker fusion improved waiting.
D162 independent currentcap4prefix CPUframe counts NOTpercentages:
controller538businesssamples/71JSONloads/23fileinventory/20execution_copy/
17execution_bundle; GPUHOSTinventory128/552,116/531,89/529,98/527.
D144inflightsharing,D151omitstaging,D153singlefileinventory already implemented.

D1707Bwarm1536+16measured, backed ef38f257; commonreference NOTgloballyfrozen.
Input≤616/>616 thresholds2.990091890/4.635325166s,TPOT.059646198/.085345274s.
Referencebatch8/slots8/seqs8/rank16/.92 differsfromFullcap4/slots4/rank64/.70,
disclose; do not relabel it Full or relax thresholds. 3Breferencepending.
D166/D16744snapshots19888tensors matchcheckpoint+GPUslot; NOTper-token
executionmap/fullpool/n_correct. D63looseprobabilitycontrols acceptwrongadapter
andbase, do not repeat or pretend numericalqualification.
3B500zero,7B498zero+2nonzero/4weightSHA; noauthoritynewweights.
Legacy7BmeanTTFT.564local/.657remote,3B.881local/1.087remote; contracts incomplete,
cannot reconstruct physicalU/n_correct fromCE or discounts.
Current3BD138mean12.588627/P9545.713688s stillUNACCEPTED. Newmetricold/newPrime,
numeric, commonwarm/Residentbudget/G1G2 remainOPEN, notjustpreviouscandidategain.
D145fourRPCtimeoutcauses unresolved; laterzero failures do not close cause.
Neverwholeload/reproject D137/D145/D14815GB original data.

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
- Current3Bcap8/slots8/cpu32/gpu.72; current7Bcap4/slots4/cpu24/gpu.70;
  7B D157 service/preparation/length profiles replace oldD89cap2 timings.
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

## D368 storage gate recovery — closed outcome logs archived

2026-10-09: The initial preflight observed 159230300160 available bytes on the
inference filesystem, below the unchanged 150-GiB start floor
(161061273600 bytes). No serving process, GPU worker, or remote service was
started. A guarded utility was added in commit `4ec3e27` and pushed to
`faaslora_origin/retry14_continuous_queue_v2` before it was run. It operated in
an owned 4-GiB, no-swap auxiliary scope and touched only four already-closed
7B W0 outcome files: D309, D323, D326 and D336.

For each file, `launch.json` required a passed run, removed service path and
confirmed native GPU release; an open-descriptor/active-command check ran before
compression. `gzip -1 -k`, `gzip -t`, two independent decompressed SHA-256
passes and a final source hash check completed before the uncompressed duplicate
was unlinked. The gzip archive remains beside the original physical-deployment
path, and a per-run receipt remains in each raw run directory. The aggregate
receipt is
`paper_results/ieee_tc/storage/20261009_closed_outcomes_lossless_compression.json`
(SHA-256
`0d9667a36b32a3f3424e0212993905b70766d791aa357e57006b327ae189b631`). It records
2,333,794,304 allocated bytes reclaimed and final available space
161563312128 bytes (about 150.47 GiB), above the start floor. The historical
curated audit SHA for D323/D326/D336 differs from the current raw source SHA;
the receipt records that provenance caveat and these archives are not promoted
to reusable result identities. No weights, traces, adapter pools, environments,
curated results, old figures or user-owned files were changed.

The compression scope is inactive. The next permitted action is to publish the
receipt/checkpoint, re-run exact disk/memory/GPU/resource gates, and then launch
the single unchanged-input D368 7B W0 replay for the asynchronous request-plan
preparation candidate. No 3B, baseline, figure or formal G1/G2 claim follows
from this storage maintenance.

## D368 launch identity — ready, not yet started

The D368 campaign identity is committed and pushed in `a71d854bd69ac1d2b5f40103aa77c8ee5cca66c1` (remote branch equality verified). The new configuration SHA-256 is
`336aeb0acd1bf256be532c181baa17f8cc0b8749f14ed1a5b4a4c6fe0e990400`.
Compared with D367, only the campaign output directory, NVMe cache directory,
HOST cache directory and explanatory comment differ; the YAML diff was checked
explicitly. The replay will use the native external W0 path and therefore the
per-arrival single-worker plan-preparation executor from commit
`5934093c22f891bb6f5d3ade597804a0f24ffbb1`, with preparation wait retained in
the dispatch/request timeline.

At this checkpoint no D368 GPU process, tmux replay, or remote artifact service
is active. The exact launcher will recheck the plan SHA, frozen D367 preparation
binding, branch equality, cgroup identity (`4 GiB`, no swap, auxiliary CPUs),
disk/memory/GPU gates, and the absence of an existing D368 result before
starting. The 7B request plan remains a development replay: its outcome can
measure the startup-barrier change, but cannot by itself establish formal G1/G2
without complete correctness and the frozen common warm/reference manifests.

## D368 execution and evidence closure — diagnostic only

2026-10-10: The D368 7B Full/W0 replay was launched only after the exact
auxiliary cgroup, branch, frozen-input, disk, memory, GPU and remote-health
gates passed. The replay used the already published read-only gzip artifact
cache and the same trace, adapter subset, request-map hash and effective time
scale as D367. It completed 4,000/4,000 native-terminal requests with no
terminal failure. The service was stopped exactly once, the remote monitor and
all owned local processes were collected, and the post-run GPU/remote checks
were clear. The raw closure files and result remain under
`results/ieee_tc/p2_backend_qualification/d368_20261009/7b_full_w0/`; the
result source SHA-256 is
`fd749ffe3f0cabb0305340089e1f8d51a08ba776aca5566bec3f30c1d7b4c73e`.
The collection receipt records 13,059.590304143727 physical GPU-seconds and
4,000 complete native-contract records. The D368 configuration SHA is
`336aeb0acd1bf256be532c181baa17f8cc0b8749f14ed1a5b4a4c6fe0e990400`; the
frozen metric and plan references remain
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` and
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.

This replay is not a formal G1/G2 result. Correctness fields were not
complete, the old development 5,000-ms SLO field is not the frozen joint
TTFT/TPOT SLO, and no reference manifest was rerun. Consequently no curated
CSV, ranking, figure or paper claim was generated from D368.

The paired raw comparison with D367 is:

| development replay | TTFT mean / P95 (ms) | E2E mean / P95 (ms) | dispatch-window mean / P95 (ms) | physical GPU-s |
|---|---:|---:|---:|---:|
| D367 bulk preparation | 1309.9 / 3176.1 | 5016.1 / 10447.2 | 172.2 / 125.3 | 13701.34 |
| D368 per-arrival preparation | 1574.0 / 3539.1 | 5350.0 / 11127.4 | 465.8 / 644.4 | 13059.59 |

The new path removed the synchronous preparation barrier for the first
request: its overall TTFT fell from 25,916.2 to 6,253.8 ms and its E2E fell
from 29,472.5 to 9,545.7 ms. The first 500 matched requests also improved
in aggregate. That local improvement did not establish a full-trace gain:
over all 4,000 paired request IDs, D368 minus D367 was +264.1 ms mean overall
TTFT, +334.0 ms mean E2E and +293.6 ms mean dispatch-window time, while
physical GPU-seconds fell by 4.69%. The largest degradation is attributable
to a state/residency event rather than remote packing: request `req_00961`
changed from GPU-resident/no mismatch in D367 to HOST-resident with an
adapter--replica mismatch in D368, incurred 35,341.3 ms LoRA I/O and 35,543.8
ms admission-to-engine-dispatch time, while the backend's native TTFT was only
253 ms. Both replays observed 132 remote transfers and zero remote packing
time, so this outlier is not evidence that the new path made remote packaging
slower.

The request-plan receipt for D368 reports 4,000 plans, a single-worker
per-arrival executor, 101.0001 s accumulated wait and 1.5824 s maximum wait.
Block-level inspection shows the first 500 requests benefit, followed by a
dispatch/residency-sensitive regression around requests 500--1,250. The
evidence therefore supports only the narrower observation that moving plan
preparation off the event loop removes the initial bulk barrier. It does not
support the stronger hypothesis that the current single-worker implementation
improves the complete workload. The next test must isolate executor-induced
queueing from natural tier/residency variation (starting with a CPU-only
bounded-worker diagnostic and state-transition audit on the same trace) before
any new full GPU replay. No blind concurrency increase, adapter-capacity
change, 3B run, baseline run or figure production is authorized by this
diagnostic alone.

## D368 follow-up CPU diagnostic and stale-identity audit

2026-10-10: A no-GPU diagnostic used the same 7B frozen 4,000-request trace
and the native fixed-output prompt planner. The tokenizer was initialized
before timing; each request then executed the real
`_prepare_request_execution_plan` path. Across 4,000 requests, preparation
time was 4.661 ms mean, 5.111 ms median, 6.830 ms P95, 7.564 ms P99, 8.947 ms
maximum and 18.644 s total. A deterministic FCFS simulation using the
measured per-request service times and the recorded arrival offsets produced
the following total executor wait (seconds):

| arrival compression | one worker | two workers | four workers |
|---:|---:|---:|---:|
| 1x (frozen W0) | 0.743 | 0.013 | <0.001 |
| 8x | 2.675 | 0.217 | 0.002 |
| 64x | 22.939 | 2.063 | 0.063 |

The observed D368 preparation receipt instead reports 101.0001 s accumulated
wait and 1.5824 s maximum wait. The bounded single-worker executor is
therefore not sufficient to explain the full observed delay; changing its
worker count without a new causal observation would be an unsupported tuning
patch and is not authorized.

The raw request evidence identifies a separate, testable path. For D368
`req_00961` and `req_00636`, the selected HOST copy was promoted or replaced by
a GPU copy, yielding one legitimate `required_source_changed` response. The
controller then re-observed the native state but did not carry the newly
observed native `source_id` into the next demand intent. Under concurrent
owner-epoch advances, the intent consequently entered repeated
`stale_snapshot` rechecks (41 and 20 respectively), with 35,323.9 ms and
17,278.1 ms excluded before native loading. The eventual native loads were
only 17.4 ms and 0.9 ms; the delay was control/state re-resolution, not GPU
copy or remote transfer time. D367 had only three requests with any stale
recheck (maximum three); D368 had three, but two reached 41/20 retries.

This observation supports a narrow first-principles hypothesis: after a fresh
native source observation, retaining its immutable copy identity lets the
serialized owner tolerate unrelated epoch changes while still rejecting a
real copy/tier change. The existing native contract already permits this
identity-based validation, and the unit test for a held HOST copy surviving an
unrelated epoch advance exercises that invariant. The next code change, if
the regression test passes, is limited to propagating the freshly observed
native `source_id` into the retry intent; it must not add sleeps, retry caps,
capacity increases or worker-count tuning. A small unit/regression test must
show both cases: unrelated epoch changes complete without repeated retries,
while a genuine HOST→GPU change still causes one re-observation and a tier
update. Only after that evidence is complete may a new GPU validation replay
be considered.

## D368 closure maintenance and source-identity fix validation gate

2026-10-10: The closed D368 outcome was archived before any new GPU work because
the inference filesystem was below the unchanged 150-GiB start floor. The
one-shot utility `scripts/ieee_tc/compress_d368_closed_outcome.py` was committed
and pushed in `c62e9e5` before execution. It accepted only
`d368_20261009/7b_full_w0`, verified `launch.json` closure/release, checked for
active commands and open descriptors, and ran `gzip -t` plus two independent
decompressed SHA-256 passes before removing the 501-MiB uncompressed duplicate.
The source hash and both round trips matched. The raw per-run receipt is
`d368_20261009/7b_full_w0/compression_receipt.json`; the aggregate receipt is
`paper_results/ieee_tc/storage/20261010_d368_closed_outcome_lossless_compression.json`
(SHA-256 `21dd9658f456130eccea571ae9ddb516614fc9dc2f0744228840f40f75b09b0f`).
The operation ran in the owned 4-GiB, no-swap auxiliary scope
`primelora-tc-aux-68ef25fda2eb459e9215f9052792c5a3.scope` and reclaimed
462,921,728 allocated bytes. Available space increased from 160,599,535,616 to
161,062,445,056 bytes; the scope is inactive and no GPU or remote serving
process is active. No weights, traces, adapter pools, environments, curated
results, figures or the user-owned manifest were changed.

The source-identity propagation fix is committed and pushed in `68eb627`.
The targeted lifecycle tests pass (3/3), the stale/host/held GPU-reference
subset passes (37/37), and the broader native/host/stale/source/admission
selection passes (266/266, 76 deselected). The fix carries a freshly observed
native `source_id` into a retry intent after a legitimate tier change, while
dropping the identity only when the source is no longer a native HOST/GPU
owner. The regression test also advances an unrelated owner epoch to ensure
the old repeated-stale loop would fail. No retry cap, sleep, capacity change or
worker-count tuning was introduced.

The next permitted action is a 100-request D369 GPU validation on the frozen
D368 input, with new raw/output/cache directories and no curated result or
figure publication. It must verify native terminal completion, absence of the
41/20 stale-recheck pattern, GPU release and remote-service closure before any
full D370 replay is considered.

## D367 storage-margin recovery and D369 validation readiness

2026-10-10: After the D368 archive, available space briefly fell about 6 MiB
below the unchanged 150-GiB inference-node start floor. No GPU or remote
service was started in that state. The already-closed D367 outcome was therefore
archived with the same guarded, lossless procedure in commit `e825bc3` before
resuming the experiment line. The one-shot wrapper accepted only
`d367_20261009/7b_full_w0`; closure/release and open-descriptor checks passed,
`gzip -t` and two decompressed SHA-256 passes matched, and the uncompressed
duplicate was then removed. The aggregate receipt is
`paper_results/ieee_tc/storage/20261010_d367_closed_outcome_lossless_compression.json`
(SHA-256 `6d579240cde08541eafc49dbf7b2fc40791f0bb709d5672f0026f2b3435651c4`).
It records 472,297,472 allocated bytes reclaimed. The operation ran in
`primelora-tc-aux-841cff404aef4a629a5ac44dc7f5bb1b.scope` with 4-GiB
`MemoryMax`, no swap and CPUs `2,3,26,27`; the scope was empty and stopped.
No unique weights, trace, adapter pool, curated result, source code or user
manifest was touched.

D369 is a bounded 100-request validation of commit `68eb627` (native
`source_id` propagation after a legitimate source/tier change). It reuses the
frozen D368 7B trace, adapter subset, fixed-length greedy contract, remote
prepublished-gzip delivery and preparation binding, but uses new output/cache
directories and the new campaign identity
`d369_20261010/7b_stale_identity_validation_100`. The dispatcher/configuration
identity is committed and pushed in `e4beb1a`; configuration SHA-256 is
`9452fa822d321c67e008dc20e3d7c06bc755e5b5be45a4e73ace3cbfc22b14c8`. The
source plan SHA is
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`, and the
frozen V1 digest is
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`. No
curated result or figure is permitted from this validation. Pre-launch gates
passed: branch HEAD equals the remote branch at `e825bc3`, GPU process census
is empty, available memory is about 104 GiB, available disk is
161,526,620,160 bytes, remote services/ports are inactive, and the D369 raw
result directory is absent. The launcher must record these identities again
and reject any mismatch before starting.

## D369 source-identity validation closure

2026-10-10: The bounded D369 replay was launched only after the plan/protocol
hash, branch equality, disk/memory/GPU, output-directory and remote-service
gates passed. It used the frozen D368 7B input and the new code path, with
`FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=100`; no configuration was selected from
the validation outcome. The runner completed 100/100 planned requests and
100/100 native-contract terminal returns with zero terminal failures. The
immutable source is
`9f0c0a9dc5b1a1af0bc5fe7635afcb6d7c2d5bd63d1969804f99c5e3c92bd6d7`, and the
launch receipt SHA is
`2b459df092c8b9d972b69f06d4526ac11414053c2e8971f83546ba8c7c8cce62`.

The primary diagnostic result is that all 100 request records report
`stale_rechecks=0`; the sum, maximum and nonzero-row count are all zero, and
there are no `cache_source_rechecks` entries. This directly exercises the
source-identity propagation fix under the same external W0 path that produced
D368's 41- and 20-recheck outliers. Every record includes the frozen
`readiness_tier_before_dispatch` field. The validation therefore supports the
narrow causal claim that carrying the freshly observed native `source_id`
prevents the repeated stale-snapshot loop; it does not establish a latency,
GPU-s/request or SLO improvement.

The closure receipt records 661.9514837537427 physical GPU-seconds for the
100-request development validation, 100 native-contract completions, remote
transfer SHA
`47806240f42d552e541dc1a8b5b471fac3c7511a8ac9ba64ce88bf02efbea99d`, remote
monitor SHA
`95caa1fad158932c7f868355818d26e284dcc575f13110bf1f85772c1db5ac5c`, and
`native_gpu_context_release_confirmed=true`/`service_path_removed=true`.
The remote stop was executed once; all three owned remote units are inactive,
ports 18080/18081 are absent, local GPU census is empty, and the collection
scope reports zero OOM/high/swap events. The raw closure is
`results/ieee_tc/p2_backend_qualification/d369_20261010/`; no curated CSV,
ranking, figure or formal G1/G2 result was produced. `n_correct` remains
unknown because the development analyzer does not supply the frozen numerical
correctness and common warm/Resident reference fields.

The next permitted Prime-first action is a full 7B W0 replay on the fixed code
(`D370`) only after this closure is backed up and the same resource/remote gates
are rechecked. D370 remains development evidence until the frozen correctness,
joint-SLO and resource-reference contracts are available; no 3B, baseline,
ablation or figure work follows from D369 alone.

Before the next full replay, the closed D364 development outcome was archived
to restore a safe disk margin. The one-shot wrapper added and pushed in
`183a665` accepted only `d364_20261009/7b_full_w0`, verified closure/release and
two gzip round trips, and reclaimed 465,145,856 allocated bytes. Its aggregate
receipt is
`paper_results/ieee_tc/storage/20261010_d364_closed_outcome_lossless_compression.json`
(SHA-256 `a861d60d0c091d5c8240b9718b03f974bb45e8ac7b4083192f5796759cdcac67`).
The operation used the owned 4-GiB/no-swap scope
`primelora-tc-aux-bcdbb0cf04e14649bcda4a46a821fda9.scope`; its task count was
zero before stop. Available space is now 161,955,876,864 bytes, above the
inference-node start floor with approximately 0.83 GiB of additional margin.
