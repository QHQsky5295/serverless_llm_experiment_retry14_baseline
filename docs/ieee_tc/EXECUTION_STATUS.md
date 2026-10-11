# IEEE TC execution status — D423 full vLLM 7B W0 preparation

Updated 2026-10-11. This is a current checkpoint, not a replacement plan.
The former 10,982-line status is preserved byte-for-byte in
`EXECUTION_HISTORY_THROUGH_D402.md`, SHA
`cbfd51321c2bb2fdc0e2c767054cc5e68a16176360235f6a855ec214cd5b9037`.
It includes every earlier D-series entry and D402 start/closure. Search it and
the related reports before changing code. Old NEXT sections are historical.
No approved experiment or failure evidence was removed. This split prevents
whole-status reads from truncating current instructions after compaction.

## Authority and unchanged evaluation

- Read `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` fully at task start,
  after compaction and before experiments. It and the full-name snapshot
  `PLAN_APPROVED_20261010_FULL_NAMES.md` have SHA
  `f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5`.
- Before comparisons/configuration selection read `METRIC_PROTOCOL_FROZEN_V1.md`
  (SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`)
  and `METRIC_PROTOCOL_SINGLE_RUN_V2.md`
  (SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`).
- G1: all4,000 correct + common joint TTFT/TPOT SLO>=95%, then minimum real
  lifecycle GPU-s/request. G2: same correctness/SLO and U<=0.75U_ref, then W1/W2
  P95TTFT and observed joint-SLO difference. CE supplementary. Relative TPOT
  regression alone is not failure; never change the evaluation ruler to win.
- One performance run per frozen key, no repeated means/CIs or best-repeat
  selection. Preserve development/reuse identity, all failures and old results.
- Order: sealed7B PrimeLoRA ->7B baselines, ServerlessLLM first -> comparison and
  necessary optimization ->3B -> ablations/mechanisms/sensitivities.
- Full names PrimeLoRA/ServerlessLLM; historical IDs unchanged. No figures until
  near target, no13B/new SGLang/paper edits/new workload or weight pools.

## Reusable PrimeLoRA evidence

D397 seals D371 as reused development W0, not unseen formal test:4,000 native/
source/adapter binding contracts,3,908 joint passes(97.70%),14,482.523924GPU-s,
3.620631GPU-s/request; mean/Type-1 P95TTFT1150.798/2837.896ms.
See `D397_7B_SINGLE_RUN_SEAL.md` and its curated seal. Actual132 published HTTP
deliveries,zero packing,one phase-attribution join caveat. Mostly-zero/repeated
LoRA content limitation remains. D379(94.425%) and D386(94.85%) are retained
below-threshold alternatives. D394 freezes7B warm thresholds; valid Resident
U_ref is OPEN. D395/D396 have measurement/ownership gaps and cannot set it.

## D402 — terminal, affected path passed; scheduler defect prevents final freeze

Raw exclusive root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d402_20261011`.
Execution commits: main348c5f3e750fd78a2f877c25bd4c06d85b4806bc;
baseline d0fea52483f2906a5e68a1ec96aed42135498c67. Existing trace,pool,
checkpoint,native vLLM0.10.2,D401 source view. No live GPU job; do not restart.

- 100/100 native contracts,0failed,17,369 output tokens. E2E/TPOT/native maximum
  reconstruction error0ms. Independent numerical LoRA qualification not added.
- All29 real published-gzip transfers join request/receipt/content/server:
  50,650,865 wire bytes,724,665,100 verified materialized bytes,71cache-hit
  requests,0request-time packs. This is not500-ID coverage or full W0.
- Physical four-card native-store pool closes at1544.433269GPU-s,no open leases.
  Raw n_correct=null and full/numerical qualification flags remain unchanged.
- Mean TTFT138.039s/router queue135.379s. Four logical instances include one
  stuck starting allocation;3engines served. No comparison of this prefix with
  PrimeLoRA's4,000-request observation or selection as a baseline optimum.
- Executed scheduler SHA
  b7779e41f451a3328eb73337533d6d9b117aafe6ee6b78f9b28e9b5f5e2b0567
  enumerates pending items then pops stale list indices. Its unaltered control-
  loop AST with3pending requests reproduces IndexError,2resolved futures,wrong
  leftover item and no metadata update. Live logs match that boundary but lack
  an exception traceback; retain this distinction, not an all-delay attribution.

Report: `D402_SERVERLESSLLM_7B_PATH_RESULT.md`.
Audit: `paper_results/ieee_tc/baseline_audit/d402_7b_path/audit.json`, SHA
`b1102a8b839ac198d512381ac6b0709fb7ab1bedebe6d38143aa4e1b05275ba8`.
Companion evidence_manifest binds77 retained files. Recipe reuses existing
analyzer/native validators. CPU audit passed; baseline native/router/launcher
regression suite70tests passed in7.699s,4-GiB/zero-swap scope. No performance repeat.

## Exact next action and remaining scope

D423 PREPARATION 2026-10-11. Full PLAN/STATUS/V1/V2 reread; fresh GPU census
empty, no tmux, disk163657142272B and hostavailable112072163328B. Existing
D422 recipes reused for original4000/132IDs/458224tokens, same native config,
no parameter search or runtime code change. Report D423_VLLM_FULL_W0.md
preregisters one full W0 and Resident-reference candidacy, contingent on full
contract/resource audit; no U_ref yet. Exclusive raw vllm_qualification/
d423_20261011, fresh private /tmp/tc-d423-7bv, neither GPU nor remote started.
HTTP configSHA32b80e2227a8826943d03abf66de98dfd230f1288dc39840e6973c3d1b9b9ced;
native config unchanged361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5.
NEXT backup preregistration -> fresh resource/protected/input/cache identity
gates -> unchanged owned remote health -> launch ONCE -> actual release/remote
closure/preserve/full4000audit/backup -> full-contract comparison. D419/D421/
D422 are passed and not repeated. PrimeLoRA D371/ServerlessLLM D414 sealed;
W1/W2/G2/remaining7Bbaselines/3B/ablations/sensitivities/docs/figures OPEN.
Full goal ACTIVE. Older NEXT/RUNNING paragraphs below are historical.

Final D422 cache-receipt checkpoint ff88422acba5344c8a08eff8dab7e3e8f5f07d06
pushed to faaslora_origin/retry14_continuous_queue_v2;remote SHA verified.
Exact3taskfiles/diff/secrets checks passed;only3protected user files remained
tracked-dirty before this postpush status update. Fresh GPU census empty,
disk163659489280B. No livejob or D423 preparation yet. Continue one complete
4000W0 after fresh full-plan/status/protocol/resource checks;do not rerun D422.
Full goal ACTIVE;this final confirmation is subsequent task-owned unstaged status.

D422 CLOSURE BACKED UP / PRIVATE OBJECTS RECLAIMED 2026-10-11~17:09.
Main730ce7cc0c6f47064f7ac8dcf9e160df8037c894 andbaseline
e646f9106c8fd6fd15f54b69041bd54234fa47e1 pushed;bothremoteSHAs verified.
Baselinefirstpush hadtransient TLSfailure;bounded HTTP/1.1 retry succeeded,
no TLScheck disabled/no force. All133evidenceSHAs/bytes verified,exact task
files only,diff/secrets checks passed;protecteduserSHAs unchanged.
Afterbackups,reclaimed ONLY /tmp/tc-d422-7bv/remote_artifacts/objects:
600files/1474429899payloadB,observedfreed1476624384B,diskafter163660283904B.
Retainedoriginalpool bytes rehashed/index/differentinodes,symlinks/mounts,
metadata copy SHAs,readableownedprocess FD/cwd/root+fuser andfuturetracked
config/scriptreferences checked. Unreadable sameUIDproc entries2628,4156156,
4157313disclosed;not a complete systemwide scan. Allraw/receipt/journal/remote
immutablecache preserved. EmptyownedcleanupscopeInvocation088c7544381b4b0eb6a16c49165badec
stopped;prior133-filemanifestunchanged. Rawcache_cleanup_before/after.json and
curatedstorage/20261011_d422_download_cache.json retain receipts.
No liveGPU/remotejob. NEXT freshfullplan/status/protocol/resourcegates ->
preregister one full4000W0 withunchangedD419/D421/D422nativeconfig (D423notyet
prepared/started). Reuse allpassed100/500/1000gates and existingnumericalevidence,
no repeats/parametersearch. PrimeLoRA D371/ServerlessLLM D414 sealed;
U_ref/W1/W2/G2 andallremainingapprovedscope OPEN;fullgoal ACTIVE.

D422 TERMINAL/AUDITED 2026-10-11, actual release before17:01:50 Asia/Shanghai.
1000 offered/submitted/terminal/native/source/adapter/content/timing contracts,
0 failures,60 IDs,122510 native tokens;E2E/TPOT/native recompute error0ms.
CommonD394 TTFT991/TPOT1000/joint991=99.1%;mean/P95TTFT455.189460/2118.832637ms,
mean/P95TPOT24.974164/27.243896ms,meanE2E3490.662821ms. Four actual workers and
closed physical leases,U4601.208847534144GPU-s,4.601208847534144/request.
This original1000 compatibility view is NOT fullW0/ranking/U_ref;raw n_correct
and numerical flags remain unchanged. Native ready notice+40.900..41.959s;
retained compiler disk cache differs fromD419,not a free engine preload.
60HTTP/74953014wireB/1474429899verifiedpayloadB,940cachehits,0packs.
Peakservice12567740416B,hostavailablemin100156542976B,diskmin162226290688B;
high/max/OOM/swap0. Actual GPU release/service removal/exit0 confirmed;empty
ownedaux stopped,3ownedremoteunits STOPPED by InvocationID/MainPID0/portsabsent.
Remote copies SHA:transfer c5457d777af44f3cc0c9ca2ae5f4ed7c253912285a4d1a7e146122ba3e7fb317,
monitor ce3ba5fe1bc49a7ce210b276fbd1a3c2a5bdfcf801db2aea29fe2cb5c9af5282.
147 protectedentries unchanged,postSHA c86a0aa5f6624cb83c71f1497d9f235c3d42ddc4267f8c11e8c32e6a0f7e75ad.
Existing strict prefix1000/resident_rr_v1 audit passed;raw auditSHA
43f07985abd12189288597587fc0128939c779e1dad65f6875afbdfe82b81517,
133file/67717384B manifestSHA42f06cbed073ec760b056db59fef4807e57187202f086c78c0db51f5d108abc8.
Original first100 independentD385 witness reused:97HF exact/3existing margin
witnesses,93same-as-D419/7changed;all4weightclasses. No newGPU oracle or exact
match gate;other900 independent numeric proof not inferred. New analysis-only
recipe initially failed on nonexistent contract arrival field/no output; uses
existing FrozenReplayPlan original order now,not a performance retry.
Report D422_VLLM_1000_COMPATIBILITY.md;curated d422_7b_compatibility/.
NEXT verify/seal backups,optional safe downloadedobject-only reclamation and
fresh resource gates -> one full4000W0,unchanged native configuration. No100,
500 or1000 repeat/parameter search. PrimeLoRA D371/ServerlessLLM D414 sealed;
U_ref/W1/W2/G2/remaining7Bbaselines/3B/A1-A5/S1-S13/CE/docs/figures OPEN.
This turn completed a monitored live run and strict audit,concrete progress,
not goalcompletion. Older RUNNING/NEXT paragraphs below are historical.

D422 VERIFIED LIVE checkpoint 2026-10-11 16:45:22 Asia/Shanghai. Same active
service InvocationID9d9487f825874614acb5d7a94e0f29a7/tmux tc-d422-7b, not restarted.
51 created/sent,47 raw/validated responses,1000 registered contracts,0 failures.
Watchdogsample167:hostavailable101595738112B,service10290659328B,
disk163301314560B,high/max/OOM/swap0,no abort. Deployment notice1215401.25616516,
business origin1215461.25616516,original last arrival1216540.6704450978.
Run is not terminal. Running documentation backup e1074cdbef55094a4f83f4929696ca201cc0e662
pushed to faaslora_origin/retry14_continuous_queue_v2 and remote SHA verified;
not the execution commit. This postpush checkpoint remains task-owned unstaged.
NEXT monitor SAME handle, no live tests/hashscan/cleanup/remote changes -> actual
release -> raw stop_remote.sh/preserve_evidence.py -> existing prefix1000 audit
with D394 and resident_rr_v1 -> seal/backup -> one full4000W0. Do not repeat
D419/D421 or treat this wait as a blocker. PrimeLoRA D371/ServerlessLLM D414
sealed; all remaining approved scope open and full goal active.

D422 RUNNING since 2026-10-11 16:42:25 Asia/Shanghai, tmux tc-d422-7b.
Exactly one heavy job. Main execution 9f334caab8a2c305f7e29909532a65f7cc8db1d1
pushed/remote SHA verified; baseline d9971a14e18eca3391665d9da1244594e69b6041
remote SHA reverified. No baseline code change. 27 existing CPU analysis tests
passed 0.102s before launch in 4GiB/zero-swap scope; exact two docs backed up,
diff/secrets/protected-user-SHA checks passed. No retained test-file receipt
for this terminal-only execution; prior D421 retained test receipt remains.
Service primelora-tc-svc-243b5724c09b451c8fb6d376df874c9a.scope,
InvocationID 9d9487f825874614acb5d7a94e0f29a7; auxiliary
primelora-tc-aux-d4220000000000000000000000000001.scope.
Original1000/60IDs/122510tokens/viewSHA verified. HTTP config SHA
6679196e85f9ec68a551e2117b7749a855cbc0e1657a23c0c6f4b236118c0244;
native config unchanged361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5.
Current native compiler disk cache628files/54387041B, SHA
92cc5289f1f91ebf0149086a133d359549ee311ca65d6bc2ebc380dba25a58d6;
not GPU preload, fresh LoRA cache. Preflight passed; protected seal SHA
2261680e6079e611ea4fda05fe60d5fad2eff2c523d8e0ade2e518ff8493cfa8.
Unchanged remote published protocol/link1000/full;7B clock
remote-process-monotonic:e7c781165a1d4f3ab78e0d70d57011d0.
Remote3B/7B/monitor InvocationIDs a1a995d5c9f2409e81c67465303d6312 /
2918a274f5b44d9eaf8bdde5802382b4 / 5895cf0ad5e948f3b8dd405333a300f3.
Watchdogsample60:hostavailable102122684416B,service9150644224B,
disk163730849792B,high/max/OOM/swap0,no abort. No terminal/performance verdict.
NEXT monitor SAME handle/local journals -> actual GPU release -> D422 owned
remote stop/preserve evidence -> strict prefix1000 audit with D394 -> backup ->
one full4000 W0. Do not restart or repeat passed100/500 gates. No live remote
changes, hashscan, cleanup, tests, parameter tuning or second heavy job.
PrimeLoRA D371/ServerlessLLM D414 remain sealed; U_ref/W1/W2/G2/rest OPEN.
This turn made concrete progress by preparing/backing up/launching D422;
full goal ACTIVE. Older RUNNING/NEXT entries below are preserved history.

D422 PREPARATION 2026-10-11. Full PLAN/STATUS/V1/V2 reread; actual GPU census
empty and no tmux job. D421 whole-pool qualification/closure/backup is complete,
not repeated. Existing D421 recipes reused with original trace_prefix_v1,
1000 requests/60 actual IDs/122510 target tokens, unchanged native config.
Exclusive raw vllm_qualification/d422_20261011; report D422_VLLM_1000_COMPATIBILITY.md.
No baseline code change or parameter search. NEXT preregistration backup,
fresh resource/protected/input/cache identity gates, unchanged owned remote
health -> one compatibility run -> release/close/audit/backup -> one full W0.
No GPU or remote operation started at this preparation checkpoint. PrimeLoRA
D371/ServerlessLLM D414 remain sealed; U_ref/W1/W2/G2/remaining matrix OPEN.
Earlier RUNNING/NEXT entries below are historical, not live commands.

D421 CLOSURE BACKEDUP / CACHE RECLAIMED2026-10-11~16:34Asia/Shanghai.
Mainclosurec95996159218d3fa1cf7a5397b446c20383fbf04 pushedtofaaslora_origin/
retry14_continuous_queue_v2 andremoteSHAverified;baselineclosure d9971a14e18eca3391665d9da1244594e69b6041
alsoverified. Exacttaskfilesonly,diff/secrets/27CPUtests/568evidenceSHAchecks
passed;all3protecteduserdirtySHAsunchanged. No liveGPU/remotejob.
Afterbackups,reclaimedONLY /tmp/tc-d421-7bv/remote_artifacts/objects:
5000files/12985984450payloadB,observedfreed13004410880B,
diskafter163735080960B(about152.5GiB). Originalpoolbytesrehash/index/inodes,
no symlinks/nestedmounts,currentmetadata copies,readableownedprocess FD/cwd/root
andfuser,futuretrackedconfig/scriptreferenceschecked. Threeunreadable sameUID
proc entries disclosed(2628,4156156,4157313),notclaimedcomplete systemscan.
Allrawlogs/receipts/journals,uniquepoolandremoteimmutablecache retained.
Ownedemptycleanupauxscope stopped;old568-filemanifestnotrewritten.
Rawcache_cleanup_before/after.json andcuratedstorage/20261011_d421_download_cache.json
retainreceipts. Do notredownloadwholepooljusttorepeatpassedbyteaudit.
NEXT freshplan/status/resourcegates -> original1000compatibility usingunchanged
D419/D421nativeconfig/trace_prefix_v1,newexclusivekey -> closure/audit/backup ->
onefull4000W0. No100/500repeat or baselineparametersearch. D422notpreparedorstarted.
PrimeLoRAD371/ServerlessLLMD414sealed;U_ref/W1/W2/G2/allremainingplanOPEN.
Thisturnconcreteprogress,fullgoalACTIVE. PriorRUNNING/NEXTentrieshistorical.
Finalcache-receiptbackup9e2a4db784559d439680a73074d7c24e3a714d53 pushedto
faaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified. Exact3taskfiles,
diff/secretscheckspassed. Freshcompute census empty;disk163734810624B. This
postpushconfirmationis subsequentunstagedtask-ownedstatus;protectedusermanifest
remainsuntouched. No livejobornewperformanceexperimentleftunreported.

D421 TERMINAL/AUDITED2026-10-11~16:25Asia/Shanghai. No liveGPUjob. 500/500
offered/submitted/terminal/native/source/adapter/content/timingcontracts,
exact500IDs=pool,0failures,68925native tokens. E2E/TPOT/nativeerror0ms.
Fouractualworkers/closedphysicalleases,U2722.8814599129837GPU-s. Functional
viewonly,notW0/ranking/U_ref;raw n_correct=null/numericalflagsunchanged.
Mean/P95TTFT644.341327/2135.383901ms,meanE2E4069.208638ms descriptiveonly.
500HTTP/420740410wireB/12985984450verifiedpayloadB,0cachehits/0packs.
Peakservice24619839488B,hostavailablemin99819220992B,high/max/OOM/swap0.
ActualGPUreleased/servicepathremoved/returncode0;emptyownedauxstopped;
threeownedremoteunitsSTOPPEDbyInvocationID/MainPID0/portsabsent. Remote
copySHAstransferf32c2c5e02569f9e3992807a265f43e1719d062346fbbf3ef2ca86f3b5dfd407,
monitor5dc7d58101458d1bd5397aa95d4cdf39c60165625cc496500a9e132121e543cd.
147protectedentriesunchanged;postSHAd8e8dd5339a45c80ba5244320665b404e678ef7e1f168e4db335188c7d506466.
AuditSHA6539a15a293aad9b0e63f4ee1e41b2689847498940468c44b8188a0c583d6277;
568file/35339691BmanifestSHA293c40bd75e447a36945dfb548759acbe3da50cfc336829053f91ec2bcaa341e.
Newpost-releaseCPUreceipt27pass0.174s withlog,notretroactiveD419testreceipt.
D420advisoryreviewbackup399508381a4ce9e864d55ce121ad013b55cc428b pushed/
remoteSHAverified. All568D421evidencefileSHAs/bytesverified. Baselineclosure
backupd9971a14e18eca3391665d9da1244594e69b6041 pushedtoorigin/main/remoteSHAverified;
exactone taskdoc,no runtime/userdirtychanges. MainD421closurebackuppending. Diskabout140GiB,
below150GiBNEWheavyfloor;objectsnotreclaimedyet. NEXTverify/backupclosure ->
safeverifiedD421downloadobject-onlyreclamation -> original1000compatibility ->
onefull4000W0. No passed100/500repeat;PrimeLoRAD371/ServerlessLLMD414remainsealed.
U_ref/W1/W2/G2/other7Bbaselines/3B/A1-A5/S1-S13/CE/docs/figuresremainOPEN.
ThisturnD420inventory+advisoryreview+D421actualgate+auditmadeconcreteprogress;
notgoalcompletion. EarlierRUNNING/NEXTparagraphshistorical,NOTlivecommands.

D421 VERIFIED-WAIT16:20:29Asia/Shanghai:sameactiveInvocationIDbelow,
243created/sent,238raw/validatedresponses,500contracts,0failure. Watchdog399:
hostavailable100258103296B,service17561776128B,disk157769797632B,
high/max/OOM/swap0,noabort. Diskbelow150GiB forbidsNEWheavyjob;currentrunabove
100GiBfloor. No livecleanup/test/hashscan/remoteoperation/configurationchange.
Runningdocbackupc12a9225ed51a86f4ad33d5ba503b08839bf9829 pushed/remoteSHAverified;
notexecutioncommit. D420freshreviewCOMPLETE,WARNforboundedscopeandmissingold
CPUtestexecutionreceipt;103sourcefiles/numbersverified,no narrowreuseblocker.
Fullprompt/interim/finaltrace in.aris/traces/experiment-audit/2026-10-11_run01;
reportD420_EXPERIMENT_AUDIT.md andmachineJSONsaved. Same-family/provisional;
missing skillreferencesdisclosed. Rawnulln_correctisnotrewritten;V2allows
evidence-backednewderivedmetrics,notautomaticpromotionfromHTTP200. Preserve
oldtesttimingclaimasuncorroboratedbyfile;captureNEWtestreceiptAFTERlivegate.
NEXTsameD421monitor/terminalclosure/audit/backup,notanotherqualificationrepeat.

D421 RUNNING2026-10-11 16:13:38Asia/Shanghai,tmux tc-d421-7b. Exactlyoneheavyjob.
Mainexecutionaa76e000127a7cc19d275f8140d0bd8366b3ebc1 pushedtofaaslora_origin/
retry14_continuous_queue_v2 andremoteSHAverified;baselinea56d79088ab1e431e12b5cfed95f53a1581d7037
alreadybackedup/reverified. No baselinecodechange. 27existinganalysisCPUtests
passed0.096s/4GiB/zeroswapbeforelaunch;diff/secretscheckspassed/exact4taskfiles.
Serviceprimelora-tc-svc-b6d8540e824142b5b26918dff1919836.scope,
InvocationID2afee10ae3244d309ce4ec5070cd5738;aux
primelora-tc-aux-d4210000000000000000000000000001.scope.
Deploymentnotice1213672.816455652;commonbusinessorigin60slater.
500existingIDs/68925targets/coverageviewSHArevalidated;no newtraceorweights.
Currentnativecache604files/53278235B,notD419prelaunchidentity;sealedbeforelaunch.
Preflightpass/protectedsealSHA
c6a29035ba17b9ac0f86f24bccfa30ad20324e3808078730462f87d66931cc42.
Unchangedpublishedremote/link1000/full;7Bclock
remote-process-monotonic:f67428c4594f496caac0e76859f00c6d.
Owned3B/7B/monitorInvocationIDsa2530da2e4324d48866d6252d9e66fc6,
7d39db34241a4ce8ad759b81dda0f21c,76eb13f3f47845628da86220d02d8801.
Initialwatchdogsample10available109869613056B/service2008920064B,
disk163776249856B,high/max/OOM/swap0,noabort. No terminalresult/coverageclaimyet.
NEXT monitorSAMEhandle/localjournals -> actualGPUrelease -> ownedremoteclosevia
D421stop_remote.sh -> preserveevidence -> existingstrict500audit/backup ->
original1000compatibility -> onefull4000W0. No liveconfiguration/remotechange,
hashscan/cleanup/tests/retry/secondheavyjob. KeepPrimeLoRAD371/ServerlessLLMD414
sealed;U_ref/W1/W2/G2/remainingplanOPEN. EarlierNEXT/PREPARATIONhistorical.

D420 read-only inventory COMPLETE2026-10-11:32 selected legacy7B vLLM DP4/
Resident replayJSONs;maximum132actualIDs,none500. D213/D395/D396 retainR2
input/native-length/diagnostic reuse,not currentnative timing/resource gates.
InventorySHA2b222ea2d2860e9b71853fc8a439e9cfb359cfc2cca54562ebe9dff81fdf9ede.
SeeD420_VLLM_COVERAGE_REUSE.md. D419100pathpassed andNOTrepeated. Independent
read-onlysame-family provisionalexperiment-audit review requested;referenced
skillshared-references absentlocally,limitationdisclosed. Reviewisadvisory.
D421PREPARATION:existingadapter_coverage_v1/500existingIDs/68925targettokens,
view286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d;
sameD419publicengineconfig/source,no baselinecodechanges/no newtraceorweights.
Rawvllm_qualification/d421_20261011 hasreusedrunner/remote/closure recipes.
No GPU/remotejobstarted. NEXT backup/preflight/protectedseal/inputandnativecache
identity -> unchangedpublishedremotehealth -> one500functionalgate -> actual
release/remoteclose/audit/backup -> original1000compatibility -> onefull4000W0.
KeepPrimeLoRAD371/ServerlessLLMD414sealed;U_ref/W1/W2/G2/remainingplanOPEN.
All earlier NEXT/RUNNING paragraphs below are historical.

D419 TERMINAL/AUDITED2026-10-11~15:54Asia/Shanghai. No liveGPUjob. Original
100/100arrivals/submissions/terminals/native/source/adapter/content/timing
contracts,0failure,17369native tokens;4actualservedworkers. RecomputeE2E/TPOT/
nativeerror0ms. Fourphysicalownersclosed,U1019.2586401619483GPU-s. Do notrestart.
29HTTP/50650865wireB/724665100verifiedpayloadB,71cachehits,0packs. CommonD394
TTFT47/TPOT100/joint47 of100;shortprefixNONFEASIBLE,notfullW0/reference/ranking.
Mean/P95TTFT18634.907506/74980.162867ms,TPOT25.349100/27.612873ms,E2E23055.655692ms.
D385independentreference reused:100inputidentities,92exactsequences,8differences;
3knownfirst-divergence/top2witnesses reused,5marginsunknown. All4weightclasses.
No newnumericalexperiment/exactmatchgate;raw n_correct=null/unknownflags intact.
D415runtimechecks nativeadapterpath butJSONretainsname/int-IDonly;offlinepath
contract binds frozenhooksource/workerconfig/receipts,notinventedoutputfield.
ActualGPUreleased/servicepathremoved/returncode0;freshcompute census empty.
Emptyownedauxstopped,3ownedremoteunitsSTOPPEDbyInvocationID/portsabsent. Remote
copySHA verified:transfer0c8fc0a2c5dec5405a8d00277d1630acefeae63d6da14b10bcb2ce95c1c19469,
monitor41e4692831319a3da982c28f653a49d0ff9d002234ecfa4617ca9f053a598df3.
Worker/resolver/remoteevidence retained;no objectcache reclaimed. Peakservice
11266482176B,hostavailablemin100241502208B,high/max/OOM/swap0,noabort.
Protected147entriesunchanged,postSHAec1fc6173b03701d132548ef0b615ade8fe6611d5a9cc9a693a25db94a394316.
Existingmainanalyzer explicitly extends resident_rr_v1 with exactworker/source/
ordinal/physicalbinding;ServerlessLLMrulesunchanged.27CPUtests pass0.090s.
Audit d419_7b_native_path/audit.json SHA
353cf28d452bbe4a6db72450fa557afe02448abdf016dba83d656268b9fe1149;
103file/18591245BmanifestSHA9830f47c7ddaa5393ceeb11bfd8b8c595966085a7ea71c7edf5959c7e345ed14.
All103evidencefileSHAs verified. Baselineclosurebackup
a56d79088ab1e431e12b5cfed95f53a1581d7037 pushedtoorigin/main/remoteSHAverified;
not the D419executioncommit. Exactlyone taskdoc,no runtime/userdirty changes.
Initialmanifestrecipe failed on relativeSLOpath (nooutput); resolvedagainstmain,
thenpassed. This wasofflinepackaging,not anew/retriedperformanceexperiment.
Mainclosurebackup bfb349b2a939e31f436602dca5a29851164c0f57 pushedto
faaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified. Exactly6taskfiles,
diff/secrets/27CPUtests/103fileSHAchecks passed. Onlyprotectedusermanifest remained
tracked-dirty beforethispostpushcheckpoint. BaselineuserSHAs unchanged. Fresh
compute census empty;diskabout153GiB. No livejob/cachecleanup/newperformance run.
This turn completedD417failureclosure,D418repair andD419qualification/audit,
concreteprogress,notgoalcompletion. Fullgoal remainsACTIVE.
NEXT focusedR0-R3inventory ofoldvLLM500-IDfunctional/
1000compatibility evidence. D395/D396 only132observedIDs,notwholepool;D80HTTP
coverage and6contentclasses cannot themselves establish500nativeIDexecution.
Reusevalidscope; ifmissing useexisting adapter_coverage_v1 functionalview and
original1000compatibility,notnewweights/trace,thenonefull4000W0/reference. Do not
repeat100gatetofindbetterSLO/exactoutput,retuneServerlessLLM orjump3B. PrimeLoRA
D371 andServerlessLLMD414 remainsealed. U_ref/W1/W2/G2/allremainingplan OPEN.
All RUNNING/NEXT paragraphs below are historical,notactivecommands.

D419 RUNNING2026-10-11 15:49:40Asia/Shanghai,tmux tc-d419-7b. No secondheavyjob.
Main execution2387d7d3b40b28c668718ed050503f51bfa52641 andbaseline
3bd6088ce988a4108f397378c5b59a8e586cbc8f pushed/remoteSHAverified.
Service primelora-tc-svc-9b776a0104514a5287475d2cb68baaeb.scope,
InvocationIDde34d82b27444e87963dc76239ba148e;aux
primelora-tc-aux-d4190000000000000000000000000001.scope.
Deploymentnotice1212230.458994817,businessorigin1212290.458994817.
Original100/29IDs/17369tokens/viewaaf917bd816e10c4ad6f8138e5365aa9b2976b4300143cc712c12b84443cb020
revalidated; HTTPconfigSHA04780143176f16c380295b289aa5a2d5126592649b449f9085a87dc0cb54b105;
nativeconfigSHA361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5.
Relevantnativecache540files/52117804B SHA
73c603234980f64ac50fd4a699fd88d5b68cbe2873aafa0f46199e96ae66aa3e frozenbeforelaunch
(D417observedprefixesplusSM86FlashInfer);not a newwhole-rootseal/noGPUpreload.
FreshLoRAcache. Remote unchangedpublishedprotocol/link1000/full;7Bclock
remote-process-monotonic:a2540f3662264ad380014273f8782e0b.
Owned3B/7B/monitor InvocationIDs5443d7afb2df49918e4183a3acf85f3c,
0df94271e6f04a6d94b40732b1a91464,cd564e7011644e798dbb1b816bf64238.
Preflight passed; freshprotectedseal
3374b30f97dd09ee2f2b8a96f5e1c92edc2ab4031d03614fe7cd4c93c2f8b79f.
Watchdogsample14:available107842158592B,service3991015424B,disk164532068352B,
high/max/OOM/swap0,noabort. No terminal/performance result yet.
NEXT monitorSAMEhandle/localjournals -> actualGPUrelease -> exactownedremote
stopusingD419stop_remote.sh -> preserveworker/resolver/remoteevidence -> strict
native/content/physicalaudit/save/backup ->qualificationreuse/fullW0. No live
remotechanges/cleanup/tests/largehashscan/retry/otherheavyjob. D417NOTrestarted.
All earlierNEXT/PREPARATION entries below are historical.

D418 CPU REPAIR COMPLETE 2026-10-11; no liveGPUjob. Existing vLLM runner now
binds the declared interpreter's bin (without resolving its Python symlink)
to workerPATH, probes installed ninja beforeGPUallocation, and verifies actual
worker tool SHA/PATH hash/MAX_JOBS. Explicitbuild_max_jobs2 implements approved
nativebuild safeguard; no servingpolicy/engineparameter change orfallback.
125CPUtests pass7.756s/4GiB/zeroswap/CPUs2,3,26,27. Actualdeclaredvenv probe
withGPUhidden passes ninja--version; installedFlashInferAST confirms-j2 with
buildcommand mocked. No compile/preload/installation. Officialfixedtag/source
andPythonvenv docs checked; D214retry6/7 history explicitly reviewed.
ReportsD418 in bothrepos; D419 preregistration and reusedrawscripts prepared.
Baseline testedcheckpoint3bd6088ce988a4108f397378c5b59a8e586cbc8f pushed to
origin/main andremoteSHAverified. Exact4taskfilesonly,diff/secretschecks passed;
all3protecteduserdirtySHAs andPLAN/V1/V2unchanged.42D417evidencefilesSHAverified.
NEXT backupD417closure/D418testedcode -> freshresource/input/protection/cache
identity gates -> startunchangedownedremote -> oneD419affected100pathgate ->
release/stop/audit/save -> requiredqualificationreuse/full7BW0. D419notstarted.
PrimeLoRAD371/ServerlessLLMD414 remainsealed; alluncompletedplan scope remains.

D417 TERMINAL/FAILURE AUDIT COMPLETE 2026-10-11. No live GPU job. All4native
workers failed at FlashInfer sampling warmup: installed ninja was absent from
inherited PATH. This is protocol_or_launcher_error, not a baseline performance
result. History D214retry6 had the same cause; D214retry7 explicitly fixed PATH.
D416 failed to carry that environment requirement into its new entrypoint.
Final publisher:100contracts,27created,24headerssent,27failed,0native responses;
unarrived73 not fabricated timeouts. Earlier physical snapshot has26terminals,
4openownerleases,U_obs499.795498GPU-s,U=null; it precedes publisher cleanup.
Do not rewrite journals to fabricate a complete lifetime/cost. External launcher
confirms actualGPUrelease/servicepathremoved; freshcompute census empty.
Threeownedremoteunits inactive/MainPID0/portsabsent; originalstop script exited1
only at no-matching-transfer-journal grep AFTER successfulstop. Separateclosure
receipt verifies actualstop and copiedmonitorSHA
edbde45fe9287f58c43d45cecd5e7c090414bb25eba48db9dc2784e4600cceeb.
Emptyownedaux stopped. Servicepeak21910249472B,hostavailablemin102155436032B,
diskmin164548575232B,high/max/OOM/swap0,noabort. Protected147entriesunchanged;
protected_afterSHAac3fee55ff35ceb5900808374b4393f368e21376e7a8b44e22add2df050e983e.
Curated d417_7b_startup_failure/audit.json SHA
f6610f2e465eda26fc2a5adde58aeff535fa9bd9bfe6ba96bdf79b8b57e8a53b;
42filemanifest SHA62d9c47024f37263bfdeaed70cdae14f9415da8d3cdece5d0e254555806d5b82.
NEXT D418: bind nativeworker PATH to declaredvenv beforeGPUallocation, failclosed
on missingtool, recordidentity and explicit boundednativeJIT concurrency;CPUtests
andhistory/officialsourcecheck -> backup -> newkey affectedpathqualification.
No disablingFlashInfer,alternateenvironment,policychange orD417restart. Preserve
PrimeLoRA D371/ServerlessLLM D414;then7BfullW0/remainingbaselines,not3Bjump.
All earlierRUNNING/NEXT paragraphs below are historical, not live jobs.

D417 RUNNING 2026-10-11 15:33:21 Asia/Shanghai, tmux tc-d417-7b.
Main execution cb2561938bc7871a2b0cfb23117417223a92d25a and baseline
5c0cfd8b4d4ead40437d6dc852a9e60395075cf3 pushed and remoteSHA verified.
Service primelora-tc-svc-2bd13b0f6b7b4343b4b80e1dc1cf0dd7.scope,
InvocationID7062c07a378a4278b92564529feaa4f9; auxiliary
primelora-tc-aux-d4170000000000000000000000000001.scope.
Deploymentnotice1211251.664325373, business t0=1211311.664325373.
Original100 prefix/29IDs/17369tokens/viewSHA
aaf917bd816e10c4ad6f8138e5365aa9b2976b4300143cc712c12b84443cb020 verified.
ConfigSHA203699817819b2632c2900c8b6f96f0ec2736539510fe823541e1e0780d07dce,
HTTPconfigSHA04780143176f16c380295b289aa5a2d5126592649b449f9085a87dc0cb54b105.
Native compiler diskcache221763files/7506097394B, aggregateSHA
42cdcdbff103f9cf09cd860d9411bb6071f588087d4837eb84d40d386160ce29;
scanned before notice, no engine/GPU preloading. Scan completed under bounded
auxiliary scope; no concurrent inference. Private LoRA cache newly created.
Remote current1Gbps/full, unchangedD78/D80publishedcache;7Bclock
remote-process-monotonic:0c5cc5135f7743dc9430a585ebb898ab.
Owned3B/7B/monitor InvocationIDs692a4f71f5504122904d977127362fff,
3457a88c8ead4a2a9733560ce148a93a,4e81b78655cd4983ba154d2a66ac7eb6.
Initial native readback: four actual contained EngineCore workers, FP16/TP1,
FLASH_ATTN2, inherited limits. Watchdog sample49:available105165381632B,
service15905505280B,disk164598992896B,high/max/OOM/swap0,noabort.
No terminal/performance verdict yet. Do not restart or overlap a heavy job.
NEXT monitor SAMEhandle/local journals only -> actualGPUrelease -> ownedremote
stop via D417stop_remote.sh -> preserve resolver/worker/remote evidence ->
native timing/source/content/physical audit and backup -> fullW0 after necessary
qualification reuse. No live remote change/hashscan/cleanup/tests/parameter search.
Earlier PREPARATION/NEXT paragraphs are historical and superseded by this run.

D416 CPU INTEGRATION COMPLETE 2026-10-11. No live GPU job at this checkpoint.
The existing vLLM engine runner now has explicit native-http-service/worker
modes, shared pre-ready HTTP publication, service-owned published artifact
resolution, original request-ordinal RR, D415 native observations and per-card
physical owners with real process/pidfd/NVML release. No PrimeLoRA policy or
userdirty replay imported; no installed backend change. Native LoRARequest/LRU
replaces the historical external API-registration wrapper; this deployment
difference is disclosed, not relabeled as byte-identical D395.
Ten new service tests plus previous suites:122 pass8.068s (earlier122 pass7.997s),
4GiB/zeroswap/CPUs2,3,26,27. Main physical lifecycle32tests pass1.316s.
Actual0.30.0 args/SamplingParams/API import with GPUs hidden passed; no engine
constructed. Official fixed-version AsyncLLM and LoRA documentation checked.
See D416_VLLM_NATIVE_SERVICE_INTEGRATION.md in both repos. This is NOT GPU,
whole-pool/numerical qualification, performance evidence or a frozen U_ref.
Baseline checkpoint5c0cfd8b4d4ead40437d6dc852a9e60395075cf3 pushed toorigin/main
and remoteSHA verified. Exact5taskfiles only; diff/secret checks passed.
Protected147entries reverified unchanged against D414preseal; all3userdirty
SHAs unchanged. Plan/frozenV1/singleV2 hashes unchanged. No remote operation,
cachecleanup/republication/newtrace/pool/figure occurred in D416.
Fresh local readback15:20:root available164596801536B (>150GiB),MemAvailable
112081690624B,compute census empty. This is not a later launch guarantee.
D417 PREPARATION: original100-request affected-path gate, inherited7B public
config, exclusive raw vllm_qualification/d417_20261011; run7b/start/stop/health
scripts reuse existing procedures, no GPU or remote service started yet.
NEXT: save/backup preparation, full plan/status/protocol+fresh resource and
protected/input/cache identity checks, start unchanged ownedpublished service,
one D417 GPU-path gate -> actualrelease/remoteclose/evidenceaudit -> necessary
reuse qualification and one fullW0. Do not repeatD395/D396,retuneServerlessLLM
or jump3B. PrimeLoRAD371/ServerlessLLMD414 sealed;U_ref/W1/W2/G2 stillOPEN.
All older NEXT paragraphs below are preserved history, not current commands.

D415 CPU COMPONENT COMPLETE 2026-10-11, following D414 closure. Previous turn
made concrete source changes but was interrupted before tests/status; this turn
verified them, not a wait or blocker. No live GPU job. Baseline existing
ieee_tc_serverless_measurement.py now has explicit install_v030_snapshot and
V030NativeRequestObservation, reusing the common native timeline. No installed
backend edits, serving-policy change, userdirty replay edit or new framework.
Actual pinned producer/collector methods reproduce mutable stats/token aliases
and token-only merge; snapshots retain matched token-event times and adapter
name/ID/path, including a separate completion tail. Unknown source/version and
missing native evidence fail closed. Official v0.30.0 output_processor/outputs/
stats downloaded read-only: all3 byte SHA and AST equal local installation.
13 new checks + existing measurement/launcher/router/SSE regression:112tests
pass7.803s in4GiB/zeroswap/CPUs2,3,26,27. First11-test attempt had one incorrect
test expectation about native exception propagation; test corrected to require
the actual raised exception, no native behavior change. Earlier combined76
passed5.663s. Actual0.30.0 import/idempotent double-install passed withGPUhidden;
before/after compute census empty. Full details in D415_VLLM_NATIVE_OBSERVATION.md
in both repos. This is NOT native GPU/path/performance qualification and has
not yet been wired into the vLLM service runner. Resident U_ref remains OPEN.
NEXT: integrate tested observation through existing runner/common launcher with
service-owned ordinary routing/fetch, pre-ready independent HTTP arrivals and
complete physical leases/actual release; CPU contract tests -> safely reclaim
verified closed D414 objects -> fresh resource gates -> affected GPU path gate
and one full W0. Do not rerun legacy D395/D396 unchanged, retune ServerlessLLM,
jump3B or call this a comparison result. PrimeLoRA D371 and ServerlessLLM D414
remain sealed. Disk~146.9GiB below150GiB NEWheavy gate; no cachecleanup/remote
operation/newtrace/pool/figure occurred. Existing D78/D80 read-only published
cache is reused under user's reconfirmed approval, not regenerated.
Baseline tested checkpoint aba520572d019743320416b0015b1ba701716ee0 pushed to
origin/main and remoteSHA verified. Exact3taskfiles only, diff/secrets checks
passed; userdirty replay/relayserve SHA unchanged. Common protected147entries
verified zerochange against D414preseal; D415 protected_after SHA
6b4deabad0cf41b4f928ab8d0069edd8ae445128fb75786a9739f4e9baf11a70.
Main report/status/protection checkpoint209b6dc8a00b8d9b4f667ea267f9ff423e00a54d
pushed tofaaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified.
Exact3taskfiles only;diff/secrets/protection checks passed. FrozenV1/singleV2
andprotecteduserSHA unchanged. This postpushconfirmation is subsequentunstaged
task-ownedstatus; no stagedfiles or liveprocess left. Fullgoalactive.
Read-only next-entry reconnaissance: legacy fair runner defaults ordinary
round_robin, not least-loaded; preserve actual chosen prior workpoint, do not
silently substitute a new policy. Existing replay_vllm_engine_trace.py is a
synchronous after-ready loop, not a qualified replacement either. Common HTTP
publisher/validator are in prepare_ieee_tc_serverless_stack.py and have explicit
schema/physical-lifecycle restrictions; reuse only after explicit tested vLLM
contract integration, not by relabeling native_store_pool. All remaining scope intact.
Earlier NEXT/RUNNING paragraphs below are retained history, not live jobs.

D414 TERMINAL/AUDITED2026-10-11 05:57 Asia/Shanghai. Original4000offered/
submitted/terminal/native/source/prompt/adapter/content/timing contracts,
0failed,458224actualtokens. Do NOT restart D414 or oldercompletedruns.
CommonD394 joint3925/4000=98.125%,TTFT3929,TPOT3996;observedbindingfeasible=true.
U16089.762743524276GPU-s,4.022440685881069perbinding-validrequest;fourphysical
leasesclosed,5servedepochs. Mean/P95TTFT973.686780/1052.246361ms,TPOT26.736734/
32.929275ms,E2Emean4000.733946ms;E2E/TPOT/native recomputeerror0ms.
132HTTP/131480060wireB/3300789780verifiedpayloadB,3868cachehits,0packs.
Raw n_correct=null andindependentnumericalflags unchanged;reuse D406wholepool/
D409boundedreference,not fullarithmeticproof. No cross-systemranking until
separate fullcontractjoin;not unseen/formal,W1/W2/G2 orcross-scenariooptimum.
ActualGPUreleased,servicepathremoved,returncode0;freshGPUcensusempty.
D414loaderRESTOREDfromownreceipt,remote3ownedunitsSTOPPEDbyInvocationID and
portsabsent,emptyownedauxstopped. No livejob. Allworker/resolver/remoteevidence
preserved underD414raw;source/copy remoteSHA match. Servicepeak57570185216B,
high/max/OOM/swap0. Protected147entrieszerochange;
protected_afterSHA6b4deabad0cf41b4f928ab8d0069edd8ae445128fb75786a9739f4e9baf11a70.
Existingstrictfull4000analyzerpassed;23CPUtestspass0.095s/4GiB/zeroswap.
Audit paper_results/ieee_tc/baseline_audit/d414_7b_full_w0/audit.json
SHA714a2fdb01b2d5a0e2e6f3a9b0faf7cb23b29632efbf5b343822b7dfe4b7dd74;
237-file/399017519B evidence_manifest
SHA8912ab5259d7b10e990ab3b6233b789ce4432c0af2c1544498b7587b700904fd.
ReportD414_SERVERLESSLLM_FULL_W0.md;all237manifestfilesSHAverified. Baseline
closurebackup600cd4bf835db7fdfdd3fd72ea31c677507de10e pushedtoorigin/main and
remoteSHAverified;notexecutioncommit. Mainclosurebackup
3af56b3bb4c76fe0c755d6198237c67851ab5f75 pushedtofaaslora_origin/
retry14_continuous_queue_v2 andremoteSHAverified05:59. Exactly4taskfiles staged;
diff/secrets/23CPUtests/237-filehashchecks passed. All3protecteduserdirtySHAs
unchanged;onlythispostpushconfirmation is subsequentunstagedtaskwork.
No downloadedcache reclaimed yet. Disk~147GiB below
150GiB NEWheavy startfloor. Do notlaunchbefore safeverifiedcache-onlycleanup
andfreshresourcegates. NEXT7BvLLM D412measurement integration;D412readagain,
stillNOTimplemented/qualified:neednativeengine-token timing,service-owned
fetch/routing,pre-readyopen-loop ingress,completephysicalleases/commonlauncher.
Preserveuserdirtyreplay,existingrunners/assets;do notsimplyrerunD395/D396.
No nextGPUjobstarted. Returnto7Bmainline,
notServerlessLLMparametersearch/newshortgates or3Bjump. PrimeLoRA D371sealed;
allotherapprovedcomparisons,Residentreference,W1/W2/G2,3B,A1-A5/S1-S13,CE/docs/
figuresremainopen. Previousgoalturnverifiedwait;thisturn completedfullrunclosure
andstrictaudit,concreteprogress,notgoalcompletion. Fullgoalactive.
All RUNNING/NEXT paragraphs below are preservedhistoricalcheckpoints.

D414 VERIFIED-WAIT continuation2026-10-11 05:44:09 Asia/Shanghai: same active
serviceInvocationID2a7b6ec60c284d0981714c35296b037e/tmux tc-d414-7b confirmed
at every poll from05:33to05:44.3590created/sent,3589raw/validatedresponses,
all4000contracts,0HTTPfailures;notterminal. Watchdogsample3391 age0.023s,
hostavailable52.326GiB,service53.183GiB,disk147.339GiB,high/max/OOM/swap0,
noabort. Originalarrivalwindow remaining587.815s. Previous05:33checkpoint
had3046sent/3045responses;05:25had2511sent/2506responses;05:19had2026sent/
responses;05:13had1483sent/1482responses;05:06had1030sent/1025responses,
all0failures. FrozenV1 andsingleV2 fully reread duringthiswait;existingterminal
audit andownedremote-stop entrypoints inspectedread-only,notexecuted.
Rawjournals retain all observations. Prior turns and this turn are verified
waits,not blockers. FullPLAN/STATUS were read afterlastcompaction;currentPLAN
SHA rechecked unchanged,latestSTATUS andD414contract reread. monitor-experiment
skill used;no finalperformancecomparison from incomplete data.
User reconfirmed once-only deliverycache: existingD78/D80 reused,no republish,
no newpool,real reads/transfers/competition/materialization remain counted.
No retries,restarts,livechanges,tests,hashscans,remoteoperations,cleanup or
newperformanceexperiment. No performanceclaim.
NEXT: keep monitoring SAMEhandle -> terminal release/restore/stop/preserve/
audit/backup ->7BvLLM. Diskbelow150GiB forbidsnewheavyjobs;100GiB runtimefloor
unchanged. Keepgoalactive. Taskownedlivecheckpoint remainsunstaged;critical
main/baselinebackups alreadyverified below,protecteduserchangesuntouched.

D414 VERIFIED-WAIT checkpoint2026-10-11 04:58:55 Asia/Shanghai. Same live
tmux tc-d414-7b/serviceInvocationID2a7b6ec60c284d0981714c35296b037e repeatedly
polled from04:51to04:58;not restarted.529created/sent,527raw/validatedresponses,
4000registeredcontracts,noHTTPfailures,terminalreceiptstillabsent. Watchdog
sample713 age0.510s,hostavailable55.231GiB,service48.374GiB,disk149.317GiB,
high/max/OOM/swap0,noabort. Originalarrivalwindow remaining3301.718s.
Previous goalturn madeprogress by launching/backingup D414;this turn is a
verifiedwait on an actuallylivehandle,not blocked or a completedperformancepoint.
Only small local log/status reads;no code/config/test/hash/cleanup/remotechange.
NEXT remains monitor SAMErun toterminal thenrelease/restore/stop/preserve/audit/
backup and7BvLLM. Do notlaunchanotherheavyjob orrestartonobservationtimeout.

D414 live checkpoint2026-10-11 04:50:41 Asia/Shanghai: same tmux/service identity
below remainsACTIVE;89created/sent,87raw and87validatedresponses,4000registered
contracts,no HTTPfailure recorded. Four worker receipts present;fullrun is NOT
terminal. Watchdogsample225 age0.851s,hostavailable56.605GiB,service46.538GiB,
disk149.794GiB,high/max/OOM/swap0,noabort. No premature performance verdict.
Disk below150GiB now forbids NEW heavylaunch;currentrun staysabove100GiB floor.
No livecache cleanup,remoteadministration,hashscan,test or configurationchange.
Monitor same handle untilterminal;do notrestart. Then closure sequence below.
Running-checkpoint backups pushed andremoteSHAverified:main
82e980f70c852dbc99939418f3056f3b03307b75;baseline
8d62172c720bd7a0a3073f86894404f0ba4ebf79. These documentationcommits are NOT
the executioncommits below. Exact taskdocs only,diff/secrets checks passed.
This final livecheckpoint is subsequentunstaged task-ownedstatus;userdirties
preserved. Fullgoalactive,no blocker,no completedperformanceclaim.

D414 RUNNING since2026-10-11 04:46:28 Asia/Shanghai, tmux tc-d414-7b.
Main execution2860b53740bfea5fc8afae7e2139a13e2855a50f pushed/remoteSHAverified;
baseline execution35dda117faecee5375c146a2c23b16fc2562d82a remoteSHAreverified.
Service primelora-tc-svc-6fc1e91952774c4dbf89415c84e35a60.scope,
InvocationID2a7b6ec60c284d0981714c35296b037e; aux
primelora-tc-aux-d4140000000000000000000000000001.scope,
InvocationID018dc11883334acf8eddaa38a6f921db.
Deploymentnotice1172460.705107463,business t0=1172520.705107463.
Actual ingress log confirms single_request_loopback_no_retry_v1 in service
domain. Existing external watchdog active;sample26 nohigh/max/OOM/swap/abort,
hostavailable84793475072B,service26035859456B,disk161509662720B.
Source4000/132IDs/458224tokens/viewSHA all revalidated; no new trace/pool.
Remote link1000Mb/s/full, unchanged published protocol;7Bclock
remote-process-monotonic:a8f7bdc1c5b54aef9e5f1dd2e84c38eb.
Remote3B/7B/monitor InvocationIDs respectively96a1fa0166b3408eabe6e2c9ac0715f6,
3af63d67d4e9495dac306a5e227d9e44,8e70bbed701040e7a7665fc59ce936ae.
Loader INSTALLED with exact D414 overlay7b_repaired/install.json/backups;
restore only AFTER actual GPU release. Do not stop/restart services in business.
NEXT: monitor same livehandle/local journals only. On terminal actualrelease ->
restore exactoverlay -> D414stop_remote.sh -> preserve resolver/worker/remote
evidence -> full4000audit with D394 ->save/backup ->7BvLLM. No concurrent tests,
hashscan,cleanup,remote administration,parameter search or second heavyjob.
No final performance result yet. Goalactive,PrimeLoRA D371sealed;W1/W2/G2open.

D414 PRELAUNCH checkpoint 2026-10-11 04:44 Asia/Shanghai. No live GPU job.
See D414_SERVERLESSLLM_FULL_W0.md. Only D413 connection-lifetime repair differs
from D411; same original4000/public native configuration/commonD394SLO.
Raw d414_20261011 under existing serverless_qualification root contains reused
runner/remote scripts, identical HTTP config, new source view, resource preflight
and protected snapshot. No GPU gate/parameter sweep. Native compile diskcache
3858files/104734877B, SHA5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791.
D411 downloadedobjects ONLY reclaimed with evidence/reference checks:
1320files,3305664512B observed release; diskafter161526784000B. All raw records,
original pool and published remote cache retained. Three unreadable sameUID
proc entries disclosed in cleanup receipts. Do not redo cleanup/byteaudit.
Baseline execution35dda117faecee5375c146a2c23b16fc2562d82a already backed up.
NEXT: backup this preparation; start unchanged published remote services and
health-check; install exact loader-only overlay using NEW D414 backups/receipt;
fresh gates/input validation -> launch one full W0 in tmux tc-d414-7b.
Loader installer /home/qhq/relayserve_serverless_llm/scripts/
relayserve_v4_3_apply_serverlessllm_vllm_store_overlay.py, SHA
da6a14276f92186457c44c215d45d1ca9d18f7bbea9cdec6665737743f9c5978;
preflight already passed, NOT yet installed. Actual release precedes restore.
Follow D414 doc for closure, then7BvLLM. Do not restart D411 or new shortgates.
User reconfirmed one-time deliverycache approval: reuse D78/D80,not republish.
Earlier next-action paragraphs below are retained historical checkpoints.

D413 CPU-only checkpoint2026-10-11 04:35 Asia/Shanghai: no live GPU/remote job.
Bounded real-loopback event-order test reproduced a failed secondPOST when the
peer closes after pool selection; unchangedproduction test1 passed0.159s.
This proves the possible mechanism,not the unrecorded historical socket IDs
ofD411. Baseline extraingress now owns oneconnection perPOST viaforce_close=True,
limit0; no retries,timeouttuning,backend/policy/scaler/loadingchange. Records
single_request_loopback_no_retry_v1 andnumeric errno whenavailable.
Newtest has2successfulPOSTs/distinctconnections/no reuse; ambiguousnativeclose
afterPOST remainsonevisiblefailure/no retry.8ingresstests passed0.345s; full
measurement/launch/router92CPUtests passed7.538s,4GiB/zeroswap/CPUs2,3,26,27.
Reports D413_SERVERLESSLLM_INGRESS_REPAIR.md inmain and
D413_INGRESS_CONNECTION_LIFETIME.md inbaseline. No newperformancevalue/claim.
AllD411failuredata unchanged;oldhelperSHAdb0c5232692215a1dd3b8ded3787b7f0dd6dc546e5e8a33bdfc9fbaed5353740
verifiedfromD411executioncommit,notmutablecurrenthelper. Currentdisk~158.2GB
below150GiB newheavyfloor;cacheNOTreclaimedyet.
Baseline testedcheckpoint35dda117faecee5375c146a2c23b16fc2562d82a pushed to
origin/main andremoteSHAverified. Exact4taskfilesonly;diff/secretschecks passed.
Protected3userdirtySHAs unchanged;plan/frozenV1/singleV2/D411curated3SHAunchanged.
Main D413report/status checkpoint03938d3c857b5768deb20e0a723d069734ff79f9
pushed tofaaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified.
Onlythis postpushconfirmation is subsequentlyunstaged taskwork;protecteduser
changes remainuntouched. No GPUexperiment/remoteoperation/cachecleanupoccurred
inD413. Fullgoal remainsactive;this is oneboundedmeasurementmilestone.
ImmediateNEXT: safelyreclaimonlyverifiedD411downloadobjects withreferencechecks;
thenone new-key full4000W0 withsameD411configuration/commonD394SLO andreal
resourceledger. Reuse D406wholepool/D409-D410backendqualification,not another
100/500/1000GPUgate. D411isNOTrestarted,not eligible ranking. PrimeLoRAsealed;
afternewfullclosure/audit/save return7BvLLM,notlocalparametersearchor3Bjump.
The older ImmediateNEXT below is preservedhistory and superseded byD413.

TERMINAL checkpoint2026-10-11 04:22 Asia/Shanghai: D411 completed original
4,000 arrivals/submissions/terminals,3,997validatedresponses and3failures.
No retry/restart/liveconfiguration change. Modelqualification passed=false,
error "external HTTP qualification failed; do not label binding as correctness".
Outerlaunch service_returncode1, native_gpu_context_release_confirmed=true,
service_path_removed=true; fresh nvidia-smi computeprocess census empty.
Nativeoverlay RESTORED from exact D411receipt,3ownedremoteunits STOPPED by
InvocationID and portsabsent,emptyownedauxscope stopped. No live heavyjob.
Do NOT restart D411 or launch newGPUwork below150GiB disk startfloor.
Physicalsummary measurement_complete=true,N_plan=N_terminal4000,
n_native_contract_complete3997,n_correct=null,noopenleases,U16090.164119827561
GPU-s; below offlineaudit has now revalidated ledger. Do not label this G1feasible.
Resolver journals/receipts/run_contract,executed scheduler and Ray worker/raylet
logs copied exclusively intoD411raw; no objectcache reclaimed yet. Remote
transfersSHA d4a2e99fd5fd353743a84fc79a6b0a14f3d22f5316414f4fe69f995dcb104e39
and monitorSHA31cec0d2961f29f3563816e40d56b9389d82aa1472f79e40f35747bab28c314c
match source. Protectedbefore/after same96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d:
only unchanged historical userreplay mismatch; no newprotectedresult change.
Offline audit COMPLETE: main existinganalyzer adds explicit
--outcome complete_with_failures,defaultsuccess still rejectsfailedruns.
23CPUtests passed0.119s/4GiB/zeroswap,actualD411audit passed.3,997 native/source/
adapter/content/timing contracts,error0ms,457845tokens. CommonD394 joint3938/
4000=98.45%,TTFT3939,TPOT3996/3997;observed_feasible=false due3failures.
Conditional mean/P95TTFT950.811265/730.630836ms,TPOT26.413121/31.833669ms,
E2Emean3940.953967ms. U16090.164120GPU-s,4.022541/offered;not eligible ranking.
132HTTP/131480060wireB/3300789780verifiedpayloadB,3865cachehits,0packs.
5actualservedepochs,allreadyreceipts verified,no pendingallocations;
not5physicalGPUs. Fullrawunknown numerical/correctness flags unchanged.
Audit paper_results/ieee_tc/baseline_audit/d411_7b_full_w0_failure/audit.json SHA
eb00a0e729ac9e43b53f67494c25ac99425e72b0c5b88283b783f1517dee617f.
Companion234-filemanifest SHA7de3171d8e8ea7655aa4a94965e80d69355d15bb49fd1a503ef5907309c2e00d,
allhashesverified; ingress_failure_evidence SHA
a6186d3168456deaeafc41670c92b509fb4047511cb74012894ca87e72c26021.
Servicepeak57622568960B,high/max/OOM/swap0. No cache reclaimed yet.
Baseline closure documentation checkpoint020414e367e063131909a51e2cee82c2f2415abc
pushed toorigin/main and remoteSHAverified;not the D411executioncommit.
Main closure checkpoint902a52428f501406a27e781261c68377b84976b5 pushed to
faaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified04:28. Exact7
taskfiles staged (2docs,analysis,tests,3curatedJSON),diff/secrets/23CPUtests and
234-filehashverification passed. Baselinebackup above separately verified.
Only preexisting userdirties remained before this finalcheckpoint. No cache
cleanup or transportfix occurred. This is progress,not goalcompletion.
ImmediateNEXT: CPU-only bounded keepalive hypothesis test in
existing ingress tests; retain failed D411 and unchangedsuccessgates. No
transportfix or newperformance attempt yet. Keep PrimeLoRAsealed;7Bnextbaseline
follows closure and evidence-based decision,not an automatic win claim.
All RUNNING paragraphs below are retained history, superseded by this checkpoint.

D411 verified-wait continuation checked2026-10-11 04:11:39 Asia/Shanghai:
same active service/InvocationID426554ae3dcf4dd2999ceeecf4a2e012,
3642created/sent,3638rawresponses,3635validatedresponses,all4000contracts,
3HTTPfailures. Watchdogsample3484 is0.174s old:hostavailable52.273GiB,
service53.191GiB,disk147.697GiB,high/max/OOM/swap0,noabort.
Remaining originalarrivalwindow493.951s;no terminalresult yet. Same livehandle
polled repeatedly from03:26to04:11 including bounded verifiedwaits;do notinterpret
waiting as stopped or restart.
This is incomplete progress,not a performance verdict. Disk below150GiB forbids
a new heavylaunch;currentrun remainsabove100GiB runtimefloor. No livecleanup,
hashscan,remoteadministration,cachepublication,tests or secondheavyjob.
NEW failure evidence: req_02460 and req_02751 returned502 from service-owned
loopback ingress, ServerDisconnectedError after forwarding in1.202/1.117ms;
no retries. Neither ID appears in native serve.log as checked03:56. Ingress
failures are5.000948/5.001383s after earlier completedresponses, consistent
with but NOT proof of keepalive-close reuse races (aiohttp3.13.3 client default
15s, actual nativeenvironment uvicorn0.41.0 server default5s). No socket-ID
trace exists. Do not attribute to remoteHTTP, GPUcapacity or baselinepolicy.
See D411_SERVERLESSLLM_FULL_W0.md incident section for exact evidence/sources.
Keep the live run unchanged; terminal helper will retain failed qualification
and physical terminals. Existing fullanalyzer asserts success: do not bypass
that assertion or promote3998responses to4000. After cleanup classify failure,
retain correct/failure counts and actualresource ledger, then if needed verify
transport hypothesis with CPU-only existing ingress tests before any repair.
No hidden retry, same-key performance rerun or new threshold is authorized.
Third failure recorded04:11: req_03635, ClientOSError after1.136ms forwarding,
5.000160s after req_03628 response; no failed-ID native serve.log entry. Same
unproven keepalive-race hypothesis, errno/message and socket-ID not logged.
Retain this newfailure; no midrun patch/retry or alteredperformance contract.
04:00 read-only analysis-entry audit: reuse baseline
summarize_serverlessllm_replay.audit_tc_http_journal (terminal-only, preserves
failed offered rows), native.validate_http_observation/audit_backend_receipts
for actual successes, and main frozen_slo_summary (failed rows stay in offered
denominator, observed_feasible false). Existing tests explicitly cover failed
offered rows; not rerun during business. Physical terminal recorder already
preserves failures. Current audit_d402 integration still asserts allsuccess,
accesses metrics on every rawresponse and assumes timing keys on every slice
row: do not bypass these assumptions with filled/removed rows. If integration
is extended after release, add an explicit diagnostic-failure outcome branch,
keep successful-path validation unchanged and test missing/failed/zero-success
populations; retain raw unknown numerical flags. No new parallel framework.
D412_VLLM_MEASUREMENT_REUSE_AUDIT.md records read-onlynext-entrypoint audit:
oldvLLM payload/EOF timing,serviceactivityinauxdomain,latepublisher and envelope
lease gaps remain. ExistingSSEparser alone is insufficient:clienttokenreceipt
is not D371engine-token-event timing. Official/local0.30.0 timingbuilder's
TTFT is scheduled-to-first,not offered-arrival-to-first. Reusepuremeasurement/
commonlauncher,not PrimeLoRAcontrolpolicies;noimplementation/newqualification.
D411terminalclosure remains the immediate next action below.
D412 documentation/status backupbe55e25bd4ac7b2334961bb9173f4ba453227834
was pushed tofaaslora_origin/retry14_continuous_queue_v2 andremoteSHAverified.
This continuation adds monitorcheckpoint and read-only incident analysis;no
code/configuration change,new experiment,comparison result,cleanup or
remoteoperation. User reconfirmed one-time deliverycache approval; already
implemented D78/D80 is reused, not republished.
D411 incident documentation backupc46aa4654a342bab874c531e681092849b661552
pushed tofaaslora_origin/retry14_continuous_queue_v2;remoteSHAverified03:58.
Exactly two docs staged;diff/secrets checks passed,no runtimecode/bundles changed.
No GPU smoke or broadhashscan during livebusiness. Latestmonitorcheckpoint and
analysis-entry notes in this status are the only subsequent task-owned
uncommitted change AT THAT CHECKPOINT; closure has since added D411report,
explicit offlinefailureanalysis/tests/curated evidence. Preserve usermanifest
dirty. Prior goal turn was verifiedwait;currentturn closed the run and completed
failure-preserving offline audit,not blocked and not goalcomplete.

D411 RUNNING since2026-10-11 03:12:24 Asia/Shanghai, tmux tc-d411-7b.
Do NOT restart or launch another heavy job. Main execution
896af899396983350c4544a9d0277283bc28339d, pushed and remoteSHAverified;
baseline execution2a5dfd1154ba93c82d9b7076fde0a8521543933b.
Baseline running-checkpoint backupf8af38bcb7a401114e3782e54bd5b1396a40fc76
pushed toorigin/main andremoteSHAverified;not the executioncommit.
Service primelora-tc-svc-cdae00875efc4791b2fad4ef286ef88f.scope,
InvocationID426554ae3dcf4dd2999ceeecf4a2e012;aux
primelora-tc-aux-d4110000000000000000000000000001.scope,
InvocationIDe1a1176feced4bd5ac90df1c38152e09.
Deploymentnotice1166817.112284243,business t0=1166877.112284243.
Full source beforelaunch validated4000/4000,132actualIDs,458224targettokens,
arrivalwindow3963.9089031219482s (~66min). No final performance result yet.
Remote link1000Mb/s/full,publishedcache reused;7Bclock
remote-process-monotonic:158e79ec85e345c19ea6fb6250184f23.
Remote3B/7B/monitor InvocationIDs respectively
3dadc35fa0e8430d8ef0a231e943546c,f5a284ee1ad643d6b4417e3987500387,
43c6e563e84f43689b0b27a570ea1255. Keep unchanged during business.
Loaderoverlay INSTALLED with exact7-memberreceipt inD411raw;restore AFTER
actualGPUrelease from that receipt. Existingwatchdogactive,4000requestcontracts
registered;no pressure/OOM/swap in initialsample. Continuousguard authoritative.
Preflight passed;protected147-entryhashsame96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d,
sole unchanged historicaluserreplaymismatch. No oldresultchange.
NEXT: monitor existingtmux/localjournals only,no live remoteadministration or
hash/cleanup. Afterterminal actualGPUrelease ->restoreoverlay ->stopownedremote
via D411stop_remote.sh ->preserveworker/resolver/remoteevidence ->auditfull4000
with existinganalyzer/commonD394 ->save/backup ->7Bnextbaseline. Do not repeat
this key for a better result,retunePrimeLoRA,or jump3B beforecomparison.

D411 preparation checkpoint follows. One full original
4000W0 development observation with unchanged D410 public graph configuration.
See D411_SERVERLESSLLM_FULL_W0.md. Existing analyzer now explicitly validates
full4000 count/source_count/W0/rate1 and exact offered view;19CPUtests passed
0.095s in4GiB/zero-swap scope. No online baseline code/algorithm change.
Raw root `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d411_20261011`;
existing runner/config/source view prepared, no new trace/pool/checkpoint.
Baseline execution2a5dfd1154ba93c82d9b7076fde0a8521543933b.
Native compilation diskcache now3858files/104734877B,aggregateSHA
5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791;
recorded beforelaunch, distinct from cold LoRA/HOST/KV/GPU state.
D410 downloadedobjects ONLY reclaimed afterbyte/source/inode/reference checks,
600files/1474429899payloadB,1476268032B observedrelease;
diskafter161905242112B. Originalpool/rawlogs/remoteimmutablecache retained.
Three inaccessible sameUIDproc entries disclosed; receipts inD410raw.
Preparation andlaunch steps completed above. No100/1000 repeat,
no PrimeLoRA tuning or3B jump. The following D410 next-action is superseded.

D410 TERMINAL, started 2026-10-11 02:40 Asia/Shanghai, released before03:00.
Original1,000 W0 requests,
no adapter remapping, fixed_length_greedy_v1, public enforce_eager=false as D409;
no new policy or backend-limit changes. This is development validation, not
full4,000 qualification or a baseline optimum. Do NOT restart the closed run.
Raw root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d410_20261011`.
Tmux `tc-d410-7b`; execution main d6fbaaf245f236e6f05babb2f1b7f7919e8e02f7,
baseline336576bc8a3e7aca34febad52372911289a9ca87.
Service primelora-tc-svc-1913f29ca4e94a49accc7c3057578dc1.scope,
InvocationID e29a72bb3c284637a1b6f62d655fd6da; aux
primelora-tc-aux-d4100000000000000000000000000001.scope.
Deployment notice1164859.37841325, business t0=1164919.37841325;
unchanged60s notice/1800s qualification protection/commonD394SLO.
Preflight passed, actual contained launch/watchdog receipt retained. GPU released,
service domain removed,emptyaux stopped;overlayRESTORED,remote3ownedunitsSTOPPED
by exactInvocationID and portsabsent. No live job.
Remote7Bclock remote-process-monotonic:672039983ce04b839837c414cf6adaf3;
3B/7B/monitor InvocationIDs respectively b00f633737b34674904ea8415758db2c,
4f9d03b0696440d88229ab2742a73596,5c3bd0ca8fb0484598261d680cd81186.
Authenticated health confirms unchanged published-gzip protocol; no new cache
publication, packing or local pre-materialization. Current remote link1Gbps/full.
Cold per-run LoRA cache; native compiler diskcache retained separately:
75,240,459B/290files, aggregateSHA
f8c62659046c116bea0ce79501c52c8824979ca8a676bb5c04cb74671cf7b9a8.
Four actual native compiler-cache hits verified,3.538..3.685s loads and2s/.09GiB
captures each. Actualready notice+138.113..176.535s;all startup/capture counted.
D408 eager1000 is the development comparison; D406500-ID/D409 bounded
reference evidence reused, no repeat just to obtain exact-token agreement.
1000/1000native/source/adapter/content/timing contracts,0fail,122510tokens,
60IDs,4actualservedengines;native/E2E/TPOTrecomputeerror0ms.
TTFTmean2672.734439/P9513276.391335ms;TPOTmean27.846560/P9536.115468ms;
E2Emean5996.914935ms. CommonSLO TTFT915/TPOT1000/joint915=91.5%NONFEASIBLE.
PhysicalU4611.030857GPU-s,4.611031/binding-validrequest,completeleases.
60HTTP/74953014wireB/1474429899verifiedpayloadB,940cachehits,0packs.
All1000D408/D410input/adapter/prompt/target/content contracts matched.
Descriptive joint improvement24.1pp/meanTPOT reduction56.19%,butUincreased0.58%:
nativefourcardstorehold remains,not claimedGPU savings orstrictcausalestimate.
Source0-99/100-499/500-999joint46/100,400/400,469/500;all85missesareTTFT.
Raw unknown numerical flags unchanged. Full4000 ranking remainsOPEN.
Report D410_SERVERLESSLLM_GRAPH_VALIDATION.md;
audit paper_results/ieee_tc/baseline_audit/d410_7b_graph_validation/audit.json,
SHAfa1be0a21dd3b9f842eb41a75b618cf59e30509ad069de29c69e69ee51e3d44a.
Companionruntime_evidence and161-filemanifest,manifestSHA
93811678b42a7fc5658985dba25b5acd180d75b10a5baf6a3de4ee6094f5509a.
15analysisCPUtests passed0.041s. Protectedbefore/after147entries unchanged,
same96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d.
Baseline closurebackup2a5dfd1154ba93c82d9b7076fde0a8521543933b pushed to
origin/main and remoteSHAverified;not the D410 executioncommit.
Next: completeevidencebackup -> reclaim only verified D410downloadobjects and
recheckdisk -> extendexistingpopulationanalyzer tofull4000 withCPUtests ->
one complete4000W0 observation viaexistingqualifiedrunner/publicgraphpath.
Currentdisk160430882816B is below150GiB startgate;do notlaunchuntilrechecked.
Do not repeat100/1000 gates or use95%failureasreason todelete this observation.
No new local parameter sweep, no PrimeLoRA retuning or3B jump. All entries below
are retained historical checkpoints; their NEXT instructions are superseded.

D409 TERMINAL 2026-10-11 02:27 Asia/Shanghai: existing 100-request affected-path
qualification, original W0 prefix, explicit --no-enforce-eager; no other public
backend/scaler fields changed. Raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d409_20261011`.
Tmux `tc-d409-7b`; main execution5a3dac21ffe52e0651c988584feecda003f1a64d,
baseline d6dcf14449e595f461bb933007b62c87423377d2.
Service scope primelora-tc-svc-8d8865fa28cd4c50bbcc940747b9fff2.scope,
InvocationID df291d0a5a9c42b2a14a0866a110f3b9; aux
primelora-tc-aux-d4090000000000000000000000000001.scope.
No live job; do not restart D409. Retained
launch7b_repaired.launch/watchdog.jsonl,replay.jsonl,service.log and native logs.
Native vLLM0.10.2 already
sets inductor compile_threads=1; no new thread parameter/build introduced.
Remote reused D78/D80 read-only published gzip cache,link1000Mb/s/full,
7Bclock remote-process-monotonic:ad5c8352384b424e9db29edaaa2a1f88.
Three owned remote units STOPPED by matchingInvocationID/portsabsent; GPU
released/service removed/emptyaux stopped; overlay RESTORED from exact backups.
No cache republication/request packing. Worker/resolver/remote evidence retained.
100/100 native/source/adapter/timing contracts,0failed,17369tokens,4servedengines.
TTFTmean63240.114772/P95111377.363558ms;TPOTmean31.699636/P9534.570687ms;
jointSLO0/100,TTFT0/100,TPOT100/100. Shortgate NONFEASIBLE,not a fullW0 optimum.
U1033.029004GPU-s,29HTTP/50650865wireB/724665100verifiedpayloadB,71cachehits,
0packs. Four actual PIECEWISE capture completions;no silent eager retry.
Independent D385 reference reused offline:97/100exact,3differences kept with
existing first-divergence/top2 witnesses;raw unknown numerical flags unchanged.
Audit_v2 SHA9c35cbb2661567d718f9671e605cf69b38099fdbcefbebbf9c688f2200db1218,
paper_results/ieee_tc/baseline_audit/d409_7b_graph/audit_v2.json.
See D409_SERVERLESSLLM_GRAPH_RESULT.md.15analysisCPUtests pass0.052s.
Evidence_manifest126retainedfiles allSHAverified;manifestSHA
52c597fe13b0b0bb2782b1a533eba8d12a10a1b2ad5f2a26518107fac2a30fcb.
Baseline closure backup336576bc8a3e7aca34febad52372911289a9ca87 pushed to
origin/main and remoteSHA verified;this is not the D409 execution commit.
Next: original1000 graph/public-config validation on a new key; record retained
native compile diskcache separately from cold per-run LoRA cache,all actual
startup cost remains counted. Do not expand unrelated knobs or repeat100 gates.
Nativecache path ~/.cache/vllm/torch_compile_cache/1c637ca91b/rank_0_0/backbone.
Keep PrimeLoRA D371; then full4000 W0. Disk at gate minimum161426763776B;
recheck>=150GiB+growth before another heavy job. No cache deleted this gate.
The CPU-only D409 entry below is a preserved earlier checkpoint.

D407 COMPLETE 2026-10-11: no GPU repeat. Existing D404100request outputs joined
to independent D385 HF/PEFT reference with source/input/adapter/content hashes:
100/100 identities,94exact sequences,all4weight classes. Six differences kept;
only3share D385's independently recorded divergence position/token and top2
margin0/.015625. Other3margins unknown,not assumed rounding. Eleven CPU tests
pass0.050s. Audit paper_results/ieee_tc/baseline_audit/d407_7b_reference/audit.json
SHA7843794689b7aeaee4ac8ed067de5871ef858729130c7f185a0c9566e6c0b5d4.
See D407_SERVERLESSLLM_REFERENCE_REUSE.md. Frozen V1/D393 do NOT require exact
cross-backend tokens; do not reintroduce D391's superseded full-exact oracle.
Bounded numerical evidence is reusable; raw unknown fields stay unchanged.
D408 COMPLETE (2026-10-11 02:07 Asia/Shanghai): original1000 W0 requests,
trace_prefix_v1,no coverage remapping,60actualIDs,target4/min1/max4/keep10,
same D406 native backend/eager mode. Raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d408_20261011`.
Execution main242b3c455e13ac10920f0af8a04695af05f52a7f,
baseline25848cfb57ef15351f800daf7062cffb2a0d3807. No new weights/traces/cache.
1000/1000 native/source/adapter/content/timing contracts,0fail,122510tokens,
4actual served engines,live scheduler/no exception/empty allocations.
TTFTmean4683.000724/P9536802.779816ms;TPOTmean63.561872/P9568.849439ms;
E2Emean12420.357748ms. Common D394 TTFT828/TPOT800/joint674 of1000:
67.4% joint,NONFEASIBLE,not a4000-request comparison/config optimum.
Physical U4584.427947GPU-s,4.584428/binding-valid request,complete releases.
60HTTP/74953014wireB/1474429899verified payloadB,940cache-hit requests,0packs.
Raw n_correct=null/numerical flags unchanged; reuse bounded D407 separately.
Audit v2 SHA3a159e40315aa5291e70972b621dd77a1222f46c0b308785022c514cfcb28ca4,
paper_results/ieee_tc/baseline_audit/d408_7b_compatibility/audit_v2.json;
earlier audit.json retained. Companion evidence_manifest189files.
Report D408_SERVERLESSLLM_1000_RESULT.md. Offline analyzer15CPUtests pass0.050s.
GPU contexts released/service removed/empty aux stopped/overlay RESTORED.
Remote3ownedunits stopped by InvocationID,MainPID0/portsabsent; transferred
remote logsSHAverified,resolver+worker evidence retained. Protected147entries
before/afterSHA96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d
unchanged,sole historical user replay mismatch. Health prelaunch corrected from
system Python lackingnumpy to existing nativeenv before any business request.
No live job; do NOT restart D408 or earlier completed gates.

D409 CPU-QUALIFIED, NOT GPU-RUN: D408 startup and steady decode both matter.
Ready receipts notice+167.977..168.029s; first100meanTTFT35.417s. Source100-499
and500-999 jointSLO78.5%/70.6%,TPOT~63ms. Helper had fixed eager=true, whereas
official pinned backend exposes false/hybrid. Existing helper now has explicit
--no-enforce-eager,default unchanged,only one public backend config field differs.
No RR/scaler/loader-policy change or silent eager retry. Baseline89CPUtests pass
7.234s in4GiB/zero-swap scope; see baseline docs/ieee_tc/D409_SERVERLESSLLM_PUBLIC_EXECUTION_MODE.md.
Baseline checkpoint d6dcf14449e595f461bb933007b62c87423377d2 pushed to origin/main;
remote branch SHA verified. D408 evidence manifest189files SHA
dbaa0ab5343027b29b0bcc964c1601a8d876f42c80fad63d1dd71ba220ed06b3,all189verified.
Next: full plan/status + safety reread, one new-key affected-path100 graph/native
timing/adapter/memory gate, then original1000 compatibility if valid. Reuse D406
content/full-ID evidence, verify graph slot execution rather than assuming it.
Keep target4/min1/max4/keep10,backendlimits,FP16,common60s/1800s unchanged.
Hypothesis is decode reduction versus capture startup/memory trade-off,not proven.
Do not expand unrelated parameters; retain failure and return to full4000W0
after this bounded public-config check. Do not repeat numerical probes merely
for token exact-match selection. Preserve PrimeLoRA D371 throughout.

D405 COMPLETE 2026-10-11: audited historical replay files: clean7B, earlier7B,
3B each4000reported successes but132actual logical adapter IDs. Original7B
first500/1000 touch46/60; D404 native100 touches29. D80 whole-pool HTTP evidence
does NOT establish whole-pool engine execution. No historical500engine gate
found in the audited sources. Inventory JSON SHA
62b649e75ca1f6bfd5fd38367716d17e3d78231d1991b512aa10a0c9fa6ed0eb.
Existing baseline helper now has opt-in adapter_coverage_v1, strictly500,
qualification-only/nonformal. Existing first500 prompt/arrival/target rows,
sorted static pool IDs, explicit original/derived row provenance; no new trace
or weights. Publisher/physical ledger/deadline share the exact view.87CPU tests
passed7.478s in4GiB/zero-swap auxiliary scope. Baseline checkpoint
7a37e1f9b3220b26f9f0166e36e2821e84aa1d4b pushed and remote hash verified.
See D405_SERVERLESSLLM_COVERAGE_INVENTORY.md and baseline detailed report.

D406 TERMINAL 2026-10-11; former tmux `tc-d406-7b` ended. Exclusive raw root
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d406_20261011`.
Execution main6eed716bed1ed994ab1d35e7a00c6ecf0469e4d7,baseline7a37e1f9b3220b26f9f0166e36e2821e84aa1d4b.
500/500responses/native contracts,exact500logical IDs,0failure,68,925tokens;
fouractualservedengines,live scheduler/no exception/empty queue. Native/E2E/TPOT
maximum recompute error0ms.500realHTTP transfers420740410B,verifiedpayload
12985984450B,0packs/0cache-hit requests. Physical GPU-s2717.3239195765927,
complete leases. Raw n_correct=null/independent numerical=false preserved.
Audit SHA24da017f3612e2e15c8533b03d30ffa3ad612c8b52e6dfa3a7bd2c445cb419d7;
main paper_results/ieee_tc/baseline_audit/d406_7b_coverage/audit.json.
Evidence manifest602files allSHAverified,manifest SHA
676f15919945163bd88196c582fa6ee1d19dfaeb85d1854ae6973c15c977d7dc.
Baseline evidence backup25848cfb57ef15351f800daf7062cffb2a0d3807 pushed/verified.
Report D406_SERVERLESSLLM_500_ADAPTER_RESULT.md. ActualGPU released,overlay
RESTORED,remote3ownedunits stopped/portsabsent,empty ownedauxscopes stopped.
Protected147entriesbefore/afteridentical;sole old mismatch user's replay edit.
Remote link1000Mb/s/full. Fullbyteaudit passed before cache-only reclamation:
objects /tmp/tc-d406-7br/remote_artifacts/objects removed;originalpool/remote
cache/receipts/journals/rawlogs retained. Observed reclaimed13003907072B;
diskafter162353836032B. Raw cache_cleanup_before/after.json record checks.
Reauditing removed cache bytes requires rematerialization; do not redownload
wholepool merely to repeat an already verified gate. Independent numerical
qualification/actual content diversity remain distinct from native binding.
View SHA286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d.
Public target4 matches existing engine max-seqs4/max-loras4; min1/max4/keep10
unchanged. D404 observed overlap2/engine supports capacity validation, not GPU
saturation. Functional mapping is not a performance comparison or optimum.
Next audit reusable numerical qualification,then original1000 compatibility/
public configuration validation and frozen full4000 W0. Existing analyzer's
explicit500view validator:6CPUtests pass0.036s; D404 original-population
reanalyzed numeric fields unchanged, no performance repeat. No live job.
Do not restart completed D406,D404 or old D72/D73 pair.

D404 TERMINAL 2026-10-11:100/100 native contracts,0failed,17,369tokens;
four actual served engines/ready receipts; exact repaired source and contained
actor readback,live loop/no exception/empty allocation queue. No repeat needed.
Raw `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d404_20261011`.
Execution main abd53201f0ed5959d554541a4b86beae2cfef108,baseline3cc25a8.
Source scheduler SHA7f42b0cce82f970392b3d484767a13563fb2d29f6c6e870aa8f29b86fb972b91.
TTFT mean82.940567s/P9587.723942s;queue82.332853s;serviceTTFT433.976789ms.
All29real HTTP/50,650,865B/724,665,100verified payload bytes;71cache hits,0packs.
Native/timing recompute error0;physical GPU-s1362.0847175046802,closed leases.
Report D404_SERVERLESSLLM_QUEUE_QUALIFICATION.md;curated d404_7b_queue/audit.json
SHAde80d7f8adbe92e4aaec8a6a4950d905ffd3da263a5c7acdff31c4c514290f3e;
companion manifest125files.100request contracts identical to D402;startup timing
differs,so do not attribute all latency change to queue repair or compare to D371.
GPU/service/aux released;remote3ownedunits stopped;loader overlay fully restored.
Baseline evidence checkpoint6f432c8a079fde9c8db656d8063f6ceef2ae20dd pushed.
Protected before/after unchanged. No live job. Full baseline/config optimum OPEN.

1. D403 opt-in `request_identity_v1` repair implemented in existing baseline
   source adapter; vendor/historical views untouched. Original ordering,placement,
   scaling,loading,migration and cadence preserved. Official sibling FCFS source
   supports identity removal. 81 CPU tests passed in7.055s,4GiB/zero-swap scope.
   See main `D403_SERVERLESSLLM_QUEUE_REPAIR.md` and baseline detailed report.
   Baseline checkpoint3cc25a87f3895a408a16efe2ef9c75d0e8780ff3 pushed/verified.
2. D404 scale-out and D406 whole-pool native execution gates passed above.
   Do not repeat them or D72/D73 polling pair. D72/D73's1000 prefixes remain
   R2/local/failed,not new-contract compatibility;reuse diagnostics/assets.
3. Audit/reuse independent numerical correctness evidence before promoting
   full-system qualification; original1,000 compatibility and reasonable public
   configuration validation/freeze ->one matched full W0.
   Historical target2/max-seqs4/min1/max4/keep10 is diagnostic,not selected optimum.
4. Preserve D371. No PrimeLoRA retuning without comparison evidence or3B jump.

Still open:complete7B comparisons,Resident reference,W1/W2/G2,3B optimization/
comparisons,A1-A5,S1-S13,CE/cost supplements,review/design documents,later figures.
Main roles:vLLM,S-LoRA,ServerlessLLM,HydraServe,Loquetier,ElasticLocality;
dLoRA only approved3B scope. HydraServe/Loquetier not automatically qualified.
The full plan and preserved history contain all approved matrix details;
this status is not a shortened experiment plan.

## Safety,remote,data protection

- One heavy job. Inference disk start max(150GiB,100GiB+1.5growth),stop100GiB;
  service memory high72/max80GiB,swap2GiB,CPUs4-23,28-47;aux4GiB,swap0,
  CPUs2,3,26,27. Actual worker containment/watchdog/cleanup precede GPU runs.
- D406 service peak64,123,109,376B,swap0,high/max/OOM0;host available minimum
  59,220,758,528B,disk minimum149,497,323,520B. Recheck before heavy work.
- All D402/D404/D406/D408/D409 workers/service scopes released. Loader overlay
  was restored to exact preimages after each run;empty aux scopes stopped.
  D410 has also released GPU/service/aux,restored overlay and stopped ownedremote
  units by exactInvocationID. D411 is now terminal with GPU/service/aux released,
  overlayRESTORED and3ownedremoteunitsSTOPPED. No live heavyjob;see closure above.
- Strict SSH alias primelora-artifact-174,lab14@10.199.227.174:8122;
  data192.168.4.174 ports18080/18081. Credentials outside Git.
  Actual current link1000Mb/s/full;historical100Mb/s was a different state.
- Reuse D78/D80 immutable gzip cache;no republication/request packing. Real reads,
  transfers,competition,local preparation remain in latency;no cumulative-span
  subtraction or live remote administration. Artifact-only reserve+growth disk
  gate does not relax inference gates.
- Small resolver journals/receipts,Ray logs and scheduler copies preserved under
  raw D402/D404. Private caches `/tmp/tc-d402-7br` and `/tmp/tc-d404-7br` remain;
  only verified re-creatable objects may be reclaimed after reference checks,
  never raw records,unique weights or frozen traces.
- D408 after audit reclaimed only600 downloaded object files at
  /tmp/tc-d408-7br/remote_artifacts/objects. Original source hashes/different
  inodes,closed ownedruntime,readable same-UID fd/cwd/root+fuser and tracked
  future config references checked;3inaccessible proc entries disclosed.
  Retained allraw/receipt/journal and remoteimmutablecache. Observed release
  1476124672B;diskafter162219474944B. Rawcache_cleanup_before/after.json.
  Re-auditing removed bytes requires rematerialization;do not redownload just
  to repeat a passed check. Recheck>=150GiB plus predictedgrowth before D409.
- D408servicepeak52653588480B,high/max/OOM/swap0;hostavailablemin59081170944B.
- Before D410, D409's downloaded objects only were reclaimed after byte/source
  hash, distinct-inode, closed-runtime and readable process/reference checks.
  Removed /tmp/tc-d409-7br/remote_artifacts/objects; retained original pool,
  remote immutable cache, rawlogs, receipts/journals and source symlink targets.
  Three inaccessible same-UID proc entries disclosed in raw cleanup receipts.
  Reclaimed725557248B; diskafter162078203904B. See D409 raw
  cache_cleanup_before.json and cache_cleanup_after.json. Historical evidence
  manifest predates this cleanup and is not rewritten. Do not redownload for
  a repeated already-passed byte audit. D410 launch rechecked growth/resource
  gates; continuous watchdog was active throughout that run.
- D410 terminal safety:servicepeak52803334144B,high/max/OOM/swap0;
  hostavailableminimum58742841344B,diskminimum160506687488B.
  Watchdog closed normally afteractualrelease. D410objects were reclaimed before
  D411 after completed byte/source/inode/reference audit as recorded above;
  originalpool,rawlogs,receipts andremoteimmutablecache remain. Do not repeat
  this cleanup or redownload solely to repeat the completed byteaudit.
- Protected147-entry checks before/after identical,SHA
  96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d.
  Sole historical mismatch:user replay edit,unchanged in this campaign.
- Never stage main `configs/generated/lora_manifest_1000.json` or baseline
  `scripts/replay_openai_trace.py` (04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22)
  and `scripts/run_serverlessllm_relayserve_continuation.sh`
  (337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499).
  Preserve all unrelated tracked/untracked work.
- Tested backups:main faaslora_origin/retry14_continuous_queue_v2;baseline
  origin/main separately. No force,secrets,raw-large-data. Update this status
  before yielding;full goal remains active until genuinely complete.
