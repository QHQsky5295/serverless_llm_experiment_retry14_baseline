# IEEE TC execution status

## CURRENT — D171 provisional 7B timing-gap analysis COMPLETE/SEALED; backup next

20:34 +08 seal PASS: curated SHA361bb4bc1ff7c49ac23449227d4dd31d7829cc87fbc86d7847c0658bb5dd7a15;
22-member 329560-byte bundle SHA89fcdff6f3bf85b65e9b2ccb649972909ff4f1d62fc6de69aadec8010b481a3c.
147 protected entries/Plan/V1/input SHAs/32 tests/table counts PASS. Existing
analyzer AST identical except new offline helper; serving code unchanged.
verify1/session20987 CLOSED0, scope absent, final memory events/swap zero.
Actual verify invocation a72547e3c7e04dcfa0d6775c5ad7b699. CSV exports use Python
csv's standard CRLF; default Git whitespace check flags CR on each record.
Explicit cr-at-eol check passes with other whitespace checks retained; sealed
CSV bytes are not rewritten. Scoped secret check PASS: 11 files/33 entries.
Doc/curated/CSVs now sealed, do not rewrite. NEXT scoped secrets/diff/checksum
backup. No live jobs, no new GPU candidate launched, all qualification caveats remain.

Reused D164 retained 34 MB request projection with D170 candidate thresholds.
No GPU replay, no new data/cache, no serving/config/remote changes. All 4000 native
timelines reconstruct; 2110 timing-joint passes = 52.75% upper bound, NOT formal
joint SLO. TTFT-only 1660, TPOT-only 91, both-miss 139; at least 1690 additional
timing passes needed for 95%. 1547/1799 TTFT misses exhaust their deadline before
engine dispatch. Mean arrival→admission 2.850748 s, admission→dispatch 1.386105 s,
together 90.80% of mean TTFT. Conditional tier tables are not causal effects.
Native numerical correctness/n_correct, globally frozen warm, Resident budget,
old Prime new-metric comparison and G1/G2 acceptance remain OPEN. No CI at n=1.
Doc D171_PROVISIONAL_TIMING_GAP.md + JSON/2 CSV tables complete. 32 tests PASS
(8 new + 24 existing). tests1 actual f9e83061d5124bfd8e66c7e2f199ec8e/session80654
and analyze1 actual 5f637545fc1d49c69ce7064a9c122b6b/session44069 CLOSED0;
both scopes absent, memory events/swap zero; analysis 9.66 s / RSS1102156 KiB.
No live jobs. Disk ~149.86 GiB below new-heavy gate; bounded offline only.
NEXT seal + scoped backup; then ONE evidence-grounded pre-engine/queue candidate,
with historical D161/D162 evidence and current official sources, after audited
space recovery before any GPU work. Do not reanalyze/replay this completed D171.
Prime7B before3B beforeexternalbaselines; later matrices still OPEN. Goal ACTIVE.

## CURRENT — D170 7B common warm measurement COMPLETE/SEALED and BACKED

20:18 +08 backup COMPLETEef38f25735c0d781330b35f4cb254ca1b06d7f7c, pushed
faaslora_origin/retry14_continuous_queue_v2; independentremoteHEADmatched.
Push38893/verify60903CLOSED0.11explicitfiles/64staged+archiveentries passed
secrets/diff/checksum/projectionfixtures/nativecuration/147protectedchecks;
userdirtymanifest excluded. No liveGPU/remote/analysis/tooljobs. Onlythis
post-backupreceipt newlydirty. GoalACTIVE/thisturnPROGRESS, notmodelacceptance.
DoNOTrepeatD170measurement/projection/curation/seal orD169probe/index/cache.
NEXT ONEprovisional7Btiming-gap analysis from sealedD16434MBprojection with
D170candidate thresholds. Correctnessunknown staysunknown: timing-only bounds
are notformaljointSLO; commonbatchcontractstillnotgloballyfrozen. No new GPU
run orservingcandidate selected. Needactualnew-metricoldPrimegap/numerical/
Residentbudget/G1G2 acceptance before3Boptimization/externalbaselines.
Disk160926638080B(~149.87GiB), below150GiBnewheavygate; do notlowerthreshold,
deleteuniqueevidence orstartheavy. Boundedofflineanalysis canproceed. Audit
reconstructiblecache orlosslessarchival onlybetweenheavyjobs, notrepeatedhashes
ofgiantoldD137/D145/D148. Alllatermatrixandfailedcauses remainOPEN.


20:16 +08 sealPASS, doc/curated/tables now immutable, doNOTedit/re-curate.
CuratedSHA0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c.
53member83467B evidencebundleSHA
b532d480f7d8fa7064e46c2c14d99d6bc3b43a73b78613d3456be3261a7e90d3.
Verify1 actual2bceee7e0d16449a99b5701fa9757fd9/session73018CLOSED0,
events/swap0/automaticremovalverified. All source/configbyteidentical6a15274;
147protected/Plan/V1/sources/tables/projectionfixturesPASS. AllpriorCPUscopes
absent withfinalevents0, includingretainedfailedprepare1/2. No livejobs.
NEXT scopeddiff/secrets/checksumbackup then new provisionaltiminggap from D164
retainedprojection. Commonreferencegloballyunfrozen/numeric/G1G2 stillOPEN.


20:14 +08 curation COMPLETE1552/1552nativechecks (1536measurement+16warmup),
194batches/allrequiredGPUready/KV/drain/release/input/token/timingchecksPASS.
WarmTTFT598.018378/927.065033ms,TPOT29.823099/42.672637ms by≤616/>616nativeinput.
Candidate5T/2P thresholds2990.091890/4635.325166ms,59.646198/85.345274ms.
NOT globallyfrozencommonreference, n_correct/numerical/fullpool/G1G2remainOPEN.
U1429.338594330GPU-s/singlelease released;1499resourcesamples/peak12596813824B/
minhost99741794304B/observedmemoryevents+swap0/warnings0. No remote use.
Projection69103CLOSED0, actual56719c5d94164b6f99c6dacbb834321e, twofixturesPASS;
raw1763970424B streamedONCE→29031137B projection. Scopeabsent/events0.
Curate75104CLOSED0, actualf3601432fb844fb6b8927a75d53ab63e, scopeabsent.
Tables/doc complete; exact finalCPU counters/seal/backup pending. No liveGPU,
remoteoranalysisjob. DoNOTreprojectraw/replay/repeatprobe. Postwriteavailable
160967503872B slightlybelow150GiBnew-heavyfloor; recoveronlyauditedspace before
nextheavy, nofloorrelaxation. NEXT verify/seal/scopedbackup→provisionaltiminggap
fromretainedD164Fullprojection, notformalG1G2; nevergiantoldrawreparse.


20:07 +08 SAME run terminal, launch.pass=true/service+watchdog0/physicalrelease
andservicepathremoved confirmed. ExactPID/GPU/scopeabsence20:06:54, no remote
service was started. All194batches ended; complete1536+16nativechecks still
await independent curation, not inferred from progress messages. No replay.
One boundedstreamingprojection launched in primelora-d170-project1-20261002.scope,
session69103 LIVE. Twoexactprojectionfixtures/AST PASS; raw1.7GiB is never
whole-loaded. Project source once; doNOTrestart whileactive orreparseafterpass.
FullPlan(repairedtruncatedmiddle)/V1/currentledger and monitor/analyze/plotting/
github-sync skills read thisturn. PreviousgoalturnPROGRESS; currentPROGRESS.
No serving/config/remote changes. NEXT projectionterminal/cleanup→curation→
reference table/doc/seal/backup. G1G2/numerical/oldPrimecomparison stillOPEN.


19:52 +08 continuation checkpoint: SAME run live, exactserviceInvocationID
b078bcdf6d1f4b428ea90d6ae05e0a13 reverified/218tasks. First512-request round
complete; serial80 = 632/1536 measured requests plus16warmup completed.
Sample611:current5441118208B/peak5706108928B,host106796007424B,events/swap0,
no warnings/aborts. These are incomplete observations, NOT final reference or
G1G2 acceptance. Four50s-spaced light polls completed, exec8583 CLOSED normally;
no live tool session, actualtmux/service experiment still LIVE. DoNOTrestart.
Prepared, NOTexecuted: exactowned cleanup, D96-style streaming projection,
two structural projection fixtures, D169-style curation with exactinput/native
checks and per-group/per-round means. No serving code or frozenconfig changes.
No GPU replay/cachegeneration/remoteoperation/analysis during inference.
NEXT monitor SAMEhandle toactualterminal, then cleanup→projectionfixtures and
ONEboundedprojection→curation→table/seal/backup. D170notyetcommitted; last
backedruntime6a15274. This goalturnPROGRESS(actualmeasurementlaunch); goalACTIVE.
3B/externalbaselines PAUSED until actual7B/new-metric acceptance; entire later
matrix remains pending, no partial sample can close n_correct/G1G2/oldPrimegap.

19:43 +08 ONE actual measure1 started19:40:44 in tmux tc-d170-7b-measure1.
Service primelora-tc-svc-3bfbb824545248308ed14a73a3a2ceb3.scope,
actualb078bcdf6d1f4b428ea90d6ae05e0a13; aux
primelora-tc-aux-7e68b0a182db4f0bb50cc2c1ac84de69.scope,
actualb1e4d789b9254280a4729087e7a6afc5. Native721356/start44853369 exactservice
CPU4–23,28–47; watchdog719257/start44850432 exactauxCPU2,3,26,27.
Actual72/80GiBswap2 and3/4GiBswap0 verified. Sample71 peak5706108928B,
host108504145920B,events/swap0/no warnings; firstwarmupbatch complete.
Prepare3finalevents0/automaticremovalverified. SAME run, doNOTrestart.
Exactterminalcleanup and bounded streaming projection PREPARED, NOTexecuted.
No source/config/remote changes or heavy analysis during inference.


2026-10-02 19:41 +08. Goal ACTIVE; Prime7B only. D169 sealed/backed6a15274,
not repeated. Full Plan/status/V1 and run/monitor/analyze/plotting skills read.
Reuse D168 index + exact D169 batch8 config/source, 1536 measured+16 warmup
calls/194 batches. No serving changes/remote operations/new cache or inputs.
Raw results/ieee_tc/p2_backend_qualification/d170_20261002. Launch measure1
prepared, NOT yet started. Output linear sizing1.846GB; preserve observations,
use bounded analysis; unchanged service72/80GiB and all disk/host gates.
Prepare1 failed generator len(), actualf22a674d90eb4a859ef3cabaa4ff580a/89956exit1;
prepare2 stat.st_size() error, actualebd3690006f144cda840436a2200ac5f/71949exit1;
both retained scripts/logs, no GPU launch. Prepare3 passed, actual
929e6fdb22b0407fa654e6639f6a06f8/8014exit0. Final counters and absence to verify.
No threshold/native numerical/G1G2 acceptance yet. 3B/externalbaselines PAUSED.
NEXT launch once and monitor exact identity→cleanup→reference table→backup.

## CURRENT — D169 common warm runtime/probe COMPLETE, SEALED and BACKED

Backup COMPLETE6a15274af4062c3a66fe7d51e7d7f910bcac895f, pushed to
faaslora_origin/retry14_continuous_queue_v2; independentremoteHEADmatches.
Push72927/verify75520 CLOSED0.11explicitfiles/62staged+archiveentries passed
secrets/diff/checksum/test/protected checks; usermanifestexcluded. No livejobs.
Onlythispost-backupreceipt newlydirty. GoalACTIVE/thisturnPROGRESS, notmodel
acceptance. NEXT actual1536warmmeasurement+16warmup, using D168sealedindex,
D169qualifiedconfig and existing preflight native_warm_reference measure phase.
DoNOTrepeat32probe/CPUtests/content/index/cache work. Beforelaunch boundoutput
growth (probe37MB), retain4GiBanalysislimit/useexistingstreamprojectionifneeded;
no giantwhole-load. No launch/newcandidate yet;7B before3B/baselines unchanged.

19:29 +08 seal COMPLETE. CuratedSHA
6a288d09587488d3bb74d9fbf448c9e9cd6a153186dcffac0f5a6e234b31b3ad;
51member124183B bundleSHA
b5bae030b9c04207f5ead189b3f4a576ce0f2085392f821c879190490a77942f.
Only4newpreflighthelpers+3opt-inwiringfunctions; allotherpreflightAST/serving
sourceunchanged,147protected/Plan/V1/sourceSHAsPASS. Doc/results nowSEALED.
verify1 failed at log-parser assertion: CLI-negative tests print usage before
separate ok line; original verifier/log retained. Corrected verifier counts72
unique preflight tests plus sole named warm error against complete85 summary;
no GPU/test rerun or metric weakening. verify2 actual
3f5ef9007f034e4ea6b198be6ce04d37/session64083CLOSED0/events0/absenceverified.
No livejobs; original37MBraw retained andbound, not addedtoGit. NEXT scoped
secrets/diff/checksum/backup then warm1536+16warmup via SAMEindex/config/mode.
DoNOTrepeatcompleted32probe, content/index audits, or recreate remote cache.
Check measurement-output size before fullcalibration; reuse bounded projection
if needed, no giant whole-load or relaxed4GiBanalysisguard. G1G2stillOPEN.

19:22 +08 actualprobe COMPLETE32/32native,4/4batches. All8 GPUready and actual
decodeintersection .192102/1.590025/.061170/2.270834s. KVrequired283/336/453/496,
actualfree1022before/after,block16. No queuecarryover; originalprompt/nativehash/
tokens/timeidentity/releasechecksPASS. U117.769190706GPU-s/singlelease released;
144samples/peak7570870272B/minhost106752929792B/events+swap0. No performanceranking.
Service/watchdog0; exactscope/PID/GPUabsenceverified19:21. Curate1 actual
fe7f086914274edfbfbe67bd7808ca9a/session7297CLOSED0/events0/automaticremovalverified.
37MBraw parsedONCE; curated JSON/CSV + doc nowcomplete.147protectedPASS.
No live GPU/remote/analysis/tooljob. NEXT scopedverification/secrets/backup,
then D1681536warmmeasurements+16warmup withsameconfig; doNOTrepeat32probe or
tests without sourcechange. Full/numeric/commonwarmthresholds/G1G2stillOPEN.

19:20 +08 warmtests2 PASS13/.459s, actuala3a04c54c1bb454cbb6a41f905247a6d,
session82790CLOSED0/events0/automaticremovalverified. Prepare1 passed sealed
input/configdiff/fullpoolrank16/147protected; actual62d4775b020f4d7a8ee4ac5c45bc3e63,
session74626CLOSED0/events0/automaticremovalverified. ONEprobe32requests/4batches
launched tmux tc-d169-7b-probe1; no reference measurement yet. Service
primelora-tc-svc-b027ea1e83d5417d81acde55031c4c3c.scope actual
f2a5be6accb749adac2fea5f75710159; aux
primelora-tc-aux-e693a130b54c4c7c824d4fe6d4b54b24.scope actual
a3f31da0adac4b9cadbad95de6185895. Limits72/80GiBswap2 and3/4GiBswap0
read back. Prelaunchdisk162762813440B/host112554930176B. No remote operations.
NEXT monitorSAMEprobe→actualrelease/cleanup→table/seal/backup; no restart.

2026-10-02 19:15 +08. Goal ACTIVE, Prime7B only. Full Plan/status/V1 read;
run-experiment/monitor/vLLM + optimization reference used. D168 sealed/backed,
not repeated. Remote once-only cache approval fulfilled D78/D80; no recreation.
New opt-in existing preflight mode binds D168 index and measures GPU-ready
native batches without Prime pending admission. Full/serving code unchanged.
Candidate reference config explicitly8slots/seqs,rank16,.92officialdefault,
same TP1FP16/prefixoff/external envelope; NOT newFull config. Needactualprobe.
tests1 ran85:72existing+13warm;1warmasync error because OS Python lacks
asyncio.timeout. Native runtime is3.12; warm tests will use existing3.12CPUenv,
no implementation fallback. Originalfailure retained, noGPUallocation.
tests1 actualce19c751a2c34c32b05e3d5e268fe73a/session1037 CLOSED1/events0;
scope automaticremoval/inactive/emptyID/emptyCGverified. No livejobs currently.
NEXT warmtests2→input/config/protectedchecks→onegatednativebatch8probe→table
andbackup. No full1536measurement yet; commonwarm/numeric/G1G2 remainOPEN.
No repeatD166/D167/D168 orFullwithoutnewhypothesis. Baselines/3BPAUSED.

## CURRENT — D168 common warm-reference input index COMPLETE and BACKED

Backup COMPLETE547142f901853cb94233a0a6755eacca18a74afd, pushed to
faaslora_origin/retry14_continuous_queue_v2; independent remoteHEADmatches.
Push44629/verify39335 CLOSED0.10scopedfiles/30staged+archiveentries passed
secrets/diff/checksum/smoke/protected checks; usermanifest excluded. No livejobs.
Only this post-backup receipt newlydirty. GoalACTIVE/thisturnPROGRESS, NOTmodel
acceptance. NEXT actual warm runtime/batch feasibility implementation, using
the sealed D168 index; doNOTrepeat79tests/indexanalysis/contentchecks orcache.

Seal COMPLETE, doc/index/CSV now immutable. Verification actual
bf8a40de1d814350864e14a307c12acf exited0/events0; automatic-removal/inactive/
emptyID/emptyControlGroup verified. Pre-existing preflight AST and serving
source byte-identical;147protected/Plan/V1/sourceSHAsPASS.20member131799Bbundle
SHAde0116fbe9c98b8e4e2b891a35d2db5e807f00158c99339b065b3ab90648b458,
checksumPASS. No livejobs. NEXT scopedstagedchecks/backup; doNOTretest/reindex.

2026-10-02 18:49 +08. Goal ACTIVE/PROGRESS, Prime7B only. Full Plan/status/V1
and run/analyze-results/vLLM/academic-plotting/github-sync read after compaction.
D167 sealed/backed881d048, not repeated. User cache approval already fulfilled
D78/D80; no recreation or remote operations. No GPU run or serving optimization.

New pure preflight index reuses D164 retained34MB projection ONLY for executed
input lengths/identity, never its latency or success ranking. All4000 source42
rows bound; Type1 quartiles616/760/760 merge to two groups1039/2961. Each selects
256 distinct original requests, targets>=2, for three future rounds (1536 calls,
NOT already run). Actualadapter counts88/94; batch8 needs up to8IDs, total declared
contexts5311/7865tokens. Actualbatch feasibility/SLOthresholds remainUNQUALIFIED.
Doc/table D168_WARM_REFERENCE_INPUT_INDEX.md; indexSHA
563abfbdeb74c1bc03e42a26c34d48d634c342ac4fa43bd00c5ff8dd82269793.

Tests79PASS1.153s(7new+72preflight), actual8a0da61436e4425f85aa6e8a211ced41,
session75567CLOSED0; analysis actual5272fa78d0794325b53062f3edb867d7,
session12442CLOSED0. Both3/4GiBswap0CPU2,3,26,27/events0, automatic-removal/
inactive/emptyInvocationID/emptyControlGroup verified.147protectedPASS.
No liveGPU/remote/analysis/tooljob. Disk162781646848B, hostavailable112212520960B;
recheckunchanged150/100GiB and102GiBidle gates before newheavywork.

NEXT verify/secrets/scopedbackup, then actual warm-reference implementation and
batch8 physicalfeasibility via existing preflight/runtime; no newframework, no
oldprofile relabel or repeatedindexanalysis. Request→adapter executionmapping,
numericfullpool/n_correct/commonwarm/Resident/G1G2/oldPrimeacceptance stillOPEN.
7Bactualacceptance before3B/externalbaselines unchanged. HistoricalNEXTbelow
superseded; do not repeat completed content checks or Full without newhypothesis.

## CURRENT — D167 checkpoint-derived content check COMPLETE and BACKED

Backup COMPLETE 881d0485716a091fabd300e830afa280951381aa, pushed to
faaslora_origin/retry14_continuous_queue_v2; independent remote HEAD matches.
Push31593/verify15958 CLOSED0. Ten scoped files / 31 staged+archive payloads
passed secrets/diff/checksum checks; user manifest excluded. No live GPU/remote/
analysis/tool jobs. Only this post-backup receipt is newly dirty. Disk available
162,800,771,072 B; recheck unchanged 150/100GiB gates before new heavy work.
Goal ACTIVE / this turn PROGRESS; D167 complete is NOT 7B G1/G2 acceptance.
NEXT remaining actual execution-map evidence and common warm-reference via
existing preflight/runtime interfaces, then new-metric old/new Prime comparison.
Do not recreate published cache, rerun 22-prefix, repeat D167 analysis, or launch
Full without a new falsifiable serving hypothesis. 7B before 3B/baselines.

Offline reuse COMPLETE, no GPU replay or remote operations. All 44 D166 snapshots
and 19,888 tensors match independently derived checkpoint/config bytes; 0 failed
tensors. Wrong medical/zero-rank8/omitted-scaling controls rejected with 256/256/
128 tensor mismatches. 14 artifact tests PASS (7 new); existing 72 preflight
tests PASS. Serving source and pre-existing preflight AST unchanged; only two
offline helper additions. Correctness table in D167_CHECKPOINT_SLOT_CONTENT.md.
The observed snapshots are qualified; per-token execution mapping, full-pool,
n_correct, common warm-SLO, Resident/G1G2 and old/new Prime acceptance remain OPEN.
This is evidence progress, NOT a performance experiment or model acceptance.

Tests actual 8ec9011e2f9d453080a6799e86ae3fa2 / session61057 CLOSED0;
analysis actual 8cf8638ef21648a0a31975b0637f228c / session87673 CLOSED0,
18.84s/RSS1,180,204KiB. Both 3/4GiB swap0 CPU2,3,26,27/events0, automatic
removal verified. D166 raw parsed once for NEW digest question, not repeated
old curation. All inputs/protected147 and Plan/V1/source SHAs checked. No new
weights/trace/cache, no deleted evidence. Verification97714 CLOSED0 and bundle
checksum PASS; exact verification receipt below. A convenience read used the
figure-data cwd and returned missing paths; corrected to repo root, no data
change or rerun. NEXT secrets/diff/backup, then remaining 7B correctness and
common measurement using existing scaffolding. Do NOT repeat D166/D167 checks.

18:33 +08 verification actual 92d3d426a1be4da48cccbe7c3563cd9b, events0,
automatic-removal/inactive/emptyID verified. Curated SHA
b5c660a626c2fd75736e5140fbf5510d13d56af622ab7eb405ec2f3cf6a1996d;
21-member 122,569-byte bundle SHA
74b3b12abcc516a74a526d87715596244a71a62fc5acda899a4409b706a5a3d9.
Document/results now sealed; no further edits/reanalysis. No live GPU/remote/
analysis/tool jobs. Goal ACTIVE, current turn PROGRESS; not model acceptance.

### D167 preparation history (superseded by completed record above)

Full authoritative Plan/status read after compaction. D166 remains sealed/backed;
no repeat GPU prefix, ordinary Full, remote cache preparation or pool generation.
Latest once-only cache approval is already fulfilled by D78/D80 and remains in
the Plan. Current task reuses D166 actual slot digests to independently derive
expected bytes from existing safetensors/config (FP16 cast then B scaling, QKV
packing, padding/absent modules). Offline preflight helper and negative controls
added, NOT yet tested/run. No serving mechanism/formula changes. No performance,
per-token execution-map, full-pool, common SLO or G1/G2 claim. Only Prime7B;
3B/external baselines remain paused. After bounded test/analysis: table, protected
checks and backup, then remaining correctness/common-reference mainline.

## CURRENT — D166 actual 7B slot-content COMPLETE, tabled; seal/backup pending

18:13 +08 BACKUP COMPLETEe35ef2c6a4021ff118fc9b5651f2670b0fac4d46,
pushedfaaslora_origin/retry14_continuous_queue_v2; independentremoteHEADmatch.
Push5916/verify14533CLOSED0.11explicitfiles/90staged+archiveentries secrets/
diff/checksum/17targeted+815regression/147protectedPASS; userdirtymanifestexcluded.
No liveGPU/remote/analysis/tooljobs. Onlythispostbackupreceipt localdirty.
ThisgoalturnPROGRESS(actualGPUcontentqualification+diagnosticlayoutfix+table+backup),
NOT7B or goalacceptance. Completedcachemaintenance and prefixchecks NOTtorepeat.
NEXT remainingcheckpoint→native registration and per-tokenexecution-map evidence,
then existingwarm-reference scaffolding for frozenV1 commonmeasurement; don't
reopenD63looseprobabilitycontrols or rerunFullwithoutnewhypothesis. Actual7B
G1G2/NewmetricoldPrimecomparison before3B/externalbaselines unchanged.

18:12 +08 seal COMPLETE, document/result SEALED; doNOTedit/retest.
CuratedSHA54b318efcfd8df839d5741de20d487ec7b0e95155294bb8cd35b24064440604a;
79member123612B bundleSHA81eeac4e75d0eddd2be2dd14ff28804686d969317fd69ac13ae731f5fc370514.
147protected/Plan/V1/allcuratedsourceSHAsPASS; onlytwooptindiagnosticfunctions
changed, othermonitorfunctions and ordinaryrunner/preflight/planner/residency
byte/ASTsame. Seal7099CLOSED0, actual545444b0782f4374840cc0e4f535636a/events0;
automaticremovalverified. Initialcleanupconveniencecall fromfiguredata cwd used
relative rawpath and failedreadonly; correctedfromreporoot, no rerun/evidencechange.
No liveGPU/remote/analysisjobs. NEXT stagedsecrets/diff/bundlecheck andbackup,
then remainingcheckpoint/execution-map/commonwarm work; notanother22prefix.

18:10 +08 attempt3 COMPLETE22/22;44snapshots x452tensorcomparisons=19888,
mismatched_elements0,absent8624,actualslots0/1/2/3,12IDs/4weightSHAs/ranks8,16.
Registry→actualGPUslot qualified for these snapshots ONLY; checkpoint loader/
per-tokenmapping/fullpool/n_correct/commonwarm/G1G2 stillOPEN. No performanceclaim.
Onelease387.762329310s released;399resourcesamples/peak5702578176B/
minhost106507923456B/events-swap0. Actualservice404399e467314cc6bb45913a2ec127e3,
service/watchdog0; exactaux/service/GPUabsenceverified18:08:14. No remotejobs.
LaunchSHAac4c035919614b30b9b6efbf6b1ca36727d64c9163479a0839050365af689b68.
Curate52192CLOSED0/12.41s/RSS1265692KiB; actuala584ba31a3344fd7b82e89e9d4c825c3,
events0/automaticremovalverified. Allidentity/nativecount/content/protected147PASS.
Raw96,810,390B retained; no needreparse. CuratedCSV44rows+JSON includesbothfailures.
Doc D166_NATIVE_SLOT_CONTENT_RUN.md updated with exacttable/boundaries.
NEXT seal/secrets/backup then remaining actual7B correctness/commonreference
work; doNOTrepeat22prefix/CPUtests orstart3B/baselines beforeactualG1G2acceptance.

18:06 +08 SAMEattempt3 remainsLIVE, exactservice404399e467314cc6bb45913a2ec127e3
unitprimelora-tc-svc-c975c7834dee4d28b9264cee67c25f35.scope; auxactual
212c979624ea4498ad2a4c8659e8524a. Watchdog290882/start44250200 inauxCPUs;
native292534 verifiedexactservicecgroup/CPU4–23,28–47. Banner15/22 passed
before+aftercontent/nativecount checks; no prematureclaimbefore finalcuration.
Servicepeak5702578176B,host108473991168B/events+swap0/no warning.
No remote/serviceconfig/cleanup duringrun. Preparedcuration/seal/checkstaged,
NOTexecuted; samecurationhandlesfailedattempt1/2 plus eventual3, allretained.
NEXT monitorSAMErun→actualphysicalrelease→cleanup→44snapshotvalidation/table→backup.

18:01 +08 regression815PASS74.700s, actualb4a53016e7cf48db8935573fd9341f61,
events0/automaticremovalverified; sourceSHAsunchangedaftertests. Prefix/config
unchanged; attempt3 launchedfresh tmux tc-d166-7b-slot3 using corrected checker.
Auxunitprimelora-tc-aux-bd7ca50155b842e4b8ab5be2a4b888d2.scope, actualIDsawaitcapture.
Curate.py preparedNOTexecuted; preservesbothfailures and remaining correctness/
SLO boundaries. No remote launch, no ordinaryFull replay. NEXTmonitorSAMErun.

17:56 +08 attempt2 COMPLETE/FAILED at first slot_content_before, beforegeneration.
Native actual inventory452tensors includes raw lm_head + embedding3D-A/4D-B;
D165 checker only handled list/tuple. This is unsupportedcheckerlayout, NOT
weightmismatch. OneGPU48.535512155s released, native249914/proxy248914 exited;
service2/watchdog0/physical+scopeabsence verified17:52:37. RawSHA
0cd93e13150957ec4ad0dc5ae456f01f3aa41f3b3cc4b7be91ca7c704fa7ce38, failuresretained/tabled.
Explicit absent-unpacked content branch added to diagnostic ONLY; mandatory
officialresetter/version/layout HOST contract unchanged, populatedraw rejected.
Read officialembeddingonline + installedfixedlogits (webfetchfailedhonestly).
No native setter/servingpolicy/IEEEformula changes. Fournewnegative/layouttests.
Targeted17PASS.044s; actual54ea86d00fbf436fbebc929367779e02/84599CLOSED0,
events0/cleanupverified. Existingregression LIVE tmux tc-d166-regression1.
NEXT regressionterminal→table→fresh attempt3 withsame22/config after tests;
no blindretry/guardweakening or baselineadvance. GoalACTIVE/PROGRESS.

17:52 +08 attempt2 LIVE since17:50:54 in tmux tc-d166-7b-slot2. Same original
22prefix, fresh explicit HOST config verified exactly against D156spec/D164Full.
Prepare2actual89afe00e0e764ddda0376b8e114da59d/39311CLOSED0/absenceverified;
inputSHA56d474d9d69e4724162ccb0a5f938b32d9415a8e7a61576ccd5bf5fdd8f66a16.
Serviceprimelora-tc-svc-397e164e6cd64684abd386483b711dbb.scope,
actual5e512e3734c94f2c94e661abba8e3e36,72/80GiBswap2readback;
auxprimelora-tc-aux-e0283de8c8ee47c89c5028b79d7bd852.scope,
actual275a808c8bf54f609c8dc71529530d68,3/4GiBswap0. Watchdog247898/start44190216.
Latestservice3.114GB/peak3.718GB/host109.214GB/events0,no warning. No remote
service/newweights/newtrace. Nativeworker/startup pending, not qualificationyet.
NEXT monitorSAMEattempt→exactrelease→contentanalysis/status table/backup.

17:50 +08 attempt1 failed BEFORE native runtime/GPU allocation: frozen model
parent omitted HOST allocator policy, whereas inherited environment declared it.
D156 source spec normally adds four HOST contract fields; isolated mode doesnot.
No guard weakened/no serving edit. Fresh config explicitly reuses exact four
D156/D164Full HOST fields; inputs2 verifier prepared/started39311, notyetterminal.
Attempt1 rawSHA b3fa1a0721d888690ee20f7147a9e1d3dab3351f3c05b0c6701c9d9499370ab0,
launch3efe6ba68f64397cf922c83da5382ccdaa3d88bfc0ac8ad67d559c35db06f867.
Serviceactual67f7a71e34004794aa3f93d9d1427061 returned2,watchdog0;
physicalabsence/aux+serviceautomaticremoval verified17:49:55. Failuretabled.
Next attempt2 only after freshconfig verification; originalprefix22 unchanged.

2026-10-02 17:46 +08. FullPlan/status/V1 read after compaction; run/monitor/
academic-plotting/github-sync used. User once-only remote cache approval already
fulfilled D78/D80; reuse, no regeneration or remote activation for local diagnostic.
D165 remains SEALED/BACKED17468f24; no repeated CPU tests/Full or serving edits.
Reused D152 strict public-download auditor: reviewed new allowlistSHA
1a8e073911745e4c99528c080eaef1ad34e93ee4220b3dcb553d256ef8f64557,
5publicarchives+5headers removed,2192306176allocatedB freed. Installedenvs,
compiler cache,models/LoRA/traces/rawresults untouched; allguardsPASS.
Audit28224/apply7625CLOSED0, actualafa6b35235c64ce09994457bf49a8726 /
5df38101aa634183a8666a9c142f70de, automaticremovalverified. Disk162941956096B.
Inputprepare81133CLOSED0, actual4dd19987a78a4a45b278f5966b55b377;
22originalprefix/12IDs/4weightSHAs/ranks8,16/147protectedPASS.
Shortestprefixcontainingfirstmedical + finance, no output-based selection.
InputSHA5cc582ee4ff99715a4da6a5544a1cae5cc5deb357dd32a517aa1a56f420d5007.
Launchscript prepared via existing D156 gated native runner, unchangedcap4/
nativeenv and D165 opt-in native_slot_content. NOT YET LAUNCHED.
NEXT one actualGPUdiagnostic→actualrelease→contenttable/backup, thenremaining
checkpoint/execution-map/commonwarm evidence. G1G2/7BacceptanceOPEN;3B/baselinesPAUSED.

## CURRENT — D165 isolated native slot-content qualification interface

17:32 +08 BACKUP COMPLETE17468f24b517eb255eb1df67b9999dcda3608834;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push16894/verify88374CLOSED0.13explicitfiles/47staged+archivepayloads secrets/
diff/checksum/syntax/147protectedPASS; userdirtymanifestexcluded. No livejobs.
ThisgoalturnPROGRESS (implemented/tested/tabled/backed diagnostic), NOT7Bacceptance.
Onlythispostbackupreceipt localdirty. Disk160766468096B below150GiBnew-heavy;
recheck/audit only reconstructible cache before necessary native run, nofloorrelaxation.
NEXT actual7B slot-content diagnostic using new opt-in existing preflight, no
repeat CPUtests/D164/cachecreation; then remaining checkpoint/execution-map and
commonwarm measurement. 3B/externalbaselines stillPAUSED until actualmodelgoals.

17:31 +08 evidence seal COMPLETE. 147protected/Plan/V1/D164/sourceSHAs PASS;
584pre-existing functions outside explicit diagnostic wiring AST-identical,
planner/residency/pool source unchanged. Actualseal7557fbff25a045da8f91aebc0c8c1d09,
session50238CLOSED0/finalevents-swap0/automaticremovalverified. CuratedSHA
c25031d5725c7ee4d15a446c55934b5f50f8d56c10cb1f8683a6aaf0ec60b8ed.
34member389114B bundleSHAae6d995e4a4d0026c65e32bd3b7e7c6b95e28a2025a29b27cb741f49506a8bd7.
D165doc/result nowSEALED, doNOTedit/retest. Onlycheckpoint/backup pending.
No actualGPUqualification/newperformance or commonwarm result claimed.

17:26 +08 checker CPU qualification COMPLETE, no live GPU/remote/test jobs.
targeted13PASS.173s(command10.92s/RSS1164888KiB), actualb62c9e31b4164cd7a1a99c9602ec33eb,
session97069CLOSED0; regression811PASS74.901s(command87.26s/RSS1199536KiB),
actual2946c2b01387419ba4c1d1e826293f3f/tmuxended; preflight72PASS1.045s
(command1.22s/RSS39552KiB), actual5d9a53f19cea42b1ba4d7d6c7a6e36e5/session70565CLOSED0.
All3/4GiBswap0CPU2,3,26,27/events0; exact automatic-removal receipts verified.
Counts overlap (13new included in broader suites), not896 independent checks.
No failed test attempt. Prepared nonexistent guessed source-profiling test name
replaced BEFORE execution by existing source-profile modules; no serving workaround.
Doc status table updated; actualGPU/native numerical/commonwarm/G1G2 stillOPEN.
NEXT evidence checks/backup, then necessary actual7B slot-content diagnostic and
remaining checkpoint/execution-map/commonreference evidence. DoNOTrepeatD165CPUtests
without changed source; no Full rerun for this opt-in non-performance checker.

2026-10-02. Goal ACTIVE, only Prime7B; no new GPU/remote run. Full Plan/status/V1
read after compaction; run-experiment/vLLM/academic-plotting/github-sync followed.
Latest once-only cache approval already fulfilled D78/D80, no regeneration.
D164 remains sealed/backed70029ba; do NOT repeat its analysis or membership tests.
New opt-in worker content audit and existing preflight native_slot_content mode
implemented; CPU tests NOT yet run. Checks registered native CPU tensors against
actual GPU slots including padding/absent slices; default Full unchanged.
Not a performance optimizer, checkpoint/numerical execution mapping/full-pool
qualification remain explicitly false. D63 loose-probability diagnostic NOT repeated.
Doc D165_NATIVE_SLOT_CONTENT_QUALIFIER.md records scope/primary sources/status.
No warm-reference implementation yet; no new thresholds or serving configuration.
Disk160723996672B below150GiBnew-heavyfloor; only3/4GiBswap0 CPU tests next.
NEXT qualify checker→status table→checkpoint backup→necessary actual7B measurement;
7B actual G1/G2 acceptance before3B/externalbaselines unchanged.

## CURRENT — D164 ordinary 7B Full COMPLETE, tabled and SEALED

17:02 +08 BACKUP COMPLETE70029ba3ff68c4914e1d4a08c40b18298758aa56;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push48238/verify29514CLOSED0.13explicitfiles/73staged+archivepayloads secretsPASS;
syntax/11comparisonrows/refs/bundlePASS. Initial post-seal conveniencecheck used
unavailable hashlib.file_digest in systemPython and stopped before hashes;
corrected to bounded SHA256 chunks and all checks passed, no result changes.
Commit/push proceeded after original bounded evidence verification; convenience
recheck was completed afterward. CSV final blank line/CRLF retained to preserve
sealed hashes; whitespacecheck excludes only those layout conditions.
Userdirtymanifestexcluded. No serving edits or new run. Only backupreceipt dirty.
Read-only numerical-control history confirms prior D63 matching also accepts
wrong adapter/base; do not repeat text/loose-tolerance tests or invent qualification.
Existing warm-reference reuse audit read; no new reference implementation yet.
NEXT necessary current-source numerical-path/common-warm measurement work;
old fields stay unknown. Actual7B acceptance before3B/baselines unchanged.

2026-10-02 16:59 +08. No live GPU/remote/analysis jobs. Same run completed,
not repeated; serving1acd8f4 unchanged. FullPlan1537/status1340/V1312 and
analysis/academic-plotting/github-sync read after compaction. Latest cache
approval already fulfilledD78/D80: reuse, no regeneration/per-request packing.

Native4000/0fail; mean/P95/P99TTFT4.666023437/11.316413800/16.880609884s,
vsD160 mean-0.268%,P95+1.913%,P99+0.749%; TPOTmean44.134842ms/P9581.865101ms
(+0.179%/+4.626%),U15929.859233GPU-s(-0.229%). n1noCI, NOT established gain.
All4leasesreleased; n_correct/commonwarm/Resident/G1G2/oldPrimecomparisonOPEN.
15nonoutputfields4000matchD160;115outputhashdifferences require diagnosis.
Timingerrors0ms/dispatchtiers4000/conflicts0. Remote132UUIDpairs/131480060wireB/
pack0;4048resourcesamples/peak20787318784B/minhost92473409536B/events-swap0.
Stageandmultimetric tables in D164_FULL_W0_FULL1.md; comparisonCSV retains
all regressions. Keep qualified membership simplification as representation
cleanup, not serving contribution; do not repeat same hypothesis.

Analysis allCOMPLETE: metadata2 actuala5ba057fa5134d96a4d8af54b5ad344a/24321exit0,
directlosslesscompact130090703B under unchanged128MiB. NOmetadata1failure.
Projection334e3a43738e4ac1a4b8c35bef995f24/91709exit0/80.27s/RSS99072KiB;
275085675Boriginal streamedONCE, SHA355f5676102038367e991f0ab4101635dad54f96e55aa353d0dee1046cf52567.
Curationd82c0e9d2afa4c259e49859e75bb728c/79185exit0/7.79s/RSS471788KiB.
Occupancy41b14bc2a7624ead84b036495673c77f/83485exit0/3.26s/RSS464408KiB,
finalevents0; allabsenceverified. No failedGPU/analysisattempt. Monitorpattern
exit1 alreadydistinguished from actualrunreturn0 in terminalhistory below.

Seal3b337ac40564426fb56a22e78ee7f0df/2939CLOSED0; absence16:58:49.
322frozen/46rehashedcurated/147protectedPASS; largeoriginalhash+statreused,
verificationSHA178c46b79eeff7b28ba13c0e2b4732eff7a02f1431209b6c610074fa1d9f12d4.
CuratedSHA7076db88971a1a1de7ece273ba82aa8882e7ad92b2cdf32e35d121d105b98bab.
Bundlebf859733dd4745738239fc5f1c45afd4 returned0/absenceverified16:59,
60members39904B/SHAf06c2450d9ac44146d71cb803c96b9191796bcc370fa9032364f4252fcd1e380.
Doc/curated SEALED, doNOTedit/reanalyze. Backup pending.
Disk160741892096B below150GiBnewheavyfloor; no new heavy launch or relaxation.
NEXT backup then close the actual numerical/common-reference measurement
gaps using existing interfaces, not another repetition of the same microchange.
No new candidate/run yet. 7B acceptance before3B/externalbaselines unchanged.
ThisgoalturnPROGRESS (analysis/table/seal), NOTmodel or whole-goal completion.

## D164 terminal analysis history — superseded by completed record above

2026-10-02 16:45 +08. SAME D164 completed, NOT relaunched. All4000 banner/native
terminal successes,0failure; actual service/replay/watchdog return0, all4physical
leases released and service path removed. LaunchSHA
3edc1c176d6db880861f266529e7085001281d4b2de35d3788cea8b4039bd3c1.
Exactlocalabsence16:42:48, exactremote3services stopped16:42:49(success/MainPID0),
stop56851CLOSED0. Remote journal2e2b62949b034b22a73e6d924042e107 matches healthclock;
copy37538CLOSED0, bothSHAsmatched. No live GPU/remote service. Monitor8270 ended
poll57 when final logtail had no Live line (exit1); rechecked actualscopes/logs/
terminal, no false restart. This was an observation-pattern exit, NOTrunfailure.

Full Plan/status/V1 and monitor/analyze-results/academic-plotting/github-sync
read this turn (truncatedPlan gap repaired). Preliminary U15929.859233295952,
132publishedUUIDpairs/131480060wireB/pack0,4048samples/peak20787318784B/
minhost92473409536B,events-swap-warnings0. n_correct/commonSLO/G1G2stillOPEN.
Metadata2 actuala5ba057fa5134d96a4d8af54b5ad344a/session24321CLOSED0, direct
losslesscompact130090703B under unchanged128MiBguard; absenceverified16:44:30.
ONEprojection running session91709, actual334e3a43738e4ac1a4b8c35bef995f24,
unitprimelora-d164-full1-project-20261002.scope. Reuse after success; never
reproject original275085675B. Curation/occupancy/table/seal/backup pending.
Serving1acd8f4 unchanged. NEXT finishsameprojection→exactcleanup→curation/table.
ThisgoalturnPROGRESS(completedrun/cleanup/metadata), not7Bacceptance. Earlier
LIVE entries below are historical. Baselines/3B remainPAUSED.

## D164 observation history — superseded by terminal record above

15:53 +08 VERIFIED WAIT: same experiment still LIVE. Eight light polls50s
apart15:47:05–15:52:57 each verified exact service InvocationID44e8b9abc6b74796a4e19095f1e78d97.
Monitor cell8262 CLOSED normally after8polls; NOT experiment termination.
Latest banner1013arrived/966reported-success/0fail of4000,532service tasks.
Sample1135 service18176622592B/host95120326656B,swap/events0/no warnings;
disk159967277056B below150GiBnew-heavy but above100GiBrunningfloor.
No new experiment, remote operations, heavy analysis, hashing, cleanup or
serving/configuration changes. No outstanding toolsession; actual tmux/service/
remote experiment remains live. Prepared8postrun helpers remain UNEXECUTED.
NEXT monitor SAMErun until terminal/release, then exact-owned closure and
prepared bounded analysis. This turn PROGRESS(preparation)+VERIFIED WAIT,
not model/goal completion; actual7B acceptance before3B/baselines unchanged.

15:46 +08 continuation PROGRESS/verified same run, no relaunch. Full Plan1537,
status1279 and V1312 read after compaction; run/monitor skills followed.
Exact service/aux InvocationIDs revalidated active15:43:46 (532/5 tasks).
All four native cores3838044/3843356/3843473/3843637 captured in service cgroup,
CPU4–23,28–47, watchdog sample591. Sample717 service17605308416B,
host95368097792B, swap/events0, no warning/abort; disk160299859968B is below
new-heavy150GiB but above running100GiB. No cleanup/threshold relaxation.
Banner519/4000 reported successes,528arrived,0fail; incomplete, not acceptance.
Latest user once-only delivery-cache approval already fulfilled D78/D80; reused
published artifacts, no regeneration/remote operations during inference.

Prepared NOT executed: exact local absence helper using actual seven owned PIDs;
seven metadata/projection/curation/failure/occupancy helpers reused from D160.
Shell5/AST3 syntax PASS. Metadata directly removes only indentation/newlines
from original into compact2, preserving128MiB guard; no invented pass1/source.
Curator compares sealed D160 projection/curated SHAs, NOT D162 prefix or old
giant originals; retains current runtime1acd8f4 and sole D163 candidate identity.
Removed old nonexistent/failed-pass1 references, not measurement checks.
cleanup_metadata_full1.sh still needs actual future analysis InvocationIDs;
do not copy old IDs. No serving edits, remote changes, GPU launch or analysis.
NEXT monitor SAME live run, then terminal/release/exact cleanup/copy before
bounded analysis/table/seal/backup. Actual7B G1G2 remains OPEN before3B/baselines.

2026-10-02 15:36 +08. PreviousgoalturnPROGRESS(D163qualified/backed1acd8f4).
ThisturnPROGRESS: ONEordinaryFull launched15:33:46 in tmux tc-d164-7b-full1.
Serving/evidenceHEAD1acd8f43676e5e1fc25afada5736c0607032ebb0 alreadyBACKED.
Full1537Plan/1236status/312V1 and run-experiment/monitor-experiment read.
4000/source42/W0/fixednativecontract, sameD160cap4/D157profiles except three
freshownedpaths. D163 immutable-membership issolecandidate; no observer/prefix,
capacity/profile/SLO changes orremote cache recreation. Formal0; G1G2stillOPEN.
Raw results/ieee_tc/p2_backend_qualification/d164_20261002.

- serviceprimelora-tc-svc-f75b863601e84d63be61ef2cc1a575cc.scope,
  actual44e8b9abc6b74796a4e19095f1e78d97,72/80GiBswap2verified.
- auxprimelora-tc-aux-d985f0241abd4a50b5613132abf467fb.scope,
  actual31de7ad1b7594d3f80574796b816f03f,3/4GiBswap0verified.
  AuxPIDs3834703/3834731/3834809, watchdog3834809/start43367491;
  3834731/watchdog CPU2,3,26,27 actualcgroupverified. Firstnative3838044 actual
  servicecgroup/CPU4–23,28–47 verified. OthernativeIDs pendingcapture.
  Sample95:service7149887488B/host104975814656B/swap-events0/no warnings.
  Initialdisk161465225216B passes150GiBnewheavyfloor; running100GiBunchanged.
- Externalnotice433675.168805268/t0433735.168805268 exactly60s;
  W0viewa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
- Remote3B3872178/c13237495e774bfda9f69aef180fa7f5;
  7B3872180/876f672e3dae4383b7a705e8f427153d;
  monitor3872183/3f09563c3d994bfdb56d7c5a725a4e8e,
  unitprimelora-artifact-monitor-d164full1.service.
  7Bclockremote-process-monotonic:95be067eaf8c4f2795a9bbac708e021f.
  Monitor/home/lab14/primelora_remote/tc/d164_20261002/remote_monitor_7b_full_full1.log.
  BothNIC1000/full;bothhealthPASS; publishedcache unchanged.
- prelaunch322refs/147protected/Plan/V1PASS;
  SHAe56efbb96ac900c54486e2fc04b8c30e79ba239e877f7fd9b1593095942eb785.
  actual77957351fd2e42fc9fded3952a1551e0/session52329CLOSED0;
  healthactualca8f8f97474e4600be5d926e7b7189ce/returned0;
  bothscopesautomaticallyremoved/inactive/emptyIDverified.

Prepared NOTexecuted: exact-owned remote stop/copy3helpers boundtoactualIDs/
clock/service; bashsyntaxPASS. Localabsencehelper awaitsallactualnativePIDs.
One absent guessedcleanupfilename returnedreadonlysederror, no statechange.
NEXT monitorSAMErun→actualterminal+physicalrelease→exactremote stop/SHAcopy→
boundedmetadata/ONEprojection→D160comparison/table→backup. No remoteoperations,
heavyanalysis,bulkhash,cleanup orsecondoptimizer duringinference. 7Bactual
acceptance before3B/externalbaselines; numerical/commonreferences remainOPEN.

## CURRENT — D163 single immutable-plan membership candidate qualification

15:29 +08 BACKUP COMPLETE1acd8f43676e5e1fc25afada5736c0607032ebb0;
pushedfaaslora_origin/retry14_continuous_queue_v2, independentremoteHEADmatches.
Push81958/verify18970CLOSED0.10explicitfiles/43staged+archivepayloads secrets/
checksum/scopeddiffPASS;822regressionsPASS, userdirtymanifestexcluded.
No liveGPU/remote/test/tooljobs. Onlythispostbackupreceiptlocaldirty.
ThisgoalturnPROGRESS (onecandidateimplemented/qualified/tabled/backed), NOT
modelacceptance. NEXT D164ordinaryFull4000W0 sameD160cap4/D157profiles, fresh
threeownedpaths/currentHEAD1acd8f4; noobserver/prefix/newprofiles/cachecreation.
Fullnotyetprepared/launched. ReuseD160fullrunner/cleanup/analysis withnewpaths
andcomparesealedD160projection/curated, doNOTreparseitsoriginalorretargetD162
diagnostic asmainresult. ReadfullPlan/status/V1beforelaunch, recheck150GiBdisk.
7BactualG1G2/numeric/commonreferences/oldPrimeacceptance OPEN before3B/baselines.

15:28 +08 sealCOMPLETE. 147protected/Plan/V1/D160/D162 sourceSHAs PASS;
original worker/formula/exportAST and runner/planner/residency filesunchanged.
CuratedSHA0ae2f98f02371cc924d80f6ea4c42756a48ed0efbdfe427d1360b3e9b1730c41;
verification1aa4856d1ca1f96e52378e87925897fee8d6968230c853e9081b2427fa8cfb0b.
33member57037B bundleSHAaea1c9824a78dbd854b46541195ccb7ef709f6a65f05aaabb548da048ef919fc.
verify1 actualbf835056738d433b90aa76959f19b273/session17482CLOSED0,finalevents0,
automaticremoval verified15:27:56. Doc/result SEALED, doNOTeditorretest.
No livejobs. Candidate/source SHA94e16c8af9957d1faa5f8b1bee8a8cba9de80d6bb09ff27251d9163d3695763e.
Disk161471758336B(~150.38GiB): recheck150GiBnewheavyfloor; no relaxation.
NEXT exactstagedchecks/backup thenD164ordinaryFull sameD160cap4/D157profiles,
freshthreeownedpaths andcandidateidentity; no observer/prefix. Notpreparedyet.

15:27 +08 qualificationCOMPLETE, no liveGPU/remote/testjobs. New6PASS.003s,
regression822PASS148.908s(command161.57s/RSS1199924KiB). Actualtargeted
afd163bbde244bd285484cb943739890/session19958CLOSED0; regression
1ab89f8ffefb48b0bb4f688ecf4068d6/tmuxended; micro
b11171097c2c4b339920d34e3e396726/session58075CLOSED0. All exactcleanupverified,
3/4GiBswap0CPU2,3,26,27/finalmemoryevents0. No failedexecutionattempt.
Existingtinyfixture paired3x100:residency235.725→120.170us, handoff207.494→
103.454us; exactdecodes2→1/outputsame. NOTservingbenefit/CI/modelacceptance.
DocD163 tablecomplete; candidateonlykeyindex/membership, otherformulas/livechecks
unchanged. NEXT seal/backup then ordinaryFull4000 sameD160parameters/profiles.
DoNOTrepeatD162diagnosis/D163tests orremote cachecreation. G1G2 stillOPEN.

2026-10-02. Goal ACTIVE; Prime7B only. D162 sealed/backed2765a51, not repeated.
Full authoritativePlan/status/V1 and run-experiment/analyze-results/academic-
plotting/github-sync read. CPython/asyncio/vLLM primarysources recheckedonline.
One candidate only: immutable key index avoids payload materialization for
membership; original worker formulas/validation/execution/live checks unchanged.
Six new representation tests and existing actual-worker check prepared, notyet
run. Reuse D159 boundedtest protocol/D153 component method/existingfixtures.
Raw d163_20261002. No GPU/remote launch, no cachecreation, cap/profile/SLO change.
NEXT minimalqualification→table/seal/backup→ordinaryFull4000 sameD160config.
7B G1G2/numerical/commonreference/oldPrimeacceptance stillOPEN;3B/baselinesPAUSED.
Only own two source/test files plus D163 doc/status edited; usermanifestuntouched.

## CURRENT — D162 current 7B prefix diagnostic COMPLETE/SEALED and BACKED

15:08 +08 backup COMPLETE2765a51bd844bd38f4b6bfde79905757c9dacdb2;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push91685/verify68573CLOSED0.8explicitfiles/53staged+archivepayloads secrets/
syntax/checksum/scopeddiffPASS; userdirtymanifestexcluded. No livejobs/tools.
Onlythispostbackupreceipt localdirty; servingruntime11a5432 unchanged.
ThisgoalturnPROGRESS (completed diagnostic/newcause evidence/table/backup),
NOT7Bacceptance. NEXT ONEimmutable-plan membership candidate qualification,
thenordinaryFull; notyetimplemented. No repeatedD162analysis orobservertests.
ReadfullPlan/status/V1 first; actual7Bacceptance before3B/baselines remains.

2026-10-02 15:06 +08. Goal ACTIVE/PROGRESS. SAME run completed1000/1000native,
0fail,4physicalleasesreleased/0quarantine. No live GPU/remote/analysis/tool jobs.
No serving edits:11a5432 runtime/ccde021 launch; cap4/D157profiles unchanged.
Full Plan/status/V1 and run/monitor/analysis/academic-plotting/github-sync read.
Once-onlydeliverycache reused, no recreation. Do NOT repeat completed D162
observer qualification/run/analysis or old D161/D160 closure.

U4499.054580241034GPU-s diagnostic ONLY; initial1/natural3;60publishedUUIDpairs,
74953014wireB/1474429899logicalB/packing0. Resources1202samples/peak19012448256B/
minhost93334138880B,events/swap/warnings0. Nativecount is NOT numericaladapter,
commonSLO, G1/G2 or oldPrimeacceptance. No per-request latencyidentityaudit or CI.
69,442,442BnormalJSON preserved without reparse; compactmetadata48,238,669B
allfields/ASCIIindent-only, unchanged128MiBguard. Protected147/frozen322PASS.

New diagnosis:10roles=controller1/planner1/frontends4/GPUcores4. Controller
538businesssamples:71RPCjsonloads,23fileinventory,20execution_copy via
Mapping.__contains__→__getitem__→snapshot,17execution_bundle_copy. The source
checks source_view membership then opens bundle; redundant whole-plan decode
is a concrete next candidate, NOT established servinggain. GPU HOST inventory
inclusive128/552,116/531,89/529,98/527 remains another cue, not simultaneous
optimizer. Mainthreadcomplete/partial0;7allperiodtruncated,4businessGPUtruncated.
Occurrences NOT CPUpercentages; no causal attribution of all waiting.
Doc/table D162_CURRENT_PREFIX_DIAGNOSTIC.md nowSEALED, doNOTedit.
CuratedSHAea9ae817196f8f8f5ccf2a5869469690ceb5e4aa53e5c53a606e337c21f9c8d5.

Closure: monitor8176 endedafter8polls with terminalreceipt, not interruption.
Localabsence14:56:17; exactremote stop14:56:18(session70429CLOSED0).
Copy93347CLOSED0/journal d3723247174c40b393dc4a3eded26297, exactclock/SHAverified.
CPUcollector df46528c3eba495d8c5ddfc6853c56ce exit0/.52s/RSS21888KiB;
metadata a1cfe3262f004f43b3b65472825016e0/session67819 exit0/1.57s/RSS171896KiB;
curation e403304ef362436bb8156fe4a57bbae4/session89501 exit0/1.14s/RSS26248KiB.
All3/4GiBswap0CPU2,3,26,27/finalevents0; automaticremoval/emptyID/emptyCGverified.
Seal c448b24179fc4b469c657c76a177f5be/session12024CLOSED0,finalevents0/absent.
VerificationSHA40723305e277ee7e30733723be50d0fcf07b97ca3f6838c5df0450e44b6ee504,
44newrefs/322frozen/147protectedPASS.45member58471BbundleSHA
2725446ed386a59354388162ff3081662b61432d59227bf307f23eb5bed24805.
Read-only guessed historical filenames returned sed errors; no files changed.
No failed GPU or analysis attempts. Current disk161498959872B(~150.41GiB),
recheckbeforeheavy; neverlower150/100GiBfloors. Remote remains stopped.

NEXT scopedsecrets/checksum/diff/syntax backup; then ONE immutable-plan membership
candidate: avoid materializing a complete certified snapshot for key presence.
CurrentCPythonMapping source/asyncio primarydocs checked; first preserve key/
missing/unhashable/export/certificate semantics, then component/minimalregression,
then ordinary Full4000. Not yet implemented; don't also changeRPCcoding, tensor
inventory, cap/profile/deadline or statefreshness. Actual7B G1G2/commonreference/
numeric/oldPrimeacceptance stillOPEN, before3B and externalbaselines. Later full
matrix remains outstanding. This goalturn is PROGRESS, not completedgoal.

## D162 observation history — superseded by completed state above

2026-10-02 14:42 +08. Goal ACTIVE/PROGRESS. ONE run launched14:35:07 in
tmux tc-d162-7b-prefix1, still same active run; DO NOT restart. No serving
changes: D159 runtime11a5432, evidenceHEADccde021, cap4/D157profiles and D160
Full configuration except three fresh owned paths and explicit prefix1000.
Existing D122 source42 index + unchanged D143 python_frames_v1 observer;
formal0, diagnostic only, NOT a new full performance result or G1/G2 acceptance.
Full Plan1537/status1074 read after compaction; run/monitor skills followed.
Once-only remote delivery-cache approval already fulfilledD78/D80: reuse, no
new weights/traces/whole-pool copy or per-request packing. No remote operations
after inference launch. Current question: what current control operations are
observed during remaining pre-source/completion waits, including GPU-ready hits?
Old D143 cap2 occurrence percentages cannot answer current-version attribution.

- Raw results/ieee_tc/p2_backend_qualification/d162_20261002.
- Service primelora-tc-svc-6a3cd29c678841fea006eb99dfd594a1.scope,
  actual ada0c5f86b6e42d1a1d79078f3b94d82; aux
  primelora-tc-aux-c20065e2272e4effa0392a8171f0fcd7.scope,
  actual f8ad4587fa8a4ce6aaa0c7e6083f877d. Both active14:41:23,544/5tasks.
  Replay3577391/start43015572 CPU2,3,26,27; controller3577602/start43015664.
  Native cores3580427/3584939/3585110/3585252 observed sample405 in exact service
  cgroup, CPU4–23,28–47. Sample405 memorycurrent17408774144B/peak19012448256B,
  host95487356928B/swap-events0/no warning/abort/foreigncompute.
  Disk160717312000B below150GiB new-heavy gate but above100GiB runningfloor;
  do NOT clean/compress during run or relax either threshold.
- notice430156.606610995/t0430216.606610995 exactly60s;
  prefixview0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981,
  source4000/count1000/W0. Frame metadata recorded for controller/planner/frontends/
  GPU-core processes; role/sample coverage to validate after terminal.
- Remote3B3808104/cd1c30c0a032486db0dc078c898f6527;
  7B3808106/7dd6d787d6854afbac6fd741bbfca6c5;
  monitor3808109/51f1b364a54a47c7a023b7fa21405ff9,
  unit primelora-artifact-monitor-d162prefix1.service.
  Frozen7Bclockremote-process-monotonic:cc74da05e0764097b028e0bf87ad3d35.
  Monitor/home/lab14/primelora_remote/tc/d162_20261002/remote_monitor_7b_full_prefix1.log.
  BothNIC1000/full, bothhealthPASS, published cache unchanged.
- prelaunch322refs/147protected/Plan/V1PASS,
  SHA91e70ed08347531997985008aa6f3be42ba6b5694cb4f1018d0a1ed24752105d.
  Actual61df1b1049214cdc895430722d868a66/session41189CLOSED0;
  healthactualf608d2196e8a4e44b8a168697afb2940 returned0; bothscopes absent.

NEXT monitor SAME handle; prepare exact-owned terminal cleanup/copy and reuse
bounded CPU-frame/metadata analysis, execute ONLY after actual physicalrelease.
No Cwatchdog/ptrace/new observer/cap/profile/deadline change or second optimizer.
Then table/interpretation/seal/backup before selecting one source-grounded change
and ordinary Full. Both models actual numeric/commonreference/G1G2 remainOPEN;
7B acceptance before3B/externalbaselines. No outstanding tool session, actual
tmux/service/remote run LIVE. Prior entries below are completed history.

## PRIOR — D161 retained RPC/source-stage diagnosis SEALED and BACKED

14:29 +08 backupCOMPLETEccde02142b1846965795551d3317979086c6b150;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push18317/verify88539 CLOSED0;12explicitfiles/25payloads secrets/checksum/
scopeddiff/syntaxPASS; usermanifestexcluded. No livejobs/toolsession.
Onlythispostbackupreceipt localdirty; runtime11a5432 unchanged. Thisgoalturn
PROGRESS,newRPC+conditional-sourceanalysis/table, NOT7Bacceptance.
Disk161776226304B(~150.66GiB), recheckbeforeheavy; don'trelax150/100GiBfloors.
NEXT ONEcurrent7Bprefixdiagnostic. Notyetprepared/launched. ReuseD143observer
andD160currentcap4/D157profiles, no repeatedD161closure/tests/cachecreation.
FullPlan/status/V1beforelaunch. BothmodelsactualG1G2stillOPEN;7Bbefore3B/baselines.

14:28 +08 sealCOMPLETE:8source refs/weightedTTFT-TPOTreconciliation/147protected/
Plan/V1/unchangedserving+observer/syntaxPASS. VerificationSHA
a59a909c4a62a8b01578dc0cb7dd23143a980cad08e9e27c9c43fc8c4dd4bd90.
13member10008B smallbundleSHA
5f5aeb22811bd50b1d333a28aafba4be1140be1d30acb08dc4b73f5496955081.
Sealactual2eb88d93d3e6458da8b26a0445746e5a returned0/finalevents0;
automaticremoval/inactive/emptyID/emptyCG verified. No livejobs.
Doc and curated nowSEALED, doNOTedit/reanalyze. BundlechecksumPASS.
Nextstagedchecks/backup thennewcurrent-prefixdiagnostic; notprepared/launched.

2026-10-02 14:27 +08. Goal ACTIVE/PROGRESS; no live GPU/remote/analysis jobs.
Serving unchanged D159 11a5432. FullPlan1537/status/V1312 and analysis/vLLM/
plotting/github-sync skills read. User once-onlycache approval alreadyfulfilled;
no cache creation or remote operation. D160/D158 closures NOT repeated.

New offline recovery reuses D16034MBprojection and sealed timeline only:
RPC22fields x4000 valid; pickupmean/P95478.953/1909.015ms, nativeworkerqueue
2.238/1.703ms, channel.014650/.017668ms; no remote-network/isolatedCPU attribution.
Selectedgroups GPU1399/HOSTnative1533/HOSTfile8/NVMe922/remote138; GPUgroup
meanTTFT3.417842s,gate→source1.792049s/source→dispatch.163136s. HOSTnative
source→dispatch2.005812s. Conditionalgroups NOTcausal tier contrasts.
Doc D161_RETAINED_CONTROL_DIAGNOSIS.md includes exacttables/limits/nextquestion.
No mainlatency/U/numerical/SLO values changed. 7B G1G2/numeric/commonreferences
remainOPEN;3B/externalbaselinesPAUSED. No servingcandidate selected.

RPC scope95b35c5c2da245a089cb9564023684bc exited0/.78s/RSS130944KiB;
source scope8243061947c14fc296adfaa6354decb2/session18453 CLOSED0/12.16s/
RSS1182776KiB. Both3/4GiBswap0CPU2,3,26,27,finalmemoryevents0 observed;
automaticremoval/inactive/emptyIDverified. Microstatistics+4000identity/source/
timelinereconciliationPASS. Analyzerunchanged;no repeated D159regressions.
Readonly absent-path lookups caused sed/rg errors only, no data altered.

NEXT seal/checksum/secrets/backup D161, then ONE current7B1000prefix CPU-frame
diagnostic using unchangedD143observer/currentD160cap4/D157profiles/Fulllauncher.
Rationale: current pre-source and completionwaits persist evenforGPUready, but
retainedlogs lack operationCPU attribution; oldD143cap2stacks notcurrentevidence.
No Cwatchdog/ptrace/newinstrumentationframework/cap/deadlinechange. Diagnostic
NOTyetprepared/launched. After evidence chooseONEoptimizer,thenminimaltests+
ordinaryFull; don'tloopoldcomponenttests or skipactualper-modelacceptance.

## PRIOR — D160 ordinary 7B Full W0 analyzed/tabled/SEALED and BACKED

14:10 +08 backup COMPLETE1267d15c0dad1ffd3aae983777cfc574f96a94c9;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push14812/verify44045 CLOSED0.12explicitfiles/76staged+archivepayloads checked;
AST7/shellsyntax/checksum/scopeddiff/311sources/147protectedPASS.
Userdirtymanifestexcluded. No livejobs, no servingchanges beyond D159.
DoNOTrepeatD160analysis/seal/replay, D159tests, D157profiles orremote cache.
Onlythispost-backupreceipt localdirty. Goal ACTIVE, thisturnPROGRESS.

Read-only next-path inspection: _ieee_request_snapshot still gathers fresh
native sources then selected-copy protection separately; HOST preparation takes
a fresh source view before demand_load_and_acquire. Confirmed identity/epoch
and cancellation ownership are required; no decision to remove those checks.
Native routing D151 already omits unused staging output but keeps live graph
validation. Current source paths are scripts/run_all_experiments.py,
faaslora/memory/residency_manager.py and faaslora/experiment/instance_pool.py;
three guessed runtime file paths and faaslora/engine.py were absent (readonly
rg errors only, no evidence altered). No new optimizer/configuration/GPUrun
selected. NEXT use D160 stage evidence plus source/history/primary-reference
diagnosis, select ONE falsifiable bottleneck; do not recycle D159 hypothesis or
infer CPUpercentages from old D143 samples. Preserve current full sequence:
actual7B G1G2/numeric/commonreference acceptance before3B/externalbaselines.

2026-10-02 14:06 +08. Goal ACTIVE/PROGRESS. D160 completed4000native/0failure;
no live GPU/remote/analysis/tool jobs. D159 sole candidate runtime11a5432 unchanged.
Mean/P95/P99TTFT4.678565588/11.104037948/16.755107238s, versusD158 -27.010%/
-35.356%/-44.232%. TPOTmean/P9544.056091963/78.245302140ms:mean+0.921%,P95-5.251%.
U15966.353155035002GPU-s (+0.069%); all4released. Numericadapter/commonSLO/
G1G2 remainOPEN,n_correctnull; n1noCI, NOTmodelacceptance orresourcegain.
Dispatch2.876250918s=61.477%TTFT; source→dispatch1.382736770s;
native4.979539395/last→controller.567400209/controller→terminal.400696384s.
Gate→terminal8.939861999s UPPERenvelope, notexactpermitrelease. 15nonoutput
fields4000matchD158;127outputhashdifferencesOPEN. Timingerrors0ms,tiers4000/conflicts0.
Remote132published/131480060wireB/packing0, noonce-onlycache recreation.
Resources4055samples,peak20444213248B/minhost92474380288B,events/swap/warnings0.
Planning1858receipts:1856owned(1855completed+1cancelleddiscarded),2initcompleted;
CPU1016.950546913+0.063301133s. Objectiveworker substage75.659msmean,606.791msP95.
Control1763inwindow/1007queued/943belowreadycapacity; notproofGPUidle.

Analysis COMPLETE: metadata1 c7aac6c0e3804d279e064ed520f5a1d1/32011exit1
BEFOREload:indentation-only135945146B exceedsunchanged128MiBguard. Retained.
metadata2 b4b1a4cfdba2440eaa7b52d891df0af3/83258exit0/4.52s/RSS461168KiB:
addlosslessJSONnewline removal,131304262B, allfields/numericlexemesoriginalunchanged.
No memorythresholdchange orGPUrerun. Projection58476exit0/80.83s/RSS99072KiB,
actualef961cfb849c42609350e8d18211d703;274117163Bsource streamedONCE.
ProjectionSHA9ea2d9c097bd16b0b75cdb3b272f2bf36833a4b999d17434d1dd46153cf5d6fb.
Curation4206exit0/8.22s/RSS476012KiB actual3fe8bed426ec497fb7656a9bf10e5e52.
Occupancy60326exit0/3.21s actual567e107576964b3eb5ef750897f6c20d,finalevents0.
Verify79148exit0 actual625b2ac11bfa42068a4b6ccb333b07b1:311frozen/51curated/
147protectedPASS,SHA133aac17696c52e0a71c0e38f427a06c0181cc79230400d9c0ff697fb208f03c.
Bundle64members40989B SHAfc86c9e0129d5ae750935c6465ab8e66cab4e8dcf036549db3400c06df87f781,
actual0269581790af4673afa386a0d1b24a67/exit0. Allscopeabsencesverified.
Doc D160_FULL_W0_FULL1.md andcurated nowSEALED, doNOTedit. CuratedSHA
67db3cf4b7a3f7a26d6cbd8f355b398945439f855fc9e1cd0254104fbdc9d777.
Disk161795526656B (~150.68GiB); recheckbeforeheavy, nofloorrelaxation.
NEXT scopedsyntax/secrets/checksum andbackup; thenread-onlysource/history/primary
reference diagnosis ofremainingnon-generationwait. No newcandidate/runselected.
DoNOTrepeatD159tests/D160projection/calibration/cachepublication. Actual7Bacceptance
before3B/externalbaseline remainsmandatory. Alllatermatrixretained.

## D160 terminal history — superseded by completed analysis above

2026-10-02 13:50 +08. SAME D160 finished, not relaunched. 13:49:27 launch
pass=true; service/replay/watchdog return0, native GPU release and service path
removal confirmed. LaunchSHA98f21b50605ba155862074d86ad26d61236a8c8d4f38631a85a72c93b611f6c2.
Banner4000/4000success,0failure is NOT numeric/SLO/G1G2 acceptance.
Exact local scope/PID/GPU absence verified13:49:42; final cgroup counters after
automatic removal unavailable, not claimed zero. Exact remote services and
monitor stopped13:49:43, MainPID0/Resultsuccess; stop46354 CLOSED0.
Copy41970 CLOSED0; remote journal09547072eefc885b04940592b504378da500edc2d811a80950baa1bf1e8dcd6c
and monitor039af26b3554c5e13d32a5a0718232de49eac2049e887f75e56ad4e4aa63879d
match remote source SHAs. No live inference/remote/tool session. Monitoring
cell8043 ended normally after63polls; no interventions during inference.
Full Plan/status/V1 and analyze-results/academic-plotting/monitor skills read.
NEXT boundedmetadata→ONEstreamingprojection→curation/occupancy→comparative
table/interpretation/seal/backup. No numerical gain claimed before validation.
7B actual acceptance remains OPEN; 3B/externalbaselines PAUSED. Goal ACTIVE,
this turn PROGRESS. Do not recreate remote cache/retest D159/reproject D158.

## D160 observation history — superseded by terminal state above

12:55 +08 VERIFIED WAIT thisgoalturn; previousgoalturnPROGRESS(D160launch).
FullPlan1537lines/status andmonitor-experiment read; SAMEtmux/service/aux
revalidatedlive12:47:53,then8lightpolls50sapart through12:55:03.
ExactserviceInvocationID b02b72c8dcf247dfa7f7d7a85605befc checkedoneachpoll,
533tasks. Latestbanner741/4000success,747arrived,0fail. Sample871:
service16.695663GiB,host88.703667GiB,disk150.199482GiB,swap/events0,
no warning/abort/foreignGPU. These areincompleteobservations,NOTacceptance.
Monitorcell8040 CLOSEDnormally after8polls; NOTexperimenttermination.
No newcode/config/remote changes, heavyanalysis, bulkI/O, cleanup orrestart.
No outstandingtoolsession; actualtmux/service/remoteexperimentstillLIVE.
NEXTmonitorSAMErun untilterminal, then exactownedcleanup+preparedanalysis.
3B/baselinesstayPAUSED until actualper-modelgoals; no Plan/V1change.

12:46 +08 continuation state: SAMErun remains active, exactserviceInvocationID
b02b72c8dcf247dfa7f7d7a85605befc reverified/533tasks. Banner166/4000success,
170arrived,0fail; incomplete, NOTnative/numerical/SLOacceptance. Sample346:
service16.2029GiB,host88.9966GiB,swap/events0,no warnings/foreigncompute,
disk150.5935GiB. Nativecores3068435/3073231/3073612/3073930 observedallinside
exactservicecgroup withCPU4–23,28–47. Replay/watchdogremainseparate.
No remote change/hash/cleanup/build orsecondoptimizer duringinference.
PreparedNOTexecuted: exactremote stop/copy, exactlocalabsence/auxcleanup,
boundedmetadata/ONEprojection/curation/failure/occupancy scriptsreusedfromD158;
shell/4ASTsyntaxPASS. Newcurator compares sealedD158projection/curatedSHAs and
auditsD159execution_objectives receiptinsideworker; no metricweakening.
Postrunanalysiscleanup identities MUSTcome fromactualfuturelaunches,notguesses;
cleanup_metadata_full1.sh andevidence/sealhelpers NOTprepared yet.
No livefunctions/toolsession; actualtmux/service/remoteexperimentstillLIVE.
ThisgoalturnPROGRESS (D160 launched), NOTcompletedmodel. NEXTmonitorSAMEhandle;
fullcleanup→validation→table→backup onlyafteractualterminal/physicalrelease.

2026-10-02 12:41 +08. ONE Full launched12:40:20 in tmux tc-d160-7b-full1.
Runtime/evidenceHEAD11a5432ecac7ca6214417ffc471c834b6edeb3e3 alreadyBACKED.
4000/source42/W0/fixednativecontract, sameD158 configuration except threefresh
ownedpaths. D159 CPUexecutionplacement issolecandidate; no secondoptimizer,
profiler, diagnosticprefix, capacity/profilechange or remote cache recreation.
ReadfullPlan/status/V1 and run-experiment/monitor-experiment skills thisturn.
Raw results/ieee_tc/p2_backend_qualification/d160_20261002. DoNOTrelaunch.

- serviceprimelora-tc-svc-e8f24d386a694532b07696e17a696e1f.scope,
  actualb02b72c8dcf247dfa7f7d7a85605befc,72/80GiBswap2verified.
- auxprimelora-tc-aux-cf939d83ef2549fab79a51edac046699.scope,
  actual6cf2dfe92a9342a0ad98533cf5e17191,3/4GiBswap0verified.
  Replay3065490/start42326796 andwatchdog3065497/start42326882 exactaux
  CPU2,3,26,27. Nativeworkers pendingcapture. Sample35:service3288608768B,
  host108774916096B,swap/events0,no warnings/foreigncompute.
  Initialdisk162672603136B passes150GiBnewheavygate;running100GiBunchanged.
- Externalnotice423269.070846577/t0423329.070846577 exactly60s;
  traceviewa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
- Remote3B3654299/4123e27451d74ed1a3535defb7a93d42;
  7B3654301/211a92065e964fbcbe0aa3ec8551fcf7;
  monitord160full1 PID3654304/04f89d32746441eabfb33e167db3be3c.
  7Bclockremote-process-monotonic:e6112f99e41e40a3b01e6bd7fceadfff.
  Monitor/home/lab14/primelora_remote/tc/d160_20261002/remote_monitor_7b_full_full1.log.
  BothNIC1000/full; bothhealthPASS; immutablepublishedcache unchanged.
- prelaunch311refs/147protected/Plan/V1PASS,
  SHA9c8daf92fbd74e2b889002d0118782f2194bef917fd3f1004f404f0a6e34d237;
  actual07b5ff508aff4f09a04a929f678a7cd4/session92584CLOSED0.
  healthactuala9793de97e8e4b5280f9499944388c1a returned0;
  bothscopesautomaticallyremoved/absenceverified. No failedlaunchattempt.

NEXT monitor SAMErun→actualterminal/physicalrelease→exact-ownedremote stop and
SHAverifiedcopy→boundedmetadata/ONEprojection→D158comparisontable→backup.
No remoteoperations/hash/cleanup/build duringinference. DoNOTretestD159,
reprojectD158 orrecollectD157profiles. Actual7B G1G2/numeric/commonreferences
remainOPEN;3B andexternalbaselinesPAUSED. Runcompletion isnotmodelacceptance.

## PRIOR — D159 fused execution objectives QUALIFIED/SEALED and BACKED

12:32 +08 backup COMPLETE11a5432ecac7ca6214417ffc471c834b6edeb3e3;
pushedfaaslora_origin/retry14_continuous_queue_v2 and independentremoteHEADmatch.
Push10625/verify51294 CLOSED0.10explicitfiles/53staged+archivepayloads checked;
userdirtymanifestexcluded,diff/checksum/816regressions/147protectedPASS.
No livejobs. This goalturn PROGRESS, NOT7Bperformanceacceptance.
DoNOTretest/reseal/recreateonce-onlyremote cache. NextD160ordinaryFull4000W0
sameD158cap4/newD157profiles, freshpaths andcurrentruntimeHEAD11a5432.
No newFull launch/config yet. ReuseD158launch+postprocesshelpers, adaptfrozen
priorcomparisonD158 and newsourcecheckpoint; readfullPlan/status/V1beforelaunch.
Onlythispost-backupreceipt dirty. 7Bactualacceptancebefore3B/baselines unchanged.

2026-10-02 12:30 +08. Candidate qualification COMPLETE, no live GPU/remote/test.
regression2 816PASS139.897s(command152.11s/RSS1199084KiB), actual
940d747b12dc4f309fa208d6ca29e03f exited0 and scopeabsence verified12:27:20.
Do NOT repeat targeted/regression checks or old D158 projection/calibration.
Source original selector/objective functions byte-identical to bd3fc57;
only execution placement/envelope changed. Correctness is NOT servinggain.
Doc D159_EXECUTION_OBJECTIVES.md nowSEALED with failure and qualificationtable.
CuratedSHA b9aa4069a588467463ebb3961e625822f5b8bd3be07e5e002b8df6704b57f663.
43member326352B evidencebundleSHA
f50f811325809bb5f783cc6c3fa02e71892ecb63ab6025794625843986a58500.
Verify1 actualee37c5f7a48043cc946fb5626aaa81d6 exited0/9.05s,147protected/
Plan/V1/D158/sourcechecksPASS, absenceverified12:29:53. Bundle excludes its
active verify log; terminalreceipt preservedoutsidebundle and inthisledger.

Disk maintenance COMPLETE: reused unchangedD152 public-wheelaudit+guards;
reviewedallowlistSHA2be67f416990c47040b5e3f3b3180bdefe0b6a3fa9d191f13a196a3b496451d3.
5publicdownloads+5headers removed,allocated2334081024B. RegistrySHA/size/HEAD,
openfile/link/mount/reference/exactfileidentity andprojectgitguardsPASS.
Installedenvs/vLLMcompilecache/models/LoRA/traces/rawresults unchanged.
Auditactuala1e0937698bf42af992481eee8cf3dab/applye0c016ac8c3c45b0b88c34095aea883c
both0/scopesabsent. Available162690375680B(~151.52GiB), recheckbeforeheavy.
NEXT scopedsecrets/diff/membercheck+backup, thenD160 ordinary7BFull4000W0 with
sameD158 cap4/profiles/controls/trace andfreshownedpaths. No morebootstrap/profile
collection; candidate requires actualnetbenefit test, no performanceclaimyet.
Bothper-modelgoals/numeric/commonwarm/Resident/G1G2 remainOPEN;3B/baselinesPAUSED.

### D159 qualification history (superseded by closure above)

2026-10-02 12:24 +08. Only Prime7B optimization; no GPU or remote service run.
Latest user once-only delivery-cache approval is already implemented D78/D80;
reuse immutable published objects, NEVER rebuild a second pool or put packing
back on request path. Plan/V1 unchanged. D158 remains sealed/backed.

Single candidate: original pure GPU/file execution-objective functions now run
inside the existing single planning worker transaction; no extra worker/RPC for
ordinary owned epochs. Pre-init actual-snapshot binding goes to SAME worker.
All live registration/owner/epoch/reservation/commit/cleanup checks unchanged.
History D153 caller observations + D158 stage table and official vLLM/Python
sources documented in D159_EXECUTION_OBJECTIVES.md. No performance claim yet.

targeted1 9tests/1error retained: handoff does not contain residency size-edge
field; now explicitly supplied by frozen profile, not an invented default.
Actualf1b1fdec2e7247a0922bef443e259c60 exited1, absence verified.
targeted2 9PASS62.157s,command72.51s/RSS1176392KiB;
actualf765b6ff3c4e411190510880ffc6c323 exited0, absence verified.
regression1 816tests/2fail5error retained: all7 legacy file-only paths lack a
native profile by design. Explicit file-only contract now supplies N/A classes;
GPU targets still reject absent classes. No test weakening/timeout changes.
Actual1aba46f627c641b2b05ed59e2e39cdcd exited1, absence verified.
regression2 was launched in tmux tc-d159-regression2, bounded scope
primelora-d159-regression2-20261002.scope; completedabove. Raw d159_20261002.

Disk160359530496B below150GiBnewheavyfloor. Readonly storage inspection found
public pip downloads; reused D152 strict public-wheel auditor wrappers prepared
but NOTexecuted. No deletions, no vLLM compilation cache change. Need reviewed
allowlist/SHA/open-file/link/reference checks before any permitted reclamation.
NEXT regressionterminal→exactcleanup→qualificationtable→auditedcachemaintenance
→seal/backup→ordinaryFull4000W0 sameD158cap4profiles. No repeatedcalibration.
7B actualG1G2/numeric/commonreferences OPEN; 3B and externalbaselines PAUSED.

## PRIOR — D158 cap4 Full COMPLETE, analyzed/tabled/sealed and BACKED

12:04 +08 backup COMPLETEbd3fc57f4f93dfcafd8add8582514c58f281b2be;
pushedfaaslora_origin/retry14_continuous_queue_v2,remoteHEADindependentlymatches.
Push69311/verify94767CLOSED0.12explicitfiles/73payloadscredential-syntax-member
hashchecksPASS;userdirtymanifestexcluded. Initialgit-addinvocationhadwrongcwd
andstagednothing; correctedfromreporootbeforechecks. No measurementrerun.
CurrentnextworkREADONLY: post-gate execution-objective synchronousCPU audit,
reusingD153historicalcallerobservationsandD158stagetable; no newcandidate
implemented orGPUrunselected. OfficialvLLMCPU-processseparationblog/Python3.12
blocking-codeguidancecheckedonline. GoalACTIVE,7BacceptanceOPEN,3B/baselinesPAUSED.

Next-path inspection only: three synchronous calls remain in current runner
at17637/17697/17878 (owned GPU/file objective derivation). Existing pure worker
supports owned_execution_epoch/validate_execution only. Functions compute frozen
costs/candidates/deepcopies/JSON hashes+original validators; live registration,
reservation and commit occur later. Possible offloading/fusion must preserve
those exact values and post-await ownership/cancellation/staleness checks;
do not add a synchronous fallback, extra workers, blindcapacity, or deletechecks.
This is NOT yet a selected/measured optimizer. D153's34inclusivehistoricalframes
cannot be used as current CPUpercentages. One ad-hoc jq schema inspection returned
null-plan error without writing files; no result or cause inferred from it.
Use bounded/scoped inspection if actual plan payload needed, neverwhole-load
giant normalresults. AllD158analysis/closurealreadycomplete; doNOTrepeat them.
Onlythispost-backupstatusreceipt isnewdirty; servingcodeunchanged. No tool
sessions leftlive. NextcontinuationreadfullPlan/status/V1 thenfinishone evidence-
based candidate selection. Before any newheavylaunch recheck150GiBdiskgate;
near-floor state does not authorize thresholdrelaxation or deletingrawresults.

12:01 +08. GoalACTIVE/PROGRESS, no live GPU/remote/analysis job. D158 runnotrepeated.
4000native/0failure, numericadapter/commonSLO/G1G2 stillOPEN,n_correctnull.
U15955.403561064915GPU-s/all4released. Mean/P95/P99TTFT6.409877344/17.177111366/
30.044413724s; TPOTmean/P9543.653935559/82.581508708ms. VsD154 meanTTFT-93.831%,
P95-91.839%,U-0.607%,BUTTPOTmean+18.470%/P95+47.117%; NOTmodelacceptance.
Dispatch4.347203608s=67.820%TTFT; service2.062673737/native.438309931s.
Phasesarrival→gate2.485875241;gate→source1.861328367;source→dispatch1.624363805;
native4.913926683;last→controller.679276340;controller→terminal.475202973s.
Gate→terminal9.554098167s UPPERenvelope;native/gatemeanconcurrency4.942781/9.610198,
max15/16. Controls1706allinwindow,1058queued/945belowreadycapacity.
Planning1866receipts:1865completed+1cancelleddiscarded,906.173312workerCPU-s.
15nonoutputfields4000matchD154;123outputhashchangesOPEN. Timingerrors0ms;
dispatchtiers4000/conflicts0. Remote132pairs131480060wireB/allpublished/packing0.
Resources4053samples/peak20526383104B/minhost93204672512B/swap-events0/no warning.

Copy87848CLOSED0; remotejournal+monitorSHAverified. Metadata43359CLOSED0/4.18s/
RSS434120KiBactual48bfa7388feb464e9cec23713f37302c. Projection79391CLOSED0/
81.25s/RSS99840KiBactual0e8723c64b83473bad043b471219222a;272793836Bsource
streamedONCE. ProjectionSHA794da4a0a41a07c782f065247eac475f131903ecb48b2d018527352015e3aad4.
Curation74957CLOSED0/8.58s/RSS452120KiBactual8b4d78acd8724ef7b930246f524e270e.
Occupancy40419CLOSED0/3.19sactual7821916901924e1ebceaef5967ff6ae0,finalevents0.
Verification42829CLOSED0actual18c75473e26148c1a9191412aa60530a,300frozenrefs/
47curatedrefs/147protectedPASS,SHA d12158e0ffede7eb7c4aa10472ddf8d456f4b3617383d469f7c00838cb64e25c.
Bundleexit0actualdfd20a4be1d840e4b72d0195d16d10c7,61members39629B,
SHA6ca3d02e043ff919a09c0b20fe8b9111e9d98673d608902f67031fbd7b5f98f7.
Allscopeabsencesverified. Compact126043991B retains128MiBguard.
CuratedSHA bfaed07c833b33dc64f2a4768eda53990068c0ed17baf93c318eecb76d420c23.
Doc D158_FULL_W0_FULL1.md SEALED,DONOTedit. Fullcomparative/stagetablesready;
noCI/numericqualification. Servingunchanged39cc3c1. Backup checkpointpending.
NEXT backup then source/history/primary-reference audit of remainingpostgate
control/preparationdelay. Retaincap4asdevelopmentcandidate,notformalwinner;
doNOTrepeatcalibration orblindcapincrease. Actual7Bacceptancebefore3B/baselines.

## D158 terminal history — completed and superseded by closure above

11:47 +08 terminal recheck: SAME D158 finished06:49,4000/4000 banner success.
launch.pass=true, service/replay/watchdog return0, physicalrelease/pathremoved
confirmed; launchSHA900180c159d9d055b29fd7c4129c9d48ae6a9bb0f0b77b94aa26e244d3f12223.
Earlier monitoring messages were last observations, NOT current live state.
No repeated launch/profile/cache preparation. tmux/scopes/PIDs/GPU absent;
verify_local_full1_cleanup.sh PASS11:47:28. Exact remote services+monitor stopped
11:47:29, MainPID0/Resultsuccess; stop session80090 CLOSED0. Remote monitor has
post-inference idle tail until shutdown: preserve, but do not count tail as the
inference measurement window. No remote changes occurred during inference.
Actual journal transfers-a06e30014e084e43a1c5e118ac620adc.jsonl in remoteD80/7b;
frozenclock37295317e6ed45eb821274319e387def. SHAverifiedcopy pending next.
Nativecontract/numericaladapter/commonSLO/G1G2 still pending; banner latency and
legacy cost/SLO NOT final evaluation. NEXT copy→boundedmetadata→ONEprojection→
curation/table/figure→interpretation/backup. Baselines/3B remain PAUSED.

## D158 observation history — superseded by terminal state above

05:56:37 +08 verified wait: full Plan/status read this turn; run-experiment and
monitor-experiment followed. SAME tmux tc-d158-7b-full1 and exact service
InvocationID f06a18ca5374415a83378e405226042a remain active,533tasks.
Eight 50-second-spaced light polls completed (exec cell7881 CLOSED normally);
this is NOT experiment termination. Latest banner797/4000success,802arrived,
0fail. Watchsample924 service16.8318GiB,host88.6918GiB,events/swap0,no warning;
disk149.0542GiB above100GiBrunningfloor/below150GiBnew-heavyfloor. No new GPU
run, source/config/remote changes, analysis, bulk hashing, or cleanup performed.
Previous and current goal turns VERIFIED WAIT, not blocked/not completed.
NEXT poll SAME live handle; only after terminal+physicalrelease run the prepared
exact-owned cleanup and bounded postprocessing. No outstanding monitor/tool
session, but tmux/service/remote remain LIVE. Do not restart or reprofile.

05:48 +08 continuation: full Plan/status/V1 and run-experiment reread; SAME
tmux run remains live, no repeated launch/profile/test. Live banner285/4000
reported successes,306arrived,0fail; incomplete, NOT final native/numeric/SLO
acceptance. Four native cores1292310/1297311/1297612/1297737 observed in exact
service cgroup with CPU4–23,28–47. Replay1288932/watchdog1288936 inaux.
Service/aux actualInvocationIDs revalidated active,533/5tasks. Sample435:
servicecurrent16.2017GiB/peak18.0157GiB,host88.9905GiB,swap/events0,no warning.
Disk149.4211GiB is below new-heavy150GiB gate but above running100GiB floor;
do NOT clean/compress during inference, recheck after owned native cleanup.
Actual service.log now present at launch.launch/service.log; prior absence
was not a stopped-run finding. No remote changes/hash/extra probes.

Prepared ONLY, NOT executed: exact-owned remote stop/copy and local-absence
helpers adapted from D154 actual IDs;7 metadata/projection/curation/occupancy
helpers reused with fresh paths and sealed D154 comparison SHAs. Shell/AST
syntax PASS. Keep128MiB metadata guard, ONE streaming projection, all failure
and numerical/SLO caveats. Post-run analysis cleanup InvocationIDs must be
captured from actual launches (no guessed IDs). Source serving unchanged.
This continuation is a VERIFIED WAIT, not a newly completed experiment.

2026-10-02 05:41 +08. PreviousgoalturnPROGRESS (D157complete/backed2c3d4d6).
ThisturnPROGRESS: freshFullprelaunchPASS and ONEactualFullstarted05:40:59 in
tmux tc-d158-7b-full1. Rawresults/ieee_tc/p2_backend_qualification/d158_20261002.
Serving source39cc3c1 unchanged; evidenceHEAD2c3d4d6bab50275991025d2c79b175909aaaa75f.
Same4000/source42/W0/fixednativecontract, no profiler/prefix. ExactD157parent
configuration except threefreshownedpaths; versusD154 onlycap2→4+newmeasured
profiles/lengthbinding. AllD88/D154 controlthresholds/window/cooldown unchanged.
No oldprofile relabel; no newtrace/weights/cachegeneration. G1G2 stillOPEN.

- serviceprimelora-tc-svc-dfd79a7266164dd6b618f5de5176f088.scope,
  actualf06a18ca5374415a83378e405226042a.
- auxprimelora-tc-aux-7fbea46f21c142679502de2f4a04aef8.scope,
  actual41e91347bc764ddeb9c465ef7470b5f6;3/4GiBswap0 verified.
  Replay1288932/start39810757/CPU2,3,26,27 inaux. Nativeworker pendingcapture.
  Watchsample18 no warning/abort,service1673850880B/host110702579712B.
- Externalnotice398108.706786477/t0398168.706786477 exactly60s;4000plan,
  viewa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
- Remote3B2969058/79a234da230b4e0a9a56b5f82f9229e8;
  7B2969060/474129f7ef96401fbdaa8c71ea7c1154;
  monitord158full1 PID2969063/6ea9834c984f452dbb15be64acae72b1.
  7Bclockremote-process-monotonic:37295317e6ed45eb821274319e387def.
  Monitor/home/lab14/primelora_remote/tc/d158_20261002/remote_monitor_7b_full_full1.log.
  BothNIC1000/full, healthbothPASS, publishedservicesunchanged.
- prelaunch300refs/147protected/fullPlan/V1PASS;
  SHA f6e99d47bb58a2b056ee33f10c1b231d116fa7533375327be43be21b19815335.
  prelaunchactual475edc9624064353a901519012a19c7c/67746CLOSED0;
  healthactual67bab28245994700b54d289dac07187e returned0; bothscopesremoved.
  Initialdisk161490014208B (~150.40GiB) passes150GiBnewheavyfloor.

NEXT monitor SAMErun→actualterminal/physicalrelease→exactownedremote stop and
SHAverifiedcopy→boundedmetadata/ONEprojection→D154comparison/table→backup.
No remotechange/hash/cleanup/build duringinference; no secondoptimizer/replay.
Baselines/3Bpaused until actualper-modelacceptance. No formalSLO/numeric claim.

## PRIOR — D157 cap4 source calibration COMPLETE and BACKED

05:36 +08 backupCOMPLETE2c3d4d6bab50275991025d2c79b175909aaaa75f, pushed
faaslora_origin/retry14_continuous_queue_v2; independentremoteHEADmatches.
Push79107/verify37131CLOSED0.23explicitfiles/81payloadsecretsPASS; checksum/
46analysis-tests/syntax/sealedrefs/diffPASS. Usermanifestexcluded. No livejobs.
This goal turn PROGRESS (actual184samplecalibration,completecoverage,strictnew
profiles,mainassembly,figures,backup), NOT7Bacceptance. DoNOTrepeatcompletedD157.
NextD158 ordinaryFull4000W0: reuseD157raw/7b_main_config_full1.yaml, changingONLY
threeownedoutput/NVMe/HOSTpaths; usecurrentHEAD2c3d4d6, unchangedserving39cc3c1.
ReuseD154launcher/prelaunch/remotehealth/cleanup withnewpaths/identities and
newprofileprovenance. D154same-configcomparisonassert needsdeclaredcap/profile
differences, not oldcap2 assertion. Profilesalreadyqualified, no more GPU probes.
No Full D158 prepared/launched yet. Beforeheavy readfullPlan/status/V1 and
recheckdisk150GiB: current161490214912B (~150.40GiB). Neverweakenfloor.
G1G2/numericadapter/commonwarm/Resident references andoldPrimecomparison open;
3B/externalbaselinesstayPAUSED. Onlythispost-backupreceipt isnewlocaldirty.

05:35 +08 evidenceverificationPASS SHA61b606b02363a85952fd53bf7d408c19de40875393468530c846aefed7934158.
Doc/result/figures/profiles nowSEALED, doNOTedit. Complete58member35338B bundle
20261002_d157_analysis_sources_complete3.tar.gz SHA45e457596cb2eef648786edb7dc2decf71f7c85c052758d383460303ee51fe11.
Bundle3 actualbc77804a4e634354979611b5c7b29871 exited0/events0/scopeabsent.
Seal1 actuald876501e1fac4d4baba0de17256b3385 exit1: symlinkrelativepath issue;
seal2 actual5a03c69c6ac54790973fcd28f3762bf1 verified all evidence then packaging
failed at old check_health.sh name (actualcheck_health1.sh). Partial6925B bundle
retained20261002_d157_analysis_sources.tar.gz, NOT complete/notforpublication;
complete3 only is selected. Bothfailure scripts/logs included in completebundle.
No GPU repeat or changed measureddata; backup pending. NextordinaryFull needs
freshD158paths+HEAD, same generatedD157Fullconfig except threeowned paths.

05:30 +08 goalACTIVE/PROGRESS. No live GPU/remote/analysis jobs.
Source1 PASS184/184 (180representative),30/30serviceclasses,24preparationclasses
each6observations,500static/sixexactfilecontentclasses. StrictprofileexportPASS.
RawSHA542b258cdf2d95035ad86597ae79f597d81475e23333a34ed05ea1ec25bae4b3.
U585.0443832949968GPU-s/singlelease released;184remoteUUIDpairs/995431947wireB,
pack0;591resourcesamples/peak5413052416B/observedmemoryevents+swap0.
No numericaladapter/commonSLO/G1G2 qualification, n_correctnull. No serving edit.
Sourcecontrols retained D88 exact; onlycapacity+newmeasured initialization.
Physical/local absence and exactremote stop/copy complete; journal48e3e6b96db34456b33087b0b2c5b579,
remoteclock81ff915daec54d238ee3347b8044aebf. stop97057/copy56013closed0.
Analysis1 ff49d00cec8744a98c80d1eb31f2783d/85216exit1: wrong plotting environment;
analysis2 13f9aeac24604d6c831f74ecd92b76ad/93219exit1: oldpurpose label rejected.
Both retained/no GPU rerun. Analysis3 992688a8be01418b98c72551c9a89608/52934CLOSED0,
scope absent/finalmemoryevents0.46testsPASS7.498s; explicitplotpurpose interface,
all strictdata checks unchanged; baseconda plotting,existingCPUenv analysis.
Plot3/summary3/export3/assembly3 all0; manual3PNG/TNR/PDF QA complete.
Newprofiles d157_7b_initialization/service d7b359b9...; preparation57b2d1d5...;
main_assembly3.json PASSactualmain4000/500/onecap4/fourcap16, noengine/network.
Doc/table D157_CAP4_ADMISSION_SOURCE.md; generatedFullparentconfiginD157raw.
NEXT seal/backup then ordinaryFull4000W0 usingnewprofile/parent. DoNOTreprofile,
rebootstrap, or advance3B/baselines. Current disk near150GiB: recheck beforeheavy.

## D157 launch history — completed, superseded by closure above

05:20 +08 same run active in tmux tc-d157-7b-source1; do NOT restart.
Service42471725ef7548b087ec11d390767311 actual474767ea09454474a7a2d9cfaaa8f72c;
aux115376acebd440c3ba73e074146a02ae actual3b3c851fcdc3401c942d5bd85748f77e.
NativePID1175604/start39658441 inside service CPU4–23,28–47. Sample215:
peak5413052416B,host108567822336B,swap/events0,no warnings/foreign compute.
Healthclock remote-process-monotonic:81ff915daec54d238ee3347b8044aebf.
NEXT monitor same source1→physical cleanup→exactremote stop/copy→strict
classcoverage/profile export/table/backup. No timing profile yet qualified.

2026-10-02 05:15 +08. PreviousgoalturnPROGRESS/D156completebacked26e44a2.
ReadfullPlan/status/V1 thisturn; no servingcodechanges. Use sameD156cap4config.
D15746waves/184samples(4warmup+180representative),sixexistingD84representatives,
threeinterleavedrounds/fivesources,admittedbins[1,2,4]. No newtrace/weights.
SpecSHA1d29c6fbcb3232d8d62f14d23fcfae1b342311268d2ba3ce5ec33bc047559055.
NewcontractSHA875f90f6133eb2632d0ec07db03b0c3e05f22b03b1dea14e1cf51e5400562d58
holdsD88coordination/settings/betas EXACT; onlycapacity/modelidentity/fresh
D156lengthinitializerchanged. No reuse/relabelofoldtimeprofile.
Prepare1PASS184indexed/500static/6representative/147protected/strictassembly,
actuald489d6acadcf453e804cd15f8a5c44d6,session11699closed0,scopeabsent.
Remoteactivatedsource1 exact3B2938819/e4f612301bc24c26b183baa15889154d;
7B2938821/ab1d1eee19824250984b4e508aa88f32;
monitor2938824/d74ef3a976dd4eecae3f5cd79d7ef11b,
unitprimelora-artifact-monitor-d157source1.service. HealthbothPASS.
BothNIC1000/full; no remoteconfig/cachechanges. Disk161564647424B meets150GiB,
remainingmargin~0.47GiB; running100GiBfloorunchanged.
Launch completed05:15:12; then exactcleanup→classcoverage/token/time/admission
verification→profileexport/table→backup→ordinaryFull4000. DoNOTrepeatD156.
G1G2/numericadapter/commonreferencesstillOPEN; baselines/3BremainPAUSED.

## PRIOR — D156 cap=4 bootstrap COMPLETE and backed

05:11 +08 BACKUP COMPLETE26e44a2014a9178148afddf1b18447fd6b6e5ab3;
pushedfaaslora_origin/retry14_continuous_queue_v2, exactremoteHEADverified.
Push37635/verify45769closed0;11explicitfiles/39payloadssecretsPASS; usermanifest
excluded. Diff/checksum/nativequalification/inputscope/syntaxPASS. No livejobs.
Onlythispost-backupreceipt localdirty. TurnPROGRESS:actualcap4bootstrap+new
strictlengthbinding+table+backup complete, NOT7Bacceptance. Nextadmission-enabled
sourceprofilecap4 thenFull; don'tretestsealedbootstrap orD155. GoalACTIVE.

05:10 +08 seal COMPLETE:147protected/config/trace/raw/remote/Plan/V1 SHAs PASS;
source syntax and actualbootstrap12native checks PASS.29member smallbundle12998B,
SHA08edc11c8384ae6083231e8afb13f56dd20bf055daaf0c6c924bfe70c6d070aa.
Sealactual7e3039dc2fb8415e8432dfef41e9088c,exited0/finalevents0,scopeabsent.
D156doc/result nowSEALED by verificationmanifest; don'tedit. Curation10.63s/
RSS1176060KiB. Disk161567154176B (~150.47GiB), recheck150GiBfloor before
nextheavy; no unsafe cleanup. Backup checkpoint pending; no livejobs.

2026-10-02 05:07 +08. GoalACTIVE, Prime7B only. No live GPU/remote/analysis job.
D156 actual bootstrap2 PASS:12/12 native token/source/timing/retirement checks.
Four native decode intervals intersect5.0070s(warmup),5.0506s/2.8809s(measured).
Measuredwave nativeTTFT250.893/233.430ms,TPOT34.475/33.322ms, noCI/mainranking.
StartupfreeKV278blocks vs maxdeclared4contexts need256; block16tokens/8MiB.
NotcontinuousKV/preemption or fullphysicaladmissionproof. U114.038600675GPU-s,
oneactuallease released; profileworkspaces removed. n_correct=null.
Remote12pairs/65204801onlinebytes/pack0, contentSHAverified. Peak5416239104B,
observedhigh/max/OOM/swap0. Numericadapter/commonSLO/G1G2 stillOPEN.
ResultSHA befd3161e9ef2cd81b2271ba08777dd61fc623219a4babfeec5f34b6773f84ac.
Serviceactualaea5686a4ac844bb9c5595b936894bfe / f757d22d0c054074a8e6df7179df7b7e;
auxactual5854b37b19e942a4971b1e142ca26cff / a12e3e2da10d430688acd5eaba3d18c6.
Bothscopesautomaticallyremoved/nativePID1117836absent. Exactremote3services
stopped05:04:35,success. Remotejournal359475af4bf745b8b0694993b8ce48f3,
clock695cea944a90406fadb1f81dd97d6b86; copySHAsmatched. Stopcommand52458
exit127 ONLY because subsequentrgnotinstalled; grep fallback foundexactjournal,
copy28521closed0. No remote restart. Preserve this toolingerror.
Curation83185closed0,actuald63097717f94459fab73b420a9a1be56,scopeabsent.
Newcompleted-length audit passed unchanged strictmeasured_admission_initializer;
source-onlyGPUtimings NOT Full/admissionprofile. NewCSV/table/doc D156.
Bootstrap1 wasonly31-digitauxname rejectedpreGPU; source/configunchanged in2.
NEXT seal/backup thisqualification; then prepareadmission-enabled 3round source
matrix forcap4 usingexistingrepresentatives/waves/[1,2,4]admittedbins, no fresh
workload/weights oroldtimingrelabel. Aftercoverage→ordinaryFull4000W0.
DoNOTrepeatD155/bootstrap2 oradvance3B/baselines beforeactualmodelacceptance.

## D156 preparation history — superseded by completed state above

2026-10-02 05:01 +08. D155 remains complete/backed; not repeated.
D156 candidate config changes ONLY max_num_seqs/runtime_concurrency_cap 2→4,
SHA4aba5abd28e11cb8518b2d4fc5938276b5b67e52f4f077f871d9d730fbfbe733.
Bootstrap spec SHA10d85d3354222b7c7e222b80099d6e62b156a56454fb5c9ad3d1bbbb41420959:
three four-lane GPU-ready waves (one retained warmup), six existing D84
representatives, 12 observations; source42 unchanged; no new weights/trace.
Source-only completed lengths bootstrap, not a Full run/admission profile/SLO.
No serving source edits. Native-source existing collector reused unchanged.
Inputs/config-diff/147protected/Plan/V1 PASS under3/4GiB/swap0:
prepare1 actualaec9b1c8610c4251933e315ed752dd54, session25287closed0,
scope automatically removed. Raw d156_20261002/ bootstrap_inputs.json.
Remote exact3B2924062/a486fad1f16a42c49ebe070f569874a5;
7B2924064/882bd27170b34b40b79ea268c1785b92;
monitor2924068/87ba3159b45f4e65b1050e210ff7eff8,
unitprimelora-artifact-monitor-d156bootstrap1.service.
BothNIC1000/full; published cache reused unchanged.
First health probe raced listener startup (connection refused, exit1); retained
empty health_3b.json. Both listener presence then verified; fresh health2 files
PASS, no service restart, no GPU attempt yet. Health scopes removed.
NEXT launch ONE guarded tmux bootstrap; check actual KV/native overlap/tokens/
retirement, then table; fresh admission-enabled class profile still pending.
G1/G2/numericadapter/commonreference and 3B/baseline work remain open/paused.

## PRIOR — D155 capacity audit COMPLETE; selected cap=4 qualification

04:51 +08 BACKUP COMPLETE6399ad2b0c60dacabfcffde46791af2dc9302a2e,
pushedfaaslora_origin/retry14_continuous_queue_v2; independentremoteHEADmatches.
Push62340/verify4501closed0.10explicitfiles/19payloadssecretsPASS; curated/source
checksums,32tests,bundle,syntax and CRLF-preservingdiffcheckPASS. Usermanifest
notstaged. Onlylocalbackupreceipt addedaftercommit. No livejobs, GPUidle.
Disk161616203776B (~150.52GiB): close tonew-heavy150GiBgate, recheck before
qualification and accountforcompiler/profilegrowth; do not weakenfloor.
Nextcap4 remains ONE selectedcandidate, no servingconfigurationedited/launched.
This goal turn PROGRESS (newcompletecapacityaudit+tests+table+backup), not
modelacceptance. NEXT prepare/qualify4concurrent native-source index with
freshmodelidentity; no repeat D155 analysis/tests or D154 fullreplay.

04:50 +08 D155 seal complete:147protectedPASS, exactanalysis/scopeabsence,
Plan/V1/sourceSHAs/syntax/tests/bundlechecksPASS.10memberbundle5259B,
SHA91223deed5c1dab7ed29c976355a3ff2542ce4ac2f767619a9df79cf6faaeca3.
Sealactualec104dad15704de0941c90b9db87c8d7 exited0; finalmemoryevents0,
nowinactive/emptyID. D155doc sealed by verificationmanifest; doNOTedit it.
Backup was pending atseal time; completedabove. Only offlineanalyzer changed.

2026-10-02 04:49 +08. GoalACTIVE, Prime7B only, no liveGPU/remote/analysis job.
D154 remains SEALED/BACKED d410447; no repeated replay or giant projection.
D155 extends existing offline analyzer only; serving source39cc3c1 unchanged.
32 tests PASS (8new+24timeline),0.141s body; command8.86s/RSS1082172KiB.
Audit10.88s/RSS1445280KiB; actualscopea9946ef3c54f4866973bed192b9b1f75,
3/4GiBswap0CPU2,3,26,27; finalmemoryevents0, automaticremoval verified.
Execsession95513 CLOSED0. Raw results/ieee_tc/p2_backend_qualification/d155_20261002.
408 distinct admission snapshots/816occurrences,408consistentduplicates:
4replicas minimumfreeKV178/180/178/178blocks, each16tokens/8MiB; maxfree304,
admittedmax2. Event-selected, NOTcontinuousKV/preemption evidence; totalsunknown.
Curated paper_results/ieee_tc/p2_backend/20261002_d155_admission_capacity/summary.json
SHAe179c24144e84bc9b2c4e70106c6809ea9e2d06acfd7e03c6d8d2e85a87fffe9.
Doc/table D155_ADMISSION_CAPACITY_AUDIT.md has fullreasoning and nextqualification.

Single next hypothesis SELECTED, not implemented/qualified:7B max_num_seqs and
runtime/requested concurrency2→4, otherwise same .70/TP1/FP16/slots4/rank64/
maxlen1024/batchtokens1024/prefixfalse. Candidate4 boundedby4LoRAslots and
floor(observedmaxfree304/ceil(1024/16))=4. This is candidate-screening arithmetic,
NOT paperformula/onlinepolicy or safetyproof. Reobserve actualKVatstartup.
Reuse native_source_matrix with existing4concurrentrequest/artifact indices;
mustprove actualnativeoverlap, correcttokens/source, retirement and resources.
New capacity changesruntime/profileidentity; doNOTrelabelD89cap2timings or weaken
measured_admission_initializer's strictidentity. Collectnewconfigurationevidence
then builditsadmission/service/preparationprofiles and fullclasscoverage before
ordinaryFull4000. No newweights/traces, no deadlineincrease, no secondoptimizer.
If capacitycandidatefails preservefailure; don'tblindretry. Allcommonreference/
numericadapter/G1G2/oldPrimeacceptance stillOPEN; baselines and3BstayPAUSED.
NEXT prepare ONEcap4qualification; D155secrets/backup completedabove.
Skills analyze-results/academic-plotting/vLLM/github-sync read; vLLM official0.30
source+docscheckedonline. GenericskilltargetspeedsNOTouracceptanceprotocol.

## PRIOR — D154 ordinary 7B Full sealed and BACKED; 7B goal still open

04:32 +08 backup COMPLETE d4104476451037a6b019a015732284af5b312afc,
pushed faaslora_origin/retry14_continuous_queue_v2; remote HEAD independently
verified. Push68724/verify92865 CLOSED0. Twelve explicit files/76payloads
syntax/secrets/bundlePASS; usermanifest excluded. Existing CSV CRLF preserved
byte-for-byte with sealed SHAs; Git whitespace check with cr-at-eol passed
(default check flagged only CSV line terminators). No data rewrite to pass check.
No live GPU/remote/analysis/exec job; all D154 closure complete, don't repeat.
Serving source still39cc3c1, d410447 is evidence-only checkpoint.

Read-only next-path review: global dispatch limit is runtime_groups*runtime_cap,
7B maximum8, and its permit covers source preparation, native generation AND
acknowledged cleanup (run_one serve finally). Slot.active_requests is likewise
released only after pending/native GPU/HOST acknowledgements. Do not conflate
these limits with raw native iteration capacity, remove cleanup, or free at last
token. D154 gate→terminal remains an upper envelope, not exact release tracing.
Pure file/GPU execution-objective construction still synchronous on mainloop at
run_all_experiments.py:17637/17697/17878; already-created CPU planning worker
currently supports owned_execution_epoch/validate_execution only. This is an
audit observation, NOT yet selected candidate or proof of dominant bottleneck.
Historical D143 frame counts are not current CPU percentages. Relevant official
vLLM performance blog (2024-09-05-perf-update) and Python3.12 asyncio blocking
guidance rechecked online; their speedups are not borrowed as ours. NEXT choose
one measured structural hypothesis, test minimally, then Full; no baseline or
3B transition until actual model acceptance, no replay without new hypothesis.

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
