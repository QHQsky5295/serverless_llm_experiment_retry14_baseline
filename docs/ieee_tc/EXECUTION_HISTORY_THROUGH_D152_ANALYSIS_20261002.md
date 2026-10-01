# IEEE TC execution status

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
acceptance test. Apply immediately; synchronize Plan after D152 provenance
closure as already recorded. Frozen V1 itself is not changed.

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
This directive is effective immediately. Synchronize it into the authoritative
Plan AFTER D152 closure/provenance checks, preserving this run's original Plan
identity and documenting the revision; do not edit frozen V1 in place.

## CURRENT — D152 ordinary 7B Full finished; bounded analysis in progress

2026-10-02 02:30 +08. Full finished02:20:50: launchPASS, all three returncodes0,
4000 native-contract completions/0fail, physical15994.237882167974GPU-s,
4allocations/allreleased/0open. n_correct remainsnull; no numericaladapter or
formalSLO/G1G2 acceptance. Local service/aux automatically removed; exact
absence/PID/GPU checks saved02:22:41. Do NOT run manual retained-ID auxcleanup.
Exactremote3B/7B/monitor stop completed02:21:23; copy source/local SHA matched,
sessions37834/2660 CLOSED0. No GPU/remote service. Cachepublication never repeated.

Metadata64105 CLOSED0/4.05s/RSS435640KiB, actual6cbb7e2178634059b4a0298ef5376ce9
automaticabsence verified02:23:40. Compact126001682B fits unchanged128MiBguard.
Preliminary4062samples:peak20379181056B,minhost93121208320B,swap/events0,
warnings0. Remote132pairs/131480060B/3300789780logical,allpublished,packing0;
1initial+3natural/0quarantine. These are not formalperformance qualification.

ONEprojection8221 CLOSED0/78.68s/RSS99840KiB; actual8bbc6bc7aa2e41edbd158a9af114026b,
unitprimelora-d152-full1-project-20261002.scope nowinactive/Invocationempty.
Original271163137B streamedONCE. Never reproject or read oldgiant originals.
Exact absence receipt next, then prepared boundedcuration/failure/occupancy,
table/interpretation/seal/backup. No liveexec/toolhandle; curation NOTyetlaunched.
FullPlan/status/V1 and applicable skills reread. FrozenV1/Plan unchanged;
latest userdirectives above apply now, Plan amendment deferred until closure.
GoalACTIVE/PROGRESS; baselinesPAUSED; stay7B until actual per-modelgoal.

## D152 launch/monitor history — superseded by CURRENT

02:14:04 latest user-clarification handling: SAME service observed active533tasks;
3807arrived/3728done/3728ok/0fail,79backlog. Sample3700 service20297916416B,
host94713053184B,swap/events0,no warnings,disk159893733376B. Observercell7465
deliberately TERMINATED only to persist latest directive; underlying tmux,
service/replay/watchdog UNCHANGED and live. 7463 also deliberately terminated;
earlier7460/7457/7453 closed. No live observerhandle now. Resume monitoring SAME
run, never relaunch. No serving/config/remote changes or new optimizer.

02:04 user update handling: monitorcell7463 deliberately TERMINATED after
same-invocation polls (latest3388arrived/3179done/3179ok/0fail). This stopped only
the lightweight observer, NOT the tmux/service/watchdog. Exact service/aux IDs
reverified after observer termination. No live toolhandle; actualrun continues.
Resume observation of SAME run, never restart. No config/source/remote change.

01:58:18 continuation VERIFIED WAIT: SAME service/aux IDs active, no terminal.
Livebanner3053arrived/2821done/2821ok/0fail,232backlog,4runtimes. Sample2766
service19613138944B,host95341821952B,swap/high/max/oom/oomkill0,no warning/abort,
disk160568356864B. Disk below150GiB new-heavy gate but above100GiB running
stop; no maintenance during measurement, no new heavy launch. Plan/V1 SHA and
HEAD26fa0a086bf91895e667a1996662971a526c860b unchanged at01:58.
Monitorcell7460 completed24x50s polls and CLOSED normally; 7457/7453 CLOSED.
No live toolhandle/execsession; actualrun remains SAMEtmux. NEVER restart on
monitor-cycle completion. Previous and current goalturns VERIFIED WAIT, actual
request progress confirmed; goalACTIVE/notblocked. No new source/config,
remote operation, analysis, optimization or repeated tests. Prepared postcleanup
helpers remain unexecuted. NEXT true terminal/physical release -> exactcleanup
-> bounded analysis/table/interpretation/backup. No performance closure yet.

01:36:59 continuation VERIFIED WAIT: SAME service/aux actualInvocation IDs
confirmed active; no terminal launch.json. Livebanner1428arrived/1365done/
1365ok/0fail,63backlog,4runtimes. Sample1504 service18442121216B,
host95632056320B,swap/high/max/oom/oomkill0,no warning/abort,
disk161284857856B. Monitorcell7457 completed20x45s polls and CLOSED normally;
7453 also CLOSED. No live toolhandle/execsession. Actualrun remains SAMEtmux;
cycle limit is NOT completion and must NOT cause restart. Previousgoalturn
PROGRESS(launch), currentgoalturn VERIFIED WAIT, goalACTIVE/notblocked.
FullPlan/status/V1 were read in preceding continuation, hash rechecked unchanged
01:21; currentledger/monitor skill reread and actual process state verified.
No newexperiment/optimizer, source/config/timeout/cap/remoteaction, partial-result
parse/hash, completedtest repeat or giantoriginal processing. Preparedpostcleanup
helpers remain UNEXECUTED. NEXT unchanged: terminal+release -> exactcleanup ->
bounded analysis/table/interpretation -> backup. Baselines PAUSED; allformal,
numericadapter and bothmodelperformance questions remain OPEN.

01:20:19 latest: SAME actual service/aux identities verified active. Livebanner
391arrived/361done/361ok/0fail,30backlog,4runtimes. Watchdogsample518 service
17444343808B,host95850770432B,swap/high/max/oom/oomkill0,no warning/abort,
disk162023829504B. No terminal launch.json. Monitorcell7453 completed six45s
polls normally and is CLOSED; this is NOT experiment completion. No liveexec
session/toolhandle; actualrun remains SAMEtmux/service. Do NOT relaunch.
This goalturn PROGRESS (maintenance complete and ordinaryFull launched), then
VERIFIED WAIT. Prepared postcleanup helpers remain unexecuted. No partial
results parsed/hashed, oldgiantfiles read, tests repeated, remote management,
configuration/deadline/cap changes or second optimizer. Goal ACTIVE/incomplete.

2026-10-02 01:15 +08. ONE ordinary Full launched01:11:34 in tmux
tc-d152-7b-full1, source26fa0a086bf91895e667a1996662971a526c860b backed.
Same4000/source42/W0/config/D89profiles, no profiler/prefix. D151 fresh native
routing observation is the only candidate; no source/config/remote changes
during inference. Baselines PAUSED; goal ACTIVE/incomplete. Turn PROGRESS.

- Service primelora-tc-svc-378dc55711df47d59f8699da609543bf.scope,
  actual0a626e9388db4e86954dc86b0493f5df,72/80GiBswap2 readback.
- Aux primelora-tc-aux-05f670e5caff478ab52afc329cb531d0.scope,
  actual5228f8f54fda45418d921a7ecfa0edb1,3/4GiBswap0 readback.
  Launcher88941/watchdog89175 verified in exactaux/CPU2,3,26,27.
- Nativecores91976/97136/97341/97684 verified by watchdog sample184 in
  exact service domain/CPU4-23,28-47. Sample184 current15654191104B,
  peak19414773760B,host97114370048B,swap/events0,no warning/abort,
  disk162598662144B. Latestbanner55arrived/51done/51ok/0fail. No terminal.
- Remote3b2610666/c6a527173f3f406193a6509819e24d01;
  7b2610668/6123604301b94a06b32719c81e1e8e32;
  monitor2610671/d081c6b2b366400f91918a9ea2d2a774,
  unitprimelora-artifact-monitor-d152full1.service. BothNIC1000/full.
  7Bclockremote-process-monotonic:92250332ccd74cc0b131ca97d18ffbcb.
  Remote log /home/lab14/primelora_remote/tc/d152_20261002/remote_monitor_7b_full_full1.log.
  ImmutableD78/D80 cache unchanged. No remote management/hash/cleanup duringrun.
- Preflight255reference entries/147protected/Plan/V1 PASS;
  SHAd8c811c440b525eac7d14183ed8527403db349ee05f0c8ea6e4f6d41b1fefcf3.
  Actualb4a13d90bd174d12a53cdb18263da210 and healthc24c207a7c0c4264a9ad63edc2ab7d38
  exited0/automatically removed, absence receipts01:11:18; no final event
  counters available after removal. Preflight3907 CLOSED0, health returned0.
- Reused D150 postcleanup metadata/projection/curation/failure/occupancy/copy
  helpers, changing freshpaths/runtime/actualunit/remoteclock; comparison uses
  sealed D15034MB projection+curated SHA, never its original for comparison.
  Prepared exact localaux/remote stop helpers, NOT executed. ShellsyntaxPASS.
  No postprocessing/partial-output parse/hash and no previous tests repeated.

NEXT monitor SAME tmux/service -> terminal/physical release -> exact remote
stop/copy and auxiliary cleanup -> bounded metadata/ONEprojection/curation/
table/interpretation -> backup. No second optimizer or GPU run during this one.
No formal SLO/numericadapter/performance gain yet. Bothmodelperformance,
warm/Resident/M1M2/A1-A5/S1-S13 and outputhash issues remain OPEN.

## D152 capacity and preparation — completed history

2026-10-02 01:10 +08. D151 serving candidate remains backed26fa0a086bf91895e667a1996662971a526c860b;
tests/qualification COMPLETE, do NOT repeat. D152 maintenance apply47955 CLOSED0,
12.78s/RSS1083656KiB. Exact actual4f7ad32afd194d70994a4aa6a53eecd9 automatically
removed; absence verified01:09:43 and cache_apply1_cleanup.txt saved. Audit47553
CLOSED0/actuala45fa937f0e94eabab6e071a0115b060 already absence-verified. No live
exec/session/GPU/remote artifact service. Final memory.events unavailable after
automatic removal, not fabricated. Raw d152_20261002.

Only10 public NVIDIA download archives+10 headers removed,2151706624 allocated
bytes; exact official PyPI SHA/size/HEAD and no open/project/environment refs
verified. Reviewed auditSHA664ae330c129daebbd4a951e8e4d92415376175be0c263bbdf9dbabbb3c44d53.
Four repositories' tracked states unchanged; installed environments, unique
models/LoRAs/traces/results untouched. D148 prior35-archive cleanup NOT repeated.
Fresh available163134590976B/MemAvailable110187456KiB; unchanged150GiB new-heavy
gate now feasible, actual preflight still required. No compression/cache rebuild.

D152 reused D150 Full config/launch/preflight/remote helpers, only candidate
identity and fresh owned paths differ. Ordinary4000/source42/W0, same profiles;
no profiler, diagnostic prefix, new trace/weights or configuration tuning.
All shell helpers syntax checked individually. Not yet launched. NEXT bounded
preflight -> remote activation/health -> exact scope cleanup -> ONE Full tmux.
Full Plan/status/V1 and run/monitor skills read; goal ACTIVE/PROGRESS. Baselines
PAUSED; both-model performance, numeric adapter/output hash, warm/Resident,
M1M2/A1-A5/S1-S13 remain OPEN. User cache permission already fulfilled D78/D80.

## D151 backed completion — completed history

01:00:40 +08 BACKUP COMPLETE26fa0a086bf91895e667a1996662971a526c860b;
retry2334 CLOSED0, remoteverify85251 CLOSED0/exactHEADmatched. InitialTLS
failure preserved below; no forcepush/credential changes. No liveexec/session,
GPU/remoteartifact/analysisjob. All tests/verification are COMPLETE, donotrepeat.
NEXT safeprovenrebuildablecapacity audit then ordinary7BFull with unchanged
config/profiles; no D152prepared/launched yet. Latestfree160984645632B,
MemAvailable110145304KiB; newheavy gate150GiBstillnotmet. No deletion orlarge
compression, immutablecache untouched. Thisnoteonlynewledgerdeltaafterbackup.
GoalACTIVE/PROGRESS; allformal/numeric/bothmodelperformanceissues remainOPEN.

01:00 +08 localcheckpoint26fa0a086bf91895e667a1996662971a526c860b COMPLETE;
12explicitfiles/55payload secrets/syntax/checksums/diffPASS, usermanifestexcluded.
Verify13382 CLOSED0 and actual5d8e17d9fbca458d97d899b22bde5045 absence verified;
cleanup_verify.txt saved. Firstpush79172 CLOSED128/GnuTLS(-110). Read-only
remotecheck49728 CLOSED0 shows old805bf611, so remote backup NOTyetcomplete.
One bounded retry LIVE execsession2334; poll SAMEsession, no duplicatepush.
No GPU/remoteartifact/analysisjob. D151tests/verification/closure mustNOTrepeat.

Capacityread-onlyfollowup: D148 retained52 pip downloads lack x-pypi headers;
zipcentralmetadata identifies many public nvidia wheels, plus CUDA-specific
torch/customarchives. This is identity discovery, NOT deletion authorization
proof. No bodyhash/PyPISHA/HEAD/openfile/ref audit performed fornewcandidates,
no deletion. Reuse D148 cleanup helper with explicit metadata-derived package
identity and exact officialSHA/size/HEAD verification; keeporiginalallowlist,
bound candidate list/newwrites. DoNOTrerun prior35-file cleanup; completed.
Installedenvs/uniqueartifacts/rawfailedresults stayprotected. NoD152 prepared.

2026-10-02 00:57 +08. Goal ACTIVE/incomplete, turn PROGRESS. ONE candidate:
fresh readonly native routing observation omits unused staging/allocator payload,
but preserves owner refresh/invariants, registered graph validation and all
physical/planning source_snapshot consumers. No cache/TTL, formula/config/cap/
timeout/lease/workload/backend/remote changes. Source parent805bf611. See sealed
D151_NATIVE_ROUTING_OBSERVATION.md. All tests/verification CLOSED; no live
GPU/remote/analysis/exec jobs. Ordinary Full NOT prepared/launched, no gain claim.

- RED5:2failures/2errors; firstGREEN one faulty cache-view mutation injection;
  correctedtestGREEN2 5PASS/.012s. Regression1 737/14fail/75errors and
  regression2 737/2fail/1error were missing operation mappings/injections in
  old CPU fixtures. Production candidate unchanged after first implementation.
  Final737PASS24.735s/command35.43s/RSS1211732KiB. All attempts retained.
- Actual scopes3/4GiBswap0CPU2,3,26,27, automatically removed/absence verified.
  Final cgroup events unavailable after removal, not invented. All handles
  76025/99956/66320/98941/28675/1397/13382 CLOSED. Verify actual
  5d8e17d9fbca458d97d899b22bde5045 exit0; cleanup receipt next.
- Verification233frozenrefs/147protected/Plan/V1/sourcevariants/syntax/secrets
  PASS. CuratedSHA787af8b2ae362f5de084fbd7632a1883d8931682acbdd44f5ddc9b7353e697da;
  44member441749B bundleSHA46e6712b6b9cfc093ce2a7490b131ce18f766e9cfd26582ff293d496e613ae9b.
  Rawd151_20261002. Verification's own log/time/cleanup local, not in prior bundle.

NEXT explicit12file diff/checksums/secrets -> commit/push; usermanifest excluded.
Then safe provenrebuildable capacity audit (free160986337280B <150GiB), no new
heavy launch before fresh safety gate. No deletion/largecompression this turn.
Do NOTrepeat tests/D150analysis/immutablecache creation. Bothmodelperformance,
numericadapter/103outputhash/warm/Resident/M1M2/A1-A5/S1-S13 remainOPEN;
baselinesPAUSED. FullPlan/status/V1 and applicable skills readFULL this turn.

## D150 backed completion — completed history

00:35:11 +08 BACKUP COMPLETE805bf611c4da5ca38575889d8b32eb6481b8d82e,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEAD verified.
12explicitfiles/77payload syntax/secrets/checksums/diffPASS; usermanifestexcluded.
Push62703/verify46831 CLOSED0. No liveGPU/remote/analysis/exec handles. Current
servingruntime remainsD1499c73e33; no sourcechange inD150closure. Thisnote is
theonlyledgerdelta aftercheckpoint. Do NOTrepeat D150closure/analysis/tests or
once-only deliverycache creation. GoalACTIVE/incomplete; turn PROGRESS.
NEXT source/history/primary-reference diagnosis ofremainingnon-nativewaiting;
do notblindlyrerun/tunecap/deadline. No nextoptimizer implemented/GPUprepared.
Beforefutureheavy, safeprovenrebuildablecapacityaudit required (lastfree~149.91GiB).
No deletion/largecompression yet. Bothmodelperformance/numericadapter/warm/
Resident/M1M2/A1-A5/S1-S13 remainOPEN; baselinesPAUSED. New103outputhashchanges
vsD148 mustremainvisible, notexplainedbyassertion. FinaltablesinD150doc sealed.

## D150 sealed closure history — completed

00:34 +08 verification38217 CLOSED0:233frozenrefs/44curatedrefs/147protected,
Plan/V1/Full/phases/contracts/cleanup PASS. SHA
9f8367a8cb426802ef39e9c8abd9590b5c870021dd0a04bdc8ad42e78e81308a.
Actualf7d39e71a2354258ab749c5c5f47075c auto-removed/absenceverified/logsaved.
Tiny66member54506B evidencebundle produced andmemberSHAre-readPASS,
SHAbff2cba36ce53305f0634f2da36c78a069d13f7efd454a85da8aea52b7d19799;
actual5864f6f231004181b5324eb316d1e6ea auto-removed/absenceverified00:33:57.
No largecompression/deletion/duplicate dataset; onlyboundedbackupbundle.
D150_FULL_W0_FULL1.md nowsealed, doNOTeditwithoutnewseal. No livehandles/jobs.
Freshdisk160962781184B (~149.91GiB), stillnonewheavy; smallanalysiscompleted.
NEXT explicit12file syntax/secrets/checksum/diff -> commit/push, then remaining
non-inference waiting diagnosis with existingevidence. Do NOTrepeatD150tests,
projection/analysis/seal/cachecreation. GoalACTIVE/PROGRESS, baselinesPAUSED.

00:31 +08 allanalyses COMPLETE, no liveGPU/remote/exec/session. Projection94650
closed0/79.69s/RSS100224KiB,34390795B; actuala6e1f9e15d58448ea1d8f6d1e3c70a1f
auto-removed/absenceverified. Curation4836 closed0/7.22s/RSS416112KiB,
actualff8b1f86b3104b979f6bcc362626a19e auto-removed/absenceverified; failure0.
Occupancy39507 closed0/3.14s/RSS408620KiB; actualc285f200b424480fb295354ad0d25268
auto-removed/absenceverified, events0 printedatend in tooltranscript.
Do NOT rerun analysis or reproject/hash oldgiantfiles. Earlier LIVE below ishistory.
CuratedSHAee01b56e9f44d8eced77528e99b488e3efb1515d7edba0d0fd9e5a3c729f72d2.
4000native/0fail,timingerrors0,confirmed4000/conflicts0;remote132/131480060B.
Mean/P95TTFT296.677792/553.254599s,GPU17958.297896,TPOT41.273491/69.890503ms.
VsD148meanTTFT-49.874%,P95-51.511%,GPU-12.483%;nativeTTFT+15.589% retained.
Inputs/countsallmatch;103outputhashdifferencesOPEN. No numeric/SLO/G1G2claim.
Occupancy native4.867887s,gate→terminal8.732492s upperenvelope (notpermitrelease),
meanconcurrency4.345730/7.795796; servicewait still294.697270s,99.332%TTFT.
Controls1986allinwindow/0outside;planning1829/1828complete/1cancelled_result_discarded.
DocD150_FULL_W0_FULL1.md finaldiagnostictables/interpretation prepared, UNSEALED.
NEXT evidenceverification/smallbundle -> explicitbackup. No servingchange,
newoptimizer orGPU prepared. FullPlan/status/V1 readFULL; four skills read.
GoalACTIVE/PROGRESS; allformal and3Bissues OPEN; baselinesPAUSED.

## D150 terminal and processing history — superseded by CURRENT

2026-10-02 00:20 +08 terminal evidence: launch PASS, service/replay/watchdog
returncodes0; native release and service-path removal confirmed. 4000 native
contract completions/0 failures, physical17958.297896314005GPU-s,4 allocations
all released/0open. Numeric adapter and commonSLO qualification remain OPEN.
Local service AND aux automatically inactive/removed; actualInvocation empty,
TasksCurrent notset, aux PIDs3701037/3701314 and cgroup path absent at00:20:27.
Do NOT execute prepared manual auxcleanup (its retained-ID prerequisite no
longer applies). No liveexec/session/monitor;7305 and earliercells CLOSED.
Normal source272774929B/main_outcome169417509B, not yet analyzed or hashed.
Metadata completed0/3.65s/RSS391908KiB; actualfd0027a85a9e41f3a4e3fd8c8db4e1e2
automatically removed, absence verified bycleanup_metadata_full1.sh/log.
Final analysis cgroup counters unavailable after automatic removal, not invented.
Compact112703435B under unchanged128MiBguard. Preliminary4547samples:
peak20423520256B/minhost94001831936B/service swap/events0/no warnings;
remote132/131480060B bothends/allpublished/packing0;1initial+3natural/0quarantine.
ONE projection LIVE session94650, unitprimelora-d150-full1-project-20261001.scope,
actuala6e1f9e15d58448ea1d8f6d1e3c70a1f. Actual3/4GiBswap0CPU2,3,26,27;
sameD96streamer, onlythis272.8MBoriginal. No partialoutputread/hash.
Curation/occupancy helpers remainprepared NOTexecuted. ProvisionalD150status
table saved; no comparative latency or gain yet.
Remote exact-owned stop COMPLETE00:25:45: all3 units inactive/MainPID0/success.
Session88414 closed0. Matching journal transfers-6244054724e5427ba64434ae54892b12.jsonl
copied after stop; source/copySHA d82dae7a136ccbacd13ece31c6d5889e834e3f27388f28e9c2091ec77a2a5319;
monitor f4b28409adf078d7b895f37d15fb725506c47e0f502fedf49a1ccd8491a9d33a.
Copy session32080 closed0. Remote has no rg; fallback grep located frozenclock.
Aftermetadata disk160999014400B slightlybelow150GiB. No newheavyGPU/build/
compression permitted; smallboundedCPUanalysis (<256MiB outputexpected) only,
100GiBstop unchanged. No deletion/compression; futureheavy requires newcapacitycheck.
NEXT remote stop/copy -> bounded metadata -> ONEprojection -> curation and
occupancy -> diagnostic table/interpretation -> protected/secrets/backup.
Do not reparse D137/D145/D148 giant originals or repeat candidate tests/cache
publication. Goal ACTIVE; turn PROGRESS (ordinary Full complete). Baselines
PAUSED; bothmodelperformance/warm/Resident/M1M2/A1-A5/S1-S13 remainOPEN.

## D150 launch and monitoring history — superseded by CURRENT

2026-10-01 23:07 +08. ONE ordinary Full launched23:02:08 in tmux
tc-d150-7b-full1; source9c73e33ad4c68d27d881f47ac81af8d579ec0f3b backed.
Same4000/source42/W0/config/D89profiles; no profiler/prefix. D149 fresh validated
source projection is the sole new runtime candidate. No serving/config/timeout/
cap change during this run. Baselines PAUSED; goal ACTIVE/incomplete.

- Service primelora-tc-svc-86bc8a80fc0a444ca42ec035ce5ea83b.scope,
  actuald34e597e8fac446fa486de007b4eefac,72/80GiBswap2 read back.
- Aux primelora-tc-aux-5470a89aabd847c49a40b7b5c427b3b0.scope,
  actual13efd326658647e38e9832a3ab55515e,3/4GiBswap0 read back.
  Replay3701037,startticks37417601 inauxCPU2,3,26,27.
- Nativecores3704297/3709703/3709806/3709916 verified by watchdog sample282
  in EXACT service domain and CPU4-23,28-47; all4 physicalGPUs held.
  Sample282 service16987086848B,peak18867638272B,host96245424128B,
  swap0/high/max/oom/oomkill0,no warning/abort; disk160922296320B.
- Remote3B2429650/04d60463d9a34e31ba533689e5cf91c7;
  7B2429652/c7f02aae303a4d24b0f39d09b2a34150;
  monitor2429655/64859df2a13c4685be679d504e8a0015,
  primelora-artifact-monitor-d150full1.service. BothNIC1000/full atprelaunch.
  7Bclock remote-process-monotonic:cc2dbc89e23543d8a7e37305d21d21f0.
  Remote log /home/lab14/primelora_remote/tc/d150_20261001/remote_monitor_7b_full_full1.log.
  ImmutableD78/D80 cache unchanged; no management/hash/cleanup during inference.
- Preflight233reference-path entries/147protected/Plan/V1 PASS,
  SHA601597db10aeb812999dd0732e8c7350fd1f542a76d610fcfe5a91a75afc4df9.
  Actual1c15f97e7263425ca8925400457068fe and health839477bd151d4413ab81ba6cde7aa3e6
  completed/empty/events0/stopped. Localprelaunch161787949056B passed150GiB
  gate withpredicted32GiBgrowth; runningstop100GiB unchanged.
- Raw results/ieee_tc/p2_backend_qualification/d150_20261001.
  launch.json notyetpresent. No exec cell/session; actualrun is tmux/systemd.
  Exact owned localaux and remote stop helpers prepared, NOTexecuted.

NEXT monitor SAME actualinvocation -> true terminal/physical release -> exact
owned remote stop/copy and auxcleanup -> bounded metadata/projection/curation/
table/interpretation -> backup. No analysis/hash ofpartialresults; never reparse
D137/D145/D148 giant originals, repeatqualification or rebuildremote cache.
Bothmodelperformance/numericadapter/warm/Resident/M1M2/A1-A5/S1-S13 remainOPEN.
Latestgoalturn VERIFIED WAIT with ownership checks; launchturn PROGRESS.

23:10 continuation: fullPlan/status/V1 reread aftercompaction; run-experiment,
monitor-experiment,analyze-results,academic-plotting skills readFULL.
Prepared eight postcleanup helpers by reusing D148 (metadata/projection/
curation/failure/occupancy/stopped-remote-copy), changing only D150 paths,
actual service identity, remoteclock and backedruntime. Curator now compares
to SEALED D148 projection/curated SHA, not D137's multiple-change endpoint;
n=1 still cannot establish statistical superiority. ShellsyntaxPASS. NOTRUN.
No oldgiantJSON read/hash or partial result analysis, no newservingchange.
Watchdog3701314 confirmed alongside replay3701037 in exactauxCPU2,3,26,27.
Remote journal filename remainsunknown; discover by frozenclock ONLYafterstop.
No live toolhandles; SAME tmux/service/aux remainslive. Currentturn evidence:
resourceownership and postcleanup preparation; no performanceclosure claim.

23:15:23 latest: SAME service actuald34e597e8fac446fa486de007b4eefac active533tasks.
Livebanner arrived576/done552/ok552/fail0/backlog24,4runtimes; no terminal
launch.json. Sample783 service17645625344B,host95732461568B,swap0,
high/max/oom/oomkill0,no warning/abort,disk160552693760B.
Monitorcell7283 finished its six45s polls normally and is CLOSED; this is NOT
the serving run's completion. No exec session/live toolhandle remains.
SAME tmux/service/aux continue; never restart. No source/config/remotechanges,
old giant reparse, cache rebuild or partial-result analysis. Currentgoalturn
VERIFIED WAIT; priorgoalturn launch PROGRESS. GoalACTIVE, notblocked.
NEXT unchanged: monitor exactrun -> true terminal+release -> exactcleanup ->
prepared bounded analysis/table/interpretation -> backup. Baselinespaused.

2026-10-02 00:15:50 continuation VERIFIED WAIT: SAME service/aux IDs confirmed,
allactive; monitorcell7303 completed its20polls and CLOSED normally (7301/7297/7292 also
CLOSED). This is NOT underlyingruncompletion. No liveexecsession/toolhandle;
servingis SAMEtmux. Livebanner4000arrived/3894done/3894ok/0fail,4runtimes,
106backlog. Allarrived since00:09:48; draincontinues. Sample4362
service20332969984B,host95269961728B,swap/events0,no warning/abort,
disk158337064960B. No terminalreceipt. Plan/V1SHAunchanged,
HEAD9c73e33unchanged; same monitoring task, no newexperiment/optimization.
FullPlan/status were read preceding continuation; currentliveledger/monitor
skill reread and actualstate checked. Previousgoalturn VERIFIED WAIT;
currentalso VERIFIED WAIT, notblocked. Preparedpostcleanuphelpers remain
UNEXECUTED; no new source/config/remoteaction, no partialresult analysis.

## D149 QUALIFIED AND BACKED — completed history

22:57:45 +08 BACKUP COMPLETE9c73e33ad4c68d27d881f47ac81af8d579ec0f3b,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
Sevenexplicitfiles/30payload syntax/checksums/secrets/diffPASS, usermanifest
excluded. First precommit scanner matched its own public private-key-marker
test expression; corrected scanner to actual anchored PEM-header syntax,
no secret printed or source/evidence changed. Push34154/verify61531 CLOSED0.
All CPU/GPU/remotejobs and toolhandles CLOSED; no nextGPU prepared/launched.
Do NOT repeat candidate qualification, D148 closure or once-only cachecreation.
NEXT reuse ordinaryFull helpers, same config/profiles and fresh safety check.
Freshfree161788661760B/MemAvailable110736028KiB; no newdeletion/compression.
This backupnote is the onlynew ledgerdelta after checkpoint. Goal ACTIVE,
turn PROGRESS; bothmodelperformance and allformal questions remain OPEN.

2026-10-01 22:56 +08. ONE source candidate: the two selected-source rechecks
now use the existing freshly/full-validated frontend projection. No source
cache/TTL, native ownership, physical admission/planner, equations, caps,
timeouts, profiles, workload, backend or immutable delivery changes. See
D149_SELECTED_SOURCE_PROJECTION.md (sealed). Baselines PAUSED, goal ACTIVE.
No GPU/remote task launched; all CPU tests and verification are closed.

History/source audit used D148 phases, D143 samples, D132 boundaries and primary
vLLM/asyncio references. D148 still NOT accepted Full improvement. D149 hypothesis
is reduced centralized message/validation overhead; no whole-system gain yet.
Do NOT repeat D132 old component probe, D148 analysis, or remote publication.

- RED two tests:3failures/1error (old direct graph boundary plus incorrect test
  expectation of returned failure); original test patch/log preserved. Corrected
  expectation to existing ValueError, not production error semantics.
  Actual a4f61695257046018467c3ce96c73c13 empty/events0 stopped22:51:03.
- GREEN2PASS/.098s, actual eb758d8eaa2d4cc9beb3803d6c1f71be empty/events0
  stopped22:52:43. Four source tiers retain source/lease behavior; bad graph
  rejected after HOST hold with cleanup. Fixture adds real producer UUID field.
- Affected regression700PASS25.238s/command36.70s/RSS1210360KiB;
  actual6db71f8dc93e48c2842b689cf66499de empty/events0 and stopped.
  Includes actual dedicated RPC, equal projected states, ownership/cancellation,
  native retirement, storage graph, routing and basic smoke. Do NOT repeat.
- Verification225frozenrefs/147protected/Plan/V1/source/test/secrets/syntax PASS.
  actual9da8f7f8df2243eebd22ec1a0d27d73f empty/events0 stopped; all actual
  scopes3/4GiBswap0CPU2,3,26,27. Sessions41241/59973/10797/52723 CLOSED.
- CuratedSHA8a3025bbb0cc5ece1d576d9897f9d9b4171964f9a611f4ed77e65b91469c2700;
  bundle23members310528B SHAc7531cc7bf37d99a7c19baf9693a8709958b142b3228f9efdaa2b80b5b0e5a3d.
  Verification's own log/time/cleanup remain local post-bundle, not fabricated.
  Raw results/ieee_tc/p2_backend_qualification/d149_20261001; new doc state table.

NEXT explicit7file diff/checksum/secrets -> commit/push; then reuse D148 ordinary
Full helpers with current frozen config/profiles/4000/no profiler, fresh safety
checks and owned output paths. No nextGPU prepared/launched. 1107B outputhash
changes,3BTPOT/old gap,numericadapter,warm/Resident and allformal work remainOPEN.
User manifest and unrelated files untouched/neverstage. No disk deletion or
compression except small evidence bundle; last freshfree161800941568B needs
recheck before newheavy. Current goalturn PROGRESS (candidate qualified).

## D148 COMPLETE AND BACKED — completed history

22:41:45 +08 BACKUP COMPLETE09989a0e7592491e66369c027d971fb5fed4ad45,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
12explicitfiles/85payload secrets+syntax+checksums+diffPASS; usermanifestexcluded.
Push27269/verify63775 CLOSED0; allownedGPU/analysis/remotejobs stopped, no live
handles. No servingedit since1eaf15a4. Do NOT repeat D148 analysis/sealing/
tests/projection or once-onlyremote publication. Thisbackupnote is theonlynew
ledgerdelta aftercheckpoint. Lastdisk161804890112B/hostavailable110689460KiB;
freshcheckbeforefutureheavywork. No newdeletion/compression since documented
publicwheelcache cleanup; uniqueoriginals protected.

NEXT majorremainingnon-nativewaiting diagnosis using existing D148 phases,
D143 historical profiles and native source consumers; selectONE evidence-backed
candidate after source/primaryreference verification, minimalqualification then
ordinaryFull. No nextGPUprepared/launched, no newcandidateimplemented. 1107B
outputhashchanges remainunresolved;3BTPOT andoldperformancegapalsoOPEN. Do not
close performancegoal or resumebaselines on zero failures alone. GoalACTIVE;
thisturnPROGRESS (fullphaseanalysis/table/seal and remoteverifiedbackup).

## D148 closure history — completed

2026-10-01 22:40 +08. No liveexec/session/analysis/GPU/remotejob. D148 ordinary
Full4000native/0fail, allcleanup; projection65337/monitors7135,7175 CLOSED,
curation18032 CLOSED0 (50.37s/RSS395052KiB), failurecollector0/.53s;
actual56c5ac2607cf403c9e142ee6eac3e5f1 empty/events0 stopped22:33:14.
Occupancy13635 CLOSED0; actuald13fd7de43374752a77243f587fde700 empty/events0
stopped22:33:48, cleanup_occupancy_full1.log retained. Do NOT repeat anyanalysis.

CuratedSHAa0f8d157150b73fb4d36fa2e6d1a9549da05a0e4ee8de50577fa919d4706d535.
TTFTmean/P95591.861802/1140.989671s; TPOT42.494351/71.297526ms.
Dispatch589.561252s=99.6113%meanTTFT, service2.300550/native.353400s.
20519.801810GPU-s; all4000 timingerrors0,132remote/131480060B/packing0.
RelativeD137:meanTTFT-1.952%,P95-.030%,GPU+.985%,TPOTmean+8.918%/P95+13.376%.
NOTanacceptedFullperformancegain; n1/multiplesourcedeltas/noCI. D1453996/4
conditionalpopulation preserved. D137inputs/countsallmatch,110outputhashchanges
OPEN; numericaladapter/warm/Resident/formalqualificationstillOPEN.
Occupancy native4.879046s vs gate→terminal9.998352s upperenvelope, NOT exactpermit
release. Meanconcurrency3.814354/7.816538; sampledheldGPUutil40.9638%, notcausal.
Controls2015/2014inwindow/1after retained. Planning2047receipts:2046complete,
1cancelled_result_discarded withCancelledError inendingreceipt; no requestfailure.

Verification6116 CLOSED0/PASS225frozenrefs/44curatedrefs/147protected, original
15.29GB hashreusedwithstat (neverreparse). Actual2af0278972fe4f45baf014ffff427af4
empty/events0 stopped22:38:24. VerificationSHA
bf94e667b3b100ebc985a5b3347b677a2aec7dc9eb0170484a87b7e3e4c7acb8.
Bundle73members83655B SHAd06b202016e83a4ad9fdae7b17770668f853fc70878acb3fc185d8ed6c13df5f,
membersre-readPASS; actual9182c20663e54caea8dcf14d74ad8513 empty/events0 cleaned.
Firstcleanupwrapper usedwrongcwd andfailed beforeopeninglog/executingcleanup;
correctedrootcommand ranONCE; no repeatedexperiment or bundle. Cleanupownfinal
logs arelocal,notreconstructed. D148_FULL_W0_FULL1.md nowfinaldiagnostic tables/
interpretation,SHApinned; don'teditafterseal. Sourceaudit/maintenance evidence
included asboundedexplicit bundlemembers. No servingchange/newcandidate.

NEXT explicit12file secrets/checksum/diff review -> commit/push; usermanifest/
unrelateduntrackedneverstage. Then majorremainingnon-nativewaiting diagnosis,
notanotherdevicequerymicrotest orblindcap/deadline increase. BaselinesPAUSED;
3BperformanceandallformalM1M2/A1-A5/S1-S13 remainOPEN. GoalACTIVE; thisturn
PROGRESS (finishedcuration,phaseanalysis,finaltables,sealedclosure).

## D148 curation history — superseded by CURRENT

2026-10-01 22:32 +08. Projection65337 EXIT0, monitor7175 CLOSED (7135alsoCLOSED).
44:46.59/RSS99456KiB,34391135B output. Exactb950242006254cc986cfc50f9ff71342
empty/events0/stopped22:31:58; cleanupstdout in tooltranscript, notinventedlog.
Do NOT wait/restart thosehandles or reproject/re-hash original foranalysis.
ONEcuration LIVE execsession18032, primelora-d148-full1-curate-20261001.scope,
actual56c5ac2607cf403c9e142ee6eac3e5f1. Actual3/4GiBswap0 CPU2,3,26,27,
MemoryCurrent393990144B/6tasks lastchecked. This also runsfailurecollector.
Prepared cleanup_curation_full1.sh boundtoactual identity, NOTexecuted.
Freshdisk161737080832B/hostavailable110885604KiB; nonewGPU/remoteoperation.
Next truecurationexit -> exactemptycleanup -> preparedoccupancy -> finaltable/
interpretation/seal/backup. Goalturn PROGRESS (projectioncomplete, curationlaunched).
FullPlan/V1 remainunchanged; allformal/3B/numericqualification issuesOPEN.

## D148 postprocessing history — superseded by CURRENT

2026-10-01 21:48 +08. Full native4000/0fail and exactcleanup done. Metadata and
remote pairing COMPLETE (below). ONE streaming projection launched21:46:54,
exec_command session65337; unitprimelora-d148-full1-project-20261001.scope,
actualInvocationb950242006254cc986cfc50f9ff71342. Actual3/4GiBswap0,
CPU2,3,26,27;2tasks/events0 lastchecked. NoGPU/remote/cachemaintenance live.
Monitor SAME session; never restart or parse/hash partial output.
Latest21:51:26 SAME actualInvocation verifiedactive2tasks/12214272B;
jqPID3383655 sourceoffset1589997568B (was607166464B), steadily advancing.
Monitorcell7132 completed/closed; actualprojection session65337 STILL LIVE.
Currentgoalturn VERIFIED WAIT (plus closure-protocol applicability inspection),
previousgoalturn PROGRESS. No duplicateprojection/remoteoperation/newGPUrun.
Original15289629214B is read ONCE by unchanged D96 streaming projector. Prior same-size
projection took~44min; zero output during reduction is not proof of a stall.
Latest21:53:32 SAME invocation remainsactive2tasks/17145856B/events0.
Monitorcell7135 LIVE, polling SAME session65337 every45s, max40polls, yielding
each poll. Resume with functions.wait(7135); do NOT start another monitor or
projection. A monitorcycle limit/observation timeout is NOT terminal evidence.
22:00:18 latest SAME projectionactive2tasks/31145984B/events0;7135 stillLIVE.
Currentturn also gained a bounded read-only source finding, preserved in raw
postcleanup_readonly_source_audit1.md: request compact evidence drops only
native_footprints, not native_staging_footprints; D132 projection presently
routing-only. Existing D143 source_snapshot44 samples belong to residency_manager;
that module has TWO same-name definitions, so caller/line evidence is needed
before classifying all44 as file or native. Not all can be called GPU queries.
No newaggregation/GPUprobe/servingedit/
optimizerselection; waitforD148phases. Do not infer onlinecost from savedJSONsize.
Next: projection completion -> exactempty cleanup -> prepared curation/failure
and occupancy -> finaltable/interpretation/seal/backup. No new optimizer yet.

22:13:31 continued VERIFIED WAIT on SAME actualprojection b950242006254cc986cfc50f9ff71342:
active2tasks/61423616B; jq sourceoffset9147240448B of15289629214B (was7620227072B
at22:09:01), steadily advancing. Monitorcell7135 stillLIVE, SAME session65337;
do NOT duplicate monitor/projector or parse/hash partialoutput. Cyclelimit is
not terminal. Disk161789595648B,hostavailable110587028KiB; Plan/V1SHAunchanged.
NoGPU/remoteoperation/maintenance or servingchange. FullPlan/status/V1 reread;
monitor/vLLM+optimization/analyze-results/academic-plotting/github-sync skills
readfull. Sourceaudit now records primarytaggedvLLM worker/model_manager/
lora_weights links and mutation-boundary caveats; no cross-callinventorycache
approved, no candidate selected. AllcompleteD147tests,metadata/copy/cachecleanup
remaincomplete; do NOTrepeat. ProvisionalD148doc/sourceaudit unsealed/unbacked,
servingstillbacked1eaf15a4. GoalACTIVE; wait -> curation/table/backup remainsNEXT.

22:19:00 nextgoalturn VERIFIED WAIT: exactsameprojection Invocationactive2tasks,
72867840B,high/max/oom/oomkill0; sourceoffset11010625536B, progressing.
Monitor7135 continues polling65337; no terminaloutput yet. Previousgoalturn
also VERIFIED WAIT with read-onlyprimarysourcefinding. No blocker: actual
streamingjob islive. No newexperiment/optimizer/replay/analysis selected;
no partialoutputread/hash, no oldgiantfiles processed. Preparedcurator,
failurecollector andempty-domaincleanup reread; no changes/noexecution.
Next unchanged: trueprojectionexit -> exactcleanup -> curation -> occupancy ->
finaltable/interpretation/backup. Bothmodelperformance/formal goals remainOPEN.

22:25:59 monitor7135 CLOSED normally atcyclelimit; actualprojector65337 remains
LIVE, exactInvocationb950242006254cc986cfc50f9ff71342 active2tasks/87977984B,
sourceoffset13369278464B. ONEreplacementmonitorcell7175 nowLIVE pollingSAME65337
every45s,20pollmax; each observation uses set-e and actualInvocationcheck.
Resume7175 with functions.wait; never wait7135 or duplicateprojector/monitor.
Currentgoalturn VERIFIED WAIT, not blocked; no newserving/GPU/remote actions.

Maintenance COMPLETED (do NOT repeat): audit87641/actualf503c2466aa54f5abab696a5020d375a
closed0; exactempty stopped21:45:24,high11754/max0/OOM0 (filecache reclamation,
maintenance only). Reviewed auditSHA3b7ebafd2420aa73f5575eafbac26ad87a1b4e9e098b5651d7c576314356cd5b.
Apply90031/actual36546d70ce7f42d28f43ee85ff3ad8b9 closed0,empty/events0 stopped21:46:54.
Only35 publicdownload archives+35headers removed,14663053312allocatedB;
officialPyPI SHA and HEAD proof, no open refs/environment symlinks/direct project
configrefs; fourrepos trackedstates identical. Environments/JIT/model/artifacts/
rawresults untouched. CompletionSHA537784465d396ce18eae77fec0c8cabe3e07a1693a09770b1e2e38993ff4d339.
Preprojection free161806897152B/host113311686656B,256MiB predictedoutputgrowth,
Plan150GiBstartgate PASS. No compression or originaldata deletion.
D148_FULL_W0_FULL1.md has provisionalstatus table and maintenance disclosure.
Plan/V1 SHAunchanged; bothmodels/numericadapter/warm/Resident/formal allOPEN;
baselinesPAUSED. GoalACTIVE; thisturnPROGRESS: metadata/remoteverification,
safe recovery of capacity and once-onlyprojection launched. Servingbacked1eaf15a4.

## D148 terminal/metadata history — superseded by CURRENT

2026-10-01 21:34 +08. SAME ordinary Full finished normally; do NOT relaunch.
launch.json pass=true, service/replay/watchdog returncodes0, native GPU release
and service path removal confirmed. All4000 native-contract completions,0fail;
numeric adapter/commonSLO qualification remains OPEN (n_correct=null).
Physical summary measurement_complete=true:20519.80180997704GPU-s,
4allocations/4released/0open. Pre-arrival37.74262093100697,
arrival15781.683877044823,drain4610.408991132164,
cleanup89.9663208690472GPU-s. No detailed latency claim yet.
Normal source15289629214B; main_outcome156879054B; terminals2448936B.
Do NOT whole-load giant source; existing bounded metadata/streaming projection
helpers prepared only. D137/D145 giant originals must not be reprocessed.
Local service inactive/GPUempty21:32:19; exact aux empty/events0/stopped21:32:21.
Remote3B/7B/monitor exact identities stopped, allinactive/MainPID0/success21:32:20.
No live exec cell/session/GPU/remote/analysis job. Historical LIVE below superseded.

Remote journal identified by frozen health clock:
/home/lab14/primelora_remote/tc/d80_20260927/7b/transfers-7127877529df4979ab820945e2084aa4.jsonl.
Next copy stopped-service evidence with existing helper; then proven-owned
rebuildable-cache capacity audit (current free147358121984B/~137.24GiB,
MemAvailable110671612KiB). No deletion/compression performed. Do not launch new
heavy work below150GiB; safe light audit/copy allowed. Preserve unique raw data.
Then metadata -> ONE projection -> curation/occupancy -> table/interpretation
-> scoped backup. Runtime remains backed1eaf15a4; no new optimizer selected.
Plan/status/V1 reread FULL after compaction; analyze-results/academic-plotting
skills read. Goal ACTIVE; this turn PROGRESS (normal terminal and exactcleanup).
BaselinesPAUSED; bothmodelperformance/3B/numeric/warm/Resident/formal allOPEN.

21:44 postcleanup progress: remote evidence copied and source/copy SHA matched.
Journal8695bb610be0de46c0868c028592c5af4a0f59ab5edf71481131130165b441c1;
monitor eb17ec98addc0b9a84e9352eb48b4fa8f971ab84d8b2ce818af1758938a486f5.
Metadata actual5a33144d8c984a1ea5854865ead8883e completed0/3.55s/RSS364260KiB,
exactempty/events0 stopped21:39:17. Compact104450418B fits unchanged128MiBguard.
PreliminarySHAa44eb56573a3d75d0c71f2c2c3be3f84a607c3963d4f1fee218e21e40e02e96c:
4000success/0fail,132matchedpublished transfers131480060B/packing0;
6039resourcesamples,peak54801080320B,minhost73518608384B,swap/events0/no warnings;
1initial+3naturalscaleout,0quarantine. D148 provisionalstatus table created.
No latency/SLO claim yet. Only LIVE CPU maintenance is bounded public-download
cache audit (session87641), actualf503c2466aa54f5abab696a5020d375a,
primelora-d148-cache-audit1-20261001.scope,3/4GiBswap0/CPU2,3,26,27.
Read-only candidate verification against official public wheelSHA/HEAD;
NOdeletion yet, no environment/uniqueartifact/result modifications. This audit
has memory.high reclamation events from filecache, not OOM; it is NOT a measured
inference experiment. Waitforaudit, reviewexplicitallowlist/SHA, then scoped
maintenanceapply if valid. Existing finished run cache dirs mostlyempty; failed
D133/D115 remainprotected, not deleted. No giantprojection started yet.

## D148 launch/save history — superseded by CURRENT

2026-10-01 19:49:05 +08. Launched ONCE in tmux tc-d148-7b-full1;
ordinary 4000/source42/W0, backed runtime1eaf15a4. No profiler/prefix.
Preflight PASS225 reference-path entries/147 protected/Plan/V1, SHA
454b8d102e011cd91baeae4eb5a27f019fa1cc0de042bc32270c355902031c59.
Actual preflight815784d3b1de48f29b5d46b58975c116 and health
1d8e7218e5194d7bb391bf59f73fdc37 both empty/events0/stopped; handles closed.
Fresh disk162934022144B/host113522024448B passed planned32GiB growth gate.
No deletions/compression or new cache publication.

- Service primelora-tc-svc-a5ca47c39263418daa5f50c917161233.scope,
  actual40ca6d259e6c469ea6e8d0cfa3826723; limits72/80GiB,swap2 read back.
- Aux primelora-tc-aux-8af03bdfb52a4e19ae9ce74839358f58.scope,
  actual92607fdb22fd404b82a5ba55c99a8403; limits3/4GiB,swap0 read back.
  Replay2849543/watchdog2849558 are in aux CPU2,3,26,27.
  Verify spawned native workers in service CPU4-23,28-47 as they appear.
- Remote3b2185271/75347dc8fc124e1aab086524fd1c05ee;
  7b2185273/49916ec17998419388a5048948c875a6;
  monitor2185276/9caeedcc963d4fdc90aa93cf7a8b7485,
  primelora-artifact-monitor-d148full1.service. Both NIC1000/full.
  7B healthclock remote-process-monotonic:c42c17f372ff4e81a9a35242467fcb4a.
  Remote log /home/lab14/primelora_remote/tc/d148_20261001/remote_monitor_7b_full_full1.log.
  Exact stop/cleanup helpers prepared, NOT run. No remote management/hash/
  cleanup during inference. Immutable D78/D80 cache unchanged.

NEXT monitor SAME invocation -> terminal/physical release -> exact remote stop
and evidence copy -> bounded analysis/table/interpretation -> backup. No
second GPU run, new optimizer or configuration/deadline/cap change during it.
No terminal result/performance gain yet. Formal/numeric/3B/warm/reference
items below all remain OPEN. This turn PROGRESS: ordinary Full launched.

21:26:55 latest: SAME run still LIVE saving/cleanup, exact Invocation verified,
Tasks125. All4000 arrived/done/ok,fail0/backlog0 in final live banner.
No terminal launch.json yet; never relaunch or postprocess incomplete outputs.
GPU compute list empty at21:17:04 but terminal release receipt still pending.
main_outcome.json156879054B exists, NOT parsed/hashed; request_terminals2448936B.
Normal source JSON is WRITING:10344944069B at21:26:55, steadily growing;
do NOT read/hash/project until terminal and source stable.
Legacy console table is NOT frozen V1 performance/cost/SLO qualification.
Sample5793:service49707405312B,hostavailable74224029696B,swap0/events0,
no warning/abort. Latest watchdog disk152371572736B;
running stop threshold100GiB unchanged (150GiB is new
heavy-launch threshold, not a reason to abort an otherwise healthy live run).
Monitorcells7077/7080/7083/7086/7088/7090/7092/7095 completed and closed;
no live tool handles. Actual serving run remains live as above.
Actual experiment remains tmux/service/aux above, not an exec session.
Prior and current goalturns VERIFIED WAIT on live actual service; launchturn
was PROGRESS. Goal ACTIVE, no blocker/completion claim. Plan/V1 SHA unchanged.
No remote management, hash/copy, analysis, new optimization or second run.
Four native GPU worker PIDs2852708/2857503/
2857803/2858012 verified in exact service domain and CPU4-23,28-47 at19:51.
Sample172:service15343702016B,hostavailable98156994560B,swap0,
high/max/oom/oom_kill0,no warning/abort. No terminal result yet.
Prepared (not executed) D145 metadata/projection/curation/failure/occupancy/
stopped-remote-copy helpers by reusing existing scripts, changing only D148
paths, actual service identity, remote clock and backed runtime. All shell
syntax PASS. Failure context is already retained by unchanged D96 projection
and existing failure-observation collector; no new projection or serving edit.
D137 sealed34MB projection remains the reusable input audit, not its15GB raw.
D145 comparison will retain its3996/4 population difference, not fabricate
complete paired output identity. No data analyzed/old giant files reread.
Do NOT run postprocessing or remote stop/copy before actual terminal cleanup.

## D148 preparation — superseded by LIVE above

2026-10-01. D147 backup remains 1eaf15a4eb82abd847b6b8cd08eff97b06b6ef24.
Prepared D148 by reusing D145 launcher/config/preflight/remote helpers under
results/ieee_tc/p2_backend_qualification/d148_20261001. Nothing launched yet.
Same 4000/source42/W0, config/profile/generation/real immutable delivery;
only three fresh owned output/cache paths and backed runtime identity differ.
No profiler/prefix, no new weights/traces/cache publication. D146 observation
and D147 query implementation are the declared source deltas. This remains
development validation, not formal SLO/ranking or an isolated causal estimate.
Plan/status/V1 and run-experiment/monitor-experiment skills read in full.
Next: bounded preflight, remote activation/health, exact empty-domain cleanup,
then ONE ordinary 7B Full. Check actual resource limits and worker ownership.
All previous performance/numeric-adapter/warm/reference/formal items OPEN;
baselines PAUSED. Do not repeat D147 tests or completed microtests.

## D147 qualified/backed — completed history

19:41 +08 BACKUP COMPLETE1eaf15a4eb82abd847b6b8cd08eff97b06b6ef24,
pushedfaaslora_origin/retry14_continuous_queue_v2; exactremoteHEADverified.
Sevenexplicitfiles/56payload secrets/checksum/diffPASS; usermanifestexcluded.
All test/microtest/verification handles CLOSED; no liveGPU/remote/CPUjob.
NEXT ordinaryFull preparation; do not repeat qualification. Recheck capacity
using Plan formula, not an extra invented gate: with prior predictedgrowth32GiB,
required=max(150,100+1.5*32)=150GiB. Current~151.7GiB is above that threshold;
earlier 'cleanup needed' was caution, NOT a demonstrated launch prohibition.
If fresh full preflight passes all disk/memory/identity/protected gates, ordinary
Full may proceed with unchanged watchdog. Reclaim only provenrebuildable owned
cache when needed; no broad deletion/scan, no compression concurrent with run.
No ordinaryFull scripts prepared/launched yet. All openperformance/formal items
below remainOPEN. ThisturnPROGRESS: evidence-backed querycandidate qualified,
firstsemantic mismatch corrected and bothattempts preserved, sourcebacked.

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
