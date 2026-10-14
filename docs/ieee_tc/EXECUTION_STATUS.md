# IEEE TC execution status — D527 Full100 qualification PREPARED

Updated2026-10-14. Prior status preserved BYTE-FOR-BYTE in
EXECUTION_HISTORY_THROUGH_D519.md, SHA
e0ed5415b779effec429b48ea716abde2e3e9b1a379d5abbc11073cdca2f3a99.
Older RUNNING/NEXT paragraphs there are historical, not current process state.

## Current authoritative handoff

D527 PREPARED, NOT LAUNCHED: new original100 Full config binds exact D526
service/preparation timings; all other D523/D476 controls unchanged. 308 CPU/
configuration/entry tests PASS; runtime child identity loads both profiles,
non_target_config_equal=true, CUDA not initialized. Actual qualification scope
recursiveempty/CLOSED/pathgone, peak3083395072B, events0/swap0. RAW
results/ieee_tc/p2_backend_qualification/d527_20261014; curated
paper_results/ieee_tc/p2_backend/d527_metadata_full_qualification/.
Read D527_FULL_METADATA_QUALIFICATION.md. Reused start/remote/close helpers
prepared; backup then ONE original100 via existing runner d527. Must audit
native contracts and actual publisher installation/accounting/use, then ONE
new4000 W0 if qualified. No CPU/device/calibration/baseline reruns.
Latest delayed user reply again confirms k3s is unrelated; existing isolated
HydraServe D510 fulfills adaptation, no shared cluster operations. No inference
or remote job live. Full remaining scope below stays ACTIVE.

Prior D526 completed handoff (superseded only for next-action preparation):

2026-10-14 13:46 UTC: D526 TERMINAL SUCCESS, strict source/calibration export
COMPLETE. This supersedes all RUNNING/prepared/not-tested statements below.
All1669 calls pass (34warmup/1635representative),200/200service+24preparation
classes, no missing content sources. Metadata installed/triggered:5060updates,
99040B actual pinned increment,28392544copiedB,no poison/no buffer replacement.
1669 bilateral fetches/3406922201wireB,zero request packing. Calibration only:
2112.197101198137GPU-s,not Full W0 performance; n_correct remainsnull.
Source SHA837915f97f7719abe9137d7aad54f3ac009b0c1dc07d00f9b1c2fc0341e8ed62;
source execution commit6b22e02bfbc4a513e7e5ccf8b27715c2fcebe3cb.
Actual GPU release/pathgone and launcher auxiliary recursiveempty/pathgone
confirmed; remote bothservices/monitorinactive and downloadSHAs match remote.
58 affected CPU tests PASS. Analysis/seal/export scopes all recursivelyempty,
closed/pathgone; peaks989442048/363069440/950407168B,events0/swap0.
Servicepeak5708759040B,minhost106395848704B,all memoryevents0/swap0.
New profiles: paper_results/ieee_tc/p2_backend/d526_metadata_profiles/initialization/.
Service SHA1d349c001d37b9a9f8516cf50a2b3643bd8b535b68ee09f49c02e23b31f532a2;
preparation SHAfd7f0c7e9c7d6c69c796c1fb1ab3e4a29aaec32fd50d1cdacfe95e3a12e054ed.
Frozen D524 inputs reused unchanged, no D524 failed timing reused. Read
D526_METADATA_TIMING_RESULT.md. Backup COMPLETE: checkpoint
a0081007e683aac892065af8558e9cd5027394c8 PUSHED/remote-equal,19explicitfiles,
147protected/history/userdirt/secrets/archive-roundtrip checks. Actual receipts
RAW/backup_receipt.json and backup_closure.json; scope recursiveempty/pathgone.
No need to repeat calibration/export/tests/backup. Only this post-backup handoff
is uncommitted alongside the untouched user's manifest.
Then NEW original100 Full config with these exact profile bindings; ONE4000 W0
if qualified. Do not repeat calibration, CPU/device qualification or baselines.
No Full100 config/launch prepared yet; no GPU/remote/analysis job remains live.
Full7B W1/W2/G2,3B,ablation,sensitivity and reviewer deliverables remain ACTIVE.

Historical live-monitoring and preparation notes, superseded above:
D526 RUNNING exactly once in tmux tc-d526-source, existing D298 collector.
Source checkpoint6b22e02bfbc4a513e7e5ccf8b27715c2fcebe3cb PUSHED/remote-equal.
RAW results/ieee_tc/p2_backend_qualification/d526_20261014; exact unchanged
D524 config/spec135waves1669calls; original failed D524 remains immutable.
No second job/tests/build/compression/hashscan/backup/remote changes during
measurement. This is affected initialization, NOT a Full performance point.
Remote bothservices/monitor ACTIVE; 7Bclock
remote-process-monotonic:8b4095a8b39f4571a70144d720c2cf64;
monitorInvocation9df46df004a54b03b2c5b7183dc003de. start_receipt.json confirms
launch; check actual state/launcher identity before any next action.
Starter scope CLOSED/recursiveempty/pathgone,peak31965184B/events0/swap0.
Launcher aux-ff0bf133da534ed6919fcfdd770f064e.scope Invocation
43613bbe2f164eaeb21678c0bf6a9ccf, inode24923990 captured live.
At watchdog sample153, actual source generation reached wave8 (native_host),
beyond D524's first-wave failure; servicepeak5708759040B/events0/swap0,
no abort. Partial successful generation is NOT complete calibration or speedup.
2026-10-14 13:18 UTC follow-up: D526 remains LIVE in its original tmux;
533 native_source_profile_sample records observed of1669 planned calls.
At watchdog sample670: servicepeak5708759040B, all memory events0,swap0,
hostavailable108169105408B,no warning/abort; root filesystem161GiB available.
No source1.json/source1_launch.json terminal result existed at this check.
No second job, code change, hash scan, backup or remote maintenance performed.
Latest user reply confirms shared k3s belongs to another project and authorizes
appropriate adaptation. Existing isolated HydraServe D510 already fulfills that
platform adaptation; preserve its result/limitations, do not redo or touch k3s.
NEXT monitor -> exact local release/recursiveempty closure -> owned remote stop
and dual-side evidence -> strict source/profile/metadata audit. No D524 failed
timings may enter exports. Then original100 Full
using new timing and ONE4000 W0 if qualified; full scope remains ACTIVE.

2026-10-14 13:25 UTC: verified same live tmux panePID2172010/python3;
922/1669 source sample records, watchdog1105,peak5708759040B,events0/swap0,
hostavailable108103266304B,no warning/abort. Terminal launch/result still absent.
Previous goal turn was a VERIFIED WAIT, not a restart or completed experiment.
While measurement remains live, ONLY offline exporter/tests/helpers were edited;
no serving collector/backend/model/config/input changes and no tests executed.
Existing export_d298_profiles now has d526 case: frozen D524 input directory,
NEW d526_metadata_profiles output/seal directory. Extracted unchanged seal
checks into audit_measurement_seal; six new fixture tests protect separate input/
measurement paths, exact source/commit identity and tamper rejection. UNTESTED
until actual measurement closure; no qualified export or new performance claim.
Reused D362 raw helpers via mechanical apply_patch adaptation: D526 analyze.py,
run_analysis.py,seal_measurement.py,export_profiles.py. Do NOT execute now.
On terminal source/launch: verify nativeGPUrelease/pathgone; stop_remote.sh;
then ONE bounded auxiliary run of run_analysis.py (it includes collect_closure.py,
affected CPU tests and analyze.py). Do NOT separately collect first: open('x')
identities/receipts are single-write. Close with close.py analysis; then bounded
seal_measurement.py -> close.py seal -> bounded export_profiles.py -> close.py
export. Reinspect helpers/actual receipts before action; preserve any failure.
No commit/push yet because backup/tests must not overlap inference. Checkpoint
6b22e02 remains the actual measurement source. Full remaining scope unchanged.

D525 CPU qualification COMPLETE:393 PASS/0failure/error/skip, actual installed
LoraState produces np.int32 and exactly reproduces the preserved D524 failure.
Three installed official0.30 source files byte-identical; no CUDA initialized.
Lossless Integral/index normalization only; bool/float/tensor/arbitrary coercions
still rejected, no fallback or formula/budget/SLO change. Existing96 nativeCPU
oracle cases included; native producer->publisher path now covered.
Actual CPU scope recursively empty/closed/pathgone; peak2312175616B,events0/swap0.
Read D525_NATIVE_INTEGER_CONTRACT.md. RAW d525_20261014; curated
paper_results/ieee_tc/p2_backend/d525_native_integer_contract/.
D525 checkpoint backup COMPLETE,10explicitfiles/protected147/dirt/secrets/archive
checks. Firstarchive guard rejected3.49MB remote monitor before staging; failure
and partial archive retained. Archival-only member limit8MiB used in backup2;
no serving/resource limit changed. backup2 closed/recursiveempty/pathgone,
peak32534528B/events0/swap0. No need to repeat qualification or backup.
Historical pre-launch handoff (superseded above): D526 runner case prepared:
new raw d526_20261014 reuses exact D524 config/spec, no input regeneration.
Actual source calibration NOT launched; export must use new output directory.
After qualified new timing, new original100 Full then ONE4000 W0. Do not repeat
completed CPU/device checks or any baseline. Full remaining scope ACTIVE.

D524 is TERMINAL FAILED at source_profile_wave:0, before any successful source
sample (14 records,0pass). Native worker HOST attachment installed the candidate:
cpu_live_mapping_v1,99040B/two banks/1024rows/14slots; actual pinned63300->162340B
within unchanged native2GiB/global16GiB budgets. This proves installation only,
NOT generation/performance. Root worker error: lower_dense_lora_metadata rejects
index_mapping because its elements are not exact built-in int; native V2
LoraState.make_lora_inputs uses tuples of np.int32. The profile-reservation error
is a fatal-worker consequence, not the root failure. No timing export allowed.
source1.json SHA ef1a60e0e68e168727a81bb90fbac1db61ac19bdfdef1121e7522ce9b24b98f6.
Actual local_cleanup.json confirms GPU release, recursively empty auxiliary and
both captured service/auxiliary paths gone; remote_stop.json confirms all owned
units inactive, no ports18080/18081 and downloaded evidence hashes matching.
All jobs CLOSED; no GPU/CPU/remote experiment remains active. Do not repeat D524.

Historical D525 intention (completed above): preserve failure and verify official/native CPU input types, then
qualify an exact lossless integer-normalization correction with native LoraState
fixtures and strict rejection of bool/float/tensor/unknown IDs. No new weights,
traces, budgets, formulas or SLOs. Subsequent affected timing
calibration requires a new execution key; previous D362 timing cannot be relabeled.
Then new original100 Full and ONE4000 if qualified. Full remaining scope active.

Historical launch handoff below is superseded by the terminal outcome above:
D524 source calibration RUNNING in tmux `tc-d524-source`, exactly once via
existing `run_d298_capacity_source.sh d524`. Check current process state before
any next action; do not launch another job, tests, build, compression, hashscan,
backup or remote changes during inference. RAW d524_20261014/source1*. It is
initialization evidence, not Full performance. Never relabel its timings as W0.
Source checkpoint d4cddb5d1429098942362dcc50ebf79e51981799 PUSHED/remote-equal;
15 explicit files,322tests/protected147/userdirt/secrets/archive checks passed.
Backup scope closed/recursiveempty/pathgone, peak30003200B/events0/swap0.

First management starter failed importing numpy with host stdlib Python before
tmux/inference. Its source and failure record preserved in start1_failure.json.
Reused D362 stdlib-only HTTP health import; no dependency install, no remote
restart/config change. resume_start.py/start2_gate.json retain exact remote
InvocationIDs; start_receipt.json establishes actual launch. Both starter scopes
closed/recursiveempty/pathgone. Remote NIC actual1000Mb/s/full;
7B clock remote-process-monotonic:36160e2111304032b3cca1eb369d92a7.
Remote services/monitor ACTIVE for this run; do not stop until local release.
launcher_identity.json: aux-f00ce6e6e6a94dc4865d2810067dca63.scope,
Invocation b40303de811c42ec887b0664484b456d, inode24914216.

NEXT monitor actual measurement -> exact local release/recursiveempty closure
and owned remote stop -> retained bilateral evidence -> measured class/profile
and metadata-installation audit/export. Then new original100 Full using exact
new profiles, ONE4000 if qualified. No rerun of old D522/D523/baselines.

D523 original100 Full attempt TERMINAL before model initialization or request
submission: `FrozenServiceProfiles.load` rejected model/configuration/context
mismatch. The sole new `ieee_lora_metadata_policy=cpu_live_mapping_v1` changes
runtime timing identity; D523 still referenced D362 timing initialization.
This is a launcher/profile-binding error, NOT a measured performance regression.
Do not bypass exact identity, remove the field, relabel old timing, or rerun D523.
Preserve its config/output and all failed evidence.

RAW `results/ieee_tc/p2_backend_qualification/d523_20261014/`:
`7b_cpu_frontend_validation_100/launch.json` pass=false, service_returncode=1,
GPU-context release confirmed and service path gone. `local_cleanup.json` and
`start_closure.json` confirm exact owned auxiliary recursive processes empty;
actual captured paths gone, events/swap zero. `remote_stop.log` confirms both
artifact services and monitor inactive/MainPID=0; no listeners remained.
NIC before launch was 1000Mb/s/full. No request transfer matched this remote
process clock, consistent with pre-start rejection. No live GPU/CPU/remote job.

NEXT D524: refresh affected service/preparation initialization via EXISTING
D298 native source collector. Reuse exact D362 135 waves/1669 calls, existing
inputs/published artifacts. Only completed fixed-output lengths may transfer;
fresh timing must carry the exact metadata-policy identity. Common warm SLO,
resource budgets, nine equations and performance workload remain unchanged.
Collector's actual global HOST attach can additionally verify native publisher
installation/accounting/counters. Then a NEW original100 Full key with qualified
profiles, followed by ONE4000 W0 if eligible. Do not repeat completed D522
device/CPU checks or any baseline.

D524 preparation COMPLETE:322 affected CPU tests PASS/0failure/error/skip,
exact135waves/1669calls retained, next Full model identity verified equal.
No CUDA initialization. Actual auxiliary recursive processes empty and path gone;
peak2285068288B, high/max/OOM/OOMkill/swap0. New spec SHA
847182165fcbe44bf132dce1f67ee61c844371ef5cce6ff9ffdabffd5936a179.
Read D524_METADATA_PROFILE_REFRESH.md. Metadata installation receipt/counter
audit is prepared; actual native installation/timing remain unmeasured.
Backup and launch now COMPLETE as above; no timing result yet.

D522 checkpoint cc80e58b1a495e90fda3eecc657012bc3ba36b83 PUSHED/remote-equal.
Ten explicit files; source/tests/protected147/userdirt/secrets/archive-roundtrip
checks passed. RAW/backup_receipt.json and backup_closure.json retain evidence;
actual backup recursive processes empty, captured path gone. Do not repeat
backup or qualification. Only this post-backup status update is uncommitted.

D522 actual CUDA device qualification COMPLETE, device2.json pass. Ten cases
(one/mixed/reassigned/all14slots/allbase, specialization off/on) have exact
native metadata/eager nonzero LoRA/pre-existing CUDA Graph output agreement.
Each mode queued32 update/replay/output-copy operations without per-update
readback; all64 outputs exactly match native eager oracles. Stable GPU buffers;
each publisher adds99040 actual allocator-accounted pinned bytes. No weights,
adapter pools or traces created. This is bounded arithmetic, NOT full-model,
native-worker-installation or performance qualification; n_correct staysnull.
Firstattempt import-only failed before CUDA/cases; corrected preflight sys.path,
allfailed logs retained. Both actualservicepaths gone and exactaux recursiveempty
closed/pathgone. Read D522_METADATA_DEVICE_QUALIFICATION.md.

Added read-only publisher counters to existing source/worker observation (no
extra RPC/readback). CPU regression521 PASS/0failure/error/skip; peak1059352576B,
high/max/OOM/OOMkill/swap0, actualrecursiveempty/pathgone. D521 tests remainvalid
historical evidence, not newperformance. RAW d522_20261014; curated
paper_results/ieee_tc/p2_backend/d522_metadata_device/. Backup COMPLETE above.

Historical D523 pre-launch intention (superseded by terminal outcome above): existing runner
d523 case and configs/ieee_tc/20261014_d523_7b_cpu_frontend_validation_100.yaml
prepared, NOT launched. Exactly D475/D476 non-target config plus
ieee_lora_metadata_policy=cpu_live_mapping_v1; no parameter tuning. This uses
Full's actual global HOST reservation/worker attach, unlike direct preflight
which does not attach HOST. Must verify installed publisher, actual pinned
accounting/update evidence and all100 nativecontracts before ONE new4000 W0.
No D523 remote/start helpers prepared yet. Remote inactive; no liveGPU/CPU job.
Do NOT repeat D522device/CPU or completedbaseline runs.

## Prior D521 checkpoint (superseded next-action text below)

D521 checkpoint0bad961c2d0f15eb5360752e5658a3ed2a933598 PUSHED/remote-equal.
RAW/backup_receipt.json confirms10explicitfiles/source/tests/protected147/
userdirt/diff/secrets/archive-roundtrip checks; backupactualrecursiveempty
closed/pathgone,peak33509376B/events0/swap0. Do NOTrepeatbackup orCPUqualification.
Onlypost-backupstatushandoff below is uncommitted; user'smanifest remainsuntouched.
Freshpostbackupdisk161GiB,MemAvailable107172MiB,GPUprocessesempty.
Nextnativequalification ONLY as below; no livedevicejob/configprepared.

D521 default-off `cpu_live_mapping_v1` integrated via existing runner/worker
HOST-budget attachment; no config/run enabled yet. Read
D521_METADATA_PUBLICATION_CONTRACT.md. Fixed pinned staging K*(48N+24L+32),
K=actualnativeconcurrency2; D476 N1024/L14 gives99040B. Exactcopyeventbefore
reuse, sameexecutionstream/thread, nativeGPUaddresses/CPUflags preserved,
currentmapping+slotlayout invalidation/activation/LRU/admission untouched.
ExistingHOSTallowance must fit staging plus unchangedloadworkspace; actual
postallocation check and futureallocatorwide checks charge allpinnedbytes.
Noinstalleddependency edit, newweights/traces/pools or graphreplacement.

D521 CPU qualification COMPLETE:441tests PASS/0fail/error/skip in cpu2/.
Firstattempt441 had3errors fromtwo oldfakeworkerfixtures lacking nativecfg;
new13tests passed. Fixtures nowinclude actualdefaultempty additional_config;
productionchecks notweakened. Failedattempt retained. ActualtwoCPUscopeclosure
recursiveempty/pathgone; cpu2peak2288648192B,high/max/OOM/OOMkill/swap0.
RAW results/ieee_tc/p2_backend_qualification/d521_20261014; curated
paper_results/ieee_tc/p2_backend/d521_metadata_publication. NoCUDAinitialized,
no realDMA/graph/model qualification/performanceclaim. BackupNEXT; actual
backup_receipt/closure will establishcompletion. No livejob.

NEXT ONLY minimal native device metadata/graph/nonzero-LoRA qualification,
including realworkerinstallation/HOSTaccounting and updatecount evidence.
Then ONEordinaryoriginal4000W0 at frozenD476 plussolemetadatastrategy ifqualified.
Do NOTrepeat D520/D521CPUtests, D519diagnosis, HydraServe orbaselinefullruns.
Allfullmatrix remainsACTIVE. No GPU qualification config preparedyet.

D520 checkpoint0f5671ce432bd62e9cb55ca8c5a774c607e51070 pushed/remote-equal;
311CPUtests/96nativeCPUoracle/fiveofficialsources/protected147 archived.

D519 outcome4e272a8be91db44ab1c6e10d2a91e7e7d1609d4a pushed/remote-equal.
D520 read source/history and added ONLY an unintegrated pure CPU metadata helper
faaslora/utils/lora_metadata.py + tests. No serving caller, no installed dependency
edit, no allocator/policy/SLO change, no GPU/remote/k3s operation.
Read D520_NATIVE_METADATA_DEPENDENCY.md. Official native mapping currently makes
GPU reductions/unique and CPU branches from information already present on CPU.
Avoiding that round trip is a new falsifiable hypothesis, not a proved speedup.
High host-free driver time may overlap waiting; do NOT restore unlimited cache,
sweep thresholds, subtract it from step time or repeat D56–60.

D520 CPU qualification COMPLETE:311tests PASS/0fail/error/skip,96 actual installed
Torch2.13 CPU kernel-metadata oracle cases, five officialv0.30 files byte-identical.
CPU scope actualrecursiveempty closed/pathgone; peak3112230912B,events0/swap0.
Raw results/ieee_tc/p2_backend_qualification/d520_20261014; curated
paper_results/ieee_tc/p2_backend/d520_metadata_lowering/cpu_qualification.json.
Explicit tested checkpoint backup NEXT; actual backup_receipt/closure establish
completion. Do not rerun completed qualification. No live inference or CPU job.

D520's then-pending integration is now implemented byD521above. GPUqualification
and ordinary4000 stillpending; neitherCPU result constitutes scientificclosure.

## Authority and execution rules

Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this status before
work/after compaction/before experiment. PLAN SHA
d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83.
Also read STORAGE_PRIORITY_20261013.md,BASELINE_COMPLETION_ORDER_20261012.md,
BASELINE_EXECUTION_RULES_20261011.md,BASELINE_SCOPE_20261014.md,
SLO_BUDGET_RATIONALE_20261014.md. D483 storage maintenance DONE; do not repeat
broadscans. HydraServe necessary adaptation/full4000 D510 DONE; do not restart it.
Shared k3s belongs to another project: no service, credential, permission or
resource changes. Qualified HydraServe isolated rootless Podman path retained.

Published comparison: PrimeLoRA, vLLM(SOSP2023), S-LoRA(MLSys2024),
ServerlessLLM(OSDI2024), HydraServe(NSDI2026), Loquetier(NeurIPS2025).
dLoRA(OSDI2024) only3BW1/W2. ElasticLocality is ONLY a composite ablation,
not baseline/internal-strong-control; historical stronger results retained.
Do not relabel/remove failures or infer improvement from changing the roster.
Only necessary baseline compatibility/correctness/measurement/safety adaptation;
no performance sweeps. Abnormal results still require evidence-based defect review.

One performance run/frozenkey; no CI/repeat means/winner filtering. Full names
PrimeLoRA/ServerlessLLM. Figures deferred until near target. Chinese updates as
paper evidence: completed/current scientific question/remaining experiments.
No new weights/traces/pools/13B/SGLang/paper source edits/history overwrite.
Use existing runners/environments. CPU metadata fixtures are not new workloads.

## Retained D519 evidence (not a new performance point)

Production main runner restored byte-identicalD476 db44e8f068eb30c69466f32ad3d147ffbc45f9e2.
Only D518 opt-in component observer differs in existing serving modules.
ONE1000 diagnostic finished nativecontracts1000/0failure/122510tokens;
independent numerical n_correct remainsnull. All11allocationsreleased; remote
60fetch/74953014wireB/packing0. Servicepeak20621922304B/events0/swap0.
Full component qualification FAILED due64MiB cap in primaryprocess,10/11journals
complete,279/1000 whole-process request coverage. Partial analysis explicitly
retains47038 realclosedsteps;1orphan not imputed. Do NOT repeat analysis/GPU.
45513single-token steps:mean23.669581ms/threadCPU7.468172,allocation.125878,
backgroundfree10.206529,input.919288,LoRAactivation2.100393,scheduler.230154.
987multitoken:step83.052146/CPU82.809049,execute80.685168/LoRA22.072073.
Nested notadditive; no puredecode/prefill/GPUidle inference.
Sourceaf2c2680e375c3c5d14720a810d30d9e6892852b7951fcdb6e1522083005e7b1.
Resultmanifest21c9299fee58c93e993dc7c4e03b244e6a290de243c1b501c0bf23be389c5f0e.
Result doc D519_STEP_DIAGNOSTIC_RESULT.md and result1 evidence remain immutable.

D516 rejected source-query candidate:4000nativecontracts, joint94.775%<95,
U11011.303222888382,2.752825806/request, P95TTFT3321.357967ms. Not accepted;
not proof of causal harm from one run. RestoredD476, no repeats.
D517/D518/D519 diagnoses/qualifications/backups completed; do not redo them.

## Frozen evaluation and sealed 7B W0

Before comparison/configselection read FULL METRIC_PROTOCOL_FROZEN_V1.md and
METRIC_PROTOCOL_SINGLE_RUN_V2.md, SHAs
5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22,
193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
G1 all4000correct/nativecontracts + jointTTFT/TPOT>=95%, then physicalGPU-s.
G2same + U<=.75Uref, W1/W2P95TTFT/observedSLOdifference; CE supplementary.
Numerical binding/count evidence is not independent numerical proof; unknown
n_correct staysunknown.95%/75% project points, not universalindustrynorms.
Keep approved90/95/99%/.5/.75/1/multiplier sensitivities including negatives.

D513 six-system published table SHA
044c04d6bb5b894761628146a65de248a53436cdd4ba23346837eccb3473b17b:
(system, joint, GPU-s/boundrequest, P95TTFTms)
PrimeLoRA D476 .97425,2.694612200556614,2315.363336;
ServerlessLLM D414 .98125,4.022440685881069,1052.246361;
vLLM D423 .99475,4.028844290537294,1200.506860;
S-LoRA D435 .94425,3.7892694676132525,17699.067366 NONFEASIBLE;
Loquetier D445 .0435,4.249595808607468,555110.032229 NONFEASIBLE;
HydraServe D510 .68725,2.4155381525526174,8851.575365 NONFEASIBLE.
HydraServe all34workerspp1: does not prove pipeline/consolidation performance.
ElasticLocalityD453 .95775/2.473178968/2420.607829 remains ablation evidence.
PrimeLoRA U10778.448802226456 is33.011%/33.117% belowServerlessLLM/vLLM,
but lowerattainment/highertail; no all-metric/cross-scenario optimum claim.

D394 SLO SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3:
input<=616/>616 TTFT2995.348713684507/4633.345350589479ms,
TPOT59.74409694902162/85.24698166015581ms.
W0Uref16115.377162149176,B12086.532871611882,Qref1200.5068596918136.
W1/W2refsOPEN; W0 is NOT W1/W2/G2closure.
Matched4000contractSHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.

## Safety, protection and publishing

One heavy job. No build/tests/hashscan/compression/backup/remote changes during
inference. Diskstart>=max150GiB,100GiB+1.5growth; runtimestop100GiB.
Servicehigh72/max80GiB,swap2,CPU4-23,28-47; auxhigh3/max4GiB,swap0,CPU2,3,26,27.
EmptyhoststartMemAvailable>=102GiB; otherwise remaininggrowth formula.
External16GiBhoststop. Actualworker restrictions before load; no blindOOMretry,
globalGPUreset/reboot/driver changes/unknowncleanup. Closure checks exact
Invocation/inode,actual recursive cgroup.procs and capturedpathgone.
Latest read-only disk~162GiB,hostavailable~104GiB; gate freshly before next job.

MAIN retry14_continuous_queue_v2/faaslora_origin. BASEHEAD
09baa9379bdc3518f18f408d2cad3763c2754554, unchanged. Protected147 seal
originalresults/ieee_tc/slora_qualification/d435_20261011/protected_before.json
SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
NeverstageMAINconfigs/generated/lora_manifest_1000.json SHA
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
BASEscripts/replay_openai_trace.py SHA
04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
scripts/run_serverlessllm_relayserve_continuation.sh SHA
337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
Explicit paths only; checksum/tests/diff/secrets before commit/push; no rawlarge
files, credentials or forcepush. All old data/failures/userdirt preserved.

RemoteSSHprimelora-artifact-174 lab14@10.199.227.174:8122 strictknownhost;
data192.168.4.174ports18080/18081 inactiveafterD523. Token outsideGit
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
D78/D80readonlygzip1581215261B reuse, no republish/extractedpool/requestpacking.
Latestactual1000Mb/s/full; old100Mbpsdistinctstate,causeunknown. Real necessary
read/transfer/contention/materialization count; no subtracting old accumulated
spans. Remote start/health must explicitly complete before launch.

## Full remaining scope — ACTIVE

7B PrimeLoRA evidence-driven optimization;7BW1/W2/G2/commonrefs;
3Boptimization/allbaselines;dLoRA3BB6;A1–A5confirmedstate/elasticity/handoff/
hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/load/GPU/network/
workload/KV/control/backend/EOS/designsensitivities;offlinecost/CE/Pareto;
B1/B2/C1/C2/C3/formula/novelty/two-reviewdocs;lateracademicfigures/V2delivery.
No stage closure from this CPU-only result, no goal completion or scope reduction.
