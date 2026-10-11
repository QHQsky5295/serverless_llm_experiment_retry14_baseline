# IEEE TC execution status — D443 Loquetier 500-ID coverage RUNNING

Updated 2026-10-12. This is a current checkpoint, NOT a replacement experiment plan.
Full earlier status is preserved byte-for-byte in EXECUTION_HISTORY_THROUGH_D433.md;
archive SHA60aeae9c98335a1b9c8d131441cb052b0351e09ac48f6b07d57ecf7ab6560d98.
earlier history through D402 remains EXECUTION_HISTORY_THROUGH_D402.md.
Search those histories and D-series reports before changing code. Their old NEXT/
RUNNING entries are historical. No approved experiment or failed evidence was removed.

## Mandatory authority and evaluation

- Read full /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md at task start,
  after compaction and before experiments; source and approved FULL_NAMES snapshot
  SHA f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
- Read BASELINE_EXECUTION_RULES_20261011.md. No non-PrimeLoRA performance
  optimization/configuration search; only necessary compatibility, correctness,
  measurement and safety adaptations. Reuse qualified prior points and report
  their actual outcomes, including nonfeasibility; do not call unsearched points optimal.
- Read METRIC_PROTOCOL_FROZEN_V1.md and METRIC_PROTOCOL_SINGLE_RUN_V2.md
  before configuration/comparison. Their SHAs remain respectively
  5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 and
  193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
- G1: all4000 native/source/adapter contracts and common joint TTFT/TPOT >=95%,
  then minimum physical lifecycle GPU-s/request. G2: same + U<=.75U_ref,
  then W1/W2 P95TTFT and observed joint-SLO difference. CE supplementary.
  Relative TPOT regression alone is not failure. Raw unknown numeric flags stay unknown;
  derived V2 binding metrics require complete evidence, not HTTP200 alone.
- One performance run per frozen execution key, no repeated means/CI/significance
  or best-repeat selection. Preserve development/reuse identity and every failure.
- Order: sealed7B PrimeLoRA -> remaining7B baselines -> comparison/necessary
  PrimeLoRA optimization ->3B -> ablations/mechanisms/sensitivities.
- Full names PrimeLoRA/ServerlessLLM. No figures before near target; no13B,
  new SGLang, paper edits, new weights/trace or pool regeneration.

## Current action — D443 same-config 500-ID functional coverage RUNNING

Launched2026-10-12T02:51:24+08:00 once in tmux tc-d443-7b. Monitor this exact
handle to terminal, do not relaunch. Serviceunit
primelora-tc-svc-ef3947b7d68f4418b5482ae7637b052b.scope,
InvocationID9a3e8a3e71944bf38b7b2e7cfa702025; outeraux0001live.
Executed MAIN702672d48712b64df62284174b1de417f22dba4c and
BASE9e0ad55d7f051a031ee4099d59d1a4718e64a59f; both pushed/remoteSHAverified.
Inputfreeze passed, HTTPc21602e6be726229a89d4e35b7043d3db15ec6e85aa967bf9c6572d057761c5d.
FreshpreflightSHAa0e174d14b2b65f341dd4c77c1677f5b1fa1f17a011ac8eaa174286731708270 passed.
Remote1000Mb/s/full; owned3B/7B/monitorInvocationIDs
d1dee2264e2a453c9496687b7c3ac183/3c15a7030a8948cd8b66b5703e28c210/
4903f0a48ce74ea4bc1335afdafebb8b;7Bclock6e96376632fc40659614cc709ff1dddf.
Prelaunch health once used systemPython lackingnumpy before GPU/fetch; recorded
health_launcher_error.json. Samecheck passed in existingouterPython3.12;
no install/source/configchange, no performance retry. Aux0003-0006 closed by
actualrecursiveemptytree+InvocationID; peaks119046144/28311552/15212544/25968640B,
high/max/OOM/swap0. Aux0002peak49405952B,high/max/OOM/swap0.
NEXT verifiedwait -> actualGPUrelease -> exact-ownedremote stop -> evidence copy
-> unchanged Loquetier/adapter_coverage_v1 audit withoutD385 reference -> seal/backup.
No secondheavyjob/live remotechanges/baselineoptimization. Preserve failures.

Read full PLAN/status/baseline rules/metricV1+V2; reuse approved D78/D80 readonly
gzip cache, no republish. BASE D443_LOQUETIER_500_ADAPTER_COVERAGE.md preregisters
existing D433 adapter_coverage_v1:500offered/500IDs/68925targettokens,
viewSHA286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d.
Raw originalresults/ieee_tc/loquetier_qualification/d443_20261012;
private /tmp/tc-d443-7bl; exact D441config/native/runtime/analyzer unchanged.
Reused D442recipes with campaign/500view only; no newweights/trace/pool/framework.
53MAIN checks passed0.329s;18BASE passed0.583s. All578 D442sources reverified
after priorcleanup;147protectedentries and threeuser/nativepatchSHAs unchanged.
Aux0002InvocationID3acfa94a66424b9aacda8476971abc43 checked actualrecursive
cgroup.procs empty and stopped; GPUhidden/CPU2,3,26,27/high3max4GiB/swap0.
The preregistered backup/inputfreeze/freshgates/launch steps are now complete above.
NoD385numericalreference for remapped500.
NotW0/ranking/optimum; no baselineperformance search or100repeat.
Then original1000/single4000; remaining7B/HydraServe/comparison/PrimeLoRAoptimization
before3B; fullscope below OPEN, goal ACTIVE. No blocker.

## D442 real100 closed; no rerun or baseline performance search

See D442_LOQUETIER_100_QUALIFICATION.md. Actual100/100native/source/adapter/timing,
17369tokens/29requestedIDs/four125-IDshards,0failures; complete500preparation.
Native/E2E/TPOTrecompute0ms. U1720.670042644022GPU-s,17.20670042644022/request.
Mean/P95TTFT266435.488/403860.253ms;mean/P95TPOT61.958/63.926ms.
CommonD394 TTFT0/TPOT70/joint0of100: NONFEASIBLE prefix,not fullW0/ranking.
Meanrouterqueue197.878s,nativequeue68.031s,registration-to-first.331s;
ready notice+193.473/287.078/379.126/476.123s. Do not subtractstartup orcallall
waiting compute/network. D385boundedreference83exact/17differences;
3existingmargins reusable,14unknown. Raw numericalfalse/n_correctnull retained.
500HTTP/420740410wireB/12985984450materializedB,0requestpacks;
100preparationreferences are not newtransfers. Realremote prior1000Mb/s/full.
Servicepeak42126540800B,hostavailablemin82596470784B,diskmin149954486272B,
high/max/OOM/swap0. All3returncodes0,actualfourGPUrelease/servicepathremoved.
Threeexact-ownedremoteunitsstopped/MainPID0/portsabsent;dual-endlogsSHAverified.
tmux ended0andremoved. Aux0001identitye17114947b5c472c8664d97b40472ef5
peak213463040B;0008-0011 exactidentity/actualemptytrees stopped,highmaxOOMswap0.
Strict pre-registered Loquetier analyzer passed withoutpostrun runtime/analyzer changes.
auditSHA503981d404c973736d96c06395539fe288c841c364236f4759bc2a0b7084c530;
578-file36274890B evidence_manifestSHAba8e1a0bcbcafc210e0abeef564a03d4c13fb9622541709b234ac7a1a1761647,
allsourceSHA independentlychecked. Old147protectedentries/threeuserfilesunchanged.
Closurebackups MAINff8e8fe23128528c80d62a8d47c426a9b6b64989 and
BASEe320186485c9b4e928377e77354dcaf491da712e pushed/remoteSHAverified.
BASEsummarySHA246a6f8569defe93ec71e9863398c6312d9874df29eb29f387645c07636a8d9f.
Afterbackup only5000owneddownloadfiles/12985984450payloadB reclaimed;
freed13004410880B,diskafter162940325888B(~151.75GiB). Originalpool/raw/remote
publishedobjects retained. Threeunreadable sameUIDproc entries2628/4156156/4157313
disclosed,notall-system FDproof. Receiptstorage/20261012_d442_download_cache.json.
All578sources reverified AFTERcleanup; old147entriesunchanged. Cleanupaux0013
ID9e8cb52c8d1a41f5b998c7a5aa740c39 peak2417983488B/highmaxOOMswap0;
verifyaux0014ID18fa0c73e4d4485fb4605fedcb04d065 peak29679616B/highmaxOOMswap0.
EveryD442ownedauxscope actualemptytree/identitycheckedandstopped; noD442units
remain. AllGPUs15MiBbackground/0util,notmux,hostavailable104GiB. Notfreshlaunchpermission.
NEXT freshgates andsameconfig500-IDcoverage,thenoriginal1000,single4000.
No100repeat/D438capacityrepeat,
baselineperformance tuning orsealedW0rerun. FullgoalACTIVE,concreteGPUqualification
andclosureprogress,no blocker;remainingfullscope belowstillOPEN.

## D442 preregistration and launch — historical details, now closed above

Launched2026-10-12 02:25:19Asia/Shanghai in tmux tc-d442-7b, once only.
Execution MAIN279cce93294d72e21abb2fdece760a94cb073baa and
BASEd81b05e95dcbd0cfebe296a9e7e86ee4dc0ea146 pushed/remoteSHAverified.
Frozen100viewaaf917bd816e10c4ad6f8138e5365aa9b2976b4300143cc712c12b84443cb020;
HTTPe1ab96f06e8f73a7497a47fafbb0b024189934359fbd4141cc7c62aaf3d1f054.
Freshpreflight passed SHA66f4ba800404244be7643ab8ae2887bef91ee2c4547111a48bc2a6a592263a69;
free162970681344B. Aux0006/0007 identity+actualemptytree stopped,
peaks119021568/28512256B,highmaxOOMswap0.
Serviceprimelora-tc-svc-5d6c6e6a08384c499441b4b7c02d00b5.scope,
InvocationID1d5c041d28d84d90b7868f84c34605d8; outeraux0001 live.
Actualremote prelaunch1000Mb/s/full, no link changes/republish.
Remote3B/7B/monitorInvocationIDs bdf3b975a833467da391a372c51a8420/
8501a79ee7fb4476adc9498803b9201e/01dcfa121a354adfad9ad079b48b6907;
7Bclock995d25520de44ec9b856c4eebc79be50. Do not touch remote while running.
NEXT monitor this exact handle to terminal; actualrelease -> exact-ownedremote
stop -> preserve evidence -> strictLoquetier audit/boundedreference -> seal/backup.
No second heavyjob, blindretry, baseline tune or unrelated source mutation.

BASE D442_LOQUETIER_EVIDENCE_AND_100.md preregisters original100/29IDs/17369tokens,
full500 preparation and fixed D437/D441 four125-ID shards, nativebatch2/context1024.
Raw originalresults/ieee_tc/loquetier_qualification/d442_20261012;
private /tmp/tc-d442-7bl; ingress8510/router8511/workers8523,8533,8543,8553.
Existing D432 run/remote/freeze/evidence recipes adapted by campaign/backend only;
no new weights/trace/pool, no native runtime/config/performance change.
Only necessary offline summary/evidence identity support added: explicit native
Loquetier IDs/path/shard-index/forward/terminal, static destination not RR, current
source/ready and actual worker lifecycle, full500 preparation counted once.
Unknown numeric fields stay unknown; existing D385 reference remains a bounded
separate check, not cross-backend exactness or full correctness proof.

Final53MAIN evidence tests passed0.409s,18BASE summary/HTTP tests passed0.604s.
Earlier69-check development invocation included16 duplicate imported tests;
preserved, not performance repetitions. All four aux0002-0005 actualemptyrecursive
trees and InvocationIDs checked/stopped; peaks35.26/41.10/32.34/29.82MB,
high/max/OOM/swap0, GPUhidden/CPU2,3,26,27/high3max4GiB.
All147 old protectedentries unchanged; protected_prelaunch SHA
1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657.
Three user-file/native Loquetier3patch/S-LoRA10patch SHAs unchanged.
BASE preregistration summary paper_results/ieee_tc/loquetier/d442_prelaunch.json;
MAIN analyzerSHA92e2d33bd2cf4d2fbe6ab7eb93ae8a220d92ccd0dfae7b0460f929c063643377.
Repeatedonce-onlycacheapproval retains D78/D80reuse, notrepublishing.
NoGPU100 launched at this preregistration checkpoint. Freshfreeze/remotehealth/
safety gates after double-repository backup; then actual100 once, preserve failures.
DoNOTrepeat D438 capacity/sealedfour-systemW0. Fullgoal ACTIVE, no blocker.
Remaining7B/HydraServe/comparison/necessaryPrimeLoRAoptimization before3B;
W1/W2/G2/A1-A5/S1-S13/reviewdocs/laterfigures remainOPEN.

## D441 closed — guarded native worker and static service checks passed

BASE docs/ieee_tc/D441_LOQUETIER_NATIVE_WORKER.md and curated
paper_results/ieee_tc/loquetier/d441_worker_integration_summary.json record actual
nativeworker/CLI integration into existing commonservice. Fixed D437 full500,
four125-ID byte-based shards; D438 load sequence/FP16/batch2/context1024/native
SMLM retained. Fullpool preparation after commonnotice; pre-ready ingress;
sequential workerstartup; static mapping, not RR/adaptive reroute; registry/path/
process/source health and nativecensus before dispatch. Fatal errors exit with
evidence/no retry; parent keeps actual process/physicalrelease checks. No baseline
performance search or PrimeLoRA algorithm change. Runtime/GPUforward/numerical
qualification still FALSE; do not infer performance from CPU assembly checks.

OuterPython3.12 commonregressions208passed10.103s; nativePython3.9 CPU26passed
2.263s before HTTPdependency addition. Missingaiohttp found before modelimport;
added only eight absent, hash-locked binary transport dependencies, aiohttp3.13.3
same as outer. Existingtorch2.1.2cu121/Transformers4.45.2/PEFT0.17.1/nativekernel
and existingattrs/idna/typing_extensions unchanged. No build/backendupgrade.
Afteraddition49worker/HTTP/online/nativeCPU checks passed4.426s/no skips.
Frozenconfig/sourceview/API/versions validation passed. Earlier196/14checks
preserved asdevelopment. One51-test attempt wronglyincluded Python3.12parent
ingress inPython3.9 andfailed itsasyncio.timeout path; realouterparent tests passed
in208, no fallback patched. OneAPIinspection checkedRequestHandler instead of
Server; correctedinspection passed. These are retainedCPUcheck errors,notperformance.

Newsourceview raw originalresults/ieee_tc/loquetier_qualification/d441_20261012/source_view,
manifestSHA2d8e801033a4f3b4062b94dbaf34e6c59c1613f4e2427705091435c6f9e82bb7.
BASEconfig configs/ieee_tc/loquetier_7b_static_native_v1.json SHA
d9f3b81d3d90340739ec66b45d36c1c71c706640dd2cd790a19d3ea4a80a617a.
NativePython3.9 worker,existingouterPython3.12; ports8510ingress/8511router/
8523,8533,8543,8553workers. ActualHTTP/runkey still to freeze. Oldview notoverwritten.
BASEsummarySHA607e8184289531676985116afb5d2ed213a0d34836551b65cd60bf3521aa37ab.
Crossrepositorybackup recorded in baseline_audit/d441_loquetier/checkpoint.json.

All14D441auxscopes GPUhidden/CPU2,3,26,27/high3max4GiB/swap0, recordedhighmaxOOM0;
actualrecursivecgroup.procs empty andInvocationID verifiedbeforeownedstop.
Finalsourceaudit7SHAs passed; only4existingrunnerdefinitions changed asintended,
27others ASTidentical. Auditaux0014IDe9a4442ef0404d769ec70a480ca945f0,
peak23130112B. Old147protectedentries andthreeuserfiles/nativeLoquetier3patches/
S-LoRA10patches unchanged; protectedverificationSHA1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657.
NoGPUmodel/remoteaction/newpool/weightdownload/oldoverwrite/figure. Repeatedcache
approval continuesD78/D80reuse, no newpublication. Disk162973720576B lastobserved,
notfreshlaunchpermission. Concreteimplementation/test/compatibilityprogress,no blocker.

NEXT: Loquetier-specific nativeanalysis/readiness/evidence contract, actual100
HTTPfreeze and freshgates/backups -> real100 ->500-IDcoverage ->original1000 ->
single4000. DoNOTrepeat D438capacity or sealedfour-systemW0. Loquetier notyet
qualifiedW0; remaining7B/HydraServe/comparison/necessaryPrimeLoRAoptimization
before3B. W1/W2/G2, A1-A5/S1-S13/reviewdocs/laterfigures/fullgoal remainOPEN.

## D440 closed — HTTP/native-token/physical-accounting checks; preserved historical checkpoint

BASE docs/ieee_tc/D440_LOQUETIER_HTTP_CONTRACT.md records necessary commonHTTP
handler, explicitLoquetier observation/publisher/lifecycle contract, and mainphysical
collector token/schema pair. Sharedfull500publisher retained; preparationreferences
are not per-requesttransfers. Nativeinteger first/last separate fromHTTPnotification;
n=1 TPOTisN/A. Unknown numericalflags remainunknown. No nativealgorithm,precision,
batch2,staticshard orperformanceconfiguration change; no baseline tuning.
Loquetier source-view/worker/CLI/outerstaticservice still NOT integrated/qualified.

185HTTP/online/commonregressionspassed10.093s;26nativeCPU/onlinepassed2.363s,
noskip;36mainphysical-lifecycletestspassed1.417s. Loopback tokenproducer andtestleases
explicitlysynthetic, notGPUarithmetic/actualrelease proof. Earlier21CPUchecks0.441s
preserved asdevelopment,notperformance repeats. AuxGPUhidden/CPU2,3,26,27,
high3/max4GiB/swap0. Finalcommon/native/lifecyclepeaks379109376/254976000/28749824B,
high/max/OOM0. Fiveauxscopes actualemptyrecursivecgroup.procs andInvocationIDchecked,
thenstopped. Protectedcheckpeak29327360B; all147oldentriesunchanged;verificationSHA
1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657.
Threeprotecteduser-fileSHAs,nativeLoquetierthreepatches/S-LoRAtenpatchesunchanged.
NoGPUmodel,remoteaction,download,republish,poolcopy,oldoverwrite orfigures.
RepeatedcacheapprovalusesexistingD78/D80; no additionalonce-onlycache created.

BASEsummary paper_results/ieee_tc/loquetier/d440_http_contract_summary.json records
source/test/resourceidentities;SHAc87797bf3da87b1a51b34b9db050079333dab4b6ff82e4f1f151b6c229d6875b.
BASEbackupb28c993f02ccdfb0726f3c2edf9201e05872add7 pushed/remoteSHAverified.
MAINcross-link paper_results/ieee_tc/baseline_audit/d440_loquetier/checkpoint.json.
Final7sourceSHA/ASTauditpassed;runnerpreexistingdefinitionsunchanged,onlynewhandler.
Fiveexistingpublisher/validatorfunctions changedexplicitly;oldbackend163regressionspassed.
Auditaux0006ID799e422a2fe6432a8952bb8654774bb7,peak26447872B/highmaxOOM0,
actualemptytreeverifiedandstopped; allsixownedD440scopesclosed.
This continuationmade concretecode/CPUchecks/protectionprogress,notperformance.
FullgoalACTIVE,no recurringblocker; completeapprovedscope belowremainsOPEN.

NEXT: connect nativeworker guardedinitialization/requestfactory/registry+processready
receipt/fatalexit and existingouterservice staticmapping/startup/health/physicalleases.
Create newsourceviewmanifest; changedsourceadapter cannotreuse D438manifest.
Onlythen freshgates andapproved100->500-IDcoverage->original1000->single4000.
DoNOTrepeat passedD438capacity orsealedW0. LoquetierhasnoqualifiedW0rowyet.
Remaining7B/HydraServe/comparison/necessaryPrimeLoRAoptimization precede3B;
W1/W2/G2,ablations,mechanisms,sensitivities,reviewdocsandlaterfiguresstillOPEN.

## D439 closed — necessary online adapter CPU checks passed; preserved previous checkpoint

BASE docs/ieee_tc/D439_LOQUETIER_ONLINE_BOUNDARIES.md records additions to the
existing runner/measurement helper: explicit static shard mapping, single-owner
online FIFO/native batch bridge, exact native terminal, cancellation and fatal-failure
boundaries. Original ModelGeneration/SMLM/mixed modules/FP16/batch2/static shards
remain; no baseline performance search or policy optimization. No new framework.

Actual native CPU add/pre/post/KvPool/KvCache witness reproduced a lifecycle bug:
after the last request releases KV, cached_decode_kv retains dead references;
the next prefill acquires a page for the ended request. Minimal adaptation clears
only an empty batch's already-released references after native generate; active
batches untouched, unexpectedly live pages fail visibly. Original witness has2
pages vs corrected1 after the next request; no GPU/performance benefit claimed.
Native checkout/historical3patches unchanged; overall adapted behavior disclosed.

Final26LoquetierCPUtests passed2.317s,no skips,in existingnativePython3.9;
163commonServerlessLLM/vLLM/S-LoRA regressions passed19.134s, existingPython3.12.
Earlier21/23development checks preserved as development tests,notperformance repeats.
AuxGPUhidden/CPU2,3,26,27/high3max4GiB/swap0;finalnativepeak254386176B,
commonpeak800219136B,protectedcheck29315072B;high/max/OOM0.
All6ownedauxscopes exactInvocationID/actualemptytrees checked thenstopped.
Finalsource audit6sources SHAverified; all pre-existing top-level functions/classes
in both commonfiles remain ASTidentical. Aux0006ID8e191c1102b2436984bd8834eb97edd1,
peak18042880B,high/max/OOM0; no GPU arithmetic.
147protectedentries,three user-fileSHAs,nativeLoquetier/S-LoRA patchSHAs unchanged.
NoGPUload,remoteadministration,download,republish,poolcopy,oldoverwrite orfigures.
Repeatedonce-onlygzipcacheapproval recorded;continue reuseD78/D80,notrebuild.

BASE curatedpaper_results/ieee_tc/loquetier/d439_cpu_boundary_summary.json records
code/test/source/resourceidentities. Backups recorded in cross-repository checkpoint
after commit/push; D438 capacity result remains at its original execution identity.
SummarySHA13e3d95c5d74378821931814ccbc78cc11c33b7e853bfc6257416317c925bbc7.
BASEcheckpoint6204e2025a432a55d4bb815ff76ae44f7e50e629 pushed/remoteSHAverified;
MAIN cross-link paper_results/ieee_tc/baseline_audit/d439_loquetier/checkpoint.json.
This is concrete implementation/CPU evidence progress,not a new performance point.
Full goal ACTIVE;no recurringblocker;all remaining approvedscope below stillOPEN.

NEXT: nativeworker/startup/HTTP/commonremote/physicalleases and Loquetier-specific
analysis contract; these are NOT yet integrated or qualified. Create newsourceview
manifest for changedhelper,doNOT reuse D438manifest with mutablehelper symlink.
Fatal native allocation/forward failure must stop worker and verify actualrelease;
queue cleanup is not proof of releasedGPU. Then freshgates and approved100->500-ID
coverage->original1000->single4000. No repeatedcapacityprobe or sealedW0 reruns.
Loquetier still NOT a qualifiedW0 row;HydraServe/remaining7B/comparison/necessary
PrimeLoRAoptimization precede3B. W1/W2/G2 and fullmatrix stillOPEN.

## D438 native observation and single capacity witness passed — preserved previous checkpoint

BASE docs/ieee_tc/D438_LOQUETIER_NATIVE_OBSERVATION.md records necessary native
measurement hooks, exact source-view AST preservation and guarded capacity probe.
Native3historicalpatches untouched; nativeSMLM/batching/loading/FP16 preserved.
Unknown/wrong-shard adapter rejected before nativeindex0fallback/KV allocation;
startup module/name/pointer readback; actualforward groupedIDs/rootcontext check;
timestamps immediately after existing nativeinteger.item(), no per-token added
CUDA sync. Startup pointer readback is explicit and belongs in future preparation.
Structural binding is not remote-content or model numerical qualification.

New9CPUtests passed2.277s; combined14passed2.055s,no skip, existingnativeenv.
CommonServerlessLLM/vLLM/S-LoRA163regressionspassed10.995s. No performance rerun.
All aux0001-0005 closed by exactidentity/emptyrealprocess-tree; high/max/OOM0.
Old147protected entries and user changes unchanged. No remote administration,
downloads/newdata, figures or changes to sealed W0 results or evaluation ruler.
Prior turn D437 was concrete code/qualification/backup progress; no blocker.

D438 raw /home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d438_20261012.
NativeviewmanifestSHA4dc33282f5777e3462b42c887c22de82ffa00b0ef8ea2aac3af52c9f0ff9dd7a;
helperSHAa4c116150db7f66f7d7d475cabdc3951095977235c520fc95ad7a5d104e203de;
sourceadapterSHA2e7e0d2cdb4aa96b239bf08c88ed1a1b8175330631a67c2087074e95dc146e10;
probeSHAdc417b9d3847aba55afe4d942b4271ab3f2ca0e42bffce1f06b92354f0905c72.
BASE preregisteredcodecommit28ffe7ed969c19d2d8683d282b9bda90c1f70d73.
D437 MAIN checkpoint28a14a10ea704c514720498bb3d61a37d11cbf7e remains preserved.

Actual single capacity probe completed2026-10-12 about01:07-01:09Asia/Shanghai,
BASE execution28ffe7ed969c19d2d8683d282b9bda90c1f70d73,
MAIN prereg400c3c8b1f358659dc8e8bf464ace4f6e24fa5e8; both backed up before launch.
Existinggated launcher/watchdog/nativePython3.9/GPU0/service72/80GiB/swap2,
predictedgrowth2GiB; preflightpassed. D437largeststaticshard0/125IDs selected
by footprint,notworkload. All125contenthash/load and128targetmodule pointertables
passed, nativebatch2/context1024KV allocation passed. No retries/configchanges.
Finalallocated24214530048B/reserved24494735360B; staticaccounted24214509568B.
SampledNVMLGPU0max24831131648B, not exactinstantaneouspeak.
Servicepeak7744462848B,hostavailablemin105536823296B,diskmin162913902592B;
113watchdogsamples,high/max/OOM/swap/abort0. Service/watchdogexit0,
servicepathremoved, actualGPUreleased,fourcards15MiBbackgroundafter.
Serviceunitprimelora-tc-svc-381426cb9ee049f28fb10e12fe13621d.scope,
InvocationIDc0bb3150f32e4699a29530914aeb2d0b. Aux0006
ID39681dc94bd747cc8f96f6f8b7ed6f60 peak85028864B; aux0007
IDb2969aef0d00407494727e90d4d326f1 peak29364224B; both actualemptytrees
identitychecked/stopped,high/max/OOM0. Completedtmuxpane0 removed.
All147protectedentries remainunchanged; nativeLoquetier/S-LoRA historicaldiffSHAs
unchanged. No remoteaction,republish,download,poolcopy,oldresultoverwrite orfigures.
User repeatedonce-onlygzipcacheapproval is recorded; reuseD78/D80, not rebuild.

NativecapacityresultSHA3e2f0a63c2d920d5e5489fe0f93e3a65b096a7f3974604a76f149305c3404f37;
launchSHA8bf163483b63ac2ac2786b82caecbfdabaae8e36fae8a1bf97fd1b6215f11cb1.
BASE curatedpaper_results/ieee_tc/loquetier/d438_native_capacity_summary.json
SHA2e16287fcf406d0cf54c61b2511d337d717a767482be94b2be62a42a970727d3;
all9sources independentlySHAchecked. BASE closurebackup
9c7adcc88a1a576893f945272b7a94141f267d6a pushed/remoteSHAverified.
This is one localimmutablepool loading/KV witness, NOT requests/forwardworkspace,
fourworker HOST peak,numericalcorrectness,realremote or mainSLO/GPU-s evidence.
runtime_qualified=false/performance_run=false. Do NOT repeat passedcapacityprobe.
NEXT necessary commonHTTP/staticshard/remote/physicallifecycle adaptation and
100->500-ID->1000->4000 qualification; Loquetier still NOT a qualified W0 row.
HydraServe/remaining7B/comparison/PrimeLoRA optimization before3B; all fullscope OPEN.
This continuation made concrete guardednativeGPUqualification/closure/backup
progress. Full goal remains ACTIVE; no recurringblocker or completedmatrix claim.

## D437 static/CPU checks closed — preserved earlier checkpoint

BASE docs/ieee_tc/D437_LOQUETIER_STATIC_CAPACITY.md records full existing500-ID
metadata accounting, historical source/receipt verification and nativeCPU rules.
Extended the existing Loquetier gate with opt-in --static-capacity-only; no native
algorithm edit, performance search, trace generation, modelGPUload, remote action,
HTTP fetch, republishing, extracted-pool copy, old-result overwrite or figure.
User again approved once-only readonly gzip delivery cache; reuse D78/D80, do not
rebuild it. Real reads/transfers/contention/local materialization stay measured.

Actual350rank8/150rank16, homogeneous128targetmodules; deterministic input-only
byte-balanced four disjoint125-ID shards, no trace/hotness read. Batch2 preserves
historical compatibility point, context1024; not official default or optimum.
PerGPU baseFP16 13476835328B, originalPEFTFP32 5469372416B(shards0/1) or
5435817984B(2/3), preparednativeFP16 4194304000B, pointers256000B, fullcontextKV
1073741824B. Accountedtotal22.55152/22.52027GiB excludes CUDA/workspace/temporary
padding/casting/activations/allocator andHOSTpeak: runtime_qualified=false and
safe_to_launch=false. Header/config/size checks do NOT rehash all tensor payloads.
Staticcapacity JSON in BASE paper_results/ieee_tc/loquetier/d437_7b_static_capacity.json
SHA049441391b67e500328546c3c8e1001fd1ce197f0c1a93e150fc26e8bdfe4a77;
scriptSHAb0f679614432c135197867007c3fa8db62bceca4b671d88cd809a66486c1268f.

5metadata tests and4nativeCPU tests passed; precommit combined9passed2.219s,
no skip. NativeModelGeneration.post_process exact1/2/256 stopping withEOSdisabled,
batch mark/adapter preservation andgreedy config checked; CPU mockKV, notmodel
arithmetic orGPU qualification. NativeunknownID->0 reproduced: wrapper must
reject unknown/wrong-shard ID and qualify actual forward binding, nothidefallback.
ExistingnativePython3.9.25/torch2.1.2+cu121/Transformers4.45.2/PEFT0.17.1 reused;
no install/build/driverchange. NativeHEADaae33baeeb19777129c1ccbff99a898d4a0e2c63,
three historicaldirtyfiles/defaultdiffSHAde476d06c318ba2be7f65dd9dba9bfb30a2f2c7e14e8bd6e35f30206ab041a39
unchanged; kernel and old7B/3B gate/OOM receipts verified, see BASEreport.

All5 D437auxscopes GPUhidden,CPU2,3,26,27/high3max4GiB/swap0; high/max/OOM/swap0.
Each exactInvocationID/actualempty-subtree stopped. Precommit testsaux0004
ID370ccc61689744b78416cfdb596f0eca peak248307712B; protectedaux0005
ID478687e8a52a4d35b28e1f4c57d7a589 peak12562432B. All147oldprotectedentries,
three user-fileSHAs,nativeS-LoRAtenfilepatch unchanged. Protectedverification
paper_results/ieee_tc/baseline_audit/d437_loquetier/protected_verification.json
SHA1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657.
LastGPUeach15MiBgraphics/0util,notmux,hostavailable105GiB,disk162920607744B
(~151.73GiB); not a fresh heavyweight launch permission. No cleanup/deletion.
BASE checkpoint0f73477ae5bff492356e56812cfb4cdc9f78d2a0 pushed/remoteSHAverified.
D436 MAIN priorbackup440f309dc819b5f145cadd596f4cbc25c38a3ed2 remains preserved.

NEXT: reuse commonHTTP/remote resolver/physicalownership and nativeLoquetier
to supply necessary staticmapping, fixedinput, nativeinteger-boundary timing,
failure/cancel and forward-binding checks; then fresh safety gates and protected
actualcapacity/100-request qualification. Preserve originalSMLM/virtualmodel/
MixedLoraModel/ModelGeneration/batching; no baseline performance search. Existing
gate timing/counts/ready-shifted arrivals are R2, notcommoncontract performance.
Follow approved500-IDcoverage->original1000->single4000 order; failures retained.
Loquetier not yet qualified for W0 table. Then HydraServe nativequalification,
7Bcomparison/necessaryPrimeLoRAoptimization before3B. All remaining scope OPEN.
This continuation made concrete qualification/code/test/metadata/backup progress,
not a new GPU performance result and not full goal completion.

## D436 comparison closed — do not repeat sealed W0 performance runs

D436_MATCHED_7B_W0_FOUR_SYSTEMS.md records the complete offline join of sealed
D371/D397 PrimeLoRA,D414 ServerlessLLM,D423 vLLM,D435 S-LoRA. No GPU run,
baseline performance/configuration change,remote administration,republishing,
download,pool generation,old-result overwrite or figure. D78/D80 once-only
readonly gzip cache remains reused under the user's repeated approval.
Extended existing check_d299_full_context.py with opt-in --include-sealed-slora;
explicit native RPC path/index and registration for S-LoRA, no fabricated vLLM
fields. Reuse already-sealed full registry/process/content audits. Complete4000
request/source/adapter/arrival/prompt/nativeprompt/target/content match;
458224tokens/132IDs, same matchedcontract SHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.
All prior D424 row fields unchanged. Current observations joint/GPU-s per request:
PrimeLoRA97.70%/3.6206309810630626;ServerlessLLM98.125%/4.022440685881069;
vLLM99.475%/4.028844290537294;S-LoRA94.425%/3.7892694676132526 NONFEASIBLE.
PrimeLoRA resource advantage about10% over two feasible baselines, but P95TTFT
and jointSLO worse. Against S-LoRA raw resource reduction4.4504%,P95TTFT
reduction83.9658%,joint+3.275pp; not a both-feasible G1 comparison.
All rows retained; no broad optimum,G2,statistical or full numerical-proof claim.
Native phase boundaries differ: S-LoRA modelRPC registration vs vLLMengineentry.
Remote132/131480060B each for prior three;S-LoRA fullpool500/420740410B.
Preparation-strategy difference is not isolated hierarchy/confirmed-state evidence.
New paper_results/ieee_tc/main_comparison/d436_7b_w0/comparison.json SHA
59a6dfa6b3a512ac561c65fb334de4843895c5c7f120e709302d50afd6704427;
metrics.csv SHA5e204c1d8514494821b71b730e27bb5e2a8ad2d25de7fc002ee1882d0f08047a.
Resident reference reused byte-identically SHA737c84804023c9f98677fb5f19c8240b4b05cabdb52f6ecdbbfd4527964c3269;
no refreeze. D424 comparison/reference/CSV hashes unchanged. All147protected
entries,three user-file SHAs,nativeS-LoRA tenfilepatch remain unchanged.
Protected verification SHA1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657.
46CPUtests passed0.390s. Full analysis uses existing LLM/bin/python module entry,
not baseline env(which lacksijson). Two failed CPU-only entry/dependency attempts
created no analysis output; no performance retry or environment installation.
CPU2,3,26,27/high3max4GiB/swap0/GPUhidden; successfuljoinpeak1699659776B,
high/max/OOM/swap0. All five owned D436 auxscopes exactidentity/empty-subtree
closed. BASE remains94543f3d4dee1a870084ec877011f40ce1b27277 unchanged.
That turn made concrete offline comparison+validation+documentation progress.
NEXT: remaining7B Loquetier/HydraServe native qualification from existing reviews,
necessary compatibility/correctness/measurement/safety only, no baseline tuning.
Then complete7B comparison/necessaryPrimeLoRAoptimization before3B.
No repeated D435 or previous sealedW0. All approved remaining scope below OPEN.

## Historical D435 closure — completed before D436; do not rerun

D435_SLORA_FULL_W0.md preregisters one original4000/132ID/458224token W0 using
the exact D432/D433/D434 native config. No baseline runtime/config/analyzer changes.
Raw /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011;
private /tmp/tc-d435-7bs; tmux tc-d435-7b launched2026-10-11 22:58:29Asia/Shanghai.
37CPU checks passed1.696s; fresh preflight passed,147oldprotected entries unchanged.
D434578-source-file hashes still match after object cleanup; three user-file SHAs,
native config/runtime/source-view and historical tenfilepatch unchanged.
Execution MAIN84d6b4d0607944c5d6ef47bd9ada2d30817e4887 and
BASEcae13c57c012ea55a8199740f2696b9db732362f both pushed/remoteSHA verified.
Input/cache freeze passed: W0viewa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d,
httpSHAf0fe970f53c549b38cc0dbd8def0109115436f4b1d8ee8530b2ad8efcec8c6f3;
Triton2055files/62246523B, no GPU preloaded. Aux0002-0004 exactemptytree closed,
high/max/OOM/swap0. Runtime aux0001 now empty-tree stopped. Closed service
primelora-tc-svc-f300f4dd9c474b9cb7cfb41a718c5019.scope,
InvocationIDfd0490c67700443fbe7238bb4d06578a.
Notice1237981.351138279,arrival_start1238041.351138279,end1242005.260041401;
fixed arrivalwindow3963.908903122s, not ready-shifted.
Remote prelaunch1000Mb/s/full;3B/7B/monitor InvocationIDs
bb3e91ffee0242f8a82dc189a22e6ada/deb3850712654fa2a7c9945a38a07af4/
d5d4313fc78242a889b9ab44c65efc2e;7Bclock7dbcab2393a74058935a5838bffb7b06.
Monitored the same actual service/tmux through terminal2026-10-12 00:06:29;
no restart, baseline tuning, remote interference or other performance run.
4000offered/submitted/terminal/native/source/adapter/timing contracts,0failures,
458224native output tokens,132requested IDs; four actual500-ID workers served.
E2E/TPOT/native recomputation error0ms. Common D394 TTFT3783/TPOT3994/joint3777
of4000=94.425%,observed NONFEASIBLE under95%;23requests below3800threshold.
U15157.07787045301GPU-s,3.7892694676132526per binding-valid request.
Mean/P95TTFT7396.416512/17699.067366ms,TPOT31.251644/45.953041ms,
E2Emean10962.479328ms. Raw n_correctnull/numericalfalse remain unchanged.
All217TTFT violations are in first500(joint283/500); later3500 joint3494/3500.
These descriptive slices do not replace the complete primary population.
Full500 preparation500HTTP/420740410wireB/12985984450materializedB,0packs;
4000preparation references are not additional transfers. No downloaded pool copied.
Servicepeak62637105152B,hostavailablemin61200150528B,diskmin150091988992B,
high/max/OOM/swap0. Service/replay/watchdogreturncodes0,actualGPUrelease and
servicepathremoval verified. All three exact-owned remote units stopped,
MainPID0/portsabsent; dual-end transfer/monitor evidence copied with SHA checks.
Strict unchanged slora_resident_rr_v1/full4000 analyzer passed. All147protected
entries and three user-file/native tenfilepatch hashes remain unchanged.
Curated baseline_audit/d435_7b_slora_full_w0/audit.json SHA
1c9361f2d995c27e0672611728697f4de48bd0ff3c4e9f4fd778523565b6e4c0;
578-file/227275259B evidence_manifest SHA
d0f6b8b4aca2ac8b34f13230e9ebd559982ed7bce684441c7c5a998e8b0886c2.
All578source hashes/sizes independently rechecked. Aux0005-0009 empty-tree
identity-checked stopped, high/max/OOM/swap0; no D435 scope remains. Verification
aux0009ID3206a46e183e4829b8c06c9d7d2ad473 peak2875392B. No independent-review-agent
PASS asserted.
NEXT full input/generation/resource matched W0 join with
sealed PrimeLoRA/ServerlessLLM/vLLM -> remaining7B qualifications. Reuse existing
comparison code, new exclusive output; do not overwrite D424 or refreeze reference.
BASE closure backup f1e7310a16ac40bfbdd9f25d047742bc46c25f9a pushed/remoteSHAverified;
MAIN closure backup212cec45ad5a31d4363356a205a13350210637e4 pushed/remoteSHAverified.
Owned recreatable D435 objects alone reclaimed after backup and complete retained
original/metadata/reference checks:5000files,12985984450payloadB,freed13004468224B;
diskafter162938236928B(~151.75GiB). No original/raw/remote published cache removed.
Receipt paper_results/ieee_tc/storage/20261012_d435_download_cache.json; three
unreadable sameUIDprocess entries2628/4156156/4157313 disclosed, not claimed all-system FD proof.
All578 sealed source hashes rechecked AFTER cleanup,147protected entries unchanged.
Cleanupaux0010ID23ca5f2597a24f80b9d57bc58f7c4c69 peak3221852160B/high3416/maxOOMswap0;
postperformance only, not a change to D435 performance pressure. Verificationaux0011
IDd8662fde35c94ccaa85b2ef74daeb60d closed by exactidentity/emptysubtree; no D435scopes remain.
Fresh complete gates still required before any next heavy run; do not repeat cache removal.
No S-LoRA performance tuning or repeated prefixes/fullW0; no3Bjump. Goal ACTIVE.
Prior running-doc backup MAINe7d7ae0e84954e9826dccc96d7e6a3542e67328d;
execution commits remain84d6b4d/cae13c5, not the later status commits.
Current continuation reread complete plan/status/latest rules and metricV1+V2,
made verified wait through terminal plus closure/audit/seal progress. D78/D80 cache
reused, no duplicate publication/extracted pool or live remote change. No baseline
source/config changes. Next matched comparison remains separate and not yet done.

## Parallel read-only preparation — Loquetier reuse, no performance run

BASE docs/ieee_tc/LOQUETIER_REUSE_REVIEW_20261011.md now records inspected
historical gates, official/native source boundaries and required common-protocol
adaptation. Documentation-only BASE5d36776a205393242c16c5bdb579bebec4240bfd
pushed/remoteSHA verified; D435 execution BASE remains cae13c5.
No subagent, baseline code/config edit, test/build/install, pool scan, GPU load,
remote administration or second performance run. Source/receipt hashes quoted
in the new review are historical evidence; independent sealing deferred to idle time.
Existing Loquetier HEADaae33baeeb19777129c1ccbff99a898d4a0e2c63 and three
historical dirty files preserved. 7B128-ID/256-request gate has507tokens,
arrival_scale0/inputcap512/outputcap2/batch2: R2 compatibility, NOT matched W0.
3B500-ID OOM occurred during adapter preparation; no inference result and no
proof that every static-sharded deployment is infeasible. Approved static-sharded
deployment still needs complete500-ID/common input/native token/source/physical
lifecycle qualification; do not optimize/search baseline performance settings.
Native unknown adapter ->index0 behavior requires explicit binding validation.
Official ModelGeneration/virtual model/MixedLoraModel/SMLM must remain native;
do not invent a cluster scheduler or replace its core. Current review is not a
fresh experiment-audit PASS or completed numerical qualification.
That review's historical next was D435 closure, then matched W0 comparison;
both are now closed above. Remaining7B qualification is next; no3Bjump.

Current continuation also completed read-only HydraServe prerequisite review:
BASE docs/ieee_tc/HYDRASERVE_REUSE_REVIEW_20261011.md, backup
e690a0a31a987075a522e4b021fa13662b3e7c41 pushed/remoteSHA verified.
Pinned native HEAD8ae605de354ccfa2e9095514cdb4a9e9c56aa56b remains clean.
Historical missing adapter fields do not by themselves prove scheduler policy
must change; actual pipeline layer mapping and destination LoRA-manager binding
still require correctness proof. vLLM0.4.2 age is not an exclusion criterion.
Current local k3s/docker/containerd services active; podman/k3s executables and
qhq subuid/subgid present. Docker socket access/default kubeconfig not usable by
current observed user, kubectl/helm root0700. This is metadata, not actual GPU
Kubernetes qualification; no existing cluster/remote service changed or tested.
Do not repeat the old blanket claim that no Kubernetes exists, or infer access
from service existence. New review changes the next qualification questions,
not the approved baseline order, core mechanisms, frozen metrics or run identities.
That earlier goal turn made evidence progress plus verified wait; D435 now closed above.

## D434 closed — do NOT repeat

D434 original1000 compatibility completed by2026-10-11 22:44:35Asia/Shanghai.
1000offered/submitted/terminal/native/source/adapter/timingcontracts,0failures,
122510native tokens,60IDs;fouractualTP1workers served/500-ID registries.
E2E/TPOT/native recomputation error0ms. U3542.956034773262GPU-s,
3.5429560347732623/binding-validrequest, all physical leases closed.
CommonD394 TTFT750/TPOT999/joint749of1000=74.9%,NONFEASIBLE prefix,
not fullW0/ranking/reference/optimum. Mean/P95TTFT37636.062/236234.733ms,
TPOT33.501/49.954ms,E2Emean41782.526ms. Raw unknown numerical flags unchanged.
Source0-99/100-499/500-999 joint1/100,249/400,499/500; last500 meanTTFT241.488ms.
Do not drop startup from primary denominator or infer fullW0 from a short prefix.
Fullstaticpool500HTTP/420740410wireB/12985984450verifiedmaterializedB,0packs.
Preparation span95.131748s; native ready notice+221.662/320.676/421.199/518.432s.
Deployment transfers counted once,requestwire0/preparation references1000.
Servicepeak62617686016B,hostavailablemin60923981824B,diskmin150234251264B,
high/max/OOM/swap0. ActualGPUreleased/servicepathremoved/exit0;remote3ownedunits
stopped by InvocationID/MainPID0/portsabsent; resolver and remote evidence preserved.
Strict existing slora_resident_rr_v1/prefix1000 analyzer passed, no runtime/analysis changes.
ReportD434_SLORA_1000_COMPATIBILITY.md;curated baseline_audit/d434_7b_slora_compatibility/:
auditSHA05a595c69063efc92484a208325b3476fd4c49c857a6a450b281845af39660a0;
578-file/63678229B evidence_manifest SHA
bf444b0c1ce339e6a63342b6484e8cc2dd1d15fbe2d317163d9a9378c65ff7ab.
All578 source-file SHA/size entries independently verified. Closure backed up:
MAIN869fdff76c57a5b8de6efe718d453d891cec2809 and
BASE33fc840bc282839628a8275fe016671b2541c955, remoteSHA verified.
Audit/seal aux0005/0006 exactidentity/empty-tree closed, high/max/OOM/swap0.
Aux0007 and cleanup0008 now closed after exactidentity/empty-subtree checks.
Only verified private objects reclaimed:5000files/12985984450payloadB,
observed freed13004435456B,free163189366784B. Originals/raw/remote cache retained.
Cleanup-only aux peak3221749760B/high4003/max0/OOM0/swap0; this was after
performance and did not alter the D434 performance resource record.
Receipt paper_results/ieee_tc/storage/20261011_d434_download_cache.json.
147protected entries unchanged,postSHA23b9be29909ef728f3450ea7e32f592419d81e319012d383ea290ecd618cfd0d.
Three protected user-file SHAs and nested10filepatch unchanged.
Historical next action (now completed): freshgates -> preregister/backup and launch
ONCE D435 full4000W0 SAME native config. D435 is closed above; do NOT relaunch.
No100/500/1000 repeats,baselineparametersearch,oldsealedW0repeat or3Bjump.
Cleanup restored about151.98GiB; launch still requires fresh complete safety gates.
Full goal ACTIVE/all remaining approved scope OPEN. Previous turn launched D434;
this turn monitored same live handle to terminal and completed strict audit, progress.

Raw: /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d434_20261011
Private: /tmp/tc-d434-7bs. Reuse run/start/stop/health/freeze/preserve/seal recipes.
Original1000viewSHA0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981.
Nativeconfig slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d.
D430 source-view manifest SHA
51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004.
DP4/TP1/RR/sequentialstartup/packedHOST/native swap nonBMM;
input984/context1024/tokenbudget14000/batch1024/running1000.
Full500 HOST preparation afternotice, business notice+60, actualGPU allocations,
qualificationprotection1800s; no freeGPU/adapterpreload. ExistingTritoncache2055files/62246523B.
Prelaunch37CPUtests pass1.858s inexistingouterPython3.12; earlier systemPython3.10
missingnumpy attempt retained, no install/codeworkaround. No performance retry.
Execution MAIN47be5422cbf65c8c0905f78668a83c188b7b9ccf,
BASE5059cb0b86c5a5378a772adaa25aa80bb7a1b240 both pushed/remoteSHAverified.
Runningdoc backupMAIN677a53e049bea6118d8f4e9d559179ada2098e3c notexecutioncommit.
Serviceprimelora-tc-svc-b0ce62d358e94489b1d5958c0e83ede0.scope,
InvocationID18c0f4b4575a45be824a81b181b46476 now released/removed;notrestart.
Notice1235966.54994207,arrival_start1236026.54994207,end1237105.9642220077.
Remote priorlink1000Mb/s/full, unchangedD78/D80 once-only gzip cache;no republish.
Closedremote3B/7B/monitorInvocationIDs158b40360fdf46bb818a036493b0246c/
e0525a84e0af4598a1e715bfac1e7255/1028fcd55b374827a52f7d705422add0.
Remote7Bclock8e030525ed52454e89698c6a7e7b6287;copiedtransferSHA
092504a38e8e2b7f9f57be60442cd2c1068db89d36a3777c43236007f407aa6c,
monitorSHAb6c9853df74bb7721af442f618e967f46ff66a0d1c459f1e9ef4ec44ffaa798e.
Aux0001-0008 closed. No remaining D434 process scope.
No figures,paper/backend/policychange or newweights/trace.

## D433 closed — do NOT repeat

500-ID functional coverage:500 offered/submitted/terminal/native/source/adapter,
68925 native tokens,0failures,exact500IDs,four actual workers served/500-ID registries.
E2E/TPOT/native recompute error0ms. D394 TTFT268/TPOT494/joint266 of500=53.2%;
NONFEASIBLE functional view, not fullW0, ranking, reference or optimum.
U1743.8732772790827 GPU-s,3.487746554558165/request.
Fullpool500HTTP/420740410wireB/12985984450materializedB,0packs.
Deployment bytes counted once; per-request references are not extra transfers.
Peakservice62410342400B,high/max/OOM/swap0; all physical leases closed.
D433 report: D433_SLORA_500_ADAPTER_COVERAGE.md.
Curated baseline_audit/d433_7b_slora_coverage/audit.json SHA
7cc9f8e220e0b5cf761cd5c416e6d3126d22f42cb77f46494af10341feedf3f7.
576-file evidence_manifest SHA
cb03ba5ddf53241e427712c285197bdcb3f89def15c2480f45e6872a4b20003c.
All576 source hashes verified.37CPU tests passed1.636s before launch.
Execution MAIN627aebff3fdaf67822946e52eacb7c9291f2eb30,
BASE15077ab4aeeac9532b1cb8d9f18567d76883163c.
Closure and cleanup backups subsequently pushed/verified; latest MAIN
29e98482d0375181fb020f53747d29ecdb136336,
BASE219ad363a83e44487d5df4831861a0edad7e3d2f.
All GPU/remote/aux scopes closed by actual identity and empty process trees.
Only /tmp/tc-d433-7bs/remote_artifacts/objects reclaimed after references, retained
source bytes/inodes/hash and process checks:5000files,12985984450payloadB,
freed13004427264B. Originals/raw/remote cache retained. Three unreadable sameUID
process entries disclosed. Do not repeat cleanup or redownload to repeat a passed audit.
Receipt paper_results/ieee_tc/storage/20261011_d433_download_cache.json.
D432 original100 passed; bounded D385 numerical witness95exact/5differences,
3known margins/2unknown retained; no full numeric proof or exact-token gate.

## Sealed W0 comparison and reference

D397 seals PrimeLoRA D371 reused-development W0, not independent holdout:
4000/132IDs/458224 native tokens,97.70% joint,3.6206309810630626 GPU-s/request.
D414 ServerlessLLM98.125%,4.022440685881069;D423 vLLM99.475%,4.028844290537294.
PrimeLoRA GPU-s about10% lower, but P95TTFT2837.896ms vs1052.246/1200.507ms
and jointSLO not leading. All meet95%; not all-metric/cross-scenario/G2 optimality.
Allthree132HTTP/131480060wireB: no W0 remote-byte-reduction evidence.
See D424_MATCHED_7B_W0.md and paper_results/ieee_tc/main_comparison/d424_7b_w0/.
W0 U_ref16115.377162149176,Q_ref1200.5068596918136ms,B12086.532871611882GPU-s.
W1/W2 references and G2 OPEN. D395/D396 cannot set reference due measurement gaps.
D394 SLO manifest paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json,
SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
<=616input:TTFT2995.348713684507ms/TPOT59.74409694902162ms;
>616:TTFT4633.345350589479ms/TPOT85.24698166015581ms. Do not regenerate.
Mostly-zero/repeated weights:500logical IDs,498zero/2nonzero,4weightSHA/6tree classes.
Native bindings, independent numerical evidence and content diversity are distinct.

## Safety, ownership, storage and backups

- MAIN=/home/qhq/serverless_llm_experiment_retry14_baseline,branch retry14_continuous_queue_v2,
  remote faaslora_origin. BASE=/home/qhq/serverless_llm_baselines,main,origin.
- One heavy job. Disk start>=max(150GiB,100GiB+1.5growth), runtimefloor100GiB.
  Service memory.high72/max80GiB,swap2GiB,CPUs4-23,28-47.
  Auxhigh3/max4GiB,swap0,CPUs2,3,26,27. Actual workers contained before init.
  Separate replay/watchdog; stop only exact owned units after identity checks.
- Strict SSH alias primelora-artifact-174=lab14@10.199.227.174:8122.
  Data192.168.4.174ports18080/18081,inference192.168.4.178;credentials outside Git.
  Token file /home/qhq/.config/primelora-tc-d75/artifact.token, never print.
  Current pre-D433 link1000Mb/s/full; historical100Mbps separate state, cause unknown.
- D78/D80 published gzip reused, no request packing. Real reads/transfers/competition/
  local materialization counted. No live remote administration/hashscan/cleanup,
  tests/second heavy job. Never subtract cumulative spans to invent corrected latency.
- Original7B pool /home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix;
  original seed42 trace and existing environments reused.
- Protected147-entry seal D433raw/protected_before.json. Historical results unchanged.
  NEVER stage MAIN configs/generated/lora_manifest_1000.json SHA
  3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df;
  BASE scripts/replay_openai_trace.py SHA
  04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
  scripts/run_serverlessllm_relayserve_continuation.sh SHA
  337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
  Preserve nestedS-LoRA ten historical patches/default diff SHA
  7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60.
  Preserve all other unrelated tracked/untracked files.
- Explicit task-only staging, diff/tests/source/secrets checks; no force/rawlarge/secrets.
  Update current status before yielding. Do not claim full goal complete.

## Remaining full approved scope — not canceled

Remaining7B baseline qualifications/comparisons, W1/W2/G2 and references, PrimeLoRA
optimization where justified,3B optimization/comparisons,dLoRA3B,ElasticLocality,
Loquetier/HydraServe native qualification, all A1-A5 and S1-S13, CE/cost sensitivity,
two-review action/design documents and later academic figures. Reuse only same contracts.
B1/B2/C1/C2/C3 and formula/novelty evidence remain required. Hardware limits disclosed.
Full original plan + preserved history define every setting; this tracker is not a shortened plan.
