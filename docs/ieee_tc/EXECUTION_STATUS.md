# IEEE TC execution status — D455 growing controller delay localized; no policy change

Updated 2026-10-12. Current checkpoint, NOT a replacement experiment plan.
Full prior status is byte-identically archived in EXECUTION_HISTORY_THROUGH_D448.md,
SHA bec6bbaac2809ed746fb3d6a7285c16bafdd95c44512daa7d2929bd325ee69fa (cmp passed).
Earlier complete histories through D433 and D402 remain. Historical NEXT/RUNNING
entries are not current instructions. No evidence or approved task was removed.

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

## Current action — D455 retained request/CPU diagnosis completed

Read full PLAN/status/baseline rules/V1/V2 after compaction; no online code/policy,
configuration, baseline tuning, remote administration, GPU run or figure changed.
Cache reapproval reuses completed D78/D80; no republish or extracted pool.
Extended existing offline control analyzer with strict 12-stage same-monotonic-clock
partition; 8000/8000 retained rows pass, max identity error0ms. D454 matchedW0
mean/P95 remain exact. FullminusElastic meanE2E+622.783867ms, meanTTFT+147.099872ms.
Mean permit acquire->resume0.380vs356.170ms; arrival->controllerrelease207.674vs6.136;
runtime-slot281.050vs151.597; admission->nativeentry291.174vs173.306;
nativeentry->firsttoken difference only12.617ms. Decode3336.563vs3102.616ms,
worker->controllercompletion255.562vs4.157. Nested timings never added twice.

Across fixed8x500source-order groups, PrimeLoRA meanarrivaldelay40.450->406.651ms
and completionpickup49.091->456.315ms; ElasticLocality stays approximately4-9/2-8ms.
This is a retained within-run trend, not independent repeats or isolated causality.
2859completeFullplanningreceipts:2849owned_execution_epoch+10initialized_gpu_objective;
worker CPU631.339s, parentfreezeserializationmean10.038ms, workerresultpickup146.898ms.
GenerationRPCchannelwait0.026vs0.055ms; responsepickup254.727vs3.251ms; replies~1781B
both. RPCpickupusesproducerwallclock/clamp; primary12-stageidentityusesmonotonic.
Pureplanningalreadyoffloaded; no redundant offload/TTL/fallback/threshold-relaxation patch.
Relevant D93/D188/D193-D204/D328-D331/D342/D350/D355/D374-D388 history and its
prior-planning-owner references cross-checked (not every intervening experiment);
Python3.12asyncio and Llumnix observation-path officialdocs checked. Current full
file-preparation inventory is synchronous; actual share/cause remains unmeasured.
Full-history coordinator export occurs at end, notliveprogress; not a valid assumedcause.

92CPUtests passed0.498s; no skips. Firstnewentry ImportError beforedata/output retained
with exactfailedscript; corrected toexistingtiming_stats and importregression. NoGPUretry.
Curated main_comparison/d455_stage_diagnosis contains byte-identical stage/RPCplanning
summaries, closure and deliveryverification. D455_CONTROLLER_STAGE_DIAGNOSIS.md
records completephase/stagetables and limits. Projected8000rows retainedraw, notGit.
Delivery check139refs/protected147/threeuserSHAs/protocols passed. SixownedCPUdomains
and verifier7 exactInvocationID/actualrecursiveemptyclosed; allhigh/max/OOM/swap0,
maxanalysispeak477192192B. NoGPUcomputeprocess. Lastdisk161830854656B(~150.72GiB),
freshheavygate required. Explicitbackupcommit/remotematch recorded in rawD455
backup_receipt.json afterpush; BASE remains71c52cf6f7331dc2a171ec83db5f1011eaf6d5a6.

NEXT: establish current function-level cause of controller-delaygrowth. Reuseexisting
D328 python_frames_v1 wrapper/analysis for one D371-config originalprefix1000 diagnostic
if no retainedcurrentstackcoversit; no newpolicy/threshold/profile/capacity. Register
andresourcequalify beforelaunch, keep diagnosticnotranking identity; no repeatD328's
olderC8 point or speculative optimization. Then one evidence-backed formula-compatible
candidate, tests/minimalvalidation/fullrun. Remaining7BW1/W2/reference/G2,3B,HydraServe
externaldependency,A1-A5/S1-S13/docs/figures allstillOPEN. GoalACTIVE; currentturn
substantiveofflineevidence/verifieddelivery progress, not blocked or entiregoal complete.

## Previous action — D454 offline comparison and control diagnosis completed

Read full PLAN/status/baseline rules/V1/V2; no runtime/configuration/policy change,
new GPU run, baseline optimization, remote administration or figure. User again
approved once-only readonly delivery cache: reuse completedD78/D80 (1581215261B),
no republish or extractedpool. This authorization and boundaries are already inPLAN.

Extended existing matchedW0 analyzer to add sealedD453 while preserving all five
D446 rows, relative observations and Resident reference byte-for-byte. Same native
controller row/binding checks for Full and ElasticLocality; all4000inputs/458224native
tokens/132logicalIDs match, contractSHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.
D453 E2E recomputed/checked against every request, not fabricated as an oldseal field.
Raw n_correct=null and absent independent numerical proof remain explicit.
Samebackend:3812equal/188changednativeoutputSHA across80adapters; all changedIDs
retained, no first-divergence/logits proof or automatic rounding/batching explanation.

MatchedW0 jointSLO / physicalGPU-s per boundrequest / Type1P95TTFTms:
PrimeLoRA .977 /3.6206309810630626 /2837.895684875548;
ServerlessLLM .98125 /4.022440685881069 /1052.2463605739176;
vLLM .99475 /4.028844290537294 /1200.5068596918136;
S-LoRA .94425 /3.7892694676132526 /17699.067366309464 (nonfeasible);
Loquetier .0435 /4.249595808607468 /555110.0322289858 (nonfeasible);
ElasticLocality .95775 /2.4731789675242615 /2420.6078292336315.
PrimeLoRA uses46.3958% moreGPUtime than feasibleElasticLocality, with17.2390%
higherP95TTFT but77morejointgoodrequests (+1.925points). Thus currentFull is NOT
lowestresource measuredW0 point; no winnerfilter/rerun/thresholdchange. Composite
internalcontrol, not single-factor causality. Firstfour share132fetch/131480060wireB,
so thisW0 does not demonstrateFull reducingremote transfers versusElasticLocality.

Reused nativecontrol events and physical intervalunions, verified identical controller
config, commonarrival3963.908903s. FullminusElastic total4589.808054GPU-s:
arrival+4717.854033, prearrival-9.024056, drain+7.979029, cleanup-127.000953.
ArrivalmeanphysicalGPUs3.631634vs2.441432; fourcardwalltime3022.505544vs979.476777s.
Withinready>1samples, TTFTnotlow1119vs80; alllow605vs990. Blockers overlap,
samplecounts are NOT elapsedtime. Resourcegap primarilyduringbusiness, notstartup/
cleanup; feedbackpath localized, routing/preparation/admission cause NOT established.

53 CPUtests passed0.374s; actualjoin successful. First controlprojection KeyError
last_arrival_offset_s retained; v2 reads existingFrozenReplayPlan lastentry/identity
and succeeds. No performance rerun or fallback. Executedv2 script byte-identically
backedup as scripts/ieee_tc/analyze_d454_control.py. Curated
paper_results/ieee_tc/main_comparison/d454_7b_w0 and D454_MATCHED_W0_ELASTICLOCALITY.md.
Delivery verification passed137source refs/protected147/threeuserfileSHAs; all5owned
CPUdomains exactInvocationID/actualrecursiveemptyclosed,high/max/OOM/swap0.
No computeGPUprocess remains. Lastdisk161856069632B (~150.74GiB), freshheavygate
required. BASE unchanged71c52cf6f7331dc2a171ec83db5f1011eaf6d5a6. Explicitbackup
commit and remote equality are recorded in raw d454_20261012/backup_receipt.json
after push verification; never stage usermanifest/unrelatedfiles or rawlarge data.

NEXT: PrimeLoRA-only causal diagnosis of persistent TTFT scale-in feedback using
existing request stages, routing/preparation/admission events, D374-D388 history and
primary-source implementations. Do not first relax thresholds, blindly repeatC8,
or call disabling allFull mechanisms an optimization. Register one falsifiable
formula-compatible candidate only with evidence, then necessaryvalidation/fullrun.
7BW1/W2/G2 inputs/references and comparison remain open before3B; HydraServe
externaldependency/fullremaining A1-A5/S1-S13/docs retained. GoalACTIVE,
substantive offline analysis/verifieddelivery progress, not a whole-goal blocker.

## Previous action — D453 full4000 closed, audited

Same single launch completed2026-10-12 09:19; no restart or second performance run.
4000/4000 native/source/adapter/lease contracts passed,458224native tokens,
zero failed. Independent retainedD435 input witness matches all4000 canonical
prompt hashes/native prompt-ID hashes/targets/adapter/source rows. Raw n_correct
stays null; complete binding evidence is not independent kernel numerical proof.
Sourcecommit0cddb850ce4d79333057da909afb4a11ffa930f6; sourceSHA
f758f48cb1231de7a86025a1cdda777c2285dd2c892b208e9b21e5691c47fb50.

Commonjoint3831/4000=95.775%, above95% threshold. MeanTTFT1003.697864922ms,
Type1P95TTFT2420.607829234ms; meanqueue528.828450925ms,
meanservice474.869413997ms; meanTPOT29.212304974ms/P95TPOT54.523438402ms.
Actual lifecycle9892.715870097047GPU-s=2.4731789675242615GPU-s/boundrequest.
These raw resource/tail values are lower than sealedPrimeLoRAW0, while jointSLO
is lower thanPrimeLoRA97.70%. Do not hide this result or repeat until it loses.
Matched six-system contract and same-backend output-difference audits remain NEXT;
qualification package ranking_eligible=false pending those comparisons. NotG2/globaloptimum.

Dispatchremote/GPU/HOST/NVMe132/2706/315/847. Routing4014selectionattempts:
GPUhit2710/slack185/nohit1119/empty0. Proactiveplans0/fileLRUreplacements0.
Initial1 ready; natural33triggers=30runtime_ready+3cancelled, not33completed
scaleups orfirstservice. Initialactivation41.894134533s.
132realHTTPtransfers/131480060wireB, two-endUUID/SHA/bytes match;
request-timepacking0. Reconfirmedcacheapproval reusesD78/D80, no republish.
4040resourcesamples/peak20717527040B, high/max/OOM/swap0. Service/watchdogexit0,
actualGPUrelease and servicepathremoved before exactremote3B/7B/monitorstop;
portsabsent,downloadSHAverified. Start/replayaux actualrecursiveempty checked and
closed; postrunaux4/5/6 closed by recordedInvocationID; closer7 alsoactualempty/
identityclosed. Lastdisk161934061568B (~150.81GiB), newheavyneedsfreshgate.

Curated elastic_locality/d453_actual_full_w0.json preserves qualification,
independent input witness, closure, source hashes and limitations. No figures,
newweights/trace/pool,baseline tuning,oldoverwrite or remote cache regeneration.
FullPLAN/status/baseline rules/V1/V2 reread;147protected entries and userfiles
unchanged by auditors. Deliveryverification passed26independent source refs and
protected147;curatedSHA3cb2e9254941125f4f4d53465c682829fc0cac2127c1cf71c6b059b5e48be9de.
Verificationaux8 InvocationID92994ba51da84c1dab3d50dbec056376 alsoactualrecursiveempty/
identityclosed,high/max/OOM/swap0. Allownedtasksclosed. Checkpoint
cab0de35e69db34e8907c7217c75c7085ab99c83 pushed/remoteSHAequal; rawbackup_receipt.json
records this closure and delivery evidence. Only protected usermanifest remains
tracked-dirty apart from this post-push status note.
NEXT: extend existing matchedW0 analyzer with this sealed row, audit native
output digests under the same backend, preserve all five prior rows; then7B
W1/W2/G2 with qualified commoninput plumbing and justifiedPrimeLoRAoptimization,
before3B. HydraServe access dependency and fullremaining scope unchanged.
Goal ACTIVE: substantive run completion/closure/audit progress, not blocked.

## Previous action — D453 launch and verified monitoring

Previousturn substantiveprogress: D452 actual1000closure/audit/backup, no blocker.
D452 checkpoint594c3e026d2c45465f1cad462ae7cad8271c2050 pushed and remoteSHAequal.
D453_ELASTICLOCALITY_FULL_W0.md registers one unchangedconfiguration full4000:
new output/NVMe/HOST paths and existinglauncher d453/PREFIX_COUNTempty only.
No runtime/algorithm/safety/testsource change; no baselineperformanceoptimization.
Exact D435 original4000 entries,132actuallogicaladapters/458224targettokens,
fullviewSHAa5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
20 replaytests passed0.678s; shell/Python syntax/configequality/source checks pass.
Protected147/protocols/userfiles/D452sealedresult intact. CPUaux1/2 exactidentity/
actualrecursiveemptytreeclosed, high/max/OOM/swap0. Curated
elastic_locality/d453_full_input_qualification.json. Checkpoint
0cddb850ce4d79333057da909afb4a11ffa930f6 pushed and remoteSHAequal verified.
Actual launched08:11, tmux tc-d453-localityw0/panePID2346473; serving4000 after
initial engine-ready08:12:05. Fresh32GiBgrowth/preflight passed, no issues;
realremote1000Mb/s/full, D78/D80cache reused, no republish. Firstrequestsarriving,
no performance result yet; watchdog high/max/OOM/swap0/no abort in earlysamples.
Startauxd453...0003 InvocationID763fe7ffa9a847bcb7606d4e9802d308;
remote3B fe60f4af1a3c4592aba1b3d1a7313e81,7B372e01863f014a3db24e87e59e25d88d;
monitor primelora-artifact-monitor-d453-localityw0.service InvocationID
65c4da1e7f324b8cba2c6f0d1548620e. Exactidentities in launch_gate/remote_start.
Do not relaunch, test, hashscan, build, clean or remotelymaintain duringmeasurement.
Remote must stay until actualGPUrelease. Reuse close_after_release.py then existing
analyze_prefix.py and cross_contract.py prepared for4000; no result claims beforeaudit.
Afterclosureaudit, compare samecontracts and continue7BW1/W2/G2 before3B;
HydraServeexternalqualification and allfullscope retained, not repeatD450–D452.

Latest continuation: previous goalturn classified PROGRESS (D452 seal/backup and
D453 actuallaunch); this turn VERIFIED_WAIT on live tmux pane/PID2346473 python3,
rechecked with ps at processelapsed705s. FullPLAN/status/baseline rules reread;
monitoring only, no runtime/config changes, tests, heavy analysis or remote action.
At replayelapsed10:44: arrived518/4000, completed515/4000, zero failed;
live counters scale_up4/scale_down4 (not yet audited activation/first-service counts).
Latestwatchdog noabort/no warning; memory.peak13598650368B,high/max/OOM/swap0.
Lastdisk161136713728B (~150.07GiB), above runtime100GiBfloor; nextlaunch requires
fresh150GiBstartgate, no cleanup duringmeasurement. Same single run remainslive.
No final4000metrics or ranking yet; continue monitoring samehandle, never relaunch
because a polling interval elapsed. Goal ACTIVE, not a repeated blocker or complete.

Next continuation VERIFIED_WAIT: same tmux/PID2346473 confirmedlive repeatedly,
pselapsed1228s at lastcheck; replayelapsed19:25,arrived1114/4000,completed1110/4000,
zero failed. Natural livecounters scale_up10/scale_down7,current4runtimes.
Latestwatchdog high/max/OOM/swap0,no warning/abort. Backlog temporarily increased
near successive rotation/arrival phases and then drained; this is partial observed
behavior, not finalSLO or an isolated mechanism attribution. No code/config change,
tests, heavy analysis, remoteadministration, cacheclearing or rerun duringmeasurement.
Continue the same4000 run; wait is an active experiment dependency, not a goalblocker.

Ongoing same-run checkpoint: tmux/PID2346473 live, processelapsed2818s,
replayelapsed45:56,arrived3062/4000,completed3060/4000,zero failed,current3runtimes.
Live scale_up25/scale_down23 are provisional counters, not audited epoch counts.
Latestwatchdog noabort/no warning/high/max/OOM/swap0. Last explicitdisk sample
160184565760B (~149.18GiB); above100GiBruntimefloor, not a newlaunchclearance.
No live maintenance, tests, runtime changes or rerun. Same full4000 still running.

Read-only next-mainline finding during wait (no edits/tests):
faaslora/datasets/workload_generator.py FrozenReplayPlan.load already implements
approved W1 half-interval/30sphase indexing and rejects unfrozenW2 mapping.
BASE scripts/prepare_ieee_tc_serverless_stack.py load_http_replay_plan line771
calls FrozenReplayPlan.load without profile, hence current HTTP path still W0;
its qualified populations cover trace_prefix_v1 and adapter_coverage_v1.
After D453closure and matchedW0 analysis, qualify an explicit common W1 input
interface using the existing transform, preserving all baseline service policies/
configuration and old W0 identity. Do not assume setting a profile label already
changes HTTP arrivals; no new trace/weights or baseline tuning authorized.

## Previous action — D452 original1000 completed and delivery verified

Previous goal turn made substantive progress: actualD451 closure/audit and backup,
not no-progress or a wait. Commit520e23af5f9e4d82b630a252f567a2c7779df331 was pushed
and remoteSHAequal verified. FullPLAN/status/latestbaseline rules/V1/V2 reread.
D452_ELASTICLOCALITY_ORIGINAL1000.md preregisters next existing qualification:
D450 frozenconfig path-only clone, existing launcher d452/PREFIX_COUNT1000 branch;
no algorithm/runtime/control/metric changes and no baseline optimization.
Exact D434 legacy1000 entries:60adapters/122510tokens, viewSHA0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981.
Both historical source paths have samefileSHAefb903...; no regenerated input.
20 existing replay tests passed0.633s, shell/Python syntax and config equality pass;
protected147, protocol/userfiles/D451 sealedresult intact. CPUaux1/2 exactidentity/
actualrecursiveempty stopped, high/max/OOM/swap0, peaks18550784/132988928B.
Curated elastic_locality/d452_prefix_input_qualification.json. Checkpoint
57e7b13623d6c67310abc6580742ea25a7a2e495 pushed/remoteSHAequal; actual original1000
launched07:40:58 in tmux tc-d452-locality1000, panePID2272004 alive/python3.
Fresh32GiBgrowth gate passed, realremote1000Mb/s/full, existingD78/D80 cache reused.
COMPLETED same launch once:1000/1000native token/source/adapter/lease bindings,
122510tokens, zero failures. D434 retained input journal independently matches all
1000 prompt hashes/native prompt-ID hashes/targets/source rows. Raw n_correct=null
and absent independent kernel numerical proof stay unchanged. Not full4000/G1/G2.
Commonjoint878/1000=87.8% (below95%); meanTTFT1856.451311ms,
Type1P95TTFT11802.663969ms; meanqueue1238.101214ms/meanservice618.350097ms,
P95TPOT66.677047ms; physicalGPU2781.422323s. No baseline tuning or rerun.
Dispatchremote/GPU/HOST/NVMe60/646/87/207;1005selectionattempts,
GPUhit646/slack33/nohit326; proactiveplans0/fileLRUreplacements0.
Activationinitial1; natural11triggers comprise9runtime_ready and2cancelled with
nullreadytime. Do not confuse triggers with completedscaleups orfirstservice.
Initialactivation41.238201s. RealHTTP60transfers/74953014wireB; two-endUUID/SHA/
bytesmatch; request-timepacking0; D78/D80existingdeliverycache reused.
1179resourcesamples/peak19297902592B,high/max/OOM/swap0. Service/watchdogexit0,
actualGPUrelease/servicepathremoved before exact-owned remote3B/7B/monitorstop;
portsabsent, downloadedjournal/monitorSHAverified. All8 priorownedCPUauxclosed
only after exactidentity and actualrecursivecgroup.procs empty checks.
Curated elastic_locality/d452_actual_prefix1000.json retains rawaudit,inputwitness,
activationdistinction, closure and allsourceSHA. Postrun auditors passedfirstattempt.
Finaldeliverycheck passed25source refs/protected147; curatedSHA
42163de84e0ecb4e4dc7ea3d33c6b257698a4b0370b9b614f4359116413cc210.
Verificationaux9 InvocationID01dd96192242429d85d8177cbe526366 also actualempty/
exactidentityclosed,high/max/OOM/swap0. No performance tasklive.
Latestdisk162227302400B(~151.09GiB), fourGPUcomputeprocessesabsent;
freshnextheavygatesrequired. Protected147/protocols/userfilesunchanged.
NEXT: one original4000W0 at same frozenconfig; no repeatD450/D451/D452.
Sealedfive-systemW0,remaining7BW1/W2/G2,3B and fullscope unchanged.

## Previous action — D451 existing500-ID view integration and actual coverage completed

D451_ELASTICLOCALITY_500_ADAPTER_COVERAGE.md records the next required coverage,
not a repeat of passedD450. Same D450 runtime/mechanisms, path-only config changes;
existing D433/D443 adapter_coverage_v1 is integrated with guarded Unix ingress.
Derived rows preserve original arrival/prompt/target, map sorted frozen500 IDs,
and carry source/derived/pool SHA; no replacement trace, weights or pool.
CPU/input qualification passed:268 tests54.029s +4 pidfd tests0.005s. Exact legacy
500rows/viewSHA286f7547...,68925tokens; non-path D450 config equality; only main
input-loader definition changed,596others unchanged. Protected147/userfiles intact.
Six owned CPU scopes actual-recursive-empty/identity closed, high/max/OOM/swap0;
stdin-selector spawn failure retained and stopped143; no production fallback.
Curated elastic_locality/d451_coverage_input_qualification.json. Checkpoint
6cb9bf6e3d29984de24967f9d9f8c36b8a115501 pushed/remoteSHAequal verified.
07:11:38 actual500-ID started once; completed07:27 with service/watchdog exit0,
actualGPUrelease and service removal confirmed. 500/500 native token/source/adapter/
lease bindings,68925tokens; independent D433 retained request-contract witness
matches all500 canonical prompt hashes/native prompt-ID hashes/targets/derived rows.
SourceSHA4668afee08c0f5e241d684098eaebf5856b1dad06dd2b2462b5bc40fad270fea.
Raw n_correct=null remains; binding evidence is not independent numerical proof.
Full4000/W0/G1/G2 qualification NOT asserted.

Frozen common jointSLO186/500=37.2%; meanTTFT58534.608611ms,
Type1P95TTFT168610.997153ms; meanqueue16565.516369ms,
meanservice41969.092242ms; P95TPOT36.373022ms;
physicalGPU2579.828998s. Initial1/natural_scaleout3, initialactivation43.483525s.
Every adapter occurs once: dispatchremote500, GPUlocalityhits0;
693 routingattempts, proactiveplans0, fileLRUreplacements0. No locality/eviction
performance inference from this functional coverage or comparison to prefix100.
HTTP500/420740410wireB, both-endUUID/SHA/bytes equal, no request-time packing.
User reconfirmed once-only compressed-cache approval: existingD78/D80 reused,
no republish, no extracted pool, all real reading/transfer/materialization retained.

931 resource samples, peak28073377792B, high/max/OOM/swap0/no watchdog abort.
After GPUrelease: exact-owned remote3B/7B/monitor stopped; both ports absent;
downloaded transfer/monitor SHA verified. Start/replay auxiliaries and postrun9–14
closed only after exactidentity and actual recursive cgroup.procs empty checks.
No task remains running. Postrun analysis-only failures retained: wrong interpreter
path exit127 before analyzer; wrong source-item attribute exit1, separate v2 uses
existing source_sha256. No GPU rerun, runtime fallback, tuning or figure.
Curated elastic_locality/d451_actual_coverage500.json SHA
42e651c96babce8e65fad79314de5470ee7a46343d0a04662fafde80f05fcacf;
all24 audit refs independently checked. Delivery verification scope15 also
exactidentity/actualrecursiveempty closed; high/max/OOM/swap0.
Protected147, frozenPLAN/V1/V2 and three userfile hashes unchanged.
Latestdisk162334121984B (~151.19GiB); runtime already released privateNVMe cache
(empty); no manual deletion needed. Fresh safety gates before next heavy task.
NEXT: original1000 compatibility at same frozen configuration -> single4000W0;
sealed five-systemW0 and every remaining mainline task retained. No repeatD451.

## Previous action — D450 combined contract/configuration CPU qualified

Repeated user once-only cache approval reuses D78/D80 (1581215261 archive bytes),
no republish/extracted pool. Full PLAN/status/baseline rules/V1/V2 read unchanged.
D450_ELASTICLOCALITY_COMBINED_QUALIFICATION.md and
paper_results/ieee_tc/elastic_locality/d450_combined_qualification.json record
one complete explicitly named variant, slack1/no search, actual router/file-owner
binding, proactive diagnostic rejection, negative/positive mechanism auditor,
and exact non-target config equality with D371. Existing dispatcher has d450
prefix100 branch; ordinary native demand/IEEE physical control/formulas unchanged.
Runner AST: four contract/observation definitions changed, one helper new,
592 existing definitions unchanged. No baseline performance tuning.

Final621 CPU regressions passed58.387s;38 offline analyzer tests passed0.20s.
Focused sets overlap. Retained failures: LLM has no pytest (no tests); native-env
whole control suite spawned dependency build processes and was identity-stopped
at high38275/peak3441676288B, no max/OOM/swap; new synthetic control fixture
missed coordinator metrics (test-only fix); audit packaging initially queried
nonexistent missing field (fixed to existing pass/changed contract).
All9 owned auxiliaries exact-identity/actual-recursive-empty checked and stopped;
other scopes high/max/OOM/swap0. Protected147/userfiles/D371/protocols unchanged.

D450 checkpoint468411b4ef8833855c7aa694b71e7fce7ac5ce93 pushed; remoteSHAequal.
COMPLETED: originalprefix100,100/100native contract/source/adapter/lease bindings,
17369tokens; initial1/natural_scaleout3; all proactive plans/CPUwork0.
Dispatchremote/GPU/HOST/NVMe36/41/6/17. Routing105attempts (41hit/12slack/52nohit),
not105requests. FileLRUreplacements0 (no actual eviction-performance claim).
RealHTTP29transfers/50650865wireB, both-endUUID/SHA/bytes match, no request packing.
Mean/P95TTFT39240.140/83290.886ms, meanqueue38045.182ms; P95TPOT56.098ms;
commonthresholdprefixjoint17/100, physical476.847596GPU-s. Initialactivation150.331273s
exceededcommon60snotice. Waiting retained, no latency subtraction, no causal attribution
to locality, no baseline tuning. Functional/mechanism qualified, NOT SLO/G1/G2.
Raw n_correct=null and numerical-proof limitation retained. Full output-digest
comparison not done here; preserve for corresponding input comparison.
Curated elastic_locality/d450_actual_prefix100.json. SourceSHA
d47de6926690aea881d88f65f39a0aeb3ddff64bfaf2db63639abe234b5563e0.
258resource samples, peak19277647872B, high/max/OOM/swap0. ActualGPUrelease and
service removal passed, then exactremote monitor/3B/7B stopped; source/log SHAs verified.
Start/replay auxiliaries closed by reusedclosure;collection11 andanalysis12 stopped
after exactidentity/actualrecursiveemptytree. No task remains running.
Latestdisk162432045056B,fourGPUs15MiBbackground/0util;fresh nextheavygatesneeded.
Protected147 unchanged. No oldresultoverwrite,newweights/trace/pool,republishing,figures.
NEXT: 500-IDcoverage on existing input view (check native runner compatibility),
then original1000 ->single4000. Do not repeat D450 prefix without an affected-contract
reason. Existing five-systemW0 stays sealed. W1/W2/G2/3B and fullscope still open.
Raw: /home/qhq/serverless_llm_experiment/results/ieee_tc/elastic_locality_qualification/d450_20261012.
Goal ACTIVE, substantive implementation/qualification progress; no whole-goal blocker.

## Previous checkpoint — D449 implementation/CPU boundary checks completed

Read full PLAN/status/latest baseline rules/V1/V2. D448 MAIN backup
ba9775e2be55ea67cf76ae96f1ee47fab821da15 previously pushed/remoteSHAverified.
BASE remains71c52cf6f7331dc2a171ec83db5f1011eaf6d5a6 unchanged.
D78/D80 approved readonly delivery cache reused; no remote action or republish.

See D449_ELASTICLOCALITY_FILE_CACHE.md. Existing demand file admission had no
ordinary eviction when a PrimeLoRA preparation objective was absent. Added explicit
opt-in demand_lru to existing LocalSourceReferences and ExperimentStack. Default
planned_only and Full benefit replacement/native transactions unchanged.
Publication/new read lease update recency; observations/repeated same lease/release
do not. Victims selected completely before deletion under physical allocation lock;
pinned/moving/nonexclusive/unsettled bytes excluded, archive+payload counted,
HOST native/activation allowances retained. Insufficient/oversized transfers fail
explicitly; no early partial eviction, hidden budget bypass or fake rollback.
Ordinary LRU rejects proactive file plans/objectives and owned planner entry.
Assembly requires explicit IEEE ownership/no_handoff/hierarchy false/coordination false.
Runner change only adds file-cache policy/replacement metadata.

Initial10 checks failed in test setup because the helper tried replacing its frozen
manifest; corrected only by constructing a fresh test client. Failure retained.
Focused10passed0.322s. Integrated612passed58.935s. Then two further HOST/deletion
checks plus replacement regressions183passed11.460s; overlaps, not795 unique tests
or performance repeats. No skips. Actual tiny file allocation/publication plus
synthetic HTTP/native fixtures; NOT GPU arithmetic/realremote/performance proof.
AST compared to D448:127other file-owner/module definitions,43stack and576runner
definitions unchanged. Formula/benefit/native-demand/control functions unchanged.

Raw originalresults/ieee_tc/elastic_locality_qualification/d449_20261012.
Curated paper_results/ieee_tc/elastic_locality/d449_file_cache_boundary.json.
147 protectedentries,three user-file hashes,D371 config unchanged.
Tests/analysis GPUhidden/CPU2,3,26,27/high3max4GiB/swap0; first6 exactInvocationID
and actualrecursiveemptytree-checked scopes stopped, high/max/OOM/swap0.
Peaks573165568/573853696/2218246144/573853696/122744832/31297536B.
Packaging/sealing scopes0007/0008 also exactidentity/actualemptytree stopped,
InvocationIDs6be917428b074b41b50764f3edc6f4c8/aff081462a154f26819cd5bdcc5260eb.
All22 source records independently verified; archivedstatus byte-identical toD448.
All8ownedscopes closed, high/max/OOM/swap0. No task remains running.
D449 code/evidence checkpoint48265b8cb4edad98f99e90c2035b6e20824b704e pushed
tofaaslora_origin/retry14_continuous_queue_v2; remoteSHAequalverified.
Onlyprotecteduser manifest remains trackeddirty. Lastdisk162459295744B(~151.31GiB),
fourGPUs15MiBbackground/0util; freshheavygatesrequired, notautomaticlaunchpermission.
No GPU performance, baseline tuning, new weights/trace/pool, oldoverwrite orfigures.
Goal ACTIVE: substantive implementation/CPU evidence progress, not a blocker
and not completed ElasticLocality or complete experimental goal.

NEXT (D450): integrate/freeze one full ElasticLocality contract: D448 gpu_locality,
slack1(no search), D449 demand_lru, no_handoff, hierarchy/coordination false.
Jointly verify automatic startup/steady preparation exclusion, diagnostic bypasses,
native demand/physical safety, unchanged Full autoscaler/backend/resource/input,
mechanism metadata and existing main analyzer/launch identity. No runnable whole
campaign has been published yet. Then affected actual100 ->500-IDcoverage ->
original1000 ->single4000, using existing runners and frozen inputs.
Do not repeat sealed Full/five-systemW0 or standalone passed witnesses without
new changes. Remaining7B/W1-W2/G2/necessaryPrimeLoRAoptimization before3B.

## Sealed W0 evidence — do NOT rerun or declare global optimum

D446_MATCHED_7B_W0_FIVE_SYSTEMS.md and main_comparison/d446_7b_w0/ retain all
five rows. Matched4000requests/458224native tokens/132logical IDs,
contractSHA384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.
D371/D397 PrimeLoRA: joint97.70%,3.6206309810630626GPU-s/request,P95TTFT2837.8957ms.
D414 ServerlessLLM:98.125%,4.022440685881069,1052.2464ms.
D423 vLLM:99.475%,4.028844290537294,1200.5069ms.
D435 S-LoRA:94.425%,3.7892694676132526,17699.0674ms; NONFEASIBLE.
D445 Loquetier:4.35%,4.249595808607468,555110.0322ms; NONFEASIBLE.
PrimeLoRA about10% less resource than two feasible baselines, but tail/joint worse.
Not all-metric optimum, independent holdout, full numerical proof or G2.
Raw numerical unknown flags remain; logical/native/source binding is distinct.
Firstthree132fetches/131480060wireB each; lattertwo full500/420740410B.
Preparation difference is not isolated hierarchy/confirmed-state evidence.
500logicalIDs,498zero/2nonzero,4weightSHA/6treeclasses: disclose diversity limits.
Loquetier native static4x125 shards, batch2/context1024; no performance tuning.
D432-D435 and D438-D445 qualifications/receipts remain archived, not repeated.
Owned old download caches reclaimed only after backups/seals; originals/raw retained.

D394 common SLO: <=616input TTFT2995.348713684507/TPOT59.74409694902162ms;
>616 TTFT4633.345350589479/TPOT85.24698166015581ms.
Frozen manifest protocols/20261010_7b_slo_d394_frozen.json,
SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
W0 Uref16115.377162149176,Qref1200.5068596918136ms,B12086.532871611882.
W1/W2 references and G2 remain OPEN. No refreeze or short-run extrapolation.

## HydraServe — limited external dependency; continue other mainline work

D447 actual rootlessPodman3.4.4 metadata succeeded: no images/containers,9CDInames,
cgroupv2 memory/pids delegation. k3s1.36.4 active but root-owned; qhq lacks readable
kubeconfig/Docker socket, kubectl/helm root0700. No GPU container/K8s qualification.
Asked whether cluster belongs to project and for an existing qhq-readable kubeconfig
path; unanswered. Do not repeat query/probe or modify unknown cluster/permissions.
NativeHEAD8ae605de354ccfa2e9095514cdb4a9e9c56aa56b clean and preserved.
D447 CPU witness: stage-local backbone names/globalLoRA lookup differ; inert2->1
sharing witness, not nativeGPU/numericalproof. Actual adapter propagation still
needs necessary qualification/adaptation; never substitute bare vLLM and call it HydraServe.
BASE reviews HYDRASERVE_REUSE_REVIEW_20261011.md,
HYDRASERVE_LORA_IDENTITY_REVIEW_20261012.md and
D447_HYDRASERVE_PLATFORM_AND_LAYER_WITNESS.md remain required before next change.
No whole-goal blocker; remaining ElasticLocality/W1-W2 work is actionable.

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
