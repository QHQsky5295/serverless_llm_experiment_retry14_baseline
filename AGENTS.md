# PrimeLoRA IEEE TC execution contract

Current handoff 2026-11-08: D1121 COMPLETE/released/analyzed; no live GPU run.
Read D1121_7B_FIXED_FULL_W1_RESULT_20261108.md. 4000 native contracts;
joint3970/4000=99.25%, U8931.497721683234, P95TTFT1823.3526549302042ms.
Same FullD1115 saves14.8645%U but joint-0.70pp and P95+27.8595%; trade-off.
Four initial activations/closed leases, no natural resize;4000 actual selectors
match; Full soft admission/residency retained.132transfers/131480060bytes,
same as Full; no reduced-download attribution.180outputdigest differences
retained/unknown cause,n_correctnull. Native contracts not independent math proof.
Remoteinactive/GPUcomputeempty;2279samples,nohigh/max/OOM/swap. No GPU replay.
Next seal/source-protectioncheck/explicitbackup, then7B LastKnown registration;
do not reopen baseline/HydraServe deployment or switch to3B. Fullscope active.
This goal turn PROGRESS: observed terminal/release and completed bounded analysis.
All older LIVE paragraphs below are historical, not instructions to relaunch.

Latest verified wait2026-11-08 01:45:38+08: D1121 SAME panePID2381434live.
2828requests complete/success/native-contract,latestsubmittedindex2836;
sample1465events/swap0,no warning/abort/foreign/escapedcompute;heldGPUs4,
available96.998680GiB,disk157.866856GiB. PriorgoalturnVERIFIED WAIT;thisturn
VERIFIED WAIT, no new run or production edits. Continue same run; no relaunch.
Fullscopeactive. Do not treat phase completion as final result/early G1 verdict.

LIVE handoff2026-11-08 01:21+08: D1121 FixedFull7BW1 launchedONCE,
Latest01:22:39sample105: same run,29terminalrecords,4allocationfiles;
no warning/abort/events/swap/foreigncompute,available106162388992B.
tmux tc-d1121-fixed-full-7b-w1 panePID2381434 live. Do NOT relaunch.
Checkpoint3cf8d0ab8db8d3a818524857c647b86aafcec73e pushed/remotematched.
Service36a01c1a0b1a418096b0cab1f54f9176, invocationffd56661026a423f927b595ce1c64196;
aux50b8648653534387829e16bd959cc612; notice3579300.885729723,
replay3579360.885729723. Remoteclockf4c2de47a858483fa4eb96e73207b82e.
OwnedremoteInvocationIDbf7d9c419df548fb80c7079eccff4d4d/PID1351838.
Sample42noabort/high/max/OOM/swap/foreigncompute,disk172112363520B.
Monitor SAME run through actualrelease; no tests/productionchanges/maintenance/
hashscans/push/remotechanges during measurement. Existing D1113/D299/D455 after
release; fourinitial/noresize checks and sameFullD1115. No complete result yet.
ThisgoalturnPROGRESS: D1120sealed, D1121registered/tested/backedup/started.
PriorNOTSTARTEDnotes below are historical. Fullremaininggoal active.

Current handoff: D1121 current-version7B FixedFullW1 REGISTERED, not started.
Read D1121_7B_FIXED_FULL_W1_REGISTRATION_20261108.md. Only min1->4 plus
unique paths from D1100; max4/scenarioFull/serving mechanisms unchanged.
Four focused CPU tests passed; no production edit. D1120sealed/pushed3185c417,
remoteHEADmatched. Next explicit checkpoint then fresh resources and launchONCE
using existing7argumentD1100runner in tc-d1121-fixed-full-7b-w1. No Full/baseline
replay/no3Bswitch. Afterrealrelease collectjournal/metrics and fourinitial/noresize
mechanism evidence; retain trade-offs and unknownnumericalcorrectness. Fullscopeactive.

Latest handoff 2026-11-08: D1120 COMPLETE, analyzed; no live GPU run.
Read D1120_7B_ELASTIC_LOCALITY_W1_RESULT_20261108.md first. Do not rerun.
4000 native contracts; joint3977/4000=99.425%; U8497.62170464173;
P95TTFT1092.9982741363347ms. FullD1115 saves10.5176%U but joint-0.875pp,
P95TTFT+113.2967%; neither max4 point meets W1G2budget. Trade-off, not all-metric win.
4000 actual locality selectors pass; proactive plans all0;132realHTTPtransfers,
131480060bytes, same as both references. 173outputdigest differences vsFull
retained/unknowncause; n_correctnull remains. Twelve leases released,remoteinactive,
no GPUcompute,high/max/OOM/swap0. Comparison reuses D1119 ElasticOnly unchanged.
Next seal/backup then register current-version7B FixedFull W1 using existing gate;
not Full/baseline replay, no3Bswitch. Remaining A1-A5/S1-S13/docs/figures ACTIVE.
Older LIVE/NOTSTARTED notes below are historical and superseded.

Latest verified wait00:33:19+08 Nov8: D1120 SAME panePID2234747 live,
1162arrived/1155done/1155ok/native-contract,7backlog;phases1/2 complete500each.
Sample727events/swap0,disk158.936485GiB,available99.516819GiB,noabort/foreigncompute.
Current/previousturn VERIFIED WAIT. Continue SAME run; no relaunch,
changes/maintenance/push. Full remaining scope remains active.

LIVE handoff2026-11-08 00:22+08: D1120 launched ONCE, same panePID2234747
confirmed live00:21:59. tmux tc-d1120-elastic-locality-7b-w1. DO NOT relaunch.
Read updated D1120 registration. Checkpoint7efef8de9bed03d96c02a3fdcab64a9377d833b7
pushed to V2 and remoteHEAD matched. Service456975b4e4004f8a8bbec5bb966d1d92,
aux22dd636a092f41ed813158d0fa9a5bea;notice3575710.468862955,replay3575770.468862955.
Remoteownedb3e9398cc2a14d119ffcae71d4a4a2ae,clock8cd21813c05b4f88ab02803caa4e11e5.
Sample56 noabort/high/max/OOM/swap/foreigncompute;disk172449480704B.
No final performance result. Monitor same handle to terminal/real release;
no production edits/tests/maintenance/hashscans/push/remote changes while measuring.
Afterwards matching-clockjournal and registered existing analyzers/comparisons.
Prior goal turn and this turn are PROGRESS; full remaining scope active.
Below NOT STARTED notes are historical, not new launch instructions.

Current handoff 2026-11-08: D1120 current-version7B ElasticLocality W1
REGISTERED, NOT STARTED. Read D1120_7B_ELASTIC_LOCALITY_W1_REGISTRATION_20261108.md.
Existing gpu_locality/slack1; no production change or tuning. Same D1100/max4
non-targets as sealed FullD1115 and ElasticOnlyD1119. 23 targeted CPU tests pass;
144 protected files/user manifest/source result SHA unchanged. Next backup,
fresh existing resource/remote checks, launch ONCE in tc-d1120-elastic-locality-7b-w1
using registered seven-argument D1100 runner. No Full/baseline replay/3B switch.
After real release use explicit-scenario D1113/D299/D455, same-clock journal,
actual gpu_locality recomputation and two retained references. Preserve negative
results and native/numerical-correctness limitations. Full scope stays active.
Earlier D1119 next-registration instructions below are superseded by this note.

Latest handoff: D1119 analysis COMPLETE, no live GPU run. Read
docs/ieee_tc/D1119_7B_ELASTIC_ONLY_W1_RESULT_20261107.md FIRST.
4000 native contracts, joint3987/4000=99.675%, U8575.626704086084,
P95TTFT1035.5770019814372ms. FullD1115 joint98.55%, U7603.874677798245:
Full saves11.3316% physical GPU time but loses1.125pp joint and125.1237% P95.
Trade-off, not an all-metric win. Both max4 points fail W1 G2budget; cap3 remains
separate. Inputs4000match,179 output digest differences retained/unknown cause.
Selectors4000each match;ElasticOnly proactivefile/GPU/HOST/residency all0,
naturalepochs10vsFull19. Bothremote132transfers/131480060bytes; no download-saving
attribution. 30targetedCPUtests,43sourceSHA,144protectedfiles/userdirty unchanged.
All11leases released;remoteinactive;memoryevents/swap0. Next checkpoint backup,
then current-version7B ElasticLocality A1 (existing slack1, non-target equality),
not Full/baseline replay or3B switch. Remaining A1-A5/S1-S13/docs/figures ACTIVE.
Older LIVE/analysis-pending paragraphs below are historical.

Current handoff 2026-11-07 22:56+08: D1119 COMPLETE, pane1953868 dead/status0,
4000 successful, all11 leases released, GPU compute census empty. Do NOT restart.
U8575.626704086084; joint SLO not yet computed. Remote matching-clock journal
copied to raw/remote_transfers.jsonl; both-side SHA93f4ec19a36fd8b7a8a305f344e94d119200bf39102841f2735a4c201e7381ff.
Next bounded CPU D1113/D299 explicit-scenario analysis, then D1115 Full comparison
and D1057 mechanism-analysis reuse. No serving change, no new GPU run, no new
baseline reproduction or switch to3B. Old LIVE paragraphs below are historical.

Latest verified wait:2026-11-07 22:44:51+08 D1119 SAME panePID1953868 live.
3031arrived/3026done/3026ok,0fail,5backlog;sample1571 no protection events/swap.
Disk157.987644GiB,available98.086304GiB; no final performance result yet.
Continue the existing run, no restart or new launch. Last turn was monitoring
only; no serving changes/heavy maintenance/push. Read live registration below.

LIVE handoff (2026-11-07 22:18 +08): D1119 started ONCE; do not relaunch.
tmux tc-d1119-elastic-only-7b-w1, panePID1953868; verified live at22:19.
Read updated D1119 registration. Checkpoint02bccc884f155c295208557eed31238910431a28
pushed to V2, remoteHEAD matched before launch. 17 targeted tests,144 protected
files unchanged. Own scenario tc_ieee_elastic_only; same D1100/max4 as D1115.
Service8f3642ca72f942588447b751d6ab3950, auxb5f3fc5c3706466589f987e0d70237f2;
notice3568345.783442678, replay_t03568405.783442678, W1view103ac4add35a43c4...
Raw d1119_20261107/7b_elastic_only_w1 under serverless_llm_experiment/results/ieee_tc/p2_backend_qualification.
Remoteowned3a5aa97080a94984ad26ddb9265c6d25, clock69ab6e9a4e864acda9c3607846d854a5.
Sample54 no abort/high/max/OOM/swap, disk172370706432B. No performance conclusion.
Monitor SAME live run through terminal and real release, then collect journal and
reuse existing analyzers with explicit scenario; compare D1115, no Full replay.
No code edits/tests/maintenance/hashscans/push/remote changes during measurement.
Only lightweight status handoff updates. Older NOT STARTED text is historical.

Current handoff (2026-11-07): D1119 7B ElasticOnly W1 REGISTERED, not yet launched.
Read D1119_7B_ELASTIC_ONLY_W1_REGISTRATION_20261107.md before next action.
Same D1100/max4 configuration as sealed D1115 Full, changing only approved
ElasticOnly gates, own scenario and isolated output/cache roots. No serving
algorithm change. D1100 runner accepts optional seventh scenario; old default
unchanged. 17 focused CPU tests pass (initial wrong-working-directory test
command retained separately); 144 protected files and Full source SHA unchanged.
Next backup this checkpoint, normal fresh resource checks, launch ONCE in tmux
tc-d1119-elastic-only-7b-w1 with registered command. Do not repeat Full/baselines.
After actual release reuse 7B collector/review with explicit scenario support,
check mechanisms/complete inputs vs D1115; do not rename raw JSON as Full.
All other A1-A5/S1-S13/docs/figures remain active; no whole-goal completion.
Older D1118 next-action notes below are historical, not duplicate instructions.

Current handoff (2026-11-07): D1118 W2 COMPLETE/released; no live GPU run.
Read D1118_W2_RESULT_AND_BUDGET_WORKING_POINT_20261107.md FIRST.
4000 native contracts; joint3900=97.5%; U9873.292173262686;
P95 TTFT3053.004479035735ms; frozen budget12086.669343876012 passes.
D1117 W1 + D1118 W2 use identical cap3 config and close this budget-feasibility
gap. Retain max4 resource_opt D1114/W0,D1115/W1,D1100/W2 separately.
Next is current-version 7B A1, starting ElasticOnly vs D1115 Full; inspect/reuse
existing gates and non-target equality. Do not rerun Full/baselines, sweep max2,
switch to3B, weaken controls or call the whole objective complete.
Curated d1118_budget_cap3_w2/result2; result1 is earlier CPU analysis only,
not another performance attempt. 22 tests,49 source SHA,144 protected files pass.
Remote inactive;18 closed leases,4072 samples,high/max/OOM/swap0,GPU census empty.
Five W2 baselines reused;159 output digests differ vsD557 without numeric cause.
497 remote transfers,494 request-level linked;3 retained in preparation metadata.
All limitations and remaining A1-A5/S1-S13/docs/figures remain active.
Older LIVE/NOT STARTED notes below are historical, not new launch instructions.

LIVE handoff (2026-11-07 18:41 +08): D1118 cap3 W2 HAS STARTED once.
tmux tc-d1118-7b-budget-w2, panePID1417227; do not repeat/restart.
Read updated D1118 registration and completed D1117 result first.
Checkpoint70cdac92606bec01458f53e03169864fac138a47 pushed, remoteHEADmatched.
Service1396f87cc78845db88722e2b49c6eadd, auxiliary025dcbaae7524e48b474cb47cc0002ae;
owned remote invocationbe5ec5e0b6d846c387f4b48d54107c42, PID949742.
Notice3555284.928629879, replay_t03555344.928629879; exact existing W2 source/view.
Raw /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1118_20261107/7b_budget_cap3_w2.
Initial available121583210496B/disk173307752448B; sample34noabort, memoryevents0.
Monitor same run to actual terminal/release; no maintenance/hashscans/push/heavy
analysis during measurement. Afterwards matching-clock remote journal and existing
collector/W2 baseline join; no new baseline run. D1117 already complete, no rerun.
Older NOT STARTED D1118 and LIVE D1117 notes below are historical.

Current handoff (2026-11-07 18:38 +08): D1117 W1 COMPLETE/released.
Read D1117_W1_RESULT_20261107.md and D1118_7B_BUDGET_CAP3_W2_REGISTRATION_20261107.md.
4000 native contracts, joint3903/4000=97.575%, U6545.884585417341,
P95 TTFT2469.504281ms; W1 budget6756.571050141065 passes by210.686465.
Versus D1115: U -13.913829%, P95 +5.926865%, joint -0.975pp, all retained.
Five existing W1 baseline rows reused; 214 output digests differ from D536,
unknown independent numerical cause. No independent arithmetic proof/CI claimed.
All leases closed, service/replay/watchdog0, remote inactive, GPU census empty.
38 source hashes, 144 protected publication files and user manifest verified;
7 targeted tests pass. Result directory d1117_budget_cap3_w1/result1.
Next D1118 W2 uses EXACT same D1117 YAML/code; no per-scenario tuning.
Registered but NOT STARTED as of this note; backup then normal launch checks.
Do not repeat W1, alter resource_opt, restart baselines or switch to3B.
Full scope remains active. Older LIVE D1117 notes below are now historical.

LIVE handoff (2026-11-07 17:54 +08): D1117 cap3 W1 HAS STARTED once.
tmux `tc-d1117-7b-budget-w1`, pane PID1300738; DO NOT duplicate/restart.
Registration checkpoint 13b94a6de06c625e58e22ac9dcc67b6c33733674 was pushed
and remote HEAD matched before launch. Read updated D1117 registration first.
Service scope 3865ea7edd1344129b8f54aa8327cbdb, auxiliary 315c2c79d9254d74a41e37f204ff3211.
Owned remote invocation caf52e7ce72d44e39ef56fb2ad30306b, data NICs 1Gbps/full.
Raw `results/ieee_tc/p2_backend_qualification/d1117_20261107/7b_budget_cap3_w1`
under `/home/qhq/serverless_llm_experiment`; replay_t0=3552593.786908827,
last planned arrival +2186.992129s. Sample54 no warning/abort; no final result.
Next monitor this same run, then actual release and existing collector/review/
W1 comparison. No maintenance/remote changes/parallel heavy work while measuring.
Below NOT STARTED notes are historical registration, not a launch instruction.

Current handoff (2026-11-07): D1116 offline W1 budget diagnosis COMPLETE.
Read `docs/ieee_tc/D1116_W1_BUDGET_DIAGNOSIS_20261107.md` then
`docs/ieee_tc/D1117_7B_BUDGET_CAP3_REGISTRATION_20261107.md`.
No GPU experiment is running. D1115 arrival held four GPUs for 1061.571773 s,
mean 3.447244 physical GPUs; non-arrival cost only 64.779336 GPU-s. Twenty
closed leases, 19 successful scale-ups/17 releases; 9 releases followed by
another activation within 38 s. Request interval complement is NOT idle time
or guaranteed savings. Existing source inputs/results remain unchanged.
Next is registered D1117 7B G2 cap3 W1, NOT another audit/replay of D1115.
Config `20261107_d1117_7b_budget_cap3.yaml` SHA
34562e22c517fc62033da1c3437ef95df616fd54c8ae0dcb20a3e59db289eb53.
Only max ready+pending 4->3 plus unique output/cache roots; all 4 physical
devices visible, same D1100 serving code/controller/equations. Runner now
accepts an optional sixth explicit config path, old 0/5-argument mode unchanged.
53 targeted tests and shell syntax pass. Before actual launch, backup and
normal resource/remote checks; launch once in tmux per D1117 instructions.
If W1 budget+joint timing feasible, use the SAME configuration for W2; do not
tune per scenario, replace resource_opt, conflate cap3 with physical budget,
change thresholds, suppress failed history, or repeat rejected guard/no-release
directions. D619 W1 success and D621 W2 failure both retained as context.
All M1/M2/A1-A5/S1-S13/docs/figures remain active; D1117 not yet started.

Completed handoff (2026-11-07): D1115 current-version 7B W1 COMPLETE/released.
Read `docs/ieee_tc/D1115_W1_RESULT_AND_THREE_SCENARIO_20261107.md` first.
4000 native-contract requests, joint SLO 98.55%, 7603.874678 GPU-s,
P95 TTFT 2331.329520 ms. No repeat; tmux ended and owned remote is inactive.
Raw: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1115_20261107/7b_same_version_w1`.
D1114 W0 / D1115 W1 / D1100 W2 now form the same-config 7B three-scenario
comparison. All three meet the timing SLO, but W1 exceeds frozen G2 budget
6756.571050 GPU-s by 847.303628 (12.5404% of budget; 11.1431% reduction needed).
Next: investigate this specific 7B G2 gap using existing ownership/queue/control
logs and prior budget candidates/rejected directions, then literature and one
falsifiable hypothesis if a new budget working point is necessary. Do not repeat
W0/W1/W2, restart baseline reproduction, switch back to 3B, change thresholds,
or declare all objectives complete. Existing A1-A5/S1-S13/docs/figures remain open.
Five W1 baseline audits reused; all inputs matched, 218 output digests differ
from D536 without independent numerical attribution. All differences retained.
50 source hashes and 144 protected publication files verified; user manifest unchanged.

Completed handoff (2026-11-07): D1114 7B W0 is COMPLETE and released.
Read `docs/ieee_tc/D1114_W0_RESULT_20261107.md` first. Do not rerun it.
4000 native-contract requests, 3954 joint SLO (98.85%), 10541.548234 GPU-s,
P95 TTFT 1868.328236 ms. Matched existing W0 baselines; 164 output digests
differ from vLLM D423, retained without an assumed numerical explanation.
144 protected publication files and user manifest unchanged. No serving changes.
Its result backup f8a52faa was pushed and verified before D1115 started.
W2 D1100 already exists; no W0/W2 rerun or new baseline reproduction.
D989 is older-source W0; full main/ablation/sensitivity scope remains open.

Current handoff (2026-11-07): read
`docs/ieee_tc/D1113_BASELINE_CONTRACT_RECONCILIATION_20261107.md`, then
`docs/ieee_tc/D1112_W2_OUTCOME_AND_HANDOFF_20261107.md`, before older headings.
D1103/D1106/D1108 are R2 diagnostics, NOT common-contract main comparisons.
Reuse already completed D555/D557/D591; D1113 matched all 4,000 inputs/native
contracts to D1100 and verified the common notice/resources. D593/D810 already
contain both-model baseline matrices; do not restart baseline reproduction.
Next missing current-configuration work is 7B D1100 W0, then W1, not another W2.
D1084 and D1112 both completed same-version 3B W2; do not rerun either or
select by victory. D1112 is below the frozen joint SLO and remains disclosed.

Latest user clarification (2026-10-25, concurrent main-line progress): storage
auditing/reclamation must progress alongside the experiment main line, not become
an indefinite replacement for it. Perform proven-safe compression/cleanup between
GPU measurements; use existing logs/code for the current 3B optimization question
while CPU/I/O maintenance runs. Never overlap heavy maintenance with a measured
GPU replay. Unreadable/shared or unresolved other-project data remain protected;
do not wait for a perfect whole-disk inventory before making in-scope progress.

Latest user update (2026-10-25): explain whole-disk storage attribution, not only
this project's usage. Audit other projects and shared/system permission gaps;
delete only proven obsolete/rebuildable contents, or losslessly compress closed
evidence with restoration/source records. Size, age, failed-run labels and cache
names alone are insufficient. Do not present a timed-out/partial scan as a full
inventory, modify shared services, or change filesystem reserves to mask growth.
See `docs/ieee_tc/STORAGE_PRIORITY_20261025.md`. Storage first, then resume the
same experiment main line without removing outstanding scope.

Latest user clarification (2026-10-22): explicitly check prior rejected and
ineffective optimizations before every new candidate. Do not repeat a closed
direction unless a concrete new observation changes its premise; document that
new evidence. Preserve all failed measurements, keep one falsifiable candidate,
and return to the complete experiment main line after its bounded evaluation.

Latest user update (2026-10-22): avoid excessive defensive decisions and useless
extra gates. Reuse valid completed checks; choose tests proportional to the changed
path. Do not repeat broad qualifications or add diagnostic replays without a
specific unresolved question. Preserve the established memory/disk containment,
correctness and measurement contract. If a native mixed-batch profile collection
already validates the changed backend path, do not add a redundant short Full
replay before the required complete workload. Record limitations, not new hurdles.

Before any task, after context compaction, and before every experiment, read
`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`docs/ieee_tc/EXECUTION_STATUS.md`. The source plan is authoritative; its
historical "plan mode / no execution this turn" sentences describe approval
time, not the user's subsequent explicit execution authorization.

Latest user update (2026-10-14): read
`docs/ieee_tc/BASELINE_ADAPTATION_STOP_RULE_20261016.md` first for the newer
bounded-adaptation rule. Once necessary correctness, measurement and machine
adaptation are verified and observed performance has mechanism/workload evidence,
stop baseline performance optimization. Low SLO alone is not a reason for another
search. Complete the frozen matrix, then focus on PrimeLoRA; retain all limitations
and unresolved concrete defects. This does not remove any planned experiment.

For the 2026-10-14 baseline scope also read
`docs/ieee_tc/BASELINE_SCOPE_20261014.md` and
`docs/ieee_tc/SLO_BUDGET_RATIONALE_20261014.md`.
External baseline tables contain published-paper systems only. ElasticLocality
is NOT an external baseline; preserve historical data and use it only as a
PrimeLoRA composite ablation variant. No relabeling as another paper system,
deleting evidence, or claiming performance improved because the table changed.
Core-mechanism preservation/evidence matters, not an exact hardware replica.
95%/75% are frozen project points, not universal industry norms; keep the
approved threshold/budget sensitivity without changing online policy or history.

Latest user order (2026-10-12): read
`docs/ieee_tc/STORAGE_PRIORITY_20261013.md` FIRST for the newer storage-first
instruction: audit and safely reclaim proven obsolete/rebuildable data or
losslessly compress closed evidence before another build/new experiment.
Retain unique assets, failed evidence, user dirt and unrelated projects.
After this maintenance, resume the baseline order below; no experiment scope
is removed. Read
`docs/ieee_tc/BASELINE_COMPLETION_ORDER_20261012.md`. Finish/audit/save the already
running D476 once, then complete HydraServe necessary isolated adaptation/native
qualification and obtain actual results BEFORE another PrimeLoRA optimization
candidate. Compare the full available baseline set, then optimize PrimeLoRA.
This supersedes older priorities deferring HydraServe, not metrics or full scope.

Latest user goal update (2026-10-11): also read
`docs/ieee_tc/BASELINE_EXECUTION_RULES_20261011.md`. Except necessary compatibility,
correctness/measurement and safety adaptations, do not optimize non-PrimeLoRA
systems or search their performance configurations. Retain their observed results;
focus optimization on PrimeLoRA. This supersedes older baseline-tuning permission,
not the full matrix, frozen metric rules or historical execution identities.

- Work on `retry14_continuous_queue_v2`; preserve pre-existing dirty files.
- Never stage `configs/generated/lora_manifest_1000.json` or unrelated user work.
- Use existing runners, environments, artifacts and traces. Do not regenerate
  adapter pools, duplicate full datasets, overwrite old results, or run 13B.
- G1: correct full workload + common joint SLO, then minimum lifecycle GPU-s.
  G2: common budget + SLO, then tail TTFT. CE is supplementary.
- Before every system comparison, configuration selection or comparative figure,
  read docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md in full. Record its SHA and the
  separately frozen warm/reference manifests. Development 5000ms/live CE is not
  final SLO qualification. Do not alter V1 in place after its freeze; revisions
  require a new protocol identity and an explicit impact assessment.
- Every optimization: historical logs/code + current primary-source literature
  and code + falsifiable bottleneck hypothesis + validation + full replay.
- Latest user direction (2026-10-10): single performance run per frozen key,
  no repeated-run mean/CI; preserve all old attempts and development identity.
  Read METRIC_PROTOCOL_SINGLE_RUN_V2.md alongside frozen V1. First close a
  reusable 7B PrimeLoRA point and compare 7B baselines (ServerlessLLM audit first),
  then 3B. W0 success alone does not close W1/W2 or G2.
- Preserve IEEE's nine equations and semantics. Prefer measured, causal online
  state; model-specific frozen configuration is allowed, test-point tuning is not.
- First validate resource containment and protected results. No GPU performance
  campaign until actual workers, playback separation, watchdog and cleanup pass.
- ServerlessLLM audit comes first among baseline tasks. Latest user naming rule
  (2026-10-10): always write PrimeLoRA and ServerlessLLM in full in reports,
  new documentation and figures. Preserve historical raw identities/paths.
  Authorized polling repair must preserve RR/scaling/loading semantics.
- One heavy run at a time. No blind OOM retries. Do not stop unrelated processes.
- Latest user direction (2026-10-06) supersedes plan section 11's immediate
  plotting cadence: before PrimeLoRA meets the performance targets, do not generate
  figures. Each run: cleanup -> validate/save data -> concise interpretation ->
  next main-line task. Do not spend optimization time on presentation work.
  Later publication figures still follow section 11's typography/layout rules.
- Report periodically in Chinese as paper evidence: completed experiments,
  current question and reason for remaining here, outstanding experiments.
  Do not disguise engineering checks as completed performance experiments.
- Commit/push tested checkpoints to faaslora_origin/retry14_continuous_queue_v2;
  baseline work goes to its own repo. No force push, credentials, or raw large data.
- Record evidence and next action in EXECUTION_STATUS before yielding. Keep the
  full goal active until all specified experiments and deliverables are verified.
