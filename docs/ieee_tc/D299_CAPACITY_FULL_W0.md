# D299 — full W0 replay of the D298 capacity candidate

2026-10-07. This is the already preregistered full replay, NOT a new optimizer
or accepted configuration. D298 source248 and Full100 are complete. Retain
the one capacity package, measured profiles, completed-length initializer and
D297 control laws. Only request count/scenario/output/cache identities change.
No new weights/trace, numerical repeat, warm reference,3B/baseline or figures.

## Preceding bounded evidence

Same100original inputs, actual native counts and control bindings allpass.
D298 vsD296: meanTTFT1.140273vs2.129072s,P95TTFT2.339925vs10.771356s;
meanTPOT29.797313vs34.997670ms,P95TPOT39.253765vs44.405718ms.
PhysicalGPU-s780.572493vs727.912111: **7.23% higher**, despite shorter wait.
Window totals38.293372pre-arrival,672.402798arrival,32.529846drain,
37.346477cleanup. Four physical GPUs released; no resource abort/high/OOM/swap.
All29remote transfers join,50650865wire bytes, no request-time packing.
9of100output hashes change; same input/binding checks pass, numerical cause
not established. Do not reopen the already bounded D295 diagnostic.
Provisional timing upper bound97/100, not formal joint SLO orN_correct.
Single observations, noCI/capacity-alone causal attribution or overall victory.

These data support testing the same candidate on4000W0, not accepting it as a
resource improvement. Main questions: does lower waiting survive longer churn,
doesTPOTremain acceptable, and what happens to actual GPU lifecycle? Compare
withD297's complete4000, not D296's100, using frozenV1 and provisionalD170
thresholds without altering them. Keep raw delay/cleanup and failures intact.

## Execution and acceptance boundaries

Config `configs/ieee_tc/20261007_d299_7b_full_w0.yaml`, existing launcher argd299.
OriginalsourceSHAefb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c,
4000requests/500logicaladapter pool/W0/no prefix, fixed greedy output.
Existing remote published gzip cache, native1GbE, no delay injection or packing.
Fresh run-owned local caches only. Common60spreparation counts toward resources;
unchanged72/80GiBservice,3/4GiBaux,150/100GiBdisk, one heavy job at a time.
Push the tested checkpoint first, read fullplan/status/V1, then start the exact
owned remote monitor and verify authenticated direct health before deployment.
On completion verify actual GPU release, stop only captured remote invocations,
join transfer journals and preserve resource/timing evidence. Data-only analysis.

G1/G2, numericalcorrectness and commonwarm/Residentreference remainOPEN.
No final performance assertion until the full replay is complete and validated;
no promotion to3B or baselines merely for exceeding the previous candidate.

## Live receipt

Checkpoint1fa3f708ee434dccd58509bea56ba097f2a669bd pushed before03:27:50launch.
ConfigSHA803693f4cccbdf4a5d4719b6d77f3b611cb380fe5e4342583f7d19e7be2ff10d.
Actualreplay4000W0/rate1/source4000/no prefix; one guarded tmuxprimelora-d299-7b.
Service a9a8e9715d484ac5939ffd02427acf48, invocation323ed5b94d1644dcbcc5a9d63f4e46f8.
Remote health2 succeeded beforedeployment on sameapprovedinvocations afterfirst
pre-listen refusal; no restart or performance rerun. Remoteclock5998fee5b2904ea78256f4aaaff2d3a2.
Owner identities and current evidence are inEXECUTION_STATUS; do not launchagain.

## Completed evidence — 2026-10-07 04:46+08

All 4,000 native-contract requests completed with zero reported failures. Four
physical leases closed, no native GPU compute processes remain, and the launch,
replay and watchdog returned zero. The owned remote monitor and both artifact
services were stopped after local release; all are inactive/PID0/success.
The immutable delivery cache was reused, not regenerated.

| Metric | D219 earlier Full | D297 preceding Full | D299 capacity package |
|---|---:|---:|---:|
| Mean TTFT (ms) | 1508.362 | 1527.145 | 1306.468 |
| P95 TTFT (ms) | 3619.799 | 3359.722 | 2884.702 |
| Mean TPOT (ms) | 37.566 | 37.884 | 32.919 |
| P95 TPOT (ms) | 60.479 | 62.066 | 62.483 |
| Mean E2E (s) | 5.848 | 5.873 | 4.904 |
| Physical GPU-s | 15911.713 | 15860.687 | 15845.533 |
| Provisional D170 joint timing upper bound | 93.975% | 94.650% | 95.700% |

Each column is one complete development W0 run, not an independent held-out
replicate. No CI or statistical superiority claim. D219 has additional serving
version differences; D297→D299 changes the capacity package and corresponding
measured initialization together, not one isolated mechanism.

Relative to D297, mean/P95 TTFT improve by 14.45%/14.14%; mean E2E improves
16.49%. Physical GPU-s improves only 0.09554%, which is not evidence of a
meaningful resource reduction from a single run. P95 TPOT worsens 0.67137%.
Under unchanged provisional D170 thresholds, TTFT timing attainment is 98.475%,
TPOT conditional attainment 97.125%, and joint 3828/4000. D297 TPOT attainment
was 98%; the decode-tail regression is retained. The joint figure is a timing
upper bound, not formal SLO qualification: N_correct, final shared references,
Resident budget and G1/G2 acceptance remain open. The terminal's development
SLO and simulated 4318-GPU-s cost projection are not used for this comparison.

### Measurement and input checks

- All4000 request/adapter/arrival/source-output/target/prompt/generation/native
  token-source fields match D297; trace and subset hashes also match.
- All4000 control/native slot/lease/content binding checks pass.213 output-token
  hashes differ from D297 and are retained; these checks alone do not establish
  kernel arithmetic qualification. Do not reopen bounded D295 without new cause.
- Dispatch tiers: GPU1782, HOST1179, NVMe902, remote137. These conditional
  populations are not common-request causal comparisons.
-132remote transfers join by UUID, clock, content/archive identity and bytes;
  total131480060wire bytes, no request-time packing or temporary archive.
-4046resource samples; service peak20523708416B, host minimum94323630080B,
  high/max/OOM/swap0. GPU windows:37.234403pre-arrival,15761.706890arrival,
  7.776115drain,38.815669cleanup; their sum equals15845.533076GPU-s.
- Historical protected seal remains146/147 unchanged with only the previously
  recorded baseline replay-script edit; that file and the user's manifest were
  not touched.35native timing/provisional-bound tests pass. No new figures.

### Resource bottleneck evidence and next direction

All1887control observations contain only three scale-ups and no scale-down.
1335observations have low queue and low active saturation but fail the latency
predicate. Every1885known P95 exceeds the281.278ms lower threshold; minimum is
1122.457ms. Unlike D297, the observed queue-low/active-low/no-pending sequences
now include six sampled spans of at least38s; longest61.728443s (D29716.882191s).
Span durations are61.728443,61.400676,49.912262,61.554602,39.929604,55.687869s.

This is evidence to examine paper-consistent scale-in eligibility using measured
service conditions; it is **not** a prediction of saved GPU-s. A changed control
decision changes queues and later observations. Before any new candidate, check
the IEEE semantics, earlier control history and primary implementation sources;
do not merely lower cooldown or tune a threshold until the current trace wins.
Keep this capacity package as a latency-improved development candidate, not an
accepted final configuration. Next: checkpoint data, recover the unchanged
150GiB launch margin safely, then one evidence-led resource/control hypothesis.
No3B, external baselines, warm rerun or extra figures yet.

### Reproducibility and operational notes

Curated native/stage/control audit: `paper_results/ieee_tc/p2_backend/d299_full_w0/validated1/`.
Context/remote/input comparison: sibling `context1/`. Existing D297 analyzer was
extended with an explicit frozen D299 identity; the thin context checker reuses
native and protected-seal helpers. Both ran after GPU release under3/4GiB bounds.
Context analysis peak1086193664B, high/max/OOM0; empty owned scopes were stopped.

- D299source SHA: `df066b365fa13ecc25b225a793c362706157224965b8a7b360981d8cf45887ab`.
- D299audit SHA: `c5414f1c90b05ba9e9cb7d09bdb7a89058f91e4371adf33ae22756858e42d9fe`.
- Context SHA: `ee7a2f23ad8d85b9b270fdbe680346c52152bfac0a90841fea3ca98f1e87ab05`.
- D219source reverified SHA: `e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`.
- Reused historical comparison SHA: `7495e9269bf961330120e06bf96d2b7fe9d7ace704f445fe98c0372d49aedb36`.
- Control CSV SHA: `35271d728a8843b934815147a1761e7d52ee4d8c8857be7015031a62468bff38`.
- Remote journal actual filename: `transfers-155042291b0c4c42b03059a33b722f92.jsonl`;
  SHA`ab63d1f84e3a6e43079e69f74e933fae2cdefe4fd0e99a3434ae8a07eb1c5224`.
- Remote monitor SHA`59ce220b19fe7b1c187db6d3b8c1ab90769e4ac04475061d52f963d7d4d8017a`.
  Both remote/local SHA pairs agree; the journal filename UUID is not its clock.

After successful remote shutdown, a directory listing piped to head returned141
(SIGPIPE); subsequent direct state checks confirmed all captured services stopped.
A discarded streaming-projection fixture and an exploratory CSV check treating
integer pending counts as booleans failed before interpretation. The validated
projection uses explicit stream paths, and the corrected control calculation
uses int(pending)==0, recovering D297's recorded16.882191s. No serving change,
performance restart, altered threshold, or overwritten result resulted.
