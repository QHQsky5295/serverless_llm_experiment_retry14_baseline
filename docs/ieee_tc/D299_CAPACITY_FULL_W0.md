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
