# D203 — 7B ordinary Full after physical HOST representation qualification

2026-10-03. Complete ordinary Full, launched at 19:10+08 and ended by 20:19+08.
All 4,000 terminal records report success and matched native token contracts;
numerical adapter correctness and performance qualification remain pending.
The launcher passed with service/replay/watchdog exit codes zero, confirmed
native GPU release and removed service domain. tmux `tc_d203_full1`, both local
domains and all six recorded replay/watchdog/GPU PIDs are absent.
Only after local cleanup, the three exact-owned remote services were stopped
and the remote journal/monitor copied with SHA verification. The current task
is bounded offline analysis, not another replay. Full evidence identities and
analysis handles are in EXECUTION_STATUS.md.

## Question and fixed conditions

Does D202's graph-only physical HOST observation reduce whole-request waiting
and improve tail TTFT/joint provisional service attainment, without degrading
TPOT or changing lifecycle GPU consumption? Component timings are not the answer.

Reuse D195's ordinary 4,000-request W0 and D157 cap4 initialization. Only three
owned output/cache paths change in the YAML. Runtime/evidence HEAD is
`2fc3630d684e73f5bfbae1b49f1ac937488444bf`. D202 qualification is sealed at
`paper_results/ieee_tc/p2_backend/20261003_d202_physical_host.json`, SHA
`0590051fcffff3eaaed25eb107789a95a51923d49db6deccbcb909ffbb3296ae`.
Intervening D198 diagnostic hooks are present but disabled; D199 changes only
offline analysis. No detailed frame/control observer, no prefix, no new optimizer.

Keep model/backend, frozen source42 trace/subset, fixed native output contract,
60-second deployment notice, open-loop arrival, natural control, 1,800-second
qualification protection, resource envelope and all nine equations unchanged.
Reused readonly D78/D80 published remote objects: no per-request packaging and
no recreation of weights, traces, or delivery cache. Start/health-check the owned
remote services before measurement and make no remote management changes during
the replay. Copy remote evidence only after confirmed local resource release.

Existing D195 launcher, preflight, cleanup and analysis are reused under unique
`results/ieee_tc/p2_backend_qualification/d203_20261003` paths. Never overwrite
D195, protected results or the user's generated manifest.

## Interpretation and mainline

Single development replay: native count/identity contract is separate from
numerical adapter qualification. Compare complete results against D195, with
all regressions and failures retained, no run-level CI from n=1. Evaluate the
same provisional D170 7B thresholds descriptively, not formal G1/G2 acceptance.
Keep numeric identity, prior output-hash differences, common warm/Resident,
old/new Prime G1/G2 gaps open. After cleanup, validation and an exact table,
backup this checkpoint. Proceed 7B acceptance -> 3B -> external baselines;
all M1/M2, A1-A5, S1-S13 work remains in the authoritative plan.

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
Metric V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.

## Complete result and descriptive comparison

The same 4,000 offered request IDs, adapter assignments, arrivals, prompts,
input/target/native token counts and generation/timing contracts match D195.
All 15 non-output fields are identical. Output token hashes match for 3,874
requests and differ for 126; the full ID list is retained. This is not numerical
adapter proof. Timing identities have zero reconstruction error in the main
curator and at most 2.78e-17 seconds in the independent timing normalization.

One run per version, no CI or isolated causal estimate. The exact CSV contains
all 11 pre-existing comparison metrics; negative reduction means regression.
An exact table is preferred to an apparent significance/ranking chart.

| Metric | Unit | D195 | D203 | Lower-is-better reduction |
|---|---|---:|---:|---:|
| Mean TTFT | s | 2.290593 | 1.741609 | 23.967% |
| P95 TTFT | s | 5.207960 | 3.853586 | 26.006% |
| P99 TTFT | s | 8.059697 | 5.923501 | 26.505% |
| Mean internal E2E | s | 6.895221 | 6.173824 | 10.462% |
| Mean dispatch/admission wait | s | 1.356397 | 0.956163 | 29.507% |
| Mean service TTFT | s | 0.934195 | 0.785446 | 15.923% |
| Mean native TTFT | ms | 328.901459 | 293.641974 | 10.720% |
| Mean TPOT | ms | 40.262617 | 38.587409 | 4.161% |
| P95 TPOT | ms | 66.609260 | 63.038563 | 5.361% |
| Mean response pickup | ms | 261.817723 | 243.802989 | 6.881% |
| Physical lifecycle GPU time | GPU-s | 15915.263362 | 15926.245594 | -0.069% |

## Same provisional reference, not formal SLO qualification

Reuse D170's measured, not globally frozen 7B candidates without changing them:
TTFT 2.990091890/4.635325166 seconds and TPOT 0.059646198/0.085345274 seconds
for input lengths at most 616/above 616. D170 uses batch/slots 8, rank 16 and
GPU fraction 0.92; Full keeps cap/slots 4, rank 64 and fraction 0.70. Disclose
these conditions, do not relabel this reference Full or loosen its thresholds.

| Timing-only observation | D195 | D203 |
|---|---:|---:|
| Joint timing passes / 4,000 | 3,476 | 3,711 |
| Joint timing-only upper bound | 86.900% | 92.775% |
| Additional timing passes needed for 3,800 | 324 | 89 |
| TTFT-only misses | 397 | 205 |
| TPOT-only misses | 93 | 75 |
| Both timing limits missed | 34 | 9 |
| Request failures | 0 | 0 |

This is +235 timing passes/+5.875 percentage points, still below 95%.
Unknown numerical correctness means these values are upper bounds, not final
joint SLO or goodput. Resident budget and GPU-s/correct-request remain unknown.
The new value does not prove superiority to legacy Prime, whose input/backend/
lifecycle/correctness evidence does not satisfy the new metric contract.

## Mechanism, resource and transfer evidence

- Selected-replica pre-generation admission/tier evidence covers 4,000 requests,
  zero conflicts: GPU 1,189, HOST 1,802, NVMe 873, remote 136. These endogenous
  tier populations are conditional diagnostics, not matched causal groups.
- One initial and three natural scale-out activations; four physical leases,
  all released; zero quarantine/replacement events. This is not full A4 acceptance.
- 132 remote UUID pairs, all published; server/client bytes both 131,480,060;
  content-verified logical bytes 3,300,789,780; packaging zero; no byte mismatch.
  Transfer count and bytes match D195, not evidence that all remote delays match.
- 4,063 resource samples: service peak 20,148,785,152 bytes, minimum host available
  92,553,940,992 bytes, high/max/OOM/swap/warning counts zero.
- Physical GPU time: pre-arrival 37.124660491, arrival 15,779.052960364,
  drain 6.238266296, cleanup 103.829706560 GPU-s; total 15,926.245593711.
  Cost is not inferred from utilization or discounted idle time.
- Source observations: 4,337 reads, 4,145 collections, 16,511 RPCs, 192 joins,
  274 stale rejections. Planner receipts: two initialization and 2,797 completed
  owned epochs; owned planner CPU 708.411902008 seconds versus D195's
  565.902328697 seconds. More completed planning work is a retained side effect,
  not automatically a speedup or a new algorithmic contribution.
- Mean arrival-to-admission is 0.956162505 seconds and admission-to-engine is
  0.491804487 seconds, together 1.447966992 seconds. For 161 requests this
  observed combined span alone exceeds the applicable TTFT threshold.
  These are measured spans, not a claimed counterfactual saving.

## Analysis execution and preserved size-guard failure

The first streaming extraction succeeded (522,819,369-byte original to
173,117,160-byte metadata), but its 128 MiB input-size guard stopped the wrapper
before Python loaded it. Keep the failed wrapper/source/output; no GPU rerun.
Its jq peak RSS was 728,448 KiB; final removed-scope counters are unavailable.

The subsequent collector reused that exact projection with a 192 MiB per-run
input cap, retaining the same 3/4 GiB RAM and zero-swap process limit. Prior
same-schema collector/curator RSS was 431,896/450,836 KiB at 121,405,834 input
bytes; a conservative eightfold expansion of the new cap is 1.5 GiB.
Measured new collector peak was 610,660 KiB, all captured memory events/swap zero.
No host safety threshold or runtime/analyzer policy was weakened.

The unchanged control analyzer still uses its 128 MiB guard. It receives an
817,902-byte exact projection of its two consumed control/quarantine fields;
all complete mechanism records remain in the parent/original. Its 1,872 control
samples are all within the request observation window. Current request
projection is 34,389,917 bytes; no whole-load of historical large raw results.

All successful analysis domains were released with terminal receipts.
The original metadata-size failure and unavailable final counters stay explicit.
No serving code, formulas, SLOs, weights, traces or runtime configuration changed
while analyzing D203. Prior qualified analysis tests are reused only after
unchanged source/test hashes are checked; actual D203 data checks also ran.

## Decision and next evidence

Retain D202 as the current development candidate: D203 observes lower mean/tail
TTFT and TPOT under unchanged workload/configuration. Do not claim causality,
statistical superiority, resource savings or model acceptance from this pair.
No new repeated Full is justified without a new hypothesis.

After evidence sealing/backup, use the retained full request-stage evidence,
D200/D201 control-boundary history and current official implementation to
identify one remaining bottleneck; verify the next hypothesis minimally before
another Full. Numerical identity/output differences, common warm/Resident,
legacy/new Prime G1/G2 and 3B acceptance remain open. Do not relax thresholds,
skip 7B acceptance, resume baselines early or repeat already rejected hypotheses.

Curated result: `paper_results/ieee_tc/p2_backend/20261003_d203_7b_full_w0_full1.json`,
SHA `8b92e298f7093295a5b062f45e8502d4c68b9d71e93aa606049116be044763e3`.
Request projection SHA `3e5250e0b34c152f06e7c520bd84469cfa483e2f6414da5d428f848929ffc8cc`.
Complete timing/occupancy CSV/JSON and this table are preserved; no old result is
overwritten. Final verification and Git backup identities are recorded in the
execution ledger.
