# D185: ordinary 7B Full replay of the D184 pending/GPU pipeline

Status: COMPLETE development execution and analysis (2026-10-03).
Evidence verification binds this document's hash; sealing/backup status is in EXECUTION_STATUS.
Development n=1, not a formal comparison or model qualification.

## Question and unchanged conditions

D183 located substantial waiting before engine entry; D184 removed one
controller roundtrip for selected-GPU requests while preserving native pending
registration before reference acquisition. These are two ordered owner operations,
not an atomic transaction. HOST/file paths, formulas, source validation, retry
and uncertainty ownership rules are unchanged. D184 passed 1,005 regression
tests and is backed at `2615d240b567043e4ca92b0debc30a0480e0b347`.

This replay measures net waiting/TTFT/TPOT and physical GPU time against D182
(`a4f4c0630534ed2c4b6f20f4f84e1f262884904f`). It uses the same existing 4,000
source-seed42 W0 requests, 500-adapter subset, fixed-length native generation,
D157 cap4 profiles, 60-second notice, 1,800-second protection, and CPU/RAM/GPU
envelope. Only fresh owned output/HOST/NVMe paths differ in the YAML.
No detailed CPU profiler, new weights, trace regeneration, or added optimization.
The D78/D80 immutable remote delivery cache is reused, never recreated.

## Prelaunch storage maintenance

The inference filesystem was below the unchanged 150 GiB new-heavy floor.
Reused D179's public-download audit/apply workflow, inspecting remaining
30–100 MiB pip HTTP-cache archives with the same package allowlist, exact public
registry SHA/size/HEAD, ownership, open-file, reference and environment-link checks.
Custom-version downloads and ambiguous metadata were retained. This does not
delete installed packages, compiled runtime caches, model data or evidence.

| Check | Observed outcome |
|---|---|
| Selected public archives | 26; 52 body/header cache files |
| Removed allocated bytes | 1,741,856,768 |
| Audit SHA | `4f18950100f5730ef093bdf59c93257f2620cdc450b574a0beda8e50821cab25` |
| Audit/apply scope limits | 3/4 GiB RAM, swap 0, CPUs 2,3,26,27 |
| Audit actual invocation | `721b1a15db7e4b52a75447f8c680b044` |
| Apply actual invocation | `0fbd5fb123464c6eb6499b02ba043bdf` |
| Final memory events/swap | all zero |
| Tracked projects / installed environments | unchanged / untouched |
| Post-maintenance available bytes | 162,534,543,360 |

Receipts and adapted scripts are under
`results/ieee_tc/p2_backend_qualification/d185_20261003/cache/`.
This is storage maintenance, not an inference performance experiment.

## Launch evidence

Prelaunch verified 397 source references and all 147 protected entries; Plan and
frozen V1 unchanged. Receipt SHA
`ce8fd3e0239845d94a805fde9e63f1fbfab22c6fbfa2b5c1bb1b311bb1577647`.
Prelaunch invocation `d183c4959b8f48839cb319473557fd5f` and health invocation
`23a1cdfead81410f84bc4ac859f8b982` finished successfully; scopes absent.

Service `primelora-tc-svc-51a2cfd9da0a43d4a2759a021354820b.scope`, actual
`32633b091d6d41f29f73021c68d20395`, has 72/80 GiB and 2 GiB swap limits.
Auxiliary `primelora-tc-aux-d8ff48c09d684b42986740377304b0b2.scope`, actual
`2fe30e1cd7004c0698b50ea4a0cfacf8`, has 3/4 GiB and zero swap limits.
The ordinary external replay confirms 4,000 requests/W0/source42, not a prefix.
Follow the single tmux session `tc_d185_full1`; do not relaunch.

Both NICs report 1000/full. Unchanged D80 services reused D78 immutable objects.
Remote actual identities: 3B `2dfc7c9cfb6c450a9d84ed7faecd29a2`/PID831145;
7B `fdfc05450c31432ab395873993a47cd4`/PID831147; monitor
`primelora-artifact-monitor-d185full1.service`,
`5796faeb622a4e968a9c3c2f58656626`/PID831150.
7B health clock is `remote-process-monotonic:111e3eb939bd4b44be2de436021a5ef1`.
The 3B artifact service is not a 3B inference experiment.

## Terminal and preliminary evidence

All 4,000 requests returned with native-count contracts matched; numerical
adapter correctness is not thereby established. Final launch receipt SHA
`5b3732c2f543453c385cc4f9ef536da17bd7de56772ba05ca213c4ddb4bf8599`
has service/replay/watchdog return codes zero and confirms physical context
release. At 09:49:42, both local scopes and replay PID 4159968 were absent;
GPU compute-process census was empty. Final cgroup event files were unavailable
after automatic removal; 4,047 saved resource samples have zero high/max/OOM,
swap and warning events. Peak service memory was 20,740,968,448 bytes.

Only after local release, exact-owned remote services were stopped at 09:49:43;
all three became inactive with PID 0. The transfer journal was selected by the
frozen health clock, copied after shutdown and matched the remote SHA.
Journal: `transfers-4ed3ca07af20480b997851c2702f64cc.jsonl`, 109,256 bytes,
SHA `d8aa2024a4d2525b45d337a5ac4894008d1fc3a9232c58bf63812e8650d645e0`.
Monitor: 68,421,037 bytes,
SHA `abf17d3d3538e91e65d31e7d56af27d0715a70760b82944669544b3bc5ba2ad7`.

| Preliminary post-cleanup check | Observed outcome |
|---|---:|
| Planned / terminal / native-count matched | 4,000 / 4,000 / 4,000 |
| Request failures | 0 |
| Physical GPU allocation/release journals | 4 / 4 |
| Physical lifecycle GPU-s | 15,930.13597129297 |
| Paired remote transfer UUIDs | 132, all published |
| Client/server wire bytes | 131,480,060 / 131,480,060 |
| Verified uncompressed payload bytes | 3,300,789,780 |
| Request-path packing operations | 0 |

These are preliminary terminal/resource checks, not G1/G2 acceptance. Metadata
analysis reused D182's streaming selector with the same 128 MiB bound, in a
3/4 GiB, zero-swap CPU scope on CPUs 2,3,26,27. Actual invocation
`bd87025894134f1bb7ef40c483e1c3d8` exited zero; automatic scope removal and final
zero memory events/swap were verified. Subsequent request projection and latency/
timing/contract comparison completed below; live legacy CE/SLO is not used.

## Full diagnostic comparison

Same-input developmental runs, one per version; no CI or isolated causal claim.
Positive reduction means lower is better. All eleven comparison metrics are
retained, including regressions. This table is more appropriate than a selective
single-metric performance figure for the present mixed result.

| Metric | D182 | D185 | Lower-better reduction |
|---|---:|---:|---:|
| mean_ttft (s) | 2.730194 | 2.730961 | -0.028% |
| p95_ttft (s) | 6.375893 | 5.987517 | 6.091% |
| p99_ttft (s) | 8.681467 | 8.760348 | -0.909% |
| mean_internal_e2e (s) | 7.389124 | 7.456483 | -0.912% |
| mean_dispatch_admission_wait (s) | 1.735268 | 1.711301 | 1.381% |
| mean_service_ttft (s) | 0.994926 | 1.019660 | -2.486% |
| mean_native_ttft (ms) | 340.622429 | 343.121378 | -0.734% |
| mean_tpot (ms) | 39.631695 | 40.462495 | -2.096% |
| p95_tpot (ms) | 68.328108 | 70.101109 | -2.595% |
| mean_response_pickup (ms) | 437.978163 | 448.251904 | -2.346% |
| physical_gpu_seconds (GPU-s) | 15928.394194 | 15930.135971 | -0.011% |

Under the **same D170 provisional 7B threshold candidates**, timing-joint passes
are 3,189/4,000 (79.725%), versus D182's 3,200/4,000 (80.000%): −0.275 percentage
points. There are 658 TTFT-only misses, 98 TPOT-only misses and 55 joint misses.
Another 611 timing passes would be needed to reach 3,800. This is a timing-only
upper bound: numerical `n_correct` remains unknown, common reference is not
globally frozen, and Resident budget/G1/G2 qualification remains open.

All 4,000 requests have matching non-output contract fields (15 fields), zero
time-decomposition errors at the existing 1 ms tolerance, and complete confirmed
selected-source dispatch tiers with zero conflicts. Output token hashes agree
on 3,890 requests and differ on 110; these differences remain for numerical
audit, not dismissed as proof of harmlessness or changed workload. Dispatch
tiers are GPU 1,266 / HOST 1,730 / NVMe 868 / remote 136. These selected-request
populations differ from the 132 actual transfer UUIDs; no one-to-one inference.

Mean pre-engine time is 2.387840 s (87.436% of mean TTFT); 563 requests already
exceed their provisional TTFT deadline before native engine dispatch. The
1,752 in-window control observations include 754 with a positive queue and
727 with a queue below the recorded ready capacity. These are sampled control
observations, not continuous GPU-idle evidence or proof of a single RPC cause.

Planning has 1,960 receipts: 2 initialization and 1,958 owned execution epochs,
all completed; owned worker CPU totals 1,091.006240 s. Source observations have
4,564 requests, 4,233 collections, 16,852 RPCs, 331 joins and 494 stale rejections.
68 successful requests retain 70 source retries (maximum 2). None of these
mechanism counts alone establishes an end-to-end causal gain.

Decision: **net benefit is not established**. P95 TTFT improves in this pair,
but mean TTFT is unchanged, P99/E2E/TPOT are higher, timing-joint attainment is
slightly lower, and physical GPU time is effectively unchanged. Do not rerun
the same candidate to seek a favorable result, declare model acceptance, or
advance 3B/baselines. First seal/back up this outcome, then use the retained
phase/RPC evidence to identify the remaining control-path bottleneck before
choosing another optimization. No new candidate is selected by this report.

## Analysis provenance and resource closure

D182's final metadata/projection/curator/timing/occupancy workflow was reused.
No old full-result reparse, new performance run, changed threshold, or analysis
helper modification. All CPU tasks used 3/4 GiB, swap 0, CPUs 2,3,26,27; every
listed task exited zero, saved zero memory events/swap, and its actual scope
was verified absent afterward.

| Task | Actual invocation |
|---|---|
| Metadata | `bd87025894134f1bb7ef40c483e1c3d8` |
| Request projection | `9664d638b276453291d8e7e3c99d751b` |
| Curation/failure table | `7dc74abbb35f4d57a4cd836e79fad3f4` |
| Timing/comparison tables | `0d648ef3fac349ad8d69d8abb955e38f` |
| Occupancy/control analysis | `83ff438244e64af89b16aebf4348874d` |

Request projection is 34,357,745 bytes, SHA
`7cac6bb4b2007e9fa81ea63b308e4d7780d9eb0dd9c2905668c241396caf0ce9`.
Curated result SHA
`159cfc9c5b32c1f909f6118bc7a9137ebe735998703a8fc08ee0b0790d6bc21e`.
Exact-integer decoding was reused without rounding, with two positive and ten
negative representation checks; this projection has 4,000 integer native
output/prompt counts, each cross-checked against retained request counts.
All 147 protected historical entries and all 397 frozen prelaunch sources
were verified unchanged by the curator.

Outputs are `paper_results/ieee_tc/p2_backend/20261003_d185_*`: full curated
JSON, failure breakdown, timing groups/requests/gap, eleven-metric comparison
CSV, and the control-occupancy directory. Raw provenance remains in the unique
gitignored `d185_20261003` directory. The evidence verifier binds this document
and all referenced results; after successful verification the document is sealed.
Git backup status is recorded separately in `EXECUTION_STATUS.md`.

## Remaining mainline

Fresh full preflight, remote health and guarded worker launch passed.
While live, no remote maintenance, source changes, installation or bulk analysis.
Physical release and exact-owned remote stop/journal collection are complete.
Analysis, complete diagnostic tables and interpretation are complete.
No performance acceptance is claimed. The closure ledger records evidence
verification/bundle and backup without changing these observed results.
Numerical adapter correctness, globally frozen common reference, Resident budget,
old/new Prime under V1 and G1/G2 remain open; 3B and baselines remain paused.
