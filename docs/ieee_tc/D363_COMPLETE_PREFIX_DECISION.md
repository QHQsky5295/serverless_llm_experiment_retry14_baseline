# D363 — complete prefix validation, not a performance win

2026-10-09. Execution commit `a5a2355beb8747d0efe2c4b550e998a39c165a7c`.
Source SHA `4228717a06b169ce308d8241e967513dbda6424e7bb2fa6d1b1b796f42a442ff`.
No serving changes were made during or after this measurement.

All 100 offered requests completed with native fixed-output contracts, no
fallback, matched input identities and auditable acquisition/release receipts.
GPU contexts were released, the local service domain removed, and three owned
remote units stopped. This is execution/measurement qualification, not numerical
`N_correct` or formal G1/G2 qualification. The original failed CPU test attempt
is retained: OS-guard fixtures were invoked in the model interpreter; the actual
watchdog uses system Python and passed all 50 tests there. All 430 other tests
passed in the model environment. No guard or serving policy was weakened.

| Prefix observation | D357 retained reference | D363 |
|---|---:|---:|
| Completed requests | 100 | 100 |
| Mean TTFT, ms | 799.660 | 909.898 |
| P95 TTFT, ms | 1,790.372 | 2,441.887 |
| Mean TPOT, ms | 26.618 | 26.234 |
| P95 TPOT, ms | 33.618 | 31.492 |
| Physical GPU-s | 654.716 | 803.253 |

Relative to the descriptive prefix reference, mean/P95 TTFT increased by
13.79%/36.39%, and GPU occupancy by 22.69%; mean/P95 TPOT decreased by
1.44%/6.32%. These unfavorable observations are retained. Different measured
initialization profiles and one run per version prevent an isolated causal
effect or confidence-interval claim. Prefix GPU-s is never extrapolated to W0.

Under unchanged provisional D170 timing bounds, 98/100 passed both timing
conditions; two missed TTFT and none missed TPOT. These remain timing-only
decisions, not a formal joint-SLO pass with numerical correctness established.

The new file-demand audit covers all 100 requests: 29 Remote-file, six NVMe-file,
61 native-GPU and four native-HOST acquisitions. All 35 file transactions bind
the held verified file, its lifetime fence and the native acknowledgement. Four
transactions saw intervening owner-epoch changes (maximum three), with current
native constraints rechecked and no old capacity snapshot reused. Native paths
do not carry file witnesses. Control interleave is observed across four owners.
This establishes exposure, not an avoided-RPC count or measured causal speedup.

Remote evidence: 29 matched transfers, 50,650,865 wire bytes, zero request-time
packing. Actual link negotiation was 1000 Mbps/full. Service peak memory was
19,176,263,680 bytes; high/max/OOM/swap were zero. After shutdown disk space
returned above the unchanged 150 GiB launch threshold without manual deletion.
After release, 62 offline regression checks and complete request/remote/timing/
binding/resource audits passed. No figures were produced.

## Decision

Proceed to D364 complete 4,000-request W0, without a second optimization or
retuning. D363 validates the candidate execution but does not support accepting
it as a performance improvement. The predeclared target includes the second
500-request phase that this prefix does not contain. Compare complete D358,
D353 and D336 measurements; retain all regressions. Examine the full timing
outcomes, physical GPU-s and preparation/recheck mechanism evidence before the
next implementation decision. Final common warm, numerical correctness and
Resident budget qualification remain open; 7B is not closed.

Mainline remains 7B → 3B → paused baselines → M1/M2 → ablations/sensitivities.
Plan and frozen V1 identities are unchanged from the D363 preparation document.
