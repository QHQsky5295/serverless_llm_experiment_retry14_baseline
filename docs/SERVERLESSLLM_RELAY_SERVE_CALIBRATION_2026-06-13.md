# ServerlessLLM RelayServe Workload Calibration

## Scope

This calibration uses the official ServerlessLLM control plane and vLLM
backend on the disjoint 512-request calibration trace at the frozen 1.00x
arrival rate. It does not use any request from the formal4000 trace.

The 3B calibration trace SHA-256 is
`11dbc15d0bc793c3d74f054e82dc871b6a9526332b1c3bf223af6ef1bd952667`;
the 3B formal trace SHA-256 is
`a90fddd1c6937a6ee8687c79b8006c3d6d24c5bb505ff6b9488377a5b05bb513`.
Their request-ID intersection is empty. The corresponding 7B trace hashes are
`35e1b5216b13c907129be294bba512b8dafe164c66e3cd1f230d9c4586652ae0`
and
`184029a4954fdf2029a31c040f8811c787fe4d0d64874b55f03efbf136988394`,
also with an empty request-ID intersection.

The only scheduler-facing parameter varied for the 3B calibration is the
official deployment `target` concurrency. All candidates use four available
RTX 3090 GPUs, `min_instances=1`, `max_instances=4`, and `keep_alive=60`.
The 5 ms allocation poll replaces client-visible one-second polling latency;
the official one-second autoscaler cadence is unchanged.

## Frozen Selection Rule

The primary selection key is the smallest worst normalized P95 SLO ratio:

```text
max(TTFT_P95 / paper_TTFT_P95, TPOT_P95 / paper_TPOT_P95)
```

Ties are resolved by higher TTFT SLO attainment, higher joint TTFT/TPOT
attainment, and then higher CE. A selected candidate is not automatically a
gate pass; formal results still report both P95 gates independently.

## 3B Calibration Result

| Target | TTFT P95 | TPOT P95 | TTFT attainment | Joint attainment | CE | Worst ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 244.00 ms | 13.600 ms | 85.55% | 82.81% | 205.6254 | 1.3556 |
| 12 | 228.49 ms | 13.999 ms | 87.50% | 83.79% | 237.8763 | 1.2694 |
| **16** | **228.29 ms** | **14.070 ms** | **87.30%** | 82.62% | 231.0821 | **1.2683** |
| 20 | 531.55 ms | 15.830 ms | 84.57% | 76.37% | 253.6732 | 2.9531 |
| 24 | 525.91 ms | 15.793 ms | 83.40% | 74.41% | 251.4568 | 2.9217 |
| 32 | 550.82 ms | 15.882 ms | 83.79% | 75.00% | 246.0118 | 3.0601 |

`target=16` is the frozen 3B winner under the pre-registered ordering. The
`target=12` refinement is practically tied on the primary key: its TTFT P95 is
only 0.20 ms higher, while its TPOT and secondary attainment metrics are
better. The exact primary key nevertheless remains 1.2683 for `target=16`
versus 1.2694 for `target=12`, so changing the winner after observing those
secondary metrics would violate the frozen rule. The `target=20` refinement
confirms the overload boundary and is rejected despite its higher CE.

The selected candidate still misses the 180 ms paper TTFT P95 gate, and its
14.070 ms TPOT P95 is 0.5% above the 14 ms gate. Selection is therefore a
configuration freeze, not a claim that calibration passed both paper gates.

The higher-CE `target=20`, `target=24`, and `target=32` candidates are rejected
because their TTFT P95 is more than 2.9x the paper target. Selecting any of
them solely for CE would be an invalid post-hoc trade of latency quality for
cost.

## 7B Calibration Result

| Target | TTFT P95 | TPOT P95 | TTFT attainment | Joint attainment | CE | Worst ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4259.77 ms | 24.872 ms | 77.93% | 77.73% | 52.5163 | 9.6813 |
| **8** | **531.76 ms** | **28.110 ms** | **90.82%** | **90.04%** | **61.8567** | **1.2085** |
| 12 | 3384.09 ms | 33.645 ms | 84.77% | 80.66% | 58.4957 | 7.6911 |
| 16 | 4124.51 ms | 33.999 ms | 83.59% | 79.10% | 56.9958 | 9.3739 |

`target=8` is the frozen 7B winner. It is the only candidate that keeps TPOT
P95 below the 32 ms paper target while also avoiding the multi-second TTFT
tail seen at targets 4, 12, and 16. Its 531.76 ms TTFT P95 is still 20.85%
above the 440 ms paper target, so this freeze is again a configuration
selection rather than a claim that both calibration gates passed.

The non-monotonic result is produced by the official ServerlessLLM autoscaler
and lifecycle path on the frozen burst trace. Higher target values did not
force earlier four-instance readiness; targets 12 and 16 retained long
dispatch-admission tails and also crossed the TPOT P95 target. All four runs
completed 512/512 requests and remain recorded rather than pruning the
unfavorable candidates.

## Evidence

- Machine-readable table:
  `paper_artifacts/relayserve_v4/serverlessllm_3b_calibration.csv`
- Machine-readable 7B table:
  `paper_artifacts/relayserve_v4/serverlessllm_7b_calibration.csv`
- Frozen policy:
  `configs/relayserve_serverlessllm_calibration_policy_v1.yaml`
- Table generator:
  `scripts/build_serverlessllm_calibration_table.py`
- Verifier:
  `scripts/verify_serverlessllm_calibration.py`

The table records absolute raw-record and source-summary paths together with
their SHA-256 hashes. Runtime outputs remain excluded from Git, while the
tracked evidence table makes any later mutation detectable.
