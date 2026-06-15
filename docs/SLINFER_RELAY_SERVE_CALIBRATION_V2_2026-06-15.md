# SLINFER RelayServe Calibration V2

## Status

SLINFER is now correctly mapped to the RelayServe comparison contract and has
completed a clean cross-rate calibration:

- Llama-3.2-3B: 0.67x, 1.00x, and 1.30x, each 512/512.
- Llama-2-7B: 0.67x, 1.00x, and 1.30x, each 512/512.
- Total: 3072/3072 completed, zero request failures.
- No host-memory guard breach, kernel OOM, NVRM error, or GPU Xid.

The immutable evidence table is
`paper_artifacts/relayserve_v4/slinfer_cross_rate_calibration_v2.csv`.

## Semantic Correction

The V1 harness passed RelayServe's external paper SLO values into SLINFER's
internal scheduler deadline fields. That was not the official SLINFER
semantics. SLINFER computes an input-aware internal TTFT budget:

```text
min(max(0.475s, input_tokens / 512 * 0.95), 7.6s)
```

and uses an internal TPOT budget of 0.2375s. V2 restores those defaults.
RelayServe's 180/14ms and 440/32ms targets are now used only for external
evaluation. V1 evidence remains in the repository for audit, but its formal
eligibility conclusion is superseded by V2.

## Frozen Topologies

| Model | Logical workers/GPU | Keep-alive | Reason |
|---|---:|---:|---|
| Llama-3.2-3B | 2 | 10s | Best smoke-tested latency-cost tradeoff; all three rates clean |
| Llama-2-7B | 1 | 10s | One safe resident 12.6GB model per 24GB GPU; all three rates clean |

The 7B topology is deliberately not increased to two workers per GPU. During
calibration a resident model plus KV cache reached roughly 22.8GB on a 24GB
RTX 3090. A second resident model is not physically safe, while another
logical endpoint would not add model capacity.

## Calibration Results

| Model | Rate | TTFT P95 | TPOT P95 | CE |
|---|---:|---:|---:|---:|
| 3B | 0.67x | 231.142ms | 27.111ms | 242.159427 |
| 3B | 1.00x | 228.561ms | 32.040ms | 291.575978 |
| 3B | 1.30x | 264.755ms | 33.977ms | 357.034166 |
| 7B | 0.67x | 1496.496ms | 104.027ms | 32.968349 |
| 7B | 1.00x | 1882.548ms | 136.533ms | 32.652657 |
| 7B | 1.30x | 909.763ms | 56.664ms | 97.327259 |

All runs are execution-clean but fail the RelayServe paper P95 targets. This
is a valid external-baseline result, not a reason to relax the targets.

## Formal Gate

The frozen V2 configuration was used exactly once for the nominal formal4000
run per model. No formal result was used to tune worker topology, keep-alive,
scheduler deadline values, or paper SLOs.

| Model | Completed | Native failures | TTFT P95 | TPOT P95 | CE | External-main eligible |
|---|---:|---:|---:|---:|---:|---|
| Llama-3.2-3B | 4000/4000 | 0 | 359.640ms | 59.463ms | 259.965506 | yes |
| Llama-2-7B | 3976/4000 | 24 `deadline_violation` | 1373.594ms | 123.153ms | 72.688786 diagnostic only | no |

The 3B run is execution-clean and therefore qualifies for the external
comparison evidence pool, although it fails the separate RelayServe paper
SLO. The 7B run fails the strict zero-failure contract and is represented only
in the feasibility and gate-results tables. Its CE is computed over successful
requests for diagnosis and must not be used as a main-comparison value.

The 7B replay itself produced all 4000 immutable raw records. A shell
control-flow bug caused the strict nonzero replay exit to bypass summary
finalization. The existing raw evidence was finalized offline without replaying
the trace or changing any frozen parameter. The runner now captures the replay
pipeline status and always completes validation, summary, and manifest
finalization when raw evidence exists.

The audited result is
`paper_artifacts/relayserve_v4/slinfer_formal_gate_results.csv`.
