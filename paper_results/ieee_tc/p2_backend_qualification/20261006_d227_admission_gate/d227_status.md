# D227 status — explicit admission gate (not run)

| Item | Value |
|---|---|
| Status | prepared; GPU/remote execution not started |
| Config | `configs/ieee_tc/20261006_d227_7b_admission_gate_22req.yaml` |
| Config SHA256 | `76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73` |
| Reference config | `configs/ieee_tc/20261005_d219_7b_confirmed_snapshot_cache_full_replay.yaml` |
| Reference SHA256 | `5072d681dc961a44b03ce9f3794b5c13d878f902c6f962cfba13999d5d3a0b63` |
| Model | Llama-2 7B, existing local model path |
| Workload | existing W0 trace prefix, seed 42, 22 requests |
| Adapter set | existing frozen 500-adapter subset and content index |
| Generation | `fixed_length_greedy_v1`; native token-ID audit required |
| Gate under test | `effective_capacity_admission_enabled=true` |
| New result/cache roots | D227-specific and absent at preparation time |
| Current resource decision | no launch: `MemAvailable` below unchanged 109.5 GiB physical-startup gate |

The configuration is a preparation artifact only.  It does not enter any
performance, SLO, G1/G2, or baseline denominator until a bounded run passes
the full execution and cleanup audit.
