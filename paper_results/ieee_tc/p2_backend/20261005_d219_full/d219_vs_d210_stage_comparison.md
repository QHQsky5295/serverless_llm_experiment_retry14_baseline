# D219 versus D210 stage comparison

This is an offline, descriptive comparison of two complete 7B W0 replays using
the same seed-42 trace and fixed-length generation contract. It is not a paired
multi-seed estimate and does not qualify G1/G2. D210 is the preceding ordinary
Full run; D219 enables the confirmed-snapshot route candidate. The values below
come from the request-level fields in the two immutable result JSON files.

| Metric | D210 | D219 | Relative change (lower is better) |
|---|---:|---:|---:|
| Mean routing/control path (ms) | 325.863 | 84.724 | −74.00% |
| P95 routing/control path (ms) | 1,075.326 | 362.985 | −66.25% |
| Mean dispatch/admission wait (ms) | 938.404 | 814.074 | −13.25% |
| Mean ingress queue wait (ms) | 701.857 | 583.843 | −16.81% |
| Mean runtime-slot wait (ms) | 615.830 | 485.884 | −21.10% |
| Mean TTFT (ms) | 1,650.511 | 1,508.362 | −8.61% |
| P95 TTFT (ms) | 3,415.442 | 3,619.799 | +5.98% |
| Mean E2E (ms) | 6,034.115 | 5,848.015 | −3.08% |
| P95 E2E (ms) | 12,391.989 | 12,025.636 | −2.96% |
| Physical GPU-s | 4,317.011 | 4,317.050 | +0.00091% |
| Legacy TTFT-only SLO display | 98.375% | 98.275% | −0.10 pp |

The route candidate therefore reduces the measured control path and several
mean queue components, but the observed P95 TTFT is worse. The request-level
tail does not support a claim that route-state caching alone improves the
service tail. D219 also changes the observed dispatch-window and ingress-tail
mix, so this single replay cannot identify the causal source of the P95
trade-off. The next comparison must use the frozen held-out blocks and report
the joint warm-derived TTFT/TPOT SLO, numerical correctness, and paired
run-level intervals.

Source result hashes:

- D210 result JSON: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d210_20261004/7b_outputs_full1/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d210_7b_full_w0_full1.json`
- D219 result JSON: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d219_20261005/7b_confirmed_snapshot_cache_full_replay/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d219_7b_confirmed_snapshot_cache_full_replay.json`

The D219 curated status remains authoritative for lifecycle, delivery, request
invariant, and qualification gates.
