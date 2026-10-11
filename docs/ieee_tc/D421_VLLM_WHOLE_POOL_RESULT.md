# D421: native vLLM 7B whole-pool functional qualification

2026-10-11. No baseline source/configuration change after the qualified D419
path. Execution baseline a56d79088ab1e431e12b5cfed95f53a1581d7037,
main aa76e000127a7cc19d275f8140d0bd8366b3ebc1. Existing 0.30.0 native RR/LRU,
common pre-ready ingress, 60-second notice, fixed output and physical ownership.

Raw `/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d421_20261011`.
Existing adapter_coverage_v1 uses the original first500 prompts/arrivals/targets
and sorted existing500 IDs. No new trace/weights. Not a W0 performance workload.

- 500 offered/submitted/terminal/native/source/adapter/content/timing contracts;
  exactly500 distinct IDs equal to the frozen pool,0failures,68925native tokens.
- E2E/TPOT/native boundary recomputation error0ms;4actualworkers,4closedphysical
  leases,total2722.8814599129837GPU-s. Raw n_correct=null unchanged.
- 500published-object HTTP transfers,420740410wireB,12985984450verifiedpayloadB,
  zero request-time packs. D78/D80 delivery cache reused,not republished.
- Mean/P95TTFT644.341327/2135.383901ms,meanE2E4069.208638ms are functional-view
  diagnostics only,not rankable against fullW0 or the D419 startup prefix.
- ActualGPUrelease,service removal,returncode0;ownedremote3units stopped by
  InvocationID/ports absent;emptyownedaux stopped. Peakservice24619839488B,
  high/max/OOM/swap0. Protected147entries unchanged.
- Main's existing analyzer resident_rr_v1/adapter_coverage_v1 passed;27CPUtests
  passed0.174s after release with a new retained test log/receipt. No new runtime
  policy,weight pool,trace or performance repetition.

Main report docs/ieee_tc/D421_VLLM_500_ADAPTER_QUALIFICATION.md and curated
paper_results/ieee_tc/baseline_audit/d421_7b_coverage/ contain the full audit.
AuditSHA6539a15a293aad9b0e63f4ee1e41b2689847498940468c44b8188a0c583d6277;
568-filemanifestSHA293c40bd75e447a36945dfb548759acbe3da50cfc336829053f91ec2bcaa341e.

Next: verified cache-only reclamation/fresh safety gate, original1000-request
compatibility, then one complete4000W0. Do not repeat passed100/500 gates;
whole-pool binding is not full numerical/task-quality proof or U_ref.
PrimeLoRA D371/ServerlessLLM D414 remain sealed. No3B jump or figures yet.
