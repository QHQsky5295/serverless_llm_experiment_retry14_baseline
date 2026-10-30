# D967: overlap-preserved activation and lifecycle-accounting audit

## Purpose

This 500-request run audits the first-principles candidate that preserves the
existing pre-initialization handoff work while using the per-request confirmed
owner snapshot for dispatch. It also verifies the lifecycle accounting change
added for IEEE-confirmed scale-out runtimes. The run is a diagnostic prefix,
not a formal G1/G2 point and not a baseline comparison.

## Frozen input and execution identity

- Model: Llama-2-7B.
- Trace: existing W2 shared trace; no new workload or adapter set was created.
- Adapter universe: existing 500-adapter map.
- Remote delivery: existing prepublished gzip artifact service and no-delay
  local-sim transport, unchanged from the D955/D965 diagnostic contract.
- Runner/config: `20261030_d955_7b_route_overlap_validation_100.yaml` and the
  existing D955 wrapper; only the run id/tag and diagnostic prefix count differ.
- Run id: `d967_20261031`.
- Git commit: `d2255fc6` (overlapped activation preparation restored) with the
  lifecycle-accounting parent commit `04919d3e`.
- Prefix: first 500 requests; `FAASLORA_FORMAL_RUN=0`.

## Observed outcome

The prefix completed 500/500 requests with zero request failures. The recorded
request-level values were average TTFT 734.50 ms, P95 TTFT 1,648.21 ms, average
TPOT 30.02 ms, and TTFT SLO attainment 100% for this diagnostic. These are
diagnostic observations only: the result does not contain an independent
`n_correct` measurement and therefore does not certify the joint correctness
criterion required by G1/G2.

The run produced four natural scale-up events and three physical scale-down
events. Five physical runtime lifecycles were present in the final accounting:
one initial runtime and four confirmed dedicated scale-out runtimes. Their
startup intervals were 45.29 s for the initial runtime and 31.35, 32.24,
32.04, and 32.34 s for the four scale-outs. The lifecycle summary reported
2,913.76 GPU-s in total (173.26 startup GPU-s and 2,772.84 ready GPU-s), an
average of 3.254 allocated replicas, and a maximum of four.

The lifecycle records include the physical device id, creation and readiness
offsets, scale-down or shutdown reason, and per-lifecycle GPU seconds. This is
the evidence that the IEEE-confirmed path is no longer silently omitted from
resource accounting. The result also keeps the normal request-triggered
transfer and preparation work in the measured path.

## Interpretation

The result supports the mechanism hypothesis that cancelling and rerunning the
whole handoff plan at worker readiness is unnecessary. Preserving overlap
restores scale-out startup to roughly 31--32 s, whereas the D965 late-replan
candidate had 40--62 s activation intervals and a worse full replay. Dispatch
still obtains a fresh confirmed owner/tier snapshot for each request, so
correctness does not rely on the trigger-time preparation plan being a current
routing prediction.

The result is not evidence that PrimeLoRA has met G1/G2. A formal run still
needs the frozen full request count, independent correctness evidence, common
TTFT/TPOT SLO, and the complete lifecycle/resource audit.

## Artifact paths

- Result JSON:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d967_20261031/7b_route_overlap_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d967_7b_overlap_preserved_lifecycle_500.json`
- Physical deployment outcome:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d967_20261031/7b_route_overlap_validation_100/launch.launch/physical_deployment/main_outcome.json`

