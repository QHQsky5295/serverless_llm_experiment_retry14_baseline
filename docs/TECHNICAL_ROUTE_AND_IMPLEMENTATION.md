# Technical Route And Implementation

This is the current technical overview for PrimeLoRA/FaaSLoRA. Historical
retry-by-retry notes have been removed from this active document; use
`docs/对比实验日志.md` for detailed historical analysis.

## Problem

In serverless multi-LoRA inference, a request is not ready for generation just
because a backbone replica exists. The selected replica also needs the target
adapter in an executable state. Under bursty arrivals, elastic scale-out, and
shifting adapter popularity, requests can repeatedly hit adapter-replica
mismatch, causing adapter preparation to enter the TTFT critical path.

## Contribution 1: Hit-Aware Placement And Scale-Out Preparation

FaaSLoRA routes requests using adapter readiness, runtime load, and predicted
service cost. During scale-out, newly activated replicas are prepared with
likely-needed adapters so that the first requests they absorb are less likely to
fall onto cold adapter paths.

Observable evidence should come from:

- `TTFT_e2e` and `TTFT_service`
- `dispatch_admission_wait_ms`
- scale-up affected request metrics
- GPU-ready / warm / cold tier diagnostics where available

## Contribution 2: Hierarchical Adapter Residency

FaaSLoRA manages adapter placement across GPU, host, local storage, and remote
storage. The goal is to keep high-value adapters close to execution without
crowding out KV cache and batched inference working sets.

Observable evidence should come from:

- adapter tier hit rates
- LoRA I/O latency
- warm/gpu-ready request subsets
- ablation runs that disable residency or migration logic

These fields are mechanism metrics and should not be forced into cross-system
headline tables when baselines cannot observe them.

## Contribution 3: Coordinated Resource Control

FaaSLoRA coordinates adapter movement, runtime concurrency, and online
inference resource usage. This avoids treating adapter loading, KV cache, and
batched generation as independent knobs.

Observable evidence should come from:

- `TPOT`
- throughput
- cold-start and scale-up diagnostics
- GPU-second / active-idle resource metrics
- ablations that remove coordinated control

## Formal Metric Boundary

Cross-system headline metrics are limited to fields shared by all systems:

```text
TTFT_e2e avg/p95
E2E_e2e avg/p95
TPOT
Throughput_tok_s
Cost/req
CE
```

FaaSLoRA-only mechanism fields are for mechanism figures, ablations, and
diagnostics.

## Current Implementation Pointers

- Autoscaling and scale-out control: `faaslora/coordination/`
- Experiment stack and instance lifecycle: `faaslora/experiment/`
- Adapter preloading: `faaslora/preloading/`
- Residency and GPU memory monitoring: `faaslora/memory/`
- Routing and scheduling: `faaslora/scheduling/`
- Main runner: `scripts/run_all_experiments.py`

## Current Formal Execution

Formal cross-system execution is delegated to:

```text
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

The baseline harness owns shared trace generation, system sequencing, cleanup,
resume markers, and final comparison.
