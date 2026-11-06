# D1077 — 3B load-only routing candidate audit (2026-11-06)

> Correction (D1080): this run uses the NoRouting selector and a 240-second
> preparation window, not paper Full under the frozen 60-second main protocol.
> The value previously labelled physical GPU-seconds below is a scenario
> infrastructure estimate; the physical lease union is 6596.153619553894 GPU-s.
> Use [D1080](D1080_DIAGNOSTIC_IDENTITY_AND_RESOURCE_CORRECTION_20261106.md)
> for the corrected identity, Type-1 quantiles, and next action. The historical
> narrative below is retained for provenance, not current acceptance guidance.

## Decision

D1077 is an exploratory PrimeLoRA mechanism candidate, not a formal ranking
point.  It completed the full 4,000-request native replay without transport,
runtime, or OOM failure and reached the frozen 3B timing gate.  It does **not**
yet establish that replacing service-bin routing with load-only routing caused
the improvement, because the run also enabled effective-capacity admission,
whereas D1074 and D1047 (the historical service-bin Full controls) recorded that
gate as disabled.  A matched service-bin control with the same admission gate is
therefore required before attributing the gain to routing.

The run remains useful as a qualified candidate for the next control: it shows
that the load-only key is compatible with the current handoff, hierarchical
residency, confirmed-arrival prefetch, and admission implementation on the 3B
trace.

## Frozen identity and provenance

| Item | Value |
|---|---|
| Repository | `/home/qhq/serverless_llm_experiment_retry14_baseline` |
| Branch | `retry14_continuous_queue_v2` |
| Config commit | `39e65069bc8569e545d1690733a869663e0a5cd4` |
| Config SHA-256 | `7362ff534e5cad2feb575c9b68e78dafe19fcab7815b9899cb1f9e2e9a0b2884` |
| Run key | `d1077_3b_load_only_full_w1_replay240_retry2` |
| Raw result | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1077_20261106/d1077_3b_load_only_full_w1_replay240_retry2` |
| Result SHA-256 | `bec4b261076e065d42629954452e1e82e46175b17e236251b341a872ca2af075` |
| Launch SHA-256 | `f97fb3982647a008b6f1b847fe214bf57bf4f137cd027d7d2db088d112dad1a8` |
| Physical summary SHA-256 | `efbd8a4be6f079630e05ceb271fb9849ee876391ec3696540e66cfc318ac3` |
| Trace | shared W1, 4,000 requests, seed 42, source SHA recorded in result metadata |
| Generation | `fixed_length_greedy_v1`; prompt cap 759; output cap 256; temperature 0; top-p 1 |
| Artifact path | real HTTP client with read-only `prepublished_gzip_v1` delivery |
| Trace role | exploratory; `formal_run=false`; no publication figure generated |

Two earlier D1077 launch attempts are retained as invalid prelaunch records:
the first violated the ordinary-file LRU ownership contract, and the second
used incorrect frozen profile-context digests.  Neither reached request
execution and neither is included in the metrics below.  The `retry2` key is
the only completed run.

## Mechanism configuration

The completed candidate recorded:

- `ieee_selection_strategy=load_only`;
- confirmed-arrival routing policy;
- scale-out handoff enabled;
- hierarchical residency enabled;
- effective-capacity admission enabled;
- forecast scale-down disabled;
- one to three dedicated single-GPU replicas, `max_num_seqs=8`, and the same
  W1 trace, model, profiles, cache reset, and remote artifact contract used by
  the candidate configuration.

The historical D1074 and D1047 service-bin controls recorded
`ieee_selection_strategy=service_bins` and
`effective_capacity_admission_enabled=false`.  Consequently, the numerical
differences below are descriptive candidate-vs-history differences, not a
one-factor causal estimate of confirmed-tier or load-only routing.

## Observed outcome

| Quantity | D1077 load-only candidate |
|---|---:|
| Offered / completed / failed | 4,000 / 4,000 / 0 |
| Average TTFT (ms) | 460.859 |
| P95 TTFT (ms) | 1,069.542 |
| Average TPOT (ms) | 16.943 |
| Average E2E (ms) | 2,287.304 |
| P95 E2E (ms) | 4,964.606 |
| Monetary CE | 380.073 |
| Infrastructure CE | 314.409 |
| Physical GPU-seconds | 6,952.670 |
| Average allocated GPUs | 2.737 |
| Scale-up events | 2 |
| Remote transfers | 132 |
| Cache hit / GPU hit | 96.70% / 81.0% |

For reference, D1074 service-bin history reported average TTFT 498.168 ms,
P95 TTFT 1,110.493 ms, average TPOT 18.032 ms, average E2E 2,412.221 ms,
monetary CE 354.760, infrastructure CE 292.260, and 7,092.233 physical
GPU-seconds.  D1047 reported average TTFT 454.156 ms, P95 TTFT 1,114.122 ms,
average TPOT 17.417 ms, average E2E 2,323.526 ms, monetary CE 353.431,
infrastructure CE 300.657, and 7,157.322 physical GPU-seconds.  These values
are retained as historical context, not as a clean causal comparison.

## Common timing gate

The D648 thresholds were applied without recalibration:

- prompt tokens `<=506`: TTFT `<=1466.0508 ms`, TPOT `<=29.9219 ms`;
- prompt tokens `>506`: TTFT `<=2539.6487 ms`, TPOT `<=36.2155 ms`.

The result has 3,983/4,000 TTFT passes, 3,858/4,000 TPOT passes, and
3,844/4,000 joint timing passes (96.10%).  The required count is 3,800.
This is a timing-only upper bound: the physical summary and result metadata
correctly retain `n_correct=null` and do not infer numerical adapter
correctness from fixed-length termination or HTTP success.

The append-only native occupancy audit is at
`/home/qhq/serverless_llm_experiment/paper_results/ieee_tc/p2_backend/d1077_load_only_full_w1_audit/native_timeline/summary.json`.
It verified 4,000 native-success terminal records, a single timing domain, no
failed requests, and 2,160 resource samples.  Occupancy values are descriptive
request intervals and are not substituted for physical GPU billing.

## Next required control

Run a new, unique-key `service_bins` candidate using the D1077 configuration
with only `ieee_selection_strategy` changed to `service_bins`, retaining
effective-capacity admission enabled and every other frozen field.  Compare
that control with D1077 on the same W1 trace.  Only if this matched comparison
reproduces the short-output tail improvement may the paper attribute the effect
to the placement key; otherwise report a joint routing-plus-admission trade-off.

No old result, figure directory, or user-modified manifest was overwritten.
