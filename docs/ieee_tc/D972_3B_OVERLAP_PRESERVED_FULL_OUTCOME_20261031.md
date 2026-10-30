# D972: 3B overlap-preserved full development outcome (2026-10-31)

## Scope and status

D972 is a complete 4,000-request development run for the Llama-3.2-3B workload after restoring the pre-init artifact-preparation overlap. It is not a formal comparison point: the run uses the frozen seed-42 trace and records `formal_run=false`. The result is retained as implementation and measurement evidence; it must not be used as a seed-level confidence interval or as a baseline ranking claim.

The run uses the IEEE confirmed-state path with the existing vLLM backend, dedicated single-GPU runtimes, the historical 3B shared trace and 500-adapter map, and the prepublished-gzip remote artifact endpoint on the artifact node. The 3B endpoint was served on port 18080. The 7B endpoint was not involved in this run.

## Reproducibility records

- Run tag: `d972_3b_overlap_preserved_lifecycle_full`
- Result: `results/ieee_tc/p2_backend_qualification/d972_20261031/3b_overlap_lifecycle_full/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d972_3b_overlap_preserved_lifecycle_full.json`
- Physical outcome: `.../launch.launch/physical_deployment/main_outcome.json`
- Workload: `llama32_3b_auto500_formal4000_s8`, 4,000 requests, 500-adapter universe, seed 42
- Generation contract: `fixed_length_greedy_v1`
- Transport: real HTTP artifact client with `prepublished_gzip_v1`; request-time packing count 0
- Bandwidth mode: local-sim no-delay (`bandwidth_mib_s=0`), therefore no physical-link capacity claim is made
- Remote content manifest: `ac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e`
- Remote transfers: 505 published transfers; every recorded transfer is content-verified by the owner-publication contract

## End-to-end outcome

| Metric | D972 |
|---|---:|
| Completed / total | 4,000 / 4,000 |
| Failed | 0 |
| Mean TTFT (arrival-to-first-token) | 798.9 ms |
| P95 TTFT | 2,316.1 ms |
| P99 TTFT | 3,309.3 ms |
| Mean service TTFT | 478.6 ms |
| Mean TPOT | 16.63 ms |
| Mean E2E | 2,649.7 ms |
| P95 E2E | 5,511.4 ms |
| TTFT SLO attainment at 5,000 ms | 99.925% |
| Throughput | 1.009 req/s |
| Total output tokens | 458,224 |

The run therefore demonstrates complete execution and native timing coverage under the development configuration. It does not by itself establish that PrimeLoRA is better than a baseline or that the result generalizes across seeds.

## Scale-out and lifecycle accounting

The run began with one GPU and naturally scaled to three GPUs through two confirmed scale-up events. Both activations preserved preparation overlap:

- activation 1: 35.37 s engine activation, submitted at request 208;
- activation 2: 36.09 s engine activation, submitted at request 221.

The physical lifecycle ledger contains three records (initial GPU plus two scale-outs), all with explicit creation, ready, shutdown, startup, and ready-state intervals. The measured lifecycle totals are:

- physical lifecycle GPU-seconds: 12,194.16;
- startup GPU-seconds: 120.04;
- ready GPU-seconds: 12,074.12;
- active GPU-seconds: 7,667.98;
- idle-ready GPU-seconds: 4,406.14;
- average allocated GPUs: 2.827;
- maximum allocated GPUs: 3.

The lifecycle fields are now emitted by the IEEE dedicated scale-up path itself. This is a measurement correction relative to older development results whose scale-out lifetime records were incomplete; it is not a performance optimization.

## Request-contract audit

All 4,000 request records satisfy the execution contract:

- `success=true`: 4,000/4,000;
- `output_contract_match=true`: 4,000/4,000;
- native TPOT timing observed: 4,000/4,000;
- `generation_contract=fixed_length_greedy_v1`: 4,000/4,000;
- requested and observed completion-token counts match: 4,000/4,000;
- canonical prompt SHA-256 present: 4,000/4,000;
- completion token-ID SHA-256 present: 4,000/4,000;
- completion token source is vLLM native token IDs: 4,000/4,000.

These checks establish contract and timing integrity. They do not establish semantic numerical correctness of the generated text, so this run intentionally reports no `n_correct` value and must not be interpreted as a semantic-accuracy audit.

## Interpretation for the optimization sequence

D972 confirms that restoring pre-init preparation overlap removes the broad regression seen in the late-ready full-replan candidate while retaining confirmed owner/tier snapshots at request routing and admission. The run is therefore suitable for the next cross-model correctness audit and for deciding whether a new formal configuration is warranted. Before any baseline comparison, the 3B configuration still requires the planned validation/freeze procedure and an independent semantic-correctness check; no configuration should be selected from this single development outcome by performance alone.

