# D767: reuse the existing S-LoRA 3B GQA/BMM path

2026-10-23. Necessary measurement-entry adaptation, not performance tuning.
The historical `run_slora_fair_experiment.sh` selects native BMM for GQA;
Llama-3.2-3B has 24 query heads and 8 KV heads. The 7B-only packed HOST
representation is not applicable. Preserve the existing native GQA source,
four TP1 replicas, round-robin, sequential startup, swap and token capacities.

`validate_slora_service_config` now accepts an explicit Boolean `bmm=true`
only with `host_layout=native_v1`. It does not silently switch algorithms or
retry a failed configuration. Measurement already checks the actual BMM engine,
adapter binding and native output events. No kernel or source-tree patch added.
52 CPU configuration/measurement/service tests pass, including the new negative
layout/type tests. Actual 3B GPU execution and complete current-contract results
are still pending; CPU success is not numerical correctness qualification.

The fixed [upstream RPC](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)
contains the BMM load/offload and forward branch. Existing local GQA shape
adaptations remain explicitly disclosed; this is not unmodified upstream GQA
support. In particular, BMM adapter tensor transfers are not the packed BGMV
unified-adapter-paging path and must not be described as proving all S-LoRA
paper mechanisms. No performance-based baseline configuration search is added.

Full source and reuse evidence:
`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D767_SLORA_3B_ENTRY_20261023.md`.
Raw CPU checks:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d767_20261023/`.
