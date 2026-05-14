# True-remote vs local-sim comparison

- local: `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/backend_portability/table_backend_portability_data.csv`
- true remote: `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/backend_portability_real_remote_v1_7b3b/table_backend_portability_data.csv`

| model_key | system_key | system | TTFT avg % | E2E avg % | Cost/req % | CE % |
| --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | sglang | SGLang | +19.44% | +1.79% | -1.06% | -0.70% |
| llama2_7b | primelora_sglang | PrimeLoRA-SGLang | +12.40% | +2.08% | -0.63% | -1.42% |
| llama2_7b | vllm | vLLM | -3.22% | -2.24% | +11.68% | -8.40% |
| llama2_7b | primelora_vllm | PrimeLoRA-vLLM | +16.53% | +3.10% | +0.40% | -3.40% |
| llama32_3b | sglang | SGLang | +89.12% | +7.66% | +0.00% | -7.11% |
| llama32_3b | primelora_sglang | PrimeLoRA-SGLang | +144.86% | +19.53% | +3.99% | -19.55% |
| llama32_3b | vllm | vLLM | +45.72% | +6.62% | +0.80% | -6.95% |
| llama32_3b | primelora_vllm | PrimeLoRA-vLLM | +23.36% | +11.67% | +1.62% | -11.88% |
