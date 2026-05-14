# True-remote vs local-sim comparison

- local: `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main_remote_fair_local_sim_v4_7b3b/table1_end_to_end_data.csv`
- true remote: `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main_remote_fair_real_remote_v1_7b3b/table1_end_to_end_data.csv`

| model | system_key | system | TTFT avg % | E2E avg % | Cost/req % | CE % |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2 7B | sglang | SGLang | +14.99% | +1.80% | +0.61% | -2.37% |
| Llama-2 7B | vllm | vLLM | -0.10% | -0.07% | +10.77% | -9.66% |
| Llama-2 7B | slora | S-LoRA | +23.27% | +8.73% | +10.80% | -17.00% |
| Llama-2 7B | serverlessllm | ServerlessLLM | +0.18% | +0.18% | +0.13% | -0.31% |
| Llama-2 7B | faaslora | PrimeLoRA | +16.53% | +3.10% | +0.40% | -3.40% |
| Llama-3.2 3B | sglang | SGLang | +58.45% | +5.97% | -0.28% | -5.37% |
| Llama-3.2 3B | vllm | vLLM | +40.82% | +5.90% | +0.47% | -6.01% |
| Llama-3.2 3B | slora | S-LoRA | +5.75% | +6.67% | +26.82% | -26.08% |
| Llama-3.2 3B | serverlessllm | ServerlessLLM | +0.43% | +0.42% | +0.58% | -0.99% |
| Llama-3.2 3B | faaslora | PrimeLoRA | +23.36% | +11.67% | +1.62% | -11.88% |
