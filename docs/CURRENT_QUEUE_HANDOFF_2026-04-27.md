# Current Queue Handoff: 2026-04-27

This note records the live baseline-harness state for the PrimeLoRA paper
experiments. The broader paper handoff lives in:

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-04-27.md
```

## Active Queue

As of `2026-04-28 09:59 CST`, the next long queue is running in tmux:

```text
tmux session: paper_adapter_pool_p0
queue id:     20260428_095850_adapter_pool_p0
profile:      adapter_pool_p0
systems:      sglang serverlessllm vllm slora faaslora
section:      07_sensitivity_adapter_pool
planned:      a100/hot16, a200/hot24, a300/hot32, a400/hot40
```

The canonical `a500/hot48` right endpoint is the already completed Llama-2 7B
`s8` main round unless `adapter_pool_full_p0` is explicitly requested later.

Active files:

```text
queue env:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260428_095850_adapter_pool_p0/queue.env

first round:
/home/qhq/serverless_llm_baselines/results/paper_experiments/07_sensitivity_adapter_pool/20260428_095850_adapter_pool_p0_llama2_7b_r4000_a100_seed42_z1p0_hot16_rot500_s8_sensadpool_v1
```

Monitor:

```bash
tmux capture-pane -p -t paper_adapter_pool_p0 -S -120
```

Attach:

```bash
tmux attach -t paper_adapter_pool_p0
```

## Completed Queue

As of `2026-04-28 03:41 CST`, the queue completed successfully:

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
profile:      load_operating_p0
systems:      sglang serverlessllm vllm slora faaslora
section:      06_sensitivity_load_operating
completed:    s12 and s10 operating-load rounds
```

Key files:

```text
queue env:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/queue.env

queue log:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/logs/06_sensitivity_load_operating_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1.log

round dir:
/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating/20260427_112832_load_operating_p0_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1
```

Completed rounds:

```text
s12:
/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating/20260427_112832_load_operating_p0_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1

s10:
/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating/20260427_112832_load_operating_p0_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1
```

Both rounds have `90_compare.done` and include all five systems. Combined with
the existing s8 main point, the data support keeping Fig. 8 as operating-load
CE/cost-latency sensitivity. The result should not be written as latency
dominance: SGLang remains lower-latency, while FaaSLoRA has higher CE through
lower lifecycle cost.

## Monitor

The tmux session may still be open at a completed shell prompt:

```bash
tmux capture-pane -p -t paper_load_operating_p0 -S -120
```

Attach:

```bash
tmux attach -t paper_load_operating_p0
```

## Rule

Do not rerun this completed queue unless a later audit finds a concrete data
integrity issue. Do not drop ServerlessLLM from any paper-facing sensitivity
rerun.
