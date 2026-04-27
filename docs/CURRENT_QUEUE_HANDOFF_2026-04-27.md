# Current Queue Handoff: 2026-04-27

This note records the live baseline-harness state for the PrimeLoRA paper
experiments. The broader paper handoff lives in:

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-04-27.md
```

## Live Queue

As of `2026-04-27 13:03 CST`, the queue is active and healthy:

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
profile:      load_operating_p0
systems:      sglang serverlessllm vllm slora faaslora
active tag:   llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1
section:      06_sensitivity_load_operating
```

Key files:

```text
queue env:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/queue.env

queue log:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/logs/06_sensitivity_load_operating_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1.log

round dir:
/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating/20260427_112832_load_operating_p0_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1
```

Latest observed progress:

```text
done:         about 3768/4000
fail:         0
ETA:          about 6 minutes for the active system stage
SLO@5000 ms:  100%
```

The round state directory contained only `00_prep.done`; the first system stage
was still running.

## Monitor And Resume

Monitor:

```bash
tmux capture-pane -p -t paper_load_operating_p0 -S -120
```

Attach:

```bash
tmux attach -t paper_load_operating_p0
```

Resume only if the tmux session or queue log proves a real failure:

```bash
cd /home/qhq/serverless_llm_baselines
PAPER_QUEUE_ID=20260427_112832_load_operating_p0 \
PAPER_QUEUE_PROFILE=load_operating_p0 \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_long_experiment_queue.sh
```

## Rule

Do not drop ServerlessLLM from this queue. If that stage fails, fix the root
cause and resume with the same workload, trace, adapter subset, and queue id.
