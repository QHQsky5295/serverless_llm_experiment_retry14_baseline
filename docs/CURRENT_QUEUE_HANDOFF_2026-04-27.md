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

## Crash Update: 2026-04-30

The active long queue has moved to backbone robustness:

```text
tmux session before reboot: paper_backbone_robustness_p0
queue id:                   20260429_115544_backbone_robustness_p0
profile:                    backbone_robustness_p0
section:                    08_backbone_robustness
```

The server was physically rebooted after becoming unreachable. After reboot,
`tmux ls` reported no active sessions and all GPUs were idle. The queue had
finished Qwen2.5-7B SGLang and ServerlessLLM, then crashed in the Qwen2.5-7B
vLLM stage.

Root cause from journal and runner logs:

- vLLM/Qwen2.5-7B V0/eager exhausted host memory because the long replay kept
  too much host-side LoRA/runtime state alive across four independent replicas.
- Linux OOM killed vLLM APIServer/engine processes during replay. The partial
  replay is invalid and must not be used.
- The 500 LoRA value remains the formal sampled adapter pool. The vLLM launch
  spec still exposes all 500 LoRA modules, and the shared trace has `4000/4000`
  LoRA-bound requests.
- A temporary `dp2/tp2` workaround avoided the host OOM but cut the number of
  independent service replicas from four to two. A real 384-request probe then
  produced unusable queueing (`TTFT` above 100s), so `dp2/tp2` is not acceptable
  for the Qwen2.5-7B vLLM paper baseline.

Applied runner fixes:

- Qwen2.5-7B vLLM now keeps `dp4/tp1` on the same four-GPU budget so all four
  GPUs remain independent serving replicas.
- Qwen2.5-7B vLLM formal stage raises the execution envelope to
  `max_num_seqs=8`, `max_loras=8`, `max_num_batched_tokens=4096`,
  `max_cpu_loras=16`.
- vLLM `max_cpu_loras` remains a bounded per-replica CPU LoRA cache; the full
  500-adapter sampled universe is still registered through `--lora-modules`.
  The Qwen2.5-7B formal override keeps this cache at `16/500` so the replay
  stays above the host-memory guard while still registering all 500 adapters.
- vLLM runner now checks host `MemAvailable` during launch and replay and aborts
  before system OOM.
- vLLM runner monitors server PIDs during replay and rejects partial runs when a
  replica dies.
- replay now supports bounded preflight through `--max-requests` and aborts as
  soon as repeated request failures prove a run invalid.
- full-round summary discovery no longer assumes `dp4_tp1`, so TP=2 backbone
  runs validate correctly.
- Real-host Qwen2.5-7B vLLM verification passed after the final fix:
  `dp4/tp1,max_cpu_loras=16` completed a bounded 384-request replay with
  `ok=384/384`, `fail=0`, `TTFT=1661.8ms`, `TPOT=73.0ms`, and `Tok/s=106.21`.
  Cleanup released all GPUs. Traceback lines in the vLLM logs are from
  controlled API-server shutdown after successful replay, not request failures.
- Llama-2 13B and Qwen2.5 14B preflight generated 4000 LoRA-bound requests and
  500-adapter subsets. vLLM and S-LoRA dry-runs for these backbones both expose
  500 adapters and resolve to `dp2/tp2`.

There is also a separate remote-access issue: `frpc.service` is repeatedly
failing to connect to `120.26.187.54:7000`, and an unrelated-looking
`yk0Wk9DV.service` repeatedly tries to execute missing `/bin/YXt5BHl6`. These
services should be inspected outside the experiment harness; if SSH depends on
frp, remote access can fail even when GPUs and experiments are idle.
As of `2026-04-30 19:12 CST`, `systemctl --failed` lists no failed units, but
both services are still in `activating (auto-restart)`, so they remain separate
remote-access noise to resolve outside the paper runner.

Resume command, using the same queue id. If `paper_backbone_robustness_p0` is
still just an attached shell, run the command there; otherwise create a new
tmux with the same name:

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_backbone_robustness_p0

PAPER_QUEUE_ID=20260429_115544_backbone_robustness_p0 \
PAPER_QUEUE_PROFILE=backbone_robustness_p0 \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_backbone_robustness_queue.sh
```

Monitor:

```bash
tmux attach -t paper_backbone_robustness_p0
tmux capture-pane -p -t paper_backbone_robustness_p0 -S -160
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
```
