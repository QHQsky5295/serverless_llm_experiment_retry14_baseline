# Historical Queue Handoff: 2026-04-27

> Historical queue handoff. This queue is no longer active. For the current
> PrimeLoRA/FaaSLoRA project state, start from
> `/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-05-25.md`.

This note records the live baseline-harness state for the PrimeLoRA paper
experiments. The broader paper handoff lives in:

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-04-27.md
```

## Historical Active Queue

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
  `max_cpu_loras=16`, `lora_registration_mode=dynamic`, and
  `dynamic_lora_routing=adapter_hash`.
- vLLM no longer statically registers all 500 LoRA modules into every Qwen2.5-7B
  OpenAI API replica. Static registration with `max_cpu_loras=16` passed a
  384-request probe but tripped the host-memory guard around 900 completed
  requests; shrinking to `max_cpu_loras=8` still failed a 1200-request probe.
  The current fix keeps the same 500-adapter sampled universe and 100% LoRA-bound
  trace, but loads adapters through vLLM's runtime LoRA API on first use. The
  dynamic mode avoids per-replica static registration while preserving the
  `dp4/tp1` serving envelope.
- 2026-05-04 复查进一步确认：单纯 dynamic runtime LoRA + request round-robin
  仍会让同一 adapter 随请求流落到多个 endpoint，长 replay 下会重复加载并继续
  推高 host-side footprint。根因修复不是继续收缩 `max_cpu_loras`，而是给
  standalone vLLM dynamic 模式加入 adapter-sticky endpoint selection。当前正式默认
  `adapter_hash`：同一个 adapter 固定归属一个 endpoint，避免 full 4000 下
  request round-robin 或 hot-pair routing 的 lifetime endpoint-adapter 加载爆炸。
  这样保留四个独立 serving replicas 和 vLLM 的 `max_num_seqs=8/max_loras=8`
  性能包络，不通过降低并发、resident 上限或 DP/TP 拓扑换稳定性。该选择只是
  vLLM baseline 的运行时 LoRA 注册负载约束，不引入 PrimeLoRA 的
  readiness-aware routing、scale-out warmup、residency 或 admission 机制。
- 同日追加横向排查：PrimeLoRA/FaaSLoRA 虽然也使用 vLLM 后端，但走的是
  `AsyncLLMEngine + LoRARequest` 直连路径，并由自身 residency/path resolver
  给每个请求传入 adapter 路径；standalone vLLM baseline 走的是
  OpenAI API server + `/v1/load_lora_adapter` runtime registration 路径。此前
  Qwen2.5-7B 崩溃发生在后者的 OpenAI API frontend multiprocessing + runtime
  LoRA 组合上，不等价于所有 vLLM 后端都会崩。为消除同族风险，Qwen-family
  standalone vLLM dynamic LoRA 现在默认增加 `--disable-frontend-multiprocessing`，
  保持 `dp/tp`、`max_num_seqs`、`max_loras`、`max_cpu_loras` 不变，只改变
  OpenAI API frontend 进程形态。
- 2026-05-04 真实主机 1200-request Qwen2.5-7B vLLM preflight：
  `dp4/tp1,max_num_seqs=8,max_loras=8,max_cpu_loras=16`,
  `lora_registration_mode=dynamic`,
  `dynamic_lora_routing=adaptive_hot_pair_hash`,
  `disable_frontend_multiprocessing=1`。结果为 `ok=1200/1200, fail=0`，
  无 `trace_expected` token fallback，越过旧 static 路径约 900 请求处的失败区间；
  吞吐约 `108.85 tok/s`，`TTFT avg/p95 = 6682.7/41820.8 ms`，`TPOT avg = 130.6 ms`。
  这轮是 bounded stability preflight，不是论文正式 4000-request 结果；formal
  gate 正确拒绝将 `completed=1200,total=4000` 的 summary 写入论文对比。
- 2026-05-04 真实主机 Qwen2.5-14B vLLM smoke：
  `dp2/tp2,max_num_seqs=2,max_loras=2,max_cpu_loras=8`,
  `lora_registration_mode=dynamic`,
  `dynamic_lora_routing=adaptive_hot_pair_hash`,
  `disable_frontend_multiprocessing=1`。两个 TP=2 replica 均成功启动，分别通过
  `/v1/load_lora_adapter` 加载 LoRA 并完成 1-token smoke request；smoke-only
  结束后四张 GPU 均回到约 15 MiB，未残留 vLLM server 进程。
- 2026-05-06 追加 Qwen2.5-14B vLLM smoke：在当前默认
  `dynamic_lora_routing=adapter_hash` 下，两个 TP=2 replica 均成功启动、加载 LoRA
  并完成 1-token 请求；smoke-only wrapper 已跳过 formal compare，命令以 0 退出，
  GPU/进程清理干净。
- 不要用旧 `paper_backbone_robustness_v2` 终端判断当前修复是否有效。该终端是
  2026-05-01 启动的旧队列，失败 launch spec 中没有
  `lora_registration_mode: dynamic`，也没有
  `disable_frontend_multiprocessing: true`，因此属于旧 static/OpenAI
  frontend-multiprocessing 路径；它在约 900 completed 后触发 32 GiB
  host-memory guard 并中止是预期保护行为，不能作为修复后的正式结果。
- vLLM runner now checks host `MemAvailable` during launch and replay and aborts
  before system OOM.
- vLLM runner monitors server PIDs during replay and rejects partial runs when a
  replica dies.
- replay now supports bounded preflight through `--max-requests` and aborts as
  soon as repeated request failures prove a run invalid.
- full-round summary discovery no longer assumes `dp4_tp1`, so TP=2 backbone
  runs validate correctly.
- Real-host Qwen2.5-7B vLLM verification history: static `dp4/tp1,max_cpu_loras=16`
  completed a bounded 384-request replay with `ok=384/384`, `fail=0`,
  `TTFT=1661.8ms`, `TPOT=73.0ms`, and `Tok/s=106.21`, but later failed the long
  replay. A `max_cpu_loras=8` long probe also failed, so it is not the root fix.
  The current validation target is
  `dp4/tp1,lora_registration_mode=dynamic,dynamic_lora_routing=adapter_hash`.
  Traceback lines in the vLLM logs are from controlled API-server shutdown after
  successful replay, not request failures.
- Llama-2 13B and Qwen2.5 14B preflight generated 4000 LoRA-bound requests and
  500-adapter subsets. vLLM and S-LoRA dry-runs for these backbones both expose
  500 adapters and resolve to `dp2/tp2`.
- 2026-05-06 更新：当前 S-LoRA 上游实现只提供 Llama/Llama2 model backend。
  Qwen2.5 正式启动会报 `can not support qwen2 now`，因此 backbone queue 对
  Qwen-family profile 显式写 `40_slora.unsupported` 并跳过 S-LoRA；Llama-2
  13B 仍继续运行 S-LoRA。不要把 Qwen 上缺失的 S-LoRA 行当作实验失败或
  静默替换结果。外层 long-queue 的完成校验也使用 profile-aware supported
  systems，Qwen-family compare 不再强制要求 S-LoRA 行。

There is also a separate remote-access issue: `frpc.service` is repeatedly
failing to connect to `120.26.187.54:7000`, and an unrelated-looking
`yk0Wk9DV.service` repeatedly tries to execute missing `/bin/YXt5BHl6`. These
services should be inspected outside the experiment harness; if SSH depends on
frp, remote access can fail even when GPUs and experiments are idle.
As of `2026-04-30 19:12 CST`, `systemctl --failed` lists no failed units, but
both services are still in `activating (auto-restart)`, so they remain separate
remote-access noise to resolve outside the paper runner.

Backbone robustness 建议使用新的 queue id 重新跑，避免旧 static launch 目录和
旧失败日志继续干扰判断。若确实要断点续跑旧 queue，必须先确认新生成的 vLLM
launch spec 中包含 `lora_registration_mode: dynamic` 与
`disable_frontend_multiprocessing: true`。

推荐启动命令：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_backbone_robustness_p0

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
