# vLLM + LoRA Notes On RTX 3090

This file records current runtime lessons that still matter. Older
model-specific rollback notes were removed.

## Current Role

vLLM appears in two places:

1. FaaSLoRA uses a vLLM-based serving path internally.
2. The baseline workspace runs standalone vLLM as a separate paper baseline.

The standalone vLLM baseline is documented under:

```text
/home/qhq/serverless_llm_baselines/vLLM_project
```

## Current Stable Rules

- Llama-2 7B single-GPU LoRA replica paths can use CUDA graph when
  `enforce_eager: auto` resolves to `false`.
- TP or known-risk model paths may still require conservative eager execution.
- `VLLM_USE_V1`, `VLLM_ATTENTION_BACKEND`, and
  `VLLM_USE_FLASHINFER_SAMPLER` must be explicit in formal harnesses.
- Token accounting must come from actual response usage or local tokenizer
  counting of the generated text. It must not fall back to raw trace expected
  tokens in formal results.

## Current Known Risk

If TPOT suddenly returns to the 60ms range for Llama-2 7B on this host, first
check whether a path accidentally enabled eager mode or failed to propagate the
vLLM runtime env into tmux-launched processes.

## 2026-05-04 Qwen vLLM Baseline Boundary

Qwen2.5-7B 的 standalone vLLM baseline 使用 OpenAI-compatible server 的
runtime LoRA API 来避免启动时静态注册 500 个 sampled adapters。这里的
500 adapter 是正式采样池大小；每个 replay 请求仍然只绑定一个 LoRA adapter。

本次 backbone robustness 队列失败的根因不是“请求一次绑定了 500 个 LoRA”，也
不是 PrimeLoRA/FaaSLoRA 内部 vLLM 后端路径。问题出在 standalone vLLM OpenAI
runtime LoRA API：`/v1/load_lora_adapter` 加载过的 adapter 不会自动按照
`--max-cpu-loras` 从 API server registry 中释放，长尾请求会让每个 endpoint
的已注册 adapter 数持续增长，最终触发 host-memory guard。

baseline harness 已在 `/home/qhq/serverless_llm_baselines` 中修复：动态 LoRA
replay 现在维护 per-endpoint LRU registry，并在加载新 adapter 前通过
`/v1/unload_lora_adapter` 卸载 inactive adapter。Qwen2.5-7B vLLM baseline
默认使用 `max_cpu_loras=24` 和
`dynamic_lora_max_loaded_per_endpoint=24`。这保留 500-adapter universe 与
请求级 LoRA 语义，同时把 standalone vLLM OpenAI server 的 host-memory 使用
限定在可控范围内。

2026-05-06 的继续排查说明：后续队列再次失败不是因为 vLLM 请求链路继续报错，
而是因为 baseline harness 把 `VLLM_HOST_MIN_MEM_GB=32` 当成硬停止线。当前
Qwen2.5-7B DP4 baseline 会在同一台 125 GiB host 上启动四个单卡 vLLM runtime，
这会自然消耗大量 host memory；`MemAvailable` 接近 32 GiB 时仍然不等价于
Linux OOM，也不等价于 SSH 即将失联。因此 baseline workspace 已把 32 GiB 调整
为预警线，把 16 GiB 作为硬停止线，并为每轮 vLLM stage 输出
`*_vllm_mem_watch.csv`。这个修改不降低 vLLM 的 `max_loras`、
`max_cpu_loras`、`max_num_seqs` 或 GPU 拓扑，因此不会通过牺牲 vLLM baseline
性能来换取“更容易跑完”。

随后对 Qwen2.5-7B DP4 vLLM 长跑的 RSS 监控显示，`/v1/models` 中的 resident
LoRA 数已被限制在 24，但 vLLM runtime-LoRA unload 不会立刻释放 host allocator
内存；如果继续使用 `adaptive_hot_pair_hash`，热 adapter 会复制到两个 endpoint，
从而增加每个进程曾经加载过的 adapter 集合。baseline workspace 已将 Qwen2.5-7B
vLLM 的默认动态 LoRA 路由改为 `adapter_hash`，让每个 adapter 固定归属一个
endpoint，降低重复加载和 unload churn。这仍然保持 500-adapter 采样池与
100% LoRA-bound 请求不变，也不降低 vLLM 的并发、resident 上限或 DP4/TP1 拓扑。

PrimeLoRA/FaaSLoRA 自身使用直接的 `AsyncLLMEngine + LoRARequest` 控制路径和
系统内 adapter residency/resolution，不走 standalone OpenAI API server 的
runtime registry。因此这个问题不直接作用于 PrimeLoRA 的当前后端，但 Qwen
family 的正式实验仍应保留 GPU/host-memory preflight 与小规模 probe，防止
未来配置变更引入相同类别的问题。

## 2026-05-06 Qwen PrimeLoRA Capacity Gate

Qwen2.5-7B publicmix 的 PrimeLoRA/FaaSLoRA profile 曾被临时收紧到
`max_num_seqs=2` / `runtime_concurrency_cap=2`。这一路径可以避免崩溃，但在
backbone robustness 正式队列中会把四个 runtime 的总并发压到 8，导致 4000-request
s8 replay 早期形成持续 backlog，TTFT tail 被系统层排队放大。因此这不是可用于论文的
性能包络。

已在真实主机上用同一 Qwen2.5-7B model profile、同一 500-adapter subset、同一
formal s8 trace 前 800 个请求做 cap4 gate：

```text
FAASLORA_RUNTIME_CONCURRENCY_CAP=4
FAASLORA_MAX_NUM_SEQS=4
FAASLORA_MAX_NUM_BATCHED_TOKENS=4096
FAASLORA_MAX_LORAS=6
```

结果为 `800/800` 成功、`fail=0`，退出后四张 GPU 均回到约 15 MiB，主机
`MemAvailable` 回到约 117 GiB，无 vLLM/FaaSLoRA worker 残留。落盘结果文件：

```text
results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_qwen2p5_7b_probe800_cap4_s4096_faaslora.json
```

该 probe 不是论文正式数据，只作为容量与稳定性 gate。它证明 cap4 在当前
4xRTX3090 Qwen2.5-7B publicmix 路径上稳定，并且避免 cap2 对 PrimeLoRA 的明显低估。
因此正式 backbone robustness 队列使用 cap4 配置继续运行；后续若修改 Qwen family
profile，仍需先通过 bounded probe，再进入 4000-request formal replay。
