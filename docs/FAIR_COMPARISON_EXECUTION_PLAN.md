# 对比实验执行规范

本文档是当前 baseline 工作区的权威执行入口。旧的 `e2e_v1/e2e_v2`、
三系统复现计划、SkyServe/Punica 优先级计划已经废弃；如与历史日志冲突，
以本文档和当前脚本为准。

## 1. 当前正式实验范围

当前正式主线服务 PrimeLoRA/FaaSLoRA 论文横向对比：

- 场景：serverless / serverful many-LoRA inference。
- 请求：100% LoRA requests。
- 主负载：4000 requests，500 sampled adapters。
- 调试负载：500 requests，仅用于 bring-up/debug，不写主结论。
- 热度：Zipf hotness，hot set cap = 48。
- 热点轮换：4000 请求主负载每 500 请求轮换一次热点。
- GPU 预算：默认 4 张 RTX 3090，系统之间不得并发运行。
- 指标口径：`metric_schema_version=e2e_v3`。

当前正式 Llama-2 7B round tag：

```text
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
```

## 2. 当前系统顺序

完整 round 默认按以下顺序执行：

```text
prep -> SGLang -> ServerlessLLM -> vLLM -> S-LoRA -> FaaSLoRA -> compare
```

各系统定位：

- `FaaSLoRA`: 本文主系统。
- `SGLang`: serverful many-LoRA serving engine baseline。
- `vLLM`: standalone general serving baseline。
- `ServerlessLLM`: general serverless LLM baseline。
- `S-LoRA`: serverful multi-LoRA paper baseline。
- `Punica`: Llama-2 7B scoped auxiliary baseline，不进入完整四 backbone 主表。

## 3. 权威脚本

启动新的完整 round：

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

从任意目录恢复未完成 round：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

查看恢复目标但不启动：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh --dry-run
```

指定恢复某个 round：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<round_dir>
```

如果上一次 tmux session 已经失败并停在 shell 提示符，保留断点但重建
session：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<round_dir> \
  --restart-session
```

## 4. 断点续跑规则

每个 round 目录包含：

```text
round.env
state/*.done
logs/*.log
shared_artifacts/
raw/
comparison/
```

规则：

1. 已完成阶段写入 `state/<stage>.done`。
2. 失败后修复代码，再运行 `resume_fair_round_tmux.sh`；如果旧 tmux
   session 仍存在但已经停在失败后的 shell，使用 `--restart-session`。
3. 续跑会跳过已完成阶段，从第一个未完成系统开始。
4. 每个系统运行前必须清理已知遗留进程，并检查 GPU compute 进程。
5. 不手工移动或删除 `state/*.done`，除非明确要重跑某个阶段。
6. FaaSLoRA 原始结果目录可能是 symlink。runner 查找 FaaSLoRA 结果时必须
   跟随 symlink，并在断点恢复时优先收集已有合法结果；如果实验已经完成但
   post-collection 失败，不应重复重跑 FaaSLoRA。
7. GPU 清洁检查不能把 `nvidia-smi` 错误输出解析为 PID。严格模式下如果
   `nvidia-smi -L` 不可用，应直接失败并提示检查 driver，而不是继续跑正式
   round。

## 5. 共享输入规则

同一 round 内所有系统必须使用同一份：

- shared trace JSON
- shared adapter subset JSON
- model profile
- dataset profile
- workload profile
- prompt/token guard
- cost model

禁止：

- 每个系统重新采样 adapter subset。
- 每个系统各自生成 trace。
- 为某个系统单独放宽 prompt/token budget。
- 将失败系统静默 fallback 到另一个 backend。

所有系统在 replay 结束后、summary 生成前必须通过统一 gate：

```text
scripts/validate_replay_results.py
```

该 gate 检查 `ok == total_requests`，并拒绝
`prompt_token_source=trace_expected` 或
`completion_token_source=trace_expected` 的正式结果。这样可以避免某个系统
产生半成功 replay、空成功请求或 token fallback 后仍写出看似完整的
summary。`run_full_fair_round.sh` 还会在每个系统阶段后做 summary schema
audit，这是第二道保险。

## 6. 主指标

主表使用以下字段：

```text
System
TTFT_e2e_avg_ms
TTFT_e2e_p95_ms
E2E_e2e_avg_ms
E2E_e2e_p95_ms
TPOT_ms
Throughput_tok_s
Cost_per_request_usd
CE
```

定义：

```text
TTFT_e2e = scheduled trace arrival -> first output token/chunk observed
E2E_e2e  = scheduled trace arrival -> full response completion observed
TPOT     = per-request service decode time per generated token
Cost/req = total monetary cost / completed requests
CE       = 1 / (avg_E2E_e2e_seconds * Cost/req)
```

`Cost/1M tokens`、GPU-second、active/idle GPU ratio 是审计指标，不替代
`Cost/req` 主成本口径。

## 7. 成本模型

当前 monetary cost 使用 cloud-style active/idle differential billing：

```text
serverful_cost = P_gpu * lifecycle_gpu_seconds

serverless_cost =
  P_gpu * (startup_gpu_seconds + active_gpu_seconds
           + idle_factor * idle_ready_gpu_seconds)
  + invocation_cost * completed_requests
```

当前默认：

```text
serverless_idle_gpu_cost_factor = 0.2380952381
```

该值来自 Alibaba Function Compute Tesla GPU idle/active CU factor：
`0.5 / 2.1`。如果更换云厂商或价格假设，必须显式改配置并同步文档。

## 8. 系统特定规则

### SGLang

- 主公平拓扑：`DP=4, TP=1`，四个单卡 serverful replicas。
- `TP=4` 只能作为 serverful model-parallel upper-bound 附表，不进入主公平表。
- 使用原生 `/generate` 和 `input_ids`，避免 OpenAI text/chat 入口的 prompt 解释偏差。

### vLLM

- 使用 standalone OpenAI-compatible server。
- Llama-2 base tokenizer 无 chat template，因此正式路径使用 `/v1/completions`。
- 成功请求不得回退到 raw trace expected tokens；否则脚本必须失败。
- vLLM OpenAI streaming replay 必须显式请求 `stream_options.include_usage=true`，
  并设置 `min_tokens=1`。正式 trace 中所有请求都有正的 expected output
  token budget；若不设置 `min_tokens`，vLLM 可能合法地首步 EOS，返回
  `200 OK` 但没有 generated text、usage 或可定义的 first-token event。
  这类结果不是有效 TTFT 样本，必须通过 replay contract 规避或由 audit 拒绝。
- 若仍出现 `HTTP 200` 空成功，wrapper 允许少量 retry，但 retry 时间仍归入
  同一请求延迟窗口；重试后仍为空则本阶段失败，不能生成 summary。

### ServerlessLLM

- 优先使用 vLLM backend。
- `enforce_eager: auto` 必须正确解析：Llama-2 单卡 scale-out LoRA 路径允许
  CUDA graph，TP 或已知风险模型保持 conservative eager。
- deploy config 必须写出 runtime env，例如 `VLLM_USE_V1`、`VLLM_ATTENTION_BACKEND`
  和 `VLLM_USE_FLASHINFER_SAMPLER`。
- 不能静默 fallback 到 transformers；如果 vLLM backend 不可用，必须记录根因。

### S-LoRA

- 使用官方 CUDA 11.8 / PyTorch 2.0.1 兼容环境。
- 正式 replay 走 native `/generate_stream`。
- prompt guard 必须按 S-LoRA 服务端 `tokenizer.encode(prompt)` 语义计入 special tokens。
- 当前正式结果中，S-LoRA 已通过 replay gate，但 served output tail
  明显长于其他系统。使用 S-LoRA 的 `TPOT`、`Tok/s` 或 token-normalized
  cost 写论文强结论前，必须先检查 EOS / ignore-eos / max-new-token
  请求语义是否与其他系统对齐；若选择保持 official wrapper 语义，也必须在
  结果分析中标注为 paper-faithful 复现边界。

### FaaSLoRA

- 由 `run_full_fair_round.sh` 调用 FaaSLoRA shared-artifact wrapper。
- 机制指标只用于 FaaSLoRA 内部图和消融，不进入跨系统主表。
- `HOST` adapter tier 必须使用 tmpfs/ramfs 等内存背书文件系统。当前正式
  默认路径是 `/dev/shm/faaslora_host_cache/<scenario>`；若结果 JSON 中
  `host_cache_memory_backed` 不是 `true`，该轮不能进入论文结果。
- FaaSLoRA 启动日志应出现类似
  `[HOST tier] path=... fs=tmpfs available=... required=...` 的 preflight 行。
- FaaSLoRA scale-out 使用 predictive target refinement：autoscaler 决定是否扩容，
  handoff predictor 根据 ready-time queue 和 runtime capacity 决定一次补足几个
  runtime。该机制必须保持 `scale_up_predictive_target_enabled=true`，避免正式
  burst 前沿中退回 `current+1` 逐步扩容而拖高早期 E2E/CE。
- `scale_up_startup_parallelism=auto`：低压时为前台 adapter load 保留余量；
  高压 scale-out 时可以用满 `max_concurrent_loads` 并行启动 runtime，但仍受
  `max_instances` 和 GPU 清洁检查约束。
- 2026-04-25 已完成同 trace 500-request 回归闭口
  `llama2_7b_r500_a500_seed42_s8_predictive1_faaslora`：
  `500/500` 成功，`TTFT_e2e=1395/10052/16366ms`，
  `TTFT_service=412/573/674ms`，`TPOT=28.1ms`，
  `E2E_e2e=4037ms`，`Cost/req=$0.003084`，主 `CE=80.324`。
  该结果只作为 FaaSLoRA 修复非回归证明，不替代 4000-request 正式结论。

## 9. 结果保存

正式 round 结果保存在 baseline 工作区：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<timestamp>_<run_tag>/
```

其中：

- `shared_artifacts/`: 本轮 trace 与 adapter subset。
- `raw/`: 各系统 replay、summary、launch specs、logs。
- `logs/`: 阶段级日志。
- `state/`: 断点续跑 markers。
- `comparison/`: 最终横向对比产物。

FaaSLoRA 自身原始结果可同时在主项目结果目录保留，但论文横向取数以这个
timestamped round 目录为准。

## 10. 当前推荐动作

2026-04-26 当前 Llama-2 7B 4000-request 五系统正式 round 已闭合：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
```

主结论：

- FaaSLoRA 主 CE 高于 SGLang、vLLM、S-LoRA 和 ServerlessLLM；
- SGLang 延迟最低，符合强 serverful runtime 的预期；
- FaaSLoRA 以更低 lifecycle monetary cost 赢得 CE；
- ServerlessLLM 的主要瓶颈是 dispatch/admission wait，而不是单请求
  service path；
- S-LoRA 输出 token 尾部分布偏长，进入论文强结论前需要单独审计或说明
  official wrapper 语义边界。

因此当前不建议继续围绕 500-request debug round 或单个 headline 数字做局部
调参。下一步应优先按 `PAPER_EXPERIMENT_TODO.md` 推进论文实验序列：
引言图、motivation、ablation、workload/adapters scale、资源/成本图。若要
把 S-LoRA 写进主表强对比，则先做一次 targeted EOS/输出语义审计。

## 11. 长期数据队列

2026-04-26 已新增长期队列脚本：

```text
/home/qhq/serverless_llm_baselines/scripts/run_paper_long_experiment_queue.sh
```

当前默认 profile 为 `load_p0`，连续运行完整五系统 high-pressure sensitivity：

```text
06_sensitivity_load / Llama-2 7B / s6 / sglang serverlessllm vllm slora faaslora
06_sensitivity_load / Llama-2 7B / s4 / sglang serverlessllm vllm slora faaslora
```

2026-04-27 审计后，`s6/s4` 被降级为 stress diagnostic：它们会把
PrimeLoRA 推入持续高负载边界，dispatch/admission wait 放大后 CE 被 SGLang
反超，不适合作为主文“负载稳健性优势”图。为验证 serverless 合理运行区间，
新增 `load_operating_p0`：

```text
06_sensitivity_load_operating / Llama-2 7B / s12 / sglang serverlessllm vllm slora faaslora
06_sensitivity_load_operating / Llama-2 7B / s10 / sglang serverlessllm vllm slora faaslora
```

这两个点与已有 s8 主 round 组成低/中/名义负载三点：s12 约 `0.67 rps`、
s10 约 `0.81 rps`、s8 约 `1.01 rps`。选择依据是 s8 下 PrimeLoRA
`ActiveGPU%≈0.65` 且 `DispatchWait≈111 ms`，仍是可解释的 serverless
名义负载；s10/s12 则把 active 占比外推到约 `0.52/0.43`，分别对应中、
低负载。s6/s4 由于 dispatch wait 明显放大，只作为 stress diagnostic。

该队列只改变 `SLLM_TIME_SCALE_FACTOR`，保持已闭合主 round 的模型、请求数、
adapter pool、seed、Zipf、hot set 和 rotation 语义。`run_full_fair_round.sh`
已同步透传 `SLLM_TIME_SCALE_FACTOR` 到 shared trace prepare 阶段，避免
run tag 与真实 trace scale 不一致。

如果为了快速探路显式覆盖 `PAPER_QUEUE_SYSTEMS="sglang vllm slora faaslora"`，
该结果只能标注为 partial sensitivity，不能作为完备横向对比。后续必须补跑
ServerlessLLM 并重新生成 compare。2026-04-27 已修复队列断点逻辑：即使
queue-level `.done` marker 已存在，只要 compare JSON 缺少当前
`PAPER_QUEUE_SYSTEMS` 中的系统，队列也会重新进入该 round；底层
`run_full_fair_round.sh` 会跳过已完成系统，只补缺失系统并重写 compare。

注意：不要在某个 bash 脚本仍在 tmux 中执行时热修改该脚本文件。bash 会按需
继续读取脚本内容，热修改可能导致当前进程在后续行遇到不一致内容并报
`unexpected token`。若必须修脚本，先让当前阶段失败/停止并清洁 GPU，再用
同一 queue id 断点续跑。

启动命令：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_load_p0

PAPER_QUEUE_PROFILE=load_p0 \
scripts/run_paper_long_experiment_queue.sh
```

如果目标是生成可进入主文的低/中/名义负载 sensitivity，使用：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_load_operating_p0

PAPER_QUEUE_PROFILE=load_operating_p0 \
scripts/run_paper_long_experiment_queue.sh
```

队列会写出：

```text
results/paper_experiments/00_queues/<queue_id>/queue.env
results/paper_experiments/06_sensitivity_load/<queue_id>_<run_tag>/
```

失败后建议显式指定完整系统列表继续，避免旧 partial `queue.env` 把
ServerlessLLM 再次排除：

```bash
cd /home/qhq/serverless_llm_baselines
PAPER_QUEUE_ID=<queue_id> \
PAPER_QUEUE_PROFILE=<load_p0_or_load_operating_p0> \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_long_experiment_queue.sh
```

已完成且 compare 完整的 round 不会重跑；已完成但缺系统的 round 会自动补齐。
