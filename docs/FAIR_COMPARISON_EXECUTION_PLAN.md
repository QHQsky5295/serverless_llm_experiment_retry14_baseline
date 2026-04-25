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

当前不建议继续围绕 500-request debug round 做局部调参。500 请求闭口已经证明
FaaSLoRA 的 HOST tmpfs 真实性 gate、scale-out predictive target 和 startup
parallelism 没有破坏服务路径，并带来小幅 CE 改善。下一步应回到正式 round：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh --dry-run
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

如果需要从零启动新 round：

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

正式判断以 timestamped round 目录中的 `comparison/` 和各系统 `summary.json`
为准，而不是单个系统的中途 live 输出。
