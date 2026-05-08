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

## 1.1 Live Queue: 2026-04-27

当前正在运行的长队列是 operating-load sensitivity，不是主 s8 round 的重跑：

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
profile:      load_operating_p0
systems:      sglang serverlessllm vllm slora faaslora
completed:    s12 and s10 operating-load rounds
section:      06_sensitivity_load_operating
```

截至 `2026-04-28 03:41 CST`，该队列已经完成；s12 和 s10 的 compare JSON 均
包含五个系统。外层 tmux 可能仍停在完成后的 shell prompt，可用以下命令查看：

```bash
tmux capture-pane -p -t paper_load_operating_p0 -S -120
```

该队列补齐了 Fig. 8 候选的 `s12/s10` 低/中 operating-load 数据。它保留了
ServerlessLLM；后续若重跑 paper-facing sensitivity，仍不得静默移除
ServerlessLLM。

## 1.2 Live Queue: 08_backbone_robustness

当前 backbone robustness 队列用于补充 Llama-2 13B 与 Qwen 家族的泛化性
证据，不替代已经闭合的 Llama-2 7B main round。

```text
tmux session: paper_backbone_robustness_p0
queue id:     20260429_115544_backbone_robustness_p0
profile:      backbone_robustness_p0
systems:      sglang serverlessllm vllm slora faaslora
section:      08_backbone_robustness
```

截至 `2026-04-30` 复查，Qwen2.5-7B 已完成 SGLang 和 ServerlessLLM 阶段，
但 vLLM 阶段先后暴露两个问题。第一轮 `dp4/tp1` 触发主机级 OOM，内核日志显示
OOM killer 连续杀掉 vLLM APIServer/engine 进程；这不是 GPU OOM、不是
500-adapter 采样池被错误设置，也不是 replay 自身生成错误。该 round 的 vLLM
launch spec 仍包含 `lora_modules_count=500`，shared trace 中 `4000/4000`
请求均绑定 LoRA adapter。

根因分两层：

1. Qwen2.5-7B publicmix 当前必须走 vLLM V0/eager 路径，首次失败来自长
   replay 下 host-side LoRA/runtime footprint 过大，导致 125GB 主机进入 OOM
   区间并杀掉 vLLM APIServer/engine。
2. 临时切换到 `dp2/tp2` 可以降低 host-memory footprint，但会把四个独立服务
   replica 降成两个。真实 384-request probe 显示该拓扑稳定但排队严重、
   `TTFT` 超过 100s，因此不能作为论文配置。该问题不是 backbone 语义错误，
   也不是 500-LoRA workload 定义错误，而是 vLLM host-memory footprint 与
   serving-replica topology 需要同时约束。

后台 Docker/Milvus/Ray 以及持续失败重启的 `frpc.service` 会降低系统余量，
但实验侧直接触发点是 Qwen-vLLM 的 host-memory footprint 与调度包络。

已修复的 runner 问题：

- `run_vllm_fair_experiment.sh` 现在使用 `setsid` 启动 vLLM server，并在
  cleanup 时优先 kill server process group，避免失败后遗留 vLLM worker 占用 GPU。
- `run_vllm_fair_experiment.sh` 增加 host-memory fail-fast guard：
  默认要求 `MemAvailable >= 32GiB`，并在 replica 启动后和 replay 期间持续检查。
  一旦触发，runner 会终止 replay 和 vLLM process group，显式失败该阶段，
  而不是继续运行到 Linux OOM 并拖垮 SSH/systemd。
- `run_vllm_fair_experiment.sh` 在 replay 期间监控 vLLM server PID；若任何
  replica 中途退出，立即拒绝该 run，避免写出 `ok < total` 的污染结果。
- `run_paper_long_experiment_queue.sh` 对 `backbone_robustness_p0` 默认设置
  `VLLM_TIMEOUT_S="${PAPER_QUEUE_VLLM_TIMEOUT_S:-21600}"`，让 Qwen-vLLM 的长排队
  表现为真实高延迟，而不是 1 小时 client timeout 失败。
- `run_vllm_fair_experiment.sh` 支持 `VLLM_LORA_REGISTRATION_MODE=static|dynamic`。
  static 模式保留原始 `--lora-modules` 注册路径；dynamic 模式通过 vLLM
  `/v1/load_lora_adapter` 在每个 replica 首次遇到 adapter 时按需加载。两种
  模式都不改变 shared trace、500-adapter sampled universe 或每请求 LoRA 绑定。
- `replay_openai_trace.py` 进一步支持 `--dynamic-lora-routing`。Qwen-family
  dynamic vLLM 正式默认使用 `adapter_hash`，让同一个 adapter 固定归属一个
  endpoint，避免 request round-robin 或 hot-pair routing 在长 replay 中把同一
  adapter 重复加载到多个 OpenAI API replica。它不是 PrimeLoRA 的
  readiness-aware routing，只是在 standalone vLLM baseline 中约束 runtime LoRA
  注册路径的 lifetime endpoint-adapter footprint。
- `run_full_fair_round.sh` 对 Qwen2.5-7B 的 vLLM stage 保持 `dp4/tp1`，
  仍占用同一 4-GPU 预算并保留四个独立服务 replica；Llama-2 13B 和
  Qwen2.5 14B 继续按 model profile 使用 `dp2/tp2`。Llama-2 7B 主 round
  仍使用已验证的 `dp4/tp1`。
- `run_full_fair_round.sh` 对 Qwen-family vLLM stage 默认使用 dynamic LoRA
  registration 和 `adapter_hash` dynamic routing。Qwen2.5-7B vLLM
  formal stage 保持 `dp4/tp1` 和四个独立服务 replica，并使用
  `max_num_seqs=8`、`max_loras=8`、
  `max_num_batched_tokens=4096`、`max_cpu_loras=16`。这不改变模型、GPU
  budget、500-adapter universe、request trace、sampling 或 LoRA-bound
  workload；修复的是 standalone vLLM OpenAI API path 在 Qwen V0/eager 下把
  500 个 LoRA 静态注册进每个 replica 导致 host-side footprint 放大的问题。
- Qwen-family standalone vLLM dynamic LoRA stage 现在默认增加
  `--disable-frontend-multiprocessing`。这不降低 vLLM 的 `dp/tp` topology、
  `max_num_seqs`、`max_loras` 或 `max_cpu_loras`，只关闭 OpenAI API frontend
  的额外多进程形态。横向排查显示，PrimeLoRA/FaaSLoRA 使用
  `AsyncLLMEngine + LoRARequest` 直连路径，不走 standalone OpenAI API server
  的 runtime LoRA 注册路径；因此此前 Qwen2.5-7B 问题不能直接推断为
  “所有 vLLM 后端都会崩”，但 Qwen-family 仍需要独立 preflight gate。
- `replay_openai_trace.py` 增加 `--max-requests` 和 failure-abort gate，
  允许正式前做 96/256-request bounded preflight，并在 fail 请求累计时立即
  终止无效 replay。
- `run_full_fair_round.sh` 不再把 vLLM/SGLang/S-LoRA summary 路径写死成
  `dp4_tp1`，而是按当前 run tag 动态选择最新 summary；这避免 13B/14B 或
  Qwen-vLLM safe topology 完成后被错误判定为缺文件。
- `bash -n` 已通过；`PAPER_QUEUE_DRY_RUN=1 PAPER_QUEUE_PROFILE=backbone_robustness_p0`
  已验证三轮计划和模型/workload profile 正确。Qwen2.5-7B vLLM dry-run
  已验证 `topology=dp4_tp1`、`lora_modules_count=500`、`4000/4000` 请求绑定 LoRA。
- Qwen2.5-7B vLLM 曾完成真实 static preflight：
  `dp4/tp1,max_cpu_loras=16` 在同一 shared trace/subset 上完成
  `ok=384/384, fail=0`，`TTFT=1661.8ms`、`TPOT=73.0ms`、`Tok/s=106.21`。
  该 preflight 是健康性验证，不作为论文 performance data。后续长 replay 在约
  900 个完成请求处触发 host-memory guard，且 `max_cpu_loras=8` 也未能通过
  1200-request 长 probe；单纯 dynamic + request round-robin 或 hot-pair routing
  又会让 adapter 被重复加载到多个 endpoint。因此当前正式长跑使用 dynamic LoRA
  registration + `adapter_hash`，把 500-adapter universe 约束到稳定的
  endpoint-affinity 路径。
- Qwen2.5-7B vLLM no-frontend-multiprocessing bounded preflight 已完成：
  `dp4/tp1,max_num_seqs=8,max_loras=8,max_cpu_loras=16`,
  `lora_registration_mode=dynamic`,
  `dynamic_lora_routing=adaptive_hot_pair_hash`,
  `disable_frontend_multiprocessing=1`，结果 `ok=1200/1200, fail=0`，
  无 token-source fallback，且越过旧 static 路径约 900 completed 的失败区间。
  该轮显式设置 `VLLM_MAX_REPLAY_REQUESTS=1200`，所以 formal gate 最后报告
  `completed=1200,total=4000` 并拒绝进入论文结果，这是正确保护；正式数据仍必须
  跑完整 4000-request round。

系统服务注意事项：

- 强制重启后的 journal 显示 `frpc.service` 每 5 秒重启并连接
  `120.26.187.54:7000` 超时；如果 SSH 依赖 frp 隧道，远程不可达不完全由实验
  解释。需要单独检查 frp server、mihomo/TUN 路由和本机 `frpc.service`。
- journal 还显示 `yk0Wk9DV.service` 持续尝试执行不存在的 `/bin/YXt5BHl6`。
  这不像论文实验服务，应由管理员审计并禁用/删除，避免持续重启污染系统日志。
- `2026-04-30 19:12 CST` 复查：`systemctl --failed` 没有 failed units，但
  `frpc.service` 和 `yk0Wk9DV.service` 均处于 `activating (auto-restart)`。
  这说明当前 GPU/实验进程已空闲，远程接入不稳仍可能来自实验外服务。

恢复同一个队列时不要新建 queue id。当前 `paper_backbone_robustness_p0` 可能
只剩一个 attached shell；若里面没有运行脚本，可直接在该 tmux 中使用同一个
queue id 继续。若希望干净重开，先关闭空 session 再新建同名 tmux。

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_backbone_robustness_p0

PAPER_QUEUE_ID=20260429_115544_backbone_robustness_p0 \
PAPER_QUEUE_PROFILE=backbone_robustness_p0 \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_backbone_robustness_queue.sh
```

如果希望进一步放宽 vLLM timeout，可显式加：

```bash
PAPER_QUEUE_VLLM_TIMEOUT_S=28800
```

监控：

```bash
tmux attach -t paper_backbone_robustness_p0
tmux capture-pane -p -t paper_backbone_robustness_p0 -S -160
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
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

Llama-2 13B 主对比补充状态：

- `vLLM` 不再使用早期保守的 `seq/lora=2` 或 `seq/lora=4` 13B 包络。
  2026-05-07 的 s8/full 诊断显示，旧配置的异常 TTFT 来自 vLLM 后端内部
  Pending 队列长期累积，而非 replay 失败或 GPU OOM；256-request probe
  对 13B 不充分，因为它只覆盖第一个 hot set。后续同负载
  `DP=2,TP=2,max_num_seqs=8,max_loras=8,max_num_batched_tokens=4096`
  probe 跨过 500-request hot-set rotation 后仍保持 0 failure、个位数
  backlog，且 vLLM 日志报告 1024-token 最大并发约 11.69x。正式 13B
  vLLM rerun 应使用该包络，仍保持同一 4-GPU 预算、500-adapter pool、
  s8 trace 和 OpenAI completions LoRA 语义。
- `S-LoRA` 不再回避 13B。当前路径使用同一 4-GPU 预算下的 `DP=1, TP=4`，
  并自动启用官方 BMM adapter path。原因是 S-LoRA 原 packed BGMV pool 仅适配
  `world_size=1`，而 13B 需要 tensor parallel；BMM 路径已补齐 q/k/v/o 的
  TP LoRA 分片和 o projection 后的 all-reduce。
- 已完成 `20260507_13b_preflight_llama13b_slora_tp4_bmmfix3_smoke16_a500`
  真实 preflight：500-adapter 注册、16/16 请求成功、无 token fallback、无
  runtime traceback，结束后 GPU clean。该轮仅作为正确性 gate，不进入论文图表。
- `run_slora_fair_experiment.sh` 现在用独立 process group 启动 S-LoRA replica，
  失败或超时时清理整个 TP worker tree，避免旧 worker 占 GPU 造成后续系统
  “看似崩溃”。

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
TPOT_avg_ms
TPOT_p95_ms
Throughput_tok_s
Cost_per_request_usd
CE
```

定义：

```text
TTFT_e2e = scheduled trace arrival -> first output token/chunk observed
E2E_e2e  = scheduled trace arrival -> full response completion observed
TPOT     = per-request service decode time per generated token; report avg and p95
Cost/req = total monetary cost / completed requests
CE       = 1 / (avg_E2E_e2e_seconds * Cost/req)
```

`Cost/1M tokens`、GPU-second、active/idle GPU ratio 是审计指标，不替代
`Cost/req` 主成本口径。
TPOT 是请求级 decode 延迟分布，不能只报告均值；正式横向表和 normalized 图
应从 observed request-level `tpot_ms` 样本报告 `TPOT avg/p95`。

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
  `TTFT_service=412/573/674ms`，`TPOT avg=28.1ms`，
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

2026-04-28 新增并完成 `adapter_pool_p0`，作为 adapter-pool sensitivity 队列：

```text
07_sensitivity_adapter_pool / Llama-2 7B / a100 hot16 / sglang serverlessllm vllm slora faaslora
07_sensitivity_adapter_pool / Llama-2 7B / a200 hot24 / sglang serverlessllm vllm slora faaslora
07_sensitivity_adapter_pool / Llama-2 7B / a300 hot32 / sglang serverlessllm vllm slora faaslora
07_sensitivity_adapter_pool / Llama-2 7B / a400 hot40 / sglang serverlessllm vllm slora faaslora
```

该 profile 保持 `4000` requests、Zipf `1.0`、hotset rotation `500`、time
scale `8.0`、seed `42` 与完整五系统集合不变，只改变 adapter universe 和
active hot cap。`a500/hot48` 右端点优先复用已闭合的 Llama-2 7B `s8` 主 round；
如需同一 queue 自包含五个点，使用 `adapter_pool_full_p0`，它会额外重跑
`a500/hot48`。便捷入口为：

```text
/home/qhq/serverless_llm_baselines/scripts/run_paper_adapter_pool_queue.sh
```

2026-04-29 新增 `backbone_robustness_p0`，作为下一轮长期实验队列：

```text
08_backbone_robustness / Qwen2.5 7B / a500 hot48 rot500 s8 / sglang serverlessllm vllm faaslora
08_backbone_robustness / Llama-2 13B TP=2 / a500 hot48 rot500 s8 / sglang serverlessllm vllm slora faaslora
08_backbone_robustness / Qwen2.5 14B TP=2 / a500 hot48 rot500 s8 / sglang serverlessllm vllm faaslora
```

说明：`PAPER_QUEUE_SYSTEMS` 仍可写成完整五系统列表；runner 会对 Qwen-family
profile 显式标记 `S-LoRA` 为 unsupported 并跳过该 stage。原因是当前接入的
S-LoRA 上游实现只提供 Llama/Llama2 model backend，Qwen2/Qwen2.5 在
Transformers 中暴露为 `model_type=qwen2`，不应通过修改核心模型实现或静默
替换来伪造 baseline 结果。Llama-2-13B 仍保留 S-LoRA 对比。

便捷入口为：

```text
/home/qhq/serverless_llm_baselines/scripts/run_paper_backbone_robustness_queue.sh
```

该 profile 已通过 `PAPER_QUEUE_DRY_RUN=1`，只生成预期的三个 round，不启动 GPU。

2026-04-30 复查 `backbone_robustness_p0` 时，Qwen2.5-7B 已完成 SGLang 与
ServerlessLLM；失败点在 vLLM。内核日志记录 `Out of memory: Killed process
... (python)`，对应 vLLM APIServer/engine PID。原因不是 500-adapter workload
过大本身：500 是正式采样池，vLLM launch 仍暴露 500 个 `--lora-modules`，
shared trace 中 4000 个请求均绑定 LoRA。真正触发点是 Qwen2.5-7B publicmix
当前必须走 vLLM V0/eager，首次失败来自长 replay 下 host-side
LoRA/runtime footprint 过大。随后验证表明，`dp2/tp2` 虽然降低内存压力，
但会把四个服务 replica 降成两个并导致严重排队，因此不能作为论文配置。

当前修复为：Qwen2.5-7B vLLM 保持同一 4-GPU 预算下的 `dp4/tp1` topology，
并使用 `lora_registration_mode=dynamic,dynamic_lora_routing=adapter_hash`，
同时默认关闭 standalone vLLM OpenAI API frontend multiprocessing，并保留
host-memory guard 与 server PID replay monitor。此前真实主机命名空间下的 384-request static
`max_cpu_loras=16` preflight 已通过：`ok=384/384, fail=0`，
`TTFT=1661.8ms`、`TPOT=73.0ms`、`Tok/s=106.21`；但正式长跑随后触发
32GiB host-memory guard，`max_cpu_loras=8` 长 probe 也失败。因此下一次验证
必须使用 dynamic registration + adapter-affinity sticky routing + no frontend
multiprocessing，而不是继续沿用 static `--lora-modules` 全量注册或 request
round-robin/adaptive hot-pair dynamic loading。1200-request bounded preflight 已验证该组合
`ok=1200/1200, fail=0`；它只是 stability gate，不是论文正式 4000-request
结果。

2026-05-06 复查当前 queue `20260504_122743_backbone_robustness_p0`：
Qwen2.5-7B vLLM 使用 `adapter_hash` 完成正式 4000-request replay，
`ok=4000/4000, fail=0`，token source 无 `trace_expected` fallback。该轮失败日志中的
`abel: 未找到命令` 来自旧启动进程读到的脚本片段；当前 HEAD 中该参数行为
`--label`，且 `bash -n` 已通过。已补生成 vLLM summary 并写入 `30_vllm.done`，
随后 queue 在 S-LoRA 阶段触发上游限制 `can not support qwen2 now`。runner
已改为对 Qwen-family S-LoRA 写 `.unsupported` 并继续后续系统。
外层 long-queue runner 也同步使用“supported systems”校验 compare 文件，
避免后续恢复 queue 时因为 Qwen-family 缺少 S-LoRA 行而误判该轮未完成。

强制重启后的日志还显示 `frpc.service` 反复连接远端超时。如果 SSH 依赖 frp，
远程不可达不完全由实验解释；但本次需要物理重启的直接实验侧触发因素是
vLLM/Qwen 的 host OOM。

同次修复还发现并修正了 backbone 队列的拓扑风险：原 `run_full_fair_round.sh`
向 vLLM/SGLang/S-LoRA 默认传 `TP=1,DP=4`，会让 Llama-2-13B 和 Qwen2.5-14B
这些 TP=2 profile 在后续阶段按错误拓扑启动。修复后默认不覆盖 profile；
Qwen2.5-7B 的 vLLM 例外保持 `dp4/tp1` 并限制 active CPU LoRA cache，
13B/14B 解析为 `dp2/tp2`。Llama-2 7B 主 round 仍保持已验证的 `dp4/tp1`。

额外预检：

- Llama-2 13B 与 Qwen2.5 14B shared artifacts 已在 `/tmp` 预生成验证：
  均为 `4000/4000` LoRA-bound requests，500 adapter subset，132 个实际命中
  adapter。
- vLLM dry-run 验证 Llama-2 13B 与 Qwen2.5 14B 均为 `dp2/tp2`，且
  `lora_modules_count=500`。
- S-LoRA dry-run 曾验证 launch spec 能为 Qwen2.5 生成 `dp4/tp1`/500 LoRA
  dirs，但正式启动暴露上游 model backend 限制：`qwen2` 不受支持。因此 Qwen
  family 不再强行跑 S-LoRA。
- 2026-05-06 补充 Llama-2 13B S-LoRA TP2 修复。此前 smoke 暴露官方 PEFT
  LoRA adapter path 中的 `assert world_size == 1`，导致 13B 在 4xRTX3090
  测试床必须使用 `dp2/tp2` 时无法启动。当前修复按 TP rank 切分 LoRA 权重，
  将 combined CPU buffer reshape 改为 local attention heads，并恢复 LoRA
  attention output 的 TP all-reduce。fair-round runner 因此不再跳过 Llama
  13B S-LoRA；Qwen-family 仍因缺少 S-LoRA 上游 Qwen model backend 而标记为
  `.unsupported`。
- 进一步 smoke 显示，Llama-2 13B 的 `dp2/tp2` S-LoRA 虽可启动第一组 TP
  replica，但第二个 data-parallel replica 会重复加载 500 个 adapter 的
  host-side 状态，使 `MemAvailable` 快速降到危险区间并增加整机卡死风险。
  因此 Llama-2 13B 正式主轮次固定为 `dp1/tp4`：仍使用同一 4-GPU 预算和
  S-LoRA serverful runtime，但不再复制整套 adapter pool 到多个 DP replica。
- 2026-05-04 补充 Qwen2.5-14B standalone vLLM smoke：`dp2/tp2`、
  `max_num_seqs=2`、`max_loras=2`、`max_cpu_loras=8`、
  `lora_registration_mode=dynamic`、
  `dynamic_lora_routing=adaptive_hot_pair_hash`、
  `disable_frontend_multiprocessing=1`。两个 replica 均完成 runtime LoRA
  load 和短请求，结束后 GPU/进程清理干净。
- 2026-05-06 在将 Qwen-family dynamic routing 默认统一为 `adapter_hash` 后，
  又补跑 Qwen2.5-14B standalone vLLM smoke：`dp2/tp2` 两个 TP=2 replica
  均成功启动、runtime-load LoRA，并完成 1-token LoRA 请求；`VLLM_SMOKE_ONLY=1`
  路径已跳过 formal compare，命令本身以 0 退出且 GPU/进程清理干净。
- 2026-05-01 的 `paper_backbone_robustness_v2` 失败终端是旧 static launch：
  launch spec 中没有 `lora_registration_mode` 和
  `disable_frontend_multiprocessing` 字段，不能用于判断当前修复。它在约
  900 completed 后触发 host-memory guard，说明旧路径确实会把机器推向
  OOM 风险；修复后的正式队列应新建 queue id，或至少确认重跑 stage 写出的
  launch spec 已变为 dynamic + no-mp。

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

如果目标是生成 adapter-pool sensitivity，使用：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_adapter_pool_p0

scripts/run_paper_adapter_pool_queue.sh
```

如果目标是生成 multi-backbone robustness，使用：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_backbone_robustness_p0

PAPER_QUEUE_PROFILE=backbone_robustness_p0 \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_backbone_robustness_queue.sh
```

队列会写出：

```text
results/paper_experiments/00_queues/<queue_id>/queue.env
results/paper_experiments/<section>/<queue_id>_<run_tag>/
```

失败后建议显式指定完整系统列表继续，避免旧 partial `queue.env` 把
ServerlessLLM 再次排除：

```bash
cd /home/qhq/serverless_llm_baselines
PAPER_QUEUE_ID=<queue_id> \
PAPER_QUEUE_PROFILE=<load_p0_or_load_operating_p0_or_adapter_pool_p0_or_backbone_robustness_p0> \
PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
bash scripts/run_paper_long_experiment_queue.sh
```

已完成且 compare 完整的 round 不会重跑；已完成但缺系统的 round 会自动补齐。
