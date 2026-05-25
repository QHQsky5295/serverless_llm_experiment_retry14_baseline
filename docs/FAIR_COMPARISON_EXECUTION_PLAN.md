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

## 2026-05-11 local-sim remote fairness rerun

目标：修正本地模拟 remote 工件链路的 baseline 公平性。此前 PrimeLoRA
路径已经承担 remote artifact 准备成本，但部分 baseline 仍更接近“本地已可见
工件目录”口径；这会让 PrimeLoRA 在 cold/readiness 阶段凭空多承担一段
remote 传输延迟。新的 remote-fair round 要求所有 baseline 在 GPU/HOST/NVMe
不可用时也必须经过 remote 源准备 adapter。

注意：remote-fair 不是“每个请求都从远程重新下载 adapter”。同类系统和工业
实践通常把远程对象仓库或模型仓库作为 backing store，同时在 worker 节点保留
本地缓存或启动期 staging。例如 ServerlessLLM 论文强调利用 GPU 服务器附近的
本地存储和内存来减少 remote checkpoint downloads，并按 locality 做调度；
NVIDIA Dynamo LoRA 文档支持从 `file://`、`s3://`、`hf://` 动态加载 adapter，
同时将已下载 adapters 本地缓存以避免重复下载；S-LoRA 官方语义是“all adapters
in main memory, active adapters to GPU”，而不是每次请求都访问远端。因而公平
口径应是：所有系统都从同一 remote backing store 出发，first-touch/cache-miss
或启动期 registry 准备要计入 lifecycle/TTFT 路径；一旦该系统语义允许本地缓存，
重复命中就不应强行重新走远程。PrimeLoRA 的论文贡献也不应表述为“唯一拥有
缓存”，而应表述为将 remote/NVMe/HOST/GPU residency 与 selected-replica routing、
scale-out handoff 和 GPU admission 协调起来，从而减少 service-readiness gap。
参考入口：vLLM LoRA 文档（startup 与 runtime LoRA loading、S3 resolver）、
SGLang LoRA 文档（`max_loaded_loras`、LRU eviction）、ServerlessLLM Store
文档（DRAM/SSD/HDD multi-tier loading）和 ServerlessLLM OSDI/Arxiv 论文
（near-GPU storage 减少 remote checkpoint downloads）。
对应公开链接：
`https://docs.vllm.ai/en/stable/features/lora/`、
`https://sgl-project.github.io/advanced_features/lora.html`、
`https://docs.nvidia.com/dynamo/user-guides/lo-ra-adapters`、
`https://github.com/ServerlessLLM/ServerlessLLM` 与
`https://arxiv.org/abs/2401.14351`。
`https://docs.nvidia.com/dynamo/latest/user-guides/lo-ra-adapters`、
`https://github.com/ServerlessLLM/ServerlessLLM`。

根因修复：旧的 `file://` local-sim 逻辑虽然按带宽 sleep，但 sleep 之后仍做
真实本地目录复制。这样会引入本地磁盘与 OS page cache 顺序效应：同一个
500-adapter subset，先跑的系统可能承担真实 copy 时间，后跑的系统可能受益于
热缓存。当前修复将 local-sim materialization 改为“按源大小 sleep +
hardlink/symlink materialization”，只保留可控的带宽延迟，不再把本地复制速度
混入远程传输模型。真实 HTTP remote 路径不变。

新增/修改入口：

- `scripts/materialize_remote_adapter_subset.py`：用于需要启动前完整 adapter
  registry 的 baseline，例如 S-LoRA。local-sim 下按源大小 sleep 后用
  hardlink/symlink 物化目录，并把这段准备计入 startup/lifecycle。S-LoRA
  官方 router/model 初始化会遍历所有 `--lora-dirs` 读取 LoRA config/weights，
  因此不能在不改核心机制的情况下改成请求期 remote miss。
- `scripts/replay_openai_trace.py`：用于支持动态 LoRA load 的 runtime，例如
  SGLang 和 vLLM。首次触达 adapter 时按 remote endpoint 拉取到 round-local
  cache；local-sim 同样只模拟带宽延迟并用 link 物化。2026-05-12 进一步
  增加 `--request-remote-adapter-map`，用于 ServerlessLLM-vLLM 这类“router
  在每个请求中传入 backend-visible LoRA path”的系统：deploy 阶段只注册空
  cache 目标路径，请求首次触达 adapter 时再从 remote 拉取，耗时进入
  dispatch/admission wait。
- `scripts/run_remote_fair_main_rounds.sh`：固定按模型顺序运行
  Llama-2 7B、Llama-2 13B、Llama-3.2 3B；按系统顺序运行
  SGLang、ServerlessLLM、vLLM、S-LoRA。PrimeLoRA-vLLM 和
  PrimeLoRA-SGLang 不在本轮重跑，后处理时使用已经闭环的 PrimeLoRA 结果。

验证：

```bash
python3 -m py_compile scripts/replay_openai_trace.py scripts/materialize_remote_adapter_subset.py
bash -n scripts/run_remote_fair_main_rounds.sh scripts/run_vllm_fair_experiment.sh \
  scripts/run_sglang_fair_experiment.sh scripts/run_serverlessllm_fair_experiment.sh \
  scripts/run_slora_fair_experiment.sh
```

单 adapter smoke 显示一个约 20.9 MiB adapter 在 250 MiB/s local-sim 下约
83 ms 完成，符合带宽模型，materialization mode 为 link 而不是本地 copy。

`paper_remote_fair_local_v2` 在 2026-05-12 被中止并标记为诊断轮：SGLang/vLLM
已经是请求期 `dynamic_remote`，但 ServerlessLLM 仍是启动前整池 staging。
正式 local-sim rerun 从 `local_sim_v3` 开始，旧的
`11_remote_fair_main_local_sim_v2` 和更早的 `11_remote_fair_main` 都不能进入
论文表图。

当前有效执行轮次：

```bash
REMOTE_FAIR_MODE=local-sim \
REMOTE_FAIR_SECTION=11_remote_fair_main_local_sim_v4 \
REMOTE_FAIR_MODEL_LIST="llama2_7b llama2_13b llama32_3b" \
REMOTE_FAIR_SYSTEMS="sglang serverlessllm vllm slora" \
REMOTE_FAIR_BANDWIDTH_MBPS=250 \
REMOTE_FAIR_STAGE_WORKERS=1 \
KILL_KNOWN_GPU_RESIDUALS=1 \
bash scripts/run_remote_fair_main_rounds.sh
```

运行中的 tmux：

```bash
tmux attach -t paper_remote_fair_local_v4
tail -f /tmp/paper_remote_fair_local_v4.log
```

上一轮 `11_remote_fair_main/20260511_182342_*` 使用旧 local copy 逻辑，只能
作为诊断废数据，不能进入论文表图。

### 2026-05-12 local-sim v3 诊断结论与 v4 正式规则

`local_sim_v3` 完成了 Llama-2 7B baseline 诊断，并完成了 Llama-2 13B 的
SGLang 与 ServerlessLLM 阶段，但 vLLM 在 13B `dynamic_remote` 请求期 LoRA
加载路径上卡住。该失败不是 replay 结果，不能进入论文表图。

横向日志证据：

- vLLM 13B TP2 服务器仍能响应 `/v1/models`，但 replay 只完成少量请求后长期
  不再前进，未写出有效 summary。
- vLLM server log 已显示前几个 adapter 通过 `/v1/load_lora_adapter`
  运行期加载成功，随后请求路径无进展。
- 同一 adapter subset 在 SGLang、ServerlessLLM 和 7B vLLM remote-fair 中可用，
  因此根因不是工件损坏或 shared trace 错误。
- vLLM 自身日志提示运行期动态加载/卸载 LoRA 主要面向 local development；
  13B TP2 长 replay 下该路径不适合作为正式 baseline 路径。

正式修复规则：

- `dynamic_remote` 保留为 vLLM 诊断路径；除已经证明静态注册会触发资源边界的
  Llama-3.2 3B 外，不作为 remote-fair 正式默认。
- vLLM remote-fair 正式默认改为 `static_remote`：先按同一 remote endpoint
  将 selected adapter subset 物化到 round-local cache，再用官方稳定的
  `--lora-modules` 静态注册路径启动 vLLM。
- 旧 Llama-2 7B main vLLM 没有显式 `lora_registration_mode` 字段，但实际启动
  仍是 500 个 selected adapters 的 `--lora-modules` 静态注册；旧 Llama-3.2 3B
  main vLLM 才是显式 dynamic LoRA。不要把“字段缺失”误判为 dynamic。
- vLLM remote staging 时间通过 `VLLM_REMOTE_STAGE_SEC` 传给 summarizer 的
  `--predeploy-startup-sec`，计入 lifecycle/cost，而不是被隐藏在实验前处理。
- S-LoRA 也同样将 remote staging 时间通过 `--predeploy-startup-sec` 计入
  lifecycle。S-LoRA 官方启动期必须读取完整 `--lora-dirs` registry，因此它的
  remote-fair 口径是 remote 到 host/local tier 的启动期准备。
- `run_slora_fair_experiment.sh` 从 v4 patch 起把 S-LoRA 的 runtime startup
  和 remote artifact stage 分开传给 summarizer：`--static-startup-sec` 只记录
  S-LoRA server ready 时间，`--predeploy-startup-sec` 记录 remote stage 时间。
  这样避免后续补跑把 remote stage 同时算进 static startup 和 predeploy startup。

因此，`local_sim_v3` 需要按系统细分使用，而不是整轮废弃。有效保留项：

- Llama-2 7B 的 SGLang、ServerlessLLM、S-LoRA 均已完成 4000/4000 且满足
  当前 remote-fair 口径。S-LoRA 虽然旧 summary 中
  `predeploy_startup_sec=0`，但当轮 runner 已把 `remote_stage_sec` 加入
  `static_startup_sec`，因此 cost/lifecycle 没有漏计。
- Llama-2 13B 的 SGLang、ServerlessLLM 已完成 4000/4000，可保留。

需要补跑项：

- Llama-2 7B vLLM：v3 使用 `dynamic_remote` 并完成，但正式规则已改为
  `static_remote`，为避免同一论文表中 vLLM 7B/13B 使用不同 LoRA 注册路径，
  需要补一个 7B `static_remote` vLLM 结果。
- Llama-2 13B vLLM：v3 `dynamic_remote` 卡住，无有效 summary，必须补跑
  `static_remote`。
- Llama-2 13B S-LoRA：v3 未运行到该阶段，必须补跑。
- Llama-3.2 3B：后续已完成 SGLang、ServerlessLLM 和 vLLM，其中 vLLM
  `static_remote` 触发 host-memory guard 后改为模型粒度 `dynamic_remote`；
  剩余 S-LoRA 正在补齐/验证。

新的补缺候选数据从 `11_remote_fair_main_local_sim_v4_patch` 开始，按上述最小
集合执行，不再重跑已经有效的 SGLang/ServerlessLLM/S-LoRA 7B 或
SGLang/ServerlessLLM 13B。后续生成表图时将 v3 有效结果、v4 patch 补缺结果
和已闭环 PrimeLoRA 结果合并，并在 manifest 中记录每个系统的来源路径。

2026-05-12 13:54 已完成第一项补缺：Llama-2 7B vLLM `static_remote`。
该 round 位于：

`results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_124128_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`

有效性检查：

- replay `ok=4000/4000`，`fail=0`；
- `validate_replay_results.py` 通过，无 `trace_expected` token fallback；
- TTFT avg/p95 = 501.18/1179.94 ms；
- service TTFT avg = 487.81 ms，dispatch wait avg = 13.38 ms；
- E2E avg = 3153.23 ms，TPOT avg = 26.00 ms，Tok/s = 104.61；
- Cost/req = 3.671 mUSD，CE = 86.39。

与旧闭环 vLLM 对比：旧 Llama-2 7B main vLLM 为 TTFT avg 517.36 ms、
E2E avg 3223.25 ms、TPOT 26.57 ms、Tok/s 104.54、Cost/req 3.641 mUSD、
CE 85.20；因此 `static_remote` 补缺结果与旧主线同量级且未损坏 vLLM
性能。与 `local_sim_v3 dynamic_remote` 对比，dispatch wait 从 40.25 ms
回到 13.38 ms，说明 vLLM 的正式 remote-fair 口径应使用 `static_remote`，
而不是不稳定的 request-time dynamic registration。

2026-05-12 15:06，第二项补缺 Llama-2 13B vLLM `static_remote` 完成：

- 路径：
  `results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_135444_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，无 token fallback；
- remote stage = 75.21 s，写入 `predeploy_startup_sec`；
- TTFT avg/p95 = 3187.10/17941.12 ms；
- service TTFT avg = 3172.49 ms，dispatch wait avg = 14.61 ms；
- E2E avg = 8333.40 ms，TPOT avg = 70.35 ms，Tok/s = 98.82；
- Cost/req = 3.736 mUSD，CE = 32.12。

与同一 Llama-2 13B `local_sim_v3` 的有效 baseline 对比：SGLang 为
TTFT avg 430.33 ms、service TTFT 402.07 ms、dispatch wait 28.26 ms、
E2E avg 3262.26 ms、TPOT 33.13 ms、Tok/s 101.83、Cost/req 3.581 mUSD、
CE 85.61；ServerlessLLM 为 TTFT avg 236128.50 ms、service TTFT
501.29 ms、dispatch wait 235627.22 ms、E2E avg 239429.97 ms、TPOT
32.59 ms、Tok/s 91.84、Cost/req 3.365 mUSD、CE 1.24。vLLM 13B 的
dispatch wait 很低，说明 remote-fair patch 没有引入新的排队问题；它的差距
主要来自 vLLM 在当前 13B/TP2/LoRA 配置下的 service TTFT 和 TPOT，而不是
artifact 传输口径。该结果有效，但论文若合并 13B 需结合 PrimeLoRA 13B
是否能取得合理 CE 第一来决定。

2026-05-12 19:27，第三项补缺 Llama-2 13B S-LoRA `local-sim remote-fair`
完成：

- 路径：
  `results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_135444_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，`validate_replay_results.py` 通过；
- 无 `trace_expected` token fallback；
- 配置为官方兼容折中 `DP=1, TP=4, BMM=1`；
- TTFT avg/p95 = 5.232e6/9.943e6 ms；
- service TTFT avg = 5.232e6 ms，dispatch wait avg = 24.18 ms；
- E2E avg/p95 = 5.274e6/9.994e6 ms，TPOT avg = 340.95 ms；
- Cost/req = 13.124 mUSD，CE = 0.0144。

与旧 `03_main_comparison/20260507_llama13b_main_cap8_core_*` 的 13B S-LoRA
结果相比，新的 remote-fair 补缺略好但仍同属“有效但极慢”区间：旧结果为
TTFT avg 5.926e6 ms、E2E avg 5.972e6 ms、TPOT 367.89 ms、Cost/req
14.008 mUSD、CE 0.012。两轮共同说明瓶颈不是 remote-fair patch，也不是
replay 崩溃，而是当前 4x RTX 3090 上 S-LoRA 公开实现的 13B TP4/BMM
服务路径本身。该结果可用于复现边界和附录说明；若主文合并 13B，仍必须以
PrimeLoRA 13B 是否能在同口径下取得合理 CE 第一作为纳入条件。

### True-remote SGLang diagnostic

已完成的 `10_remote_artifact_diagnostic/20260511_162422_*` 使用真实远程
artifact node 测试 Llama-3.2-3B 的 SGLang-DP4。与同一 3B workload 的本地
main SGLang 结果相比：

| Setting | TTFT avg | Service TTFT avg | Dispatch wait avg | E2E avg | TPOT avg | Tok/s | CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local artifact path | 120.87 ms | 103.78 ms | 17.09 ms | 1379.85 ms | 11.00 ms | 113.09 | 199.88 |
| True remote artifact | 227.90 ms | 105.56 ms | 122.34 ms | 1487.18 ms | 11.08 ms | 112.91 | 187.63 |

结论：真实远程 remote 主要增加 adapter readiness/dispatch 阶段等待，backend
service TTFT、TPOT 和 token throughput 基本不变。这说明 remote artifact
链路没有改坏 SGLang 的生成性能，也说明 local-sim remote 在所有系统同口径
启用时可以作为可控的远程传输模型；正式论文表图仍以重新闭环的
`local_sim_v3` 结果为准。

2026-05-12 20:37，Llama-3.2 3B SGLang `local-sim remote-fair` 补缺完成：

- 路径：
  `results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_192609_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，`validate_replay_results.py` 通过；
- 无 `trace_expected` token fallback；
- TTFT avg = 144.27 ms，service TTFT avg = 104.36 ms，dispatch wait avg =
  39.91 ms；
- E2E avg = 1401.83 ms，TPOT avg = 10.99 ms，Tok/s = 113.09；
- Cost/req = 3.636 mUSD，CE = 196.20。

与旧 3B 本地 artifact SGLang 主结果相比，旧结果为 TTFT avg 120.87 ms、
service TTFT 103.78 ms、dispatch wait 17.09 ms、E2E avg 1379.85 ms、
TPOT 11.00 ms、Tok/s 113.09、Cost/req 3.626 mUSD、CE 199.88。两者的
service TTFT、TPOT 和吞吐几乎一致，local-sim remote-fair 主要把首次
adapter 准备体现在 dispatch/readiness 侧。与真实 remote 诊断相比，真实
remote 的 dispatch wait 更高（122.34 ms）且 CE 更低（187.63），方向一致。

2026-05-12 21:49，Llama-3.2 3B ServerlessLLM `local-sim remote-fair`
补缺完成：

- 路径：
  `results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_192609_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，`validate_replay_results.py` 通过；
- 无 `trace_expected` token fallback；
- TTFT avg/p95 = 236435.81/469526.08 ms；
- service TTFT avg = 502.26 ms，dispatch wait avg = 235933.55 ms；
- E2E avg/p95 = 238086.45/471037.56 ms，TPOT avg = 14.91 ms；
- Tok/s = 107.34，Cost/req = 2.278 mUSD，CE = 1.8438。

与旧 3B main ServerlessLLM 对比，旧结果为 TTFT avg 235944.29 ms、
service TTFT 485.66 ms、dispatch wait 235458.63 ms、E2E avg 237592.78 ms、
TPOT 14.90 ms、Tok/s 107.38、Cost/req 2.244 mUSD、CE 1.8759。两者几乎
完全同型：ServerlessLLM 的 backend service path 正常，差距仍由
dispatch/admission backlog 主导。local-sim remote-fair 没有引入新的失败模式，
也没有改变该 baseline 的基本趋势。

2026-05-12 21:52，修复 vLLM remote-fair 补跑配置优先级：

- 旧主线中 Llama-3.2 3B standalone vLLM 的 launch spec 使用
  `lora_registration_mode: dynamic`，这是为了避免 500-LoRA 静态注册在
  4x RTX 3090 上造成过高 host-memory/启动压力。
- remote-fair 正式口径在 2026-05-12 已改为 vLLM `static_remote`：先从同一
  remote endpoint 将 selected 500 adapters materialize 到 round-local cache，
  再通过官方 `--lora-modules` 启动 vLLM，并把 remote stage 记入 lifecycle。
- 发现 `run_full_fair_round.sh` 的 Llama-3.2 safe topology override 会在
  已显式传入 `VLLM_LORA_REGISTRATION_MODE=static_remote` 时覆盖回 `dynamic`。
  该逻辑已修复为“显式实验口径优先，模型默认只在未指定时生效”。
- 因此，3B vLLM `static_remote` 补跑重启为新 round
  `20260512_215256_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`。
  旧的 3B dynamic 主线仍是有效历史结果，但不混入 remote-fair 正式表。
- 该 `static_remote` 尝试在 2389/4000 时触发 host-memory fail-fast guard：
  `MemAvailable=32753 MiB < 32768 MiB`。这是预期中的根因确认：3B vLLM
  静态注册 500 adapters 会把主机内存压到安全阈值边缘，不能通过调低 guard
  硬跑。正式 3B vLLM remote-fair 因此改为模型粒度 `dynamic_remote`：首次触达
  adapter 时从同一 remote endpoint 拉取到 round-local cache，同时保留 7B/13B
  已验证的 `static_remote`。新 round 为
  `20260512_223735_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`。

2026-05-12 23:47，Llama-3.2 3B vLLM `dynamic_remote` 补缺完成：

- 路径：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_223735_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，无 `trace_expected` token fallback；
- fleet 记录 `lora_registration_mode: dynamic_remote`、`dynamic_lora_routing: adapter_hash`、
  `dynamic_lora_max_loaded_per_endpoint: 24`；
- TTFT avg/p95 = 323.34/975.57 ms；
- service TTFT avg = 299.41 ms，dispatch wait avg = 23.93 ms；
- E2E avg/p95 = 2421.49/5653.53 ms，TPOT avg = 20.02 ms；
- Tok/s = 112.83，Cost/req = 3.593 mUSD，CE = 114.95。

与旧 3B main vLLM 对比，旧结果为 TTFT avg 312.47 ms、service TTFT
295.40 ms、dispatch wait 17.07 ms、E2E avg 2405.04 ms、TPOT 20.13 ms、
Tok/s 112.79、Cost/req 3.581 mUSD、CE 116.11。新结果只增加了小幅
request-time remote materialization/dispatch 成本，service path 和 TPOT 基本
一致，说明 `dynamic_remote` 是当前 3B/vLLM/500-LoRA/4x3090 组合的正确
remote-fair 口径。该结论只适用于 Llama-3.2 3B；Llama-2 7B 与 Llama-2 13B
仍使用已验证的 `static_remote`。

2026-05-13 01:01，Llama-3.2 3B S-LoRA `local-sim remote-fair` 补缺完成：

- 路径：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4_patch/20260512_223735_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1`
- replay `ok=4000/4000`，`fail=0`，`validate_replay_results.py` 通过；
- 无 `trace_expected` token fallback；
- TTFT avg/p95 = 289.23/465.89 ms；
- service TTFT avg = 273.04 ms，dispatch wait avg = 16.19 ms；
- E2E avg/p95 = 7273.13/18260.11 ms，TPOT avg = 126.64 ms；
- Tok/s = 119.20，Cost/req = 3.691 mUSD，CE = 37.25。

与旧 3B main S-LoRA 对比，旧结果为 TTFT avg/p95 308.72/556.92 ms、
service TTFT avg 292.69 ms、dispatch wait avg 16.03 ms、E2E avg/p95
7877.40/20244.87 ms、TPOT avg 137.31 ms、Tok/s 118.96、Cost/req
3.623 mUSD、CE 35.04。新结果在 TTFT、E2E、TPOT 上略好，成本略高，
总体仍是同量级有效结果；remote-fair patch 没有破坏 S-LoRA 3B 路径。

2026-05-13 已生成不覆盖旧结果的 Llama-2 7B + Llama-3.2 3B
`local-sim remote-fair` 论文候选表图：

- 输出目录：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main_remote_fair_local_sim_v4_7b3b`
- 表格：
  `table1_end_to_end.tex`、`table_ttft_decomposition.tex`
- 图：
  `fig7_lifecycle_cost.pdf`
- 数据：
  `table1_end_to_end_data.csv`、`table_ttft_decomposition_data.csv`、
  `fig7_lifecycle_cost_data.csv`

该目录只保存本轮 remote-fair local-sim 的有效最终合并结果，不覆盖旧主线图表。
其中 Llama-2 7B 使用已验证的 SGLang/ServerlessLLM/S-LoRA local-sim v3、
vLLM static_remote v4、PrimeLoRA 旧闭环主线；Llama-3.2 3B 使用
SGLang/ServerlessLLM local-sim v4、vLLM dynamic_remote v4、S-LoRA
dynamic batch round、PrimeLoRA 旧闭环调优主线。两个模型的表内 PrimeLoRA
CE 均为第一：7B 为 123.02，3B 为 241.20。

## 2026-05-13 true-remote artifact validation

真实两节点 remote artifact 复查使用 GPU 节点 `192.168.4.178` 和 artifact
HTTP 节点 `192.168.4.174`。endpoint 对应关系：

- Llama-2 7B: `http://192.168.4.174:18081`
- Llama-2 13B: `http://192.168.4.174:18082`
- Llama-3.2 3B: `http://192.168.4.174:18080`

baseline 队列：

```bash
tmux attach -t remote_fair_real_v1
tail -f /tmp/remote_fair_real_v1.log
```

只读对比 watcher：

```bash
tmux attach -t remote_fair_compare_watch_v1
tail -f /tmp/remote_fair_real_v1_compare_watch.log
```

该 watcher 每 5 分钟读取已生成 summary，并与
`figs/paper/main_remote_fair_local_sim_v4_7b3b/table1_end_to_end_data.csv`
对比；只写 `/tmp` 中的 compare log/csv/md，不修改实验数据。

2026-05-13 06:36，Llama-2 7B true-remote baseline round 已闭环：

- SGLang：4000/4000，0 fail，TTFT Avg 274.48 ms，E2E Avg 2427.36 ms，
  Cost/req 3.599 mUSD，CE 114.47。相比 local-sim：TTFT Avg +14.99%，
  E2E Avg +1.80%，Cost/req +0.61%，CE -2.37%。
- ServerlessLLM：4000/4000，0 fail，TTFT Avg 236575.70 ms，
  E2E Avg 239156.82 ms，Cost/req 2.692 mUSD，CE 1.55。相比 local-sim：
  TTFT Avg +0.18%，E2E Avg +0.18%，Cost/req +0.13%，CE -0.31%。
- vLLM：4000/4000，0 fail，TTFT Avg 500.68 ms，E2E Avg 3151.02 ms，
  Cost/req 4.066 mUSD，CE 78.04。相比 local-sim：TTFT Avg -0.10%，
  E2E Avg -0.07%，Cost/req +10.77%，CE -9.66%。
- S-LoRA：4000/4000，0 fail，TTFT Avg 325.50 ms，E2E Avg 3790.54 ms，
  Cost/req 4.151 mUSD，CE 63.56。相比 local-sim：TTFT Avg +23.27%，
  E2E Avg +8.73%，Cost/req +10.80%，CE -17.00%。

四个系统均无 `trace_expected` fallback。true-remote 对 serverful 静态/staging
系统的 Cost/req 影响更明显，但没有破坏 service path；ServerlessLLM 的主要
瓶颈仍是 dispatch/admission backlog。队列随后进入 Llama-2 13B true-remote
round。

2026-05-13 06:40，Llama-2 13B true-remote SGLang 首轮被立即标记为无效：
早期 `finance_lora` 请求出现 `RemoteDisconnected`，replay 已产生失败请求，
不满足正式结果 gate。根因不是 SGLang generation 或 GPU OOM，而是远端 artifact
节点打包 `finance_lora` 时遇到该目录中的绝对 symlink
`/home/qhq/serverless_llm_experiment/models/...`；真实 remote 节点没有该路径，
旧 artifact server 对 symlink 执行 `resolve(strict=True)` 导致
`FileNotFoundError`，HTTP 连接以 `Empty reply` 断开。已修复
`remote_artifact_node/server.py`：打包时跳过不可解析或越过 artifact root 的
非可移植 symlink，只发送 LoRA adapter payload 与本地真实文件。该修改不改变
实验 trace、adapter subset、routing、admission 或统计口径，只修复真实远端工件
服务的可移植性。修复后必须先用同一 13B trace/subset 做 `SGLang_MAX_REPLAY_REQUESTS`
smoke 且 `fail=0`，再恢复 4000-request 正式队列。

2026-05-13 07:00，13B SGLang true-remote smoke 又暴露一个独立的启动竞态：
SGLang 在未显式设置 `nccl-port` 时会通过 `get_free_port()` 随机选择内部
NCCL/torch distributed 通信端口，而 HTTP/Uvicorn 端口要等 engine 初始化后才
绑定。该随机端口可能碰巧选中计划中的 HTTP 端口 `8353`，导致 scheduler 先
监听 `127.0.0.1:8353`，随后 Uvicorn 报
`[Errno 98] address already in use`。这不是 remote artifact、LoRA 权重或
GPU OOM 问题，而是 SGLang 启动端口分配竞态；此前正式 round 能运行只是因为
随机端口没有撞上 HTTP 端口。已修复 `scripts/run_sglang_fair_experiment.sh`：
每个 replica 生成 launch spec 时显式写入 `nccl-port`，默认取
`http_port + 10000`，也可用 `SGLANG_NCCL_PORT_BASE` 覆盖。后续所有
SGLang fair runs 都必须保留该规则。

随后使用完整 `run_full_fair_round.sh` 包装执行同一 13B trace/subset 的
`SGLANG_MAX_REPLAY_REQUESTS=120` smoke：两个 TP2 replicas 均成功启动，
launch spec 中分别记录 `port=8353,nccl-port=18353` 与
`port=8354,nccl-port=18354`；replay 结果 `ok=120/120, fail=0`，且无
`trace_expected` fallback。为方便后续 smoke，`run_full_fair_round.sh`
的 summary gate 已支持 `SGLANG_MAX_REPLAY_REQUESTS`，正式 4000-request
round 不受影响。

2026-05-13 07:20，恢复 13B true-remote 正式 SGLang 后又发现第三个独立根因：
replica r0 在约 300+ 请求后退出，后续 replay 对 `127.0.0.1:8353` 的
`/load_lora_adapter` 调用出现 `Connection refused`，该轮已停止并标记为无效。
r0 server log 的真正异常是 SGLang scheduler 内部断言
`assert len(cur_uids) <= self.max_loras_per_batch`。这说明该 replica 的
运行 batch 中同时出现的 LoRA uid 数超过了启动时声明的
`max-loras-per-batch`。根因不是真实 remote 下载、artifact 内容或 GPU OOM，
而是 fair harness 只设置了 `max-loras-per-batch`，没有把 profile 中的
`max_num_seqs` 同步传给 SGLang 的 `max-running-requests`；13B TP2 解码更慢，
burst 阶段 running requests 可超过 LoRA batch 上限，从而触发断言。

修复规则：

- SGLang launch spec 必须写入 `max-running-requests=model.max_num_seqs`；
- `max-loras-per-batch` 必须至少覆盖 `max(max_loras, max-running-requests)`；
- replay 期间必须监控 SGLang replica PID，任一 replica 退出即终止 replay；
- formal SGLang 默认 `SGLANG_ABORT_AFTER_FAILURES=1`，任何请求失败都 fail-fast，
  不允许继续写半污染结果。

该修复对齐 SGLang 官方参数语义：`max-loras-per-batch` 是 running batch 内
可同时出现的 adapter 数上限，必须与 running request 上限一致。修复后先执行
同一 13B true-remote trace/subset 的 `SGLANG_MAX_REPLAY_REQUESTS=700` smoke，
必须越过原先 300+ 请求断言区且 `fail=0` 后，才能恢复 4000-request 正式队列。

2026-05-13 07:41，`12_remote_fair_smoke_real_remote_sglang_fix3` 已通过：
同一 13B true-remote trace/subset 的 SGLang smoke 完成 `ok=700/700`、
`fail=0`，无 `AssertionError`、无 `Connection refused`、无
`trace_expected` fallback。失败的 partial formal round 已写入
`INVALID_DO_NOT_USE.txt`，不得用于论文表图。随后恢复
`12_remote_fair_main_real_remote_v1` 的 13B/3B 4000-request 正式队列。

2026-05-13 08:51，修复后的 Llama-2 13B true-remote SGLang 正式阶段已完成：

- round：
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_074336_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- summary：
  `raw/replay/llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_sglang_dp_tpprofile_summary.json`
- replay gate：`completed=4000/4000`，`fail=0`，无 `trace_expected` fallback；
- 核心结果：TTFT Avg 493.81 ms，TTFT P95 1471.09 ms，E2E Avg 3253.71 ms，
  E2E P95 7666.36 ms，TPOT Avg 31.61 ms，Tok/s 100.50，
  Cost/req 3.581 mUSD，CE 85.83。

与最近的 13B local-sim remote-fair SGLang 有效轮
`11_remote_fair_main_local_sim_v3/20260512_084402_*` 对比：TTFT Avg +14.75%，
TTFT P95 +89.78%，E2E Avg -0.26%，Cost/req 基本不变，CE +0.26%。因此真实
remote 主要影响 first-touch/readiness tail，没有破坏 SGLang 的 service path
或总吞吐趋势。该阶段结果有效；队列已进入 Llama-2 13B true-remote
ServerlessLLM 阶段。

2026-05-13 10:03，Llama-2 13B true-remote ServerlessLLM 阶段已完成：

- round：
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_074336_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- summary：
  `raw/replay/llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_serverlessllm_summary.json`
- replay gate：`completed=4000/4000`，`fail=0`，无 `trace_expected` fallback；
- 核心结果：TTFT Avg 236555.75 ms，TTFT P95 470160.22 ms，
  Service TTFT Avg 501.33 ms，Dispatch Wait Avg 236054.43 ms，
  E2E Avg 239834.22 ms，TPOT Avg 32.44 ms，Tok/s 91.78，
  Cost/req 3.366 mUSD，CE 1.239。

与最近的 13B local-sim remote-fair ServerlessLLM 有效轮
`11_remote_fair_main_local_sim_v3/20260512_084402_*` 对比：TTFT Avg +0.18%，
E2E Avg +0.17%，Cost/req +0.01%，CE -0.18%，Service TTFT +0.01%，Tok/s
-0.07%。因此 true-remote 与 local-sim 在 ServerlessLLM 上趋势一致：主要瓶颈
仍是 serverless admission/backlog，而不是真实 remote artifact 传输本身。
该阶段结果有效；队列已进入 Llama-2 13B true-remote vLLM 阶段。

2026-05-13 11:25，Llama-2 13B true-remote vLLM 阶段已完成：

- round：
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_074336_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- summary：
  `raw/replay/llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_vllm_dp2_tp2_summary.json`
- replay gate：`completed=4000/4000`，`fail=0`，无 `trace_expected` fallback；
- 核心结果：TTFT Avg 17389.09 ms，TTFT P95 85658.39 ms，
  Service TTFT Avg 17374.06 ms，Dispatch Wait Avg 15.03 ms，
  E2E Avg 23962.66 ms，TPOT Avg 85.15 ms，Tok/s 98.99，
  Cost/req 4.154 mUSD，CE 10.047。

与最近的 13B `local-sim remote-fair` vLLM 有效轮
`11_remote_fair_main_local_sim_v4_patch/20260512_135444_*` 对比：TTFT Avg
+445.6%，E2E Avg +187.5%，TPOT +20.9%，Tok/s +0.17%，Cost/req +11.2%，
CE -68.7%。该轮有效但不替代已闭环 local-sim 主结果；它说明在真实远程冷
artifact staging 下，13B vLLM 的主要放大项出现在 service path 内部队列/LoRA
load 长尾，而不是上游 dispatch/admission wait，也不是请求失败。队列随后进入
Llama-2 13B true-remote S-LoRA 阶段。

2026-05-14，`12_remote_fair_main_real_remote_v1` 已全部闭环：

- 7B、13B、3B 三个 true-remote baseline round 均完成 SGLang、
  ServerlessLLM、vLLM、S-LoRA。
- 每个正式 summary 均满足 `completed=4000/4000`、`fail=0`，且没有
  `trace_expected` fallback。
- 对比文件保存在
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/_comparisons/`。
- 7B 与 3B true-remote 主表由 FaaSLoRA 仓库合并生成；当前完整 mirror 位于
  `figs_remote_full_real_remote_v1/` 与
  `paper_results/final_remote_full_real_remote_v1/`。早期
  `figs/paper/main_remote_fair_real_remote_v1_7b3b/` / `figs_remote/` 路径只保留为
  历史 candidate 记录；13B 只作为诊断数据保留。

最终 baseline 侧 true-remote 摘要：

- 7B SGLang/vLLM/S-LoRA/ServerlessLLM CE 分别为 `114.47`、`78.04`、
  `63.56`、`1.55`。
- 13B SGLang/vLLM/S-LoRA/ServerlessLLM CE 分别为 `85.83`、`10.05`、
  `0.013`、`1.24`。13B 不进入当前主表。
- 3B SGLang/vLLM/S-LoRA/ServerlessLLM CE 分别为 `185.66`、`108.04`、
  `27.54`、`1.83`。

该 true-remote 队列修复了三个根因并记录为复现规则：remote artifact server
跳过不可移植 symlink，SGLang 显式设置 `nccl-port`，SGLang many-LoRA 的
`max-running-requests` 与 `max-loras-per-batch` 对齐。以后所有同类 round
必须保留这些规则。
