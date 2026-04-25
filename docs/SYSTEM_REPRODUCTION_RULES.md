# 系统复现规则与踩坑清单

本文档是后续复现任何新系统、维护已有 baseline、或把新论文系统接入
`FaaSLoRA` 公平对比链路时必须遵守的工程规则。它不是实验结果分析文档，
而是“如何复现一个系统才算合格”的执行标准。

核心目标只有一个：让每个对比系统在同一问题、同一负载、同一 LoRA 子集、
同一 GPU 预算、同一指标口径和同一成本模型下，输出可审计、可回退、
可写进论文的结果。

## 1. 复现目标定义

复现一个系统不等于“能跑一次 demo”。正式复现至少必须满足：

1. 能使用当前 shared trace。
2. 能使用当前 shared adapter subset。
3. 能处理正式 many-LoRA 请求路径。
4. 能在同一 GPU 预算下运行。
5. 能输出 `metric_schema_version=e2e_v3`。
6. 能进入统一 compare 表。
7. 失败时能明确报错，而不是生成全 0 或半空 summary。
8. 复现边界、不可观测指标、已知限制必须写入文档。

9. 如果系统声明存在 Host/CPU-memory、NVMe、Remote、GPU 等物理层级，
   wrapper 必须验证该层级的真实 backing store。禁止把普通磁盘目录当作
   Host memory 层，或把 fallback 路径伪装成论文机制。

不满足这些条件的系统只能标记为：

- `smoke only`
- `limited auxiliary baseline`
- `paper-faithful reimplementation`
- `related work only`

不能包装成正式主 baseline。

### 1.1 物理层真实性 gate

正式系统实验必须遵守“名称对应物理实现”的 gate：

- `GPU` 层必须对应真实 GPU runtime/cache/adapter admission；
- `HOST` 或 `main memory` 层必须对应 DRAM-backed 路径或真实进程内内存；
- `NVMe` 层必须对应本地持久化存储路径；
- `REMOTE` 层必须对应远端源仓库、对象存储模拟目录或跨节点传输路径；
- 成本中的 active/idle/lifecycle GPU seconds 必须来自真实 runtime lifecycle 或统一模拟审计，而不是硬编码常量。

对于 PrimeLoRA/FaaSLoRA，`HOST` adapter tier 通过 vLLM `LoRARequest`
仍需要普通文件路径，因此正式实现使用 `/dev/shm/faaslora_host_cache`
下的 tmpfs 目录承载 adapter 文件。启动时必须校验 mount、filesystem type、
可写性和容量余量，并把这些元数据写入结果 JSON。若校验失败，正式实验
必须停止，而不是静默使用 `artifacts/host_cache` 等 ext4 目录。

## 2. 新系统准入检查

复现任何新系统前，必须先检查：

1. 是否有论文或明确系统定位。
2. 是否有官方开源代码。
3. 是否支持当前目标模型族：
   - `Llama-2 7B`
   - `Llama-2 13B`
   - `Qwen 7B`
   - `Qwen 14B`
4. 是否支持 LoRA 或至少能以 paper-faithful 方式适配 LoRA。
5. 是否能处理 `500/4000 requests + 500 adapters` 的 many-LoRA 场景。
6. 是否能对齐 4 GPU 预算。
7. 是否能暴露或通过客户端真实观测得到：
   - `TTFT_e2e`
   - `TTFT_service`
   - `E2E_e2e`
   - `E2E_service`
   - `TPOT`
   - `Cost/req`
   - `CE`
   - `Cost/1MTok`
   - GPU-second resource metrics
8. 如果某些机制指标无法真实观测，调试审计中是否能明确说明缺失；正式论文图表不能依赖这些字段，不能用 `null`、估计值或近似值进入主 TODO。

如果这些问题没查清，禁止直接写运行脚本。

## 3. 项目化目录规则

每个新系统必须建立独立项目入口：

```text
System_project/
  README.md
  repo -> official upstream repo or symlink
  scripts/
  docs/
  configs/
  results/ or symlink
  logs/ or symlink
  models/ or symlink
```

要求：

- `README.md` 写明系统定位、源码位置、环境、正式入口、复现边界。
- `scripts/` 中必须有用户可直接调用的 wrapper。
- `docs/` 中必须记录复现计划、已知坑、指标能力边界。
- 官方源码优先保持原样，公平适配逻辑放在 harness 层。
- 不复制多份源码，不让结果散落在临时目录。

已有事实：

- `Punica` 已完成 `Llama-2 7B` 受限复现，不再当作新系统重复规划。
- `SGLang` 已接入正式 shared artifact / e2e_v3 链路。
- `ServerlessLLM` 已接入正式 shared artifact / e2e_v3 链路。
- `vLLM` 已接入 standalone OpenAI-compatible server 正式 harness。
- `S-LoRA` 已建立 `S-LoRA_project`，使用独立 `slora_official_cu118`
  环境和项目 wrapper 接入 shared artifact / `e2e_v3` replay。由于该系统
  属于官方 serverful multi-LoRA paper baseline，复现只能做环境、输入和
  观测口径适配，不修改其核心调度或 paging 机制；若真实 replay 报错或效果
  异常，应保留原论文语义并在横向分析中说明复现边界。

## 4. 公平输入规则

所有正式对比系统必须使用：

- 同一个 frozen sanitized pool。
- 同一个 shared trace artifact。
- 同一个 shared adapter subset artifact。
- 同一个 total request 数。
- 同一个 selected adapter 数。
- 同一个 sampling seed。
- 同一个 time scale。
- 同一个 prompt/token budget guard。

禁止：

- 每个系统各自采样 LoRA。
- 每个系统各自生成 trace。
- 一个系统使用原始冻结池，另一个系统使用 sanitized pool。
- 一个系统使用 chat template 裁剪，另一个系统不裁剪。
- 一个系统静默减少 LoRA 数量。
- 一个系统失败后自动 fallback 到另一个 backend，却仍用原系统名字记录。

shared trace 必须记录：

- `model_profile`
- `dataset_profile`
- `workload_profile`
- `total_requests`
- `selected_num_adapters`
- `sampling_seed`
- `configured_time_scale_factor`
- `effective_time_scale_factor`
- `load_profile`

缺少这些字段的结果不能进入论文主表。

## 5. LoRA 选择语义

不同系统的 LoRA 请求语义不同，必须逐个对齐，不能凭字段名猜。

### vLLM

vLLM OpenAI server 的 LoRA 语义是：

- 启动时用 `--enable-lora`。
- 用 `--lora-modules name=path` 注册 LoRA。
- 请求时把 LoRA adapter name 放在 OpenAI request body 的 `model` 字段。

因此 replay 必须：

```text
adapter_id -> model
```

### SGLang

当前 SGLang harness 使用 `lora_path` 或等价字段选择 adapter。

必须注意：

- replay 脚本会先 drop body fields，再做 adapter mapping。
- 如果 source field 放在 body 中又被 drop，就会导致 LoRA 选择丢失。
- 因此正式链路必须使用 trace 顶层 `adapter_id` 作为 source。

已踩坑：

```text
--adapter-source-field lora_adapter_name
--drop-body-field lora_adapter_name
```

这会让 adapter mapping 丢失，导致 baseline 可能实际跑 base model。正确方式是：

```text
--adapter-source-field adapter_id
```

### ServerlessLLM

ServerlessLLM 通过 deploy config 和模型注册路径识别 adapter。

要求：

- deploy config 必须由 shared adapter subset 生成。
- backend 必须明确标注，例如 `ServerlessLLM-vLLM` 或
  `ServerlessLLM-transformers`。
- `SLLM_BACKEND=auto` 只能用于 bring-up/smoke，不能进入正式主表。

## 6. Prompt 与 API 入口规则

不能把 OpenAI-compatible 都当成同一个接口。不同模型和 endpoint 的 prompt
语义不同。

### vLLM Llama base tokenizer 坑

已踩坑：

```text
/v1/chat/completions
Llama-2-7B-HF base tokenizer
transformers >= 4.44
```

会报：

```text
ValueError: default chat template is no longer allowed
HTTP 400 Bad Request
```

原因：Llama-2 base tokenizer 没有 chat template。

修复规则：

- vLLM standalone 正式 harness 使用 `/v1/completions`。
- replay 将 shared trace 中的 `messages` 渲染为 `prompt`。
- prompt guard 必须与 SGLang / FaaSLoRA 对齐。
- 不能把 chat endpoint 失败写成性能结果。

### vLLM usage/token fallback 坑

已踩坑：

```text
vLLM /v1/completions streaming
metrics_source_counts={"missing": 500}
prompt/output token stats fall back to raw trace expected tokens
```

现象：

- `TTFT/E2E` 是真实 client-observed timing，仍可用于初步判断；
- `prompt_tokens / completion_tokens / TPOT / Cost/1M tokens` 会被 raw trace token
  budget 污染；
- 典型表现是 vLLM 的 `total_output_tokens` 明显偏离 SGLang、ServerlessLLM 和
  FaaSLoRA。

修复规则：

- replay 在 prompt guard 后记录 `local_guarded_prompt` token count。
- 当 server `usage` 缺失时，replay 从实际响应文本中用同一 tokenizer 计算
  `completion_tokens`。
- 对 vLLM standalone OpenAI streaming replay，正式 wrapper 必须显式加入
  `stream_options.include_usage=true`，让 vLLM 在最终 SSE chunk 中返回真实
  token usage。
- 对正输出 trace 请求，正式 wrapper 必须设置 `min_tokens=1`。否则 vLLM
  OpenAI completion 可能首步 EOS，返回 `200 OK`、`finish_reason=stop`、
  但没有 generated text；这种请求没有可定义的 first-token event，不能作为
  `TTFT` 样本进入正式结果。
- replay 写出 `prompt_token_source` 与 `completion_token_source`。
- summary 写出 `prompt_token_source_counts` 与
  `completion_token_source_counts`。
- `HTTP 200` 但没有任何可观测成功负载的响应不能当作成功。可观测成功
  负载至少包括 `usage`、server metrics 或实际生成文本之一；只有 SSE
  结束帧、空白 chunk、不可解析 body 或空 JSON envelope 都不能用于正式
  token/latency/cost 统计。可对这类协议级空响应做少量 client retry，
  retry 时间仍计入同一个请求的延迟窗口，重试后仍为空则必须失败。
- OpenAI-compatible replay 必须优先按 JSON object 解析非流式响应；只有行首
  `data:` 的 Server-Sent Events 才能按流式响应处理。不能用
  `if "data:" in raw_text` 这类全文子串判断 SSE，因为模型输出文本本身可能
  包含 `data:`，例如代码、URL 或普通英文句子。
- 正式结果中不允许 token source 仍大面积为 `trace_expected`；否则只能作为
  timing bring-up，不能进入 token/TPOT/cost-per-token 图。

### SGLang prompt guard

SGLang 正式 replay 必须：

- 将 `messages` 渲染为 prompt。
- 应用 FaaSLoRA 风格 prompt guard。
- 保证 max model length、max input length、max output cap 与 profile 一致。

### Chat endpoint 使用条件

只有当以下条件同时成立时，才能使用 `/v1/chat/completions`：

1. tokenizer 或 server 明确有可用 chat template。
2. 所有系统都能以同一消息渲染语义处理请求。
3. smoke 测试证明至少一个 LoRA 请求能返回成功。

否则必须使用 prompt/completions 路径。

### S-LoRA native endpoint

S-LoRA 官方 `/generate_stream` 不是 OpenAI-compatible 请求体。正式 wrapper 必须
生成如下语义：

```text
prompt/messages -> inputs
max_tokens      -> parameters.max_new_tokens
adapter_id      -> lora_dir 真实路径
```

注意：

- S-LoRA `/v1/chat/completions` 当前源码没有把 LoRA adapter 显式传入底层
  generate path，不能作为正式 LoRA 对比入口。
- S-LoRA 请求中的 `lora_dir` 必须与 server 启动时 `--lora-dirs` 注册的字符串完全一致。
- S-LoRA `/generate_stream` 服务端在 `httpserver/manager.py` 中按
  `tokenizer.encode(prompt)` 计算输入长度，默认会计入 special tokens。正式
  prompt guard 必须按同一语义做 budget；如果沿用
  `add_special_tokens=False`，会在 `984/985` 边界上出现系统性越界并导致
  `HTTP 200 + Response ended prematurely`。
- S-LoRA 在正式接入前必须先跑单 adapter smoke，证明不是静默 base-model fallback。
- S-LoRA 正式 wrapper 必须在 replay 后检查 `prompt_token_source` 与
  `completion_token_source`；如果仍回退到 `trace_expected`，直接失败，不能进入
  TPOT / token cost / Cost-per-token 图。

## 7. 指标输出规则

正式结果只接受：

```text
metric_schema_version=e2e_v3
```

共同主指标：

- `TTFT_e2e`
- `TTFT_service`
- `dispatch_admission_wait_ms`
- `E2E_e2e`
- `E2E_service`
- `TPOT`
- `Tok/s`
- `RPS`
- `SLO`
- `Cost/req`
- `Cost/1MTok`
- `CE`
- `InfraCost`
- `InfraCE`
- GPU-second resource metrics

每个关键延迟必须尽量输出：

- `avg`
- `p50`
- `p95`
- `p99`

机制指标只能真实观测，不能估计：

- cold start
- LoRA I/O
- GPU-ready hit
- warm pool hit
- scale-up affected requests
- residency hit/miss
- preload hit/miss

如果 baseline 没有该机制，或无法真实观测，调试审计可以记录为缺失或 `null`，不能填 `0`
来暗示“没有开销”。正式论文图表和实验 TODO 不使用这类字段，只使用所有参与系统都能真实输出的共同字段。

## 8. TTFT / E2E 分层规则

必须区分：

```text
scheduled trace arrival
  -> replay dispatch
  -> system admission / backend start
  -> first generated token
  -> response complete
```

定义：

```text
dispatch_admission_wait_ms =
  backend/service start - scheduled trace arrival

TTFT_service =
  first token - backend/service start

TTFT_e2e =
  dispatch_admission_wait_ms + TTFT_service

E2E_service =
  response complete - backend/service start

E2E_e2e =
  dispatch_admission_wait_ms + E2E_service
```

注意：

- 如果后端暴露 `backend_started_at`，则 `request_received_at -> backend_started_at`
  属于 queue/admission，不属于 service TTFT。
- 如果没有服务端 token 时间，只能使用客户端 streaming chunk 时间。
- 非流式完整 JSON 响应不能真实拆 first token，`TPOT` 必须标记为不可观测或诊断。

## 9. 成本与资源规则

主成本使用 monetary lifecycle cost，而不是旧 token proxy cost。

统一公式：

```text
Cost_serverful =
  P_gpu * GPUSeconds_lifecycle

Cost_serverless =
  P_gpu * (GPUSeconds_startup + GPUSeconds_active
           + alpha_idle * GPUSeconds_idle_ready)
  + C_invocation * N_completed

Cost/req = Cost / N_completed
CE       = 1 / (avg E2E_e2e seconds * Cost/req)
```

保留诊断项：

- `TokenProxy/req`
- `TokenProxyCE`
- `InfraCost`
- `InfraCE`
- `Cost/1MTok`
- `GoodTok/GPU-s`
- `ActiveGPU_s`
- `IdleReadyGPU_s`

解释规则：

- `Cost/req` 是主 CE 的成本项。
- `Cost/1MTok` 用于补充说明 token-normalized monetary cost。
- `TokenProxy` 只是 legacy 诊断，不能作为论文主结论。
- Serverful runtime 必须计入 launch-to-ready 和全生命周期 GPU-second。
- Serverless runtime 可按差分 idle factor 计入 idle-ready GPU-second。

禁止：

- 为了让某个系统更好看而改成本公式。
- 某系统少算启动 GPU-second。
- 某系统少算 idle-ready GPU-second。
- 用 token proxy cost 冒充 cloud lifecycle cost。

## 10. 资源拓扑规则

同一主实验默认对齐 4 GPU 预算。

推荐主表设置：

- `SGLang`: `DP=4, TP=1`
- `vLLM`: `DP=4, TP=1`
- `FaaSLoRA`: 四卡 serverless pool
- `ServerlessLLM`: 四卡 worker pool

说明：

- `TP=4` 是 serverful model-parallel upper bound，可作为附表，不进入主公平表。
- 对 13B 或 Qwen14B，如模型 profile 需要 TP=2，则应清楚记录：
  - `tensor_parallel_size`
  - `data_parallel_replicas`
  - `runtime_gpu_count`
  - `gpu_per_request`

每个 summary 必须能 audit：

- visible GPUs
- runtime GPUs
- TP
- DP
- GPU/request
- max billing GPUs
- lifecycle GPU-seconds

## 11. 失败处理规则

任何请求失败都不能沉默。

正式 harness 必须：

1. replay 后检查 `ok == total_requests`。
2. 如果失败，打印至少前几个失败样本：
   - `request_id`
   - `adapter_id`
   - `status_code`
   - `error`
3. 失败时退出非 0。
4. 不生成或不继续使用全 0 summary。
5. 在生成 summary 前统一运行
   `scripts/validate_replay_results.py --system <name> --replay <path>
   --expected-total <N>`。
6. `prompt_token_source` 或 `completion_token_source` 只要仍为
   `trace_expected`，正式阶段必须失败；这说明 token、TPOT 或成本诊断不再是
   后端真实观测/本地 guarded prompt 的结果。

当前 `SGLang`、`ServerlessLLM`、`vLLM`、`S-LoRA` wrapper 都必须在 replay
后、summary 前经过同一个 replay gate。`run_full_fair_round.sh` 的 summary
schema audit 是第二道保险，不能替代 wrapper 本地 gate。

已踩坑：

- vLLM chat template 错误导致所有请求 400，但 live 指标全 0。
- 如果不做成功数检查，summary 可能看起来像“系统极差”，实际是 serving 失败。
- vLLM OpenAI completion 可能返回 `HTTP 200` 但空输出、无 usage 和无
  first-token event；这类“空成功”必须在 replay gate 被拒绝，不能进入
  summary。

因此：

```text
all-zero metrics = first suspect serving/replay failure, not performance.
```

## 12. 日志与 tmux 规则

每个系统必须有：

- tmux session 名称。
- attach/switch 指令。
- server log。
- replay json。
- summary json。
- launch/deploy/fleet spec。

tmux 规则：

- 不要并发跑多个正式系统。
- 不要在 systemd closing session 中跑长任务。
- 如果当前 session `State=closing`，必须换新 SSH/TTY 或使用 user-scope runner。
- tmux 嵌套时用 `tmux switch-client`，非 tmux 中用 `tmux attach`。

日志排查顺序：

1. tmux capture-pane。
2. server log tail。
3. replay json 失败样本。
4. summary json。
5. launch/deploy/fleet spec。
6. shared trace / adapter subset。

## 13. 测试门槛

任何新系统 harness 在交给用户跑正式实验前，至少必须通过：

```text
bash -n scripts/*.sh
python3 -m py_compile replay / summary / compare scripts
dry-run launch spec generation
one small smoke if GPU 状态允许
summary schema smoke
compare script smoke
git diff --check
```

如果无法做真实 GPU smoke，必须明确说明原因，并至少完成 dry-run 与 schema
smoke。

## 14. 文档更新规则

每接入或修复一个系统，必须同步更新：

- `System_project/README.md`
- `System_project/docs/*`
- `docs/FAIR_COMPARISON_EXECUTION_PLAN.md`
- `docs/UPSTREAM_REPO_STATE.md`
- 必要时更新 `docs/CODEX_INTERACTION_RULES.md`

文档必须记录：

- 当前系统状态。
- 正式入口脚本。
- 复现边界。
- 指标能力。
- 已踩坑。
- 下一步动作。

## 15. Git 与回退规则

重要复现阶段必须形成可回退点。

提交前必须：

```text
git status
git diff --stat
bash -n
py_compile
git diff --check
```

禁止：

- `git add -A` 盲加结果文件。
- 提交无关模型、日志、大结果。
- 未经要求执行破坏性回滚。
- 把临时 debug 改动混进正式 harness。

## 16. 当前已知踩坑汇总

### vLLM

- Llama base tokenizer 没有 chat template，不能直接用 `/v1/chat/completions`。
- 正式路径应使用 `/v1/completions`。
- LoRA 选择必须是 `adapter_id -> model`。
- serverful lifecycle cost 必须计入 launch-to-ready。
- 失败请求必须阻断 summary。
- server `usage` 缺失时，必须用 prompt guard 后的本地 prompt 和实际响应文本
  计算 token；如果 token source 仍是 `trace_expected`，正式脚本必须失败。
- standalone streaming replay 必须请求 `stream_options.include_usage=true`，
  并对正输出 trace 设置 `min_tokens=1`，避免首步 EOS 造成不可观测 TTFT 的
  `200 OK` 空成功。

### SGLang

- LoRA source field 不能被 drop 后再读取。
- 正式路径使用顶层 `adapter_id`。
- 主表用 `DP=4, TP=1`，`TP=4` 只做附表。
- prompt guard 必须与 FaaSLoRA 对齐。

### S-LoRA

- 正式路径使用官方 `/generate_stream`，不能使用缺少 LoRA 选择语义的 chat endpoint。
- 请求映射必须是 `adapter_id -> lora_dir` 真实路径。
- 启动时必须用 `--lora-dirs` 注册 shared subset 中全部 LoRA。
- 默认公平拓扑为 `DP=4, TP=1`，按 serverful 静态 runtime 成本计费。
- 当前 harness 已通过 dry-run；`slora_official_cu118` 环境 preflight 已通过。
- `fix5` 已通过 `Llama-2 7B / 500 requests / 500 adapters` debug gate：
  `500/500` 成功，`metric_schema_version=e2e_v3`，无 `trace_expected`
  token fallback；下一步正式结论仍需跑 `4000 requests`。
- S-LoRA native guard 必须与服务端 `tokenizer.encode(prompt)` 语义一致，
  包括 special tokens；否则长 prompt 会在服务端被重新计成 `+1 token` 并触发
  `the input prompt token len ... is too long`。
- S-LoRA 环境必须使用 `nvidia/label/cuda-11.8.0`，禁止默认 `nvidia` channel
  混装 CUDA 12/13 组件。
- 真实结果必须先通过 shared-subset GPU replay 和 replay success/token-source gate。

### ServerlessLLM

- `SLLM_BACKEND=auto` 不能进入正式主表。
- vLLM backend probe 失败不能静默 fallback。
- deploy config 必须来自 shared adapter subset。
- 读超时过短会造成尾部失败，需要与正式 workload 对齐。

### FaaSLoRA

- FaaSLoRA 是主系统，baseline 复现不能破坏其环境或工件池。
- 任何优化必须对齐三大贡献：
  - LoRA readiness / preload
  - residency / local caching
  - serverless scale-out / admission / placement
- 不能只调指标或成本公式来替代系统优化。

### 成本

- token proxy cost 不能作为主成本。
- 主 CE 使用 monetary lifecycle `Cost/req`。
- `Cost/1MTok` 是补充指标。
- serverful 必须计入全生命周期 GPU-second。
- serverless idle-ready 可使用外部有据可查的 idle factor。

## 17. 最终可进入论文主表的检查清单

只有以下全部满足，结果才可以进入论文主表：

1. 同一 shared trace。
2. 同一 adapter subset。
3. 同一 model profile。
4. 同一 dataset/workload profile。
5. 同一 GPU budget。
6. 同一 metric schema。
7. 请求全部成功，或失败率按统一 deadline/service-rate 明确统计。
8. 没有 backend fallback。
9. 没有 base-model-only 误跑。
10. 没有全 0 summary。
11. 成本模型一致。
12. launch/deploy/fleet spec 可审计。
13. replay raw 可审计。
14. summary json 可审计。
15. compare 表可复现。
16. 文档已记录复现边界和已知限制。

如果任何一项不满足，必须先修复链路，不能进入结果分析。
