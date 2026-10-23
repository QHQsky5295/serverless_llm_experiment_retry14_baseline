# dLoRA 3B：历史复用与必要计量适配核查

日期：2026-10-23。此项是在 D773 S-LoRA 3B W2 运行期间进行的轻量只读核查，
不是新增 GPU 实验，也没有修改任何被测源码、配置或远端服务。下一主线仍为
完成 D773 后的 dLoRA 限定 3B W1/W2；不新增 dLoRA 7B 或性能参数搜索。

采用 analyze-results 的“原始观测—解释—研究含义—下一步”结构；历史合同
与当前合同不同，不能计算系统优劣的相对改善。最新单轮 V2 覆盖旧重复/CI规则。
当前指标依据：冻结 V1 SHA
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
单轮 V2 SHA
`193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

## 1. 原始记录及可复用资产

基线仓库：`/home/qhq/serverless_llm_baselines`，下称 BASE。
历史登记：`BASE/DLoRA_project/evidence/formal_period_mig_full4000_3b_2026-05-21.json`。
历史 run ID：`20260521_dlora_remote_mig3_full4000_g2u92_s4_v1`。

完整结果目录：

```text
BASE/results/paper_experiments/15_new_serverless_baselines_remote_v1/
20260521_dlora_remote_mig3_full4000_g2u92_s4_v1_llama32_3b_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_dlora/
```

该目录的 `MANIFEST.json`、`raw/replay/*_replay.json` 和 `*_summary.json`
保持原样。原生服务日志：

```text
BASE/results/logs/new_serverless_baselines_remote_v1/dlora/formal/
20260521_dlora_remote_mig3_full4000_g2u92_s4_v1/llama32_3b_server.log
```

可复用源码：`BASE/vendor_new_baselines/dLoRA_artifact_main_20260519`，官方
commit `75f1c439446fe194b1df8a24982ef9067841fab5`。历史 MANIFEST 中实际环境为
`dlora_medusa_clone_20260519`，不是仅凭目录名称推测的环境；下一步核对现有
环境及实际导入路径，不先安装或重建。原模型、500-adapter 池、源 trace、已有
W1/W2 参数视图和远端 D78/D80 发布对象继续复用，不生成第二套权重或负载。

历史配置：DP2/TP1、GPU 0/1、`migration_type=3`（PERIOD_MIG）、
`gpu_memory_utilization=0.92`、`max_num_seqs=4`、
`max_num_batched_tokens=1024`、`gpu_capacity=8`、`swap_space_gb=8`。
旧 manifest 的 Ray object-store/CPU 值为 null；不能解释成新协议已核验的
8 GiB 合计对象存储或共同 CPU 隔离。实际迁移事件也不能由配置枚举代替。

## 2. 原始观测及限制

| 项目 | 历史原始记录 | 当前结论 |
|---|---:|---|
| 返回成功/失败 | 4000/0 | 支持旧合同完整完成，不自动支持新合同 |
| 输出 token 总数 | 458533 | 与当前固定生成目标 458224 不同 |
| 计量版本 | e2e_v3 | 不是当前原生时钟协议 |
| 每请求 stream_event_count | 全部 4000 条均为 1 | 客户端没有逐步输出观测 |
| tpot_observed=true | 0/4000 | 数字型 TPOT 不代表实际观测 |
| 非空 server_metrics | 0/4000 | 不能从这些记录恢复原生首末 token 时间 |
| 旧平均“TTFT” | 11162.4256 ms | 接近完成返回时间，不能直接作为新 TTFT |
| 旧平均 E2E | 11162.7033 ms | 保留旧语义，不跨合同排名 |
| 旧 GPU 时间字段 | 9445.63637 GPU-s | 尚未证明满足当前实际 allocation/release 积分 |

以上 stream/TPOT/server_metrics 计数直接由约 9 MB 的旧 replay JSON 统计，
不是只引用 README。第一条记录的 `tpot_ms=0.0036368542620665487`，但同时
`tpot_observed=false`；因此不能把该数用于共同 TPOT 达成率。原生 usage
计数来自输出 token IDs 的长度，值得复用其计数路径，但数量来源正确不等于
生成合同一致、时间正确或独立数值正确性已经证明。

服务日志中存在 `tot_time`、`exec_time`、`wait_time` 和
`wait_time_per_output_token`。源码 `llm_engine.py` 在请求终止时打印这些
完成态统计；其中等待均摊值不是首末 token 间隔，不能改名为 TPOT。本次
定向检索未发现 first_token/last_token/token_ts/token_timestamp 原生日志。
这是一项已查文件范围内的结论，不冒称整个历史存储绝无其他可恢复证据。

## 3. 原因与官方来源

当前源码 `vllm/engine/async_llm_engine.py` 的
`RequestTracker.process_request_output` 只在 `request_output.finished` 时
把输出送入请求流。`engine_step` 虽然逐次取得原生 RequestOutput，未完成
输出没有进入该返回流。`vllm/entrypoints/api_server.py` 的本地兼容层返回
text/usage/request_id/finished，没有原生 token 时间字段。

已核对固定官方 commit 的同一条件：
[dLoRA 官方 RequestTracker 源码](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/async_llm_engine.py)。
因此，单次返回不是仅由本地 HTTP 回放器漏记流式 chunk 造成；不能简单调整
客户端统计就宣称恢复了 TTFT/TPOT。这也不证明 dLoRA 的调度/迁移算法存在
性能缺陷，更不能用旧完成态延迟支持 PrimeLoRA 的首 token 优势。

## 4. 复用裁决与下一步边界

1. 对当前 G1/G2 和 W1/W2，旧完整运行归为 **R2**：复用资产、环境、兼容经验
   与诊断，不混入新主表。旧 W0、输出总量和资源/准备合同均不同，不能仅靠
   改分析器成为 R0/R1。保留旧结果及其当时身份，不原地改数字。
2. 继续使用已有 DP2/TP1 路径，不重复 DP4/TP1 已知主机 OOM；不因旧 TP2
   更慢再开拓扑搜索，不重开已关闭的 7B 零 KV-block 方向。
3. 必要适配集中在原生首末 token/实际 IDs、canonical 输入与固定输出合同、
   请求/adapter 绑定、物理 GPU 生命周期及共同资源限制。原生周期迁移、
   调度、放置和 merge/unmerge 机制必须保留并记录实际触发范围。
4. 测量需在原生生成事件处取得时钟，并跨请求迁移保持同一请求身份；不能
   将最后一次 HTTP 返回当作首 token，也不能把迁移后局部首 token 重置为
   全局首 token。优先检查既有计量适配可复用部分，再做受影响路径的最小验证。
5. 旧 wrapper 含无项目身份限定的 `ray stop --force`、`RAY_ADDRESS=auto`，
   不能原样运行。复用现有受限服务/外置回放和精确所有权收尾接口，不触碰
   其他项目 Ray/k3s，也不以限制启动 CLI 冒充限制真实 workers。
6. 历史 `formal_500_adapter_runtime_compat_20260520.patch` 实际涉及
   `engine_manager.py` 初始化放置，不能只引用旧“wrapper-only”说明。
   下一步逐处核对该补丁和原生事件，明确必要适配范围；尚未完成此项核验，
   不提前宣称所有核心路径已验证。保留旧补丁，不能静默覆盖历史来源。
7. D773 完整结束、资源释放、结果封存和重复缓存回收后才进入适配执行。
   不因本次记录而新增无问题的重复资格、不扫描性能参数、不扩大比较范围。
   新合同的 W1/W2 必须各自完整执行，不能用旧 W0 替代。

本次只新增本记录并更新执行状态；大型哈希、环境导入、构建、性能实验、
清理和 Git 备份均不与 D773 性能测量并行。后续正常检查点将本记录一并备份。

## 5. 既有适配与实际机制的补充核查（同日，D773运行中）

已完整读取两份历史补丁，以及当前 `engine_manager.py`、原生 LoRA 容量分配、
请求流迁移和输出结构。未执行环境导入、数值测试或源码修改；本节是静态核查
加已有运行事件，不把它写成新的资格通过。

| 路径 | 具体观察 | 复用/适配边界 |
|---|---|---|
| PEFT加载 | 明确加载q/k/v/o、alpha/r缩放和GQA形状；未加载的MLP LoRA初始化为零 | 复用已有加载器及历史切片核验；仍须与冻结池target modules对应，不把切片通过升级为完整数值证明 |
| selected-column einsum | 仅去掉当前adapter_mapping全零的列 | 对有限输入在代数上保持求和对象；FP16/merge后的数值证据仍独立，不用代数说明替代原生核验 |
| 放置评分 | list成员查询改为同步set查询，评分表达式相同 | 不是另一个放置目标；需保持list/set同步和tie-break |
| 放置终止 | 无成功更新时不再仅因存在未分配项无限重试；初始随机adapter遍历所有引擎后可推进到下一adapter | 实际改变无进展/卡住边界，不能标为纯wrapper；合理性是终止性适配，不宣称所有输入逐步等价 |
| 周期迁移 | 原生周期、KV/负载触发、ILP和请求流迁移路径保留 | 参数值不代表实际触发；下述旧事件可复用为历史机制证据 |

### 5.1 生效GPU容量不是命令行的8

旧服务日志第65行记录：`num_lora_replicas 501 engine_lora_counts [251, 250]`；
第66–67行分别为1143、1173个GPU KV blocks。当前原生
`LlamaLoRaEngine.allocate_gpu_lora_weight` 用放置列表长度重新赋值
`lora_config.gpu_capacity`，然后分配相应张量。
固定[官方LoRA源码](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/lora_engine.py)
也有相同容量更新。

因此，旧 manifest 的 `gpu_capacity=8` 是输入参数，不是该运行实际GPU槽位
上限。不能据此解释为“只缓存8个adapter”；下轮保留原生放置语义，记录
实际放置、张量容量与KV数量，不人为强制裁成8个来降低资源或改变机制。

### 5.2 旧迁移并非只有配置标志

同一旧服务日志第2966–2971行包含：KV满触发组0与组1平衡、41条请求/2组/
500模型、ILP求解约0.369267秒、1085变量/1047约束，以及
`move 2 requests from 0 to 1`。这支持旧运行实际进入求解与迁移决策分支，
不能再说“没有任何迁移证据”。但打印发生在fetch/insert之前，没有这两条
请求的完整关联回执时，不能把它升级成迁移完成及数值连续性已证明。

当前 `pop_requests`/`add_requests` 搬移的是同一个 AsyncStream 对象；序列
数据另经fetch/deepcopy/insert。这为最小计量提供现成所有权载体：在原生
RequestOutput的token累计量变化时记录首末观测，随同一请求流保留，不在
迁移后重置首token。正式实现仍需核对事件边界/时钟与取消的实际owner；
不通过替换调度或关闭迁移让计量更容易，不默认HTTP完成时间就是token时间。

### 5.3 不扩大为基线优化

对照[固定官方EngineManager](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/engine_manager.py)
和本地补丁后，后续只核查变化路径的终止性、输入/adapter绑定、迁移事件连续性
及资源归属。保留原生算法常量和已选DP2/TP1配置，不启动新的吞吐/副本/迁移
参数搜索。旧加载probe位于 `BASE/scripts/probe_dlora_real_adapter_loader.py`，
先复用它和现有TC运行/计量接口；不能原样启动带全局Ray清理的旧wrapper。

独立数值正确性、当前环境可运行性、实际共同80GiB包络和新合同W1/W2仍未
验证。本节收窄需要适配的具体范围，不将这些未知补成通过，也不延误当前
D773完整回放、封存和后续主线。

## 6. 旧加载 probe 的实际覆盖范围（2026-10-24只读核查）

完整读取 `BASE/scripts/probe_dlora_real_adapter_loader.py` 后，不能将
`GATE_2026-05-20_REAL_ADAPT.md` 中“q/k/v/o均已核对”的描述直接作为该脚本
覆盖证明。本次未执行probe，也尚未核对其历史原始JSON；以下是当前脚本
明示的检查边界，不是否定旧加载器或推断旧运行发生错误。

| 检查对象 | 当前脚本实际实现 | 可支持的范围 |
|---|---|---|
| 输入集合 | `--limit`默认2，并调用全部所选adapter的加载 | 加载所选集合，不等于逐adapter比值核验 |
| 权重比较对象 | `adapter_paths[0]`和`engine.cpu_lora_weights[0]` | 只比较第一个adapter |
| 层及模块 | `layers.0`的q_proj、o_proj A/B，B含alpha/r缩放 | 第一层q/o的所测切片 |
| 其他模块 | 检查所选第一层gate/up及down缓冲非零数为0 | 不是k/v加载核验，也不是推理数值参考 |
| passed判定 | 四个q/o最大差为0且两个MLP非零计数为0 | 不证明全部层、GQA k/v、全部adapter或迁移后输出 |

后续复用该probe骨架，必要时补其缺失的所选adapter/层/声明target模块
及GQA切片检查，不重新生成权重，不为已有未改变路径重建整套资格流程。
独立数值正确性仍与权重搬运、native token数量、完整HTTP成功分别记录；
计量适配也不借此替换dLoRA的放置/迁移策略。D773仍优先完整收尾，本节
没有启动新资格或改变在测服务源码。
