# EuroSys '27 V2 审稿意见处理与实验协议

更新时间：2026-07-22

适用分支：`retry14_continuous_queue_v2`

本文档是 PrimeLoRA/FaaSLoRA 针对 EuroSys '27 Spring Paper #907 审稿意见的 V2 工作入口。它规定新增实验如何运行、哪些结论可以写、哪些旧结果不得改变；它不是论文正文，也不授权覆盖历史结果。

## 1. 证据层级与使用边界

### 1.1 论文规范语义

V2 解释以投稿论文的设计目标为起点，但不把设计目标自动视为代码事实。论文提出三类机制：M1 命中感知放置与扩容准备（readiness-aware routing、NVMe preparation 和 scale-out handoff 是同一机制的组成部分）、M2 分层驻留与动态迁移、M3 协调准入。实现中仍保留细粒度 feature gates 供负向审计，但 Fig. 9 只按这三类论文机制构造四行累积消融。只有复合 gate 与相应在线触发证据同时成立时，才允许归因性能变化。

正式结论遵循以下顺序：

1. 先在 validation trace 上选择双方配置；
2. 冻结配置、trace、adapter subset 和生成契约；
3. 再运行 held-out seeds；
4. 按 seed 统计 paired difference 与置信区间；
5. 只报告门槛通过的完整运行，不按胜负筛选 seed 或系统配置。

运行器把 seed 41/42/43--45 分别机器标记为 `validation`、`smoke`、`heldout`。正式模式只接受 seed 41 的 1,000-request validation，或 seeds 43--45 的 4,000-request held-out；seed 42 只能是非正式 smoke。每个系统另保存排除 seed、trace/subset 和执行顺序的 resolved-config SHA，正式三个 held-out seed 必须一致。正式运行还要求 campaign manifest 为 `complete` 且代码来自 clean committed source；FaaSLoRA 只豁免用户已有的 `configs/generated/lora_manifest_1000.json` 修改。

### 1.2 当前实现证据

创建本文档时，代码已包含以下可核对的基础机制：

- [`InstanceSlot`](../../faaslora/experiment/instance_pool.py) 保存 per-slot GPU/HOST/NVMe tier hints、活动请求数、活动 adapter 引用、handoff 提示和完成请求的服务成本；`Router` 用这些状态构造 lexicographic routing key。
- [`ScenarioRunner`](../../scripts/run_all_experiments.py) 在实例选择并成功 reserve execution lane 后、LoRA path resolution 之前冻结 `readiness_tier_before_dispatch`，并把它写入请求级结果。
- [`ExperimentStack`](../../faaslora/experiment/experiment_stack.py) 连接 registry、residency manager、preloading manager、hotness tracker 和 resource coordinator。
- [`ResourceCoordinator`](../../faaslora/scheduling/resource_coordinator.py) 在候选 GPU promotion 时重新计算 effective capacity、KV 预测、load pressure 和实际 GPU memory pressure。
- [`PreloadingPlanner`](../../faaslora/preloading/preloading_planner.py) 对有界 candidate snapshot 使用 MiB 粒度 0--1 knapsack；过大问题回退到 greedy，因此不存在一般性的全局最优保证。
- V2 local-sim Remote->NVMe miss 使用“按实际目录字节在进程级共享链路上先 reserve、再 hardlink 落地”的统一语义；PrimeLoRA 与 ServerlessLLM-new 不再分别叠加不同的本地磁盘 copy 延迟。
- 正式单场景 V2 运行把 primary runtime 初始化开始至 preload 完成的 replay 前 GPU 占用计入 lifecycle/startup 成本，但不计入请求 TTFT/E2E；结果同时保留 preload wall、bytes、injected wait 和 GPU-s 以供 A4 分解。

同时必须保留以下审计边界：

- 当前 `ObservedRequestCost` 是请求完成时更新的算术均值，不是投稿文字所称的 EWMA。
- 当前 `CoordinationMetrics.queued_loads` 是累计进入等待路径的 load 次数；`InstanceSlot.load_queue_depth` 从该字段刷新，不能据此声称测得了瞬时 loading-queue depth。
- 单节点实验中的 HOST/NVMe 是 node-local shared caches；GPU hints 是 per-slot。全局 `ArtifactMetadata.storage_tier` 是单值元数据，不是强一致的多副本位置目录。
- `ResidencyManager` 的 GPU tier 是控制面 residency state；真正的 adapter executable state 最终由 backend 的 LoRA load/invocation path建立。V2 readiness 证据必须使用 pre-dispatch snapshot 和实际请求结果交叉验证，不能只用 registry 标签。

这些边界在本目录各专题文档中展开。V2 脚本变更完成后，应再次进行实现审计并更新“当前实现证据”，不能反向修改“论文规范语义”以掩盖差异。

本机拓扑预检还显示四张 RTX 3090 之间为 `NODE/SYS` PCIe 路径，`nvidia-smi topo -m` 没有 `NV#` 链路；论文主实验的每个 runtime 为 TP=1 单 GPU。因此 V2 可以客观说明当前主机制实验没有使用跨 GPU tensor-parallel/NVLink 数据通路，但不能据此推断 H100 或更大集群的绝对性能。原始预检保存在 `paper_results/eurosys27_v2/preservation/hardware_topology_preflight.txt`。

## 2. 审稿意见 action matrix

| 编号 | 决定 | V2 交付物 | 可写结论的门槛 |
|---|---|---|---|
| A2：收益是否主要来自 elasticity | 采纳 | 按论文三类机制做四组累积消融：`ElasticOnly -> +HitAwarePreparation -> +HierarchicalResidency -> Full(+CoordinatedAdmission)` | 变体只差目标机制 gate；对应触发计数与开关一致；每个 7B 增量有 3 个 held-out seeds |
| A3：Fig. 9 解读 | 采纳 | 新建 V2 Fig. 9，显示绝对值、常规方向的相对差值和关键点 95% CI | lower-is-better 使用 `(reference-value)/reference`；不覆盖旧 Fig. 9 |
| A4：CE 合理性 | 保留 CE，补充诊断 | CE 分解、Pareto、GPU-s/request、SLO-goodput/$、cost/token、参数/idle-factor sensitivity | 离线计算保留原始指标；不把单一 CE 当作完整 trade-off |
| A5：dLoRA、Chameleon、ELORA | 部分采纳 | V2 新实验只使用一个 `ServerlessLLM-new` 行；其他系统给出可复现性记录 | 不自行重写论文核心；不把 3B-only 或 gate-only 结果包装成 3B+7B 正式基线 |
| A6：测试床、网络与规模 | 部分采纳 | 真实设备附近的 aggregate bandwidth sweep 和 fetch microtest | 明示应用层限速；不称为物理 100/400GbE；`no-delay` 只称乐观上界 |
| B1：场景与算法是否完整 | 论文补充 | [B1/B2 设计澄清](B1_B2_DESIGN_CLARIFICATION.md) 的状态机、伪代码和场景覆盖表 | 论文规范语义与当前实现证据分列；不靠本轮代码改动回避差异 |
| B2：budget、位置、最优性、动态性 | 论文补充 | 同上 | 明确 operator/runtime 各自责任；只声明预算可行性，不声明全局最优 |
| B4：GPU-ready/transition/EWMA | 本轮不处理 | 无 | 不新增实验或专门改写 |
| C1：queue/affinity signals | 文档采纳 | [queue/affinity 状态说明](C1_QUEUE_AFFINITY_STATE.md) | 明确存储、owner、event-driven/sampled 更新和 stale fallback |
| C2：active-LoRA feasibility | 文档采纳 | [active-LoRA feasibility 定义](C2_ACTIVE_LORA_FEASIBILITY.md) | 与 residency、request concurrency、queue length 严格区分 |
| C3：为何 admission 时再查 budget | 文档采纳 | [plan 与 admission 分层说明](C3_PLAN_VS_ADMISSION.md) | planner 只选择价值；admission 对 momentary feasibility 重新检查 |
| C4：500 adapters/rotation 500 的代表性 | 采纳 | 保留投稿的 legacy-overlap 主点，并增加 stationary、abrupt、Zipf、gradual-overlap sensitivity | 同一 arrival/token trace；报告实际 unique adapters 与 churn 指标 |
| C5：Prime E2E 优于 S-LoRA | 重新验证 | `fixed_length_greedy_v1` matched-output 对比及 E2E 阶段分解 | 每请求 token/prompt 契约一致，4,000/4,000 成功，无 token fallback |
| Readiness 是否真是 dispatch-time | 主动修正证据 | pre-dispatch tier audit 与 multi-cycle scale-out diagnostic | tier 字段完整率 100%，不变量冲突 0，first-service 样本满足预注册门槛 |
| 新 GPU/NVLink | 部分回应 | runtime topology/NCCL/NVLink smoke metadata | 只说明实测拓扑；不推断 H100 性能 |

基线纳入理由和证据位置见 [BASELINE_REPRODUCTION_NOTES.md](BASELINE_REPRODUCTION_NOTES.md)。

## 3. 冻结的正式实验协议

### 3.1 Worktree 与数据保护

- 正式工作只使用 `/home/qhq/serverless_llm_experiment_retry14_baseline` 的 `retry14_continuous_queue_v2` 分支。
- 创建本文档前已存在用户修改 `configs/generated/lora_manifest_1000.json`；任何 V2 commit 都不得 stage 或修改该文件。
- 旧 `paper_results/final_v2/`、旧 `figs/paper/`、投稿 7B/3B local frozen/local-sim 结果保持只读。
- 新原始结果使用唯一 `campaign_id/model/seed/run_tag`，写入 gitignored `results/eurosys27_v2/` 或 baseline harness 的新 round；已存在 run-key 不原地重跑。
- 新汇总写入 `paper_results/eurosys27_v2/<topic>/`，新图写入 `figs/eurosys27_v2/<topic>/`。
- 每个 campaign 保存有效配置、trace/subset hash、source revision、环境/拓扑、完成状态和原始结果 hash。

### 3.2 调参与统计

- validation：seed 41、1,000 requests；只在此处选择 PrimeLoRA 和对手配置。
- compatibility smoke：seed 42；只检查接口和旧结果兼容，不作为新结论的 held-out 样本。
- formal held-out：seed 43、44、45；配置冻结后不得按正式结果改动。
- 完整 sensitivity matrix 先跑 seed 43；支撑主结论的端点再跑 seed 44/45。
- 统计单位是 seed。对同 seed、同 workload 的系统计算 paired difference；3 seeds 给出均值与双侧 95% Student-t CI，同时保留每个 seed 的原值。
- 4,000 个请求只用于计算单次运行内的 latency distribution，不把请求当作 4,000 次独立实验重复。
- 若正式结果不支持预期结论，只能：检查预注册门槛；修复共同的测量/语义 bug；回到 validation trace 对双方执行同等调参；冻结新版本后完整重跑 43/44/45。禁止只换 PrimeLoRA、筛 seed 或隐去失败结果。

Standalone A2/A3 消融运行器把成功的 seed41 `v2_full` validation manifest
登记到 family-scoped、加锁原子写入的 registry。登记前会核对 source commit、配置文件
SHA、1,000-request trace、场景集合和 non-feature frozen hash；held-out 启动前必须从
registry 解析唯一候选，随后把 per-round immutable evidence 写入
`protocol/seed41_validation_evidence.json`。每个 family 的第一个成功正式 seed41 manifest
即为唯一选定记录；第二个不同 hash（或不同 manifest）的正式 validation 会被拒绝。
所有候选调参只能使用 non-formal scratch rounds/registry，不能在看到多个正式 seed41
结果后再选择。已冻结 family 不能切换 hash。formal
analyzer 会再次核对 evidence、seed41 manifest、validation 原始结果及其 byte/SHA，
不会只信 held-out manifest 的摘要。该链只冻结非机制配置；四行机制 gate 仍由场景定义
和触发计数独立审计。

PrimeLoRA Full 与 ServerlessLLM-new 的正式主比较由
`scripts/analyze_v2_full_vs_serverless.py` 单独发布。它只接受 completed held-out
campaign manifests，并强制 2 models x 3 seeds x 2 systems 的 12 个 identity、
`v2_full`/`serverlessllm_fair` 官方场景、上述投稿 legacy workload 轴、共享
trace/subset SHA 以及跨 seed frozen-config 一致性；输出 per-run 原值、同 seed
paired difference、方向统一的 improvement 和 95% t-CI。`serverlessllm_fair`
是 ServerlessLLM-new 官方 harness 保留的结果-schema 场景名，不代表同时保留了
一个旧 ServerlessLLM 比较行。

### 3.3 A2/A3 累积消融

主点严格冻结为 7B、4,000 requests、500-adapter universe、local-sim
250 MiB/s、time-scale 8 和投稿真实主 workload：Zipf 1.0、active-adapter
cap 48、每 500 请求轮换、`rotation_mode=legacy`、相邻生成器 hot set
目标 overlap 0.75，generation contract 为 `legacy`。四行必须记录同一个
`non_feature_frozen_config_sha256`；该摘要排除下列机制开关，但覆盖其余 runtime、
autoscaler、billing 与 tuning 配置；同时排除 seed、request count、trace、bandwidth
和 workload sensitivity 轴，formal analyzer 遇到跨行漂移即拒绝发布。被摘要排除的
实验轴仍由 formal matrix 逐项精确检查，不能借摘要排除而漂移。

1. `ElasticOnly`：保留同一 autoscaler、runtime、billing 和 backend 按需 LoRA load；least-loaded routing；关闭 PrimeLoRA 的三类机制。
2. `+HitAwarePreparation`：加入机制一，即 readiness-aware placement、NVMe preparation 与 scale-out handoff。三者共同构成论文中的“命中感知放置与扩容准备”，不再拆成多个伪机制行。
3. `+HierarchicalResidency`：在机制一之上加入机制二，即 GPU/HOST/NVMe 分层驻留与动态迁移；不启用协调准入。
4. `Full(+CoordinatedAdmission)`：加入机制三，即 effective-capacity coordinated GPU admission。

运行矩阵：

- 7B seed 43/44/45：四个变体全部运行，以 seed-level paired difference 为每个机制增量计算均值和 95% t-CI。
- 3B seed 43/44/45：`ElasticOnly`、`Full`。
- 所有变体共享 Full 的 frozen non-feature 配置，不为每个变体单独调参。

机制触发门槛：

- `ElasticOnly`：readiness-aware routing selection、handoff、tier-promotion 和 admission 触发必须为 0；为测量 dispatch-time readiness 而在选定 replica 后冻结 tier 的 instrumentation 仍保留，但不参与排序。
- `+HitAwarePreparation`：readiness lookup/route decision、planned adapters、实际 warmup 和 first-service 计数可对齐；HOST/GPU 动态迁移必须仍为 0。
- `+HierarchicalResidency`：在上一行门槛之外，HOST/NVMe population 与 online HOST/GPU promotion 或 tier transition 计数大于 0。
- `Full`：GPU admission decision 计数大于 0，并同时报告 admit/defer/reject。

Full 的 CE、TTFT 与 E2E 优势只允许在 seed 41 validation trace 上通过公平参数搜索争取；配置冻结后，seed 43/44/45 的任何结果都进入汇总，不能以“Full 未胜”作为删除或筛选 run 的理由。

Fig. 9 四个主面板为 P95 TTFT、平均 E2E、Cost/request 和 CE；数据表另含平均 TTFT、TPOT、throughput 和触发次数。若增量 CI 跨 0，论文写作使用“not statistically distinguishable under this testbed”，不写成确定性改进。

### 3.4 A4 CE 补充包

优先离线复用旧主结果和新 held-out 结果，生成：

- latency、cost、throughput、CE 原值；
- GPU-s/request 及 active/startup/idle-ready GPU seconds；
- SLO attainment、SLO-goodput/$、cost per 1M output/total tokens；
- cost--latency Pareto；
- `Delta ln(CE)` 的 latency/cost 对数贡献；
- `CE_(alpha,beta)=1/(L^alpha C^beta)`，`alpha,beta in {0.5,1,2}`；
- idle billing factor `{0, 0.238095, 0.5, 0.75, 1.0}` 及相对基线的 break-even factor。

除非生命周期字段不足以精确重算，否则不为价格 sensitivity 重跑 GPU。该包只进入 `paper_results/eurosys27_v2/a4_ce/`，是否写入论文另行决定。

### 3.5 A6 带宽 sensitivity

正式 sweep 的配置速率均为 aggregate application-layer MiB/s：

| 配置 MiB/s | 展示标签 |
|---:|---:|
| 11.9209 | 0.1 Gbit/s |
| 29.8023 | 0.25 Gbit/s |
| 59.6046 | 0.5 Gbit/s |
| 119.2093 | 1.0 Gbit/s |
| 250 | 2.097 Gbit/s |
| no-delay | local-sim 乐观上界 |

- 7B seed 43 跑全部六点；11.9209、119.2093、no-delay 再跑 44/45。
- 3B 只跑上述三个关键点的 seed 43 锚点。
- PrimeLoRA 与 ServerlessLLM-new 使用同一 trace、adapter subset、cold cache 与限速语义。
- 另做四并发 fetch microtest：小/中/大 adapter，各三次；记录 configured/achieved MiB/s、总 bytes、wall time 和 injected wait。
- microtest 的必要条件为 `wall_time >= total_MiB/configured_MiB_s`，允许调度误差使实际更慢，不允许 aggregate throughput 系统性高于配置速率。
- 实际 frozen pool 审计显示 7B 只有两档 adapter 目录大小；因此三档 microtest 从同机 3B frozen pool 中按实际目录字节数确定性选择 26.718/36.468/53.718 MiB 三组（每组四个不同 adapter），而不是把同尺寸文件重命名为“小/中/大”。原始结果保存在 `paper_results/eurosys27_v2/a6_bandwidth/microtest_1gbit_actual_adapters.json`。

论文只声明“在本机 1GbE 附近及更低带宽的等价应用层限制下的敏感性”；不把 250 MiB/s 或 no-delay 称为物理 100/400GbE。

### 3.6 Dispatch-time readiness

- 主证据使用请求级 `readiness_tier_before_dispatch`，不使用事后 `cache_tier` 代理。
- diagnostic 为 4,000 请求、8 phases x 500；phase 间 idle 2s，并使用现有 scale-down/scale-up 路径。
- 7B：`ElasticOnly`、`+HierarchicalResidency(no coordination)`、`Full`；3B：`Full`。
- 必要门槛：4,000/4,000 成功；dispatch tier 完整率 100%；tier invariant conflict 0；每个 model/variant 至少 8 个 scale-up events；first-service 样本至少 20。
- 若单次 first-service 少于 20，只允许增加一次完全相同配置的独立 cold run并合并；不得改样本定义或阈值。

### 3.7 C4 workload sensitivity

固定 7B、4,000 请求、500-adapter universe、arrival/token trace，测试：

1. stationary Zipf 1.0，无轮换；
2. abrupt rotation 100；
3. `submitted_main_legacy_rot500_overlap75`：投稿主点，legacy rotation 500、目标 overlap 0.75；
4. `abrupt_rot500_overlap0`：额外的完全突变 sensitivity 点，不冒充投稿主点；
5. abrupt rotation 2000；
6. Zipf 0.6、abrupt rotation 500；
7. Zipf 1.4、abrupt rotation 500；
8. gradual rotation 500、相邻 hot set 50% overlap。

Formal C4 矩阵共有 37 个 identity。Full 与 ServerlessLLM-new 在八种
profile 上运行 seed 43；stationary、rotation 100、投稿 legacy-rotation-500
主点再运行 44/45，并在这三个关键 profile 加入 ElasticOnly seed 43/44/45。
单独的 abrupt-500/overlap-0 只运行 seed 43。

`final_v2` 投稿主 trace 的实际请求序列并不是相邻窗口完全不重叠：按 500
请求划窗后，相邻窗口中“实际出现的 adapter 集合”的 Jaccard similarity 约为
0.55--0.61（对应 observed-set turnover 约 0.39--0.45，均值约 0.416）。因此旧
主点必须称为 `legacy rotation / overlap 0.75`，不得改称 abrupt。报告名义
universe 以外，还必须报告 actual unique adapters、effective adapter count、
entropy/Gini、first-touch ratio、reuse-distance quantiles、基于实际观察 adapter
集合计算的相邻窗口 turnover，以及 remote miss ratio；该 observed-set turnover
不冒充生成器内部 hot-set ground truth。

### 3.8 C5 PrimeLoRA--S-LoRA matched-output

`fixed_length_greedy_v1` 契约：

- `target_i=min(source_expected_output_tokens_i,256)`；
- `temperature=0`、`top_p=1`、`ignore_eos=true`、无 stop sequence；
- 相同 canonical prompt，prompt content 上限 759 tokens，并逐请求保存 prompt hash；
- S-LoRA 从 native SSE 的整数 `token.id` 计数；PrimeLoRA 从 vLLM `token_ids` 计数；文本重分词不能作为主 token count。

先用 seed 42、每模型每系统 100 请求 smoke；正式运行 7B/3B x seed 43/44/45 x Prime/S-LoRA，共 12 个 run，并按 seed 交替系统运行顺序。

硬门槛：

- 每 run 4,000/4,000 成功且无 fallback；
- 每请求 `actual_tokens == target_tokens`；
- 同 seed 的 request/adapter/arrival/target map/prompt hash 完全一致；
- E2E 分解恒等式和 TPOT 重算误差不超过 1 ms；
- 输出 dispatch/admission、service TTFT、decode 三段，以及 token throughput、cost 和 CE。

## 4. 产物与发布验收

新目录固定为：

```text
results/eurosys27_v2/<campaign_id>/<model>/<seed>/<run_tag>/
paper_results/eurosys27_v2/{full_vs_serverless,a2_a3_ablation,a4_ce,a6_bandwidth,c4_workload,c5_slora,readiness}/
figs/eurosys27_v2/{full_vs_serverless,a2_a3_ablation,a4_ce,a6_bandwidth,c4_workload,c5_slora,readiness}/
```

发布前必须：

1. 运行 GPU idle/cleanup gate；失败 run 保留原始记录但不混入汇总。
2. 校验 result completion、trace/subset hash、generation contract 和 dispatch-tier completeness。
3. 对旧 `paper_results/final_v2/` 与旧 `figs/paper/` 比较前后 hash，必须零变化。
4. 记录 `nvidia-smi topo -m`、runtime visible GPU、TP/DP、NCCL/NVLink 计数；只报告实测。
5. 执行 `git diff --check`、内部 Markdown link 检查、bundle checksum 和 secrets scan。
6. 明确排除用户的 `configs/generated/lora_manifest_1000.json` 修改。

## 5. 本轮明确不做

- 不改投稿 LaTeX/PDF。
- 不处理 B4。
- 不做 13B。
- 不把应用层限速称为物理 100/400GbE。
- 不模拟 H100 性能。
- 不从论文重新实现 dLoRA、Chameleon 或 ELORA 的核心机制。
- 不覆盖旧结果或旧图。
- 不按正式结果胜负选择配置、seed 或报告项。

## 6. 可直接用于英文论文的实验方法段落

> **Revision protocol.** We preserve the submitted 3B and 7B local-simulation results and report all revision experiments as a separate, non-overwriting evidence set. We tune each system only on a 1,000-request validation trace (seed 41), freeze the resulting configuration, and evaluate it on three held-out traces (seeds 43--45). A full sensitivity matrix is screened on seed 43, while endpoints that support a main claim are repeated on all three held-out seeds. We treat the trace seed, rather than individual requests within a replay, as the unit of replication and report paired per-seed differences with 95% Student-t confidence intervals. Failed or contract-violating runs are retained for audit but excluded by predeclared validity gates, not by their relative performance.

> **Artifact discipline.** Every revision run records the resolved configuration, source revision, trace and adapter-subset hashes, hardware topology, completion status, and checksums of its raw inputs and outputs. Revision data and figures are written to new versioned directories; the submitted result bundle and figures remain immutable. This protocol lets multiple analyses reuse one validated run without rerunning or selectively replacing historical measurements.
