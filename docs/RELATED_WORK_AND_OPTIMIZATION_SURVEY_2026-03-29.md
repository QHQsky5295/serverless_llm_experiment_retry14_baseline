# FaaSLoRA 相关工作与优化调研记录（2026-03-29）

本文档记录 2026-03-29 这轮围绕 `FaaSLoRA` 的机器化文献调研、实验可比性判断、差距归因，以及下一阶段 TODO 排序建议。

用途：

- 固化本轮调研结论，避免以后重复搜索与重复判断
- 为后续论文 related work、motivation、实验对比与系统优化提供一手材料
- 为新会话提供“为什么下一步先做什么、不该做什么”的权威入口

若本文与当前代码实现、`docs/PROJECT_PROGRESS.md`、`docs/SESSION_HANDOFF_2026-03-13.md` 冲突，以后两者和当前代码为准；本文主要负责记录调研事实、可比性边界和规划建议。

> 2026-04-12 补充：
>
> - 当前 active mainline 仍是 `retry14_continuous_queue_v2`。
> - 当前 7B 可信 checkpoint 仍是 `retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`。
> - 当前 14B 正式负载最新可信结果已推进到 `retry14_continuous_queue_v2_qwen14b_r500_a500_main_baseline45_poststartup_elapsed @ 500`。
> - 当前新结论与本文调研逻辑是对齐的：
>   - 14B bring-up 与正式负载都表明，命中感知的扩容首批接管机制并没有停留在单一 7B 场景
>   - `scaleup_first_service_planned_match_rate = 1.0` 与 `gpu_hit_rate = 1.0` 说明第一项贡献已跨到更大模型
>   - 剩余问题已从“首批接管预测错误”转向“后续接管请求的层级驻留与资源协同不足”，也就是更接近第二、第三项贡献
> - 因此，本文第 6 节 TODO 排序的当前正式理解应更新为：
>   - 7B：保持 soft-close
>   - 14B：第一项贡献对应的 handoff/control 语义链可软收口
>   - 下一条 active 主线：`Mistral 7B V2 publicmix @ 500 adapters`
>   - 如果另一模型家族复现同样的后续接管冷路径问题，再回到跨模型共通的 C2/C3 主线继续收口

> 2026-04-11 补充：
>
> - 当前 active mainline 仍是 `retry14_continuous_queue_v2`。
> - 当前最新已正式分析、且最可信的 7B checkpoint 是 `retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`。
> - 当前正式判断进一步收紧为：
>   - `baseline35` 不能再被当成真实上界，因为其部分优势来自更乐观的 `bootstrap / ready-time` 语义
>   - `baseline44` 已经把最近几轮由语义错位和并发 fan-out 带来的结构性回归收回
>   - 7B 当前更合理的工程状态是“可信 checkpoint + 准备做跨模型迁移验证”，而不是“继续在单一 7B 上无限刷局部最好数字”
> - 这一步与本文调研结论一致：
>   - 同类论文不会把某个单模型单机型上的一次局部最优，当成系统已经方法学收口的唯一证据
>   - 更合理的下一步是转向 `14B` 和另一模型家族，验证三项贡献是否具备跨模型可迁移性
> - 如果后续重开 7B，当前最符合本文调研逻辑的重开方向不再是 handoff exactness，而是第三项贡献中的：
>   - `warm pool retention`
>   - `受控资源保留`
>   - 也就是进一步降低真实物理冷启动带来的代价
> - 因此，本文第 6 节 TODO 排序的当前正式理解应更新为：
>   - TODO `#2R`：不需要重开
>   - 7B 主线：当前可 soft-close
>   - 下一条 active 主线：`Qwen 14B TP=2`
>   - 紧接着：另一模型家族的迁移验证
>   - TODO `#4/#5`：继续后置

> 2026-04-09 补充：
>
> - 当前 active mainline 仍是 `retry14_continuous_queue_v2`。
> - 当前最新已正式分析、且属于 `continuous_queue_v2` 主线的 7B 有效结果是 `retry14_continuous_queue_v2_qwen7b_r500_baseline34_multiruntime_routeaware @ 500`。
> - 当前正式判断已经从“先把 7B 某个单点数字刷到最低”收紧为：
>   - 7B 当前已达到一个可冻结的 soft-close checkpoint
>   - 它说明本文调研所强调的 `continuous online queue + handoff-aware scale-up` 大方向是对的
>   - 但尚未证明 7B 上的 TODO `#3` 在所有 headline 指标上完全收口
> - 最新结果信号是：
>   - `baseline34` 相比 `baseline33` 已改善 `TTFT_scaleup_affected / TTFT_scaleup_first_service / TPOT / E2E / tok/s`
>   - 但相对 `baseline30` 仍未完全拿回 `TTFT_overall / TTFT_comparable / GPU_hit_rate`
> - 因此，本文第 6 节 TODO 排序的当前正式理解应更新为：
>   - TODO `#2R`：不需要重开
>   - 7B 上的 TODO `#3`：暂不作为唯一 active 主线无限延长
>   - 下一条 active 主线：`Qwen 14B TP=2` 的 `continuous_queue_v2` bring-up
>   - TODO `#4/#5`：继续后置
> - 这一步符合本文调研结论中的“不要闭门造车”原则：
>   - 先把当前更合理的 runner 语义冻结为可回退 checkpoint
>   - 再在更大模型上验证同一机制链，而不是长期停留在单一 7B 局部调参
>
> 2026-04-03 补充：
>
> - 当前 active mainline 已经正式切到 `retry14_continuous_queue_v2`，`substrate_v1` 只保留为历史基线。
> - 当前最新已正式分析、且属于 `continuous_queue_v2` 主线的有效结果是 `retry14_continuous_queue_v2_qwen7b_r500_baseline4_cadencefix @ 500`。
> - `baseline4_cadencefix` 相比 `baseline3_realtiming` 已明显改善：
>   - `TTFT_overall`
>   - `TTFT_comparable`
>   - `TTFT_scaleup_affected`
>   - `Cold_start_latency`
>   - `Throughput_req/s`
>   - `Throughput_tok/s`
> - 这说明当前与本文调研结论对齐的关键 runner 问题，已经不再是“batch-submission substrate 本身是否合理”，而是回到：
>   - `continuous_queue_v2` 上的 scale-up cold path / preload coverage / handoff plan
> - 当前正式 workload 语义也已经收紧：
>   - `Azure real trace arrivals + Azure token distribution + ShareGPT prompts`
>   - `time_scale_factor = 1.0`
>   - 若真实 Azure / ShareGPT 数据缺失，则 fail-fast，不允许回退到 `synthetic_poisson` 或 embedded prompts
> - 因此，本文第 6 节 TODO 排序的当前正式理解应更新为：
>   - TODO `#2R`：当前主线可冻结
>   - TODO `#3`：当前唯一 next active 主线
>   - TODO `#4/#5`：继续后置
>
> 2026-03-31 补充：
>
> - 当前最新已正式分析结果是 `retry44_fix7_cleanrun2_baseline @ 500`；系统运行形态正常，没有新的 crash 型结构性 bug。
> - 但 TODO `#3` 仍未收口：相对 `retry43`，`retry44_fix7_cleanrun2` 仍明显更差于 `TTFT_overall / TTFT_comparable / TTFT_gpu_ready / Runtime_TTFT / GPU_hit_rate / avg_lora_io_ms`，且 `scaleup_affected` 请求仍没有真正进入 `GPU-ready` 覆盖。
> - 这说明当前主瓶颈仍是 cold-path / preload coverage，而不是运行健壮性。
> - 2026-03-30~31 一度出现过偏离主线的越界修改：把 `primary runtime` 也改成 subprocess；这条改动已经被明确判定为不符合当前 clean-tree 主线，并已收回。
> - 当前对本文第 6 节 TODO 排序的正式理解应进一步收紧为：
>   - TODO `#2`：已收口，不再回头扩改 runtime-forward / runtime-shape
>   - TODO `#3`：当前唯一 next active 主线
>   - TODO `#4/#5`：继续后置，不提前进入
> - 当前 latest local code 不再继续沿 `retry43 -> retry44_fix6 -> retry44_fix7` 的更激进 working-set 扩张链修补；其最新真实状态是把 scale-up warmup preferred set 与 dynamic preload budget target 收回到 `live hotset` 语义，同时保留 foreign GPU consumer fail-fast guard，等待下一轮正式验证。
>
> 2026-04-01 补充：
>
> - 当前最新已正式分析结果已推进到 `retry44_fix15_baseline @ 500`；当前最近局部最优正式结果是 `retry44_fix12_baseline @ 500`。
> - `retry44_fix8 -> retry44_fix12` 说明，从 `retry43 -> fix6 -> fix7_cleanrun2` 的激进 working-set 扩张链回撤之后，headline `TTFT / throughput / avg_lora_io_ms` 已明显回正，但 TODO `#3` 仍未收口。
> - `retry44_fix8 -> retry44_fix15` 的 request-level 正式结论进一步收紧为：
>   - `scaleup_affected` 请求仍主要来自 scale-up 新实例的 `host/nvme` tier
>   - 它们并没有被真正推向 `GPU-ready`
> - `retry44_fix15` 明确证明“继续放大 frontier / submitted-window coverage”是错误方向：
>   - scale-up frontier 扩到 `18` 个 adapter
>   - 新实例实际 warmup 也到 `18` 个 adapter
>   - `Cold_start_latency` 升到 `93145.9 ms`
> - 因此，当前对本文第 6 节 TODO 排序的正式理解应进一步收紧为：
>   - TODO `#2`：已收口，不再回头扩改 runtime-forward / runtime-shape
>   - TODO `#3`：当前唯一 next active 主线，但其最终方向不再是 `live hotset vs working set` 二选一，而是 `readiness-aware exact scale-up handoff plan`
>   - TODO `#4/#5`：继续后置，不提前进入
> - 当前必须保持冻结、不在 TODO `#3` 中混改的实验口径包括：
>   - `time_scale_factor = 0.02`
>   - `active_adapter_cap = 48`
>   - `hotset_rotation_requests = 100`
>   - `scale_decision_interval = 25`
>
> 2026-04-02 补充：
>
> - 当前最新已正式分析、且仍属于 `substrate_v1` 历史基线的结果已推进到 `retry44_fix16_baseline @ 500`；它相对 `fix15 / fix12` 继续回正，但 TODO `#3` 仍未真正收口。
> - 更重要的是，当前已经正式确认：旧 runner 的 `arrival/backlog 在线 + submission/decision batch` 语义，并不对齐本文所调研的同类论文与工业实践。
> - 这意味着：
>   - `substrate_v1` 可以作为历史实验 substrate 保留
>   - 但不能再被当成最终 production-correct 运行形态继续积累论文主结论
>   - TODO `#2` 也不能再被表述为“方法学上已经最终收口”
> - 因此当前对本文第 6 节 TODO 排序的最新正式理解是：
>   - `retry14_rebuild`：冻结 `substrate_v1` 历史状态
>   - `retry14_continuous_queue_v2`：开启 `TODO #2R = continuous online queue substrate v2`
>   - 现有 TODO `#3` 的 readiness-aware handoff plan 后续要迁移到 `substrate_v2` 上重做
> - 当前 4GPU 代码适配已经完成；新分支必须继承 `4 × RTX 3090 24GB` 的设备与配置语义，不允许回退到旧的 2GPU 代码假设。
>
> 2026-03-30 补充：
>
> - `retry42_fix4` 已将 TODO `#2` 的 runtime-local topology / background-forward 回退问题实质收口。
> - 当前 next active TODO 已切换为 `#3`：`scale_up_preload_mb=1024` 的 headroom-aware 动态预算。
> - 本文第 6 节给出的排序仍成立，但现在应理解为：
>   - TODO `#2`：当前已收口
>   - TODO `#3`：当前正式进入

## 1. 当前 FaaSLoRA 基线与问题边界

本轮调研时的当前已验证主线基线为 `retry40_baseline @ 500`：

- 仓库：`/home/qhq/serverless_llm_experiment_retry14_baseline`
- 分支：`retry14_rebuild`
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- profile：`qwen_7b_main_v2_publicmix`
- workload：`qwen_7b_auto500_main`
- dataset：`azure_sharegpt_rep1000`
- adapters：`500`
- requests：`500`
- mode：`auto`
- max instances：`2`
- 硬件：`2 × RTX 3090 24GB`

headline 指标：

- `TTFT_overall = 7196.9 ms`
- `TTFT_comparable = 8122.7 ms`
- `TTFT_scaleup_affected = 8722.8 ms`
- `TTFT_gpu_ready = 8069.4 ms`
- `Runtime_TTFT = 7042.5 ms`
- `TPOT = 50.5 ms`
- `E2E_latency = 10393.8 ms`
- `Throughput_req/s = 0.13455`
- `Throughput_tok/s = 17.205`
- `SLO_attainment = 17.8%`
- `Cold_start_latency = 49529.5 ms`
- `Monetary_cost = $0.0034435 / req`

当前调研的核心问题不是“系统是否已经可运行”，而是：

1. 同类论文通常报哪些指标、在什么范围内
2. 当前 `FaaSLoRA` 为什么比这些系统的绝对数字更弱
3. 哪些差距来自问题设定不同，哪些差距来自系统仍有可优化空间
4. 在不破坏当前 clean-tree 主线和三项贡献的前提下，下一步该怎么排 TODO

## 2. 调研方法与选文原则

本轮只采用一手来源：

- 论文 PDF / arXiv / USENIX / MLSys 正式页面

筛选原则：

1. 优先 LoRA serving、多租户 LoRA serving、serverless LLM inference、serverless LoRA inference
2. 只保留对当前三项贡献至少有一项形成直接参照的系统
3. 明确区分：
   - 可直接借鉴设计思路的论文
   - 只能参考方向、不能直接横比数字的论文

本轮核心参考：

1. Punica: Multi-Tenant LoRA Serving
2. S-LoRA: Serving Thousands of Concurrent LoRA Adapters
3. dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving
4. ServerlessLLM: Low-Latency Serverless Inference for Large Language Models
5. ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs
6. FASTLIBRA: Improving the Serving Performance of Multi-LoRA Large Language Models via Efficient LoRA and KV Cache Management
7. LoRAServe: Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems
8. Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead

## 3. 可比性边界

在阅读这些论文时，必须先接受下面这个事实：

- `FaaSLoRA` 当前不是 always-on 单进程共享 serving 系统，也不是大规模多节点集群；它是一个单节点双 GPU、带真实 scale-up / scale-down / warmup / 分层工件路径 / serverless 控制面的 LoRA 原型。

因此，大多数论文数字都不能直接和 `retry40` 的绝对值横比。最常见的不可比来源包括：

1. 对方是 `A100 / A800 / L40S / A5000 / A40`，而当前是 `2 × RTX 3090`
2. 对方是 always-on serving，主要优化 steady-state throughput / decode
3. 对方没有 serverless 冷路径，也没有物理 runtime scale-up
4. 对方的 latency 定义可能是：
   - `s/token`
   - `average request latency`
   - `startup latency`
   - `first-token latency`
   而不是当前 clean-tree 已固定的 `TTFT_overall / TTFT_scaleup_affected / Cold_start_latency`
5. 对方 often 使用更大 GPU 集群或单卡大显存，使 memory pressure 结论天然不同

所以，正确做法不是“看到别人 0.x 秒就认定自己一定做坏了”，而是：

- 先判断论文属于哪一类系统
- 再看它优化的具体瓶颈和本系统是否同构
- 最后只借鉴对当前三项贡献有帮助、且不破坏主线的部分

## 4. 逐篇调研记录

### 4.1 Punica

来源：

- MLSys 2024 PDF: https://proceedings.mlsys.org/paper_files/paper/2024/file/054de805fcceb78a201f5e9d53c85908-Paper-Conference.pdf

系统类型：

- 多租户 multi-LoRA serving
- 共享 GPU 集群
- 非 serverless

关注问题：

- 如何让不同 LoRA adapter 在 decode 阶段也能高效 co-batch
- 如何在共享 backbone 的前提下尽量提高 GPU 吞吐与利用率

硬件 / 模型 /实验方式：

- 使用 `Llama2 7B / 13B / 70B`
- 使用 `NVIDIA A100` GPU 集群
- 重点做 single-GPU text generation comparison、70B tensor parallel comparison、cluster deployment

论文重点结果：

- 在固定 GPU 资源下，相比 state-of-the-art LLM serving systems，`throughput` 可到 `12×`
- 对 token latency 的额外代价很小，摘要中给出“只增加约 `2 ms` 每 token”
- 核心收益来自 custom CUDA kernel + 跨 adapter batching + GPU consolidation

对 FaaSLoRA 的可比性判断：

- 弱可比
- 适合拿来说明“decode batching 对 multi-LoRA 系统至关重要”
- 不适合直接拿它的 throughput 或 token latency 和当前 `FaaSLoRA` 横比

可直接借鉴的点：

1. decode 是长期性能瓶颈，steady-state 不应只看 load path
2. 当系统已经不是结构性 bug，而是 steady-state 性能问题时，应优先识别 decode / occupancy 缺口

不应直接照搬的点：

1. fused kernel / 单进程全局 batching 方案
2. 以牺牲 serverless 隔离和 runtime 独立性为代价的实现路径

### 4.2 S-LoRA

来源：

- MLSys 2024 PDF: https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf

系统类型：

- 单机或多 GPU 的 scalable LoRA serving
- 非 serverless
- 重点在 unified paging、heterogeneous batching、LoRA TP

关注问题：

- 如何在一个 serving runtime 内高效支撑上千 adapters
- 如何同时管理 adapter weights 与 KV cache 的动态显存争用

硬件 / 模型 / 实验方式：

- `Llama-7B / 13B / 30B / 70B`
- 单卡 `A10G 24GB`、`A100 40GB/80GB` 以及多 GPU A100
- synthetic workloads + sampled real traces
- 单 GPU 下对比 `S-LoRA / vLLM-packed / PEFT`

论文重点结果：

- 摘要给出相对 `vLLM naive LoRA` 最多 `4× throughput`
- Table 3 中，单张 `A100 80GB` 上 `Llama-7B` 场景下，`S-LoRA` 在 `5 / 100 / 1000 / 2000` adapters 时仍能维持约 `8.05 / 7.99 / 7.64 / 7.61 req/s`
- 强调 served adapters 数量和 throughput 的数量级优势

对 FaaSLoRA 的可比性判断：

- 中弱可比
- 它和当前系统在“LoRA + KV cache 争用”“多 adapter memory management”上高度同构
- 但它不是 serverless，也没有真实 scale-up cold path

可直接借鉴的点：

1. LoRA weights 和 KV cache 不应分离地被优化，要用统一视角看争用
2. 当 workload 存在 adapter 异构时，仅按 hotness 做缓存决策是不够的

不应直接照搬的点：

1. 以单 runtime 为前提的 unified paging 设计
2. 需要改动 runtime kernel 的方案

### 4.3 dLoRA

来源：

- OSDI 2024 PDF: https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf

系统类型：

- cluster-scale LoRA serving
- 非 serverless
- worker-level + cluster-level 动态编排

关注问题：

1. merge / unmerge 何时切换
2. request 与 adapter 如何共同迁移以处理 replica 间负载不平衡

硬件 / 模型 / 实验方式：

- 四节点集群，共 `32 × A800 80GB`
- 每节点 `128 CPU`、`2048 GB` host memory、`200 Gbps InfiniBand`
- `Llama-2-7B / 13B / 70B`
- 基于 `ShareGPT` 和 `Azure Function Trace 2019/2021`
- 论文主指标为 average latency 与在给定 SLO 下的 max throughput

论文重点结果：

- 相比 `vLLM`，throughput 最高 `57.9×`
- 相比 `PEFT`，throughput 最高 `26.0×`
- 相比 concurrent work `S-LoRA`，平均 latency 最多 `1.8×` 更低
- SLO 口径为 `0.5 s`，比较的是在 SLO 下的最大吞吐

对 FaaSLoRA 的可比性判断：

- 方法学可参考，数字不可直接比
- 这篇论文最有价值的部分，是它明确承认：
  - 仅按 adapter popularity 或均匀分发做 replica 负载均衡不够
  - variable input/output lengths 会放大 replica 间 load imbalance

可直接借鉴的点：

1. request-level tail 和 adapter placement 需要联合考虑
2. 当系统已经没有结构性 bug 后，load imbalance 会成为决定性性能问题

不应直接照搬的点：

1. ILP 驱动的 cluster-level co-migration
2. merge / unmerge 重 runtime 逻辑
3. 面向大集群的重调度机制

### 4.4 ServerlessLLM

来源：

- OSDI 2024 PDF: https://www.usenix.org/system/files/osdi24-fu.pdf

系统类型：

- serverless LLM inference
- 不是 LoRA 专项系统
- 重点是 model loading / startup / live migration / loading-aware scheduling

关注问题：

1. serverless 冷启动为什么慢
2. 如何利用 in-server multi-tier storage 减少 startup latency
3. 如何用 locality-aware scheduling / live migration 减少资源浪费

硬件 / 模型 / 实验方式：

- testbed (i): `8 × NVIDIA A5000`
- testbed (ii): 四节点集群，每节点 `4 × A40`
- 用 `OPT / LLaMA-2 / Falcon`
- dataset 用 `GSM8K + ShareGPT`
- workload 用 `Azure Serverless Trace`
- 指标主要是 `model startup latency / mean latency`

论文重点结果：

- 对 `OPT-6.7B`，平均 startup / serving latency 可达约 `0.8 s`
- `OPT-13B ShareGPT` 也在约 `1.6 s` 量级
- 对 LoRA adapter loading 还单独给了一个小结论：`rank=32, size=1GB` 的 `LLaMA-70B` adapter，加载可到 `83.5 ms`
- 相比 Ray Serve / Ray Serve with Cache，整体 latency 改善很明显

对 FaaSLoRA 的可比性判断：

- 强可比于“serverless 冷路径设计思路”
- 弱可比于“绝对 TTFT 数值”
- 这是当前最能说明“冷路径就该打到秒级，而不是几十秒级”的非 LoRA 方向参照物

可直接借鉴的点：

1. 要把 startup latency 当成单独主问题，而不是附带问题
2. locality-aware scheduling 和 loading-aware estimation 必须有可解释成本模型
3. multi-tier storage 的价值在 serverless 场景里比 steady-state serving 更显著

不应直接照搬的点：

1. 直接把多模型 checkpoint startup 经验等同于 LoRA artifact startup
2. 直接拿其绝对 latency 和当前 LoRA serverless 场景横比

### 4.5 ServerlessLoRA

来源：

- arXiv 2025: https://arxiv.org/abs/2505.14468

系统类型：

- serverless LoRA inference
- 方向上与 FaaSLoRA 最接近

关注问题：

1. backbone redundancy
2. LoRA artifact cold-start latency
3. bursty workload 下的 contention

硬件 / 模型 / 实验方式：

- 单机 testbed：`8 × NVIDIA L40S`
- 多机 testbed：四节点，共 `16 × L40S`
- 使用 industrial workloads
- 强调 `TTFT` 与 `Monetary cost`

论文重点结果：

- 摘要给出：`TTFT` 最多降低 `86%`
- `Monetary cost` 最多降低 `89%`
- 论文动机里直接写明用户期望 `sub-second` first token

对 FaaSLoRA 的可比性判断：

- 强方法学可比
- 中等数值可比
- 它明确说明：如果一个 LoRA serverless 系统的冷路径仍然很长，那么即使 steady-state 还可以，论文说服力也会受限

可直接借鉴的点：

1. pre-loading 必须覆盖的不只是 adapter 文件，还包括真正阻塞首 token 的 preparatory steps
2. cold-start、TTFT、cost 必须联立看
3. contention-aware batching / offloading 是 serverless LoRA 系统后续自然方向

不应直接照搬的点：

1. 如果它依赖更强隔离模型或更重多节点机制，不应直接搬入当前双 3090 单节点主线
2. 不应为了追绝对 TTFT 数字而破坏当前 clean-tree 的三项贡献边界

### 4.6 FASTLIBRA

来源：

- arXiv 2025: https://arxiv.org/abs/2505.03756

系统类型：

- multi-LoRA caching system
- 非 serverless

关注问题：

- 现有 multi-LoRA serving 在优化 `TTFT` 时忽略 LoRA 与 KV cache 的 usage dependency

论文重点结果：

- 摘要给出：平均 `TTFT` 降低 `63.4%`

对 FaaSLoRA 的可比性判断：

- 方向参考价值高
- 数字弱可比

可直接借鉴的点：

1. `LoRA + KV` 不应各自独立优化
2. cache manager / swapper 的 utility 应该比“只看 hotness”更细

### 4.7 LoRAServe

来源：

- arXiv 2025: https://arxiv.org/abs/2511.22880

系统类型：

- distributed LoRA serving
- 强调 heterogeneous rank variability

关注问题：

- heterogeneous adapter rank 会造成 co-batching skew 与 GPU under-utilization

论文重点结果：

- 最高 `2× throughput`
- 最高 `9× lower TTFT`
- 在 SLO 下可用更少 GPU

对 FaaSLoRA 的可比性判断：

- 当前仅作方向参考
- 它最重要的启发不是数字，而是：adapter 异构不是噪声，而是独立主矛盾

与当前 workload 的关系：

- 当前 `lora_manifest_1000.json` 中 500 adapters 的 rank 分布约为：
  - `rank 4`: `83`
  - `rank 8`: `375`
  - `rank 16`: `42`
- size 约在 `18MB - 45MB`
- 因此，未来若只按 tier / hotness 做 utility，确实可能低估 rank / size 异构带来的真实代价

### 4.8 Compress then Serve

来源：

- arXiv 2024/2025: https://arxiv.org/abs/2407.00066

系统类型：

- 通过 shared basis / compression 服务大量 LoRA

论文重点结果：

- 在上千 LoRA 下，可维持单 LoRA serving throughput 的约 `80%`

对 FaaSLoRA 的可比性判断：

- 仅作远期参考
- 它更偏表示压缩 / model representation 方向，不适合当前 clean-tree 主线

结论：

- 不建议纳入当前 TODO
- 未来若论文要扩成“系统 + adapter representation co-design”，可再回看

## 5. 综合判断：为什么当前 FaaSLoRA 数字仍偏弱

### 5.1 不应误判为“系统已经失败”的部分

以下差距主要来自问题设定更难，不应被误判成系统一定做坏了：

1. 当前是单节点双 3090，不是 A100/A800/L40S 集群
2. 当前 headline 指标含真实 serverless scale-up / warmup / GPU-ready 区分
3. 当前系统保留了函数实例 / 物理 runtime / 多层工件路径 / 资源协同控制的完整链路
4. 当前没有走 fused-kernel、单进程全局 batching 这类“更快但会抹掉贡献边界”的捷径

### 5.2 当前真实还该优化的地方

本轮调研后，真正值得继续优化的点有三类。

#### A. 冷路径仍重

证据：

- `Cold_start_latency = 49.5 s`
- `TTFT_scaleup_affected = 8.7 s`
- serverless 相关论文通常把 startup / TTFT 打到秒级或次秒级

当前实现对应点：

- 固定 `scale_up_preload_mb` 仍在 [scripts/run_all_experiments.py](/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/run_all_experiments.py#L2757)
- scale-up warmup 候选仍主要按 `recency + hotness + value` 选，在 [experiment_stack.py](/home/qhq/serverless_llm_experiment_retry14_baseline/faaslora/experiment/experiment_stack.py#L584)

结论：

- 下一主攻点应继续围绕 cold path，而不是再回头加大 TODO `#1` 控制面

#### B. 异构 adapter 的真实代价还没有被完整吸收

证据：

- 当前 workload 有 rank/size 异构
- 相关论文开始直接把 heterogeneity 当主问题

当前实现对应点：

- router 的 bucket 主要按 `lora_gpu / lora_host / lora_nvme / lora_any` 聚合，在 [instance_pool.py](/home/qhq/serverless_llm_experiment_retry14_baseline/faaslora/experiment/instance_pool.py#L180)
- GPU admission utility 当前仍近似为 `hotness * locality`，在 [resource_coordinator.py](/home/qhq/serverless_llm_experiment_retry14_baseline/faaslora/scheduling/resource_coordinator.py#L534)

结论：

- 这不是当前最先动的方向，但很适合成为 `TODO #4`

#### C. LoRA load / KV / decode 的争用模型仍偏粗

证据：

- S-LoRA / FASTLIBRA 都把 LoRA + KV 统一建模
- 当前系统虽然已稳定，但 decode/load contention 仍可能继续拖 `TPOT / E2E / SLO`

当前实现对应点：

- contention path 中仍有按 `50ms` 轮询的等待，在 [resource_coordinator.py](/home/qhq/serverless_llm_experiment_retry14_baseline/faaslora/scheduling/resource_coordinator.py#L246)

结论：

- 这是未来更晚一步的性能精炼方向，更适合作为 `TODO #5`

## 6. 本轮调研后的 TODO 排序建议

以下顺序已经按“主指标相关性 + 不破坏当前主线 + 与三项贡献对齐程度”排序。

### TODO `#2`

清理残留 `device 0` 拓扑硬编码

定位：

- correctness / topology hygiene
- 不是 headline 性能主瓶颈
- 但必须先清掉，避免后续多 GPU 观测、预算、压力口径继续失真

### TODO `#3`

将 `scale_up_preload_mb=1024` 升级为 headroom-aware 动态预算

定位：

- 这是当前最值得继续优化的下一刀
- 直接打 `TTFT_scaleup_affected / Cold_start_latency / TTFT_overall / E2E / Throughput`
- 严格对齐贡献 2 和贡献 3

约束：

1. 只使用系统已有观测值
2. 不能做实例级 / 单轮级 / 单 adapter 级特判
3. 不能破坏 `retry40` 已收口的 live scale-up 主线

### TODO `#4`

rank / size-aware observed utility

范围：

- routing
- preload candidate selection
- GPU admission utility

定位：

- 不是当前 first move
- 但有充分调研依据，且与 workload 异构真实匹配

### TODO `#5`

decode-aware contention control

范围：

- LoRA load / KV / decode 争用的更细粒度控制

定位：

- 晚于 `TODO #4`
- 只有在 `#3/#4` 之后仍有明显 `TPOT / E2E / SLO` 瓶颈时再进入

## 7. 当前明确不建议纳入 TODO 的方向

1. Punica / S-LoRA 式 fused kernel 与单进程全局 batching
2. dLoRA 式 cluster-level ILP co-migration
3. Compress then Serve 式表示压缩主线
4. 任何以牺牲 serverless 隔离、函数实例语义或真实 scale-up 路径为代价的方案

原因：

- 这些方向要么超出当前 clean-tree 主线
- 要么会削弱当前论文三项贡献
- 要么实现风险明显高于当前阶段收益

## 8. 未来使用建议

如果以后要继续 related work / 论文撰写 / 系统调优，建议按下面顺序使用本文：

1. 先看第 3 节，确认可比性边界
2. 再看第 4 节，按论文类别挑参考
3. 进入系统优化前，先看第 5-7 节，避免把不该做的方向加入 TODO

当前最重要的一句话总结：

- `FaaSLoRA` 当前已经把 TODO `#1` 收口，接下来真正该打的是更可解释、更贴近真实运行路径的 cold-path 和异构争用优化，而不是回去破坏已收口的 live scale-up 主线。
