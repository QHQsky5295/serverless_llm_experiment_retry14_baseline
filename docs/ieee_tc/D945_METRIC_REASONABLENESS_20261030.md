# D945：物理生命周期 GPU-s 与成熟服务指标的口径裁决

日期：2026-10-30  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 1. 裁决摘要

物理生命周期 GPU-s 不是把延迟和成本相乘得到的自定义“效果分数”，而是对实验期间真实 GPU 分配区间的直接资源时间积分。因此，它适合回答本文的 G1 问题：在正确完成相同工作负载并满足共同服务要求的前提下，系统实际占用了多少 GPU 时间。

但它不是可以单独替代服务质量的万能指标，也不能在不同资源计量合同之间直接比较。最终评价采用三个相互约束的维度：

1. **可行性**：所有请求正确完成，并报告共同 TTFT/TPOT 联合 SLO 及 goodput；
2. **资源效率（G1）**：在可行性成立后，比较同一物理计量合同下的 GPU-s/request；
3. **弹性质量（G2）**：在预先登记的共同物理预算下，比较扩容和 adapter churn 场景的 P95 请求级 TTFT/E2E，并同时报告联合 SLO/goodput。

美元成本、每百万 token 成本、平均延迟、吞吐和 CE 作为补充解释，不得覆盖前述可行性或资源结论。

## 2. 与同类工作的关系

同类系统通常将服务质量和资源/费用分开报告，而不是依赖单一乘积指标：

- ServerlessLLM 的论文关注 serverless 推理中的低延迟、分层制品/内存管理、迁移和按使用付费的资源机会；其核心实验同时呈现延迟、冷启动/迁移行为和资源成本，而非把 CE 作为普适标准。见 [ServerlessLLM (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/fu)。
- HydraServe 的论文将冷启动延迟和 SLO 达成作为 serverless serving 的主要服务结果，并结合按使用付费场景讨论系统收益。见 [HydraServe (NSDI 2026)](https://www.usenix.org/conference/nsdi26/presentation/lou)。
- vLLM 的 goodput 定义把“请求在延迟目标内完成”作为质量约束，并允许同时指定 TTFT、TPOT 和 E2E 目标；Kubernetes inference-perf 也采用 TTFT/TPOT 联合 goodput 语义。见 [vLLM goodput schema](https://vllm-project.github.io/guidellm/main/reference/guidellm/schemas/benchmark/goodput/) 和 [inference-perf goodput](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/goodput.md)。

由此得到的客观结论是：本文不应声称“GPU-s 是领域统一首要指标”，但可以把它作为与 PrimeLoRA 资源目标直接对应、可审计的资源计量轴，并用成熟的联合 SLO/goodput 和尾延迟共同约束资源比较。

## 3. 冻结公式与资格条件

对物理设备或明确 GPU 资源身份 (d)，定义其被服务实际分配的并集时间区间为 (A_{s,d})。系统 (s) 的物理生命周期资源为

\[
U_s=\sum_d\int \mathbf{1}[t\in A_{s,d}]\,dt,
\qquad
u_s=U_s/N,
\]

其中 (N) 是完整计划中的请求数，不能只用成功请求数作分母。分配区间必须来自可追溯的 allocation/release 或 lease census；实例数量乘以客户端 wall time、逻辑副本寿命或账单折算值不能自动升级为物理 GPU-s。

一条结果只有同时满足以下条件，才可进入 G1 资源排名：

- 请求完整终止，失败、截断、取消和重试均有逐请求归因；
- 输入、adapter 集合、到达时间和生成合同与比较对象一致；
- TTFT/TPOT 联合 SLO 定义和阈值一致，并报告 goodput；
- 同一运行的物理分配、释放、设备身份和结束清点完整；
- 运行期间资源边界、远端交付合同和计量方式未被改变。

缺少任一项时，仍可报告 TTFT/SLO 诊断，但 G1 资源列必须为 unavailable，不能用 CE、`infra_gpu_seconds_total` 或“估计生命周期”补齐。

## 4. 与论文研究边界的对应

请求触发的制品读取、远程传输、共享带宽竞争、本地接收和必要物化属于用户请求路径，计入请求级 TTFT/E2E；发布前一次性生成的只读压缩交付缓存属于部署准备，单独记录，不计入请求路径。该划分对所有系统使用同一冻结合同，但不强制不同系统产生相同的获取次数或缓存命中率，因为这些正是缓存和驻留机制的研究对象。

物理 GPU-s 同样只计入实验服务生命周期内实际分配给被测系统的 GPU 时间。远端制品节点的离线打包、清理、哈希扫描和管理操作不计入推理机 GPU-s，也不得在请求级时间线中伪造为应用路径延迟；远端请求服务端为完成一次合法对象读取所消耗的服务时间则保留在阶段分解中，并与本地接收和物化分别报告。

## 5. 当前证据裁决

PrimeLoRA D931 已有 `physical_gpu_resource_summary_v1`，物理 GPU-s 为 11,716.0641，且计划/终态及未关闭 lease 清点完整。ServerlessLLM D941 当前的 16,899.3785 来自 `simulated_gpu_second_deployment_lifecycle_v3_capped`；其五个一 GPU 实例有生命周期估计，但现有结果没有同格式的物理 allocation/release、设备身份和 lease census。因此该值只能作为部署生命周期诊断，不能与 D931 直接计算 G1 节省比例。

这不是对 GPU-s 指标本身的否定，而是对证据合同不一致的否定。D944 已撤回此前由两种字段直接相除得到的 PrimeLoRA 相对 ServerlessLLM 资源优势；在 ServerlessLLM 的物理计量闭合前，不生成 G1 资源排名图，也不据此继续调整 PrimeLoRA。

## 6. 后续执行规则

1. 优先离线检查 ServerlessLLM 现有日志能否重建同一物理合同；不能重建时，保留 TTFT/SLO baseline 行，并明确 G1 resource unavailable。
2. 只有当比较所需的物理证据确实缺失且该缺失会阻断主张时，才增加一次同合同物理资源侧证运行；不盲目重跑已经完整的 PrimeLoRA 或 ServerlessLLM 请求回放。
3. 在资源合同闭合前，继续推进不依赖 G1 资源列的基线资格和请求级服务指标，但不把诊断值写成最终资源排名。
4. 所有后续表格优先列出：完整完成率、联合 SLO/goodput、P95 TTFT/P95 E2E、物理 GPU-s/request（可用时）和成本/1M token；CE 只作敏感性或解释性列。

