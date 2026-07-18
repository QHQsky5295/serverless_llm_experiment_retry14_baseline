# A5：基线复现性记录与 V2 纳入决定

审计日期：2026-07-16

本文件解释为什么 V2 新实验只增加一个 `ServerlessLLM-new` 系统行，而不强行加入 dLoRA、Chameleon 或 ELORA。这里的“不纳入”不是性能判断，也不是声称这些系统不可用；它只表示在当前 4xRTX 3090、3B+7B、4,000-request、500-real-PEFT-adapter 契约下，无法在不改论文核心机制的前提下得到一组对称、完整、可审计的正式结果。

## 1. 论文规范语义

### 1.1 正式基线的必要条件

一个系统只有同时满足以下条件，才进入 V2 正式对比表：

1. 有作者/官方公开 artifact，或项目中已有可审计的官方 artifact checkout；
2. 在当前硬件上不修改其 routing、adapter/KV management、migration、cache policy 等核心机制即可运行；
3. 同时覆盖 Llama-3.2 3B 与 Llama-2 7B；
4. 能接受同一 4,000-request trace、500 个真实 PEFT adapters、per-request adapter identity 和生成契约；
5. 两个 backbone 均通过完整性、token-source、failure 和 metric-schema gate；
6. 使用相同 GPU budget、arrival/token trace、adapter subset 与计费边界；
7. 结果不是 simulation、source audit、component smoke 或只覆盖一个 backbone 的 limited evidence。

允许的 adaptation 仅限依赖 pin、API/构建兼容、路径/manifest 接线和不改变语义的 replay wrapper。若需要重写 scheduler、cache manager、migration、KV policy 或 adapter execution path，所得系统不再是该论文的忠实 baseline。

### 1.2 V2 唯一新增行

V2 正式新实验只使用一个 `ServerlessLLM-new` 行：

- 来源为 [ServerlessLLM 官方仓库](https://github.com/ServerlessLLM/ServerlessLLM)；论文为 [OSDI '24 ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu)。
- formal campaign 启动前记录实际使用的 upstream SHA、compatibility patch hash 和 harness SHA；“current official” 不是运行中跟随 `main` 漂移。
- 保持 ServerlessLLM 的 request-driven serverless scaling 语义；`min_instances=4 + wait90` 等把系统变成常驻 serverful 的诊断配置不进入正式表。
- 同一张 V2 表中不同时保留旧 `ServerlessLLM` 与 `ServerlessLLM-new` 两行。旧投稿结果只归档，新 V2 选择当前官方复现路径。

### 1.3 不纳入不等于否定

对 dLoRA、Chameleon 与 ELORA，只报告：

- 是否有官方 artifact；
- 本地已经达到的最强复现 gate；
- 还缺哪一项正式契约；
- 为什么继续适配会越过“wrapper/compatibility”边界。

不比较它们的性能，不用 PrimeLoRA 组件替代其缺失模块，也不把失败 gate 当作论文系统本身的负面结论。

## 2. 当前实现与 artifact 证据

### 2.1 决定总表

| 系统 | 官方来源/artifact | 当前本地证据 | 正式契约缺口 | V2 决定 |
|---|---|---|---|---|
| ServerlessLLM-new | [论文](https://www.usenix.org/conference/osdi24/presentation/fu)、[官方仓库](https://github.com/ServerlessLLM/ServerlessLLM) | 本地 audited checkout `9f50241baa5386e06a9321c51f19a9ef5f964c2b`；已有 3B/7B、4,000-request true-remote replay bundle | V2 需按新 seed/契约重跑并在 manifest 固定实际 SHA | **唯一新增正式基线** |
| dLoRA | [论文](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)、[官方 artifact](https://github.com/LLMServe/dLoRA-artifact) | 3B official `migration_type=3` 完整 4,000/4,000；7B 多个 DP/TP envelope 均无法达到 HTTP readiness | 当前 24GB/GPU envelope 下缺完整 7B；继续需改 dLoRA/vLLM core memory layout | 不进 3B+7B正式表；保留 limited/gate evidence |
| Chameleon | [MICRO '25 论文页](https://research.ibm.com/publications/chameleon-adaptive-caching-and-scheduling-for-many-adapter-llm-inference-environments)、[作者 PDF](https://tianyin.github.io/pub/chameleon.pdf) | 截止审计日，论文页/PDF/arXiv 与作者定向仓库检索未定位到官方实现；本项目没有可运行 checkout | 需要从论文重建 per-instance adapter cache、compound eviction 和 multi-queue adapter-aware scheduler | 不复现、不报性能 |
| ELORA | [HPCA '26 出版记录](https://researchportal.hkust.edu.hk/en/publications/elora-efficient-lora-and-kv-cache-management-for-multi-lora-llm-s/)、[预印本](https://arxiv.org/abs/2505.03756) | 截止审计日，出版页/预印本与作者定向仓库检索未定位到官方实现；本项目没有可运行 checkout | 需要从论文重建 unified LoRA/KV pool、dependency-aware manager 和 performance-driven swapper | 不复现、不报性能 |

“未定位官方实现”是有日期和检索范围的审计结论，不是永久性的“没有代码”。若作者后续发布 artifact，应重新执行 source/build/workload gate，而不是沿用本表。

### 2.2 ServerlessLLM-new 证据

当前 local source：

```text
/home/qhq/serverless_llm_baselines/repos/ServerlessLLM
upstream: https://github.com/ServerlessLLM/ServerlessLLM.git
audited base SHA: 9f50241baa5386e06a9321c51f19a9ef5f964c2b
```

已有 non-overwriting evidence：

- [`paper_results/new_serverless_baselines_remote_v1/README.md`](../../paper_results/new_serverless_baselines_remote_v1/README.md)
- [`serverlessllm_new_metrics.csv`](../../paper_results/new_serverless_baselines_remote_v1/tables/serverlessllm_new_metrics.csv)
- baseline harness 的 `results/paper_experiments/15_new_serverless_baselines_remote_v1/`

已知边界：

- ServerlessLLM 主要管理 model/runtime startup，并非以 PrimeLoRA 相同方式实现 per-request multi-LoRA readiness control；V2 应把它标为 serverless LLM baseline，不暗示机制等价。
- 已有 harness 通过 wrapper 映射正式 adapter workload 与 `e2e_v3` 指标。正式 V2 必须继续保存 wrapper patch/provenance，且不得把 wrapper 的适配逻辑称为 upstream feature。
- [`SERVERLESSLLM_NEW_OPTIMIZATION_ANALYSIS_2026-05-21.md`](../SERVERLESSLLM_NEW_OPTIMIZATION_ANALYSIS_2026-05-21.md) 已证明 warm-min4 会改变 serverless 语义，因此只保留为诊断，不作为“优化版基线”。

正式 V2 使用流程：

1. fetch official upstream并记录 resolved SHA；
2. 对 compatibility patch做 diff audit，确认未改 router/controller/backend policy；
3. 运行 100-request smoke；
4. 冻结 validation-selected documented parameters；
5. 对 3B/7B held-out seeds运行完整相同契约；
6. 每个结果写 upstream SHA、patch SHA、deploy config、trace/subset hash 与 no-fallback gate。

### 2.3 dLoRA 证据

本地 official artifact checkout：

```text
/home/qhq/serverless_llm_baselines/vendor_new_baselines/dLoRA_artifact_main_20260519
upstream: https://github.com/LLMServe/dLoRA-artifact.git
audited base SHA: 75f1c439446fe194b1df8a24982ef9067841fab5
```

详细 gate 位于：

- [`paper_results/new_serverless_baselines_remote_v1/gates/dlora/README.md`](../../paper_results/new_serverless_baselines_remote_v1/gates/dlora/README.md)

客观结论：

- compatibility layer 能加载真实 PEFT adapters，未替换 dLoRA scheduling/migration；
- Llama-3.2 3B 的 official `migration_type=3` 路径已完成 4,000/4,000 true-remote replay；
- 对称的 Llama-2 7B 在 DP2/TP2 下受 host-memory startup envelope阻塞；G1/TP4 即使降低 GPU adapter capacity并提高 utilization，仍得到 0 GPU cache blocks，未达到 HTTP readiness；
- 解决 7B 需要改变 dLoRA/vLLM 的 cache allocation、adapter placement、rank/quantization 或内存布局，已不是普通 wrapper adaptation。

因此只报告 3B 会形成 asymmetric baseline coverage；把 7B gate failure填成性能值更不成立。V2 将其保留为复现性附录证据，不放入要求 3B+7B完整覆盖的正式表。

### 2.4 Chameleon 证据

Chameleon 是 many-adapter runtime system，核心包括 per-instance GPU adapter cache、compound eviction 和 non-preemptive multi-queue adapter-aware scheduling。其机制与本审稿意见高度相关，但：

- 官方论文页、作者 PDF、arXiv 记录没有给出本项目可定位的 official artifact link；
- 截止 2026-07-16，对论文标题和作者的 GitHub 定向检索没有找到可确认的官方实现；同名 `facebookresearch/chameleon` 和 `chameleon-llm` 是不同项目，不能误用；
- 用 vLLM/S-LoRA/PrimeLoRA 重新实现其 cache 与 scheduler，会把 baseline 变成我们的解释性复刻，性能和正确性无法归因给 Chameleon。

因此合理处理是：相关工作中保留机制对比；复现表中记录“official runnable artifact not located as of audit date”；不生成 Chameleon 性能行。

### 2.5 ELORA 证据

ELORA 的核心是 LoRA/KV usage dependency、unified caching pool 和根据 GPU idle/busy 状态执行的 performance-driven swapper。当前审计发现：

- HPCA '26 出版记录与预印本可以确认系统及其机制；
- 截止 2026-07-16，出版页、预印本和作者定向 GitHub 检索没有给出可确认的 official runnable artifact；
- 将现有 PrimeLoRA ResourceCoordinator 改造成 ELORA unified LoRA/KV manager会直接重做其核心贡献，不能作为忠实复现；
- 即便只实现一个 cache heuristic，也无法声称等价于论文中的 dependency-aware manager/swapper。

因此 ELORA 只进入 related-work/reproducibility note，不进入性能表。

## 3. 复现性审计模板

若未来重新检查任何候选，必须保存：

```text
system/title:
official paper URL:
official artifact URL:
artifact SHA/tag/date:
license:
hardware/runtime prerequisites:
native 3B support:
native 7B support:
native PEFT/per-request adapter identity:
500-adapter materialization gate:
100-request smoke:
4000-request 3B result:
4000-request 7B result:
token-source/fallback status:
compatibility patch SHA and scope:
core mechanism changed? yes/no:
formal decision and reason:
```

没有 artifact URL 或两 backbone 完整 result 时，table value 留空并写 reason；不得用 simulation、部分请求或另一系统的模块填充。

## 4. 当前证据与论文规范的区分

- **论文规范语义**是“正式行必须对称覆盖共同 workload，且不改核心机制”。这是一项公平性规则。
- **当前实现证据**是具体 artifact SHA、local compatibility patches、成功/失败 gates。它只说明当前机器与当前版本的复现状态。
- dLoRA 7B 的失败不是“dLoRA 无效”；Chameleon/ELORA 未定位 artifact也不是“系统不存在”。论文必须使用 reproducibility limitation语言，不作性能推断。
- ServerlessLLM-new 的现有 3B/7B结果证明运行路径可行，但 V2 seed/生成/带宽协议改变后仍需重跑或通过严格 hash/contract 等价检查，不能自动复用为新 held-out 结果。

## 5. 可直接用于英文论文的文字

### 5.1 Baseline selection

> **Baseline reproducibility.** We include the current official ServerlessLLM path as the additional serverless-system baseline because its public artifact can be executed end to end for both evaluated backbones under our common replay contract. We record the exact upstream revision, compatibility-patch hash, deployment configuration, trace and adapter-subset hashes, and we retain ServerlessLLM's request-driven scaling policy. We do not include the diagnostic configuration that pre-creates and waits for all four instances, because that changes the baseline from elastic serverless serving to a fully warm deployment.

### 5.2 dLoRA limitation

> We also audited the official dLoRA artifact. Its period-migration path completes the full 3B workload after compatibility-only adaptation, but the corresponding 7B path cannot reach service readiness within the 4x24-GB testbed envelope: the tested DP/TP configurations either exhaust the host startup envelope or leave no GPU KV-cache blocks after loading the adapter pool. Resolving this would require changing dLoRA/vLLM's core memory or adapter-placement layout. We therefore retain this as single-backbone reproduction evidence rather than reporting an asymmetric 3B-only row in a table that compares both backbones.

### 5.3 Chameleon/ELORA limitation

> Chameleon and ELORA are directly relevant many-adapter systems, but as of our artifact audit on July 16, 2026, their official publication and author pages did not expose a runnable implementation that we could validate under the shared workload. Reimplementing Chameleon's cache/scheduler or ELORA's unified LoRA--KV manager from the paper would reproduce their central mechanisms through our own code and would not constitute a faithful performance baseline. We therefore discuss their mechanisms and state the artifact limitation explicitly, without inferring performance from an unofficial reimplementation.
