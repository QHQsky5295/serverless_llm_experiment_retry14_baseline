# D225：D210/D219 输出差异审计与数值正确性准入门

日期：2026-10-05
状态：**离线审计完成；不是性能实验，也不把 D219 升级为 `N_correct` 已知**。

## 1. 审计问题

D219 是 7B、4,000 请求的 confirmed-snapshot full replay；D210 是此前同一
工作负载的普通 Full replay。两次运行都满足固定长度生成合同，但输出摘要并非
完全相同。需要区分三种可能：

1. 运行时批处理、CUDA graph/FP16 数值路径或实例状态导致的合法非确定性；
2. adapter/slot 内容或请求绑定错误；
3. 仅摘要不同但缺少逐 token 真值，当前证据不足以判定前两者。

本审计不把“摘要不同”改写成系统错误，也不把“请求成功且 token 数正确”改写
成数值正确。

## 2. 输入身份和复核方法

| 项目 | D210 | D219 | 结论 |
|---|---|---|---|
| 计划/终端请求 | 4,000 / 4,000 | 4,000 / 4,000 | 数量一致 |
| request ID 交集 | \multicolumn{2}{c|}{4,000} | 完全配对 |
| adapter ID | \multicolumn{2}{c|}{4,000/4,000 相同} | 无输入身份差异 |
| canonical prompt SHA | \multicolumn{2}{c|}{4,000/4,000 相同} | 无 prompt 差异 |
| input tokens | \multicolumn{2}{c|}{4,000/4,000 相同} | 无输入长度差异 |
| target/requested output | \multicolumn{2}{c|}{4,000/4,000 相同} | 无工作量差异 |
| 实际 output token 数 | \multicolumn{2}{c|}{4,000/4,000 相同且合同通过} | 仅证明生成长度 |
| token source | `vllm_token_ids` | `vllm_token_ids` | 计量来源一致 |
| completion token-ID SHA | — | — | 136 行不同，3.40% |

复核直接以两次不可变 JSON 的 `detailed_results.tc_ieee_full.requests` 为输入，
按 request ID 配对；没有使用性能结果或自身输出构造 ground truth。

输入文件 SHA：

- D210：`942948559cea51aebecbcc6f8fdfd55acfb455e4afc33dfa01782b86b2b3ecb1`
- D219：`e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`

## 3. 观察到的差异及不能作出的归因

- 136 个摘要差异请求全部使用了不同的 `instance_id`；这是两次新启动的预期
  事实，不能单独证明实例导致错误。
- 在 136 个差异请求中，cache tier 有 55 个不同；但在 3,864 个摘要相同的
  请求中也有 1,408 个 cache tier 不同。因此 tier 变化与摘要变化不是一一
  对应关系，不能把本表当作 confirmed-tier 的因果消融。
- 136 个差异请求中有 33 个 GPU-ready/replica-mismatch 字段发生变化；同样的
  字段变化也出现在大量摘要相同的请求中。
- 远端 cold/mismatch 字段在差异组和对照组没有系统性变化（差异组仍为
  132 个非 cold、4 个 cold），不能把 D219 的摘要变化归为远端交付。
- D219 的 `effective_capacity_admission_enabled=false`，且 admission/planning/
  promotion counters 为零；所以即便输出摘要相同，也不能据此证明论文 Full
  的 admission/hierarchy 全路径。

因此，当前最强的客观结论是：两次运行的**输入与生成计量合同一致，但输出
序列一致性未知**。D219 的 `N_correct`、联合 SLO 和 G1/G2 资格保持未知。

## 4. 可证伪假设和最小验证

### H0：运行时非确定性

在相同后端、相同权重内容、相同请求/adapter、相同 `fixed_length_greedy_v1`
和单请求受控调度下，独立重复的 token 序列会稳定；D210/D219 的差异来自
并发批处理、CUDA graph/FP16 或实例状态，而非错误 adapter。

### H1：绑定或算术错误

即使在上述受控调度下，输出仍与从原始 checkpoint 独立构造的 adapter reference
不一致，或明确出现另一 adapter/base 的 token/logit 结果。

### 最小验证（下一次 GPU 资源门槛通过后才执行）

1. 复用 D166/D167 已核验的 22 个请求、四个独立权重内容类别和既有 trace；不
   生成新工件、不重跑 4,000 请求。
2. 保留 D207 的 native request/slot/owner 绑定与 D208 的 shrink/expand 算术
   检查；新增一个**诊断专用、显式 opt-in**的逐 token ID（必要时逐步 logits）
   采集，不改变普通 Full 路径的调度、公式或计量。
3. 对相同请求做三次受控重复：固定单请求/批大小、同一 adapter 内容、同一
   prompt hash、同一目标长度；同时记录 native adapter integer ID、slot 内容
   SHA、token 序列 SHA、请求绑定和 execution-mode。
4. 用原始 safetensors/config 独立构造的 reference（不是被测服务输出）检查
   非零 adapter；包含 finance、medical、code、writing，另含显式 base/no-LoRA
   对照。零权重 adapter 只验证身份映射，不声称数值可区分。
5. 预先登记判据：逐 token ID 全部相等才记为 exact；若 GPU 数值路径确实
   产生 tie 或合法非确定性，必须保存逐步 logits/最大 margin 并单独报告，不能
   事后放宽 token 判据或以摘要相似替代 reference。

该验证的读回/同步时间不进入性能结果；它的作用是为后续完整回放提供
`N_correct` 资格证据，而不是制造 Prime 的性能优势。

## 5. 当前决定和后续门

| 项目 | 当前状态 | 处理 |
|---|---|---|
| D210/D219 输入合同 | 通过 | 可作为差异审计的配对输入 |
| 输出长度/原生 token 计量 | 通过 | 仅证明 fixed-output 合同 |
| adapter 内容（22 请求） | D166/D167 已通过 | 不重复 |
| 执行映射（22 请求） | D207 已通过 | 不重复 |
| 算术坐标（22 请求） | D208 已通过 | 不重复 |
| 4,000 请求数值正确性 | 未知 | 等待上述最小逐 token reference gate |
| D219 性能/G1/G2 | 不合格为正式排名 | 不改变 |

在本机 `MemAvailable` 低于计划的 109.5 GiB 物理启动门槛期间，不启动该
GPU 验证，也不通过清 cache、停止无关进程或降低门槛强行运行。远端 artifact
服务保持停止。门槛通过后，先执行本节最小验证，只有 H0/H1 结果明确且记录
完整资源释放，才考虑新的 7B Full 留出块；3B 和外部 baseline 继续暂停。

## 6. 与冻结指标/远端范围协议的关系

本审计不修改冻结协议：

- `METRIC_PROTOCOL_FROZEN_V1.md` SHA：
  `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
- `METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` SHA：
  `485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`。

远端链路/打包边界仍按 V2：部署前的一次性只读 gzip 发布不进入请求延迟；
请求触发的真实 read/send/receive/write、缓存 miss 和本地 materialize 进入
用户路径并单独记录。D77 的 100 Mb/s 与 D78 后复查的 1 Gb/s 是两个不同的
链路状态，不合并为同一性能点，也不从 E2E 中事后扣除发布/打包时间。

## 7. 证据边界

本文件是离线审计和下一实验门，不是新性能样本、不是图表排名、不是
`N_correct=4000` 的证明。它使“为什么不能把 D219 直接用于主比较”变成可
复查的证据链，并给出只改变一个未闭合事实的下一步。
