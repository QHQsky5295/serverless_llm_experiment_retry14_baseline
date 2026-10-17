# D645 — 当前 3B 实测时间配置导出

2026-10-17。状态：离线导出与原生加载器校验通过；不是 Full 性能结果。
遵循完整 PLAN、冻结指标 V1 和单轮 V2，不改变 IEEE 九公式或 G1/G2。

## 本轮解决的问题

直接复用 D644 已封存的准入启用测量，将实测服务时间和准备时间接入当前
3B 的初始化配置。未重跑 368 用例，未新建权重、负载或工件池；没有用
历史 D89/D138 或 7B 时间替代当前 3B，也没有重新选择控制参数。

| 项目 | 实际证据 |
|---|---:|
| 保留请求用例 | 368，含 8 条预热 |
| 服务时间样本 / 观测类别 | 360 / 40，类别覆盖完整 |
| 准备时间样本 / 类别 | 288 / 96，类别覆盖完整 |
| 原静态逻辑 adapter / 文件内容类别 | 500 / 24 |
| 文件按需事务 | 216；37 次观察到 owner epoch 前进，最大 4 |
| 本轮新增 GPU 运行 | 0 |
| CPU 回归测试 | 26 项通过 |

24 个文件内容类别不是 24 个独立训练权重；静态池仍只有两个独立权重 SHA。
相同请求/adapter/prompt/target 的 24 组中有 2 组输出 token hash 不完全一致。
此处保留差异和原生计数/绑定审计，不据此声称逐 token 数值一致，也不筛掉样本。
完整回放正确性沿用当前协议，不能将本导出视为额外的全模型数值证明。

## 首次导出拒绝及证据范围

首次离线导出在 `capacity_before` 字段处失败，原记录和资源关闭回执保留。
原因经实际采集代码核实：`qualify_native_source_matrix` 仅在输入含
`capacity_probe_contract` 时采集逐波 `capacity_before/after`。本次是既有
3B 并发范围内的来源测量，没有扩大容量探测，故只存在全流程首尾 scheduler
快照。不是中途丢字段，也不能将不存在的逐波触发次数补为零。

导出器显式选择 `source_run_endpoints`，使用原有同 owner/clock、严格时序、
计数单调和子集关系检查。默认逐波路径仍要求全部快照；没有自动缺省回退。
endpoint 路径拒绝有容量探测合同或已有逐波快照的输入，防止丢弃细粒度证据。
所有波次仍须完整有序；服务/准备类别、准入/KV、引用释放、远端双端身份及
配置绑定的校验不变。新增反例测试覆盖缺失、错 owner、倒序及非法计数。

本次首尾累计差为 12,914 input turns、5,897 handled controls、3,150 次交还
原生循环、其中 1,557 次有排队输入。这只支持“该机制在测量期间确实执行”；
逐波计数、模型 step 数、GPU 空闲时间和因果加速均未知，不由这些计数推出。

## 绑定、资源与保护

- D644 源 SHA：`d0c6d29b839196c7f96936d5943554e72daf30bd52fb6362fb255304bc995b65`。
- 实际执行版本：`143b5ac066a3716ef14dc42d5187a5bfeb4b1f01`；测量备份版本：
  `f9e595b456f48b3f719d3b26e4741dcaa236bd17`。
- service SHA：`87371bee3f7c6af4fea25c9691ba0838938c41a694317e19d0089020cead8a42`。
- preparation SHA：`a1c38c6dd9f55b0df9310c2adcbde9443b5a2b0b5515907fdc46c3fbde94d6b1`。
- manifest SHA：`972d8c740f9421a530e064c0449ae8c48b20bf623b66bf0c488e7f3a570fe960`。
- 使用既有 `FrozenServiceProfiles` / `FrozenPreparationProfiles` 重新加载验证；
  模型配置、实际后端、输入与资源包络均以 SHA 绑定，EWMA beta 仍为 0.5。
- 远端只读核验已部署 server SHA 和资源配置，服务 inactive；沿用 D644 实际
  启动时 1000 Mb/s/full 记录。没有启动服务或重新发布缓存。
- 两次离线过程均限 CPU 2,3,26,27，内存 high/max 为 3/4 GiB，swap 禁止。
  峰值分别 727,191,552 / 726,351,872 B，high/max/OOM/OOMkill/swap 均为 0；
  无 CUDA 初始化，资源域已递归空并移除。
- 147 个投稿保护文件和用户 manifest SHA 未变；失败证据不覆盖。

原始记录位于 `results/ieee_tc/p2_backend_qualification/d645_20261017/`；
小型导出位于 `paper_results/ieee_tc/p2_backend/d645_3b_profiles/`，其中
`initialization/manifest.json` 给出全部源 SHA、覆盖、估计和绑定。

## 结论和下一主线

当前 3B 的实测初始化配置可供后续 Full 准备使用，但仍是开发配置：
`full_qualified=false`、`production_profile_frozen=false`、`common_slo_qualified=false`。
D643 保留的历史控制参数不是本轮重标定的共同 SLO 或最优工作点。

下一步复用现有 warm-reference 输入索引/采集器流程，为 3B 建立其自身共同
warm 阈值，再完成 Resident 资源参考与 Full W0/W1/W2。不能借用 7B 阈值，
不能由本次 source 微测宣告 G1/G2 合格。3B 全部基线、dLoRA 限定比较、
A1–A5、S1–S13、成本/审稿文档及最终图表保持未完成；不回到 7B 重复资格。
