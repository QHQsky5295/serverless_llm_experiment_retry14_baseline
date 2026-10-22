# D703：3B full W0 结果与资格边界

> 2026-10-22 更正：下文 99.875% 是 TTFT≤5000 ms 的开发单项比例，旧审计
> 错误地标为共同联合 SLO。相关达标/相对 SLO 提升解释撤回，等待 D714
> 原生逐请求重算；其他原始记录不覆盖。见
> [D714 纠错记录](D714_COMMON_SLO_CORRECTION_20261022.md)。

## 结果

D703 使用 D701 的同规格 native profile、D693 生产父版本、冻结 3B W0
4,000-request trace、500-adapter 子集和 `fixed_length_greedy_v1` 完成一次完整
回放。严格审计回执为
`paper_results/ieee_tc/p2_backend/d703_3b_full_w0_d701profile/audit.json`，原始
结果保留在
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0`。

| 量 | D703 观测 |
|---|---:|
| 计划/终态/完成/失败 | 4,000 / 4,000 / 4,000 / 0 |
| 固定生成契约匹配 | 4,000 / 4,000 |
| 联合 SLO 达成 | 3,995 / 4,000 = 99.875% |
| 平均 TTFT | 734.351 ms |
| P95 TTFT | 1,784.727 ms |
| 平均 TPOT | 16.278 ms |
| 平均 E2E | 2,608.428 ms |
| 物理生命周期 GPU 时间 | 14,864.808 GPU-s |
| 物理 GPU-s/request（按 offered 请求） | 3.7162 |
| 远端传输 | 132 次；预发布 gzip；请求期打包 0 |

相对同一 3B W0 的 D682，D703 的联合 SLO 提高 7.525 个百分点，P95 TTFT
降低 10.425%，物理生命周期 GPU 时间降低 6.489%。这说明把此前不完整的
D645 profile 替换为 D701 的同 source-spec、无插值导出，确实消除了一个可观测的
qualification/运行画像缺口；不能把这个差异归因于新的算法补丁，因为 D703
恢复的是 D693 生产父版本。

## 已验证的边界

- D701 的 160 个 service class 和 96 个 preparation class 能被运行时加载；本次
  4,000 个请求没有 nearest-class、插值、默认值或 profile fallback。
- 每个请求都有 adapter/content/lease 绑定、confirmed dispatch snapshot、原生
  vLLM token-id 计数和 release receipt；所有 allocation journal 已释放。
- watchdog 没有 `memory.high`、`memory.max`、OOM、OOM-kill 或 swap 事件；远端
  3B 服务已停止，链路状态只读记录为独立 post-run witness。
- 132 次远端交付都绑定冻结 content SHA，使用 `prepublished_gzip_v1`，
  `remote_pack_performed=false`；远端请求读取、网络、物化和必要解包仍保留在
  请求时间线中，部署前发布/管理员运维没有被加入或事后扣除。

## 尚未宣称的内容

固定长度终态、HTTP 成功、adapter 绑定和请求完成不能单独证明数值 LoRA
正确性。当前 physical summary 的 `n_correct` 仍为未知，严格审计因此明确
`numerical_adapter_correctness_qualified=false`、`ranking_eligible=false`。
D703 是完整 native service/SLO qualification 和相对历史候选的诊断改善，不是
正式 G1 排名点，也不关闭 W1/W2、G2 或 baseline 比较。

下一步应在不改写 D703、不重新运行同一 execution key 的前提下，建立独立的
数值正确性证据（同一 canonical prompt/adapter/content SHA，与可信的同后端
参考路径逐请求核对），然后再决定 3B 是否进入正式 baseline 矩阵。若没有独立
参考，必须继续把 `n_correct` 保持为未知，不能用 D703 的 99.875% SLO 代替它。

图表暂不生成；本文件和机器可读 audit 只作为候选状态与论文实验记录。
