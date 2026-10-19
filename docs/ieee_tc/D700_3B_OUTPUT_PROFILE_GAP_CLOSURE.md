# D700：3B declared-output profile gap closure audit

日期：2026-10-19
执行键：`D700_20261019`
原始结果：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d700_20261019/source1.json`
审计产物：`paper_results/ieee_tc/p2_backend/d700_3b_output_profile_gap_closure/closure.json`

## 结论

D700 完成了一个受限的 3B 原生 service-profile gap measurement：31 个波次、61/61 请求完成，原生输出长度逐请求等于目标值，所有 lease 已释放；61 条交付记录均为已发布 gzip，`remote_pack_performed=false`，没有请求期远端打包。D700 因此有效地补齐了 D698 指出的类别层缺口。

将既有 D644/D645 观测和 D700 观测按冻结的 D698 候选 output bins `[89, 152, 179, 256]` 做严格集合审计，结果为：

| 项目 | 数量 |
|---|---:|
| 需要覆盖的 service classes | 160 |
| D644/D645 既有观测覆盖 | 130 |
| D698 缺口（D700 前） | 30 |
| D700 新观测带来的类别数 | 49 |
| 合并后覆盖 | 160 |
| 合并后仍缺失 | 0 |

这只说明“类别域”已经闭合，不能写成 3B profile 已经可用于 production 或性能排名。

## 为什么不直接导出 profile

当前 profile collector 的合并契约要求各次 native collection 使用完全相同的 source specification。D645 使用 24 个 exact-content representative adapters；D700 是为 30 个 gap 构造的 bounded run，只使用 6 个 representative adapters，且 source-spec SHA 不同。强行把两者的样本拼接后导出，会丢失 source-spec、adapter identity 和 runtime-input provenance，因此被拒绝。

D700 的正式状态保持为：

- `class_domain_closed=true`；
- `production_profile_written=false`；
- `profile_export_authorized=false`；
- 不改变 D645、D644 或任何历史运行；
- 不把 D700 当作 PrimeLoRA 或其他系统的性能结果。

下一步只有在同一 source-spec、同一 runtime/input contract 下完成 exact-content source coverage 和 service-class coverage，才能生成新的 development profile；任何缺失类别均不得插值、补零或按邻近类别填充。

## 计量边界

D700 继续遵循远端交付冻结口径：请求触发的真实读取、传输、竞争、本地物化和必要解包属于测量路径；发布前准备及远端管理操作不属于请求路径。D700 的原始记录已经证明本轮没有请求期远端打包，所以不存在需要从 TTFT/E2E 中事后扣除的外力时间。

该审计脚本为 `scripts/ieee_tc/audit_d700_3b_output_profile_gap_closure.py`，采用 append-only 输出；它只生成 provenance receipt，不修改 runtime profile、论文结果或旧图表。
