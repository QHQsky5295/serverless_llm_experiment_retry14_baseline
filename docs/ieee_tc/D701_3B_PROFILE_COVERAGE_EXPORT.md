# D701：3B profile coverage 与严格导出审计

执行键：`D701_20261019`
性质：开发期资格证据；不是完整工作负载性能实验，也不是论文排名结果。

## 为什么必须做这一步

PrimeLoRA 的 admission/planner 需要 service profile 和 preparation profile 来
估计不同 prompt 长度、输出长度、adapter rank、驻留层级和并发 admission edge
下的代价。若某个可达类别没有真实测量，使用相邻类别、平均值、插值或零值会把
“没有观测到的代价”伪装成系统行为；之后的 TTFT、SLO 或生命周期 GPU-s 可能
因此来自 profile 缺失，而不是来自 PrimeLoRA 的调度机制。

D698 的离线审计发现原有 3B profile 在预登记的输出长度边界下缺少 30 个
service classes。D700 虽然以新的六个代表 adapter 观测到这些类别，但它的
source specification 与 D645 的 24-representative specification 不同，不能
按 SHA 约束安全拼接。D701 因此在一个新的、追加写入的同规格 source contract
中保留 D645 的所有输入和代表 adapter，并追加预登记的 gap waves。覆盖审计和
profile export 是把这个 provenance 缺口闭合，而不是为了制造一个有利的性能
数字。

## 冻结输入与请求路径

- 复用既有 1,000-request W0 trace 前缀、500-adapter frozen pool、3B 权重和
  content index；不生成新 trace、adapter 或权重。
- 24 个 exact-content representatives，76 个 waves，其中 75 个为
  `representative_measurement`，1 个为保留的 GPU warm-up；总计 423 个请求。
- 生成合同为 `fixed_length_greedy_v1`，prompt 上限 759，输出上限 256，
  canonical renderer 为 `role_lines_v1`。
- 远端只提供预部署的只读 gzip 对象。请求触发的读取、传输、本地物化和必要
  解包仍属于服务路径；部署前归档、整池扫描、清理和运维打包不属于请求路径。
  D701 的所有传输都验证为 `prepublished_gzip_v1` 且没有请求期打包。

## 严格验证

导出脚本 `scripts/ieee_tc/export_d701_3b_profile.py` 在写出文件前依次验证：

1. native run 完整、gated launch 和 watchdog 正常退出；
2. 423 个请求均成功、目标 token 与实际 token 一致、reservation 已释放；
3. physical allocation 已释放、profile workspace 已删除、服务路径已移除；
4. raw/spec/config/trace/runtime receipt 的 SHA 链一致；
5. 每个远端 transfer 已发布、内容已核验、无 request-time packing；
6. 由现有 `source_profile_inputs` 重新解析 24 个 adapter identity；
7. 重新计算 reachable class domain，要求没有缺失 service class 或 content/source
   组合；
8. 通过既有 `measured_source_profile_payloads`，再分别由
   `FrozenServiceProfiles.load` 和 `FrozenPreparationProfiles.load` 重算并加载。

脚本使用 exclusive 文件创建。任一检查失败都不会用 fallback 继续导出，也不
会覆盖既有结果。

## 审计结果

| 项目 | D701 结果 |
|---|---:|
| native requests | 423/423 |
| reachable service classes | 160/160 |
| missing service classes | 0 |
| missing exact content/source pairs | 0 |
| measured preparation classes | 96 |
| representative service samples | 415 |
| preparation samples | 332 |
| remote transfers | 423 |
| request-time remote packing | 0 |
| profile loader checks | passed |

导出目录为
`paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/`，包含
`service.json`、`preparation.json`、`coverage.json` 和 `manifest.json`。

## 证据边界

本结果只证明：在 D701 的 native 采样域内，PrimeLoRA 后续可以读取一个由真实
观测构成、类别完整且 provenance 可追溯的开发期 profile。它不证明：

- 数值 adapter 正确性或任务质量；
- 4,000-request Full workload 的正确完成率；
- 共同 TTFT/TPOT SLO、G1 物理 GPU-s 或 G2 尾延迟；
- PrimeLoRA 相对于任何 baseline 的优越性；
- profile 在所有未来模型、硬件或 workload 上的全局最优性。

因此本产物不生成论文性能图，也不能替代后续 3B PrimeLoRA 的完整资格回放。
它的作用是确保后续性能数字不会由未测类别或错误的跨运行合并产生。
