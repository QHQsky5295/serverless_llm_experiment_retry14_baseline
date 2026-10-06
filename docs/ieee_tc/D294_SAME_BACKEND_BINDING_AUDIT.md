# D294：D290/D293 同后端输出差异的已有证据审计

2026-10-06；离线工作，无 GPU/远端服务启动，无新性能样本。
输入为 D293 审计中绑定 SHA 的两个原始结果，以及既有冻结 500-adapter 内容清单。
复用 `plot_d290_diagnostic.py --binding-audit-only`，不重新生成权重、负载或输出。

## 结果

| 记录级检查 | D290 | D293 |
|---|---:|---:|
| 全部请求 | 100 | 100 |
| adapter 名称与 intent/receipt/source 一致 | 100 | 100 |
| 选择、加载、native token timing、释放的 owner 一致 | 100 | 100 |
| intent/acquisition/native lease 与 adapter integer ID 一致 | 100 | 100 |
| acquisition 后槽位与 source 槽位一致 | 100 | 100 |
| 文件清单独立重算内容 SHA 与三处回执一致 | 100 | 100 |
| acquisition 早于 native dispatch，同一 clock | 100 | 100 |
| acquired/released 完整且 token digest 传递一致 | 100 | 100 |
| 数值正确性资格 | 未建立 | 未建立 |

六条输出不同请求 `req_00005/00059/00074/00077/00084/00091` 的原生 prompt
token-ID SHA 在两次运行中全部相同。它们包含相同与不同 pre-dispatch tier，
因此没有证据把这些差异直接归因于远端获取或某一层级。

本表只验证**已经保存的控制与执行入口记录自洽**，不是重新读回 GPU tensor，
也没有观察每个 kernel 的实际数值操作。不能把 100/100 binding 检查写成
`N_correct=100`，更不能推广到 4,000 请求。

## 官方实现与历史核查

- D225 已提出同后端受控重复、实际 slot/content 与独立 reference 的区分。
- D246 是一次 22 请求 native sequence transport；D247 是跨后端 HF/PEFT，
  strict exact-match 为 19/22。D248/D249 保留 near-tie 与旧回执隔离边界。
  **这些不是三次同后端、受控单请求的重复实验**，不能把它们改名算作已完成。
- vLLM 0.30.0 官方文档说明默认不保证输出完全可复现，online 调度若要求
  batching-independent 输出需独立启用相应机制。
  [官方说明](https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/)。
  官方支持的硬件条件本身不能证明当前 Multi-LoRA 组合已验证；本轮没有启用
  batch invariance、改 attention、降低精度要求或修改生产调度。

当前候选解释是批组合/FP16 运算路径变化，但 D290/D293 没有保存完整 token-ID
序列、逐步 logits 或实际 batch membership。首分歧位置不能由 SHA 恢复；
也不能用调度重叠时间冒充实际 batch membership。相关字段在产物中保持 null/false。

## 下一步与主线边界

沿用已有 token-ID opt-in 和 native qualification 路径补**受控同后端重复**，
而不是再次盲跑 4,000 请求或重复 D247 跨后端 reference。使用既有请求索引与
现有非零/零权重类别，冻结请求、prompt、adapter、生成数量和执行配置；观察
单请求受控重复与合法并发情形的差异，保存原生序列及必要的首分歧证据。
结果必须区分 exact-match、内容/绑定错误、数值路径未辨明，不能事后放宽阈值。
执行前仍需精确登记最小子集和次数并通过磁盘/内存门，不在此文中声称已执行。

只有该问题得到可核查结论后再推进完整 7B 资格；共同 SLO、预算、旧 Prime
新口径比较仍缺，3B 未达标问题仍保留。所有外部 baseline、正式比较和后续
消融/敏感性不提前恢复。

## 产物与验证

`paper_results/ieee_tc/p2_backend/d294_output_binding_audit/` 保存所有 200 行检查、
六条差异、来源 SHA、冻结内容清单 SHA 和脚本 SHA；状态表适合本项，不画性能图。
20 项 CPU 测试通过，覆盖错误 owner/lease/adapter/slot/content/clock/digest
以及 D293 的时延与字段检查；首轮一个测试尾块位置错误已修复后完整重跑通过。
使用 3/4-GiB、swap=0、CPU 2,3,26,27 的独立资源域，不改变推理环境或公式。
