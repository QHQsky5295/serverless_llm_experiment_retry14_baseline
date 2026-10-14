# D531：完整 W1 到达视图的必要测量适配

2026-10-15。`trace_w1_v1` 调用主仓库既有 FrozenReplayPlan 的 W1：8×500，
phase内时间减半、phase间30秒。只接受SHA绑定的完整4000源行，不生成新trace，
不改变prompt/adapter/target、后端、缓存、路由或资源限制。W0及功能覆盖保持。
拒绝混用标签、速率缩放、短前缀和更大源trace的截取；publisher、service、
physical ledger 共享一个loader。配套主仓库分析器显式使用full4000_w1。

103项跨仓库CPU检查通过；实际输入4000条、132ID、458224目标token、到达
2186.9921293258667秒，和PrimeLoRA同源同view。资格记录在主仓库
paper_results/ieee_tc/main_comparison/d531_w1_input/qualification.json。
这是输入资格，不是已完成GPU性能、数值正确性或W1预算参考。下一D532
复用vLLM已冻结DP4/TP1原生配置；不搜索或优化baseline性能参数。
Shared k3s未触碰，HydraServe已完成的隔离适配和W0不重跑。
