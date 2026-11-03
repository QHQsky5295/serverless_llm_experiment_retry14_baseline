# D1049：保留正确性边界，进入机制消融

2026-11-04。承接D1048的同配置3B三场景、20个已测系统点比较。本轮没有新增
性能结果，也不重新开启HydraServe/k3s适配。当前目标仍active。

## 证据审阅与执行决定

采用experiment-audit的新会话只读审阅；指定旧模型和shared-references在本机
不可用，因此记录为可用继承模型、same-family/provisional fallback，不是正式
或跨模型家族认证。完整裁决与来源见
`paper_results/ieee_tc/experiment_audit/d1049/EXPERIMENT_AUDIT.{md,json}`。

审阅复算20行登记来源、41个直接来源及三份4000请求阶段表，未发现D1048的
指标值不符。三场景3914/3836/3833条联合时延达标以及物理GPU-s观测保留。
这不能自动升级为全量数值算术正确性或全局最优性。

3B既有500逻辑adapter池为零更新结构负载。原生request/content/slot/lease
绑定与加载证据可报告，但输出不能区分不同零adapter或证明非零LoRA效果。
不生成新权重掩盖这一限制，也不反复做无法解决数值不可辨识性的负对照。
旧D980/D981仅提供旧配置100请求、29个adapter的独立实现序列见证：95条全等、
5条差异保留。当前槽数/capture等配置已变化，不能直接把这个比例移植到当前
Full；7B D990/D991也不能作3B证明。原始n_correct=null不改写。

最小勘误在本文件追加，不改旧raw：

- D981原始limitations模板的“7B”不正确；实际backbone/config/pool为3B。
- D981文档的“两条finance gap=0”应为req_00006/finance_lora与
  req_00078/research_lora_0008。95/100与五条first-divergence top-2数值不变。
- 4000条fresh descriptor身份记录是路径/库身份，不是4000次张量算术证明。

以上边界不要求4000请求HF/PEFT逐token全等门槛，也不要求重复warm、预算、
绑定或长度资格。按冻结协议推进同合同机制消融；若将来要主张当前版本数值
一致，先离线核对已有完整输入键和参考摘要，再决定具体缺口是否需要最小测量。

## 已完成无损归档

复用现有compress_closed_main_outcomes.py，仅归档已实际关闭释放的D927、D932、
D958、D1043四份main_outcome。每份原SHA、gzip SHA、两次解压SHA、原路径和
恢复方式均保存；删除的是验证后的未压缩副本，不删除成功/失败证据。

- 实际分配空间回收2,886,549,504 B（约2.69 GiB）。
- df可用从160,388,222,976 B到163,272,794,112 B；两者差额不冒充精确归档收益。
- 辅助域峰值1,657,311,232 B；high/max/OOM/kill/swap均0；已退出exit0。
- 前后147个旧投稿文件与两项用户dirty SHA一致；无GPU测量与维护重叠。

回执为`paper_results/ieee_tc/storage/d1049_closed_outcomes_20261104.json`及
`d1049_closed_outcomes_verification_20261104.json`。只恢复到新路径，不覆盖旧证据。
本次不重新进行全盘扫描；D848已记录的其他项目/权限缺口继续保留，不删除。

## 下一主线

ablation-planner的有界审阅见D1049_ABLATION_ENTRY_REVIEW_20261104.md；完整
A1–A5没有删减。先执行D1050的3B W1 NoHandoff，复用D1047 Full。后续再补
现有缺失的load-only selector、LastKnown信息视图、NoHOST复用边界及
CapacityOnly激活入口透传，按各自最小差异测试，不叠成一轮生产优化。

D1050仅配置变化和过时测试fixture的async接口修正，无生产策略修改。
三项目标测试首轮保留2个fixture错误，修正await/AsyncMock后二轮3/3通过。
不得把测试修正写成性能收益。全部主比较/正确性范围、A1–A5、S1–S13、
CE/Pareto/价格敏感性、两轮审稿文档、IEEE图表和最终FaaSLoRA V2集成仍开放。
EXECUTION_STATUS.md含用户修改且头部陈旧，不覆盖；当前交接见本文件及D1050。
