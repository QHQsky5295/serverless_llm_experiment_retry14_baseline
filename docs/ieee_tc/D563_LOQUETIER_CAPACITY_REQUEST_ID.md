# D563：修复容量诊断的跨场景请求身份冲突

2026-10-16。D562失败先已封存：MAIN9b68e76128e6835f8eef28f6487792b50e45acf5，
BASE08321ad37ecab9cc379a2558032ea3b093960af9，均已推送。D562结果不是通过；
8条同时prefill可执行，混合场景未执行，req_00015输出差异仍未知。

## 第一性原因与最小改动

D562两场景使用同一ModelGeneration，原生测量合同在整个generator生命周期内
保存request ID，防止终态后二次冒用。容量函数却只用源request ID构造诊断ID，
导致第二场景第一条登记即被拒绝。源请求是复用输入，不是同一诊断执行事件。

仅修改现有run_loquetier_trace_gate.py的诊断ID：
capacity-all-prefill-sourceID 与 capacity-mixed-prefill-decode-sourceID。
输出中的source_request_id仍为原D445身份，prompt/adapter/target没有改变。
不清除原生历史集合，不删除唯一性检查，不改SMLM、模型、原生生成、serving配置。
这不是遇错回退：从事件身份定义消除两种执行情况的冲突，重复同场景仍失败。

## 真实路径回归

在现有test_ieee_tc_loquetier_measurement.py新增一项回归：

- 实际LoquetierNativeBatchBridge和测量版ModelGeneration.add/pre/post路径。
- 实际CPU KvPool/KvCache、原生registry和终态结果提取；只用可控整数代替模型算术。
- 两个adapter，先完整all-prefill，再同generator混合prefill/decode，均完整drain。
- 源身份不变，四个执行ID不同，KV和active状态归零。
- 再重复首场景仍被原有合同拒绝，不分配KV，不放宽检查。

修复前同一新增测试确实失败于原生登记guard，原始源码/失败日志已留存。
修复后17项容量/所有权测试＋16项原生测量测试，合计33项全部通过、无跳过。
两组使用新解释器隔离source-view导入，不重新安装环境、重建kernel或使用GPU。
前后测试资源域实际递归空成员关闭，峰值284,299,264/284,540,928B；
high/max/OOM/kill/swap均0。所有历史与用户修改保留。

## 边界与主线下一步

此次只证明诊断请求身份修复；不证明batch8数值正确、混合GPU算术已验证、
全500资格或性能收益。D562第132token分歧不能靠这项修复解释。
PyTorch2.1官方数值说明允许批形状导致舍入差异，但不能作为本次错误的结论：
https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html
官方实现来源为固定aae33baeeb19777129c1ccbff99a898d4a0e2c63，
其generation_strategy接收原生next_token_scores；原代码尚未改动。

下一新身份的最小原生诊断，应复用相同8条输入和125-ID分片，记录目标请求在
分歧前缀处的原生logits、top候选间距及执行分段，比较batch2/8与混合情况。
只观察原生greedy结果，不强制token、不事后跳过差异，不用诊断时间作吞吐。
必须验证观测本身不改变score或返回token、卸载/指针与原生绑定仍完整。
这项logit观测尚未实现，也没有新GPU运行。若不能解释差异，资格继续未通过。

更高容量通过后才接在线所有权生命周期适配、完整500功能覆盖和新W0/W1/W2；
保留旧D445/D544、D562失败和所有退化。随后返回全部7B比较及PrimeLoRA优化、
3B、A1–A5/S1–S13、两轮审稿文档与后期图表，不把局部资格当作全计划完成。

RAW：/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d563_20261016。
本轮按结果分析技能分离诊断缺陷与未确证数值差异，按资源技能核查真实限制，
按github-sync技能保存经测试的双仓库检查点；不修改任何论文或冻结指标。
