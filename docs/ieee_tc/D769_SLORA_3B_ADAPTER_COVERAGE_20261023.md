# D769：S-LoRA 3B完整500-adapter功能覆盖

## 问题与复用边界

D768已验证100请求原生输入/输出合同，实际覆盖29个adapter。本轮回答剩余
500个既有逻辑adapter是否都能通过同一原生执行路径，不能由前缀检查代替。
直接复用D768原生配置、source_view、57项CPU检查及成功GPU证据；没有新
源码、调度/缓存策略、backend升级或baseline性能调参。

这是原计划§6.7的功能覆盖，不是4000请求主性能点。当前仅完成输入绑定，
尚未启动GPU；独立数值正确性仍未知，不由HTTP成功推导正确性或G1/G2排名。

## 冻结输入与执行

- 原3B seed42 trace、500-adapter pool与真实远端一次性发布工件不变。
- 复用既有`adapter_coverage_v1`，与ServerlessLLM D758相同：500请求、500
  实际adapter IDs、68925目标输出tokens。
- view SHA：`4a9f63b08a600c40af7f5acd8066fb518aa21ed48ee8d5c75d7b3e2e8a51088e`。
- S-LoRA保持3B GQA/BMM、native HOST、四TP1 worker、RR、顺序启动；
  canonical token输入边界只修复D767的旧tokenizer重分词差异。
- BMM路径不等于7B packed BGMV分页机制，不宣称原论文所有机制均已覆盖。
- 共同60秒部署通知、真实准备/读取/传输/物化计入时延；请求中不打包。
- 原生token计数、adapter绑定、实际worker与GPU释放计量均复用既有检查。

服务72/80GiB、swap2GiB，独立回放3/4GiB、swap0，CPU隔离。启动使用原
150GiB/102GiB准入规则与独立watchdog；不会为过门槛修改资源或性能条件。
D768关闭的重复下载缓存已回收，唯一模型/工件/trace和历史结果不变。
新RAW为`results/ieee_tc/slora_qualification/d769_20261023`；小型输入绑定
备份于`paper_results/ieee_tc/baseline_audit/d769_slora_3b_coverage_binding/`。

## 解释与下一步

成功要求完整500条原生合同、目标token和500-ID覆盖，不要求SLO达95%才
保存。低SLO原样报告，不单独触发baseline优化；出现具体输入/计量/适配
错误才按既定边界修复受影响路径。完成后保存与释放资源，继续1000兼容、
完整W0/W1/W2和其余3B基线，之后再集中优化PrimeLoRA。
全部A1–A5、S1–S13及审稿证据交付保留；共享k3s不动，当前不画图。
