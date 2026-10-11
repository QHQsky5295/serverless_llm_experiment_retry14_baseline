# D420：vLLM 7B 历史资格复用边界

2026-10-11。只读离线核查；不产生性能运行、不修改历史结果。

检查 baseline 结果树中由公开路径选择式定位的 32 份 7B vLLM DP4／IEEE TC
Resident 请求级 replay JSON。逐文件保存 SHA、请求数、报告成功数、原生 token
来源数、输出合同匹配数及实际 adapter 集合摘要，未重新计算旧延迟或成本。
清单：`paper_results/ieee_tc/baseline_audit/d420_vllm_coverage_inventory.json`，SHA
`2b222ea2d2860e9b71853fc8a439e9cfb359cfc2cca54562ebe9dff81fdf9ede`。
分析配方保留于 raw `vllm_qualification/d420_20261011/inventory.py`。

| 历史证据 | 可复用范围 | 不能替代什么 |
|---|---|---|
| 32 份被检查 replay，实际 adapter 最多 132 | 原始记录、输入、环境与已观察 ID 的执行诊断 | 500-ID 全池原生执行 |
| D213 三次完整 4000 | 已有后端生成/输入诊断；保留三次，不择优 | 真实远端、共同到达和物理资源合同 |
| D395/D396 各4000、132 ID、原生输出长度匹配4000 | 既有工件和原生长度诊断 | 修正后的事件计时、服务资源归属、完整 U_ref |
| D419 100 请求、29 ID | 当前入口、原生事件及现有边界的已通过检查；不重跑100 | 全池与1000兼容、4000主比较 |
| D80 500-ID交付、D406 ServerlessLLM全池执行 | 共同已发布对象内容／其所属系统的证据 | vLLM 0.30.0 本身的500-ID执行 |

无一份被检查旧 replay 实际覆盖500个逻辑ID。该结论限定于列出的检索范围，
不声称遍历服务器所有文件。D397/D412已经确认的计时和归属缺口不能通过
统计补齐，所以这些旧数据按R2复用，不能改称当前入口1000兼容或R0主结果。
不以重复权重类别覆盖代替逻辑ID执行，也不把500逻辑ID写成500独立训练权重。

下一步只补缺失的D421全池功能检查：复用既有`adapter_coverage_v1`，原前500
请求的prompt/arrival/target保持不变，逐项对应排序后的500个既有ID，不写新
trace、不生成权重。这是功能视图，不是性能W0。随后原1000兼容和单轮完整W0。
PrimeLoRA D371、ServerlessLLM D414保持封存；不跳3B、不重调ServerlessLLM。

按experiment-audit技能另请全新只读审阅者复核D419及历史复用依据，只有文件
路径和清单传给审阅者。语义审阅为同模型家族、暂定、建议性；附属shared-references
在本机缺失，不能声称完整独立认证。审阅报告/trace单独保存，不冒充性能数据。

run-experiment继续使用现有入口，serving-llms-vllm仅用于部署/原生边界检查，
其通用示例阈值不替代冻结V1/V2。github-sync保存可回退节点；所有未完成矩阵保留。
