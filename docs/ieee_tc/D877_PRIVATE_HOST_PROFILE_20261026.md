# D877：同一私有HOST候选的服务/准备初始化测量

D876在业务到达前被严格配置身份检查拒绝，未产生性能点。D875已通过真实
原生构造、HOST提交、GPU内容、固定输出及释放核验。本次不重做这些资格，
不再增加短Full或原因profiling，只补齐改变执行路径所必需的时间初始化。

复用D725最终成功的既有collector、24个代表工件、76 waves、423条原请求
及严格exporter。保持原工件/输入/输出，加入已经验证的
`private_cpu_core_commit_v1`。继承当前D747描述器、isal解包、真实已发布
远端交付和原资源包络。无新权重、无重新抽样、请求中无打包。

固定输出的既有完成长度可用于初始化同一输入桶；这是长度证据转移，不是
时间复用。显式记录graph及HOST身份变化，重新实测所有时间类；不移除严格
profile身份检查，不给D725旧时延换标签。共同SLO D648、Resident D657、
95%和75%不变。新profile仅供同候选下一次完整3B W2使用，不视为完整性能。

本次只扩展已有prepare/runner的显式接口与长度绑定检查。原始依据再次核查：
[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的CPU构造与cache提交为不同步骤；
[engine core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
管理原生执行循环。本次不再修改生产候选，九公式不变。

raw：`results/ieee_tc/p2_backend_qualification/d877_20261026/`；
curated：`paper_results/ieee_tc/p2_backend/d877_private_host_profile/`。
结果以实际completion/export为准；未完成不声称通过。独立数值正确性仍未知。
后续唯一主线：导出匹配初始化 → 新键完整4000请求W2 → 按冻结指标接受/撤销。
同版本两模型全场景、其余baseline/dLoRA、A1–A5/S1–S13、数值证据、审稿图文
及最终V2均仍开放；目标前不绘图，不操作共享k3s。
