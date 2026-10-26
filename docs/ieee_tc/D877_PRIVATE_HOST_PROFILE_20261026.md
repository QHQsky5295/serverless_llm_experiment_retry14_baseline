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

## 实际失败、收尾与最小修正

18项长度初始化测试通过后，实际source测量完成32/423条，停止在第5组
native_host受控准备。core只读观察显示started=settled=in_flight=0，当前
native源为空。调用链中collector使用未注册的`profile-setup/4`计划身份，
新private begin要求非空计划必须是权威注册计划，因而该调用不能启动。
这里不是新增线程死锁的证据，也不是性能退化结论。

仅中止此资格父进程，资源释放已验证。取消没有生成最终source1.json；
保留32条样本日志、原生观察及全部失败/启动/释放证据，不构造缺失结果。
服务峰值5720973312B，high/max/OOM/swap0；远端全部本轮服务停止。
首次CPU入口的非UUID scope名拒绝亦记入失败说明，不冒称一次无误执行。

修正是明确区分测量wave标签与生产计划授权：空cache受控setup没有运行
planner，故调用plan_id=None；原生容量检查不变，staging仍需真实计划。
仅改变collector，不修改生产计划验证或Full机制。新增1项调用语义测试
通过，原18项和D874/D875证据复用。下一新键D878继续同423 cases，完成后
严格export再完整W2；不再追加无关资格或短Full。
