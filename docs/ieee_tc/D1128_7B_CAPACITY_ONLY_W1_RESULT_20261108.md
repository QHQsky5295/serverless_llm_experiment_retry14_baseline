# D1128：7B软准入消融结果与结论边界

2026-11-08。对应已登记的A1 CapacityOnly W1；唯一功能差异是关闭主动准备
软准入，物理保护、victim/replacement、路由、handoff、层级驻留及后端/
autoscaler不变。Full仍引用主表D1115，不重跑、不换较有利Full来源。

## 1. 完整结果

| 指标 | Full：D1115 | CapacityOnly：D1128 |
|---|---:|---:|
| 原生生成合同匹配/计划请求 | 4000/4000 | 4000/4000 |
| 联合TTFT/TPOT达成率 | 98.550% | 98.250% |
| 生命周期物理GPU-s | 7603.874678 | 7971.358070 |
| GPU-s/offered-request | 1.900969 | 1.992840 |
| P95 TTFT(ms) | 2331.329520 | 2333.273327 |
| 平均TTFT(ms) | 988.900322 | 962.114954 |
| P95 TPOT(ms) | 56.037299 | 57.092136 |
| 平均TPOT(ms) | 36.318603 | 35.906185 |
| 平均E2E(ms) | 5141.060583 | 5005.769222 |
| 真实远端传输次数/线上字节 | 132 / 131480060 | 132 / 131480060 |

Full相对本次CapacityOnly少用4.61004749%物理GPU时间，联合达成率高0.30
个百分点；P95 TTFT低0.08330814%，仅约1.943807ms。不能将这个单轮微小
尾延迟差写成显著或稳健改善。Full平均TTFT、平均TPOT、平均E2E反而更高，
辅助指标代价完整保留。两点均超过冻结W1预算6756.571050141065 GPU-s；
本比较不是G2同预算胜出。

4000请求输入/adapter/arrival/prompt/target合同匹配。同后端输出digest有
192处不同，全部保存，不预先归因为舍入或错误adapter。原始n_correct仍
为null；原生数量和绑定证据不等于独立数值正确性证明。结果是已知seed42
开发回放的合同/时序/资源观测，不冒称独立留出、CI或正式全局排名。

## 2. 机制真的改变了什么

现有比较器逐个验证GPU准备计划的capacity_only标志：Full=false，
CapacityOnly=true；metadata中其他feature gates一致。4000次路由均按
同一service-bin语义重算匹配；同快照load-only shadow不生成反事实时延。

| 准备尝试分类 | Full | CapacityOnly |
|---|---:|---:|
| 已记录准入放行 | 1327 | 1348 |
| 已记录软容量不足延后 | 1 | 0 |
| 准入之前快照过期 | 648 | 631 |
| 已完成复用、无native load调用 | 46 | 72 |
| 缺少准入回执 | 15 | 22 |
| 已观测软余量不足的准入决策 | 1 | 1 |

Full唯一一次延后属于residency，对象education_lora_0091；候选footprint
82264064B、effective_capacity=0、physical_increment=0、batch_pressure=1、
load_pressure=1/3，返回defer_effective_capacity。该attempt为deferred，
其plan最终completed；该回执native_load_invoked未记录，保留null。

CapacityOnly的一次软不足对象是security_lora_0048，同样footprint
82264064B、effective_capacity=0、physical_increment=0、batch_pressure=1，
但load_pressure=0；它被放行且native_load_invoked=true、attempt completed。
这证明开关影响了实际策略，**不是两轮相同候选、相同时刻、相同完整状态
的配对实验**。不能把两次事件相减当作一个反事实加载干扰估计。

缺回执不算成功/延后/物理加载。Full的15条分别为observing4、superseded11，
所属plan为cancelled2、superseded13；CapacityOnly为observing8、superseded14，
所属plan为cancelled2、superseded20。逐条原始attempt及plan上下文已保存。
尝试/计划计数不等于独立物理加载次数。

## 3. 闭环轨迹与其他机制

- Full有20次分配，包含19次natural scale-out；CapacityOnly有16次分配，
  包含15次natural scale-out。两者natural first-service均完整有首token，
  分别19/15，无须为凑样本重复同键。
- CapacityOnly的16个GPU持有区间均正常结束。更多/更少激活次数本身不
  决定总GPU-s，不能用次数代替实际持有积分。
- Full/CapacityOnly分别有1529/1652次稳态residency记录，GPU准备计划
  1287/1396、文件准备计划1292/1406，非目标机制保留。
- CapacityOnly dispatch前tier计数：GPU2572、HOST429、NVMe862、remote137；
  native acquisition来源：GPU2580、HOST414、file999、staging7。两类
  快照时间不同，不把其计数差直接称为不变量冲突。
- 两轮远端次数和线上字节相同，因此不能将本次收益归因于减少下载。

## 4. 可支持与仍未支持的结论

执行闭口：该唯一运行已完整结束、资源释放并分析，保留原始数据与所有
观测；不重跑同键选更好结果。

子结论：在本次共同95%时序要求下，Full观测资源更少、联合达成略高，
但P95 TTFT几乎相同，平均服务指标有代价。不能由仅一次软不足延后证明
全部367.483393 GPU-s差距由准入造成，也不能称普遍的尾延迟提升。

机制覆盖：实际软不足与策略分歧已观测到；自然闭环的候选、激活和时序
随反馈改变。计划内S3应继续用相同候选、初态和触发时间，对照准备与
decode争用，并覆盖延后工作的后续成本。不是为扩大效应而给CapacityOnly
加等待、减少容量或改变替换策略。

系统主结论：本单点不关闭G2或整个系统的优越性论证；独立数值正确性
限制仍披露。继续既定累积节点和机制实验，不进入无界调参/重复循环。

## 5. 执行、来源和保护

原始结果：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1128_20261108/7b_capacity_only_w1/`。
result SHA `ca76e70cb2105cf09b1db0f226fa14e78b672e013471ad61f74f0059bd5c63f7`；
launch SHA `23a7db73ec9bcc0c51d3a2fa0ae9728a4d7ed86ce6efadb79aaf4b2254b34bcd`。

回放14:46前完成收尾，service/replay/watchdog退出码均0，service域已移除，
GPU compute census为空；owned远端inactive/MainPID0。2273次资源样本中
high/max/OOM/swap均0，主机最小available101729792000B。远端journal的132条
记录均匹配本轮clock，无请求内打包，复制前后SHA一致，保存来源说明和
原分析所需parent链接。

启动前checkpoint `7c30e66d8e720c708195fc95e4db4edac048b95d` 已推送V2。
分析复用D1113、D299、D455及D1127/D1052方法，全部在CPU受限资源域执行，
没有GPU重测。来源校验68个唯一文件、旧投稿144文件和三个用户dirty文件
均通过。单元级执行/管理收尾区别见analysis_attempts.md；既有inotify
问题不误标为GPU实验或统计断言失败，也不修改共享系统设置。

完整数据、分类与来源位于
`paper_results/ieee_tc/ablation/d1128_7b_capacity_only_w1/`。
下一步为当前7B RoutingOnly、RoutingHandoff，再按既定主线完成受控A2/A4、
S2/S3、M2/S5及其余机制/敏感性、3B同版本缺项、CE/Pareto、审稿文档和
IEEE图表。整份计划继续active；不重新部署HydraServe或修改共享k3s。
