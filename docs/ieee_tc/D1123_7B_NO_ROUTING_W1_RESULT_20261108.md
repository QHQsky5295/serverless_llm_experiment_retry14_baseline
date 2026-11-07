# D1123：7B 服务时间感知路由的单因素消融

2026-11-08。既定 A1 的一次 W1 测量已完整结束并释放资源。Full 继续引用主表
同一份 D1115，不重跑 Full、不修改阈值或对照机制，不按胜负筛选结果。
本项完成不等于全部消融、两个模型或整个论文实验已经完成。

## 条件、唯一改动与证据身份

Llama-2-7B、vLLM 0.30.0、FP16/TP1、min1/max4、相同14个slot/seqs与D362
profile。复用500逻辑adapter集合、seed42源trace、W1的8×500阶段/30秒间隔、
60秒共同部署通知及fixed_length_greedy_v1。真实HTTP从已发布只读gzip获取，
两端1Gbps/full，无人工等待，不在请求路径打包。

NoRouting仅将同一可行快照上的service-bin优先排序替换成live load排序。
可行性、来源授权、active-LoRA约束、原子预留、稳定tie-break、handoff、
GPU/HOST/NVMe驻留、软准入和扩缩容控制规律均保留。共同观测仍采集，因此
它隔离的是决策规则，不是最低观测开销，也不是LastKnown的信息新鲜度实验。
两次自然闭环的队列、激活次数和缓存位置可以不同，不能视为相同未来状态。

启动前检查点：`0a4270d21a83fd10371675300d3370c8218170b4`，已推送V2。
登记见 [D1123 registration](D1123_7B_NO_ROUTING_W1_REGISTRATION_20261108.md)。
原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1123_20261108/7b_no_routing_w1`。
结果SHA：`b0f9223926f19a5156a52e35cef374f5512aa85656af956e14d2d42c4f7a84e0`。
Full SHA：`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
原始scenario为tc_ieee_full，展示必须标NoRouting，不并入canonical Full。

## 完整观测结果

| 指标 | Full（D1115） | NoRouting（D1123） |
|---|---:|---:|
| 完成且原生合同匹配 | 4,000 | 4,000 |
| TTFT达标数 | 3,963 | 3,989 |
| TPOT达标数 | 3,978 | 3,988 |
| 联合达标数 | 3,942 | 3,977 |
| 联合达成率 | 98.55% | 99.425% |
| 生命周期物理GPU-s | 7,603.874678 | 8,614.702449 |
| GPU-s/offered request | 1.900969 | 2.153676 |
| 平均TTFT（ms） | 988.900322 | 784.159238 |
| P95 TTFT（ms） | 2,331.329520 | 1,866.566116 |
| P99 TTFT（ms） | 4,090.440717 | 3,108.861995 |
| 平均TPOT（ms） | 36.318603 | 31.128132 |
| P95 TPOT（ms） | 56.037299 | 49.301456 |
| 平均E2E（ms） | 5,141.060583 | 4,268.501289 |
| 远端传输次数 | 132 | 132 |
| 线上字节 | 131,480,060 | 131,480,060 |

Full的物理GPU时间低11.7338%，联合达成率低0.875个百分点，P95 TTFT高24.8994%。
两组满足冻结的95%联合时间要求；可描述为本轮观测到的资源—服务质量取舍，
不是所有指标领先。两者均超过W1 G2预算6,756.571050 GPU-s，因此这两点不能
证明共同预算下尾延迟优越。D1117 cap3另存，不能拿其成本拼接这里的延迟。

4,000条arrival、prompt、adapter、target、原生输入和生成合同逐项匹配。
其中176条输出token digest不同，完整保留；没有独立证据将差异归因于特定
数值原因。原始n_correct=null不改写；请求/内容/slot/lease绑定与原生长度
不是独立kernel算术证明，故表中使用per-offered，不伪造per-correct正式资格。
本轮是已知seed42的单轮开发观测，没有独立留出、CI或统计显著性结论。

## 排序确实改变选择，其他机制仍然触发

直接调用现有Router纯selector重算每条不可变候选快照：Full与NoRouting各
4,000次实际选择全部匹配。Full的3,889条、NoRouting的3,934条请求存在多个
可行副本。各自实际状态下，service-bin与load-only shadow分别有2,867和
2,666次选择分歧；这些不是两次运行之间可一一对应的反事实执行。
shadow不产生反事实TTFT，也不证明未选副本当时的全局物理真值。

dispatch GPU/HOST/NVMe/Remote计数分别为：

- Full：2,590 / 435 / 840 / 135。
- NoRouting：2,290 / 630 / 944 / 136。

Full更多请求选择GPU-ready来源，但这并不自动保证更短TTFT；负载集中、
实际保留卡数及闭环反馈需同时考虑。两组实际下载次数/字节相等，因此不能
用减少远端下载解释本项资源差异，也不能把135/136个remote tier当成下载次数。

| 实际机制 | Full | NoRouting |
|---|---:|---:|
| initial activation | 1 | 1 |
| natural-scaleout activation | 19 | 10 |
| 完成scale-up | 19 | 10 |
| scale-down | 17 | 7 |
| 闭合物理持有区间数 | 20 | 11 |
| 文件准备计划 | 1,292 | 1,511 |
| GPU准备计划 | 1,287 | 1,509 |
| HOST准备记录 | 1,018 | 1,003 |
| residency epoch | 1,529 | 1,730 |
| 明确软admission判定attempt | 1,328 | 1,345 |
| 其中有效容量不足而defer | 1 | 0 |

NoRouting的GPU准备均保留capacity_only=false；1,345次明确准入判定均admit。
它还有11个file-handoff、10个GPU-handoff、62个HOST handoff记录。计划数不是
物理复制次数，也不是被first-service使用次数；记录存在仅证明非目标路径运行。
通用legacy feature_gates不能替代IEEE实际selector及typed receipt。

Full的19个、NoRouting的10个自然epoch均有dispatch及首token，initial另列；
first-service按第一条实际dispatch选取，不按成功或最快请求替换，不为凑数重跑。

## 资源与延迟阶段解释

| 互斥GPU-s时间窗 | Full | NoRouting |
|---|---:|---:|
| 到达前 | 34.866169 | 36.267863 |
| 到达窗口 | 7,539.095342 | 8,529.242682 |
| drain | 2.544181 | 8.689345 |
| 清理 | 27.368987 | 40.502559 |

总差1,010.827772 GPU-s中，990.147340来自业务到达窗口；不是仅由收尾或
部署通知造成。实际区间按物理设备并集累计，重叠准备不重复计量。
NoRouting更少的scale-down并不意味着资源更少，持有时长也决定GPU积分。

平均runtime slot等待为274.531486 / 212.770076ms；admission至engine dispatch
为239.654052 / 204.969471ms；native decode为3,955.958026 / 3,341.246320ms
（Full / NoRouting）。阶段恒等式最大误差0ms。这与资源更多、延迟更低的
闭环取舍一致，但不构成这些分项各自独立的因果效应或理论必然性。

## 收尾、交付与下一步

03:55左右同一运行结束；service/replay/watchdog均0，11个持有区间全部闭合，
GPU compute为空，服务资源域移除，远端所属服务inactive/MainPID0。
2,275次资源采样无high/max/OOM/oom_kill，实验swap为0；主机最低可用
101,642,117,120B。匹配clock的远端journal两侧SHA：
`e4db0a099bd87f4167b4b7ab0db696162942f06dc9df2358fb8e8b41b62defed`。
132次传输中131次由最终请求直接关联，剩余记录保留并核对完整交付元数据，
不删除、不从时延或成本扣减。

复用D1113/D299/D455；比较脚本仅在D1121结构上替换为D1055已有load-only审计。
生产代码未改；启动前10项定向CPU测试仍适用。分析限定CPU2,3,26,27、内存
3/4GiB和swap0，数据表已保存，不为展示延误主线。experiment-audit指定模型
gpt-5.6-sol及其共享参考在本会话不可用，独立语义审计记unavailable；执行者的
确定性核对不冒称独立审计通过。源文件、旧投稿和备份校验另存closure_manifest。

产物：`paper_results/ieee_tc/ablation/d1123_7b_no_routing_w1/`。
下一项为当前7B NoHandoff/R0H0及其余A1/累积节点，随后A2–A5、S1–S13、
离线CE/成本/Pareto、两轮审稿文档与IEEE图表。已有五个外部baseline的两模型
矩阵不重开，不切回3B或改共享k3s。完整目标继续active。
