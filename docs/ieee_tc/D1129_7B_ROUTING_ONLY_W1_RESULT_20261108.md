# D1129：7B累积路由消融结果与结论边界

2026-11-08。已批准A5链的RoutingOnly节点：只在D1119 ElasticOnly上
增加既有IEEE service-bin路由；按需缓存、autoscaler、后端和其他非目标
配置相等。Full继续引用主表D1115，未重复或选择更有利的Full运行。

## 1. 完整观测

| 指标 | ElasticOnly D1119 | RoutingOnly D1129 | Full D1115 |
|---|---:|---:|---:|
| 原生合同匹配/计划请求 | 4000/4000 | 4000/4000 | 4000/4000 |
| 联合TTFT/TPOT达成率 | 99.675% | 97.625% | 98.550% |
| 生命周期物理GPU-s | 8575.626704 | 7204.901377 | 7603.874678 |
| GPU-s/offered-request | 2.143907 | 1.801225 | 1.900969 |
| P95 TTFT(ms) | 1035.577002 | 1868.047928 | 2331.329520 |
| 平均TTFT(ms) | 557.647797 | 808.267828 | 988.900322 |
| P95 TPOT(ms) | 46.535648 | 57.641736 | 56.037299 |
| 平均TPOT(ms) | 27.602703 | 35.084678 | 36.318603 |
| 平均E2E(ms) | 3532.720622 | 4653.093401 | 5141.060583 |
| 远端传输次数/线上字节 | 132/131480060 | 132/131480060 | 132/131480060 |

RoutingOnly相对ElasticOnly少用15.983967% GPU时间，但joint低2.05个百分点、
P95 TTFT高80.387158%。这支持共同最低95%时序要求下的观测资源收益，
同时有明显质量代价；不能称路由全面改善服务。

Full相对RoutingOnly多用5.537526% GPU时间、P95 TTFT高24.800305%，
joint高0.925个百分点。平均TTFT/TPOT/E2E也更高，P95 TPOT较低。
所以当前Full并不在本组主要维度上支配RoutingOnly，不能关闭系统全面
优越性或所有主动管理机制的优越性论证。不得削弱对照来改变这一排序。

三点都超过共同W1预算6756.571050141065 GPU-s，本组不是G2同预算比较。
它们都超过95%不代表实际服务质量相同。不能把别的cap3工作点资源值与
这里的时延拼接，也不能按单轮差异声称显著、稳健或全局最优。

## 2. 路由和主动机制的实际证据

逐请求重算D1129的4000次路由均匹配service_bins；其中3844次有多个
可行副本，2830次所选副本与相同快照的load-only影子不同。该影子只记录
决策，不生成反事实TTFT。

| D1129相同状态中的选择 | 实际service-bin | load-only影子 |
|---|---:|---:|
| GPU tier | 2543 | 1865 |
| HOST tier | 383 | 295 |
| NVMe tier | 942 | 1708 |
| remote tier | 132 | 132 |
| 选择时admitted数量为0 | 426 | 1843 |

实际策略更常选GPU-ready副本，但更少选择admitted为零的副本。这与
locality和当前负载之间的取舍一致；不能仅凭这一相关性将全部尾延迟差
归因给某一条排序规则，尤其自然闭环还改变了激活和服务轨迹。

D1129文件/GPU/HOST主动准备和稳态residency记录均为0；19个activation
全部no_handoff、preparation disabled且无plan SHA。普通按需缓存、合法
HOST复用和物理安全仍在，不能把HOST命中误算为主动准备旁路。

D1129共有19次真实分配并全部释放：1次initial和18次natural scale-out，
18个自然epoch都有实际first dispatch及首token；16次在线scale-down。
ElasticOnly/Full分别有10/19个自然epoch。激活次数不能替代GPU-s积分；
也不能把路由影子当作重演了另一条控制轨迹。

远端次数、线上字节三点相同，不支持本组收益来自减少下载的解释。
请求时打包为false，未事后扣除真实传输或其他请求等待。

## 3. 计量、来源与限制

- 4000个请求的输入/adapter/arrival/prompt/target与同一Full来源匹配；
  ElasticOnly与RoutingOnly的共同输入关系通过已封存的同一Full锚点验证。
- D1129与Full有208个输出digest不同，已完整保留；未预先归因为数值舍入
  或错误adapter。未进行ElasticOnly与RoutingOnly直接输出digest对比。
- 原始n_correct仍为null；4000原生数量和绑定检查不替代独立kernel数值
  正确性证明。GPU-s分母明确为offered，保留开发/已知seed42身份。
- 原生TTFT/TPOT/E2E重算误差均为0；联合3905，其中仅TTFT失败47、仅TPOT
  失败44、两者失败4。单token数0。原始计量和结果未修改。
- 2244个资源样本无high/max/OOM/swap，最小主机可用内存103102689280B。
  本机服务已释放，GPU compute为空，远端专属服务inactive/MainPID0。
- 132条远端journal均匹配本轮时钟，复制前后SHA一致。84个来源SHA、
  144个旧投稿文件、3个用户dirty文件通过核验。

原始结果SHA：
`29b41d3863f513df6856a772c18b0ff75e20ac5705f415b0fb1376df52cf093c`。
分析复用D1113、D299、D455、D1119/D1120比较及D389 first-service语义，
保存于`paper_results/ieee_tc/ablation/d1129_7b_routing_only_w1/`。

## 4. 闭口与唯一下一动作

执行闭口：该唯一运行完整结束，结果与负面取舍封存；不重复同键找赢家。
子结论：路由实际改变了选择，可支持条件资源收益及质量取舍；本组不支持
Full全面最优，也不单独证明confirmed相对last-known信息的贡献。

下一项仍为已批准的7B RoutingHandoff：沿既有handoff与普通按需缓存组合，
检验主动激活准备的增量；先按`ROUTING_HANDOFF_COMPOSITION_REVIEW_20261108.md`
实现受影响边界及针对性测试，再唯一运行。不改论文九公式、评价协议，
不重复Full/ElasticOnly，不因本结果跳回baseline或3B资格。

受控A2/A4、S2/S3、共同预算/规模、其余敏感性和3B同版本缺项继续保留。
完成累积链后按直接机制证据判断收益与代价，再选择有新证据的有界优化，
不把本次结果用于保证下一节点必胜。完整图表、成本/CE/Pareto、两轮审稿
文档与V2交付均未删减；当前不宣告总目标完成。
