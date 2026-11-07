# D1122：7B 路由来源信息新鲜度的单因素对照

2026-11-08。既定 A1/A3 的一次 W1 测量已完整完成并释放资源。本报告不表示
整个项目完成，也不把小幅单轮差异写成显著性或全局最优。Full 保持主表同源
D1115；没有另跑 Full、改变指标、调低对照或重放至获胜。

## 条件与证据身份

Llama-2-7B、vLLM 0.30.0、FP16/TP1、min1/max4；既有500逻辑adapter集合、
seed42源trace、W1的8×500阶段及30秒间隔、60秒共同部署通知、固定输出greedy。
真实HTTP、已发布只读gzip、两端1Gbps/full、无人工等待。除隔离路径外，
LastKnown仅新增0.5秒请求触发的路由来源sample-and-hold；非目标配置完全相等。
所有4,000条arrival、prompt、adapter、target、内容身份及生成合同匹配。

Full是具有确认来源的历史缓存观测，不是全局瞬时物理真值。LastKnown仅保持
排名类中的tier/representation/footprint；可行性、Q、pending、utilization、
EWMA、owner、registry、planner/admission和实际选中来源保护保持live。两侧
都有真实层级缓存，旧排名视图不授予执行引用、不改写dispatch tier。

启动前检查点 `59409b6391ae15f397bd3b2c06082e5d73285ffd` 已推送V2。
运行登记见 [D1122 registration](D1122_7B_LASTKNOWN_W1_REGISTRATION_20261108.md)。
原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1122_20261108/7b_last_known_w1`。
结果SHA：`1af9eda45a488c5815862f5ac439f968c33375f833d7c31a82d74ce0a6c3d047`。
Full D1115 SHA：`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
本轮原始scenario仍为tc_ieee_full，展示必须标LastKnown，不能并入主表Full。

## 完整结果

| 指标 | Full（D1115） | LastKnown（D1122） |
|---|---:|---:|
| 完成且原生合同匹配 | 4,000 | 4,000 |
| TTFT达标数 | 3,963 | 3,967 |
| TPOT达标数 | 3,978 | 3,960 |
| 联合达标数 | 3,942 | 3,929 |
| 联合达成率 | 98.55% | 98.225% |
| 生命周期物理GPU-s | 7,603.874678 | 7,898.095640 |
| GPU-s/offered request | 1.900969 | 1.974524 |
| 平均TTFT（ms） | 988.900322 | 949.563493 |
| P95 TTFT（ms） | 2,331.329520 | 2,308.962506 |
| P99 TTFT（ms） | 4,090.440717 | 3,714.024665 |
| 平均TPOT（ms） | 36.318603 | 35.694602 |
| P95 TPOT（ms） | 56.037299 | 56.431541 |
| 平均E2E（ms） | 5,141.060583 | 4,989.729017 |
| 远端传输次数 | 132 | 132 |
| 线上字节 | 131,480,060 | 131,480,060 |

Full观测GPU时间低3.7252%、联合达成率高0.325个百分点，但P95 TTFT高0.9687%，
平均TTFT/E2E也更高。两者都超过95%联合时间要求；不能据此宣称Full全面优越。
两点都超过冻结W1 G2预算6,756.571050 GPU-s，不是预算内尾延迟比较。
cap3结果D1117独立保留，不拿其资源值与本表延迟拼接。

4,000条同输入中有178条输出token digest不同，逐条保留，原因尚无独立数值
证据。原始n_correct=null不改写；绑定、内容与原生输出合同不等于独立kernel
数值正确性证明，因此使用per-offered而非伪造per-correct合格排名。
两组为已知seed42上的单轮开发观测，不声称独立留出、CI或可重复性证明。

## 信息变量确实触发，但影响范围有限

4,000条最终请求的LastKnown实际选择全部通过既有D297专用重算；不能拿收到
的原始视图直接复算旧视图选择。全部最终事件在不可变journal中唯一对应。

- 共4,107次路由事件；含107次先前尝试，不能当作额外请求。
- 4,000次最终选择中29次与同状态下收到视图的shadow不同，占0.725%。
- 12,990个可行候选观测中87个来源类不同；全journal中bootstrap/refresh/hold
  分别为859/11,063/1,394，另有163个不可行候选。
- controller保持年龄均值0.022357秒、P95 0.207734秒、最大0.497784秒。
- 底层来源观测年龄均值3.126098秒、P95 13.722858秒、最大55.243057秒。
  因此0.5秒不是物理状态陈旧程度上界，不能混称为原生广播周期。
- 42次排名认为GPU而后续获取不是GPU，36次相反。这是选中副本跨时刻的不一致，
  不等于同时刻全局false-ready/missed-ready，更不能全部归因于保持视图。
  未选中候选没有同时刻权威物理真值；shadow不生成反事实TTFT。

这项实验区分了“有层级缓存”与“路由所用信息的额外新鲜度限制”，但目前仅
表明该替代信息设计会改变少量选择，并呈现有限的资源—延迟取舍。不能据此
断言其他论文系统均缺少状态传播、该机制解释全部收益，或0.5秒是最优周期。

## 非目标机制、first-service和资源分解

| 实际机制 | Full | LastKnown |
|---|---:|---:|
| initial activation | 1 | 1 |
| natural-scaleout activation记录 | 19 | 16 |
| 完成scale-up记录 | 19 | 15 |
| scale-down | 17 | 14 |
| 闭合物理持有区间数 | 20 | 17 |
| 文件准备计划 | 1,292 | 1,370 |
| GPU准备计划 | 1,287 | 1,364 |
| HOST准备记录 | 1,018 | 1,016 |
| residency epoch | 1,529 | 1,601 |
| 明确admission判定attempt | 1,328 | 1,383 |
| 其中有效容量不足而defer | 1 | 3 |

LastKnown GPU准备保持capacity_only=false，判定中1,380次admit、3次defer；
17个file-handoff、15个GPU-handoff及118个HOST handoff记录实际存在。
这些计数证明路径仍启用，不自动证明每个计划执行了物理复制或被first-service使用。
旧通用feature_gates中的false标签不能替代此IEEE路径实际selector及typed receipt。

LastKnown有16个自然activation，其中14个有dispatch及首token；1个ready但未用，
最后1个在回放结束附近触发后取消、无engine-ready。其资源没有扣掉，不把15个
完成scale-up与16个触发混写；没有为凑20个样本追加运行。Full的19个自然epoch
均有dispatch和首token。initial/controlled与自然样本严格分开。

互斥GPU-s时间窗（Full / LastKnown）：

- 到达前：34.866169 / 35.935647。
- 到达窗口：7,539.095342 / 7,735.573660。
- drain：2.544181 / 5.244293。
- 清理：27.368987 / 121.342040。

总差294.220962 GPU-s中，196.478318来自到达窗口，93.973054来自清理；不能把
全部资源差额写成信息传播的直接收益。重叠准备不重复累计，也不事后去掉收尾。
平均runtime slot等待274.531486/255.848223ms；admission至engine dispatch
239.654052/227.630277ms；native decode 3,955.958026/3,839.760459ms。
阶段恒等式最大误差0ms；这些为闭环观测，不是独立分项因果效应。
dispatch GPU/HOST/NVMe/Remote分别为2590/435/840/135和2581/445/839/135。
下载次数、字节相等，不能以减少下载解释本项结果。

## 收尾、复用与下一步

02:57左右完成，service/replay/watchdog均返回0；17个lease全部释放，GPU compute
为空，服务资源域删除，远端inactive/MainPID0。2,315个资源采样无high/max/OOM，
实验swap为0；最低主机可用101,076,148,224B。远端同clock日志双边SHA一致：
`a1358eacda9cae238a88c4daea405faed80af351e9b2de06eee6bbbf0d673453`。
132次服务端传输中130次由最终请求直接关联，另外2次未计作丢失或删除；完整
交付关联检查保存在本组delivery_linkage中。启动前短暂健康连接失败已保留，
服务ready后才发部署通知，没有请求fallback或测量期间重启远端。

分析复用D1113/D299/D455及D1062 LastKnown回执，7B比较由D1121脚本小幅派生，
没有改生产代码。CPU分析限3/4GiB、swap0、CPU2,3,26,27；启动前20项定向测试
仍适用。按用户指令先保存数据表，不为展示延误主线。结果技能用于分离观测、
解释及边界；experiment-audit指定的gpt-5.6-sol不可用，独立语义审计状态为
unavailable/pending，不将执行者校验冒称独立通过。备份检查另存closure_manifest。

完整产物：`paper_results/ieee_tc/ablation/d1122_7b_last_known_w1/`。
下一项为当前7B A1 NoRouting，其后其余单因素/累积消融、A2–A5、S1–S13、
CE/成本/Pareto、两轮审稿设计文档与最终图表。五个外部baseline的两模型矩阵
已存在，不重开HydraServe或ServerlessLLM适配、不改共享k3s、不重复本轮或Full。
