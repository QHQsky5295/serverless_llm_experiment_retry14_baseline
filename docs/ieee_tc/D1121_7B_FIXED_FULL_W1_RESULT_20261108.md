# D1121：7B 常驻 Full 与弹性 Full 的资源—服务质量取舍

2026-11-08。本轮按登记完成一次 4,000 请求 W1 测量，原始结果、失败计数和
输出差异全部保留。没有重跑 Full、调整正式阈值或修改生产算法。完整项目仍
在执行，本报告只封存一个 A1 对照，不表示全部实验或最优性目标完成。

## 条件和来源

Full 复用 D1115；FixedFull 为 D1121。相同 Llama-2-7B、vLLM 0.30.0、FP16、
TP=1、max4、profile、路由/准备/admission、真实远端已发布 gzip、500 逻辑
adapter、固定生成合同及 seed42 W1。除隔离输出/cache 路径，只改 min_instances
从1到4。源配置逐字段相等检查通过；4,000条输入的 arrival、prompt/token、
adapter、target、内容身份及生成合同匹配。共同60秒部署通知不按 ready 平移。

登记与命令见 [D1121 registration](D1121_7B_FIXED_FULL_W1_REGISTRATION_20261108.md)。
运行检查点 `3cf8d0ab8db8d3a818524857c647b86aafcec73e` 已在启动前推送。
本轮没有改算法代码；分析复用 D1113、D299、D455，并从 D1120 比较脚本与
D1053 四副本检查派生本轮比较脚本。未另建实验框架。

原始 D1121：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1121_20261108/7b_fixed_full_w1`。
结果 SHA：`4fef1c9aba0063e2ac14033235ec4891691ed209c6b7d0fd642a8bc28106a9ea`。
Full D1115 SHA：`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
本轮原始 scenario 仍是 `tc_ieee_full`；展示身份必须为 FixedFull，不并入主表 Full。

## 完整观测结果

| 指标 | 弹性 Full（D1115） | FixedFull（D1121） |
|---|---:|---:|
| 完成且原生合同匹配 | 4,000 | 4,000 |
| TTFT 达标数 | 3,963 | 3,996 |
| TPOT 达标数 | 3,978 | 3,974 |
| 联合达标数 | 3,942 | 3,970 |
| 联合达成率 | 98.55% | 99.25% |
| 生命周期物理 GPU-s | 7,603.874678 | 8,931.497722 |
| GPU-s/offered request | 1.900969 | 2.232874 |
| 平均 TTFT（ms） | 988.900322 | 787.884369 |
| P95 TTFT（ms） | 2,331.329520 | 1,823.352655 |
| P99 TTFT（ms） | 4,090.440717 | 2,774.087010 |
| 平均 TPOT（ms） | 36.318603 | 31.895841 |
| P95 TPOT（ms） | 56.037299 | 50.666838 |
| 平均 E2E（ms） | 5,141.060583 | 4,358.792021 |
| P95 E2E（ms） | 10,955.810484 | 9,343.784872 |
| 远端传输次数 | 132 | 132 |
| 线上字节 | 131,480,060 | 131,480,060 |

冻结 V1 与单轮 V2、D394 长度分组阈值不变。两点均超过95%联合时间要求；
Full 相比 FixedFull 节省 **14.8645% GPU 时间**，但联合达成率低 **0.70个百分点**，
P95 TTFT 高 **27.8595%**。这是同峰值容量、同 adapter 管理下的弹性取舍，
不能称 Full 在所有指标领先，也不能将 FixedFull 的延迟和 Full 的资源拼成一个点。

两点均超过 W1 冻结 G2 预算 6,756.571050 GPU-s，不是预算内尾延迟比较。
D1117 cap3 工作点保持独立，不拿其资源值替换本表。原始 `n_correct=null`
不改写：内容/lease/原生 token 证据通过不等于独立 kernel 数值证明，因此本表
明确使用每 offered 请求，不冒充 per-correct 或无条件的最终合格排名。

## 实际机制而非配置标签

| 机制证据 | Full | FixedFull |
|---|---:|---:|
| initial activation | 1 | 4 |
| natural scale-out activation | 19 | 0 |
| scale-down | 17 | 0 |
| 物理 allocation/release 完整区间 | 20 | 4 |
| 已复算匹配实际选择的 IEEE 路由请求 | 4,000 | 4,000 |
| 文件准备计划 | 1,292 | 1,486 |
| GPU 准备计划 | 1,287 | 1,479 |
| HOST 准备记录 | 1,018 | 1,180 |
| residency epoch | 1,529 | 1,651 |
| 有明确 admission 判定的 attempt | 1,328 | 1,658 |
| 其中有效容量不足而延后 | 1 | 2 |

FixedFull 的四张卡各有一个闭合持有区间，覆盖完整到达窗口；没有自然 resize。
其1,088次控制评估均为 no_action。四个初始 epoch 均有真实 dispatch 和首token，
不冒称自然扩容 first-service 样本。Full 的19个自然 epoch 均有 dispatch/首token，
不为补样本数重跑。FixedFull 的引擎 ready 晚于业务原点，首个请求 TTFT 为
4,562.238484ms；仍按共同原点计量，没有给常驻版本免费预热。

两组均实际使用 `ieee_confirmed/service_bins`，GPU 准备保留 `capacity_only=false`。
历史通用 feature_gates 中 readiness/effective_capacity 两项为false，它们分别
检查旧 `adapter_affinity` 名称和旧配置项，不是此 IEEE 路径的实际执行开关。
本报告依据逐请求 selector 复算与 typed admission receipt 判断，不用旧标签
掩盖或否认已执行路径。源标签原样保留，不追改原始结果。

FixedFull 四个初始 file-handoff 计划保留，但本轮没有初始 GPU-handoff 计划；
其 GPU/HOST 准备均来自稳态 residency。计划存在或 completed 不自动证明
物理拷贝发生、早于 ready 或被 first-service 使用，不据计数宣称这些额外贡献。

## 阶段解释和不可归因部分

物理 GPU-s 的互斥时间窗（Full / FixedFull）：

- 到达前：34.866169 / 142.537134。
- 到达窗口：7,539.095342 / 8,747.968517。
- drain：2.544181 / 2.824355。
- 清理：27.368987 / 38.167715。

各组分项之和等于总量，重叠准备不重复计费。差异主要发生在业务窗口内，
不只是移出或缩短准备窗口。平均 runtime slot 等待为274.531486 / 221.392603ms，
admission→engine dispatch 为239.654052 / 185.531139ms，native decode 为
3,955.958026 / 3,422.873365ms。阶段恒等式最大误差0ms。这些是闭环后果，
不是独立单因素因果分解；不能直接把分项差额都归为某一个缓存机制。

dispatch tier（GPU/HOST/NVMe/Remote）为2590/435/840/135与2756/334/772/138。
不同实际路由形成的条件集合不能当成同请求的因果分组。132次远端传输和字节
完全相同，故本轮不能声称弹性 Full 减少了远端下载。router 的 locality shadow
只作同状态决策诊断，不生成反事实 TTFT。

4,000条相同输入中有180条输出 token digest 不同，逐条保留，未预设为浮点
舍入，也未直接当成错误或删除。这里只证明已核验的合同与绑定，不声称独立
数值正确性。两组都是已知 seed42 上的单轮开发观测，无独立留出、无CI、无
全局最优/跨硬件结论；初始并发、稳态批处理等闭环变化均保留。

## 安全、备份和主线下一步

01:59左右实际结束；service/replay/watchdog返回0，四个lease释放，服务资源域
删除，GPU compute为空，远端inactive/MainPID0。2,279个资源采样，无high/max/
OOM/kill，实验swap为0；最低主机可用101,560,262,656B。请求计时重算误差0ms。
远端同clock journal已保存，两端SHA均为
`e5afe9b1e09d7f8bed0d93f28d227fc6fa1c8a85afe5760ad81761c4b57b0e9d`。
启动前CPU wrapper的既有inotify警告仍见登记，不改写为不存在；本轮运行及
四次CPU分析均正常结束。分析限制3/4GiB、swap0、CPU2,3,26,27，未启动GPU。

产物在 `paper_results/ieee_tc/ablation/d1121_7b_fixed_full_w1/`，包括完整指标表、
机制记录、输入/输出差异、阶段统计和来源manifest。沿用用户绘图延期规则，
先保存数据和表格。封存后的哈希/旧投稿保护检查另存 closure_manifest。

下一项仍为当前7B A1的 LastKnown，其后其余单因素/累积消融、A2–A5、S1–S13、
CE/成本/Pareto、两轮审稿文档和最终图表。外部五baseline矩阵已存在，不重开
HydraServe或ServerlessLLM适配，不回切3B，不重跑本轮或Full，不削弱消融对照。
