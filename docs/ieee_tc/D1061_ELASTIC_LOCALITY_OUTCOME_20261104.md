# D1061：3B W1 普通 locality 的收益及 Full 反证

2026-11-04。按唯一执行键完成原4,000请求；不重跑 Full 或 ElasticOnly。
本项是 PrimeLoRA 复合消融，不是外部 baseline，也不是 Full 新优化。
目前3B A1已完成9/11配置；LastKnown、NoHOST仍待实现和测量。

## 1. 原始观测

三组使用同一 source42/W1 到达、adapter、prompt、target 和生成合同；原生
输出均458,224 token。Full直接复用D1047，ElasticOnly复用D1057封存投影，
本轮只执行D1061。共同D648阈值、D657预算及V1/单轮V2不变。

| 指标 | Full：D1047 | ElasticOnly：D1057 | ElasticLocality：D1061 |
|---|---:|---:|---:|
| 成功终态 / offered | 4000 / 4000 | 4000 / 4000 | 4000 / 4000 |
| 联合TTFT/TPOT达成 | 3836 / 4000 | 3928 / 4000 | 3954 / 4000 |
| 联合达成率 | 95.90% | 98.20% | 98.85% |
| 生命周期物理GPU-s | 6182.715971 | 6032.623108 | 6087.726524 |
| GPU-s / offered-request | 1.545679 | 1.508156 | 1.521932 |
| 平均TTFT（ms） | 454.156048 | 337.218147 | 314.377860 |
| P95 TTFT（ms） | 1114.091001 | 695.139877 | 658.336802 |
| 平均TPOT（ms） | 17.417030 | 15.140558 | 14.595355 |
| P95 TPOT（ms） | 31.774061 | 27.148174 | 24.071394 |
| 平均E2E（ms） | 2323.526111 | 1940.005663 | 1900.053384 |
| 远端获取次数 | 132 | 132 | 132 |
| 线上字节 | 306360162 | 306360162 | 306360162 |
| dispatch GPU / HOST / NVMe / Remote | 3327 / 2 / 538 / 133 | 3188 / 0 / 680 / 132 | 3577 / 0 / 291 / 132 |

本轮TTFT达标3997、TPOT达标3956、联合3954；不能用边际率乘积替代联合。
本轮P95 E2E=4036.745357ms。分位数为Type-1；TTFT、TPOT、E2E重算及阶段
恒等式误差均0ms。全部4,000条固定输出计数与目标一致。

原始 `n_correct=null`、独立非零LoRA数值证明未知、formal ranking=false继续
保留。既有3B池为零更新结构负载，路径/绑定/原生长度合格不等于证明非零
LoRA算术。因此表中明确使用offered分母，不改写为已证明的correct-request。
三组都达到观测时延95%参考点，不据此宣告正式G1/G2或全局最优。

## 2. 对审稿问题的结论

1. 相对ElasticOnly，ElasticLocality的P95 TTFT低5.2943%，联合SLO高0.65
   个百分点，GPU时间高0.9134%。这是普通GPU locality与负载回退的单轮
   资源—延迟取舍，不是全部指标同时优越，也不称显著或稳健改善。
2. Full相对ElasticLocality的P95 TTFT高69.2281%，联合SLO低2.95个百分点，
   GPU时间高1.5603%。相对ElasticOnly的旧反例仍然成立。本轮不能支持
   “Full优于简化管理”或“更高管理复杂度必然带来收益”。
3. 三组远端fetch与线上字节相同，不能把本case差值归因为Full减少远端传输。
   Full较ElasticLocality GPU命中还低250条；命中与开销均需进一步解释。
   该观测不证明所有churn、容量、带宽场景均如此，也不否定尚未完成机制实验。
4. Full平均TTFT高139.778187ms；阶段差为arrival→admission +74.593407、
   admission→acquired +16.401055、executable→engine +19.516981、
   engine-entry→queue +22.423177、native queue +1.602733、prefill
   +5.240836ms。较大差距发生在前端等待与控制/交付路径，而非只在prefill。
   这是定位线索，复合消融还不能单独证明某个planner或状态RPC是唯一根因。

保持九公式、同一后端和评价尺子；不通过取消Full核心思想、削弱对照、隐藏
反例或换阈值取得表面胜出。先补齐已登记信息新鲜度/HOST边界，再综合同版本
单因素与复合证据确定一个可证伪的底层优化方向，不开展盲目参数搜索。

## 3. 机制确实执行了什么

逐请求重建原始不可变快照：D1061全部4000个选择均符合
`Router.select_locality_snapshot(..., slack=1)`，输入与被选副本一致。
3834条请求有多个可行副本；在D1061自己的相同快照上，Full公式shadow与
实际locality选择分歧1723/4000。D1047自身快照上分歧2083/4000。
两者是不同自然轨迹的同状态决策诊断，不生成反事实时延，不合并成独立重复。

D1061全部file/GPU/HOST主动准备和residency epoch计数为0，9个activation
均no_handoff/disabled、plan SHA为空；请求驱动获取、原生cache与物理保护仍在。
Full保留2397个file plans、1267个GPU plans、730个HOST准备、2389个residency
epochs。其1042个主动attempt中707有类型化admission决策且均admit，312为
stale-snapshot；没有观察到软headroom不足。不能仅凭开启标记归因软admission收益。

D1061自然scale-up=8、scale-down=7，加initial共9个物理allocation，全部
release。initial1/natural8均有first-dispatch和首token，歧义0、unused0；
Full为initial1/natural9。均不是controlled；不能替代同plan的A2/A4或为凑
20个样本追加本执行键。first-service选择不筛成功后补选另一请求。

## 4. 输入、输出和资源证据

- Full↔D1061直接逐请求匹配4,000组adapter、canonical/native prompt SHA、
  source item、target/actual、generation和arrival plan；同后端输出hash
  3792组相同、208组不同，全部差异保留。尚不能把这些差异自动定性为错误，
  也不能在没有数值证据时断言只是浮点误差。未过滤差异请求。
- D1057复用时核验原comparison manifest、同一Full指标完全一致及非目标
  配置；通过同一Full输入见证建立三方输入等同。没有声称本脚本直接做过
  ElasticOnly↔ElasticLocality输出hash比较。
- 132个真实远端transfer的ID/字节/hash双边关联通过，无请求中打包；复用
  D80已发布gzip。远端模式为existing_unmonitored，没有连续CPU/干扰监控，
  不能把本机无外部GPU事件扩大为远端全程无干扰保证。
- service/replay/watchdog返回码均0，实际服务域消失，native GPU context
  释放确认。2244个资源采样无foreign compute、escaped worker或保护中止。
  服务内存峰值20286279680B，high/max/OOM/kill/swap均0。
- 两个离线分析域已退出，峰值分别721432576B、636432384B，保护事件/swap0。
  运行缓存已空；147个旧投稿文件和两项用户dirty SHA保持不变。
- 现有共享k3s未修改；HydraServe独立rootless适配及D807–D809结果复用，
  pp1观测不冒称pipeline/consolidation证明，不重新开展基础适配。

## 5. 来源、备份与下一项

执行提交：`509a4c29fde27bfb5e986e4960242ca0d1a544be`。
配置SHA：`6c5cbc48cb6e03da94010a429bb40dc46c45cee965e6410d6b0f24c3553557cc`。
raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1061_20261104/`。
curated：`paper_results/ieee_tc/ablation/d1061_3b_elastic_locality_w1/`，含completion、
comparison、原始投影、逐请求stage和mechanism、执行脚本snapshot及SHA manifest。

本次没有新生成权重/trace，没有改生产源码或论文，没有新绘图。按单轮V2保存
全部观测，不做CI或重复寻优。随后以显式路径备份，成功提交/远端指针以raw的
`backup_completion.json`为准；独立backup不等于已整合FaaSLoRA V2分歧。

下一是D1062 LastKnown：只改变router看到的来源信息，实际owner/residency/
admission和selected-source revalidation不变，最小CPU验证后一次完整3B W1。
NoHOST、7B A1与两累积节点、受控A2–A5、M2工作点、S1–S13、CE/Pareto与
计费敏感性、两轮审稿文档/公式解释、IEEE图表和最终V2集成全部保持未完成。
存储按既有明确关闭路径无损归档，不与GPU测量并行，不重开全盘扫描。
