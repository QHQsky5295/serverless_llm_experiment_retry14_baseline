# D1129：当前7B RoutingOnly W1累积消融

2026-11-08，执行前登记；尚未启动GPU回放。D1128准入消融已封存并推送
e83ab2e3abb9eeaa1779b33a601eed7b46290920。本项推进已批准PLAN §8 A5的
第一个缺失中间节点，不重做Full、ElasticOnly或任何外部baseline。

## 科学问题与对照

在同一扩缩容、按需缓存、物理资源和后端下，仅加入IEEE确认状态的服务
时间分桶路由，能否改变请求排队/选中层级，并改善资源—质量关系？
第一差值是RoutingOnly与D1119 ElasticOnly；同时列D1115 Full，后续和
RoutingHandoff、D1128 CapacityOnly组成完整累积链。它不是外部baseline，
也不单独证明confirmed对last-known信息新鲜度的贡献；该问题仍归A3/S2。

若路由起作用，需观察真实选择分歧、选中副本tier与队列分布，再检查GPU-s、
联合SLO和P95 TTFT/TPOT。收益不是配置开关存在的必然结果；不预设幅度，
不加等待或减少对照缓存/资源来获得排序。shadow仅比较同状态决策，不
生成反事实延迟。累积链次序的增量不等于与其他机制无关的普适贡献。

参照复用：

- D1119 ElasticOnly：4000原生合同、联合99.675%、U8575.626704086084、
  P95 TTFT1035.5770019814372ms；结果SHA
  `9d06cf7d69ff99085fa6d3f6f9902d4b5ccbb5b22edfe216871cd6576c542ca3`。
- D1115 Full：4000原生合同、联合98.55%、U7603.874677798245、
  P95 TTFT2331.3295198604465ms；结果SHA
  `99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
- 两者已呈资源—质量取舍，不能因本项尚未完成重新选择较有利的Full。
  不混入D1117 cap3工作点，不将旧3B数值作为本项参照。

## 唯一功能变化与实现边界

以D1119配置为源，只将`ieee_selection_strategy: load_only`改成既有
`service_bins`，准确登记`routing_only_v1`/`tc_ieee_routing_only`，另改
三个独占输出/NVMe/HOST路径。配置SHA为
`279e9df7382afa5d4cb73e3795eb3b1f5ac6690db87e73154ef6c1f19b825e8c`。

- `demand_lru`、no_handoff、禁用稳态hierarchy/主动软admission/forwarding
  保持不变。原生按需加载、LRU、必要staging及合法复用继续保留。
- 相同严格可行集、原子reservation、引用/取消、物理保护和稳定tie-break；
  只恢复式(2)/(3)对应的服务分桶排序，不改变九公式或Full语义。
- serving代码只扩展既有组合身份和scenario映射；不更改selector公式、
  autoscaler、缓存owner、准备器或vLLM。旧Full/ElasticOnly/ElasticLocality
  行为保持；原D1100 runner显式scenario接口直接可用，无新启动框架。
- 与ElasticOnly归一化后的完整配置相等，检查覆盖所有非目标参数，不仅
  手挑几个字段。实际机制计数须回放后验证，配置检查不冒称性能证据。

公共条件：Llama-2-7B、vLLM0.30.0 FP16/TP1、14slots/seqs、1024context/
batch、0.92utilization、prefix=false、D362 profiles、D1100 predictive
autoscaler、min1/max4、5秒需求窗口、27.43356142744625ms service bin。
复用source42的4000请求/500逻辑adapter W1，8×500phase、相位内间隔
减半、相位间30秒且不等待drain；固定greedy原生token合同、共同60秒
部署通知。无新增工件/trace，真实远端仍用D78/D80预发布只读gzip。

## 历史、文献与有限检查

已核对D1049入口复核、D1057实现及其负结果、D1119实现/结果、当前
variant contract、demand-LRU owner、实际launch、路由fixture及相关git
修改历史。没有恢复已被否决的legacy路由或把RoutingOnly改名成Full。

再次核查[vLLM0.30.0官方LoRA按需LRU源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
关闭主动管理不应取消正常缓存。[dLoRA作者提供原文](https://xinjin.github.io/files/OSDI24_dLoRA.pdf)
说明request/adapter分布和负载平衡相互影响，因此需固定缓存/弹性条件
观察路由增量；它不证明本项必然获益，也不是PrimeLoRA式(3)的来源。
USENIX PDF本次访问403，作者PDF成功；未使用打不开的链接推断内容。

ablation-planner指定评审模型当前不可用，采用本地已批准方案可行性复核，
标记`[pending external review]`；不是独立审阅，不增实验门槛或改矩阵。
32项针对性CPU检查已通过：新配置单因素、实际service-bin/文件owner与
scenario、旧组合/Full回归、同快照服务分桶、load-only、可行集及pure
shadow。日志与实际CPU域收尾另记prelaunch_tests.md。不新做无理由的
GPU smoke、500-ID资格或Full重跑；没有独立数值正确性结论。

## 唯一执行键与保护

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1129_20261108/7b_routing_only_w1 \
  d1129_7b_routing_only_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1129_20261108/7b_nvme_routing_only_w1 \
  /dev/shm/tc-d1129-7b-routing-only-w1 \
  configs/ieee_tc/20261108_d1129_7b_routing_only_w1.yaml tc_ieee_routing_only
```

tmux `tc-d1129-routing-only-7b-w1`。启动前显式备份、检查三路径未用，GPU
无compute、空服务MemAvailable≥102GiB、磁盘≥150GiB；运行保护仍为
服务72/80GiB swap2GiB、辅助3/4GiB swap0、独立watchdog与原CPU隔离，
运行停止线16GiB内存/100GiB磁盘。只管理本项目远端服务的Invocation，
不改共享k3s。测量期间不测试/压缩/推送/改代码或远端。旧投稿144文件、
三个用户dirty文件保护，源manifest与raw结果不得覆盖。

## 收尾、解释与后续

终态和实际释放后复用D1113/D299/D455显式scenario接口、D1119/D1120
比较方法；先保存同clock远端journal与parent链接。4000实际选择应符合
service_bins；handoff/稳态主动file/GPU/HOST计划应为0，按需活动不误算
主动准备。保留自然激活/first-service、真实资源积分、dispatch tier、
阶段分解、完整输入合同和输出digest差异。缺失与未知不填造。

指标沿冻结V1、单轮V2、D394 SLO、D594 W1预算6756.571050141065GPU-s。
PLAN SHA `b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
source42开发身份/单轮保留；n_correct未知则null，GPU-s分母明确offered。
全部系统/变体都展示，不以95%筛行；G1条件资源和G2同预算质量分开判断。
较少资源且更差尾延迟是取舍，不是系统全面胜出。首个完整结果封存，
不按输赢重跑同键。下一项RoutingHandoff需明确handoff-only与demand-LRU
的组合，不能简单关闭owner检查或恢复全部稳态hierarchy。后续受控A2/A4、
S2/S3、M2/S5、剩余敏感性、3B同版本缺项、成本/CE/Pareto、审稿文档与
IEEE图表全部保留；本项不关闭总目标。
