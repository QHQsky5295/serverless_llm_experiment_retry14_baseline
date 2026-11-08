# D1128：当前 7B CapacityOnly W1 单因素消融

2026-11-08，执行前登记。承接已完成并备份的D1127；当前7B A1尚缺此项，
随后才是RoutingOnly和RoutingHandoff。复用D1115 Full，不新增Full或baseline
回放，不重开HydraServe/共享k3s适配。目标整体继续active。

## 问题与单因素边界

保持确认状态路由、handoff、层级驻留、HOST复用、后端、物理容量、替换策略
及autoscaler不变，只关闭主动准备的软准入E(t)判断。问题是：其实际放行/
延后能否减少加载争用，从而改善共同服务要求下的物理GPU占用及尾延迟？
如果有贡献，应先看到实际准入决策变化，再检查完整工作负载的服务和资源
后果；不预设效应方向或最低百分比。未触发、负结果和取舍都保存。

现有实现已有完整开关，本轮不改生产代码。唯一功能差异为
`resource_coordination.coordination_enabled=false`；另改output/NVMe/HOST
三个独占路径。initial/natural-scaleout/steady-residency计划均透传
`capacity_only=not self._coordination_enabled`。保留native slot、KV、工作
空间、物理headroom、引用/租约和实际释放检查，不关闭物理安全，不给对照
增加等待或减少容量。需求驱动加载保留；CapacityOnly不是无约束加载。

已核对D1052的最小接线提交69cfb9e940bbd5ed4f9fdb30f89d49cd0bd50e97、
当前三个准备入口及既有测试。旧3B D1052没有观测到软余量不足，不能作为
7B收益证据或因果结论；只复用它的测试/分析方法，配置始终来自当前7B。
相关官方物理可行性依据沿用已核对的[vLLM 0.30.0 scheduler源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)。
此次不是新优化，未修改九公式/控制逻辑或重新估计profile。

采用ablation-planner核查单因素，run-experiment/gpu-status复用既有执行与
资源流程；指定评审模型不可用，标记[pending external review]，不新增实验
门禁、不声称完成独立评审。github-sync用于启动前显式备份。

## 冻结执行身份

- Llama-2-7B，vLLM0.30.0，FP16/TP1；14 slots/seqs、1024 context/batch、
  0.92 utilization、prefix caching=false；D362 profiles和D1100 predictive
  controller、min1/max4、5秒需求窗口、27.43356142744625ms service bin不变。
- 配置：`configs/ieee_tc/20261108_d1128_7b_capacity_only_w1.yaml`；SHA
  `e136c37406075476fc1f14b5b0b37d49df3f835277aeb8bb8d21623361fa44a2`。
- 参照配置D1100 SHA
  `386a34c84ec70e5d0d366f6d3bcaf0891536be503ba1359d583a93bd5ecf303f`；
  Full固定D1115 W1，结果SHA
  `99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
- 既有source42、4000请求/500逻辑adapter，W1的8×500 phases、相位内
  到达间隔减半、相位间30秒，不等待drain；相同输入/adapter/target及
  fixed_length_greedy_v1，不生成新工件或完整负载。
- 共同60秒部署通知；真实远端既有D78/D80只读gzip交付，无请求内打包。
  读取、传输竞争、本地物化均保留计时。仅启停本项目owned服务。
- 冻结V1+单轮V2、D394共同SLO、D594 W1预算6756.571050141065 GPU-s；
  当前PLAN SHA
  `b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
- raw scenario仍为tc_ieee_full以复用入口，展示身份必须为CapacityOnly。
  已知开发回放formal_run=false，不改称独立留出，不计算运行间CI。

## 执行与保护

现有CPU测试覆盖配置单因素、initial/natural入口、软策略/相同victim与
物理保护。检查回执单独保存，不新做GPU smoke或全量后端资格。启动前
核查旧投稿144文件、用户dirty文件、Full及输入来源；正常提交/推送显式
文件后刷新资源，只启动一个新的执行键。命令：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1128_20261108/7b_capacity_only_w1 \
  d1128_7b_capacity_only_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1128_20261108/7b_nvme_capacity_only_w1 \
  /dev/shm/tc-d1128-7b-capacity-only-w1 \
  configs/ieee_tc/20261108_d1128_7b_capacity_only_w1.yaml tc_ieee_full
```

tmux为tc-d1128-capacity-only-7b-w1。三路径必须未用，GPU无compute，空服务
时MemAvailable≥102GiB，磁盘≥max(150GiB,100GiB+1.5×32GiB)。保持服务
72/80GiB、swap2GiB及辅助3/4GiB、swap0的既有CPU划分和独立watchdog；
运行停止线为16GiB内存/100GiB磁盘。测量期间不测试、推送、压缩、哈希
扫描、改代码/配置或重启远端。保护用户manifest、两份dirty状态文件及
旧投稿，不stage无关文件、不删除唯一原始证据。

## 分析、闭口与剩余范围

实际终态/GPU释放/远端收尾后，复用D1113 collector、D299 review、D455
阶段分解、D1127当前7B比较和D1052 admission提取方法，不运行硬编码旧
PLAN/旧Full的封存脚本。保存同clock远端journal及parent链接。

逐请求核对输入/adapter/arrival/prompt/target/生成和原生token合同、输出
digest差异、dispatch tier与实际allocation/release。n_correct未知仍为
null，分母明确offered；不能以HTTP或数量当数值正确性。actual plans的
capacity_only应为true；分别记录typed admit/defer、资源不足、stale、
复用、取消及缺回执。缺回执不等于延后或成功加载。Full使用同一提取法，
实际未触发就不归因；自然闭环轨迹不同，不能冒充同候选受控S3。

执行完成即封存该键；G1资源优势、G2预算内P95/joint优势和机制因果解释
分别判断。较少GPU但较高尾延迟只能表述条件收益/取舍，不能当全面胜出。
保存平均/尾部TTFT、TPOT、E2E与完整阶段表，不增全辅助指标第一门禁，不
反复同键找赢家。图表按最新要求延后。

后续当前7B RoutingOnly→RoutingHandoff，再受控A2/A4、S2/S3、M2/S5必要
点及3B真正同版本缺项。全部A1–A5/S1–S13、CE/成本/Pareto、两轮审稿文档、
IEEE图表和V2交付保持，不因本项完成而关闭总目标。
