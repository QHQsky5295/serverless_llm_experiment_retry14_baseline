# D1127：当前 7B NoHOST W1 单因素消融

2026-11-08，执行前登记。完整 PLAN 及已完成实验闭口台账已核对；此前台账
检查点2657c902已推送并核验。下一主线是既定A1的NoHOST，不重跑Full或外部
基线、不切回3B资格、不操作共享k3s。本页不提前宣称性能结果或备份成功。

## 问题、预期与边界

在相同GPU/NVMe预算、路由、autoscaler、handoff、hierarchy和admission下，
关闭独立受管HOST复用，观察实际来源、准备工作、联合SLO、P95 TTFT和物理
GPU时间如何改变。若独立HOST复用有效，应看到实际HOST来源/最终目标减少，
以及相应准备路径和服务结果的差异；不能仅凭开关或CPU准备条数归因。
收益不预设正负，也不预设最低改善百分比。没有真实远端字节差时，不把
结果解释为减少远端下载。

沿用D1063定义及D1089实际实现：保留GPU-active backing、请求/计划的必要
staging、HOST物理包络、GPU与NVMe容量、page cache及物理安全；不把空出的
HOST预算转给其他层，不关闭整个层级规划，不强制清缓存或附加等待。
NoHOST不是“没有主机内存”。GPU驱逐后的后继来源仍按真实NVMe/Remote计价，
保留精确引用、释放和预算约束。现有路径已在3B运行验证，本轮不改服务代码。

已核对D1063实现边界/结果、D1089结果、现有NoHOST安装与规划测试、相关
98712498/7a96e201及后续公共planner修改历史。此为已批准机制的另一模型
缺项，不是新优化假设；不重复后端资格、profile或整套242项既有检查。
run-experiment用于复用执行入口，ablation-planner用于单因素边界核查；指定
外部评审模型不可用，标记[pending external review]，不将其作为额外门禁。

## 固定参照与执行键

- 模型：Llama-2-7B，vLLM0.30.0，FP16/TP1。
- 配置：`configs/ieee_tc/20261108_d1127_7b_nohost_w1.yaml`。
- 配置SHA：`cf9318a6fc3d4213f15e1e9b893d8515b6d7d96da6139c26c7c49f67fd91d8ac`。
- Full参照固定D1115，原始结果SHA：
  `99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
- 只改变`storage.preloading.independent_host_reuse=false`和同一scenario
  的该开关，以及output/NVMe/HOST三条独占路径；其余与D1100配置完全相同。
- 保持14 slots/seqs、1024 context/batch tokens、0.92 utilization、prefix
  cache false、D362 profiles、27.43356142744625ms bin、min1/max4、5秒需求
  窗口、D1100 predictive controller。九公式不变，不借用cap3资源结果。
- W1沿用source42、4000请求、500逻辑adapter；8×500，相位内到达间隔减半、
  相位间30秒，不等drain，不按Full行为重排；fixed_length_greedy_v1。
- 共同60秒部署通知；真实远端D78/D80已发布gzip，无注入等待、无请求内打包。
  请求引发的读取、传输、竞争与本地物化计时，不新造工件或完整负载。
- 已知开发回放，formal_run=false，单轮；无CI/独立留出或全局最优主张。
- 冻结V1＋单轮V2、D394共同SLO、D594 W1预算6756.571050141065 GPU-s。
  当前PLAN SHA：`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。

## 执行和保护

配置单因素一致性、7B子进程的两个冻结profile绑定、现有NoHOST安装/预算/
backing测试在独立CPU域进行；实际回执另存prelaunch_tests.md，不提前写通过。
之后正常提交/推送显式文件，刷新资源，并只启动一次以下命令；不新增GPU smoke。

tmux名：`tc-d1127-nohost-7b-w1`。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1127_20261108/7b_nohost_w1 \
  d1127_7b_nohost_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1127_20261108/7b_nvme_nohost_w1 \
  /dev/shm/tc-d1127-7b-nohost-w1 \
  configs/ieee_tc/20261108_d1127_7b_nohost_w1.yaml tc_ieee_full
```

三路径必须未用；GPU无compute任务；空服务时MemAvailable≥102GiB；磁盘
≥max(150GiB,100GiB+1.5×预计新增32GiB)。服务72/80GiB、swap2GiB，辅助
3/4GiB、swap0，既有CPU划分和独立watchdog。运行中止线16GiB内存/100GiB
磁盘不变。只启停本项目owned远端服务，正式测量期间不测试、推送、哈希扫描、
压缩清理、改代码/配置或重启远端。现有未用磁盘不为凑清理动作而删除。
用户dirty manifest/状态文件、旧投稿和所有失败证据保护，不暂存无关文件。

## 分析、闭口与下一步

真实终态/GPU释放/远端收尾后，复用D1113 collector、D299 review、D455
stage分解及D1063 NoHOST机制分析方法；本轮与旧Full的不同PLAN绑定明确保存，
不追改旧来源。保存同clock远端journal及既有分析入口所需的parent链接。

逐请求匹配输入、adapter、prompt、arrival、target、生成和原生数量，检查
dispatch tier与物理allocation区间；未知n_correct保持未知，分母用offered
明确标注，不以HTTP成功或数量推定数值正确，同后端输出digest差异完整保留。
NoHOST须有实际owner回执，独立HOST dispatch/acquisition/最终目标为零；
GPU backing/staging另报，非目标路由/handoff/residency/admission保持可执行。
记录实际trigger、来源和准备工作，不仅统计配置字段。

按照最新闭口规则分开判断：执行有效即封存该键；G1是共同最低服务要求下的
资源收益；G2需同工作点在预算内且P95更低、joint不低。少GPU而更高尾延迟
不能单独称系统全面优越。辅助均值/TPOT/E2E披露，不另设全指标第一门禁。
输出完整状态表，保留取舍、失败、未知和不支持范围，不重复同键筛赢家。

结束/分析/封存/备份后依次当前7B CapacityOnly→RoutingOnly→RoutingHandoff，
再原受控A2/A4、S2/S3、M2/S5缺点及3B真正同版本缺项。S1–S13、CE/成本/
Pareto、两轮审稿文档、IEEE单栏图表和V2交付全部保留，目标继续active。
