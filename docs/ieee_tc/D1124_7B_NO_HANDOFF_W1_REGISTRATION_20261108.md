# D1124：当前7B NoHandoff W1单因素登记

2026-11-08，性能运行前登记，尚未启动。承接已封存并推送的D1123，继续既定
A1主线，不重新复现baseline，不切换3B，不重跑Full。Full唯一参照为D1115。
使用ablation-planner核对单因素与可行性；指定评审模型不可用，标记
`[pending external review]`，不冒称独立审阅完成。

## 问题、预期与优先级

检验activating replica的主动准备是否影响新副本首服务、完整服务质量与
GPU生命周期。若handoff有贡献，应看到Full真实提前工作与后续使用，以及
完整资源/延迟差异，而非只统计非空plan或成功扩容。结果允许无效、退化或
trade-off，不能改变准入、请求负载或其他机制以确保Full获胜。

优先级1，现有W1一次4,000请求；预计约40分钟、2–3 GPU-hours，仅为排程
估计。复用已有开关，不新增生产机制或额外GPU smoke。结束并分析封存后
再做R0H0，构成已批准的四格；其后DelayedHandoff/NoHOST/CapacityOnly及
累积节点、A2–A5和S1–S13仍保留，不静默缩减范围。

## 唯一差异与非目标保留

新配置：`configs/ieee_tc/20261108_d1124_7b_no_handoff_w1.yaml`。
SHA：`c1cef50fe9f37eb8b7a2888b993bab961671f48027e58a3d14fb86934098b5b2`。
相对D1100/当前Full，仅：

- `ieee_handoff_policy: full → no_handoff`；
- output、NVMe、HOST三条独占路径。

保留Full service-bin排序、确认来源缓存、可行集与预留、当前需求/已知历史、
请求驱动获取、ready后的层级驻留、主动GPU软准入和物理安全。不会清每阶段
缓存，不改变min1/max4、预测/反应式控制或让NoHandoff使用legacy入口。
共用观测/profile；自然闭环反馈可以改变激活、队列和后续准备目标，因此
不能将本项称为同计划受控A2/A4，亦不能生成反事实TTFT。

scenario保持`tc_ieee_full`，外部变体标NoHandoff，不混入主表canonical Full。
Llama-2-7B、vLLM0.30.0、FP16/TP1、14slots/seqs、1024上下文/batch tokens、
GPU utilization .92、prefix cache false、D362 profiles、service bin
27.43356142744625ms、5秒需求窗口及完整D1100控制配置不变。

输入仍为既有seed42/500逻辑adapter/4,000请求，W1 8×500/30秒间隔/相位内
到达压缩一半，60秒共同部署通知及fixed_length_greedy_v1。真实已发布gzip
HTTP、无注入等待，不重建工件或负载。保留exploratory/formal_run=false。
冻结V1+单轮V2、D394共同SLO和D594预算6,756.571050GPU-s不变。

## 历史、源码与外部依据

复用D1050已存在的NoHandoff语义和CPU测试，不复用其3B性能或profile。
历史D1050中Full相对NoHandoff资源和TTFT更差，联合SLO仅高0.05个百分点；
该负结果保留，不将本次登记写成“必然改善”。近期fdeea80f/00f2a873的
零收益/零需求规划优化在两侧共享，不重新开启已否定优化方向。

源码核查：`scripts/run_all_experiments.py`中activation仅在policy不是
no_handoff时创建handoff计划；无计划仍正常启动engine，准备状态为disabled。
现有ready稳态路径继续调用`_schedule_ieee_residency_epochs`，不因关闭handoff
而同时关闭hierarchy。生产源码不改；新测试只断言全部非目标配置相等。

2026-11-08核查[vLLM官方LoRA文档](https://docs.vllm.ai/en/stable/features/lora/)：
运行时动态加载/按请求解析与启动时加载是可区分能力。它支持保留请求驱动
路径的对照边界，不证明本系统的提前准备一定有收益；本机仍冻结0.30.0，
不因在线文档更新更换后端或引入新插件。部署限受信隔离实验环境。
HydraServe官方仓库仅作启动优化背景核查，不重开其Kubernetes部署或改共享集群。

## 最小检查与执行

四项CPU检查：非目标配置完全相等、无activating准备入口、关闭预加载时
仍正常初始激活、NoHandoff保留ready驻留且不重复epoch。结果另记prelaunch。
CPU2,3,26,27，MemoryHigh3GiB/Max4GiB/swap0，隐藏GPU；不重做未改变的资格。

四项断言全部通过（16.444s），主进程status0；run-u373的systemd wrapper却因
stop-sigterm超时返回1（125.402s），不能标为生命周期全部通过。已观察MainPID0、
资源域为空，随后unit收回。journal记录control/memory inotify ENOSPC，属于
D1121/D458已有的监听额度限制，不能仅凭该回执断言fixture泄漏、磁盘不足或
OOM。完整日志保留；未修改测试fixture、生产代码、内核上限或其他进程。
沿用D458已验证的独立watchdog直接读取cgroup/memory.events/NVML及显式空域
收尾；实际启动仍核对首次样本。告警不被抹去，也不声称性能影响恒为零。

性能运行前先备份，刷新实际GPU空闲、可用内存102GiB和磁盘150GiB门槛。
服务72/80GiB/swap2，辅助3/4GiB/swap0、独立watchdog和16/100GiB停止线保持。
已有runner管理仅属于本实验的远端服务；不重复发布缓存、不改共享k3s。
唯一tmux：`tc-d1124-no-handoff-7b-w1`。下方是登记命令，不代表已经执行：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1124_20261108/7b_no_handoff_w1 \
  d1124_7b_no_handoff_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1124_20261108/7b_nvme_no_handoff_w1 \
  /dev/shm/tc-d1124-7b-no-handoff-w1 \
  configs/ieee_tc/20261108_d1124_7b_no_handoff_w1.yaml tc_ieee_full
```

仅在目录/执行键未用时启动一次；随后记录PID、scope/InvocationID、remote
clock和真实回执，禁止重复启动。测量期间只轻量监测，不推送、维护、构建、
哈希扫描或运行第二项实验；不以阶段性胜负决定停止或重复。

## 收尾证据与下一项

终态和实际释放后复用D1113/D299/D455及D1050机制分析，收集同clock远端journal。
检查4,000完整输入/原生输出/adapter内容/lease/时间线及全部dispatch tier。
报告绝对资源、联合与边际SLO、TTFT/TPOT/E2E、互斥GPU窗口和真实传输。

NoHandoff每个activation应policy=no_handoff、plan_sha=null、preparation disabled；
file/GPU/HOST的handoff记录应为0，ready后的驻留与准入按实际触发记录。
Full需区分计划、实际ready前完成、first-dispatch前完成及使用；不能只以
计划存在归因。first-service绑定最早实际dispatch，initial/natural/controlled
分开，unused/失败保留，不为凑20样本追加同配置运行。

同后端token digest差异须原样保留审计；独立数值原因未知仍未知，n_correct
不补造。单轮描述，无CI/显著性。完成状态表和解释、保护核对、封存备份后
进入当前7B R0H0；不重复本项/Full/基线，不回到无依据的局部调参。
两轮审稿设计文档、CE/成本/Pareto、IEEE图表和最终V2交付仍待完成。

PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
不stage用户manifest、已有dirty状态全文或其他未关联文件。
