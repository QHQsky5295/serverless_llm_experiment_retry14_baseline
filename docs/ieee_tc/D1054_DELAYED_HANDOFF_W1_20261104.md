# D1054：3B W1 DelayedHandoff，自然闭环的准备时机消融

2026-11-04。上一 goal turn 为 PROGRESS：D1053 FixedFull 已完成、分析并以
e214a8d028febb23a288e8b103b8d82d5213c258 备份；其资源—服务取舍保留。
当前沿 A1 主线，不重跑 Full、不回到基线无限适配或性能调参。完整目标 active。

## 研究问题和唯一差异

将同一 activation 的准备任务推迟到 engine-ready 后才执行，保留原有候选
选择、预算、routing、hierarchy、软 admission、autoscaler 和物理保护。
检验提前准备时机是否影响自然扩容的 first-service、完整联合时延和 GPU 时间。
不预设 Full 胜出，不以计划数量代替真实提前执行或性能收益。

从 D1047 Full 派生配置，唯一非路径差异为
`resource_coordination.ieee_handoff_policy: full → delayed`。实例范围仍 1–3；
不沿用 FixedFull 的四实例边界。保留 D1031 profiles、D747 fresh descriptor、
vLLM 0.30.0、FP16/TP1、32 slots、c8、4096 batch tokens、1024 context、
GPU utilization .72，以及到达驱动的已确认来源预取。

代码路径已经存在：activation 冻结计划后同时启动 runtime 和 preparation；
delayed preparation 在实际文件/GPU movement 之前等待 activation-ready。
等待期间不重新采样 h/d、选择集或计划预算。ready 后仍执行普通实时安全校验，
不绕过 owner、reservation、reference 或 physical-capacity 保护。

这是 A1 自然闭环单轮，不是 A2/A4 受控同计划实验。不同运行的反馈会使扩容
时机、需求快照和准备目标变化，跨运行计划 hash 不要求相同；受控实验仍待做。
文件准备 record 的 started_at 是计划登记时刻，**不是物理复制开始时间**。
收尾审计必须采用实际 movement/transfer/native 回执，不据 started_at 判
delayed 提前执行，也不把 engine-ready、first-request TTFT、activation-to-token
三个量混称启动延迟。

## 已通过的有限检查

复用 D1049 入口审阅和现有两项 CPU 测试，2/2 通过（9.324549 秒）：

1. Delayed 保留原 activation epoch，ready 前没有文件复制，ready 后按原计划执行。
2. Full 文件复制确实可以先于 runtime-ready，HOST allowance adoption 无预算空隙。

测试只使用真实本地文件 owner/队列及模拟 engine，不是 GPU 性能结果。
没有修改生产代码或测试 fixture；测试峰值 626,331,648 B，high/max/OOM/swap
均 0。现有初始池/自然控制/生命周期检查按原版本复用，不增加 blanket smoke。

curate 已逐字段验证与 Full 仅上述一项策略及三条私有路径不同。源 trace 与
W1 view SHA、4,000 条输入、Full 来源、生产源码、147 个旧投稿文件及两项
用户 dirty SHA 均核验。远端现有 3B 只读 gzip 服务健康，不重启、不发布新池。

PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
指标沿用冻结 V1 + 单轮 V2、D648 共同 SLO、D657 W1 预算，不改阈值和分母。
原生生成 fixed_length_greedy_v1，源 seed42 是已知开发负载，不包装成留出。
仍保留 n_correct=null、independent_numerical_proof=false 和正式排名资格未知；
3B 全零更新池不证明非零 LoRA 运算，同后端输出摘要变化另行记录。

## 唯一执行键

配置：`configs/ieee_tc/20261104_d1054_3b_delayed_handoff_w1.yaml`。
原始根：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1054_20261104/`。
整理结果：`paper_results/ieee_tc/ablation/d1054_3b_delayed_handoff_w1/`。
prelaunch/ 已保存测试全文、配置差异、健康状态、来源与 SHA；不覆盖历史结果。

复用原 runner 和 tmux，指定：

```text
FAASLORA_D1000_RUN_ID=d1054_20261104
FAASLORA_D1000_RUN_TAG=d1054_3b_delayed_handoff_w1
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1054_3b_delayed_handoff_w1.yaml
FAASLORA_D1000_WORKLOAD=W1
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

完整 4,000 请求，W1 8×500、phase 间 30 秒、共同部署通知提前 60 秒。
无 prefix 诊断、无详细 profiling、无新权重/负载/数据池。原始 scenario
仍 tc_ieee_full，外部显示 DelayedHandoff，不混进 canonical Full 来源。
正式开始前仍须由现有 launcher 验证磁盘 150 GiB、主机余量 102 GiB、
GPU 空闲和实际 72/80 GiB 服务资源域；停止线 100 GiB/16 GiB 不变。
一次一个重型任务，不与压缩、构建、推送或远端维护重叠。

## 收尾与后续范围

本文件登记时尚未启动 GPU，不预填性能结果。完成后先确认实际释放，再复用
D1053 collector、D1050/D1053 的匹配输入/共同 SLO/生命周期/first-service
分析。核对所有 initial/natural activation 的 delayed policy、实际移动时序、
ready 后 residency、软 admission、远端传输和原生 token；保留取消、未使用
准备、无 dispatch 和输出变化，不补写未知数值资格。

原自然控制可能改变计划，阶段差异只能作为该工作点的单轮描述，不冒称
同目标受控因果效应。existing_unmonitored 远端限制不变，不宣称已排除
所有外部干扰。未触发/退化保留，不通过重跑同键寻找胜出。

随后继续 A1 的缺失入口与剩余变体；Full/NoHandoff/CapacityOnly/FixedFull
按同合同复用。两模型全部 A1、两个 7B 累积节点、A2–A5、S1–S13、CE/价格/
Pareto、两轮审稿设计文档、IEEE 图表和最终 FaaSLoRA V2 集成均未删减。
HydraServe 隔离适配及基线比较已完成，不改共享 k3s；不做 13B/SGLang，
不改论文 LaTeX/PDF。EXECUTION_STATUS.md 的用户修改保持，本文件为当前交接。
