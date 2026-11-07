# D1123：当前版本 7B NoRouting W1，隔离服务时间排序

2026-11-08，性能运行前登记。按既定 A1 顺序承接已封存的 D1122，
不是新的 Full 优化或基线复现；Full 唯一参照仍是 D1115，不重复回放。
本登记时尚未启动。ablation-planner 指定独立评审模型不可用，本地可行性
核查标为 `[pending external review]`，不冒称独立评审完成。

## 问题、假设与范围

在保留相同可行性、来源授权、handoff、层级驻留、准入和弹性控制规律时，
服务时间感知排序是否改善物理 GPU 占用与联合 SLO/尾部延迟的权衡？
若该排序重要，应有实际决策差异，并在完整服务结果和阶段证据中体现。
若没有改善或出现权衡，原样保留；不能降低对照的容量或关闭其他机制来造优势。

优先级 1，一轮既有 4,000 请求 W1。排程估计约 40 分钟、2–3 GPU-hours，
不是保证。本次仅使用已有 selector；不新增生产机制，不追加冗余 GPU smoke。
本项与 LastKnown 不同：它检验是否使用服务时间排序，而非信息新鲜度。
同一控制规律不意味着两次闭环运行产生相同扩缩容、缓存位置或准备计划。

## 单因素合同

Full 在同一不可变快照的可行副本上使用：

`(floor((D+T+O)/delta), admitted_requests, pending_loads, gpu_utilization_pct, last_dispatch_at, replica_id)`。

NoRouting 仅移除 service-bin 首项：

`(admitted_requests, pending_loads, gpu_utilization_pct, last_dispatch_at, replica_id)`。

保留 live Q、相同稳定 tie-break、active-LoRA 可行性、原子预留、来源保护、
取消和释放。选中后的来源重查仍真实执行；无可行候选仍排队。
实际路由和 scale-up 路由投影均使用同一 load_only；不偷偷保留 Full 投影排序。
共同 profile/快照采集仍保留，因此隔离决策规则，不是最小观测成本实验。

配置 `configs/ieee_tc/20261108_d1123_7b_no_routing_w1.yaml`，SHA
`00729d927fad71131f4797d08cd86ffde2c1ec0f9bd9957fa1dd864175602ca4`。
相对 D1100/同版本 Full，除独立 output/NVMe/HOST 路径，只新增：

```yaml
ieee_selection_strategy: load_only
```

`routing_policy=ieee_confirmed`、handoff=full、confirmed snapshot cache=true、
min1/max4 保持。不改成 legacy least_connections，不用 ElasticOnly 开关。
实际场景仍 tc_ieee_full，外部变体明确 NoRouting，不并入 canonical Full。

Llama-2-7B、vLLM0.30.0、FP16/TP1、14slots/seqs、1024batch tokens/context、
GPU utilization .92、prefix cache false、D362 profiles、27.43356142744625ms
service bin、预测和反应式控制均与 D1115 相同。沿用 seed42 源 trace、500逻辑
adapter、W1 8×500、phase间30秒、业务前60秒通知及 fixed_length_greedy_v1。
保持 exploratory/formal_run=false；不是新独立留出数据。

## 历史、实现与原始资料

复用 D1055 的 selector（提交25583709160bd93d8fd72f2b8d0d203cb9d8ea09）和
`Router.select_load_snapshot`，不改生产代码。历史3B对照曾体现资源与延迟
权衡，保留该反向结果；只复用方法，不将3B profile或性能混入7B。
近期 fdeea80f/00f2a873 的零收益/零需求规划优化及 NoHOST 集成在 Full/本项
同版本共享，不因为消融重新开启已否定的优化方向。

2026-11-08复核 [vLLM Router 官方说明](https://raw.githubusercontent.com/vllm-project/router/main/docs/load_balancing/README.md)：
负载感知与缓存感知是可区分的路由策略；本项不是复现其随机二选一算法。
[dLoRA 原论文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
讨论请求负载与 adapter 位置的协同。因此普通负载排序不是新贡献，NoRouting
单项也不能证明其他系统缺少所有协调；本项仅隔离 PrimeLoRA 自身排序。

10项定向CPU测试通过（run-u365，main/wrapper均0，测试0.071s、服务10.257s）：
六项纯selector合同、一项真实runner/profile透传、两项来源授权/取消、
一项全部非目标配置严格相等。CPU2,3,26,27、MemoryHigh3GiB/Max4GiB/swap0，
GPU隐藏。测试中torch报告没有CUDA runtime是此CPU检查环境，不是GPU资格。
没有新后端/负载或生产路径，不重做整套资格；实际机制仍在完整运行中核验。

## 唯一执行与安全

备份后刷新 GPU 空闲、102GiB可用内存、150GiB可用磁盘；停止线16/100GiB。
服务72/80GiB/swap2、辅助3/4GiB/swap0、CPU分离、独立watchdog保持。
已有runner在通知前自动启动并认证核验远端已发布gzip/timing_v2，仅停止自己
拥有的服务；不重复发布缓存，不触碰共享k3s，不重开已完成的HydraServe适配。
测量期间不维护远端、不做哈希扫描/压缩/构建/测试/推送或另一个性能运行。

唯一tmux `tc-d1123-no-routing-7b-w1`，独立路径不存在时仅启动一次：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1123_20261108/7b_no_routing_w1 \
  d1123_7b_no_routing_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1123_20261108/7b_nvme_no_routing_w1 \
  /dev/shm/tc-d1123-7b-no-routing-w1 \
  configs/ieee_tc/20261108_d1123_7b_no_routing_w1.yaml tc_ieee_full
```

启动后另记checkpoint、panePID、scope/InvocationID、remote clock及保护样本。
上述命令仅是登记，不能当作已执行。完整原始和失败记录不覆盖、不按胜负剔除。

## 分析与进入下一项的条件

实际终态和资源释放后收集同clock远端journal，复用D1113/D299/D455/D297。
逐请求校验同输入、原生输出数量、adapter/来源/lease、时间线和dispatch tier。
逐一用完整可行快照重算实际load_only选择，核对计数；与Full纯shadow比较
决策分歧，但不生成反事实TTFT。真实handoff、层级promotion、soft admission
和自然扩缩容必须有触发证据；只配置开启不能作为机制已运行证明。

冻结V1+单轮V2、D394共同SLO和D594预算不变。报告联合/边际SLO、
物理GPU-s及互斥窗口、GPU-s/offered、TTFT/TPOT/E2E、实际下载字节和次数。
W1预算6756.571050141065GPU-s；不把max4与另一个cap3工作点拼接。
同后端输出digest变化原样审计，独立数值原因未知则仍未知；n_correct=null
不补造，不用原生长度通过推断独立数学正确性。单轮描述，不计算CI或显著性。

首次有效结果封存、输出状态表/解释并备份后推进7B NoHandoff/R0H0等其余
A1变体和累积节点。A2–A5、S1–S13、离线CE/成本/Pareto、两轮审稿设计文档、
IEEE图表和最终V2交付全部保留。整个目标仍active，不重复Full或基线矩阵。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
其余source/test/protocol SHA与旧投稿保护核查见同组prelaunch.json；
不stage用户manifest或已有dirty状态全文。
