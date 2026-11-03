# D1052：CapacityOnly 3B W1 单因素消融

2026-11-04。当前为 A1 自然闭环消融，不是重新调优 Full。承接 D1049
实现审阅及 D1050/D1051 handoff 对照；完整目标保持 active。源计划
SHA 为 d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结指标 V1 与单轮 V2 继续生效，不改共同 SLO、预算或评价方向。

## 问题、假设和唯一差异

问题：在相同路由、handoff、层级驻留、后端及扩缩容策略下，论文软 admission
E(t) 是否减少主动加载对服务的干扰，并改善完整工作负载的资源—SLO 关系？
如果机制有效，应出现软策略的 admit/defer 差别及服务结果变化；单有推迟
动作不能证明收益，必须同时计入后续请求承担的加载和持卡成本。无触发则
不归因；CapacityOnly 更好或呈 trade-off 时也保留结果，不筛选运行。

从 D1047 Full 配置只增加 resource_coordination.coordination_enabled=false。
另改输出/NVMe/HOST 独占路径。其余配置逐字段一致。保留原始可执行 scenario
tc_ieee_full，分析和图表必须标为 CapacityOnly，不能误记为另一轮 Full。

D1049 已发现激活入口未透传 capacity_only，原默认 False 会让禁用软准入
只覆盖稳态准备。此次生产代码仅增加一行：

```python
capacity_only=not self._coordination_enabled,
```

该行位于 _add_ieee_dedicated_instance 的文件准备计划调用，覆盖 initial 和
natural_scaleout。Full 的有效参数仍为 False，保持既有语义；不修改九公式、
软容量计算、native victim/replacement、引用、槽位、KV 和物理容量检查。
请求驱动加载、handoff 目标、ready 后驻留及真实远端交付继续启用。

历史核查定位至激活路径引入提交 4af6a9cc5，并检查所有 _coordination_enabled
使用位置，确认 IEEE 入口不会因该开关切换为 legacy routing/control。
这是一处消融接线修正，不是基于 NoHandoff 较好而修改 Full 策略。

官方 vLLM 0.30.0 scheduler 的 active-LoRA 限制及 allocate_slots 路径说明
执行可行性仍依赖原生槽位/KV 分配，不能把“关闭软准入”实现为关闭物理保护。
参考：[vLLM 0.30.0 scheduler 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)。
这一源码对应关系不证明本项目 E(t) 带来性能收益；收益由消融观测决定。

## 已完成检查及其范围

- ActivationPreparation 与 AdmissionEquationContract 共 21 项 CPU 测试通过；
  新测试覆盖 initial/natural × Full/CapacityOnly 四种组合、相同计划与预算收尾。
  本组保存工具观测摘要，不冒称保留了完整 stdout 或峰值内存记录。
- 三项原生 owner CPU 测试通过：Full 压力延后不驱逐、CapacityOnly 使用相同
  victim 且保留引用、所有槽位 pinned 时不能绕过物理拒绝。保存完整工具输出。
- 复用 D1047 共同配置、运行入口、W1 source/view SHA、SLO/profile 资格。
  逐字校验生产文件相对父版本恰好多上述一行，而非跳过全部来源检查。
- 147 个旧投稿保护文件和两个用户 dirty 文件保持原 SHA；不覆盖历史结果。

prelaunch 第一次检查使用了不符合既有 UUID 约定的 scope 名称，在创建输出前
被拒绝；未启动 GPU。改用既有合法命名后继续，失败回执保留。没有修改安全
条件来绕过检查。所有 CPU 检查限定 3/4 GiB、swap=0、CPU 2,3,26,27、CUDA 隐藏。

预检查产物：paper_results/ieee_tc/ablation/d1052_3b_capacity_only_w1/prelaunch_v2/。
首版 prelaunch/ 保留；提交检查发现 YAML 末尾多余空行，规范格式后新建 v2
来源回执，配置含义不变，未重复 CPU 测试或运行 GPU。首次提交检查在 commit
之前停止，未向远端推送不合格版本。
原始工作目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/
p2_backend_qualification/d1052_20261104/。

## 唯一完整回放与收尾

复用 scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh：

```text
FAASLORA_D1000_RUN_ID=d1052_20261104
FAASLORA_D1000_RUN_TAG=d1052_3b_capacity_only_w1
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1052_3b_capacity_only_w1.yaml
FAASLORA_D1000_WORKLOAD=W1
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1
```

不设 prefix 或详细 profiling。使用既有 4,000 请求、500 逻辑工件池、真实
远端已发布 gzip、冷本轮本地缓存、共同 60 秒准备和 W1 30 秒 phase 间隔。
同 vLLM 0.30.0、FP16/TP1、32 slots、c8、max_batched_tokens=4096、
max_model_len=1024、GPU utilization=.72、1–3 实例及 D1031 profile。
不重建输入/工件，不新增 Full 回放或 GPU smoke。

启动仍检查磁盘至少 150 GiB、内存余量、GPU 空闲，沿用服务 72/80 GiB、
swap 2 GiB、独立回放/watchdog、磁盘 100 GiB 和主机内存 16 GiB 停止规则。
一次一个重型运行，测量期间不压缩、安装、推送或操作共享集群/远端配置。

完成后先释放资源，再检查全部请求、原生 token、输入映射、输出 digest、
dispatch tier、原生 admission/movement、生命周期 GPU 占用及资源事件。
重点提取 initial/natural/稳态准备是否都使用 capacity_only=True；demand
路径与物理拒绝单独统计，不把“软策略未执行”当作“无安全检查”。
与 D1047 Full 比较联合 SLO、GPU-s、P95 TTFT/TPOT 和完整阶段分解，不将
不同自然闭环计划声称为同候选受控 S3。S3 仍需按原计划单独完成。

## 证据边界和后续范围

本轮仍为已知 source42 开发负载的一次描述性消融，无运行间 CI。保留
n_correct=null、independent_numerical_proof=false、formal_ranking_eligible=false。
3B 工件全零更新与独立数值可区分性限制继续披露。此前 Full/NoHandoff 的
206 个输出 digest 差异及其有限参考解释详见 D1051；不默认新差异均正确，
也不通过反复回放获取相同输出。输入/绑定/长度检查不冒充数值正确性证明。

HydraServe 已使用独立 rootless Podman 适配；共享 k3s 不属于本项目，
不得修改其 kubeconfig、服务或资源。3B W0/W1/W2 已有 D807–D809 回放，
不因权限问题的迟到回复重新部署或回放；pp1 未覆盖 pipeline/consolidation
的限制保留。

尚未完成的 A1 十一变体、两个 7B 累积节点、A2–A5 confirmed-state/受控
handoff/first-service、S1–S13、CE/价格/Pareto、两轮审稿设计文档、IEEE 图表
与最终 V2 集成都不删减。EXECUTION_STATUS.md 为用户预存 dirty 文件，
不覆盖；本文件承接 D1051，不能让其旧 D998 页首覆盖当前阶段。

本文件预登记时尚未启动 D1052 GPU；启动/完成事实另附回执，不预写结果。

## 启动后交接：2026-11-04 03:20 +08:00

本轮为 PROGRESS：最小接线、24 项 CPU 检查、配置/旧结果保护校验及备份完成，
随后已启动唯一完整 W1 回放。不是被阻塞，也不是整个目标已完成。

代码提交 69cfb9e940bbd5ed4f9fdb30f89d49cd0bd50e97 已 push 并 ls-remote
核验 origin/retry14_continuous_queue_v2 及
faaslora_origin/backup/d1052-capacity-only-20261104。后者正式 V2 分支存在
分叉，未 force push，不能声称已完成最终 V2 集成。备份回执为 raw/backup.json。
提交只包含此次 14 个文件，用户 manifest 与 EXECUTION_STATUS 均未 stage。

运行位于上述 raw 根目录下 d1052_3b_capacity_only_w1/，tmux 会话
tc-d1052-capacity-only。launcher/replay/service/watchdog PID 分别为
2012259/2012268/2012269/2012272；服务 scope 为
primelora-tc-svc-3ca78ec3cff04357a1c1b56539953500.scope。当前均在运行，
不得重复启动同一键。first 108 个监控样本中的最新快照：主机可用约
90.34 GiB、服务峰值约 16.72 GiB、磁盘约 150.55 GiB；high/max/OOM/kill
事件均为 0、swap 为 0、无外部 compute、无越界 owned worker、无安全中止原因。
已产生 37 条请求终态记录，仅表示运行已进入服务阶段，不作为最终性能值。
精确观测保存为 raw/live_checkpoint_001.json，当前 GPU 尚未释放。

下一次只继续观察这个运行。完成并确认资源释放后，复用 D1050 的 collect_closed_evidence.py
和比较逻辑建立新的 D1052 分析目录；检查软准入实际参数、物理安全、请求
合同和完整指标，再与原 D1047 Full 作单轮描述性对照。不要在测量期间编辑
运行源码、压缩/清理、推送、重启远端或开展另一 GPU 实验。源 PLAN、冻结指标、
Full 和 NoHandoff 结果均不变。所有后续消融、敏感性和交付仍保持待办。

### 后续观测：03:25 +08:00

上一 goal turn 为 PROGRESS；本 turn 为已核实 live 的等待及收尾统计入口准备。
同一 launcher/replay/service/watchdog 仍在运行，first phase 500/500 完成、
日志请求失败 0；后续 phases 未完成，不能出完整性能结论。live 面板的
slo@5000ms 是开发显示，不用于 D648 共同 SLO 判定。

raw/collect_closed_evidence.py 复用 D1050 collector，仅更新运行、配置 SHA、
变体和提交身份；raw/compare.py 复用 D1050 比较，改为 CapacityOnly 单开关
核查并增加 GPU preparation plan 的 capacity_only、attempt/admission reason、
实际 load 与观测软余量不足字段。Full/CapacityOnly 都保留 handoff；缺回执、
提前拒绝和真正评估的软准入分开，计数不冒充独立物理加载或反事实收益。
两脚本已通过轻量 AST 语法检查，尚未对未完成输出运行，也未宣称分析通过。
精确进程/安全快照和检查回执为 raw/live_checkpoint_002.json。

待本轮正常结束且 GPU 释放后，依次在既定辅助资源域执行上述 collector、
compare；不要重启本轮、重新生成 Full、修改运行中源码或使用部分数据排名。
此次只增加 raw 统计脚本与本文交接，没有推送/压缩/远端服务变更，也未删除
剩余实验。源码备份仍为 69cfb9e9；运行后记录待收尾时一起备份。

## 完成结果：同输入单轮观测，不支持软准入收益归因

D1052 已正常完成，4,000/4,000 请求成功，458,224 个原生输出 token；全部
请求的实际数量等于规定目标。launcher、回放、服务和 watchdog 均退出 0，
服务资源域已移除，11 次物理 GPU allocation 全部具有 release 证据，无残留
GPU compute。没有重跑 Full，也没有重复本轮或修改共同评价阈值。

| 指标 | Full D1047（复用） | CapacityOnly D1052 |
|---|---:|---:|
| 完整原生数量合同通过 | 4,000/4,000 | 4,000/4,000 |
| 联合时延 SLO 通过数 | 3,836 | 3,848 |
| 联合时延 SLO 达成率 | 95.90% | 96.20% |
| 生命周期物理 GPU-s | 6,182.715971 | 6,371.761305 |
| GPU-s/offered-request | 1.545679 | 1.592940 |
| 平均 TTFT（ms） | 454.156048 | 439.785864 |
| P95 TTFT（ms） | 1,114.091001 | 1,042.205312 |
| 平均 TPOT（ms） | 17.417030 | 17.345311 |
| P95 TPOT（ms） | 31.774061 | 31.488817 |
| 平均 E2E（ms） | 2,323.526111 | 2,292.407132 |
| 真实远端传输次数 | 132 | 132 |
| 线上传输字节 | 306,360,162 | 306,360,162 |
| initial / natural-scaleout 数 | 1 / 9 | 1 / 10 |

Full 的物理 GPU-s 比 CapacityOnly 低 2.966924%，但 P95 TTFT 高
6.897459%，联合 SLO 低 0.30 个百分点。这是单轮观测的资源—时延取舍，
不是 Full 在所有目标上胜出。两点的观测时延达成均超过 95%，资源均低于
D657 W1 预算 6,754.783565 GPU-s；这不解除独立数值正确性限制，不宣称
正式 G1/G2 排名、显著差异或全局最优。表中分母明确为 offered，不把未知的
N_correct 偷换成 4,000。

4,000 条 request/adapter/source-item/arrival-plan/prompt/native-prompt/target/
generation/actual-token 映射逐条匹配。输出 digest 相同 3,807 条，不同 193 条。
性能日志未保存全部原生 token 序列，不能由摘要判定这些差异的数值正确性，
也不能将 D1051 的有限参考检查泛化至这 193 条。n_correct=null、
independent_numerical_proof=false、formal_ranking_eligible=false 原样保留。
既有 3B 零更新工件池的不可区分性与 source42 开发负载身份同样保留。

### 实际机制：软余量不足未触发

| 准备 attempt 分类 | Full | CapacityOnly |
|---|---:|---:|
| 进入 typed resource-admission 且放行 | 707 | 796 |
| 进入 typed resource-admission 且延后 | 0 | 0 |
| 观测 footprint 超过有效软容量 | 0 | 0 |
| 进入资源准入前因 stale snapshot 延后 | 312 | 328 |
| 已复用、未调用 native load | 23 | 24 |
| attempt 无 native receipt | 0 | 2 |
| attempt 合计 | 1,042 | 1,150 |

所有被记录的 GPU preparation plan 的 capacity_only 与目标开关一致，包含
natural-scaleout 和稳态 residency；initial 没有 GPU preparation attempt，
不虚构 initial 的运行时软准入决策。参数透传的 initial 覆盖来自已保存的
CPU 测试，与上述实际触发范围分开。

CapacityOnly 的两条无 receipt attempt 在 journal 中仍标为 observing；其
所属 plan 明确记录 CancelledError、state=cancelled、closed=true，关闭时
live native/HOST/staged leases 均为 0，pending preparation targets 为空。
原始 attempt 状态不补写。它们不算已评估的 admission、不算成功物理加载，
也不凭“缺 receipt”推断发生了软延后。保留完整小型原始 plan/attempt 摘录。
聚合中的 receipt_reasons.not_recorded 仅表示无顶层 reason，绝不能误读成
730/822 条 receipt 缺失；实际缺失只有上述两条。

因此，这一自然 W1 工作点没有观测到 Full 软 E(t) 检查阻止加载；不能把
2.97% GPU-s 差异解释为该机制收益，也不能把 CapacityOnly 较低尾延迟解释为
软策略确定有害。两轮自然反馈的扩容、准备计划和调度轨迹不同，且没有独立
运行重复。原计划 S3 的同候选加载争用对照仍需要执行；不人为注入等待、
更改阈值或反复运行来制造软准入触发。

Full−CapacityOnly 的平均 TTFT 阶段差（ms）为：arrival→admission +9.485075、
admission→acquired +0.263892、executable→engine +1.104971、
engine-entry→queue +2.630387、native queue +0.117428、prefill +0.768430。
这些只定位观测等待分布，不提供 E(t) 的因果分解。实际 dispatch tier 完整：
Full GPU/host/NVMe/remote=3327/2/538/133，CapacityOnly=3333/1/535/131；
不能由 dispatch remote 计数替代双方实际 132 次传输。

### 计量、安全和分析来源

- 2,266 个资源样本；service high/max/OOM/kill 事件均 0，swap=0；服务峰值
  22,642,708,480 B。最低主机可用 93,233,377,280 B，最低磁盘可用
  156,977,467,392 B；没有外部 compute 或越界 owned worker。
- 132 次传输均能对应服务端/客户端 ID、线上字节与 SHA；请求中无打包。
  远端 observation mode 为 existing_unmonitored，没有连续远端 CPU/外部
  干扰监控，不能宣称已排除所有远端干扰。
- 本轮工作缓存已清空；147 个旧投稿保护文件及两份用户 dirty 文件 SHA
  不变。没有压缩原始本轮证据、覆盖旧图或修改论文。
- completion/ 复用 D1050 完整计量 collector；comparison/ 复用既有冻结
  指标与 actual-first-dispatch 分析。首次离线比较把机制 activation 的
  category 误写成控制事件的 activation_kind，在发布输出前失败；依据
  实际 schema 改正后通过，失败回执保留，未重跑性能实验。
- verification/ 进一步区分 typed decision、stale snapshot、复用和真正
  缺 receipt，原始 comparison 输出不覆盖。所有分析在既定 3/4 GiB、
  swap=0、CPU 2,3,26,27、CUDA 隐藏的辅助域中执行，无内存保护事件。
- D1050 的四项 actual-first-dispatch 测试及其源文件 SHA 未变化，直接复用
  已通过回执，不再重复 broad qualification。

产物根目录：paper_results/ieee_tc/ablation/d1052_3b_capacity_only_w1/。
原始 result SHA 为 a66578172e42c3860450af1cd8355cd7bc9aff8c709b2f4e8b4431bded1ab6da，
full4000 audit SHA 为 e5d580def9bf057df240f1eabdad32f0e8af9dfdbbd676d057f307db3cc0517f。

## 当前交接与下一主线

本轮为 PROGRESS：完成一个 A1 CapacityOnly 自然 W1 点、完整计量和机制
提取，不能把“未触发”包装成正贡献。Full、NoHandoff、CapacityOnly 的
3B W1 已有数据，完整目标仍 active；不因此重新调优 Full 或重开基线适配。

下一项仍按 A1，选择 FixedFull 3B W1：只将初始/最小/最大实例固定为四个，
保持 Full 的 router、handoff、hierarchy、admission 与动态控制执行路径。
它直接回答两轮审稿关于 elasticity 与 adapter-management 的区分，不应以
关闭 dynamic_scaling 顺便关闭驻留调度。实施前复用 D1049 入口核查、做
相应最小配置/初始池检查，并重新读计划及资源余量；本交接尚未启动该实验。

其他 A1 节点、7B 两个累积节点、A2–A5 的 confirmed-state/受控 handoff/
first-service、S1–S13（包括 S3）、CE/价格/Pareto、两轮审稿设计文档、
IEEE 图表和最终 FaaSLoRA V2 集成全部保留。按最新规则先保存状态表，
不以反复正确性总审计或展示工作替代实验主线。
