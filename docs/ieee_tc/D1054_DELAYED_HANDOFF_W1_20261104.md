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

## 已启动交接：2026-11-04 05:08 +08:00

本轮为 PROGRESS：配置、两项 CPU 时机检查、唯一差异与保护清单验证完成，
并已真实启动 D1054 唯一完整回放。启动前提交
e3ae4f327a0d370b48a7092db2ab8acf13a55b84 已推送并 ls-remote 验证至
origin/retry14_continuous_queue_v2 和 faaslora_origin/backup/
d1054-delayed-handoff-20261104。未强推实际 FaaSLoRA V2，最终集成仍待完成。
备份回执 raw/backup.json；配置 SHA
14c55659bbee6b3e3d6668a42b11c286c4579361e0aea3601f98e86313efbb74。

tmux=tc-d1054-delayed-handoff。launcher/replay/service/watchdog PID 分别为
2277621/2277633/2277671/2277684，三次实际进程查询均 live；service scope
primelora-tc-svc-123ef765757f40dcb0ff7917307d31ff.scope，InvocationID
6d852c3b9ccc4d29a3f24c2085963e8b。不要因观察窗口结束而重启同键。

最新部分进度 arrived=68/4000、done=65、success=65、fail=0；实际三副本，
自然 scale-up=2、scale-down=0。第 136 个资源样本：主机可用
96,224,133,120 B，服务峰值 18,077,335,552 B，磁盘可用 163,484,983,296 B；
high/max/OOM/kill/swap、外部 compute、越界 owned worker 均 0，无中止原因。
这些只是运行中观察，不是最终联合 SLO 或性能结论；完整观察和监测命令
已保存 raw/live_checkpoint_001.json。

raw/collect_closed_evidence.py 已从 D1053 collector 按本轮路径、配置 SHA
和提交身份适配；尚未执行，不读取未完成结果、不预填资格。之后先继续
监测同一 live handle，等 launch.json 正常终态与实际 GPU 释放，再运行
collector。比较入口仍需按 DelayedHandoff 的时机证据适配，不能运行
D1053 的 fixed4 断言。正式比较前重新核对冻结 V1/V2。

测量期间无生产代码/远端服务/共享集群修改，无压缩、构建或推送。当前
只有本次追加交接尚未备份，待正常收尾一起保存，不与推理测量重叠。
后续全部实验及交付范围保持，本轮不将任何单项启动写成整体完成。

### 收尾分析入口准备（回放期间，仅编辑离线脚本）

已从 D1053 compare.py 复用共同指标、输入逐条匹配、阶段表及 actual-first-dispatch
逻辑；仅将配置断言改为 handoff full/delayed，核验所有 activation 的相应
policy，不沿用 fixed4/零扩缩容断言。所有软 admission、预算参考和正式资格
未知字段保留。collector/compare 已通过 AST 检查，回执
raw/analysis_syntax_receipt.json；没有导入或运行分析，没有读取未完成大结果。

后续时机核查直接复用 D1051 audit.py 的既有证据来源，不重复 HF/PEFT 历史
数值参考审阅、不增加 GPU 回放，也不从当前性能结果反推正确性。具体区分：

- queue 在 `coordination_after_shutdown.ieee_movements` 保存每个 job 的
  subscriptions、attached_at、attempt started_at/finished_at/output_ready_at。
  metadata 的 activation_id/target_replica/plan_id 用于对应准备所属 activation。
- 同一个物理 job 可以先由 demand/residency 发起，DelayedHandoff 在 ready
  之后加入。不能把旧 job 的起点归给后加入的 handoff subscriber，也不能
  因旧 job 在 ready 前开始就判 delayed 违规。创建者与订阅者须分开记录。
- queue attempt 开始只是 executor 进入，不自动等于真正复制；completed
  也可能只是复用已存在内容。实际 remote 关联按 adapter、local owner、
  loading clock 和 loading/published interval；只将自己拥有的 I/O 归因给
  发起意图，不重复分摊成多个物理加载。
- native HOST/GPU 的实际加载使用 receipt.native_load_invoked=true 及其
  load_started/load_completed（GPU 带 native_ 前缀）时间，核验 owner/clock。
  未加载、无回执、取消和 reused 状态保留，不能以计划已完成填零或补事件。
- first-service 继续用 epoch 第一条实际 dispatch；准备完成后同 adapter
  命中只是关联，缺连续驻留证据时不声称此命中必然来自该次 handoff。

这些是收尾统计解释规范，尚无本轮的完整时机结论。运行期间不修改生产
源码、远端配置或性能测量，不执行压缩/构建/推送。

### 继续观察：前两阶段 1,000 请求完成

本 goal turn 为 verified wait，并完成离线比较入口准备；没有新 GPU 启动。
持续查询同一组 PID，当前 launcher/replay/service/watchdog 仍全部 live。
最新 arrived=done=success=1,000/4,000、fail=0，实际三副本，scale-up=3、
scale-down=1；第 648 个资源样本：主机可用 95,476,318,208 B、服务峰值
18,934,026,240 B、磁盘可用 162,028,720,128 B。high/max/OOM/kill/swap、
foreign compute、escaped worker 均为 0，无中止原因。前两阶段成功不替代
后六阶段或最终指标，尚无终态 launch.json。

此次短期监测 cell 4114 已结束，但 tmux 中的实验和独立 watchdog 没有停止。
该 cell 的临时 store 在终止后未保留，首次拼接观察回执失败于写文件之前；
随即重新实查同一 PID/资源记录并保存 raw/live_checkpoint_002.json。旧轮询
仍有工具输出 chunk 引用，不虚构为已持久保存的完整历史。实验原始 watchdog
日志继续由原进程完整写入，没有丢失或重启测量。

下一 turn 直接继续监测同一运行，不再等待已关闭的 cell 4114，不启动第二轮。
准备好的 collector/compare 均未执行；终态及 GPU 实际释放后才运行，随后
完成上述 delayed 时序/共同输入/输出摘要核查并备份。本交接暂不与测量
重叠推送；原生产提交和全部未完成范围不变。

### 继续观察：完成前四阶段；真实准备时机分析入口已备妥

本 goal turn 为 verified wait with retained-receipt audit preparation。再次读取
目标、完整 PLAN、AGENTS、当前交接和指标 V1/V2；EXECUTION_STATUS 中旧 D998
状态不覆盖 D1048–D1054 已完成工作。用户确认共享 k3s 不属于本项目，沿用
已完成的 HydraServe 隔离适配，不修改共享集群权限、配置或工作负载，不重做
已经完成的适配/基线回放。

同一 launcher/replay/service/watchdog PID 仍全部 live。最新 arrived=2,093、
done=success=2,092/4,000、fail=0，前四个 500-request 阶段已全部完成。
scale-up=5、scale-down=3 是在线逻辑计数，不能代替物理释放证据。
第 1,155 个 watchdog 样本：主机可用 99,531,735,040 B，服务峰值
20,073,291,776 B，磁盘可用 160,987,860,992 B；high/max/OOM/kill/swap、
foreign compute、escaped worker 均 0，无安全中止原因。当前持有两 GPU。
尚无终态 launch.json；部分完成不等于最终 SLO 或性能结论。实际观察和
监测命令保存 raw/live_checkpoint_003.json，不依赖未完成 cell 的临时 store。

从 D1051 retained-receipt 分析复用并准备 raw/audit_delayed_timing.py，
SHA b7925e8c2cb5f356468472a9df55e8bcc4f2dcb1044563066dc04f4fafeccf2b。
只通过 AST 语法检查，未导入/执行，没有读取本轮未完成大结果。脚本在正常
终态、物理 GPU 释放、collector 和 comparison 完成后才运行，使用既有
3/4 GiB、swap=0、CPU 2,3,26,27、CUDA hidden 的离线分析资源域。

分析直接读取原始 handoff attempts 和 movement subscriptions，区分任务
登记、executor 进入、真实 native load 与 remote publication。创建者按
唯一最早 attached_at 确定，不依赖可能经 JSON 排序的字典键序；时间并列
保留未知。多个订阅者共享的 native load span 按 owner/clock/adapter/起止
去重，避免把一次物理加载重复统计为多个计划收益。Delayed 的订阅必须不早于
其 engine-ready；只有动作创建者属于该 handoff 时才将其 executor/remote
起点归给该 handoff。取消、缺回执、未执行及复用保留，不能填零充当加载。

没有修改生产代码、运行配置、测试、远端服务或共享集群；没有压缩、构建、
推送或第二个 GPU 实验。下一步继续同一 live run，正常收尾后收集/比较/
核查实际时序并备份，再继续 A1。两模型剩余 A1/累积节点、A2–A5、S1–S13、
CE/价格/Pareto、审稿设计文档、IEEE 图表及最终 FaaSLoRA V2 集成仍全部保留。

## 完整结果：单轮完成，不支持本点的提前重叠收益归因

D1054 已正常完成 4,000/4,000 请求。launch.json 的 pass、service_path_removed、
native_gpu_context_release_confirmed 均为 true；原四 PID 均已退出，GPU census
为空。collector、comparison、timing 三个离线步骤均退出 0，其辅助资源域
也已消失。本 goal turn 从 verified wait 推进到完整测量、离线分析和机制核查；
没有重跑同键、重启远端或改动生产实现。

### 共同指标与绝对值

| 指标 | Full（D1047，复用） | DelayedHandoff（D1054） |
|---|---:|---:|
| 完成请求 | 4,000 | 4,000 |
| 联合达标数 | 3,836 | 3,851 |
| 联合 SLO | 95.900% | 96.275% |
| 物理 GPU-s | 6,182.715971 | 6,190.253722 |
| GPU-s/offered-request | 1.545679 | 1.547563 |
| 平均 TTFT（ms） | 454.156048 | 445.199272 |
| P95 TTFT（ms） | 1,114.091001 | 1,080.769681 |
| 平均 TPOT（ms） | 17.417030 | 17.505360 |
| P95 TPOT（ms） | 31.774061 | 31.868809 |
| 平均 E2E（ms） | 2,323.526111 | 2,315.197003 |
| 远端获取次数 | 132 | 132 |
| 线上字节 | 306,360,162 | 306,360,162 |

Full 相对 DelayedHandoff 的 GPU 时间少 **0.121768%**，联合 SLO 低
**0.375 个百分点**，P95 TTFT 高 **3.083110%**。两者均超过冻结的 95%
联合时延要求，且低于同一 W1 GPU 时间上限；这些是已知 source42 负载上的
单轮观测，不作显著性、非劣或全局最优结论。数值资格未知，因此上表明确
使用 offered 分母，不把该值冒称 GPU-s/correct-request。

逐请求 input/adapter/prompt/原生输入 token 摘要/target/实际输出数量/
generation 全部匹配；输出序列摘要 3,785 相同、215 不同，完整变化表保留。
既有 all-zero 3B pool 的非零 LoRA 算术不可区分限制不变，n_correct=null、
independent_numerical_proof=false、formal_ranking_eligible=false 不回填。
不重复 D1051 已关闭的旧 100-request HF/PEFT 参考核查，不用长度或名称替代
独立数值证明，也不把摘要变化直接定性成错误 adapter。

六段 TTFT 的 Full−DelayedHandoff 平均差分别为：到达→admission +5.829981 ms、
admission→acquired +1.388059 ms、executable→engine −0.791216 ms、
engine→queue +1.924084 ms、native queue −0.276348 ms、prefill +0.882214 ms。
阶段差描述路径差异，不将其全部因果归给 handoff。dispatch tier 分别为
Full remote/GPU/NVMe/HOST=133/3327/538/2，DelayedHandoff=132/3328/539/1；
真实远端获取仍两边各 132 次，不能把 dispatch tier 计数当作 fetch 次数。

### 实际时机，而非仅统计计划数量

| 证据 | Full | DelayedHandoff |
|---|---:|---:|
| initial / natural activation | 1 / 9 | 1 / 8 |
| 有实际 first dispatch / 首 token 的 epoch | 10 / 10 | 9 / 9 |
| handoff native HOST 实际加载 | 58 | 52 |
| handoff native GPU 实际加载 | 57 | 51 |
| handoff native 加载早于 engine-ready | 0 | 0 |
| 上述加载完成早于所属 epoch first dispatch | 27 | 52 |
| handoff movement jobs（含复用，非物理加载次数） | 132 | 108 |
| handoff subscriptions 早于 engine-ready | 0 | 0 |
| 匹配到 handoff job 的实际远端传输 | 0 | 0 |

两个变体的 handoff movement 均在 native owner 域，未观测到共同文件层的
handoff 复制。DelayedHandoff 的时机约束核查 **0 违例、0 未决**；取消、复用、
未调用 native load 的 attempt 仍保留。Full 也没有在本点触发可归因的
engine-ready 前实际加载，因此自然 W1 的对照没有充分隔离所声称的启动重叠
收益。不能将上述 0.12% 资源差归因为提前准备，也不能据此否定所有其他
负载下的 handoff 作用。现有受控 A2/A4 仍必要，且已经在原计划中，不另加
重复性能点或人为 sleep。

自然闭环的扩容数、计划和 first request 本身不同，27/52 等计数不是同一
批准备任务的受控因果效应。first-service 使用第一条实际 dispatch，未按成功
或 adapter 命中重新选样；不为凑 20 个样本追加同配置运行。ready 后某次同
adapter 命中不能证明该 handoff 加载持续驻留。

软 admission 的有效决策分别为 Full 707、DelayedHandoff 728，全部 admit，
没有观测到软 headroom deficit。与 D1052 一致，此点不新增 E(t) 延迟拒绝的
因果证据；原 S3 真实争用实验继续保留，不在这里重新调 admission。

### 完整性、资源、来源与交付

- 4,000 个原生 profile binding 和 fresh descriptor 通过；时延重算及阶段
  恒等式误差为 0 ms。完整 trace/subset/合同保持，全部来源 SHA 在 manifest。
- 132 次真实远端传输的客户端/服务端 ID、字节和内容核验匹配；请求中无打包。
  远端 existing_unmonitored 限制保留，不声称排除了全部外部干扰。
- 9 个物理 allocation 全部 release；2,264 个资源样本中无 foreign/escaped
  compute，无 high/max/OOM/kill/swap。服务峰值 21,822,636,032 B。
- 本轮工作缓存已按原 runner 清理，旧投稿 147 文件和两项用户 dirty SHA 不变。
  收尾磁盘可用约 163.64 GB（十进制），恢复到下一任务 150 GiB 启动线以上；
  不因此跳过下一次实际资源检查。本轮未额外压缩或删除历史证据。
- 三项离线输出分别位于本目录族 completion/、comparison/、timing/，保留
  脚本快照和 source refs。timing 分析采用实际 publication.content_sha256
  验证单 adapter 内容；content_manifest_sha256 是整个工件池清单，二者不能
  混同。这一字段错误在首次执行前通过源码核对修正，未发生失败 GPU 回放。
  最终 timing 脚本 SHA 为
  64040339d27f830eb9f46a38213c83e0c40c786acfdb749cab489264da7a2040。

下一主线是完成 A1 尚缺的 load-only selector（NoRouting/ElasticOnly/R0H0
共用的最小入口），保持 live feasibility、reservation 和稳定 tie-break；
先对照 D1049、既有 routing 代码/测试和原始文献，再作限定实现。不是因
DelayedHandoff 本轮某些指标更好而修改 Full，也不重跑已封存的 Full。
LastKnown、NoHOST、ElasticLocality、两个 7B 累积节点及完整 A2–A5/S1–S13/
设计文档/图表/最终 V2 集成仍在总目标中。接下来先备份本结果，再进入下一项。
