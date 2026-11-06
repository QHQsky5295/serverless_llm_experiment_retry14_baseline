# D1083：消融定义的只读复核，不改正在运行的版本

2026-11-06。当前唯一完整 3B W0 正在运行，本项只读小型历史记录、
配置和源码；没有新 GPU 实验、生产修改、测试、远端操作或备份。
本项不是新增消融计划，也不是独立 reviewer 的审计通过证明。

## 最新要求与执行边界

本次完整读取的目标为
`/home/qhq/.codex/attachments/2357f95d-3420-4a51-a932-4da462c01c17/goal-objective.md`。
新增要求允许在 Full 不及消融时检查消融代码、按贡献修正。
执行解释：如变体误保留了应被移除的贡献，或非目标配置与 Full 不一致，
据定义修复并只重测受影响项；不能以使结果下降为验收条件。
不加入人为等待、不削减共同资源、不禁用合法按需缓存、不改 SLO/分母，
不删除既有反例。正确关闭一个机制可能使结果变好，也可能变差，均须保留。

PLAN SHA 保持
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
冻结指标 V1 与单轮 V2 已完整重读；没有因本要求改动九公式或评价尺子。
共享 k3s 仍不属于项目，继续已有 rootless Podman 隔离适配；不申请其
管理员配置，不重新部署已完成的 HydraServe 基础路径。

## 检查范围与发现

| 检查 | 当前证据 | 裁决 |
|---|---|---|
| Full→ElasticOnly 配置 | D1047 与 D1057 YAML 的差异仅为独占路径、准确身份及批准的 load-only/no-handoff/demand-LRU/关闭主动管理组合 | 未见另换后端、模型、容量或 autoscaler 的非目标差异 |
| ElasticOnly→ElasticLocality 配置 | D1057 与 D1061 仅换独占路径/身份、gpu_locality 及原已声明 slack=1 | 不根据胜负重新选择 slack |
| 路由实际选择 | instance_pool.py 的 load-only 不按 tier/D/T/O 排序；locality 只用 GPU-hit 和 admitted-request slack | 是不同的预定义复合变体，不是同一个单因素 |
| 主动规划入口 | _schedule_ieee_residency_epochs 在 hierarchy=False 时直接返回；activation 在 no_handoff 时不产生准备计划 | 所读入口没有主动规划泄漏证据 |
| 运行中的机制事实 | D1057/D1061 原比较记录均为 4000 条 selector 重算一致；主动 file/GPU/HOST 计划和 residency epoch 为零，activation 均 disabled/null plan | 支持这两次历史执行已关闭目标机制；本次不冒称重读了全部大原始日志 |
| 请求驱动准备 | _arrived_route_confirmed_file_preparation 由已到达请求触发，沿用当前变体 selector，只开始共享 NVMe movement，不授予 GPU slot 或 source admission | 三组原合同共同保留，不能事后把正常需求流水线改成 Full 独享 |
| 最新共享优化 | 462f9987 改 gpu_monitor.py 的预算用 HOST 清点；两处调用省略无消费者的逐 adapter 表，仍作新鲜物理 union/alias/pinning 检查 | 不按 system variant 开关，涉及这些公共路径的消融也应使用相同实现 |

load-only 的精确定义是保留论文 Q 元组：admitted requests、pending loads、
GPU utilization，再按 last-dispatch 与 replica ID 稳定决胜；不是“只看
admitted 数”的另一种算法。ElasticLocality 则只看 admitted 数及二元 GPU
命中。这个区别已在既有实现/登记中声明，不能看到胜负后改定义；若论文
描述只写“least-loaded”，最终文档应明确实际排序字段。

共同来源/物理可行性观察仍存在，不等于继续使用 Full 的 readiness 排序。
这两项复合消融本身不能隔离 confirmed-state propagation；该问题仍由
既定 LastKnown、同状态 shadow/A-A 和受控机制实验回答。

## 依据与限制

核对：D1057_ENTRY_AUDIT、D1057_IMPLEMENTATION/OUTCOME、D1061_OUTCOME、
D1064_RETAINED_A1_OVERHEAD、D1082_FULL_W1_OUTCOME，以及实际配置 diff。
当前源文件：

- `scripts/run_all_experiments.py`：1532、14676、15830、16624、19441 附近；
  SHA `82490fbd40bbfd189ec11298c742d4e819043a8fa72b8591551aed983114b93b`。
- `faaslora/experiment/instance_pool.py`：2044、2127、2159 附近；
  SHA `f051c0e4a07bd1ebe5af008edbbe4be6db650e0a083604700502656595b15629`。
- `faaslora/experiment/experiment_stack.py`：demand_lru 组合检查。
- `git show 462f9987 -- faaslora/memory/gpu_monitor.py`：公共预算消费者投影。

结论限定为所列配置、入口与已有机制摘要，不宣称穷尽全部旁路。
experiment-audit 技能引用的 shared-references 文件在本机缺失，故按技能
不可用回退规则完成本地只读检查；没有独立 reviewer、审计 PASS 或虚构 trace。
本次没有新优化候选，不以本检查替代新候选所需的历史/官方资料/最小验证。

## 对实验主线的影响

当前未发现足以授权修改这两个消融语义的具体缺陷，保持其定义和历史结果。
D1082 Full 已改变公共底层实现；它与旧 D1057/D1061 的比较只能作为开发
参照，不能冒称同版本的严格消融归因。依既定顺序完成当前 D1083 W0、
同版本 W2，再更新受影响 A1（优先这两个已知反例），无需新增旁支实验。
共享优化应给所有实际使用该路径的变体，不独占于 Full。

W0 收尾后先核算并封存，再备份本检查点；测量期间不提交/推送、清理或
更换代码。两项用户 dirty 文件和原投稿结果保持不动。G2、同版本 A1、
7B A1/累积节点、受控 A2–A5、S1–S13、补充分析、论文证据文档、图表及
实际 FaaSLoRA V2 集成继续在范围内；总体目标 ACTIVE。
