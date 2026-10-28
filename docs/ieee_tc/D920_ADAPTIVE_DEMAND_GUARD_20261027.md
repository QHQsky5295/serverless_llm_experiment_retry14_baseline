# D920：7B IEEE 扩容的当前需求守卫候选

日期：2026-10-27。性质：单一候选的代码与配置资格检查，尚未产生新的
GPU 性能结果。

## 1. 动机与可证伪假设

D917 是当前代码、真实远端发布工件、固定长度生成合同下的 7B W2 完整运行：
4,000/4,000 请求完成，平均 TTFT 1,124.790 ms，P95 TTFT 2,672.268 ms，
物理生命周期占用 11,507.847 GPU-s。其控制事件中，最早三次扩容均发生在
`queue_depth=0`、`active_requests=0`，而 `p95_ttft_ms=13,669.073` 使
TTFT 归一化压力超过 1；随后才出现由 active/queue 压力触发的扩容。该证据
支持一个窄的、可证伪假设：历史端到端 TTFT 窗口在当前工作已完成时不应单独
启动新的物理副本，否则会把已结束请求的延迟压力转换成空闲 GPU 生命周期。

候选策略不是把 7B 的副本上限改成 3。配置仍使用四张可用 GPU 作为物理
`max_instances=4`，实际副本数由 IEEE 控制器的 live queue、active capacity、
TTFT 窗口和已在途 activation 决定。候选只在以下条件同时成立时抑制一次
扩容执行：

1. IEEE max-score 已提出扩容；
2. queue 和 active 两个归一化压力均未越过 1，唯一超限分量是 TTFT；
3. 当前 live work 与已有 ready/pending capacity 的 ready-time projection
   已覆盖；
4. handoff projection 没有等待到达新副本的请求。

任一 queue/active 压力越界均沿原 IEEE 扩容路径执行。没有读取未来 trace 行，
没有修改九个公式、SLO、预算、工件、负载或后端。

## 2. 实现边界

论文规范语义保持为 IEEE §3.1 的 max-score 扩容和 all-low/cooldown 缩容。
`IEEEReplicaControl` 仍先计算原始 score；`ScenarioRunner` 只在动作执行前
调用已有的当前需求/ready-time handoff 投影，将本次动作标记为
`no_action_current_demand_covered`。控制事件中额外保存
`scale_up_guard=current_demand_projection` 与 ready-time queue 数，便于审计
“提出了扩容但为何没有激活”。

## 3. 资格检查

- `tests/test_basic_smoke.py` 新增两个边界测试：TTFT-only 压力且当前容量已
  覆盖时抑制；queue 压力越界时不抑制。
- 目标测试通过：3/3。
- `python -m py_compile scripts/run_all_experiments.py` 通过。
- 候选配置 `configs/ieee_tc/20261027_d920_7b_adaptive_demand_guard_full_w2.yaml`
  与 D917 共享模型、trace、adapter、远端目录、生成合同、SLO 和控制阈值，
  只新增 guard 开关、独立 output/NVMe/HOST 路径；`max_instances=4` 与
  `hardware.gpu_count=4` 一致。
- 更大 IEEE 集成测试在本机已有环境限制下于启动阶段因 NVML 不可用而停止；
  未产生 GPU 结果，未将该环境错误归因给候选实现。

## 4. 下一步判定

按既定安全包络、GPU idle/cleanup、真实远端服务冻结和单轮执行键运行 D920
一次。结束后先做资源释放、合同/计量审计和控制事件统计，再决定是否保留：

- 若正确完成、共同 TTFT/TPOT 条件与物理 GPU-s 同时改善，候选进入 PrimeLoRA
  开发基线；
- 若只减少 GPU-s 但尾延迟或联合 SLO 明显恶化，保留为 trade-off 证据并回退
  到 D917 语义；
- 若候选没有触发 guard，说明 D917 的触发条件不再出现，不据此声称性能改进。

不重复运行 D917，不按结果修改 guard 规则，不用本次结果选择对手或改变共同
比较协议。

## 5. D920 终态与修复

D920 在约 446 个请求终态、约 6 分钟回放后失败，不能进入任何性能汇总。失败
栈位于新守卫为 ready-time 投影构造 route-aware 模拟 Router 的路径；实际
IEEE Router 所需的 `service_bin_ms`、selection strategy 和 locality slack
没有从真实 Router 复制到模拟 Router，导致其错误拒绝构造。服务资源域被
正常释放，OOM、swap、high 和外部远端服务故障均未出现。

这不是放宽合同或重跑同一键的理由。修复为让模拟 Router 使用真实 Router 的
同一策略参数，保持模拟与实际选择语义一致；修复通过已有 route-aware/guard
定向测试和 Python 编译检查。下一次运行使用新的 execution key，D920 的失败
原始日志和 `launch.json` 永久保留。

## 6. D921 终态与第二个语义修复

D921 使用独立 execution key 重新触发了扩容路径，但在约 268 个请求终态后
停止，仍不能进入性能汇总。第二次失败不来自远端制品服务、GPU、磁盘、内存或
回放器；服务资源已正常释放，`launch.json` 将其记录为失败运行并保留原始日志。

失败位置为 `ieee_confirmed` 路由的 ready-time 投影：D921 已经复制了副本的
缓存与负载状态，却仍以没有 `ieee_snapshot` 的方式调用正式 Router。正式
Router 拒绝这种调用是正确的，因为它会把“最近一次 hint”误当成已提交的
候选状态。因而 D921 暴露的是预测器与正式路由合同之间的缺口，而不是证明
当前需求守卫无效。

修复内容是为投影中的每个候选副本构造不可变的
`ReplicaRoutingSnapshot`：使用复制状态中的活动请求、活动 adapter、待处理
负载、GPU 利用率和最近调度时间，使用冻结工件身份中的 rank/载荷身份，并由
冻结服务 profile 分类出对应的 D/T/O 类。扩容产生的尚未拥有 engine 的副本
只继承 profile 定义，不继承其他副本的在线 EWMA 或缓存提示；缺少 profile 或
工件身份时显式失败。随后投影仍调用 `Router.select_instance(...,
ieee_snapshot=...)`，因此没有回到 legacy routing，也没有放宽
`confirmed-tier` 约束。模拟副本的活动计数和 pending 集合同时与真实副本
分离，避免预测过程改变下一次真实控制快照。

这一定义保留了原假设的可证伪性：守卫只可能抑制“TTFT 单项超限且当前需求已被
已有容量覆盖”的动作；若 queue/active 压力越界，仍执行原 IEEE 扩容。修复后计划
使用新的 execution key。D922 已在服务 profile 装载阶段停止，尚未进入回放：新
配置中的 native-host workspace 审计摘要有一个字符错误，冻结 profile 正确拒绝了
model/configuration/context 不一致的配置。该失败同样不进入性能表；原始
`launch.json` 和服务日志保留。D923 将只修正这一配置摘要并使用独立输出路径，
继续验证前述 snapshot 语义。D920/D921/D922 的失败记录不用于筛选结果或改变
guard 规则。

## 7. D923 终态与容量边界修复

D923 在 280 个请求终态、全部成功且原生合同匹配的前缀后终止。终止发生在
下一次扩容需求投影，而非请求执行、远端制品读取、GPU、内存或磁盘路径；资源
域随后完整释放。投影对一个已达到运行并发上限的副本仍构造了
`admitted_after_accept = active_requests + 1` 的服务类别。该类别落入冻结画像
的 admitted-request 上界之外，因此被正确判定为未测量类别，但投影错误地把
这个不可能的准入继续交给类别查找并抛出异常。

修正保持确认路由的语义：当投影副本的活动请求数达到运行并发上限时，直接生成
`available_slots=0` 且没有服务类别/成本的不可行 `ReplicaRoutingSnapshot`；
确认路由随后排除该副本，仍对有容量副本使用冻结的 D/T/O 画像。该路径不插值、
不复制最近 EWMA，也不把未测量类别当作零成本。回归测试覆盖“饱和副本不生成
新画像类别”的边界；D923 原始日志和中止记录永久保留，不计入任何性能汇总。

## 8. D924/D925 终态与请求合同对齐

D924 在 577 个请求终态后停止，D925 在 37 个请求终态后停止；两轮已终止的请求均满足原生 token 合同，但整轮计划均未完成，故两轮都不具备正式比较资格。D924/D925 的失败均发生在 IEEE 扩容投影的服务类别查找，而不是远端读取或 GPU 执行。D925 的诊断证据给出了触发键：原始 trace 上界为 prompt=1099、output=379，而固定长度生成合同在真正请求执行前已将其规范化为 prompt 不超过 759、output 不超过 256。冻结服务画像覆盖的是后者。

此前投影直接读取 `expected_*_tokens`，把请求源数据上界误当成后端实际声明类别，造成控制面和执行面合同分叉。这不是画像不足，也不能用最近类别估计来绕过。修复让投影复用与请求路径相同的输入提示和固定输出规范化：使用统一的输入上界、`fixed_length_greedy_v1` 的 `min(source, cap)` 规则，再按冻结画像分类。D925 的完整诊断错误、577/37 个成功前缀及资源释放证据永久保留；D926 使用独立执行键验证该修复。

## 9. D926 终态与确认源线性化修复

D926 已通过 D924/D925 的类别边界并完成 2,229/4,000 个请求，所有已记录终态请求均成功，随后在异步 residency 任务中暴露 `ConfirmedSourceConflict: confirmed file source owner/epoch changed`。这轮同样不能进入正式性能汇总，但它把问题定位到文件源选择与确认租约之间的 TOCTOU：进程内 `LocalSourceReferences` 路径先取 source snapshot，之后才进入锁内执行 `acquire_confirmed`；另一准备/迁移任务可在这段间隙发布或回收同一 adapter，因而旧 epoch 被拒绝。

修复不重试旧计划、不接受过期源，也不改变 tier 优先级。进程内路径现在在同一 owner lock 内完成 snapshot、HOST/NVMe 源选择和确认租约安装，与已有的进程隔离 owner `acquire_best` 保持相同的原子语义。D926 的 2,229 个成功前缀、6 个中断计数、6,279.19 GPU-s 观测值及完整失败栈保留，D927 以新执行键验证该线性化。

## 10. D927 终态：合同闭合但未改善 PrimeLoRA 7B W2

D927 使用独立执行键验证了 D926 的线性化修复，不能与 D924--D926 合并为一轮
结果。运行完成全部 4,000 个请求，终态成功 4,000/4,000，唯一请求 ID 4,000
个，固定长度目标、vLLM 原生 token ID 来源和 `ieee_tc_native_v1` 计时合同均
完整；服务、回放器、watchdog、GPU lease 和清理均正常结束。原始主结果文件
SHA-256 为
`f87e6681bd23c2002504f100d000ddd58d5de32d98e6e67ed0871067c46d1f36`，请求终态
文件 SHA-256 为
`8afeba4246b5ee34680691120ab3c5225dca9d838923246ae886342d7d8c4960`。

按已冻结的 7B D394 分组阈值（TTFT=2995.3487/4633.3454 ms，TPOT=
59.7441/85.2470 ms）对同一结果文件逐请求重算，得到：

| 观测 | D927 | D917 直接参考 |
|---|---:|---:|
| 计划/终态成功 | 4000/4000 | 4000/4000 |
| TTFT 边际达标 | 3835/4000 | — |
| TPOT 边际达标 | 3979/4000 | — |
| 联合达标（时延合同） | 3816/4000 = 95.40% | — |
| 平均 TTFT | 1380.589 ms | 1124.790 ms |
| P95 TTFT | 3695.678 ms | 2672.268 ms |
| 平均 TPOT | 29.401 ms | — |
| 平均 E2E | 4822.053 ms | — |
| 物理生命周期 GPU-s | 14550.866 | 11507.847 |

因此 D927 只证明“当前需求守卫、请求契约和文件源线性化能够完成一次完整回放”，
没有证明该守卫带来性能收益。相对 D917，GPU-s 增加 26.44%，平均 TTFT 增加
22.74%，P95 TTFT 增加 38.42%；这一方向与当前 PrimeLoRA 优化目标相反，D927
不能作为新的开发基线或论文主结果。D917 仍保留原身份，D927 作为失败的性能候选
和完整生命周期诊断证据保留，不能按胜负删去。

控制面诊断显示 D927 发生 10 次 scale-up、7 次 scale-down；请求记录中有
795 条被标记为 scale-up affected、20 条 first-service，1,706 条在 dispatch
前未处于 GPU-ready，506 条发生 remote cold/mismatch。下一优化必须先用这些
已观测阶段分解定位造成 GPU-s 和尾延迟增加的控制路径，不能只继续调整守卫开关。

`summary.json` 和物理资源摘要仍将 `n_correct` 保持为 `null`，
`correctness_qualification` 为 `not_inferred_from_token_contract`。这是冻结协议
对数值 adapter 正确性的明确边界：4,000 个请求的原生身份、token 数量和时间合同
不等于独立的数值输出 oracle。D295 的受控数值重复也只发现近并列 token 的跨启动
变化，未建立完整池的 `N_correct`，因此本轮不把 `n_correct` 人工补成 4,000。

该结果的执行文件实际位于 D927 专属启动目录；实验结果 JSON 因复用冻结配置仍
落在 D924 结果目录下，但文件名含唯一 `d927_7b_adaptive_demand_guard_full_w2`
标签且 SHA 已登记。该路径差异不改变结果内容，也不覆盖 D924 原文件；后续分析须
同时记录启动目录和结果文件路径。
