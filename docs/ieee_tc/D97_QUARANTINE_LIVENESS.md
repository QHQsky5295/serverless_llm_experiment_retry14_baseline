# D97：停止服务后的副本必须仍有退出路径

2026-09-28。Prime IEEE Full 开发阶段。已在 CPU 反例基础上修改副本退出
实现；**没有新的 GPU 性能结果**。D96 完整失败证据已备份至 `9b23a49`。

## 问题与历史

D96 的 4,000 请求中仅 1,417 个成功。后期全部 ready 副本消失，但四个
物理 owner 一直到整轮收尾才退出；1,207 次在线扩容决策均无空闲设备。
D94 正确地禁止仍持卡的 draining 成员被视为空卡，但当时没有实现
未决原生操作的在线恢复流程。D95/D96 分别修正发送边界与并发状态采集，
不能替代这个生命周期步骤。

可证伪假设：保留请求 ownership 的路径把成员设为 draining，在线
dead-pruning 却只扫描 running；因此停止路由同时使该成员脱离退出检查。

## 实际控制路径反例

复用 `tests.test_ieee_tc_launch.DeploymentTerminalIntegration.owned_pool_runner`，
执行实际 `_retain_runtime_request_reservation`、`_prune_dead_instance_slots`、
`_retire_failed_slot`、`_select_dedicated_device_id` 和真实 InstancePool。
模型对象、native-reference intent 端点和 shutdown 完成端点是明确的 CPU mock；
这里不声称验证原生进程退出或物理释放。

| 起始条件（四个成员） | 在线退出调用 | 保留成员 | 设备黑名单 | 可分配设备 |
|---|---:|---:|---:|---|
| 未决 reservation→draining；engine alive | 0 | 4 | 0 | 无 |
| 未决 reservation→draining；engine dead | 0 | 4 | 0 | 无 |
| 对照：running；engine dead | 4 | 0 | 4 | 无 |

三个检查全部按预期复现，进程退出码 0。第二行证明仅仅等待 dead 标记
不能使现有检查重新访问 draining 成员。第三行是控制组：检查并非根本
不能退出成员，但现有 failure 语义会封禁设备。因此只把 get_slots 改为
get_all_slots 也不足以恢复容量。不能删除物理选卡保护来绕过该问题。

这是结构性活性缺口的证据，不证明 D96 每个取消的起因，也不证明修正
后全部 4,000 请求会成功；前期状态冲突／队列积压仍需独立验证。

## 原始来源核查与实现边界

- [vLLM v0.30.0 AsyncLLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)：
  `abort` 更新输出管理并发送 EngineCore 取消；`shutdown` 另行关闭 engine core
  和输出处理。请求取消与 runtime 退出不是同一事件。本地仍须用已有原生
  终态、进程身份和物理租约证明，不能从返回一次 RPC 推断整卡空闲。
- [Kubernetes Pod 生命周期](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)：
  terminating endpoint 不参与普通流量，同时仍保留终止状态并由生命周期
  管理完成进程退出。这里仅借鉴状态职责分离，不声称 Prime 部署了 Kubernetes，
  也不复制其默认超时作为新调参常数。

资料核查于 2026-09-28。九个公式、路由目标、planner/admission、工作负载、
模型 profile、1,800 秒请求保护及真实远端交付方式均未改变。

## 实现与本地复核（local-only）

退出分为三个有不同证据要求的步骤：

1. 原有未决操作检测将副本置为 draining，立即拒绝新的准入；每个真实
   controller reservation 有可查询的生命周期，不以请求队列为空冒充退出。
2. 每个所属 runtime 只有一个退出任务。等待仍绑定该副本的请求按原期限
   完成、取消或在已释放 reservation 后重新选副本；不主动取消正常 sibling。
   释放和请求 wrapper 退出均发送事件，没有新增轮询间隔或等待参数。
3. 复用已有准备任务收尾、engine shutdown、native worker pidfd 和物理租约
   释放流程。只有物理 allocator 确认释放，才注销未决 controller 记账及本地
   source pin，之后从实例池移除成员。软件隔离不新增永久设备黑名单。

请求 cancel 不负责取消独立退出任务；整池 shutdown 先 join 已有退出，避免
同时关闭同一 engine。退出异常通过在线控制和最终收尾传播，保留失败与物理
owner；不会清空未确认 RPC、改成请求成功或对未知执行盲目重试。

原生引用的历史 `unresolved` 证据不重写为成功 ACK；物理退出另记
`physical_runtime_retirement_v1`，相应引用状态为 `retired_with_runtime`。
新的状态日志写入正常结果和异常 `main_outcome`，供完整回放验证。

复核还发现现有 selected-source conflict 会在同一请求内释放并重置
reservation：仅等请求 wrapper 结束会漏掉离开旧副本的事件。新增实际
`_finish_runtime_request_reservation`＋`retry_known_conflict` 测试复现超时，
随后将通知绑定到实际 reservation 释放；这不是新增重试策略。

| 检查 | 状态／边界 |
|---|---|
| 原实现四副本活性反例 | 已复现；三种起始条件见上表 |
| 新退出实现前七项测试 | 7 errors；缺少退出任务／方法，原日志保留 |
| 第一轮定向测试 | 54 项通过，0.672 秒 |
| 重新选副本的退出通知反例 | 2 项中 1 个超时，取消保留测试通过 |
| 模型环境相关回归 | 1,138 项通过（包括新增 9 项退出测试） |
| 系统 Python 护栏回归 | 63 项通过，0.999 秒；与上一行合计 1,201 个不同测试 |
| 真实 GPU 退出、复用和 Full4000 | 尚待新完整回放验证 |

测试复用真实 InstancePool、实际请求 wrapper、reservation 和退出控制；
engine shutdown／physical release 端点为明确的 CPU fixture，不将它们称为
真实 GPU 退出证据。测试包括物理 ACK 前禁止复用、并发单一退出 owner、
保留正常 sibling、失败／取消保留 owner、local pin 及整池 join。

首次全组调用误将护栏测试也交给模型环境：1,201 项运行耗时 56.400 秒，
其中 4 项因该 Python 缺少 `signal.pidfd_send_signal` 失败。原失败日志保留。
按已冻结的解释器分工，未改任何生产／测试代码，只在 `/usr/bin/python3`
重跑 63 项护栏测试并全部通过；不是删除失败测试或放松保护。147 个历史
保护条目校验通过，计划与指标协议 SHA 未改变。

## 文件与下一步

原始目录：`results/ieee_tc/p2_backend_qualification/d97_20260928/`。
`quarantine_probe.json` 保留了原始 stdout 日志前缀，不能直接当纯 JSON 读取；
其末尾结构化对象已解析校验，未为纠正格式重复运行实验。
结构化摘要：`paper_results/ieee_tc/p2_backend/20260928_d97_quarantine_counterexample.json`。

下一步完成相关回归、来源／保护检查和本修正的可回退备份，然后返回
3B 完整 W0 主线。不堆叠全局 source epoch 等另一条优化；7B、baseline、warm SLO、Resident 与正式
M1/M2/A/S 均未因本 CPU 检查而获得资格。
