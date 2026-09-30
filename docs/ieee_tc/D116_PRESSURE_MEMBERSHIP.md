# D116：以运行时生命周期管理共享传输成员

日期：2026-09-30。D115 完整失败证据已在 `d36581421413e0a4bc70354bdf191fa06375435e`
备份后才开始本项。范围：一个 CPU 因果反例及其候选修复，不是 GPU 性能实验。
不改变 IEEE 九个公式、路由、驻留选择、admission 压力定义、资源预算或超时。

## 历史证据与可证伪假设

D115 的 4,000 请求均完成，但一个初始副本的驻留 epoch 在最后请求终态之后
0.512 s 失败。错误来自 `SharedFileTransferDomain.attach`。最终四成员均
`retired`，没有 `uncertain`，且所有实际传输已结束。原记录没有保存报错瞬间
的具体成员身份/状态，因此不能仅凭相同异常字符串宣布完整历史根因已证明。

源码将 pool 的 running 副本列表先复制到本地，然后逐个 `await attach`。
在这些等待之间，另一个任务可以把某副本改为 draining 并完成 retire；旧列表
仍包含该副本。假设是：**非目标副本的正常退出，错误阻断了仍有效目标的准备。**
与之相反，目标副本自己的失效必须阻止新操作；不能笼统忽略 attach 异常。

`79e0a6e` 引入的共享域已经具备正确的原语：attach 时重放未结束区间、
retire 时阻止新参与并等旧操作收尾、run 时在同一把锁下确定参与成员。
本轮修正的是调用方绕过该生命周期、重新遍历旧 routing 列表的做法，
不是另建一个缓存、增加轮询频率或放宽保护。

## 原始最小验证

复用 `tests.test_ieee_tc_transfer_pressure.TransferPressure`，调用真实
`ScenarioRunner._run_ieee_file_transfer`、`SharedFileTransferDomain`、
`InstancePool` 和 native journal。只有 native RPC/file IO 边界使用既有 fixture。
两个运行时先正常订阅；事件固定等待与退出顺序，无计时 sleep、GPU 或实际工件。

| 条件 | 修改前 | 候选修改后 | 应有语义 |
|---|---|---|---|
| 两副本均可用 | 操作完成，两者压力各为 1 | 相同 | 保留共享资源竞争 |
| 等待期间仅非目标退出 | 相同异常，操作未开始 | 操作完成，目标压力 1，退出者 0 | 其他副本可以继续工作 |
| 等待期间目标退出 | 拒绝，操作未开始 | 仍拒绝，操作未开始 | 不复活已退出目标 |

这是执行顺序的反例与修复测试，不是对 D115 的统计重复，也没有速度百分比。

## 候选实现与约束

1. 运行时在激活/加入服务池前订阅共享域，原路径已实现此顺序，保持不变。
2. 文件传输只确认显式目标；不再重新订阅之前取得的全体 running 列表。
3. 共享域在 transfer publication 的同一锁中重查 `required_engine`。
   目标在 attach 后退出也不能开始 IO。target 为 None 的预激活文件操作仍合法；
   后加入的运行时必须重放已有压力区间，原规则不变。
4. 同一锁下读取参与成员；retiring/retired 非目标不接收新区间，但旧区间
   继续等待真实 finish。uncertain 成员仍失败，不吞异常、不重试未知操作。
5. HOST 字节 reservation、原生 owner/clock 校验、取消 settle、physical GPU
   release 全部保留；订阅退出不等于 GPU 已释放。

额外回归覆盖：目标在 attach 返回与 run 之间退出、从未订阅的显式目标、
重复逻辑 slot、预激活后加入、取消、丢失 finish、未决 native reply、真实
slot cleanup。350 项 transfer/native-retirement/request-lifecycle 测试通过，
15.045 s。另 401 项 launch/legacy 回归通过，47.844 s；合计 751 项。
这仍不替代完整 GPU Full 回放。

## 证据收尾

- 前/后 CPU probe 均退出 0，wall time 7.91/7.93 s、峰值 RSS
  930464/982236 KiB，主要包含模块导入，不把这些时间解释为服务性能。
- 汇总：`paper_results/ieee_tc/p2_backend/20260930_d116_pressure_membership/summary.json`，
  SHA256 `92a50dce49971f57fe1d654fb51ee253701d6de71a2cb0268cc859ed52241a54`。
- D115 的 69 个冻结来源中，3 个引用按预期改变（runner 两种路径写法及
  transfer 测试）。其余未变；resource_coordinator 不在旧 69 项清单内，
  本次用运行前 Git 内容与运行后 SHA 独立核验，不能声称旧清单已覆盖它。
- 首次 curator 对旧清单覆盖范围的断言错误，在资源清理/生成汇总之前停止。
  原脚本和日志保留为 attempt1；只改正该来源假定，未修改测量或重跑测试。
- 147 个历史对象及冻结计划/指标零变化。30 成员小型来源归档逐成员校验通过，
  `SHA256SUMS` 验证汇总、来源归档与成员清单；无大型原始数据或凭据入库。
- 五个 probe/test/curator 资源域均按 exact-owned/empty 证据停止；high/max/OOM
  全部 0。没有 GPU、远端或分析任务遗留。D115 失败缓存根仍保留，未删除。

## 原始实现资料核查

- [Python 3.12 asyncio 取消与 shield](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)：
  shield 不表示外层调用未取消；必须保存并等待仍执行的操作。本次不取消该
  等待约束，也不以捕获所有异常来宣布退出成功。
- [vLLM 0.30 AsyncLLM shutdown 源码](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/engine/async_llm/#vllm.v1.engine.async_llm.AsyncLLM.shutdown)：
  后端退出涉及 engine-core 与后台通信清理。成员生命周期和压力记录应在
  关闭这些对象之前收尾；调用 shutdown 本身不替代本项目的物理释放证明。

借鉴的是显式资源所有权和退出顺序，不声称这些资料提出了本文算法，也不
把跨副本准备/退出问题包装为新的论文贡献。

## 另一个已发现但本候选未修改的问题

D115 `scale_down_event_log` 还包含请求总数为 4,000 的终尾缩容；对应
`ScenarioRunner.run` 在单 phase 结束后的历史 `_scale_down_one_instance`
调用，而非最后的 IEEE 在线控制决策（后者为 no_action）。随后还调用
`_cleanup_extra_instances`。它们与最终 `_shutdown_instance_pool` 的统一
收尾顺序需要对齐，且不能冒称自然 scale-out/scale-down 证据。

下一步限定为检查这个明确的终尾分支和同一生命周期合同，不再泛泛优化
3B 局部性能；不改变历史记录。完整重放之前须确定 IEEE 请求终态后进入
统一收尾，而不是额外注入旧控制决策。然后按主线验证 Full/7B、warm/Resident
及已暂停的 baseline。D116 候选尚未通过完整 Full，不宣布 G1/G2 领先。
