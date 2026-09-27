# D94：路由退出不等于物理 GPU 释放

2026-09-28。阶段：Prime IEEE Full 开发资格；不是主性能结果。

## 历史证据与可证伪假设

D93 的 3B W0 完整回放 attempt2 仅完成 320/4000 请求。29 条请求到达
1800 秒保护期限后，部分副本被标为 draining；随后四次新激活因原 GPU
仍有 compute/owned/unknown contexts 而失败。所有 GPU 最终均已释放。
这支持检查设备占用状态，不能把早期长排队全部归因于该终止错误。
原始结果和条件时延见 D93 文档及
`paper_results/ieee_tc/p2_backend/20260928_d93_3b_full_w0_attempt2.json`。

历史提交 `857fc7e` 为关闭流程增加了 `InstancePool.get_all_slots()`，
但选卡仍使用仅包含 running 的 `get_slots()`。此外，缩容等路径在
`await shutdown()` 前移除成员。假设是：控制器把停止服务错误地当作
设备已归还，因而与最终的物理安全检查冲突。

## 外部依据与本地边界

- [Ray 资源文档](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
  区分调度使用的逻辑资源与物理使用状态。本地设计必须保留这一区分；
  这不证明本次具体故障由 Ray 导致。
- [vLLM v0.30.0 EngineCore 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  的 shutdown 显式关闭 executor/scheduler 并清理分布式状态和内存。
  从路由列表删除不是该过程的替代品。本项目仍以既有原生进程退出及
  GPU census 证据确认物理释放，不以此源码推断本机释放时刻。

以上原始资料于 2026-09-28 核查。未引入新公式、超时、人工重试或配置。

## 修正的生命周期不变量

| 阶段 | 是否参与路由 | 是否阻止同 GPU 新激活 | 退出条件 |
|---|---|---|---|
| running | 是 | 是 | 缩容、异常或未决原生工作 |
| draining / 正在关闭 | 否 | 是 | 原有 shutdown 成功返回 |
| 关闭失败或取消 | 否 | 是 | 保留成员及原有未决证据，不冒充成功 |
| 已确认关闭并移除 | 否 | 否 | 后续选卡仍经过既有物理检查 |

选卡读取完整保留成员，同时继续排除 pending 和 failed 设备。
IEEE 缩容、通用缩容、异常退休、额外副本收尾、整体关闭和已发布激活
回滚都先设 draining，待原有关闭/预算归还成功后再移除成员。
整体关闭某个副本失败时，仍处理其他副本；失败成员不消失。
不会自动恢复未决原生请求，不把所有副本 draining 当作免费空卡。

论文九个公式、路由排序、admission、原生引用、失败传播、物理安全
检查和生命周期计费均保持不变。此修正是所有权实现一致性，不是
本系统的新算法贡献，也不能单独证明性能改善。

## CPU 反例与检查表

以下使用真实 InstancePool 和 runner 控制路径；模型/关闭完成事件是
显式 CPU fixture，未使用 GPU，也未产生推理性能数据。

| 检查 | 修正前 | 修正后 |
|---|---|---|
| 四个 draining 副本仍持卡 | 错误选择 GPU 0 | 不返回可用设备 |
| 缩容关闭尚在等待 | 成员已被删除、状态 stopped | 保留 draining；确认后才可复用 |
| 一个关闭失败、另一个成功 | 失败成员也被删除 | 保留失败成员，成功成员可释放 |
| 既有退出/控制检查组 | — | 19 项通过，0.471 秒 |

红测试实际结果：3 项中 2 failure、1 error；error 是失败成员消失后的
`None.status`，不是测试依赖安装失败。后续完整回归记录保存在
`results/ieee_tc/p2_backend_qualification/d94_20260928/`。
最终相关回归 **841 项通过，49.844 秒**；使用既有 CPU 测试环境，
3/4 GiB high/max、零 swap、辅助 CPU 集合、离线模型访问。147 个历史
保护项和冻结计划/指标协议 SHA 不变。机器仍无推理任务，未重启远端。
结构化回执：`paper_results/ieee_tc/p2_backend/20260928_d94_physical_owner_retirement.json`。

## 剩余主线

尚未解释 D93 成功请求中平均约 517.5 秒的 dispatch/admission wait。
下一步检查：未发送 generation 却尝试退休的边界，以及并发全副本
source_snapshot 的过期/冲突放大。不得仅凭本次 CPU 检查重跑原配置，
或宣称完整 4000、共同 SLO、GPU-s 领先。7B 与基线仍暂停。

实现检查采用 experiment-bridge 的先反例、后最小验证流程；本轮
review 为 local-only，没有独立审稿代理结论。正式性能图暂不适用，
本表是当前正确性证据交付。
