# D132：新鲜路由观测的进程边界投影

状态：开发期候选；组件对照完成，普通 Full 验证尚未执行。基线仍暂停。
未修改 IEEE 九个公式、输入/权重、缓存生命周期、生成合同、并发容量或超时。

## 问题与可证伪假设

D131 的 7B W0 全部 4,000 请求均到达并获得终态，只有 3,749 条原生合同成功，
不能进入主比较。成功子集的平均 TTFT 为 902.616520 秒，入服务前等待占主要部分。
这不是数值 adapter 身份或最终 SLO 已通过的证据。

历史 D123 的采样指向 RPC/footprint 处理，但不能将旧 CPU 采样比例当作 D131
当前耗时比例。D127 单独更换 JSON decoder 的组件结果已经归档，本轮不重复。

当前路由向每个副本请求完整原生存储图，再在集中路由进程解码和验证；实际上
路由只消费已验证的 source identity、tier、实际 footprint/representation 与 GPU UUID。
假设：在已有独立后端 frontend 中验证新鲜完整图、再投影必要字段，可以减小
集中路由的消息与同步工作，而不改变候选排序和执行前的物理复核。

## 边界与不变量

- 每次仍执行原有 native `source_snapshot`，不使用 TTL 或完成后的观测缓存。
- GPU core 构造完整观测的路径不改；完整 `_footprints` 验证在独立 frontend 中执行，
  不把额外验证塞入 GPU core 的执行循环。
- 新只读 RPC `ieee_routing_sources` 在完整验证后输出
  `native_lora_routing_sources_v1`。它不授予引用、reservation 或 GPU 可执行权。
- 接收方继续检验类型、owner/epoch/clock、capture time、ID 覆盖、GPU copy confirmation、
  已知 source 的容量与表示。共享存储总量保留 distinct union，不把 adapter footprint 相加。
- 同一进行中的采集仍可共享；成员变更、过期 epoch、每个请求自己的 live capacity
  和原子 reservation 均按原逻辑重新检查。
- planner、replacement、admission 和实际加载前的 native ownership/physical checks
  仍使用完整接口。新端点缺失/错误不静默退回其他路径。
- 投影是可信本机 worker 的类型化消息边界，不是密码学证明，也不是缓存的“验证通过”布尔值。

设计参考 [vLLM CPU/GIL 与进程隔离分析](https://vllm.ai/blog/2024-09-05-perf-update)、
[v0.30.0 异步 engine 客户端源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core_client.py)
及 [Python asyncio 阻塞工作说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)。
这里只借鉴执行上下文和消息边界，不借用其硬件性能数字或将投影包装成新的调度算法。

## 正确性与安全

第一轮 262 项测试通过（4.168 秒；整条命令 12.15 秒，最大 RSS 985,156 KiB）。
覆盖原生身份、共享存储、HOST/GPU 表示、未知/未确认 source、过期 epoch、成员替换、
并发共享观测、取消、真实 dedicated worker/proxy 通信及只读取消不产生未决 native 所有权。
原始图每次都重新验证，坏 alias/exclusive-capacity 和错误 clock 被拒绝。
后续补入同源 A/A 排序检查，不改已测 serving 代码。最终 925 项回归全部通过
（131.360 秒；整条命令 142.72 秒，最大 RSS 1,177,496 KiB），包括 A/A 排序、
planner/admission、原生生命周期、外置回放和基础 smoke。147 个历史保护对象
校验通过；计划、冻结指标与已测 serving 源码 SHA 均一致。

测试与组件测量均在实际 3/4 GiB、swap=0、CPU 2,3,26,27 的资源域内运行；
子进程继承同一边界。没有 GPU 或远端服务启动。已结束的两个域均确认空后关闭，
high/max/OOM/oom_kill 事件均为 0。

## 组件对照结果

复用 D121/D127 已保存的 7B 原生观测：2 个 registered adapter、512 个 HOST allocation。
使用真实独立 frontend 进程、production dedicated TCP 协议与接收方代码；只有
原生 GPU 采集被同一历史观测替代。不复制权重、不生成新负载。

两种方法分别保留首个冷 RPC，随后 3 轮交替顺序，每方法每轮 10 次。
全部 62 次返回的不可变 source state 与同一完整验证结果精确相等。
下表是 30 次暖组件观测的算术均值；不是独立 workload seeds，不出系统级 CI。

| 指标 | 完整消息＋父进程验证 | worker 验证后投影 | 相对下降 |
|---|---:|---:|---:|
| RPC＋接收方验证（ms） | 19.809265 | 1.776892 | 91.0300% |
| 接收方验证（ms） | 0.944488 | 0.086930 | 90.7960% |
| 每次父循环 heartbeat 最大迟到（ms）的均值 | 13.518934 | 0.626171 | 95.3682% |
| RPC 响应字节均值 | 479,168 | 1,881.967 | 99.6072% |

解释：此历史组件上集中进程同步工作与消息明显减少；完整验证没有被省略。
响应字节的微小变动来自诊断时间数值字段。heartbeat 每毫秒采样，仅属于本次
组件对照，不进入正式服务。worker 正常退出，整条测量命令 19.45 秒，最大 RSS
1,133,944 KiB。处理时间包含 producer validation/projection、TCP 和 receiver 工作，
但不含真实 GPU 状态采集、加载、生成或多副本竞争。

## 判定与下一步

允许进入最终回归和一个普通 7B W0 Full 验证，不宣布系统性能提升已成立。
两 adapter 的小状态不能外推满 CPU cache，更不能把约 18 ms 的组件改善直接解释
为消除了约 900 秒的排队。Full 仍需按全部 4,000 请求检查成功/失败、阶段时延、
GPU 生命周期、远端交付与实际资源事件；保留任何退化。
特别检查独立 frontend 中的同步验证是否推迟其 token 事件处理：工作被移动而非
消失，GPU core 独立并不保证 frontend 没有竞争，必须同时查看 TPOT 和终态返回阶段。

冻结指标依据 `METRIC_PROTOCOL_FROZEN_V1.md`，SHA
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
共同 warm/reference 数值尚未冻结，不能将开发期显示的阈值当成正式 SLO。

原始目录：`results/ieee_tc/p2_backend_qualification/d132_20261001/`。
来源与每次组件观测保留在 `routing_projection_probe.json`；最终 curated CSV/JSON
位于 `paper_results/ieee_tc/p2_backend/20261001_d132_validated_routing_projection/`，
含 62 条观测、4 行汇总及完整最终验证回执。小型源码包包含 24 个文件，56,079 字节，
SHA `ffba6aa1eaca73dbf6c396ce72650d7b544fe2b9d09908c74b6ac03c4320c8af`。
没有覆盖旧结果。
