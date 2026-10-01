# D151 — 路由专用的实时原生观测边界

2026-10-02；P2/P3 开发验证，尚非性能接受。父版本 `805bf611`。

## 问题与可证伪假设

D150 已完成并备份，不重复分析。其 4000 请求原生生成契约完成，平均
TTFT 296.678 s、P95 553.255 s；平均 dispatch wait 294.697 s，仍占
TTFT 99.332%。平均实际原生服务 4.868 s，进入执行门至终态上包络 8.732 s
（不是精确 permit 持有时间）。这些数值提示服务外等待，但不能单凭它们
将全部排队归因于某个函数。103 个输出 hash 差异和数值 adapter 资格仍未闭合。

D149 已缩减 frontend→controller 的来源重检查信息。源码核查进一步发现，
GPU core→frontend 的该类读取仍调用完整 `source_snapshot`，构建并传送
staging 张量清单和 pinned allocator 观测。接收者 `NativeSourceSnapshot`
仅使用已注册 `native_footprints`、实际 owner/epoch/source/slot/clock，
不使用前两项。完整预算检查却确实需要它们，不能全局删除。

假设：以显式只读操作仅取得路由所需的完整实时注册图，省掉没有消费者的
staging/allocator 观测，可减少原生核心同步管理工作和消息体，而不改变
论文信息、路由公式或物理保护。若普通 Full 未改善或出现退化，保留结果，
不由组件调用次数推断系统增益。

## 单一候选与边界

新增 `routing_source_snapshot`，仍调用原 owner `source_snapshot()`，每次
执行 `_refresh`，保留实际来源、slot、epoch、保护集合、staged source identity
及其全部不变量；随后重新测量已注册 HOST 图和实际 GPU slot 图。frontend
仍完整校验注册图并投影，不在 GPU 核心增加解析校验，不缓存旧快照。

仅此只读操作不构建 `native_staging_footprints`、不读
`native_host_allocator`。完整 `source_snapshot` 的内容与行为保持不变；
所有真实 allocation、HOST budget、planner/admission 路径仍执行原完整检查。
旧路径偶然在路由时发现的 staging/allocator 观测错误，不再由路由信息读取
负责发现；这不授权跳过真实预算/分配检查，也不吞掉 owner 或注册图错误。
传输失败照常向上抛出，不回退旧快照；纯观测失败不伪称持有未知 mutation。

不改九个公式、lease/保护、缓存策略、并发上限、timeout、profile、模型、
权重、输入、远端交付或资源限制。只维护这个候选。

## 历史与原始来源

- D132：完整图验证保留在独立 frontend；不把解析工作搬进 GPU loop。
- D144：同一次观测消除重复遍历；未授权跨调用 tensor inventory 缓存。
- D149/D150：选定来源复核仍必须保留，完整回放改进不能外推为正式最优。
- [vLLM 官方 CPU/异步优化说明](https://vllm.ai/blog/2024-09-05-perf-update)：
  CPU 处理和同步控制可以限制 GPU 服务。这仅支持检查方向，不移用其收益。
- [vLLM 0.30.0 model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  adapter 注册含原地 packing/optimization，不能以 adapter ID 相同假定张量
  元数据永久不变。保留逐调用观测，未采用跨调用缓存。

## 验证状态

| 检查 | 状态 |
|---|---|
| 实时来源、两次状态变化、注册图和 owner 错误 | 通过；新状态实际重读，非法图与 source 替换拒绝 |
| staging/allocator 不在路由路径，完整端点保持 | 通过；路由调用 0 次，完整端点保留非空 staging 检查 |
| 真实 frontend/RPC 转发与只读失败 | 通过；五项定向检查 0.012 s |
| 受影响回归 | 737 项通过，24.735 s；命令 35.43 s，峰值 RSS 1211732 KiB |
| 普通 4000 请求 Full | 待执行；先满足磁盘启动门槛 |

正确性阶段使用状态表，不把模拟对象检查画成 GPU 性能图。原始日志保存到
`results/ieee_tc/p2_backend_qualification/d151_20261002`，不覆盖 D150。

检查均在独立 CPU 资源域（3/4 GiB、swap 0、CPU 2,3,26,27）完成，无 CUDA
执行、无新模型/trace/工件。每轮实际 InvocationID 已保存，结束后的资源域
自动删除；逐一核验不存在，不伪造删除后已不可读取的 memory.events。

失败与修正保留：RED 五项为 2 failures/2 errors，完整旧端点检查通过。
首次 GREEN 仅 owner 替换注入失败：测试修改的是 cache 的只读映射副本，
改用缓存本身的赋值后五项通过。第一次回归 14 failures/75 errors，源于
旧测试替身将新 worker 操作名直接转给 owner；补齐真实 worker 映射后第二次
仍有三处旧操作名拦截（2 failures/1 error），更新注入条件后 737 项全部通过。
全过程没有为通过测试再修改生产候选、放宽断言或改 timeout。原失败日志、
测试 diff 和各次源文件 SHA 全保留，不称失败测量为系统性能实验。

结论：只读信息边界获得 CPU 正确性资格，可以进入同配置普通 Full；尚不
接受任何整体性能收益。未新增第二个候选，也不重复 D150 已封存分析。
7B/3B 性能、数值 adapter、输出 hash 差异、warm/Resident 标定以及主比较、
消融、敏感性仍未闭合，baseline 继续暂停。
