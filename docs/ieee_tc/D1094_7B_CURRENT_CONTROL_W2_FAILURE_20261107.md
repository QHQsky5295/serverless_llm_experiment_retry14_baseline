# D1094：7B current-control W2 运行失败审计

执行键：`d1094_20261107 / 7b_current_control_w2_allocator_contract`

## 结果性质

D1094 通过了 allocator 启动契约和 GPU worker 初始化，但不是完整性能运行。
服务在第 36 个到达请求附近退出；回放记录为 `N_created=36`、
`N_submitted=35`，随后 `replay_incomplete`。不生成延迟、GPU 生命周期或 SLO
正式统计，也不进入 baseline 比较。

## 观测到的失败

服务日志同时出现：

```text
RemoteArtifactError: download failed for code_lo
RemoteArtifactError: download failed for educati
RuntimeError: failed movement requires explicit owner recovery, not an automatic retry
```

allocator 冲突没有再次出现；模型 worker 已经创建并运行过短时间。随后受控服务
scope 完整退出，`native_gpu_context_release_confirmed=true`，远端服务停止回执为
`inactive`。

## 根因判定

D1094 启动前检查确认了远端 endpoint 为 inactive，但错误地把“inactive”当成了
可继续条件；本次 launcher 没有先启动
`primelora-artifact-tc-7b.service`，而 runner 又不会替调用者启动该服务。因此
请求触发真实远端读取时 endpoint 不可达，下载失败是基础设施编排错误，不是
PrimeLoRA 的 residency、routing 或 admission 机制退化。

这也解释了为什么失败集中发生在第一次需要远端移动之后，而不是在 preload 或
allocator 初始化阶段。该执行的 raw 证据保留完整，不能修补为成功运行或把失败
请求计入分母。

## 下一步边界

新执行键必须在进入 GPU qualification 前完成：

1. 远端 user service start；
2. bearer health 返回 `ok=true` 且 delivery/timing contract 与 frozen 配置一致；
3. 运行结束（成功或失败）无条件 stop service 并保存停止回执。

除上述远端生命周期握手外，不改变 D1094 的模型、trace、adapter subset、
allocator marker、代码版本或运行参数。
