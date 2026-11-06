# D1093：7B current-control W2 启动失败审计

执行键：`d1093_20261107 / 7b_current_control_w2_user_manager_clean`

## 结果性质

D1093 不是性能运行。服务完成了 500 个 adapter 的准备、冻结 trace 的读取和
Full 场景初始化，但在第一个 preload 请求前退出；外部回放记录为
`N_arrived=0, N_submitted=0`。因此该执行不进入任何延迟、GPU 生命周期或 SLO
统计分母。

## 失败位置

失败栈位于 `SubprocessInferenceEngineProxy._worker_inputs` 调用
`_ieee_native_allocator_environment`，异常为：

```text
ValueError: native HOST allocator candidate conflicts with inherited allocator tuning
```

GPU worker 尚未创建，launch receipt 的 GPU cleanup 证据保持为空；远端 artifact
服务在回放未开始后已健康停止，停止回执为 `inactive`。

## 根因判定

这不是 adapter、trace、远端传输或 PrimeLoRA 调度机制的失败。D1093 清理了
systemd user manager 中的 allocator 变量，并且故意没有在 launcher 中设置
`FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY`/`PYTORCH_ALLOC_CONF`。然而当前
`run_all_experiments.py` 的导入契约是：若 launcher 没有显式 native policy，导入
阶段会设置兼容默认值
`PYTORCH_ALLOC_CONF=expandable_segments:False` 和
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`；随后 frozen model config
要求 `uncached_background_v1`，严格校验因此拒绝把两个 allocator policy 混在同一
个新进程中。

这个判定由隔离的无 GPU 环境探针复核：没有 marker/allocator alias 时会触发导入
默认；在进程启动前同时提供 marker 与精确的
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True` 时，校验通过。
因此可行的下一步是建立新的执行键，恢复历史 D931 已验证的“启动前显式设置”
契约；不修改算法、不放宽校验，也不重用 D1093 的执行键。

## 保留证据

- raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1093_20261107/7b_current_control_w2_user_manager_clean/`
- 失败服务日志：`launch.launch/service.log`
- 回放记录：`launch.launch/replay.jsonl`
- 清理前后 manager 环境摘要：`local_user_manager_allocator_env_before.txt`、
  `local_user_manager_allocator_env_after.txt`
- 远端健康与停止回执：`remote_post_failure_health.json`、
  `remote_service_stop.txt`

## 禁止的解释

不能把 D1093 写成 7B Full 的退化、不能据此调整 PrimeLoRA 参数，也不能把
`N_arrived=0` 当成低吞吐或高延迟证据。
