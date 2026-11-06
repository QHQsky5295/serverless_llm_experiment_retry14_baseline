# D1091：7B current-control W2 第二次启动失败审计

执行键：`d1091_20261107 / 7b_current_control_w2_retry`  
状态：**启动资格失败；不属于性能结果；不进入比较**

D1091 已正确注入冻结 W2 trace 的 500-adapter subset，并完成远端交付健康检查；
随后在 native HOST worker 启动阶段退出，4,000 请求中 0 个到达、0 个提交，GPU
没有进入服务。原始证据保留在：

```text
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1091_20261107/7b_current_control_w2_retry/
```

服务日志的唯一错误为：

```text
ValueError: native HOST allocator candidate conflicts with inherited allocator tuning
```

当前代码的 `_ieee_native_allocator_environment()` 正确拒绝了继承的 allocator
设置，避免 native HOST 实验在未冻结的环境下运行。D1091 的输入衔接已经通过：
日志显示 `500 adapters ready`，shared subset 路径和 W2 map SHA 均正确；因此这次
失败不是空 adapter、远端服务、模型性能或请求路径问题。

远端服务健康回执为 `ok=true`、`prepublished_gzip_v1`、`artifact_timing_v2`，
没有请求期传输；服务已在失败证据收集后停止。D1091 不产生 TTFT、SLO、GPU-s、
成本或 baseline 比较数字，也不覆盖 D1090。

## 下一步

使用新的执行键 D1092，在启动命令中显式清空
`PYTORCH_ALLOC_CONF`、`PYTORCH_CUDA_ALLOC_CONF` 和 `PYTORCH_HIP_ALLOC_CONF`，
再由冻结配置唯一注入 `uncached_background_v1`。不修改 serving code、模型配置、
工件、trace、adapter map 或评价口径。
