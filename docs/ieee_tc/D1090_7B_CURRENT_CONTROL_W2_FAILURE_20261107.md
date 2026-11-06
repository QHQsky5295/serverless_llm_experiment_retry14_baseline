# D1090：7B current-control W2 启动失败审计

执行键：`d1090_20261107 / 7b_current_control_w2`  
状态：**启动资格失败；不属于性能结果；不进入任何比较**

本轮在 2026-11-07 启动远端 7B 只读交付服务后进入受限 launcher，但在模型/远端
存储设置阶段退出。4,000 请求中 0 个到达、0 个提交，GPU 未加载，service 和
replay 返回码均为 1。原始 `launch.json`、`launch.launch/`、远端健康和停止回执
全部保留在：

```text
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1090_20261107/7b_current_control_w2/
```

## 根因

当前 runner 通过 `FAASLORA_SHARED_TRACE_PATH` 传入了冻结 W2 trace 和 adapter map，
但没有传入与该 trace 配对的
`FAASLORA_SHARED_ADAPTER_SUBSET_PATH`。因此当前 `run_all_experiments.py` 没有把
不可变 500-adapter subset 注入 `lora_adapters.adapters`；IEEE 远端输入合同在
`setup_remote_storage()` 中正确拒绝了空 adapter ID 集合：

```text
ValueError: IEEE input setup requires nonempty unique adapter IDs
```

这说明当前代码的保护逻辑正常工作，失败来自本次启动命令的输入衔接缺失，不是
PrimeLoRA 的 7B 服务性能或远端带宽问题。远端服务自身健康回执为
`ok=true`、`prepublished_gzip_v1`、`artifact_timing_v2`，随后已停止；没有请求期
读取、打包或传输污染。

## 处理

- D1090 只保留失败证据，不重用执行键，不从它计算 TTFT、SLO、GPU-s 或成本。
- 不修改旧 D931、不修改 trace/subset/工件池、不修改生产算法。
- 新的重验证必须使用新的执行键，并显式传入 trace 对应的
  `*_adapter_subset.json`；重验证结果与 D1090 分开封存。
