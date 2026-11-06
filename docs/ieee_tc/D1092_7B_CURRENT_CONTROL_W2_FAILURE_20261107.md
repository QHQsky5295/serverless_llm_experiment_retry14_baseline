# D1092：7B current-control W2 allocator 清理尝试失败审计

执行键：`d1092_20261107 / 7b_current_control_w2_allocator_clean`  
状态：**启动资格失败；不属于性能结果；不进入比较**

D1092 已验证 adapter subset 衔接正确，并在 launcher 环境中显式传入三个空的
`PYTORCH_*_ALLOC_CONF` 变量，但 native HOST worker 仍报告 allocator 冲突；0/4,000
请求进入回放，GPU 没有完成模型初始化。原始证据保留在：

```text
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1092_20261107/7b_current_control_w2_allocator_clean/
```

启动前保存的 user-manager 环境显示，systemd user manager 仍持有：

```text
FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
```

因此空变量只清除了 tmux 子命令环境，未清除 systemd user manager 在受限 scope
中的继承值。当前代码拒绝这种未明确隔离的 inherited tuning，避免把 7B 运行放在
不可复核的 allocator 状态下。D1092 不产生 TTFT、SLO、GPU-s、成本或 baseline 数字。

远端服务健康回执正常，未发生请求期传输；服务已在失败证据收集后停止。该失败
进一步说明问题是本机 launcher 的 user-manager 状态污染，不是远端链路或 PrimeLoRA
机制退化。

## 下一步

在新的 D1093 执行键前，先记录并显式清除 user-manager 中的三个 allocator 别名和
PrimeLoRA allocator marker；随后由 frozen model config 唯一注入
`uncached_background_v1`。清除动作只作用于本项目当前用户的实验 scope，不改生产
代码、模型、远端服务或评价口径。
