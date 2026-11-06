# D1094：7B current-control W2 allocator contract retry

执行键：`d1094_20261107 / 7b_current_control_w2_allocator_contract`  
前序：D1090–D1093 启动审计；D931 历史 current-control W2 候选

## 唯一修正

D1093 已确认失败原因是 launcher 没有把 frozen native policy marker 传入新进程，
导致导入阶段的兼容默认 allocator 与模型配置冲突。D1094 恢复 D931 已验证的
启动前契约：

```text
FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
```

配置、模型、vLLM 0.30.0、W2 trace、500-adapter subset、remote endpoint、固定
生成合同、资源包络和 D648 审计规则均不变。该 marker 不是运行时调参，而是让
新进程不注入 `expandable_segments:False` 的启动身份字段；worker 仍需通过现有
严格 readback。

## 启动前验证

在申请 GPU 和启动远端服务前，用同一 venv 做无 GPU 隔离探针，确认：

1. marker 与精确 allocator setting 可被 nested systemd scope 继承；
2. `_ieee_native_allocator_environment` 接受当前 frozen model config；
3. 不存在 `PYTORCH_CUDA_ALLOC_CONF` 或 `PYTORCH_HIP_ALLOC_CONF` 旁路 alias。

探针只写入本执行键的 `prelaunch_allocator_probe.txt`，不改变生产代码和旧结果。

## 验收

- 4,000/4,000 成功，W2 回放和 adapter map 完整；
- 请求级阶段恒等式、物理 GPU allocation-union、联合 TTFT/TPOT SLO、P95 TTFT、
  native HOST readback 和 `n_correct` 按现有 D648 规则审计；
- 若仍在请求前启动失败，保留 raw、停止远端服务并终止该 allocator 分支，不能
  重用执行键或放宽校验；
- 若完成，先做完整性/指标审计，再决定是否进入 baseline 阶段。

## 预期解释边界

D1094 只用于把同一 7B current-control 候选迁移到当前代码版本的可比运行；它
不能被表述为新的算法改进，也不能与 D1090–D1093 的启动失败混入性能统计。
