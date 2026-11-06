# D1092：7B current-control W2 allocator 环境隔离登记

执行键：`d1092_20261107 / 7b_current_control_w2_allocator_clean`  
参照：D1090、D1091 启动失败审计；D931 历史 current-control 候选

## 唯一修正

D1092 保持 D1091 已验证的 500-adapter subset、W2 trace、remote endpoint、模型、
服务参数和资源包络不变。唯一改变是在进入受限 launcher 前显式设置：

```text
PYTORCH_ALLOC_CONF=
PYTORCH_CUDA_ALLOC_CONF=
PYTORCH_HIP_ALLOC_CONF=
```

这样消除 tmux/systemd user manager 遗留的 allocator 环境，随后当前模型配置中的
`ieee_native_host_allocator_policy: uncached_background_v1` 负责注入唯一候选值。
若仍检测到非空继承值，launcher 必须失败，不得绕过检查。

## 新输出与闭合

- raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1092_20261107/7b_current_control_w2_allocator_clean`
- curated：`paper_results/ieee_tc/p2_backend/d1092_7b_current_control_w2_allocator_clean/`
- 远端服务仅在回放期间 active；健康回执、transfer journal、停止回执和环境摘要
  写入 raw。
- 4,000 请求、物理 allocation-union、联合 TTFT/TPOT、P95 TTFT、阶段恒等式和
  `n_correct` 仍按原冻结协议审计。
- D1090/D1091 的失败不进入性能分母；D1092 若失败仍保留为启动审计，不重用执行键。

除上述三个 allocator 环境变量外，D1092 不改变任何配置、输入、后端、生成合同、
远端交付或 PrimeLoRA 机制。
