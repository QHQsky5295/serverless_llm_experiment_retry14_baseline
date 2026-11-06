# D1093：7B current-control W2 user-manager 隔离登记

执行键：`d1093_20261107 / 7b_current_control_w2_user_manager_clean`  
参照：D1090–D1092 启动失败审计；D931 历史 current-control 候选

## 唯一新增处理

在启动前保存 `systemctl --user show-environment` 的非敏感 allocator 摘要，并执行
以下局部清理：

```bash
systemctl --user unset-environment \
  PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF \
  FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY
```

清理只针对当前用户的 systemd manager 环境，避免旧实验的 allocator 状态泄漏到
本轮 scope。随后 runner 不使用环境变量覆盖 allocator；当前 frozen model config
唯一注入 `uncached_background_v1`，worker readback 必须与配置一致。

## 其余合同

模型、vLLM 0.30.0、D931 YAML、W2 trace、500-adapter subset、remote endpoint、
fixed-length greedy、资源包络和 D648 指标全部不变。新的 raw 为：

```text
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1093_20261107/7b_current_control_w2_user_manager_clean/
```

D1093 成功闭合后才进行性能审计；若再次启动失败，保留 raw 并停止此 allocator
分支，不修改 PrimeLoRA 算法或重复同一执行键。
