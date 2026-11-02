# D1012：3B pressure-adaptive arrival-prefetch smoke 结论

执行键：`d1012_smoke_20261102 / d1012_3b_pressure_adaptive_prefetch_w1_smoke`  
配置：`20261102_d1012_3b_pressure_adaptive_prefetch_w1.yaml`  
日期：2026-11-02

100-request smoke 已完成，且满足进入唯一 4,000-request 回放的前置条件：

- 100/100 native terminal，100/100 successful，0 failed；
- output contract 与 native timing contract 完整；
- `remote_mismatch=0`；
- 29 个请求产生 `confirmed_prefetch_published` reconciliation，71 个请求没有
  需要等待的 arrival movement；
- P95 TTFT 982.62 ms，smoke 级 SLO attainment 100%；
- GPU lifecycle summary 正常生成，退出后未见残留 dedicated worker 或 compute
  process。

Smoke 只验证 policy wiring、资源释放和请求级证据，不作为 4,000-request 性能结论，
也不覆盖 D1000。full 回放使用新的执行键和同一冻结 W1 trace/adapter map；若 full
运行中发现服务级协议或资源释放异常，立即按失败分类保存，不将其改写成性能结果。

