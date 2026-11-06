# D1095：7B current-control W2 remote-handshake retry

执行键：`d1095_20261107 / 7b_current_control_w2_remote_handshake`  
前序：D1094 allocator contract retry；D931 frozen current-control W2

## 唯一修正

D1094 已证明本地 native allocator 启动契约可用；唯一新增的是远端 artifact
生命周期握手。launcher 在受限实验启动前通过已配置的 SSH alias 启动
`primelora-artifact-tc-7b.service`，轮询 bearer health，只有在
`ok=true`、`delivery_mode=prepublished_gzip_v1` 和 `timing_contract=artifact_timing_v2`
同时满足时才允许 GPU qualification。运行退出时（包括失败和信号退出）执行
远端 stop，并保存 pre/post health、start 和 stop 回执。

这不是调整请求路径或算法的补丁：远程 artifact 仍按原 frozen transfer contract
计入请求等待；只有服务生命周期从人工前置条件变成可审计的 launcher 前置条件。

## 其余合同

模型、vLLM 0.30.0、D931 YAML、W2 trace、500-adapter subset、allocator marker、
remote artifact 根目录、固定生成合同、资源包络和 D648 指标均不变。D1094 的
部分请求和失败 movement 不复用。

## 验收

- 4,000/4,000 成功并完成 W2；
- remote transfer journal、阶段恒等式、物理 GPU allocation-union、联合
  TTFT/TPOT SLO、P95 TTFT、HOST allocator readback 和 `n_correct` 按现有规则审计；
- 远端服务停止回执必须为 inactive；
- 任何不完整运行仍只作为启动/基础设施审计证据，不进入性能分母。
