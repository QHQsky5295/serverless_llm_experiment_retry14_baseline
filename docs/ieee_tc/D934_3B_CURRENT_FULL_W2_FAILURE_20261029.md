# D934：3B 当前代码 W2 回放失败审计

日期：2026-10-29  
执行键：`d934_3b_current_full_w2`  
系统：PrimeLoRA（IEEE Full，Llama-3.2-3B，vLLM 0.30.0）

## 裁决

D934 不是性能运行，也不进入 3B 的 SLO、GPU-s、G1/G2 或基线排名。外部回放
在第一次 handoff 制品准备时终止；首个失败不是推理 worker 或 GPU 资源故障，
而是当时 3B 远端制品服务 `192.168.4.174:18080` 未运行，导致远端获取返回
`RemoteArtifactError`。随后当前代码按所有权协议保留失败 movement，并拒绝对
同一 owner 自动重试；因此日志末端的 owner-recovery 异常是第一失败的后果，而
不是根因。该失败证据保留，不能用 0 个完成请求构造低成本结果，也不能把它解释
成 3B 模型的服务性能。

## 已确认事实

- 启动合同、计划哈希、3B W2 trace、adapter map 和本地配置中的远端发布缓存身份
  核验通过；但运行开始前没有核验 3B HTTP 服务处于 active/listening 状态。
- 两个 dedicated worker 均完成模型加载、CUDA graph capture、KV cache 初始化，
  并分别写入 `ready.json`；模型初始化本身没有报 CUDA OOM。
- GPU 生命周期、服务 cgroup 和看门狗均正常收尾；`launch.json` 的 GPU release
  和 service path removal 均为真，服务 OOM/high/max/swap 事件未由本轮产生。
- 失败发生在 `execute_gpu → prepare_gpu_source → _queue_ieee_file_preparation`
  路径。最早的三个 NVMe movement 在 2795700.526--2795700.529 单调时刻失败，
  `main_outcome` 记录的类型是 `RemoteArtifactError`；稍后才由
  `PreloadingManager.submit` 拒绝同一 owner 的失败 movement：
  `failed movement requires explicit owner recovery, not an automatic retry`。
- `n_terminal`、`n_correct` 和共同 SLO 均不适用于本轮；没有有效性能人口。

## 失败链条

1. 回放开始后，3B dedicated runtime 正常启动并接收首批请求。
2. 运行时进入 IEEE residency/file-preparation epoch；远端→NVMe→GPU 的 source
   准备触发 movement queue。
3. 当时 3B HTTP listener 不存在；最早的多个 NVMe 获取立即返回
   `RemoteArtifactError`，movement 被标为 failed。
4. 当前实现按设计拒绝对失败 movement 自动重试，向上抛出显式 owner-recovery
   错误；外部回放收到失败并终止，没有使用内部定时器兜底。

因此，表面上的 “engine_dead” 是后果标签，不足以证明 vLLM engine 自身崩溃；
原始 worker 日志显示模型和 CUDA graph 初始化完成，最终退出由远端制品获取失败
触发的服务收尾产生。D934 也暴露了启动协议缺少“服务已监听”的运行前核验，
但不支持修改 owner 队列重试语义。

## 与历史 D886 的关系

D886 的完整 3B W2 观测为 3,750/4,000 联合通过、物理 GPU-s 11,936.650，
但仍未达到 95% 共同服务门槛。D934 使用 D886 的模型、trace、W2 adapter map、
profile 和资源配置，只使用新的输出/cache 路径及当前代码执行身份；因此 D934
不能覆盖或修正 D886，也不能证明当前代码比 D886 更差的服务性能。它只说明当前
代码在相同 3B 准备合同下暴露了一个需要先解决的 owner-lifecycle 失败。

## 下一步

1. 保留 D934 所有 raw worker、service、replay、watchdog 和 launch 回执；不重启
   同一执行键，不将失败结果混入任何汇总。
2. 在新的执行键前，启动已批准的 3B artifact service，核验 authenticated
   `/health`、500-entry manifest 和端口监听；该步骤只恢复远端服务，不改变
   推理机配置、缓存、trace 或带宽。
3. 不修改 owner 队列重试语义、不添加兜底重试、不放宽共同合同。远端服务健康
   后建立新的 D935 执行键；D934 不重跑、不覆盖。

## 原始证据

- `results/ieee_tc/p2_backend_qualification/d934_20261029/3b_current_full_w2/`
- worker failures：
  `results/worker_failures/faaslora_worker_297i5jgy_engine_dead_1793288496/`
  和
  `results/worker_failures/faaslora_worker_gl7hri6__engine_dead_1793288499/`
- 运行结果不具备 `n_correct`、联合 SLO 或 G1/G2 资格。
