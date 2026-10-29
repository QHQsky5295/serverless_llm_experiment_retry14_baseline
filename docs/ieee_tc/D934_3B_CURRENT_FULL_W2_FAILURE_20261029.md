# D934：3B 当前代码 W2 回放失败审计

日期：2026-10-29  
执行键：`d934_3b_current_full_w2`  
系统：PrimeLoRA（IEEE Full，Llama-3.2-3B，vLLM 0.30.0）

## 裁决

D934 不是性能运行，也不进入 3B 的 SLO、GPU-s、G1/G2 或基线排名。外部回放
在早期请求阶段被显式终止，原因是原生 worker 生命周期与文件准备事务的 owner
状态失配；系统没有执行隐藏重试或 fallback。该失败证据保留，不能用 0 个完成
请求构造低成本结果，也不能把它解释成 3B 模型的服务性能。

## 已确认事实

- 启动合同、计划哈希、3B W2 trace、adapter map 和远端发布缓存核验通过。
- 两个 dedicated worker 均完成模型加载、CUDA graph capture、KV cache 初始化，
  并分别写入 `ready.json`；模型初始化本身没有报 CUDA OOM。
- GPU 生命周期、服务 cgroup 和看门狗均正常收尾；`launch.json` 的 GPU release
  和 service path removal 均为真，服务 OOM/high/max/swap 事件未由本轮产生。
- 失败发生在 `execute_gpu → prepare_gpu_source → _queue_ieee_file_preparation`
  路径，`PreloadingManager.submit` 拒绝了一个已经失败的 movement：
  `failed movement requires explicit owner recovery, not an automatic retry`。
- `n_terminal`、`n_correct` 和共同 SLO 均不适用于本轮；没有有效性能人口。

## 失败链条

1. 回放开始后，3B dedicated runtime 正常启动并接收首批请求。
2. 运行时进入 IEEE residency/file-preparation epoch；远端→NVMe→GPU 的 source
   准备触发 movement queue。
3. worker EngineCore 已完成 ready，但 owner 事务在后续 movement 中被标记失败。
4. 当前实现按设计拒绝对失败 movement 自动重试，向上抛出显式 owner-recovery
   错误；外部回放收到失败并终止，没有使用内部定时器兜底。

因此，表面上的 “engine_dead” 是后果标签，不足以证明 vLLM engine 自身崩溃；
原始 worker 日志显示模型和 CUDA graph 初始化完成，最终退出由服务失败收尾触发。

## 与历史 D886 的关系

D886 的完整 3B W2 观测为 3,750/4,000 联合通过、物理 GPU-s 11,936.650，
但仍未达到 95% 共同服务门槛。D934 使用 D886 的模型、trace、W2 adapter map、
profile 和资源配置，只使用新的输出/cache 路径及当前代码执行身份；因此 D934
不能覆盖或修正 D886，也不能证明当前代码比 D886 更差的服务性能。它只说明当前
代码在相同 3B 准备合同下暴露了一个需要先解决的 owner-lifecycle 失败。

## 下一步

1. 保留 D934 所有 raw worker、service、replay、watchdog 和 launch 回执；不重启
   同一执行键，不将失败结果混入任何汇总。
2. 只读比对 D886/D883 的成功 movement owner、worker shutdown 顺序、source
   lease/settlement 回执与当前 `PreloadingManager.submit` 失败状态，定位是
   owner 注册、失败状态清理还是当前 worker retirement 顺序差异。
3. 在没有具体因果证据前，不修改重试语义、不添加兜底重试、不放宽共同合同，
   也不重新运行 3B 4000 请求。
4. 找到可验证的生命周期修复后，才建立新的 D935 执行键；若无法证明修复，
   3B PrimeLoRA 保持“历史结果可用但当前候选未通过”的状态。

## 原始证据

- `results/ieee_tc/p2_backend_qualification/d934_20261029/3b_current_full_w2/`
- worker failures：
  `results/worker_failures/faaslora_worker_297i5jgy_engine_dead_1793288496/`
  和
  `results/worker_failures/faaslora_worker_gl7hri6__engine_dead_1793288499/`
- 运行结果不具备 `n_correct`、联合 SLO 或 G1/G2 资格。
