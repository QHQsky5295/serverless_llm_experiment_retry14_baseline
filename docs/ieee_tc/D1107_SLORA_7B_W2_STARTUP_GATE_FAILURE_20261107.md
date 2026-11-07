# D1107：S-LoRA 7B W2 启动资格失败（不属于性能结果）

D1107 使用与 D1103/D1106 完全相同的 7B W2 trace、500-adapter view 和
`fixed_length_greedy_v1` 合同，目标是进入 S-LoRA--PrimeLoRA 同合同对照。它在
开始 replay 之前被资源安全门阻止，因此不产生任何性能结论。

## 已完成部分

- 共享 trace 和 adapter subset 身份检查通过。
- 500/500 adapter 通过 frozen content manifest 严格 materialize；manifest digest
  为 `684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050`，staging
  用时 101.371757 s。
- S-LoRA 官方环境 preflight 通过，未修改 S-LoRA 核心调度或 adapter 管理逻辑。
- replica 0 和 replica 1 分别通过 `/health`，启动耗时 169.219004 s 和
  143.331089 s；replica 2 开始加载但尚未 readiness。
- replay、summary 和任何请求记录均未生成；因此不存在部分性能样本混入问题。

## 资源门结论

D1107 使用了为 vLLM/ServerlessLLM 设定的 72 GiB `MemoryHigh`、80 GiB
`MemoryMax` scope。S-LoRA 的官方语义是在每个副本启动时把 500 adapters 常驻
主机侧；当第三副本加载期间，scope memory 达到约 77.3 GiB，`memory.events.high`
累计 454 次，尚未出现 `max`、`oom` 或 `oom_kill`。继续启动第四副本会违反本机
内存安全预算，因此在 replay 前停止该候选。

这不是 S-LoRA 性能退化，也不是通过改 S-LoRA 算法规避的问题；它说明 S-LoRA
需要独立于 vLLM 的宿主机内存 envelope。下一候选只调整 cgroup 安全上限以容纳
官方 resident-adapter 语义，保持模型、输入、adapter 集合、GPU 拓扑、生成合同
和 S-LoRA 源码不变。

## 收尾

停止过程中 transient scope 的 systemd cleanup 超时，随后使用 scope 级 SIGKILL
完成收尾；GPU 上没有残留 serving worker，D1107 未进入 replay。完整 runner、
server logs 和 staging cache 保留在
`/home/qhq/serverless_llm_baselines/results/paper_experiments/20_slora_w2_v1/d1107_20261107_slora_w2/`，
不得把 D1107 写入 baseline 性能表。
