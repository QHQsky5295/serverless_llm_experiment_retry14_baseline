# D213：普通 vLLM Resident 参考路径语义审计

日期：2026-10-04  
状态：通过只读审计；尚未形成 Resident 性能参考

## 目的

V1 的 Resident 参考必须表示普通 vLLM 的静态多副本服务，而不能暗含
PrimeLoRA 的 confirmed-tier 路由、handoff、层级驻留、扩缩容或有效容量
准入控制。D211 的候选因仍经过 `faaslora_full`/`ieee_confirmed` 被拒绝；本
审计针对其后的独立 vLLM runner 和已生成的启动规格。

## 审计对象与证据

| 对象 | 证据 |
|---|---|
| runner | baseline `scripts/run_vllm_fair_experiment.sh`，SHA256 `af0f6f356140eedb2ab712d32f1ad0b3e857fb22bf88b0b5acd03beb91122e54` |
| replay client | `scripts/replay_openai_trace.py`，SHA256 `46537ecd5df55be115e1d240f286a0bf072396d5d317c9f10f0596128e75c270` |
| baseline checkpoint | `7400e9d9f9e86179c79dcbd7855a426134da43ec` |
| launch specification | `results/ieee_tc_resident_audit_dryrun/shared_inputs/d212_vllm_resident_audit_seed42_vllm_dp4_tp1_launch.yaml`，SHA256 `340d6b0f49f0ae7b88169e8a944bef1886e13ac01beec2f2c2406282a763eb15` |
| input | 既有 W0 trace/subset；不生成新权重、工件或负载 |

## 语义检查结果

1. 对 runner 和 replay 源文件进行不区分大小写检索，未发现
   `faaslora_full`、`ieee_confirmed`、Prime confirmed-tier、handoff、
   hierarchical residency、forwarding 或 Prime admission/scale-control
   路径。
2. 启动规格仅包含普通 vLLM 参数：4 个 DP 副本、TP=1、每副本 1 张 GPU、
   `enable_lora`、`max_loras=4`、`max_cpu_loras=16`、固定 500-adapter
   静态注册，以及 vLLM 自己的 batch/KV/cache 选项。规格中的
   `dynamic_lora_*` 是 vLLM runner 的普通配置字段；静态注册时不启用
   Prime 调度控制，也不改变副本选择。
3. replay 的 round-robin URL 选择只负责把同一 trace 分发到四个普通
   HTTP endpoint；它不读取 tier registry、queue/affinity 状态或 admission
   决策。固定输出合同由 D212 记录，token source 为 vLLM 原生
   `token_ids`。
4. 该资格路径使用本地 frozen adapter pool；没有远端 artifact endpoint，
   因而不会把 PrimeLoRA 的远端准备路径引入 Resident 参考。远端传输仅在
   PrimeLoRA/Serverless 共同交付实验中按冻结协议处理。

## 结论与限制

只读审计通过，普通 vLLM 路径可以作为后续 Resident 参考的候选实现。该
结论不等同于数值资格：仍必须在同一 trace、固定输出合同和共同资源约束下
完成 4,000/4,000 正确请求，验证 warm SLO、physical GPU-s/request 和
V1 §7 的三次独立完整运行。Resident 的参考占用和 G2 budget 只能在这些
完整运行完成后冻结；D212 的 100 请求短回放不能替代它们。
