# D937：3B W2 内容驱动 NVMe 容量候选结果

日期：2026-10-30  
执行键：`d937_20261030/3b_full_w2_nvme_extent`  
协议提交：`5f0d4378`  
计划 SHA-256：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 1. 运行身份与完整性

本轮沿用 D935 的 Llama-3.2-3B、500 个 LoRA 工件、4,000 请求 W2 trace、adapter map、固定长度贪心生成契约和远端 `prepublished_gzip_v1` 交付语义。唯一改变是把 NVMe 容量从 16,384 MiB 提高到 24,576 MiB；GPU 数量、TP、路由、准入、HOST 16 GiB 上限、传输限制和请求顺序均未改变。容量候选由冻结内容索引的 18.156 GiB 分配字节乘以 1.25 后向上取整得到，不是根据本轮结果反向选择的。

| 检查项 | 结果 |
|---|---:|
| 计划请求 / 终止请求 | 4,000 / 4,000 |
| 成功请求 / 错误请求 | 4,000 / 0 |
| `output_contract_match` | 4,000 / 4,000 |
| 实际 token 数等于目标 token 数 | 4,000 / 4,000 |
| TPOT 观测完整 | 4,000 / 4,000 |
| generation contract | `fixed_length_greedy_v1` |
| token 计数来源 | vLLM `token_ids` |
| GPU context release / service path removed | true / true |
| GPU OOM / watchdog / replay 失败 | 0 / 0 / 0 |

本轮结果仍不能证明独立的数值正确性：物理部署摘要的 `n_correct=null`，资格字段为
`not_inferred_from_token_contract`。因此下文的“成功”仅表示共同请求、输出长度和时序契约完成，不将其写成答案正确率。

## 2. 主要结果

联合 TTFT-SLO 阈值为 5,000 ms。

| 指标 | D935（16 GiB） | D937（24 GiB） | 变化 |
|---|---:|---:|---:|
| TTFT-SLO 达成 | 3,995/4,000 = 99.875% | 3,999/4,000 = 99.975% | +0.100 个百分点 |
| 平均 TTFT | 803.767 ms | 775.708 ms | −3.49% |
| P95 TTFT | 2,302.041 ms | 2,204.690 ms | −4.23% |
| P99 TTFT | 3,504.860 ms | 3,311.315 ms | −5.52% |
| 平均 service TTFT | 483.394 ms | 467.577 ms | −3.27% |
| 平均 TPOT | 16.537 ms | 16.341 ms | −1.19% |
| 平均 E2E | 2,635.175 ms | 2,604.454 ms | −1.17% |
| P95 E2E | 5,449.011 ms | 5,406.884 ms | −0.77% |
| 物理生命周期 GPU 时间 | 11,936.550 GPU-s | 11,941.411 GPU-s | +0.041% |
| 物理 GPU 时间 / offered request | 2.984138 | 2.985353 | +0.041% |

容量扩展使 dispatch 前 `remote-cold` 请求从 511 降为 501（12.775% 降为
12.525%），远端工件传输次数从 505 降为 497。缓存层级计数从
`gpu/host/nvme/remote = 1772/1085/632/511` 变为
`1797/1058/644/501`。这说明候选容量确实减少了重复远端获取，并伴随可重复观测的
TTFT 尾部改善；但改善幅度有限，不能把 NVMe 容量描述为解决全部尾延迟问题。

## 3. 物理资源与安全边界

物理资源摘要记录 3 个 GPU allocation journal 全部释放，`measurement_complete=true`，
`n_terminal=4000`、`n_native_contract_complete=4000`、`open_lease_ids=[]`。运行期间未触发
OOM、watchdog、交换区或磁盘停止条件；运行后 NVMe 候选目录已清理，远端 3B 工件服务也已
停止并确认 18080 端口关闭。运行开始前的可用磁盘空间为 167,758,385,152 B，超过计划的
160 GiB 启动下限；结束后保留的根文件系统空间仍约 155 GiB。内存判定使用
`MemAvailable`，而不是把 Linux `MemFree` 单独当作压力结论。

需要区分两个 GPU 时间字段：运行器中的 `infra_gpu_seconds_total=4,317.335` 是部署账单/生命周期
字段，而本节候选验收使用 physical owner union 摘要中的 `11,941.411 GPU-s`。后者是本计划
规定的物理生命周期字段；两者不能混为同一成本口径。

## 4. 结论

D937 支持以下有限结论：在相同 3B W2 workload 和相同 GPU/路由/准入设置下，把 NVMe 容量
按冻结内容集合扩展到 24 GiB，可以小幅减少 remote-cold 和远端传输，并把 P95 TTFT 从
2,302.041 ms 降到 2,204.690 ms，同时联合 TTFT-SLO 提高到 99.975%；物理 GPU 时间基本
不变（增加约 0.041%）。因此该容量候选可作为 3B PrimeLoRA 当前 Full 的候选配置进入后续
对照，但它不是单独足以证明全局最优的优化，也没有证明 PrimeLoRA 已经在 P95 TTFT 上胜过
所有基线。

本轮不重复执行同一执行键、不筛选请求、不改变共同评价口径。下一步应在不改变生成契约的
前提下，先把该 3B 候选与统一口径的基线重接并检查物理 GPU 时间和 TTFT-SLO；若比较显示
尾延迟瓶颈仍主要来自 scale-out 首服务或 dispatch/admission，而非 remote-cold，则后续优化
应针对这些可观测阶段，而不是继续盲目增大 NVMe。

## 5. 证据索引

- 原始结果：
  `results/ieee_tc/p2_backend_qualification/d937_20261030/3b_full_w2_nvme_extent/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d937_3b_full_w2_nvme_extent.json`
- 物理资源摘要：
  `results/ieee_tc/p2_backend_qualification/d937_20261030/3b_full_w2_nvme_extent/launch.launch/physical_deployment/summary.json`
- 启动记录：
  `results/ieee_tc/p2_backend_qualification/d937_20261030/3b_full_w2_nvme_extent/launch.json`
- 执行脚本：`scripts/ieee_tc/run_d937_3b_full_w2_nvme_extent.sh`

