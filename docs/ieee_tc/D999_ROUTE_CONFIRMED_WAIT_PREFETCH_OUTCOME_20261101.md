# D999：route-confirmed wait preparation 候选的结果

日期：2026-11-01  
模型：Llama-3.2-3B-Instruct  
负载：既定 W2、4,000 请求、500 adapters、`fixed_length_greedy_v1`、真实远端预发布 gzip 交付  
候选配置：`configs/ieee_tc/20261101_d999_3b_route_confirmed_wait_prefetch_w1.yaml`

## 结论

D999 保留为完整、可审计的失败候选结果，但不作为 PrimeLoRA 的改进版本，也不进入主结果。候选机制在本次请求轨迹中 **0 次触发**：它被实现为“确认路由后、每副本运行槽预留失败时”才启动 NVMe 准备；实际路径中路由器会先过滤没有可行运行槽的副本，剩余副本通常在原子预留时直接成功。因此，D999 没有测量到候选机制的因果效果。

这不是把结果解释成“未显著”，而是机制未执行，故不能据此支持或否定该优化假设。冻结共同 SLO 审计同时显示 D999 退化，因而保留 D992 作为当前 3B retained reference，并否决 D999 作为候选版本。

## 冻结审计结果

| 指标 | D992 retained reference | D999 | D999 相对 D992 |
|---|---:|---:|---:|
| 正确完成 | 4,000/4,000 | 4,000/4,000 | — |
| 联合 TTFT/TPOT SLO | 3,678/4,000 (91.95%) | 3,594/4,000 (89.85%) | -84 requests |
| P95 TTFT (ms) | 2,197.8597 | 2,364.5892 | 更差 |
| 平均 TTFT (ms) | 737.904 | 801.924 | 更差 |
| 物理生命周期 GPU-s | 8,268.6399 | 8,436.2703 | 更高 |
| GPU-s/请求 | 2.067160 | 2.109068 | 更高 |
| 运行时扩容次数 | 5 | 5 | — |

D999 的 `full_replay=true`，但 `observed_latency_feasible=false`、`n_correct=null`，因此不满足 G1/G2 的排名准入条件。所有 4,000 请求的原生计时重算误差为 0；该结论不是由计量错误造成的。

## 机制触发审计

请求级 `gpu_reference_evidence` 统计：

| 证据 | 次数 |
|---|---:|
| `route_confirmed_file_preparation` | 0 |
| `remote_preparation` | 505 |
| `file_preparation_settlement` | 505 |
| `source_admission` | 4,000 |

因此 D999 的 505 次远端准备仍然发生在正常的 admission 后路径；并没有发生“在运行槽等待期间提前移动到 NVMe”。D999 与 D992 的远端准备次数基本相同（D992 为 504/505 级别），不能把 D999 的差异归因于候选机制。

## 为什么候选没有触发

当前 IEEE 路径的顺序是：

1. 读取确认的路由快照；
2. `Router.select_instance_with_decision` 只从 `feasible` 副本中选择；
3. 若没有可行副本，返回 `slot=None`；若有可行副本，原子运行槽预留通常直接成功；
4. D999 的候选调用位于“`slot is not None` 且原子预留失败”分支。

D992/D999 的请求证据中，存在 143/141 个带有历史路由尝试的请求，但没有一个已记录快照是“所有副本 `available_slots=0`”。这说明实际等待主要发生在确认快照重试、请求入口/运行时队列和 admission 后的文件准备，而不是候选所假设的原子槽预留失败窗口。

## 下一候选的边界

下一候选不复用 D999 执行键，也不把 D999 结果改写为有效消融。若继续优化，应只改变一个可证伪点：在**确认路由已经选出、且选中源 tier 为 Remote 之后、原子运行槽预留之前**启动一次有 owner 的 NVMe 文件准备；若后续确认源已经转为本地，则取消并完整回收；若 admission 成功，则 join 同一 movement 并由正常 source admission 再验证。该位置能覆盖 D992 中实际存在的 509 次 Remote 选中事件，同时仍不做 pre-routing 或未来请求预取。

下一候选必须使用新 execution key，并报告候选触发次数、取消次数、复用次数、远端传输次数和共同 SLO；没有触发证据时不得宣称优化生效。

