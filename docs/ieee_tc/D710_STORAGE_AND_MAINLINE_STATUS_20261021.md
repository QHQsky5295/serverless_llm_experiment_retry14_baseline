# D710：存储回收与 PrimeLoRA 主线状态

日期：2026-10-21  
分支：`retry14_continuous_queue_v2`  
基线提交：`efeca418167af61f8d825323ac3e7b4471df6d02`  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 本检查点的作用

本记录只保存存储保护和实验主线选择的证据，不改变既有结果、评价协议、用户文件、
模型或 LoRA 工件。推理机的 150 GiB 磁盘门槛、内存/OOM 门槛和单重型任务约束继续
适用。远端工件来源审计是内部资格证据；它不改变当前实验顺序，也不作为论文新增
结论。

## D131 旧结果的可逆归档

D131 的大型 JSON 是可由归档恢复的旧开发结果。删除原始副本前先完成 gzip 归档、
归档完整性检查和 round-trip SHA 校验；归档仍位于原结果目录，未删除任何模型、
LoRA、trace、配置或唯一实验记录。

| 项目 | 值 |
|---|---|
| 原始路径 | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d131_20261001/7b_outputs_full1/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d131_7b_full_w0_full1.json` |
| 原始字节数 | 13,292,883,550 |
| 原始 SHA-256 | `5e235216fa1364f4c894d7ec53132db49039619883c10bb8e621c141b77fc69a` |
| 归档路径 | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d131_20261001/7b_outputs_full1/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d131_7b_full_w0_full1.json.gz` |
| 归档字节数 | 304,767,413 |
| 归档 SHA-256 | `9954472a73c39cd892d737c792a0bf5e83b5872def50a54cfde66093bb28ae96` |
| round-trip SHA-256 | `5e235216fa1364f4c894d7ec53132db49039619883c10bb8e621c141b77fc69a` |
| 原始副本状态 | 已移除；归档可恢复 |
| 请求路径影响 | 无；归档和清理均在测量窗口之外 |

最新复核的推理机可用空间为 `175,966,367,744` 字节。下一次启动前仍须重新检查
`df -B1`、内存增长门槛、GPU 空闲状态和父级 cgroup 容量；本记录的数值不是未来
运行的预留保证。

## 为什么当前先做 7B

这是执行顺序的选择，不是由远端工件数值审计触发的暂停。7B 拥有当前代码路径下
的成熟开发参照（D379）和已记录的容量/队列反例（D386、D343、D624 等），能够先
针对请求级排队与生命周期账本提出可证伪的容量假设。3B 的 D703 已完成一次完整
服务回放，D705/D706 又定位了其控制器阈值与 scale-down 参考之间的具体错配；因此
3B 不是被放弃，而是按计划排在 7B 当前容量假设之后。7B 与 3B 都尚未取得独立
数值正确性和最终 G1/G2 资格，不能把任一模型写成已经达标。

## 下一项 7B 候选：容量包络假设

D379 的请求级离线分析显示，按当前共同参考阈值有 223 条联合时序失败；其中 124
条发生时没有有效 scale-up，失败请求的平均 ingress/dispatch 等待远高于 service
TTFT。D386 将 `queue_upper` 从 8 降到 4 后，平均和 P95 用户 TTFT、dispatch wait
以及物理 GPU-s 均上升，因此不采用更激进的队列触发。D343 的 C14 原生容量探针又
说明增加并发槽可以提高重叠，但会付出 TPOT 代价。由这些观察得到一个单一、可证伪
的候选：在 D379 的 profile、路由、准备、admission、trace、工件集合和生成合同
均保持不变时，使用更大的原生容量包络（`max_num_seqs`、`max_loras` 和
`runtime_concurrency_cap` 同步变化）是否能降低 ingress/dispatch 排队，并在
联合 SLO 与物理 GPU-s 之间形成更好的 Pareto 点。

候选不会通过提高超时、删掉失败请求、改变联合 SLO、替换 trace、按结果筛选执行
或复用旧 profile 来“修复”结果。首先进行一次小规模原生资格/资源安全检查；只有
该检查证明容量、显存、合同和释放路径成立，才登记新的完整 7B 执行键。完整执行
仍只运行一次，保留失败和退化证据，暂不生成展示图。

## 当前阶段边界

已完成：D708/D709 远端 3B/7B 来源审计、D131 可逆归档、D379/D386 7B 请求级瓶颈
复核，以及推理机磁盘门槛复核。远端节点没有被修改，服务路径没有被重启，D131
原始结果没有被改写。

正在进行：把上述容量假设整理成新的、独立结果身份，并在启动前完成资源与旧结果保护
检查。

尚未完成：新的 7B 容量资格检查和完整回放、3B 下一候选、ServerlessLLM/vLLM/
S-LoRA/Loquetier/HydraServe 的主比较、核心消融、G2 场景、成本与 CE 补充分析、
最终图表和论文改写。未完成项不会由本检查点的归档或来源审计隐式关闭。

