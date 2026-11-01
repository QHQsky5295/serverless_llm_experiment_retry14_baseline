# D1007：3B 最小两实例与预测性释放候选结果

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化（未进入 baseline 比较）  
配置身份：`d1007_3b_min2_forecast_release_w2`  
运行身份：`d1007_full_20261102`  
工作负载：Llama-3.2-3B、W2、4,000 请求、500-adapter universe、seed 42 共享 trace  
状态：完整运行、冻结审计通过；开发候选，未取得排名资格

## 1. 可证伪假设

D1006 表明永久禁止预测性释放会显著增加 ready-but-idle GPU 时间；D632--D636 的 7B 结果表明在扩缩容场景中保留至少两个实例可能改善联合服务质量。因此 D1007 只改变 3B 的 `min_instances` 为 2，保留已有预测性释放逻辑，并保持 `max_instances=3`、路由、后端、trace、W2 和其他控制参数不变。预期是避免 3B 在低谷阶段回落到单实例，同时仍允许 3→2 释放。

## 2. 完整性与协议检查

- 计划请求数：4,000；native terminal：4,000；成功：4,000；失败：0。
- `launch.json` 的 `pass=true`；物理部署 6 个 allocation journal 均已释放，`open_lease_ids=[]`。
- 物理计量完整：`physical_gpu_seconds_observed=8406.161231717095`。
- 共享 trace、adapter map、W2、generation contract 和 common-SLO SHA 与候选配置一致。
- 请求级 TTFT/TPOT/E2E 重算误差均为 0 ms。
- 数值 adapter correctness 尚未完成认证，因此 `ranking_eligible=false`；本结果不能作为最终性能排名或论文主结果。

审计文件：

`paper_results/ieee_tc/p2_backend/d1007_3b_min2_forecast_release_w2_full/full4000.json`

原始运行目录：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1007_full_20261102/d1007_3b_min2_forecast_release_w2_full`

## 3. 冻结口径结果

| 指标 | D1007 | D1000 当前最佳 | D1006 禁止预测释放 |
|---|---:|---:|---:|
| 联合 TTFT/TPOT SLO | 3720/4000 = **93.00%** | 3690/4000 = 92.25% | 3668/4000 = 91.70% |
| TTFT 达标率 | 95.825% | 95.675% | 94.675% |
| TPOT 达标率 | 96.775% | 96.15% | 96.45% |
| 平均 TTFT | 742.91 ms | 751.90 ms | 796.72 ms |
| P95 TTFT | 2058.61 ms | 2112.67 ms | 2196.07 ms |
| 平均 TPOT | 16.94 ms | 17.36 ms | 16.61 ms |
| P95 TPOT | 30.27 ms | 31.57 ms | 31.31 ms |
| P95 E2E | 5444.80 ms | — | 5486.0 ms |
| 物理 GPU-s/request | **2.101540** | **2.002078** | 2.984105 |
| 物理 GPU-s 总量 | 8406.161 | 8008.312 | 11936.420 |

相对 D1000，D1007 将联合达标率提高 0.75 个百分点，P95 TTFT 降低约 54.06 ms，但物理 GPU-s/request 增加约 4.97%。因此它改善了服务质量方向，却没有满足 G1 的“共同服务要求下进一步减少生命周期物理 GPU-s/request”。

## 4. 机制观测

- 4 次 scale-up、4 次 forecast-based scale-down；实例数量始终不低于 2，未出现 2→1 的释放。
- scale-up 发生在请求序号 1、633、997、3017；新增实例准备时间分别约为 37.14、36.35、36.66、39.04 s。
- scale-down 均以 `ieee_forecast_capacity_release` 触发，说明最小实例约束确实生效，同时预测释放仍在工作。
- `physical_gpu_seconds` 的增加来自额外的长期 ready 生命周期，而非请求失败或遗留 lease；D1006 的永久保留进一步证明“完全禁止释放”会造成更大资源代价。
- D1007 的 93% 联合达标率仍低于冻结的 95% 服务要求，剩余违约分布在多个 500-request phase，并非单一启动点即可解释。

## 5. 裁决

**拒绝作为 G1 最终候选，保留为 G2/机制分析的开发观测点。**

理由不是单看 TPOT 或旧 CE，而是同时检查共同联合 SLO、物理生命周期 GPU-s/request、完整性和数值正确性。D1007 相比 D1000 的质量改善不足以抵消资源代价，且尚未通过 adapter 数值正确性认证；因此不能据此提前结束 3B PrimeLoRA 优化，也不能开始 baseline 比较。

## 6. 后续最小实验方向

下一候选不能继续简单提高 `min_instances` 或永久禁用释放。应先利用 D1007 的事件与 phase 结果，检验“仅在可观测的 scale-up hazard/高排队阶段临时保持第二实例，低负载阶段按预测容量释放”的控制条件；配置、触发条件和验收指标必须在新 run-key 中预先登记。若没有新的可证伪条件，不重复 D1007 或 D1006。

