# D1163：A4 成本效率补充与计费敏感性

日期：2026-11-09。该产物只做离线重算，不启动 GPU、不修改运行配置、不覆盖
`final_v2/`、旧 `figs/paper/` 或历史原始结果。它回应 A4/CE 质疑，但不改变
冻结的 G1/G2 主判据，也不把 CE 重新提升为系统优越性的唯一判定标准。

## 1. 口径

主指标仍为：

* G1：在联合服务要求 `A >= 0.95` 下，生命周期物理 GPU 持有时间 `U/N` 越低越好；
* G2：在共同预算 `U <= B` 下，比较 P95 TTFT，并同时报告实际联合达成率；
* 联合 SLO 达成率、P95 TTFT、完成/失败和真实 GPU 持有区间不能由 CE 抵消或隐藏。

CE 仅作为成本—延迟的补充描述。沿用历史论文口径：

\[
  C = \frac{\text{monetary lifecycle cost}}{N},\qquad
  CE = \frac{1}{\bar L_{E2E}(s)\,C},
\]

其中 `L` 使用平均端到端延迟（秒），`C` 使用每请求美元成本。广义敏感性为

\[
  CE_{\alpha,\beta}=\frac{1}{\bar L_{E2E}^{\alpha}C^{\beta}},
  \quad \alpha,\beta\in\{0.5,1,2\}.
\]

这两个表达式不替代 G1/G2；它们允许成本和延迟互相补偿，因此不能单独证明用户
体验、尾延迟或 SLO 改善。

## 2. 当前 7B PrimeLoRA 封存点的重算

本轮使用已封存 D1115（W1 Full）、D1117（W1 cap3）和 D1118（W2 cap3）。
三者均是已有完整 4,000 请求结果；没有重跑。当前实验记录的价格率为
`0.0008 USD/GPU-s`，调用费为 0。

原始字段中的 `monetary_idle_charge_gpu_seconds` 是已经乘过历史 idle factor 的
“等价 GPU-s”，不是原始 ready-idle 区间。因此重算先使用同一记录中的
`infra_idle_ready_gpu_seconds`，再施加新的 `f`：

\[
 C(f)=\frac{0.0008\,[G_{active}+fG_{idle-ready}]}{N}.
\]

这使得 `f=0.238095` 精确回到运行记录中的货币成本（允许最后小数位舍入），而
不是把已折扣字段再次折扣。

| 运行 | 负载 | `f=0` CE | `f=0.238095` CE | `f=0.5` CE | `f=0.75` CE | `f=1` CE |
|---|---|---:|---:|---:|---:|---:|
| D1115 | W1 Full | 112.699 | 108.028 | 103.319 | 99.191 | 95.380 |
| D1117 | W1 cap3 | 154.281 | 146.702 | 139.182 | 132.689 | 126.775 |
| D1118 | W2 cap3 | 95.197 | 92.464 | 89.634 | 87.089 | 84.685 |

在当前 PrimeLoRA 观测点内，从 `f=0.238095` 增至 `f=1`，CE 分别下降约
11.7%、13.6% 和 8.4%；从 `f=0.238095` 降至免费 idle-ready（`f=0`），CE
分别上升约 4.3%、5.2% 和 3.0%。因此 0.238095 会影响绝对值，但不是一个可以
任意制造数量级优势的开关。

完整逐因子数据见
[`primelora_current_idle_factor.csv`](../../paper_results/ieee_tc/a4_ce_supplement/d1163_20261109/primelora_current_idle_factor.csv)，
广义指数结果见
[`primelora_current_generalized_ce.csv`](../../paper_results/ieee_tc/a4_ce_supplement/d1163_20261109/primelora_current_generalized_ce.csv)。

## 3. 与外部系统的边界

已有的匹配 7B seed-42 主比较 CE 包位于
`paper_results/eurosys27_v2/a4_ce/legacy_7b_seed42/`，包括：

* 原始 latency、cost、CE、token cost 和 goodput；
* `f in {0, 0.238095, 0.5, 0.75, 1}` 的 serverless idle-factor 敏感性；
* `CE_(alpha,beta)` 的 3×3 组合；
* `Delta ln(CE)` 的延迟/成本贡献分解；
* PrimeLoRA 相对匹配基线的 break-even factor。

该包仍然有效，但它属于早期匹配的 seed-42 主比较，不能与 D1115/D1117/D1118
拼成同一轮的新主表。D1115/D1117/D1118 的当前 baseline 对照审计只封存了物理
GPU 持有时间和延迟，没有逐系统的 active/ready-idle 货币分解；因此本轮不从
`U` 反推 baseline 的货币成本，也不把不匹配的旧成本字段嫁接到当前工作点。

这保留了两个层次的可复核证据：匹配旧轮用于回答“计费假设是否改变 CE 排名”，
当前封存点用于回答“新的 PrimeLoRA 工作点对 idle factor 有多敏感”。在缺少共同
成本分解的当前 baseline 行上，结论保持未知，而不是用估算填空。

## 4. 对 A4 质疑的裁决

1. “CE 完全没有先例”不成立；已有材料中的 ServerlessLoRA 指标核查和旧匹配包
   说明成本—延迟乘积有同类先例。
2. “CE 足以证明 PrimeLoRA 全面优于其他系统”也不成立。CE 可以奖励成本下降并
   容忍延迟上升，尤其不能替代 P95 TTFT、联合 SLO 或生命周期 GPU-s。
3. 当前论文写作应把 CE 放在成本敏感性/补充分析中；主结论使用 G1/G2 和原始质量
   指标。若正文保留 CE，应同时展示 `f` 敏感性和 `Delta ln(CE)` 分解，明确哪些
   系统的优势来自成本、哪些来自延迟。
4. 本轮没有发现需要为了 CE 单独改变 PrimeLoRA 配置的证据；继续优化应由 G1/G2
   或机制实验中的可证伪瓶颈驱动，而不是针对 CE 排名调参。

## 5. 产物和复用

本产物的机器可读清单为
[`manifest.json`](../../paper_results/ieee_tc/a4_ce_supplement/d1163_20261109/manifest.json)。
旧 A4 文件保持原位且未被覆盖；本轮没有生成 PDF/图，待主要比较和 3B 证据接近
封存后再按统一学术绘图规范出图。

