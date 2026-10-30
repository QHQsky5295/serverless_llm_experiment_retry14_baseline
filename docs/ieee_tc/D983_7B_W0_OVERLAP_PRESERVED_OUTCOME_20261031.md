# D983：7B W0 overlap-preserved PrimeLoRA 完整结果与瓶颈裁决

日期：2026-10-31  
身份：PrimeLoRA 开发诊断点；不是正式 G1/G2 封存点。  
执行键：`d983_20261031/7b_overlap_preserved_w0`  
代码提交：`5ca81a832880487fe838cea2676d10129916de07`  
正式运行标志：`FAASLORA_FORMAL_RUN=0`

## 1. 执行合同

本轮复用既有 Llama-2-7B、4,000 请求、500-adapter W0 trace 和 adapter
subset，没有生成新权重、负载或第二套工件池。W0 使用既有
`rotation=500`/bursty arrival trace；固定长度 greedy/native-token 合同、
confirmed-tier 路由、overlap-preserved activation preparation、TP=1 和
四张 RTX 3090 保持不变。

配置、输入和代码来源如下：

| 来源 | SHA/路径 |
|---|---|
| 配置 | `configs/ieee_tc/20261030_d955_7b_route_overlap_validation_100.yaml`，SHA-256 `6c12310b2a3078b9dff9b34ce86321a98a5b42458e4b2619455d372370090941` |
| trace | `.../llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json`，SHA-256 `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` |
| adapter subset | 同目录 `_adapter_subset.json`，SHA-256 `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016` |
| 结果 | `.../d983_20261031/7b_overlap_preserved_w0/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d983_7b_overlap_preserved_w0.json` |
| 物理摘要 | `.../launch.launch/physical_deployment/physical_resource_summary.json` |

## 2. 完成性和正确性

4,000/4,000 请求到达终态，4,000/4,000 `success=true`，失败数为零。固定
长度生成、原生 token 计数、prompt hash、native timing 和输出合同均完整。
这些字段证明请求执行合同完整，但不等价于 LoRA 数值语义正确性。

本轮没有独立的语义 oracle，因此：

- `n_correct` 保持 `null`；
- `eligible_correctness=false`；
- `formal_comparison_qualified=false`；
- 不能把 `GPU-s/correct-request`、G1 资源排名或跨系统结论从本轮推导出来。

这不是额外的性能门禁，而是当前执行合同实际没有测量语义正确性的事实。D981
的 100-request 数值见证不能替代本轮 4,000-request 的独立正确性计数。

## 3. 观测结果

| 指标 | D983 W0 |
|---|---:|
| 完成/计划 | 4000/4000 |
| 平均 TTFT（arrival→first token） | 1205.236 ms |
| P50 TTFT | 622.858 ms |
| P95 TTFT | 3435.241 ms |
| P99 TTFT | 13764.721 ms |
| 平均 service TTFT | 573.018 ms |
| P95 service TTFT | 1508.593 ms |
| 平均 TPOT | 33.596 ms |
| 平均 E2E | 5000.575 ms |
| P95 E2E | 10959.787 ms |
| TTFT-SLO（5,000 ms） | 95.750% |
| 平均 dispatch/admission wait | 632.218 ms |
| P95 dispatch/admission wait | 2069.416 ms |
| 平均 ingress queue wait | 599.426 ms |
| P95 ingress queue wait | 2017.816 ms |
| 平均 GPU-ready TTFT | 942.660 ms |
| P95 GPU-ready TTFT | 2034.990 ms |
| P95 scale-up-affected TTFT | 9745.667 ms |
| scale-up first-service 样本 | 26 |
| 平均 LoRA I/O | 95.063 ms |
| cache hit rate | 96.700% |
| GPU hit rate | 61.150% |
| scale-up / scale-down 事件 | 26 / 25 |
| 最大物理 GPU 数 | 4 |

请求级远端读取、传输、竞争和本地物化仍在 arrival→first-token 的主路径中；
发布前一次性压缩缓存、远端管理和无关维护没有混入请求计时。

## 4. 物理资源字段裁决

本轮结果 JSON 同时包含两套不同语义的 GPU-s 字段：

- `infra_gpu_seconds_total=14911.190036` 是旧的逻辑实例/部署生命周期投影；
- `physical_resource_summary.json` 的
  `physical_gpu_seconds_total=9720.1728154351` 是
  `physical_gpu_resource_summary_v1` 的 owner-union 观测值；
- `gpu_seconds_observed/4000=2.4300432039` 只能称为
  `GPU-s/offered-request`，不能称为 `GPU-s/correct-request`。

两者不应相减、平均或择优拼接。按照 D240、D944 和冻结指标协议，若讨论本轮
物理生命周期资源，只保留 owner-union 字段；旧投影另作为诊断字段保存，不用于
G1 排名。该差异是计量合同差异，不是本轮系统性能收益。

## 5. 与同工作点历史 PrimeLoRA 的客观比较

D624 是同一 7B W0 rotation=500 工作点的历史完整观测，D983 不与 W2 的
D969 拼接比较。D624 的联合达成率为 95.250%、平均 TTFT 1253.977 ms、
P95 TTFT 3090.315 ms、物理 GPU-s 9910.774533。相对 D624：

| 指标 | D983 相对 D624 |
|---|---:|
| 平均 TTFT | 下降约 3.89% |
| P95 TTFT | 增加约 11.16% |
| TTFT-SLO | 增加 0.50 个百分点 |
| owner-union GPU-s/offered-request | 2.4300，对应 D624 的 2.4777，下降约 1.92% |

这是一项单轮、已知工作负载上的描述性改善，不能表述为 G1/G2 已达标。它也
说明 overlap-preserved activation preparation 没有消除 W0 的尾部问题：平均资源
和平均 TTFT 有改善，但 P95 TTFT 仍由少数扩容/churn 请求主导。

## 6. 阶段证据与可证伪瓶颈假设

后端服务阶段的 P95 TTFT 为 1508.593 ms，而 P95 arrival→first-token 为
3435.241 ms；P95 ingress queue wait 为 2017.816 ms，P95 dispatch/admission
wait 为 2069.416 ms。与此同时，scale-up-affected 请求的 P95 TTFT 达到
9745.667 ms，scale-up first-service 样本的 GPU hit rate 只有 19.23%，而
全体 GPU hit rate 为 61.15%。这组证据把主要问题定位在扩容就绪期间的前端
排队、首次服务和 tier 准备包络，不能归因于 decode 内核本身。

控制事件还显示，在单个 arrival burst 中会连续提交多个自然扩容；部分提交记录
带有尚未就绪的 pending activation，单次激活的 engine-ready 时间约 31--44 秒。
`IEEEReplicaControl` 已显式计入 ready/pending 实例数，因此当前不能简单声称
“忘记了 pending 实例”。更谨慎、可证伪的假设是：**现有 max-score 控制虽然
知道 pending 数，但只在离散采样点根据当前队列和过去 45 秒 TTFT 反应，未将
实测 activation-ready 时间、近期到达率和已观测服务率组合成“就绪时刻的可用
容量”预测；因此 burst 到达时，扩容请求提交与真正可服务容量之间存在较长
activation debt。**

该假设尚未转化为代码修改。它可以被以下最小验证证伪：从 D983 的请求和
activation 事件中离线计算每个扩容提交到 ready 的时间、提交时的 queue/active、
就绪前到达量和就绪后实际可排空量；若 tail 请求并不落在 activation debt 区间，
则不应修改 autoscaler，而应转向 adapter 准备或本地准入阶段。

历史候选 D930 的 current-demand guard 实际抑制次数为零，D965 的就绪边界完整
重规划反而退化；因此本轮不重复这两类无效方向，也不通过调大并发、延长超时或
直接改变固定阈值来掩盖尾延迟。

## 7. 结论和下一步

D983 证明了 overlap-preserved PrimeLoRA 路径可以在该 W0 负载下完整完成 4,000
请求，并取得相对 D624 的平均 TTFT 和物理 owner-union GPU-s 描述性改善；它同时
暴露了扩容/adapter-churn 尾部的持续问题，且没有独立语义正确性计数。因此：

1. 原始 JSON、请求终态、watchdog、物理 allocation journals 和日志永久保留；
2. 不生成论文图，不把 D983 放入正式 baseline 或 G1/G2 表；
3. 不启动 baseline，也不推进 3B 主比较；
4. 下一步先做无 GPU 的 activation-debt 阶段审计，确认上述单一假设；
5. 只有当审计支持新假设时，才设计一个由观测 arrival/service/ready 数据驱动的
   控制候选，并用新 execution key 做一次完整回放；否则保持当前实现并转向
   仍未解释的具体阶段。

本轮进程已结束，GPU lease、worker 和临时缓存均已进入正常收尾；推理机磁盘
仍保持约 165 GiB，原有 150/100 GiB、内存和 OOM 保护规则不变。

