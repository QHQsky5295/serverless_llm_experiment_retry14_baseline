# PrimeLoRA IEEE TC：HydraServe 对齐与真实远端边界审计

日期：2026-10-06  
性质：`METRIC_PROTOCOL_FINAL_V2.md` 的证据审计附录。本文不改变 V1、
`METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` 或 `METRIC_PROTOCOL_FINAL_V2.md`
中的公式、门槛、分母和主判据；后续正式运行必须同时保存本文件要求的证据。

## 1. 结论先行

PrimeLoRA 的主评价口径保持不变：

1. G1：在全部请求正确完成且联合 TTFT/TPOT SLO 达成率至少 95% 的可行点中，
   最小化物理整卡生命周期 GPU-s/request；
2. G2：在相同资源预算和联合 SLO 约束下，比较扩容与 adapter churn 工作块的
   绝对 P95 用户 TTFT，并检验联合达成率非劣；
3. CE、美元价格、GPU memory-time、goodput 和阶段分解是补充解释量，不能把
   不合格运行提升为“冠军”。

这不是为了预设 PrimeLoRA 必然领先。若没有达到正确性、联合 SLO 或释放证据，
结果只能标为诊断/不合格；不得以 CE、平均 TTFT 或较低的计费估计抵消这些失败。

## 2. HydraServe 文献核查

核查来源为 HydraServe 的 NSDI 2026 正式公开论文和官方仓库：

- [NSDI 2026 论文 PDF](https://www.usenix.org/system/files/nsdi26-lou.pdf)
- [USENIX 论文页面](https://www.usenix.org/conference/nsdi26/presentation/lou)
- [官方实现仓库](https://github.com/LLMServe/hydraserve)

原文的评价逻辑与 PrimeLoRA 有可比部分，但不能照搬为本机成本模型：

| HydraServe 的公开评价 | 本项目的对应物 | 关系 |
|---|---|---|
| cold-start TTFT 及分阶段启动延迟 | dispatch/readiness、service TTFT、E2E 阶段分解 | 可对齐展示，不能混合时钟或阶段定义 |
| warm 请求测量后用 5×TTFT、2×TPOT 构造服务阈值 | 共同 warm reference 与联合 SLO | 作为启发式参考，不称为行业标准；本项目必须按自己的后端、长度组和资源合同重新测量 |
| TTFT/TPOT SLO attainment | (A_T)、条件 TPOT 达成率、(A_{joint}) | 本项目保留联合值，避免两个边际率掩盖交集失败 |
| GPU memory-time 比例成本 | 可核验的 GPU memory-time 补充列 | 不等于本机物理整卡 hold；本机 G1 使用 GPU allocation 区间并集 |
| pipeline/worker 选择对 TTFT、TPOT 和成本的权衡 | Prime 的 residency、handoff、admission 对 TTFT、TPOT 和 GPU-s 的权衡 | 只能用共同工作负载的观测量归因 |

HydraServe 的测试床使用多 GPU 服务器、较高的服务器网络带宽和独立远端模型
存储；这些硬件条件不能被写成 PrimeLoRA 已经测过的集群能力。其论文报告了
冷启动、SLO 和资源/成本 trade-off，而不是 (1/(L C)) 形式的 CE 主指标。因此
“采用 HydraServe 的 SLO 评价维度”不等于“采用 HydraServe 的数值、硬件或
memory-time 计费假设”。

## 3. 冻结的指标层级

### 3.1 资格层

每个 offered 请求保留 `N_plan`、`N_arrived`、`N_submitted`、`N_terminal`、
`N_correct` 和 `N_good`。正确完成必须同时满足正常终态、adapter 身份、目标
输出数量、原生 token-ID 计数和完整的可靠时间线。失败、未到达、协议错误、
环境中止和缺字段不能静默删除。

### 3.2 服务层

用同一时间线计算用户指标：

\[
  TTFT_r=f_r-a_r,\qquad
  TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
  E2E_r=c_r-a_r.
\]

其中 $a$ 是计划到达，$f/l$ 是首/末 token，$c$ 是完成通知。`service TTFT`
或 `dispatch wait` 只能作为阶段列，不能替代用户 TTFT；重叠阶段不能在 E2E
之外再相加。

共同 warm reference 必须在同一最终后端、资源包络、长度分组、GPU-ready
adapter 和固定 prefix-cache 条件下测量。阈值采用

\[
  \theta^T_r=5T^{warm}_{\mu\kappa(r)},\qquad
  \theta^P_r=2P^{warm}_{\mu\kappa(r)},
\]

但这两个倍率只表示与 HydraServe 的对齐约定，不是普适标准。warm 标定未完成
前，任何运行都不能进入 G1/G2 排名。

### 3.3 资源层

主资源是可核验的物理 GPU 持有区间并集：

\[
  U=\sum_d\int a_d(t)\,dt,\qquad R=U/N_{correct}.
\]

GPU memory-time 只有在 allocated/reserved/used 的遥测语义和释放边界都明确时
才补报；它不能取代整卡不可共享时的生命周期占用。GPU-s、CPU/HOST/远端字节、
goodput、cost/token 和 CE 必须分列，不能互相冒充。

## 4. 真实远端的计量边界

### 4.1 计入请求用户时间线

请求触发的远端 `open/read/send`、实际线上字节、竞争与 back-pressure、本地
`receive/write`、必要的 gzip 解压和 materialize，以及由系统策略造成的 miss、
fetch、迁移、驱逐和重新获取，全部属于服务路径。它们对 TTFT/E2E、SLO 和 GPU
生命周期的影响必须保留。

不同系统在共同链路上可以产生不同的传输次数、字节数和等待时间；这不是不公平，
而是缓存/驻留机制的可检验结果。公平约束是相同的工件内容、请求 trace、生成
合同、部署窗口、资源上限和链路状态，而不是强行相同的 miss 数。

### 4.2 部署前、单独审计但不进入请求 E2E

用户已批准的“一次性只读 gzip 交付缓存”属于共同部署前准备：对既有 500 个
adapter 集合统一创建、完成后只读发布、记录 archive/content SHA，再通知实验
开始。请求路径不重新打包、不重新压缩、不扫描完整池，也不建立第二套解压工件池。

以下内容同样不进入请求 E2E，且不得事后从 E2E 中扣除：远端管理员操作、整池
hash 扫描、日志拷贝、失败收尾、清理，以及请求之外的压缩压力测试。扣除这些
时间会制造一个真实不存在的“净延迟”。相反，请求引发的读取、传输、接收、
解压和本地 materialize 不能被移出 E2E。

## 5. 100 Mbps 与 1 Gbps 的客观处理

### 5.1 观测事实

- D77 的证据记录远端 `eno1` 当时协商为 100 Mb/s/full，且直接对端只通告
  10/100；推理机 `eno1np0` 显示 1000 Mb/s/full。
- D78 重启后的证据记录两端均通告并协商 1000 Mb/s/full；2026-10-05 的只读
  复核仍为 1000/full，错误计数为零。
- 没有证据确认是哪一次交换机、端口、线缆、自动协商或重启状态造成了变化；
  Codex 没有强制速率、重置 NIC 或修改 qdisc。

因此，100 Mbps 不能被描述为远端 NIC 的物理上限，也不能被重标为 1 Gbps；
当前正式运行必须保存实际协商状态和路由端点，历史 D77 与 D78 不能合并为一个
未标注的工作点。

### 5.2 是否污染结论

这取决于运行用途，而不是取决于结果是否有利：

| 运行 | 处理 | 能否进入 G1/G2 |
|---|---|---|
| D75--D77 100 Mbps 功能/覆盖与取消检查 | 保留为历史交付诊断；不是推理性能 profile | 否 |
| 当前两端 1 Gbps/full、完整请求路径且合同冻结 | 作为实际远端工作点，传输开销照计 | 可以，仍须通过全部资格和 SLO 门槛 |
| 应用层限速或 `no-delay` | 明确标注为敏感性/乐观控制，不称物理网速 | 只能在对应敏感性问题中使用 |

如果将来正式运行确实在 100 Mbps 下完成，则它必须作为单独的 `link_state`
execution key 报告，不能和 1 Gbps 结果池化。100 Mbps 会放大远端 miss 的时间
代价；这可能改变绝对数值，但不构成“造假”，前提是所有系统共享该状态、请求
触发的传输均计入，并且论文只把结论限定在该环境。若目标是回答 1-Gbps 条件，
则不得把 D77 数据代替 D78 重跑。

### 5.3 最低链路证据

每次正式 run 保存：`local_interface`、`remote_interface`、`speed_mbps`、
`duplex`、`autoneg`、`link_detected`、错误计数、运行时段、remote boot ID、
实际 wire bytes、传输并发和 achieved throughput。应用层 `FAASLORA_STORAGE_
BANDWIDTH_MIB_S` 只能表示进程级限速，不可写成物理 NIC 速度。

## 6. 运行后必须生成的审计列

每个请求或传输至少保留：

```text
request_id, adapter_id, content_sha256, transfer_id,
delivery_mode, remote_read_ms, remote_send_ms,
client_receive_ms, client_write_ms, client_extract_ms,
wire_bytes, logical_bytes, cache_tier_before, cache_tier_after,
miss_or_hit, link_state_key, prompt_hash, target_tokens,
native_token_count, ttft_ms, tpot_ms, e2e_ms, correct, failure_class
```

跨主机时钟未校准时只报告各自时钟域的 span，不直接相减；跨阶段重叠由时间线
和事件 ID 解释，不能用多个 span 的简单求和替换 E2E。

## 7. 对当前阶段的执行决定

本附录不授权新的 GPU replay，也不把 D269 的 500-request diagnostic 当作性能
或 G1/G2 结果。继续执行顺序为：

1. 保持已发布只读 gzip cache 和当前 1-Gbps/full 状态，不重启、不重配远端；
2. 解决 D266 identity conflict 的精确 source witness；
3. 完成共同 warm reference、Resident budget 和 Prime 7B readiness；
4. 只有在 Prime 7B 通过新口径后，才推进 3B 和外部 baseline。

本文件的作用是防止后续把打包时间、历史 100 Mbps、CE 或 HydraServe 的
memory-time 口径混入主结论；它不为 PrimeLoRA 预先保证任何胜出结果。

