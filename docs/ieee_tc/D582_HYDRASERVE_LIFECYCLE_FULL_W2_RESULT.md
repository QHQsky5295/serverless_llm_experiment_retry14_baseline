# D582：HydraServe 完整 W2 观测与输入核对

2026-10-16。仅一次既有负载回放；没有重跑推理、修改原生策略或按结果选点。
沿用 D573/D574 已资格化的生命周期适配及 D576/D579 原生配置。

## 完整结果

| 指标 | 本次观测 |
|---|---:|
| 请求／原生合同完整请求／失败 | 4000／4000／0 |
| 实际 adapter 数／原生输出 token | 497／458224 |
| 联合 SLO | 2640/4000，66.000% |
| TTFT 达成率／TPOT 条件达成率 | 87.750%／72.400%（分母4000） |
| 生命周期物理 GPU-s | 9909.915261428105 |
| GPU-s/原生合同完整请求 | 2.477478815357026 |
| 平均／P95／P99 TTFT | 1980.788／11435.129／18186.318 ms |
| 平均／P95 TPOT | 71.785／118.107 ms |
| 平均／P95 E2E | 9433.580／19885.308 ms |
| 真实远端获取／线上字节 | 497／418365205 B |

联合达成低于冻结95%档位，因此不进入达标资源排名；结果完整保留。
4000条原生合同检查覆盖输入、adapter路径/索引/worker、正常终态、实际token
数量、原生时钟与资源区间，不等同于独立数值正确性证明。后者仍为unknown，
原始n_correct仍为null；formal_run=false、independent_holdout=false不追改。

平均E2E五段依次为提交4.038、分配等待1113.651、分配至首token863.098、
decode7311.700、完成通知141.093 ms。artifact resolve平均92.792 ms属于
嵌套span，不再叠加。仅此分解不能把尾延迟因果归于网络或某个原生机制。

## 同一 W2 输入下的现有四系统观测

复用D557输入核对方法，将D582逐请求与vLLM对接，再复用其已经SHA验证的
PrimeLoRA/ServerlessLLM对应证据。4000条request、adapter、target、prompt
hash及token数、原生输入token hash、source item及工件内容/manifest hash
均一致；共同W2 plan及到达索引一致。不是跨后端输出逐token相同的证明。

| 系统 | 联合 SLO | 物理 GPU-s | P95 TTFT（ms） |
|---|---:|---:|---:|
| PrimeLoRA（D554） | 89.550% | 14641.258982 | 5827.475134 |
| ServerlessLLM（D555） | 93.900% | 15893.645978 | 9516.719258 |
| vLLM（D557） | 98.225% | 16115.559125 | 1751.295472 |
| HydraServe（D582） | 66.000% | 9909.915261 | 11435.128737 |

PrimeLoRA相对HydraServe：联合SLO高23.55个百分点，P95 TTFT低49.039%，
但GPU时间高47.744%；两者均未达到95%。此表中仅vLLM跨过该服务档位。
这不是完整baseline表：S-LoRA、Loquetier的W2缺项仍须完成。共同预算参考
在该视图中尚未冻结，不据此宣布G2完成、跨场景最优或统计显著。
后端版本、配置及初始化/编译历史不同；输入一致不等于完整初态一致，也不是
隔离单一算法的因果对照。保留单轮、已知负载的开发观测身份。

## 机制、交付与资源范围

- 实际32个原生worker均pp1/batch8；32次分配和32次释放逐项绑定，按物理GPU
  区间并集独立积分。最后调度器快照的3个实例不是全程worker总数。
- 仍是单物理推理机四张RTX3090、独立rootless Podman；不接入其他项目k3s，
  不冒称物理双机。没有据此验证pipeline/consolidation收益。
- 启动核查实际链路1000Mb/s/full。既有远端只读gzip发布复用；请求中打包0，
  真实获取、竞争和本地物化保留，未事后扣除延迟。
- 本次服务峰值61766651904 B，high/max/OOM/kill事件和swap均0。实际GPU及
  本地服务/辅助组已释放，远端本轮服务已停止；远端日志SHA核对通过。
- 主审计分析峰值379674624 B、内存事件及swap0，并已实际关闭。该次分析
  使用systemd AllowedCPUs，但没有独立记录进程affinity验证；不将其配置值
  冒称已核实的affinity。它发生在推理与远端测量结束后，不影响性能样本。
- 跨系统离线核对首次在明确affinity断言处停止，未产生比较结果；保留源文件、
  identity、失败及关闭回执。第二次增加显式taskset，保持检查规则/内存包络
  不变，使用独立分析身份并通过。没有重复推理或改变性能结果。

## 来源与下一步

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d582_20261016`。
结果SHA：`e9518e38884310f76fddfafe182d0a3231a36c81362bb04f899e5cc05efdaef3`。
对应`full_w2_evaluation.json`、`full_w2_slo_rows.json`、`request_metrics.json`、
`cross_contract.json`和全部原始日志，旧结果零覆盖。PLAN及冻结V1/单轮V2
SHA与预登记文件一致，沿用D394共同阈值。

下一步保存和备份本次证据，再作当前来源/引用可核验的缓存维护，然后完成
Loquetier新适配完整W0/W1/W2及剩余7B比较。PrimeLoRA的G1/G2/跨场景优化、
3B、dLoRA、A1–A5、S1–S13、成本/CE、审稿文档及最终图表均未删减。
