# D1057：3B W1 ElasticOnly 完整观测及 Full 比较

2026-11-04。状态：单次回放、收尾与离线比较完成；完整目标仍 active。
不重跑本键，不改 D1047 Full，不修改冻结阈值，也不把反例筛除。
本次依照 analyze-results 分开记录观测、解释、主张边界和下一步；沿用现有
分析器，只扩展场景身份和对照配置检查，没有新建实验框架或改生产策略。

## 问题和共同合同

本项回答：相同弹性控制与后端下，Full 的主动 adapter 管理是否提供净收益？
ElasticOnly 是复合消融，不是外部论文 baseline，也不是单独的状态传播消融。

- Full 复用 D1047，ElasticOnly 为唯一 D1057，均为 3B W1、同 4,000 请求。
- 相同 source42 trace、W1 arrival view、adapter/prompt/原生 prompt token SHA、
  target map、fixed_length_greedy_v1；逐请求 4,000/4,000 输入比较通过。
- 相同 vLLM 0.30.0、模型/profile、1–3 实例 autoscaler、资源预算、准备窗口、
  原生按需加载与已到达请求驱动的来源准备；不使用未来请求，不按 phase 清缓存。
- 除独占结果/cache 路径与准确 scenario 名外，配置差异仅为批准的组合开关：
  load-only 排序、no-handoff、关闭主动协调/分层规划/软 admission/forwarding/
  HOST promotion，文件层使用 demand LRU。物理安全及原子 reservation 保留。
- V1/单轮 V2、D648 共同阈值和 D657 W1 预算均保持冻结。预算为
  6754.783565160236 GPU-s；95% 是项目冻结要求，不是行业通用标准。

source trace SHA：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`。
W1 view SHA：`8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f`。
PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
冻结协议 SHA、配置及其他来源 SHA 全部在 comparison/manifest.json 和
comparison/comparison.json 中，不以相同显示名称代替来源校验。

## 原始指标和直接结论

| 指标 | Full / D1047 | ElasticOnly / D1057 |
|---|---:|---:|
| 成功响应、绑定及原生长度核验 | 4000/4000 | 4000/4000 |
| TTFT 达标数 | 3972 | 3996 |
| TPOT 达标数（分母 4000） | 3858 | 3930 |
| 联合达标数 | 3836 | 3928 |
| 联合 SLO | 95.900% | 98.200% |
| 物理生命周期 GPU-s | 6182.715971 | 6032.623108 |
| GPU-s/offered-request | 1.545679 | 1.508156 |
| 平均 TTFT（ms） | 454.156048 | 337.218147 |
| P95 TTFT（ms） | 1114.091001 | 695.139877 |
| 平均 TPOT（ms） | 17.417030 | 15.140558 |
| P95 TPOT（ms） | 31.774061 | 27.148174 |
| 平均 E2E（ms） | 2323.526111 | 1940.005663 |
| P95 E2E（ms） | 5001.507240 | 4069.537885 |
| 真实远端 fetch | 132 | 132 |
| 线上传输字节 | 306360162 | 306360162 |
| 自然 scale-up / scale-down | 9 / 8 | 8 / 7 |

Full 相对 ElasticOnly：GPU 时间增加 **2.488020%**，P95 TTFT 增加
**60.268608%**，联合 SLO 低 **2.300 个百分点**。两点均满足本次时延条件及
共同预算，但独立数值正确性未知，因此不是无条件正式 G1/G2 排名。

这项单轮观测不支持“Full 比仅弹性控制更好”。结果不仅是 CE 或价格假设问题，
在冻结的物理资源与服务指标上也未显示净收益。不能用达到 95% 就关闭机制优势
问题，不能换 Full 展示点、放宽阈值或重复本配置寻找胜出。

## 实际开关和机制证据

| 观测 | Full | ElasticOnly |
|---|---:|---:|
| 按实际策略逐请求重算一致 | 4000 service_bins | 4000 load_only |
| 各自同状态下 service_bins/load_only 选择分歧 | 2079 | 1993 |
| 文件准备计划记录 | 2397 | 0 |
| GPU 准备计划记录 | 1267 | 0 |
| native HOST 准备记录 | 730 | 0 |
| residency epoch | 2389 | 0 |
| 实际 typed 软 admission 决策 | 707，全部 admit | 0，主动路径关闭 |
| 已观察软 headroom 不足 | 0 | 不适用：未评估 |
| dispatch GPU tier | 3327 | 3188 |
| dispatch HOST / NVMe / Remote tier | 2 / 538 / 133 | 0 / 680 / 132 |

ElasticOnly 的 9 个 activation（1 initial + 8 natural）全部 no_handoff、
preparation disabled、plan SHA=null；所有主动计划均为零，不是只改了标签。
Full 为 1 initial + 9 natural。所有 epoch 都有第一条实际 dispatch 和首 token，
无歧义、unused 或 controlled event。自然 first-service 只有 9/8 个，不能按
旧规则补同配置运行凑数，也不冒称已完成受控 A2/A4。

计划数不等于独立物理加载数。Full 的 1267 个 GPU 计划包含 816 superseded、
449 completed、2 cancelled；1042 个 attempt 中 312 个 receipt 标 stale_snapshot，
707 个 typed admission 都 admit。这些说明存在待核查的管理工作和失效计划，
但不能直接把它们当成浪费时间总量，或据此宣称软 admission 已避免争用。

Full GPU dispatch 命中高 **3.475 个百分点**，但没有减少本轮 fetch 或线上字节，
也没有转化为更好尾延迟。按需 NVMe 缓存本身已保留有用工件；本轮不能将
“较少 remote 下载”当成 Full 相对 ElasticOnly 的实测收益。Remote tier 请求
数与独立 fetch 数不同，不是计量矛盾。此结论仅限当前 workload/容量/权重集合，
不能外推为层级管理对任何 churn、带宽或缓存压力均无效。

## 阶段定位：观测不等于单因果归因

Full−ElasticOnly 的平均 TTFT 为 +116.937901 ms，对应现有六段差值：

| 阶段 | 差值（ms） |
|---|---:|
| arrival → admission | +68.446453 |
| admission → acquired | +5.083218 |
| executable → engine | +13.795494 |
| engine entry → queue | +21.842072 |
| native engine queue | +1.160212 |
| native prefill | +6.610453 |

最明显差异在进入推理前，但自然闭环同时改变负载分配、缓存、启动与批次。
不能将 +68.446 ms 全部归因于“路由函数 CPU 时间”，也不能把六段分位数相加。
同快照 shadow 只比较选择，不生成反事实 TTFT。两组输出 token hash 相同
3795 条、不同 205 条，差异逐请求保留；目标长度均一致。批次/数值执行差异是
可能解释，不是本轮已证明的唯一原因；hash 不同不自动判错，亦不证明正确。

## 证据边界、安全和复用

- source42 是已用于开发的固定负载；每配置单次运行，无运行间 CI、显著性、
  独立留出或全局最优主张。全部反例及旧候选永久保留。
- 既有 3B 工件全零更新不能验证非零 LoRA 算术可区分性。`n_correct=null`、
  `independent_numerical_proof=false`、`formal_ranking_eligible=false` 不回填。
  因此资源表分母明确为 offered，不冒充已独立证明的正确请求数。
- 远端复用 D80 已发布 gzip，无请求内打包、无 fallback；双端传输 ID、字节及
  内容 SHA 核验通过。没有连续远端 CPU/外部干扰监测的限制仍保留，不声称
  任何外部干扰都已排除。本轮没有重启服务或修改链路配置。
- D1057 的 9 次物理 allocation 全部释放，2243 次 watchdog 采样无
  high/max/OOM/kill/swap、foreign 或 escaped compute。服务峰值
  20295069696 B，主机最小可用 95057092608 B，磁盘最小可用156948242432 B。
- 资源域在正常收尾后退出，独占工作缓存为空；未删除原始测量。147 项旧投稿
  保护文件及两份用户 dirty 文件 SHA 不变。没有因性能结果差而提前中止。
- collector 及 comparison 首次执行通过，CPU-only 辅助域 3/4 GiB、swap0、
  affinity 2/3/26/27；保存时峰值分别 710287360 / 636633088 B，内存事件为零。
  两域已退出；这些是分析检查，不是额外性能实验。
- 原启动前 health 按原字节保留为 remote_health_prelaunch.json，后收集 health
  新建，避免覆盖。同原始场景 tc_ieee_elastic_only 贯穿四个统计入口。

## 文件与当前下一步

- 原始：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1057_20261104/`。
- 汇总：`paper_results/ieee_tc/ablation/d1057_3b_elastic_only_w1/completion/`。
- 比较：同级 `comparison/`，包含完整 CSV、JSON、逐请求输出差异、机制记录及
  复用分析脚本 snapshot；不复制大原始 JSON 入 Git。
- 原始结果 SHA：`28ef4e1465586d47629fe3de64d56b8725ecd5ed5ae71f54d047e9533d1fff47`。
- 完成审计 SHA：`cbb4daf69ccd39e245d330f7098116000fd1b77bcf2bdb5f85a959d83b32b946`。
- 实际运行代码检查点：`432435be36b9d43c46075512c4f6eb65acd80251`。

先备份本次完整结果，再按 A1 主线完成 ElasticLocality、LastKnown、NoHOST，
不回到无界参数扫描、不重开已关闭的 HydraServe 基础适配。下一项
ElasticLocality 优先复用现成 GPU-locality + demand-LRU 组合及既有测试；
必要的新配置仍须公开 locality slack 的批准选择依据，不看正式胜负调 slack。
Full 优势不足作为后续机制改进的明确问题；优化前继续按历史日志/失败方向、
原始资料和可证伪假设定位，不能通过删机制并仍标 Full 来达到目标。

当前 3B 自然 W1 的 Full、NoHandoff、CapacityOnly、FixedFull、DelayedHandoff、
NoRouting、R0H0、ElasticOnly 已有单轮记录，不等于其机制收益全部成立。
剩余 A1、7B 十一变体/两个累积节点、受控 A2–A5、S1–S13、M2资源配置问题、
CE/价格/Pareto、B1/B2/C1/C2/C3/公式推导/两轮审稿文档、IEEE 图表和最终
FaaSLoRA V2 集成全部保留。按最新指令此阶段先保存表格，不画优势尚未成立的图。

共享 k3s 不属于本项目，继续不接管、不改权限、不索取 kubeconfig。HydraServe
已有隔离 rootless Podman 路径及 D807/D808/D809 三种 3B 完整回放；pp1 范围
不能冒称已经验证 pipeline/consolidation。没有必要为本次答复重新部署。

本检查点使用 github-sync 白名单检查备份。实际 FaaSLoRA V2 的已有分歧不强推；
专属备份分支与 origin V2 的备份不等于完成最终集成。用户 manifest 和
EXECUTION_STATUS 原字节保留，不暂存；本页及 D1057 实现文档承担最新交接。
