# D505: HydraServe 原始 1000 请求兼容性完成

2026-10-13。完成既定 100 → 500-ID → 1000 → 4000 顺序中的 1000 前缀。
未调整 baseline 性能配置，没有新增权重、工件池或负载，没有访问其他项目的 k3s。
这是兼容性结果，不是完整 W0 性能点，也不是 HydraServe 的最优配置。

## 原始观测与证据范围

| 指标 | 本轮观测 |
|---|---:|
| 计划 / 到达 / 终态 / 原生合同核验响应 | 1000 / 1000 / 1000 / 1000 |
| 失败 | 0 |
| 实际请求的逻辑 adapter 数 | 60 |
| 真实远端获取 / 线上字节 | 60 / 74,953,014 |
| 请求中打包 / 临时归档 | 0 / 0 |
| 平均 / P95 TTFT | 302606.358 / 523997.129 ms |
| 平均 / P95 TPOT | 62.884 / 79.803 ms/token |
| 平均 / P95 E2E | 310140.195 / 527501.027 ms |
| 物理 GPU 生命周期区间并集 | 6554.059487593 GPU-s |
| GPU-s / offered request | 6.554059488 |

原始 `trace_prefix_v1` 的前 1000 个 arrival/prompt/adapter/target 不变，
view SHA `0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`。
它不同于 D504 的 500-ID 覆盖视图，不能混成配对性能结果。500 个静态逻辑 ID
仍是既有 498 零 / 2 非零适配器、4 个权重 SHA 类别；不宣称独立训练多样性。

全部请求通过 canonical input、adapter/native index、实际来源内容 SHA、原生
输出长度、worker/时钟身份及计时分解核验。所有远端 transfer ID 与本地回执
匹配。两个既有非零请求的输出与 D502 一致，仅为一致性见证。
`n_correct=null` 与独立数值正确性未证明标记保留；不将名称、HTTP200 或长度
单独等同完整数值证明，也不追加跨后端逐 token 完全一致的替代验收规则。

## 解释边界

平均提交滞后 4.031 ms，提交至实例分配等待 260795.201 ms，分配至首个原生
输出 41807.126 ms，decode 7491.086 ms。artifact resolve 平均 225.439 ms，
嵌套在服务路径中，不重复相加。这表明本轮长延迟主要在分配等待及后续服务
路径被观测到，不能将其全部称为网络时间、模型启动时间或已经证实的机制缺陷。
本轮不是隔离原因的消融；不据此前缀推算 4000 请求 SLO 或修改 baseline 参数。

物理 GPU-s 互斥窗口：到达前 0，到达窗口 4106.776171554，drain
2270.325859637，终态后回收 176.957456402，总和 6554.059487593。
计时沿用 D503 的实际容器创建前 reservation 边界，不冒称更早的 native
bookkeeping 已被恢复；本次 pp1 claim/create 无中间 await。多 rank 顺序创建、
cache reservation 若后续被触发，需单独核验边界。

原 Scheduler、modified vLLM0.4.2、model tuple(13,9000,200,1000000,1)、
batch8、USE_CACHE0、SLOW_EXPR0 及原生 30 秒 idle retirement 与 D503/D504
一致。实际创建并自然释放四个 pp1 实例，只有一个物理计算节点。
不强制 pipeline/consolidation；其完整数值与 KV migration 资格仍未证明。

## 执行、安全与来源

- 原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d505_20261013`。
- 第一 CPU prepare 因错误假定 request_view 缺省而中止，未生成输入、未启动
  服务或 GPU；prepare2 检查原配置明确的 trace_prefix_v1。原错误和代码保留。
- 远端 start 成功后，独立认证 health、来源和资源检查实际通过并读取，随后
  才启动测量。不存在 D504 的未读取前置失败就启动问题。
- 原生执行、实际释放、远端收尾整体 tmux exit0；没有重试性能运行。
- service Invocation `9af869eb405a4d339f67d6cc4497562e`，inode24109586 已移除；
  launch auxiliary Invocation `cd65124d387f4e138a2a61461b1f0852`，inode24109386
  已确认递归为空并关闭。所有实际 GPU/worker/provider/storage/pause 已释放。
- 服务峰值 49,888,940,032B；high/max/OOM/OOM-kill/swap 全零。
  保持共同72/80GiB、swap2GiB和CPU包络。
- 远端三个服务按启动 Invocation 精确核验，在本机释放之后关闭，18080/18081
  不再监听。远端日志下载 SHA 匹配；请求中没有管理、打包、扫描或压缩。
- 启动前真实链路 1000Mb/s/full；历史100Mb观测的物理原因仍未知。
- 收尾磁盘可用 185,855,791,104B；下一次启动重新核验，不据此放宽150/100GiB。

curated evidence 位于 `paper_results/ieee_tc/hydraserve/d505/`。复用未变源码的
D504 88项 CPU smoke，备份前另作 fresh smoke。旧投稿147文件、用户修改及
原始 vendor checkout 仍须由归档/备份检查逐项核验；无凭据和权重入库。

## 下一项主线

保存并备份本轮后，沿用同一已验证原生配置进行一次完整4000 W0，不搜索
性能参数。明确列出 native runtime 目标与所有系统共同的 D394 evaluation
阈值，不能把原生9000/200ms当作共同验收阈值，也不称该公开工作点为最优。
结果无论是否达标均保存，然后核对全部可用 baseline，再优化 PrimeLoRA。
不因原始数值正确性字段未知而伪造值，不通过不必要的跨后端逐字重复拖延主线。
7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/文档和最终图表仍未完成。
目前不画图、不修改论文、不覆盖历史结果。
