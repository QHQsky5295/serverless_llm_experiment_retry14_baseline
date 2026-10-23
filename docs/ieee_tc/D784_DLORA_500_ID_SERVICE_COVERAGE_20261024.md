# D784：dLoRA 3B 的完整 500-ID 实际服务覆盖

## 问题与继承证据

D783 已经闭合完整池启动和 100 请求检查：两 worker 均加载 500 个 adapter，
但请求实际仅访问 29 个 ID。D784 回答剩余的全 ID 功能覆盖问题，不重复该短
前缀，不以它的低 SLO 为理由搜索 baseline 参数，也不当作 W0 性能比较。

复用 D769 的 `adapter_coverage_v1` 索引变换：既有 3B 源 trace、500 个既有
adapter、既有 prompt 与输出目标，不创建新权重、完整 trace 或新工件池。
输入视图 SHA 为 `4a9f63b08a600c40af7f5acd8066fb518aa21ed48ee8d5c75d7b3e2e8a51088e`，
500 请求对应 500 个 ID、68,925 个目标 token。

## 不变条件

- D782 已核验的 CPU 零 MLP 视图；manifest
  `be6c1f20a1f7febd3a505b25584a8cfebc6ecf7a14be730aa125b81c7b98b78e`。
- dLoRA 原生 DP2/TP1 EngineManager、PERIOD_MIG、500 池、CPU KV 8 GiB；
  沿用 D783 的所有服务参数，不改原调度、batch、迁移或超时。
- 真实远端已发布 gzip；请求路径不打包，cold per-run 本地缓存。
- 部署通知后 60 秒按既有计划到达；原生固定长度 greedy 与共同 1,800 秒保护。
- 服务 72/80 GiB、swap 2 GiB；辅助 3/4 GiB、swap 0；CPU 与 watchdog 原配置。
- 源 trace、旧投稿和用户 dirty 不变；只回收审计后确认可重建的重复 objects。
- 冻结 V1 与单轮 V2 不变；数值正确性与原生绑定/输出数量分开判断。

## 执行和交付

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d784_20261024`。
启动前复用原 preflight 和输入校验；一轮 tmux 执行，结束后先释放资源和停止
所属远端服务，再用 D783 已适配的 dLoRA analyzer 审计并保存数据。运行中
不做构建、哈希扫描、压缩、远端调整或第二个重任务。

接受功能覆盖需要全部 500 ID 实际正常服务、逐请求原生 token 数符合目标、
绑定和物理计量完整。低 SLO 本身不否决功能覆盖，不声明独立数值正确性。
运行失败保留原始证据；不按胜负重跑或改变门槛。

下一主线为原计划 1,000 请求兼容、dLoRA 限定 3B W1/W2；之后 Loquetier /
HydraServe 3B、PrimeLoRA 完整对照和必要优化、同版本 Full、A1–A5、S1–S13、
成本补充、审稿文档及最终图表均保留。共享 k3s 属其他项目，不操作其资源。

## 本轮实际结果（单轮，已审计）

| 证据项 | 观测值 |
|---|---:|
| 计划／提交／正常终态 | 500／500／500 |
| 实际服务的不同 adapter ID | 500 |
| 原生输出 token | 68,925，逐请求符合目标 |
| 物理 GPU 生命周期 | 1,998.414445841 GPU-s |
| 平均／P95 TTFT | 442,047.077／461,939.953 ms |
| 平均／P95 TPOT | 66.035／78.254 ms |
| 平均／P95 E2E | 451,169.154／472,904.513 ms |
| 共同冻结阈值联合达成 | 0／500 |
| 原生时间、E2E、TPOT 重算误差 | 0 ms |
| 服务内存峰值 | 77,310,185,472 bytes |
| memory.high／max／OOM／OOM-kill | 1552／0／0／0 |
| 服务 swap 峰值 | 986,509,312 bytes |
| 主机最低可用内存 | 34,585,317,376 bytes |

两原生 worker 的初始 GPU adapter 容量分别 251／250，CPU 池均为 500，
GPU KV blocks 为 1143／1173，CPU KV blocks 均为 4681。保留原生双引擎
EngineManager 和 PERIOD_MIG；仅配置迁移不等于已证明迁移触发或收益。

平均提交滞后 5.617 ms、提交后 dispatch 等待 158,179.802 ms、dispatch 至
首 token 283,861.658 ms、decode 9,094.551 ms、完成通知 27.526 ms。
dispatch 至首 token 包含原生后端内部排队，不能将整个量称为 GPU 计算时间。
这些记录显示明显启动／服务积压；覆盖视图与 D783 的 100 前缀不同，不把
两者延迟差异解释为某个机制的因果效果，也不与 4000 请求 PrimeLoRA 排名。

完整 500 池实际从已发布远端读取，500 次传输共 1,160,474,851 线上 bytes，
本地逐文件核验 19,482,573,296 bytes，请求路径打包次数为 0。准备引用只计
一次真实传输，不能按 500 条请求再次累计。低 SLO 如实保留，不触发 baseline
性能搜索；这是全 ID 功能覆盖通过，不是完整 W1/W2 性能或独立数值正确性证明。

service、replay、watchdog 均返回 0，GPU 上下文与实际服务资源域释放；所属
远端两个工件服务及监测服务已停止，54 个原生 Ray 日志文件和原始请求／资源／
transfer／resolver 证据保留。共享 k3s 未修改。

审计复用 D783 已测试的 dLoRA 消费者和 D769 已批准的 coverage 视图；本轮
没有修改 runtime 或统计代码，GPU 只运行一次、离线审计一次通过。原始
`n_correct` 及独立数值正确性仍为未知，不因正常终态和 token 数量而补写。

已按原规则逐文件核对原池和下载副本，回收本轮 4000 个重复 objects，实际
释放 19,497,287,680 bytes，磁盘可用 162,792,108,032 bytes。147 项投稿与
用户修改保护 SHA 不变；唯一工件、原始请求和失败证据保留。引用检查范围为
可读的同 UID 进程与原有 fuser／挂载检查，三个不可读进程项在回执中明确保留，
不宣称系统级所有进程均可完整检查。存储辅助域已实际关闭。

下一 D785 为 1000 请求兼容，尚未创建或启动；随后为
限定 W1/W2，整个实验计划与 G1/G2 最优性仍未完成。
