# D1064：完整消融后的额外等待定位

2026-11-05。复用 D1047 Full、D1057 ElasticOnly、D1061 ElasticLocality 的
各 4,000 请求及规划事件；新增 GPU 实验为 0。没有修改服务机制、九公式、
模型配置、阈值、工件、负载或任何基线，不重跑已完成的 NoHOST。
用户 dirty 的 EXECUTION_STATUS.md 不编辑、不暂存，本页承担本次交接。

## 1. 当前主线和 Kubernetes 边界

用户确认现有 k3s 非本项目所有。这一边界已在既有
HYDRASERVE_ISOLATION_DECISION_20261012.md 中落实；HydraServe 使用独立 rootless
Podman 和其官方专用后端。D807/D808/D809 的 3B W0/W1/W2 已完成，继续复用，
不修改共享集群、不申请其管理员配置、不重新部署。pp1 未证明 pipeline /
consolidation 的完整性能收益，该限制仍保留。

两模型的已测外部比较已汇总；3B A1 十一组已齐。现在的问题不是简单的
“有没有达到 95%”，而是 Full 在相同 W1 下未获得相对简化变体的净收益。
继续按 G1/G2 判断，不改回 CE 第一，不把 ElasticOnly/ElasticLocality 当外部
论文基线，也不因其更好而删除结果。

## 2. 原始观测及等待分布

三组共享 W1 输入和固定输出合同，逐请求身份及完整时间分解再次核对。
下表为各单轮的请求算术均值，单位 ms；不是多轮均值或置信区间。

| 观察 | Full | ElasticOnly | ElasticLocality |
|---|---:|---:|---:|
| 联合时延达成率 | 95.900% | 98.200% | 98.850% |
| 物理 GPU-s | 6182.716 | 6032.623 | 6087.727 |
| 平均 TTFT | 454.156 | 337.218 | 314.378 |
| 到达释放迟到 | 20.640 | 5.563 | 5.339 |
| 释放至 acquire | 24.577 | 19.138 | 19.187 |
| acquire 至恢复 | 23.170 | 7.563 | 7.182 |
| runtime-slot 区间 | 130.026 | 97.702 | 92.111 |
| admission 至 engine dispatch | 88.171 | 69.292 | 52.253 |
| engine entry 至队列 | 56.152 | 34.310 | 33.729 |
| native queue | 2.590 | 1.430 | 0.987 |
| prefill 包络 | 108.831 | 102.220 | 103.590 |
| decode 包络 | 1799.707 | 1569.061 | 1551.824 |
| worker 完成通知 | 45.972 | 30.423 | 30.118 |
| worker 至 controller 完成 | 23.691 | 3.303 | 3.734 |

前八个时间分项之和为 TTFT；加 decode 和两段完成通知得到 E2E。
runtime-slot 区间含状态获取、预留和等待，不等于纯容量排队；prefill/decode
是实际墙钟包络，不是纯 GPU kernel 时间。P95 不能像均值这样相加。

另外保留以下嵌套观察，**不再加到上述分解上**：

| 嵌套观察（ms） | Full | ElasticOnly | ElasticLocality |
|---|---:|---:|---:|
| 已记录 routing span | 9.893 | 16.204 | 11.295 |
| native load-and-acquire | 17.160 | 19.445 | 9.854 |
| RPC 完成后的 pickup 延迟估计 | 22.801 | 2.394 | 2.825 |
| RPC channel acquire | 0.023 | 0.016 | 0.020 |
| RPC send/flush | 0.121 | 0.137 | 0.146 |

routing 字段不是完整控制 CPU 时间：源码中快照返回 None / membership 冲突后
提前 continue 的尝试不累加该计时，而且跨度包含 await。故“Full 的已记录
routing 值更小”不能证明路由总成本更低，也不能由 runtime-slot 大就判断公式
本身太慢。RPC pickup 使用已有同主机 wall-clock 与 producer clamping，不能
把它冒称独占事件循环阻塞时间。

Full 在八个预先定义的 phase 中，arrival-release、runtime-slot、decode 和
RPC pickup 的均值均高于两个简化变体；差距不是仅有首次初始化一项。三组
远端次数/线上字节均为 132 / 306360162，不能用多下载解释本次 Full 差距。
仍然可能存在路由选择、活跃请求形态、控制执行与实际准备的共同影响；复合
消融不固定这些闭环轨迹，当前分解不是它们各自的因果贡献。

## 3. 已保存事件中的主动准备工作

Full 保留 2407 个规划 CPU 事件：2398 个 owned_execution_epoch、9 个
initialized_gpu_objective；2406 completed、1 cancelled_result_discarded。
简化变体均为 0，符合它们关闭主动管理的消融定义。

| 每规划事件的观察 | 均值(ms) | P95(ms) | 累计(s) |
|---|---:|---:|---:|
| 输入冻结/序列化 | 8.515 | 12.065 | 20.495 |
| 等待串行 planner | 174.352 | 632.240 | 419.664 |
| worker wall | 210.557 | 476.199 | 506.810 |
| worker CPU | 210.525 | 476.132 | 506.734 |
| 结果完成至 join | 14.399 | 25.990 | 34.659 |

纯规划已经在独立 worker 中执行，不能把其 506.7 CPU 秒说成主请求线程阻塞，
也不能从请求 E2E 扣掉这些累计值。等待、并发请求和规划相互交叠。

进一步直接复用共享文件观察的 round_id：2389 条 residency 记录引用了
1048 次不同的物理文件观察，而不是 2389 次独立扫描；同 round 的记录完全
一致，引用数与 replicas 字段一致。每次文件观察平均 61.643 ms、P95
76.658 ms，合计 64.602 s。实际执行入口是进程内 preparation_snapshot，
它在事件循环调用中同步读取并构造文件状态；这提供了具体待核对的执行边界。

1048 次观察中有 121 个不同的 (owner, epoch)，927 个相邻观察 owner/epoch
相同。**这不意味着可直接缓存或省掉 927 次校验**：文件 extent、外部变化与
完整物理容量的合同不能只由 epoch 相等保证。当前数据既没有证明安全缓存，
也没有证明 64.602 s 可以转化为某个 TTFT 或 GPU-s 改善。

## 4. 历史与官方资料约束下的判断

本次核对 D998（只读控制延后退化）、D1036/1037（扩大 HOST 容量退化与已完成
控制仲裁修正）、D863/1040（原生控制/HOST 清点）、D1042（文件 owner 直接
计时）、D737–739（规划消息延后解析退化）、D740/741（已完成消费视图）、
D731（已实现同控制轮共享文件观察）。这些方向不能因再次看到等待就换名重做。
同样不重开 whole-owner 线程、stat 批处理、全图缓存或容量网格。

在线重新核对 [Python 3.12 asyncio 执行边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
和 [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)。
前者解释同步工作为什么会延迟同循环的任务；后者表明后端输入/utility 与
engine-step 的执行位置。它们支持检查这些边界，**不证明本项目哪一函数造成
全部差距**，也不授权再次应用已被完整回放否定的线程化/延后候选。

本轮裁决：尚无足够具体的新前提来登记一个服务性能修改。保持 Full 代码，
不凭总耗时直接调参，不再补同键画像。下一项限定为源码和已保留事务的依赖
核查：找出同一冻结准备事务内尚未处理的重复展开/校验或可消除的串行依赖，
并与 D163/D735/D741/D747/D751/D869 的已完成优化逐项区分；只有得到精确
语义等价依据，才做一个有界 CPU 验证与新的完整回放。不关闭核心机制冒充
Full 优化、不以未知因果关系启动下一轮长实验。

## 5. 分析产物、验证和交接

复用 analyze_d455_stages.py 和既有 native 时间分解，只添加显式封存 comparison
输入模式；支持完整来源 SHA、三组输入匹配、缺字段拒绝、阶段恒等式及旧摘要
数值复核。新增嵌套字段检查不填零、不创造反事实延迟。44 项针对性/既有分析
测试全部通过，含 4 项新增；1 项既有 pytest 配置警告。CPU 检查不是 GPU 实验。

产物：`paper_results/ieee_tc/p2_backend/d1064_retained_a1_overhead/`。
包含 summary.json、stages_and_nested.csv、file_rounds.json 及检查点 manifest。
原始小型分析脚本和 tests.xml 位于
`results/ieee_tc/p2_backend_qualification/d1064_20261105/`，不复制三份大日志。

summary SHA：304fe266e7481fa873338f1b7f561dbb167f28509831c25628efdc5bac0af798。
file_rounds SHA：d3039a8daa3e3fa834c9431218fec235a89fa034746a09a0fea84a33b9f161fe。
PLAN、冻结 V1、单轮 V2 均保持原 SHA。原始 n_correct=null、数值证明限制和
formal_ranking_eligible=false 不改；不生成 CI，不把开发回放改称独立留出。

分析使用 3/4 GiB、swap0、CPU2/3/26/27、CUDA 隐藏资源域。测试及主分析域已
inactive/dead、正常退出；没有保存其最终 memory.peak，不虚构为零。文件观察
分析记录峰值 30887936 B、high/max/OOM/swap 均零。两项用户 dirty SHA 未变。
无 GPU compute 进程，磁盘约 150.9 GiB；未删除唯一证据或进行重型清理。

按 github-sync 的明确路径、旧投稿保护清单、来源 SHA、diff、测试和凭据检查
备份。实际推送成功以本轮 raw backup.json 为准，不预写成功；origin V2 与
FaaSLoRA 独立备份分支核验，不强推分叉的实际 FaaSLoRA V2。下一检查点同时
纳入 D1063 已完成的后置备份交接。

首次备份在任何 stage/commit/push 前停止：校验器把 JUnit 的 aggregate
tests=61 当成 pytest 收集的44个测试方法。实际 XML 有44个 testcase，全部无
failure/error/skip；现按 testcase 数检查并单独保存 header。失败回执保留，
没有重跑测试、分析或性能实验，也没有改写原 XML。
第二次备份停在提交前的换行检查：Python csv 标准 CRLF 被默认 whitespace
规则标记。原 CSV 和 manifest 的字节/SHA 保留；续传只将 CR 作为合法行尾
检查，其余行尾空白、文件末尾空行和 tab 前空格检查不变。没有性能结果重试。

仍待完成：Full 净收益优化、7B 十一组 A1 与两个累积点、受控 A2–A5、M2、
S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3 与两轮审稿证据、IEEE 图表、最终
同版本来源统一与实际 V2 集成。HydraServe 不重新部署，整个目标保持 active。
