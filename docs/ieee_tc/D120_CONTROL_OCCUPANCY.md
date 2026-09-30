# D120：7B Full 的并发占用与失败边界分析

2026-09-30。开发诊断，复用 D119，未启动新的 GPU/远程实验，未修改在线策略。
依据冻结指标 V1；共同 warm SLO、数值 adapter 身份和 Full 资格仍未通过。
源运行完整保留 4,000 个终态：2,890 个原生生成合同成功、1,110 个失败。
本表不是成功系统的性能排名，不计算单次运行 CI。

## 1. 成功请求的阶段分解

下表每行 n=2,890；单位秒。阶段不含失败请求，不能当作全体服务时间。
阶段 P95 不可相加；请求并发占用不是物理 GPU-s。

| 阶段 | 均值 | P95 | 对整个观察窗口的平均并发贡献 | 峰值 |
|---|---:|---:|---:|---:|
| 计划到达→dispatch gate | 957.259510 | 1787.339988 | 483.482943 | 1180 |
| gate→source admission | 3.008479 | 10.492889 | 1.519492 | 8 |
| source admission→native dispatch | 2.841817 | 11.691951 | 1.435316 | 7 |
| native dispatch→末 token | 3.739423 | 8.936358 | 1.888670 | 8 |
| 末 token→controller completion | 1.824671 | 8.619586 | 0.921586 | 8 |
| controller completion→outer terminal | 0.737410 | 2.675211 | 0.372443 | 8 |
| gate→outer terminal（上述后五项包络） | 12.151800 | 26.292820 | 6.137508 | 8 |

观察窗口 5,721.980519 s。gate 的末端采用 release 之后的 outer terminal，
所以最后一行是占用上包络，不是已单独测量的 gate-release 时刻。
controller completion 和 outer terminal 都不是已确认的客户端接收事件。
5,646 个窗口内资源样本的持卡 GPU utilization 样本均值为 33.795006%，
不是时间加权平均、kernel 饱和度证明或上述成功请求子集的专属利用率。

## 2. 独立控制观测：积压早于故障恢复

以业务开始为零点，首个失败终态为 2,878.662652 s；异常收集时刻为
2,878.663673 s；首个 quarantine 为 4,098.851197 s。

| 时段 | 控制样本 | 平均 active | 平均 queue | queue>0 样本 | queue>0 且 active<capacity |
|---|---:|---:|---:|---:|---:|
| 首个失败终态之前 | 923 | 6.503792 | 649.770314 | 910 | 584 |
| 首个失败之后、首个 quarantine 之前 | 375 | 6.714667 | 1510.058667 | 375 | 240 |
| 首个 quarantine 之后 | 641 | 2.684867 | 682.173167 | 639 | 143 |

`active_requests` 是当前 running/routable slots 中已绑定的请求，不包括 draining
slots，也不是全局 dispatch-gate 占用或 GPU 正在生成的请求数。
`queue_depth` 包含其余尚未完成请求，不能全解释为未提交到 GPU 的队列。
首个失败前 899/923 个样本的 ready capacity 为 8；不能从未满采样点推断连续
空闲资源。quarantine 后 190 个样本的 ready capacity 为零，不等于 GPU 已释放。
不能用 active 减去成功子集 native occupancy 推导所有非生成工作的占用。

## 3. 观察、解释与下一步

1. 积压已在首次请求失败和 quarantine 之前出现。后续恢复不能单独解释此前等待。
2. 成功请求在 native 生成前后存在多秒等待，而原生生成区间均值为 3.739 s。
   这支持检查准备、确认、清理与并发容量之间的耦合；尚不证明某条控制路径是原因。
3. 当前 7B runtime cap=2、max_num_seqs=2。先核查该配置的历史依据和 KV 约束，
   再选择一个最小可证伪验证；不直接提高 timeout、取消所有权保护或盲跑 Full2。

vLLM 将运行请求数和每轮 token 预算分别约束；增大前者不自动保证可行或更快。
见 [v0.30.0 优化文档](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/)
及 [对应 scheduler 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)。
取消终态不自动等于下层任务释放；见 [Python 3.12 timeout 语义](https://docs.python.org/3.12/library/asyncio-task.html#timeouts)。
这些是机制依据，不代替本机因果测量。Full/warm/Resident 完成前 baseline 仍暂停。

## 4. 分析实现、失败保留与来源

扩展已有 `scripts/analyze_control_path_overhead.py`，显式 `--allow-failed` 才接受
完整失败运行；默认仍要求全部原生成功。所有 4,000 个 ID 留在 CSV；失败的默认
零阶段字段输出为空，不虚构为零延迟。严格区分 task exception 在 outer finally
之后收集，与返回型 execution error 在 finally 之前构造两种边界。

首次分析错误地要求所有失败观察早于 terminal，因而拒绝 D119 数据；脚本、日志
和退出状态完整保留。修正 analyzer 的生产者边界判断，不改原始时间戳或指标。
第二次 21 项定向测试通过（0.176 s）；分析 2.26 s、峰值 RSS 213,704 KiB、exit 0。
仅加载 29,123,606 B projection 和 60,920,330 B 正常 outcome，不重读整个 9.48 GB JSON。
所有分析资源域 high/max/OOM/swap 事件为零；没有重复 GPU 回放或交付缓存重建。

数据目录：`paper_results/ieee_tc/p2_backend/20260930_d120_control_occupancy/`。
`summary.json` SHA256：`26ea0ce8153066693e44118fb29a0fd701089dd5efde15520201e1cc96141e73`。
原始分析脚本和日志：`results/ieee_tc/p2_backend_qualification/d120_20260930/`。
复用的 D119 结果及失败分类见 [D119 完整表](D119_FULL_W0_FULL1.md)。
