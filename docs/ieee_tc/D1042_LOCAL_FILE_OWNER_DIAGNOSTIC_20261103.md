# D1042：当前进程内文件 owner 的直接计时

2026-11-03。接续 D1041，主线仍为 3B PrimeLoRA 优化。本轮不是性能候选，
不修改九公式、缓存容量、路由、驻留、准入、扩缩容或文件校验算法。
EXECUTION_STATUS.md 含用户修改，保持原字节；本文件为独立交接记录。

## 1. 为什么只补这一项观察

完整父观测 D1033：4000 条原生输入/adapter/token/时间合同通过，联合 SLO
3772/4000=94.30%，物理 U=8155.283735964913 GPU-s，P95 TTFT
1956.042105332017 ms。独立数值正确性未闭合，n_correct=null、
ranking_eligible=false；不能仅凭 token 数与绑定回执宣称 G1/G2 已达标。
D1040 的前缀 955/1000 不替代完整结果。

已有 D1033 的违约 Remote 请求显示平均发布包络约 461 ms，发布后至 native
load 约 412 ms，但没有当前进程内锁的直接计时。D1039 选了独立进程命令
观测器，而实际采用 LocalSourceReferences；其缺失不是零耗时。D1040 已
回答真实 D747 融合 descriptor 下的 HOST 清点问题，本轮不重复该剖析。

唯一问题：当前文件发布、快照、源租约路径，多少时间处于获得 owner 锁之前，
多少处于持锁阶段？持锁调用线程实际消耗多少 CPU？这能否为下一项优化提供
新的、可证伪的依据？未测到前不恢复进程 owner、不移动 SHA、不扩大缓存。

已核对历史：D628–631 的 SHA 移锁、D825 去重复 SHA、D1026 并行目标校验、
D838–840 解码器、D850 提前通知、D953→D957→D958 的进程 owner 后续，以及
D1041 列出的 HOST 闭合方向。D958 确实完整且干净退出，但为 7B 联合因素
诊断，不证明当前 3B 打开进程选项必然更好。

## 2. 观测定义与限制

`local_lock_spans_v1` 包装现有 threading.RLock，保持原 acquire/release。
默认未启用时仍为原生 RLock，无 wrapper/import；仅非正式且实际服务域内的
既有前缀诊断资格允许开启。不是通过改成另一种文件执行方式来测量。

每次记录 attempt/acquired/end 的主机单调时间与同线程 CPU，sequence、父
sequence、嵌套深度、PID/TID、调用位置和有界 adapter/transfer/lease/plan ID。
不记录权重、帧、完整参数、返回值、路径内容或异常消息。以已有 journal
限制 64 MiB/200000 records；截断、写入失败、未闭合均显式保留。

- acquired−attempt 包含锁竞争、OS 调度和 GIL 返回，不称纯锁等待。
- end−acquired 包含该持锁操作及嵌套操作；区间用并集/父子排除分析，不相加。
- thread CPU 只覆盖调用线程；D1026 并行 SHA reader 的 CPU 不在其中。
  低线程 CPU 比例不能直接解释为纯 I/O 或无计算成本。
- journal 写入开销不从请求 TTFT/E2E 扣除。外层写入在释放后；嵌套写入仍落在
  父区间，不能假装免费。诊断结果不进入性能排名。
- journal footer 的 owner_closed=false 表示未声明文件 owner 整体关闭，不等于
  资源泄漏；journal 完整性与服务真实释放分别验证。
- 只通过完全一致的 ID、时钟域和实际区间关联请求，不用最近时间猜配对。

依据为 [Python 3.12 RLock 语义](https://docs.python.org/3.12/library/threading.html#rlock-objects)
及 [同线程 CPU 时钟](https://docs.python.org/3.12/library/time.html#time.thread_time_ns)。
这些定义支持观测边界，不预先证明本机哪个阶段是瓶颈。

## 3. 最小验证与唯一诊断身份

20 项针对性测试通过：默认关闭、错执行路径拒绝、真实服务资格复用、递归锁、
父子序号、并发非阻塞失败与实际等待、异常保持、日志写失败不覆盖原异常、
截断/未闭合、真实本地 source/lease 以及 runner/export/config 一致性。
CPU 验证在既有 LLM 环境与 3/4 GiB、swap0、CPU2/3/26/27、CUDA 隐藏域内。
前两次辅助启动在任何测试前失败（未取得指定 affinity、使用错误 service
unit 类型），均已保留日志；随后使用既有 UUID scope＋taskset 成功。
没有由此启动 GPU 或修改系统委派。systemd 的 inotify watch 警告不等于磁盘
容量耗尽，实际限制和资源事件另有回读。

prelaunch 证据：`paper_results/ieee_tc/p2_backend/d1042_local_file_owner/prelaunch/`。
147 项旧投稿文件和两项用户 dirt 的 SHA 均不变。

仅一次 W2 既有源 trace 前 1000 请求，配置相对 D1033 只改三个独占路径：

```bash
FAASLORA_D1000_RUN_ID=d1042_20261103 \
FAASLORA_D1000_RUN_TAG=d1042_3b_local_file_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1042_3b_local_file_prefix1000.yaml \
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_LOCAL_FILE_PROFILE=local_lock_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

使用 run-experiment skill 的现有入口和 tmux；后端仍为已合格 vLLM 0.30，
不是该 skill 历史默认环境。不开 native/core/HOST profiler；显式保持进程内
file owner。D1037 当前仲裁修正仍在源码，完整性能尚未单独验证，不能把 D1033
历史成绩归给新源码。真实远端只读归档、共同准备60秒、外置回放及资源包络
均不改变。不重新生成工件/负载、不清远端 page cache、不在线重启服务。

启动后尽早确认 journal 的实际 PID/cgroup、local mode 与事件；若根本未触发，
记录测量错误，不把空输出当零。完成后先真实释放，再校验完整 footer、原生
合同/生命周期及逐操作分解。若持锁/等待不足以解释问题，关闭此方向；不自动
追加诊断、进程切换或参数扫描。任何性能策略需要另有新证据和执行身份。

## 4. 变更清单、保护与后续主线

| 文件 | 本轮变更 |
|---|---|
| faaslora/memory/file_state_process.py | 复用有界 journal，增加本地锁观测 wrapper |
| faaslora/memory/residency_manager.py | 仅显式 local 诊断时包装既有锁 |
| scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh | 本地文件诊断 opt-in，排除并行 profiler |
| tests/test_ieee_tc_file_owner_timing_journal.py | 8 项本地锁/真实 owner 测试 |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 3 项配置/实际 export 测试 |
| configs/ieee_tc/20261103_d1042_3b_local_file_prefix1000.yaml | 同策略、独占路径 |

诊断默认关闭；后续性能运行不带该 flag，必要时可单独撤销这些观测改动。
PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。

HydraServe 隔离适配与 D810 的18项 baseline 观测已完成，不接管共享 k3s，
不重开 baseline 寻优。仍须完成：最终同版本两模型 W0/W1/W2、G1/G2、独立
数值正确性及最终外部比较；A1–A5、S1–S13、CE/Pareto/价格、B1/B2/C1/C2/C3
及两轮审稿文档、达标后 IEEE 图表和最终 V2 集成。没有任何范围被此诊断替代。
结果领先由冻结协议和实际数据判断，不预先保证。完整目标保持 active。

## 5. 2026-11-03 19:21 运行中交接

源码检查点 `5df5addec6046c3ce7d2597e94eaea0996d57dd7` 已提交并通过
ls-remote 核验：origin/retry14_continuous_queue_v2 与
faaslora_origin/backup/d1042-local-file-owner-20261103。实际 FaaSLoRA V2
分支仍有历史分歧，未 force、未宣称最终集成。备份回执为本轮 raw backup.json。

19:18:55 在 tmux `tc-d1042-local-file` 启动一次；不是已完成结果，不重启。
raw 根为 `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1042_20261103/`。
运行目录 `d1042_3b_local_file_prefix1000/`；完整 launch.json 要在终态才出现，
当前 exec receipt、watchdog、replay、文件 journal 均在 `launch.launch/`。

- service：primelora-tc-svc-6af9ba6a6fb04b6e81606ced8ba7b46e.scope；
  InvocationID 8239967c532a45a4bf466c9756e19087。
- aux：primelora-tc-aux-ec75f1064b9c42dcb7f789c21a230cc4.scope。
- 控制器 PID847156/start_ticks321198452；watchdog PID847158；
  首个实际 GPU worker 观测 PID848705。
- early_identity.json 验证一个实际 local journal 的 PID、启动身份、cgroup、
  mode、首个事件和实际环境：file_state_process=0；D747 descriptor；没有
  core/step/HOST 详细计时。这里只核验控制器选择，最终仍需请求级 worker 身份。
- 本次轻量快照时27条 request terminal，非完整样本；不据此计算性能结论。
  service memory.events 全0，诊断未结束。运行前可用磁盘约153 GiB、RAM104 GiB；
  实际启动与持续安全判断以既有 preflight/watchdog 为准。
- 复用健康的既有远端服务，未重启/调整；未启动额外远端监控，本项限制保留。

下一步只监视现有运行，结束后确认实际 GPU/service/aux 释放再离线分析。
复用现有请求/阶段/资源分析，增加本地锁区间与嵌套关系的离线汇总，验证 footer
和截断标志，不把 journal owner_closed=false 当成资源未释放。没有新优化候选，
不重开 HOST 清点诊断，不修改在跑配置，不在 GPU 测量期间压缩、构建或 push。
该交接追加尚未推送，等待运行结束后与实际结果一起备份；已测源码备份完整。

## 6. 19:39 完成：直接观察不支持“发布主要在等待锁”

现有唯一回放已结束，不重启。launch/service/watchdog/replay 均正常退出，真实
GPU context 已释放、服务资源域已移除，HOST/NVMe 工作缓存已清空。146 次真实
远端传输共 338,845,661 bytes；逐 transfer ID 对齐远端和本地的内容 SHA、字节
与终态，全部一致。没有请求内打包、服务重启或远端配置变动。没有额外远端
CPU 干扰监测，因此不宣称已经完整排除远端 CPU 外力。

| 本次前缀诊断 | 观测值 |
|---|---:|
| 原生请求/adapter 绑定/固定输出合同通过 | 1000/1000 |
| TTFT 达成 | 990/1000 |
| TPOT 达成 | 972/1000 |
| 联合达成 | 965/1000（96.50%） |
| 前/后500请求联合达成 | 494 / 471 |
| TTFT mean / Type-1 P95 | 490.042290 / 1244.290639 ms |
| TPOT mean / Type-1 P95 | 16.101034 / 30.408441 ms |
| 生命周期物理 GPU 时间 | 2445.491293 GPU-s |
| 服务内存峰值 | 22,214,553,600 bytes |
| high/max/OOM/kill 与本服务 swap | 全0 |

这些数值包含诊断开销，且只有1000请求。n_correct=null、ranking_eligible=false：
独立数值正确性仍未闭合；本结果不替代 D1033 的4000请求、94.30%，不能宣告
完整 G1/G2 达标、性能提升或优于任何 baseline。实际请求逐条使用 D747 fresh
descriptor，未启用跨调用状态缓存；没有复用 D1040 的时间值填充本轮字段。

### 6.1 当前进程内直接计时

同一实际文件 owner 的44,567次记录全部闭合，无丢失、截断或孤立父序号。
submitted=observed=records=44,567。journal 记录的自身开销为5.620736秒，
不从用户时延扣除。实际持锁区间并集88.562446秒、acquisition 区间并集
13.641285秒；两种并集之间仍有交叠，不能相加成可节省服务时间。

| 直接观察对象 | 次数 | 平均获得锁(ms) | 平均持锁(ms) | 持锁调用线程CPU(ms) |
|---|---:|---:|---:|---:|
| 真实远端 publish_transfer（完全匹配 ID） | 146 | 2.573938 | 133.237942 | 56.433877 |
| preparation_snapshot（外层） | 525 | 1.313135 | 54.581739 | 54.476792 |
| inventory（含嵌套） | 545 | — | 24.256215 | 24.184026 |
| source_snapshot（含嵌套） | 10580 | — | 1.210397 | 1.192729 |

发布包络平均136.777919ms；分成锁前0.291983、acquisition 2.573938、持锁
133.237942、释放边界后0.674056ms。持锁时间占主要部分，并不是竞争等待主导。
获得锁的Type-1 P95仅1.895295ms；均值受少数较慢事件影响，完整分布保留。
caller CPU 不包括并行 SHA reader，不能把133.24−56.43ms解释成纯 I/O。

141条 Remote 请求取得精确的发布关联，其中4条 TTFT 违约：其对应发布平均
acquisition 0.003848ms、持锁253.651344ms。另1条 Remote 请求缺精确关联，
保留 req_00994 和未知原因，不按 adapter 或最近时间补齐。完整 transfer 可能
服务多个请求；表中 span 未裁剪到单请求 admission，不用它们相加推算 E2E。

preparation_snapshot 平均54.58ms中的 inventory 为嵌套工作，不能把两行
叠加。外层扣除直接子调用完整包络后平均29.744612ms，表明并非只有文件读取
成本；但本次没有进一步拆分这29.74ms，不给未测子函数分配虚构占比。

28条 TPOT 违约请求的平均 decode 窗口364.640601ms，与文件持锁并集重叠
78.355107ms。这是控制进程与异步推理的时间重叠，不证明GPU停顿，更不支持
直接扣掉78.36ms。两类请求的输出长度不同，不把原始重叠均值作为因果比较。

### 6.2 对下一步的约束

当前证据不支持以“锁竞争”为理由重做 D628 的 SHA 移锁、D850 提前通知，
也不自动恢复 D851–855 whole-owner 线程迁移或 D958 的进程 owner。D844/D856
stat 批处理、D860/D861 来源投影及 D862 撤回记录已再次核对；看到 inventory
或 source_snapshot 有成本，不是重做这些负面候选的新前提。

本轮问题已经得到回答，不追加同配置诊断。下一主线先复用 D1033 完整请求的
违约阶段与本轮直接计时，核查持锁处理/规划成本是否存在**尚未处理的具体
重复工作或串行依赖**；必须能说明与上述已关闭方向的差别，才登记一个新
性能候选。没有新前提则不做局部参数扫描。保留 D1037 当前仲裁修正尚缺完整
性能验证这一事实；不把历史父运行归给当前源码。

## 7. 离线工具、来源保护与备份范围

仅扩展既有 analyze_control_path_overhead.py：本地嵌套锁区间统计、header/
时钟/进程身份、完整/截断状态及精确 transfer 关联。31项既有及新增分析测试
通过（新增8项），原有 core/HOST 计时测试不退化。分析、远端记录取回和汇总
均在实际 GPU 释放后串行执行，辅助域3/4GiB、swap0、CPU2/3/26/27，CUDA隐藏；
没有新增生产机制改动或GPU回放。测试/分析中的CUDA不可见提示不代表GPU任务。

复用 D723/D1016/D703 的原生请求、资源、内容绑定与远端交付核查，不重建框架。
raw 中保留 analysis1/file_timing/lock_spans.csv 和完整journal；小型可查阅数据：
`paper_results/ieee_tc/p2_backend/d1042_local_file_owner/completion/`。
manifest保存每份汇总的SHA；各汇总继续绑定原始证据、分析源码和模板SHA。
147项旧投稿保护文件及两项用户dirty文件再次核对不变，不编辑或暂存它们。

HydraServe 无需共享 k3s 权限；已完成隔离适配与基线观测，不返回部署支线。
§4列出的全部未完成实验和交付继续保留。本轮结论是关闭一个瓶颈诊断问题，
不是完成整个优化目标；下一次任务从本节接续，不使用旧运行中状态重启 D1042。
