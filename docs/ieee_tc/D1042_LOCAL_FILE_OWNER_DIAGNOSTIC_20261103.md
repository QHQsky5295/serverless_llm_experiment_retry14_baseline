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
