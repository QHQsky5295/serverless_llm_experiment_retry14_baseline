# D1016：当前 3B 短输出停顿的单次原生诊断

2026-11-02，接续 D1015；本文件先登记问题与执行身份，结果另行追加。
当前是 baseline 必要适配和完整观测之后的 PrimeLoRA 优化阶段。HydraServe
继续使用已验证的隔离部署，不接管用户已有共享 k3s，不重开基础适配。

## 问题与终止条件

D1000 完整 W2 为 3690/4000 联合达标（92.25%），仍未达到冻结 95%。
D1014 减少查询的候选已拒绝；D1015 逐请求复用表明其净减少 38 条达标请求，
资源差仅 −0.381829 GPU-s，不采用该策略。D1000 的 154 条 TPOT 违约中
147 条输出不超过 64 token，现有完整请求日志只有阶段包络，不能分出控制
事务与 engine step。D863 的原生诊断早于文件 owner、物理清点和到达预取
修改，不能用旧占比冒充当前瓶颈。

本次唯一问题：当前短输出的首末 token 窗口内，同步控制调用、input 非控制、
engine step、其余时间分别是多少？先测这个缺口，再决定一个有证据的新候选；
不是再测试已拒绝的只读 deferral、files-first、提前等待、实例数或 token 预算。
若窗口覆盖不足，报告缺失，不扩观察上限或重跑凑样本。只有新依赖证据才可
重开已关闭方向；没有证据就不堆叠补丁。

## 执行合同

- 配置是保留 D1000 的精确克隆，仅 output/NVMe/HOST 三个独占路径不同。
- 复用现有 D1000 runner 和 D328 原生观察 wrapper；正式默认路径不变。
- 完整 source42、W2 map 先全量验证，再执行原序前 1000 请求。保留完整源
  4000 身份，不重建任何工件、到达、prompt、target 或 adapter 映射。
- `formal_run=0`，`diagnostic_prefix_v1`；不以短诊断 SLO 替代完整回放。
- 只启用既有 `host_spans_v1`；不启用 Python 栈、step 子组件或 HOST 清点
  子计时，不修改观察器、线上算法、原生后端或 IEEE 九公式。
- 每 native 进程 64 MiB / 200000 spans 上限不变。记录覆盖与截断；主机
  span 不是 CUDA kernel 时间。观察开销保留，不做事后扣除或无观察器胜负比较。
- 仍为 stdlib 解码、真实远端已发布 gzip、共同固定输出、D648 SLO 与 V1/V2。
  原始 `n_correct=null` 不改写成独立全量数值正确性通过。

执行键：`d1016_20261102/d1016_3b_native_prefix1000`。启动命令：

```bash
FAASLORA_D1000_RUN_ID=d1016_20261102 \
FAASLORA_D1000_RUN_TAG=d1016_3b_native_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1016_3b_native_prefix1000.yaml \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原始目录位于 `/home/qhq/serverless_llm_experiment/results/ieee_tc/`
`p2_backend_qualification/d1016_20261102/`。独占创建，不覆盖旧 run。

## 远端与安全边界

本次只读检查发现 3B port 18080 已由 lab14 的 PID 1984520 提供服务，
2026-10-31 启动，位于 session-8529.scope。认证 health 确认为既有 3B 根目录、
`prepublished_gzip_v1`、`artifact_timing_v2`，clock ID
`remote-process-monotonic:b7a04b62f38c4ae6adabb4da5c0fd8dc`。
对应用户 unit 当前 failed，原因为 Address already in use，不是 HTTP 服务
失效。不能把该 standalone PID 冒称在 unit 的资源限制内或本次启动成功。
沿用健康且配置一致的服务，不杀监听者、不重复启动，不停止另一个 7B 服务。
本轮未拥有其启动身份，结束后也不停止该既存服务。远端 NIC 当前协商 1 Gbit/s，
不据此声称应用实际吞吐等于 1 Gbit/s。测量期间不改远端或额外读取工件。

既定服务 MemoryHigh/Max 72/80 GiB、swap 2 GiB、CPU 4–23/28–47；外置
回放与监控 3/4 GiB、swap 0、CPU 2/3/26/27。空启动 MemAvailable≥102 GiB，
磁盘≥max(150 GiB,100 GiB+1.5×新增峰值)，运行低线 16/100 GiB 不变。
启动前可用内存约 104 GiB、磁盘约 150.55 GiB，无 GPU 计算进程。沿用已通过
的进程树限制、watchdog、cleanup，不重复广泛资格；不并发压缩、测试或构建。
不删除其他项目、未知数据或唯一证据。

## 官方来源、验证与变更清单

本轮复核 [vLLM 0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
与 [profiling 文档](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/)。
输入处理和 engine step 是不同边界，主机 step 还包含输出处理等工作，不能
等同纯 GPU 计算；工具观察可能影响性能。官方结构支持分解方法，不证明本机
当前瓶颈。已有 atomic_control_yield 不重新当作待实现的优化。

37 项定向测试通过，含配置仅三路径变化、opt-in/default、W2 原序前缀完整验证、
open-loop 与原生计时语义；两 shell 的 bash -n 通过。不是性能成功证据。

按 system-profile 的变更清单：

| 文件 | 变化 |
|---|---|
| configs/ieee_tc/20261102_d1016_3b_native_prefix1000.yaml | D1000 三路径克隆 |
| scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh | opt-in 复用原生观察 wrapper，默认不变 |
| scripts/ieee_tc/profile_d328_controller_python.sh | 登记本诊断 tag/output |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 2 项定向合同测试 |
| 本文档 | 问题、身份、边界、结果记录 |
| 生产策略、core_timing、后端 | 无修改 |

PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
V1 SHA：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
V2 SHA：`193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
D648 SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。

保存、清理与分析复用 D863/D723 的原生窗口分析，不另建框架。147 项旧结果与
用户修改保持零变化；不 stage 用户 manifest 或 EXECUTION_STATUS。显式提交
本轮文件并推送 origin V2 与 FaaSLoRA 专用 backup，不强推分歧 V2。此独立
文档承接状态，不覆盖用户修改的历史 tracker。

尚未完成：最终同版本 W0/W1/W2、G1/G2 主比较，A1–A5、S1–S13、正确性范围
汇总、审稿证据文档/图表、最终 V2 集成。当前诊断不替代这些任务，不宣布最优。

## 本次实际结果：归属检查误判导致提前中止

入口提交 `08eec2d40fd2c70e92a9d33f276c4ecd32e46c55` 已在启动前推送并
核验 origin V2 和 FaaSLoRA 的 `backup/d1016-current-native-prefix-20261102`。
tmux 启动于 17:30:58，确实进入模型执行；不是只有启动命令成功。

这次没有完成 1000 请求。外置回放记录 N_plan=1000、N_arrived=56、
N_submitted=55；服务收到 55，记录 45 个终态，其中成功且原生合同匹配 44、
中断 1。未到达/未完成部分不补成成功或超时。完整源仍为 4000。
观察到的 GPU 时间为 348.282748 GPU-s，完整 GPU 时间为 null，不参与资源
排名。没有运行要求完整前缀的分析器，不用中途 live 5000 ms SLO 当冻结指标。

最早错误是旧 `_assert_clean_gpu_environment` 在 live_progress 报告
`gpu2 pid=1445406 [No data] 7326MiB` 为 foreign。核对发现：

- 同 PID、start_ticks=311918514 在独立监控的 74 个样本中属于本轮服务资源域。
- 原生 allocation journal 也绑定相同 PID/出生身份和 GPU；没有外部计算进程。
- 本轮在线控制实际发出一次自然缩容，原因 `ieee_all_low_cooldown`，
  started=3119266.967794040、released=3119275.314833835；未接纳请求、无
  unresolved request。worker 日志显示正常 shutdown 请求及 SIGTERM 收尾。
- 旧检查先取 nvidia-smi 表、再读瞬时 parent/children 集合。退出阶段可能
  已丢失父子链，但先前 GPU 表仍保留该 PID；这不构成外部作业的证据。
- 所以早期 commentary 中的“后端退出”只描述现象；不是已经证明后端崩溃。
  不能将这次误判归因为 LoRA 数值失败、CUDA OOM、观察上限或外部抢卡。

183 个资源样本无保护中止；服务峰值 17284587520 B，high/max/OOM/swap
均为 0。launcher pass=false，service/replay 返回 -15/1；watchdog 返回 0。
最终三个原生进程已释放 GPU、服务资源域移除，147 项旧文件和两个用户 dirty
文件 SHA 不变。远端既存服务未停止、未重启。三个原生观察文件正常写入 close，
但只有部分负载，不能完成原先的瓶颈问题。

保留产物：`paper_results/ieee_tc/p2_backend/d1016_interrupted_native_prefix/summary.json`，
绑定原始回放、终态、完整失败 outcome、worker 日志、原生计时和资源来源。
原始 `analyze.py` 是预备的完整分析入口，未对失败输入执行；`close_failed.py`
只整理这次失败。其首次辅助启动因 scope 名不符合既有 UUID 身份格式，在读取
实验输入/生成产物前失败；使用原规范名字后完成，未放松资源检查。

## 修正决定与验证，不是性能优化结论

使用 [psutil 官方说明](https://psutil.readthedocs.io/stable/#psutil.Process.children)
核对父进程消失会丢失后代链；并检查现有 `NativeGPUCensus`：它已经按服务
cgroup、PID/start_ticks 判断归属，保留未知身份、阻止外部计算进程、检查逃逸，
最终 GPU release 也由它见证。它是本轮真实启用的独立每秒监控，不是新写的
“忽略错误”分支。旧祖先链检查来自历史提交 `60737ddf`，早于该资源协议。

因此只在已核验的 TC 服务中由既有独立监控承担 GPU 归属保护：

- 每次先调用现有 verify_current_service，核对 live watchdog 出生身份、
  服务资源域和实际资源限制；环境标志本身不构成授权。
- 必须存在非测试的原生 GPU census 与已锁定 NVML binding 身份，否则拒绝。
- 非 TC 路径保留原有外部 GPU 检查；不扩大允许 PID 列表、不忽略 `[No data]`
  或未知进程、不加重试/sleep，不改变缩容、路由、admission 或后端公式。
- supervisor 持续检查 watcher 退出；外部 GPU/逃逸/监控失败和实际释放规则
  均未改。不同采样视图不能再以父子链缺失推定“外部”。

新增修改：`scripts/run_all_experiments.py` 的 GPU 环境检查与
`tests/test_basic_smoke.py` 的 4 项合同测试。6 项相关 runner 检查通过；7 项
实际 supervisor 解释器的身份/退出检查和 9 项既有 native census 检查通过。
首次将 supervisor 测试误用 serving venv，7 项中 1 项因其 signal 模块缺少
pidfd_send_signal 失败；改用既定 `/usr/bin/python3` 后原测试通过，未改测试
或监控语义。保留该失败，不当作生产 watcher 功能缺失。

下一步是修正测量错误后的新 attempt 身份，不覆盖本次失败，不做同键择优。
只完成原登记的 1000 请求问题，不追加另一套性能 smoke 或扩大诊断样本。
重新启动前仍检查磁盘/内存、源与备份；当前尚未运行修正后的 GPU attempt。
最终主线与全部未完成项不变。

## 修正测量错误后的 attempt 2 登记

2026-11-02：新身份为 `d1016_20261102_attempt2/`
`d1016_attempt2_3b_native_prefix1000`，不覆盖上一中止目录。复用原 D1016
配置（SHA `1f6ecbbff539862823bafe4964335ca7267149cea2099f24f3cc0db807cef286`），
原 runner 用新 run ID/tag 覆盖 output、NVMe、HOST 三个独占路径；不复制
工件或负载、不改变任何性能参数。wrapper 只新增该 tag 的观察输出目录。
这是修复已确认测量错误后的新 attempt，不是追加胜负/覆盖率采样。

```bash
env -u FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER \
  PYTHONPATH=/home/qhq/serverless_llm_experiment_retry14_baseline \
  FAASLORA_D1000_RUN_ID=d1016_20261102_attempt2 \
  FAASLORA_D1000_RUN_TAG=d1016_attempt2_3b_native_prefix1000 \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1016_3b_native_prefix1000.yaml \
  FAASLORA_D1000_PREFIX_COUNT=1000 \
  FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

继续复用上一轮 37 项诊断合同检查与修正后的 22 项 GPU 归属/监督检查；
本轮只检查新输出映射和 shell 语法，不重复广泛资格。上一中止记录、完整
分析的输入完整性要求、观察上限、资源保护、未知数值正确性均保持不变。
本节登记时尚未启动，不预写成功或瓶颈结论。用户修改的总 tracker 不覆盖；
本文件继续承接实际状态。完整主比较、A1–A5、S1–S13及最终交付不删减。

attempt 2 启动前：新映射和既有两项 D1016 合同共 3 项通过，两个 shell
语法通过；147 项旧结果核验不变。可用磁盘 161612382208 B（约 150.51 GiB），
MemAvailable 111611817984 B（约 103.95 GiB），没有 GPU 计算任务或 tmux
实验。原生前缀最多 32 GiB 新增预算沿用旧值，不改变 150/100 GiB 门槛。
再次只读认证 health 成功，既有 remote clock ID 与交付模式不变，不重启服务。
GPU 归属修正已在提交 `b9fc395e6fa7ac4a245dfa94727047fc1d5fd1f2` 保存；
本次只登记新输出路径，不再次修改监督或生产策略。

### attempt 2 实际启动与运行中交接

入口提交 `26429f75c5457cf6e0b94b7ea6036e9c6711a052` 已在测量前 push，
两端 ls-remote 均核对一致：origin V2 与 FaaSLoRA 的上述 backup 分支。
2026-11-02 17:50:14 启动 tmux `tc-d1016-attempt2-3b-native-prefix1000`。
真实服务为 `primelora-tc-svc-fecc66262d8d4f97af92e126828b8c9d.scope`；
外置资源域为 `primelora-tc-aux-1822f485f3144d22aedb480530c855af.scope`，
watchdog PID 1486284 / start_ticks 312026384，使用既定原生 NVML census。
回放 receipt 已确认 source_count=4000、count=1000、W2、原 frozen map，
实际观察文件已产生，不是只有启动命令成功。

sample 230：提交至 req_00091，已有 90 条终态，持卡由三张减为两张，
foreign/unresolved 均为空；资源保护未触发，high/max/OOM/OOM-kill/swap
为零，服务记录峰值 17250828288 B。旧误判时间点已越过，但这不代替
完整运行验收。当前仍 RUNNING，没有 launch.json 终态或 1000 请求结果。
不要重启、追加同键或对部分观察运行完整分析。测量期间不 push、测试、
压缩、维护远端或新增另一重型任务；这个轻量交接追加在结束后统一备份。
下一步监控同一 tmux/原始目录至终态，再按既有 D723/D863 分析入口核对
完整性、GPU 释放及原生窗口分解。不能先写出控制或解码的瓶颈结论。

## attempt 2 完成：当前控制调用与短输出停顿的观测

2026-11-02：上述 attempt 已结束，不再是 RUNNING。1000 条计划请求均实际
到达、提交并成功终结；launcher、service、replay、watchdog 均正常退出。
原生 GPU 上下文释放和服务资源域移除已确认，工作缓存为空。五个原生进程
的观察文件均正常关闭，无截断或孤立区间；1000/1000 个首末 token 窗口
均有完整观察。此前失败 attempt 单独保留，不参与这次统计。

小型结果位于 `paper_results/ieee_tc/p2_backend/d1016_current_native_prefix/completion/`。
`manifest.json` 绑定十五份结果/分析入口的 SHA；原始大日志不入 Git。
原生 adapter 绑定、实际 token 数和计时重算全部通过，但这不替代独立数值
正确性，`n_correct=null`、`ranking_eligible=false` 继续保留。

| 本次原序 1000 请求诊断 | 观测值 |
|---|---:|
| TTFT 达标 | 967/1000 |
| TPOT 达标 | 939/1000 |
| 联合达标 | 913/1000（91.3%） |
| 仅 TTFT / 仅 TPOT / 二者违约 | 26 / 54 / 7 |
| 平均 / P95 TTFT | 702.285 / 1798.746 ms |
| 平均 / P95 TPOT | 18.849 / 38.632 ms |
| 平均 / P95 E2E | 2788.716 / 5684.214 ms |
| 物理 GPU 生命周期 | 2520.674594 GPU-s |
| GPU-s/offered request | 2.520675 |

不将这次带观察器的前缀 91.3% 与 D1000 无该观察器的完整 92.25% 作性能
胜负比较；不线性外推 4000 请求成本，不宣布 G1/G2 达标。共同 D648 阈值、
source42/W2 map、生成合同不变；无 CI 或独立留出主张。

### 首末 token 窗口分解

下表是各组逐请求窗口交集的均值（ms），不是全服务独占时间。client 是
控制/输入处理区间，已从 input 中剔除，不能再次把嵌套区间相加。

| TPOT 违约组 | 请求数 | 窗口 | client | input 非 client | engine step | 其他 |
|---|---:|---:|---:|---:|---:|---:|
| 全部输出长度 | 61 | 646.245 | 349.998 | 11.811 | 267.992 | 16.443 |
| 2–32 token | 57 | 534.160 | 285.289 | 8.029 | 228.599 | 12.243 |

61 条 TPOT 违约中，57 条输出不超过 32 token，59 条不超过 64 token。
全部违约组的 client 占窗口均值约 54.16%，是大于 engine step 的观测项。
这回答了本诊断登记的问题：当前控制调用仍是短输出窗口的重要组成，而非
仅引用 D863 旧版本占比。不能推断删掉控制时间便能取得对应加速；host span
不是 kernel 时间，可能与异步 GPU 工作重叠，其他项也不是推定的 GPU 空闲。

对 59 条不超过 64 token 的 TPOT 违约，进一步按实际控制调用分解：

| 控制操作 | 有交叠的请求数 | 每组请求平均交叠 ms | 占本组 client 交叠 |
|---|---:|---:|---:|
| demand_load_and_acquire | 38 | 128.618 | 42.05% |
| source_snapshot | 19 | 58.924 | 19.26% |
| ieee_prepare_host | 21 | 45.954 | 15.02% |
| ieee_pending_gpu_reference | 18 | 35.330 | 11.55% |
| routing_source_snapshot | 12 | 16.710 | 5.46% |
| request_source_snapshot | 24 | 6.270 | 2.05% |

表中请求集合可以重叠，分母固定为本组全部 59 条；剩余操作在完整 CSV 中
保留，不把这六行重命名为全部开销。进入控制操作时仍有推理工作的主进程
中，按需加载、完整/路由来源快照、HOST 准备和 pending 引用的线程 CPU 时间
均接近相应墙钟时间。这支持继续检查事务内 CPU 工作，尚未定位某一子函数
或测量 GIL 占用，也不能把长短输出组的不同工作量忽略后直接比较绝对均值。

### 交付、安全与分析变更

远端日志在 GPU 测量结束后才取回。146 次实际请求交付均为已发布工件，
客户端与服务端 transfer ID、内容 SHA、字节和读/发记录匹配，线上字节共
338845661 B，无请求时打包或临时归档。现有 3B 服务未重启/停止；没有新增
远端 CPU 监控，因此不声称排除了所有远端 CPU 外部干扰。旧的分析阶段
`delivery_audit_status=pending` 不追改，最终闭环见 `delivery_closure.json`。

1194 个资源样本，服务峰值 21798617088 B；high/max/OOM/OOM-kill/swap
均为零。离线分析和整理分别在既定 CPU 辅助域内完成，峰值 2014310400 B
与 1576501248 B，相关事件及 swap 为零；没有与 GPU 测量并发。147 项旧
结果及两个用户 dirty 文件 SHA 均不变。整理后磁盘可用 161246617600 B
（约 150.17 GiB），下一重型任务前须重新核验，不能忽略空间已接近启动线。

本次新增的分析文件只有原始 attempt2 目录的 `analyze.py` 和 `curate.py`
以及本结果目录中的快照/CSV/JSON。分析分别复用 D1016 首次预备入口和
D723 的合同、共同 SLO 与原生窗口方法，D863 的 client 区间交集方法，以及
D703 的客户端/服务端交付关联检查；未改统计公式。生产策略、观察器、vLLM
实现和 IEEE 九公式本次没有修改。依照 system-profile 保留上述完整变化说明，
依照 analyze-results 区分观察、解释与未证实的下一步；目标前不生成图形。

### 决策与下一主线

**本次诊断问题已回答，关闭这个 profiling 执行键，不追加相同前缀。** 当前
尚未接受新的性能候选，D1000 仍是保留的完整 3B W2 开发参考。

已回查 D860/D862、D864–D869、D884–D887 及 D998/D1014：整图 C++、
同事务物理图复用、staged 联合图投影和只读 deferral 均已有实际记录；不能
仅凭当前控制占比大就重新跑这些方向。源码中现有 registered 激活重检查、
staged 联合图投影与私有 HOST 构造仍须保留，不再当作尚未实现的优化。
当前 `routing_source_snapshot` 仍包含完整 HOST 图清点；这一事实本身不
证明可安全删除它。下一步只围绕同一事务的状态/存储依赖继续核对：哪些
消费者确实需要全图，哪些检查在没有状态变化的区间内重复；发现明确新依赖
才登记一个候选。不得用陈旧 snapshot、无依据缓存或延后同步检查替代确认。

之后仍须回到完整 4000 请求及共同指标评价，而不是无限展开组件微优化。
已完成 HydraServe 隔离适配不重开、不操作共享 k3s。最终同版本两模型
W0/W1/W2、G1/G2、baseline/dLoRA 剩余点、A1–A5、S1–S13、独立数值正确性
范围、审稿文档/图表与最终 V2 集成均继续开放；本次诊断不替代它们。
