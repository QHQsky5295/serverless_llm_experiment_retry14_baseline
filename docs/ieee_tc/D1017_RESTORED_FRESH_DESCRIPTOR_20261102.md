# D1017：恢复启动路径遗漏的已保留 HOST 描述读取

2026-11-02，接续已完成的 D1016。上一轮为实质进展：当前 1000 请求原生
窗口已经完整测量和备份，不追加同前缀 profiling。主线仍为 3B PrimeLoRA
在已完成 baseline 适配/观测后的优化；完整范围不缩减。

## 新证据，而不是重复一个已关闭优化

逐请求检查实际 GPU 引用回执，得到：

| 来源 | 完整请求数 | Python 描述读取 | D747 融合描述读取 | 缺字段 |
|---|---:|---:|---:|---:|
| D1000 完整 W2 | 4000 | 4000 | 0 | 0 |
| D1016 attempt2 诊断 | 1000 | 1000 | 0 | 0 |

来源、字段、SHA 和提取身份见 `paper_results/ieee_tc/p2_backend/`
`d1017_restore_descriptor/entry.json`。计数只取每请求实际获取回执，不将
嵌套重复出现的 metadata 当作独立调用。最初误对 main_outcome 使用结果文件
的请求前缀得到零行；该探查未作为结论，改用相邻完整结果后人数精确匹配。

D747/D748 的融合实现逐 view 读取当前 Tensor/storage/pinning，减少接口
封装，不缓存状态。D749 完整 W0 从 D742 的 3727/4000 提高到 3787/4000，
资源从 14868.152527 降到 14847.164967 GPU-s；当时明确保留为后续父版。
D886 仍明确使用 D747 库，且旧 runner 设置其路径与 SHA。

当前 D1000 runner 没有这两个导出项；至少 D934 从 D886 配置转入新 runner
时已缺失（当前文件 SHA `86eb33ab6f59d95c9528149c0ce65d291626dd66c0038e52a8eafe76fe7662b9`）。
已查 D88x–D1016 记录，没有找到撤回这个逐 view 实现的决定。默认 loader
确实在未配置时选 Python，实际回执也证实了这一点。因此这是**已保留实现
未被当前启动配置启用**，不是证明 D747 算法失败后再试一次。D866 整图模式
与它不同，仍不启用；也不顺带恢复已关闭的 ISA-L 或其他参数。

## 唯一假设与公式边界

假设：当前同步控制仍消耗显著 CPU 时间，恢复已验证的实时描述读取可以
减少控制事务开销，并改善完整 W2 的共同 SLO。D1016 的窗口占比支持检查
这条路径，但不能预测改善幅度；旧 D749 的收益也不能移作当前结果。

九公式、tier 确认、实时 pinning、alias/storage 容量、allocator/workspace、
准入和所有权不变。没有删除检查、跨调用缓存、兜底回退或人为延迟；不改
到达预取、扩缩容、GPU 数、token 预算、热度、阈值或负载。库失败继续显式
报错。当前源码不改，复用 SHA 已核对的 D747 195688 B 库和 D1000 runner。

本轮联网复核 [vLLM 0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)
的共享 Tensor 语义、[PyTorch 2.13 pinning](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
的当前 storage 查询，以及 [EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的输入/执行边界。这些支持保留动态检查和减少接口调用的实现边界，不提供
本机加速百分比。vLLM 技能的通用性能数字/参数不替代本项目协议。

## 验证、执行与终止条件

直接复用 D747 的实际 CUDA 九状态、四种图及 D748 的原生构造/clone 资格。
本次只复核当前 loader 的九项既有配置/图接口测试，以及实际 ABI 能载入
同一库并立即读到同 Tensor 的 storage 变化。没有改 C++ 或生产算法，不
重建库、不新增模型 smoke、不重复组件性能微测。

通过后只执行一次完整 4000 请求 W2：新键
`d1017_20261102/d1017_3b_restored_fresh_descriptor_w2`。沿用 D1000 YAML，
由现有 runner 按新键覆盖 output/NVMe/HOST 三个路径；不创建新配置副本、
工件或 trace。没有 Python 栈/host-spans profiling，解压保持 stdlib。

```bash
env -u FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER \
  -u FAASLORA_D1000_PREFIX_COUNT -u FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT \
  -u FAASLORA_D1000_NATIVE_PROFILE \
  PYTHONPATH=/home/qhq/serverless_llm_experiment_retry14_baseline \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_D1000_RUN_ID=d1017_20261102 \
  FAASLORA_D1000_RUN_TAG=d1017_3b_restored_fresh_descriptor_w2 \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261101_d1000_3b_route_confirmed_arrival_prefetch_w1.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

实际 worker 回执必须确认 `fresh_native_tensor_descriptor_v1` 和上述 SHA。
仅看到启动环境不算生效。完整结果按冻结 D648、V1/单轮 V2 比较 D1000
3690/4000、8008.311805 GPU-s；不以 D1016 前缀作性能对照。保留所有
正负结果与旧运行，不追加同键择优。数值证明缺口 `n_correct=null` 不补造。
若没有完整改善，记录当前组合的实际边界，不继续调此库/GC/整图实现。

启动前核验原 150/100 GiB 磁盘、102 GiB 空启动内存、原 72/80 GiB 服务
与 3/4 GiB 辅助约束、GPU idle、健康的既有远端。测量期间不压缩/构建/
测试/push/维护远端。旧 147 项、用户 manifest 和 EXECUTION_STATUS 零覆盖。
共享 k3s 不动，HydraServe 已有隔离适配继续复用。原远端服务不重启或停止。

PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

当前登记尚无 D1017 性能结果。system-profile 变更清单：只新增本文、entry.json
及原始验证记录；生产代码、观察器和库均未改。后续按 run-experiment 复用
已受限长任务入口，github-sync 在测量前备份。目标前不绘图。

## 入口验证已完成

当前 loader 的 `NativeDescriptorSelection` / `ProductionDescriptorPath` 共
9 项既有测试全部通过；实际 D747 库成功在 torch 2.13.0+cu130 / CPython 3.12
载入。对同一个 CPU Tensor 的 storage 从 64 B 增至 256 B，紧接着读取可见
新容量及当前指针/pinning，CUDA 未初始化。这只验证当前 ABI 和实时读取，
实际 GPU 九状态仍复用 D747，不声称又测了 GPU 性能。

受限验证资源峰值 616013824 B，high/max/OOM/OOM-kill/swap 均为 0；验证
任务已退出。原始来源与 SHA、资源身份保存于同目录 `validation.json`。
没有生产源修改、额外 profiling 或新资格重跑。下一步为备份后一次完整 W2；
实际生效以本次 worker 回执为准，结果未出前不作收益判断。

未完成的最终同版本 W0/W1/W2、G1/G2、baseline/dLoRA 剩余点、A1–A5、
S1–S13、正确性范围、审稿文档/图表和最终 V2 集成全部保留，不以此次恢复
启动配置代替完成系统优化。

## 实际启动交接（进行中，不是最终结果）

2026-11-02 18:35:19 Asia/Shanghai，已按上述唯一命令启动完整 W2。
启动前提交 `0698da0260b856402ad105d674cb497e8c9f4582` 已分别推送并
由 `ls-remote` 确认：`origin/retry14_continuous_queue_v2`、
`faaslora_origin/backup/d1017-restored-fresh-descriptor-20261102`。
后者为恢复点备份，不冒称完成分叉目标分支的最终集成；没有 force push。
Git 中仅新增本文及 entry/validation 两份 JSON；147 个旧投稿文件验证不变，
用户 manifest 和 EXECUTION_STATUS 的 SHA 仍与入口相同，未 stage。

- tmux：`tc-d1017-3b-restored-fresh-descriptor-w2`。
- 原始根：`/home/qhq/serverless_llm_experiment/results/ieee_tc/`
  `p2_backend_qualification/d1017_20261102/d1017_3b_restored_fresh_descriptor_w2`。
- 服务 scope：`primelora-tc-svc-f9a253c12b254cabaa5c291dda3aeeea.scope`，
  invocation `aa12de1938f046baa925321593db6e4b`。
- 辅助 scope：`primelora-tc-aux-2aa5a863094d4fa2bebbc23fb1c9d265.scope`。
- `launch.launch/exec_receipt.json` 已实际放行，外置 publisher 计划为 W2/4000，
  source SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`，
  view SHA `5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174`。
- 共同部署通知 monotonic `3122969.628762803`，业务开始 `3123029.628762803`，
  最后计划到达 `3126993.537665925`；60 秒准备未更改。
- 早期检查已有 14 条成功终态，原生 token 合同匹配；这是进度而非最终分母。
  sample 110 的服务内存约 7.65 GiB，主机可用约 96.27 GiB，磁盘约
  150.09 GiB，high/max/OOM/swap 为 0，无外部计算 GPU 进程和保护中止。
- 实际 GPU worker PID 1595087 的 `/proc/maps` 已加载指定 D747 库，
  进程环境中库路径/SHA 匹配。这证明实际载入，不替代结束后逐请求 kind/SHA
  的完整核验；不为取回中间回执而新增控制调用或 profiler。
- 既有 D998 辅助 scope 名称仍显示 active，但 `TasksCurrent=0` 且
  `cgroup.procs` 为空；没有据名称误杀，也没有另一个重型实验并行。

远端 health 在启动前确认同一已发布协议与 clock ID
`remote-process-monotonic:b7a04b62f38c4ae6adabb4da5c0fd8dc`；未重启服务。
本轮不新增远端 CPU profiler，因此结束后仍须明确该外部干扰观测范围，不能
据 health 正常声称已完全排除远端 CPU 争用。

下一轮继续监测此既有句柄，直到真实退出、4000 终态与资源释放后再做统计。
不因工具超时或新对话重复启动，不在测量期间测试/压缩/哈希大文件/push。
结束后复用 D1000/D1016 分析入口，核对 descriptor 实际回执、生成/绑定、
SLO、物理 GPU 生命周期和远端关联；仍不能用 `slo@5000ms` 或 tokenproxy CE
判定达标。运行中没有最终 SLO 或 GPU-s 结论，目标保持 active。

### 继续监测与已核对的结束分析入口

2026-11-02 18:42:02 +08:00 再次确认同一 tmux 和服务 scope 实际存活。
watchdog sample 395 年龄 0.48 秒：233 条终态均成功且原生合同匹配，实际
GPU PID 1595087 归属本次服务；high/max/OOM/swap 为 0，主机可用约
95.93 GiB、服务峰值约 12.28 GiB、磁盘约 148.22 GiB。150 GiB 是**新启动**
门槛，本轮启动时已满足；运行中按原 100 GiB 底线保护，不人为中止后重跑，
也不在测量期间清理或压缩。上一 goal turn 为 PROGRESS（备份和实际启动），
本次监测属于针对具体存活句柄的 verified wait，不是重复实验。

本轮已只读核对现有 `scripts/ieee_tc/audit_d703_3b_full.py` 的实际接口：

1. `--raw RUN --expected-count 4000 --reevaluate-common-slo OUT/full4000.json`
   完整核对 native timeline/输入/生成和释放证据，使用冻结 D648。
2. `--raw RUN --metric-audit OUT/full4000.json --diagnose-common-slo OUT/stages`
   输出全部请求的同口径阶段表，不选择慢请求子集替代总样本。
3. `--reference-stage-audit /home/qhq/serverless_llm_experiment/results/ieee_tc/profile_output/d1015_20261102/d1000_stages`
   配合 `--stage-audit OUT/stages --compare-stage-audits OUT/paired_d1000`。
   已从 D1015 的 sealed `diagnosis.json` 确认这个真实来源，而不是猜测
   D1000 curated 目录下存在 `stages`。来源 diagnosis SHA 为
   `1dfa91c791546f080342a9c3fbc5f0ebacf7eb828ace7329f3945ae3adcb903d`，
   requests CSV SHA 为
   `1ba1b2601d9b24003ae1f225b0b55dbc98f60e59a0c374c455c85a7e359817cc`。
   结束后由原分析器再次核验，不重复投影已有完整阶段表，更不新增 GPU 重跑。
4. descriptor kind/library SHA 从本次完整请求结果逐条投影，4000 条必须全部
   有相同实际选择；该检查不是只看环境或 `/proc/maps`。
5. 远端关联复用 D1016 `curate.py` 的日志快照/ID筛选及 D703 原有字节/SHA/
   no-pack 检查。其通用 `audit_closed_outcome` 还硬引用旧次的 remote_stop/
   monitor 文件，不能为通过它虚构本次 stop/monitor；分别记录本次确有的
   客户端/服务端关联证据和缺失的外部 CPU 监控范围，沿用 D1016 的处理。

`RUN` 为上节真实根目录，`OUT` 使用同一 curated 目录的全新 `completion`
子目录。以上为复用入口核对，**尚未运行结果分析**。实际分析只在本次真实
退出与资源释放后，于原 3/4 GiB、swap0、CPU 2/3/26/27 的辅助域执行。
不修改现有 analyzer、生产算法或正在运行的配置；不用不完整数据提前接受
候选。D1000 的旧 `adapter_replica_mismatch` 字段解释已被同名
`D1000_METRIC_LABEL_CORRECTION_20261101.md` 修正：这里只能称“所选副本
dispatch 前非 GPU-ready”，不能无替代副本证据就称路由失配。

18:43 左右的继续观察：service 日志已记录 2 次自然 scale-up、3 个 runtime，
最近日志完成 348/4000、失败 0。watchdog sample 496 的实际 compute PID 为
1595087/1612104/1612177，分别位于 GPU 0/1/2，均归属本次服务；逐进程
`/proc/maps` 均映射同一 D747 库。此证据覆盖扩容进程的实际载入，不把它
扩大为全部请求 receipt 已通过。该采样年龄 0.11 秒，内存峰值约 18.59 GiB、
主机可用约 90.07 GiB、磁盘约 147.72 GiB，high/max/OOM 为 0，无保护
告警或中止。实验仍在同一 tmux/scope 中继续，尚未生成最终评价。

约 18:46，服务日志达到 **500/4000 完成、500 成功、0 失败**，累计两次
扩容和一次缩容，当前两个 runtime。watchdog sample 671 年龄 0.89 秒，
high/max/OOM/swap 为 0，无外部 compute 进程或保护告警，磁盘约
147.00 GiB。这是同一完整回放的首个 500 请求进度点，不是独立实验或正式
SLO 结果；不从 live 的 5000ms 单阈值显示推导共同联合达成率。继续等待
原执行键到 4000 及真实释放，当前不运行分析/测试/压缩/push。

18:55:47 +08:00，连续监测同一运行到 1116 条终态：1116 成功、1116 原生
合同匹配、失败 0。实际服务仍 active、436 tasks，watchdog sample 1209
年龄 0.58 秒；服务峰值约 21.31 GiB、主机可用约 89.42 GiB、磁盘约
144.15 GiB，high/max/OOM/swap 为 0，无保护告警。最近日志累计三次扩容、
一次缩容、三个 runtime。约第 1000 请求附近原 trace 的突发队列曾到 27，
随后回落；没有因此改到达、缩短负载、重启或改配置。该 1116 是进行中进度，
不是独立前缀实验，也没有产生最终 SLO/GPU-s 结论。

本次 goal turn 为 verified wait：多次核对同一实际存活服务、tmux 和新鲜
watchdog，而非依据旧状态文件推定运行。目标/PLAN 已核对，PLAN SHA 未变，
前文完整范围和下一步不变。测量中仅追加此交接，不测试、压缩、推送或新开
GPU 作业；尚未完成的主比较、A1–A5、S1–S13、正确性和交付不删减。

19:08:21 +08:00，完整回放越过一半：2023 条终态均 success/native-contract
匹配，失败 0；同一服务仍 active、337 tasks。累计四次 scale-up、三次
scale-down，当前两个 runtime。watchdog sample 1953 年龄 0.64 秒，
服务峰值约 24.88 GiB、主机可用约 92.12 GiB、磁盘约 140.62 GiB，
high/max/OOM/swap 为 0，无保护告警。仍在原 4000 请求到达窗口内，未到
最终结果节点；没有改参数、替换 trace、并发维护或重新启动。本 goal turn
继续属于 verified wait，完整目标不缩减、不标 complete/blocked。

19:14:57 +08:00：2546 条终态均成功、原生合同匹配，失败 0。同一 tmux/
服务持续存活；最近日志累计五次扩容、四次缩容，当前两个 runtime。
watchdog sample 2343 年龄 0.88 秒，服务峰值约 26.84 GiB、主机可用
约 92.01 GiB、磁盘约 138.18 GiB，high/max/OOM/swap 为 0，无保护
告警。当前依旧为 verified wait；未分析不完整结果、未改配置、未重跑、
未开展并行维护，完整 4000 请求和释放后的评价仍待完成。

19:22:05 +08:00：3056 条终态均成功、原生合同匹配，失败 0。服务实际
active、337 tasks，最近日志仍为五次扩容/四次缩容、两个 runtime。
watchdog sample 2765 年龄 0.64 秒，服务峰值约 26.84 GiB、主机可用
约 91.85 GiB、磁盘约 135.83 GiB，high/max/OOM/swap 为 0，无保护
告警。此 turn 为 verified wait；另完整重读冻结 V1、单轮 V2 和 D648
数值协议以准备结束后的既有分析，没有执行结果统计或更改指标。相同任务
的 PLAN 在当前上下文已完整读取，SHA 复核无变化；保持原定 4000 请求、
远端交付和全部后续范围，不将四分之三进度冒作实验完成。

## 最终结果：完整运行已结束，服务质量改善但尚未达标

2026-11-02 19:43 后最终 launch 回执已生成：4000/4000 请求成功、失败 0；
service/watchdog/replay returncode 均为 0，service_path_removed 和
native_gpu_context_release_confirmed 均为 true。实际 GPU compute 进程为空；
本次 tmux 已退出，不得重复启动。当前 turn 从 verified wait 转为 PROGRESS：
完成整轮、完整指标/阶段配对、机制与真实远端关联检查。目标仍 active。

| 同一 3B W2 合同的单轮观测 | D1000 原完整参照 | D1017 恢复描述读取 |
|---|---:|---:|
| 联合 SLO | 3690/4000，92.25% | 3738/4000，93.45% |
| TTFT 达标请求 | 3827 | 3831 |
| TPOT 达标请求 | 3846 | 3894 |
| 平均 TTFT (ms) | 751.901293 | 709.541028 |
| P95 TTFT (ms，Type-1) | 2112.674413 | 2131.337589 |
| 平均 TPOT (ms) | 17.359275 | 16.413181 |
| 平均 E2E (ms) | 2653.940846 | 2529.352222 |
| 完整生命周期 GPU-s | 8008.311805 | 8084.812476 |
| GPU-s/offered request | 2.002078 | 2.021203 |

联合达标净增 48（+1.20 个百分点），平均 TTFT/TPOT/E2E 分别下降
5.63%/5.45%/4.69%；但 P95 TTFT 增加 18.663176 ms，GPU-s 增加
76.500671（0.9553%）。这是质量—资源权衡，不是全面改进或最终最优。
距离冻结 3800 个联合达标仍差 62，不宣告 G1/G2 完成。`n_correct=null`
继续保留：原生数量、请求/adapter/内容绑定通过，不补造独立数值正确性。
来源为已知源 trace 的开发观测，无 CI、显著性或独立留出结论。

### 实际机制与阶段证据

- 全部 4000 个实际获取回执均为 `fresh_native_tensor_descriptor_v1`，
  library SHA 均为已登记 D747；不是仅据环境变量或映射表推断生效。
- TTFT 六段分解与原生 TTFT/TPOT/E2E 重算最大误差均为 0 ms。
  同一输入逐请求配对：195 个请求由联合不达标变达标，147 个反向变化，
  净增 48；不隐藏反向变化。
- 平均到达至准入、准入至可执行、可执行至引擎、引擎入口至原生排队分别
  减少 8.293、9.695、12.677、9.590 ms；这些是闭环单轮观测，不把其和
  全部解释为此描述函数的独立因果收益。
- 现有剩余违约：156 个仅 TTFT、93 个仅 TPOT、13 个二者；相比 D1000
  的 156/137/17，改善主要体现为 TPOT。106 个 TPOT 违约中 103 个属于
  1–64 输出 token，169 个 TTFT 违约中 114 个 dispatch 前为 Remote。
  这些是条件诊断，不是“另外一副本 GPU-ready”的路由错配证明，也不把
  控制/准备时间全部视为可消除时间。
- 同一真实远端 505 次传输、1172100920 线上字节，全部完成客户端/服务端
  transfer ID、adapter、SHA、bytes 和 no-pack 关联；未重启、停止或改配
  远端。没有新增远端 CPU monitor，仍不声称完全排除远端外部 CPU 干扰。
- 6 个 allocation 全部释放；5 次扩容、4 次自然缩容。GPU 时间增加主要
  来自共同到达窗口（+79.458146 GPU-s），不是将清理时间漏掉或加上。

### 收尾与分析错误均保留

4059 个资源采样完整，含最终 GPU census 和 service-domain-gone；服务
峰值 29844418560 B，high/max/OOM/OOM-kill/swap 全为 0，无外部 compute
进程或 escaped-owned 进程。HOST/NVMe 本次工作缓存由原收尾清空，147 个
受保护历史文件与两个用户 dirty 文件 SHA 不变。没有删除原始或失败证据。

复用 D703 的指标、六阶段和配对入口全部执行一次通过。首个分析 wrapper
随后把 receipt 的实际 `sha256` 误写成 `library_sha256`，仅该投影失败；
保留失败脚本/资源回执，后续使用实际字段核对，不重跑已完成指标和性能。
另一个旧闭合分析仅允许 `ieee_all_low_through_cooldown`，遇到本轮四个
`ieee_forecast_capacity_release` 拒绝。实际四段时序均有序，当前冻结
D1000 YAML 已启用预测释放，原 autoscaler 和 D989 历史均有此原因。
新 curation 只显式承认这两个既有原因，保留全部字节/SHA、计数、时序、
资源与保护检查，不修改生产代码、原始事件或 D703 helper。

最终 curation 通过；所有分析 scope（0005–0008）已 inactive、无 cgroup。
指标分析资源峰值 2440650752 B，成功 curation 峰值 1718280192 B，受限
于原 3/4 GiB、swap0 和指定辅助 CPU。没有新增模型测试或图片。

完整产物：`paper_results/ieee_tc/p2_backend/d1017_restore_descriptor/completion/`。
manifest SHA `a70ddb8f947f0a7b16e41eb74e7bc6af7c37116e227f66accaf8de2a17423c9e`。
原始结果 SHA `eaa83764ddb9e364c33d8f33734b667a7cf07dca2b640b34ee566812eddcf0b6`；
launch SHA `c12e6550b7c18d96bd719fc9dbd684acb580f766e12cf2033d669821b66e8685`。

### 决定与下一主线

保留 D1017 为更高联合达成的开发候选，同时保留 D1000 的资源/尾延迟参照；
不能将它写为已通过资源排名的最终工作点。恢复逐 view 实现的假设已完成
一次有界评价，不再调此库、GC、整图或追加相同配置重复。下一步先使用本轮
已保存的违约、阶段、当前控制证据识别一个尚未关闭的瓶颈；每个新候选仍
须历史、官方实现和九公式核对。避免重复已经关闭的线程/解压/路由重试等
方向，没有新证据时不新增前缀 profiling。

7B/3B 最终同版本 W0/W1/W2、G1/G2、baseline/dLoRA 剩余点、A1–A5、
S1–S13、正确性范围、审稿文档/图表及最终 V2 集成全部仍未完成，不以
这项局部改善删减范围，也不重启已完成的 HydraServe 隔离适配或共享 k3s。
最后可用磁盘 160264855552 B，略低于 150 GiB 新重型启动门槛；下次 GPU
实验前按已有工具安全压缩闭合证据或核查可重建缓存，不删唯一工件，不在
测量期间维护。此文件承接当前状态；用户 dirty 的 EXECUTION_STATUS 保持
原字节、未 stage，不能用其中旧 D998 交接重新启动历史实验。
