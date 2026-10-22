# D734：当前 3B 请求接收等待的有界函数观察

## 问题与停止边界

D733 已复用 D732 全部请求，发现文件观察只与累计提交→接收等待的
18.989% 重叠；182 条 TTFT 违约中仅23条在这段等待中与文件观察重叠。
这不是因果扣减，但足以说明尚不能直接将整个文件 owner 异步化当作首选。
本次只问：当前3B controller中，哪些同步调用与接收/返回等待并存？

复用原1000请求前缀一次，不生成负载、不改来源/到达/生成/工件。不增加
Full性能重复，不调控制参数，不改九个公式，不重做原生HOST profiling。
历史D456/D603是旧7B；D684只有native spans，不能替代当前controller观察。
若已有采样分辨率仍不能定位，报告限制，不无限追加局部诊断。

## 方法与解释限制

- 复用 `profile_d328_controller_python.sh` 和既有 `python_frames_v1`；
  每2秒记录Python线程frame与现有source RPC边界，不新增插桩。
- 保留当前D732全部系统配置，只换输出、NVMe工作缓存和HOST缓存的独占路径。
- 禁用额外native core/step/HOST清点观察；外置原到达回放和安全监控不变。
- 采样受GIL和调度偏差影响；frame频次不是无偏CPU时间，也不证明独占因果。
  记录实际采样间隔和observer开销，不从请求时延中扣除。
- 1000前缀不是4000主结果；带观察运行不参与G1/G2排名，也不与D732宣称性能胜负。
  独立adapter数值正确性未知仍为未知，不由token计数或HTTP成功推定。

本次复核[Python asyncio官方线程边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
与[vLLM官方CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)，只采用
按实际组件区分同步工作、异步等待和GIL竞争的方法，不移用其加速比。

## 启动、保存与恢复

使用既有run-experiment入口及tmux；服务72/80GiB、swap2GiB、CPU4–23/28–47；
外置回放/监控3/4GiB、swap0、CPU2/3/26/27。既有磁盘150/100GiB和内存
余量公式不变。每次只有一个重型任务，不在运行中备份、扫描、压缩或改远端。
真实远端复用D80已发布只读gzip；请求不打包，不复制工件。保留真实传输。

运行结束沿用实际GPU释放后按InvocationID关闭本轮远端服务；旧数据不覆盖。
配置、启动脚本快照、检查结果与来源放入
`paper_results/ieee_tc/p2_backend/d734_3b_controller_prefix1000/prelaunch/`；原始日志
在 `results/ieee_tc/p2_backend_qualification/d734_20261022/`。
启动前显式路径提交备份，V2分歧不force，不stage用户manifest。

## profiling变更清单

| 文件 | 本次改动 |
|---|---|
| scripts/ieee_tc/profile_d328_controller_python.sh | 只登记D734使用已有stack observer |
| scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh | 登记1000前缀、3B输入/远端、已有observer入口 |
| configs/ieee_tc/20261022_d734_3b_controller_prefix1000.yaml | D732配置，仅三个独占工作路径变化 |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 两项针对性合同检查 |
| raw/d734_20261022/ | 复用原启动/关闭/health脚本；保存检查与运行回执 |
| 本文与EXECUTION_STATUS | 问题、边界、状态与下一步 |

没有修改生产推理/路由/规划代码。该观察仅由D734诊断入口启用，后续普通
性能入口不启用；保留脚本作来源追溯，无须删除历史证据来“清理插桩”。

## 共享集群与完整主线

用户已确认旧k3s不是本项目；已有HydraServe独立rootless Podman适配及7B
W0/W1/W2证据继续复用，不访问旧集群、不索取kubeconfig、不重做基础适配。
3B优化尚未达到95%，后续W1/W2、baseline/dLoRA、7B最终版本一致性、
A1–A5、S1–S13、CE/审稿文档和最终图表全部保留；当前不画图。

## 终态结果：一次诊断已经完成，不重复运行

实际执行源码为 `94bacfcf85775572bb35336f6560612d395319b4`。1,000 条原前缀
请求全部正常终态，原生数量/绑定/计时与远端内容关联检查通过；独立 adapter
数值正确性仍未确认，`n_correct=null`、`ranking_eligible=false`。终端旧口径
的 100% 只是 runtime TTFT-only，不能替代 D648 共同联合 SLO。

| 指标 | D734，1,000 请求、带观察器 |
|---|---:|
| TTFT 达标 | 974/1,000 |
| TPOT 达标 | 962/1,000 |
| 联合达标 | 939/1,000（93.9%） |
| TTFT-only / TPOT-only / 两项均违约 | 23 / 35 / 3 |
| 平均 / P95 TTFT | 676.314 / 1,649.169 ms |
| 平均 / P95 TPOT | 17.539 / 33.395 ms |
| 平均 / P95 E2E | 2,717.720 / 5,831.559 ms |
| 实际持有 GPU 时间 | 3,406.608804 GPU-s |
| 实际分配 / 自然 scale-up / scale-down | 4 / 3 / 0 |

这不是新的完整 W0 点，不能与 D732 的4,000条结果计算改善率，也不能线性
外推 GPU 时间。当前未插桩的完整3B结果仍为 D732：联合92.500%，
U=15,903.076489 GPU-s；尚未达到共同95%，不进入它的W1/W2验收。

### 请求阶段证据

下表是同一运行的条件性诊断，单位 ms；不是跨系统因果对照。

| 阶段均值 | 全部1,000 | TTFT违约26条 | TPOT违约38条 |
|---|---:|---:|---:|
| 计划到达→admission | 303.925 | 1,074.613 | 621.104 |
| 提交→本机接收 | 38.501 | 99.840 | 119.724 |
| 本机接收→admission | 262.257 | 972.019 | 499.910 |
| admission→GPU引用取得 | 134.762 | 900.154 | 149.104 |
| backend-executable→引擎入口 | 67.765 | 187.660 | 75.392 |
| 引擎入口→原生队列 | 56.367 | 119.784 | 187.253 |
| 原生队列 | 3.083 | 1.056 | 15.183 |
| 原生prefill | 110.412 | 151.173 | 114.546 |

提交→接收及接收→admission是到达→admission的细分，不与该父项重复相加；
计划到达→提交余量仍计入用户时延。违约请求的主要等待不只发生在网络接收
之前；必须同时保留路由/引用取得/原生执行证据，不能把全部违约归给一个函数。

### 函数观察与控制消息

观察覆盖1个controller、4个frontend、1个planner和4个native core。
controller全程578个样本，业务窗口536个；业务窗口缺失/截断均0，全程有
3个截断样本，不能声称所有采样完整。该进程最后样本累计CPU 386.941秒，
observer截至倒数边界累计CPU 0.409秒（不含末样本）；这不是每函数CPU分摊。
最大采样调度滞后0.446秒。采样偏差和观察开销不从请求时延扣除。

| 业务窗口中的函数 | 样本数/536 | 含义 |
|---|---:|---|
| `_ieee_publish_planning_sources` | 39 | inclusive，包含下列校验，不可相加 |
| `NativeSourceSnapshot._footprints` | 30 | 最近的项目函数；完整HOST分配/别名图校验 |
| `_decode_native_rpc_frame` | 18 | 最近的项目函数；收到的消息解码 |
| `execution_bundle_copy` | 14 | 最近的项目函数；冻结结果的私有执行副本 |
| `preparation_snapshot` | 8 | 最近的项目函数；共享文件状态观察 |

这些是frame出现次数，不是独占时间、延迟贡献比例或可获得的加速比。
`main`作为最近项目函数的387个样本包含事件循环等待，不能把它称为CPU热点。

477条已插桩的routing-source RPC都有完整成功链，部分消息为0。其区间：

| 区间 | 平均 ms | P95 ms |
|---|---:|---:|
| frontend发送→native开始 | 58.185 | 237.244 |
| native函数体 | 9.988 | 15.945 |
| native完成→frontend收到 | 9.765 | 31.167 |
| frontend准备返回→controller收到 | 24.986 | 123.478 |
| 整条调用 | 109.192 | 362.641 |

原生函数体线程CPU均值9.251ms。该表只涵盖这477条特定调用，不是全部控制
操作、全部规划调用或全部GPU时间。返回区间含传输、调度和解析，不能称纯IPC。

## 解释与下一唯一候选方向

1. D733已排除“仅文件观察解释全部接收等待”。D734提供了新的具体证据：
   业务controller确实执行完整native图校验；纯规划进程随后又对同一冻结
   图执行 `owned_preparation_inputs → NativeSourceSnapshot.from_native`。
   代码中的第二个消费者不是新的物理观测，因此应研究如何共享**同一次
   验证的不可变结果**，而不是删掉验证或跨epoch缓存物理容量。
2. 历史 `af0b771` / D330引入双消费者发布以减少路由的重复原生查询，其收益
   目的仍成立；不能简单关闭它。D329禁止owner/epoch-only的物理表示缓存；
   D514/D516的identity-only预算候选已失败；D687/D688的HOST局部重写退化
   后撤销。后续不得重复这些已关闭方向。
3. 下一实现假设：在现有独立规划进程中，完成一次完整输入校验后，同时产生
   不可变规划结果及紧凑路由发布视图；controller只做当前slot/engine/GPU、
   原始epoch/capture time和迟到结果检查，然后发布。复用现有worker与取消
   所有权，不增加另一排队服务、任意TTL、轮询频率或控制阈值。
4. 这可能延后路由获得规划测量的时刻，不能宣称严格时序等价或预言收益。
   不能把子进程完成时间改写成新观测时间；取消/过期结果不得发布，实际
   dispatch引用、资源reservation、别名/共享容量及执行重检查全部保留。
   先检查冻结输入/目标/投影等价、损坏图拒绝和迟到/取消测试，再对新版本
   做一次未插桩完整W0。若受影响路径已有原生资格，不追加无问题的短回放。
5. 本轮只完成证据与候选选择，**未实施该候选**。消息解码、结果复制以及
   原生排队仍是明确剩余开销，不能由一次控制器改动保证95%或GPU资源目标。
   不再追加相同的1,000前缀诊断，不扩大为全文件owner重构或baseline调参。

本次重新核对[Python asyncio的阻塞调用边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)、
[vLLM官方CPU/推理进程分离说明](https://vllm.ai/blog/2024-09-05-perf-update)和
[vLLM 0.30.0 core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)。
它们支持把CPU工作和服务调度隔离的设计依据；不证明本候选有效，不借用其
加速数字。九公式及其输入语义、共同阈值、工件/负载/生成和真实传输均不改。

## 收尾、来源与修改范围

服务/回放/watchdog返回码均0，四次物理GPU分配全部释放；本轮3B/7B远端
服务与monitor全部inactive/MainPID0。60次真实传输、139,254,428线上字节，
两端ID/字节/内容SHA关联通过，没有请求中打包。服务峰值20,598,931,456B，
high/max/OOM/swap全0；分析域峰值745,914,368B，同样无上述事件。
145个保护项及用户manifest未变，未删除或压缩任何历史证据。

结果保存在 `paper_results/ieee_tc/p2_backend/d734_3b_controller_prefix1000/completion/`：
原生指标、阶段CSV、stack/caller统计、控制消息区间及文件SHA清单。
本轮分析只复用既有collector/analyzer，新增raw `complete.py`、
`collect_cpu.py`、`curate_callers.py`，其快照随结果保留；另有封存脚本及回执。
没有新增运行时插桩、生产代码或公式修改。普通性能入口不启用D734观察器。
下一步回到上述单一PrimeLoRA候选；共享k3s与已完成HydraServe适配保持不动。
