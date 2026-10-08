# D351：区分 native core 控制工作与生成推进

2026-10-08，7B 优化主线；基于 D350 已封存诊断。不是正式性能点，不绘图。

## 问题、历史与反证条件

D349 相对 D336 的 request-associated 原生加载减少28.71%，但 TPOT
未通过暂定共同阈值的请求由80增至239。短输出的退化与其他请求
scheduled→first 包络重叠；该包络包括 CPU 控制、加载和等待，不能
等同纯 GPU prefill。D328 的旧 source RPC/栈采样不覆盖当前配置下
完整 core 轮转，已有 HOST 清点优化不重复。D307/D314 的固定512预算
已显示资源/TTFT代价；D348/D349不支持继续任意扫容量或软目标。

可证伪假设：成串的控制/加载 utility 占用 native core 线程，延后
已有解码的下一次生成推进。若长间隔主要位于 step 而非输入/utility，
或 utility 占用不足以解释观察到的停顿，则不接受此归因；下一项应
分析迭代内 prefill/decode，不先改队列、取消同步保护或调 cooldown。

官方核查（2026-10-08）：

- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  在输入循环调用 client handler，再执行 engine step；utility 返回
  Future 时 handler 的返回不代表该异步工作已经完成。
- [UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  的 collective RPC 在本进程调用 worker。实际安装源码另存 SHA，
  不能以版本号替代身份。
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  为混合 prefill/decode 干扰提供相关机制背景，不证明本机当前瓶颈，
  不照搬其工作点或收益。

## 单一诊断合同

复用 D293 入口和 D328 观察 wrapper；D351 配置与 D349 相比只更换
结果/NVMe/HOST 路径。C14、软目标、生成、200/24类 profile、初态、
真实远端发布缓存、控制公式、SLO、CPU/内存限制均不改变。
源负载仍4000请求；只提交同顺序前1000请求，标明 diagnostic_prefix_v1，
不能把前1000的资源外推成4000。无新权重/trace/解压池。

`FAASLORA_TC_CORE_TIMING=host_spans_v1` 仅在 resource gate 放行后启用。
必须 nonformal、正的 prefix 长度、实际处于 receipt 指定服务 cgroup。
默认不开启；不启用旧栈采样或强同步数值观察器。九个公式与所有 serving
策略保持不变，计时开销不从观测时延中事后扣除。

## 观察含义与边界

- 同一 core 主线程记录 input_queue、client、step、schedule 的起止
  monotonic ns、thread CPU ns、父子 span、返回/抛出。
- input_queue 含合法 idle wait；`unfinished_before/running_before/
  waiting_before` 用于区分入队前空闲，不能把整个输入 span 都称阻塞。
- utility/worker operation 仅输出白名单常量；不保存 RPC 参数、prompt、
  request ID、token、路径、秘密或异常消息。未知类别统一 other。
- schedule 输出总 token 数、请求数及分配多于1 token的请求数；后者
  不冒称纯 prefill 数，step 主机包络也不冒称 GPU kernel 时间。
- 异步 GPU 可与 utility 重叠；host 占用只证明主线程不能推进新的 step，
  不自动证明 GPU 空闲或每个请求损失同等时间。需与已有请求时间线关联。
- 嵌套 span 不直接相加；同线程 exclusive 时长由父子区间求差。
  `returned` 不叫成功，因为 native utility 自己可能返回失败或 Future。
- 每进程最多200000个 span、64MiB记录；达到任一上限写截断标志，
  不停止原生工作。每行直接写入，普通 shutdown/atexit 写 close；异常
  终止可能缺当前嵌套 span/close，保留未知。无 close 不宣称全程完整。
- 记录观测器自身与写入累计主机时间；它们是观察开销估计，不是完整
  零扰动保证。所有带观察器的结果不进入普通性能排名或正式 G1/G2。

## 验证与执行顺序

1. CPU 测试：默认禁用、非正式/cgroup/prefix gate、调用顺序/结果/异常、
   嵌套关系、类别脱敏、单线程所有权、幂等安装、容量截断、短写失败、
   shutdown、旧诊断与原生 scheduler 回归。
2. 逐字段校验 D349→D351 仅三类路径变更；保存实际安装 core/UniProc/
   scheduler 源码 SHA；复核历史保护清单与用户 manifest。
3. 测试通过后 exact-path checkpoint 到 V2，再按现有磁盘/内存/GPU idle
   gate 启动；不为本次诊断放松150GiB或OOM规则。
4. 一次前1000真实远端回放；服务结束先释放、核验 token/终态/资源、
   封存源结果与观察文件。完整性失败不伪造零值。
5. 复用现有请求投影，检验短请求停顿区间的 input/client/step 时间分配，
   所有请求和未知状态保留。不用条件重叠相关性充当因果净收益。
6. 依据诊断选择一项有机制依据的优化或否定当前假设；最小验证后完整
   4000回放。最多两轮最小验证且必须有新证据，不长期停留于观察器。

准备阶段：CPU最初49项及最终202项通过；最终配置绑定、历史保护和资源准入
通过。准备脚本最初误要求继承的 shutdown 必须在
子类体内定义，检查失败后按实际继承关系修正；日志通过标记的转义检查
也在写出绑定前修正。两次均未启动GPU、未修改实验结果；原测试/失败
准备脚本保留。准备检查点15e66c9已推送并独立核验。
N_correct、正式共同warm/Resident参考与G1/G2仍OPEN；不进入3B/基线。

## 实际结果：一次前1000诊断已完成

运行时间2026-10-08 18:00–18:20 CST；D349配置、原始前1000次序、
真实远端已发布交付缓存保持不变。12个原生core生命周期均有完整关闭记录，
171284个host span无截断、无孤立子span；1000/1000解码窗有完整绑定。
这些是进程生命周期，不是12张GPU或12次独立重复。全部1000请求原生
token合同、adapter内容/来源绑定与终态检查通过；数值正确性仍未由此确立。

| 观测值 | D351诊断结果 |
|---|---:|
| 完成原生合同请求 | 1000/1000 |
| 数值正确数N_correct | 未确立 |
| 平均/P95 TTFT | 1922.098 / 10327.093 ms |
| 平均/P95 TPOT | 46.742 / 116.659 ms |
| 完整观察的物理GPU-s | 3297.097 |
| 详细计时自身累计主机开销 | 48.551 s（含写入39.647 s） |
| 所有进程观察包络时长之和 | 2663.368 s |
| 服务内存峰值 | 21160337408 bytes |
| high/max/OOM/swap与保护告警 | 均0 |

开销约为进程包络之和的1.82%，不是对请求时延扰动的完整上界。
不扣除此值、不与普通Full作加速对比、不外推4000请求资源。
106项新增及既有分析回归通过；分析峰值509108224 bytes，资源事件0。
GPU实际释放、12个物理allocation journal关闭、远端三个所属服务停止，
传输/监控日志下载SHA一致。远端打包次数0；没有重建交付缓存或工件。

### 解码窗的互斥host时间分解

以下是请求条件分组，不是随机处理组。TPOT判断使用同一D170暂定尺子，
并非最终共同warm资格。所有请求保留；同一core区间可服务多个请求，
所以不得把请求列相加当GPU成本。

| 输出长度/TPOT状态 | 请求数 | 平均解码ms | client处理ms | input非client ms | step ms | 其他ms |
|---|---:|---:|---:|---:|---:|---:|
| 2–32，未超暂定阈值 | 204 | 515.375 | 119.584 | 3.154 | 386.721 | 5.915 |
| 2–32，超过暂定阈值 | 108 | 1359.664 | 569.394 | 17.907 | 760.867 | 11.496 |
| 33–256，未超暂定阈值 | 657 | 5836.285 | 1203.723 | 41.563 | 4497.275 | 93.725 |
| 33–256，超过暂定阈值 | 31 | 14356.894 | 5975.525 | 123.760 | 8160.324 | 97.285 |

短输出TPOT失约组的client时间占组均值解码时间41.88%。这说明控制处理
与解码停顿确实交叠，不能把它全部归因为GPU prefill；但不是可消除损失
的估计，也不支持从该组减去另一组来宣称因果提升。

### 控制操作定位

按父input进入时原生请求数非零的work组，聚合不同core的完成调用。
这是host串行开销，不是GPU kernel时间；异步GPU可以同时运行。

| 操作 | 次数 | 平均host ms | 总host s | 线程CPU/host |
|---|---:|---:|---:|---:|
| source_snapshot | 767 | 119.553 | 91.697 | 99.625% |
| pending_gpu_reference | 558 | 131.201 | 73.210 | 99.647% |
| demand_load_and_acquire | 504 | 106.739 | 53.796 | 98.294% |
| prepare_host | 243 | 199.012 | 48.360 | 99.288% |
| routing_source_snapshot | 553 | 70.370 | 38.915 | 99.815% |

work组所有client合计339.386 host-s；idle-at-entry组另有225.704 host-s，
未丢弃。后者仅描述父输入入口状态，不能把其所有子调用称为空闲无影响。
完整operation表保留unknown/other、低耗时类别以及每进程原始Type-1分位数。
不能从各进程P95的平均恢复全体P95。

## 结论与唯一下一项

支持的结论：当前昂贵utility主要占用原生core线程的CPU时间，控制查询和
预算清点是有实际证据的优化对象。尚未证明哪个内部函数消耗了全部时间，
也未证明排除它就能满足G1/G2。step仍占较大比例，不能忽略后续计算瓶颈。

2026-10-08再次核对上述官方core和worker manager，并核对
[PyTorch2.13 is_pinned实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
该查询涉及底层storage pointer属性；这不是本机逐函数耗时证据。
本地源码显示已确认GPU-hit仍经过HOST allocation checker前后两次清点；
source_snapshot及routing_source_snapshot也构建物理图。D329已经否定
跨调用epoch/model-ID缓存，D330已共享planner实测发布，不能重复这两项。

下一候选仅研究**同一个串行原生操作内部**、无HOST tensor增删/替换的
已确认GPU-hit分支，是否可以复用刚测得的物理图、同时重新读取活跃allocator
占用并完整检查预算。它不是跨请求缓存、TTL、少检查一次物理上限或变更
IEEE公式。先核查acquire/fence/回调/异步准备全部变更入口，复用已有CPU
fixture证明前后等价和不变式，再决定实现；目前尚未实现、不能先报收益。
若无法证明，该候选不放行，依据已有具体操作证据转向下一项，不再重复D351。
通过后最多两次最小验证且需新证据，随后普通完整4000回放验证净收益。

完整结果：paper_results/ieee_tc/p2_backend/d351_core_timing/analysis1；
操作聚合：同目录operation_detail1。原始source SHA
df53608d95fc85ffeca113a4ff2d8ac8ad85bfd1df64e9b9c5d00fe15ef1eb3c。
本轮未画图、未更改服务策略/公式/SLO/配置，未推进3B或外部baseline。

## 磁盘与证据保护

收集后空闲降至150GiB启动底线以下约17MiB，未放宽门槛。复用D339无损
流程，仅将已完成D341的main_outcome.json归档为gzip；原始与两次解压SHA
均匹配既有manifest，确认无进程引用后移除未压缩重复件，释放490188800 bytes。
旧投稿保护目录、唯一权重/trace、失败证据和既有统计不变；精确恢复路径
及SHA保存在本次storage回执。后续新重型任务仍须重新检查实际余量。
