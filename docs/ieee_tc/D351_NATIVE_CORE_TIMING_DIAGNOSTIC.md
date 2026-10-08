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

当前：CPU最初49项及最终202项通过；最终配置绑定、历史保护和资源准入
通过，尚未运行GPU诊断。准备脚本最初误要求继承的 shutdown 必须在
子类体内定义，检查失败后按实际继承关系修正；日志通过标记的转义检查
也在写出绑定前修正。两次均未启动GPU、未修改实验结果；原测试/失败
准备脚本保留。检查点及GPU诊断待执行。
N_correct、正式共同warm/Resident参考与G1/G2仍OPEN；不进入3B/基线。
