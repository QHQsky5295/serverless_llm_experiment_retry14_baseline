# D198：请求状态查询的同机分段诊断

2026-10-03。状态：CPU 资格验证通过；不是优化效果或模型验收。

## 问题与唯一动作

D195 完整 7B W0 的平均记录 routing 等待为 519.378 ms。D196 保留数据
不能区分控制器推进、前端处理、原生引擎控制调用与 worker 内实际 inventory。
D197 消息表示研究不支持优先做生产压缩优化，因此不实现该候选、不重复 Full。
本次只增加 opt-in 诊断边界；不改变九个公式、调度、驻留、准入、超时或配置。

复用 `python_frames_v1` 已有非正式、受限 diagnostic-prefix 准入。只有当前 PID
确实完成该准入后才启用，默认无诊断 ID、无新增诊断文件或返回字段。
仅记录 `ieee_request_sources` → `request_source_snapshot` 只读调用。
每次调用的随机 attempt ID 贯穿控制器、专用前端和原生 worker；不记录 prompt、
权重、token 内容、通用 kwargs 或错误文本。不缓存状态、不添加 CUDA 同步。

## 边界与解释

| 记录点 | 含义 |
|---|---|
| parent_begin | 控制器开始该 RPC，在连接申请之前 |
| parent_send | 请求编码后、异步传输前；附发送字节数 |
| worker_received | 专用前端读取并解析命令后，开始分派 |
| frontend_begin | 前端请求状态方法入口 |
| frontend_native_send | 原生 collective RPC 调用之前 |
| native_begin | 原生 worker 方法入口，物理观察之前 |
| native_ready | 原有 owner/物理 inventory/device 检查后，返回构造前 |
| frontend_native_received | collective RPC await 恢复后，校验回复前 |
| frontend_ready | 原有验证和紧凑响应构造完成 |
| parent_received | 控制器 roundtrip await 恢复后、解码前；附响应字节数 |
| parent_terminal | 原有连接清理后；success/error/cancelled |

所有事件保存本机 monotonic clock ID、PID、线程 ID、monotonic 秒和线程 CPU 秒。
跨进程差值仅在 clock ID 一致、身份唯一、边界齐全且顺序正确时计算。
事件按每 PID 独立文件立即写出，不依赖整轮成功。取消可出现 parent_terminal
早于迟来的原生观察，按取消调用保留，不伪造同步取消或成功。

边界间包含记录自身开销，不事后扣除。异步前端/控制器的线程 CPU 增量可能
包含其他协程，不能称该请求 CPU 时间；native 同步区间也不等于纯 CUDA 时间。
发送→收到、原生发送→native_begin、native_ready→前端恢复等区间含序列化、
传输及调度，不能分别命名为纯网络或纯排队。并行四副本查询不能把四段相加
解释请求关键路径。缺失/重复/异常顺序/不同 clock 必须单列，不补零。
该诊断不产生正式性能排名、SLO 资格、数值正确性或 G1/G2 达标结论。

## 第一性原则与参照

采用已有调用链的直接边界观测，先区分实际工作与 await 周围的等待，再决定
是否需要优化。vLLM 0.30 的异步控制调用经过 utility future/消息传输和输出
处理，并非直接的同步函数耗时，故不能以总 RPC 等待归因 inventory。
依据：[固定版本官方源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)、
[Python 时钟文档](https://docs.python.org/3.12/library/time.html#time.monotonic)。
没有引入其他论文硬件上的加速倍数或通用吞吐目标。

## 资格、资源与下一步

复用原 CPU scope/test wrapper；3/4 GiB、swap 0、CPU 2,3,26,27。
测试默认关闭、实际准入进程、跨进程时钟、短写、非法 ID、native owner 检查、
实际 loopback worker/proxy、成功/失败/取消和后续新读；随后运行既有回归。
CPU fixture 不是 native CUDA 性能证据。原生完整数据和失败证据均保留。

| 资格检查 | 测试数 | 结果 | 测试主体秒 |
|---|---:|---|---:|
| 新增诊断及相关状态/传输检查 | 40 | 全通过 | 49.586 |
| 完整请求/owner/规划/取消及诊断回归 | 1100 | 全通过 | 156.099 |

两组有重叠，不是 1,140 次独立实验重复。新增诊断测试 10 项，其余为既有回归。
两次测试均无 memory high/max/OOM 事件和 swap 使用，结束后资源域已消失。
没有失败测试或针对测试结果修改生产逻辑。本次交付资格状态表，不制作性能
提升图。完整历史 ledger 逐字归档并校验 SHA，旧实验数值未改写。

新 GPU 诊断尚未运行；必须先通过资格、提交备份及新磁盘/资源门槛。
只允许一次当前配置的 1,000 请求前缀诊断，以测量缺口决定下一候选，不连续
堆叠优化。7B 数值正确性、128 个输出 hash 变化、共同参考/预算和旧 Prime
新口径对照仍未闭合；7B 达标后才推进 3B，再恢复基线及完整后续矩阵。
