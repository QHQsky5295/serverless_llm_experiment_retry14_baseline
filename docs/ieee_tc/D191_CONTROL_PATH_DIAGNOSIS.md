# D191：D190 剩余控制等待的离线分解

2026-10-03。离线诊断已完成，不是新增推理运行；服务代码保持不变。
D190 已封存并备份为 `6ee3883a916a834a08e342154659a88bd924bd89`，
运行源码仍是 `8a106bb5cc84317a02a089cfddb609065fc23ad7`。
原 Plan 和冻结 V1 均不修改；仅推进 Prime 7B，3B 和外部基线仍暂停。

## 1. 问题与方法

D189 去除静态工件描述的重复构造后，D190 单轮平均 TTFT 比 D185 下降
10.907%，但同一临时阈值下的联合时延上界仍仅 84.375%，未达到 95%。
当前问题不是证明候选已经成功，而是确定剩余等待的位置，避免盲目重跑。

复用 D186 的既有入口及 `analyze_rpc_breakdown`，使用 D190 已封存的
34,378,192-byte 请求投影，旧对照直接复用 D186 对 D185 的阶段表。
4,000 请求身份证据、完整字段、时钟和分解恒等式通过；输出 4,000 行
控制阶段、65 行分组和完整 RPC 字段汇总。无新权重、trace 或原始大文件重解析；
未访问远端，D78/D80 一次性发布缓存不重建。

## 2. 完整非重叠阶段

单位 ms；变化为 D190−D185。阶段均值可相加，阶段 P95 不可相加。
每个版本仍只有一次开发回放，不生成 CI，不把差值视作孤立机制因果。

| 非重叠阶段 | D185 均值 | D190 均值 | 均值变化 | D190 Type-1 P95 |
|---|---:|---:|---:|---:|
| 计划到达→服务端接收 | 342.659 | 283.278 | -59.380 | 1459.798 |
| 接收→全局 gate | 273.549 | 204.150 | -69.399 | 1015.797 |
| gate→源准入：已记录 routing | 620.808 | 563.304 | -57.504 | 1888.943 |
| gate→源准入：其余残余 | 474.286 | 415.139 | -59.147 | 1578.075 |
| 源准入→generation handoff | 601.934 | 562.536 | -39.398 | 2287.255 |
| generation handoff→native dispatch | 74.605 | 72.948 | -1.657 | 396.621 |

引擎前平均时间从 2387.840 降至 2101.355 ms，减少约 286.485 ms。
其中 gate→源准入合计从 1095.093 降至 978.443 ms；
已记录 routing 从 620.808 降至 563.304 ms，其他残余从 474.286 降至
415.139 ms。入口和 generation 交付前也有观测下降，不只一个阶段变化。

残余含未记录的拒绝/重试、pending 登记、选中源保护、容量等待与循环恢复，
不能全部归为某一个 RPC；routing 含 await，不是独占 CPU 时间。
393 个请求在原生引擎开始前已越过临时 TTFT 阈值。相对于平均 TTFT
2433.103 ms，引擎前部分仍占约 86.365%，说明优化不能只看 GPU kernel。

## 3. 条件分组与本机通信

GPU-selected 请求 D185 有 1,266 个、D190 有 1,232 个。
下表是各自实际选中集合的条件诊断，不是同请求子集的配对因果比较。

| GPU-selected 阶段，ms | D185 均值 | D190 均值 |
|---|---:|---:|
| runtime_slot_wait_ms | 1219.170 | 1090.817 |
| recorded_routing_ms | 630.915 | 565.537 |
| runtime_wait_outside_recorded_routing_ms | 588.255 | 525.280 |
| source_to_generation_handoff_ms | 1.236 | 1.213 |
| generation_handoff_to_native_ms | 103.150 | 98.999 |
| dispatch_admission_wait_ms | 1781.610 | 1530.451 |

即使选中已在 GPU 的 adapter，仍观察到源准入之前的等待。
源保护发生在该边界之前，不能把后续 source→handoff 约 1.213 ms
解释成零保护开销，也不能据此排除真实传输对其他请求的影响。

全体请求的计划到达→task 均值 1.982 ms、task→submit 0.815 ms、
submit→server receive 280.481 ms。源码确认这一入口走本机 Unix socket，
不是 174 节点工件链路。子段含发送、背压、接收与调度，不把全部时间
误称为网络传输或 CPU 计算；它们与总体表第一项重叠，不重复相加。

generation RPC 终态回复平均 1,774.431 bytes，channel 获取 0.012 ms、
send flush 0.110 ms、response pickup 352.058 ms。
pickup 使用旧 producer 同机 wall-clock 且对负差 clamp，限制保留；
它不等于 source RPC 耗时，亦不与 monotonic 分解直接相加。
结构性零 resolve/admission/thread-resume 字段不是机制零开销证明。

## 4. 当前源码、历史与官方资料

D144 在途共享、D151/D174 路由投影、D153 单次文件观察、D159 规划 worker、
D163 成员查询、D172 解码、D176/D178/D181/D184 的事务与目标范围查询，
以及 D189 静态描述复用均保留，不重复旧资格或叠加另一项猜测修改。

D187 观察到 controller 的准备快照、RPC 解码和执行 bundle 复制，以及
frontend 的 RPC 编码；这些是调用链采样次数，不是 CPU 时间。
D189 只改变准备描述部分，不能把 D187 的整个热点排序当成 D190 新测结果。
未改的 `_encode_rpc_frame` 当前仍调用 stdlib JSON→Unicode→UTF-8；
dedicated worker 的成功/错误/进度回复及 parent 请求共用它。
D172 仅改解码，不能声称生产端编码已经被那一实验验证。

本轮重新核查：
[asyncio 的同步工作说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明同一循环上的同步任务会推迟其他工作；
[vLLM 官方 CPU 分析](https://vllm.ai/blog/2024-09-05-perf-update)
提供状态表示和控制/推理分离的方向，不借用其加速值。
[vLLM 0.30.0 序列化源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)
使用高效编码及显式类型处理，但其 MessagePack/张量缓冲不能直接当成
本项目 newline-JSON 的等价替换。

[msgspec 官方类型语义](https://msgspec.dev/supported-types) 明确：
其 JSON 编码会将非有限浮点变为 null，也支持 bytes 等扩展类型。
因此**拒绝直接替换编码器就认定正确**；既有非法值拒绝、精确整数/浮点、
Unicode、字典键、嵌套结构和 8 MiB 帧边界都必须保留或明确验证。
不以隐藏 null、缺省值或 legacy fallback 达成表面通过。

## 5. 下一项有界问题与判定

下一项只研究**生产端状态编码的工作量与语义**，不是重复 D172 解码优化：
复用既有实际来源快照和小型请求/回复，不启动 GPU 或重新生成数据。
先核查真实字段类型和消息形态，再测“原编码”与“显式有限原语合同下的
编码”总开销，包含所有新增验证，不只计 C 编码器本体。
本阶段不改变线上协议或服务代码；最大两轮最小验证的原规则继续适用。

如果完整语义检查抵消收益、实际消息不代表当前路径，或无法保持错误/
取消/所有权边界，就归档该方向，不进行又一次 Full 来碰运气。
若组件证据确实支持一个候选，再实现单项变更、资格与完整同合同回放。
仍须将组件收益和端到端收益分开，不以这份离线报告宣布采纳优化。

7B 数值正确性及 130 条输出 hash 变化、共同 warm/Resident、旧 Prime
新指标对照、G1/G2 仍 OPEN；目前不进入 3B、baseline 或主消融。
后续 M1/M2、A1–A5、S1–S13 完整保留。

## 6. 资源、产物与交付

离线任务实际身份 `80469b22e671495bb5998c17570adecf`，
3/4 GiB、swap0、CPU2,3,26,27；wall9.98 s，RSS1103248 KiB。
退出0，保存的 high/max/OOM/swap 为零，实际 scope/身份/路径已不存在。
147 项旧结果前后保护通过；服务代码及用户 manifest 没有改变。
本机可用 160,664,141,824 bytes，低于新重型任务的 150 GiB 门槛：
不启动新 GPU/build，下一重型任务先做已授权的可重建缓存审计与新准入，
不降低推理机门槛。远端无运行或维护。

`analyze-results` 区分观察、解释和未证实因果；
`academic-plotting` 按计划选完整精确阶段表，不给 n=1 装饰性显著性图。
原始入口和日志：`results/ieee_tc/p2_backend_qualification/d191_20261003/`；
数据表：`paper_results/ieee_tc/p2_backend/20261003_d191_control_path/`。
独立重算及 SHA 校验完成后封存本文件，再备份到既定 V2 分支。
