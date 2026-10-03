# D196：D195 剩余控制等待的离线分解

2026-10-03。离线诊断已完成，不是新增推理运行，服务代码未修改。
D195 结果已封存并备份为 `a42f250785ce697750ebc3a7963ff06ef30f0f32`；
运行源码为 `a0c4a9d24b4fe42625a4fb9bd80f6c5a8f4d8fed`。
Plan、九个公式及冻结指标 V1 不变。只推进 Prime 7B，3B 和外部基线暂停。

## 1. 问题和复用范围

D194 改变初始化后 planner 使用的观测表示，D195 是其唯一普通完整开发回放。
平均 TTFT 比 D190 下降 5.857%，但 P99 TTFT 和平均 TPOT 退化；
物理 GPU 生命周期占用变化仅 −0.02854%，没有建立资源节省证据。
同一 D170 暂定参考的联合时间达成上界为 3476/4000（86.9%），
距 3800 仍差 324；数值 adapter 正确性尚未完成，不能称正式 SLO 合格。

本次复用 D191 入口及现有 `analyze_rpc_breakdown`，
只读 D195 已封存的 34,407,428-byte 请求投影；
旧对照直接复用已封存的 D191/D190 阶段表，不重解析旧大文件。
4,000 请求、完整字段、时钟与阶段恒等式检查通过，
交付 4,000 行请求阶段、65 行分组以及生成 RPC 汇总。
无新推理、工件、负载、profile 或远端操作。
D78/D80 两模型一次性只读交付缓存已完成，本轮不重复制作。

## 2. 完整非重叠阶段

单位 ms；变化为 D195−D190。阶段均值可相加，阶段 P95 不可相加。
每版只有一次开发运行，差异仅是观察，没有 CI 或孤立机制因果结论。

| 非重叠阶段 | D190 均值 | D195 均值 | 均值变化 | D195 Type-1 P95 |
|---|---:|---:|---:|---:|
| 计划到达→服务端接收 | 283.278 | 223.604 | -59.675 | 1201.852 |
| 接收→全局 gate | 204.150 | 198.569 | -5.581 | 1360.616 |
| gate→源准入：已记录 routing | 563.304 | 519.378 | -43.927 | 1643.708 |
| gate→源准入：其余残余 | 415.139 | 414.847 | -0.292 | 1506.078 |
| 源准入→generation handoff | 562.536 | 539.380 | -23.155 | 1986.519 |
| generation handoff→native dispatch | 72.948 | 65.913 | -7.034 | 364.534 |

引擎前均值 2101.355→1961.691 ms，减少 139.663 ms；
相对于平均 TTFT 2290.593 ms，仍占 85.641%。
gate→源准入合计 978.443→934.224 ms；
已记录 routing 降低 43.927 ms，其余残余只降低 0.292 ms。
321 个请求在进入原生引擎前已经越过暂定 TTFT 阈值。

残余包括被拒绝且未计入 routing 的读、重试、pending 登记、
选中源保护、容量等待和协程恢复；不是单个 RPC 的耗时。
routing 中含 await，也不是独占 CPU 时间。
入口和源准入后各段均有变化，不能把整个改善归给某一函数。

## 3. GPU-selected 条件分组与本机通信

GPU-selected 集合为 D190 的 1232 条与 D195 的 1165 条，
不是完全相同的请求子集，以下只用于定位，不是配对因果比较。

| GPU-selected 阶段，ms | D190 均值 | D195 均值 |
|---|---:|---:|
| runtime_slot_wait_ms | 1090.817 | 1051.791 |
| recorded_routing_ms | 565.537 | 519.888 |
| runtime_wait_outside_recorded_routing_ms | 525.280 | 531.903 |
| source_to_generation_handoff_ms | 1.213 | 1.219 |
| generation_handoff_to_native_ms | 98.999 | 93.366 |
| dispatch_admission_wait_ms | 1530.451 | 1429.614 |

GPU-selected 在 gate→源准入仍平均耗时 1051.791 ms；
源准入之后交给 generation 接口只需 1.219 ms。
这说明不能只以远端传输解释其等待，但保护已经发生在源准入边界之前，
不能据此宣称保护免费或真实网络对其他层级无影响。
该条件组残余反而增加 6.623 ms，不能隐藏。

全体计划到达→task 为 1.856 ms，task→submit 为 0.864 ms，
submit→server receive 为 220.884 ms。
源码中的这段入口是本机 Unix socket，不是 174 的工件 HTTP 链路；
包含发送、背压、接收和调度，与上表第一项重叠，不重复相加。

generation RPC 终态回复均值 1774.345 bytes；
channel 获取 0.011719 ms、send flush 0.110260 ms、
response pickup 261.817723 ms。
pickup 有同机 wall-clock 和负差 clamp 限制，且不是 source RPC 耗时。
结构性零 resolve/admission/thread-resume 不能解释为机制开销为零。

## 4. 历史、当前实现与证据边界

D144 在途共享、D151/D174/D181 观测投影、D159 worker、
D163 成员查询、D176/D178/D184 源事务、D189 静态描述及 D194
完整规划图投影均已存在。不重复这些资格，也不重新尝试已拒绝的 D192 编码器。

当前 dispatch permit 已采用事件驱动 FIFO，授予时先转移计数再唤醒。
不能未经证据把它改成更短轮询来“优化”。
当前 request snapshot 对各可行副本并发查询，只共享尚未完成的同目标读取；
动态状态没有 TTL 缓存。D195 共 16806 次此类原生查询，不能将其数量
乘以 generation RPC 的时间来估算 source 查询成本。

源码还显示 request-source 每次保留完整 GPU pool inventory；
它不仅构造张量布局，也校验当下 active、slot 与 registered 关系。
因此“固定分配池”不意味着整个结果恒定，不能按 manager ID 缓存整个返回值。
本轮仅确认依赖，尚无该函数的当前独占耗时证据，也未决定采用优化。
旧 D187 栈采样不是当前 CPU 百分比，不能直接沿用其热点排序。

本轮核对 [vLLM 官方 CPU 分析](https://vllm.ai/blog/2024-09-05-perf-update)
与 [Python asyncio 说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：
控制/表示工作可能延迟请求推进，但文献不能证明本机某一函数的贡献。
[固定版本 LoRA manager 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
提供原生槽位生命周期核查入口；本机已适配实现必须另行核对，
不把官方代码直接当成本机未修改的证据，不借用其他硬件的加速数字。

## 5. 下一项有界调查与主线

下一项先核查请求来源观测中“物理布局”和“动态槽位状态”的依赖边界，
结合已有实测快照及本机/官方 setter 生命周期，判断是否存在重复构造。
这只是一个待证伪问题，不是已接受的缓存方案或已发现主瓶颈。

若不能证明复用边界、对象替换检测、别名/物理容量检查和动态失效均保持，
或没有相关组件收益证据，就停止该方向，不为了验证猜测重复 Full。
不得缓存旧 tier、删安全检查、另加 TTL、调整共同 SLO 或改变配置。
只有一个合格候选才进入普通完整回放；当前没有新候选代码。

实际 7B 数值正确性、128 条输出 hash 变化、共同 warm/Resident、
按新指标的旧 Prime 对照、G1/G2 仍 OPEN。
先完成这些实际验收，再进入 3B 和外部基线；
M1/M2、A1–A5、S1–S13 全部保留。

## 6. 资源和交付

分析实际身份 `026c90f010da4e07a38efc71c16ef3fa`；
3/4 GiB、swap0、CPU2,3,26,27，wall10.41 s，RSS1102588 KiB。
退出0，保存的 high/max/OOM/swap 为零；实际 scope/身份/路径已消失。
147 项旧结果前后保护通过，没有更改服务代码或用户 manifest。
本机可用约 160.72 GB，低于新重型任务 150 GiB 门槛；
本次只做受限 CPU 分析，下一 GPU/build 前必须审计回收并重新准入，
不降低推理机内存/磁盘护栏，也不删除唯一证据。

`analyze-results` 用于区分观察、解释与因果限制；
`academic-plotting` 按计划选择完整精确阶段表，不制作单次显著性图。
原始入口/日志：`results/ieee_tc/p2_backend_qualification/d196_20261003/`；
汇总：`paper_results/ieee_tc/p2_backend/20261003_d196_control_path/`。
独立数值重算、来源 SHA、文档表和旧结果校验完成后封存，
按 github-sync 检查并备份到既定 V2 分支。
