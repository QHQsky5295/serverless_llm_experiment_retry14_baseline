# D186：D185 的控制路径分解与下一步定位

2026-10-03。离线诊断，不是新增 GPU 性能运行；服务实现未改。
复用已封存 D185 的 34 MB 请求投影及完整请求身份证据，旧比较直接复用
D183 对 D182 的阶段表，不重建投影、不扫描旧大文件、不改原数字。
当前主线仍是实际 7B 达标；3B、外部 baseline、M1/M2、A1–A5、S1–S13
全部保留待做。一次性交付缓存已在 D78/D80 完成，本轮未访问远端或重建缓存。

## 1. 本次问题与方法

D184 将选中 GPU 后的 pending 登记与引用获取放在同一前端顺序执行，
减少控制器中间一次往返。D185 的完整回放 P95 TTFT 降低，但均值、TPOT、
GPU 生命周期没有一致改善，不能将 D184 宣称为净收益已成立。

本次检查既有非重叠阶段，回答“缩短的部分是否被其他阶段抵消”。复用
`analyze_control_path_overhead.py::analyze_rpc_breakdown` 及 D183 分解入口，
仅更换冻结输入和输出目录。全部 4000 请求的字段、时钟、原生合同和分解
恒等式通过；输出 4000 行控制阶段、65 行分组汇总和完整 RPC 字段表。
没有用缺省零、裁负值或文本 token 估算替代缺失证据。

## 2. 完整总体阶段表

单位 ms；变化为 D185−D182，负值表示等待减少。每个阶段的均值可以相加；
各阶段 P95 不能相加。每次回放 n=1，不给 CI 或因果增益断言。

| 非重叠阶段 | D182 均值 | D185 均值 | 均值变化 | D185 Type-1 P95 |
|---|---:|---:|---:|---:|
| 计划到达→服务端接收 | 326.534 | 342.659 | +16.125 | 1692.086 |
| 接收→全局 gate | 299.478 | 273.549 | -25.929 | 1787.442 |
| gate→源准入：已记录 routing | 582.821 | 620.808 | +37.987 | 2120.123 |
| gate→源准入：其余残余 | 526.435 | 474.286 | -52.149 | 1915.413 |
| 源准入→generation handoff | 583.696 | 601.934 | +18.237 | 2573.788 |
| generation handoff→native dispatch | 70.607 | 74.605 | +3.997 | 402.997 |

合计引擎前均值为 2389.572→2387.840 ms，仅减少 1.732 ms。
gate→源准入合计 1109.256→1095.093 ms，减少 14.163 ms：
已记录 routing 增加 37.987 ms，未覆盖残余减少 52.149 ms。
source→handoff 增加 18.237 ms，handoff→native 增加 3.997 ms。
所以“减少一次往返”不等于全部请求等待按某个 RPC 平均时间等额下降。

残余包含未记录的重试、pending、选中源保护、容量等待与事件循环恢复，
不是独立的 GPU acquire 或 pending RPC 时间。routing 计时包含 await，
也不等于独占 CPU 时间；返回 None 的拒绝观察不计入该 timer。
D185 的 native TTFT 均值约343.121 ms，故平均总 TTFT 2730.961 ms 中
约87.436%仍发生在进入原生引擎之前。

## 3. 条件分组与通信线索

D185 实际 GPU-selected 为1266条，D182为1300条；不是相同请求子集。
不能由该组差值证明优化的反事实效果，完整所有 tier 数据均保留。

| GPU-selected 阶段 | D182 均值 ms | D185 均值 ms |
|---|---:|---:|
| gate→源准入 | 1254.938 | 1219.170 |
| 已记录 routing | 587.842 | 630.915 |
| 其余残余 | 667.096 | 588.255 |
| 源准入→generation handoff | 1.267 | 1.236 |
| generation handoff→native dispatch | 96.305 | 103.150 |
| 到达→源准入 | 1777.912 | 1781.610 |

源保护已在源准入之前发生；1.236 ms 不能解释成 GPU acquire 零成本。
其后的103.150 ms与此前的1.267 ms边界不同，不能混作同阶段退化。

D185 全请求入口还可细分为：计划→task均值1.867 ms、task→submit
0.803 ms、submit→server receive339.988 ms。源码确认入口为同机 Unix
socket，而非远端174工件下载。这些子段包含发送、背压、接收与调度，
不能将339.988 ms全部称为网络传输或 controller CPU。
它们与上表第一项重叠，不再次求和。

generation RPC 回复约1774.417 bytes，channel取得均值0.019 ms，
send flush0.111 ms，response pickup448.252 ms。
pickup使用旧producer的同机wall-clock且负差clamp，保留该限制；
它不是 source RPC 的耗时，也不能和monotonic阶段重复相加。
仅说明“命令通道取得/发送慢”没有得到这组generation记录支持，
不能推出所有控制通信无成本。结构性零的resolve/admission/thread-resume
字段不作为没有开销的证据。

## 4. 历史、源码与原始资料核查

D144在途共享、D151 staging投影、D153单次文件观察、D159目标构造worker、
D163成员查询、D172解码、D174 storage graph、D176 HOST transaction、
D178 identity recheck、D181请求范围footprint和D184均已进行过验证。
本轮不重复这些资格，也不叠加新的快照缓存、并发或后端配置。

源码中 `_ieee_request_snapshot` 在最后一次await后同步完成文件来源观察、
状态提交及决策；后台 `preparation_snapshot` 仍需新鲜全文件预算/来源观察，
并在协作锁内构造规划输入。两者有不同一致性责任，不能为了快把完整预算
换成目标adapter的footprint，或把新鲜确认换成长期缓存。
目前证据尚未量化它们在D185中的实际占比。

[Python asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
解释同步CPU工作为何可能延迟同一循环的其他任务。
[vLLM 官方 CPU 性能分析](https://vllm.ai/blog/2024-09-05-perf-update)
展示分离控制/API与推理执行的诊断思路；其硬件加速值不借作本机预期。
[vLLM v0.30.0 core_client 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
采用异步跨进程接口。Prime已存在独立前端/GPU core，不能仅重复“分进程”
口号而不定位当前串行段。上述原始来源于2026-10-03重新核查。

D162 CPU帧诊断早于多次实现修改，出现次数也不是CPU百分比。
把它的旧热点顺序直接当作当前瓶颈，会导致继续围绕失效假设优化。

## 5. 决定、下一步与验收边界

本次没有接受新的服务优化，也不撤改D185证据或反复重跑寻找更好数字。
D184仍是当前被测实现，性能采纳状态为“净收益未成立”，不是成功完成。

下一项有界定位应复用既有 `python_frames_v1` 观察与1,000请求前缀入口，
在当前D185实现/冻结配置上定位业务期CPU调用链，区分controller、planner、
frontend、GPU core；这是改变实现后的诊断，不重测已经完成的功能资格。
先核验资源与启动合同，保持原输入索引、真实交付、60秒准备、1800秒保护。
不生成新trace或工件，不为观察改SLO/资源上限，不把带profiler的结果
混入普通Full性能比较。先由当前热点支持一个可证伪候选，再最小验证与
普通Full；若已有日志无法支持具体归因，就保持未知。

当前D185仅原生生成数量与身份记录通过：数值adapter正确性、110条输出
hash差异审计、共同warm冻结、Resident预算和旧Prime新指标对照仍开放。
暂定timing-only联合上界3189/4000=79.725%，不是正式SLO，距离3800仍611条。
G1/G2未达标，不推进3B或外部基线。

## 6. 安全、产物与技能影响

本分析 actual InvocationID `22f373af898749b68c2955bf3a5848dc`，
使用3/4GiB、swap0、CPU2,3,26,27；wall10.71s、RSS1103164KiB，退出0，
memory high/max/OOM/swap均0，退出后资源域路径消失、身份为空。
前后147项保护文件通过；Plan/V1 SHA不变，服务代码与用户manifest未改。
没有GPU/远端运行或维护。本机可用161771876352B，贴近新重型任务150GiB
门槛；下一项启动需新准入，不据此承诺资源足够、不降低门槛。

`analyze-results`用于区分观察、解释与未证实归因；
`academic-plotting`按Plan§11.2选择精确阶段表，无需把n=1画成显著性图；
优化技能要求当前瓶颈证据，因此本轮停在诊断，不加入猜测性实现。

产物：`paper_results/ieee_tc/p2_backend/20261003_d186_control_path/`；
原始入口、受限运行日志：`results/ieee_tc/p2_backend_qualification/d186_20261003/`。
D185/D183封存文件保持不变。本文件经独立重算、来源核验和备份后封存。
