# D123：7B 原始前 1,000 请求的控制路径剖析

状态：请求、资源、CPU、逐请求字段分析及证据封装完成；Git 备份待提交。
这是开发期诊断，不是 4,000 请求 Full 资格、正式 SLO 或系统排名。

## 问题与执行条件

D119 的 7B 完整回放存在明显积压和超时；D120 阶段分析不能证明单一
瓶颈，D121 保存状态微测也不代表实际 7B 请求路径。D123 因此仅执行一次
当前实现的 CPU 剖析，使用原 W0 源 trace 的前 1,000 个请求索引视图，
保留完整 500-adapter 可访问集合，不生成新负载或权重。

执行版本 `d4825f64ae51096dd619cd9b26dc09b166c8dc7e`；与 D119 相同的 7B
配置、D89 初始化 profile、60 秒部署通知、固定输出生成契约、计划到达起
1,800 秒期限、资源边界和真实远程已发布工件。只有诊断人口与本轮独占
输出路径不同。没有新增服务优化，baseline 仍暂停。

py-spy 0.4.2 对父控制进程采样：100 Hz、GIL-only、线程标签及完整文件名。
不包括原生子进程的完整 CPU/GPU 成本，没有测得 profiler 自身开销。
不能把该诊断与无 profiler 的完整回放作单因素性能增益比较。

## 完整诊断状态表

| 项目 | 实测值 | 解释边界 |
|---|---:|---|
| 原始源 / 本轮计划请求 | 4,000 / 1,000 | 显式 diagnostic prefix，不替代 Full |
| 已提交 / 已终态 | 1,000 / 1,000 | 所有失败保留 |
| 原生契约成功 / 超时 | 916 / 84 | 数值 adapter 正确性及正式 SLO 未合格 |
| 物理 GPU 占用 | 11,695.308336 GPU·s | 不作为达标资源排名 |
| 物理租约 / 已释放 | 5 / 5 | 不等同五个成功服务实例 |
| runtime-ready / cancelled activation | 4 / 1 | 初始四个为 initial 1、natural 3 |
| 隔离事件 / 已释放 | 2 / 2 | 并非请求失败数量 |
| 远端传输 ID 配对 | 60 / 60 | 同一远端进程时钟与工件身份 |
| 发送 / 接收字节 | 74,953,014 / 74,953,014 | 线上压缩字节，非解包字节 |
| 请求内打包 | 0 | 复用一次性发布缓存 |
| 服务内存峰值 | 20,695,658,496 B | 3,247 个资源样本 |
| 主机最小可用内存 | 92,699,725,824 B | 本轮无安全告警 |
| high / max / OOM / swap | 全部 0 | 不证明不存在控制路径瓶颈 |
| 服务、回放、监控退出码 | 均为 0 | 执行收尾成功，不等于请求全部成功 |
| GPU 与本轮 HOST/NVMe 工作区清理 | 完成 | 唯一工件和旧结果未删除 |
| 1 ms 计时一致性要求 | 未通过：3 个请求 | 最大偏差 3.948905 ms，未放宽门槛 |

## 成功请求条件下的时延与失败证据

以下是 916 个成功请求的原始报告列，单位为秒，TPOT 单列另标毫秒。
不包含失败服务时延、不构成全体请求性能排名。分位数使用 Type-1；不同
阶段 P95 不能相加。3 个请求存在下述小间隙，原始数字没有被覆盖。

| 指标 | 平均 | P95 |
|---|---:|---:|
| 原始用户 TTFT | 700.900821 | 1,692.545729 |
| 原始 controller-result E2E | 707.796039 | 1,701.858600 |
| dispatch/admission 等待 | 694.305124 | 1,687.286159 |
| dispatch-window 等待 | 684.191716 | 1,672.513399 |
| runtime/source-slot 等待 | 8.330000 | 23.875586 |
| arrival release lateness | 1.783407 | 5.650350 |
| service TTFT | 6.595697 | 25.823560 |
| admission → native dispatch | 6.330967 | 25.613134 |
| 原生后端 TTFT | 0.264730 | 0.565928 |
| 原生 decode | 3.657414 | 7.744426 |
| worker → controller 完成通知 | 3.212812 | 9.833613 |
| TPOT（ms） | 31.420830 | 50.771127 |

`controller-result E2E` 和外层 terminal 都不是客户端响应接收事件；不能
因此宣称正式客户端 E2E 计量已经完成。GPU 生命周期使用独立物理租约。

全部 916 个成功请求的 prompt/hash/原生 token 数及生成契约核对完成；
dispatch 前选中来源身份、时钟、顺序一致：GPU 333、HOST 317、NVMe 211、
Remote 55，tier 冲突为零。这是成功子集证据，不是所有失败请求的 dispatch
完整覆盖，不是数值 adapter 正确性或完整 A4 研究。

84 个失败均为 `TimeoutError`。首个为 `req_00369`，观察 offset 为
2,249.536751 秒。全部 84 个没有记录 generation-submission/source-admission
证据；缺字段、空 tier 和默认零值不能证明“没有 dispatch”或“等待为零”。
没有返回的 ownership-unresolved 错误；两次 runtime 隔离仍须单独保留。

### 计时一致性问题：保留失败，不放宽 1 ms

首次 curator 在 `req_00063` 的 2.792140 ms 偏差处停止，原脚本、日志和
退出码保留。第二版仅把阈值检查转为完整违例清单，明确
`pass_tolerance=false`，未改数据或阈值。三个请求为：

| 请求 | TTFT / controller E2E / dispatch 的共同偏差（ms） |
|---|---:|
| `req_00063` | 2.792140 |
| `req_00330` | 3.205084 |
| `req_00539` | 3.948905 |

service/dispatch 内部分解及 TPOT 重算误差为零。源码中 `run_one` 记录全局
准入结束 G；`_exec_request_in_reservation` 随后才记录 slot 等待起点 S，
而选定 source-admission 为 T。当前原始 dispatch=(G−A)+(T−S)，相对直接
边界 T−A 少计 S−G；同一缺口随分段求和进入 TTFT 和 controller E2E。
这是具体计时边界线索，不是数百秒积压的解释，也尚未单独测得间隙内各
操作耗时。后续需按权威绝对边界核对/修正；不能把本轮标为计时合格。

## CPU 采样表

有效样本 279,902；报告采样错误 67，栈读取警告同为 67。以下是每个样本
最内层项目函数的互斥归属，排除具有 `save_results` 祖先的样本后展示热点；
**比例分母仍是全部 279,902 个样本**，不是请求墙钟时间或用户延迟。

| 最内层项目函数 | 样本数 | 全部样本占比 |
|---|---:|---:|
| `owned_preparation_inputs` | 59,998 | 21.44% |
| `_run_ieee_file_preparation_plan` | 54,296 | 19.40% |
| `_send_rpc_on_channel` | 37,606 | 13.44% |
| `_footprints` | 31,492 | 11.25% |
| `_local_file_inventory` | 12,329 | 4.40% |
| `validate_ieee_execution_plan` | 9,643 | 3.45% |

另有保存结果祖先样本 24,896、import 祖先样本 1,811、其他 253,195；三类
互斥且合计全部样本。inclusive 调用栈统计允许重叠，不能相加作为耗时分解。
GNU time 包含 profiler 及其等待的后代，不作为孤立控制进程 CPU/RSS。

## 当前可支持与不可支持的结论

1. 当前准备输入和文件准备执行路径是控制进程的重要 CPU 热点，足以指导
   下一次针对真实数据结构和调用频率的核查。
2. 尚不能断言它们解释了全部排队、84 次超时或某一百分比的 E2E。需要
   逐请求阶段证据，以及保留所有权、时效性和字节预算检查的因果验证。
3. 不能以提高 deadline、删除失败、关闭物理保护或盲目增加并发作为修复。
4. 先关闭本轮证据、图表与备份，再选择一个可证伪优化；验证后返回普通
   Full 回放，再继续 warm/Resident、baseline、M1/M2 与消融/敏感性主线。

本轮采用计划第 11 节与 academic-plotting 的数据证据规范，单次资格/瓶颈
诊断优先表格，不制造排名图、显著性或 n=1 置信区间。

## 证据位置

- 原始目录：`results/ieee_tc/p2_backend_qualification/d123_20260930/`。
- 请求原始文件 2,886,956,618 B，使用未修改 D96 streaming projection；
  从不整体加载重复的 native inventory。
- 初步表：`prefix1_preliminary.json`；CPU 表：`prefix1_cpu_samples.json`。
- 远端日志：`remote_7b_prefix1_transfers.jsonl`，SHA-256
  `a51c18941e6c2479fead2d76dd5bead30c5d7167f4913ba490df7e8047f50502`。
- 远端监控：`remote_monitor_prefix1_final.log`，SHA-256
  `d7b111494a2aad73a341f87a60abdb68db68fcd5ff559f63e7fcaaf4c9704dd1`。
- 两端 SHA 已相等；远端服务和本轮本地 auxiliary 已按精确身份关闭。

最终 curated：`paper_results/ieee_tc/p2_backend/20260930_d123_7b_full_w0_prefix1.json`，
SHA-256 `93614036c20e9b978fe5e98c99163f77235ee1e7a7770110e36bcea63c271b65`；
失败清单 `20260930_d123_7b_full_w0_failure_breakdown.json`，
SHA-256 `9af5c025465fada4704961c834d7ae74bd0e2c2b26e66ba40f5880b59d2e1fa6`。
122 个冻结来源及 147 个保护文件在 curator 中核验通过。原始请求文件只
流式计算一次 SHA；后续封装复用该 SHA 加文件 stat，不重复读取整树。
原始 SHA 为 `70c3a0fe24dd3edf1108183090ba5dcc8d762ca18ebd014b733b9b1c7848be70`。

71 项生命周期、回放和 token 计量测试通过；这里证明证据处理一致，不将
3 个实测计时违例改为通过。最终验证清单 SHA 为
`280e9b565d5ca439e61b6d0bd6946db64cbce433d13ef26e862126ecdbcb76d3`。
75 个小型脚本/配置/回执经逐成员 SHA 验证，压缩包为 50,029 B，SHA 为
`f0a2f617d7c33a00fe9a6b519fd9eec1965c3f4c28bad72f4e2c79b79ca84bc5`。
大结果、权重、凭据均不入该包。全部分析资源组已按精确身份、空进程组
关闭，内存事件均为零；没有遗留 GPU、远端或分析任务。
