# D162：当前 7B 等待路径诊断

2026-10-02。开发诊断，非正式性能比较。当前 serving 版本为
`11a5432ecac7ca6214417ffc471c834b6edeb3e3`，启动证据版本为
`ccde02142b1846965795551d3317979086c6b150`。复用 D160 的 cap4、D157
初始化资料、IEEE Full、60 秒部署通知和真实远端已发布工件，只取 source42
既有 W0 前 1,000 请求，并启用已验证的 `python_frames_v1` 观察器。
没有新权重、负载、交付缓存或 serving 修改；不能替代普通 Full 4,000 请求。

## 1. 完成、资源与远程边界

| 检查 | 本次结果 |
|---|---:|
| 计划 / 终态 / 原生生成合同匹配 | 1,000 / 1,000 / 1,000 |
| 请求失败 / runtime quarantine | 0 / 0 |
| 物理 GPU 租约 / 已确认释放 | 4 / 4 |
| 物理 GPU-s，诊断值 | 4,499.054580 |
| 业务前 / 到达窗口 GPU-s | 35.798983 / 4,259.321597 |
| drain / 终态后至释放 GPU-s | 105.953333 / 97.980667 |
| 远程获取 / 已发布并核验内容 | 60 / 60 |
| 客户端收到 / 远端写出线上字节 | 74,953,014 / 74,953,014 |
| 核验后的逻辑内容字节 | 1,474,429,899 |
| 按请求打包次数 | 0 |
| service 内存峰值，B | 19,012,448,256 |
| 主机最低可用内存，B | 93,334,138,880 |
| high / max / OOM / swap / 告警 | 0 / 0 / 0 / 0 / 0 |

资源监控 1,202 条；物理窗口互斥并与总 GPU-s 一致。activation 为 initial 1、
natural scale-out 3，不充当受控 A4 或足够 first-service 样本。
实际释放后才停止本轮远端服务并复制日志，60 个 transfer UUID 一一核对。
远端日志 SHA `b1be986fa1443f57de64419ea71448115d19f851682c11f121a7a0c68af66a33`；
监控 SHA `ebabfbce5d7605a0f24f159727c4111914f2a1a9d323098f40ad6eff31b9913e`。
沿用 D78/D80 的一次性交付缓存，不重复构建或在请求中打包；真实传输和竞争保留。

原生数量与请求身份记录通过不等于数值 adapter 正确性已验证。共同 warm SLO、
Resident 预算、数值身份和 G1/G2 仍开放；不计算达标资源排名，不使用开发
5,000 ms 或 CE 代替共同阈值。本次也不以单轮数据计算 CI。

## 2. 观测覆盖与限制

| 实际角色 / PID | 全期样本 | 业务期样本 | 最大采样延后，s |
|---|---:|---:|---:|
| 控制器 3577602 | 590 | 538 | 0.967671 |
| 规划器 3578498 | 584 | 547 | 0.448229 |
| 前端 3579676 | 566 | 550 | 0.523779 |
| 前端 3583879 | 544 | 536 | 0.490973 |
| 前端 3583952 | 547 | 534 | 0.498833 |
| 前端 3584314 | 549 | 532 | 0.602717 |
| GPU core 3580427 | 559 | 552 | 0.375775 |
| GPU core 3584939 | 537 | 531 | 0.368408 |
| GPU core 3585110 | 540 | 529 | 0.271675 |
| GPU core 3585252 | 542 | 527 | 0.398266 |

业务期是共同计划到达开始至最后请求终态，包含 drain。控制器另有业务前
28、请求后 24 个样本；保存结果阶段不算在线等待。全部主线程都存在，末行
没有残缺。全期 7 条栈达到 100 帧上限，其中业务期 GPU core 4 条；控制器
3 条截断发生在业务期之外。保留截断，不补造未观察到的 caller。

这些是 GIL 调度下的栈出现次数，不是 CPU 时间百分比或请求耗时分解。
idle 栈保留，C/CUDA 内部不覆盖，inclusive 调用链有重叠。观察器本身会
产生扰动，因此不直接和 D160 比较延迟或资源优越性。前端同时出现 controller
文件名不表示存在额外控制器；按实际入口区分为 1/1/4/4 个角色。

## 3. 当前证据，不是唯一原因判定

控制器业务期最近项目帧包含以下操作；完整数据与调用链保存在 curated JSON。

| 操作 | 出现次数 / 538 个业务样本 |
|---|---:|
| RPC 响应 JSON 解码，`_send_rpc_on_channel:5313` | 71 |
| 本地文件清单，`_local_file_inventory` | 23 |
| 计划 `execution_copy` | 20 |
| 规划来源观察，`_planning_source_observations` | 20 |
| `preparation_snapshot`，最近项目帧 | 18 |
| `execution_bundle_copy` | 17 |

第一项当前源码确为 `json.loads(raw.decode(...))`，不是把 await 时间直接算作
CPU；但采样仍不能给出该操作的总耗时。D161 的生成回复约 1.8 KB，不能据此
推断其他 source/control 回复同样小，也不能将它归为远程 artifact 网络。

更具体的一条冗余路径已由源码和本轮调用链同时证实：20 个 `execution_copy`
样本均经 `Mapping.__contains__ → __getitem__ → snapshot` 到达整份计划的
`pickle.loads`。当前文件执行入口只是在判断 `'source_view' in plan`，随后
`execution_preparation_bundle` 又需打开同一不可变计划。前一个操作逻辑上
只需要键集合，不需要物化全部 source view 和目标数据。

这条发现支持下一步检查不可变计划接口的表示成本，而非删除公式、验证、
实时来源检查或资源保护。需要先证明成员查询、缺失键、不可哈希键、导出隔离
和已验证 envelope 的语义与原来一致，再测组件成本及普通 Full；本文件没有
宣称修改已实现或收益已经测得。也不因此顺带改 RPC 编码、增加并发或超时。

四个 GPU core 的 HOST inventory inclusive 出现次数依次为 128/552、116/531、
89/529、98/527；直接 HOST `add` caller 计数为 113、100、75、91。该路径仍有
开销线索，但不能套用 D143 旧版比例，也不能与上述控制器计数相加。D144 的
同次观察复用已生效，不重复提出同一候选或跨状态复用旧 residency 判断。

## 4. 原始资料与下一步

[CPython 3.12 的 Mapping 实现](https://raw.githubusercontent.com/python/cpython/3.12/Lib/_collections_abc.py)
说明默认成员查询会调用取值方法；本机观察对应这条已知接口语义，不是推测。
[asyncio 文档](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明同步工作可能延迟同一事件循环的其他任务，但不能替代本机收益测量。
后续仍沿用 [vLLM 的控制路径性能分析思路](https://vllm.ai/blog/2024-09-05-perf-update)，
不借用其硬件性能或加速数字。

复用 D143 已通过六项测试的统计器，只替换 campaign 标签，未重复旧资格。
CPU 汇总 0.52 s / RSS 21,888 KiB；终态/资源/remote 核验 1.57 s / 171,896 KiB；
当前 caller/table 汇总 1.14 s / 26,248 KiB。三者分别运行于 3/4 GiB、swap0、
CPU2,3,26,27，末尾内存事件为零，退出后确认 inactive/空身份/空资源域。
322 项冻结来源与 147 项保护结果核验通过，IEEE 公式及 serving 源码未改。

按 `analyze-results` 区分观察、解释和待验证假设；按 `academic-plotting` 与
计划 §11 使用精确诊断表，不把栈次数画成 CPU 时间堆叠贡献。原始 69,442,442 B
请求结果只保存，本项没有重新解析其逐请求时延，所以不冒称完整时延恒等式
已经校验；原始 metadata 的完整字段通过只去 ASCII 缩进的方式读取，未删字段。

- 原始目录：`results/ieee_tc/p2_backend_qualification/d162_20261002/`。
- curated：`paper_results/ieee_tc/p2_backend/20261002_d162_prefix_diagnostic.json`，
  SHA `ea9ae817196f8f8f5ccf2a5869469690ceb5e4aa53e5c53a606e337c21f9c8d5`。
- 角色表：`20261002_d162_prefix_roles.csv`，SHA
  `fd0fc31dcd53f53ff96e78e9430cd3fee61d810d008460ff2c24fed8def2abf4`。

先封存、备份本次诊断，再验证一个明确的冗余操作候选；随后普通 Full。
7B 实际 G1/G2 达标前不推进 3B 或外部 baseline。两模型数值身份与共同参考、
旧 Prime 的新指标对照、正式 M1/M2、A1–A5、S1–S13 均保留为未完成任务。
