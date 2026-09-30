# D115：3B Full10 完整请求回放，驻留任务收尾失败

实验日期：2026-09-29；证据收尾：2026-09-30。状态：完整流式提取及汇总已完成。
不是 Full 验收通过，不是正式 SLO 或主比较结果。

## 已确认事实

| 项目 | 本轮观测 | 解释 |
|---|---:|---|
| 计划/终态请求 | 4,000 / 4,000 | 全部请求进入终态 |
| 成功且原生输出合同匹配 | 4,000 | 不替代 adapter 数值身份验证 |
| 请求失败 | 0 | 不等于后台机制无失败 |
| 物理 GPU 占用 | 16,103.305610239011 GPU-s | 四个物理分配均有释放记录 |
| 到达前 / 到达窗口 | 48.10997727900394 / 15,754.069256878807 GPU-s | 互斥时间窗 |
| drain / GPU 清理 | 204.41446136019658 / 96.71191472100327 GPU-s | 不丢弃尾部及清理 |
| 驻留 epoch | completed 151；superseded 308；failed 1 | 失败保留 |
| 文件准备计划 / GPU 准备计划 | 各 1 条 failed | 与上述失败链相关，不能当三个独立故障 |
| 服务退出码 | 1 | 整体未通过 |
| 启动器验收 | false | 不能只取 4,000 成功数作为通过条件 |
| GPU 释放 / 服务资源域移除 | true / true | 不代表本轮缓存工作区已清理 |
| 缓存工作区清理 | 未完成，保留 | `shutdown_unresolved`，不强制删除后冒称正常 |

采用状态表而非性能排名图，是因为这轮没有通过 Full 机制验收。

## 完整请求时延与资源表

全部 4,000 请求，分位数为 Type-1；以下均为内部观测，不是客户端接收时间。
原始控制器 E2E 与外层终态分开报告，未修改冻结指标 V1 或历史字段。

| 指标 | 均值 | P95 | 单位 |
|---|---:|---:|---|
| TTFT | 128.577329 | 296.831144 | s |
| 控制器结果 E2E | 134.607933 | 304.333317 | s |
| 外层请求终态时延 | 137.004243 | 307.454002 | s |
| 控制器结果至外层终态 | 2.396280 | 8.514045 | s |
| dispatch/admission 等待 | 120.398556 | 289.287844 | s |
| 其中：dispatch window | 106.042384 | 271.484581 | s |
| 其中：runtime slot | 11.749672 | 33.437971 | s |
| 其中：arrival release lateness | 2.606500 | 12.567280 | s |
| service TTFT | 8.178774 | 41.145050 | s |
| 其中：native 提交前 | 7.654598 | 40.961233 | s |
| 其中：native vLLM TTFT | 0.524176 | 1.648090 | s |
| TPOT | 33.906219 | 78.432320 | ms |

均值的互斥项可以分解；各项 P95 不能相加。LoRA IO 是嵌套区间
（均值 6.542618 s），不额外叠加。顶层 parent response pickup/thread resume
字段样本数为 0，记 N/A，不能用缺字段推断零延迟。

| 原生阶段 | 均值（ms） | P95（ms） |
|---|---:|---:|
| engine entry → queue | 278.105241 | 1,066.541628 |
| native queue | 37.187090 | 284.598995 |
| prefill | 208.883371 | 464.261259 |
| decode | 3,230.776515 | 8,744.123839 |
| worker completion notification | 56.307307 | 131.661801 |
| worker → controller completion | 2,743.520146 | 16,291.734545 |

| 完整性与安全检查 | 结果 |
|---|---|
| 请求/输入/prompt/原生 token 合同 | 4,000/4,000；全部 ≥2 tokens |
| 时间恒等式最大误差 | 0.196654 ms；TPOT 重算误差 0 |
| pre-generation selected-source 确认 | 4,000/4,000；tier/clock/order 冲突 0 |
| 已确认层级 | GPU 2,428；HOST 732；NVMe 690；Remote 150 |
| activation | initial 1；natural scale-out 3；replacement/quarantine 0 |
| 远端 UUID 配对/内容发布 | 132/132；线上字节两端均 306,360,162 |
| 解压内容验证字节 | 5,139,892,912；请求路径打包 0 |
| 资源采样 | 4,752；服务峰值 39,357,952,000 B |
| 主机最小可用内存 | 82,589,106,176 B |
| high/max/OOM/swap/保护告警 | 全部 0 |
| 冻结来源 / 历史保护 | 69 个来源核验；147 个历史对象零变化 |

## 开发期对照与解释

仅与既有普通 Full9（D112）作描述性核对，各 n=1，不做 CI/正式排名。
同输入、配置及 D89 profile，唯一生产修改是 D114 非阻塞 native RPC；
运行顺序和闭环状态并未随机化，不能用两点推断全部变化的因果效应。

| 指标 | D112 Full9 | D115 Full10 | 方向 |
|---|---:|---:|---|
| 平均 TTFT（s） | 271.850703 | 128.577329 | 下降 52.70% |
| P95 TTFT（s） | 507.604630 | 296.831144 | 下降 41.52% |
| 物理 GPU-s | 17,920.278385 | 16,103.305610 | 下降 10.14% |
| 平均 TPOT（ms） | 32.354962 | 33.906219 | 上升 4.79% |
| 请求合同成功 / 后台收尾失败 | 4,000 / 0 | 4,000 / 1 | 新失败必须解决 |

1. **观察：** 等待仍主导平均 TTFT，原生 vLLM 首 token 阶段不是唯一瓶颈；
   有时延下降，也有 TPOT 上升及后台失败，全部保留。
2. **解释：** D114 最小 TCP 因果测试证明消除了特定 executor 依赖；本轮
   完整回放说明候选能服务全部请求，但不能据此宣布整个 Full 合格或最佳。
3. **研究含义：** 暂不能作为 G1/G2 胜出证据；warm SLO、adapter 数值身份、
   完成通知共同边界仍未合格。资源积分完整不等于正确性资格通过。
4. **下一步：** 对下面这一条驻留失败做最小因果验证，然后返回 Full/7B 主线。

## 失败时序和当前解释边界

唯一失败驻留 epoch 的记录：

- target：`ieee-activation-8c3c1220292d452ab5d3c43e3e1eb2b5`。
- 开始：单调时钟 `124205.662886491`。
- 最后请求终态：`124266.217899361`。
- epoch 失败终态：`124266.729466565`，晚约 `0.511567204 s`。
- 原因：`native file pressure subscription is not available`。
- 路径：residency → 文件准备 → GPU 准备的 native HOST staging → 共享
  文件压力域 `attach`；收尾保留该异常并报
  `IEEE shutdown preserved a failed residency epoch`。

失败与收尾时间相邻，提示需检查准备任务与订阅退出之间的协作，但时间关系
不是根因证明。`SharedFileTransferDomain.attach` 对非 attached 状态报错；
仍需区分目标订阅失效、旁路广播目标退出和其他状态不确定原因。
不能删除该检查、吞掉异常、延长等待或盲目重跑来获得通过。

## 来源与数据保护

- 运行提交：`2697758908ab644f84c191a0318f01cbea8f7a55`，已经备份。
- D114 仅改变 native proxy 通信方式；D115 沿用 D112 的输入、配置和
  初始化 profile，仅使用新的输出与缓存工作目录。
- 原始目录：`results/ieee_tc/p2_backend_qualification/d115_20260929/`。
- 请求成功后收尾失败，原始请求保存在
  `3b_full_w0_full10/launch.launch/physical_deployment/main_outcome.json`
  的 `completed_scenario_windows`，**不是正常运行的 standalone result**。
- 原文件 `9187872294` 字节，不一次性载入内存、不重写、不重新生成请求。
- 元数据在最终顶层属性前的精确偏移 `35696635` 字节；有边界断言的
  元数据检查耗时 0.75 s、峰值 RSS 176116 KiB、退出 0，high/max/OOM 均 0。
- 元数据初步汇总：`full_full10_metadata_preliminary.json`，SHA256
  `8c3fd75be251a70b0cfac1961e4170b4cb0a52a2311841ba5d9553c867290239`。
- D96 流式 request 投影只适配已存在的失败保留 schema；所有原始错误、
  非 request 元数据及全部请求均保留，重复 native inventory 不进入内存。
- 先前准备的正常成功路径分析脚本从未执行，已明确标记 `NOT_RUN_normal_path`。
  不将其断言失败视为新的实验失败，也不将它用于加载该大文件。
- 远端 3B/7B/监控服务已于 01:50:30 按实际 invocation/PID 停止；本地空
  auxiliary 资源域随后停止。发布缓存未改动。
- 远端 transfer journal 已复制一次，SHA256
  `57f6b8c6e1ec130fbbf108b521fe30006d4b42b32178b64dc22cc9f6e7cb6f59`；
  monitor SHA256 `24ce8f978527f589c3f9b0caed4e687e7957ff983acc0b2ff710c0d78f342ab3`。
  两者与远端原件 SHA 一致，随后已完成上述 132 条逐 UUID/字节关联。
- 唯一流式投影完成：1,124.05 s，峰值 RSS 277632 KiB，退出 0；输出
  `full_full10_outcome_projection.json` 为 69,694,371 B，SHA256
  `1683109776c5c0352ee7ec14f2064b82b10e688c1e7614a14cad0c17bd90e7fa`。
- 完整汇总完成：29.46 s，峰值 RSS 200264 KiB，退出 0；原始大文件仅顺序
  哈希，不整文件 JSON 加载。三个分析 scope 均 exact-owned 清理，high/max/OOM 0。
- 最终汇总：`paper_results/ieee_tc/p2_backend/20260929_d115_3b_full_w0_full10.json`，
  SHA256 `fd3e460a1cd676f2c7cc282dde1a2b7074b81efa1a758b95345a615b7db9768b`。
  原始、投影和最终汇总均保留，不再次提取或重跑来覆盖失败。

## 发布前检查

- 70 个生命周期/回放/原生计量检查通过（1.968 s）；真实 jq 小型 fixture
  验证失败状态、metadata、全部请求及 source proof 保留，不重提取原始数据。
- 再核验 69 个冻结来源、34 个小型/中型 curated 来源与 147 个保护对象。
  9.19 GB 原件复用已经完成的顺序哈希，并核验大小/mtime；不声称二次哈希。
- `20260930_d115_evidence_verification.json` SHA256
  `d95a3d95c6b70877ad4699a9c77444294cc173233070673f212c8c66f36e4d06`。
- 37,754 B 的 `20260930_d115_analysis_sources.tar.gz` 保存 50 个小型来源，
  每个成员 SHA 校验通过；不包含大型原始结果、权重、token 或私钥。
  SHA256 `6c949b4ad21ae26b28e70736fbf482fac0fcb43b3cf093d7c9462a5433ac5355`。
- 首次打包检查错误匹配了检查代码自己的字符串，未完成的压缩包及脚本另存
  attempt1。修正为匹配真实 PEM 行头后打包成功；没有重跑实验或改变结果。
- 最后 evidence scope 已验证 exact-owned/empty 后停止，内存事件全部 0。
  没有遗留 GPU、远端服务或分析任务；保留的缓存工作区不冒称清理完成。

## 下一步

备份上述证据；然后对上述具体失败提出最小可证伪验证。当前不接受 D114
为完整 Full 改进，不启动第二轮 GPU 回放；7B 与正式主矩阵仍待推进。
