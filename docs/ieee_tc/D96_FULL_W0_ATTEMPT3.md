# D96：3B Full W0 attempt3 完整失败回放

2026-09-28。开发资格验证，不是正式 M1/M2 比较。全部终态、请求阶段、
状态读取、物理占用与远端传输已完成交叉校验；下一步是因果验证，而非性能排名。

## 立即状态表

| 项目 | 本次观察 | 含义 |
|---|---:|---|
| 计划／实际提交请求 | 4,000／4,000 | 同一完整 W0 到达序列，未缩短负载 |
| 请求终态 | 4,000 | 没有把尚未到达或未完成请求移出分母 |
| 成功且原生输出长度匹配 | 1,417（35.425%） | 不等于 adapter 数值判别资格 |
| 原定 1,800 秒期限超时 | 2,581 | 不是仅离线 SLO 违约 |
| 返回失败、未生成匹配输出 | 2 | req_01312、req_01455；RuntimeError，未发送新的 generation |
| 全部成功要求 | 不通过 | 不进入达标资源／时延排名，不推进 7B 为已合格 |
| 物理 GPU 占用 | 23,090.882559 GPU-s | 完整失败运行的实际资源记录，非成功请求成本排名 |
| 物理租约 | 4 个，全部已释放 | 无 open lease；实际 GPU 进程检查为空 |
| 计量完整性 | measurement_complete=true | 与 eligible_correctness=false 明确区分 |
| 启动器与服务收尾 | 返回码 0，清理通过 | 仅表示执行与清理完成，不表示系统性能合格 |
| 远端与外置监控 | 本次所属服务已停止 | 原 invocation 核验后停止；未触及其他任务 |
| 远端传输关联 | 78／78 UUID 匹配 | 线上 181,071,856 bytes；内容校验后逻辑字节 3,075,482,472；请求打包为零 |
| 服务内存峰值／主机最低可用内存 | 24.559／85.545 GiB | 6,016 次样本；high/max/OOM/OOM-kill 与服务 swap 均为零 |
| 旧投稿保护清单 | 147／147 未变 | 原始失败证据保留，没有重写旧结果 |

## 条件时延与状态证据

下表只覆盖成功的 1,417 个请求（不是完整负载性能，不能用于系统排名或 SLO 达标结论）。
请求内相加的恒等式是用户 TTFT＝调度／准入等待＋service TTFT；分位数不直接相加。

| 量 | 平均值（秒） | Type-1 P95（秒） |
|---|---:|---:|
| 用户 TTFT | 842.922768 | 1,676.405844 |
| 调度／准入等待 | 839.632860 | 1,676.275735 |
| Service TTFT | 3.289907 | 11.242809 |
| 原生 vLLM TTFT | 0.183401 | 0.523259 |
| 用户 E2E | 846.580648 | 1,678.779776 |
| TPOT | 0.018064 | 0.027302 |

- 成功请求中 1,026 个保留了 selected-source 重选记录，合计 13,655 次，最大 124 次。
  这不是所有失败 snapshot 的总数，不能把未保留的失败请求推断为零次重选。
- Source observation：206,707 次调用、102,599 次 collection、4,220 次 RPC invocation、
  104,108 次加入已有 in-flight observation；membership rejection 32，stale rejection 2,461。
  collection 不等于原生 RPC，尤其无可服务成员时；RPC invocation 也不等于已确认原生执行。
- 2,325 次控制观察中，1,210 次为 ready=0 且 queue>0；其中 1,207 次 scale_up
  **全部**返回 no_free_device，另 3 次 no_action。不能解释为扩容器没有发出扩容决策。
- 四次 activation 达到 runtime_ready；residency 64 完成／71 superseded／1 cancelled，
  file plan 68／71／1，GPU plan 27／71／1。superseded 本身不等于错误。
- 两个 RuntimeError 的完整文本均为 `native RPC ownership unresolved; new generation withheld`，
  generation_submission=not_submitted。它们没有新的原生输出；不能由此推出同副本的其他
  RPC 已结束，亦不能据此提前释放该 GPU。

观察支持两个需分别验证的问题：已有成功样本的主要用户等待发生在推理前；
后期 routing 不再接纳副本，但物理 owner 尚未退出，使新的扩容没有设备可用。
当前数据尚不能量化其中每种状态冲突的因果贡献。

## 已观察、尚未归因

- 早期持续积压。四个副本均实际完成过请求，因此不能从旧 dashboard
  的缓存计数或瞬时利用率推断只有单卡服务。
- 05:50 首次观测到请求期限超时。req_00501、req_00521 的终态相对
  计划到达为 1,804.382699／1,800.680911 秒；终态可以晚于取消触发。
- 后续出现两条返回失败，日志类别为 `native RPC ownership unresolved;
  new generation`。终态的 error_type=null 不等于没有失败。
- 成功数于约 06:04 停在 1,417；随后日志中的可服务副本降至零。
  该状态不等于物理资源已经释放；实际释放发生在全部请求终态后的收尾。
- 不能把长排队、状态冲突、取消后的可服务性下降直接归为同一个已证明的
  根因。完整结果中需分别核对状态读取计数、重选、失败观测和副本生命周期。
- 控制记录中有 1,210 次“无 ready 副本且队列非空”的观察，其中 1,207 次
  决策为 scale_up。首条对应 queue_depth=2,061、目标 1 个副本、
  outcome=no_free_device、scheduled=0。因此不能把恢复失败直接解释为
  扩容控制器未响应；需核查被保留的物理 owner 如何退出并使设备重新可用。
- D94/D95/D96 包含多个修正，不能把跨版本变化全部归因于状态读取合并。

## 执行与保护

代码：`dff50a1a34f79887ea14b7283db7b65865fec18b`。未在回放中改动源码、
输入、模型配置、期限或远端服务。部署通知与 D88/D89 配置保持不变。
远端使用已批准的一次性只读发布缓存，无请求路径打包；78 个关联传输已全部校验。

原始目录：`results/ieee_tc/p2_backend_qualification/d96_20260928/`。
终态与物理记录：`3b_full_w0_attempt3/launch.launch/physical_deployment/`。
启动器：`3b_full_w0_attempt3/launch.json`。
完整请求结果：`3b_outputs_attempt3/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d96_3b_full_w0_attempt3.json`。
远端清理：`remote_stop_full_attempt3.log`；本地空辅助组清理：
`local_aux_cleanup_attempt3.log`。远端最终监测与本次传输日志已独立保存。

请求结果约 3.7 GiB，不整份载入 Python 或普通 jq 对象树；使用独立 4 GiB
资源域中的流式投影，保留原始文件。旧 D93 curator 的字段选择与校验可复用，
但不能沿用其 2,417／320 等硬编码计数、interrupted_replays 假设或不完整计量标签。

流式投影退出码 0，耗时 803.62 秒、峰值 RSS 67,584 KiB；分析不占用推理 GPU。
Curated：`paper_results/ieee_tc/p2_backend/20260928_d96_3b_full_w0_attempt3.json`，
SHA256 `4cbd37e0f7d4cd01720b53694ad1f7aa2c2db0f4d56173f84f4dc859a209573a`，
21 项源文件 SHA，包含未修改的完整请求文件与分析脚本。
独立再次核验 21 项 SHA 全部一致；物理生命周期／外部回放的 47 项 CPU smoke
测试通过（1.996 秒）。这些测试验证计量与请求边界，不证明本次推理性能合格。

下一步：备份证据后，以实际保留 reservation、draining 成员、原生退出和物理释放
建立一个 CPU 因果测试。只有真实退出才能重新分配设备；不移除安全检查、不把发送
abort 当完成、不以盲目重启替代生命周期逻辑。暂不重跑 7B／基线、不改变请求期限。
