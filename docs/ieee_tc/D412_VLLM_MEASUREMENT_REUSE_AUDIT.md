# D412：vLLM 旧结果复用与下一入口的计量核查

日期：2026-10-11。状态：**只读审计完成，未实现、未启动新的 vLLM 实验**。
这是 D397 更正后的具体入口核查，不是替代完整实验计划，也不改变实验顺序。
D411 ServerlessLLM 完整 W0 仍在运行；先完成其收尾、完整分析和备份，再进行
vLLM 的受影响路径修正与资格验证。本次没有修改运行源码、配置、环境或远端服务。

## 结论与复用边界

旧 vLLM 的权重、源 trace、环境、输入与原生 token 数量证据可以保留复用，
但 D395/D396 缺少合格的首末 token 时间和完整服务资源归属，不能离线补出
共同预算参考。不能选较快的 D396，也不应原样启动第三轮。

只接入已经存在的 SSE token 解析器也不足以完成修正：**客户端收到 token 的
时间和后端原生 token 事件时间是不同边界**。PrimeLoRA D371 使用后者。
下一入口必须保留原生事件和到达时钟的对应证据；客户端流式接收另作交付
诊断，不能换个字段名后与 D371 的计时直接视为等价。

共同 SLO 仍是 D394 冻结阈值，G1/G2、95%、75%、60 秒准备窗口和单轮规则
均不变。计量修正不授权变更 vLLM 的普通路由、原生缓存或后端调度策略。

## 本次核对的身份和历史

- 主仓库检查时 HEAD：`f39130e34a5570aca465b30beb358d196c23f128`。
- baseline 仓库检查时 HEAD：`f8af38bcb7a401114e3782e54bd5b1396a40fc76`。
- baseline 的 `scripts/replay_openai_trace.py` 是既有用户修改，未编辑、未 stage；
  其历史保护身份见 EXECUTION_STATUS。源码行号描述该工作文件，不冒充纯 HEAD。
- 历史更正：[D397 Resident 审计](D397_RESIDENT_AUDIT_CORRECTION.md)；
  更早合同差异：[D213](D213_RESIDENT_REFERENCE_PROTOCOL_AUDIT.md)。
- baseline 历史依次包括共同业务时钟 `dd1ca72`、并发启动 `effb149`、实际限制
  记录 `1ebe507`、CPU 归属 `1115b16`、SSE 解析准备 `fecf41f`。这些改进保留，
  但各自不能替代完整计量资格。
- 只读检查既有环境
  `/home/qhq/.venvs/primelora_vllm0300_tc_20260925/lib/python3.12/site-packages/vllm/`；
  没有导入推理后端、安装依赖或升级版本。

## 入口与证据缺口

以下 baseline 路径相对 `/home/qhq/serverless_llm_baselines`，主仓库路径单独标明。

| 边界 | 实际代码/历史证据 | 处理结论 |
|---|---|---|
| 首 token | `replay_openai_trace.py:1383` 在首个非空 HTTP payload 设置时间，SSE 内容在响应结束后才解析 | 旧日志无逐事件接收时钟时无法离线恢复；不能把 HTTP 非空当生成 token |
| 末 token | 同文件 `:1653` 的 fallback 使用响应完成减首 payload | 完成通知尾部不能放进 TPOT；`tpot_observed` 布尔值不是原生边界证明 |
| 下载和注册 | 同文件 `DynamicLoRALoader.acquire` 调用 `RemoteArtifactFetcher.ensure` 并向加载 API 注册；runner `:1468` 将整个 replay 放进辅助域 | 下载、解包、注册、请求路由应归服务域；只有到达发生和外置观测留在辅助域 |
| 初始排队 | `run_vllm_fair_experiment.sh` 先完成服务 ready 检查，随后在 `[4/5]` 启动 replay | 需要先有独立请求发生器和部署前就绪的接收入口，不能等全部 engine-ready 再注入 |
| GPU 生命周期 | `resident_protocol_receipt.py:init/finalize` 使用整体包络，清点前取 release 时间，并直接设置 `release_after_worker_exit=true` | 保留旧记录为诊断；使用共同 owner/lease、进程出生身份和真实释放见证，不由显存阈值制造释放事件 |
| 编译初态 | D395/D396 编译与启动日志不同，历史缓存未形成同一冻结初态 | 复用合法磁盘编译缓存可以，但启动前登记其身份；不可将其当成 GPU/HOST/LoRA 已就绪或免费准备 |
| 资源保护 | 旧 wrapper 有部分内存/CPU 限制，不等于共同独立 watchdog 完整通过 | 复用现有共同安全入口；核验实际 worker、服务辅助任务、PSI、swap、磁盘和退出 |

当前 `run_vllm_resident_reference_v1.sh` 的注释声称 protocol-compliant，
不是验收证据。它转入上述 fair runner；D397 对其结果的资格更正继续有效。

## 官方字段核查：不要将 service TTFT 当作用户 TTFT

核对 vLLM v0.30.0 官方源码及既有安装中的同名函数：

\[
\texttt{time\_to\_first\_token\_ms}
=1000(t_{first}-t_{scheduled}),
\]

\[
\texttt{mean\_itl\_ms}
=1000(t_{last}-t_{first})/(n-1).
\]

前者不包含到达至调度的全部等待，不能直接填入本项目的
\(TTFT=1000(t_{first}-a)\)。即使同时取得 engine queue span，仍需证明
到达、入口、物化、后端提交之间没有遗漏或重复。该 API 的相对时长字段
本身也不提供完整绝对时钟关联。
[官方 timing builder](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/entrypoints/generate/base/serving.py)

官方 completion stream 区分输出 `token_ids` 与 prompt token IDs；启用 usage
后可在尾部发送没有输出 token 的统计消息。chat stream 还存在 role-only
消息。因此必须解析原生输出身份后才更新时间，不能计首字节或最后一个消息。
[completion 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/entrypoints/openai/completion/serving.py)、
[chat 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/entrypoints/openai/chat_completion/serving.py)

本地相应位置为 `entrypoints/generate/base/serving.py:50`，以及
`entrypoints/openai/completion/serving.py:465`。行号与远端格式不同，未假设整文件
字节一致；下一执行身份仍须单独记录实际文件和观测适配。

## 已有组件如何复用，不另建框架

1. **原生时间线**：主仓库 `faaslora/metrics/metrics_collector.py` 的
   `NativeV1TokenTimeline` 已验证 queued/scheduled/first/last、原生输出数量、
   clock ID、终态和完成尾部。复用这类纯测量校验，不引入 PrimeLoRA 控制逻辑。
   原生 mutable stats 必须在与输出更新对应的时刻快照，不能读取后来变化的对象。
2. **客户端诊断**：baseline `scripts/native_stream_timing.py` 已有增量 UTF-8/SSE
   解析和七项 CPU 测试源码。它记录收到输出 token 的首末时间、事件数和响应尾部；
   当前 replay 尚未调用它。它不独自验证目标长度、正确 adapter、成功终态或全部
   请求，调用方仍须核验；单 token TPOT 为 N/A，多 token chunk 不代表 kernel
   逐 token 时间。本轮只读测试源码，没有重新运行测试。
3. **独立到达与保护**：主仓库 `scripts/ieee_tc_preflight.py:gated_launch` 已有
   外置 publisher、先于部署的启动握手和独立 watchdog。其 HTTP 路径当前显式
   检查 `prepare_ieee_tc_serverless_stack.py` 身份，不是任意 helper 的即插即用接口。
   复用时必须检查并测试入口约束，不能只更换命令名宣称已经适配。
4. **服务侧接收和物化**：baseline 的 `PreReadyHTTPIngress`、发布缓存客户端及
   既有普通 vLLM loader 提供可复用边界。应把 ordinary routing、必要加载与转发
   放在服务预算内，并保持原先公开策略；不是把整个客户端移入服务组。
5. **物理占用**：复用共同 `PhysicalGPUDeployment` / `PhysicalGPUAllocation`
   的 owner/lease 和 `bind_process`/release 证据。同卡多个进程取区间并集；
   服务守护进程仍持卡时继续计量，不以请求结束关闭账本。

上述是可复用性结论，不是已完成集成。特别是 ServerlessLLM 的原生快照适配
`install_v1_snapshot` 有 vLLM 0.10.2 文件身份检查，不能原样套到 vLLM 0.30.0。
应按已安装版本验证最小观测接点，不复制其 router、autoscaler 或 native store 策略。

## 进入下一次 GPU 资格前必须关闭的事项

- 保留用户 dirty replay 文件；在现有、明确版本化的适配入口扩展测量，不能
  将用户改动悄悄纳入提交，也不能另复制一个完整 replay 框架。
- 先证明 role/usage/空消息/完成尾部不会改变原生边界；源输入、prompt、adapter、
  输出 token 数量和终态逐请求关联；异常保留，禁止无原生计时 fallback。
- 对比客户端接收边界与后端原生边界，明确各自用途；主比较和共同 warm SLO
  使用相容的原生边界，不把不同测点间的未知差额设成零。
- 部署前接收入口、独立 open-loop 到达、共同 60 秒、1,800 秒资格保护，以及
  下载/解包/加载所属服务域必须通过小型非 GPU 合同测试和实际进程读回。
- 确认实际 HTTP fetch 未注入模拟等待。wrapper 的历史 `250` 默认值不等于
  生效限速；同时检查调用链、有效配置和等待日志，不能仅凭变量名下结论。
- 短资格后只启动一个事先登记的完整参考；冻结原生版本、资源包络、普通路由、
  后端参数、编译缓存初态和冷 LoRA 缓存。不得用同键重复选择最快值。
- D411 实际结束、GPU 释放、overlay 恢复、远端所属服务收尾、完整审计和备份
  先完成。届时重新读完整计划及状态，核查启动磁盘和内存，才允许下一重型任务。

## 本轮交付限制

仅新增这份审计和当前状态记录；没有新的性能数据、成功资格、预算参考或排名。
不改论文、九个公式、冻结指标、旧结果、权重或负载；不生成图。
使用 serving-llms-vllm 技能定位部署/流式边界，并以固定版本官方及本地源码
核查；技能中的示例阈值、默认配置和通用吞吐目标没有成为本项目评价标准。
