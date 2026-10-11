# D415 — vLLM 0.30.0 原生输出事件观测

日期：2026-10-11。状态：CPU 组件验证通过；尚未接入完整服务入口，
尚无新 GPU 性能运行，不是合格的 Resident 预算参考。

## 问题、历史与边界

主仓库 D412 已确定 D395/D396 的首 HTTP payload/响应结束计时不能与
PrimeLoRA D371 的原生 token 事件直接比较。原始结果保留；不离线编造缺失
时间戳，不再原样运行旧 runner。原生计数、时间、工件、准备期、服务资源
归属和实际 GPU 释放均完整之后，才允许新的完整参考进入比较。

本次复用现有 `scripts/ieee_tc_serverless_measurement.py` 的纯观测组件；
历史起点为 `26b1628733fbca93e4f431a24386c19b7babcefa`。ServerlessLLM
0.10.2 入口仍独立检查其原 SHA；没有将其调度、扩缩容或存储策略移植给 vLLM。
用户已有 `replay_openai_trace.py` 和 relayserve 修改完全不动。

可证伪假设：vLLM 0.30.0 的原生统计及累计 token 列表是可变对象；当异步
消费者滞后时，某次输出的对象可能对应后来状态。只复制统计而不处理 collector
合并仍会使累计 token 数与统计快照不同步。应在输出生产事件复制观测，并在
原生合并后携带同一事件的快照；不能改用客户端首字节或完成时间兜底。

## 官方代码和实际安装

2026-10-11 对三个官方 v0.30.0 文件进行在线核查。下载内容与本机既有安装
逐字节 SHA 相同，AST 也相同；没有安装、升级或修改 site-packages。

| 文件 | 官方/本地相同的 SHA-256 |
|---|---|
| `v1/engine/output_processor.py` | `e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329` |
| `outputs.py` | `346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425` |
| `v1/metrics/stats.py` | `af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3` |

原始 [output processor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/output_processor.py)
向输出提供 stats 引用；累计 completion 的 token 列表也可共享。
[RequestOutput.add](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/outputs.py)
合并/替换 completion，但不更新 metrics。
[原生 stats](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/metrics/stats.py)
区分 wall-clock arrival 和 engine-core monotonic 事件；两者不能直接相减。

## 最小观测改动

- 新增显式 `install_v030_snapshot()`，只对上述版本和三个 SHA 生效。
  未知版本/来源拒绝，不放宽旧 0.10.2 的版本门槛。只在当前 Python 进程
  包装两个输出函数，不落盘修改已安装的 vLLM。
- 复用原有 producer/collector 快照逻辑，0.30.0 额外复制累计 token ID
  列表，并保留实际输出状态的 adapter name、整数 ID 和路径。
- 仅支持已冻结的单序列、累计文本输出；无原生 stats、数量不一致、错误
  adapter、错误时序、非 length 终态均不能转作成功观测。
- 无新 token 的完成通知保留最近 token 事件的时间，即使消费者只读到合并
  后的终态。原生 stats 本身不改写，完成通知尾部仍计入 E2E，不进入 TPOT。
- 复用主仓库 `NativeV1TokenTimeline`。客户端 SSE 观测保持另一测点，不能
  与原生时钟混名。单 token 的 TPOT 仍为 N/A。
- 无调度、路由、加载、缓存、采样、超时、重试或评价阈值变更。观测适配的
  CPU 开销仍在未来服务域内，不称为性能优化或免费工作。

## 验证和保留的失败

新增测试 `tests/test_ieee_tc_vllm_measurement.py`，从上述安装文件提取并
原样编译实际 producer、collector、RequestOutput 和 stats 的方法；不导入
推理引擎、不加载权重，不把测试中的 token 事件说成性能数据。

测试确实复现了未观测版本的对象别名和只合并 token、不更新 metrics 的行为。
覆盖慢消费者、连续事件、空文本但有 token、独立/合并完成通知、单 token、
异常传播、请求身份、adapter name/ID/path、缺失统计、版本/文件身份和幂等安装。

首次 11 项测试有 1 项测试预期错误：测试期待 `get_nowait()` 返回异常对象，
但官方实现实际抛出该异常。仅将测试改为核对原异常被抛出；未修改原生行为。
随后 76 项组合测试通过（5.663 秒）。补充拒绝错误终态/时序、混合请求及实际
collector 合并后的 adapter 检查后，完整组合 **112 项通过（7.803 秒）**：

```text
tests.test_ieee_tc_vllm_measurement
tests.test_ieee_tc_serverless_measurement
tests.test_ieee_tc_serverless_launch
tests.test_ieee_tc_serverless_router
tests.test_native_stream_timing
```

CPU 测试沿用 systemd scope：MemoryHigh=3GiB、MemoryMax=4GiB、swap=0、
AllowedCPUs/affinity=2,3,26,27。在既有 vLLM 0.30.0 环境中另做实际模块
导入和两次幂等安装，CUDA_VISIBLE_DEVICES 为空，成功返回相同来源回执；
未启动 engine，前后 GPU 计算进程清点为空。没有性能实验、远端服务操作、
缓存重发布、数据/负载生成、删除或历史结果覆盖。

## 尚未完成及下一动作

此组件当前尚未被 vLLM 正式服务 runner 调用，不能提前宣告计量资格通过。
继续复用既有 runner/common launcher，完成服务域内 ordinary routing 与
按需远端获取、部署前接收/独立 open-loop 到达、完整物理 lease 与真实释放。
原生计时与 HTTP 完成在同主机可靠时钟中关联，逐请求核对输入/输出/工件。
保持既有公共策略并记录适配差异，不引入 PrimeLoRA 控制机制。

推理机现有可用约 146.9GiB，低于 150GiB 新重型任务底线；先核验 D414
已结束的下载缓存再仅回收可重建 objects，保留来源、回执、日志和原始权重。
随后受影响路径短资格及一个预登记完整 W0；不得选择更快重复。
PrimeLoRA D371 与 ServerlessLLM D414 继续封存，W1/W2/G2、3B 及完整计划
其余实验均未因本 CPU 检查而完成。一次性远端发布缓存复用 D78/D80，不重复创建。
