# D954: PrimeLoRA 7B request-path bottleneck audit

日期：2026-10-30
范围：PrimeLoRA 7B、W2、4,000 请求、`fixed_length_greedy_v1`；仅诊断，不是新的正式结果。

## 当前证据

D931 的请求级原始记录显示：

| 阶段/指标 | 均值 | P95 | 解释边界 |
|---|---:|---:|---|
| 总体 TTFT | 1,215.95 ms | 2,997.74 ms | 计划到达至首 token |
| ingress queue wait | 424.87 ms | 1,048.37 ms | 到达释放后到全局准入窗口 |
| runtime-slot wait | 251.94 ms | 836.47 ms | 已进入准入窗口后至受保护 runtime slot |
| nested confirmed routing | 42.78 ms | 234.78 ms | runtime-slot wait 内嵌，不可再次相加 |
| LoRA I/O | 237.60 ms | 1,420.77 ms | 只在实际非 GPU tier 请求上出现 |

请求级 `ingress_queue_wait_ms` 与总体 TTFT 的 Pearson 相关约为 0.892，
`runtime_slot_wait_ms` 与总体 TTFT 的相关约为 0.416。按 dispatch tier 分组，
HOST 组的 nested routing 均值约 113.7 ms，而 GPU 组约 37.0 ms；这与每次
确认路由都要经过文件 owner 的 `source_snapshot(adapter_id)` 相一致，但仍不是
单独的因果分摊。

D931 的确认快照缓存已经避免了多数副本的重复 native scoped read，但
`ScenarioRunner._ieee_request_snapshot` 仍按以下顺序执行两个互相独立的只读
观察：

1. 并发收集所有候选 replica 的 native source observation；
2. 收集结束后，再通过文件 owner 读取当前 adapter 的 confirmed file source。

随后才构造一个不可变 candidate view，并由选中副本执行既有的 owner/epoch/source
revalidation。因而当前顺序把两个独立读取的等待时间串联到了同一个请求的 routing
区间。文件 owner 的 `source_snapshot` 本身必须保留双重 stat/签名验证；本审计不
建议 TTL、last-known tier、请求外预取或绕过 selected-copy revalidation。

## 可证伪假设

在不改变九个 IEEE 公式、确认层级语义、缓存命中/传输次数、admission、预算或
生命周期记账的前提下，将 native source collection 与 confirmed file-source
snapshot 作为两个独立的只读任务并发启动，能够把请求级 routing wall time 从

\[
T_{route}=T_{native}+T_{file}
\]

降低到接近

\[
T_{route}=\max(T_{native},T_{file})+T_{join},
\]

并且不会增加 stale/membership rejection、selected-source conflict、文件 owner
错误或 native contract 失败。如果这组条件不成立，候选立即撤回，不叠加其他调参。

该假设只针对控制面等待重叠；它不把远程传输、LoRA I/O、GPU 解码或任何远端
预发布打包时间从用户延迟中移除。远端传输仍按冻结的指标协议计入请求路径。

## 语义不变量

- 两个观察均为本次请求触发的真实读取；没有 TTL、预测、最近一次层级直接路由
  或共享未来需求。
- membership/engine identity 在两项任务完成后重新核对；发生扩缩容或引擎替换
  时，整个 candidate view 被丢弃并按现有重试逻辑重新收集。
- 文件 snapshot 的 captured time、owner/epoch、来源 footprint 和签名验证
  保持原值；selected-copy protection、admission、lease、budget 和 execution
  commit 不变。
- 任一任务失败时传播原始异常；另一任务被取消并 join，不留下后台 owner/engine
  RPC。
- `routing_decision_us` 仍记录完整请求级控制路径；重叠不改变 stage identity，
  只减少其真实 wall interval。

## 外部设计依据

官方 vLLM 文档把 async scheduling 描述为降低 CPU 开销、改善延迟/吞吐的实验性
路径，同时明确其受特性约束；官方异步引擎也将请求加入 tracker 后由后台 engine
loop 处理。这里借鉴的是“独立控制/执行等待可以并发”的实现原则，不改变 vLLM
的 scheduler、LoRA kernel 或生成合同：

- <https://docs.vllm.ai/_/downloads/en/v0.6.0/pdf/>（async scheduling、LoRA、
  async output processing）

## 最小验证

代码变更仅限 `_ieee_request_snapshot` 的两个只读任务启动/汇合与异常清理，以及
选中源保护路径对同步/异步 owner 实现的能力判断；后者只修正调用约定，不改变
保护检查。现有 route/ownership/contract 单测已通过（44/44）。一个 CPU-only
夹具把 native observation 和 file snapshot 各延迟 50 ms；实测总快照时间约
52.7 ms，而串行基线应约 100 ms，说明任务确实重叠。该夹具不是性能结果，只是
验证并发结构。下一步才进行一个新的 100-request 7B validation execution key。
记录并比较 D931 的：总体/服务 TTFT、dispatch wait、runtime-slot
wait、nested routing、远程传输次数/字节、confirmed cache counters、stale/selected
conflict、4,000/100 native contract、物理 lease 释放和磁盘/内存峰值。

只有在 100-request 中观察到 routing wall time 下降且所有语义/生命周期条件不变时，
才允许一次新的 4,000-request 7B replay；否则候选撤回。单次 validation 不作为
G1/G2 或最终正确性结论，也不覆盖 D931/D935/D937 等历史结果。

## 与主线的关系

D931 的 98.1% 单一 TTFT-SLO 通过率只说明当前运行接近服务可行区间，不能替代
`N_correct`、共同 TTFT/TPOT SLO 与物理 GPU-s 的完整资格。因此继续优化的理由
不是追逐单一 TTFT，而是减少已被阶段证据定位的控制面等待，同时保留 PrimeLoRA
的层级驻留与远程 miss 减少机制。外部 baseline、图表和论文排名在本审计完成前
不启动新的重型回放。
