# D691：实时状态查询的消费者与原生控制交错复核

2026-10-18。3B PrimeLoRA 主线；仅复用 D684 已关闭的四个 core 日志，
没有新推理运行、生产修改、参数选择、远端操作或图表。当前生产保持 D689
精确回退后的实现；D688 的完整退化证据不覆盖，G1/G2 尚未达标。

## 本次结论

1. 不重复 D514 的 GPU 准备前 identity-only 改动。Git AST 核查证实它正是
   本次最初考虑的同一调用点；D516 已有完整反例，D519 已撤回。
2. 当前 D355 控制交错实际生效。在输入轮次开始时已有原生未完成请求的
   9,670 个含 utility 轮次中，每轮恰好一个 utility，没有同轮排空多个
   utility 的现象，随后全部进入原生 step。不能再将旧 D354 的队列行为
   当成当前版本尚未修复的问题。
3. 单个同步操作仍可能很长。上述轮次 utility 的 P95 为 150.207292 ms；
   其中 source_snapshot 的均值为 112.917889 ms，P95 为 180.625658 ms。
   交错生效不等于单次操作不会干扰生成。
4. 现有日志标识 native operation，不区分 controller 中的调用位置。
   不能把 591 次 source_snapshot 的全部 66.734472 s 归给某个调用点，
   也不能据此预测删掉某个查询就能节省同等时延。

## 历史反例与优化边界

D514 在 `_run_ieee_gpu_preparation_plan.execute` 将一次 `source_snapshot`
换成 `source_identity_snapshot`，其余生产 AST 不变。D515 资格后，D516
完整 7B W0 从此前 D476 的 97.425% 联合时延达成变为 94.775%，物理 GPU
时间增加 2.16037%；该候选未接受并在 D519 撤回。单轮自然闭环不证明所有
退化均由该改动造成，也不证明所有视图精简必然无效；但没有新的差异化证据，
不能把同一个改动重新包装为未尝试过的方案。

D463/D474 已精简 HOST 入队/执行身份观察，保留现状，不重复实现。D329
跨调用物理图缓存反例、D333 的 controller 可见中间状态、D352 同事务
物理核验边界继续有效。D687 元数据物化组件改善未通过 D688 完整结果，
也不能借本次查询耗时恢复该候选。此次不改 SLO 或扩缩容阈值。

## 九个现有完整查询边界

以下行号绑定 `scripts/run_all_experiments.py` 的 SHA
`f677e6010ee9a3ba9bfbd391bad03eace9719acce23e3d7e036d0dbf921ad0e4`。
AST 枚举全部显式 `source_snapshot` / `routing_source_snapshot` 调用，
不是整个 Python 程序的形式化数据流证明。

| 行 | 消费者 | 当前处理及理由 |
|---:|---|---|
| 5073 | routing source view | 保留；准备类别和 GPU 几何消费实际 footprint |
| 14045 | deferred HOST capacity wake | 保留；使用 allocator accounting 和 exclusive storage |
| 15196 | activation layout | 保留；核对实际初始化 GPU 布局 |
| 19442 | GPU plan execute | 保留；同 D514 已撤回候选，不重试 |
| 19923 | steady-state planner | 保留；全注册图、alias/capacity 与 router 发布有实际消费者 |
| 20026 | opt-in proactive diagnostic | 保留；诊断需要物理 before-state，不混作主回放开销 |
| 20336 | native file fallback protection | 保留；后续 hold_host_source 并不重新测完整物理图，不能套用“后面必有同等检查”的理由 |
| 20451 | prepare_gpu_source | 直接消费者是来源身份/失效判断；本轮没有该调用点独立耗时，不机械扩展精简 |
| 20491 | initialized GPU objective | 保留；实际布局/容量参与目标构造 |

identity endpoint 仍执行新鲜 `owner.source_snapshot()`，不等于 TTL 缓存；
但“能用更小视图表达”与“此处修改将提高完整系统效果”是两个问题。
真实准备中的 KV、E(t)、footprint、allocator、引用与释放检查不能删除。

## 已有日志的新离线结果

复用 `summarize_core_host_spans(..., include_input_turns=True,
host_inventory=True)`，检查完整 schema、嵌套、同一 owner 时间线及 close。
四个完整流共 312,160 spans，无截断；没有新增插桩或重跑 D684 前缀。
23 项既有 analyzer 回归通过，CPU 资格不当成 GPU 性能实验。

| 输入轮次分组 | 轮次数 | 同轮多个 utility | 后续为 step | step 返回 True / False |
|---|---:|---:|---:|---:|
| 全部非空轮次 | 10,706 | 473 | 10,706 | 10,441 / 265 |
| 入轮已有未完成请求的非空轮次 | 10,020 | 0 | 10,020 | 9,969 / 51 |
| 入轮已有未完成请求且含 utility | 9,670 | 0 | 9,670 | 9,619 / 51 |
| 其中含 source_snapshot | 591 | 0 | 591 | 585 / 6 |

在入轮已有工作的轮次中，首个 utility 后的额外 client 耗时总计为 0。
473 个多 utility 轮次均在入轮原生请求数为零时开始；这不是证明这些轮次
全程无业务、没有 pending admission，或不会延迟新请求。观察只在 root
入口采集 work，不能补造每个 child 时刻的工作状态。

原生 `step` 返回 True 是 `model_executed`，不等于某条请求产出 token，
也不是 CUDA kernel 或 GPU 空闲测量。51 个 False 的原因本次未作推定；
它们不能被删除，也不能在没有请求/调度证据时当作调度错误。

| 同步 utility 耗时，入轮已有工作 | 均值 ms | P50 ms | P95 ms | 最大 ms |
|---|---:|---:|---:|---:|
| 全部 9,670 个 utility 轮次 | 26.715388 | 0.662259 | 150.207292 | 1,278.086131 |
| 591 个 source_snapshot 轮次 | 112.917889 | 103.942727 | 180.625658 | 508.889159 |

有观察开销的描述性结果，非普通完整 W0 成绩。嵌套时间不相加，不从 TTFT、
TPOT 或 GPU-s 扣除；不预言吞吐或服务达成改善。独立数值正确性仍未知。

## 原始依据、变更与安全

本轮在线复核 [vLLM 0.30 UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
和 [EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
worker utility 在该执行器中同步执行，core 输入处理与模型推进存在明确边界。
这支持检查单次事务占用，而不是性能加速的承诺。本机 D355 的实际交错
须以上述日志验证，不能只引用官方原生实现判断。

system-profile 用于区分调用边界与 kernel/因果时延，analyze-results 用于
保留反例和未识别项；github-sync 用于五个明确路径的校验备份。

| 新增/修改内容 | 范围 |
|---|---|
| raw `d691_20261018/diagnose.py` | 复用原 analyzer，枚举调用点、核对历史 AST、汇总既有 input turns |
| raw `close.py` / `publish.py` | 复用精确资源身份关闭与明确路径备份，不新建实验框架 |
| 本文、执行状态、curated summary/evidence/checksums | 本次审计证据，非论文修改 |

analysis 资源域峰值 314,384,384 B，high/max/OOM/swap 均 0；已核对递归
空进程后关闭。147 个受保护投稿文件及用户 manifest SHA 未变，GPU 空闲。
不修改原共享 k3s；HydraServe 已完成隔离适配与结果，不重复基础部署。

## 收敛决定与后续主线

本次排除“当前仍在有推理工作时同轮堆积多个控制调用”的解释，也避免重做
D514。下一项必须针对**单个实际事务的工作量或发起原因**提出新证据：
优先从既有准备计划/attempt 记录区分有效准备、来源竞争失效和已就绪复用，
并核对是否存在不改变论文选择/准入语义的重复发起。先复用日志，不再增加
同一个 GPU profiling 前缀；没有新证据不做批量 identity 替换、事务盲融合
或另一轮无变化 W0。此为下一有界审计，不是已接受的新优化。

3B 尚未达到约定目标；完整跨场景/最终版本 7B、基线、dLoRA、A1–A5、
S1–S13、成本/CE、两轮审稿说明和最终图表全部保留。没有完成主线、宣称
最优或将审计冒充新的性能实验。
