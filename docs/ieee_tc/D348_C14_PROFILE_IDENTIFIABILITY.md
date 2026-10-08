# D348：既有 C14 测量不能直接识别延迟最优软目标

2026-10-08，7B 开发主线。仅复用 D343/D344 的已封存观测，未启动 GPU、
未改工件/负载/共同评价阈值、未制作图。扩展既有完整分析器，不重建框架。

## 结果与判断

| 已有证据 | 本次核查 | 可支持的结论 |
|---|---|---|
| D343 同 C14 后端的 8/14 波次 | 66 条测量均为 760 输入、256 输出 token | 支持长请求吞吐/TPOT 权衡，不覆盖短输出 |
| D344 完整 profile | 1545 条测量中 333 条 GPU-ready；实际接纳计数覆盖 1–14 | 支持论文定义的条件初始化，不等于恒定并发服务曲线 |
| 接纳计数 1 组 | 28 条；decode 期间平均又到来 9.32 个接纳 | 接纳快照不能代替完整服务期间负载 |
| 接纳计数 14 组 | 9 条；无后续接纳，平均 TPOT 33.00 ms | 不能据此推断高并发总比低并发快 |
| 短输出≤24、计数 1–6 | 75 条、15 次暂定 TPOT 越界 | 含连续新到请求干扰，非稳态低并发 |
| 短输出≤24、计数 9/10 | 6 条、0 次暂定 TPOT 越界 | 长背景＋后到短请求；与上一行不是同条件对照 |

分析保存完整 333 条记录及全部计数组。按已封存 native prompt 计数匹配
D170 暂定阈值；gap CSV 缺少的 native prompt 字段从该次原始记录恢复，
不用 canonical length 或邻居样本填充。warmup 不进入统计。

平均其他接纳请求数来自同源同波次 admission→last 区间在 decode 期间的
时间积分；peer admission→first 是服务前段包络。二者均**不是实际 kernel
batch 或纯 prefill kernel 时间**。不同 source 的相同 request ID 不混合，
不同 clock 不相减。单进程三轮不构成独立重复，不给 t-CI。

因此拒绝从当前表格插值/拟合一个“latency-optimal target”。该值明确为
null。也不因为覆盖了所有 service classes，就声称覆盖了所有控制工作点。
这不推翻现有 profile 对论文指定观测类的初始化用途。

## 主线下一步

D346 相对 D336 同时改变了真实容量、对应 profile 和容量派生的控制目标。
保留 C14 后端及其 profile，单独恢复 D336 的**绝对负载控制目标**，是比再
扫描硬容量更直接的对照。它用于检验软目标与物理容量耦合是否造成服务损失，
不是宣称从上述 profile 推导出最优点。详见 D349 工作点文档。

原始资料：2026-10-08 核查 [Ray 2.59 目标与最大在途数](https://raw.githubusercontent.com/ray-project/ray/ray-2.59.0/doc/source/serve/advanced-guides/advanced-autoscaling.md)、
[vLLM 0.30 prefill/decode 预算](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)。
它们支持区分软目标、物理容量、预填充干扰；不证明 Prime 的因果机制或最优值。

产物：`paper_results/ieee_tc/p2_backend/d348_soft_target_evidence/analysis1/`。
65 项 CPU 测试通过，分析命令成功；峰值 508760064 B，high/max/OOM/swap 均 0。
N_correct、共同参考、正式 G1/G2 仍开放；没有新增性能回放。
