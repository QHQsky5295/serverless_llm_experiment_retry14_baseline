# D1166：7B dispatch-time readiness 证据闭口

日期：2026-11-09。本文档只整理已经封存的 D278 与 D279，不启动新的 GPU
运行，不修改任何历史结果、配置、负载或远端交付缓存。

## 结论

D278 与 D279 是同一 7B readiness 协议下的两个独立冷运行。两次运行均采用
八个 500-request phase、phase 间 2 秒空闲、相同 seed-42 trace、adapter
subset、fixed-length greedy 生成契约、vLLM 后端、confirmed routing、handoff、
hierarchical residency 和 admission 配置。两次只改变了结果、cache 和运行
身份路径；没有把请求记录从一个运行复制到另一个运行。

| 证据 | D278 | D279 | 判定 |
|---|---:|---:|---|
| 计划/终态/成功请求 | 4000/4000/4000 | 4000/4000/4000 | 两次均完整 |
| 完整 phase | 8 | 8 | 两次均完整 |
| 自然 scale-up 事件 | 10 | 10 | 两次均有 |
| dispatch 前 tier 字段 | 4000/4000 | 4000/4000 | 覆盖完整 |
| tier 不变量冲突 | 0 | 0 | 未发现冲突 |
| first-service 观察 | 10 | 10 | 合并后 20 |

因此，受控 readiness 证据可以闭口为：在该协议和已测实现中，selected
replica 的 dispatch 前 tier 快照能够与 activation epoch、实际 dispatch
以及首 token 事件绑定；两个冷运行合计提供 20 个真实 first-service 观察。
这不是新的 Full 性能排名点，也不是 G1/G2、CE 或跨系统优越性证据。

## 论文可支持的语义

论文可以说明：PrimeLoRA 的 readiness 信号不是在请求完成后回填的服务时间
代理，而是在 replica reserve 之后、adapter resolve/dispatch 之前记录的
selected-replica 快照。该快照带有 epoch/activation 身份，并可与实际首个
dispatch 和首 token 事件关联。对 7B 的两个相同协议冷运行，所有 4,000
请求均有完整 dispatch-tier 字段；自然 scale-up 的 first-service 观测合计
20 条。

这支持“readiness 的观测位置和生命周期绑定是可追溯的”这一证据主张。它不
支持把 readiness 本身写成必然降低 TTFT，也不支持把 first-service 样本的
数量转换成跨运行置信区间或因果效果。

## 不支持的语义

- 不能将 D278/D279 的 first-service 样本当作新的 7B Full 主比较点；
- 不能从 20 个样本推断 confirmed tier propagation 的普遍收益；
- 不能把 dispatch-tier 完整性替代独立数值 adapter correctness；
- 不能用 readiness 证据掩盖 A3 LastKnown 的质量—资源取舍；
- 不能把探索性运行的 GPU-s、平均延迟或 CE 合并到 D1115/D1117/D1118。

## 来源与保护

- D278：`docs/ieee_tc/D278_7B_READINESS_POST_D277.md`；
- D279：`docs/ieee_tc/D279_7B_READINESS_REPLICATE_20261006.md`；
- D278/D279 原始结果及其 SHA、phase、dispatch 和 first-service 明细均保留在
  原始结果目录和对应 curated 目录中；
- 该闭口不改变 `configs/generated/lora_manifest_1000.json`、
  `docs/ieee_tc/EXECUTION_STATUS.md` 或任何旧投稿图表；
- 本项不触发新的远端服务操作、缓存生成或 GPU 运行。

## 下一项

7B 的下一项不是 readiness 重跑，而是受控机制证据：优先登记并执行 S2
（同 cache 位置/候选和 admission 输入下切换 confirmed 与 last-known 信息
视图），或在已有 runner 能产生合法 proactive candidate 时执行 S3（同一
候选、同一初态、同一触发时刻下比较 Full 与 CapacityOnly 的 admission 和
延期成本）。如果现有实现无法在不注入人工压力或 sleep 的条件下形成这些
受控事件，则保留“未受控覆盖”的边界，不把自然运行改写成因果实验。

