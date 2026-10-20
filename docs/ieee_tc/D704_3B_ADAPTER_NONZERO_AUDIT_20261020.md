# D704：3B LoRA 制品非零权重审计

日期：2026-10-20  
模型：Llama-3.2 3B  
执行类型：只读、CPU-only、无 GPU、无远端服务、无请求回放  
状态：终止并封存（审计完成）

## 目的

D703 已经完成了 4,000 请求的服务契约和调度观测，但其独立数值
`n_correct` 仍未知。D704 检查 D703 所使用的冻结 500-adapter 制品池是否包含
可用于非零 LoRA 数值核对的权重。审计只读取现有文件，不生成、改写或复制
任何模型、LoRA、trace 或压缩归档。

## 输入与可复现信息

制品根目录为：

`/home/qhq/serverless_llm_experiment/artifacts/frozen/llama32_3b_a500_v1_modelscope`

目录包含 500 个 `adapter_model.safetensors` 文件。审计使用
`/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`（Python 3.12.12，
`safetensors` 0.7.0），通过 `safetensors.safe_open(..., framework="numpy",
device="cpu")` 逐文件、逐 tensor 读取，并统计元素数量、非零元素、非有限元素
和最大绝对值。没有使用服务输出作为参考，也没有使用随机或人为生成的权重。

冻结制品文件清单（相对父目录、文件大小、SHA-256 逐行串联）的审计摘要为：

`inventory_sha256=9b3293bb9c33b3f670f53460b619c79410c99fe82b4e7c818b136199a89df81a`

## 结果

| 项目 | 结果 |
|---|---:|
| 权重文件 | 500 |
| 读取失败文件 | 0 |
| tensor 数量 | 112,000 |
| 元素数量 | 2,981,888,000 |
| 非零元素 | 0 |
| 非有限元素 | 0 |
| 最大绝对值 | 0.0 |
| 全部权重为零 | 是 |

机器可读回执：

`paper_results/ieee_tc/p2_backend/d704_3b_adapter_nonzero_audit/audit.json`

## 解释边界

该结果说明当前冻结的 3B adapter pool 在数值上是全零的。因而在不引入新的
权重、也不改变既定制品池的前提下，不能从这个 pool 构造一个能区分非零 LoRA
增量的独立数值 oracle。D703 的以下证据仍然有效：

- 请求终态、固定长度生成合同和原生 token-id 计数；
- adapter/content/lease 绑定；
- profile 加载与 160/160 service-class coverage；
- 远端预发布制品交付、请求期无打包；
- 联合 TTFT/TPOT SLO、尾延迟和物理 GPU 生命周期账目。

但这些证据不能被改写为“LoRA 数值正确性已证明”。对于当前 3B pool，输出与
base-model 参考相同最多能证明基座模型和生成合同行为，不能证明非零 adapter
路径。D703 的 `n_correct=null`、`numerical_adapter_correctness_qualified=false`
和 `ranking_eligible=false` 必须保持不变。

该审计也不支持通过人为生成非零权重、替换 adapter、改写 trace 或把零权重
结果标成正确来闭合资格。后续若要取得 3B 的独立数值正确性证据，必须使用
既有且确实非零的可信权重/参考路径，并记录其内容 SHA、后端版本、比较规则和
失败样本；在此之前不开始 3B 的 G1/G2 排名或 baseline 胜负比较。

