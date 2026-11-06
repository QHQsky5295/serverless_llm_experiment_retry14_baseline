# D1101：ServerlessLLM 7B 历史证据与重跑裁决

## 审计对象

本轮先审计已有 ServerlessLLM 证据，不把异常差结果直接当成算法结论，也不为
baseline 搜索更优参数。复核对象是：

1. D941：`ServerlessLLM-new` 的 7B fixed-length greedy 完整 4,000 请求回放；
2. D555：ServerlessLLM 7B W2 高频轮换完整 4,000 请求回放；
3. D539/D537：W1 等待、启动和调度路径因果核查；
4. 官方 commit `9f50241baa5386e06a9321c51f19a9ef5f964c2b` 及 TC 共同合同适配。

## 已有证据的有效性

### D941：完整但不是本轮 W2 对照点

D941 使用 fixed-length greedy v1、vLLM 原生 token IDs、4,000/4,000 成功请求、
500-adapter subset、同一 trace SHA 和预发布 gzip 交付；输出合同、prompt hash、
目标长度和 native token audit 均通过。其观察值为平均 TTFT 1,522.4 ms、P95
TTFT 4,665.5 ms、平均 TPOT 25.61 ms、TTFT-only SLO 95.125%、物理 GPU
16,899.38 GPU-s。它可以作为 ServerlessLLM-new 的完整 W0 资格证据。

但 D941 的远端 HTTP fetcher 记录为 `http_unthrottled`，不能用于等价带宽敏感性；
`n_correct` 也未知。更重要的是，它是 W0 工作点，不能直接替代 PrimeLoRA
D1100 的 W2 高频轮换对照。

### D555：负载匹配，但不是当前正式身份

D555 使用相同 7B W2 轨迹、500-adapter subset 和 fixed-length 语义，4,000 请求
全部完成，ServerlessLLM 联合 TTFT/TPOT 达成率 93.90%、P95 TTFT 9,516.7 ms、
物理 GPU-s/request 3.9734。它是有价值的 W2 历史观测，但执行身份仍为
`formal_run=false`，且来自较早的 baseline 运行环境；不能把它事后改写成当前
ServerlessLLM-new 的正式同合同点。

## 异常差结果的因果判断

D537/D539 的请求级分解显示，W1 的高 TTFT 主要来自 router queue；平均 service
TTFT 约 462.9 ms，而平均 router queue 约 19.9 s。D941 同样把大量时间留在
dispatch/admission wait（平均约 1.12 s、P95 约 4.37 s），不是 LoRA 前向或
decode 本身突然变慢。启动日志还显示后续 replica 的创建、ready 和 KV/profile
阶段存在排队；这支持“原始 round-robin/concurrency-threshold 控制路径在本负载
下响应较慢”的解释，但不能把所有等待归因为单一编译或 GPU 算力不足。

现有 `ready_before_wait` 修复仅改变空闲 ready replica 的轮询顺序，未替换
round-robin、容量上限、autoscaler 或加载策略；CPU AST 和真实启动证据没有发现
足以支持继续改算法的适配错误。因此本轮不做 ServerlessLLM 性能调参，也不把
较差 W1/W2 结果当成 PrimeLoRA 优势的唯一依据。

## 重跑裁决

为了让 7B PrimeLoRA D1100 与 ServerlessLLM 在同一 W2 轨迹、adapter subset、
fixed-length 合同、远端预发布交付和单轮物理资源口径下可比，下一步执行一次
新的 ServerlessLLM W2 4000-request formal key。该运行只修复共同实验合同和资源
计量衔接，不搜索 ServerlessLLM 配置，不改变其官方 router/autoscaler/load
policy；`ServerlessLLM-new` 仅作为内部身份，论文图表显示名仍为 **ServerlessLLM**。

若新运行仍出现高 dispatch wait，保留完整等待分解并如实报告；若出现合同失败，
先按生成、绑定、环境和资源归因分类，不能筛掉失败请求或改变分母。D555、D537、
D941 的历史 raw evidence 均保持不变，新结果只进入新 campaign 目录。

## 边界

- 所有结果的 `n_correct` 未知，HTTP 成功和目标 token 数量不等价于独立 LoRA
  数值正确性；
- 请求触发的真实远端读取/传输/本地物化计入 TTFT/E2E，离线发布和请求时打包不
  混入；
- 单轮结果不计算运行间 CI，不把 4,000 请求当作 4,000 个独立重复；
- 不生成图表，直到 PrimeLoRA 与同合同 ServerlessLLM 对照闭合。
