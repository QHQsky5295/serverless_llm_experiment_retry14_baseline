# D1103：ServerlessLLM 7B 同合同 W2 正式对照候选

D1103 是 D1102 启动资格失败后的唯一新执行键。它复用 D1102 已通过的 W2 输入
投影和远端交付合同，仅增加冷启动资格探针的 300 秒完成上限；不修改
ServerlessLLM 的官方调度、路由、扩缩容或 LoRA 加载策略。

## 固定实验身份

| 项目 | 值 |
|---|---|
| 模型/后端 | Llama-2-7B；ServerlessLLM 官方复现路径；vLLM 0.10.2 环境 |
| 工作负载 | 原 seed-42 W0 arrival/prompt/target，4,000 请求，W2 adapter view |
| W2 轮换 | 100 请求、48-ID 热点窗口、stride 12、邻窗重叠 75% |
| adapter 集合 | 原有 500 个；不生成、不复制新权重 |
| 生成合同 | `fixed_length_greedy_v1`，native vLLM token IDs |
| 远端交付 | `prepublished_gzip_v1`；请求期间不打包 |
| 结果目录 | `serverless_llm_baselines/results/paper_experiments/16_serverlessllm_new_remote_w2_v1/` 的 D1103 新命名空间 |

W2 map SHA 为 `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`，
source trace SHA 为 `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
本次实际 W2 view SHA 为 `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767`。

## 资格规则

必须同时满足 4,000/4,000 终态、请求/adapter/arrival/target/prompt 身份一致、
fixed-length native token 合同、无请求期打包、完整服务/远端释放和可审计资源
字段。联合服务达成率低于 95% 时，结果仍保留为真实 ServerlessLLM 观测，不重跑
筛选更有利值；先按 D1101 审计规则判断是适配问题还是机制行为。

D1103 完成并审计后，才进入 7B PrimeLoRA–ServerlessLLM 同合同比较；D1102、
D941 和 D555 的历史身份继续保留，不被重写。
