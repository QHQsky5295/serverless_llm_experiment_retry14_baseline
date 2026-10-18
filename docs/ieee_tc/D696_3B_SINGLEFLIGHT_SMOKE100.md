# D696：3B 单飞扩容候选的 100 请求原生资格审计

日期：2026-10-18  
候选：D696（在 `ieee_confirmed` 控制评估中将扩容提交限制为单飞）  
目的：验证候选不破坏既有请求、生成、adapter 绑定、远端传输和资源释放合同；不把短前缀运行当作性能评价。

## 实验身份与范围

- 模型：Llama-3.2-3B。
- 负载：既有 W0 冻结 trace 的前 100 条请求；源 trace 为 4,000 请求，未重新生成工件或负载。
- 运行目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d696_20261018/3b_full_smoke100`。
- 代码提交：`0195fd232fc86ef49d492350f84e9fd4e264402a`；D696 配置和 runner 已在该提交登记。
- 生成合同：`fixed_length_greedy_v1`，原生 vLLM token-id 计数。
- 远端：复用已发布只读压缩工件；请求路径打包计数为 0。

## 结果

100/100 请求正常终态，100/100 原生生成合同完整，目标长度均满足，adapter 绑定和内容哈希审计通过。观测到 29 次远端传输（67,350,156 bytes）、26 次 HOST→GPU lease transfer、45 次确认 GPU 命中和 29 次 fresh inventory。物理 GPU 持有时间为 231.62473052507266 GPU-s；这些数值仅描述 100 请求观察窗口。

本轮只有 1 个 `initial` activation，没有 natural/controlled scale-out；因此 D696 的单飞限制没有被触发，也没有触发 `same_transaction_proactive_host_activation_v1`。这不是候选失败，而是短资格前缀未达到扩容触发条件。它不能支持单飞候选对 W0 性能、扩容尾延迟或 SLO 的结论。

请求级观测值：平均 TTFT 529.0697 ms，P95 TTFT 1159.4374 ms，平均 TPOT 14.9009 ms，P95 TPOT 18.3399 ms。独立数值正确性仍为未知，不能参与排名。

## 闭环与资源安全

- `launch.json`、`completion_receipt.json`、远端停止证据和 native 分析回执均通过。
- 远端 3B/7B artifact 服务和 monitor 已停止，owned service scope 已释放；共享 k3s 未触碰。
- 分析在 4 GiB `MemoryMax`、`MemorySwapMax=0`、CPU 2,3,26,27 的独立 scope 中完成；峰值 137,277,440 bytes，memory events 和 swap 均为 0。
- 分析作用域已显式关闭，递归进程为空、cgroup 路径已消失。
- 原始大文件和失败的通用分析尝试均保留；通用分析失败原因是其要求短前缀必须触发 proactive host activation，不改写为候选性能失败。短资格专用审计只放宽了“必须触发机制”这一不适用条件，其余合同仍严格检查。

## 裁决与下一步

D696 作为**资格通过、性能不接受、排名不合格**的观察点封存。下一步仍需在相同冻结配置下完成一次完整 3B W0；只有完整运行实际触发扩容并完成全量 4,000 请求后，才评价单飞候选是否改善 D695 暴露的扩容提交顺序问题。若完整运行不支持假设，保留结果并回滚候选，不叠加新的无证据补丁。

详细机器可读汇总见 [`qualification.json`](../../paper_results/ieee_tc/p2_backend/d696_3b_singleflight_smoke100/qualification.json)。原始证据根目录为 `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d696_20261018`。
