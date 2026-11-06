# D1102：ServerlessLLM 7B W2 启动资格失败记录

D1102 于 2026-11-07 建立并启动，但未进入正式请求回放，因此不产生
ServerlessLLM 性能观测，也不替代 D941、D555 或后续正式 W2 键。

## 失败边界

- W2 投影输入通过：4,000 条请求、原始 trace SHA、W2 map SHA、500 个工件和
  497 个实际出现的 adapter 均已核验。
- ServerlessLLM deploy config 生成成功，远端 `prepublished_gzip_v1` 健康检查成功。
- vLLM 探针使用官方 `sllm_vllm0102_newserverless_20260518` 环境，默认
  `SLLM_VLLM_PROBE_TIMEOUT_S=120`，在冷启动的 vLLM 编译阶段超时；runner 因此
  按设计拒绝进入无正确性证据的正式回放。
- 本地独立复核未修改模型、adapter、trace、调度或后端：同一 deploy 和 W2 trace
  的 V1 probe 完成约 146 秒并输出 16 个非零 token，确认失败是资格时间上限，
  不是 ServerlessLLM 的请求性能结果或 LoRA 输出错误。
- 没有正式 replay、没有 summary、没有 GPU-s/request 或 SLO 结果；D1102 不进入
  主比较表。原始输出目录和日志保留，远端服务已停止，GPU 进程已释放。

## 允许的最小兼容修正

baseline 仓库仅在新的 W2 runner 中显式设置
`SLLM_VLLM_PROBE_TIMEOUT_S=300`，使冷启动编译能够完成；ServerlessLLM 官方
代码、路由、扩缩容、缓存和加载策略没有改变。该修正提交为
`686ee1fce68e230bfc8ace1abe6a49444ecdc50d`，不回写 D1102 输出。

后续使用新的 D1103 执行键；若 D1103 仍失败，将按实际失败类型保留，不把失败
重命名为性能结果。
