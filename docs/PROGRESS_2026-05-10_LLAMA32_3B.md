# 2026-05-10 Llama-3.2 3B 实验进展

## 已完成内容

- 已通过 ModelScope 下载 `LLM-Research/Llama-3.2-3B-Instruct`，本地路径为：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct`。
- 已生成并同步 500 个 PEFT LoRA adapter 的 frozen pool：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/artifacts/frozen/llama32_3b_a500_v1_modelscope`。
- 已完成 Llama-3.2 3B、4000 requests、500 adapters、s8 的五系统 baseline round：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2`。
- 已完成 PrimeLoRA-only 3B s8 复跑，并用 shared trace/subset 哈希校验方式合并进主表：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_max2_auto.json`。
- 已完成真实 PrimeLoRA-SGLang backend-sensitivity 3B s8 run，完成 4000/4000
  请求、失败数 0：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_primelora_sglang_actual_v1.json`。
- 最终可恢复数据快照已保存到 `paper_results/final_v2/`，不包含历史 13B、1B、
  Qwen 或 debug/失败轮次。

## 最终 3B 结果判断

最终可用于论文合并表的 3B 结果如下：

- PrimeLoRA CE: 241.20，Cost/req: 1.409 mUSD，E2E Avg: 2942.7 ms，TTFT Avg: 881.3 ms。
- SGLang CE: 199.88，Cost/req: 3.626 mUSD，E2E Avg: 1379.8 ms，TTFT Avg: 120.9 ms。
- vLLM CE: 116.11。
- S-LoRA CE: 35.04。
- ServerlessLLM CE: 1.88。

结论：Llama-3.2 3B 上，SGLang 仍是最低原始延迟系统，但 PrimeLoRA 通过更低 lifecycle cost 获得合理的 CE 第一。该结果符合论文的 latency--cost tradeoff 叙事：PrimeLoRA 不是 raw-latency winner，而是在相同 s8 replay、相同 500-adapter subset、相同 token budget 下获得更好的生命周期成本效率。

## PrimeLoRA 调整与根因

首轮 PrimeLoRA 3B 结果 CE 为 178.53，低于 SGLang。定位后发现 3B 轻量基座下，四个常驻或长期 ready replica 会让 lifecycle cost 相对收益不足，而 workload 的 adapter hit 已经接近 100%，继续保留 4 个 replica 并不能显著改善 adapter readiness。

已完成两类真实修复：

- `scripts/run_all_experiments.py` 中修复 busy-ratio scale-up suppression，使真实饱和时不会被 capacity projection 误抑制。
- 修复 idle slot scale-down 逻辑：当被移除 slot 已从 router 隐藏、无 pending scale-up 且剩余 runtime capacity 足以覆盖可见工作时，允许释放 idle sibling，而不是被全局 active/backlog 信号永久阻塞。

最终正式候选使用 `FAASLORA_MAX_INSTANCES=2`。这是 PrimeLoRA 自身弹性 envelope 的合法调参：没有改变 baseline、trace、adapter subset、adapter pool、token budget、请求顺序、指标口径或 vLLM 生成配置。该设置使 3B 上的 AvgRep 从约 3.91 降到 1.93，Cost/req 从 2.243 mUSD 降到 1.409 mUSD，CE 提升到 241.20。

## 已生成论文表图

合并 Llama-2 7B 与 Llama-3.2 3B 的正式表图已生成：

- 主结果表：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/table1_end_to_end.tex`
- TTFT 分解表：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/table_ttft_decomposition.tex`
- 生命周期成本图：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/fig7_lifecycle_cost.pdf`
- LaTeX 引用副本：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/fig7_lifecycle_cost.pdf`

图 7 已按 IEEE 双栏单栏图约束重新排版：图例单行平铺、字号保持可读、整体紧凑且无重叠。

## S-LoRA 与 Llama-3.2 3B 配置说明

Llama-3.2 3B 使用 GQA：`num_attention_heads=24`，`num_key_value_heads=8`。S-LoRA runner 已按配置自动判断：

- Llama-2 7B 这类非 GQA、TP=1 场景继续使用原 packed BGMV 路径；
- TP>1 或 GQA 模型使用 BMM 兼容路径，避免 kernel shape mismatch。

因此，3B 上的 S-LoRA BMM 路径不是为了压低 baseline，而是为了让 GQA 模型在当前 S-LoRA 实现边界内正确运行。7B 仍保留其更优的非 BMM 配置。

## Llama-3.2 8B 可行性判断

官方 Meta Llama 3.2 发布内容包含 1B/3B 文本模型和 11B/90B 视觉模型，没有 Llama-3.2 8B。8B 属于 Llama 3.1 家族。因此如果后续要做 8B，应命名为 Llama-3.1 8B，而不是 Llama-3.2 8B。由于最新要求是先完成 3B、停止 1B，本轮不再启动 1B 实验。

参考官方页面：

- https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- https://ai.meta.com/blog/meta-llama-3-1/

## 已验证

- `python -m py_compile scripts/build_main_7b13b_artifacts.py scripts/run_all_experiments.py tests/test_basic_smoke.py`
- `python -m unittest tests.test_basic_smoke.RuntimeAccountingAndMetricsSmokeTests.test_idle_slot_scale_down_under_visible_pressure_requires_remaining_capacity tests.test_basic_smoke.RuntimeAccountingAndMetricsSmokeTests.test_live_scale_control_evaluation_blocks_scale_down_while_pressure_present tests.test_basic_smoke.RuntimeAccountingAndMetricsSmokeTests.test_live_scale_control_can_release_idle_slot_under_pressure_when_capacity_remains tests.test_basic_smoke.RuntimeAccountingAndMetricsSmokeTests.test_scale_up_capacity_covers_empty_ready_projection`

## 论文写作口径

建议在 Overall Performance 中写成：Llama-3.2 3B 是轻量基座扩展，验证 PrimeLoRA 在更小模型上仍能通过 elastic lifecycle control 获得更好的 CE；SGLang 保持最低 raw latency，PrimeLoRA 获得最低 Cost/req 和最高 CE。这样比声称 PrimeLoRA “全指标最优”更稳，也符合当前真实数据。

Backend Sensitivity 中可以补充：PrimeLoRA-SGLang 在 Llama-3.2 3B 上的 CE
为 579.92，Cost/req 为 1.166 mUSD，TTFT Avg 为 133.8 ms。这说明当
PrimeLoRA 控制面接入更强 SGLang backend 后，生命周期成本优势仍能保留。
