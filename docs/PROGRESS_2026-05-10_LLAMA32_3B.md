# 2026-05-10 Llama-3.2 3B 实验进展

## 已完成内容

- 已通过 ModelScope 下载 `LLM-Research/Llama-3.2-3B-Instruct`，本地路径为：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct`。
- 已生成并同步 500 个 PEFT LoRA adapter 的 frozen pool：
  `/home/qhq/serverless_llm_experiment_retry14_baseline/artifacts/frozen/llama32_3b_a500_v1_modelscope`。
- 已完成 Llama-3.2 3B、4000 requests、500 adapters、s8 的五系统正式 round：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2`。

## 当前结果判断

该 round 所有系统均完成 4000/4000 请求，结果可用于定位和审计，但 PrimeLoRA 在该 3B round 中尚未达到 CE 第一：

- SGLang CE: 199.88
- PrimeLoRA CE: 178.53
- vLLM CE: 116.11
- S-LoRA CE: 35.04
- ServerlessLLM CE: 1.88

因此，在修复 PrimeLoRA 自身真实问题并复跑前，不能把该 round 直接写成“PrimeLoRA 在 3B 上 CE 第一”。

## 已定位的 PrimeLoRA 根因

对 PrimeLoRA 3B 结果的生命周期日志和 scale-up 事件进行检查后发现：scale-out handoff 计划在新 runtime ready 时没有根据 ready-time 队列重新释放旧 reservation。

典型现象：

- `inst_3` 在约 87s ready；
- 该实例只服务了 5 个请求；
- 最后完成请求在约 1085s；
- 直到约 3979s 才 scale-down；
- scale-up 计划仍保留旧的 `finance_lora` planned adapter，但 ready-time 队列已经为空。

这会让新 runtime 被过期 handoff prefix 锁住，形成长时间空转，拉高生命周期成本并削弱 CE。

## 已完成修复

在 `scripts/run_all_experiments.py` 中修改 `_refresh_scale_up_runtime_handoff_plan_after_startup`：

- 当 ready-time 刷新后的 first-service window 已经为空时，不再保留原始 planned-adapter prefix；
- 只有刷新后仍存在 first-service budget 时，才继承启动前的 planned prefix；
- 这样可以避免启动期间队列已被 incumbent runtime 消化后，新 runtime 仍被旧 adapter reservation 锁住。

同步更新了 `tests/test_basic_smoke.py` 中相关单元测试，验证 refresh collapse 时会释放旧 handoff budget。

后续继续定位到两类 scale-out 过度响应：

- ready-time queue 已经为空时，`_refined_scale_up_target_instances` 仍会根据 `projected_arrived - incumbent_started` 一次性 fanout 到更多 runtime；
- ready-time queue 已经为空且当前 runtime capacity 足以覆盖可见工作时，`current_instances=1` 仍会因 busy-ratio 信号过早 scale-up。

对应修复为：

- 在 refined target 计算中，若 `queue_at_ready_request_count == 0`，仅执行 autoscaler 基础目标，不做 predictive fanout；
- 在 scale-up capacity gate 中，即使当前只有 1 个 runtime，只要 ready-time queue 为空且当前/待启动容量覆盖可见工作，就抑制 busy-ratio scale-out pulse；
- 保留真实扩容能力：当可见工作超过当前 forward capacity 或 ready-time queue 非空时，仍允许扩容。

同步补充了 `RuntimeAccountingAndMetricsSmokeTests` 中的覆盖用例，并修复 `ExperimentStack.__new__` 测试路径下 GPU forwarding 缺省配置与正常初始化语义不一致的问题。

## 已验证

已通过轻量检查：

- `python -m py_compile scripts/run_all_experiments.py tests/test_basic_smoke.py`
- `python -m unittest tests.test_basic_smoke -k handoff -k router`
- `python -m py_compile faaslora/experiment/experiment_stack.py scripts/run_all_experiments.py tests/test_basic_smoke.py`
- `python -m unittest tests.test_basic_smoke.RuntimeAccountingAndMetricsSmokeTests`

## 复跑观察

已尝试两轮 PrimeLoRA-only 3B s8 复跑：

- `fix6` 验证了空 ready queue 下不会一次性 fanout 到 4 个 runtime，但仍在 early busy-ratio 信号下长期保持 2 个 runtime。中段 `tokenproxy_ce` 约 145，低于 3B SGLang 的 199.88，因此停止，不进入论文数据。
- `fix7` 进一步抑制了当前容量覆盖时的单 runtime 误扩容，验证了首段保持 1 个 runtime 且 `fail=0`。但单 runtime 下 E2E 排队上升，中段 `tokenproxy_ce` 约 113，同样低于 3B SGLang，因此停止，不进入论文数据。

结论：Llama-3.2 3B 作为轻量基座时，SGLang 的 backend execution latency 优势过强，PrimeLoRA 的 serverless readiness/cost 优势不足以在该点形成 CE 第一。该结果可作为内部诊断，不建议替代 Llama-2 13B 进入主表。

## 与 Llama-2 13B 的关系

本地已有 Llama-2 13B 结果显示，该模型规模更符合论文的 elasticity/readiness 叙事：

- SGLang 13B 已完成结果 CE 约 61.52；
- vLLM 13B 已完成结果 CE 约 36.70；
- PrimeLoRA 13B 正式候选结果最高约 76.66，具备 CE 第一的趋势；
- S-LoRA 13B 在当前 4x RTX 3090 24GB、长 Azure trace、500 adapters 环境下表现异常差。S-LoRA 论文/官方博客中的 Llama-13B 设置使用 A100 80GB 级别环境，因此该差异更像硬件与长序列压力不匹配，而不是简单代码崩溃。

## 下一步

建议回到 Llama-2 7B + Llama-2 13B 的主线，优先用已经完成且同属 Llama-2 家族的正式结果合并主表。3B/1B 可以保留为内部适配和诊断，不应作为必须证明 PrimeLoRA CE 第一的主扩展点。若继续做 1B，预期 SGLang 的 latency 优势会更强，需要先把它定位为额外诊断实验，而不是主表替代项。
