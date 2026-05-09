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

## 已验证

已通过轻量检查：

- `python -m py_compile scripts/run_all_experiments.py tests/test_basic_smoke.py`
- `python -m unittest tests.test_basic_smoke -k handoff -k router`

## 下一步

需要在同一 shared trace、adapter subset、token budget、GPU budget 下复跑 PrimeLoRA-only 3B round，并与已有 3B baseline 结果合并比较。若修复后 PrimeLoRA CE 仍不是第一，应继续按交互规则定位真实链路问题，而不能改动 baseline 或指标口径。
