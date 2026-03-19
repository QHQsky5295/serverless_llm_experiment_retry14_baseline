# 项目进度记录

## 文档用途

本文档记录：

- 当前本地仓库与 GitHub 远端的一致性状态
- 相较上一次远端同步基线的主要变化
- 当前主线实验做到哪里了
- 仍待完成的工程与扩展任务
- 下一步应该推进什么

每次同步 GitHub 前后都应更新本文件，并保证与 README 和技术说明文档口径一致。

## 当前仓库标识

- 项目名称：**FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**
- 本地仓库：`/home/qhq/serverless_llm_experiment`
- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 本次同步前基线：`5826d43`

当前结论：

- 本次同步开始前，GitHub 上的 `main` 与本地已提交基线一致。
- 本文记录的是本次同步批次中纳入的代码与文档更新。

## 本次同步纳入的更新

本次同步包含以下文件更新：

- `faaslora/cli.py`
- `README.md`
- `EXPERIMENT_GUIDE.md`
- `PROJECT_STRUCTURE.md`
- `docs/ENVIRONMENT.md`
- `docs/PROJECT_PROGRESS.md`
- `tests/test_basic_smoke.py`
- `tests/test_integration.py`

这些变更主要对应：

1. 补回 `faaslora.cli`，修复 `pyproject` console script 与 autoscaler 子进程入口断裂。
2. 新增 `unittest` 版 smoke tests，覆盖主线默认配置、500 preset serving 参数与 CLI 可执行性。
3. 把 `README` / `EXPERIMENT_GUIDE` / `PROJECT_STRUCTURE` / `ENVIRONMENT` / `PROJECT_PROGRESS` 对齐到当前主线默认配置和当前实际稳定环境。

## 相较当前 GitHub 基线的主要变化

### 代码变化

1. `faaslora/cli.py` 已提供 `faaslora coordinator --config ... --host ... --port ...` 的最小可用入口。
2. `python -m faaslora.cli --help` 与 `python -m faaslora.cli coordinator --help` 已验证可执行。
3. `tests/test_basic_smoke.py` 已提供当前稳定环境下可直接运行的基础 smoke tests。

### 文档变化

1. `README.md` 已更新为当前主线口径：
   - 主线模式是 `auto`
   - 当前主线配置是 `auto + 500 LoRA + representative 1000 requests`
   - 当前验证通过的 serving 参数是 `max_num_seqs=8`、`max_loras=8`、`runtime_concurrency_cap=8`
2. `EXPERIMENT_GUIDE.md` 已改掉旧的“启动即多槽位”“默认 500 requests”等过时表述。
3. `PROJECT_STRUCTURE.md` 与 `docs/ENVIRONMENT.md` 已对齐当前主线默认模型、结果文件形态和当前实际环境版本。
4. 本进度文档已经改为反映当前真实状态，而不是历史阶段计划。

## 当前主线实验状态

### 已完成

当前已经完成并验收的主线结果为：

- 模式：`auto`
- LoRA 数量：`500`
- 请求数：`1000 representative requests`
- serving 参数：
  - `concurrency=8`
  - `runtime_concurrency_cap=8`
  - `max_model_len=2048`
  - `max_num_batched_tokens=4096`
  - `max_num_seqs=8`
  - `max_loras=8`

对应结果文件：

- `results/experiment_results_full_vllm_auto_a500_r1000_c8_faaslora_full_seq8_lora8.json`

这组 `3B` 历史冻结基线的结论：

1. `auto` 主路径已真实跑通，扩缩容、warmup、preload、residency、resource coordination 都进入了完整链路。
2. 历史 `3B` 路线的主要瓶颈已经从 admission / defer 转移到 serving 参数调优问题，并已通过 `seq8_lora8` 基本解决。
3. `auto500 + representative1000 + seq8_lora8` 现在保留为 `3B` 的历史冻结基线，不再代表当前仓库默认入口。
4. `r3` 复现实验已经完成，当前主基线已通过一次稳定性复验。

### 当前主线关键结果

当前已验证结果的关键指标约为：

- `TTFT avg = 1409 ms`
- `TTFT P95 = 4068 ms`
- `TTFT P99 = 6023 ms`
- `E2E P99 = 22550 ms`
- `RPS = 0.364`
- `Hit = 94.6%`
- `scale_up_events = 1`
- `scale_down_events = 1`
- `contention = 0`
- `defer = 0`

当前稳定性复验结果：

- 文件：`results/experiment_results_full_vllm_auto_a500_r1000_c8_faaslora_full_seq8_lora8_r3.json`
- `TTFT avg = 1494 ms`
- `TTFT P95 = 4138 ms`
- `TTFT P99 = 6670 ms`
- `E2E P99 = 22642 ms`
- `RPS = 0.359`
- `Hit = 94.6%`

与上一轮主结果相比：

- `TTFT avg` 波动约 `+6.0%`
- `P95 TTFT` 波动约 `+1.7%`
- `P99 TTFT` 波动约 `+10.7%`
- `E2E P99` 波动约 `+0.4%`
- `RPS` 波动约 `-1.5%`

当前判断：这组波动处于可接受范围，暂时不需要继续追加 `r4` 来确认主基线稳定性。

### 14B 最新状态（2026-03-15）

当前扩展主线已经从“14B bring-up 修复”推进到“14B `representative r1000` 长跑完成”，结论需要和旧文档区分开：

- 14B 最初的主阻塞已经定位清楚：
  - 不是模型文件缺失
  - 不是一开始就 OOM
  - 而是 `scripts/run_all_experiments.py` 在 `TP=2` 路径里把 `CUDA_VISIBLE_DEVICES` 临时缩成了单卡
  - 导致 vLLM 误选 `ray_distributed_executor`，随后在单节点 Ray node startup 阶段超时

- 这个问题已经修复：
  - `TP>1` 时优先使用 `visible_device_ids`
  - 本机双卡足够时显式使用 `distributed_executor_backend="mp"`

- 当前 `qwen_14b_tp2` 默认 profile 已收敛到：
  - `distributed_executor_backend=mp`
  - `gpu_memory_utilization=0.85`

- 修复后的 quick 与完整长跑复测已经确认：
  - 14B `TP=2` 可以稳定完成 engine bring-up 并进入真实 serving
  - `gpu_memory_utilization=0.90` 时 GPU 常驻约 `22.3/24.0 GB (93%)`，会持续触发 memory pressure
  - `gpu_memory_utilization=0.80` 时 GPU 常驻约 `19.8/24.0 GB (83%)`
  - memory pressure 告警消失，GPU resident adapter 数可稳定升到 `46`
  - `representative 1000 requests` 已完整跑完，结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_p25_on.json`
  - 关键指标：
    - `1000/1000` 完成，`fail=0`
    - `TTFT avg/p95/p99 = 603 / 1006 / 1174 ms`
    - `E2E avg/p95/p99 = 13.10 / 14.91 / 15.22 s`
    - `TPOT avg = 99.3 ms`
    - `RPS = 0.148`
    - `cache hit rate = 85.2%`
    - `warm_pool_hits = 806`
    - `contention_events = 0`
    - `avg_defer_ms = 0`
  - 日志中未再出现 `Memory pressure detected`、Ray startup timeout 或 EngineCore 异常退出
  - 在此基础上补做了 `gpu_memory_utilization=0.85` 的完整 `r1000` A/B：
    - `1000/1000` 完成，`fail=0`
    - GPU 常驻约 `21.1/24.0 GB (88%)`
    - 仍未出现 `Memory pressure detected`、Ray timeout 或 EngineCore 异常退出
    - 相比 `0.80`：
      - `TTFT avg` 下降约 `0.7%`
      - `TTFT p99` 下降约 `1.1%`
      - `E2E avg` 下降约 `1.5%`
      - `E2E p95/p99` 下降约 `8.5%`
      - `TPOT avg` 下降约 `1.7%`
      - `RPS` 提升约 `1.5%`
      - 仅 `TTFT p95` 小幅上升约 `1.5%`
    - 综合判断 `0.85` 是当前 `14B` 更优的稳定参数组合

- 在此基础上继续补做了 `representative 4000 requests` 长跑：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r4000_c2_faaslora_full_qwen14b_tp2_r4000_u085_p25_on.json`
  - `4000/4000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 588 / 1013 / 1133 ms`
  - `E2E avg/p95/p99 = 13.15 / 14.90 / 15.16 s`
  - `TPOT avg = 99.6 ms`
  - `RPS = 0.1475`
  - `cache hit rate = 86.05%`
  - `warm_pool_hits = 3341`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 日志中未出现 `Memory pressure`、Ray timeout、OOM 或 Traceback

当前判断：

- 14B 已经从“启动失败 / bring-up 阻塞”推进到“最优稳定参数已收敛并完成 `r4000` 验证”
- 当前 `gpu_memory_utilization=0.85` 可作为 `14B` 的冻结默认稳定参数
- 14B 主线的下一步不再是继续调参，而是切到 `7B TP=2` 对照结论收口，并准备第二家族 bring-up；当前已由 OPT 改为 Mistral

### 7B TP=2 对照最新状态（2026-03-15）

在 `Qwen2.5-7B-Instruct` 上已经补做完：

- `qwen_7b_tp2_compare`
- `tensor_parallel_size=2`
- `distributed_executor_backend=mp`
- `max_instances=1`
- `representative 1000 requests`

结果文件：

- TP=1 基线：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_auto_r1000_p25_on.json`
- TP=2 对照：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_tp2_compare_r1000_p25_on.json`

对比结论：

- 两轮都完成 `1000/1000`，且 `fail=0`
- TP=2 相比 TP=1：
  - `RPS` 从 `0.228` 升到 `0.340`，约 `+49.2%`
  - `E2E avg/p95/p99` 分别改善约 `30.8% / 38.1% / 14.6%`
  - `TPOT avg` 改善约 `43.2%`
  - `P95 TTFT` 改善约 `36.4%`
  - 但 `TTFT avg` 变差约 `37.7%`
  - `P99 TTFT` 变差约 `5.9%`
  - `cache hit rate` 从 `94.6%` 降到 `85.2%`
  - `warm_pool_hits` 从 `896` 降到 `801`
- 两轮都没有 `contention` / `defer` / `Memory pressure` 回归

当前判断：

- 如果目标是**吞吐、平均 E2E 和 TPOT**，`7B TP=2` 明显更优
- 当前项目已按新的实验口径把 `7B` 主线默认切到 `TP=2`
- 原先 `TP=1` 的口径只作为历史阶段性结论保留，不再作为当前默认模式

### 不再作为当前主线推进的内容

以下内容的实现和配置入口继续保留，但不再作为当前主线推进对象：

- `shared`
- `dedicated`
- `28185 full trace`

它们目前的定位是：

- 内部 sanity check
- 备用补充实验
- 压力测试入口
- 后续扩展可复用接口

补充说明：

- `effective_capacity_admission_enabled` 的开关接口继续保留，用于 on/off 对照；
- 但在最新一轮 `Qwen2.5-7B r300` A/B 后，仓库默认配置已改为 `on`，不再作为默认关闭路径。

## 当前工程状态

### 已修复

1. `run_all_experiments.py` 中 preset 对显式环境变量的覆盖问题。
2. user-scope 启动脚本未透传 `FAASLORA_*` 环境变量的问题。
3. 结果文件命名冲突问题。
4. 主线 serving 参数不可显式覆盖的问题。
5. `configs/experiments.yaml` 的默认入口已固化为 `profile_selection` 路径；当前文件内默认选择为 `qwen_14b_tp2 + azure_sharegpt_rep1000 + qwen_14b_tp2_main`。
6. 默认入口复验已完成，默认命令路径与冻结主线配置保持一致。
7. `faaslora.cli` 已补回，`pyproject` 的 console script 与 autoscaler 的 `python -m faaslora.cli coordinator ...` 子进程入口不再悬空。
8. `flashinfer` 与 `torch-c-dlpack-ext` 已装入 `LLM_vllm0102`，vLLM 采样路径显式请求使用 FlashInfer。
9. 修复了 GPU 全局显存观测、ResidencyManager 监控未启动、contention/defer 记账失真等问题；后续结果统一以修复后的口径为准。
10. `Qwen2.5-7B r300` 的 `P2.5 on/off` A/B 已完成；`P2.5 on` 在修复后的口径下显著降低了 `contention_events` 与 `avg_defer_ms`。
11. `Qwen2.5-7B auto + 100 adapters + 1000 requests + P2.5 on` 已完成长跑验证，`contention=0 / defer=0`，7B 当前默认验证参数可冻结。
12. `Qwen2.5-3B auto500 + representative1000 + P2.5 on` 复验已完成；相对旧的 `seq8_lora8` frozen baseline 仅有轻微波动，说明 P2.5 对 3B 主要用于统一口径，而非带来显著性能收益。
13. 修复了 Azure dataset adapter 中字符串布尔值被 `bool(\"False\") -> True` 误判的问题，并补回归测试。
14. 修复了 `Qwen2.5-14B-Instruct TP=2` 路径里 `CUDA_VISIBLE_DEVICES` 被错误缩成单卡、从而让 vLLM 误走 Ray 的问题。
15. `run_all_experiments_user_scope.sh` 已补透传 `VLLM_*` 与 `PYTHONUNBUFFERED`，与当前 user-scope 主线启动命令保持一致。
16. 已补齐第二家族的 model/workload profile 骨架，并修复 `scripts/download_model.py` 仍按旧版单模型 YAML 结构误改 `experiments.yaml` 的问题；当前下载脚本改为输出 profile 使用提示，不再自动重写主配置。

### 仍待完成

1. 稳定环境下可执行的基础测试仍待补齐到更完整覆盖面；当前仅补上了不依赖 GPU / 外部模型的 smoke tests。
2. 仍需继续清理 README / GUIDE / 其他附属文档与当前实现之间的漂移。
3. 7B 默认参数虽已可冻结，但文档和后续模型扩展仍需以这组参数为基线继续收口。
4. 仍需继续清理部分附属脚本/注释中残留的旧口径。

## 模型与数据集扩展状态

### 当前状态

- 当前扩展主线的 Qwen 家族阶段已基本收口
- 已完成：`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Qwen2.5-14B-Instruct`
- 已完成：`Qwen2.5-7B-Instruct TP=2` 对照 profile 验证
- 已完成：`mistralai/Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000`
- 下一步：推进 `mistralai/Mistral-Nemo-Instruct-2407 + TP=2 + PEFT+finetune + 500 adapters + representative r1000`
- Gemma 暂不进入当前配置与实验轮次，但继续保留在计划列表中

### 后续扩展目标

模型家族：

- Qwen：`Qwen2.5-7B-Instruct`、`Qwen2.5-14B-Instruct`
- Mistral：已完成 `mistralai/Mistral-7B-Instruct-v0.3`，下一步推进 `mistralai/Mistral-Nemo-Instruct-2407`
- Gemma：暂挂计划列表，当前不配置、不启动实验

数据集：

- `HuggingFaceH4/ultrachat_200k`
- `lmsys/lmsys-chat-1m`
- `gsm8k`

进入这些扩展之前，需要先把当前 Qwen 主线口径稳定下来；本机现有 `Qwen2.5-3B-Instruct` 模型目录继续保留，不删除。

## 当前主线 TODO

### A. 主配置固化

1. 已完成：对历史 `3B seq8_lora8` 冻结基线做稳定性复验。
2. 已完成：将当前 serving 配置正式固化为主线默认复现实验参数。
3. 已完成：用默认入口再做一次复验，确认后续复现不依赖长串环境变量覆盖。
4. 已完成：把当前主线结果、配置和运行命令同步到所有核心文档。

### B. 工程闭环

5. 已完成：修复 CLI / packaging 断裂。
6. 继续补齐稳定环境下可跑的基础测试。
7. 清理 README / GUIDE / docs 与实现不一致的残留项。
8. 已完成：将 `model / dataset / workload` 主线切换入口收敛到 `experiments.yaml` 的 `profile_selection`。

### C. 扩展主线

8. 已完成：`Qwen2.5-7B-Instruct` 的 bring-up 与 `r300` 短测。
9. 已完成：`Qwen2.5-7B-Instruct auto + 100 adapters + 1000 requests + P2.5 on` 长跑验证。
10. 已完成：补一轮 `Qwen2.5-3B auto500 + representative1000 + P2.5 on` 复验，统一默认配置与结果口径。
11. 已可执行：基于现有结果冻结 7B 的默认验证参数。
12. 已完成：`Qwen2.5-14B-Instruct + representative 1000 requests` 的完整长跑验证。
13. 已完成：基于 `14B r1000` 的完整结果，收敛出“指标最好且稳定可复现”的参数组合。
14. 已完成：用 `14B` 最优稳定参数 `gpu_memory_utilization=0.85` 完成 `representative 4000 requests`。
15. 已完成：新增并验证 `Qwen2.5-7B-Instruct TP=2` 对照 profile（单实例双卡），并与当前默认 `7B TP=1` 路径完成效果对比。
16. 已确认：`facebook/opt-6.7b` 在当前 `vLLM 0.10.2 + LoRA` 环境下报 `OPTForCausalLM does not support LoRA yet`，因此 OPT 路线退出当前主线。
17. 已完成：切到 `mistralai/Mistral-7B-Instruct-v0.3`，并按论文主线默认使用 `PEFT+finetune + 500 adapters` 完成 `representative r1000`。
18. Gemma 暂不配置，但保留在计划列表中，等 Qwen / Mistral 稳定后再恢复。
19. 在 Mistral 主线稳定后，再接入额外数据集，新增 `gsm8k`。
20. 已决定：论文主线默认 LoRA 工件从 synthetic 切到 `PEFT+finetune`；synthetic 仅保留给 quick/debug。当前正式对比推荐走 `two_phase`：先为每个 base model 单独建好冻结工件池，再启动实验；`one_shot / bootstrap_once` 仅保留给日常调试和建库。
21. 已补 `mistral_7b_auto500_main`，后续 Mistral 7B 主线统一使用 `representative r1000 + 500 adapters`，不再沿用扩展阶段的 `100 adapters` bring-up 口径。
22. 已完成：`scripts/generate_lora_adapters.py` 默认值改为跟随 `profile_selection + model_profiles + workload_profiles` 解析，不再只读取顶层 `model.name`。
23. 已完成：`PEFT+finetune` 生成路径改为单次加载 base model 后循环生成多个 adapters，避免每个 adapter 重复 `from_pretrained`。
24. 已完成：`tests/test_basic_smoke.py` 与 `tests/test_integration.py` 已更新为校验 active profile、生成器默认值与 batch PEFT 路径；旧的不可执行/失真集成测试入口已清理。
25. 已完成：`experiments.yaml` 顶层 `model / hardware / workload` 注释已改为“兼容回退层”口径，避免误判为当前主线默认入口。
26. 已完成：`Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000`，结果稳定通过，`1000/1000` 完成且 `fail=0`。
27. 已确认：`mistral_nemo_12b_tp2_main` 现在作为论文主线工作负载 profile 固定为 `500 adapters`；`mistral_nemo_12b_tp2_bringup100_main` 仅保留给显式 bring-up / 快速排障。
28. 已完成：引入模型专属冻结工件目录（`artifacts/frozen/<model>_a500_v1`）与 `standardized_v1` 基线工件池；`Qwen 7B / Qwen 14B / Mistral 7B / Mistral Nemo 12B` 现在都能按“先建库、后复用”的方式维持论文正式对比变量一致。
29. 已完成：清理旧的 `*_v2_realistic` 目录，保留 `V1` 冻结工件不动，准备按新的正式 `V2 publicmix` 规则重建。
30. 当前下一步：`Mistral-Nemo V2 publicmix representative r1000` 已完成，且 `opt1`（`gpu_memory_utilization=0.85, max_num_seqs=2, runtime_concurrency_cap=2`）已作为敏感性实验保留、未采纳为默认参数。`Mistral 7B V2 publicmix representative r1000` 首次尝试在当前环境下因 `vLLM V1 + 异构 public LoRA` 触发 `EngineCore / CUDA illegal memory access`，默认 profile 已收紧到更保守的 `V0 + no chunked prefill + no prefix caching + lower concurrency` 路径，准备重跑；随后补齐 `Qwen 7B / Qwen 14B` 的 `representative r1000`，最后进入 `ServerlessLLM` 对比实验。
31. 已完成：新增 `scripts/prepare_publicmix_pool.py`，支持对本地公开 adapter 做兼容性验收（`validate`）并生成正式 `V2 publicmix` 清单（`plan`），避免后续建库时靠人工肉眼筛选。
32. 已完成：`Qwen 7B / Qwen 14B / Mistral 7B / Mistral-Nemo` 四个模型的 `V2 publicmix` 第一阶段验证与 manifest 已落盘；当前 accepted public 数已修正为 `0 / 4 / 1 / 4`，因为当前运行时不支持 `DoRA`，对应公开工件已在验证阶段被剔除，其余缺口通过统一 `realistic_v2 + seed=42` 规则补齐到 `500`。
33. 已完成：`scripts/prepare_publicmix_pool.py` 新增 `build` 子命令，可按 manifest 先复制公开 adapter，再只对 `generated_fill` 缺口调用生成器补齐，从而物化成真正可复用的 `V2` 冻结工件池。
34. 已完成：`Qwen 7B / Mistral 7B` 的 `V2 publicmix` 工件池已做 `DoRA` 清洗修复；当前 live 目录中的 `500` 个工件已重新验到 `use_dora=0`，避免后续 `Mistral 7B / Qwen 7B` 正式实验在 vLLM 侧因 `DoRA` 直接报错。
35. 已完成：`faaslora/utils/model_assets.py` 现已能自动修复冻结工件池在归档/恢复后产生的坏 symlink；`Qwen 7B / Mistral 7B` 的 `config.json / generation_config.json` 支持文件已重新补齐到正确本地模型路径，避免 `two_phase` 启动时在 `ensure_adapter_support_files()` 阶段再次报错。
36. 已完成：`warning elimination / runtime hygiene` 第一轮修复。项目默认不再全局启用 FlashInfer sampler；`Mistral` 系列 frozen adapter 目录现会自动补齐 `tokenizer.model.v* / tekken.json / chat_template.jinja` 等支持文件，从而避免把运行时 fallback warning 当成“正常噪声”长期遗留。

## 当前已确认的长期约束

1. 每次同步 GitHub 时，代码与文档必须一起同步。
2. 每次同步前后都要更新本进度文档。
3. 项目标题固定为：
   - **FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**
4. 模型目录只保留占位，不上传权重内容。
5. `shared / dedicated / full-trace` 的接口继续保留，但不作为当前主线默认路径。
6. `effective_capacity_admission_enabled` 的 on/off 接口继续保留，但当前默认配置已切到 `on`。

## 建议的下一步

1. 以当前已经冻结的 `Qwen2.5-14B-Instruct @ gpu_memory_utilization=0.85` 作为 14B 默认主线结果。
2. 保持 `Qwen2.5-7B-Instruct TP=1` 为默认路径，同时保留 `TP=2` 作为吞吐导向的正式对照 profile。
3. 保持 `Mistral-7B-Instruct-v0.3` 这轮 `PEFT+finetune + 500 adapters + representative r1000` 结果作为第二家族 7B 档基线。
4. 先完成 `mistralai/Mistral-Nemo-Instruct-2407` 的论文主线 `TP=2 + PEFT+finetune + 500 adapters + representative r1000`。
5. 随后补齐 `Qwen 7B / Qwen 14B / Mistral 7B` 的 `representative r1000`，统一验证四个目标模型在当前项目中的正式主线路径。
6. 仅在四个模型主线都收口后，再进入 `ServerlessLLM` 的对比实验。

补充说明：

- `mistralai/Mistral-7B-Instruct-v0.3` 与 `mistralai/Mistral-Nemo-Instruct-2407` 都是 instruct 路线，和当前 Qwen 主线更可比。
