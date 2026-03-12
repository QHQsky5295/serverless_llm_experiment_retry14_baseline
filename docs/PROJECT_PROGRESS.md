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

当前主线结论：

1. `auto` 主路径已真实跑通，扩缩容、warmup、preload、residency、resource coordination 都进入了完整链路。
2. 当前主线瓶颈已经从 admission / defer 转移到 serving 参数调优问题，并已通过 `seq8_lora8` 基本解决。
3. 当前 `auto500 + representative1000 + seq8_lora8` 可以作为项目主线的工作基线。
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

### 不再作为当前主线推进的内容

以下内容的实现和配置入口继续保留，但不再作为当前主线推进对象：

- `shared`
- `dedicated`
- `28185 full trace`
- `effective_capacity_admission_enabled`（P2.5 风格实验路径）

它们目前的定位是：

- 内部 sanity check
- 备用补充实验
- 压力测试入口
- 后续扩展可复用接口

## 当前工程状态

### 已修复

1. `run_all_experiments.py` 中 preset 对显式环境变量的覆盖问题。
2. user-scope 启动脚本未透传 `FAASLORA_*` 环境变量的问题。
3. 结果文件命名冲突问题。
4. 主线 serving 参数不可显式覆盖的问题。
5. `configs/experiments.yaml` 的默认主线路径已固化为当前验证通过的 `auto500 + representative1000 + seq8_lora8` 配置。
6. 默认入口复验已完成，默认命令路径与冻结主线配置保持一致。
7. `faaslora.cli` 已补回，`pyproject` 的 console script 与 autoscaler 的 `python -m faaslora.cli coordinator ...` 子进程入口不再悬空。

### 仍待完成

1. 稳定环境下可执行的基础测试仍待补齐到更完整覆盖面；当前仅补上了不依赖 GPU / 外部模型的 smoke tests。
2. 仍需继续清理 README / GUIDE / 其他附属文档与当前实现之间的漂移。
3. `Qwen2.5-7B-Instruct` 扩展仍待完成首轮 bring-up 验证与参数冻结。

## 模型与数据集扩展状态

### 当前状态

- Qwen 主线当前只推进 `Qwen2.5-3B-Instruct`
- 其他模型家族和额外数据集尚未开始主线接入

### 后续扩展目标

模型家族：

- Qwen：`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`
- Meta-Llama：`Llama-3.2-3B-Instruct`、`Meta-Llama-3.1-8B-Instruct`
- Gemma：`gemma-3-4b-it`、`gemma-3-12b-it`

数据集：

- `HuggingFaceH4/ultrachat_200k`
- `lmsys/lmsys-chat-1m`

进入这些扩展之前，需要先把当前 Qwen-3B 主线配置和文档稳定下来。

## 当前主线 TODO

### A. 主配置固化

1. 已完成：对 `auto500 + representative1000 + seq8_lora8` 做稳定性复验。
2. 已完成：将当前 serving 配置正式固化为主线默认复现实验参数。
3. 已完成：用默认入口再做一次复验，确认后续复现不依赖长串环境变量覆盖。
4. 已完成：把当前主线结果、配置和运行命令同步到所有核心文档。

### B. 工程闭环

5. 已完成：修复 CLI / packaging 断裂。
6. 继续补齐稳定环境下可跑的基础测试。
7. 清理 README / GUIDE / docs 与实现不一致的残留项。

### C. 扩展主线

8. 进行 `Qwen2.5-7B-Instruct` 的首轮 bring-up 验证。
9. 基于 bring-up 结果冻结 7B 的默认验证参数。
10. 再进入其他模型家族扩展。
11. 最后接入额外对话数据集。

## 当前已确认的长期约束

1. 每次同步 GitHub 时，代码与文档必须一起同步。
2. 每次同步前后都要更新本进度文档。
3. 项目标题固定为：
   - **FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**
4. 模型目录只保留占位，不上传权重内容。
5. `shared / dedicated / full-trace / effective_capacity_admission_enabled` 的接口继续保留，但不作为当前主线默认路径。

## 建议的下一步

1. 利用新的显式模型 / 硬件覆盖入口，先跑 `Qwen2.5-7B-Instruct` 的 100 adapters / 100 requests bring-up。
2. 如果 7B bring-up 稳定，再放大到更接近主线的 LoRA 规模与请求规模。
3. 扩展稳定后，再进入其他模型家族与额外数据集。
