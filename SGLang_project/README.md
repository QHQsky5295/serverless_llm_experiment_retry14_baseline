# SGLang 基线项目

本目录用于在 `/home/qhq/serverless_llm_baselines` 工作区内，以隔离方式复现
`SGLang`，并将其接入当前 many-LoRA 公平对比实验链。

## 目标

1. 不修改 `/home/qhq/serverless_llm_experiment` 主项目代码与环境。
2. 不污染 `/home/qhq/serverless_llm_experiment_retry14_baseline` 的正式实验环境。
3. 在独立项目目录中完成：
   - 官方仓库复现
   - 独立环境准备
   - shared trace / shared LoRA subset 接入
   - 指标口径对齐

## 目录说明

- [repo](/home/qhq/serverless_llm_baselines/SGLang_project/repo)：官方 `SGLang` 仓库映射，实际指向 `/home/qhq/serverless_llm_baselines/repos/SGLang`
- [docs](/home/qhq/serverless_llm_baselines/SGLang_project/docs)：中文复现说明与实验记录
- [scripts](/home/qhq/serverless_llm_baselines/SGLang_project/scripts)：项目本地入口 wrapper，转发到共享公平实验 harness
- [configs](/home/qhq/serverless_llm_baselines/SGLang_project/configs)：基线配置目录
- [results](/home/qhq/serverless_llm_baselines/SGLang_project/results)：结果目录映射，实际指向共享 baseline 结果目录
- [logs](/home/qhq/serverless_llm_baselines/SGLang_project/logs)：日志目录映射
- [environments](/home/qhq/serverless_llm_baselines/SGLang_project/environments)：环境说明目录映射

## 当前状态

- `SGLang_project` 项目目录已建立
- 已完成：
  - 上游仓库拉取：`/home/qhq/serverless_llm_baselines/repos/SGLang`
  - 隔离虚拟环境 `/home/qhq/.venvs/sglang_py310`
  - official multi-LoRA 能力核查
  - many-LoRA 公平 replay 脚本接入
  - `Llama-2 7B + sanitized shared subset` 最小真实 GPU smoke（`4/4` 成功）
  - `Llama-2 13B / Qwen 7B / Qwen 14B` 正式 profile smoke
- 后续正式对比应从本项目入口调用：
  - `/home/qhq/serverless_llm_baselines/SGLang_project/scripts/prepare_shared_round_artifacts.sh`
  - `/home/qhq/serverless_llm_baselines/SGLang_project/scripts/run_sglang_fair_experiment.sh`
  - `/home/qhq/serverless_llm_baselines/SGLang_project/scripts/audit_e2e_v3_round.sh`

这些入口只做路径归位，不复制或改写底层逻辑。真正的公平 replay、summary 和 audit 实现仍保留在 `/home/qhq/serverless_llm_baselines/scripts`，防止不同 baseline 之间出现多份 harness 漂移。

## 当前结论

截至当前版本，`SGLang` 已不再局限于 `Llama-2 7B` 最小验证。  
它当前已完成四个正式模型 profile 的 smoke，并且正式公平 replay 已统一切到：

- 原生 `/generate`
- 提交 `input_ids`

这样可以避免旧 `/v1/completions` 路径在 `Qwen` 长上下文边界上的 prompt 解释偏差。

## 复现边界

当前 SGLang 复现不修改官方 SGLang 底层 serving 逻辑。我们只在外部实验 harness 中完成：

- shared trace 和 shared adapter subset 校验；
- LoRA path materialization；
- 统一 prompt/token budget；
- 统一 `e2e_v3` 指标采集；
- 与 FaaSLoRA、ServerlessLLM 一致的 replay/summary/audit。

因此，`SGLang_project` 是项目化入口，`repos/SGLang` 是官方源码，`.venvs/sglang_py310` 是运行环境，三者不要混淆。
