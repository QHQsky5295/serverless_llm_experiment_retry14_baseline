# FaaSLoRA 实验环境说明（当前 clean-tree 基线）

本文档记录当前干净树 `serverless_llm_experiment_retry14_baseline` 的真实运行环境、启动方式和本机约束。

## 0. 2026-04-13 当前同步口径

- 当前 active 主线分支：`retry14_continuous_queue_v2`
- 当前正式论文模型家族已固定为：
  - `Qwen 7B`
  - `Qwen 14B TP=2`
  - `Llama-2 7B`
  - `Llama-2 13B TP=2`
- 当前正式综合主指标已固定为：
  - `CE = 1 / (avg_e2e_sec * avg_cost_usd)`
- 当前 live 与 summary 中需要共同展示的核心指标为：
  - `CE`
  - `TTFT_overall`
  - `TTFT_comp`
  - `TTFT_warm`
  - `TPOT`
  - `Tok/s`
  - `E2E`
  - `ColdStart`
  - `ScaleUpAffectedTTFT`
- 当前最新本地测试状态：
  - `tests.test_basic_smoke = 228/228 OK`

## 0. 2026-04-12 当前同步口径

- 当前 active 主线分支：`retry14_continuous_queue_v2`
- 当前最新已正式分析、且最可信的 7B checkpoint：`retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`
- 当前最新已正式分析、且最可信的 14B checkpoint：`retry14_continuous_queue_v2_qwen14b_r500_a500_main_baseline45_poststartup_elapsed @ 500`
- 当前最新已正式分析、且最可信的另一模型家族 7B checkpoint：`retry14_continuous_queue_v2_mistral7b_r500_a500_main_baseline3_maxloras4 @ 500`
- 当前最新本地测试状态：
  - `tests.test_basic_smoke = 228/228 OK`
- 当前下一步 active 主线将转入：
  - `Mistral Nemo TP=2`
  - 仍在同一 `continuous_queue_v2` runner 语义下进行
- 当前环境层面必须特别说明：
  - 当前机器是 `4 x RTX 3090 24GB`
  - 因此 `TP=2` 路径的物理上限是 `2` 个双卡 runtime，而不是旧两卡时期的 `1`
- 当前正式长跑建议继续使用：
  - `scripts/run_all_experiments_user_scope.sh`
  - 避免直接从即将关闭的 SSH/session scope 启动，触发 systemd closing-session guard

## 0. 2026-04-11 当前同步口径

- 当前 active 主线分支：`retry14_continuous_queue_v2`
- 当前最新已正式分析、且最可信的 7B checkpoint：`retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`
- 当前 7B 正式 workload：
  - `Qwen/Qwen2.5-7B-Instruct`
  - `4 x RTX 3090 24GB`
  - `500 adapters`
  - `500 representative requests`
  - `Azure real trace arrivals + Azure token distribution + ShareGPT prompts`
  - `time_scale_factor = 1.0`
- 当前下一步 active 主线将转入：
  - `Qwen 14B TP=2`
  - 然后转入另一模型家族的 `7B / 10B~14B` 级别验证
  - 仍在同一 `continuous_queue_v2` runner 语义下进行
- 当前正式长跑建议继续使用：
  - `scripts/run_all_experiments_user_scope.sh`
  - 避免直接从即将关闭的 SSH/session scope 启动，触发 systemd closing-session guard
- 当前本地测试状态：
  - `tests.test_basic_smoke = 228/228 OK`

## 0. 2026-04-09 当前同步口径

- 当前 active 主线分支：`retry14_continuous_queue_v2`
- 当前最新已正式分析的 7B 有效结果：`retry14_continuous_queue_v2_qwen7b_r500_baseline34_multiruntime_routeaware @ 500`
- 当前 7B 正式 workload：
  - `Qwen/Qwen2.5-7B-Instruct`
  - `4 x RTX 3090 24GB`
  - `500 adapters`
  - `500 representative requests`
  - `Azure real trace arrivals + Azure token distribution + ShareGPT prompts`
  - `time_scale_factor = 1.0`
- 当前下一步 active 主线将转入：
  - `Qwen 14B TP=2`
  - 仍在同一 `continuous_queue_v2` runner 语义下进行
- 当前正式长跑建议继续使用：
  - `scripts/run_all_experiments_user_scope.sh`
  - 避免直接从即将关闭的 SSH/session scope 启动，触发 systemd closing-session guard
- 当前本地测试状态：
  - `tests.test_basic_smoke = 194/194 OK`

## 1. 当前稳定环境

| 组件 | 当前值 | 说明 |
|---|---|---|
| Conda 环境 | `LLM_vllm0102` | 当前主线实验统一使用 |
| Python | `3.12.12` | 本机已验证 |
| PyTorch | `2.8.0+cu128` | 本机已验证 |
| vLLM | `0.10.2` | 当前主线固定版本 |
| transformers | `4.57.6` | 当前主线已验证 |
| numpy | `2.2.6` | 当前主线已验证 |
| flashinfer-python | `0.6.6` | 已安装，但当前部分 profile 显式关闭 sampler |
| torch-c-dlpack-ext | `0.1.5` | 已安装 |

当前实验统一使用：

- Python：`/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`
- 仓库：`/home/qhq/serverless_llm_experiment_retry14_baseline`

## 2. clean-tree 与历史脏树

当前研究和回退统一以：

- ` /home/qhq/serverless_llm_experiment_retry14_baseline`

为准。

历史目录：

- `/home/qhq/serverless_llm_experiment`

仍可能保留模型权重或旧工件，但不再作为正式实验主树。

## 3. 当前默认运行方式

### 3.1 通用入口

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python scripts/run_all_experiments.py --config configs/experiments.yaml
```

说明：

- `configs/experiments.yaml` 顶层 `profile_selection` 当前仍保留 `Qwen 14B TP=2` 默认入口。
- 当前 7B rollback 主线实验通常通过环境变量覆盖 profile 选择与请求数，不直接依赖顶层默认入口。

### 3.2 当前主线回归入口（Qwen 7B V2 publicmix，500 requests）

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline

FAASLORA_PROFILE_MODEL=qwen_7b_main_v2_publicmix \
FAASLORA_PROFILE_DATASET=azure_sharegpt_rep1000 \
FAASLORA_PROFILE_WORKLOAD=qwen_7b_auto500_main \
FAASLORA_TOTAL_REQUESTS=500 \
FAASLORA_PYTHON=/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python \
PYTHONUNBUFFERED=1 \
VLLM_NO_USAGE_STATS=1 \
bash scripts/run_all_experiments_user_scope.sh \
  --config configs/experiments.yaml \
  --scenario faaslora_full
```

## 4. tmux 与 shell 约定

本机 `tmux` 当前安装在：

- `/home/qhq/.local/bin/tmux`

如果当前 shell 看不到 `tmux`，先执行：

```bash
source ~/.bashrc
```

必要时也可手动补：

```bash
export PATH="$HOME/.local/bin:$PATH"
hash -r
```

## 5. 模型与工件目录

当前模型权重与工件目录存在两个事实：

1. clean-tree 是当前权威代码树。
2. 模型权重可能仍共享历史目录下的本地路径，例如：
   - `/home/qhq/serverless_llm_experiment/models/...`

这不会影响当前 clean-tree 作为代码与实验主树的地位，但在迁移机器或重建环境时要显式检查这些绝对路径。

当前 Qwen 7B V2 publicmix 主线使用：

- remote artifact dir：`artifacts/frozen/qwen_7b_a500_v2_publicmix`
- curated manifest：`configs/generated/lora_manifest_1000.json`

## 6. 当前已落地的论文主指标

终端 live 和实验结束 summary 当前已统一打印：

- `TTFT_overall`
- `TTFT_comparable`
- `TTFT_scaleup_affected`
- `TTFT_gpu_ready`
- `TPOT`
- `Throughput_req/s`
- `Throughput_tok/s`
- `E2E_latency`
- `SLO_attainment`

实验结束 summary 与结果 JSON 另外补充：

- `Cold_start_latency`
- `Monetary_cost`

其它诊断指标仍会继续写入日志和 JSON。

## 7. 环境验证命令

```bash
/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python -c "
import sys, torch, vllm, transformers, numpy
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU0:', torch.cuda.get_device_name(0))
print('vLLM:', vllm.__version__)
print('transformers:', transformers.__version__)
print('numpy:', numpy.__version__)
"
```

## 8. 当前边界

本文档只负责说明：

1. 当前 clean-tree 的真实可用环境
2. 当前主线实验入口
3. 当前 shell / tmux 使用边界
4. 当前论文主指标输出口径

如果文档与代码冲突，以当前代码实现和 `docs/PROJECT_PROGRESS.md` 为准。
