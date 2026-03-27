# FaaSLoRA 实验执行指南（当前 clean-tree）

> 权威口径以 [README.md](README.md)、[docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md) 和 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) 为准。

## 1. 当前推荐环境

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment_retry14_baseline
```

当前稳定环境见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 2. 当前主线对象

当前 rollback 主线聚焦于：

- 模型：`Qwen2.5-7B-Instruct`
- profile：`qwen_7b_main_v2_publicmix`
- dataset：`azure_sharegpt_rep1000`
- workload：`qwen_7b_auto500_main`
- adapters：`500`
- requests：`500`
- scenario：`faaslora_full`

## 3. 运行当前 baseline

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

## 4. tmux 运行方式

若希望在 tmux 中跑实验：

```bash
tmux new-session -s qwen7b_baseline
```

若当前 shell 找不到 `tmux`，先执行：

```bash
source ~/.bashrc
```

## 5. 结果位置

当前标准输出包括两部分：

1. 终端 live 面板
2. 结果 JSON

常见位置：

- launch log：`/tmp/<tag>.launch.log`
- result json：`results/*.json`

## 6. 当前 live 与 summary 主指标

终端 live 和实验结束 summary 当前统一展示：

- `TTFT_overall`
- `TTFT_comparable`
- `TTFT_scaleup_affected`
- `TTFT_gpu_ready`
- `TPOT`
- `Throughput_req/s`
- `Throughput_tok/s`
- `E2E_latency`
- `SLO_attainment`

实验结束 summary 与结果 JSON 还会补充：

- `Cold_start_latency`
- `Monetary_cost`

其余诊断指标仍会继续写入日志和 JSON，用于归因与后续优化。

## 7. 当前实验纪律

1. 当前正式对比统一以 clean-tree 为准。
2. `retry21` 与历史脏树结果不再作为正式比较对象。
3. 代码修改前先明确它服务于哪些论文主指标。
4. 结果分析时必须同时看：
   - 本轮运行情况
   - 与上一轮相比的变化
   - 造成变化的代码或系统原因

## 8. 当前未完成事项

- `REMOTE` 真实性升级暂缓，后续切到双机真实 remote 存储
- 当前仍需继续优化 GPU resident stickiness 与 tail TTFT
- 下一轮 baseline 请优先用于验证最新代码收口是否稳定
