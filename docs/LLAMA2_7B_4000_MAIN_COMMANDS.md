# Llama-2 7B 4000-request Main Comparison Commands

本文件记录 `Llama-2 7B / 4000 requests / 500 adapters / s8 / hotset rotation 500`
五系统主横向对比的执行指令。按顺序执行，不要并发跑多个系统。

优先使用连续执行脚本：

- 脚本：`/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh`
- 一个系统结束后会清理已知遗留服务并检查 GPU compute state。
- 如果某个系统失败，修复后保留同一个 `FAIR_ROUND_DIR` 重新运行脚本，会跳过已完成阶段，从失败阶段继续。
- 所有 shared artifacts、raw replay、summary、logs、FaaSLoRA 结果副本、最终 compare 都会收进同一个时间戳 round 目录。
- 默认会自动清理命令行匹配本实验工程的 GPU 残留进程；如需只检查不自动杀残留，可设置 `FAIR_ROUND_KILL_KNOWN_GPU_RESIDUALS=0`。

统一设置：

- `RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`
- `MODEL_PROFILE=llama2_7b_main_v2_publicmix`
- `DATASET_PROFILE=azure_sharegpt_rep4000`
- `WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8`
- `TOTAL_REQUESTS=4000`
- `SELECTED_NUM_ADAPTERS=500`
- `SEED=42`

共享输入：

连续 runner 会在本轮时间戳目录下生成：

- Trace: `$FAIR_ROUND_DIR/shared_artifacts/${RUN_TAG}_trace.json`
- Adapter subset: `$FAIR_ROUND_DIR/shared_artifacts/${RUN_TAG}_adapter_subset.json`

正式实验不要再使用旧的 `results/shared_rounds` 手动路径作为主输入。连续
runner 会把本轮 shared artifacts 固定保存在 timestamped round 目录中，后续
取数、复跑和审计都应以该目录为准。

进入 tmux 查看时，如果当前不在 tmux 中，使用：

```bash
tmux attach -t <session_name>
```

如果当前已经在 tmux 中，使用：

```bash
tmux switch-client -t <session_name>
```

## 0. 清理旧会话和检查 GPU

```bash
tmux kill-session -t prep_l7_4000_mainv1 2>/dev/null || true
tmux kill-session -t sglang_l7_4000_mainv1 2>/dev/null || true
tmux kill-session -t sllm_l7_4000_mainv1 2>/dev/null || true
tmux kill-session -t vllm_l7_4000_mainv1 2>/dev/null || true
tmux kill-session -t slora_l7_4000_mainv1 2>/dev/null || true
tmux kill-session -t faas_l7_4000_mainv1 2>/dev/null || true

bash /home/qhq/serverless_llm_baselines/scripts/stop_serverlessllm_stack.sh 2>/dev/null || true

nvidia-smi
```

## A. 推荐：连续执行五系统完整 round

```bash
tmux new-session -d -s l7_4000_mainv1_round '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
FAIR_ROUND_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
FAIR_ROUND_GPU_IDS=0,1,2,3 \
FAIR_ROUND_KILL_KNOWN_GPU_RESIDUALS=1 \
bash /home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] full five-system round completed successfully."
else
  echo "[failed] full five-system round failed with status=$status."
  echo "Use the FAIR_ROUND_DIR printed near the top, or source the round.env file inside the created round directory, then rerun the same script to resume."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t l7_4000_mainv1_round
```

如果当前已经在 tmux 中：

```bash
tmux switch-client -t l7_4000_mainv1_round
```

失败后断点接续优先使用：

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

下面的分段命令只作为紧急 fallback；正式运行优先用上面的连续脚本或
`resume_fair_round_tmux.sh`。

## 1. 生成 shared trace 和 adapter subset

```bash
tmux new-session -d -s prep_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq/serverless_llm_baselines

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
bash /home/qhq/serverless_llm_baselines/scripts/prepare_shared_round_artifacts.sh

echo
echo "[finished] shared artifacts ready."
exec bash
'
```

进入终端查看：

```bash
tmux attach -t prep_l7_4000_mainv1
```

如果已经在 tmux 里面，用：

```bash
tmux switch-client -t prep_l7_4000_mainv1
```

## 2. 跑 SGLang

```bash
tmux new-session -d -s sglang_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq/serverless_llm_baselines

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
SLLM_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
SLLM_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_adapter_subset.json \
SGLANG_GPU_IDS=0,1,2,3 \
SGLANG_DATA_PARALLEL_REPLICAS=4 \
SGLANG_TENSOR_PARALLEL_SIZE=1 \
bash /home/qhq/serverless_llm_baselines/scripts/run_sglang_fair_experiment.sh

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] SGLang completed successfully."
else
  echo "[failed] SGLang failed with status=$status."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t sglang_l7_4000_mainv1
```

## 3. 跑 ServerlessLLM

```bash
tmux new-session -d -s sllm_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq/serverless_llm_baselines

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
SLLM_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
SLLM_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_adapter_subset.json \
SLLM_BACKEND=vllm \
SLLM_WORKER_GPUS=0,1,2,3 \
bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_fair_experiment.sh

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] ServerlessLLM completed successfully."
else
  echo "[failed] ServerlessLLM failed with status=$status."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t sllm_l7_4000_mainv1
```

## 4. 跑 vLLM

```bash
tmux new-session -d -s vllm_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq/serverless_llm_baselines

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
SLLM_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
SLLM_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_adapter_subset.json \
VLLM_GPU_IDS=0,1,2,3 \
VLLM_DATA_PARALLEL_REPLICAS=4 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
bash /home/qhq/serverless_llm_baselines/scripts/run_vllm_fair_experiment.sh

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] vLLM completed successfully."
else
  echo "[failed] vLLM failed with status=$status."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t vllm_l7_4000_mainv1
```

## 5. 跑 S-LoRA

```bash
tmux new-session -d -s slora_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
cd /home/qhq/serverless_llm_baselines

SLLM_MODEL_PROFILE=llama2_7b_main_v2_publicmix \
SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
SLLM_WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8 \
SLLM_TOTAL_REQUESTS=4000 \
SLLM_SELECTED_NUM_ADAPTERS=500 \
SLLM_SAMPLING_SEED=42 \
SLLM_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
SLLM_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
SLLM_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_adapter_subset.json \
SLORA_GPU_IDS=0,1,2,3 \
SLORA_DATA_PARALLEL_REPLICAS=4 \
SLORA_TENSOR_PARALLEL_SIZE=1 \
bash /home/qhq/serverless_llm_baselines/scripts/run_slora_fair_experiment.sh

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] S-LoRA completed successfully."
else
  echo "[failed] S-LoRA failed with status=$status."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t slora_l7_4000_mainv1
```

## 6. 跑 FaaSLoRA

```bash
tmux new-session -d -s faas_l7_4000_mainv1 '
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment_retry14_baseline

FAASLORA_PROFILE_MODEL=llama2_7b_main_v2_publicmix \
FAASLORA_PROFILE_DATASET=azure_sharegpt_rep4000 \
FAASLORA_PROFILE_WORKLOAD=llama2_7b_auto500_formal4000_s8 \
FAASLORA_TOTAL_REQUESTS=4000 \
FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
FAASLORA_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_adapter_subset.json \
FAASLORA_RESULTS_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_faaslora \
PYTHONUNBUFFERED=1 \
bash /home/qhq/serverless_llm_experiment_retry14_baseline/scripts/run_faaslora_shared_artifact_experiment.sh \
  --num-adapters 500 \
  --full-stack

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "[finished] FaaSLoRA completed successfully."
else
  echo "[failed] FaaSLoRA failed with status=$status."
fi
exec bash
'
```

进入终端查看：

```bash
tmux attach -t faas_l7_4000_mainv1
```
