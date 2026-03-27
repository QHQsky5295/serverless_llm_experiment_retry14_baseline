# FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知 Serverless 系统

FaaSLoRA 是一个面向真实云工作负载的多 LoRA Serverless 推理研究原型。当前 clean-tree 运行在单节点双 RTX 3090 服务器上，围绕三项核心贡献展开：

1. 命中感知的请求放置与 scale-up 预加载
2. `GPU -> HOST -> NVMe -> REMOTE` 的分层工件驻留
3. LoRA 加载、KV cache 与批推理之间的资源协同控制

## 当前 clean-tree 快照（2026-03-27）

- 当前权威代码树：`/home/qhq/serverless_llm_experiment_retry14_baseline`
- 当前工作分支：`retry14_rebuild`
- 历史脏树：`/home/qhq/serverless_llm_experiment`
- 当前 rollback 主线：`Qwen 7B V2 publicmix + 500 adapters + 500 requests`

当前快照已经包含并验证过的关键修复：

- `NVMe-hit -> HOST promotion` 真实生效
- 多 GPU 口径修复不再只盯 `device 0`
- `scaleup_affected` 升级成请求级真标签
- 终端 live 与 run summary 已统一围绕论文主指标输出
- 若干暗箱 heuristics 已显式配置化，但保持当前有效行为不变

## 当前论文主指标

后续优化和论文主表统一围绕以下指标：

- `TTFT_overall`
- `TTFT_comparable`
- `TTFT_scaleup_affected`
- `TTFT_gpu_ready`
- `TPOT`
- `Throughput_req/s`
- `Throughput_tok/s`
- `E2E_latency`
- `SLO_attainment`

另外，实验结束 summary 与结果 JSON 还会补充输出两条论文辅助指标：

- `Cold_start_latency`
- `Monetary_cost`

其中：

- `TTFT_comparable` 当前定义为：非 `scale-up` 受影响，且 `cache_tier ∈ {gpu, host, nvme}` 的请求 TTFT
- `TTFT_scaleup_affected` 当前定义为：真实落到 scale-up 新增 runtime 且当时不是 `GPU-ready` 的请求 TTFT
- `Cold_start_latency` 当前定义为：dedicated scale-up 从开始创建新实例到新实例 warmup 完成的真实时延
- `Monetary_cost` 当前输出为：`avg_cost_usd` 与 `total_cost_usd`

## 当前主线运行方式

### 环境

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment_retry14_baseline
```

### 当前 7B rollback baseline 入口

```bash
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

说明：

- `configs/experiments.yaml` 顶层默认 `profile_selection` 仍保留 14B 入口。
- 当前 rollback 主线通过环境变量覆盖 profile 选择与请求数。

## 当前仓库边界

Git 跟踪：

- source code
- configs
- docs
- tests
- 当前刻意 curated 的 `configs/generated/lora_manifest_1000.json`

Git 不跟踪：

- `results/`
- `artifacts/`
- `data/`
- 模型权重
- `/tmp` 运行日志

## 重要说明

- 当前 `REMOTE` 物理路径仍是“本地目录 + 带宽仿真”的半模拟实现；双机真实 remote 存储升级已列入 TODO，但本次快照不修改。
- 论文主线当前不要再使用 `retry21` 或历史脏树结果做正式比较。
- 详细进度见 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md)。
- 当前技术口径见 [docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md)。
