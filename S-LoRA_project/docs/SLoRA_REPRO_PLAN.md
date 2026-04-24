# S-LoRA 复现计划与准入边界

## 1. 为什么复现 S-LoRA

S-LoRA 是 MLSys 2024 的 multi-LoRA serving 论文系统，官方目标是服务大量
concurrent LoRA adapters。它的核心机制包括：

- Unified Paging：统一管理 LoRA adapter weights 与 KV cache；
- Heterogeneous batching：支持不同 adapter/rank 的混合 batch；
- LoRA tensor parallelism：降低多 GPU LoRA 计算通信开销。

这些问题与 PrimeLoRA 的 adapter readiness、hierarchical residency 和资源协同
高度相关。因此，S-LoRA 应作为 serverful multi-LoRA serving paper baseline，
但不能被当作 serverless baseline。

## 2. 当前官方源码事实

- 官方 repo: `/home/qhq/serverless_llm_baselines/repos/S-LoRA`
- 本项目软链接: `/home/qhq/serverless_llm_baselines/S-LoRA_project/repo`
- 官方 README 推荐：
  - CUDA 11.8；
  - PyTorch 1.13 到 2.0.1；
  - Triton 2.1.0；
  - Ampere GPU。
- 官方 benchmark 默认模型族：
  - `huggyllama/llama-7b`
  - `huggyllama/llama-13b`
- 官方 server 原生接口：
  - `/generate`
  - `/generate_stream`
  - `/v1/chat/completions`

## 3. 已识别的适配风险

### 3.1 API 语义

S-LoRA 原生 `/generate_stream` 请求格式是：

```json
{
  "inputs": "...",
  "lora_dir": "/path/to/adapter",
  "parameters": {
    "do_sample": false,
    "ignore_eos": true,
    "max_new_tokens": 128
  }
}
```

因此不能直接复用 vLLM 的 `adapter_id -> model` 映射，也不能直接复用
SGLang 的 `adapter_id -> lora_path` 假设。正式 replay 必须把 shared subset 中的
`adapter_id` 映射到 S-LoRA 启动时注册的真实 `lora_dir` 字符串。

### 3.2 Chat endpoint 风险

官方 `/v1/chat/completions` 看起来更接近 OpenAI API，但当前源码中该路径没有把
LoRA adapter 显式传入 `httpserver_manager.generate()`。因此正式 harness 优先使用
`/generate_stream`，避免 chat endpoint 在 LoRA 选择上出现静默 base-model fallback。

### 3.3 环境风险

S-LoRA 依赖 CUDA extension、旧版 PyTorch 和 Triton。必须使用独立环境，不能污染：

- FaaSLoRA 的 `LLM_vllm0102` 环境；
- SGLang 的 `sglang_py310` 环境；
- ServerlessLLM 的 `sllm_head_official` 环境。

## 4. 正式接入步骤与当前进度

1. 建立或确认 `slora_official_cu118` 独立环境。
   - 当前本机已验证：
     `/home/qhq/anaconda3/envs/slora_official_cu118/bin/python`。
   - 该环境已使用 `nvidia/label/cuda-11.8.0` 安装一致 CUDA 11.8 toolkit。
   - 不再使用默认 `nvidia` channel 安装 CUDA，否则会混入 CUDA 12/13 CCCL，
     导致 S-LoRA 官方 extension 编译失败。
   - 不能复用 `LLM_vllm0102`，因为它是 PyTorch 2.8 / Triton 3.x，不符合官方依赖边界。
2. 编译 S-LoRA CUDA extension。
   - 官方推荐 CUDA 11.8、PyTorch 1.13 到 2.0.1、Triton 2.1.0。
   - 编译失败不能 fallback 到其它 backend，否则不再是 S-LoRA。
3. 用 1 个真实 Llama-2 7B LoRA adapter 做 smoke。
4. 用 shared adapter subset 生成 S-LoRA launch spec。
   - 已完成 dry-run 验证，当前脚本可以生成：
   - `adapter_id -> adapter_path`
   - `--lora-dirs adapter_path`
   - DP4/TP1 replica 端口：
     `8463, 8473, 8483, 8493`
5. 扩展统一 replay 或新增 S-LoRA replay wrapper。
   - 已完成：统一 replay 新增 `--slora-native-generate`。
   - 已完成：请求体按官方 `/generate_stream` 转换为：
     `inputs + parameters.max_new_tokens + lora_dir`。
   - 已完成：`adapter_id` 通过 adapter value map 转成真实 `lora_dir`。
6. replay 与 summary 必须输出：
   - shared trace -> rendered prompt；
   - prompt guard 与其它系统一致；
   - `adapter_id -> lora_dir`；
   - `/generate_stream`；
   - 真实 client-observed TTFT/E2E/TPOT；
   - 本地 tokenizer token count 审计。
7. 用 `500 requests / 500 adapters / s24` 做 bring-up。
8. 通过后才允许进入 `4000 requests` 正式主横向对比。

## 5. 指标边界

S-LoRA 横向图只使用所有系统可观测字段：

- `TTFT_e2e avg/p95/p99`
- `TTFT_service avg/p95/p99`
- `E2E_e2e avg/p95/p99`
- `E2E_service avg/p95/p99`
- `TPOT`
- `Tok/s`
- `RPS`
- `SLO`
- `Cost/req`
- `Cost/1M tokens`
- `CE`
- GPU-second resource metrics

以下字段不进入横向主图：

- S-LoRA 内部 adapter paging event；
- FaaSLoRA 的 GPU hit / preload match / scale-up first-service 等机制字段；
- 任何需要估计或填 `null` 的内部机制指标。

## 6. 下一步 gate

vLLM token/TPOT 统计隐患已在 replay 层修复，并新增正式 gate：

- 空输出但 HTTP 200 的 streaming 响应会记为 `local_generated_text_empty`，
  completion token 数为本地 tokenizer 对空文本的真实计数，而不是 raw trace expected；
- vLLM 和 S-LoRA 正式脚本都会检查 token source；
- 如果任一成功请求仍使用 `trace_expected` token source，脚本直接失败。

S-LoRA 下一步 gate：

1. 先建立 `slora_official_cu118` 独立环境并完成官方 extension 编译。
2. 用本文档中的 dry-run 指令确认 shared artifacts、端口和 launch spec 无误。
3. 再跑真实 `500 requests / 500 adapters / s24` bring-up。
4. bring-up 成功后，才能把 S-LoRA 加入全系统横向对比。

## 2026-04-24 补充：native prompt budget 根因

在 `fix4` 这一轮真实 GPU replay 中，S-LoRA 仍出现：

- `396/500` 成功
- `104/500` 失败
- 服务端真实异常统一为：
  `ValueError: the input prompt token len 985 is too long > 984`

根因不是旧 guard 没有执行，而是 guard 语义与 S-LoRA 服务端语义不一致：

- replay 之前按 `tokenizer.encode(prompt, add_special_tokens=False)` 预算输入；
- S-LoRA 官方 `/generate_stream` 服务端在
  `slora/server/httpserver/manager.py` 中使用 `tokenizer.encode(prompt)`，
  默认会计入 special tokens；
- 因此会出现：
  - 本地 guard 计成 `984`
  - 服务端真实计成 `985`

已经完成的根因修复：

- `replay_openai_trace.py` 在 `--slora-native-generate` 打开时，prompt guard 改为按
  `add_special_tokens=True` 做：
  - 初始 encode
  - decode -> re-encode 边界校验
  - `actual_input_tokens` 计算

离线验证结论：

- 对同一份 `shared trace` 全量 `500` 请求复算后：
  - `violations_true_gt_984 = 0`
  - `max_true = 984`

因此，后续 S-LoRA 正式 replay 只有在使用这条修复后的 guard 语义重跑并达到
`500/500` 成功后，才能进入五系统横向主表。

## 2026-04-24 补充：`fix5` 已通过 500-request debug gate

使用上述 native prompt budget 修复后，S-LoRA 已完成有效 bring-up：

- run tag：`llama2_7b_r500_a500_seed42_s24_vllmformal1_fix5_slora_dp4_tp1`
- summary：`/home/qhq/serverless_llm_baselines/results/replay/llama2_7b_r500_a500_seed42_s24_vllmformal1_fix5_slora_dp4_tp1_summary.json`
- completed：`500/500`
- failed：`0/500`
- `metric_schema_version = e2e_v3`
- `prompt_token_source_counts = {'local_guarded_prompt': 500}`
- `completion_token_source_counts = {'local_generated_text': 500}`

本轮可以作为 `Llama-2 7B / 500 requests / 500 adapters` debug gate 的有效 S-LoRA 结果。正式论文主结论仍需重跑 `4000 requests`，并继续遵守本文档的边界：不修改 S-LoRA 算法逻辑，不用其它 backend 冒充 S-LoRA，不使用无法跨系统观测的内部机制字段做横向图。

## 7. 推荐环境建立命令

如果本机没有 `slora_official_cu118`，按官方依赖边界优先建立独立环境。以下命令是复现入口，
不是论文性能实验本身；环境建立成功后再运行正式 fair replay。

```bash
conda create -n slora_official_cu118 python=3.9 -y
conda activate slora_official_cu118

conda install -n slora_official_cu118 -c nvidia/label/cuda-11.8.0 cuda -y
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install triton==2.1.0 rpyc aiohttp einops fastapi ninja packaging pyzmq safetensors uvloop uvicorn psutil
pip install transformers==4.34.0 tokenizers==0.14.1

cd /home/qhq/serverless_llm_baselines/repos/S-LoRA
export CUDA_HOME=/home/qhq/anaconda3/envs/slora_official_cu118
export PATH=/home/qhq/anaconda3/envs/slora_official_cu118/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"
pip install -e .
```

如果 `pip install -e .` 因 CUDA / PyTorch / Triton ABI 报错，必须修 S-LoRA 环境或
extension 编译；不能切换到其它 serving backend 冒充 S-LoRA。

已踩坑记录：

- 不要用 `conda install -c nvidia cuda-toolkit=11.8` 或默认 `nvidia` channel 安装
  CUDA。该路径会把 CUDA 13/12 的 `cuda-cccl` 等包混入环境，即使 `nvcc` 是 11.8，
  也会在编译时出现 `cuda/std/type_traits` 或 CCCL 版本错误。
- S-LoRA 官方 extension 依赖 PyTorch C++ 动态库，运行 server 时需要把
  `${CONDA_PREFIX}/lib/python3.9/site-packages/torch/lib` 放入 `LD_LIBRARY_PATH`。
  当前 `run_slora_fair_experiment.sh` 已自动设置。
- `transformers 4.57+` 会在 `torch==2.0.1` 下禁用 PyTorch backend，正式环境固定
  `transformers==4.34.0`。
