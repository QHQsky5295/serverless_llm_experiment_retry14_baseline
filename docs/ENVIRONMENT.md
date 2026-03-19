# FaaSLoRA 实验环境说明（当前稳定主线环境）

本文档记录当前项目主线实际使用并验证通过的环境组合，以及与仓库脚本之间的关系。
若文档、脚本和本机现有环境出现冲突，应以当前主线真实运行环境为准。

---

## 1. 当前稳定环境（已验证可用）

| 组件 | 版本 | 说明 |
|------|------|------|
| **conda 环境名** | `LLM_vllm0102` | 当前机器上用于主线实验的稳定环境 |
| **Python** | `3.12.12` | 本机当前实际版本 |
| **PyTorch** | `2.8.0+cu128` | 本机当前实际版本 |
| **vLLM** | `0.10.2` | 项目固定版本；当前主线实验均基于此版本完成 |
| **transformers** | `4.57.6` | 当前主线实验环境版本 |
| **numpy** | `2.2.6` | 当前主线实验环境版本 |
| **flashinfer-python** | `0.6.6` | 当前主线环境已安装；vLLM 采样路径显式请求使用 |
| **torch-c-dlpack-ext** | `0.1.5` | 当前主线环境已安装；用于补齐 FlashInfer 相关可选扩展 |

当前主线默认命令、`run_validation_bundle.sh` 和默认入口复验都已在这套环境下验证通过。

## 2. 当前环境与历史重装脚本的关系

仓库中仍保留了 `scripts/reinstall_env.sh`、`scripts/reinstall_gpu_env.sh` 等历史环境脚本。
这些脚本目前仍以旧的 `LLM / Python 3.11 / PyTorch cu124` 组合为目标，**不等于当前主线实际运行环境**。

因此当前最稳的做法是：

1. 直接复用现有 `LLM_vllm0102` 环境运行主线实验。
2. 不把历史重装脚本视为当前主线的权威环境来源。
3. 若后续确实需要重建环境，再以当前稳定环境版本为目标单独整理脚本。

---

## 3. 当前主线运行方式

当前主线实验直接使用下面这条默认入口命令：

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment
python scripts/run_all_experiments.py --config configs/experiments.yaml
```

---

## 4. 验证环境

在激活 `LLM_vllm0102` 后执行：

```bash
python -c "
import torch
import vllm
import transformers
import numpy
import sys
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('vLLM:', vllm.__version__)
print('transformers:', transformers.__version__)
print('numpy:', numpy.__version__)
"
```

当前期望输出应与本机稳定环境一致：

- `Python: 3.12.12`
- `PyTorch: 2.8.0+cu128`
- `CUDA available: True`
- `vLLM: 0.10.2`
- `transformers: 4.57.6`
- `numpy: 2.2.6`
- `pandas: 3.0.1`

---

## 5. 与项目配置的对应关系

- **pyproject.toml** 中已固定 `vllm==0.10.2`，与本文档一致。
- **scripts/run_validation_bundle.sh** 的默认 Python 入口已经固定为 `/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`。
- **configs/experiments.yaml** 当前默认入口已经切到 `mistral_7b_main_v2_publicmix + azure_sharegpt_rep1000 + mistral_7b_auto500_main`；当前主线口径为 `Qwen 7B / Mistral 7B -> TP=1 + max_instances=2`、`Qwen 14B / Mistral-Nemo -> TP=2 + max_instances=1`。
- **scripts/generate_lora_adapters.py** 当前默认会跟随 `profile_selection + model_profiles + workload_profiles` 解析 base model 与 adapter 数量；`PEFT+finetune` 模式已改为单次加载 base model 后循环生成多个 adapter。
- **scripts/run_all_experiments.py** 与 **faaslora/serving/vllm_wrapper.py** 当前默认已不再全局强制启用 FlashInfer sampler；只有显式 profile/运行时需要时才会打开。
- **当前本机 `LLM_vllm0102` 环境** 还包含一处本地兼容性修复：`vLLM` 的 Mistral sentencepiece tokenizer 适配层已显式改为 `SpecialTokenPolicy.IGNORE`，用来消除 `mistral_common` 的弃用 warning。该修复不改变解码语义，只是把当前已存在的默认行为显式化。
- 若在 3090 上仍遇 vLLM LoRA 崩溃，可参考 [docs/VLLM_RTX3090_LORA.md](VLLM_RTX3090_LORA.md)（如增大 `/dev/shm`、改用 `backend: "transformers"` 等）。

---

## 6. 当前文档边界

本文档现在只负责记录：

1. 当前主线真实可用的环境版本。
2. 当前项目默认配置与环境入口的对应关系。
3. 哪些历史环境脚本仍保留但不应当被视为当前主线权威来源。
