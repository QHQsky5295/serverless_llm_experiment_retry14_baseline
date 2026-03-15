#!/usr/bin/env python3
"""
FaaSLoRA 大模型下载脚本
========================

将 HuggingFace 开源大模型下载至本地 models/ 目录，供 vLLM 推理使用。

支持的模型（适用于 RTX 3090 24 GB）：
  - Qwen/Qwen2.5-0.5B-Instruct           (~1 GB，快速验证用)
  - Qwen/Qwen2.5-7B-Instruct             (~15 GB，Qwen 单卡档)
  - Qwen/Qwen2.5-14B-Instruct            (~29 GB，Qwen 双卡档)
  - mistralai/Mistral-7B-Instruct-v0.3   (~14 GB，当前第二家族 7B 主线)
  - mistralai/Mistral-Nemo-Instruct-2407 (~24 GB，当前第二家族 12B/13B 档主线)
  - facebook/opt-1.3b                    (~2.6 GB，仅下载用途；当前 vLLM LoRA 路线不推荐)

用法：
  conda activate LLM
  python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
  python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --token hf_xxxxx
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# 推荐模型列表
RECOMMENDED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "size_gb": 1.0,
        "desc": "快速验证用，1 GB，无需授权",
        "vram_gb": 2,
        "yaml_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "profile": None,
        "workload": None,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "size_gb": 14.5,
        "desc": "论文实验推荐，RTX 3090 单卡可运行",
        "vram_gb": 18,
        "yaml_name": "Qwen/Qwen2.5-7B-Instruct",
        "profile": "qwen_7b_main",
        "workload": "qwen_7b_auto100_main",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "size_gb": 29.0,
        "desc": "大规模对比实验，需要双 3090（tensor_parallel=2）",
        "vram_gb": 30,
        "yaml_name": "Qwen/Qwen2.5-14B-Instruct",
        "profile": "qwen_14b_tp2",
        "workload": "qwen_14b_tp2_main",
    },
    "facebook/opt-1.3b": {
        "size_gb": 2.6,
        "desc": "无需授权，适合无 HuggingFace Token 的环境",
        "vram_gb": 4,
        "yaml_name": "facebook/opt-1.3b",
        "profile": None,
        "workload": None,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "size_gb": 14.0,
        "desc": "无需授权，Mistral 当前 7B 主线（vLLM LoRA 支持）",
        "vram_gb": 18,
        "yaml_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "profile": "mistral_7b_main",
        "workload": "mistral_7b_auto500_main",
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "size_gb": 24.0,
        "desc": "无需授权，双 3090 + TP=2；Mistral 当前 12B/13B 左右主线",
        "vram_gb": 30,
        "yaml_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "profile": "mistral_nemo_12b_tp2",
        "workload": "mistral_nemo_12b_tp2_main",
    },
}


def validate_token(token: str | None):
    """Fail fast on common placeholder / encoding mistakes."""
    if not token:
        return

    try:
        token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Hugging Face token 包含非 ASCII 字符。请把示例占位符替换成真实的 hf_... token，"
            "不要直接使用“你的_hf_token”这类中文占位符。"
        ) from exc

    if not token.startswith("hf_"):
        print("  [提示] 当前 token 不是以 'hf_' 开头；请确认它是有效的 Hugging Face Access Token。")


def download_model(model_id: str, token: str = None, force: bool = False):
    """从 HuggingFace Hub 下载模型到本地 models/ 目录。"""
    from huggingface_hub import snapshot_download

    safe_name = model_id.replace("/", "--")
    local_dir  = MODELS_DIR / safe_name

    if local_dir.exists() and not force:
        # 检查是否有实际文件（不只是空目录）
        files = list(local_dir.rglob("*.safetensors")) + list(local_dir.rglob("*.bin"))
        if files:
            print(f"✓ 模型已存在: {local_dir} ({len(files)} 权重文件)")
            return str(local_dir)
        else:
            print("  目录存在但无权重文件，重新下载...")

    info = RECOMMENDED_MODELS.get(model_id, {})
    size_gb = info.get("size_gb", "?")
    vram_gb = info.get("vram_gb", "?")
    print(f"\n下载模型: {model_id}")
    print(f"  预计大小: {size_gb} GB  |  推理显存需求: {vram_gb} GB")
    print(f"  保存路径: {local_dir}")
    print("  (下载可能需要 3-30 分钟，视网络情况)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if token:
        kwargs["token"] = token

    try:
        snapshot_download(**kwargs)
        print(f"\n✓ 下载完成: {local_dir}")
    except Exception as exc:
        print(f"\n[错误] 下载失败: {exc}")
        print("  提示：")
        print("  1. 检查网络连接")
        print("  2. 如需授权模型，请提供 HF Token：")
        print("     python scripts/download_model.py --model MODEL_ID --token hf_xxxx")
        print("  3. 国内网络可设置镜像：")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
        return None

    return str(local_dir)


def update_yaml_config(model_id: str, local_path: str, tensor_parallel: int = 1):
    """打印 experiments.yaml 的 profile 更新提示，不直接改写主配置。"""
    yaml_path = REPO_ROOT / "configs" / "experiments.yaml"
    if not yaml_path.exists():
        return

    info = RECOMMENDED_MODELS.get(model_id, {})
    profile = info.get("profile")
    workload = info.get("workload")

    print("ℹ 当前 experiments.yaml 已改为 profile_selection + model_profiles 结构。")
    print("  为避免误改主配置，下载脚本不再自动重写整个 YAML 文件。")
    if profile:
        print(f"  推荐 model profile   : {profile}")
        print(f"  推荐 workload profile: {workload}")
        print(f"  请确认 {yaml_path} 中的 model_profiles.{profile}.model.name 为：")
        print(f'    "{local_path}"')
        if tensor_parallel > 1:
            print(f"  并确认 model_profiles.{profile}.model.tensor_parallel_size = {tensor_parallel}")
    else:
        print("  当前模型未绑定现成 profile；请按实验需要手动补充。")


def print_model_list():
    print("\n可用模型（适用于 RTX 3090 24 GB）：\n")
    print(f"  {'模型 ID':<45} {'大小':>6} {'显存':>6}  说明")
    print(f"  {'─'*45} {'─'*6} {'─'*6}  {'─'*30}")
    for mid, info in RECOMMENDED_MODELS.items():
        print(f"  {mid:<45} {info['size_gb']:>5}G {info['vram_gb']:>5}G  {info['desc']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="下载 FaaSLoRA 实验所需大模型")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace 模型 ID")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace Token（授权模型需要；默认读取环境变量 HF_TOKEN）",
    )
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="张量并行数（双 3090 填 2）")
    parser.add_argument("--list", action="store_true", help="列出可用模型")
    args = parser.parse_args()

    if args.list:
        print_model_list()
        return

    try:
        validate_token(args.token)
    except ValueError as exc:
        print(f"\n[错误] {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  FaaSLoRA 大模型下载工具")
    print("=" * 60)

    local_path = download_model(args.model, token=args.token, force=args.force)
    if not local_path:
        sys.exit(1)

    # 打印 YAML / profile 使用提示
    update_yaml_config(args.model, local_path, tensor_parallel=args.tensor_parallel)

    # 提示配置建议
    info = RECOMMENDED_MODELS.get(args.model, {})
    vram = info.get("vram_gb", 16)
    print("\n  配置建议（在 configs/experiments.yaml 中修改）：")
    print("    model:")
    print(f"      name: \"{local_path}\"")
    if args.tensor_parallel > 1:
        print(f"      tensor_parallel_size: {args.tensor_parallel}")
    print(f"      gpu_memory_utilization: {'0.85' if vram < 20 else '0.80'}")
    print("    hardware:")
    # Recommend gpu_budget_mb based on vram
    gpu_mb = 24576  # 3090 24GB
    print(f"      gpu_budget_mb: {gpu_mb}")
    model_weights_mb = int(info.get("size_gb", 7) * 1024)
    print(f"      model_weights_mb: {model_weights_mb}")
    kv_per_1k = round(info.get("size_gb", 7) / 7 * 2.0, 1)
    print(f"      kv_per_1k_tokens_mb: {kv_per_1k}")
    profile = info.get("profile")
    workload = info.get("workload")
    if profile and workload:
        print("\n  现在可以运行完整实验：")
        print(f"    FAASLORA_PROFILE_MODEL={profile} \\")
        print("    FAASLORA_PROFILE_DATASET=azure_sharegpt_rep1000 \\")
        print(f"    FAASLORA_PROFILE_WORKLOAD={workload} \\")
        print("    python scripts/run_all_experiments.py --config configs/experiments.yaml")
    else:
        print("\n  现在可以运行完整实验：")
        print("    python scripts/run_all_experiments.py --config configs/experiments.yaml")


if __name__ == "__main__":
    main()
