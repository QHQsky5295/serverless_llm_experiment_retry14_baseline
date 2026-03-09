#!/usr/bin/env python3
"""
FaaSLoRA 大模型下载脚本
========================

将 HuggingFace 开源大模型下载至本地 models/ 目录，供 vLLM 推理使用。

支持的模型（适用于 RTX 3090 24 GB）：
  - Qwen/Qwen2.5-0.5B-Instruct  (~1 GB，快速验证用)
  - Qwen/Qwen2.5-7B-Instruct    (~15 GB，论文实验推荐)
  - facebook/opt-1.3b            (~2.6 GB，无需授权)
  - meta-llama/Llama-3.1-8B-Instruct (~16 GB，需要 HuggingFace Token)

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
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "size_gb": 14.5,
        "desc": "论文实验推荐，RTX 3090 单卡可运行",
        "vram_gb": 18,
        "yaml_name": "Qwen/Qwen2.5-7B-Instruct",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "size_gb": 29.0,
        "desc": "大规模对比实验，需要双 3090（tensor_parallel=2）",
        "vram_gb": 30,
        "yaml_name": "Qwen/Qwen2.5-14B-Instruct",
    },
    "facebook/opt-1.3b": {
        "size_gb": 2.6,
        "desc": "无需授权，适合无 HuggingFace Token 的环境",
        "vram_gb": 4,
        "yaml_name": "facebook/opt-1.3b",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "size_gb": 16.0,
        "desc": "需要 HuggingFace Token，与 S-LoRA 论文相同基座",
        "vram_gb": 20,
        "yaml_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


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
            print(f"  目录存在但无权重文件，重新下载...")

    info = RECOMMENDED_MODELS.get(model_id, {})
    size_gb = info.get("size_gb", "?")
    vram_gb = info.get("vram_gb", "?")
    print(f"\n下载模型: {model_id}")
    print(f"  预计大小: {size_gb} GB  |  推理显存需求: {vram_gb} GB")
    print(f"  保存路径: {local_dir}")
    print(f"  (下载可能需要 3-30 分钟，视网络情况)")

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
        print("  2. 如需授权模型（如 Llama），请提供 HF Token：")
        print("     python scripts/download_model.py --model MODEL_ID --token hf_xxxx")
        print("  3. 国内网络可设置镜像：")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
        return None

    return str(local_dir)


def update_yaml_config(model_id: str, local_path: str, tensor_parallel: int = 1):
    """更新 configs/experiments.yaml 中的模型配置。"""
    import yaml

    yaml_path = REPO_ROOT / "configs" / "experiments.yaml"
    if not yaml_path.exists():
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 更新 model.name 字段
    import re
    new_content = re.sub(
        r'(^  name:\s*")[^"]*(")',
        f'\\1{local_path}\\2',
        content,
        flags=re.MULTILINE,
    )
    if tensor_parallel > 1:
        new_content = re.sub(
            r"(tensor_parallel_size:\s*)\d+",
            f"\\g<1>{tensor_parallel}",
            new_content,
        )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✓ 已更新 configs/experiments.yaml: model.name = {local_path}")


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
    parser.add_argument("--token", default=None, help="HuggingFace Token（授权模型需要）")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="张量并行数（双 3090 填 2）")
    parser.add_argument("--list", action="store_true", help="列出可用模型")
    args = parser.parse_args()

    if args.list:
        print_model_list()
        return

    print("=" * 60)
    print("  FaaSLoRA 大模型下载工具")
    print("=" * 60)

    local_path = download_model(args.model, token=args.token, force=args.force)
    if not local_path:
        sys.exit(1)

    # 更新 YAML 配置
    update_yaml_config(args.model, local_path, tensor_parallel=args.tensor_parallel)

    # 提示配置建议
    info = RECOMMENDED_MODELS.get(args.model, {})
    vram = info.get("vram_gb", 16)
    print("\n  配置建议（在 configs/experiments.yaml 中修改）：")
    print(f"    model:")
    print(f"      name: \"{local_path}\"")
    if args.tensor_parallel > 1:
        print(f"      tensor_parallel_size: {args.tensor_parallel}")
    print(f"      gpu_memory_utilization: {'0.85' if vram < 20 else '0.80'}")
    print(f"    hardware:")
    # Recommend gpu_budget_mb based on vram
    gpu_mb = 24576  # 3090 24GB
    print(f"      gpu_budget_mb: {gpu_mb}")
    model_weights_mb = int(info.get("size_gb", 7) * 1024)
    print(f"      model_weights_mb: {model_weights_mb}")
    kv_per_1k = round(info.get("size_gb", 7) / 7 * 2.0, 1)
    print(f"      kv_per_1k_tokens_mb: {kv_per_1k}")
    print(f"\n  现在可以运行完整实验：")
    print(f"    python scripts/run_all_experiments.py --config configs/experiments.yaml")


if __name__ == "__main__":
    main()
