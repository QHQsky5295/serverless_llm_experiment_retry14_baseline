#!/usr/bin/env python3
"""
环境依赖检查脚本
================
输出 Python、torch、transformers、peft、accelerate 等核心依赖版本，
并判断是否满足 FaaSLoRA 实验运行需求。

用法：
  python scripts/check_env.py
  或：conda activate LLM && python scripts/check_env.py
"""

import platform
import sys


def main() -> None:
    ok = True
    print("=" * 60)
    print("  FaaSLoRA 环境依赖检查")
    print("=" * 60)
    print()

    # Python
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    print(f"Python: {py_str} ({platform.python_implementation()})")
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 8):
        print("  [X] 需要 Python >= 3.8")
        ok = False
    else:
        print("  [OK] Python 版本满足")
    print()

    # torch
    try:
        import torch

        torch_str = torch.__version__
        cuda_available = torch.cuda.is_available()
        print(f"torch: {torch_str}")
        if cuda_available:
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  cuDNN: {torch.backends.cudnn.version()}")
            print("  [OK] CUDA 可用")
        else:
            print("  [X] CUDA 不可用，实验需要 GPU")
            ok = False
        if not torch_str.startswith("2."):
            print(f"  [!] 推荐 torch>=2.0，当前 {torch_str}")
            # 不强制失败，2.x 一般可用
    except ImportError as e:
        print(f"torch: [X] 未安装 ({e})")
        ok = False
    print()

    # transformers
    try:
        import transformers

        tf_str = transformers.__version__
        print(f"transformers: {tf_str}")
        # 简单版本检查（lexicographic）
        try:
            if tuple(map(int, transformers.__version__.split(".")[:2])) < (4, 30):
                print("  [!] 推荐 transformers>=4.30")
            else:
                print("  [OK] 版本满足")
        except Exception:
            print("  [OK] 已安装")
    except ImportError as e:
        print(f"transformers: [X] 未安装 ({e})")
        ok = False
    print()

    # peft
    try:
        import peft

        peft_str = getattr(peft, "__version__", "未知")
        print(f"peft: {peft_str}")
        print("  [OK] 已安装")
    except ImportError as e:
        print(f"peft: [X] 未安装 ({e})")
        ok = False
    print()

    # accelerate
    try:
        import accelerate

        acc_str = getattr(accelerate, "__version__", "未知")
        print(f"accelerate: {acc_str}")
        print("  [OK] 已安装")
    except ImportError as e:
        print(f"accelerate: [X] 未安装 ({e})")
        ok = False
    print()

    # GPU 信息
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                print(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)")
            print("  [OK] 至少 1 张 GPU")
        else:
            print("GPU: 无可用设备")
    except Exception as e:
        print(f"GPU: 无法获取信息 ({e})")
    print()

    # 总结
    print("=" * 60)
    if ok:
        print("  结论: 环境满足实验要求，可运行 run_all_experiments.py")
    else:
        print("  结论: 部分依赖缺失或不满足，请先安装/升级")
        print("  建议: pip install torch transformers peft accelerate")
    print("=" * 60)


if __name__ == "__main__":
    main()
