#!/usr/bin/env python3
"""
FaaSLoRA Dataset Downloader
=============================

Downloads and caches the datasets used in comparable LLM inference papers:

  1. ShareGPT (HuggingFace)
     - Used by: S-LoRA (SOSP'23), Punica (MLSys'24), vLLM benchmarks
     - ~90K conversations, variable-length prompts
     - Saved to: data/sharegpt/sharegpt_cache.json

  2. Azure LLM Inference Trace (already included)
     - Used by: ServerlessLLM (NSDI'24), MuxServe (SC'24)
     - ~28K records with timestamps + token counts
     - Location: data/azure_llm/AzureLLMInferenceTrace_*.csv

Usage:
  conda activate LLM
  python scripts/download_datasets.py
  python scripts/download_datasets.py --sharegpt-only
  python scripts/download_datasets.py --max-samples 10000
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def download_sharegpt(max_samples: int = 5000, force: bool = False):
    """Download ShareGPT dataset from HuggingFace and cache locally."""
    from faaslora.datasets.dataset_loader import ShareGPTLoader, SHAREGPT_CACHE

    if SHAREGPT_CACHE.exists() and not force:
        with open(SHAREGPT_CACHE) as f:
            existing = json.load(f)
        print(f"  ShareGPT already cached: {len(existing)} records at {SHAREGPT_CACHE}")
        if len(existing) >= max_samples:
            return

    print(f"  Downloading ShareGPT from HuggingFace (max {max_samples} samples)...")
    print("  (This requires internet access and may take 1-2 minutes)")
    loader = ShareGPTLoader()
    records = loader.load(max_samples=max_samples)

    if loader.get_source() == "huggingface":
        print(f"  Downloaded {len(records)} ShareGPT records")
        stats = loader.get_stats()
        print(f"  Input tokens:  mean={stats['input_tokens']['mean']:.0f}  "
              f"p50={stats['input_tokens']['p50']:.0f}  "
              f"p95={stats['input_tokens']['p95']:.0f}")
        print(f"  Output tokens: mean={stats['output_tokens']['mean']:.0f}  "
              f"p50={stats['output_tokens']['p50']:.0f}")
    elif loader.get_source() == "cache":
        print(f"  Loaded {len(records)} records from cache")
    else:
        print("  WARNING: Could not download ShareGPT, using embedded prompts")
        print("  To download, ensure 'datasets' package is installed:")
        print("    pip install datasets huggingface_hub")


def verify_azure_trace():
    """Verify Azure LLM trace files exist and show statistics."""
    from faaslora.datasets.dataset_loader import AzureTraceLoader

    loader = AzureTraceLoader()
    records = loader.load()

    if not records:
        print("  WARNING: Azure LLM trace not found at data/azure_llm/*.csv")
        print("  The trace files should already be included with the repository.")
        return False

    stats = loader.get_stats()
    print(f"  Azure LLM trace: {stats['total_records']} records")
    print(f"    Conversation: {stats['conv_records']}  Code: {stats['code_records']}")
    print(f"    Context tokens:  p50={stats['context_tokens']['p50']:.0f}  "
          f"p95={stats['context_tokens']['p95']:.0f}")
    print(f"    Generated tokens: p50={stats['generated_tokens']['p50']:.0f}  "
          f"p95={stats['generated_tokens']['p95']:.0f}")
    print(f"    Implied arrival rate: {stats['implied_rps']:.2f} req/s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download FaaSLoRA datasets")
    parser.add_argument("--sharegpt-only", action="store_true")
    parser.add_argument("--azure-only",    action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max ShareGPT samples to download (default: 5000)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cache exists")
    args = parser.parse_args()

    print("=" * 60)
    print("  FaaSLoRA Dataset Downloader")
    print("=" * 60)

    if not args.sharegpt_only:
        print("\n[1/2] Verifying Azure LLM Inference Trace ...")
        verify_azure_trace()

    if not args.azure_only:
        print("\n[2/2] ShareGPT dataset ...")
        download_sharegpt(max_samples=args.max_samples, force=args.force)

    print("\n  Done. Run experiments with:")
    print("    python scripts/run_all_experiments.py --config configs/experiments.yaml")


if __name__ == "__main__":
    main()
